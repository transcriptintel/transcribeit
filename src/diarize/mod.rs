mod ffi;

use std::ffi::CString;
use std::path::Path;
use std::sync::mpsc;
use std::thread::JoinHandle;

use anyhow::{Context, Result};
use tokio::sync::oneshot;

use crate::transcriber::Transcript;

/// A speaker-labeled time span from diarization.
#[derive(Debug, Clone)]
pub struct DiarizedSegment {
    pub start_secs: f32,
    pub end_secs: f32,
    pub speaker: i32,
}

/// Request sent to the diarization worker thread.
struct DiarizeRequest {
    samples: Vec<f32>,
    response_tx: oneshot::Sender<Result<Vec<DiarizedSegment>>>,
}

/// Speaker diarization engine using sherpa-onnx's C API directly.
/// Runs on a dedicated thread (the C types are !Send/!Sync).
pub struct Diarizer {
    request_tx: mpsc::Sender<DiarizeRequest>,
    _thread: JoinHandle<()>,
}

impl Diarizer {
    /// Create a new diarizer.
    ///
    /// - `segmentation_model`: path to pyannote segmentation ONNX model
    /// - `embedding_model`: path to speaker embedding ONNX model
    /// - `num_speakers`: number of speakers (must be > 0)
    pub fn new(
        segmentation_model: &Path,
        embedding_model: &Path,
        num_speakers: i32,
    ) -> Result<Self> {
        let seg_model = segmentation_model.to_path_buf();
        let emb_model = embedding_model.to_path_buf();

        let (init_tx, init_rx) = std::sync::mpsc::channel::<Result<()>>();
        let (request_tx, request_rx) = mpsc::channel::<DiarizeRequest>();

        let thread = std::thread::spawn(move || {
            let seg_model_c =
                CString::new(seg_model.to_string_lossy().as_bytes()).unwrap_or_default();
            let emb_model_c =
                CString::new(emb_model.to_string_lossy().as_bytes()).unwrap_or_default();
            let provider_c = CString::new("cpu").unwrap();

            let config = ffi::SherpaOnnxOfflineSpeakerDiarizationConfig {
                segmentation: ffi::SherpaOnnxOfflineSpeakerSegmentationModelConfig {
                    pyannote: ffi::SherpaOnnxOfflineSpeakerSegmentationPyannoteModelConfig {
                        model: seg_model_c.as_ptr(),
                    },
                    num_threads: std::thread::available_parallelism()
                        .map(|n| n.get() as i32)
                        .unwrap_or(4),
                    debug: 0,
                    provider: provider_c.as_ptr(),
                },
                embedding: ffi::SherpaOnnxSpeakerEmbeddingExtractorConfig {
                    model: emb_model_c.as_ptr(),
                    num_threads: std::thread::available_parallelism()
                        .map(|n| n.get() as i32)
                        .unwrap_or(4),
                    debug: 0,
                    provider: provider_c.as_ptr(),
                },
                clustering: ffi::SherpaOnnxFastClusteringConfig {
                    num_clusters: num_speakers,
                    threshold: 0.5,
                },
                min_duration_on: 0.3,
                min_duration_off: 0.5,
            };

            let sd = unsafe { ffi::SherpaOnnxCreateOfflineSpeakerDiarization(&config) };
            if sd.is_null() {
                init_tx
                    .send(Err(anyhow::anyhow!(
                        "Failed to create speaker diarization engine"
                    )))
                    .ok();
                return;
            }

            init_tx.send(Ok(())).ok();

            while let Ok(req) = request_rx.recv() {
                let result = unsafe { process_diarization(sd, &req.samples) };
                req.response_tx.send(result).ok();
            }

            unsafe {
                ffi::SherpaOnnxDestroyOfflineSpeakerDiarization(sd);
            }
        });

        init_rx
            .recv()
            .context("Diarization worker thread exited during init")??;

        Ok(Self {
            request_tx,
            _thread: thread,
        })
    }

    /// Run diarization on audio samples (16kHz mono f32).
    pub async fn diarize(&self, samples: Vec<f32>) -> Result<Vec<DiarizedSegment>> {
        let (response_tx, response_rx) = oneshot::channel();
        self.request_tx
            .send(DiarizeRequest {
                samples,
                response_tx,
            })
            .map_err(|_| anyhow::anyhow!("Diarization worker thread has stopped"))?;
        response_rx
            .await
            .context("Diarization worker dropped without responding")?
    }
}

unsafe fn process_diarization(
    sd: *const ffi::SherpaOnnxOfflineSpeakerDiarization,
    samples: &[f32],
) -> Result<Vec<DiarizedSegment>> {
    let result = unsafe {
        ffi::SherpaOnnxOfflineSpeakerDiarizationProcess(sd, samples.as_ptr(), samples.len() as i32)
    };

    if result.is_null() {
        anyhow::bail!("Diarization returned null result");
    }

    let num_segments =
        unsafe { ffi::SherpaOnnxOfflineSpeakerDiarizationResultGetNumSegments(result) };
    let sorted = unsafe { ffi::SherpaOnnxOfflineSpeakerDiarizationResultSortByStartTime(result) };

    let mut segments = Vec::with_capacity(num_segments as usize);
    if !sorted.is_null() {
        for i in 0..num_segments as isize {
            let seg = unsafe { &*sorted.offset(i) };
            segments.push(DiarizedSegment {
                start_secs: seg.start,
                end_secs: seg.end,
                speaker: seg.speaker,
            });
        }
        unsafe { ffi::SherpaOnnxOfflineSpeakerDiarizationDestroySegment(sorted) };
    }

    unsafe { ffi::SherpaOnnxOfflineSpeakerDiarizationDestroyResult(result) };

    Ok(segments)
}

/// Assign speaker labels to transcript segments by timestamp overlap.
pub fn assign_speakers(transcript: &mut Transcript, diarized: &[DiarizedSegment]) {
    for seg in &mut transcript.segments {
        let seg_start = seg.start_ms as f32 / 1000.0;
        let seg_end = seg.end_ms as f32 / 1000.0;

        // Find the diarization segment with maximum overlap
        let mut best_speaker = None;
        let mut best_overlap = 0.0f32;

        for d in diarized {
            let overlap_start = seg_start.max(d.start_secs);
            let overlap_end = seg_end.min(d.end_secs);
            let overlap = (overlap_end - overlap_start).max(0.0);

            if overlap > best_overlap {
                best_overlap = overlap;
                best_speaker = Some(d.speaker);
            }
        }

        if let Some(speaker) = best_speaker {
            seg.speaker = Some(format!("Speaker {}", speaker));
        }
    }
}
