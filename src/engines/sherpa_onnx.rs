use std::path::PathBuf;
use std::sync::mpsc;
use std::thread::JoinHandle;

use anyhow::{Context, Result};
use async_trait::async_trait;
use sherpa_onnx::{OfflineRecognizer, OfflineRecognizerConfig};
use tokio::sync::oneshot;

use crate::transcriber::{Segment, Transcriber, Transcript};

mod model;

struct RecognizeRequest {
    samples: Vec<f32>,
    response_tx: oneshot::Sender<Result<Transcript>>,
}

pub struct SherpaOnnxEngine {
    request_tx: mpsc::Sender<RecognizeRequest>,
    _thread: JoinHandle<()>,
}

impl SherpaOnnxEngine {
    pub fn new(model_dir: PathBuf, language: Option<String>) -> Result<Self> {
        let num_threads = std::thread::available_parallelism()
            .map(|n| n.get() as i32)
            .unwrap_or(4);
        let model_config = model::build_model_config(&model_dir, language, num_threads)?;

        let (init_tx, init_rx) = std::sync::mpsc::channel::<Result<()>>();
        let (request_tx, request_rx) = mpsc::channel::<RecognizeRequest>();

        let thread = std::thread::spawn(move || {
            let config = OfflineRecognizerConfig {
                model_config,
                ..Default::default()
            };

            let recognizer = match OfflineRecognizer::create(&config) {
                Some(r) => {
                    init_tx.send(Ok(())).ok();
                    r
                }
                None => {
                    init_tx
                        .send(Err(anyhow::anyhow!(
                            "Failed to create sherpa-onnx recognizer from {}",
                            model_dir.display()
                        )))
                        .ok();
                    return;
                }
            };

            while let Ok(req) = request_rx.recv() {
                let result = recognize(&recognizer, &req.samples);
                req.response_tx.send(result).ok();
            }
        });

        init_rx
            .recv()
            .context("sherpa-onnx worker thread exited during init")??;

        Ok(Self {
            request_tx,
            _thread: thread,
        })
    }
}

#[async_trait]
impl Transcriber for SherpaOnnxEngine {
    async fn transcribe(&self, audio_samples: Vec<f32>) -> Result<Transcript> {
        let (response_tx, response_rx) = oneshot::channel();
        self.request_tx
            .send(RecognizeRequest {
                samples: audio_samples,
                response_tx,
            })
            .map_err(|_| anyhow::anyhow!("sherpa-onnx worker thread has stopped"))?;
        response_rx
            .await
            .context("sherpa-onnx worker dropped without responding")?
    }
}

fn recognize(recognizer: &OfflineRecognizer, samples: &[f32]) -> Result<Transcript> {
    let stream = recognizer.create_stream();
    stream.accept_waveform(16000, samples);

    // Suppress sherpa-onnx C++ warnings printed to stderr
    let _stderr_guard = suppress_stderr();
    recognizer.decode(&stream);
    drop(_stderr_guard);

    let result = stream
        .get_result()
        .context("sherpa-onnx returned no result")?;

    let segments = match result.timestamps {
        Some(ref timestamps) if !timestamps.is_empty() && !result.tokens.is_empty() => {
            tokens_to_segments(&result.tokens, timestamps)
        }
        _ => {
            vec![Segment {
                start_ms: 0,
                end_ms: 0,
                text: result.text.clone(),
                speaker: None,
                ..Default::default()
            }]
        }
    };

    Ok(Transcript {
        segments,
        provider_metadata: None,
    })
}

/// Group per-token timestamps into sentence-level segments.
fn tokens_to_segments(tokens: &[String], timestamps: &[f32]) -> Vec<Segment> {
    if tokens.is_empty() {
        return vec![];
    }

    let mut segments = Vec::new();
    let mut seg_start_idx = 0;
    let len = tokens.len().min(timestamps.len());

    for i in 0..len {
        let is_last = i == len - 1;
        let token = tokens[i].trim();
        let is_sentence_end = token.ends_with('.') || token.ends_with('?') || token.ends_with('!');
        let has_time_gap =
            !is_last && i + 1 < timestamps.len() && timestamps[i + 1] - timestamps[i] > 3.0;

        if is_last || is_sentence_end || has_time_gap {
            let text: String = tokens[seg_start_idx..=i].join("");
            let trimmed = text.trim();
            if !trimmed.is_empty() {
                segments.push(Segment {
                    start_ms: (timestamps[seg_start_idx] * 1000.0) as i64,
                    end_ms: (timestamps[i] * 1000.0) as i64,
                    text: trimmed.to_string(),
                    speaker: None,
                    ..Default::default()
                });
            }
            seg_start_idx = i + 1;
        }
    }

    if segments.is_empty() {
        let text: String = tokens
            .iter()
            .map(|t| t.as_str())
            .collect::<Vec<_>>()
            .join("");
        let trimmed = text.trim();
        if !trimmed.is_empty() {
            return vec![Segment {
                start_ms: (timestamps[0] * 1000.0) as i64,
                end_ms: (timestamps[len.saturating_sub(1)] * 1000.0) as i64,
                text: trimmed.to_string(),
                speaker: None,
                ..Default::default()
            }];
        }
    }

    segments
}

/// Temporarily redirect stderr to /dev/null to suppress C++ library warnings.
fn suppress_stderr() -> Option<StderrGuard> {
    use std::os::unix::io::AsRawFd;

    let stderr_fd = std::io::stderr().as_raw_fd();
    let saved = unsafe { libc::dup(stderr_fd) };
    if saved < 0 {
        return None;
    }

    let devnull = std::fs::File::open("/dev/null").ok()?;
    let devnull_fd = devnull.as_raw_fd();
    unsafe { libc::dup2(devnull_fd, stderr_fd) };

    Some(StderrGuard {
        saved_fd: saved,
        stderr_fd,
    })
}

struct StderrGuard {
    saved_fd: i32,
    stderr_fd: i32,
}

impl Drop for StderrGuard {
    fn drop(&mut self) {
        unsafe {
            libc::dup2(self.saved_fd, self.stderr_fd);
            libc::close(self.saved_fd);
        }
    }
}
