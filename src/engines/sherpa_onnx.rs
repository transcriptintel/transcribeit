use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::thread::JoinHandle;

use anyhow::{Context, Result};
use async_trait::async_trait;
use sherpa_onnx::{
    OfflineModelConfig, OfflineMoonshineModelConfig, OfflineRecognizer, OfflineRecognizerConfig,
    OfflineSenseVoiceModelConfig, OfflineWhisperModelConfig,
};
use tokio::sync::oneshot;

use crate::transcriber::{Segment, Transcriber, Transcript};

struct RecognizeRequest {
    samples: Vec<f32>,
    response_tx: oneshot::Sender<Result<Transcript>>,
}

pub struct SherpaOnnxEngine {
    request_tx: mpsc::Sender<RecognizeRequest>,
    _thread: JoinHandle<()>,
}

/// Detected model architecture based on files present in the model directory.
enum ModelArch {
    Whisper {
        encoder: PathBuf,
        decoder: PathBuf,
    },
    Moonshine {
        preprocessor: PathBuf,
        encoder: PathBuf,
        uncached_decoder: PathBuf,
        cached_decoder: PathBuf,
    },
    SenseVoice {
        model: PathBuf,
    },
}

fn detect_model_arch(model_dir: &Path) -> Result<ModelArch> {
    // Check for Moonshine files first (most specific)
    let moonshine_preprocess = probe_model_file(model_dir, "preprocess");
    let moonshine_encode = probe_model_file(model_dir, "encode");
    let moonshine_uncached = probe_model_file(model_dir, "uncached_decode");
    let moonshine_cached = probe_model_file(model_dir, "cached_decode");

    if let (Ok(preprocessor), Ok(encoder), Ok(uncached_decoder), Ok(cached_decoder)) = (
        moonshine_preprocess,
        moonshine_encode,
        moonshine_uncached,
        moonshine_cached,
    ) {
        return Ok(ModelArch::Moonshine {
            preprocessor,
            encoder,
            uncached_decoder,
            cached_decoder,
        });
    }

    // Check for Whisper files (encoder + decoder pair)
    if let (Ok(encoder), Ok(decoder)) = (
        probe_model_file(model_dir, "encoder"),
        probe_model_file(model_dir, "decoder"),
    ) {
        return Ok(ModelArch::Whisper { encoder, decoder });
    }

    // Check for SenseVoice (single model file)
    if let Ok(model) = probe_model_file(model_dir, "model") {
        return Ok(ModelArch::SenseVoice { model });
    }

    anyhow::bail!(
        "Could not detect model architecture in {}. Expected Whisper (encoder+decoder), Moonshine (preprocess+encode+cached_decode+uncached_decode), or SenseVoice (model) ONNX files.",
        model_dir.display()
    )
}

impl SherpaOnnxEngine {
    pub fn new(model_dir: PathBuf, language: Option<String>) -> Result<Self> {
        let arch = detect_model_arch(&model_dir)?;
        let tokens = probe_tokens_file(&model_dir)?;

        let (init_tx, init_rx) = std::sync::mpsc::channel::<Result<()>>();
        let (request_tx, request_rx) = mpsc::channel::<RecognizeRequest>();

        let thread = std::thread::spawn(move || {
            let num_threads = std::thread::available_parallelism()
                .map(|n| n.get() as i32)
                .unwrap_or(4);

            let model_config = match arch {
                ModelArch::Whisper { encoder, decoder } => OfflineModelConfig {
                    whisper: OfflineWhisperModelConfig {
                        encoder: Some(encoder.to_string_lossy().into_owned()),
                        decoder: Some(decoder.to_string_lossy().into_owned()),
                        language: language.or(Some(String::new())),
                        task: Some("transcribe".into()),
                        tail_paddings: -1,
                        ..Default::default()
                    },
                    tokens: Some(tokens.to_string_lossy().into_owned()),
                    num_threads,
                    provider: Some("cpu".into()),
                    ..Default::default()
                },
                ModelArch::Moonshine {
                    preprocessor,
                    encoder,
                    uncached_decoder,
                    cached_decoder,
                } => OfflineModelConfig {
                    moonshine: OfflineMoonshineModelConfig {
                        preprocessor: Some(preprocessor.to_string_lossy().into_owned()),
                        encoder: Some(encoder.to_string_lossy().into_owned()),
                        uncached_decoder: Some(uncached_decoder.to_string_lossy().into_owned()),
                        cached_decoder: Some(cached_decoder.to_string_lossy().into_owned()),
                        ..Default::default()
                    },
                    tokens: Some(tokens.to_string_lossy().into_owned()),
                    num_threads,
                    provider: Some("cpu".into()),
                    ..Default::default()
                },
                ModelArch::SenseVoice { model } => OfflineModelConfig {
                    sense_voice: OfflineSenseVoiceModelConfig {
                        model: Some(model.to_string_lossy().into_owned()),
                        language: Some(language.unwrap_or_else(|| "auto".into())),
                        use_itn: true,
                    },
                    tokens: Some(tokens.to_string_lossy().into_owned()),
                    num_threads,
                    provider: Some("cpu".into()),
                    ..Default::default()
                },
            };

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
            }]
        }
    };

    Ok(Transcript { segments })
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
            }];
        }
    }

    segments
}

/// Find an ONNX model file, preferring int8 variant.
fn probe_model_file(model_dir: &Path, component: &str) -> Result<PathBuf> {
    let candidates = [
        format!("{component}.int8.onnx"),
        format!("{component}.onnx"),
        format!("*-{component}.int8.onnx"),
        format!("*-{component}.onnx"),
    ];

    for name in &candidates[..2] {
        let path = model_dir.join(name);
        if path.exists() {
            return Ok(path);
        }
    }

    for pattern in &candidates[2..] {
        let glob_pattern = format!("{}/{}", model_dir.display(), pattern);
        if let Some(path) = glob::glob(&glob_pattern)
            .ok()
            .and_then(|mut paths| paths.find_map(|p| p.ok()))
        {
            return Ok(path);
        }
    }

    anyhow::bail!(
        "{component}.onnx (or {component}.int8.onnx) not found in {}",
        model_dir.display()
    )
}

/// Find tokens file (tokens.txt or *-tokens.txt).
fn probe_tokens_file(model_dir: &Path) -> Result<PathBuf> {
    let direct = model_dir.join("tokens.txt");
    if direct.exists() {
        return Ok(direct);
    }

    let glob_pattern = format!("{}/*-tokens.txt", model_dir.display());
    if let Some(path) = glob::glob(&glob_pattern)
        .ok()
        .and_then(|mut paths| paths.find_map(|p| p.ok()))
    {
        return Ok(path);
    }

    anyhow::bail!("tokens.txt not found in {}", model_dir.display())
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
