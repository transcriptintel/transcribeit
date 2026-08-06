use std::path::{Path, PathBuf};

use anyhow::{Result, ensure};
use sherpa_onnx::{
    OfflineModelConfig, OfflineMoonshineModelConfig, OfflineQwen3ASRModelConfig,
    OfflineSenseVoiceModelConfig, OfflineWhisperModelConfig,
};

enum ModelArch {
    Qwen3Asr {
        conv_frontend: PathBuf,
        encoder: PathBuf,
        decoder: PathBuf,
        tokenizer: PathBuf,
    },
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

pub(super) fn build_model_config(
    model_dir: &Path,
    language: Option<String>,
    num_threads: i32,
) -> Result<OfflineModelConfig> {
    let config = match detect_model_arch(model_dir)? {
        ModelArch::Qwen3Asr {
            conv_frontend,
            encoder,
            decoder,
            tokenizer,
        } => {
            ensure!(
                language.is_none(),
                "Qwen3-ASR ONNX auto-detects language; --language is not supported"
            );
            OfflineModelConfig {
                qwen3_asr: OfflineQwen3ASRModelConfig {
                    conv_frontend: path_string(conv_frontend),
                    encoder: path_string(encoder),
                    decoder: path_string(decoder),
                    tokenizer: path_string(tokenizer),
                    max_new_tokens: 512,
                    ..Default::default()
                },
                num_threads,
                provider: Some("cpu".into()),
                ..Default::default()
            }
        }
        ModelArch::Whisper { encoder, decoder } => OfflineModelConfig {
            whisper: OfflineWhisperModelConfig {
                encoder: path_string(encoder),
                decoder: path_string(decoder),
                language: language.or(Some(String::new())),
                task: Some("transcribe".into()),
                tail_paddings: -1,
                ..Default::default()
            },
            tokens: path_string(probe_tokens_file(model_dir)?),
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
                preprocessor: path_string(preprocessor),
                encoder: path_string(encoder),
                uncached_decoder: path_string(uncached_decoder),
                cached_decoder: path_string(cached_decoder),
                ..Default::default()
            },
            tokens: path_string(probe_tokens_file(model_dir)?),
            num_threads,
            provider: Some("cpu".into()),
            ..Default::default()
        },
        ModelArch::SenseVoice { model } => OfflineModelConfig {
            sense_voice: OfflineSenseVoiceModelConfig {
                model: path_string(model),
                language: Some(language.unwrap_or_else(|| "auto".into())),
                use_itn: true,
            },
            tokens: path_string(probe_tokens_file(model_dir)?),
            num_threads,
            provider: Some("cpu".into()),
            ..Default::default()
        },
    };
    Ok(config)
}

fn detect_model_arch(model_dir: &Path) -> Result<ModelArch> {
    let qwen_conv = probe_model_file(model_dir, "conv_frontend");
    let qwen_encoder = probe_model_file(model_dir, "encoder");
    let qwen_decoder = probe_model_file(model_dir, "decoder");
    let qwen_tokenizer = model_dir.join("tokenizer");
    if let (Ok(conv_frontend), Ok(encoder), Ok(decoder), true) = (
        qwen_conv,
        qwen_encoder,
        qwen_decoder,
        qwen_tokenizer.is_dir(),
    ) {
        return Ok(ModelArch::Qwen3Asr {
            conv_frontend,
            encoder,
            decoder,
            tokenizer: qwen_tokenizer,
        });
    }

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

    if let (Ok(encoder), Ok(decoder)) = (
        probe_model_file(model_dir, "encoder"),
        probe_model_file(model_dir, "decoder"),
    ) {
        return Ok(ModelArch::Whisper { encoder, decoder });
    }
    if let Ok(model) = probe_model_file(model_dir, "model") {
        return Ok(ModelArch::SenseVoice { model });
    }

    anyhow::bail!(
        "Could not detect model architecture in {}. Expected Qwen3-ASR (conv_frontend+encoder+decoder+tokenizer), Whisper (encoder+decoder+tokens), Moonshine (preprocess+encode+cached_decode+uncached_decode+tokens), or SenseVoice (model+tokens).",
        model_dir.display()
    )
}

fn probe_model_file(model_dir: &Path, component: &str) -> Result<PathBuf> {
    let candidates = [
        format!("{component}.int8.onnx"),
        format!("{component}.onnx"),
        format!("*-{component}.int8.onnx"),
        format!("*-{component}.onnx"),
    ];
    for name in &candidates[..2] {
        let path = model_dir.join(name);
        if path.is_file() {
            return Ok(path);
        }
    }
    for pattern in &candidates[2..] {
        let glob_pattern = format!("{}/{}", model_dir.display(), pattern);
        if let Some(path) = glob::glob(&glob_pattern)
            .ok()
            .and_then(|mut paths| paths.find_map(|path| path.ok()))
        {
            return Ok(path);
        }
    }
    anyhow::bail!("missing {component} ONNX file in {}", model_dir.display())
}

fn probe_tokens_file(model_dir: &Path) -> Result<PathBuf> {
    let direct = model_dir.join("tokens.txt");
    if direct.is_file() {
        return Ok(direct);
    }
    let pattern = format!("{}/*-tokens.txt", model_dir.display());
    if let Some(path) = glob::glob(&pattern)
        .ok()
        .and_then(|mut paths| paths.find_map(|path| path.ok()))
    {
        return Ok(path);
    }
    anyhow::bail!("tokens.txt not found in {}", model_dir.display())
}

fn path_string(path: PathBuf) -> Option<String> {
    Some(path.to_string_lossy().into_owned())
}

#[cfg(test)]
mod tests {
    use super::build_model_config;

    fn qwen_dir() -> tempfile::TempDir {
        let root = tempfile::tempdir().unwrap();
        for name in [
            "conv_frontend.onnx",
            "encoder.int8.onnx",
            "decoder.int8.onnx",
        ] {
            std::fs::write(root.path().join(name), []).unwrap();
        }
        std::fs::create_dir(root.path().join("tokenizer")).unwrap();
        root
    }

    #[test]
    fn qwen_is_detected_before_whisper_and_does_not_require_tokens() {
        let root = qwen_dir();
        let config = build_model_config(root.path(), None, 3).unwrap();
        assert_eq!(config.num_threads, 3);
        assert!(config.tokens.is_none());
        assert_eq!(config.qwen3_asr.max_new_tokens, 512);
        assert!(config.qwen3_asr.conv_frontend.is_some());
    }

    #[test]
    fn qwen_rejects_unsupported_language_hint() {
        let root = qwen_dir();
        let error = build_model_config(root.path(), Some("en".into()), 2).unwrap_err();
        assert!(error.to_string().contains("--language is not supported"));
    }
}
