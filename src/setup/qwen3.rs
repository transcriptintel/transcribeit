use std::path::PathBuf;

use anyhow::Result;

use crate::artifacts::ArtifactIntegrity;
use crate::models::models_dir;

pub(crate) const QWEN3_ASR_ARCHIVE: &str = "sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25";

pub(crate) async fn setup_qwen3_asr(output_dir: Option<PathBuf>) -> Result<String> {
    let install_root = output_dir.unwrap_or_else(models_dir);
    let check_dir = install_root.join(QWEN3_ASR_ARCHIVE);
    let url = format!(
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/{QWEN3_ASR_ARCHIVE}.tar.bz2"
    );

    super::download_and_extract(
        &url,
        &install_root,
        &check_dir,
        "Qwen3-ASR 0.6B int8 ONNX model",
        ArtifactIntegrity {
            sha256: "393f8a14e2f5fb96746aaab342997a40641001fbd5bf9592a080a8329178ee96",
            size_bytes: 878_702_423,
        },
    )
    .await
}
