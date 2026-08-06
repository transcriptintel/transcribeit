use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, ensure};

use crate::artifacts::{ArtifactIntegrity, seal_installed_directory};

const SHERPA_ONNX_VERSION: &str = "v1.13.4";

pub(crate) async fn setup_sherpa_libs(output_dir: Option<PathBuf>) -> Result<String> {
    let os = std::env::consts::OS;
    let arch = std::env::consts::ARCH;
    let archive_suffix = match (os, arch) {
        ("macos", "x86_64") => "osx-x64-shared-lib",
        ("macos", "aarch64") => "osx-arm64-shared-lib",
        ("linux", "x86_64") => "linux-x64-shared-lib",
        ("linux", "aarch64") => "linux-aarch64-shared-cpu-lib",
        _ => anyhow::bail!(
            "Unsupported platform: {os}-{arch}. Download sherpa-onnx shared libraries manually."
        ),
    };

    let archive_name = format!("sherpa-onnx-{SHERPA_ONNX_VERSION}-{archive_suffix}");
    let integrity = sherpa_library_integrity(archive_suffix)?;
    let url = format!(
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/{SHERPA_ONNX_VERSION}/{archive_name}.tar.bz2"
    );
    let install_root = super::absolute_path(output_dir.unwrap_or_else(|| PathBuf::from("vendor")))?;
    let check_dir = install_root.join(&archive_name);
    let status = super::download_and_extract(
        &url,
        &install_root,
        &check_dir,
        "sherpa-onnx shared libraries",
        integrity,
    )
    .await?;
    let repaired = repair_macos_onnxruntime_signatures(&check_dir)?;

    Ok(format!(
        "{status} ({archive_suffix}, lib={}, macos_signatures_repaired={repaired})",
        check_dir.join("lib").display()
    ))
}

pub(super) fn sherpa_lib_dir_hint(install_root: &Path) -> Option<PathBuf> {
    let install_root = super::absolute_path(install_root.to_path_buf()).ok()?;
    let expected_prefix = format!("sherpa-onnx-{SHERPA_ONNX_VERSION}-");
    let entries: Vec<_> = std::fs::read_dir(&install_root)
        .ok()?
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| path.is_dir() && path.join("lib").exists())
        .collect();

    entries
        .iter()
        .find(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with(&expected_prefix))
        })
        .or_else(|| entries.first())
        .map(|path| path.join("lib"))
}

fn repair_macos_onnxruntime_signatures(install_dir: &Path) -> Result<bool> {
    if std::env::consts::OS != "macos" {
        return Ok(false);
    }

    let mut repaired = false;
    for name in ["libonnxruntime.dylib", "libonnxruntime.1.27.0.dylib"] {
        let library = install_dir.join("lib").join(name);
        ensure!(
            library.is_file(),
            "Missing Sherpa runtime library {}",
            library.display()
        );
        let valid = Command::new("codesign")
            .args(["--verify", "--strict"])
            .arg(&library)
            .status()
            .context("Failed to verify macOS ONNX Runtime signature")?
            .success();
        if !valid {
            let status = Command::new("codesign")
                .args(["--force", "--sign", "-"])
                .arg(&library)
                .status()
                .context("Failed to repair macOS ONNX Runtime signature")?;
            ensure!(
                status.success(),
                "Failed to ad-hoc sign {}",
                library.display()
            );
            repaired = true;
        }
    }
    if repaired {
        seal_installed_directory(install_dir)?;
    }
    Ok(repaired)
}

fn sherpa_library_integrity(archive_suffix: &str) -> Result<ArtifactIntegrity> {
    let integrity = match archive_suffix {
        "osx-x64-shared-lib" => (
            "24d37d744b9f4b6b6bff618ede6cede527d7c0073fcddeb554b5d13242a4544b",
            17_696_902,
        ),
        "osx-arm64-shared-lib" => (
            "995d38d323eef0bfbfe7432dcceffda91bbd95525a15fa64fed517ed368378b9",
            15_741_101,
        ),
        "linux-x64-shared-lib" => (
            "3e7ce80379c938668f11157b1f54a0272b40972f618f445dcae71d122764d1fa",
            9_534_742,
        ),
        "linux-aarch64-shared-cpu-lib" => (
            "8993fd2ae4c435345f270231ae9af48def799638c8b1b06c0e48512e47c39e4d",
            12_290_972,
        ),
        _ => anyhow::bail!("No pinned integrity metadata for {archive_suffix}"),
    };
    Ok(ArtifactIntegrity {
        sha256: integrity.0,
        size_bytes: integrity.1,
    })
}
