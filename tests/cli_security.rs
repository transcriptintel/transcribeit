use std::process::Command;

const SENTINELS: &[(&str, &str)] = &[
    ("OPENAI_API_KEY", "openai-help-sentinel"),
    ("AZURE_API_KEY", "azure-help-sentinel"),
    ("DASHSCOPE_API_KEY", "dashscope-help-sentinel"),
    ("GEMINI_API_KEY", "gemini-help-sentinel"),
    ("NVIDIA_API_KEY", "nvidia-help-sentinel"),
    ("DEEPGRAM_API_KEY", "deepgram-help-sentinel"),
    ("S3_ACCESS_KEY_ID", "s3-access-help-sentinel"),
    ("S3_SECRET_ACCESS_KEY", "s3-secret-help-sentinel"),
    ("S3_SESSION_TOKEN", "s3-session-help-sentinel"),
    ("HF_TOKEN", "hf-help-sentinel"),
];

#[test]
fn generated_help_never_renders_environment_values() {
    let working_directory = tempfile::tempdir().unwrap();

    for arguments in [
        vec!["--help"],
        vec!["help", "run"],
        vec!["run", "--help"],
        vec!["setup", "--help"],
        vec!["download-model", "--help"],
        vec!["--version"],
    ] {
        let mut command = Command::new(env!("CARGO_BIN_EXE_transcribeit"));
        command
            .args(&arguments)
            .current_dir(working_directory.path())
            .env_clear()
            .envs(SENTINELS.iter().copied());

        let output = command.output().unwrap();
        assert!(
            output.status.success(),
            "{arguments:?} failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );

        let rendered = format!(
            "{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        for (_, sentinel) in SENTINELS {
            assert!(
                !rendered.contains(sentinel),
                "{arguments:?} exposed {sentinel}"
            );
        }
    }
}

#[cfg(target_os = "macos")]
#[test]
fn apple_speech_rejects_missing_locale_before_input_discovery() {
    let working_directory = tempfile::tempdir().unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_transcribeit"))
        .args([
            "run",
            "--provider",
            "apple-speech",
            "--input",
            "missing-input",
        ])
        .current_dir(working_directory.path())
        .env_clear()
        .output()
        .unwrap();

    assert!(!output.status.success());
    let error = String::from_utf8_lossy(&output.stderr);
    assert!(error.contains("requires --language <locale>"));
    assert!(!error.contains("No input files"));
}
