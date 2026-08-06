use anyhow::{Context, Result};
use std::path::Path;

pub(crate) fn resolve_explicit_key(
    explicit_key: Option<String>,
    key_file: Option<&Path>,
) -> Result<Option<String>> {
    if explicit_key.is_some() {
        return Ok(explicit_key);
    }
    let Some(path) = key_file else {
        return Ok(None);
    };

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = std::fs::metadata(path)
            .with_context(|| format!("Failed to inspect API key file {}", path.display()))?
            .permissions()
            .mode();
        anyhow::ensure!(
            mode & 0o077 == 0,
            "API key file {} must not be accessible by group or other users; run chmod 600 {}",
            path.display(),
            path.display()
        );
    }

    let key = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read API key file {}", path.display()))?;
    let key = key.trim();
    anyhow::ensure!(!key.is_empty(), "API key file {} is empty", path.display());
    anyhow::ensure!(
        !key.contains(['\r', '\n', '\0']),
        "API key file {} must contain exactly one key",
        path.display()
    );
    Ok(Some(key.to_string()))
}

pub(crate) fn resolve_openai_key(explicit_key: Option<String>) -> Result<String> {
    explicit_key
        .or_else(|| std::env::var("OPENAI_API_KEY").ok())
        .context("--api-key or OPENAI_API_KEY is required for --provider openai")
}

pub(crate) fn resolve_provider_key(
    provider_key: Option<String>,
    explicit_key: Option<String>,
    provider: &str,
    provider_flag: &str,
    provider_env: &str,
) -> Result<String> {
    provider_key.or(explicit_key).with_context(|| {
        format!(
            "{provider_flag}, {provider_env}, or an explicit --api-key is required for --provider {provider}"
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provider_key_error_never_suggests_openai_environment() {
        let error =
            resolve_provider_key(None, None, "gemini", "--gemini-api-key", "GEMINI_API_KEY")
                .unwrap_err();

        assert!(error.to_string().contains("GEMINI_API_KEY"));
        assert!(!error.to_string().contains("OPENAI_API_KEY"));
    }

    #[test]
    fn explicit_key_remains_an_intentional_override() {
        let key = resolve_provider_key(
            None,
            Some("explicit".to_string()),
            "deepgram",
            "--deepgram-api-key",
            "DEEPGRAM_API_KEY",
        )
        .unwrap();

        assert_eq!(key, "explicit");
    }

    #[cfg(unix)]
    #[test]
    fn reads_key_from_private_file_and_rejects_public_file() {
        use std::os::unix::fs::PermissionsExt;

        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("provider.key");
        std::fs::write(&path, "private-key\n").unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();
        assert_eq!(
            resolve_explicit_key(None, Some(&path)).unwrap(),
            Some("private-key".to_string())
        );

        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).unwrap();
        assert!(resolve_explicit_key(None, Some(&path)).is_err());
    }
}
