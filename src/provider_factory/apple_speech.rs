use anyhow::Result;

use super::{ProviderFactoryArgs, ProviderRuntime};
#[cfg(target_os = "macos")]
use crate::engines::apple_speech::AppleSpeech;

pub(super) fn build(args: &ProviderFactoryArgs<'_>) -> Result<ProviderRuntime> {
    #[cfg(target_os = "macos")]
    {
        let locale = resolve_locale(args.language)?;
        Ok(ProviderRuntime {
            engine: Box::new(AppleSpeech::new(locale)),
            analyzer: None,
            provider_name: "apple-speech".into(),
            model_name: "speech-transcriber".into(),
        })
    }

    #[cfg(not(target_os = "macos"))]
    {
        let _ = args;
        anyhow::bail!("provider 'apple-speech' requires macOS 26 or later")
    }
}

pub(super) fn resolve_locale(language: Option<&str>) -> Result<Option<String>> {
    let Some(language) = language.map(str::trim).filter(|value| !value.is_empty()) else {
        anyhow::bail!(
            "provider 'apple-speech' requires --language <locale> (for example ja-JP); use --language system to explicitly use the current macOS locale"
        );
    };

    if language.eq_ignore_ascii_case("auto") {
        anyhow::bail!(
            "provider 'apple-speech' does not support automatic language detection; pass --language <locale> or --language system"
        );
    }
    if language.eq_ignore_ascii_case("system") {
        return Ok(None);
    }

    Ok(Some(language.to_owned()))
}

#[cfg(test)]
mod tests {
    use super::resolve_locale;

    #[test]
    fn explicit_locale_is_preserved_for_apple_speech() {
        assert_eq!(
            resolve_locale(Some("ja-JP")).unwrap().as_deref(),
            Some("ja-JP")
        );
        assert_eq!(resolve_locale(Some(" ja ")).unwrap().as_deref(), Some("ja"));
    }

    #[test]
    fn system_locale_requires_explicit_opt_in() {
        assert_eq!(resolve_locale(Some("system")).unwrap(), None);
        assert_eq!(resolve_locale(Some("SYSTEM")).unwrap(), None);
    }

    #[test]
    fn missing_empty_and_automatic_locales_are_rejected() {
        let missing = resolve_locale(None).unwrap_err().to_string();
        let empty = resolve_locale(Some("  ")).unwrap_err().to_string();
        let automatic = resolve_locale(Some("auto")).unwrap_err().to_string();

        assert!(missing.contains("requires --language <locale>"));
        assert!(empty.contains("requires --language <locale>"));
        assert!(automatic.contains("does not support automatic language detection"));
    }
}
