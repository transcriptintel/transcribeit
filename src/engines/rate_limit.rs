use std::sync::LazyLock;
use std::time::Duration;

use regex::Regex;
use reqwest::header::HeaderMap;

static RETRY_AFTER_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"[Rr]etry after (\d+) seconds").unwrap());

const DEFAULT_MAX_RETRIES: u32 = 5;
const DEFAULT_REQUEST_TIMEOUT_SECS: u64 = 120;
const DEFAULT_RETRY_SECS: u64 = 10;
const MAX_RETRY_SECS: u64 = 120;

#[derive(Clone, Copy)]
pub struct ApiRequestSettings {
    pub request_timeout: Duration,
    pub max_retries: u32,
    pub default_retry_wait: Duration,
    pub max_retry_wait: Duration,
}

impl Default for ApiRequestSettings {
    fn default() -> Self {
        Self {
            request_timeout: Duration::from_secs(DEFAULT_REQUEST_TIMEOUT_SECS),
            max_retries: DEFAULT_MAX_RETRIES,
            default_retry_wait: Duration::from_secs(DEFAULT_RETRY_SECS),
            max_retry_wait: Duration::from_secs(MAX_RETRY_SECS),
        }
    }
}

impl ApiRequestSettings {
    pub fn new(
        request_timeout: Duration,
        max_retries: u32,
        default_retry_wait: Duration,
        max_retry_wait: Duration,
    ) -> Self {
        Self {
            request_timeout,
            max_retries,
            default_retry_wait,
            max_retry_wait,
        }
    }
}

/// Determine wait duration from a 429 response.
/// Checks Retry-After header first, then parses "retry after N seconds" from the body.
fn parse_retry_after(headers: &HeaderMap, body: &str, settings: &ApiRequestSettings) -> Duration {
    // Check Retry-After header
    if let Some(val) = headers.get("retry-after").and_then(|v| v.to_str().ok())
        && let Ok(secs) = val.parse::<u64>()
    {
        return Duration::from_secs(secs.min(settings.max_retry_wait.as_secs()));
    }

    // Parse "retry after N seconds" from error body
    if let Some(caps) = RETRY_AFTER_RE.captures(body)
        && let Some(secs) = caps.get(1).and_then(|m| m.as_str().parse::<u64>().ok())
    {
        return Duration::from_secs(secs.min(settings.max_retry_wait.as_secs()));
    }

    settings.default_retry_wait.min(settings.max_retry_wait)
}

/// Result of checking a response for rate limiting.
pub enum RateLimitCheck {
    /// Response is OK, body bytes included.
    Ok(bytes::Bytes),
    /// Non-429 error.
    Error(reqwest::StatusCode, String),
    /// Rate limited, should retry after this duration.
    RetryAfter(Duration),
}

/// Check a response for rate limiting. Returns the appropriate action.
pub async fn check_response(
    resp: reqwest::Response,
    settings: &ApiRequestSettings,
) -> RateLimitCheck {
    if resp.status().is_success() {
        match resp.bytes().await {
            Ok(body) => return RateLimitCheck::Ok(body),
            Err(e) => return RateLimitCheck::Error(reqwest::StatusCode::OK, e.to_string()),
        }
    }

    let status = resp.status();
    let headers = resp.headers().clone();
    let body = resp.text().await.unwrap_or_default();

    if status == reqwest::StatusCode::TOO_MANY_REQUESTS {
        let wait = parse_retry_after(&headers, &body, settings);
        RateLimitCheck::RetryAfter(wait)
    } else {
        RateLimitCheck::Error(status, body)
    }
}

/// Send a request with retry on 429 responses.
///
/// `build_request` is called for each attempt and must return a ready-to-send request.
/// Returns Ok(body_bytes) on success, or the error status+body on non-retryable failure.
pub async fn send_with_retry<F>(
    settings: &ApiRequestSettings,
    api_label: &str,
    mut build_request: F,
) -> Result<bytes::Bytes, (reqwest::StatusCode, String)>
where
    F: FnMut() -> std::pin::Pin<
        Box<dyn std::future::Future<Output = anyhow::Result<reqwest::Response>> + Send>,
    >,
{
    for attempt in 0..=settings.max_retries {
        let resp = match build_request().await {
            Ok(r) => r,
            Err(e) => {
                return Err((
                    reqwest::StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Failed to send request to {api_label}: {e}"),
                ));
            }
        };

        match check_response(resp, settings).await {
            RateLimitCheck::Ok(body) => return Ok(body),
            RateLimitCheck::RetryAfter(wait) => {
                if attempt == settings.max_retries {
                    return Err((
                        reqwest::StatusCode::TOO_MANY_REQUESTS,
                        format!(
                            "Rate limited after {} retries, last wait was {}s",
                            settings.max_retries,
                            wait.as_secs()
                        ),
                    ));
                }
                eprintln!(
                    "    Rate limited, retrying in {}s (attempt {}/{})...",
                    wait.as_secs(),
                    attempt + 1,
                    settings.max_retries
                );
                tokio::time::sleep(wait).await;
            }
            RateLimitCheck::Error(status, body) => return Err((status, body)),
        }
    }

    Err((
        reqwest::StatusCode::INTERNAL_SERVER_ERROR,
        "Retry loop exited unexpectedly".into(),
    ))
}
