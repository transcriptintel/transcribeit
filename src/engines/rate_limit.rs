use std::sync::LazyLock;
use std::time::Duration;

use anyhow::{Context, Result};
use bytes::{Bytes, BytesMut};
use futures_util::StreamExt;
use regex::Regex;
use reqwest::header::HeaderMap;

static RETRY_AFTER_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"[Rr]etry after (\d+) seconds").unwrap());

const DEFAULT_MAX_RETRIES: u32 = 5;
const DEFAULT_REQUEST_TIMEOUT_SECS: u64 = 120;
const DEFAULT_RETRY_SECS: u64 = 10;
const MAX_RETRY_SECS: u64 = 120;
pub const MAX_RESULT_RESPONSE_BYTES: usize = 64 * 1024 * 1024;
pub const MAX_ERROR_RESPONSE_BYTES: usize = 1024 * 1024;

#[derive(Clone, Copy)]
pub struct ApiRequestSettings {
    pub request_timeout: Duration,
    pub max_retries: u32,
    pub default_retry_wait: Duration,
    pub max_retry_wait: Duration,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RetryPolicy {
    /// Safe to repeat after ambiguous transport failures and server errors.
    Idempotent,
    /// Retry only explicit rate-limit responses; ambiguous submissions are not replayed.
    RateLimitOnly,
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
    /// Request can be retried after this duration.
    RetryAfter {
        status: reqwest::StatusCode,
        wait: Duration,
        reason: &'static str,
    },
}

pub async fn read_response_limited(
    response: reqwest::Response,
    maximum_bytes: usize,
    context: &str,
) -> Result<Bytes> {
    if let Some(content_length) = response.content_length() {
        anyhow::ensure!(
            content_length <= maximum_bytes as u64,
            "{context} Content-Length {content_length} exceeds the {maximum_bytes}-byte limit"
        );
    }

    let mut body = BytesMut::new();
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.with_context(|| format!("Failed to read {context}"))?;
        anyhow::ensure!(
            chunk.len() <= maximum_bytes.saturating_sub(body.len()),
            "{context} exceeds the {maximum_bytes}-byte limit"
        );
        body.extend_from_slice(&chunk);
    }
    Ok(body.freeze())
}

/// Check a response for rate limiting. Returns the appropriate action.
pub async fn check_response(
    resp: reqwest::Response,
    settings: &ApiRequestSettings,
    retry_policy: RetryPolicy,
) -> RateLimitCheck {
    if resp.status().is_success() {
        match read_response_limited(resp, MAX_RESULT_RESPONSE_BYTES, "API response body").await {
            Ok(body) => return RateLimitCheck::Ok(body),
            Err(e) => return RateLimitCheck::Error(reqwest::StatusCode::OK, e.to_string()),
        }
    }

    let status = resp.status();
    let headers = resp.headers().clone();
    let body = read_response_limited(resp, MAX_ERROR_RESPONSE_BYTES, "API error response body")
        .await
        .map(|body| String::from_utf8_lossy(&body).into_owned())
        .unwrap_or_else(|error| error.to_string());

    if status == reqwest::StatusCode::TOO_MANY_REQUESTS {
        let wait = parse_retry_after(&headers, &body, settings);
        RateLimitCheck::RetryAfter {
            status,
            wait,
            reason: "rate limited",
        }
    } else if status.is_server_error() && retry_policy == RetryPolicy::Idempotent {
        RateLimitCheck::RetryAfter {
            status,
            wait: settings.default_retry_wait.min(settings.max_retry_wait),
            reason: "server error",
        }
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
    retry_policy: RetryPolicy,
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
                let error = e.to_string();
                if retry_policy == RetryPolicy::RateLimitOnly {
                    return Err((
                        reqwest::StatusCode::SERVICE_UNAVAILABLE,
                        format!(
                            "Request to {api_label} failed after submission became ambiguous and was not retried: {error}"
                        ),
                    ));
                }
                if attempt == settings.max_retries {
                    return Err((
                        reqwest::StatusCode::SERVICE_UNAVAILABLE,
                        format!(
                            "Failed to send request to {api_label} after {} retries: {}",
                            settings.max_retries, error
                        ),
                    ));
                }
                let wait = settings.default_retry_wait.min(settings.max_retry_wait);
                eprintln!(
                    "    Request to {api_label} failed, retrying in {}s (attempt {}/{})...",
                    wait.as_secs(),
                    attempt + 1,
                    settings.max_retries
                );
                tokio::time::sleep(wait).await;
                continue;
            }
        };

        match check_response(resp, settings, retry_policy).await {
            RateLimitCheck::Ok(body) => return Ok(body),
            RateLimitCheck::RetryAfter {
                status,
                wait,
                reason,
            } => {
                if attempt == settings.max_retries {
                    return Err((
                        status,
                        format!(
                            "{api_label} returned retryable {status} ({reason}) after {} retries, last wait was {}s",
                            settings.max_retries,
                            wait.as_secs()
                        ),
                    ));
                }
                eprintln!(
                    "    {api_label} returned {status} ({reason}), retrying in {}s (attempt {}/{})...",
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

#[cfg(test)]
mod tests {
    use super::{RetryPolicy, read_response_limited};

    #[test]
    fn non_idempotent_policy_is_distinct_from_idempotent_retries() {
        assert_ne!(RetryPolicy::RateLimitOnly, RetryPolicy::Idempotent);
    }

    #[tokio::test]
    async fn response_reader_rejects_declared_oversize_body() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            use tokio::io::{AsyncReadExt, AsyncWriteExt};
            let mut request = [0_u8; 1024];
            let _ = socket.read(&mut request).await;
            socket
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 6\r\n\r\nabcdef")
                .await
                .unwrap();
        });

        let response = reqwest::get(format!("http://{address}")).await.unwrap();
        let error = read_response_limited(response, 5, "test response")
            .await
            .expect_err("declared oversize response must fail");

        assert!(format!("{error:#}").contains("exceeds the 5-byte limit"));
        server.await.unwrap();
    }
}
