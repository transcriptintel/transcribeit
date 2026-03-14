use std::time::Duration;

use regex::Regex;
use reqwest::header::HeaderMap;

const MAX_RETRIES: u32 = 5;
const DEFAULT_RETRY_SECS: u64 = 10;
const MAX_RETRY_SECS: u64 = 120;

/// Determine wait duration from a 429 response.
/// Checks Retry-After header first, then parses "retry after N seconds" from the body.
fn parse_retry_after(headers: &HeaderMap, body: &str) -> Duration {
    // Check Retry-After header
    if let Some(val) = headers.get("retry-after").and_then(|v| v.to_str().ok())
        && let Ok(secs) = val.parse::<u64>()
    {
        return Duration::from_secs(secs.min(MAX_RETRY_SECS));
    }

    // Parse "retry after N seconds" from error body
    if let Ok(re) = Regex::new(r"[Rr]etry after (\d+) seconds")
        && let Some(caps) = re.captures(body)
        && let Some(secs) = caps.get(1).and_then(|m| m.as_str().parse::<u64>().ok())
    {
        return Duration::from_secs(secs.min(MAX_RETRY_SECS));
    }

    Duration::from_secs(DEFAULT_RETRY_SECS)
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
pub async fn check_response(resp: reqwest::Response) -> RateLimitCheck {
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
        let wait = parse_retry_after(&headers, &body);
        RateLimitCheck::RetryAfter(wait)
    } else {
        RateLimitCheck::Error(status, body)
    }
}

pub fn max_retries() -> u32 {
    MAX_RETRIES
}
