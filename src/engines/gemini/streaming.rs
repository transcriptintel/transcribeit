use anyhow::{Context, Result};
use futures_util::StreamExt;
use serde_json::Value;

use super::GeminiApi;

impl GeminiApi {
    pub(super) async fn stream_generate_chunks(
        &self,
        url: &str,
        payload: &Value,
    ) -> Result<Vec<Value>> {
        for attempt in 0..=self.settings.max_retries {
            let response = self
                .client
                .post(url)
                .header("x-goog-api-key", &self.api_key)
                .header("Accept", "text/event-stream")
                .json(payload)
                .send()
                .await;

            let response = match response {
                Ok(response) => response,
                Err(err) => {
                    if attempt == self.settings.max_retries {
                        return Err(err).context("Failed to stream Gemini generateContent");
                    }
                    self.wait_before_retry(
                        attempt,
                        "Gemini streamGenerateContent transport error",
                        None,
                    )
                    .await;
                    continue;
                }
            };

            let status = response.status();
            if !status.is_success() {
                let body = response.text().await.unwrap_or_default();
                if is_retryable_status(status) && attempt < self.settings.max_retries {
                    self.wait_before_retry(
                        attempt,
                        "Gemini streamGenerateContent retryable response",
                        Some(status),
                    )
                    .await;
                    continue;
                }
                anyhow::bail!("Gemini streamGenerateContent returned {status}: {body}");
            }

            match read_sse_chunks(response).await {
                Ok(chunks) if !chunks.is_empty() => return Ok(chunks),
                Ok(_) if attempt < self.settings.max_retries => {
                    self.wait_before_retry(
                        attempt,
                        "Gemini streamGenerateContent returned an empty stream",
                        None,
                    )
                    .await;
                }
                Ok(_) => anyhow::bail!("Gemini streamGenerateContent returned an empty stream"),
                Err(err) if attempt < self.settings.max_retries => {
                    eprintln!("    Gemini stream failed: {err:#}");
                    self.wait_before_retry(
                        attempt,
                        "Gemini streamGenerateContent stream error",
                        None,
                    )
                    .await;
                }
                Err(err) => return Err(err),
            }
        }

        anyhow::bail!("Gemini streamGenerateContent retry loop exited unexpectedly")
    }

    async fn wait_before_retry(
        &self,
        attempt: u32,
        reason: &str,
        status: Option<reqwest::StatusCode>,
    ) {
        let wait = self
            .settings
            .default_retry_wait
            .min(self.settings.max_retry_wait);
        let status = status
            .map(|status| status.to_string())
            .unwrap_or_else(|| "n/a".to_string());
        eprintln!(
            "    {reason} ({status}), retrying in {}s (attempt {}/{})...",
            wait.as_secs(),
            attempt + 1,
            self.settings.max_retries
        );
        tokio::time::sleep(wait).await;
    }
}

async fn read_sse_chunks(response: reqwest::Response) -> Result<Vec<Value>> {
    let mut stream = response.bytes_stream();
    let mut buffer = String::new();
    let mut chunks = Vec::new();

    while let Some(next) = stream.next().await {
        let bytes = next.context("Failed to read Gemini stream chunk")?;
        buffer.push_str(&String::from_utf8_lossy(&bytes));

        while let Some((event_end, separator_len)) = next_sse_event(&buffer) {
            let remaining = buffer.split_off(event_end + separator_len);
            let event_data = std::mem::replace(&mut buffer, remaining);
            if let Some(value) = parse_sse_event(&event_data[..event_end])? {
                chunks.push(value);
            }
        }
    }

    if !buffer.trim().is_empty()
        && let Some(value) = parse_sse_event(&buffer)?
    {
        chunks.push(value);
    }

    Ok(chunks)
}

fn next_sse_event(buffer: &str) -> Option<(usize, usize)> {
    let lf = buffer.find("\n\n").map(|index| (index, 2));
    let crlf = buffer.find("\r\n\r\n").map(|index| (index, 4));
    match (lf, crlf) {
        (Some((lf_index, lf_len)), Some((crlf_index, crlf_len))) => {
            if lf_index < crlf_index {
                Some((lf_index, lf_len))
            } else {
                Some((crlf_index, crlf_len))
            }
        }
        (Some((index, len)), None) | (None, Some((index, len))) => Some((index, len)),
        (None, None) => None,
    }
}

fn parse_sse_event(event: &str) -> Result<Option<Value>> {
    let data = event
        .lines()
        .filter_map(|line| line.trim_start().strip_prefix("data:"))
        .map(str::trim_start)
        .collect::<Vec<_>>()
        .join("\n");
    let data = data.trim();
    if data.is_empty() || data == "[DONE]" {
        return Ok(None);
    }
    serde_json::from_str(data)
        .map(Some)
        .with_context(|| format!("Failed to parse Gemini SSE data: {data}"))
}

fn is_retryable_status(status: reqwest::StatusCode) -> bool {
    status == reqwest::StatusCode::TOO_MANY_REQUESTS || status.is_server_error()
}
