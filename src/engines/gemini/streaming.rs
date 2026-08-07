use anyhow::{Context, Result};
use futures_util::StreamExt;
use serde_json::Value;

use crate::engines::rate_limit::{MAX_ERROR_RESPONSE_BYTES, read_response_limited};

use super::GeminiApi;

const MAX_SSE_EVENT_BYTES: usize = 16 * 1024 * 1024;
const MAX_SSE_RESPONSE_BYTES: usize = 64 * 1024 * 1024;

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
                    return Err(err).context(
                        "Gemini streamGenerateContent submission became ambiguous and was not retried",
                    );
                }
            };

            let status = response.status();
            if !status.is_success() {
                let body = read_response_limited(
                    response,
                    MAX_ERROR_RESPONSE_BYTES,
                    "Gemini streamGenerateContent error response",
                )
                .await
                .map(|body| String::from_utf8_lossy(&body).into_owned())
                .unwrap_or_else(|error| error.to_string());
                if status == reqwest::StatusCode::TOO_MANY_REQUESTS
                    && attempt < self.settings.max_retries
                {
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
                Ok(_) => anyhow::bail!("Gemini streamGenerateContent returned an empty stream"),
                Err(err) => {
                    return Err(err).context(
                        "Gemini stream failed after the request was accepted and was not retried",
                    );
                }
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
    let mut decoder = SseDecoder::new(MAX_SSE_EVENT_BYTES, MAX_SSE_RESPONSE_BYTES);

    while let Some(next) = stream.next().await {
        let bytes = next.context("Failed to read Gemini stream chunk")?;
        decoder.push(&bytes)?;
    }

    decoder.finish()
}

struct SseDecoder {
    buffer: Vec<u8>,
    chunks: Vec<Value>,
    total_bytes: usize,
    maximum_event_bytes: usize,
    maximum_response_bytes: usize,
}

impl SseDecoder {
    fn new(maximum_event_bytes: usize, maximum_response_bytes: usize) -> Self {
        Self {
            buffer: Vec::new(),
            chunks: Vec::new(),
            total_bytes: 0,
            maximum_event_bytes,
            maximum_response_bytes,
        }
    }

    fn push(&mut self, bytes: &[u8]) -> Result<()> {
        anyhow::ensure!(
            bytes.len() <= self.maximum_response_bytes.saturating_sub(self.total_bytes),
            "Gemini SSE response exceeds the {}-byte limit",
            self.maximum_response_bytes
        );
        self.total_bytes += bytes.len();
        self.buffer.extend_from_slice(bytes);
        self.drain_complete_events()?;
        anyhow::ensure!(
            self.buffer.len() <= self.maximum_event_bytes,
            "Gemini SSE event exceeds the {}-byte limit",
            self.maximum_event_bytes
        );
        Ok(())
    }

    fn finish(mut self) -> Result<Vec<Value>> {
        self.drain_complete_events()?;
        if !self.buffer.iter().all(u8::is_ascii_whitespace)
            && let Some(value) = parse_sse_event(&self.buffer)?
        {
            self.chunks.push(value);
        }
        Ok(self.chunks)
    }

    fn drain_complete_events(&mut self) -> Result<()> {
        while let Some((event_end, separator_len)) = next_sse_event(&self.buffer) {
            anyhow::ensure!(
                event_end <= self.maximum_event_bytes,
                "Gemini SSE event exceeds the {}-byte limit",
                self.maximum_event_bytes
            );
            let remaining = self.buffer.split_off(event_end + separator_len);
            let event_data = std::mem::replace(&mut self.buffer, remaining);
            if let Some(value) = parse_sse_event(&event_data[..event_end])? {
                self.chunks.push(value);
            }
        }
        Ok(())
    }
}

fn next_sse_event(buffer: &[u8]) -> Option<(usize, usize)> {
    let lf = buffer
        .windows(2)
        .position(|window| window == b"\n\n")
        .map(|index| (index, 2));
    let crlf = buffer
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .map(|index| (index, 4));
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

fn parse_sse_event(event: &[u8]) -> Result<Option<Value>> {
    let event = std::str::from_utf8(event)
        .context("Gemini SSE event was not valid UTF-8 after complete event framing")?;
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

#[cfg(test)]
mod tests {
    use super::SseDecoder;

    #[test]
    fn preserves_utf8_split_across_network_chunks() {
        let payload = "data: {\"text\":\"café\"}\n\n".as_bytes();
        let split = payload.iter().position(|byte| *byte == 0xc3).unwrap() + 1;
        let mut decoder = SseDecoder::new(1024, 2048);
        decoder.push(&payload[..split]).unwrap();
        decoder.push(&payload[split..]).unwrap();

        let chunks = decoder.finish().unwrap();
        assert_eq!(chunks[0]["text"], "café");
    }

    #[test]
    fn rejects_oversize_event_and_response() {
        let mut event_decoder = SseDecoder::new(5, 100);
        assert!(event_decoder.push(b"data: x").is_err());

        let mut response_decoder = SseDecoder::new(100, 5);
        assert!(response_decoder.push(b"123456").is_err());
    }
}
