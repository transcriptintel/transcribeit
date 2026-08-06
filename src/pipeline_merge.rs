use serde_json::{Value, json};

use crate::transcriber::Transcript;

pub(crate) struct TranscriptChunk {
    pub index: usize,
    pub offset_ms: i64,
    pub transcript: Transcript,
}

pub(crate) fn merge_segmented_transcripts(
    provider: &str,
    mut chunks: Vec<TranscriptChunk>,
) -> Transcript {
    chunks.sort_by_key(|chunk| chunk.index);

    let mut segments = Vec::new();
    let mut metadata_chunks = Vec::with_capacity(chunks.len());
    for mut chunk in chunks {
        let provider_metadata = chunk.transcript.provider_metadata.take();
        chunk.transcript.shift_by(chunk.offset_ms);
        segments.append(&mut chunk.transcript.segments);
        metadata_chunks.push(json!({
            "index": chunk.index,
            "offset_ms": chunk.offset_ms,
            "provider_metadata": provider_metadata,
        }));
    }

    let provider_metadata = metadata_chunks
        .iter()
        .any(|chunk| !chunk["provider_metadata"].is_null())
        .then(|| segmented_metadata(provider, metadata_chunks));

    Transcript {
        segments,
        provider_metadata,
    }
}

fn segmented_metadata(provider: &str, chunks: Vec<Value>) -> Value {
    json!({
        "provider": provider,
        "schema_version": format!("{provider}.segmented-metadata.v1"),
        "data": {
            "segmented": true,
            "chunks": chunks,
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transcriber::{Segment, Word};

    #[test]
    fn merge_orders_chunks_offsets_words_and_preserves_metadata() {
        let merged = merge_segmented_transcripts(
            "deepgram",
            vec![
                chunk(1, 10_000, "second", "request-2"),
                chunk(0, 0, "first", "request-1"),
            ],
        );

        assert_eq!(merged.segments[0].text, "first");
        assert_eq!(merged.segments[1].text, "second");
        assert_eq!(merged.segments[1].start_ms, 10_000);
        assert_eq!(merged.segments[1].words[0].start_ms, 10_100);
        assert_eq!(
            merged.provider_metadata.as_ref().unwrap()["data"]["chunks"][1]["provider_metadata"]["request_id"],
            "request-2"
        );
    }

    fn chunk(index: usize, offset_ms: i64, text: &str, request_id: &str) -> TranscriptChunk {
        TranscriptChunk {
            index,
            offset_ms,
            transcript: Transcript {
                segments: vec![Segment {
                    start_ms: 0,
                    end_ms: 1_000,
                    text: text.to_string(),
                    words: vec![Word {
                        start_ms: 100,
                        end_ms: 300,
                        text: text.to_string(),
                        punctuation: None,
                    }],
                    ..Default::default()
                }],
                provider_metadata: Some(json!({"request_id": request_id})),
            },
        }
    }
}
