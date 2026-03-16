//! VAD-based speech segmentation using sherpa-onnx's Silero VAD.
//! Produces clean speech boundaries that avoid mid-word cuts.

use anyhow::{Context, Result};
use sherpa_onnx::{SileroVadModelConfig, VadModelConfig, VoiceActivityDetector};

const SAMPLE_RATE: u32 = 16_000;
const FRAME_SIZE: usize = 512; // ~32ms at 16kHz

/// A speech chunk with sample-level boundaries.
#[derive(Debug, Clone)]
pub struct SpeechChunk {
    pub start_sample: usize,
    pub end_sample: usize,
}

impl SpeechChunk {
    pub fn start_secs(&self) -> f64 {
        self.start_sample as f64 / SAMPLE_RATE as f64
    }

    pub fn end_secs(&self) -> f64 {
        self.end_sample as f64 / SAMPLE_RATE as f64
    }

    pub fn duration_secs(&self) -> f64 {
        (self.end_sample - self.start_sample) as f64 / SAMPLE_RATE as f64
    }
}

/// Detect speech segments in audio using Silero VAD.
pub fn detect_speech_chunks(samples: &[f32], vad_model_path: &str) -> Result<Vec<SpeechChunk>> {
    let config = VadModelConfig {
        silero_vad: SileroVadModelConfig {
            model: Some(vad_model_path.to_string()),
            threshold: 0.5,
            min_silence_duration: 0.25,
            min_speech_duration: 0.1,
            window_size: FRAME_SIZE as i32,
            max_speech_duration: 30.0,
        },
        sample_rate: SAMPLE_RATE as i32,
        num_threads: 1,
        provider: Some("cpu".into()),
        debug: false,
        ..Default::default()
    };

    let vad = VoiceActivityDetector::create(&config, 60.0)
        .context("Failed to create VAD (check vad_model_path)")?;

    let mut chunks = Vec::new();
    let mut cursor = 0usize;

    while cursor < samples.len() {
        let end = (cursor + FRAME_SIZE).min(samples.len());
        let frame = &samples[cursor..end];
        vad.accept_waveform(frame);

        while let Some(seg) = vad.front() {
            chunks.push(SpeechChunk {
                start_sample: seg.start() as usize,
                end_sample: seg.start() as usize + seg.n() as usize,
            });
            vad.pop();
        }

        cursor = end;
    }

    vad.flush();

    while let Some(seg) = vad.front() {
        chunks.push(SpeechChunk {
            start_sample: seg.start() as usize,
            end_sample: seg.start() as usize + seg.n() as usize,
        });
        vad.pop();
    }

    Ok(chunks)
}

/// Add padding around each chunk to protect word boundaries.
pub fn pad_chunks(
    chunks: &[SpeechChunk],
    total_len: usize,
    pad_samples: usize,
) -> Vec<SpeechChunk> {
    chunks
        .iter()
        .map(|c| SpeechChunk {
            start_sample: c.start_sample.saturating_sub(pad_samples),
            end_sample: (c.end_sample + pad_samples).min(total_len),
        })
        .collect()
}

/// Merge chunks separated by less than max_gap_samples.
pub fn merge_close_chunks(chunks: &[SpeechChunk], max_gap_samples: usize) -> Vec<SpeechChunk> {
    if chunks.is_empty() {
        return Vec::new();
    }

    let mut sorted = chunks.to_vec();
    sorted.sort_by_key(|c| c.start_sample);

    let mut merged = Vec::new();
    let mut cur = sorted[0].clone();

    for next in sorted.into_iter().skip(1) {
        let gap = next.start_sample.saturating_sub(cur.end_sample);
        if gap <= max_gap_samples {
            cur.end_sample = cur.end_sample.max(next.end_sample);
        } else {
            merged.push(cur);
            cur = next;
        }
    }

    merged.push(cur);
    merged
}

/// Split chunks that exceed max duration, cutting at the lowest-energy point.
pub fn split_long_chunks(
    samples: &[f32],
    chunks: &[SpeechChunk],
    max_chunk_secs: f32,
) -> Vec<SpeechChunk> {
    let max_len = (max_chunk_secs * SAMPLE_RATE as f32) as usize;
    let mut out = Vec::new();

    for c in chunks {
        let mut start = c.start_sample;
        while c.end_sample.saturating_sub(start) > max_len {
            let target = start + max_len;
            // Search ±500ms around the target for the quietest spot
            let search_radius = (SAMPLE_RATE / 2) as usize;
            let left = target.saturating_sub(search_radius).max(start);
            let right = (target + search_radius).min(c.end_sample);

            let cut = find_low_energy_cut(samples, left, right).unwrap_or(target);

            out.push(SpeechChunk {
                start_sample: start,
                end_sample: cut,
            });
            start = cut;
        }

        if start < c.end_sample {
            out.push(SpeechChunk {
                start_sample: start,
                end_sample: c.end_sample,
            });
        }
    }

    out
}

/// Find the sample position with the lowest energy in a window.
fn find_low_energy_cut(samples: &[f32], start: usize, end: usize) -> Option<usize> {
    let window = 320; // 20ms window
    if end <= start + window || end > samples.len() {
        return None;
    }

    let mut best_pos = None;
    let mut best_energy = f32::INFINITY;

    let mut i = start;
    while i + window <= end {
        let energy: f32 = samples[i..i + window].iter().map(|x| x * x).sum::<f32>() / window as f32;

        if energy < best_energy {
            best_energy = energy;
            best_pos = Some(i + window / 2);
        }

        i += window / 2; // 50% overlap
    }

    best_pos
}

/// Full VAD pipeline: detect → pad → merge → split.
/// Returns clean speech chunks ready for STT.
pub fn vad_segment(
    samples: &[f32],
    vad_model_path: &str,
    max_chunk_secs: f32,
) -> Result<Vec<SpeechChunk>> {
    let raw = detect_speech_chunks(samples, vad_model_path)?;

    // 250ms padding to protect word boundaries
    let pad_samples = (SAMPLE_RATE as f32 * 0.25) as usize;
    let padded = pad_chunks(&raw, samples.len(), pad_samples);

    // Merge chunks separated by <200ms gap
    let merge_gap = (SAMPLE_RATE as f32 * 0.20) as usize;
    let merged = merge_close_chunks(&padded, merge_gap);

    // Split oversized chunks at low-energy points
    let final_chunks = split_long_chunks(samples, &merged, max_chunk_secs);

    Ok(final_chunks)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pad_extends_boundaries() {
        let chunks = vec![SpeechChunk {
            start_sample: 1000,
            end_sample: 5000,
        }];
        let padded = pad_chunks(&chunks, 10000, 500);
        assert_eq!(padded[0].start_sample, 500);
        assert_eq!(padded[0].end_sample, 5500);
    }

    #[test]
    fn pad_clamps_to_bounds() {
        let chunks = vec![SpeechChunk {
            start_sample: 100,
            end_sample: 9900,
        }];
        let padded = pad_chunks(&chunks, 10000, 500);
        assert_eq!(padded[0].start_sample, 0);
        assert_eq!(padded[0].end_sample, 10000);
    }

    #[test]
    fn merge_combines_close_chunks() {
        let chunks = vec![
            SpeechChunk {
                start_sample: 0,
                end_sample: 1000,
            },
            SpeechChunk {
                start_sample: 1100,
                end_sample: 2000,
            },
            SpeechChunk {
                start_sample: 5000,
                end_sample: 6000,
            },
        ];
        let merged = merge_close_chunks(&chunks, 200);
        assert_eq!(merged.len(), 2);
        assert_eq!(merged[0].start_sample, 0);
        assert_eq!(merged[0].end_sample, 2000);
        assert_eq!(merged[1].start_sample, 5000);
    }

    #[test]
    fn split_cuts_long_chunks() {
        let samples = vec![0.0f32; 80000]; // 5 seconds at 16kHz
        let chunks = vec![SpeechChunk {
            start_sample: 0,
            end_sample: 80000,
        }];
        let split = split_long_chunks(&samples, &chunks, 2.0);
        assert!(split.len() >= 2);
        for chunk in &split {
            assert!(chunk.duration_secs() <= 2.5); // some tolerance for cut point
        }
    }
}
