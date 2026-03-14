use std::io::{Cursor, Read, Seek};

use anyhow::{Context, Result};
use hound::WavReader;

/// Read WAV bytes and return f32 samples normalized to [-1.0, 1.0].
///
/// Validates that the file is mono and 16kHz, bailing with an ffmpeg hint if not.
pub fn read_wav_bytes(data: &[u8]) -> Result<Vec<f32>> {
    let cursor = Cursor::new(data);
    let reader = WavReader::new(cursor).context("Failed to parse WAV bytes")?;
    read_wav_reader(reader)
}

fn read_wav_reader<R: Read + Seek>(reader: WavReader<R>) -> Result<Vec<f32>> {
    let spec = reader.spec();
    if spec.channels != 1 {
        anyhow::bail!(
            "Expected mono audio, got {} channels. Pre-process with: \
             ffmpeg -i input -ac 1 -ar 16000 output.wav",
            spec.channels
        );
    }
    if spec.sample_rate != 16000 {
        anyhow::bail!(
            "Expected 16kHz sample rate, got {}Hz. Pre-process with: \
             ffmpeg -i input -ac 1 -ar 16000 output.wav",
            spec.sample_rate
        );
    }

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let max_val = (1 << (spec.bits_per_sample - 1)) as f32;
            reader
                .into_samples::<i32>()
                .map(|s| s.map(|v| v as f32 / max_val))
                .collect::<std::result::Result<_, _>>()
                .context("Failed to read WAV samples")?
        }
        hound::SampleFormat::Float => reader
            .into_samples::<f32>()
            .collect::<std::result::Result<_, _>>()
            .context("Failed to read WAV samples")?,
    };

    Ok(samples)
}

/// Encode f32 PCM samples to a 16-bit mono 16kHz WAV in memory.
pub fn encode_wav(samples: &[f32]) -> Result<Vec<u8>> {
    let mut cursor = std::io::Cursor::new(Vec::new());
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 16000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer =
        hound::WavWriter::new(&mut cursor, spec).context("Failed to create WAV writer")?;

    for &sample in samples {
        let clamped = sample.clamp(-1.0, 1.0);
        let val = (clamped * i16::MAX as f32) as i16;
        writer
            .write_sample(val)
            .context("Failed to write WAV sample")?;
    }

    writer.finalize().context("Failed to finalize WAV")?;
    Ok(cursor.into_inner())
}
