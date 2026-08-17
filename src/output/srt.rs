use crate::output::prepare_subtitle_cues;
use crate::transcriber::Transcript;
use anyhow::Result;
use std::io::Write;

/// Format milliseconds as SRT timestamp: HH:MM:SS,mmm
fn format_timestamp(ms: i64) -> String {
    let ms = ms.max(0);
    let total_secs = ms / 1000;
    let millis = ms % 1000;
    let hours = total_secs / 3600;
    let minutes = (total_secs % 3600) / 60;
    let seconds = total_secs % 60;
    format!("{:02}:{:02}:{:02},{:03}", hours, minutes, seconds, millis)
}

/// Write a Transcript as SRT to any writer
pub fn write_srt(transcript: &Transcript, writer: &mut impl Write) -> Result<()> {
    let cues = prepare_subtitle_cues(transcript)?;
    for (i, cue) in cues.iter().enumerate() {
        writeln!(writer, "{}", i + 1)?;
        writeln!(
            writer,
            "{} --> {}",
            format_timestamp(cue.start_ms),
            format_timestamp(cue.end_ms)
        )?;
        for line in &cue.lines {
            if let Some(speaker) = line.speaker {
                writeln!(writer, "[{}] {}", speaker, line.text)?;
            } else {
                writeln!(writer, "{}", line.text)?;
            }
        }
        writeln!(writer)?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::write_srt;
    use crate::transcriber::{Segment, Transcript};
    use std::io::Cursor;

    #[test]
    fn write_srt_outputs_timestamps_with_commas() {
        let transcript = Transcript {
            segments: vec![Segment {
                start_ms: 0,
                end_ms: 1234,
                text: " Hello ".to_string(),
                speaker: None,
                ..Default::default()
            }],
            provider_metadata: None,
        };

        let mut out = Cursor::new(Vec::new());
        write_srt(&transcript, &mut out).expect("writing srt should succeed");
        let result = String::from_utf8(out.into_inner()).expect("srt output should be UTF-8");

        assert!(result.contains("1"));
        assert!(result.contains("00:00:00,000 --> 00:00:01,234"));
        assert!(result.contains("Hello"));
    }

    #[test]
    fn write_srt_folds_zero_duration_text_and_preserves_speaker() {
        let transcript = Transcript {
            segments: vec![
                Segment {
                    start_ms: 0,
                    end_ms: 1_000,
                    text: "first".to_string(),
                    ..Default::default()
                },
                Segment {
                    start_ms: 1_000,
                    end_ms: 1_000,
                    text: "second".to_string(),
                    speaker: Some("Speaker 2".to_string()),
                    ..Default::default()
                },
            ],
            provider_metadata: None,
        };

        let mut out = Cursor::new(Vec::new());
        write_srt(&transcript, &mut out).expect("zero-duration text should be folded");
        let out = String::from_utf8(out.into_inner()).unwrap();

        assert_eq!(out.matches(" --> ").count(), 1);
        assert!(out.contains("first\n[Speaker 2] second"));
    }

    #[test]
    fn write_srt_rejects_invalid_timing() {
        for (start_ms, end_ms) in [(-1, 100), (100, 100), (200, 100)] {
            let transcript = Transcript {
                segments: vec![Segment {
                    start_ms,
                    end_ms,
                    text: "invalid".to_string(),
                    ..Default::default()
                }],
                provider_metadata: None,
            };
            assert!(write_srt(&transcript, &mut Cursor::new(Vec::new())).is_err());
        }
    }
}
