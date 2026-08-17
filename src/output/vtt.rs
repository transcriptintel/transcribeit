use crate::output::prepare_subtitle_cues;
use crate::transcriber::Transcript;
use anyhow::Result;
use std::io::Write;

/// Format milliseconds as VTT timestamp: HH:MM:SS.mmm
fn format_timestamp(ms: i64) -> String {
    let ms = ms.max(0);
    let total_secs = ms / 1000;
    let millis = ms % 1000;
    let hours = total_secs / 3600;
    let minutes = (total_secs % 3600) / 60;
    let seconds = total_secs % 60;
    format!("{:02}:{:02}:{:02}.{:03}", hours, minutes, seconds, millis)
}

/// Write a Transcript as WebVTT to any writer
pub fn write_vtt(transcript: &Transcript, writer: &mut impl Write) -> Result<()> {
    let cues = prepare_subtitle_cues(transcript)?;
    writeln!(writer, "WEBVTT")?;
    writeln!(writer)?;

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
                write!(writer, "<v {}>", speaker)?;
            }
            writeln!(writer, "{}", line.text)?;
        }
        writeln!(writer)?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::write_vtt;
    use crate::transcriber::{Segment, Transcript};
    use std::io::Cursor;

    #[test]
    fn write_vtt_outputs_header_and_timestamps() {
        let transcript = Transcript {
            segments: vec![
                Segment {
                    start_ms: 0,
                    end_ms: 1234,
                    text: " Hello ".to_string(),
                    speaker: None,
                    ..Default::default()
                },
                Segment {
                    start_ms: 5_000,
                    end_ms: 6_100,
                    text: "world".to_string(),
                    speaker: None,
                    ..Default::default()
                },
            ],
            provider_metadata: None,
        };

        let mut out = Cursor::new(Vec::new());
        write_vtt(&transcript, &mut out).expect("writing vtt should succeed");
        let result = String::from_utf8(out.into_inner()).expect("vtt output should be UTF-8");

        assert!(result.contains("WEBVTT"));
        assert!(result.contains("1\n00:00:00.000 --> 00:00:01.234"));
        assert!(result.contains("2\n00:00:05.000 --> 00:00:06.100"));
        assert!(result.contains("Hello"));
        assert!(result.contains("world"));
    }

    #[test]
    fn write_vtt_numbers_cues_from_one() {
        let transcript = Transcript {
            segments: vec![
                Segment {
                    start_ms: 0,
                    end_ms: 10,
                    text: "A".to_string(),
                    speaker: None,
                    ..Default::default()
                },
                Segment {
                    start_ms: 10,
                    end_ms: 20,
                    text: "B".to_string(),
                    speaker: None,
                    ..Default::default()
                },
                Segment {
                    start_ms: 20,
                    end_ms: 30,
                    text: "C".to_string(),
                    speaker: None,
                    ..Default::default()
                },
            ],
            provider_metadata: None,
        };

        let mut out = Cursor::new(Vec::new());
        write_vtt(&transcript, &mut out).unwrap();
        let out = String::from_utf8(out.into_inner()).unwrap();

        assert!(out.contains("\n1\n"));
        assert!(out.contains("\n2\n"));
        assert!(out.contains("\n3\n"));
        assert!(!out.contains(" 0\n"));
    }

    #[test]
    fn write_vtt_folds_zero_duration_text_and_preserves_speaker() {
        let transcript = Transcript {
            segments: vec![
                Segment {
                    start_ms: 55_000,
                    end_ms: 57_040,
                    text: "regular cue".to_string(),
                    ..Default::default()
                },
                Segment {
                    start_ms: 57_040,
                    end_ms: 57_040,
                    text: "boundary text".to_string(),
                    speaker: Some("Speaker 2".to_string()),
                    ..Default::default()
                },
            ],
            provider_metadata: None,
        };

        let mut out = Cursor::new(Vec::new());
        write_vtt(&transcript, &mut out).expect("zero-duration text should be folded");
        let out = String::from_utf8(out.into_inner()).unwrap();

        assert_eq!(out.matches(" --> ").count(), 1);
        assert!(out.contains("00:00:55.000 --> 00:00:57.040"));
        assert!(out.contains("regular cue\n<v Speaker 2>boundary text"));
    }

    #[test]
    fn write_vtt_rejects_non_monotonic_segments() {
        let transcript = Transcript {
            segments: vec![
                Segment {
                    start_ms: 1_000,
                    end_ms: 2_000,
                    text: "first".to_string(),
                    ..Default::default()
                },
                Segment {
                    start_ms: 500,
                    end_ms: 1_500,
                    text: "second".to_string(),
                    ..Default::default()
                },
            ],
            provider_metadata: None,
        };

        assert!(write_vtt(&transcript, &mut Cursor::new(Vec::new())).is_err());
    }
}
