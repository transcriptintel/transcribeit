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
    writeln!(writer, "WEBVTT")?;
    writeln!(writer)?;

    for (i, segment) in transcript.segments.iter().enumerate() {
        writeln!(writer, "{}", i + 1)?;
        writeln!(
            writer,
            "{} --> {}",
            format_timestamp(segment.start_ms),
            format_timestamp(segment.end_ms)
        )?;
        if let Some(ref spk) = segment.speaker {
            write!(writer, "<v {}>", spk)?;
        }
        writeln!(writer, "{}", segment.text.trim())?;
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
                },
                Segment {
                    start_ms: 5_000,
                    end_ms: 6_100,
                    text: "world".to_string(),
                    speaker: None,
                },
            ],
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
                },
                Segment {
                    start_ms: 10,
                    end_ms: 20,
                    text: "B".to_string(),
                    speaker: None,
                },
                Segment {
                    start_ms: 20,
                    end_ms: 30,
                    text: "C".to_string(),
                    speaker: None,
                },
            ],
        };

        let mut out = Cursor::new(Vec::new());
        write_vtt(&transcript, &mut out).unwrap();
        let out = String::from_utf8(out.into_inner()).unwrap();

        assert!(out.contains("\n1\n"));
        assert!(out.contains("\n2\n"));
        assert!(out.contains("\n3\n"));
        assert!(!out.contains(" 0\n"));
    }
}
