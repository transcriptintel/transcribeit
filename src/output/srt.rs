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
    for (i, segment) in transcript.segments.iter().enumerate() {
        writeln!(writer, "{}", i + 1)?;
        writeln!(
            writer,
            "{} --> {}",
            format_timestamp(segment.start_ms),
            format_timestamp(segment.end_ms)
        )?;
        writeln!(writer, "{}", segment.text.trim())?;
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
            }],
        };

        let mut out = Cursor::new(Vec::new());
        write_srt(&transcript, &mut out).expect("writing srt should succeed");
        let result = String::from_utf8(out.into_inner()).expect("srt output should be UTF-8");

        assert!(result.contains("1"));
        assert!(result.contains("00:00:00,000 --> 00:00:01,234"));
        assert!(result.contains("Hello"));
    }
}
