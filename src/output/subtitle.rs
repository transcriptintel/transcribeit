use anyhow::Result;

use crate::transcriber::{Segment, Transcript};

use super::validate_subtitle_timing;

pub(crate) struct SubtitleLine<'a> {
    pub(crate) text: &'a str,
    pub(crate) speaker: Option<&'a str>,
}

pub(crate) struct SubtitleCue<'a> {
    pub(crate) start_ms: i64,
    pub(crate) end_ms: i64,
    pub(crate) lines: Vec<SubtitleLine<'a>>,
}

pub(crate) fn prepare_subtitle_cues(transcript: &Transcript) -> Result<Vec<SubtitleCue<'_>>> {
    validate_subtitle_timing(transcript)?;

    let positive_indices = transcript
        .segments
        .iter()
        .enumerate()
        .filter_map(|(index, segment)| (segment.end_ms > segment.start_ms).then_some(index))
        .collect::<Vec<_>>();

    if positive_indices.is_empty() {
        anyhow::ensure!(
            transcript.segments.is_empty(),
            "subtitle output requires at least one positive-duration segment"
        );
        return Ok(Vec::new());
    }

    let mut cue_for_segment = vec![None; transcript.segments.len()];
    let mut cues = Vec::with_capacity(positive_indices.len());
    for (cue_index, &segment_index) in positive_indices.iter().enumerate() {
        let segment = &transcript.segments[segment_index];
        cue_for_segment[segment_index] = Some(cue_index);
        cues.push(SubtitleCue {
            start_ms: segment.start_ms,
            end_ms: segment.end_ms,
            lines: Vec::new(),
        });
    }

    for (segment_index, segment) in transcript.segments.iter().enumerate() {
        if segment.end_ms == segment.start_ms && segment.text.trim().is_empty() {
            continue;
        }
        let cue_index = cue_for_segment[segment_index].unwrap_or_else(|| {
            nearest_positive_cue(
                segment_index,
                segment,
                &positive_indices,
                &transcript.segments,
            )
        });
        cues[cue_index].lines.push(SubtitleLine {
            text: segment.text.trim(),
            speaker: segment.speaker.as_deref(),
        });
    }

    Ok(cues)
}

fn nearest_positive_cue(
    segment_index: usize,
    segment: &Segment,
    positive_indices: &[usize],
    segments: &[Segment],
) -> usize {
    let next_position = positive_indices.partition_point(|&index| index < segment_index);
    let previous = next_position.checked_sub(1);
    let next = (next_position < positive_indices.len()).then_some(next_position);

    match (previous, next) {
        (Some(previous), Some(next)) => {
            let previous_distance =
                distance_to_segment(segment.start_ms, &segments[positive_indices[previous]]);
            let next_distance =
                distance_to_segment(segment.start_ms, &segments[positive_indices[next]]);
            if previous_distance <= next_distance {
                previous
            } else {
                next
            }
        }
        (Some(previous), None) => previous,
        (None, Some(next)) => next,
        (None, None) => unreachable!("positive_indices is known to be non-empty"),
    }
}

fn distance_to_segment(timestamp_ms: i64, segment: &Segment) -> i64 {
    if timestamp_ms < segment.start_ms {
        segment.start_ms.saturating_sub(timestamp_ms)
    } else if timestamp_ms > segment.end_ms {
        timestamp_ms.saturating_sub(segment.end_ms)
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::prepare_subtitle_cues;
    use crate::transcriber::{Segment, Transcript};

    #[test]
    fn zero_duration_text_uses_nearest_positive_cue_in_source_order() {
        let transcript = Transcript {
            segments: vec![
                Segment {
                    start_ms: 0,
                    end_ms: 0,
                    text: "leading".to_string(),
                    ..Default::default()
                },
                Segment {
                    start_ms: 0,
                    end_ms: 1_000,
                    text: "first".to_string(),
                    ..Default::default()
                },
                Segment {
                    start_ms: 1_000,
                    end_ms: 1_000,
                    text: "boundary".to_string(),
                    ..Default::default()
                },
                Segment {
                    start_ms: 2_000,
                    end_ms: 3_000,
                    text: "second".to_string(),
                    ..Default::default()
                },
            ],
            provider_metadata: None,
        };

        let cues = prepare_subtitle_cues(&transcript).expect("cues should be normalized");

        assert_eq!(cues.len(), 2);
        assert_eq!(
            cues[0]
                .lines
                .iter()
                .map(|line| line.text)
                .collect::<Vec<_>>(),
            ["leading", "first", "boundary"]
        );
        assert_eq!(cues[1].lines[0].text, "second");
    }
}
