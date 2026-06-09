use super::{AudioSegment, SilenceInterval, compute_segments};

#[test]
fn no_silence_uses_fixed_splits_when_longer_than_limit() {
    let segments = compute_segments(&[], 30.0, 10.0);

    assert_eq!(segments.len(), 3);
    assert_eq!(segments[0].index, 0);
    assert_eq!(segments[0].start_secs, 0.0);
    assert_eq!(segments[0].end_secs, 10.0);
    assert_eq!(segments[1].index, 1);
    assert_eq!(segments[1].start_secs, 10.0);
    assert_eq!(segments[1].end_secs, 20.0);
    assert_eq!(segments[2].index, 2);
    assert_eq!(segments[2].start_secs, 20.0);
    assert_eq!(segments[2].end_secs, 30.0);
}

#[test]
fn no_silence_keeps_single_segment_when_short_enough() {
    let segments = compute_segments(&[], 8.0, 10.0);

    assert_eq!(segments.len(), 1);
    assert_eq!(segments[0].index, 0);
    assert_eq!(segments[0].start_secs, 0.0);
    assert_eq!(segments[0].end_secs, 8.0);
}

#[test]
fn silence_midpoints_guide_splitting_points() {
    let silences = vec![
        SilenceInterval {
            start_secs: 2.0,
            end_secs: 4.0,
        },
        SilenceInterval {
            start_secs: 17.0,
            end_secs: 19.0,
        },
    ];

    let segments = compute_segments(&silences, 35.0, 20.0);

    assert_eq!(segments.len(), 3);
    assert_eq!(segments[0], segment(0, 0.0, 3.0));
    assert_eq!(segments[1], segment(1, 3.0, 18.0));
    assert_eq!(segments[2], segment(2, 18.0, 35.0));
}

#[test]
fn very_short_splits_are_skipped() {
    let silences = vec![
        SilenceInterval {
            start_secs: 1.0,
            end_secs: 2.0,
        },
        SilenceInterval {
            start_secs: 2.8,
            end_secs: 3.5,
        },
    ];

    let segments = compute_segments(&silences, 40.0, 30.0);

    assert_eq!(segments.len(), 1);
    assert_eq!(segments[0], segment(0, 0.0, 40.0));
}

#[test]
fn short_split_is_kept_when_followed_by_long_gap() {
    let silences = vec![
        SilenceInterval {
            start_secs: 1.0,
            end_secs: 2.0,
        },
        SilenceInterval {
            start_secs: 20.0,
            end_secs: 21.5,
        },
    ];

    let segments = compute_segments(&silences, 40.0, 20.0);

    assert_eq!(segments.len(), 3);
    assert_eq!(segments[0], segment(0, 0.0, 1.5));
    assert_eq!(segments[1], segment(1, 1.5, 20.75));
    assert_eq!(segments[2], segment(2, 20.75, 40.0));
}

#[test]
fn consecutive_short_splits_are_merged() {
    let silences = vec![
        SilenceInterval {
            start_secs: 1.0,
            end_secs: 2.0,
        },
        SilenceInterval {
            start_secs: 2.8,
            end_secs: 3.5,
        },
        SilenceInterval {
            start_secs: 3.8,
            end_secs: 4.3,
        },
    ];

    let segments = compute_segments(&silences, 40.0, 20.0);

    assert_eq!(segments.len(), 1);
    assert_eq!(segments[0], segment(0, 0.0, 40.0));
}

#[test]
fn split_exactly_at_min_segment_threshold_is_kept() {
    let silences = vec![SilenceInterval {
        start_secs: 0.0,
        end_secs: 10.0,
    }];

    let segments = compute_segments(&silences, 15.0, 20.0);

    assert_eq!(segments.len(), 1);
    assert_eq!(segments[0], segment(0, 0.0, 15.0));
}

#[test]
fn max_segment_limit_overrides_sparse_splits() {
    let silences = vec![SilenceInterval {
        start_secs: 29.0,
        end_secs: 31.0,
    }];

    let segments = compute_segments(&silences, 40.0, 10.0);

    assert_eq!(segments.len(), 4);
    assert_eq!(segments[0], segment(0, 0.0, 10.0));
    assert_eq!(segments[1], segment(1, 10.0, 20.0));
    assert_eq!(segments[2], segment(2, 20.0, 30.0));
    assert_eq!(segments[3], segment(3, 30.0, 40.0));
}

fn segment(index: usize, start_secs: f64, end_secs: f64) -> AudioSegment {
    AudioSegment {
        index,
        start_secs,
        end_secs,
    }
}
