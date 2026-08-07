use std::io::Write;

use super::{AudioSegment, SilenceInterval, compute_segments, detect_silence, get_duration};

#[test]
fn no_silence_uses_fixed_splits_when_longer_than_limit() {
    let segments = compute_segments(&[], 30.0, 10.0).unwrap();

    assert_eq!(
        segments,
        vec![
            segment(0, 0.0, 10.0),
            segment(1, 10.0, 20.0),
            segment(2, 20.0, 30.0),
        ]
    );
}

#[test]
fn no_silence_keeps_single_segment_when_short_enough() {
    let segments = compute_segments(&[], 8.0, 10.0).unwrap();

    assert_eq!(segments, vec![segment(0, 0.0, 8.0)]);
}

#[test]
fn latest_usable_silence_before_hard_limit_is_preferred() {
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

    let segments = compute_segments(&silences, 35.0, 20.0).unwrap();

    assert_eq!(
        segments,
        vec![segment(0, 0.0, 18.0), segment(1, 18.0, 35.0)]
    );
}

#[test]
fn early_silences_do_not_create_tiny_segments() {
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

    let segments = compute_segments(&silences, 40.0, 30.0).unwrap();

    assert_eq!(
        segments,
        vec![segment(0, 0.0, 30.0), segment(1, 30.0, 40.0)]
    );
}

#[test]
fn short_final_tail_is_not_merged_past_hard_limit() {
    let segments = compute_segments(&[], 32.0, 30.0).unwrap();

    assert_eq!(
        segments,
        vec![segment(0, 0.0, 30.0), segment(1, 30.0, 32.0)]
    );
}

#[test]
fn sparse_silence_never_overrides_hard_limit() {
    let silences = vec![SilenceInterval {
        start_secs: 29.0,
        end_secs: 31.0,
    }];

    let segments = compute_segments(&silences, 40.0, 10.0).unwrap();

    assert_eq!(segments.len(), 4);
    assert!(segments.iter().all(|segment| {
        segment.end_secs > segment.start_secs && segment.end_secs - segment.start_secs <= 10.0
    }));
}

#[test]
fn invalid_durations_are_rejected_without_looping() {
    for maximum in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(compute_segments(&[], 30.0, maximum).is_err());
    }
    assert!(compute_segments(&[], 1.0, f64::MIN_POSITIVE).is_err());
    for duration in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(compute_segments(&[], duration, 10.0).is_err());
    }
}

#[tokio::test]
async fn silence_detection_fails_for_invalid_media() {
    crate::audio::extract::check_ffmpeg().expect("FFmpeg is required for this integration test");
    let mut input = tempfile::Builder::new().suffix(".wav").tempfile().unwrap();
    input.write_all(b"not a wave file").unwrap();

    let error = detect_silence(input.path(), -30.0, 0.5)
        .await
        .expect_err("invalid media must not be treated as having no silence");

    assert!(format!("{error:#}").contains("ffmpeg silencedetect exited with status"));
}

#[tokio::test]
async fn local_playlist_cannot_open_network_protocols() {
    crate::audio::extract::check_ffmpeg().expect("FFmpeg is required for this integration test");
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let mut playlist = tempfile::Builder::new().suffix(".m3u8").tempfile().unwrap();
    writeln!(
        playlist,
        "#EXTM3U\n#EXT-X-TARGETDURATION:1\n#EXTINF:1,\nhttp://{address}/audio.wav\n#EXT-X-ENDLIST"
    )
    .unwrap();

    let error = get_duration(playlist.path())
        .await
        .expect_err("network-backed playlist must be rejected");
    assert!(format!("{error:#}").contains("ffprobe failed"));
    assert!(
        tokio::time::timeout(std::time::Duration::from_millis(250), listener.accept())
            .await
            .is_err(),
        "FFprobe attempted an outbound connection from a local playlist"
    );
}

fn segment(index: usize, start_secs: f64, end_secs: f64) -> AudioSegment {
    AudioSegment {
        index,
        start_secs,
        end_secs,
    }
}
