pub mod extract;
pub mod segment;
pub mod wav;

/// FFmpeg/FFprobe protocols permitted while opening caller-supplied local media.
///
/// `pipe` is required for FFmpeg's null output target. Network, data, crypto, and
/// nested URL protocols are intentionally excluded so playlists and containers
/// cannot turn a local-file operation into an outbound request.
pub(crate) const LOCAL_MEDIA_PROTOCOLS: &str = "file,pipe";
