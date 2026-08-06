use clap::ValueEnum;

#[derive(Debug, Clone, ValueEnum)]
pub(crate) enum SetupComponent {
    /// Default STT models (GGML base)
    Models,
}
