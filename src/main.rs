mod analysis;
mod artifacts;
mod audio;
mod batch;
mod cli;
mod command_dispatch;
mod credentials;
mod engines;
mod input;
mod models;
mod output;
mod pipeline;
mod pipeline_merge;
mod pipeline_output;
mod provider_factory;
mod run_command;
mod setup;
mod storage;
mod transcriber;

use anyhow::Result;
use clap::Parser;

use crate::cli::Cli;

#[tokio::main]
async fn main() -> Result<()> {
    let arguments = std::env::args_os().collect::<Vec<_>>();
    let help_or_version_requested = arguments.get(1).is_some_and(|argument| argument == "help")
        || arguments.iter().any(|argument| {
            matches!(
                argument.to_str(),
                Some("-h" | "--help" | "-V" | "--version")
            )
        });
    if !help_or_version_requested {
        dotenvy::dotenv().ok();
    }

    command_dispatch::dispatch(Cli::parse().command).await
}
