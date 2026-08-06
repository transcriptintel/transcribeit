use anyhow::Result;

use crate::cli::{Command, SetupComponent};
use crate::models::{download_model, list_models};
use crate::run_command;
use crate::setup::{print_setup_summary, setup_models};

pub(crate) async fn dispatch(command: Command) -> Result<()> {
    match command {
        Command::Setup {
            component,
            output_dir,
            hf_token,
        } => execute_setup(component, output_dir, hf_token).await,
        Command::DownloadModel {
            model_size,
            output_dir,
            hf_token,
        } => {
            download_model(&model_size, output_dir, hf_token.as_deref()).await?;
            Ok(())
        }
        Command::ListModels { dir } => list_models(dir),
        command @ Command::Run { .. } => run_command::execute(command).await,
    }
}

async fn execute_setup(
    component: Option<SetupComponent>,
    output_dir: Option<std::path::PathBuf>,
    hf_token: Option<String>,
) -> Result<()> {
    let components = component.map_or_else(|| vec![SetupComponent::Models], |value| vec![value]);
    let mut summary = Vec::new();
    for component in &components {
        let (name, status) = match component {
            SetupComponent::Models => (
                "models",
                setup_models(output_dir.clone(), hf_token.as_deref()).await?,
            ),
        };
        summary.push((name, status));
    }
    print_setup_summary(&summary, output_dir.as_deref());
    Ok(())
}
