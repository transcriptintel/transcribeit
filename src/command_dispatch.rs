use anyhow::Result;

use crate::cli::{Command, ModelFormat, SetupComponent};
#[cfg(feature = "sherpa-onnx")]
use crate::models::download_onnx_model;
use crate::models::{download_model, list_models};
use crate::run_command;
use crate::setup::{
    print_setup_summary, setup_diarize, setup_models, setup_qwen3_asr, setup_sherpa_libs, setup_vad,
};

pub(crate) async fn dispatch(command: Command) -> Result<()> {
    match command {
        Command::Setup {
            component,
            output_dir,
            hf_token,
        } => execute_setup(component, output_dir, hf_token).await,
        Command::DownloadModel {
            model_size,
            format,
            output_dir,
            hf_token,
            vad,
            diarize,
        } => {
            match format {
                ModelFormat::Ggml => {
                    download_model(&model_size, output_dir.clone(), hf_token.as_deref()).await?;
                }
                ModelFormat::Onnx => {
                    #[cfg(feature = "sherpa-onnx")]
                    download_onnx_model(&model_size, output_dir.clone()).await?;
                    #[cfg(not(feature = "sherpa-onnx"))]
                    anyhow::bail!(
                        "ONNX model download requires the 'sherpa-onnx' feature. Build with: cargo build --features sherpa-onnx"
                    );
                }
            }
            if vad {
                setup_vad(output_dir.clone()).await?;
            }
            if diarize {
                setup_diarize(output_dir).await?;
            }
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
    let components = component.map_or_else(
        || {
            vec![
                SetupComponent::Models,
                SetupComponent::Vad,
                SetupComponent::Diarize,
                SetupComponent::SherpaLibs,
            ]
        },
        |component| vec![component],
    );
    let mut summary = Vec::new();
    for component in &components {
        let (name, status) = match component {
            SetupComponent::Models => (
                "models",
                setup_models(output_dir.clone(), hf_token.as_deref()).await?,
            ),
            SetupComponent::Vad => ("vad", setup_vad(output_dir.clone()).await?),
            SetupComponent::Diarize => ("diarize", setup_diarize(output_dir.clone()).await?),
            SetupComponent::SherpaLibs => {
                ("sherpa-libs", setup_sherpa_libs(output_dir.clone()).await?)
            }
            SetupComponent::Qwen3Asr => ("qwen3-asr", setup_qwen3_asr(output_dir.clone()).await?),
        };
        summary.push((name, status));
    }
    print_setup_summary(&summary, output_dir.as_deref());
    Ok(())
}
