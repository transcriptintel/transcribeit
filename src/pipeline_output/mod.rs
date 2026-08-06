mod cache;
mod capability;
mod manifest;
mod quality;

use anyhow::{Context, Result};

use crate::output::{create_private_file, srt::write_srt, vtt::write_vtt};
use crate::pipeline::{OutputFormat, PipelineConfig};
use crate::transcriber::Transcript;

pub(crate) use manifest::write_manifest_output;

pub(crate) fn write_transcript_output(
    config: &PipelineConfig,
    transcript: &Transcript,
) -> Result<()> {
    match config.output_format {
        OutputFormat::Text => write_text(config, transcript)?,
        OutputFormat::Vtt => write_vtt_output(config, transcript)?,
        OutputFormat::Srt => write_srt_output(config, transcript)?,
    }
    Ok(())
}

fn write_text(config: &PipelineConfig, transcript: &Transcript) -> Result<()> {
    let text = transcript.text();
    if let Some(ref dir) = config.output_dir {
        std::fs::create_dir_all(dir)
            .with_context(|| format!("Failed to create output dir: {}", dir.display()))?;
        let stem = config
            .input
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy();
        let text_path = dir.join(format!("{stem}.txt"));
        let mut file = create_private_file(&text_path)?;
        std::io::Write::write_all(&mut file, text.as_bytes())
            .with_context(|| format!("Failed to write text output: {}", text_path.display()))?;
        eprintln!("Text written to {}", text_path.display());
    } else {
        println!("{text}");
    }
    Ok(())
}

fn write_vtt_output(config: &PipelineConfig, transcript: &Transcript) -> Result<()> {
    if let Some(ref dir) = config.output_dir {
        std::fs::create_dir_all(dir)
            .with_context(|| format!("Failed to create output dir: {}", dir.display()))?;
        let stem = config
            .input
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy();
        let vtt_path = dir.join(format!("{stem}.vtt"));
        let mut file = create_private_file(&vtt_path)?;
        write_vtt(transcript, &mut file)?;
        eprintln!("VTT written to {}", vtt_path.display());
    } else {
        let mut stdout = std::io::stdout();
        write_vtt(transcript, &mut stdout)?;
    }
    Ok(())
}

fn write_srt_output(config: &PipelineConfig, transcript: &Transcript) -> Result<()> {
    if let Some(ref dir) = config.output_dir {
        std::fs::create_dir_all(dir)
            .with_context(|| format!("Failed to create output dir: {}", dir.display()))?;
        let stem = config
            .input
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy();
        let srt_path = dir.join(format!("{stem}.srt"));
        let mut file = create_private_file(&srt_path)?;
        write_srt(transcript, &mut file)?;
        eprintln!("SRT written to {}", srt_path.display());
    } else {
        let mut stdout = std::io::stdout();
        write_srt(transcript, &mut stdout)?;
    }
    Ok(())
}
