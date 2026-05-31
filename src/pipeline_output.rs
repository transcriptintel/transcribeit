use anyhow::{Context, Result};

use crate::output::manifest::{
    InputInfo, Manifest, ProcessingConfig, SegmentInfo, Stats, write_manifest,
};
use crate::output::{srt::write_srt, vtt::write_vtt};
use crate::pipeline::{OutputFormat, PipelineConfig};
use crate::transcriber::Transcript;

pub(crate) fn write_outputs(
    config: &PipelineConfig,
    transcript: &Transcript,
    total_duration: f64,
    should_segment: bool,
    processing_time: f64,
) -> Result<()> {
    match config.output_format {
        OutputFormat::Text => write_text(config, transcript)?,
        OutputFormat::Vtt => write_vtt_output(config, transcript)?,
        OutputFormat::Srt => write_srt_output(config, transcript)?,
    }

    if let Some(ref dir) = config.output_dir {
        std::fs::create_dir_all(dir)?;
        let stem = config
            .input
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy();
        let manifest_path = dir.join(format!("{stem}.manifest.json"));
        let manifest = build_manifest(
            config,
            transcript,
            total_duration,
            should_segment,
            processing_time,
        );

        let mut file = std::fs::File::create(&manifest_path)?;
        write_manifest(&manifest, &mut file)?;
        eprintln!("Manifest written to {}", manifest_path.display());
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
        std::fs::write(&text_path, text.as_bytes())
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
        let mut file = std::fs::File::create(&vtt_path)?;
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
        let mut file = std::fs::File::create(&srt_path)?;
        write_srt(transcript, &mut file)?;
        eprintln!("SRT written to {}", srt_path.display());
    } else {
        let mut stdout = std::io::stdout();
        write_srt(transcript, &mut stdout)?;
    }
    Ok(())
}

fn build_manifest(
    config: &PipelineConfig,
    transcript: &Transcript,
    total_duration: f64,
    should_segment: bool,
    processing_time: f64,
) -> Manifest {
    Manifest {
        input: InputInfo {
            file: config.input.display().to_string(),
            duration_secs: total_duration,
        },
        config: ProcessingConfig {
            provider: config.provider_name.clone(),
            model: config.model_name.clone(),
            segmentation_enabled: should_segment,
            silence_threshold_db: config.silence_threshold,
            min_silence_duration_secs: config.min_silence_duration,
            output_format: format!("{:?}", config.output_format).to_lowercase(),
            language: config.language.clone(),
            normalized_audio: config.normalize_audio,
        },
        segments: transcript
            .segments
            .iter()
            .enumerate()
            .map(|(i, s)| SegmentInfo {
                index: i,
                start_secs: s.start_ms as f64 / 1000.0,
                end_secs: s.end_ms as f64 / 1000.0,
                text: s.text.trim().to_string(),
                speaker: s.speaker.clone(),
            })
            .collect(),
        stats: Stats {
            total_duration_secs: total_duration,
            total_segments: transcript.segments.len(),
            total_characters: transcript.segments.iter().map(|s| s.text.len()).sum(),
            processing_time_secs: processing_time,
        },
    }
}
