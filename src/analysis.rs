use anyhow::Result;
use async_trait::async_trait;
use serde::Serialize;
use serde_json::Value;

use crate::transcriber::Transcript;

#[derive(Debug, Clone, Default)]
pub struct AnalysisConfig {
    pub summary: bool,
}

impl AnalysisConfig {
    pub fn is_enabled(&self) -> bool {
        self.summary
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct AnalysisResult {
    pub provider: String,
    pub model: String,
    pub schema_version: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary: Option<SummaryAnalysis>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_metadata: Option<Value>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SummaryAnalysis {
    pub short: String,
    pub detailed: String,
    pub key_points: Vec<String>,
    pub topics: Vec<String>,
    pub action_items: Vec<String>,
    pub questions: Vec<String>,
    pub follow_ups: Vec<String>,
}

#[async_trait]
pub trait TranscriptAnalyzer: Send + Sync {
    async fn analyze_transcript(
        &self,
        transcript: &Transcript,
        config: &AnalysisConfig,
    ) -> Result<AnalysisResult>;
}
