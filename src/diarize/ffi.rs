//! Raw FFI bindings for sherpa-onnx speaker diarization C API.
#![allow(dead_code)]
//! These are not exposed by sherpa-onnx-sys 0.1.10 so we bind them directly.

use std::os::raw::{c_char, c_float, c_int};

#[repr(C)]
pub struct SherpaOnnxOfflineSpeakerSegmentationPyannoteModelConfig {
    pub model: *const c_char,
}

#[repr(C)]
pub struct SherpaOnnxOfflineSpeakerSegmentationModelConfig {
    pub pyannote: SherpaOnnxOfflineSpeakerSegmentationPyannoteModelConfig,
    pub num_threads: c_int,
    pub debug: c_int,
    pub provider: *const c_char,
}

#[repr(C)]
pub struct SherpaOnnxSpeakerEmbeddingExtractorConfig {
    pub model: *const c_char,
    pub num_threads: c_int,
    pub debug: c_int,
    pub provider: *const c_char,
}

#[repr(C)]
pub struct SherpaOnnxFastClusteringConfig {
    pub num_clusters: c_int,
    pub threshold: c_float,
}

#[repr(C)]
pub struct SherpaOnnxOfflineSpeakerDiarizationConfig {
    pub segmentation: SherpaOnnxOfflineSpeakerSegmentationModelConfig,
    pub embedding: SherpaOnnxSpeakerEmbeddingExtractorConfig,
    pub clustering: SherpaOnnxFastClusteringConfig,
    pub min_duration_on: c_float,
    pub min_duration_off: c_float,
}

#[repr(C)]
pub struct SherpaOnnxOfflineSpeakerDiarizationSegment {
    pub start: c_float,
    pub end: c_float,
    pub speaker: c_int,
}

// Opaque types
pub enum SherpaOnnxOfflineSpeakerDiarization {}
pub enum SherpaOnnxOfflineSpeakerDiarizationResult {}

unsafe extern "C" {
    pub fn SherpaOnnxCreateOfflineSpeakerDiarization(
        config: *const SherpaOnnxOfflineSpeakerDiarizationConfig,
    ) -> *const SherpaOnnxOfflineSpeakerDiarization;

    pub fn SherpaOnnxDestroyOfflineSpeakerDiarization(
        sd: *const SherpaOnnxOfflineSpeakerDiarization,
    );

    pub fn SherpaOnnxOfflineSpeakerDiarizationGetSampleRate(
        sd: *const SherpaOnnxOfflineSpeakerDiarization,
    ) -> c_int;

    pub fn SherpaOnnxOfflineSpeakerDiarizationProcess(
        sd: *const SherpaOnnxOfflineSpeakerDiarization,
        samples: *const c_float,
        n: c_int,
    ) -> *const SherpaOnnxOfflineSpeakerDiarizationResult;

    pub fn SherpaOnnxOfflineSpeakerDiarizationResultGetNumSpeakers(
        r: *const SherpaOnnxOfflineSpeakerDiarizationResult,
    ) -> c_int;

    pub fn SherpaOnnxOfflineSpeakerDiarizationResultGetNumSegments(
        r: *const SherpaOnnxOfflineSpeakerDiarizationResult,
    ) -> c_int;

    pub fn SherpaOnnxOfflineSpeakerDiarizationResultSortByStartTime(
        r: *const SherpaOnnxOfflineSpeakerDiarizationResult,
    ) -> *const SherpaOnnxOfflineSpeakerDiarizationSegment;

    pub fn SherpaOnnxOfflineSpeakerDiarizationDestroySegment(
        s: *const SherpaOnnxOfflineSpeakerDiarizationSegment,
    );

    pub fn SherpaOnnxOfflineSpeakerDiarizationDestroyResult(
        r: *const SherpaOnnxOfflineSpeakerDiarizationResult,
    );
}
