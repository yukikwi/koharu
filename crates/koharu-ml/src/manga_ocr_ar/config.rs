//! Manga OCR AR configuration.
//!
//! Upstream configuration consumer (`Model.__init__` and `MangaOCR.__init__`):
//! https://huggingface.co/muscgab/manga-ocr-ar-experimental/blob/20208154a8d34dc6e5fc13ba81bc35f0a2f1a26e/infer.py

use std::{fs, path::Path};

use anyhow::{Result, ensure};
use serde::Deserialize;

const MODEL_TYPE: &str = "manga-ocr-ar";

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct MangaOcrArConfig {
    model_type: String,
    /// Maximum number of 14x14 patches in the resized crop.
    pub(crate) patch_budget: i64,
    /// Maximum number of generated characters before the decoder's EOS slot.
    pub(crate) max_length: i64,
    pub(crate) encoder: EncoderConfig,
    pub(crate) decoder: DecoderConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct EncoderConfig {
    pub(crate) dim: i64,
    pub(crate) heads: i64,
    pub(crate) ffn: i64,
    pub(crate) layers: usize,
    pub(crate) rms_eps: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct DecoderConfig {
    pub(crate) dim: i64,
    pub(crate) heads: i64,
    pub(crate) ffn: i64,
    pub(crate) layers: usize,
}

impl MangaOcrArConfig {
    pub(crate) fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let config: Self = serde_json::from_str(&fs::read_to_string(path)?)?;
        ensure!(
            config.model_type == MODEL_TYPE,
            "expected a {MODEL_TYPE} checkpoint, found {}",
            config.model_type
        );
        ensure!(config.patch_budget > 0, "patch budget must be positive");
        ensure!(config.max_length > 0, "maximum length must be positive");
        ensure!(
            config.encoder.layers > 0 && config.decoder.layers > 0,
            "encoder and decoder must have at least one layer"
        );
        ensure!(
            config.encoder.dim % config.encoder.heads == 0
                && config.decoder.dim % config.decoder.heads == 0,
            "model width must be divisible by the attention head count"
        );
        // The encoder rotary embedding splits each head into row and column halves.
        ensure!(
            (config.encoder.dim / config.encoder.heads) % 4 == 0,
            "encoder head size must be divisible by four for 2D RoPE"
        );
        Ok(config)
    }
}
