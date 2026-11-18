mod config;
mod model;
mod processor;

use anyhow::{Context, Result};
use image::DynamicImage;
use koharu_torch::Device;

use crate::backend::TryIntoDevice;

use self::{
    config::MangaOcrArConfig,
    model::Model,
    processor::{ImageProcessor, Tokenizer},
};

crate::model_repository!("muscgab/manga-ocr-ar-experimental" @ "20208154a8d34dc6e5fc13ba81bc35f0a2f1a26e" {
    CONFIG = "config.json",
    WEIGHTS = "model.safetensors",
    VOCABULARY = "vocab.json",
});

/// Crop-level Japanese manga OCR.
///
/// Port of the experimental autoregressive Manga OCR: a MonkeyOCRv2-B vision
/// tower over an aspect-preserving 14x14 patch grid, depth-fused into a
/// single-layer character decoder with greedy decoding.
#[derive(Debug)]
pub struct MangaOcrAr {
    model: Model,
    processor: ImageProcessor,
    tokenizer: Tokenizer,
}

impl MangaOcrAr {
    pub async fn load(device: crate::Device) -> Result<Self> {
        let device: Device = device.try_into_device()?;
        let (config_path, weights_path, vocabulary_path) = tokio::try_join!(
            async {
                CONFIG
                    .resolve()
                    .await
                    .context("failed to resolve Manga OCR AR config")
            },
            async {
                WEIGHTS
                    .resolve()
                    .await
                    .context("failed to resolve Manga OCR AR weights")
            },
            async {
                VOCABULARY
                    .resolve()
                    .await
                    .context("failed to resolve Manga OCR AR vocabulary")
            },
        )?;

        let config = MangaOcrArConfig::from_file(&config_path)
            .with_context(|| format!("failed to read {}", config_path.display()))?;
        let tokenizer = Tokenizer::from_file(&vocabulary_path)
            .with_context(|| format!("failed to read {}", vocabulary_path.display()))?;

        // The embedding size follows the vocabulary, so a mismatched vocabulary
        // fails as a checkpoint shape error during loading.
        let mut model = Model::new(&config, tokenizer.len() as i64, device)?;
        model
            .load(&weights_path)
            .with_context(|| format!("failed to load {}", weights_path.display()))?;

        Ok(Self {
            model,
            processor: ImageProcessor::new(config.patch_budget, device),
            tokenizer,
        })
    }

    pub fn inference(&self, image: &DynamicImage) -> Result<String> {
        koharu_torch::no_grad(|| {
            let input = self.processor.preprocess(image)?;
            let token_ids = self.model.generate(&input.patches, input.grid)?;
            Ok(self.tokenizer.decode(&token_ids))
        })
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::MangaOcrAr;

    #[tokio::test]
    #[ignore = "downloads the checkpoint and requires the LibTorch runtime"]
    async fn checkpoint_matches_upstream_character_sequence() -> anyhow::Result<()> {
        crate::init().await?;
        let model = MangaOcrAr::load(crate::Device::cpu()).await?;
        let input =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("benches/fixtures/ocr/title.png");
        let text = model.inference(&image::open(input)?)?;
        // Verified against the pinned upstream `infer.py` on CPU fp32 with the
        // identical fixture crop.
        assert_eq!(text, "対策委員会です!");
        Ok(())
    }
}
