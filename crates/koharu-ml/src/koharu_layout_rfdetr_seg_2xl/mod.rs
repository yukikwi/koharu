//! Manga layout instance segmentation with RF-DETR Seg 2XL.
//!
//! Checkpoints share one architecture and parameter tree; they differ only in
//! input resolution, which sizes the DINOv2 position embeddings:
//! https://huggingface.co/mayocream/koharu-layout-rfdetr-seg-2xl-1152/tree/aed55fdb8ca953c6bec33cf6ed6dd52a9b72bfa2
//! https://huggingface.co/mayocream/koharu-layout-highres/tree/68077eb09401a46e548190ff2786bd405113ea8b
//! RF-DETR upstream implementation:
//! https://github.com/roboflow/rf-detr/tree/4ab7c18729de9d02ffd0495795d0831b5630f01b

mod config;
mod model;
mod processor;

use anyhow::{Context, Result};
use image::DynamicImage;
use koharu_runtime::HuggingFaceFile;
use koharu_torch::Device;

use crate::backend::TryIntoDevice;

pub use self::{
    config::{KoharuLayoutRFDetrSeg2XLConfig, KoharuLayoutThresholds},
    processor::{
        KoharuLayoutDetection, KoharuLayoutDetections, KoharuLayoutMask,
        KoharuLayoutRFDetrImageProcessor,
    },
};

use self::model::Model;

/// A published KoharuLayout RF-DETR Seg 2XL checkpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KoharuLayoutCheckpoint {
    /// 1152 px input.
    Seg2XL1152,
    /// 1920 px input, fine-tuned from the 1152 checkpoint.
    HighRes,
}

impl KoharuLayoutCheckpoint {
    #[must_use]
    pub const fn repository(self) -> &'static str {
        match self {
            Self::Seg2XL1152 => "mayocream/koharu-layout-rfdetr-seg-2xl-1152",
            Self::HighRes => "mayocream/koharu-layout-highres",
        }
    }

    const fn files(self) -> (HuggingFaceFile<'static>, HuggingFaceFile<'static>) {
        let (revision, config) = match self {
            Self::Seg2XL1152 => (
                "aed55fdb8ca953c6bec33cf6ed6dd52a9b72bfa2",
                "inference_config.json",
            ),
            Self::HighRes => ("68077eb09401a46e548190ff2786bd405113ea8b", "config.json"),
        };
        let repository = self.repository();
        (
            HuggingFaceFile::pinned(repository, revision, config),
            HuggingFaceFile::pinned(repository, revision, "model.safetensors"),
        )
    }
}

#[derive(Debug)]
pub struct KoharuLayoutRFDetrSeg2XL {
    device: Device,
    model: Model,
    processor: KoharuLayoutRFDetrImageProcessor,
}

impl KoharuLayoutRFDetrSeg2XL {
    pub async fn load(device: crate::Device, checkpoint: KoharuLayoutCheckpoint) -> Result<Self> {
        let device: Device = device.try_into_device()?;
        let (config, weights) = checkpoint.files();
        let config_path = config
            .resolve()
            .await
            .context("failed to resolve KoharuLayout RF-DETR inference config")?;
        let weights_path = weights
            .resolve()
            .await
            .context("failed to resolve KoharuLayout RF-DETR weights")?;
        let config = KoharuLayoutRFDetrSeg2XLConfig::from_file(&config_path)?;
        let processor = KoharuLayoutRFDetrImageProcessor::new(&config)?;
        let mut model = Model::new(device, config.resolution);
        model
            .load(&weights_path)
            .with_context(|| format!("failed to load {}", weights_path.display()))?;
        Ok(Self {
            device,
            model,
            processor,
        })
    }

    pub fn inference(&self, image: &DynamicImage) -> Result<KoharuLayoutDetections> {
        self.inference_with_thresholds(image, self.processor.recommended_thresholds())
    }

    pub fn inference_with_thresholds(
        &self,
        image: &DynamicImage,
        thresholds: KoharuLayoutThresholds,
    ) -> Result<KoharuLayoutDetections> {
        koharu_torch::no_grad(|| {
            let pixel_values = self.processor.preprocess(image, self.device)?;
            let output = self.model.forward(&pixel_values);
            self.processor
                .postprocess(&output, image.width(), image.height(), thresholds)
        })
    }

    pub fn recommended_thresholds(&self) -> KoharuLayoutThresholds {
        self.processor.recommended_thresholds()
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use anyhow::Result;

    use super::{KoharuLayoutCheckpoint, KoharuLayoutRFDetrSeg2XL};

    #[tokio::test]
    #[ignore = "downloads the checkpoint and requires CUDA"]
    async fn checkpoint_matches_rfdetr_upstream_structured_output() -> Result<()> {
        crate::init().await?;
        let image = image::open(
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("benches/fixtures/object_detection/1.jpg"),
        )?;
        let model = KoharuLayoutRFDetrSeg2XL::load(
            crate::Device::cuda(0),
            KoharuLayoutCheckpoint::Seg2XL1152,
        )
        .await?;
        let result = model.inference(&image)?;

        // RF-DETR 4ab7c18, CUDA BF16, shape=(1152, 1152), antialias disabled.
        // CUDA kernels vary across LibTorch releases, so compare structured
        // geometry and mask area in addition to bounded confidence differences.
        let best = &result.detections[0];
        assert_eq!(best.label, "bubble");
        assert!((best.score - 0.968_856_2).abs() < 0.01);
        for (actual, expected) in best
            .bbox
            .into_iter()
            .zip([566.220_7, 550.019_5, 691.044_9, 724.043])
        {
            assert!((actual - expected).abs() < 6.0);
        }
        assert!(best.area.abs_diff(18_882) < 100);

        let lower_panel = result
            .detections
            .iter()
            .find(|detection| detection.label == "panel" && detection.bbox[1] > 700.0)
            .expect("lower page panel");
        let middle_panel = result
            .detections
            .iter()
            .find(|detection| {
                detection.label == "panel" && detection.bbox[1] > 500.0 && detection.bbox[1] < 700.0
            })
            .expect("middle page panel");
        for (actual, expected) in lower_panel
            .bbox
            .into_iter()
            .zip([69.179_69, 799.453_1, 700.820_3, 1006.171_9])
        {
            assert!((actual - expected).abs() < 7.0);
        }
        for (actual, expected) in middle_panel
            .bbox
            .into_iter()
            .zip([70.683_59, 550.546_9, 699.316_4, 782.578_1])
        {
            assert!((actual - expected).abs() < 3.0);
        }
        assert!(lower_panel.area.abs_diff(136_734) < 1_000);
        assert!(middle_panel.area.abs_diff(142_179) < 2_000);
        Ok(())
    }
}
