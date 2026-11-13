//! MiniMax-H3-Image masked reference editing.
//!
//! Sampling defaults follow the upstream stable-diffusion.cpp recipe:
//! https://github.com/leejet/stable-diffusion.cpp/blob/b9d7176bf88fe465576e0aafbe47ba253ffcfc2c/examples/server/recipes/minimax-h3-image.json
//! The source crop is both the init image, so unmasked latents are preserved,
//! and `<Picture 1>`, so the Qwen3-VL encoder sees the content to reconstruct:
//! https://github.com/leejet/stable-diffusion.cpp/blob/b9d7176bf88fe465576e0aafbe47ba253ffcfc2c/src/pipeline/image.cpp#L511-L524

mod model;

use std::time::Instant;

use anyhow::{Context, Result, ensure};
use fast_image_resize::{FilterType, ResizeAlg, ResizeOptions, Resizer};
use image::{DynamicImage, GrayImage, Luma, RgbImage};
use koharu_diffusion::{
    GuidanceParams, ImageGenerationParams, SampleMethod, SampleParams, TilingParams,
};

use self::model::{Model, ModelPaths};

// The Abiray/MiniMax-H3-Image-GGUF conversion flattens `video_patch_proj`,
// `condition_proj`, `final_layer.video_out`, and the AdaLN tables to 1D and
// relies on ComfyUI-GGUF's `comfy.gguf.orig_shape` metadata to restore them.
// stable-diffusion.cpp does not read that metadata and rejects the shapes, so
// this uses the same Q4_K_M quantization converted by stable-diffusion.cpp.
crate::model_repository!("Hearthscale/MiniMax-H3-Image-GGUF" @ "2cd99b1e7307385b61def10c5caf0cbbef3aaaa2" {
    TRANSFORMER_WEIGHTS = "minimax_h3_image_pruned-Q4_K_M.gguf"
});
crate::model_repository!("leejet/MiniMax-H3-GGUF" @ "d9c4c6312b4728a68a15a35626d84775a6523783" {
    TEXT_ENCODER_WEIGHTS = "qwen3vl_32b_minimax_h3-Q4_K_M.gguf"
});
crate::model_repository!("Comfy-Org/MiniMax-H3-Image" @ "0bb149f51e3156ed4b90dd45d5620346bfe1df8f" {
    VAE_WEIGHTS = "vae/minimax_h3_image_vae_fp16.safetensors"
});

pub const DEFAULT_PROMPT: &str = "Edit <Picture 1>. Remove all text and lettering, and reconstruct the hidden artwork and background in the same style.";

/// The checkpoint is calibrated for 1 MP; upstream aligns sizes to 32 pixels.
const TARGET_AREA: f64 = 1024.0 * 1024.0;
const SIZE_MULTIPLE: u32 = 32;
const PDD_SIGMAS: [f32; 7] = [
    1.0,
    0.967_523_7,
    0.887_548_5,
    0.75,
    0.532_773_9,
    0.232_008_87,
    0.0,
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MiniMaxH3ImageInpaintOptions {
    pub seed: i64,
}

impl Default for MiniMaxH3ImageInpaintOptions {
    fn default() -> Self {
        Self { seed: -1 }
    }
}

#[derive(Debug)]
pub struct MiniMaxH3ImageInpaint {
    model: Model,
}

impl MiniMaxH3ImageInpaint {
    pub async fn load(device: crate::Device) -> Result<Self> {
        let started = Instant::now();
        let (transformer, text_encoder, vae) = tokio::try_join!(
            TRANSFORMER_WEIGHTS.resolve(),
            TEXT_ENCODER_WEIGHTS.resolve(),
            VAE_WEIGHTS.resolve(),
        )
        .context("failed to resolve MiniMax-H3-Image model assets")?;
        tracing::info!(
            transformer = %transformer.display(),
            text_encoder = %text_encoder.display(),
            vae = %vae.display(),
            elapsed_ms = started.elapsed().as_millis(),
            "resolved MiniMax-H3-Image assets"
        );
        let model = Model::new(
            &device,
            ModelPaths {
                transformer,
                text_encoder,
                vae,
            },
        )?;
        Ok(Self { model })
    }

    pub fn inference(
        &self,
        prompt: &str,
        image: &DynamicImage,
        mask_image: &DynamicImage,
        options: &MiniMaxH3ImageInpaintOptions,
    ) -> Result<DynamicImage> {
        ensure!(
            image.width() == mask_image.width() && image.height() == mask_image.height(),
            "image/mask dimensions differ: image={}x{}, mask={}x{}",
            image.width(),
            image.height(),
            mask_image.width(),
            mask_image.height()
        );
        ensure!(
            !prompt.contains('\0'),
            "prompt contains an interior NUL byte"
        );

        let source = image.to_rgb8();
        let source_mask = mask_image.to_luma8();
        let (width, height) = processing_size(source.width(), source.height());
        let init_image = resize_rgb(&source, width, height);
        let mut mask = resize_gray(&source_mask, width, height);
        for pixel in mask.pixels_mut() {
            *pixel = Luma([if pixel.0[0] < 128 { 0 } else { 255 }]);
        }
        let masked = mask.pixels().filter(|pixel| pixel.0[0] != 0).count();
        tracing::info!(
            input = %format!("{}x{}", source.width(), source.height()),
            processing = %format!("{width}x{height}"),
            mask_coverage = %format!("{:.2}%", masked as f64 * 100.0 / f64::from(width * height)),
            seed = options.seed,
            "MiniMax-H3-Image inpainting tile"
        );

        let generated = self
            .model
            .forward(&ImageGenerationParams {
                prompt: prompt.to_owned(),
                width: i32::try_from(width)?,
                height: i32::try_from(height)?,
                init_image: Some(init_image.clone()),
                reference_images: vec![init_image],
                mask_image: Some(mask),
                sample: SampleParams {
                    guidance: GuidanceParams {
                        text_cfg: 1.0,
                        ..GuidanceParams::default()
                    },
                    sample_method: SampleMethod::Euler,
                    sample_steps: 6,
                    custom_sigmas: PDD_SIGMAS.to_vec(),
                    ..SampleParams::default()
                },
                strength: 1.0,
                seed: options.seed,
                batch_count: 1,
                // Upstream recommends VAE tiling to bound the ViT decoder's memory.
                vae_tiling: TilingParams {
                    enabled: true,
                    ..TilingParams::default()
                },
                ..ImageGenerationParams::default()
            })?
            .into_iter()
            .next()
            .context("MiniMax-H3-Image returned no inpainted image")?;
        tracing::debug!(
            output = %format!("{}x{}", generated.width(), generated.height()),
            "MiniMax-H3-Image raw output"
        );

        let mut output = resize_rgb(&generated, source.width(), source.height());
        for ((pixel, original), mask) in output
            .pixels_mut()
            .zip(source.pixels())
            .zip(source_mask.pixels())
        {
            let amount = f32::from(mask.0[0]) / 255.0;
            for (channel, original) in pixel.0.iter_mut().zip(original.0) {
                *channel = (f32::from(*channel) * amount + f32::from(original) * (1.0 - amount))
                    .round()
                    .clamp(0.0, 255.0) as u8;
            }
        }
        Ok(DynamicImage::ImageRgb8(output))
    }
}

/// Scales to the calibrated area in either direction, keeping the aspect ratio.
fn processing_size(width: u32, height: u32) -> (u32, u32) {
    let scale = (TARGET_AREA / (f64::from(width) * f64::from(height))).sqrt();
    let align = |value: u32| {
        let aligned = (f64::from(value) * scale / f64::from(SIZE_MULTIPLE)).round() as u32;
        aligned.max(1) * SIZE_MULTIPLE
    };
    (align(width), align(height))
}

fn resize_rgb(source: &RgbImage, width: u32, height: u32) -> RgbImage {
    if source.dimensions() == (width, height) {
        return source.clone();
    }
    let mut resized = RgbImage::new(width, height);
    Resizer::new()
        .resize(
            source,
            &mut resized,
            &ResizeOptions::new()
                .resize_alg(ResizeAlg::Convolution(FilterType::Lanczos3))
                .use_alpha(false),
        )
        .expect("source and destination images have the same pixel type");
    resized
}

fn resize_gray(source: &GrayImage, width: u32, height: u32) -> GrayImage {
    let mut resized = GrayImage::new(width, height);
    Resizer::new()
        .resize(
            source,
            &mut resized,
            &ResizeOptions::new()
                .resize_alg(ResizeAlg::Convolution(FilterType::Lanczos3))
                .use_alpha(false),
        )
        .expect("source and destination masks have the same pixel type");
    resized
}
