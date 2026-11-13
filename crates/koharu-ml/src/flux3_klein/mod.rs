//! FLUX.3 Image Klein masked reference editing.
//!
//! Sampling follows the distilled-model recipe in stable-diffusion.cpp:
//! https://github.com/leejet/stable-diffusion.cpp/blob/b9d7176bf88fe465576e0aafbe47ba253ffcfc2c/docs/flux3.md
//! The source crop is both the init image, so unmasked latents are preserved by
//! the denoise mask, and the `flux2`-preset reference image, so the DiT attends
//! to the content to reconstruct:
//! https://github.com/leejet/stable-diffusion.cpp/blob/b9d7176bf88fe465576e0aafbe47ba253ffcfc2c/src/pipeline/image.cpp#L511-L538

mod model;

use std::time::Instant;

use anyhow::{Context, Result, ensure};
use fast_image_resize::{FilterType, ResizeAlg, ResizeOptions, Resizer};
use image::{DynamicImage, GrayImage, Luma, RgbImage};
use koharu_diffusion::{
    CacheMode, CacheParams, GuidanceParams, ImageGenerationParams, SampleMethod, SampleParams,
};

use self::model::{Model, ModelPaths};

crate::model_repository!("leejet/flux-3-image-klein-GGUF" @ "1c2ad2f9b9247ccca6506a23bdb033fdb0873321" {
    TRANSFORMER_WEIGHTS = "flux3-image-klein-Q4_K.gguf"
});
crate::model_repository!("Qwen/Qwen3-VL-4B-Instruct-GGUF" @ "1cd86afb9a95c410a6038ab3b40d8b578c892266" {
    TEXT_ENCODER_WEIGHTS = "Qwen3VL-4B-Instruct-Q4_K_M.gguf"
});
// Upstream accepts any FLUX.2 VAE. The ungated small-decoder checkpoint is the
// one FLUX.2 Klein already downloads, so both models share it.
crate::model_repository!("black-forest-labs/FLUX.2-small-decoder" @ "a3efc24f613ef42d9428af62fdbd6f5fd8856c4a" {
    VAE_WEIGHTS = "full_encoder_small_decoder.safetensors"
});

pub const DEFAULT_PROMPT: &str = "Remove only the text and lettering. Keep the speech bubble's flat fill colour, outline and tail exactly as they are. Fill the removed text with the surrounding bubble colour.";

/// FLUX.2 Klein is calibrated for up to 1 MP; latents are 16-pixel aligned
/// (8x VAE downsampling followed by 2x2 patchification into 128 channels).
const MAX_AREA: f64 = 1024.0 * 1024.0;
const SIZE_MULTIPLE: u32 = 16;
const SAMPLE_STEPS: i32 = 16;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Flux3KleinInpaintOptions {
    pub seed: i64,
}

impl Default for Flux3KleinInpaintOptions {
    fn default() -> Self {
        Self { seed: -1 }
    }
}

#[derive(Debug)]
pub struct Flux3KleinInpaint {
    model: Model,
}

impl Flux3KleinInpaint {
    pub async fn load(device: crate::Device) -> Result<Self> {
        let started = Instant::now();
        let (transformer, text_encoder, vae) = tokio::try_join!(
            TRANSFORMER_WEIGHTS.resolve(),
            TEXT_ENCODER_WEIGHTS.resolve(),
            VAE_WEIGHTS.resolve(),
        )
        .context("failed to resolve FLUX.3 Klein model assets")?;
        tracing::info!(
            transformer = %transformer.display(),
            text_encoder = %text_encoder.display(),
            vae = %vae.display(),
            elapsed_ms = started.elapsed().as_millis(),
            "resolved FLUX.3 Klein assets"
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
        options: &Flux3KleinInpaintOptions,
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
            steps = SAMPLE_STEPS,
            "FLUX.3 Klein inpainting tile"
        );

        let generated = self
            .model
            .forward(&ImageGenerationParams {
                prompt: prompt.to_owned(),
                width: i32::try_from(width)?,
                height: i32::try_from(height)?,
                init_image: Some(init_image.clone()),
                reference_images: vec![init_image],
                reference_image_args: Some("resize_before_vae=0".into()),
                mask_image: Some(mask),
                sample: SampleParams {
                    guidance: GuidanceParams {
                        text_cfg: 1.0,
                        ..GuidanceParams::default()
                    },
                    sample_method: SampleMethod::Euler,
                    sample_steps: SAMPLE_STEPS,
                    ..SampleParams::default()
                },
                // EasyCache reuses the DiT output on steps whose input barely
                // changes; the unset threshold selects upstream's DiT default.
                cache: CacheParams {
                    mode: CacheMode::EasyCache,
                    ..CacheParams::default()
                },
                strength: 1.0,
                seed: options.seed,
                batch_count: 1,
                ..ImageGenerationParams::default()
            })?
            .into_iter()
            .next()
            .context("FLUX.3 Klein returned no inpainted image")?;
        tracing::debug!(
            output = %format!("{}x{}", generated.width(), generated.height()),
            "FLUX.3 Klein raw output"
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

/// Downscales crops above the calibrated area, keeping the aspect ratio.
/// Smaller crops keep their native resolution: upscaling them only multiplies
/// the DiT token count (doubled by the reference image) without adding detail.
fn processing_size(width: u32, height: u32) -> (u32, u32) {
    let scale = (MAX_AREA / (f64::from(width) * f64::from(height)))
        .sqrt()
        .min(1.0);
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
