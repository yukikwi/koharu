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

/// FLUX.2 Klein is calibrated for 1 MP; latents are 16-pixel aligned
/// (8x VAE downsampling followed by 2x2 patchification into 128 channels).
const MAX_AREA: f64 = 1024.0 * 1024.0;
/// Smallest working area. Below it a manual-removal crop is only a few dozen
/// tokens, the denoise mask swallows the whole bubble, and the model replaces
/// its fill with the surrounding page colour.
const MIN_AREA: f64 = 768.0 * 768.0;
/// Linear scale of the reference image relative to the working size.
const REFERENCE_SCALE: f64 = 0.5;
const SIZE_MULTIPLE: u32 = 16;

/// Core tile edge for the inpainting stage. Every crop costs a full DiT pass
/// at no less than `MIN_AREA`, so a crop of this core plus the stage's
/// 128-pixel context on each side (at most 1024 pixels) packs as many mask
/// components into one call as fit in the calibrated area.
pub const TILE_SIZE: u32 = 768;
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
        let (reference_width, reference_height) = (
            align_to_multiple(f64::from(width) * REFERENCE_SCALE),
            align_to_multiple(f64::from(height) * REFERENCE_SCALE),
        );
        let reference_image = resize_rgb(&source, reference_width, reference_height);
        let mut mask = resize_gray(&source_mask, width, height);
        for pixel in mask.pixels_mut() {
            *pixel = Luma([if pixel.0[0] < 128 { 0 } else { 255 }]);
        }
        let masked = mask.pixels().filter(|pixel| pixel.0[0] != 0).count();
        tracing::info!(
            input = %format!("{}x{}", source.width(), source.height()),
            processing = %format!("{width}x{height}"),
            reference = %format!("{reference_width}x{reference_height}"),
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
                init_image: Some(init_image),
                // Deliberate divergence from upstream, which passes the init
                // image as the reference at full size. Unmasked latents are
                // already preserved by the denoise mask, so the reference only
                // supplies context; at half scale it adds a quarter of the
                // image tokens instead of doubling the DiT sequence.
                reference_images: vec![reference_image],
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
        correct_color_drift(&mut output, &source, &source_mask);
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

/// Clamps the crop area into `MIN_AREA..=MAX_AREA`, keeping the aspect ratio.
/// Sampling cost grows with the token count, so crops already inside the range
/// keep their native resolution instead of being upscaled to the calibrated
/// area.
fn processing_size(width: u32, height: u32) -> (u32, u32) {
    let area = f64::from(width) * f64::from(height);
    let scale = (area.clamp(MIN_AREA, MAX_AREA) / area).sqrt();
    (
        align_to_multiple(f64::from(width) * scale),
        align_to_multiple(f64::from(height) * scale),
    )
}

fn align_to_multiple(value: f64) -> u32 {
    ((value / f64::from(SIZE_MULTIPLE)).round() as u32).max(1) * SIZE_MULTIPLE
}

/// Neighbourhood used to estimate decode drift; large enough to reach unmasked
/// pixels from the middle of a brush stroke inside a small bubble.
const DRIFT_RADIUS: u32 = 32;
/// Differences above this are content changes (line art, residual glyphs), not
/// VAE colour drift, and must not tint the fill.
const DRIFT_LIMIT: i32 = 40;

/// Deliberate divergence from upstream: stable-diffusion.cpp keeps unmasked
/// content only in latent space, so the decoded image is shifted in colour
/// relative to the source. Compositing it pixel-exactly against the source
/// leaves visible seams and blotches on flat bubble fills. Unmasked pixels
/// show that shift directly, so a locally smoothed source-minus-decoded offset
/// is added back to the generated pixels.
fn correct_color_drift(output: &mut RgbImage, source: &RgbImage, mask: &GrayImage) {
    let (width, height) = output.dimensions();
    let stride = width as usize + 1;
    // Summed-area tables of [dr, dg, db, count] over trusted unmasked pixels.
    let mut sums = vec![[0i64; 4]; stride * (height as usize + 1)];
    let mut total = [0i64; 4];
    for y in 0..height {
        let mut row = [0i64; 4];
        for x in 0..width {
            if mask.get_pixel(x, y).0[0] == 0 {
                let decoded = output.get_pixel(x, y).0;
                let original = source.get_pixel(x, y).0;
                let diff: [i32; 3] =
                    std::array::from_fn(|c| i32::from(original[c]) - i32::from(decoded[c]));
                if diff.iter().all(|value| value.abs() <= DRIFT_LIMIT) {
                    for c in 0..3 {
                        row[c] += i64::from(diff[c]);
                    }
                    row[3] += 1;
                }
            }
            let above = sums[y as usize * stride + x as usize + 1];
            sums[(y as usize + 1) * stride + x as usize + 1] =
                std::array::from_fn(|c| above[c] + row[c]);
        }
        for c in 0..4 {
            total[c] += row[c];
        }
    }
    if total[3] == 0 {
        return;
    }
    for y in 0..height {
        for x in 0..width {
            if mask.get_pixel(x, y).0[0] == 0 {
                continue;
            }
            let (left, top) = (
                x.saturating_sub(DRIFT_RADIUS) as usize,
                y.saturating_sub(DRIFT_RADIUS) as usize,
            );
            let right = (x + DRIFT_RADIUS + 1).min(width) as usize;
            let bottom = (y + DRIFT_RADIUS + 1).min(height) as usize;
            let at = |row: usize, col: usize| sums[row * stride + col];
            let window: [i64; 4] = std::array::from_fn(|c| {
                at(bottom, right)[c] - at(top, right)[c] - at(bottom, left)[c] + at(top, left)[c]
            });
            let local = if window[3] > 0 { window } else { total };
            let pixel = output.get_pixel_mut(x, y);
            for c in 0..3 {
                let offset = local[c] as f64 / local[3] as f64;
                pixel.0[c] = (f64::from(pixel.0[c]) + offset).round().clamp(0.0, 255.0) as u8;
            }
        }
    }
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
