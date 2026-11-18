//! Manga OCR AR image and text processing.
//!
//! Ported from `pad_grid`, `preprocess`, and `MangaOCR.decode`:
//! https://huggingface.co/muscgab/manga-ocr-ar-experimental/blob/20208154a8d34dc6e5fc13ba81bc35f0a2f1a26e/infer.py

use std::{fs, path::Path};

use anyhow::{Context, Result, bail, ensure};
use fast_image_resize::{FilterType, ResizeAlg, ResizeOptions, Resizer};
use image::{DynamicImage, Rgb, RgbImage};
use koharu_torch::{Device, Kind, Tensor};
use serde::Deserialize;

pub(super) const PATCH: i64 = 14;
const MEAN: [f32; 3] = [0.481_454_66, 0.457_827_5, 0.408_210_73];
const STD: [f32; 3] = [0.268_629_54, 0.261_302_6, 0.275_777_1];

#[derive(Debug)]
pub(super) struct ImageProcessor {
    budget: i64,
    mean: Tensor,
    std: Tensor,
    device: Device,
}

/// One preprocessed crop.
#[derive(Debug)]
pub(super) struct ImageInput {
    /// Flattened `(channel, row, column)` patches `(1, h * w, 3 * 14 * 14)`.
    ///
    /// Upstream zero-pads to `patch_budget` and masks the padding as attention
    /// keys everywhere, so padded rows never reach a valid token. A single crop
    /// therefore carries only its `h * w` grid patches.
    pub(super) patches: Tensor,
    /// Patch-grid size `(h, w)`.
    pub(super) grid: [i64; 2],
}

impl ImageProcessor {
    pub(super) fn new(budget: i64, device: Device) -> Self {
        Self {
            budget,
            mean: Tensor::from_slice(&MEAN).to_device(device),
            std: Tensor::from_slice(&STD).to_device(device),
            device,
        }
    }

    pub(super) fn preprocess(&self, image: &DynamicImage) -> Result<ImageInput> {
        if image.width() == 0 || image.height() == 0 {
            bail!("cannot recognize an empty image");
        }
        let image = image.to_rgb8();
        let (scale, h, w) = pad_grid(image.height(), image.width(), self.budget);
        ensure!(
            h > 0 && w > 0 && h * w <= self.budget,
            "image resize produced an invalid {h}x{w} patch grid"
        );
        let canvas_width = (w * PATCH) as u32;
        let canvas_height = (h * PATCH) as u32;
        // Python's `round` is round-half-to-even.
        let resized_width = (f64::from(image.width()) * scale)
            .round_ties_even()
            .clamp(1.0, f64::from(canvas_width)) as u32;
        let resized_height = (f64::from(image.height()) * scale)
            .round_ties_even()
            .clamp(1.0, f64::from(canvas_height)) as u32;
        let small = resize_bicubic(&image, resized_width, resized_height)?;

        let mut canvas = RgbImage::from_pixel(canvas_width, canvas_height, border_median(&small));
        image::imageops::replace(&mut canvas, &small, 0, 0);

        let patches = Tensor::from_slice(canvas.as_raw())
            .to_device(self.device)
            .to_kind(Kind::Float)
            .view([h * PATCH, w * PATCH, 3]);
        let patches = (patches / 255.0 - &self.mean) / &self.std;
        let patches = patches
            .view([h, PATCH, w, PATCH, 3])
            .permute([0, 2, 4, 1, 3])
            .reshape([1, h * w, 3 * PATCH * PATCH]);
        Ok(ImageInput {
            patches,
            grid: [h, w],
        })
    }
}

/// Largest uniform scale `s` with `ceil(H*s/14) * ceil(W*s/14) <= budget`.
fn pad_grid(height: u32, width: u32, budget: i64) -> (f64, i64, i64) {
    let grid = |scale: f64, size: u32| (f64::from(size) * scale / PATCH as f64).ceil() as i64;
    let (mut lo, mut hi) = (1e-4, 1000.0);
    for _ in 0..60 {
        let mid = (lo + hi) / 2.0;
        if grid(mid, height) * grid(mid, width) <= budget {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    (lo, grid(lo, height), grid(lo, width))
}

/// Per-channel `np.median` of the top, bottom, left, and right edges (corners
/// counted twice), truncated to `u8` like upstream's `astype(np.uint8)`.
fn border_median(image: &RgbImage) -> Rgb<u8> {
    let (width, height) = image.dimensions();
    let mut border = Vec::with_capacity(2 * (width + height) as usize);
    for y in [0, height - 1] {
        border.extend((0..width).map(|x| *image.get_pixel(x, y)));
    }
    for x in [0, width - 1] {
        border.extend((0..height).map(|y| *image.get_pixel(x, y)));
    }
    Rgb(std::array::from_fn(|channel| {
        let mut values = border
            .iter()
            .map(|pixel| u16::from(pixel[channel]))
            .collect::<Vec<_>>();
        values.sort_unstable();
        let middle = values.len() / 2;
        if values.len() % 2 == 0 {
            ((values[middle - 1] + values[middle]) / 2) as u8
        } else {
            values[middle] as u8
        }
    }))
}

fn resize_bicubic(image: &RgbImage, width: u32, height: u32) -> Result<RgbImage> {
    if image.dimensions() == (width, height) {
        return Ok(image.clone());
    }
    let mut resized = RgbImage::new(width, height);
    Resizer::new()
        .resize(
            image,
            &mut resized,
            &ResizeOptions::new()
                // Catmull-Rom (a = -0.5) convolution is Pillow's BICUBIC filter.
                // fast_image_resize runs the vertical pass first with coarser
                // fixed-point coefficients, so ringing near glyph edges can
                // differ: on 2x-upscaled bubble crops the mean channel error is
                // about 0.25 levels (max 24), with identical decoded text and
                // per-step logits within 0.12 of upstream on CPU fp32.
                .resize_alg(ResizeAlg::Convolution(FilterType::CatmullRom))
                .use_alpha(false),
        )
        .context("failed to resize Manga OCR AR input")?;
    Ok(resized)
}

#[derive(Debug, Deserialize)]
pub(super) struct Tokenizer {
    characters: Vec<String>,
}

impl Tokenizer {
    pub(super) fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let tokenizer: Self = serde_json::from_str(&fs::read_to_string(path)?)?;
        ensure!(
            !tokenizer.characters.is_empty(),
            "Manga OCR AR vocabulary is empty"
        );
        Ok(tokenizer)
    }

    pub(super) fn len(&self) -> usize {
        self.characters.len()
    }

    /// Joins characters up to the first EOS (`len()`), the only output id
    /// beyond the character table.
    pub(super) fn decode(&self, token_ids: &[i64]) -> String {
        token_ids
            .iter()
            .map_while(|&id| self.characters.get(usize::try_from(id).ok()?))
            .map(String::as_str)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use image::{Rgb, RgbImage};

    use super::{border_median, pad_grid};

    #[test]
    fn grid_fills_the_patch_budget_with_aspect_kept() {
        let (_, h, w) = pad_grid(1000, 1000, 324);
        assert_eq!((h, w), (18, 18));
        let (scale, h, w) = pad_grid(300, 60, 324);
        assert!(h * w <= 324);
        assert_eq!((h, w), (40, 8));
        assert!((300.0 * scale / 14.0).ceil() as i64 == h);
    }

    #[test]
    fn border_median_truncates_even_counts() {
        let mut image = RgbImage::from_pixel(2, 1, Rgb([10, 0, 255]));
        image.put_pixel(1, 0, Rgb([13, 1, 255]));
        // Six border samples per channel: 10,13,10,13,10,13 -> (10 + 13) / 2.
        assert_eq!(border_median(&image), Rgb([11, 0, 255]));
    }
}
