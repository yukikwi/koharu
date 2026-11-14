use std::{io::Cursor, sync::Arc};

use anyhow::Context;
use base64::{Engine as _, engine::general_purpose::STANDARD};
use fast_image_resize::{FilterType, ResizeAlg, ResizeOptions, Resizer};
use image::{DynamicImage, ImageEncoder, RgbImage, codecs::jpeg::JpegEncoder};

use crate::Language;

const MAX_IMAGE_DIMENSION: u32 = 2048;
const JPEG_QUALITY: u8 = 88;

#[derive(Debug, Clone, PartialEq)]
pub struct TranslationRequest {
    pub segments: Vec<String>,
    /// Page location of each segment, index-aligned with `segments`. Empty when
    /// the caller has no geometry.
    pub boxes: Vec<Option<SegmentBox>>,
    pub source_language: Option<Language>,
    pub target_language: Language,
    pub instructions: Option<String>,
    pub glossary: Vec<GlossaryTerm>,
    pub context: Vec<TranslationContext>,
    pub image: Option<Arc<DynamicImage>>,
}

impl TranslationRequest {
    #[must_use]
    pub fn new(
        segments: impl IntoIterator<Item = impl Into<String>>,
        target_language: Language,
    ) -> Self {
        Self {
            segments: segments.into_iter().map(Into::into).collect(),
            boxes: Vec::new(),
            source_language: None,
            target_language,
            instructions: None,
            glossary: Vec::new(),
            context: Vec::new(),
            image: None,
        }
    }

    #[cfg(test)]
    #[must_use]
    pub fn with_source_language(mut self, language: Language) -> Self {
        self.source_language = Some(language);
        self
    }

    #[must_use]
    pub fn with_instructions(mut self, instructions: impl Into<String>) -> Self {
        self.instructions = Some(instructions.into());
        self
    }

    #[must_use]
    pub fn with_glossary(mut self, glossary: impl IntoIterator<Item = GlossaryTerm>) -> Self {
        self.glossary = glossary.into_iter().collect();
        self
    }

    #[must_use]
    pub fn with_boxes(mut self, boxes: impl IntoIterator<Item = Option<SegmentBox>>) -> Self {
        self.boxes = boxes.into_iter().collect();
        self
    }

    #[must_use]
    pub fn with_image(mut self, image: Arc<DynamicImage>) -> Self {
        self.image = Some(image);
        self
    }

    #[cfg(test)]
    #[must_use]
    pub fn with_context(mut self, context: impl IntoIterator<Item = TranslationContext>) -> Self {
        self.context = context.into_iter().collect();
        self
    }

    pub(crate) fn prepare_image(&mut self) -> anyhow::Result<()> {
        let Some(image) = self.image.as_ref() else {
            return Ok(());
        };
        let (width, height) = (image.width(), image.height());
        let longest = width.max(height);
        if longest <= MAX_IMAGE_DIMENSION {
            return Ok(());
        }
        let resized_width =
            (u64::from(width) * u64::from(MAX_IMAGE_DIMENSION) / u64::from(longest)).max(1) as u32;
        let resized_height =
            (u64::from(height) * u64::from(MAX_IMAGE_DIMENSION) / u64::from(longest)).max(1) as u32;
        let source = image.to_rgb8();
        let mut resized = RgbImage::new(resized_width, resized_height);
        Resizer::new()
            .resize(
                &source,
                &mut resized,
                &ResizeOptions::new()
                    .resize_alg(ResizeAlg::Convolution(FilterType::Lanczos3))
                    .use_alpha(false),
            )
            .context("failed to resize translation image")?;
        self.image = Some(Arc::new(DynamicImage::ImageRgb8(resized)));
        Ok(())
    }

    pub(crate) fn remove_image(&mut self) {
        self.image = None;
    }
}

pub(crate) struct EncodedImage {
    pub(crate) data: String,
}

impl EncodedImage {
    pub(crate) fn data_url(&self) -> String {
        format!("data:image/jpeg;base64,{}", self.data)
    }
}

pub(crate) fn encode_image(image: &DynamicImage) -> anyhow::Result<EncodedImage> {
    let rgb = image.to_rgb8();
    let mut bytes = Cursor::new(Vec::new());
    JpegEncoder::new_with_quality(&mut bytes, JPEG_QUALITY)
        .write_image(
            rgb.as_raw(),
            rgb.width(),
            rgb.height(),
            image::ExtendedColorType::Rgb8,
        )
        .context("failed to encode translation image")?;
    Ok(EncodedImage {
        data: STANDARD.encode(bytes.into_inner()),
    })
}

/// A segment's axis-aligned bounds on its page as `[y_min, x_min, y_max, x_max]`
/// normalized to `0..=1000`. This is the `box_2d` grounding convention Gemini
/// and Gemma vision models are trained on, and it stays valid when the attached
/// page image is downscaled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(transparent)]
pub struct SegmentBox([u16; 4]);

impl SegmentBox {
    /// Normalizes pixel extents on a `page_width` x `page_height` page.
    #[must_use]
    pub fn from_extents(
        (min_x, min_y, max_x, max_y): (f64, f64, f64, f64),
        page_width: f64,
        page_height: f64,
    ) -> Option<Self> {
        if !(page_width > 0.0 && page_height > 0.0) {
            return None;
        }
        let scale =
            |value: f64, extent: f64| (value / extent * 1000.0).round().clamp(0.0, 1000.0) as u16;
        let bounds = [
            scale(min_y, page_height),
            scale(min_x, page_width),
            scale(max_y, page_height),
            scale(max_x, page_width),
        ];
        (bounds[0] < bounds[2] && bounds[1] < bounds[3]).then_some(Self(bounds))
    }
}

/// A source term with the exact rendering the translation must use.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GlossaryTerm {
    pub source: String,
    pub target: String,
    pub note: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct TranslationContext {
    pub source: String,
    pub translation: String,
}

impl TranslationContext {
    #[cfg(test)]
    #[must_use]
    pub fn new(source: impl Into<String>, translation: impl Into<String>) -> Self {
        Self {
            source: source.into(),
            translation: translation.into(),
        }
    }
}
