//! dots.ocr text recognition through llama.cpp.
//!
//! The GGUF files are ggml-org's conversion of `rednote-hilab/dots.ocr`
//! (now `dots-studio/dots.ocr`); the vision projector is llama.cpp's
//! `dotsocr` MTMD implementation. The prompt is upstream's `prompt_ocr`.

use anyhow::{Context, Result, ensure};
use image::DynamicImage;
use koharu_llama::mtmd::mtmd_default_marker;

use crate::{
    Device,
    llm::{
        ChatMessage, GenerationControl, GenerationOptions, Input, Llm, LoadOptions, MtmdOptions,
    },
};

crate::model_repository!("ggml-org/dots.ocr-GGUF" @ "2c093a32ca360a396bc6d87d60408636130b9d9b" {
    MODEL = "dots.ocr-Q8_0.gguf",
    PROJECTOR = "mmproj-dots.ocr-Q8_0.gguf",
});

const PROMPT: &str = "Extract the text content from this image.";
const MAX_NEW_TOKENS: usize = 512;
// The checkpoint ends a turn with this token, but the GGUF vocabulary does not
// mark it end-of-generation, so the model stops on it here.
const END_OF_TURN: &str = "<|endofassistant|>";

#[derive(Debug)]
pub struct DotsOcr {
    model: Llm,
}

impl DotsOcr {
    pub async fn load(device: Device) -> Result<Self> {
        let (model_path, projector_path) = tokio::try_join!(MODEL.resolve(), PROJECTOR.resolve())
            .context("failed to resolve dots.ocr files")?;
        let model = Llm::load_with_options(
            device,
            model_path,
            LoadOptions {
                mtmd: Some(MtmdOptions::new(projector_path)),
                ..LoadOptions::default()
            },
        )
        .await
        .context("failed to load dots.ocr")?;
        ensure!(
            model.capabilities().vision,
            "dots.ocr projector does not support images"
        );
        Ok(Self { model })
    }

    pub fn inference(&self, image: &DynamicImage) -> Result<String> {
        let prompt = self.model.render_chat_prompt(&[ChatMessage::user(format!(
            "{}{PROMPT}",
            mtmd_default_marker()
        ))])?;
        // Upstream samples at temperature 0.1 without a repetition penalty;
        // greedy decoding keeps OCR deterministic.
        let generation = self.model.inference_with_callback(
            &Input::new(&prompt).with_image(image),
            &GenerationOptions {
                max_tokens: MAX_NEW_TOKENS,
                temperature: 0.0,
                repeat_penalty: 1.0,
                repeat_last_n: 0,
                add_special: true,
                ..GenerationOptions::default()
            },
            |chunk| {
                Ok(if chunk.piece == END_OF_TURN {
                    GenerationControl::Stop
                } else {
                    GenerationControl::Continue
                })
            },
        )?;
        let text = generation.text;
        Ok(text
            .strip_suffix(END_OF_TURN)
            .unwrap_or(&text)
            .trim()
            .to_owned())
    }
}
