use anyhow::{Result, bail};
use koharu_translator::{GenerationConfig, Language};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use specta::Type;

use crate::stages::{
    Flux2KleinConfig, Flux3KleinConfig, KoharuLayoutRFDetrSeg2XLConfig, MiniMaxH3ImageConfig,
    RoremMixedConfig,
};

#[derive(Clone, Debug, PartialEq, Type)]
pub struct PipelineConfig {
    pub detection: DetectionModel,
    pub ocr: OcrModel,
    pub translation: TranslationConfig,
    pub inpainting: InpaintingModel,
    /// The Remove brush's model. It is chosen apart from the inpainting stage so
    /// switching it never changes automatic cleanup.
    pub remove: InpaintingModel,
    /// Settings for every model are kept independently of the active model.
    /// The active stage fields above only select which profile is used.
    pub processor: ProcessorConfig,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default)]
struct PipelineFile {
    detection: ModelSelection,
    ocr: ModelSelection,
    translation: TranslationConfig,
    inpainting: ModelSelection,
    remove: ModelSelection,
    #[serde(default)]
    processor: ProcessorConfig,
}

impl Default for PipelineFile {
    fn default() -> Self {
        Self {
            detection: ModelSelection {
                model: "koharu-layout-rfdetr-seg-2xl".to_owned(),
            },
            ocr: ModelSelection {
                model: "paddleocr-vl-1.6".to_owned(),
            },
            translation: TranslationConfig::default(),
            inpainting: ModelSelection {
                model: "lama".to_owned(),
            },
            remove: ModelSelection {
                model: "lama".to_owned(),
            },
            processor: ProcessorConfig::default(),
        }
    }
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(default)]
struct ModelSelection {
    model: String,
}

impl Serialize for PipelineConfig {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let detection = match &self.detection {
            DetectionModel::KoharuLayoutRFDetrSeg2XL(_) => "koharu-layout-rfdetr-seg-2xl",
        };
        let ocr = match &self.ocr {
            OcrModel::PaddleOcrVl1_6 => "paddleocr-vl-1.6",
            OcrModel::MangaOcr => "manga-ocr",
            OcrModel::BaberuOcr => "baberu-ocr",
            OcrModel::HayaiOcr => "hayai-ocr",
            OcrModel::DotsOcr => "dots.ocr",
        };
        let mut processor = self.processor.clone();
        let DetectionModel::KoharuLayoutRFDetrSeg2XL(config) = &self.detection;
        processor
            .koharu_layout_rfdetr_seg_2xl
            .get_or_insert_with(|| config.clone());
        for model in [&self.inpainting, &self.remove] {
            match model {
                InpaintingModel::Flux2Klein(config) => {
                    processor.flux2_klein.get_or_insert_with(|| config.clone());
                }
                InpaintingModel::Flux3Klein(config) => {
                    processor.flux3_klein.get_or_insert_with(|| config.clone());
                }
                InpaintingModel::MiniMaxH3Image(config) => {
                    processor
                        .minimax_h3_image
                        .get_or_insert_with(|| config.clone());
                }
                InpaintingModel::RoremMixed(config) => {
                    processor.rorem_mixed.get_or_insert_with(|| config.clone());
                }
                InpaintingModel::LaMa {} | InpaintingModel::AotInpainting {} => {}
            }
        }
        PipelineFile {
            detection: ModelSelection {
                model: detection.to_owned(),
            },
            ocr: ModelSelection {
                model: ocr.to_owned(),
            },
            translation: self.translation.clone(),
            inpainting: ModelSelection {
                model: self.inpainting.name().to_owned(),
            },
            remove: ModelSelection {
                model: self.remove.name().to_owned(),
            },
            processor,
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for PipelineConfig {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let file = PipelineFile::deserialize(deserializer)?;
        let detection = match file.detection.model.as_str() {
            "koharu-layout-rfdetr-seg-2xl" => DetectionModel::KoharuLayoutRFDetrSeg2XL(
                file.processor
                    .koharu_layout_rfdetr_seg_2xl
                    .clone()
                    .unwrap_or_default(),
            ),
            model => {
                return Err(serde::de::Error::custom(format!(
                    "unsupported detection model {model}"
                )));
            }
        };
        let ocr = match file.ocr.model.as_str() {
            "paddleocr-vl-1.6" => OcrModel::PaddleOcrVl1_6,
            "manga-ocr" => OcrModel::MangaOcr,
            "baberu-ocr" => OcrModel::BaberuOcr,
            "hayai-ocr" => OcrModel::HayaiOcr,
            "dots.ocr" => OcrModel::DotsOcr,
            model => {
                return Err(serde::de::Error::custom(format!(
                    "unsupported OCR model {model}"
                )));
            }
        };
        let inpainting = InpaintingModel::from_profile(&file.inpainting.model, &file.processor)
            .map_err(serde::de::Error::custom)?;
        let remove = InpaintingModel::from_profile(&file.remove.model, &file.processor)
            .map_err(serde::de::Error::custom)?;
        Ok(Self {
            detection,
            ocr,
            translation: file.translation,
            inpainting,
            remove,
            processor: file.processor,
        })
    }
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            detection: DetectionModel::KoharuLayoutRFDetrSeg2XL(
                KoharuLayoutRFDetrSeg2XLConfig::default(),
            ),
            ocr: OcrModel::PaddleOcrVl1_6,
            translation: TranslationConfig::default(),
            inpainting: InpaintingModel::LaMa {},
            remove: InpaintingModel::LaMa {},
            processor: ProcessorConfig::default(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Type)]
pub struct TranslationConfig {
    pub model: koharu_translator::ModelSelection,
    pub generation: GenerationConfig,
    #[specta(type = String)]
    pub target_language: Language,
    pub instructions: Option<String>,
}

impl Default for TranslationConfig {
    fn default() -> Self {
        Self {
            model: koharu_translator::ModelSelection::default(),
            generation: GenerationConfig::default(),
            target_language: Language::English,
            instructions: None,
        }
    }
}

impl PipelineConfig {
    pub fn load() -> anyhow::Result<koharu_config::Config<Self>> {
        koharu_config::load("pipeline")
    }

    pub fn detection(&self) -> Result<DetectionModel> {
        match &self.detection {
            DetectionModel::KoharuLayoutRFDetrSeg2XL(config) => {
                Ok(DetectionModel::KoharuLayoutRFDetrSeg2XL(
                    self.processor
                        .koharu_layout_rfdetr_seg_2xl
                        .clone()
                        .unwrap_or_else(|| config.clone()),
                ))
            }
        }
    }

    pub fn inpainting(&self) -> Result<InpaintingModel> {
        self.inpainting_profile(&self.inpainting)
    }

    pub fn remove(&self) -> Result<InpaintingModel> {
        self.inpainting_profile(&self.remove)
    }

    fn inpainting_profile(&self, model: &InpaintingModel) -> Result<InpaintingModel> {
        match model {
            InpaintingModel::LaMa {} => Ok(InpaintingModel::LaMa {}),
            InpaintingModel::AotInpainting {} => Ok(InpaintingModel::AotInpainting {}),
            InpaintingModel::Flux2Klein(config) => Ok(InpaintingModel::Flux2Klein(
                self.processor
                    .flux2_klein
                    .clone()
                    .unwrap_or_else(|| config.clone()),
            )),
            InpaintingModel::Flux3Klein(config) => Ok(InpaintingModel::Flux3Klein(
                self.processor
                    .flux3_klein
                    .clone()
                    .unwrap_or_else(|| config.clone()),
            )),
            InpaintingModel::MiniMaxH3Image(config) => Ok(InpaintingModel::MiniMaxH3Image(
                self.processor
                    .minimax_h3_image
                    .clone()
                    .unwrap_or_else(|| config.clone()),
            )),
            InpaintingModel::RoremMixed(config) => Ok(InpaintingModel::RoremMixed(
                self.processor
                    .rorem_mixed
                    .clone()
                    .unwrap_or_else(|| config.clone()),
            )),
        }
    }

    pub fn validate(&self) -> Result<()> {
        let _ = self.detection()?;
        let _ = self.inpainting()?;
        let _ = self.remove()?;
        if !matches!(
            self.ocr,
            OcrModel::PaddleOcrVl1_6
                | OcrModel::MangaOcr
                | OcrModel::BaberuOcr
                | OcrModel::HayaiOcr
                | OcrModel::DotsOcr
        ) {
            bail!("unsupported OCR model")
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, Type)]
#[serde(default)]
pub struct ProcessorConfig {
    #[serde(rename = "koharu-layout-rfdetr-seg-2xl")]
    pub koharu_layout_rfdetr_seg_2xl: Option<KoharuLayoutRFDetrSeg2XLConfig>,
    #[serde(rename = "flux2-klein")]
    pub flux2_klein: Option<Flux2KleinConfig>,
    #[serde(rename = "flux3-klein")]
    pub flux3_klein: Option<Flux3KleinConfig>,
    #[serde(rename = "minimax-h3-image")]
    pub minimax_h3_image: Option<MiniMaxH3ImageConfig>,
    #[serde(rename = "rorem-mixed")]
    pub rorem_mixed: Option<RoremMixedConfig>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Type)]
#[serde(tag = "model")]
pub enum DetectionModel {
    #[serde(rename = "koharu-layout-rfdetr-seg-2xl")]
    KoharuLayoutRFDetrSeg2XL(KoharuLayoutRFDetrSeg2XLConfig),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Type)]
#[serde(tag = "model")]
pub enum OcrModel {
    #[serde(rename = "paddleocr-vl-1.6")]
    PaddleOcrVl1_6,
    #[serde(rename = "manga-ocr")]
    MangaOcr,
    #[serde(rename = "baberu-ocr")]
    BaberuOcr,
    #[serde(rename = "hayai-ocr")]
    HayaiOcr,
    #[serde(rename = "dots.ocr")]
    DotsOcr,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Type)]
#[serde(tag = "model")]
pub enum InpaintingModel {
    #[serde(rename = "lama")]
    LaMa {},
    #[serde(rename = "aot-inpainting")]
    AotInpainting {},
    #[serde(rename = "flux2-klein")]
    Flux2Klein(Flux2KleinConfig),
    #[serde(rename = "flux3-klein")]
    Flux3Klein(Flux3KleinConfig),
    #[serde(rename = "minimax-h3-image")]
    MiniMaxH3Image(MiniMaxH3ImageConfig),
    #[serde(rename = "rorem-mixed")]
    RoremMixed(RoremMixedConfig),
}

impl InpaintingModel {
    fn name(&self) -> &'static str {
        match self {
            Self::LaMa {} => "lama",
            Self::AotInpainting {} => "aot-inpainting",
            Self::Flux2Klein(_) => "flux2-klein",
            Self::Flux3Klein(_) => "flux3-klein",
            Self::MiniMaxH3Image(_) => "minimax-h3-image",
            Self::RoremMixed(_) => "rorem-mixed",
        }
    }

    fn from_profile(name: &str, processor: &ProcessorConfig) -> std::result::Result<Self, String> {
        Ok(match name {
            "lama" => Self::LaMa {},
            "aot-inpainting" => Self::AotInpainting {},
            "flux2-klein" => Self::Flux2Klein(processor.flux2_klein.clone().unwrap_or_default()),
            "flux3-klein" => Self::Flux3Klein(processor.flux3_klein.clone().unwrap_or_default()),
            "minimax-h3-image" => {
                Self::MiniMaxH3Image(processor.minimax_h3_image.clone().unwrap_or_default())
            }
            "rorem-mixed" => Self::RoremMixed(processor.rorem_mixed.clone().unwrap_or_default()),
            model => return Err(format!("unsupported inpainting model {model}")),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_select_one_processor_for_each_phase() {
        let config = PipelineConfig::default();

        assert!(matches!(
            config.detection,
            DetectionModel::KoharuLayoutRFDetrSeg2XL(_)
        ));
        assert!(matches!(config.ocr, OcrModel::PaddleOcrVl1_6));
        assert!(matches!(config.inpainting, InpaintingModel::LaMa {}));
    }

    #[test]
    fn parses_phase_keyed_processor_configuration() {
        let config: PipelineConfig = toml::from_str(
            r#"
                [detection]
                model = "koharu-layout-rfdetr-seg-2xl"

                [ocr]
                model = "baberu-ocr"

                [inpainting]
                model = "rorem-mixed"

                [processor."rorem-mixed"]
                prompt = "Remove the lettering."
                negative_prompt = "letters, words"
            "#,
        )
        .unwrap();

        assert!(matches!(
            config.detection,
            DetectionModel::KoharuLayoutRFDetrSeg2XL(_)
        ));
        assert!(matches!(config.ocr, OcrModel::BaberuOcr));
        assert!(matches!(
            config.inpainting(),
            Ok(InpaintingModel::RoremMixed(config))
                if config.prompt == "Remove the lettering."
                    && config.negative_prompt == "letters, words"
        ));
    }

    #[test]
    fn missing_slots_use_defaults() {
        let config = toml::from_str::<PipelineConfig>("").unwrap();

        assert_eq!(config, PipelineConfig::default());
    }

    #[test]
    fn ignores_unknown_model_configuration_fields() {
        let config = toml::from_str::<PipelineConfig>(
            r#"
                [detection]
                model = "koharu-layout-rfdetr-seg-2xl"
                legacy_threshold = 0.5

                [ocr]
                model = "paddleocr-vl-1.6"
                legacy_language = "ja"

                [inpainting]
                model = "lama"
                legacy_resolution = 1024
            "#,
        )
        .unwrap();

        assert!(matches!(
            config.detection,
            DetectionModel::KoharuLayoutRFDetrSeg2XL(_)
        ));
        assert!(matches!(config.ocr, OcrModel::PaddleOcrVl1_6));
        assert!(matches!(config.inpainting(), Ok(InpaintingModel::LaMa {})));
    }

    #[test]
    fn parses_detection_and_generative_inpainting_options() {
        let config = toml::from_str::<PipelineConfig>(
            r#"
                [detection]
                model = "koharu-layout-rfdetr-seg-2xl"

                [inpainting]
                model = "flux2-klein"

                [processor."koharu-layout-rfdetr-seg-2xl"]
                text_threshold = 0.25
                bubble_threshold = 0.45
                panel_threshold = 0.55

                [processor."flux2-klein"]
                prompt = "Reconstruct the illustration without text."
            "#,
        )
        .unwrap();

        assert!(matches!(
            config.detection().unwrap(),
            DetectionModel::KoharuLayoutRFDetrSeg2XL(config)
                if config.text_threshold == Some(0.25)
                    && config.bubble_threshold == Some(0.45)
                    && config.panel_threshold == Some(0.55)
        ));
        assert!(matches!(
            config.inpainting().unwrap(),
            InpaintingModel::Flux2Klein(config)
                if config.prompt == "Reconstruct the illustration without text."
        ));
    }

    #[test]
    fn keeps_profiles_separate_from_active_stage_selection() {
        let config = toml::from_str::<PipelineConfig>(
            r#"
                [detection]
                model = "koharu-layout-rfdetr-seg-2xl"

                [inpainting]
                model = "flux2-klein"

                [processor."flux2-klein"]
                prompt = "saved prompt"
            "#,
        )
        .unwrap();

        let InpaintingModel::Flux2Klein(config) = config.inpainting().unwrap() else {
            panic!("expected FLUX profile")
        };
        assert_eq!(config.prompt, "saved prompt");
    }

    #[test]
    fn serializes_model_profiles_under_processor() {
        let config = PipelineConfig {
            detection: DetectionModel::KoharuLayoutRFDetrSeg2XL(KoharuLayoutRFDetrSeg2XLConfig {
                text_threshold: Some(0.25),
                ..Default::default()
            }),
            ocr: OcrModel::PaddleOcrVl1_6,
            translation: TranslationConfig::default(),
            inpainting: InpaintingModel::Flux2Klein(Flux2KleinConfig {
                prompt: "Keep the line art.".to_owned(),
            }),
            remove: InpaintingModel::LaMa {},
            processor: ProcessorConfig::default(),
        };
        let document = toml::to_string(&config).unwrap();
        assert!(document.contains("[detection]\nmodel = \"koharu-layout-rfdetr-seg-2xl\""));
        assert!(document.contains("[processor.koharu-layout-rfdetr-seg-2xl]"));
        assert!(document.contains("[processor.flux2-klein]"));
        assert!(document.contains("[translation]"));
        assert!(!document.contains("prompt = \"Keep the line art.\"\n[inpainting]"));

        let restored = toml::from_str::<PipelineConfig>(&document).unwrap();
        assert!(matches!(
            restored.inpainting().unwrap(),
            InpaintingModel::Flux2Klein(config) if config.prompt == "Keep the line art."
        ));
    }
}
