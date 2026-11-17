use anyhow::Result;
use async_trait::async_trait;
use koharu_scene::{Authored, Geometry, Glossary, LanguageTag, Origin, SourceText, Translation};
use koharu_translator::{GlossaryTerm, SegmentBox, TranslationRequest, Translator};

use crate::{Progress, Stage, TranslationConfig, progress, scope::geometry_extents};

use super::{StageInput, StageProcessor, finish, generation};

const PRODUCER: &str = "dev.koharu.pipeline.translation";

pub(super) struct Processor {
    config: TranslationConfig,
    translator: Translator,
}

impl Processor {
    pub(super) fn new(config: TranslationConfig, translator: Translator) -> Self {
        Self { config, translator }
    }
}

#[async_trait]
impl StageProcessor for Processor {
    fn model(&self) -> &'static str {
        Translator::model(&self.config.model)
    }

    fn unload(&self) -> bool {
        self.translator.unload()
    }

    async fn load(&self) -> Result<()> {
        self.translator.load_model(&self.config.model).await
    }

    async fn process(&self, input: StageInput) -> Result<koharu_scene::Patch> {
        let page = input.scene.page(input.page)?;
        let size = page.page()?;
        let mut targets = Vec::new();
        let mut regions = Vec::new();
        let mut boxes = Vec::new();
        if let Some(group) = page.text_group()? {
            for layer in group.text_layers()? {
                if !input.contains_entity(layer.id())? {
                    continue;
                }
                let content = layer.content()?;
                let Some(source) = content.source()? else {
                    continue;
                };
                if !source.text.value.trim().is_empty() {
                    // The OCR region is where the source text was read, so it is the
                    // location that maps the segment back onto the page image.
                    let region = content.source_region()?;
                    let bounds = region
                        .map(|region| region.geometry())
                        .transpose()?
                        .as_ref()
                        .and_then(geometry_extents)
                        .and_then(|extents| {
                            SegmentBox::from_extents(extents, size.width, size.height)
                        });
                    regions.extend(region.map(|region| region.id()));
                    boxes.push(bounds);
                    targets.push((content.id(), source.text.value));
                }
            }
        }
        let mut request = TranslationRequest::new(
            targets.iter().map(|(_, source)| source.clone()),
            self.config.target_language,
        )
        .with_boxes(boxes);
        if let Some(instructions) = self.config.instructions.as_deref() {
            request = request.with_instructions(instructions);
        }
        if let Some(glossary) = input.scene.project_component::<Glossary>()? {
            request =
                request.with_glossary(glossary.entries.into_iter().map(|entry| GlossaryTerm {
                    source: entry.source,
                    target: entry.target,
                    note: entry.note,
                }));
        }
        if Translator::supports_vision(&self.config.model, &self.config.generation)
            && let Some(image) = input.images.get(&input.scene, input.page, "source").await?
        {
            request = request.with_image(image);
        }
        let (provider, translated) = self
            .translator
            .translate(
                &self.config.model,
                self.config.generation,
                request,
                |retry| {
                    progress::emit(
                        input.progress(),
                        Progress::Retrying {
                            page: input.page,
                            stage: Stage::Translation,
                            model: self.model().to_owned(),
                            attempt: retry.attempt,
                            attempts: retry.attempts,
                        },
                    );
                },
            )
            .await?;
        let language = LanguageTag::new(self.config.target_language.tag())?;
        let generated = generation(PRODUCER, provider)?;
        let mut edit = input.scene.edit_as(generated.clone());
        edit.observe_project::<Glossary>()?;
        for (entity, _) in &targets {
            edit.observe::<SourceText>(*entity)?;
            edit.observe::<Translation>(*entity)?;
        }
        for region in regions {
            edit.observe::<Geometry>(region)?;
        }
        for ((entity, source), text) in targets.into_iter().zip(translated) {
            if input
                .scene
                .component::<Translation>(entity)?
                .is_some_and(|value| matches!(value.text.origin, Origin::User))
            {
                continue;
            }
            let text = if source.trim() == "\u{2026}" {
                "\u{2026}".to_owned()
            } else {
                text
            };
            edit.set(
                entity,
                &Translation {
                    text: Authored::generated(text, generated.clone()),
                    language: Some(language.clone()),
                },
            )?;
        }
        finish(edit)
    }
}
