use anyhow::Result;
use async_trait::async_trait;
use koharu_scene::{
    Authored, EntityId, Geometry, Glossary, LanguageTag, Origin, Snapshot, SourceText, Translation,
};
use koharu_translator::{
    GlossaryTerm, SegmentBox, TranslationContext, TranslationRequest, Translator,
};

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
        let language = LanguageTag::new(self.config.target_language.tag())?;
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
        let (context, observed) = previous_pages(
            &input.scene,
            input.page,
            self.config.context_pages,
            &language,
        )?;
        request = request.with_context(context);
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
        for entity in observed {
            edit.observe::<SourceText>(entity)?;
            edit.observe::<Translation>(entity)?;
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

/// Collects source and translation pairs from the nearest preceding pages that
/// have any, oldest first, so dialogue that crosses a page break keeps its
/// voice and terminology. Pages without translated text do not use up the
/// budget. The returned entities are observed so a later edit to the context
/// rejects this result instead of committing work based on stale text.
fn previous_pages(
    scene: &Snapshot,
    page: EntityId,
    limit: u32,
    language: &LanguageTag,
) -> Result<(Vec<TranslationContext>, Vec<EntityId>)> {
    let mut pages = Vec::new();
    let mut observed = Vec::new();
    if limit == 0 {
        return Ok((Vec::new(), observed));
    }
    let order = scene.pages().map(|page| page.id()).collect::<Vec<_>>();
    let current = order.iter().position(|id| *id == page).unwrap_or(0);
    for id in order[..current].iter().rev() {
        let mut context = Vec::new();
        if let Some(group) = scene.page(*id)?.text_group()? {
            for layer in group.text_layers()? {
                let content = layer.content()?;
                let (Some(source), Some(translation)) = (content.source()?, content.translation()?)
                else {
                    continue;
                };
                if translation
                    .language
                    .as_ref()
                    .is_some_and(|tag| tag != language)
                {
                    continue;
                }
                let (source, translation) = (source.text.value, translation.text.value);
                if source.trim().is_empty() || translation.trim().is_empty() {
                    continue;
                }
                observed.push(content.id());
                context.push(TranslationContext {
                    source,
                    translation,
                });
            }
        }
        if !context.is_empty() {
            pages.push(context);
            if pages.len() == limit as usize {
                break;
            }
        }
    }
    Ok((pages.into_iter().rev().flatten().collect(), observed))
}

#[cfg(test)]
mod tests {
    use koharu_scene::{At, Origin, PageDraft, Session, TextLayout, TextLayoutKind};

    use super::*;

    #[tokio::test]
    async fn context_skips_untranslated_pages_and_keeps_page_order() {
        let mut session = Session::memory().await.unwrap();
        let mut current = None;
        let patch = session
            .snapshot()
            .patch(|edit| {
                let english = LanguageTag::new("en-US")?;
                let french = LanguageTag::new("fr-FR")?;
                let pages = [
                    vec![("一", Some(("one", &english)))],
                    vec![
                        ("二", Some(("two", &english))),
                        ("三", Some(("trois", &french))),
                    ],
                    vec![("四", None)],
                    vec![("五", Some(("five", &english)))],
                    vec![("六", None)],
                ];
                for texts in pages {
                    let page = edit.add_page(PageDraft::new("page", 1.0, 1.0), At::End)?;
                    current = Some(page);
                    for (source, translation) in texts {
                        let content = edit.add_text_content(page, At::End)?;
                        edit.set(
                            content,
                            &SourceText {
                                text: Authored::user((*source).to_owned()),
                                language: None,
                            },
                        )?;
                        if let Some((text, language)) = translation {
                            edit.set(
                                content,
                                &Translation {
                                    text: Authored::user((*text).to_owned()),
                                    language: Some((*language).clone()),
                                },
                            )?;
                        }
                        edit.add_text_layer(
                            page,
                            At::End,
                            content,
                            &TextLayout {
                                origin: Origin::User,
                                kind: TextLayoutKind::Paragraph,
                                angle_degrees: None,
                            },
                        )?;
                    }
                }
                Ok(())
            })
            .unwrap();
        let scene = session.commit(patch).await.unwrap().snapshot;
        let english = LanguageTag::new("en-US").unwrap();
        let pairs = |limit| {
            previous_pages(&scene, current.unwrap(), limit, &english)
                .unwrap()
                .0
                .into_iter()
                .map(|context| (context.source, context.translation))
                .collect::<Vec<_>>()
        };
        let pair = |source: &str, translation: &str| (source.to_owned(), translation.to_owned());

        assert!(pairs(0).is_empty());
        assert_eq!(pairs(1), [pair("五", "five")]);
        assert_eq!(pairs(2), [pair("二", "two"), pair("五", "five")]);
        assert_eq!(
            pairs(9),
            [pair("一", "one"), pair("二", "two"), pair("五", "five")]
        );
    }
}
