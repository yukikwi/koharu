//! Translation: one conversation per open project, kept in memory, a turn per
//! page in page order. Changing the model, the target, or the prompt starts a
//! new one. A page is its texts as a JSON object numbered in reading order, and
//! its answer must keep the keys, held to a schema, an unusable answer being
//! asked again up to three asks in all; a page that fails never
//! enters the history, and an answer echoing the page before's is asked again
//! without the history it copied. A text given back unchanged, such as a
//! copyright line or a "!?", is in the target's script already, so it keeps
//! its own lettering. The project's glossary terms a page holds follow its
//! texts as constraints.

mod conversation;
mod model;
mod placement;
mod turn;

use anyhow::{Context as _, Result, ensure};
use base64::Engine as _;
use icu_experimental::displaynames::{
    DisplayNamesOptions, DisplayNamesPreferences,
    multi::{LanguageDisplayNames, LocaleDisplayNamesFormatter},
};
use icu_locale::LocaleExpander;
use icu_locale_core::{LanguageIdentifier, Locale, langid, subtags::language};
use indoc::formatdoc;
use koharu_llm::rig::{completion::Reasoning, driver::DynModel};
use koharu_scene::{Content, LayerId, Op, Project, Term};
use rayon::prelude::*;
use serde::{Deserialize, Deserializer, Serialize, de::Error as _};
use serde_json::json;
use tokio::sync::OwnedMutexGuard;
use tracing::{Instrument as _, info_span};

pub(crate) use self::conversation::{Conversation, Conversations};
pub use self::placement::placement;
use self::turn::Turn;
use super::{Member, Stopped, Task, action, key};
use crate::{Event, PipelineConfig, Stage};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
#[derive(specta::Type)]
pub struct TranslatorConfig {
    /// A `vendor/model` reference, as koharu-llm connects one.
    pub model: String,
    /// A language CLDR names in English.
    #[serde(deserialize_with = "target")]
    #[specta(type = String)]
    pub target: LanguageIdentifier,
    /// The user's own instructions to the model, in any language and form,
    /// such as a name's spelling or the tone; they follow Koharu's own system
    /// prompt as written, and can't replace it.
    pub instructions: String,
    /// Whether each page's turn shows the model the page itself beside its
    /// texts, as an image goes with a chat message to OpenAI's and Gemini's
    /// models; only for a model that reads images.
    pub vision: bool,
    /// How much the model reasons before it answers: off, or one of rig's
    /// portable levels after OpenAI's `reasoning_effort`, written as one word;
    /// the model's own default when unset. rig refuses off for a model that
    /// can't stop reasoning.
    #[serde(with = "koharu_llm::reasoning")]
    #[specta(type = Option<String>)]
    pub reasoning: Option<Reasoning>,
}

impl Default for TranslatorConfig {
    fn default() -> Self {
        Self {
            // On 16 Blue Archive pages 9B read names and meaning that 2B
            // missed, such as Shale for シャーレ where 2B wrote "tray", at
            // about twice the decoding time.
            model: "koharu/index-translate-9b".to_owned(),
            target: langid!("en"),
            instructions: String::new(),
            vision: false,
            reasoning: None,
        }
    }
}

/// The system prompt: Index-Translate's structured-data template in English,
/// as bilibili/Index-Translate's `docs/prompts.md` gives it
/// (93ac63774d1c6939aec05e6f0f191bf90f225365, Apache-2.0), for the JSON the
/// answer is parsed by; then the user's instructions as they wrote them,
/// whatever their form.
fn system(config: &TranslatorConfig) -> Result<String> {
    let template = formatdoc! {"
        Translate the manga text in each message into {target}. Each message is a page's texts as a JSON object, numbered in reading order: translate only the values, and never alter the structure or the keys. Output only the translated JSON, without any explanation.",
        target = name(&config.target)?,
    };
    Ok(match config.instructions.trim() {
        "" => template,
        instructions => formatdoc! {"
            {template}

            {instructions}",
            template = template,
            instructions = instructions,
        },
    })
}

fn target<'de, D: Deserializer<'de>>(deserializer: D) -> Result<LanguageIdentifier, D::Error> {
    let target = LanguageIdentifier::deserialize(deserializer)?;
    name(&target).map_err(D::Error::custom)?;
    Ok(target)
}

/// A page's texts, and the page as a JPEG in base 64 when the model sees it.
struct Page {
    /// Each block and its source's language.
    blocks: Vec<(LayerId, LanguageIdentifier)>,
    texts: Vec<String>,
    /// The glossary's terms the texts hold, as constraints.
    constraints: Option<String>,
    image: Option<String>,
}

pub(crate) async fn translate(
    task: &Task,
    config: &PipelineConfig,
    mut conversation: OwnedMutexGuard<Conversation>,
) -> Result<()> {
    let member = &task.members[0];
    let translator = config.translator.clone();
    let (read, vision) = (member.clone(), translator.vision);
    let page = tokio::task::spawn_blocking(move || {
        let _span = info_span!("read", stage = %Stage::Translate, page = read.index + 1).entered();
        page(&read, vision)
    })
    .await??;
    let Some(page) = page else {
        return Ok(());
    };
    let conversation = &mut *conversation;
    conversation.keep(&translator);
    let model = match key(Stage::Translate, config) {
        // A local model loads on the GPU thread, which only hands it over:
        // its engine runs on a thread of its own, so the turn waits off the
        // GPU thread, and the other stages' models run meanwhile.
        Some(key) => {
            let (reference, images) = (translator.model.clone(), translator.vision);
            task.inference(&[member], move |models| {
                let model = models.get(&key, |_| model::load(&reference, images))?;
                Ok(DynModel::clone(model))
            })
            .await?
        }
        None => {
            task.started(&[member]);
            conversation.hosted(&translator).await?
        }
    };
    let system = system(&translator)?;
    let turn = Turn::new(
        system,
        &page.texts,
        page.constraints.as_deref(),
        page.image.as_deref(),
        translator.reasoning,
        &member.cancel,
    );
    let retried = |attempt| {
        task.events.send(Event::Retrying {
            run: member.run,
            page: member.page,
            stage: Stage::Translate,
            attempt,
            attempts: turn::ATTEMPTS,
        });
    };
    let Some(translations) = turn
        .take(conversation, &model, retried)
        .instrument(info_span!("llm", texts = page.texts.len()))
        .await?
    else {
        return Err(Stopped.into());
    };
    let translated: Vec<(LayerId, LanguageIdentifier, String)> = page
        .blocks
        .into_iter()
        .zip(translations)
        .map(|((block, source), text)| (block, source, text))
        .collect();
    member
        .apply(Stage::Translate, move |project| {
            written(project, translated, &translator)
        })
        .await
}

/// The page's texts the task works on, and the page itself when the model
/// sees it; `None` when it has no text.
fn page(member: &Member, vision: bool) -> Result<Option<Page>> {
    // A page with no scan, such as a blank one with typed text, goes
    // without its image.
    let seen = if vision { member.scan()? } else { None };
    let (project, image) = match seen {
        Some((project, scan)) => (project, Some(sight(&scan)?)),
        None => (member.session.lock().project(), None),
    };
    let (blocks, texts): (Vec<_>, Vec<_>) = member
        .blocks(&project)
        .into_iter()
        .map(|(id, block)| {
            let text = block.source.content.text();
            ((id, block.source.language.clone()), text)
        })
        .filter(|(_, text)| !text.trim().is_empty())
        .unzip();
    let constraints = constraints(project.glossary(), &texts);
    Ok((!texts.is_empty()).then_some(Page {
        blocks,
        texts,
        constraints,
        image,
    }))
}

/// The `glossary` terms found in `texts`, ignoring case, as Index-Translate's
/// instTrans template writes constraints after the source
/// (bilibili/Index-Translate `docs/prompts.md`,
/// 93ac63774d1c6939aec05e6f0f191bf90f225365): a hard terminology mapping,
/// then the notes as a soft disambiguation. Only a page's own terms go, so a
/// long glossary doesn't crowd every turn, and they go with the page rather
/// than in the system prompt, which a local model's reused history follows.
fn constraints<'t>(glossary: impl Iterator<Item = &'t Term>, texts: &[String]) -> Option<String> {
    let text = texts.join("\n").to_lowercase();
    let terms: Vec<&Term> = glossary
        .filter(|term| text.contains(&term.source.to_lowercase()))
        .collect();
    if terms.is_empty() {
        return None;
    }
    let mapping = terms
        .iter()
        .map(|term| format!("{} -> {}", term.source, term.target))
        .collect::<Vec<_>>()
        .join("; ");
    let mut constraints = vec![format!("[Hard Constraint] Terminology mapping: {mapping}")];
    let notes = terms
        .iter()
        .filter_map(|term| Some(format!("{}: {}", term.source, term.note.as_ref()?)))
        .collect::<Vec<_>>();
    if !notes.is_empty() {
        constraints.push(format!("[Note] Disambiguation -- {}", notes.join("; ")));
    }
    let numbered = constraints
        .iter()
        .enumerate()
        .map(|(at, constraint)| format!("{}. {constraint}", at + 1))
        .collect::<Vec<_>>()
        .join("\n");
    Some(format!("[Constraints]\n{numbered}"))
}

/// The longest side of the page a model sees, in pixels: about 1,120 image
/// tokens to Index-Translate, above the 1,024 llama.cpp asks of Qwen-VL
/// models, where the whole scan took 4,050 and read no better.
const SIGHT: u32 = 1280;

/// `scan` at `SIGHT` as a JPEG in base 64. Each pixel averages the scan's
/// under it: a fraction of Lanczos's time, and the page is on the
/// translation's critical path.
fn sight(scan: &image::DynamicImage) -> Result<String> {
    let page = scan.thumbnail(SIGHT, SIGHT).to_rgb8();
    let mut bytes = Vec::new();
    page.write_to(
        &mut std::io::Cursor::new(&mut bytes),
        image::ImageFormat::Jpeg,
    )?;
    Ok(base64::engine::general_purpose::STANDARD.encode(bytes))
}

/// The ops writing each block's translation, placed, into the project as it
/// is now.
fn written(
    project: &Project,
    translated: Vec<(LayerId, LanguageIdentifier, String)>,
    config: &TranslatorConfig,
) -> Result<Vec<Op>> {
    let target = &config.target;
    // Each block's placement reads the project as it is before any of them,
    // so they're found in parallel: a balloon's room split is a flood over
    // its pixels, run for each of its texts.
    let placed: Vec<_> = translated
        .into_par_iter()
        .filter_map(|(block, source, text)| {
            let Ok(Content::Block(found)) = project.layer(block).map(|layer| &layer.content) else {
                return None;
            };
            // Its clean-up would erase the lettering for the translation to
            // redraw it in another hand.
            let unchanged = text.trim() == found.source.content.text().trim();
            let placed = (!unchanged).then(|| placement(project, block, target));
            Some((block, source, text, placed))
        })
        .collect();
    let mut ops = Vec::new();
    for (block, source, text, placed) in placed {
        let Some(placed) = placed else {
            ops.push(Op::UpdateLayer {
                layer: block.into(),
                patch: json!({"translation": null, "cleanup": {"visible": false}}),
            });
            continue;
        };
        ops.push(Op::SetText {
            layer: block.into(),
            text,
        });
        ops.push(placed?);
        ops.push(Op::UpdateLayer {
            layer: block.into(),
            patch: json!({"translation": {"language": target}, "cleanup": {"visible": true}}),
        });
        let parameters = json!({"sourceLanguage": source, "targetLanguage": target});
        ops.push(Op::SetAttribute {
            layer: block.into(),
            namespace: "c2pa".to_owned(),
            key: "translation".to_owned(),
            value: Some(action("c2pa.translated", &config.model, Some(parameters))),
        });
    }
    Ok(ops)
}

/// The target's plain English name from CLDR, such as Simplified Chinese.
/// Chinese is named by its script, which its region implies, and another
/// language drops a region its language implies, as ja-JP is Japanese. A
/// language CLDR doesn't name fails, rather than going in as its tag.
fn name(target: &LanguageIdentifier) -> Result<String> {
    let mut target = target.clone();
    let expander = LocaleExpander::new_extended();
    if target.language == language!("zh") {
        expander.maximize(&mut target);
        target.region = None;
    } else {
        expander.minimize(&mut target);
    }
    let english = DisplayNamesPreferences::from(&langid!("en"));
    let options = DisplayNamesOptions::default();
    let languages = LanguageDisplayNames::try_new(english, options)
        .context("couldn't load CLDR's language names")?;
    ensure!(
        languages.of(target.language).is_some(),
        "CLDR has no English name for {target}"
    );
    let locales = LocaleDisplayNamesFormatter::try_new(english, options)
        .context("couldn't load CLDR's locale names")?;
    Ok(locales.of(&Locale::from(target)).into_owned())
}

#[cfg(test)]
mod tests {
    use indoc::indoc;

    use super::*;

    #[test]
    fn a_language_cldr_doesnt_name_fails() {
        let language = LanguageIdentifier::try_from_str("xx").expect("a tag");
        assert!(name(&language).is_err());
    }

    #[test]
    fn only_the_terms_a_page_holds_go_with_it() {
        let term = |source: &str, target: &str, note: Option<&str>| Term {
            source: source.to_owned(),
            target: target.to_owned(),
            note: note.map(str::to_owned),
        };
        let glossary = [
            term("ピンク", "Pink", Some("a name")),
            term("アイギス", "Aegis", None),
            term("schale", "Schale", None),
        ];
        let texts = ["ピンク確保してっ！".to_owned(), "SCHALE".to_owned()];
        assert_eq!(
            constraints(glossary.iter(), &texts).expect("two terms"),
            indoc! {"
                [Constraints]
                1. [Hard Constraint] Terminology mapping: ピンク -> Pink; schale -> Schale
                2. [Note] Disambiguation -- ピンク: a name"}
        );
        assert_eq!(constraints(glossary.iter(), &["hello".to_owned()]), None);
    }

    #[test]
    fn instructions_follow_the_whole_template() {
        let template = system(&TranslatorConfig::default()).expect("English names it");
        let config = TranslatorConfig {
            instructions: indoc! {"

                シャーレはSchaleと訳す。

                敬称は残す。

            "}
            .to_owned(),
            ..Default::default()
        };
        assert_eq!(
            system(&config).expect("English names it"),
            formatdoc! {"
                {template}

                シャーレはSchaleと訳す。

                敬称は残す。",
                template = template,
            }
        );
    }
}
