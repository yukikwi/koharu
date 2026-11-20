//! The fonts the Inspector offers. The bundled ones' catalog and faces are
//! downloaded on first use.

use koharu_renderer::fonts::{self, FontFamily};
use serde::{Deserialize, Serialize};
use specta::Type;

use super::{Bytes, Result};

/// `[fonts]`: how the font pickers list the fonts. The renderer never reads
/// it.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, Type)]
#[serde(default)]
pub struct FontsConfig {
    /// Families listed first in every font picker, in the order pinned.
    pub pinned: Vec<String>,
}

impl koharu_config::Section for FontsConfig {
    const NAME: &'static str = "fonts";
}

#[tauri::command]
#[specta::specta]
pub async fn list_fonts() -> Result<Vec<FontFamily>> {
    Ok(fonts::list().await?)
}

/// `family`'s name set in it, as WebP.
#[tauri::command]
#[specta::specta]
pub async fn get_font_preview(family: String) -> Result<Bytes> {
    Ok(Bytes(fonts::preview(&family).await?))
}
