use std::{collections::HashMap, sync::Arc};

use anyhow::{Context as _, Result};
use koharu_agent::Codex;
use koharu_desktop::{CanvasState, Desktop};
use parking_lot::Mutex;
use tauri::{AppHandle, Manager as _, WebviewWindow, ipc::Channel};
use tauri_runtime_cef::CefRuntime;

use super::{
    Error,
    agent::AgentState,
    lifecycle::{Download, ModelResources},
    processing::{Job, Processing},
    project::{Project, ProjectInfo},
};

const MAIN_WINDOW: &str = "main";
const PROJECT_WINDOW_PREFIX: &str = "project-";

/// Everything one editor window owns. Each window edits at most one project, so the
/// project, its presentation, jobs, Agent conversation, and UI subscriptions live
/// together and disappear with the window. The pipeline and its admission lane stay
/// application-wide, so jobs from different windows still run one at a time.
pub(crate) struct Workspace {
    pub(crate) label: String,
    pub(crate) project: tokio::sync::Mutex<Option<Project>>,
    pub(crate) desktop: Desktop,
    pub(crate) processing: Processing,
    pub(crate) agent: AgentState,
    pub(crate) canvas_channel: Mutex<Option<Channel<CanvasState>>>,
    pub(crate) job_channel: Mutex<Option<Channel<Job>>>,
    pub(crate) download_channel: Mutex<Option<Channel<Download>>>,
    pub(crate) resource_channel: Mutex<Option<Channel<ModelResources>>>,
    pub(crate) project_channel: Mutex<Option<Channel<Option<ProjectInfo>>>>,
}

impl Workspace {
    pub(crate) fn of(window: &WebviewWindow<CefRuntime>) -> Result<Arc<Self>> {
        window.state::<Workspaces>().get(window.label())
    }

    /// Stops this window's work. Running jobs observe the stop cooperatively and
    /// discard their uncommitted result.
    pub(crate) fn stop(&self) {
        let mut stops = self.processing.stops.lock();
        for stop in stops.values() {
            stop.stop();
        }
        stops.clear();
        self.processing.jobs.lock().clear();
    }

    pub(crate) async fn project_name(&self) -> Option<String> {
        self.project
            .lock()
            .await
            .as_ref()
            .map(|project| project.name.clone())
    }
}

pub(crate) struct Workspaces {
    codex: Codex,
    windows: Mutex<HashMap<String, Arc<Workspace>>>,
}

impl Workspaces {
    pub(crate) fn new() -> Result<Self> {
        Ok(Self {
            codex: Codex::new()?,
            windows: Mutex::new(HashMap::new()),
        })
    }

    pub(crate) fn get(&self, label: &str) -> Result<Arc<Workspace>> {
        self.windows
            .lock()
            .get(label)
            .cloned()
            .with_context(|| format!("window {label} has no open workspace"))
    }

    pub(crate) fn all(&self) -> Vec<Arc<Workspace>> {
        self.windows.lock().values().cloned().collect()
    }

    pub(crate) fn remove(&self, label: &str) -> Option<Arc<Workspace>> {
        self.windows.lock().remove(label)
    }

    /// Finds the window, other than `except`, that currently has `name` open.
    pub(crate) async fn find_project(&self, name: &str, except: &str) -> Option<String> {
        for workspace in self.all() {
            if workspace.label != except && workspace.project_name().await.as_deref() == Some(name)
            {
                return Some(workspace.label.clone());
            }
        }
        None
    }

    fn insert(&self, handle: &AppHandle<CefRuntime>, label: String) -> Result<()> {
        let workspace = Arc::new(Workspace {
            label: label.clone(),
            project: tokio::sync::Mutex::new(None),
            desktop: Desktop::new()?,
            processing: Processing::default(),
            agent: AgentState::new(self.codex.clone(), handle.clone(), label.clone())?,
            canvas_channel: Mutex::new(None),
            job_channel: Mutex::new(None),
            download_channel: Mutex::new(None),
            resource_channel: Mutex::new(None),
            project_channel: Mutex::new(None),
        });
        self.windows.lock().insert(label, workspace);
        Ok(())
    }
}

/// Creates an editor window from the main window configuration. Additional windows
/// reuse the lowest free label so their saved geometry is restored on reuse.
pub(crate) fn create_window(handle: &AppHandle<CefRuntime>, main: bool) -> Result<()> {
    let label = if main {
        MAIN_WINDOW.to_owned()
    } else {
        (1_u32..)
            .map(|index| format!("{PROJECT_WINDOW_PREFIX}{index}"))
            .find(|label| {
                handle.get_webview_window(label).is_none()
                    && !handle
                        .state::<Workspaces>()
                        .windows
                        .lock()
                        .contains_key(label)
            })
            .expect("an unused window label exists")
    };
    let mut config = handle
        .config()
        .app
        .windows
        .iter()
        .find(|window| window.label == MAIN_WINDOW)
        .context("the main Tauri window configuration is unavailable")?
        .clone();
    config.label.clone_from(&label);
    handle.state::<Workspaces>().insert(handle, label.clone())?;
    let window = tauri::WebviewWindowBuilder::from_config(handle, &config)?
        .build()
        .with_context(|| format!("failed to create window {label}"));
    let window = match window {
        Ok(window) => window,
        Err(error) => {
            handle.state::<Workspaces>().remove(&label);
            return Err(error);
        }
    };
    window.show().context("failed to show the window")?;
    window.set_focus().context("failed to focus the window")?;
    Ok(())
}

pub(crate) fn focus_window(handle: &AppHandle<CefRuntime>, label: &str) {
    if let Some(window) = handle.get_webview_window(label) {
        let _ = window.unminimize();
        let _ = window.show();
        let _ = window.set_focus();
    }
}

#[tauri::command]
#[specta::specta]
pub(crate) async fn open_window(handle: AppHandle<CefRuntime>) -> std::result::Result<(), Error> {
    Ok(create_window(&handle, false)?)
}
