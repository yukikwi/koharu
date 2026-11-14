use anyhow::{Context as _, Result};
use koharu_desktop::CanvasState;
use koharu_scene::{AssetInput, AssetMetadata, AssetRole, At, PageDraft};
use serde::{Deserialize, Serialize};
use specta::Type;
use strum::{EnumMessage as _, IntoEnumIterator as _};
use tauri::{Manager as _, State, WebviewWindow, ipc::Channel};
use tauri_runtime_cef::CefRuntime;
use walkdir::WalkDir;

use super::{
    ChannelExt as _, Error, import,
    preferences::Preferences,
    processing::Job,
    project::{Page, PageSummary, Project, ProjectInfo, ProjectLibrary, ProjectSummary},
    workspace::{Workspace, Workspaces, focus_window},
};

#[derive(Clone, Debug, Serialize, Type)]
pub struct StartupState {
    pub preferences: Preferences,
    pub jobs: Vec<Job>,
    pub canvas: CanvasState,
}

#[derive(Clone, Debug, Serialize, Type)]
pub struct PageSelection {
    pub project: ProjectInfo,
    pub page: Page,
}

#[derive(Clone, Copy, Debug, Deserialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum PageImportSource {
    Files,
    Folder,
}

pub(crate) struct Initialization {
    ready: tokio::sync::watch::Sender<bool>,
}

impl Default for Initialization {
    fn default() -> Self {
        let (ready, _) = tokio::sync::watch::channel(false);
        Self { ready }
    }
}

impl Initialization {
    pub(crate) fn ready(&self) {
        self.ready.send_replace(true);
    }

    async fn wait(&self) -> Result<()> {
        let mut ready = self.ready.subscribe();
        while !*ready.borrow_and_update() {
            ready
                .changed()
                .await
                .context("startup state closed before initialization completed")?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Type)]
pub struct Download {
    #[specta(type = f64)]
    pub id: u64,
    pub state: DownloadState,
    pub name: Option<String>,
    #[specta(type = f64)]
    pub completed: u64,
    #[specta(type = f64)]
    pub total: u64,
    pub error: Option<String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum DownloadState {
    Running,
    Finished,
    Failed,
}

#[derive(Clone, Debug, Default, Serialize, Type)]
pub struct ModelResources {
    #[specta(type = f64)]
    pub process_memory: u64,
    #[specta(type = f64)]
    pub system_memory: u64,
    pub process_cpu: f32,
    pub devices: Vec<DeviceResources>,
}

#[derive(Clone, Debug, Default, Serialize, Type)]
pub struct DeviceResources {
    pub name: String,
    pub selected: bool,
    #[specta(type = Option<f64>)]
    pub memory_budget: Option<u64>,
    #[specta(type = Option<f64>)]
    pub memory_used: Option<u64>,
    pub utilization: Option<f32>,
}

impl From<koharu_pipeline::ResourceSnapshot> for ModelResources {
    fn from(value: koharu_pipeline::ResourceSnapshot) -> Self {
        Self {
            process_memory: value.process_memory_bytes,
            system_memory: value.system_memory_bytes,
            process_cpu: value.process_cpu_percent,
            devices: value
                .devices
                .into_iter()
                .map(|device| DeviceResources {
                    name: device.name,
                    selected: device.selected,
                    memory_budget: device.memory_budget_bytes,
                    memory_used: device.memory_used_bytes,
                    utilization: device.utilization_percent,
                })
                .collect(),
        }
    }
}

#[tauri::command]
#[specta::specta]
pub(crate) async fn subscribe(
    window: WebviewWindow<CefRuntime>,
    on_canvas: Channel<CanvasState>,
    on_job: Channel<Job>,
    on_download: Channel<Download>,
    on_resources: Channel<ModelResources>,
    on_project: Channel<Option<ProjectInfo>>,
) -> std::result::Result<StartupState, Error> {
    window.state::<Initialization>().wait().await?;

    let workspace = Workspace::of(&window)?;
    *workspace.canvas_channel.lock() = Some(on_canvas);
    *workspace.job_channel.lock() = Some(on_job);
    *workspace.download_channel.lock() = Some(on_download);
    *workspace.resource_channel.lock() = Some(on_resources);
    *workspace.project_channel.lock() = Some(on_project);

    let canvas = workspace.desktop.canvas_state();
    let preferences = Preferences::load()?;
    Ok(StartupState {
        preferences,
        jobs: workspace.processing.jobs.lock().values().cloned().collect(),
        canvas,
    })
}

async fn replace_project(window: &WebviewWindow<CefRuntime>, opened: Project) -> Result<()> {
    let workspace = Workspace::of(window)?;
    let snapshot = opened.snapshot();
    let page = opened.active_page();
    let info = opened.info()?;
    let title = format!("{} - Koharu", opened.name);

    workspace.agent.reset().await;
    workspace.stop();

    let previous = workspace.project.lock().await.replace(opened);

    let desktop = &workspace.desktop;
    desktop.show_page(&snapshot, page).await?;
    let canvas = desktop.canvas_state();
    drop(previous);
    workspace.canvas_channel.publish(canvas);
    workspace.project_channel.publish(Some(info));
    window.set_title(&title)?;
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub(crate) async fn get_project(
    window: WebviewWindow<CefRuntime>,
) -> std::result::Result<Option<ProjectInfo>, Error> {
    let workspace = Workspace::of(&window)?;
    Ok(workspace
        .project
        .lock()
        .await
        .as_ref()
        .map(Project::info)
        .transpose()?)
}

#[tauri::command]
#[specta::specta]
pub(crate) async fn get_pages(
    window: WebviewWindow<CefRuntime>,
) -> std::result::Result<Vec<PageSummary>, Error> {
    let workspace = Workspace::of(&window)?;
    let snapshot = workspace
        .project
        .lock()
        .await
        .as_ref()
        .context("no project is open")?
        .snapshot();
    Ok(Project::pages(&snapshot)?)
}

#[tauri::command]
#[specta::specta]
pub(crate) async fn get_page(
    window: WebviewWindow<CefRuntime>,
) -> std::result::Result<Option<Page>, Error> {
    let workspace = Workspace::of(&window)?;
    let current = {
        let project = workspace.project.lock().await;
        project
            .as_ref()
            .map(|project| (project.snapshot(), project.active_page()))
    };
    Ok(current
        .and_then(|(snapshot, page)| page.map(|page| (snapshot, page)))
        .map(|(snapshot, page)| Project::page(&snapshot, page))
        .transpose()?)
}

#[tauri::command]
#[specta::specta]
pub(crate) async fn list_projects(
    library: State<'_, ProjectLibrary>,
) -> std::result::Result<Vec<ProjectSummary>, Error> {
    Ok(library.list()?)
}

/// Reads another project's glossary without opening it in a window. A project held by a
/// window is locked, so its glossary comes from that window's snapshot instead.
#[tauri::command]
#[specta::specta]
pub(crate) async fn get_project_glossary(
    name: String,
    window: WebviewWindow<CefRuntime>,
) -> std::result::Result<Vec<koharu_scene::GlossaryEntry>, Error> {
    for workspace in window.state::<Workspaces>().all() {
        let project = workspace.project.lock().await;
        if let Some(project) = project.as_ref().filter(|project| project.name == name) {
            return Ok(project.glossary()?);
        }
    }
    let library = window.state::<ProjectLibrary>().inner().clone();
    Ok(library.open(&name).await?.glossary()?)
}

#[tauri::command]
#[specta::specta]
pub(crate) async fn create_project(
    name: String,
    window: WebviewWindow<CefRuntime>,
) -> std::result::Result<(), Error> {
    let library = window.state::<ProjectLibrary>().inner().clone();
    let opened = library.create(&name).await?;
    replace_project(&window, opened).await?;
    Ok(())
}

/// A project is edited in one window at a time; opening it again focuses that window.
#[tauri::command]
#[specta::specta]
pub(crate) async fn open_project(
    name: String,
    window: WebviewWindow<CefRuntime>,
) -> std::result::Result<(), Error> {
    if let Some(label) = window
        .state::<Workspaces>()
        .find_project(&name, window.label())
        .await
    {
        focus_window(window.app_handle(), &label);
        return Ok(());
    }
    let library = window.state::<ProjectLibrary>().inner().clone();
    let opened = library.open(&name).await?;
    replace_project(&window, opened).await?;
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub(crate) async fn close_project(
    window: WebviewWindow<CefRuntime>,
) -> std::result::Result<(), Error> {
    close_current_project(&window).await?;
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub(crate) async fn delete_project(
    name: String,
    window: WebviewWindow<CefRuntime>,
) -> std::result::Result<(), Error> {
    for workspace in window.state::<Workspaces>().all() {
        if workspace.project_name().await.as_deref() == Some(name.as_str())
            && let Some(owner) = window.app_handle().get_webview_window(&workspace.label)
        {
            close_current_project(&owner).await?;
        }
    }
    let library = window.state::<ProjectLibrary>().inner().clone();
    tokio::task::spawn_blocking(move || library.delete(&name))
        .await
        .context("project deletion task failed")??;
    Ok(())
}

async fn close_current_project(window: &WebviewWindow<CefRuntime>) -> Result<()> {
    let workspace = Workspace::of(window)?;
    workspace.agent.reset().await;
    workspace.stop();
    let previous = workspace.project.lock().await.take();
    let desktop = &workspace.desktop;
    desktop.clear().await;
    let result = desktop.canvas_state();
    drop(previous);
    workspace.canvas_channel.publish(result);
    workspace.project_channel.publish(None);
    window.set_title("Koharu")?;
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub(crate) async fn import(
    source: PageImportSource,
    window: WebviewWindow<CefRuntime>,
) -> std::result::Result<(), Error> {
    let workspace = Workspace::of(&window)?;
    let desktop = &workspace.desktop;
    let processing = &workspace.processing;
    if !processing.stops.lock().is_empty() {
        return Err(anyhow::anyhow!("pages cannot be imported while processing is running").into());
    }
    let extensions = import::Format::iter()
        .flat_map(|format| format.get_serializations())
        .collect::<Vec<_>>();
    let dialog = rfd::AsyncFileDialog::new()
        .add_filter("Images, archives, and PDF", &extensions)
        .set_parent(&window);
    let files = match source {
        PageImportSource::Files => dialog.pick_files().await.map(|files| {
            files
                .into_iter()
                .map(|file| file.path().to_owned())
                .collect::<Vec<_>>()
        }),
        PageImportSource::Folder => dialog.pick_folder().await.map(|folder| {
            WalkDir::new(folder.path())
                .follow_links(false)
                .into_iter()
                .filter_map(|entry| match entry {
                    Ok(entry) if entry.file_type().is_file() => Some(entry.into_path()),
                    Ok(_) => None,
                    Err(error) => {
                        tracing::warn!(%error, "could not inspect an import directory entry");
                        None
                    }
                })
                .filter(|path| {
                    path.extension()
                        .and_then(|extension| extension.to_str())
                        .is_some_and(|extension| extension.parse::<import::Format>().is_ok())
                })
                .collect::<Vec<_>>()
        }),
    };
    let Some(files) = files else {
        return Ok(());
    };
    if files.is_empty() {
        return Err(anyhow::anyhow!("no supported images were found in the selection").into());
    }
    let pages = tokio_rayon::spawn(move || import::import(files)).await?;

    let (commit, page) = {
        let mut project = workspace.project.lock().await;
        let project = project.as_mut().context("no project is open")?;
        let source = AssetRole::new("source")?;
        let patch = project.snapshot().patch(|edit| {
            for imported in pages {
                let page = edit.add_page(
                    PageDraft::new(
                        imported.name,
                        f64::from(imported.width),
                        f64::from(imported.height),
                    ),
                    At::End,
                )?;
                edit.set_asset(
                    page,
                    &source,
                    AssetInput::new(
                        imported.bytes,
                        imported.format.to_mime_type(),
                        AssetMetadata {
                            width: Some(imported.width),
                            height: Some(imported.height),
                            attributes: Default::default(),
                        },
                    ),
                )?;
            }
            Ok(())
        })?;
        let commit = project.session.commit(patch).await?;
        project.record(vec![commit.revision]);
        project.reconcile_page();
        let page = project.active_page();
        (commit, page)
    };
    desktop.synchronize(&commit.snapshot, page, &commit).await?;
    let canvas = desktop.canvas_state();
    workspace.canvas_channel.publish(canvas);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub(crate) async fn select_page(
    page: koharu_scene::EntityId,
    window: WebviewWindow<CefRuntime>,
) -> std::result::Result<PageSelection, Error> {
    let workspace = Workspace::of(&window)?;
    let desktop = &workspace.desktop;
    let (snapshot, project_info, selected_page) = {
        let mut project = workspace.project.lock().await;
        let project = project.as_mut().context("no project is open")?;
        project.select_page(page)?;
        let snapshot = project.snapshot();
        let project_info = project.info()?;
        let selected_page = Project::page(&snapshot, page)?;
        (snapshot, project_info, selected_page)
    };
    if desktop.show_page(&snapshot, Some(page)).await? {
        let canvas = desktop.canvas_state();
        workspace.canvas_channel.publish(canvas);
    }
    Ok(PageSelection {
        project: project_info,
        page: selected_page,
    })
}
