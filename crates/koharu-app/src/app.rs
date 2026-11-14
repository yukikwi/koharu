use anyhow::{Context as _, Result};
use tauri::{AppHandle, Manager as _, WindowEvent};
use tauri_runtime_cef::{Cef, CefRuntime};

use crate::commands::{
    ChannelExt as _,
    lifecycle::{Download, DownloadState, Initialization, ModelResources},
    project::ProjectLibrary,
    workspace::{self, Workspaces},
};

pub(crate) async fn initialize(handle: AppHandle<CefRuntime>) -> Result<()> {
    koharu_ml::init()
        .await
        .context("failed to initialize the ML runtime")?;
    let device = koharu_ml::device(false);
    let pipeline = koharu_pipeline::Pipeline::load(device)?;
    handle.manage(pipeline.clone());

    let mut resources = pipeline.subscribe_resources();
    let resource_handle = handle.clone();
    drop(tauri::async_runtime::spawn(async move {
        while resources.changed().await.is_ok() {
            let snapshot = ModelResources::from(resources.borrow_and_update().clone());
            for workspace in resource_handle.state::<Workspaces>().all() {
                workspace.resource_channel.publish(snapshot.clone());
            }
        }
    }));
    Ok(())
}

pub fn run(context: tauri::Context<CefRuntime>) -> Result<()> {
    let cef = Cef::default();
    #[cfg(debug_assertions)]
    let cef = cef.remote_debugging(tauri_runtime_cef::RemoteDebugging::Port {
        port: 4000,
        allowed_origins: Vec::new(),
    });
    #[cfg(target_os = "linux")]
    let cef = cef
        .enable_features(["Vulkan", "VulkanFromANGLE"])
        .command_line_args([
            ("--enable-unsafe-webgpu", None),
            ("use-angle", Some("vulkan")),
            ("--ozone-platform", Some("x11")),
        ]);
    tauri::Builder::<CefRuntime>::new()
        .runtime(cef)
        .plugin(
            tauri_plugin_log::Builder::new()
                .level(tauri_plugin_log::log::LevelFilter::Info)
                .max_file_size(1_000_000)
                .clear_targets()
                .target(tauri_plugin_log::Target::new(
                    tauri_plugin_log::TargetKind::LogDir { file_name: None },
                ))
                .build(),
        )
        .plugin(tauri_plugin_single_instance::init(|handle, _, _| {
            if let Some(label) = handle.webview_windows().into_keys().next() {
                workspace::focus_window(handle, &label);
            }
        }))
        .plugin(
            tauri_plugin_window_state::Builder::default()
                .with_state_flags(
                    tauri_plugin_window_state::StateFlags::SIZE
                        | tauri_plugin_window_state::StateFlags::POSITION
                        | tauri_plugin_window_state::StateFlags::MAXIMIZED
                        | tauri_plugin_window_state::StateFlags::FULLSCREEN,
                )
                .build(),
        )
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_process::init())
        .plugin(tauri_plugin_updater::Builder::new().build())
        .invoke_handler(crate::commands::bindings().invoke_handler())
        .setup(move |application| {
            #[cfg(all(target_os = "windows", not(debug_assertions)))]
            koharu_runtime::Store::configure(
                application
                    .path()
                    .resource_dir()
                    .context("failed to locate Koharu's installation directory")?
                    .join("store"),
            )?;

            application.manage(ProjectLibrary::new()?);
            application.manage(Initialization::default());
            application.manage(Workspaces::new()?);

            let handle = application.handle().clone();
            workspace::create_window(&handle, true)?;
            let initialization_handle = handle.clone();
            drop(tauri::async_runtime::spawn(async move {
                initialize(initialization_handle.clone())
                    .await
                    .expect("failed to initialize the desktop runtime");
                initialization_handle.state::<Initialization>().ready();
            }));

            let mut downloads = koharu_runtime::download::subscribe();
            let download_handle = handle.clone();
            drop(tauri::async_runtime::spawn(async move {
                loop {
                    match downloads.recv().await {
                        Ok(event) => {
                            let download = match event {
                                koharu_runtime::download::Event::Started { id, name } => Download {
                                    id,
                                    state: DownloadState::Running,
                                    name: Some(name),
                                    completed: 0,
                                    total: 0,
                                    error: None,
                                },
                                koharu_runtime::download::Event::Progress {
                                    id,
                                    name,
                                    completed,
                                    total,
                                } => Download {
                                    id,
                                    state: DownloadState::Running,
                                    name: Some(name),
                                    completed,
                                    total,
                                    error: None,
                                },
                                koharu_runtime::download::Event::Finished { id } => Download {
                                    id,
                                    state: DownloadState::Finished,
                                    name: None,
                                    completed: 0,
                                    total: 0,
                                    error: None,
                                },
                                koharu_runtime::download::Event::Failed { id, name, error } => {
                                    Download {
                                        id,
                                        state: DownloadState::Failed,
                                        name: Some(name),
                                        completed: 0,
                                        total: 0,
                                        error: Some(error),
                                    }
                                }
                            };
                            for workspace in download_handle.state::<Workspaces>().all() {
                                workspace.download_channel.publish(download.clone());
                            }
                        }
                        Err(tokio::sync::broadcast::error::RecvError::Lagged(skipped)) => {
                            tracing::warn!(skipped, "download channel fell behind");
                        }
                        Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
                    }
                }
            }));

            Ok(())
        })
        .on_window_event(|window, event| {
            // The window's project closes once its last in-flight command or job
            // releases the workspace.
            if matches!(event, WindowEvent::Destroyed)
                && let Some(workspace) = window.state::<Workspaces>().remove(window.label())
            {
                workspace.stop();
                workspace.agent.cancel_all();
            }
        })
        .run(context)?;
    Ok(())
}
