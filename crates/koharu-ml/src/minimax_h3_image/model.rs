//! Native MiniMax-H3-Image model assembly.
//!
//! Component mapping follows stable-diffusion.cpp at:
//! https://github.com/leejet/stable-diffusion.cpp/blob/b9d7176bf88fe465576e0aafbe47ba253ffcfc2c/docs/minimax_h3_image.md
//! The DiT is identified by `video_patch_proj` and `condition_proj` without
//! `audio_patch_proj`, which also requires the dedicated 2D image VAE:
//! https://github.com/leejet/stable-diffusion.cpp/blob/b9d7176bf88fe465576e0aafbe47ba253ffcfc2c/src/model_loader.cpp#L579-L583
//! https://github.com/leejet/stable-diffusion.cpp/blob/b9d7176bf88fe465576e0aafbe47ba253ffcfc2c/src/pipeline/model_builders.cpp#L162-L169

use std::{path::PathBuf, sync::Mutex, time::Instant};

use anyhow::{Context as _, Result, anyhow, ensure};
use koharu_diffusion::{Context, ContextParams, ImageGenerationParams, RgbImage};

use crate::Backend;

/// The DiT, the Qwen3-VL-32B text encoder, and the VAE together hold about
/// 31 GiB of weights. Smaller accelerators keep source parameters in RAM.
const RESIDENT_PARAMETER_BYTES: usize = 40 * 1024 * 1024 * 1024;

#[derive(Debug)]
pub(super) struct ModelPaths {
    pub transformer: PathBuf,
    pub text_encoder: PathBuf,
    pub vae: PathBuf,
}

#[derive(Debug)]
pub(super) struct Model {
    context: Mutex<Context>,
}

impl Model {
    pub fn new(device: &crate::Device, paths: ModelPaths) -> Result<Self> {
        let params = context_params(device, paths);
        tracing::info!(
            device = %device.name,
            backend = ?device.backend,
            memory_total_mib = device.memory_total / (1024 * 1024),
            memory_free_mib = device.memory_free / (1024 * 1024),
            params_backend = ?params.params_backend,
            flash_attention = params.diffusion_flash_attention,
            "creating MiniMax-H3-Image context"
        );
        let started = Instant::now();
        let context = Context::new(&params).context("failed to load MiniMax-H3-Image components")?;
        tracing::info!(
            elapsed_ms = started.elapsed().as_millis(),
            "MiniMax-H3-Image context created"
        );
        ensure!(
            context.supports_image_generation(),
            "the loaded MiniMax-H3-Image context does not support image generation"
        );
        Ok(Self {
            context: Mutex::new(context),
        })
    }

    pub fn forward(&self, params: &ImageGenerationParams) -> Result<Vec<RgbImage>> {
        let mut context = self
            .context
            .lock()
            .map_err(|_| anyhow!("MiniMax-H3-Image context lock was poisoned"))?;
        tracing::debug!("MiniMax-H3-Image request:\n{params}");
        let started = Instant::now();
        let images = context
            .generate_image(params)
            .context("MiniMax-H3-Image inference failed")?;
        tracing::info!(
            elapsed_ms = started.elapsed().as_millis(),
            images = images.len(),
            "MiniMax-H3-Image generation finished"
        );
        Ok(images)
    }
}

fn context_params(device: &crate::Device, paths: ModelPaths) -> ContextParams {
    let use_accelerator = device.backend != Backend::Cpu;
    let use_cuda = device.backend == Backend::Cuda;
    let keep_parameters_resident =
        use_accelerator && device.memory_free >= RESIDENT_PARAMETER_BYTES;
    ContextParams {
        diffusion_model_path: Some(paths.transformer),
        llm_path: Some(paths.text_encoder),
        vae_path: Some(paths.vae),
        enable_mmap: true,
        flash_attention: use_cuda,
        diffusion_flash_attention: use_cuda,
        backend: Some(device.name.clone()),
        // Text encoding, denoising, and decoding run in separate phases, so
        // parameters staged from RAM only occupy the accelerator per phase.
        params_backend: (use_accelerator && !keep_parameters_resident).then(|| "*=cpu".to_owned()),
        ..ContextParams::default()
    }
}
