//! Native FLUX.3 Image Klein model assembly.
//!
//! Component mapping follows stable-diffusion.cpp at:
//! https://github.com/leejet/stable-diffusion.cpp/blob/b9d7176bf88fe465576e0aafbe47ba253ffcfc2c/docs/flux3.md
//! The DiT selects the Qwen3-VL text encoder, the FLUX.2 VAE, the `flux2`
//! reference-image preset, and the discrete scheduler with a 6.93 flow shift:
//! https://github.com/leejet/stable-diffusion.cpp/blob/b9d7176bf88fe465576e0aafbe47ba253ffcfc2c/src/model/integrations/flux3.cpp#L18-L37

use std::{path::PathBuf, sync::Mutex, time::Instant};

use anyhow::{Context as _, Result, anyhow, ensure};
use koharu_diffusion::{Context, ContextParams, ImageGenerationParams, RgbImage, VaeFormat};

use crate::Backend;

/// The Q4_K DiT, the Q4_K_M Qwen3-VL-4B encoder, and the VAE hold about
/// 6 GiB of weights. Smaller accelerators keep source parameters in RAM.
const RESIDENT_PARAMETER_BYTES: usize = 12 * 1024 * 1024 * 1024;

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
            "creating FLUX.3 Klein context"
        );
        let started = Instant::now();
        let context = Context::new(&params).context("failed to load FLUX.3 Klein components")?;
        tracing::info!(
            elapsed_ms = started.elapsed().as_millis(),
            "FLUX.3 Klein context created"
        );
        ensure!(
            context.supports_image_generation(),
            "the loaded FLUX.3 Klein context does not support image generation"
        );
        Ok(Self {
            context: Mutex::new(context),
        })
    }

    pub fn forward(&self, params: &ImageGenerationParams) -> Result<Vec<RgbImage>> {
        let mut context = self
            .context
            .lock()
            .map_err(|_| anyhow!("FLUX.3 Klein context lock was poisoned"))?;
        tracing::debug!("FLUX.3 Klein request:\n{params}");
        let started = Instant::now();
        let images = context
            .generate_image(params)
            .context("FLUX.3 Klein inference failed")?;
        tracing::info!(
            elapsed_ms = started.elapsed().as_millis(),
            images = images.len(),
            "FLUX.3 Klein generation finished"
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
        diffusion_conv_direct: use_cuda,
        vae_conv_direct: use_cuda,
        // The small-decoder checkpoint shares the FLUX.2 latent space; naming
        // the format lets the loader infer its narrower decoder.
        vae_format: VaeFormat::Flux2,
        backend: Some(device.name.clone()),
        // Text encoding, denoising, and decoding run in separate phases, so
        // parameters staged from RAM only occupy the accelerator per phase.
        params_backend: (use_accelerator && !keep_parameters_resident).then(|| "*=cpu".to_owned()),
        ..ContextParams::default()
    }
}
