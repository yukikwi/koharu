//! Inference-only Manga OCR AR network.
//!
//! Ported from `Encoder`, `Memory`, `ARBlock`, and `Model.greedy_cached`:
//! https://huggingface.co/muscgab/manga-ocr-ar-experimental/blob/20208154a8d34dc6e5fc13ba81bc35f0a2f1a26e/infer.py
//! The encoder keeps the MonkeyOCRv2-B vision tower parameter layout.

use std::path::Path;

use anyhow::{Result, ensure};
use koharu_torch::{
    Device, Kind, Tensor,
    nn::{self, Module},
};

use super::{config::MangaOcrArConfig, processor::PATCH};

/// `RMSNorm` default epsilon used by the bridge, memory, and decoder norms.
const RMS_EPS: f64 = 1e-6;
const ROPE_THETA: f64 = 10_000.0;
const BRIDGE_DIM: i64 = 1024;
/// Two learned null memory slots precede the image tokens.
const NULL_SLOTS: i64 = 2;
const FOURIER_BANDS: i64 = 8;

#[derive(Debug)]
pub(super) struct Model {
    vs: nn::VarStore,
    encoder: Encoder,
    memory: Memory,
    blocks: Vec<ArBlock>,
    tok: nn::Embedding,
    final_norm: RmsNorm,
    cos: Tensor,
    sin: Tensor,
    eos: i64,
    bos: i64,
    slots: i64,
}

impl Model {
    pub(super) fn new(config: &MangaOcrArConfig, characters: i64, device: Device) -> Result<Self> {
        let mut vs = nn::VarStore::new(device);
        crate::backend::set_precision(&mut vs);
        let root = vs.root();
        let encoder = Encoder::new(&(&root / "encoder"), config);
        let memory = Memory::new(&(&root / "memory"), config);
        let blocks = (0..config.decoder.layers)
            .map(|index| {
                ArBlock::new(
                    &(&root / "blocks" / index),
                    config.decoder.dim,
                    config.decoder.heads,
                    config.decoder.ffn,
                )
            })
            .collect();
        // Outputs are characters + EOS; BOS is an input-only embedding row.
        let tok = nn::embedding(
            &root / "tok",
            characters + 2,
            config.decoder.dim,
            Default::default(),
        );
        let final_norm = RmsNorm::new(&(&root / "final_norm"), config.decoder.dim, RMS_EPS);

        let slots = config.max_length + 1;
        let head_dim = config.decoder.dim / config.decoder.heads;
        let angles = Tensor::arange(slots, (Kind::Float, device))
            .outer(&inverse_frequencies(head_dim, device));
        let angles = Tensor::cat(&[&angles, &angles], -1).view([1, slots, 1, head_dim]);

        vs.freeze();
        Ok(Self {
            vs,
            encoder,
            memory,
            blocks,
            tok,
            final_norm,
            cos: angles.cos(),
            sin: angles.sin(),
            eos: characters,
            bos: characters + 1,
            slots,
        })
    }

    pub(super) fn load(&mut self, path: impl AsRef<Path>) -> Result<()> {
        self.vs.load(path)?;
        crate::backend::set_precision(&mut self.vs);
        Ok(())
    }

    /// Greedy decoding with a self-attention KV cache; returns character ids
    /// without the terminating EOS.
    pub(super) fn generate(&self, patches: &Tensor, [h, w]: [i64; 2]) -> Result<Vec<i64>> {
        let device = patches.device();
        ensure!(
            patches.size() == [1, h * w, 3 * PATCH * PATCH],
            "Manga OCR AR expects one flattened {h}x{w} patch grid"
        );
        let index = Tensor::arange(h * w, (Kind::Int64, device));
        let rows = index.floor_divide_scalar(w);
        let cols = index.remainder(w);
        let states = self
            .encoder
            .forward(&patches.to_kind(self.vs.kind()), &rows, &cols);
        let memories = self.memory.forward(&states, &rows, &cols, h, w);
        let cross_kv = self
            .blocks
            .iter()
            .zip(&memories)
            .map(|(block, memory)| block.memory_kv(memory))
            .collect::<Vec<_>>();

        let mut caches: Vec<Option<(Tensor, Tensor)>> =
            (0..self.blocks.len()).map(|_| None).collect();
        let mut tokens = Vec::new();
        let mut token = Tensor::full([1], self.bos, (Kind::Int64, device));
        let output_weight = self.tok.ws.narrow(0, 0, self.eos + 1);
        for t in 0..self.slots {
            let mut x = self.tok.forward(&token).unsqueeze(0);
            let cos = self.cos.narrow(1, t, 1);
            let sin = self.sin.narrow(1, t, 1);
            for ((block, kv), cache) in self.blocks.iter().zip(&cross_kv).zip(&mut caches) {
                x = block.step(&x, kv, (&cos, &sin), cache);
            }
            let logits = self
                .final_norm
                .forward(&x.select(1, 0))
                .linear(&output_weight, None::<Tensor>);
            token = logits.argmax(-1, false);
            let next = token.int64_value(&[0]);
            if next == self.eos {
                break;
            }
            tokens.push(next);
        }
        Ok(tokens)
    }
}

fn inverse_frequencies(dim: i64, device: Device) -> Tensor {
    let exponents = Tensor::arange_start_step(0, dim, 2, (Kind::Float, device)) / dim as f64;
    Tensor::pow_scalar(ROPE_THETA, &exponents).reciprocal()
}

fn rotate_half(x: &Tensor) -> Tensor {
    let halves = x.chunk(2, -1);
    Tensor::cat(&[-&halves[1], halves[0].shallow_clone()], -1)
}

/// Applies non-interleaved RoPE in fp32 and casts back to the input kind.
fn apply_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Tensor {
    let y = x.to_kind(Kind::Float);
    (&y * cos + rotate_half(&y) * sin).to_kind(x.kind())
}

fn no_bias_linear(path: nn::Path<'_>, input: i64, output: i64) -> nn::Linear {
    nn::linear(
        path,
        input,
        output,
        nn::LinearConfig {
            bias: false,
            ..Default::default()
        },
    )
}

/// `F.rms_norm` over fp32 input and weight, cast back to the input kind.
#[derive(Debug)]
struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    fn new(path: &nn::Path<'_>, dim: i64, eps: f64) -> Self {
        Self {
            weight: path.var("weight", &[dim], nn::Init::Const(1.0)),
            eps,
        }
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let dim = *x.size().last().expect("RMSNorm input has a feature axis");
        x.to_kind(Kind::Float)
            .rms_norm([dim], Some(self.weight.to_kind(Kind::Float)), self.eps)
            .to_kind(x.kind())
    }
}

/// MonkeyOCRv2 vision RMSNorm: normalize in fp32, cast back, then scale.
#[derive(Debug)]
struct VitRmsNorm {
    weight: Tensor,
    eps: f64,
}

impl VitRmsNorm {
    fn new(path: &nn::Path<'_>, dim: i64, eps: f64) -> Self {
        Self {
            weight: path.var("weight", &[dim], nn::Init::Const(1.0)),
            eps,
        }
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let y = x.to_kind(Kind::Float);
        let variance = y
            .pow_tensor_scalar(2.0)
            .mean_dim(&[-1i64][..], true, Some(Kind::Float));
        (y * (variance + self.eps).rsqrt()).to_kind(x.kind()) * &self.weight
    }
}

#[derive(Debug)]
struct Encoder {
    /// The patchifier convolution runs as a linear layer over host-flattened
    /// patches, so its `(dim, 3, 14, 14)` weight is flattened at use.
    patch_weight: Tensor,
    patch_bias: Tensor,
    patch_norm: VitRmsNorm,
    blocks: Vec<VitBlock>,
    post_trunk_norm: VitRmsNorm,
    inv_freq: Tensor,
    dim: i64,
    heads: i64,
}

impl Encoder {
    fn new(path: &nn::Path<'_>, config: &MangaOcrArConfig) -> Self {
        let encoder = &config.encoder;
        let vit = path / "vit";
        let patchifier = &vit / "patch_embed" / "patchifier";
        let proj = &patchifier / "proj";
        let head_dim = encoder.dim / encoder.heads;
        Self {
            patch_weight: proj.var(
                "weight",
                &[encoder.dim, 3, PATCH, PATCH],
                nn::Init::Const(0.0),
            ),
            patch_bias: proj.var("bias", &[encoder.dim], nn::Init::Const(0.0)),
            patch_norm: VitRmsNorm::new(&(&patchifier / "norm"), encoder.dim, encoder.rms_eps),
            blocks: (0..encoder.layers)
                .map(|index| VitBlock::new(&(&vit / "blocks" / index), encoder))
                .collect(),
            post_trunk_norm: VitRmsNorm::new(
                &(&vit / "post_trunk_norm"),
                encoder.dim,
                encoder.rms_eps,
            ),
            // Each half of a head rotates by row and column positions.
            inv_freq: inverse_frequencies(head_dim / 2, path.device()),
            dim: encoder.dim,
            heads: encoder.heads,
        }
    }

    /// Returns `[h0, ..., h11, post-norm h12]`.
    fn forward(&self, patches: &Tensor, rows: &Tensor, cols: &Tensor) -> Vec<Tensor> {
        let size = patches.size();
        let (batch, length) = (size[0], size[1]);
        let x = patches.linear(&self.patch_weight.flatten(1, -1), Some(&self.patch_bias));
        let mut x = self.patch_norm.forward(&x);

        let angles = Tensor::stack(
            &[
                rows.to_kind(Kind::Float).outer(&self.inv_freq),
                cols.to_kind(Kind::Float).outer(&self.inv_freq),
            ],
            -1,
        )
        .flatten(-2, -1);
        let angles = Tensor::cat(&[&angles, &angles], -1).view([1, length, 1, -1]);
        let (cos, sin) = (angles.cos(), angles.sin());

        let head_dim = self.dim / self.heads;
        let mut states = Vec::with_capacity(self.blocks.len() + 1);
        states.push(x.shallow_clone());
        for block in &self.blocks {
            let qkv = block
                .qkv
                .forward(&block.norm1.forward(&x))
                .view([batch, length, 3, self.heads, head_dim]);
            let query = apply_rope(&qkv.select(2, 0), &cos, &sin).transpose(1, 2);
            let key = apply_rope(&qkv.select(2, 1), &cos, &sin).transpose(1, 2);
            let value = qkv.select(2, 2).transpose(1, 2);
            let attended = Tensor::scaled_dot_product_attention(
                &query,
                &key,
                &value,
                None::<Tensor>,
                0.0,
                false,
                None,
                false,
            )
            .transpose(1, 2)
            .reshape([batch, length, self.dim]);
            x = &x + block.proj.forward(&attended);
            let y = block.norm2.forward(&x);
            x = &x
                + block
                    .fc2
                    .forward(&(block.fc1.forward(&y).silu() * block.fc3.forward(&y)));
            states.push(x.shallow_clone());
        }
        let last = states.len() - 1;
        states[last] = self.post_trunk_norm.forward(&x);
        states
    }
}

#[derive(Debug)]
struct VitBlock {
    norm1: VitRmsNorm,
    qkv: nn::Linear,
    proj: nn::Linear,
    norm2: VitRmsNorm,
    fc1: nn::Linear,
    fc2: nn::Linear,
    fc3: nn::Linear,
}

impl VitBlock {
    fn new(path: &nn::Path<'_>, encoder: &super::config::EncoderConfig) -> Self {
        let (dim, ffn) = (encoder.dim, encoder.ffn);
        let attn = path / "attn";
        let mlp = path / "mlp";
        Self {
            norm1: VitRmsNorm::new(&(path / "norm1"), dim, encoder.rms_eps),
            qkv: no_bias_linear(&attn / "qkv", dim, 3 * dim),
            proj: no_bias_linear(&attn / "proj", dim, dim),
            norm2: VitRmsNorm::new(&(path / "norm2"), dim, encoder.rms_eps),
            fc1: no_bias_linear(&mlp / "fc1", dim, ffn),
            fc2: no_bias_linear(&mlp / "fc2", ffn, dim),
            fc3: no_bias_linear(&mlp / "fc3", dim, ffn),
        }
    }
}

/// Static depth fusion of encoder states into one memory per decoder layer.
#[derive(Debug)]
struct Memory {
    bridge_in_norm: RmsNorm,
    bridge_up: nn::Linear,
    bridge_down: nn::Linear,
    bridge_out_norm: RmsNorm,
    pos: nn::Linear,
    null: Tensor,
    norms: Vec<RmsNorm>,
    wo: nn::Linear,
    gamma: Tensor,
    depth_logits: Tensor,
}

impl Memory {
    fn new(path: &nn::Path<'_>, config: &MangaOcrArConfig) -> Self {
        let (encoder_dim, dim) = (config.encoder.dim, config.decoder.dim);
        let depths = config.encoder.layers;
        let layers = config.decoder.layers as i64;
        let bridge = path / "bridge";
        Self {
            bridge_in_norm: RmsNorm::new(&(&bridge / 0), encoder_dim, RMS_EPS),
            bridge_up: no_bias_linear(&bridge / 1, encoder_dim, BRIDGE_DIM),
            bridge_down: no_bias_linear(&bridge / 3, BRIDGE_DIM, dim),
            bridge_out_norm: RmsNorm::new(&(&bridge / 4), dim, RMS_EPS),
            pos: no_bias_linear(path / "pos", 4 * FOURIER_BANDS, dim),
            null: path.var("null", &[NULL_SLOTS, dim], nn::Init::Const(0.0)),
            norms: (0..depths)
                .map(|index| RmsNorm::new(&(path / "norms" / index), encoder_dim, RMS_EPS))
                .collect(),
            wo: no_bias_linear(path / "wo", encoder_dim, dim),
            gamma: path.var("gamma", &[layers, dim], nn::Init::Const(0.0)),
            depth_logits: path.var(
                "depth_logits",
                &[layers, depths as i64],
                nn::Init::Const(0.0),
            ),
        }
    }

    /// Fourier features `[sin r, cos r, sin c, cos c]` of patch-center coordinates.
    fn fourier(rows: &Tensor, cols: &Tensor, h: i64, w: i64) -> Tensor {
        let bands = (0..FOURIER_BANDS)
            .map(|band| std::f32::consts::PI * 2f32.powi(band as i32))
            .collect::<Vec<_>>();
        let bands = Tensor::from_slice(&bands).to_device(rows.device());
        let r = ((rows.to_kind(Kind::Float) + 0.5) / h as f64).unsqueeze(-1) * &bands;
        let c = ((cols.to_kind(Kind::Float) + 0.5) / w as f64).unsqueeze(-1) * &bands;
        Tensor::cat(&[r.sin(), r.cos(), c.sin(), c.cos()], -1)
    }

    /// Returns one `(1, 2 + h * w, dim)` memory per decoder layer.
    fn forward(
        &self,
        states: &[Tensor],
        rows: &Tensor,
        cols: &Tensor,
        h: i64,
        w: i64,
    ) -> Vec<Tensor> {
        let (last, depth_states) = states.split_last().expect("encoder returns its states");
        let bridge = self.bridge_out_norm.forward(
            &self.bridge_down.forward(
                &self
                    .bridge_up
                    .forward(&self.bridge_in_norm.forward(last))
                    .gelu("none"),
            ),
        );
        let base = bridge
            + self
                .pos
                .forward(&Self::fourier(rows, cols, h, w).to_kind(last.kind()));
        let depth = Tensor::stack(
            &self
                .norms
                .iter()
                .zip(depth_states)
                .map(|(norm, state)| norm.forward(state))
                .collect::<Vec<_>>(),
            0,
        );
        // einsum("jl,lbnd->jbnd", softmax(depth_logits), depth)
        let alpha = self
            .depth_logits
            .softmax(-1, None::<Kind>)
            .to_kind(depth.kind());
        let mut fused_shape = depth.size();
        fused_shape[0] = alpha.size()[0];
        let fused = alpha
            .matmul(&depth.flatten(1, -1))
            .view(fused_shape.as_slice());
        let memories =
            base.unsqueeze(0) + self.gamma.unsqueeze(1).unsqueeze(1) * self.wo.forward(&fused);
        let batch = last.size()[0];
        let null = self.null.unsqueeze(0).expand([batch, -1, -1], false);
        memories
            .unbind(0)
            .into_iter()
            .map(|memory| Tensor::cat(&[&null, &memory], 1))
            .collect()
    }
}

#[derive(Debug)]
struct ArBlock {
    self_norm: RmsNorm,
    qkv: nn::Linear,
    self_proj: nn::Linear,
    cross_norm: RmsNorm,
    q: nn::Linear,
    kv: nn::Linear,
    cross_proj: nn::Linear,
    ffn_norm: RmsNorm,
    gate_up: nn::Linear,
    down: nn::Linear,
    heads: i64,
    head_dim: i64,
}

impl ArBlock {
    fn new(path: &nn::Path<'_>, dim: i64, heads: i64, ffn: i64) -> Self {
        Self {
            self_norm: RmsNorm::new(&(path / "self_norm"), dim, RMS_EPS),
            qkv: no_bias_linear(path / "qkv", dim, 3 * dim),
            self_proj: no_bias_linear(path / "self_proj", dim, dim),
            cross_norm: RmsNorm::new(&(path / "cross_norm"), dim, RMS_EPS),
            q: no_bias_linear(path / "q", dim, dim),
            kv: no_bias_linear(path / "kv", dim, 2 * dim),
            cross_proj: no_bias_linear(path / "cross_proj", dim, dim),
            ffn_norm: RmsNorm::new(&(path / "ffn_norm"), dim, RMS_EPS),
            gate_up: no_bias_linear(path / "gate_up", dim, 2 * ffn),
            down: no_bias_linear(path / "down", ffn, dim),
            heads,
            head_dim: dim / heads,
        }
    }

    fn memory_kv(&self, memory: &Tensor) -> (Tensor, Tensor) {
        let size = memory.size();
        let kv = self
            .kv
            .forward(memory)
            .view([size[0], size[1], 2, self.heads, self.head_dim]);
        (
            kv.select(2, 0).transpose(1, 2),
            kv.select(2, 1).transpose(1, 2),
        )
    }

    /// One new token. Every cached position is visible, so the causal mask of
    /// upstream's fixed-size cache reduces to attending over the whole cache.
    fn step(
        &self,
        x: &Tensor,
        (memory_key, memory_value): &(Tensor, Tensor),
        (cos, sin): (&Tensor, &Tensor),
        cache: &mut Option<(Tensor, Tensor)>,
    ) -> Tensor {
        let size = x.size();
        let (batch, dim) = (size[0], size[2]);
        let qkv = self.qkv.forward(&self.self_norm.forward(x)).view([
            batch,
            1,
            3,
            self.heads,
            self.head_dim,
        ]);
        let query = apply_rope(&qkv.select(2, 0), cos, sin).transpose(1, 2);
        let key = apply_rope(&qkv.select(2, 1), cos, sin).transpose(1, 2);
        let value = qkv.select(2, 2).transpose(1, 2);
        let (key, value) = match cache.take() {
            Some((keys, values)) => (
                Tensor::cat(&[keys, key], 2),
                Tensor::cat(&[values, value], 2),
            ),
            None => (key, value),
        };
        let attended = Tensor::scaled_dot_product_attention(
            &query,
            &key,
            &value,
            None::<Tensor>,
            0.0,
            false,
            None,
            false,
        );
        *cache = Some((key, value));
        let x = x + self
            .self_proj
            .forward(&attended.transpose(1, 2).reshape([batch, 1, dim]));

        let query = self
            .q
            .forward(&self.cross_norm.forward(&x))
            .view([batch, 1, self.heads, self.head_dim])
            .transpose(1, 2);
        let attended = Tensor::scaled_dot_product_attention(
            &query,
            memory_key,
            memory_value,
            None::<Tensor>,
            0.0,
            false,
            None,
            false,
        );
        let x = &x
            + self
                .cross_proj
                .forward(&attended.transpose(1, 2).reshape([batch, 1, dim]));

        let gate_up = self
            .gate_up
            .forward(&self.ffn_norm.forward(&x))
            .chunk(2, -1);
        &x + self.down.forward(&(gate_up[0].silu() * &gate_up[1]))
    }
}
