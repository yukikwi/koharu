use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use koharu_ml::minimax_h3_image::{
    DEFAULT_PROMPT, MiniMaxH3ImageInpaint, MiniMaxH3ImageInpaintOptions,
};

#[derive(Debug, Parser)]
struct Cli {
    #[arg(short, long, value_name = "FILE")]
    input: PathBuf,

    /// White pixels are regenerated.
    #[arg(short, long, value_name = "FILE")]
    mask: PathBuf,

    #[arg(short, long, value_name = "FILE")]
    output: PathBuf,

    #[arg(short, long, value_name = "TEXT", default_value = DEFAULT_PROMPT)]
    prompt: String,

    #[arg(long, default_value_t = -1)]
    seed: i64,

    #[arg(long, default_value_t = false)]
    cpu: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let cli = Cli::parse();
    let input = image::open(&cli.input)?;
    let mask = image::open(&cli.mask)?;

    koharu_ml::init().await?;

    let pipeline = MiniMaxH3ImageInpaint::load(koharu_ml::device(cli.cpu)).await?;
    let output = pipeline.inference(
        &cli.prompt,
        &input,
        &mask,
        &MiniMaxH3ImageInpaintOptions { seed: cli.seed },
    )?;
    output.save(cli.output)?;

    Ok(())
}
