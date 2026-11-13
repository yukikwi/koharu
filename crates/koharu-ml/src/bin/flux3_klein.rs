use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use koharu_ml::flux3_klein::{DEFAULT_PROMPT, Flux3KleinInpaint, Flux3KleinInpaintOptions};

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

    let pipeline = Flux3KleinInpaint::load(koharu_ml::device(cli.cpu)).await?;
    let output = pipeline.inference(
        &cli.prompt,
        &input,
        &mask,
        &Flux3KleinInpaintOptions { seed: cli.seed },
    )?;
    output.save(cli.output)?;

    Ok(())
}
