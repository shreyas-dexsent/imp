//! `imp` — a thin client over the same Zenoh planes the UI uses (spec §16).
//! Phase 1 implements the topic-introspection verbs and `version`.

use std::io::Write;
use std::time::Instant;

use clap::{Parser, Subcommand};
use imp_bus::Bus;

#[derive(Parser)]
#[command(name = "imp", about = "imp control CLI", version)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Discover and measure any channel (wildcards ok).
    Topic {
        #[command(subcommand)]
        cmd: TopicCmd,
    },
    /// Print runtime + schema versions.
    Version,
}

#[derive(Subcommand)]
enum TopicCmd {
    /// Print messages on a key expression as they arrive.
    Echo { key: String },
    /// Measure message rate (Hz) on a key expression.
    Hz {
        key: String,
        /// Reporting interval in seconds.
        #[arg(long, default_value_t = 1.0)]
        interval: f64,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Version => {
            println!("imp        {}", env!("CARGO_PKG_VERSION"));
            println!("schemas    {} known", imp_schemas::KNOWN_SCHEMAS.len());
            println!("transport  zenoh 1.9");
        }
        Cmd::Topic { cmd } => match cmd {
            TopicCmd::Echo { key } => echo(&key).await?,
            TopicCmd::Hz { key, interval } => hz(&key, interval).await?,
        },
    }
    Ok(())
}

async fn echo(key: &str) -> anyhow::Result<()> {
    let bus = Bus::open_default().await.map_err(anyhow::Error::msg)?;
    let sub = bus.subscribe_raw(key).await.map_err(anyhow::Error::msg)?;
    eprintln!("echo {key} (Ctrl-C to stop)");
    loop {
        let msg = sub.recv().await.map_err(anyhow::Error::msg)?;
        let schema = msg.schema.as_deref().unwrap_or("<none>");
        match msg.schema.as_deref().and_then(|s| imp_schemas::decode_to_json(s, &msg.payload)) {
            Some(json) => println!("{} [{}]\n{}\n", msg.key, schema, json),
            None => println!("{} [{}] <{} bytes, undecoded>\n", msg.key, schema, msg.payload.len()),
        }
        std::io::stdout().flush().ok();
    }
}

async fn hz(key: &str, interval: f64) -> anyhow::Result<()> {
    let bus = Bus::open_default().await.map_err(anyhow::Error::msg)?;
    let sub = bus.subscribe_raw(key).await.map_err(anyhow::Error::msg)?;
    eprintln!("hz {key} every {interval}s (Ctrl-C to stop)");

    let mut ticker = tokio::time::interval(std::time::Duration::from_secs_f64(interval));
    ticker.tick().await; // fire immediately, ignore
    let mut count: u64 = 0;
    let mut window_start = Instant::now();

    loop {
        tokio::select! {
            r = sub.recv() => {
                r.map_err(anyhow::Error::msg)?;
                count += 1;
            }
            _ = ticker.tick() => {
                let elapsed = window_start.elapsed().as_secs_f64();
                let rate = if elapsed > 0.0 { count as f64 / elapsed } else { 0.0 };
                println!("{key}: {rate:.2} Hz ({count} msgs / {elapsed:.2}s)");
                std::io::stdout().flush().ok();
                count = 0;
                window_start = Instant::now();
            }
        }
    }
}
