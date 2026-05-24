//! Minimal publisher for verifying the bus + CLI without hardware.
//! Publishes a `RobotState` at 10 Hz on `imp/<station>/hal/<device>/state`.
//!
//! Run:  cargo run -p imp-bus --example demo_pub
//! Then: imp topic echo 'imp/devstation/hal/sim/state'

use imp_bus::Bus;
use imp_core::{keyexpr, QosClass};
use imp_schemas::imp::{Header, RobotState};
use imp_schemas::ImpMessage;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let station = std::env::var("IMP_STATION").unwrap_or_else(|_| "devstation".into());
    let device = std::env::var("IMP_DEVICE").unwrap_or_else(|_| "sim".into());
    let key = keyexpr::hal(&station, &device, "state");

    let bus = Bus::open_default().await.map_err(|e| e.to_string())?;
    let publisher = bus.publisher::<RobotState>(&key, QosClass::State).await.map_err(|e| e.to_string())?;
    eprintln!("publishing RobotState on {key} at 10 Hz");

    let mut seq = 0u64;
    loop {
        let msg = RobotState {
            header: Some(Header {
                seq,
                stamp_ns: now_ns(),
                schema: RobotState::schema_tag(),
                ..Default::default()
            }),
            q: vec![0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785],
            mode: "idle".into(),
            ..Default::default()
        };
        publisher.put(&msg).await.map_err(|e| e.to_string())?;
        seq += 1;
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }
}

fn now_ns() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
}
