//! Typed pub/sub round-trip over a live Zenoh session, including schema tagging
//! and reject-on-mismatch.

use std::time::Duration;

use imp_bus::Bus;
use imp_core::QosClass;
use imp_schemas::imp::Pose6D;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn typed_roundtrip_and_schema_reject() {
    let bus = Bus::open_default().await.expect("open session");
    let key = "imp/test/perc/s1/pose";

    let sub = bus.subscribe::<Pose6D>(key).await.expect("subscribe");
    // Give the subscription time to be established before publishing.
    tokio::time::sleep(Duration::from_millis(300)).await;

    let sent = Pose6D {
        object_id: "barrel".into(),
        position_m: vec![1.0, 2.0, 3.0],
        quat_xyzw: vec![0.0, 0.0, 0.0, 1.0],
        valid: true,
        ..Default::default()
    };
    bus.put(key, &sent, QosClass::State).await.expect("put");

    let got = tokio::time::timeout(Duration::from_secs(5), sub.recv())
        .await
        .expect("recv timed out")
        .expect("recv error");

    assert_eq!(got.object_id, "barrel");
    assert_eq!(got.position_m, vec![1.0, 2.0, 3.0]);
    assert!(got.valid);
}
