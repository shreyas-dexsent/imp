//! Zenoh transport for imp: a thin wrapper that applies imp's key conventions
//! (via `imp-core`), maps [`QosClass`] onto Zenoh QoS, tags every publication
//! with its schema, and rejects mismatched schemas on receive (spec §6).

use std::marker::PhantomData;

use imp_core::{Error, QosClass, Result, SchemaTag};
use imp_schemas::ImpMessage;
use zenoh::qos::{CongestionControl, Priority, Reliability};

fn map_err<E: std::fmt::Display>(e: E) -> Error {
    Error::Bus(e.to_string())
}

/// imp's default Zenoh config. Forces IPv4 TCP listening because the zenoh
/// default binds `tcp/[::]:0`, which fails on IPv4-only hosts (many containers
/// and CI runners). Set `IMP_ZENOH_CONFIG` to a json5 file to override entirely.
fn default_config() -> Result<zenoh::Config> {
    if let Ok(path) = std::env::var("IMP_ZENOH_CONFIG") {
        return zenoh::Config::from_file(path).map_err(map_err);
    }
    let mut config = zenoh::Config::default();
    config.insert_json5("listen/endpoints", r#"["tcp/0.0.0.0:0"]"#).map_err(map_err)?;
    Ok(config)
}

/// The concrete Zenoh QoS for an imp [`QosClass`] (spec §6 table).
fn qos(class: QosClass) -> (Reliability, CongestionControl, Priority) {
    match class {
        QosClass::Command => (Reliability::Reliable, CongestionControl::Block, Priority::RealTime),
        QosClass::State => (Reliability::Reliable, CongestionControl::Drop, Priority::Data),
        QosClass::Frame => (Reliability::BestEffort, CongestionControl::Drop, Priority::DataLow),
        QosClass::Telemetry => {
            (Reliability::BestEffort, CongestionControl::Drop, Priority::Background)
        }
    }
}

/// A connection to the Zenoh fabric.
pub struct Bus {
    session: zenoh::Session,
}

impl Bus {
    /// Open a session with imp's default config (peer mode, auto SHM when
    /// colocated). Honors `IMP_ZENOH_CONFIG` (a json5 file path) when set.
    pub async fn open_default() -> Result<Self> {
        Self::open(default_config()?).await
    }

    pub async fn open(config: zenoh::Config) -> Result<Self> {
        let session = zenoh::open(config).await.map_err(map_err)?;
        Ok(Self { session })
    }

    /// Underlying session, for ops not yet wrapped (queryables, liveliness, …).
    pub fn session(&self) -> &zenoh::Session {
        &self.session
    }

    /// Publish a single typed message, tagging the schema attachment.
    pub async fn put<M: ImpMessage>(&self, key: &str, msg: &M, class: QosClass) -> Result<()> {
        let (rel, cc, prio) = qos(class);
        self.session
            .put(key, msg.encode_to_vec())
            .attachment(M::schema_tag().into_bytes())
            .reliability(rel)
            .congestion_control(cc)
            .priority(prio)
            .await
            .map_err(map_err)
    }

    /// A reusable publisher bound to one key + QoS (declares the resource once).
    pub async fn publisher<M: ImpMessage>(&self, key: &str, class: QosClass) -> Result<Publisher<M>> {
        let (rel, cc, prio) = qos(class);
        let inner = self
            .session
            .declare_publisher(key.to_string())
            .reliability(rel)
            .congestion_control(cc)
            .priority(prio)
            .await
            .map_err(map_err)?;
        Ok(Publisher { inner, tag: M::schema_tag(), _m: PhantomData })
    }

    /// Subscribe and decode a known typed message. Messages whose schema tag
    /// does not match `M` are dropped (treated as a missing topic, spec §6).
    pub async fn subscribe<M: ImpMessage>(&self, key: &str) -> Result<TypedSub<M>> {
        let raw = self.subscribe_raw(key).await?;
        let expected = SchemaTag::new(M::NAME, M::VERSION);
        Ok(TypedSub { raw, expected, _m: PhantomData })
    }

    /// Subscribe to raw samples for any schema (used by `imp topic echo`/`hz`).
    pub async fn subscribe_raw(&self, key: &str) -> Result<RawSub> {
        let inner = self.session.declare_subscriber(key.to_string()).await.map_err(map_err)?;
        Ok(RawSub { inner })
    }
}

/// A declared publisher for a fixed message type.
pub struct Publisher<M: ImpMessage> {
    inner: zenoh::pubsub::Publisher<'static>,
    tag: String,
    _m: PhantomData<M>,
}

impl<M: ImpMessage> Publisher<M> {
    pub async fn put(&self, msg: &M) -> Result<()> {
        self.inner
            .put(msg.encode_to_vec())
            .attachment(self.tag.clone().into_bytes())
            .await
            .map_err(map_err)
    }
}

/// A raw, untyped subscription.
pub struct RawSub {
    inner: zenoh::pubsub::Subscriber<zenoh::handlers::FifoChannelHandler<zenoh::sample::Sample>>,
}

/// One received sample, decoded only as far as key + schema tag + bytes.
pub struct RawMsg {
    pub key: String,
    pub schema: Option<String>,
    pub payload: Vec<u8>,
}

impl RawSub {
    pub async fn recv(&self) -> Result<RawMsg> {
        let sample = self.inner.recv_async().await.map_err(map_err)?;
        let schema = sample
            .attachment()
            .map(|a| String::from_utf8_lossy(&a.to_bytes()).into_owned());
        Ok(RawMsg {
            key: sample.key_expr().as_str().to_string(),
            schema,
            payload: sample.payload().to_bytes().into_owned(),
        })
    }
}

/// A typed subscription that validates the schema tag and decodes to `M`.
pub struct TypedSub<M: ImpMessage> {
    raw: RawSub,
    expected: SchemaTag,
    _m: PhantomData<M>,
}

impl<M: ImpMessage> TypedSub<M> {
    /// Receive the next message whose schema matches `M`, dropping mismatches.
    pub async fn recv(&self) -> Result<M> {
        loop {
            let msg = self.raw.recv().await?;
            let Some(tag_str) = msg.schema.as_deref() else { continue };
            let Some(tag) = SchemaTag::parse(tag_str) else { continue };
            if !self.expected.accepts(&tag) {
                continue; // schema mismatch → dropped (missing topic)
            }
            return M::decode(&msg.payload[..]).map_err(|e| Error::Decode(e.to_string()));
        }
    }
}
