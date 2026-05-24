//! imp core: ids, errors, key-expression conventions, QoS classes, and the
//! schema-tag format. No transport here — `imp-bus` maps these onto Zenoh.

pub mod keyexpr;

use serde::{Deserialize, Serialize};

/// Top of the key namespace: `imp/<station>/...` (spec §6).
pub const ROOT: &str = "imp";

/// QoS class for a channel (spec §6). The concrete Zenoh settings are applied
/// by `imp-bus`; this enum is the transport-agnostic declaration that lives on
/// an interface descriptor (spec §7).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QosClass {
    /// Commands / trajectories: reliable, block on congestion, high priority.
    Command,
    /// Frames / masks: best-effort, drop latest-wins, low priority.
    Frame,
    /// Poses / state: reliable, drop-oldest, medium priority.
    State,
    /// Metrics / telemetry: best-effort, drop, lowest priority.
    Telemetry,
}

/// A parsed schema tag, e.g. `imp.Pose6D/1` → `{ name: "imp.Pose6D", version: 1 }`.
///
/// Carried as the key attachment on every message; subscribers reject on
/// mismatch (spec §6).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SchemaTag {
    pub name: String,
    pub version: u32,
}

impl SchemaTag {
    pub fn new(name: impl Into<String>, version: u32) -> Self {
        Self { name: name.into(), version }
    }

    /// Parse `"<name>/<version>"`. Returns `None` if malformed.
    pub fn parse(s: &str) -> Option<Self> {
        let (name, ver) = s.rsplit_once('/')?;
        if name.is_empty() {
            return None;
        }
        Some(Self { name: name.to_string(), version: ver.parse().ok()? })
    }

    /// True when name matches and the incoming version is not newer than ours
    /// (older readers reject newer writers; spec §7 pins versions per deploy).
    pub fn accepts(&self, incoming: &SchemaTag) -> bool {
        self.name == incoming.name && incoming.version <= self.version
    }
}

impl std::fmt::Display for SchemaTag {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}/{}", self.name, self.version)
    }
}

/// Direction of an I/O surface relative to its owner.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Direction {
    Publish,
    Subscribe,
}

/// The single descriptor shape used by every I/O surface — HAL topic, module
/// port, service request/response, job lifecycle (spec §7). The runtime uses
/// it for codegen, runtime validation, introspection, and UI form generation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Interface {
    pub name: String,
    pub direction: Direction,
    /// Versioned schema tag, e.g. `imp.Frame/1`.
    pub schema: SchemaTag,
    pub qos: QosClass,
    /// Nominal publish rate, where applicable.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rate_hz: Option<f64>,
}

impl Interface {
    pub fn publishes(name: impl Into<String>, schema: SchemaTag, qos: QosClass, rate_hz: Option<f64>) -> Self {
        Self { name: name.into(), direction: Direction::Publish, schema, qos, rate_hz }
    }
    pub fn subscribes(name: impl Into<String>, schema: SchemaTag, qos: QosClass) -> Self {
        Self { name: name.into(), direction: Direction::Subscribe, schema, qos, rate_hz: None }
    }
}

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("invalid schema tag: {0}")]
    InvalidSchemaTag(String),
    #[error("schema mismatch on {key}: expected {expected}, got {got}")]
    SchemaMismatch { key: String, expected: String, got: String },
    #[error("missing schema attachment on {0}")]
    MissingSchema(String),
    #[error("bus error: {0}")]
    Bus(String),
    #[error("decode error: {0}")]
    Decode(String),
}

pub type Result<T> = std::result::Result<T, Error>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schema_tag_parse_and_accept() {
        let t = SchemaTag::parse("imp.Pose6D/2").unwrap();
        assert_eq!(t.name, "imp.Pose6D");
        assert_eq!(t.version, 2);
        assert_eq!(t.to_string(), "imp.Pose6D/2");
        assert!(SchemaTag::parse("noversion").is_none());

        let reader = SchemaTag::new("imp.Pose6D", 2);
        assert!(reader.accepts(&SchemaTag::new("imp.Pose6D", 1)));
        assert!(reader.accepts(&SchemaTag::new("imp.Pose6D", 2)));
        assert!(!reader.accepts(&SchemaTag::new("imp.Pose6D", 3)));
        assert!(!reader.accepts(&SchemaTag::new("imp.Frame", 1)));
    }
}
