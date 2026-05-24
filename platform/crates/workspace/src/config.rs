//! Workspace config schemas (spec §14). Each is schema- and version-pinned;
//! the loader rejects an unknown schema or a version it doesn't understand.
//!
//! The robot/world description YAML (`robot_system.yaml`, `world.yaml`) uses the
//! `robot-algorithms` pydantic schemas and is validated where it is consumed
//! (the motion modules), not here.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// A config type that carries a pinned `schema`/`version`.
pub trait Versioned {
    const SCHEMA: &'static str;
    const VERSION: u32;
    fn schema(&self) -> &str;
    fn version(&self) -> u32;
}

macro_rules! versioned {
    ($ty:ty, $schema:literal, $ver:literal) => {
        impl Versioned for $ty {
            const SCHEMA: &'static str = $schema;
            const VERSION: u32 = $ver;
            fn schema(&self) -> &str {
                &self.schema
            }
            fn version(&self) -> u32 {
                self.version
            }
        }
    };
}

/// `stations/<id>/station.yaml`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Station {
    pub schema: String,
    pub version: u32,
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub location: String,
}
versioned!(Station, "imp.station", 1);

/// `stations/<id>/hardware.yaml`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hardware {
    pub schema: String,
    pub version: u32,
    #[serde(default)]
    pub devices: Vec<Device>,
}
versioned!(Hardware, "imp.hardware", 1);

/// One device entry: id → driver + topics + rate (spec §14).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Device {
    pub id: String,
    pub kind: String,
    pub driver: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub state: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub command: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub topic: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rate_hz: Option<f64>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub params: BTreeMap<String, serde_yaml::Value>,
}

/// `stations/<id>/processes/<id>/process.yaml`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Process {
    pub schema: String,
    pub version: u32,
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub task_type: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub robot: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gripper: Option<String>,
}
versioned!(Process, "imp.process", 1);
