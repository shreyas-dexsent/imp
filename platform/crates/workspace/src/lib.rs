//! Workspace loader (spec §14): the Station → Process → Task → Run model, with
//! schema-validated, version-pinned config and atomic writes that back up the
//! previous file to `trash/<timestamp>/` (no silent overwrites).

pub mod config;

use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::de::DeserializeOwned;
use serde::Serialize;

use config::{Hardware, Process, Station, Versioned};

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("io error on {path}: {source}")]
    Io { path: String, source: std::io::Error },
    #[error("yaml error in {path}: {source}")]
    Yaml { path: String, source: serde_yaml::Error },
    #[error("{path}: expected schema {expected}/{ev}, got {got}/{gv}")]
    Schema { path: String, expected: String, ev: u32, got: String, gv: u32 },
    #[error("not found: {0}")]
    NotFound(String),
}

pub type Result<T> = std::result::Result<T, Error>;

/// A workspace rooted at a directory (spec §14 layout).
pub struct Workspace {
    root: PathBuf,
}

impl Workspace {
    pub fn open(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// Resolve the workspace root from `IMP_WORKSPACE`, else `~/.imp/workspace`.
    pub fn from_env() -> Self {
        if let Ok(p) = std::env::var("IMP_WORKSPACE") {
            return Self::open(p);
        }
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
        Self::open(PathBuf::from(home).join(".imp").join("workspace"))
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    // ---- stations / processes / tasks -----------------------------------

    pub fn list_stations(&self) -> Result<Vec<String>> {
        list_dirs(&self.root.join("stations"))
    }

    pub fn load_station(&self, station: &str) -> Result<Station> {
        self.load_yaml(format!("stations/{station}/station.yaml"))
    }

    pub fn load_hardware(&self, station: &str) -> Result<Hardware> {
        self.load_yaml(format!("stations/{station}/hardware.yaml"))
    }

    pub fn list_processes(&self, station: &str) -> Result<Vec<String>> {
        list_dirs(&self.root.join("stations").join(station).join("processes"))
    }

    pub fn load_process(&self, station: &str, process: &str) -> Result<Process> {
        self.load_yaml(format!("stations/{station}/processes/{process}/process.yaml"))
    }

    pub fn list_tasks(&self, station: &str, process: &str) -> Result<Vec<String>> {
        let dir = self
            .root
            .join("stations")
            .join(station)
            .join("processes")
            .join(process)
            .join("tasks");
        let mut tasks = Vec::new();
        if !dir.is_dir() {
            return Ok(tasks);
        }
        for entry in read_dir(&dir)? {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("yaml") {
                if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                    tasks.push(stem.to_string());
                }
            }
        }
        tasks.sort();
        Ok(tasks)
    }

    // ---- generic load / save --------------------------------------------

    /// Load and validate a versioned config at a workspace-relative path.
    pub fn load_yaml<T>(&self, rel: impl AsRef<Path>) -> Result<T>
    where
        T: DeserializeOwned + Versioned,
    {
        let path = self.root.join(rel.as_ref());
        let display = path.display().to_string();
        if !path.exists() {
            return Err(Error::NotFound(display));
        }
        let text = std::fs::read_to_string(&path).map_err(|source| Error::Io {
            path: display.clone(),
            source,
        })?;
        let value: T = serde_yaml::from_str(&text).map_err(|source| Error::Yaml {
            path: display.clone(),
            source,
        })?;
        if value.schema() != T::SCHEMA || value.version() != T::VERSION {
            return Err(Error::Schema {
                path: display,
                expected: T::SCHEMA.to_string(),
                ev: T::VERSION,
                got: value.schema().to_string(),
                gv: value.version(),
            });
        }
        Ok(value)
    }

    /// Serialize `value` to a workspace-relative path, atomically, backing up any
    /// existing file to `trash/<timestamp>/<rel>` first (spec §14).
    pub fn save_yaml<T: Serialize>(&self, rel: impl AsRef<Path>, value: &T) -> Result<()> {
        let rel = rel.as_ref();
        let path = self.root.join(rel);
        let display = path.display().to_string();

        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|source| Error::Io {
                path: parent.display().to_string(),
                source,
            })?;
        }

        if path.exists() {
            let backup = self.root.join("trash").join(timestamp()).join(rel);
            if let Some(parent) = backup.parent() {
                std::fs::create_dir_all(parent).map_err(|source| Error::Io {
                    path: parent.display().to_string(),
                    source,
                })?;
            }
            std::fs::copy(&path, &backup).map_err(|source| Error::Io {
                path: backup.display().to_string(),
                source,
            })?;
        }

        let text = serde_yaml::to_string(value).map_err(|source| Error::Yaml {
            path: display.clone(),
            source,
        })?;
        let tmp = path.with_extension("yaml.tmp");
        std::fs::write(&tmp, text.as_bytes()).map_err(|source| Error::Io {
            path: tmp.display().to_string(),
            source,
        })?;
        std::fs::rename(&tmp, &path).map_err(|source| Error::Io {
            path: display,
            source,
        })?;
        Ok(())
    }
}

fn timestamp() -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("{nanos:020}")
}

fn read_dir(dir: &Path) -> Result<Vec<std::fs::DirEntry>> {
    let mut entries = Vec::new();
    for entry in std::fs::read_dir(dir).map_err(|source| Error::Io {
        path: dir.display().to_string(),
        source,
    })? {
        let entry = entry.map_err(|source| Error::Io {
            path: dir.display().to_string(),
            source,
        })?;
        entries.push(entry);
    }
    Ok(entries)
}

fn list_dirs(dir: &Path) -> Result<Vec<String>> {
    let mut names = Vec::new();
    if !dir.is_dir() {
        return Ok(names);
    }
    for entry in read_dir(dir)? {
        if entry.path().is_dir() {
            if let Some(name) = entry.file_name().to_str() {
                names.push(name.to_string());
            }
        }
    }
    names.sort();
    Ok(names)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(path: &Path, text: &str) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, text).unwrap();
    }

    #[test]
    fn loads_station_process_tasks_and_validates() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        write(
            &root.join("stations/st1/station.yaml"),
            "schema: imp.station\nversion: 1\nid: st1\nname: Cell 1\nlocation: Line A\n",
        );
        write(
            &root.join("stations/st1/hardware.yaml"),
            "schema: imp.hardware\nversion: 1\ndevices:\n  - { id: ur5e, kind: robot, driver: robot-mujoco-ur5e, state: hal/ur5e/state, command: hal/ur5e/command, rate_hz: 125 }\n",
        );
        write(
            &root.join("stations/st1/processes/p1/process.yaml"),
            "schema: imp.process\nversion: 1\nid: p1\nname: Pick\ntask_type: bin_picking\nrobot: ur5e\n",
        );
        write(
            &root.join("stations/st1/processes/p1/tasks/pick.yaml"),
            "schema: imp.task\nversion: 1\nid: pick\n",
        );

        let ws = Workspace::open(root);
        assert_eq!(ws.list_stations().unwrap(), vec!["st1"]);
        let st = ws.load_station("st1").unwrap();
        assert_eq!(st.name, "Cell 1");
        let hw = ws.load_hardware("st1").unwrap();
        assert_eq!(hw.devices.len(), 1);
        assert_eq!(hw.devices[0].driver, "robot-mujoco-ur5e");
        assert_eq!(ws.list_processes("st1").unwrap(), vec!["p1"]);
        assert_eq!(ws.load_process("st1", "p1").unwrap().task_type, "bin_picking");
        assert_eq!(ws.list_tasks("st1", "p1").unwrap(), vec!["pick"]);
    }

    #[test]
    fn rejects_schema_version_mismatch() {
        let tmp = tempfile::tempdir().unwrap();
        write(
            &tmp.path().join("stations/x/station.yaml"),
            "schema: imp.station\nversion: 99\nid: x\nname: X\n",
        );
        let err = Workspace::open(tmp.path()).load_station("x").unwrap_err();
        assert!(matches!(err, Error::Schema { .. }));
    }

    #[test]
    fn save_backs_up_previous_to_trash() {
        let tmp = tempfile::tempdir().unwrap();
        let ws = Workspace::open(tmp.path());
        let v1 = Station {
            schema: "imp.station".into(),
            version: 1,
            id: "st1".into(),
            name: "first".into(),
            location: String::new(),
        };
        ws.save_yaml("stations/st1/station.yaml", &v1).unwrap();
        let mut v2 = v1.clone();
        v2.name = "second".into();
        ws.save_yaml("stations/st1/station.yaml", &v2).unwrap();

        assert_eq!(ws.load_station("st1").unwrap().name, "second");
        // A backup of the first write now lives under trash/.
        let trash = tmp.path().join("trash");
        let backups: Vec<_> = walkdir(&trash);
        assert!(backups.iter().any(|p| p.ends_with("station.yaml")));
    }

    fn walkdir(dir: &Path) -> Vec<PathBuf> {
        let mut out = Vec::new();
        if let Ok(rd) = std::fs::read_dir(dir) {
            for e in rd.flatten() {
                let p = e.path();
                if p.is_dir() {
                    out.extend(walkdir(&p));
                } else {
                    out.push(p);
                }
            }
        }
        out
    }
}
