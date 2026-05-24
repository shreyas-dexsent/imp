# workspace

**Kind:** Crate &nbsp;|&nbsp; **Status:** phase 2 — implemented

Workspace loader (spec §14): the Station → Process → Task → Run model with
schema-validated, version-pinned config and atomic writes that back up the
previous file to `trash/<timestamp>/` (no silent overwrites).

- `Workspace::open(root)` / `from_env()` (`IMP_WORKSPACE`, else `~/.imp/workspace`).
- `load_station` / `load_hardware` / `load_process`, `list_*`, generic `load_yaml`.
- `save_yaml` — atomic write + trash backup.
- Config schemas in `config` (`imp.station/1`, `imp.hardware/1`, `imp.process/1`).

The robot/world description YAML (`dexsent.algorithms.robot_system`/`world`) is
validated by the motion modules, not here. Migrated from `orchestrator/storage/*`.
