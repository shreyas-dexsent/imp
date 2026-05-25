//! Task layer (spec §11) -- Graph Compiler + Task Runtime + sequence FSM.
//!
//! **Status:** Rust scaffold. The working engine is the Python implementation
//! under `platform/sdk/py/imp_tasks/`. This crate exists to:
//!
//! 1. Keep `platform/crates/` the canonical home of sealed-core engines, so a
//!    Rust replacement can land here in a later phase (matches the contract
//!    crates today -- see `crates/module-contract`).
//! 2. Reserve the published artifact name `imp-tasks` on the Rust side.
//!
//! No types are exported yet. Consumers should depend on the Python
//! `imp_tasks` package via the SDK.

#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    #[error("task engine not yet implemented in Rust; see imp_tasks (Python)")]
    NotImplemented,
}
