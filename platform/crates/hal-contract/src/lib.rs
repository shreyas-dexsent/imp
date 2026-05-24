//! HAL device contract (spec §8). A device is a node holding the vendor SDK and
//! exposing the standard topic contract from §7. Python drivers implement the
//! equivalent surface through the SDK; this trait is for Rust/C++ nodes and for
//! the Supervisor's view of a managed node's lifecycle.

use imp_core::{Interface, Result};

/// Lifecycle state of a managed node (driven by the Supervisor, spec §12).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lifecycle {
    Unconfigured,
    Inactive,
    Active,
    Faulted,
}

/// A hardware device exposed as a node.
pub trait HalDevice {
    /// Device family: `"camera"`, `"robot"`, `"gripper"`, `"plc"`, ….
    fn kind(&self) -> &str;

    /// The topics this device publishes and subscribes (spec §7/§8).
    fn interfaces(&self) -> Vec<Interface>;

    /// Acquire the vendor SDK / open the device. `Unconfigured` → `Inactive`.
    fn configure(&mut self) -> Result<()>;

    /// Begin the device's real-time loop. `Inactive` → `Active`. The robot HAL
    /// node owns the deterministic loop (spec §8).
    fn activate(&mut self) -> Result<()>;

    /// Stop the loop and fail safe. `Active` → `Inactive`.
    fn deactivate(&mut self) -> Result<()>;
}
