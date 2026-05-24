//! Functional-module contract (spec §9). A module is a pure typed function with
//! declared input/output ports. The Compute Runtime (added in a later phase)
//! does subscribe → validate → call → publish, and for motion modules fills the
//! resolved `Scene` from subscribed topics each tick before calling the op.

use imp_core::Interface;

/// Static description of a module's ports and rate. Authored via the SDK
/// `@module` decorator (Python) or this trait (Rust).
pub trait Module {
    fn name(&self) -> &str;

    /// Typed input ports (subscribed topics).
    fn inputs(&self) -> Vec<Interface>;

    /// Typed output ports (published topics).
    fn outputs(&self) -> Vec<Interface>;

    /// Nominal call rate in Hz, if rate-scheduled.
    fn rate_hz(&self) -> Option<f64> {
        None
    }
}
