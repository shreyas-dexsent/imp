//! Service and job contracts (spec §10). A service is a synchronous Zenoh
//! queryable; a job is a long-running, cancelable, monitorable op. Both are
//! declared with the same descriptor shape as everything else (spec §7).

use imp_core::SchemaTag;

/// A synchronous request/response service backed by a Zenoh queryable.
pub trait Service {
    /// Service name, e.g. `"tf.lookup"`.
    fn name(&self) -> &str;
    fn request_schema(&self) -> SchemaTag;
    fn response_schema(&self) -> SchemaTag;
}

/// A long-running operation: request in, progress stream, terminal result.
pub trait Job {
    fn name(&self) -> &str;
    fn request_schema(&self) -> SchemaTag;
    fn progress_schema(&self) -> SchemaTag;
    fn result_schema(&self) -> SchemaTag;
}
