//! Generated imp wire schemas + a thin self-describing layer.
//!
//! Every message is generated from `proto/imp.proto`. On top of the generated
//! types we add [`ImpMessage`], which ties a Rust type to its versioned schema
//! id (e.g. `"imp.Pose6D/1"`), and [`decode_to_json`], which the CLI/UI use to
//! render an arbitrary message from its wire bytes + schema tag.

pub mod imp {
    include!(concat!(env!("OUT_DIR"), "/imp.rs"));
}

pub use prost::Message;

/// Associates a generated message type with its versioned schema id.
///
/// The id is `"<package>.<Message>"` and the version starts at 1; a new field
/// is a version bump (spec §7).
pub trait ImpMessage: prost::Message + Default {
    /// Schema name without version, e.g. `"imp.Pose6D"`.
    const NAME: &'static str;
    /// Schema version, e.g. `1`.
    const VERSION: u32 = 1;

    /// Full versioned tag carried in the key attachment, e.g. `"imp.Pose6D/1"`.
    fn schema_tag() -> String {
        format!("{}/{}", Self::NAME, Self::VERSION)
    }
}

macro_rules! impl_schema {
    ($($ty:ident),+ $(,)?) => {
        $(
            impl ImpMessage for imp::$ty {
                const NAME: &'static str = concat!("imp.", stringify!($ty));
            }
        )+

        /// Decode `bytes` for a known schema (version-insensitive) into pretty JSON.
        /// Returns `None` for an unknown schema name.
        pub fn decode_to_json(schema_name: &str, bytes: &[u8]) -> Option<String> {
            let name = schema_name.split('/').next().unwrap_or(schema_name);
            match name {
                $(
                    concat!("imp.", stringify!($ty)) => {
                        let msg = <imp::$ty as prost::Message>::decode(bytes).ok()?;
                        serde_json::to_string_pretty(&msg).ok()
                    }
                )+
                _ => None,
            }
        }

        /// Every schema name imp knows about (without version).
        pub const KNOWN_SCHEMAS: &[&str] = &[ $( concat!("imp.", stringify!($ty)) ),+ ];
    };
}

impl_schema!(
    Header, BlobRef, Intrinsics,
    Frame, Roi, Mask, Detection, Detections, Pose6D, Grasp, Grasps, PointCloud, Scalar,
    TfEdge, PoseTarget,
    JointSolution, Path, Trajectory,
    RobotState, MotionCommand, GripperState, GripperCommand, Io,
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schema_tag_roundtrips_through_json() {
        assert_eq!(imp::Pose6D::schema_tag(), "imp.Pose6D/1");
        let p = imp::Pose6D {
            object_id: "barrel".into(),
            position_m: vec![0.1, 0.2, 0.3],
            valid: true,
            ..Default::default()
        };
        let bytes = p.encode_to_vec();
        let json = decode_to_json("imp.Pose6D/1", &bytes).unwrap();
        assert!(json.contains("barrel"));
        assert!(decode_to_json("imp.DoesNotExist", &bytes).is_none());
    }
}
