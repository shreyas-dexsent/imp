//! Key-expression builders for the imp namespace (spec §6):
//!
//! ```text
//! imp/<station>/hal/<device>/<signal>
//! imp/<station>/perc/<session>/<signal>
//! imp/<station>/motion/<plan>/<signal>
//! imp/<station>/tf
//! imp/<station>/svc/<service>
//! imp/<station>/ctrl/<node>/<verb>
//! ```

use crate::ROOT;

pub fn hal(station: &str, device: &str, signal: &str) -> String {
    format!("{ROOT}/{station}/hal/{device}/{signal}")
}

pub fn perc(station: &str, session: &str, signal: &str) -> String {
    format!("{ROOT}/{station}/perc/{session}/{signal}")
}

pub fn motion(station: &str, plan: &str, signal: &str) -> String {
    format!("{ROOT}/{station}/motion/{plan}/{signal}")
}

pub fn tf(station: &str) -> String {
    format!("{ROOT}/{station}/tf")
}

pub fn svc(station: &str, service: &str) -> String {
    format!("{ROOT}/{station}/svc/{service}")
}

pub fn ctrl(station: &str, node: &str, verb: &str) -> String {
    format!("{ROOT}/{station}/ctrl/{node}/{verb}")
}

/// All perception signals for a station: `imp/<station>/perc/**`.
pub fn all_perc(station: &str) -> String {
    format!("{ROOT}/{station}/perc/**")
}

/// One signal across every device: `imp/<station>/hal/*/<signal>`.
pub fn hal_any_device(station: &str, signal: &str) -> String {
    format!("{ROOT}/{station}/hal/*/{signal}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_spec_keys() {
        assert_eq!(hal("st1", "cam_d405", "frame"), "imp/st1/hal/cam_d405/frame");
        assert_eq!(perc("st1", "s1", "pose"), "imp/st1/perc/s1/pose");
        assert_eq!(motion("st1", "p1", "trajectory"), "imp/st1/motion/p1/trajectory");
        assert_eq!(tf("st1"), "imp/st1/tf");
        assert_eq!(svc("st1", "tf.lookup"), "imp/st1/svc/tf.lookup");
        assert_eq!(ctrl("st1", "cam_d405", "heartbeat"), "imp/st1/ctrl/cam_d405/heartbeat");
        assert_eq!(all_perc("st1"), "imp/st1/perc/**");
        assert_eq!(hal_any_device("st1", "state"), "imp/st1/hal/*/state");
    }
}
