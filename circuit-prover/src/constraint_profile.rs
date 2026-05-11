use serde::{Deserialize, Serialize};

/// High-level constraint profiles used to instantiate AIR variants.
///
/// Profiles are selected per proof shape / recursion layer and must be
/// applied consistently across:
/// - circuit construction,
/// - native batch proving,
/// - recursive verification.
#[derive(Clone, Copy, Default, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum ConstraintProfile {
    /// Existing behaviour: all AIRs keep their current low-degree constraints.
    #[default]
    Standard,
    /// Recursion-optimised profile.
    RecursionOptimized,
}
