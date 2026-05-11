//! User-facing call structs for adding Poseidon2 permutation rows.

use alloc::vec;
use alloc::vec::Vec;

use crate::ops::poseidon2_perm::config::Poseidon2Config;
use crate::types::ExprId;

/// User-facing arguments for adding a Poseidon2 perm row.
pub struct Poseidon2PermCall {
    /// Poseidon2 configuration for this permutation row.
    pub config: Poseidon2Config,
    /// Flag indicating whether a new chain is started.
    pub new_start: bool,
    /// Flag indicating whether we are verifying a Merkle path
    pub merkle_path: bool,
    /// MMCS direction bit input (base field, boolean).
    ///
    /// Required when `merkle_path = true`. When `merkle_path = false`, this may be omitted and
    /// defaults to 0 (not exposed via CTL).
    pub mmcs_bit: Option<ExprId>,
    /// Optional CTL exposure for each input limb (one extension element).
    /// If `None`, the limb is not exposed via CTL (in_ctl = 0).
    /// Note: For Merkle mode, unexposed limbs are provided via Poseidon2PermPrivateData (the sibling).
    pub inputs: Vec<Option<ExprId>>,
    /// Output exposure flags for rate limbs (CTL-verified against witness table).
    ///
    /// When `out_ctl[i]` is true, this call allocates an output witness expression for limb `i`
    /// (returned from `add_poseidon2_perm`) and exposes it via CTL.
    pub out_ctl: Vec<bool>,
    /// Whether to return all 4 output limbs (for challenger use).
    ///
    /// When true, outputs 2-3 are also allocated and returned, but NOT CTL-verified
    /// (they are capacity elements, constrained only by the Poseidon2 permutation itself).
    /// This is used by challenger operations that need the full sponge state.
    pub return_all_outputs: bool,
    /// Optional MMCS index accumulator value to expose.
    pub mmcs_index_sum: Option<ExprId>,
}

impl Default for Poseidon2PermCall {
    fn default() -> Self {
        let config = Poseidon2Config::BabyBearD4Width16;
        Self {
            config,
            new_start: false,
            merkle_path: false,
            mmcs_bit: None,
            inputs: vec![None; config.width_ext()],
            out_ctl: vec![false; config.rate_ext()],
            return_all_outputs: false,
            mmcs_index_sum: None,
        }
    }
}

/// User-facing arguments for adding a Poseidon2 perm row with D=1 (base field).
///
/// This variant is for D=1 configurations where we have 16 base field elements
/// instead of 4 extension field limbs.
pub struct Poseidon2PermCallBase {
    /// Poseidon2 configuration for this permutation row (must be D=1).
    pub config: Poseidon2Config,
    /// Flag indicating whether a new chain is started.
    pub new_start: bool,
    /// Optional CTL exposure for each of the 16 input elements.
    /// If `None`, the element is not exposed via CTL.
    pub inputs: [Option<ExprId>; 16],
    /// Output exposure flags for the rate elements (first RATE=8 elements).
    /// When `out_ctl[i]` is true for i in 0..8, output[i] is CTL-verified.
    pub out_ctl: [bool; 8],
    /// Whether to return all 16 output elements (for challenger use).
    /// When true, outputs 8-15 are also allocated and returned, but NOT CTL-verified
    /// (they are capacity elements, constrained only by the Poseidon2 permutation itself).
    pub return_all_outputs: bool,
}

impl Default for Poseidon2PermCallBase {
    fn default() -> Self {
        Self {
            config: Poseidon2Config::BabyBearD1Width16,
            new_start: false,
            inputs: [None; 16],
            out_ctl: [false; 8],
            return_all_outputs: false,
        }
    }
}
