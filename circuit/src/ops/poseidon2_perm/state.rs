//! Execution state and private data for Poseidon2 permutation operations.

use alloc::vec::Vec;

use crate::ops::poseidon2_perm::trace::Poseidon2CircuitRow;

/// Private data for Poseidon2 permutation.
///
/// Only used for Merkle mode operations. `sibling` holds extension limbs copied into the
/// capacity portion of the sponge state (length ≤ `capacity_ext` for the configured perm).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Poseidon2PermPrivateData<F> {
    pub sibling: Vec<F>,
}

/// Execution state for Poseidon2 permutation operations.
#[derive(Debug, Default)]
pub(crate) struct Poseidon2ExecutionState<F> {
    pub last_output_normal: Option<Vec<F>>,
    pub last_output_merkle: Option<Vec<F>>,
    /// Circuit rows captured during execution.
    pub rows: Vec<Poseidon2CircuitRow<F>>,
}
