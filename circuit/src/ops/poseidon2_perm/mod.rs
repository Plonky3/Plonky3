//! Poseidon2 permutation non-primitive operation (one Poseidon2 call per row).
//!
//! This module contains all Poseidon2 permutation related code:
//! - Builder API ([`Poseidon2PermCall`], [`CircuitBuilder::add_poseidon2_perm`])
//! - Executor ([`executor::Poseidon2PermExecutor`])
//! - Execution state ([`state::Poseidon2ExecutionState`])
//! - Private data ([`Poseidon2PermPrivateData`])
//! - Trace generation types ([`Poseidon2Params`], [`Poseidon2CircuitRow`], [`Poseidon2Trace`])
//!
//! This operation supports both standard hashing and Merkle path verification:
//!
//! - **Hashing**: Performs a standard Poseidon2 permutation.
//! - **Chaining**: Can start a new hash or continue from the previous row's output
//!   (controlled by `new_start`).
//! - **Merkle Path Verification**: When `merkle_path` is enabled, conditionally arranges
//!   inputs (sibling vs. computed hash) based on a direction bit (`mmcs_bit`).
//! - **Index Accumulation**: Accumulates path indices (`mmcs_index_sum`) to verify the
//!   leaf's position in the tree.
//!
//! [`CircuitBuilder::add_poseidon2_perm`]: crate::builder::CircuitBuilder::add_poseidon2_perm

mod builder;
pub mod call;
pub(crate) mod config;
pub(crate) mod executor;
pub(crate) mod plugin;
pub mod state;
pub mod trace;

pub use call::{Poseidon2PermCall, Poseidon2PermCallBase};
pub use config::Poseidon2Config;
pub(crate) use config::Poseidon2PermExec;
pub(crate) use plugin::Poseidon2CircuitPlugin;
pub use state::Poseidon2PermPrivateData;
pub use trace::{
    BabyBearD1Width16, GoldilocksD2Width8, KoalaBearD1Width16, Poseidon2CircuitRow,
    Poseidon2Params, Poseidon2Trace, generate_poseidon2_trace,
};
