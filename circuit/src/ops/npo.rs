use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::any::Any;
use core::fmt::Debug;
use core::hash::Hash;

use hashbrown::HashMap;
use serde::{Deserialize, Serialize};

use super::executor::OpExecutionState;
use super::poseidon2_perm::Poseidon2Config;
use crate::NpoCircuitPlugin;

/// Opaque, string-based identifier for non-primitive operation types.
///
/// Each unique (operation-kind, configuration) pair gets its own `NpoTypeId`.
/// For example, Poseidon2 with BabyBear D=4 W=16 is `"poseidon2_perm/baby_bear_d4_w16"`.
#[derive(Clone, PartialEq, Eq, Hash, Ord, PartialOrd, Serialize, Deserialize)]
pub struct NpoTypeId(String);

impl NpoTypeId {
    /// Create a new NPO type identifier.
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// The string key.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Convenience: Poseidon2 permutation type ID for a given config.
    pub fn poseidon2_perm(config: Poseidon2Config) -> Self {
        Self::new(alloc::format!("poseidon2_perm/{}", config.variant_name()))
    }

    /// Convenience: Recompose (BF→EF packing) operation type ID.
    pub fn recompose() -> Self {
        Self::new("recompose")
    }

    /// Recompose table variant that registers per-coefficient WitnessChecks receives.
    ///
    /// Used when a D=1 Poseidon2 (or similar) must read individual BF coefficients that came
    /// from this packing op. The standard [`Self::recompose`] table omits that overhead.
    pub fn recompose_with_coeff_lookups() -> Self {
        Self::new("recompose/coeff")
    }

    /// Convenience: Unconstrained (hint) operation type ID.
    ///
    /// This is kept only for profiling / debugging purposes; Unconstrained is
    /// no longer a table-backed non-primitive op and is executed via `Op::Hint`.
    pub fn unconstrained() -> Self {
        Self::new("unconstrained")
    }
}

impl Debug for NpoTypeId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "NpoTypeId({})", self.0)
    }
}

impl core::fmt::Display for NpoTypeId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Preprocessed data for non-primitive tables, keyed by operation type.
pub type NonPrimitivePreprocessedMap<F> = HashMap<NpoTypeId, Vec<F>>;

/// Registry of NPO plugins, keyed by type ID.
pub type NpoRegistry<F> = HashMap<NpoTypeId, Arc<dyn NpoCircuitPlugin<F>>>;

/// Type-erased, plugin-owned configuration for a non-primitive operation.
///
/// Each NPO plugin both produces and consumes its own typed data through
/// this wrapper. The core infrastructure never inspects the contents.
pub struct NpoConfig(pub(crate) Arc<dyn Any + Send + Sync>);

impl NpoConfig {
    /// Wrap a concrete config value.
    pub fn new<T: Any + Send + Sync>(val: T) -> Self {
        Self(Arc::new(val))
    }

    /// Downcast to a concrete config type.
    pub fn downcast_ref<T: Any>(&self) -> Option<&T> {
        self.0.downcast_ref::<T>()
    }
}

impl Clone for NpoConfig {
    fn clone(&self) -> Self {
        // Shared, immutable plugin config; cloning just bumps the refcount.
        Self(self.0.clone())
    }
}

impl Debug for NpoConfig {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "NpoConfig(<type-erased>)")
    }
}

/// Non-primitive operations representing complex cryptographic constraints.
///
/// These operations implement sophisticated cryptographic primitives that:
/// - Have dedicated AIR tables for constraint verification
/// - Take witness values as public interface
/// - May require separate private data for complete specification
/// - Are NOT subject to primitive optimizations (CSE, constant folding)
/// - Enable modular addition of complex functionality
///
/// Non-primitive operations are isolated from primitive optimizations to:
/// 1. Maintain clean separation between basic arithmetic and complex crypto
/// 2. Allow specialized constraint systems for each operation type
/// 3. Enable parallel development of different cryptographic primitives
/// 4. Avoid optimization passes breaking complex constraint relationships
///
/// Type-erased private auxiliary data for non-primitive operations.
///
/// This data is NOT part of the witness table but provides additional
/// parameters needed to fully specify complex operations. Private data:
/// - Is set during circuit execution via `NonPrimitiveOpId`
/// - Contains sensitive information like cryptographic witnesses
/// - Is used by AIR tables to generate the appropriate constraints
///
/// Each NPO plugin both produces and consumes its own typed data through
/// this wrapper. The core infrastructure never inspects the contents.
pub struct NpoPrivateData(pub(crate) Box<dyn Any + Send + Sync>);

impl NpoPrivateData {
    /// Wrap concrete private data.
    pub fn new<T: Any + Send + Sync>(val: T) -> Self {
        Self(Box::new(val))
    }

    /// Downcast to a concrete private data type.
    pub fn downcast_ref<T: Any>(&self) -> Option<&T> {
        self.0.downcast_ref::<T>()
    }
}

impl Debug for NpoPrivateData {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "NpoPrivateData(<type-erased>)")
    }
}

/// Type-erased storage for operation execution states.
///
/// This allows each operation type to maintain its own state without
/// coupling `ExecutionContext` to specific operation implementations.
pub type OpStateMap = BTreeMap<NpoTypeId, Box<dyn OpExecutionState>>;
