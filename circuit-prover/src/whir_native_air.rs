//! Multilinear AIR adapter boundary for WHIR-native circuit proofs.
//!
//! The recursive WHIR outer proof must eventually route every expanded
//! non-primitive AIR through this module: commit main columns as multilinear
//! WHIR oracles, derive public preprocessed columns, and prove AIR constraints
//! with multilinear zero-checks. Until that backend lands, callers must keep
//! non-primitive AIRs behind [`WhirNativeCircuitError::UnsupportedSoundComponent`].

use alloc::format;

use serde::{Deserialize, Serialize};

use crate::whir_native::WhirNativeCircuitError;
use crate::whir_native::WhirNativeLocalConstraintProof;

/// Public proof record reserved for a WHIR-native multilinear AIR check.
///
/// The record intentionally references the same local sumcheck/opening shape
/// used by primitive tables so the final adapter can be attached without
/// introducing STARK quotient or FRI proof data.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "EF: Serialize", deserialize = "EF: Deserialize<'de>"))]
pub struct WhirNativeAirProof<EF> {
    pub table_index: usize,
    pub main_width: usize,
    pub preprocessed_width: usize,
    pub local_proof: WhirNativeLocalConstraintProof<EF>,
}

pub fn unsupported_multilinear_air_adapter(component: &str) -> WhirNativeCircuitError {
    WhirNativeCircuitError::UnsupportedSoundComponent(format!(
        "{component} requires the WHIR-native multilinear AIR adapter before it can enter oracle-only recursion"
    ))
}
