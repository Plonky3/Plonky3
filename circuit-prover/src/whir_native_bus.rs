//! WitnessChecks bus proof boundary for WHIR-native recursion.
//!
//! The production bus proof will be a WHIR-native LogUp over tuples
//! `(witness_id, value)`, with committed inverse oracles and terminal openings
//! for source, witness, and inverse values. The current supported primitive
//! path binds table ports to the witness table through local WHIR openings;
//! Poseidon2/MMCS stays gated until the global LogUp is implemented.

use alloc::format;

use serde::{Deserialize, Serialize};

use crate::whir_native::WhirNativeCircuitError;

/// Public proof record reserved for the global WitnessChecks LogUp.
///
/// Production instances must include committed inverse-oracle openings and a
/// signed LogUp sum over `(witness_id, value)` tuples. The verifier currently
/// rejects populated records until those checks are wired in.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "EF: Serialize", deserialize = "EF: Deserialize<'de>"))]
pub struct WhirNativeWitnessBusProof<EF> {
    pub source_tuple_count: usize,
    pub witness_tuple_count: usize,
    pub claimed_logup_sum: EF,
}

pub fn unsupported_witness_bus_logup(component: &str) -> WhirNativeCircuitError {
    WhirNativeCircuitError::UnsupportedSoundComponent(format!(
        "{component} requires the global WHIR-native WitnessChecks LogUp bus proof"
    ))
}
