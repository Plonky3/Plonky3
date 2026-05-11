//! Poseidon2/MMCS boundary for WHIR-native recursive circuit proofs.
//!
//! This module owns the strict gate for Poseidon2/MMCS until the expanded
//! `Poseidon2CircuitAir` trace is proved by the multilinear AIR adapter and
//! its WitnessChecks traffic is covered by the global bus proof.

use alloc::format;
use alloc::string::String;

use p3_circuit::ops::Poseidon2Config;
use serde::{Deserialize, Serialize};

use crate::whir_native::WhirNativeCircuitError;
use crate::whir_native_air::WhirNativeAirProof;

/// Public proof record reserved for expanded Poseidon2/MMCS AIR tables.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "EF: Serialize", deserialize = "EF: Deserialize<'de>"))]
pub struct WhirNativePoseidon2AirProof<EF> {
    pub op_type: String,
    pub config: Poseidon2Config,
    pub air_proof: WhirNativeAirProof<EF>,
}

pub fn unsupported_poseidon2_message(op_type: &str) -> String {
    format!(
        "non-primitive table `{op_type}` requires an expanded WHIR-native Poseidon2/MMCS AIR proof plus the global WitnessChecks LogUp bus before it can enter comparison timing"
    )
}

pub fn unsupported_poseidon2_component(op_type: &str) -> WhirNativeCircuitError {
    WhirNativeCircuitError::UnsupportedSoundComponent(unsupported_poseidon2_message(op_type))
}

#[allow(dead_code)]
pub fn require_poseidon2_whir_native_soundness(
    op_type: &str,
    _config: Poseidon2Config,
) -> Result<(), WhirNativeCircuitError> {
    Err(unsupported_poseidon2_component(op_type))
}
