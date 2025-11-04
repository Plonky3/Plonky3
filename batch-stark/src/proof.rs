use alloc::vec::Vec;

use p3_uni_stark::OpenedValues;
use serde::{Deserialize, Serialize};

use crate::config::{Challenge, Commitment, PcsProof, StarkGenericConfig};

/// A proof of batched STARK instances.
#[derive(Serialize, Deserialize)]
#[serde(bound = "")]
pub struct BatchProof<SC: StarkGenericConfig> {
    /// Commitments to all trace and quotient polynomials.
    pub commitments: BatchCommitments<Commitment<SC>>,
    /// Opened values at the out-of-domain point for all instances.
    pub opened_values: BatchOpenedValues<Challenge<SC>>,
    /// PCS opening proof for all commitments.
    pub opening_proof: PcsProof<SC>,
    /// Per-instance log2 of the extended trace domain size.
    /// For instance i, this stores `log2(|extended trace domain|) = log2(N_i) + is_zk()`.
    pub degree_bits: Vec<usize>,
}

/// Commitments for a batch-STARK proof.
#[derive(Debug, Serialize, Deserialize)]
pub struct BatchCommitments<Com> {
    /// Commitment to all main trace matrices (one per instance).
    pub main: Com,
    /// Commitment to all quotient polynomial chunks (across all instances).
    pub quotient_chunks: Com,
}

/// Opened values for all instances in a batch-STARK proof.
#[derive(Debug, Serialize, Deserialize)]
pub struct BatchOpenedValues<Challenge> {
    /// Opened values for each instance, in the same order as provided to the prover.
    pub instances: Vec<OpenedValues<Challenge>>,
}
