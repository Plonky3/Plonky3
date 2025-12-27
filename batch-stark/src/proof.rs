use alloc::vec::Vec;

use p3_lookup::lookup_traits::LookupData;
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
    /// Data necessary to verify the global lookup arguments across all instances.
    pub global_lookup_data: Vec<Vec<LookupData<Challenge<SC>>>>,
    /// Per-instance log2 of the extended trace domain size.
    /// For instance i, this stores `log2(|extended trace domain|) = log2(N_i) + is_zk()`.
    pub degree_bits: Vec<usize>,
}

/// Commitments for a batch-STARK proof.
#[derive(Debug, Serialize, Deserialize)]
pub struct BatchCommitments<Com> {
    /// Commitment to all main trace matrices (one per instance).
    pub main: Com,
    /// Commitment to all permutation polynomials (one per instance).
    pub permutation: Option<Com>,
    /// Commitment to all quotient polynomial chunks (across all instances).
    pub quotient_chunks: Com,
    /// Commitment to all randomization polynomials (one per instance, if ZK is enabled).
    pub random: Option<Com>,
}

/// Opened values for a single instance in a batch-STARK proof, including lookup-related values.
#[derive(Debug, Serialize, Deserialize)]
pub struct OpenedValuesWithLookups<Challenge> {
    /// Standard opened values (trace and quotient).
    pub base_opened_values: OpenedValues<Challenge>,
    /// Opened values for the permutation polynomials at the challenge `zeta`.
    pub permutation_local: Vec<Challenge>,
    /// Opened values for the permutation polynomials at the next row `g * zeta`.
    pub permutation_next: Vec<Challenge>,
}

/// Opened values for all instances in a batch-STARK proof.
#[derive(Debug, Serialize, Deserialize)]
pub struct BatchOpenedValues<Challenge> {
    /// Opened values for each instance, in the same order as provided to the prover.
    pub instances: Vec<OpenedValuesWithLookups<Challenge>>,
}
