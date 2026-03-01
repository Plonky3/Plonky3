use alloc::vec::Vec;

use p3_commit::Pcs;
use serde::{Deserialize, Serialize};

use crate::StarkGenericConfig;
use crate::security::{ConjecturedSecurity, ProvenSecurity, StarkSecurityParams};

type Com<SC> = <<SC as StarkGenericConfig>::Pcs as Pcs<
    <SC as StarkGenericConfig>::Challenge,
    <SC as StarkGenericConfig>::Challenger,
>>::Commitment;
type PcsProof<SC> = <<SC as StarkGenericConfig>::Pcs as Pcs<
    <SC as StarkGenericConfig>::Challenge,
    <SC as StarkGenericConfig>::Challenger,
>>::Proof;

#[derive(Serialize, Deserialize)]
#[serde(bound = "")]
pub struct Proof<SC: StarkGenericConfig> {
    pub commitments: Commitments<Com<SC>>,
    pub opened_values: OpenedValues<SC::Challenge>,
    pub opening_proof: PcsProof<SC>,
    pub degree_bits: usize,
}

impl<SC: StarkGenericConfig> Proof<SC> {
    /// Conjectured security level (in bits).
    /// Based on the random-words formula in [2025/2010](https://eprint.iacr.org/2025/2010) §1.5.
    ///
    /// See [`ConjecturedSecurity`](crate::security::ConjecturedSecurity).
    pub fn conjectured_security(&self, params: &StarkSecurityParams) -> ConjecturedSecurity {
        ConjecturedSecurity::compute_from_params(params)
    }

    /// Proven security level (in bits).
    ///
    /// See [`ProvenSecurity`](crate::security::ProvenSecurity).
    pub fn proven_security(&self, params: &StarkSecurityParams) -> ProvenSecurity {
        ProvenSecurity::compute_from_proof(self.degree_bits, params)
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Commitments<Com> {
    pub trace: Com,
    pub quotient_chunks: Com,
    pub random: Option<Com>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct OpenedValues<Challenge> {
    pub trace_local: Vec<Challenge>,
    /// Main trace evaluated at `g * zeta`.
    ///
    /// `None` when the AIR has no transition constraints and does not access the next row.
    pub trace_next: Option<Vec<Challenge>>,
    pub preprocessed_local: Option<Vec<Challenge>>,
    pub preprocessed_next: Option<Vec<Challenge>>, // may not always be necessary
    pub quotient_chunks: Vec<Vec<Challenge>>,
    pub random: Option<Vec<Challenge>>,
}
