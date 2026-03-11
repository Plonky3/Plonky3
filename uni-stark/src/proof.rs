use alloc::vec::Vec;

use p3_air::AirClaims;
use p3_commit::Pcs;
use serde::{Deserialize, Serialize};

use crate::StarkGenericConfig;

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

impl<F: Clone> From<&OpenedValues<F>> for AirClaims<F> {
    fn from(ov: &OpenedValues<F>) -> Self {
        Self {
            main_evals: ov.trace_local.clone(),
            main_next_evals: ov.trace_next.clone(),
            preprocessed_evals: ov.preprocessed_local.clone(),
            preprocessed_next_evals: ov.preprocessed_next.clone(),
        }
    }
}
