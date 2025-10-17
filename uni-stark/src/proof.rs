use alloc::vec::Vec;

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
    pub trace_next: Vec<Challenge>,
    pub quotient_chunks: Vec<Vec<Challenge>>,
    pub random: Option<Vec<Challenge>>,
}
