use alloc::vec::Vec;

use p3_commit::Pcs;
use serde::{Deserialize, Serialize};

use crate::config::MultiStarkGenericConfig as MSGC;

type Com<SC> = <<SC as p3_uni_stark::StarkGenericConfig>::Pcs as Pcs<
    <SC as p3_uni_stark::StarkGenericConfig>::Challenge,
    <SC as p3_uni_stark::StarkGenericConfig>::Challenger,
>>::Commitment;
type PcsProof<SC> = <<SC as p3_uni_stark::StarkGenericConfig>::Pcs as Pcs<
    <SC as p3_uni_stark::StarkGenericConfig>::Challenge,
    <SC as p3_uni_stark::StarkGenericConfig>::Challenger,
>>::Proof;

#[derive(Serialize, Deserialize)]
#[serde(bound = "")]
pub struct MultiProof<SC: MSGC> {
    pub commitments: MultiCommitments<Com<SC>>,
    pub opened_values: MultiOpenedValues<<SC as p3_uni_stark::StarkGenericConfig>::Challenge>,
    pub opening_proof: PcsProof<SC>,
    /// Per-instance log2 of the extended trace domain size.
    /// For instance i, this stores `log2(|extended trace domain|) = log2(N_i) + is_zk()`.
    pub degree_bits: Vec<usize>,
}

#[derive(Serialize, Deserialize)]
pub struct MultiCommitments<Com> {
    pub main: Com,
    pub quotient_chunks: Com,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct InstanceOpenedValues<Challenge> {
    pub trace_local: Vec<Challenge>,
    pub trace_next: Vec<Challenge>,
    // one Vec<Challenge> per chunk (values for each flattened-basis column at zeta)
    pub quotient_chunks: Vec<Vec<Challenge>>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct MultiOpenedValues<Challenge> {
    pub instances: Vec<InstanceOpenedValues<Challenge>>,
}
