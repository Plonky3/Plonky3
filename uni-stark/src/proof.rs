use alloc::vec::Vec;

use p3_commit::Pcs;
use p3_matrix::dense::RowMajorMatrix;
use serde::{Deserialize, Serialize};

use crate::StarkConfig;

type Val<SC> = <SC as StarkConfig>::Val;
type ValMat<SC> = RowMajorMatrix<Val<SC>>;
type Com<SC> = <<SC as StarkConfig>::Pcs as Pcs<Val<SC>, ValMat<SC>>>::Commitment;
type PcsProof<SC> = <<SC as StarkConfig>::Pcs as Pcs<Val<SC>, ValMat<SC>>>::Proof;

#[derive(Serialize, Deserialize)]
#[serde(bound = "")]
pub struct Proof<SC: StarkConfig> {
    pub(crate) commitments: Commitments<Com<SC>>,
    pub(crate) opened_values: OpenedValues<SC::Challenge>,
    pub(crate) opening_proof: PcsProof<SC>,
    pub(crate) degree_bits: usize,
}

#[derive(Serialize, Deserialize)]
pub struct Commitments<Com> {
    pub(crate) trace: Com,
    pub(crate) quotient_chunks: Com,
}

#[derive(Serialize, Deserialize)]
pub struct OpenedValues<Challenge> {
    pub(crate) trace_local: Vec<Challenge>,
    pub(crate) trace_next: Vec<Challenge>,
    pub(crate) quotient_chunks: Vec<Challenge>,
}
