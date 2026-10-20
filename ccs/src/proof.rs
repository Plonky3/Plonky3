//! Reduction proofs and the witness claim a bare reduction leaves.

use alloc::vec::Vec;

use p3_multilinear_util::point::Point;
use p3_sumcheck::generic_degree::GenericDegreeProof;
use serde::{Deserialize, Serialize};

/// Every prover message of one CCS reduction.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CcsProof<F, EF> {
    /// Zerocheck over the constraint rows.
    pub zerocheck: GenericDegreeProof<F, EF>,
    /// Each matrix output at the zerocheck point.
    pub operand_evals: Vec<EF>,
    /// Sumcheck of the merged weights against the witness.
    pub linear: GenericDegreeProof<F, EF>,
    /// Witness value at the linear sumcheck point.
    pub witness_eval: EF,
}

/// Witness evaluation claim left by the reduction.
///
/// Sound only once opened against the witness commitment.
#[must_use = "the reduction is complete only once this claim is opened against the witness commitment"]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WitnessClaim<EF> {
    /// Opening point.
    pub point: Point<EF>,
    /// Expected opening value.
    pub value: EF,
}

/// A reduction proof together with the opening of every witness claim it leaves.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CommittedProof<F, EF, Opening> {
    /// The reduction to one witness evaluation.
    pub reduction: CcsProof<F, EF>,
    /// One commitment opening: the reduction's point, then one point per public region.
    pub opening: Opening,
}
