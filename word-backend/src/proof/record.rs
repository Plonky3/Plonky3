//! The transcript record one complete word-level proof carries.

use p3_binary_pcs::BooleanMultilinearPcs;
use p3_sumcheck::generic_degree::GenericDegreeProof;
use serde::{Deserialize, Serialize};

use crate::ShiftReductionProof;

/// Operand evaluations the vanishing check leaves for the shift reduction.
///
/// The order is the vanishing operand, then the left, right, and output operands.
pub(super) const OPERAND_EVALUATIONS: usize = 4;

/// One successful proof: the trace commitment beside its transcript record.
pub type ProvedStatement<F, EF, Pcs, Challenger> = (
    <Pcs as BooleanMultilinearPcs<EF, Challenger>>::Commitment,
    WordProof<F, EF, <Pcs as BooleanMultilinearPcs<EF, Challenger>>::Proof>,
);

/// Transcript record of one complete word-level proof.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WordProof<F, EF, Opening> {
    /// Rounds of the batched relation vanishing check.
    pub(super) zerocheck: GenericDegreeProof<F, EF>,
    /// Operand evaluations left at the point the vanishing check ends on.
    pub(super) operands: [EF; OPERAND_EVALUATIONS],
    /// Record of the reduction from shifted operands to one trace claim.
    pub(super) shift: ShiftReductionProof<F, EF>,
    /// Proof that the committed trace takes the reduced value at the reduced point.
    pub(super) opening: Opening,
}
