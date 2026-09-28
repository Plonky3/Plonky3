//! The transcript record one complete word-level proof carries.

use p3_sumcheck::generic_degree::GenericDegreeProof;
use serde::{Deserialize, Serialize};

use super::relation::OPERAND_EVALUATIONS;
use crate::ShiftReductionProof;
use crate::integer_mul::IntegerMulProof;

/// Transcript record of one complete word-level proof.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WordProof<F, EF, Opening> {
    /// Reduction of every unsigned product, present exactly when the statement declares one.
    pub(super) integer_mul: Option<IntegerMulProof<F, EF>>,
    /// Rounds of the batched relation vanishing check.
    pub(super) zerocheck: GenericDegreeProof<F, EF>,
    /// Operand evaluations left at the point the vanishing check ends on.
    pub(super) operands: [EF; OPERAND_EVALUATIONS],
    /// Record of the reduction from shifted operands to one trace claim.
    pub(super) shift: ShiftReductionProof<F, EF>,
    /// Proof that the committed trace takes the reduced value at the reduced point.
    pub(super) opening: Opening,
}
