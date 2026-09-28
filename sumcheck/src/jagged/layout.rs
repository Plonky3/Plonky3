//! Validated geometry for a capacity-free jagged commitment.
//!
//! See [ePrint 2025/917](https://eprint.iacr.org/2025/917) for the construction and equation numbering.

use alloc::vec::Vec;

use p3_challenger::fs::TranscriptField;
use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::ExtensionField;
use p3_multilinear_util::poly::Poly;
use p3_util::log2_ceil_usize;

use super::selector::JaggedSelector;
use super::transcript::{JaggedProverTranscript, JaggedVerifierTranscript};
use super::{
    JaggedDenseClaim, JaggedError, JaggedLayoutError, JaggedPoint, JaggedProof, JaggedProverOutput,
};
use crate::SumcheckData;
use crate::product_polynomial::ProductPolynomial;
use crate::strategy::{Basis, SumcheckProver, VariableOrder};

/// A sparse column layout and its contiguous dense representation.
///
/// Every column occupies one consecutive interval in the dense witness.
///
/// The suffix filling out the power-of-two envelope holds no sparse cell, and the selector ignores it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JaggedLayout {
    /// Number of variables addressing a row in the sparse view.
    row_variables: usize,
    /// Number of variables addressing the dense power-of-two envelope.
    dense_variables: usize,
    /// Start of every column followed by the total live area.
    cumulative_heights: Vec<usize>,
}

impl JaggedLayout {
    /// Builds a layout from the live height of every sparse column.
    ///
    /// Columns may be empty and heights need not be powers of two.
    ///
    /// # Errors
    ///
    /// - The column count is zero or not a power of two.
    /// - The row bound or dense area cannot be represented by `usize`.
    /// - A column height exceeds the declared row bound.
    pub fn new(row_variables: usize, heights: &[usize]) -> Result<Self, JaggedLayoutError> {
        Self::with_min_dense_variables(row_variables, heights, 0)
    }

    /// Builds a layout whose envelope holds at least a given number of variables.
    ///
    /// A folding commitment scheme refuses an arity below its factor, which a short trace falls under.
    ///
    /// Widening buys only padding, which no sparse cell reaches and no constraint binds.
    ///
    /// # Errors
    ///
    /// - Everything the smallest envelope is rejected for, and an arity no machine index holds.
    pub fn with_min_dense_variables(
        row_variables: usize,
        heights: &[usize],
        min_dense_variables: usize,
    ) -> Result<Self, JaggedLayoutError> {
        // A column point names one vertex of a Boolean cube.
        if heights.is_empty() {
            return Err(JaggedLayoutError::NoColumns);
        }
        if !heights.len().is_power_of_two() {
            return Err(JaggedLayoutError::ColumnCountNotPowerOfTwo {
                columns: heights.len(),
            });
        }

        let row_shift =
            u32::try_from(row_variables).map_err(|_| JaggedLayoutError::RowVariablesOverflow {
                variables: row_variables,
            })?;
        let row_bound =
            1usize
                .checked_shl(row_shift)
                .ok_or(JaggedLayoutError::RowVariablesOverflow {
                    variables: row_variables,
                })?;

        // Prefix sums are the sparse-to-dense bijection.
        //
        //     column y  ->  [prefix[y], prefix[y + 1])
        let mut cumulative_heights = Vec::with_capacity(heights.len() + 1);
        cumulative_heights.push(0usize);
        for (column, &height) in heights.iter().enumerate() {
            if height > row_bound {
                return Err(JaggedLayoutError::HeightExceedsRowBound {
                    column,
                    height,
                    maximum: row_bound,
                });
            }
            let area = cumulative_heights[column]
                .checked_add(height)
                .ok_or(JaggedLayoutError::AreaOverflow { column })?;
            cumulative_heights.push(area);
        }

        // The dense multilinear has the smallest power-of-two domain covering all live cells.
        // An empty trace still has one constant zero evaluation.
        let area = cumulative_heights[heights.len()];
        let capacity = area
            .max(1)
            .checked_next_power_of_two()
            .ok_or(JaggedLayoutError::DenseAreaOverflow { area })?;
        let dense_variables = log2_ceil_usize(capacity).max(min_dense_variables);

        // The envelope index must stay inside a machine word after the floor is applied.
        if dense_variables >= usize::BITS as usize {
            return Err(JaggedLayoutError::DenseVariablesOverflow {
                variables: dense_variables,
            });
        }

        Ok(Self {
            row_variables,
            dense_variables,
            cumulative_heights,
        })
    }

    /// Returns the number of variables addressing a sparse row.
    #[must_use]
    pub const fn row_variables(&self) -> usize {
        self.row_variables
    }

    /// Returns the number of variables addressing a sparse column.
    #[must_use]
    pub const fn column_variables(&self) -> usize {
        // Construction requires a nonzero power-of-two column count.
        self.num_columns().trailing_zeros() as usize
    }

    /// Returns the number of variables addressing the dense witness.
    #[must_use]
    pub const fn dense_variables(&self) -> usize {
        self.dense_variables
    }

    /// Returns the number of sparse columns.
    #[must_use]
    pub const fn num_columns(&self) -> usize {
        // The final entry is the total rather than a column start.
        self.cumulative_heights.len() - 1
    }

    /// Returns the number of live witness cells.
    #[must_use]
    pub fn area(&self) -> usize {
        self.cumulative_heights[self.num_columns()]
    }

    /// Returns the power-of-two size of the dense multilinear.
    #[must_use]
    pub const fn dense_capacity(&self) -> usize {
        // Construction rejects an exponent that does not fit in a machine index.
        1usize << self.dense_variables
    }

    /// Returns the live height of one sparse column.
    ///
    /// # Panics
    ///
    /// Panics when the column index is outside the layout.
    #[must_use]
    pub fn column_height(&self, column: usize) -> usize {
        // Adjacent prefix sums delimit exactly one column.
        self.cumulative_heights[column + 1] - self.cumulative_heights[column]
    }

    /// Returns every column boundary followed by the total live area.
    #[must_use]
    pub fn cumulative_heights(&self) -> &[usize] {
        &self.cumulative_heights
    }

    /// Proves that a virtual jagged table takes a caller-supplied value at a sparse point.
    ///
    /// The witness is the committed vector itself: this layout's live cells in column-major order, followed by padding out to the power-of-two envelope.
    ///
    /// The returned dense claim is an evaluation of exactly that vector, so passing the live cells alone would only ever open against zero padding.
    ///
    /// # Soundness
    ///
    /// The reduction contributes at most `2m / |EF|` error, where `m` is the base-two logarithm of the padded live area rather than the row bound, which may be far larger.
    ///
    /// This bound does not include the binding error of the underlying PCS.
    ///
    /// # Errors
    ///
    /// - The sparse point does not match the public layout.
    /// - The dense witness length differs from the power-of-two envelope.
    /// - The witness does not take the supplied value at the supplied point.
    pub fn prove<F, EF, Challenger>(
        &self,
        dense_witness: &[F],
        point: &JaggedPoint<EF>,
        claimed_value: EF,
        challenger: &mut Challenger,
    ) -> Result<JaggedProverOutput<F, EF>, JaggedError>
    where
        F: TranscriptField,
        EF: ExtensionField<F>,
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        // Reject malformed public geometry before touching the transcript.
        let selector = JaggedSelector::new(self);
        selector.validate_point(point)?;
        if dense_witness.len() != self.dense_capacity() {
            return Err(JaggedError::DenseLengthMismatch {
                expected: self.dense_capacity(),
                actual: dense_witness.len(),
            });
        }

        // On Boolean dense indices this selector maps the contiguous representation back to rows.
        // Only the live prefix contributes, and a base-field cell scales a weight without a full extension product.
        let selector = selector.table(point);
        let witness_value = dense_witness
            .iter()
            .zip(&selector)
            .map(|(&cell, &weight)| weight * cell)
            .sum::<EF>();

        // A reduction that invented its own statement could not be composed with the claim it was called to discharge.
        if witness_value != claimed_value {
            return Err(JaggedError::ClaimMismatch);
        }

        // The sumcheck runs over the envelope, so the padding the caller committed must ride along.
        let dense = dense_witness.iter().copied().map(EF::from).collect();

        // Both parties seed from the complete sparse statement before any challenge is drawn.
        let mut transcript = JaggedProverTranscript::<Challenger, F, EF>::new(
            challenger,
            self,
            point,
            claimed_value,
        );

        // A product sumcheck reduces the sparse evaluation to one product at a random dense point.
        let polynomial = ProductPolynomial::new_unpacked(
            VariableOrder::Prefix,
            Poly::new(dense),
            Poly::new(selector),
        );
        let mut prover = SumcheckProver::new(polynomial, claimed_value);
        let mut sumcheck = SumcheckData::default();
        let dense_point = transcript.product_sumcheck(|challenger| {
            prover.compute_sumcheck_polynomials(
                &mut sumcheck,
                challenger,
                self.dense_variables(),
                0,
                None,
            )
        });

        // Settling the final held challenge leaves one dense evaluation.
        let dense_evaluation = prover.evals().as_slice()[0];
        transcript.dense_evaluation(dense_evaluation);
        transcript.finish();

        let dense_claim = JaggedDenseClaim {
            point: dense_point,
            value: dense_evaluation,
        };
        let proof = JaggedProof {
            sumcheck,
            dense_evaluation,
        };
        Ok(JaggedProverOutput { proof, dense_claim })
    }

    /// Verifies a sparse evaluation reduction.
    ///
    /// Acceptance returns the single dense claim the underlying PCS must authenticate.
    ///
    /// # Security
    ///
    /// The terminal relation is one equation in the surviving sumcheck value and the claimed dense evaluation, and dividing the first by the publicly computable selector weight satisfies it for any claimed sparse value, over any witness or none.
    ///
    /// What rules that out is the caller opening its commitment at the returned point and finding exactly the returned value, so discarding the returned claim is not a weakened check but unconditional acceptance.
    ///
    /// The commitment must name as many variables as the dense arity and hold this layout's live cells in column-major order, while evaluations past the live area are unconstrained because the selector vanishes there.
    ///
    /// The column heights must already be bound into the transcript, normally by that commitment, before the sparse point is drawn.
    ///
    /// That ordering is on the caller when the reduction is reached through this entry point.
    ///
    /// Sealing the geometry before drawing the point makes that ordering structural.
    ///
    /// Prefer that route where one exists.
    ///
    /// The caller-obligations page under `docs` records the raw route.
    ///
    /// # Minimum non-degenerate shape
    ///
    /// The delegated sumcheck runs one round per dense variable.
    ///
    /// ```text
    ///     live area   dense variables   challenges drawn
    ///     0 or 1      0                 none
    ///     2 or more   at least 1        at least one
    /// ```
    ///
    /// The smallest shape that folds anything is a live area of two.
    ///
    /// Below it no challenge is drawn, which is harmless rather than vacuous.
    ///
    /// A one-cell dense multilinear has no interior left to test.
    ///
    /// Its terminal relation weighs the surviving evaluation by a public selector value.
    ///
    /// Both sides are public or pinned by the commitment, and the claim names that one cell.
    ///
    /// The seed still binds the whole statement, so two such instances part anyway.
    ///
    /// What the small shape does rest on is the caller opening the claim it returns.
    ///
    /// A vanishing selector leaves the terminal equation pinning nothing.
    ///
    /// # Soundness
    ///
    /// The terminal selector is evaluated independently through a width-four branching program.
    ///
    /// A false sparse claim therefore becomes a false dense claim except with probability `2m / |EF|`, for the same dense arity `m` as on the proving side.
    ///
    /// The bound is meaningful only when the extension is large enough for the target security level.
    ///
    /// # Errors
    ///
    /// - The sparse point does not match the public layout.
    /// - The proof has a malformed sumcheck shape.
    /// - A sumcheck round or the terminal product relation fails.
    pub fn verify<F, EF, Challenger>(
        &self,
        point: &JaggedPoint<EF>,
        claimed_value: EF,
        proof: &JaggedProof<F, EF>,
        challenger: &mut Challenger,
    ) -> Result<JaggedDenseClaim<EF>, JaggedError>
    where
        F: TranscriptField,
        EF: ExtensionField<F>,
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        // Reject caller-owned shape mismatches before advancing the sponge.
        let selector = JaggedSelector::new(self);
        selector.validate_point(point)?;

        // Replay against the same public statement that seeded the prover.
        let mut transcript = JaggedVerifierTranscript::<Challenger, F, EF>::new(
            challenger,
            self,
            point,
            claimed_value,
        );
        let mut terminal = claimed_value;

        // The delegated round verifier binds both proof counts before it folds anything.
        let dense_point = match transcript.product_sumcheck(|challenger| {
            proof.sumcheck.verify_rounds(
                challenger,
                &mut terminal,
                self.dense_variables(),
                0,
                Basis::Evaluation,
            )
        }) {
            Ok(point) => point,
            Err(error) => {
                transcript.abort();
                return Err(error.into());
            }
        };

        // The surviving value is transcript-bound before any algebraic rejection is returned.
        transcript.dense_evaluation(proof.dense_evaluation);
        transcript.finish();

        // Paper Eq. (4) holds only on Boolean dense indices.
        // The branching program computes the correct multilinear extension at this field point.
        let selector_evaluation = selector.evaluate(point, &dense_point);
        if terminal != proof.dense_evaluation * selector_evaluation {
            return Err(JaggedError::TerminalMismatch);
        }

        Ok(JaggedDenseClaim {
            point: dense_point,
            value: proof.dense_evaluation,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mixed_heights_form_contiguous_dense_intervals() {
        // Fixture state:
        //
        //     Heights   [3, 0, 5, 1]
        //     Prefixes  [0, 3, 3, 8, 9]
        //     Capacity  16 cells
        let layout = JaggedLayout::new(3, &[3, 0, 5, 1]).unwrap();

        assert_eq!(layout.cumulative_heights(), &[0, 3, 3, 8, 9]);
        assert_eq!(layout.area(), 9);
        assert_eq!(layout.dense_variables(), 4);
        assert_eq!(layout.dense_capacity(), 16);
        assert_eq!(layout.column_height(1), 0);
    }

    #[test]
    fn an_envelope_floor_buys_padding_and_nothing_else() {
        // Two live cells give an envelope of one variable, below what a folding schedule accepts.
        let heights = [1, 1];
        assert_eq!(JaggedLayout::new(1, &heights).unwrap().dense_variables(), 1);

        // The floor widens the envelope, leaving the live geometry underneath untouched.
        let raised = JaggedLayout::with_min_dense_variables(1, &heights, 4).unwrap();
        assert_eq!(raised.dense_variables(), 4);
        assert_eq!(raised.dense_capacity(), 16);
        assert_eq!(raised.area(), 2);
        assert_eq!(raised.cumulative_heights(), &[0, 1, 2]);

        let unraised = JaggedLayout::with_min_dense_variables(1, &heights, 1).unwrap();
        assert_eq!(unraised, JaggedLayout::new(1, &heights).unwrap());

        // An envelope wider than a machine index has no representable capacity.
        assert_eq!(
            JaggedLayout::with_min_dense_variables(1, &heights, usize::BITS as usize),
            Err(JaggedLayoutError::DenseVariablesOverflow {
                variables: usize::BITS as usize
            })
        );
    }

    #[test]
    fn malformed_geometry_is_rejected_at_construction() {
        // No Boolean point addresses three columns.
        assert_eq!(
            JaggedLayout::new(2, &[1, 1, 1]),
            Err(JaggedLayoutError::ColumnCountNotPowerOfTwo { columns: 3 })
        );

        // A two-variable row point addresses only four rows.
        assert_eq!(
            JaggedLayout::new(2, &[5]),
            Err(JaggedLayoutError::HeightExceedsRowBound {
                column: 0,
                height: 5,
                maximum: 4,
            })
        );

        // An empty height list has no sparse coordinate space.
        assert_eq!(JaggedLayout::new(0, &[]), Err(JaggedLayoutError::NoColumns));

        // The empty trace is the constant zero polynomial over one dense cell.
        let empty = JaggedLayout::new(3, &[0, 0]).unwrap();
        assert_eq!(empty.area(), 0);
        assert_eq!(empty.dense_capacity(), 1);
        assert_eq!(empty.dense_variables(), 0);
    }

    #[test]
    fn unrepresentable_geometry_is_rejected_at_construction() {
        // These guards are what keep the shift behind the dense capacity in range.
        assert_eq!(
            JaggedLayout::new(usize::BITS as usize, &[1]),
            Err(JaggedLayoutError::RowVariablesOverflow {
                variables: usize::BITS as usize
            })
        );

        // Two columns at half the index space each overflow the running prefix sum.
        let half = usize::MAX / 2 + 1;
        assert_eq!(
            JaggedLayout::new(usize::BITS as usize - 1, &[half, half]),
            Err(JaggedLayoutError::AreaOverflow { column: 1 })
        );

        // An area above the largest representable power of two has no envelope to round up to.
        assert_eq!(
            JaggedLayout::new(usize::BITS as usize - 1, &[half, 1]),
            Err(JaggedLayoutError::DenseAreaOverflow { area: half + 1 })
        );
    }
}
