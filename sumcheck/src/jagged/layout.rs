//! Validated geometry for a capacity-free jagged commitment.

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
use crate::product_polynomial::ProductPolynomial;
use crate::strategy::{Basis, SumcheckProver, VariableOrder};
use crate::{SumcheckData, SumcheckError};

/// A sparse column layout and its contiguous dense representation.
///
/// Every column occupies one consecutive interval in the dense witness.
///
/// The final power-of-two suffix is virtual zero padding.
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
        // A column point names one vertex of a Boolean cube.
        if heights.is_empty() {
            return Err(JaggedLayoutError::NoColumns);
        }
        if !heights.len().is_power_of_two() {
            return Err(JaggedLayoutError::ColumnCountNotPowerOfTwo {
                columns: heights.len(),
            });
        }

        // The row bound is used for checked height validation and allocation.
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
        let dense_variables = log2_ceil_usize(capacity);

        Ok(Self {
            row_variables,
            dense_variables,
            cumulative_heights,
        })
    }

    /// Returns the number of variables addressing a sparse row.
    #[must_use]
    pub const fn row_variables(&self) -> usize {
        // The value was validated when the layout was built.
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
        // The value was derived from the checked dense capacity.
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
        // Construction always stores one terminal prefix sum.
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
        // The stored prefix table includes both endpoints of every column.
        &self.cumulative_heights
    }

    /// Proves an evaluation of a virtual jagged table.
    ///
    /// The dense witness contains only live cells in column-major order.
    ///
    /// The returned dense claim must be opened against the commitment to the same witness.
    ///
    /// # Soundness
    ///
    /// The reduction contributes at most `2m / |EF|` error for `m` dense variables.
    ///
    /// This bound does not include the binding error of the underlying PCS.
    ///
    /// # Errors
    ///
    /// - The sparse point does not match the public layout.
    /// - The dense witness length differs from the live trace area.
    pub fn prove<F, EF, Challenger>(
        &self,
        dense_witness: &[F],
        point: &JaggedPoint<EF>,
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
        if dense_witness.len() != self.area() {
            return Err(JaggedError::DenseLengthMismatch {
                expected: self.area(),
                actual: dense_witness.len(),
            });
        }

        // The committed data occupies the live prefix.
        // The power-of-two suffix is represented only by zeros inside this reduction.
        let mut dense = EF::zero_vec(self.dense_capacity());
        for (destination, &source) in dense.iter_mut().zip(dense_witness) {
            *destination = EF::from(source);
        }

        // On Boolean dense indices this selector maps the contiguous representation back to rows.
        let selector = selector.table(point);
        let claimed_value = dense
            .iter()
            .zip(&selector)
            .map(|(&value, &weight)| value * weight)
            .sum();

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
        Ok(JaggedProverOutput {
            proof,
            sparse_value: claimed_value,
            dense_claim,
        })
    }

    /// Verifies a sparse evaluation reduction.
    ///
    /// Acceptance returns the single dense claim the underlying PCS must authenticate.
    ///
    /// # Soundness
    ///
    /// The terminal selector is evaluated independently through a width-four branching program.
    ///
    /// A false sparse claim therefore becomes a false dense claim except with probability `2m / |EF|`.
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

        // The proof must not smuggle unused grinding witnesses into a zero-difficulty reduction.
        if !proof.sumcheck.pow_witnesses.is_empty() {
            return Err(SumcheckError::PowWitnessCountMismatch {
                expected: 0,
                actual: proof.sumcheck.pow_witnesses.len(),
            }
            .into());
        }

        // Replay against the same public statement that seeded the prover.
        let mut transcript = JaggedVerifierTranscript::<Challenger, F, EF>::new(
            challenger,
            self,
            point,
            claimed_value,
        );
        let mut terminal = claimed_value;
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
    use alloc::vec;

    use super::*;

    #[test]
    fn mixed_heights_form_contiguous_dense_intervals() {
        // Fixture state:
        //
        //     heights   [3, 0, 5, 1]
        //     prefixes  [0, 3, 3, 8, 9]
        //     capacity  16 cells
        let layout = JaggedLayout::new(3, &[3, 0, 5, 1]).unwrap();

        assert_eq!(layout.cumulative_heights(), &[0, 3, 3, 8, 9]);
        assert_eq!(layout.area(), 9);
        assert_eq!(layout.dense_variables(), 4);
        assert_eq!(layout.dense_capacity(), 16);
        assert_eq!(layout.column_height(1), 0);
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

        // Keep the allocation alive until every assertion has read it.
        drop(vec![empty]);
    }
}
