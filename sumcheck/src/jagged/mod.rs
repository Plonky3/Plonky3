//! Capacity-free commitments to columns of unequal height.
//!
//! This module implements the basic sparse-to-dense reduction from ePrint 2025/917.
//!
//! A downstream multilinear PCS authenticates the live cells concatenated column by column, padded to a power-of-two envelope.
//!
//! The reduction converts an evaluation of the virtual zero-padded table into one dense claim.

mod error;
mod layout;
mod selector;
mod transcript;

pub use error::{JaggedError, JaggedLayoutError};
pub use layout::JaggedLayout;
use p3_multilinear_util::point::Point;
use serde::{Deserialize, Serialize};

use crate::SumcheckData;

/// An evaluation point in the virtual sparse row-by-column space.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JaggedPoint<F> {
    /// Coordinates addressing a row inside one sparse column.
    row: Point<F>,
    /// Coordinates selecting one sparse column.
    column: Point<F>,
}

impl<F> JaggedPoint<F> {
    /// Builds a sparse point from its row and column coordinates.
    #[must_use]
    pub const fn new(row: Point<F>, column: Point<F>) -> Self {
        // Coordinate widths are validated against a layout at protocol entry.
        Self { row, column }
    }

    /// Returns the row coordinates.
    pub const fn row(&self) -> &Point<F> {
        // Return the borrowed point without re-encoding its coordinates.
        &self.row
    }

    /// Returns the column coordinates.
    pub const fn column(&self) -> &Point<F> {
        // Return the borrowed point without re-encoding its coordinates.
        &self.column
    }
}

/// A claim on the dense multilinear authenticated by the underlying PCS.
///
/// Dropping this value without opening the commitment at its point makes the surrounding verification accept everything.
#[must_use]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JaggedDenseClaim<F> {
    /// Point produced by the quadratic sumcheck challenges.
    point: Point<F>,
    /// Dense multilinear evaluation at the produced point.
    value: F,
}

impl<F> JaggedDenseClaim<F> {
    /// Returns the dense evaluation point.
    pub const fn point(&self) -> &Point<F> {
        // The point is owned by the claim and remains in protocol order.
        &self.point
    }

    /// Returns the claimed dense evaluation.
    #[must_use]
    pub const fn value(&self) -> &F {
        // Borrowing avoids imposing a copy bound on the field element.
        &self.value
    }
}

/// Proof reducing one sparse evaluation to one dense evaluation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct JaggedProof<F, EF> {
    /// Quadratic sumcheck proving the sparse evaluation identity.
    sumcheck: SumcheckData<F, EF>,
    /// Dense evaluation left for the underlying PCS to authenticate.
    dense_evaluation: EF,
}

impl<F, EF> JaggedProof<F, EF> {
    /// Returns the delegated quadratic sumcheck proof.
    #[must_use]
    pub const fn sumcheck(&self) -> &SumcheckData<F, EF> {
        // Proof internals stay immutable after transcript construction.
        &self.sumcheck
    }

    /// Returns the dense evaluation left for PCS authentication.
    #[must_use]
    pub const fn dense_evaluation(&self) -> &EF {
        // The verifier binds this exact value before checking the terminal relation.
        &self.dense_evaluation
    }
}

/// Prover output carrying both sides of the sparse-to-dense reduction.
#[must_use]
#[derive(Clone, Debug)]
pub struct JaggedProverOutput<F, EF> {
    /// Proof consumed by the sparse verifier.
    proof: JaggedProof<F, EF>,
    /// Dense claim passed to the underlying PCS.
    dense_claim: JaggedDenseClaim<EF>,
}

impl<F, EF> JaggedProverOutput<F, EF> {
    /// Returns the sparse reduction proof.
    #[must_use]
    pub const fn proof(&self) -> &JaggedProof<F, EF> {
        // Keep ownership with the aggregate result unless the caller splits it.
        &self.proof
    }

    /// Returns the dense claim to authenticate through the underlying PCS.
    pub const fn dense_claim(&self) -> &JaggedDenseClaim<EF> {
        // The claim is the only value delegated to the commitment scheme.
        &self.dense_claim
    }

    /// Splits the output into its proof and its dense claim.
    pub fn into_parts(self) -> (JaggedProof<F, EF>, JaggedDenseClaim<EF>) {
        // Ownership passes through without copying any proof or point data.
        (self.proof, self.dense_claim)
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use p3_challenger::FieldChallenger;
    use p3_field::{BasedVectorSpace, PrimeCharacteristicRing};
    use p3_multilinear_util::poly::Poly;
    use proptest::prelude::*;

    use super::selector::JaggedSelector;
    use super::transcript::JaggedProverTranscript;
    use super::*;
    use crate::SumcheckError;
    use crate::tests::{EF, F, MyChallenger, challenger};

    // Builds an extension coordinate from four independent base coefficients.
    // A point confined to the prime subfield would hide any mistake only a true extension exposes.
    fn extension(seeds: &[u32]) -> EF {
        EF::from_basis_coefficients_fn(|index| F::from_u32(seeds[index]))
    }

    fn extension_point(seeds: &[u32]) -> Point<EF> {
        Point::new(
            seeds
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| extension(c))
                .collect(),
        )
    }

    // Equality weight of one Boolean index against a point, written straight from the definition.
    // Coordinates are most-significant first, so the leading coordinate owns the top index bit.
    fn equality_weight(point: &Point<EF>, index: usize) -> EF {
        let width = point.num_variables();
        (0..width)
            .map(|position| {
                let bit = (index >> (width - 1 - position)) & 1 == 1;
                if bit {
                    point[position]
                } else {
                    EF::ONE - point[position]
                }
            })
            .product()
    }

    // Reference evaluation of the virtual jagged table, built only from the column heights.
    // It never mentions a selector, a dense index or a sumcheck, so it cannot drift with them.
    fn jagged_evaluation(heights: &[usize], witness: &[F], point: &JaggedPoint<EF>) -> EF {
        let mut total = EF::ZERO;
        let mut start = 0;
        for (column, &height) in heights.iter().enumerate() {
            let column_weight = equality_weight(point.column(), column);
            for row in 0..height {
                total += column_weight * equality_weight(point.row(), row) * witness[start + row];
            }
            start += height;
        }
        total
    }

    // Evaluation of the committed dense vector, which is the claim the underlying PCS must answer.
    fn committed_evaluation(witness: &[F], point: &Point<EF>) -> EF {
        Poly::new(witness.to_vec()).eval_base(point)
    }

    fn fixture() -> (JaggedLayout, Vec<usize>, Vec<F>, JaggedPoint<EF>) {
        // Four unequal columns concatenate to nine live cells inside a sixteen-cell envelope.
        // The seven padding cells are nonzero, so a reduction that silently assumed zeros is caught here.
        let heights = vec![3, 0, 5, 1];
        let layout = JaggedLayout::new(3, &heights).unwrap();
        let dense = (1..=layout.dense_capacity())
            .map(|value| F::from_u64(value as u64))
            .collect();
        let point = JaggedPoint::new(
            extension_point(&[2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]),
            extension_point(&[41, 43, 47, 53, 59, 61, 67, 71]),
        );
        (layout, heights, dense, point)
    }

    fn seeded_challenge(layout: &JaggedLayout, point: &JaggedPoint<EF>, value: EF) -> EF {
        // Draw from the lent sponge exactly where the delegated sumcheck draws its first challenge.
        let mut challenger = challenger();
        let mut transcript = JaggedProverTranscript::<MyChallenger, F, EF>::new(
            &mut challenger,
            layout,
            point,
            value,
        );
        let challenge = transcript.product_sumcheck(FieldChallenger::sample_algebra_element::<EF>);

        // Every described step must be played before the transcript may be dropped.
        transcript.dense_evaluation(EF::ZERO);
        transcript.finish();
        challenge
    }

    #[test]
    fn every_public_statement_component_separates_two_transcript_seeds() {
        // Each mutation below keeps the point widths and the round count.
        // Only the seed can tell the two statements apart, so the challenge must move with it.
        let (layout, _, _, point) = fixture();
        let value = EF::from_u64(42);
        let baseline = seeded_challenge(&layout, &point, value);
        assert_eq!(baseline, seeded_challenge(&layout, &point, value));

        // Mutation: move one live row between columns.
        let moved = JaggedLayout::new(3, &[2, 1, 5, 1]).unwrap();
        assert_eq!(moved.dense_variables(), layout.dense_variables());
        assert_ne!(baseline, seeded_challenge(&moved, &point, value));

        // Mutation: shift one row coordinate of the sparse point.
        let mut shifted = point.row().as_slice().to_vec();
        shifted[0] += EF::ONE;
        let shifted_point = JaggedPoint::new(Point::new(shifted), point.column().clone());
        assert_ne!(baseline, seeded_challenge(&layout, &shifted_point, value));

        // Mutation: shift the claimed sparse evaluation alone.
        assert_ne!(baseline, seeded_challenge(&layout, &point, value + EF::ONE));
    }

    #[test]
    fn honest_sparse_evaluation_reduces_to_the_dense_multilinear() {
        // Prover and verifier begin from identical transcript states.
        let (layout, heights, dense, point) = fixture();
        let value = jagged_evaluation(&heights, &dense, &point);
        let output = layout
            .prove(&dense, &point, value, &mut challenger())
            .unwrap();
        let (proof, prover_claim) = output.into_parts();
        let verifier_claim = layout
            .verify(&point, value, &proof, &mut challenger())
            .unwrap();

        // The surviving claim is exactly an evaluation of the vector the caller committed.
        assert_eq!(prover_claim, verifier_claim);
        assert_eq!(
            *verifier_claim.value(),
            committed_evaluation(&dense, verifier_claim.point())
        );
    }

    #[test]
    fn a_value_the_witness_does_not_take_is_refused_before_the_transcript() {
        // The caller owns the statement, so a witness that disagrees with it is a caller error.
        let (layout, heights, dense, point) = fixture();
        let value = jagged_evaluation(&heights, &dense, &point);

        assert_eq!(
            layout
                .prove(&dense, &point, value + EF::ONE, &mut challenger())
                .err(),
            Some(JaggedError::ClaimMismatch)
        );
    }

    #[test]
    fn malformed_prover_input_is_rejected_before_the_transcript() {
        // Reordering these checks after the seed would leave the statement encoding ambiguous.
        let (layout, _, dense, point) = fixture();
        let mut short = dense.clone();
        short.pop();
        assert_eq!(
            layout
                .prove(&short, &point, EF::ZERO, &mut challenger())
                .err(),
            Some(JaggedError::DenseLengthMismatch {
                expected: 16,
                actual: 15
            })
        );

        // One coordinate too many in the row point addresses rows the layout does not provision.
        let mut wide = point.row().as_slice().to_vec();
        wide.push(EF::ONE);
        let wide_row = JaggedPoint::new(Point::new(wide), point.column().clone());
        assert_eq!(
            layout
                .prove(&dense, &wide_row, EF::ZERO, &mut challenger())
                .err(),
            Some(JaggedError::RowPointWidthMismatch {
                expected: 3,
                actual: 4
            })
        );

        // One coordinate too few in the column point cannot address four columns.
        let narrow = JaggedPoint::new(
            point.row().clone(),
            Point::new(point.column().as_slice()[..1].to_vec()),
        );
        assert_eq!(
            layout
                .prove(&dense, &narrow, EF::ZERO, &mut challenger())
                .err(),
            Some(JaggedError::ColumnPointWidthMismatch {
                expected: 2,
                actual: 1
            })
        );

        // The verifier runs the same two point checks on its own public inputs.
        let proof = layout
            .prove(
                &dense,
                &point,
                jagged_evaluation(&[3, 0, 5, 1], &dense, &point),
                &mut challenger(),
            )
            .unwrap()
            .into_parts()
            .0;
        assert_eq!(
            layout
                .verify(&wide_row, EF::ZERO, &proof, &mut challenger())
                .err(),
            Some(JaggedError::RowPointWidthMismatch {
                expected: 3,
                actual: 4
            })
        );
        assert_eq!(
            layout
                .verify(&narrow, EF::ZERO, &proof, &mut challenger())
                .err(),
            Some(JaggedError::ColumnPointWidthMismatch {
                expected: 2,
                actual: 1
            })
        );
    }

    #[test]
    fn changing_any_public_statement_component_rejects() {
        // The proof is generated for one layout, point and claimed value.
        let (layout, heights, dense, point) = fixture();
        let value = jagged_evaluation(&heights, &dense, &point);
        let output = layout
            .prove(&dense, &point, value, &mut challenger())
            .unwrap();
        let (proof, _) = output.into_parts();

        // Mutation: move one live row from the first column to the second.
        let other_layout = JaggedLayout::new(3, &[2, 1, 5, 1]).unwrap();
        assert!(
            other_layout
                .verify(&point, value, &proof, &mut challenger())
                .is_err()
        );

        // Mutation: change one coordinate while preserving the point width.
        let mut other_row = point.row().as_slice().to_vec();
        other_row[0] += EF::ONE;
        let other_point = JaggedPoint::new(Point::new(other_row), point.column().clone());
        assert!(
            layout
                .verify(&other_point, value, &proof, &mut challenger())
                .is_err()
        );

        // Mutation: change only the claimed sparse evaluation.
        assert!(
            layout
                .verify(&point, value + EF::ONE, &proof, &mut challenger())
                .is_err()
        );
    }

    #[test]
    fn malformed_proof_shapes_and_terminal_values_reject() {
        // Start from an honest proof, then mutate one independently checked component at a time.
        let (layout, heights, dense, point) = fixture();
        let value = jagged_evaluation(&heights, &dense, &point);
        let output = layout
            .prove(&dense, &point, value, &mut challenger())
            .unwrap();
        let (proof, claim) = output.into_parts();

        // The terminal product is vacuous wherever the selector vanishes, and every mutation below would then reject for the wrong reason.
        assert_ne!(
            JaggedSelector::new(&layout).evaluate(&point, claim.point()),
            EF::ZERO
        );

        // One missing round cannot redefine the transcript shape.
        let mut short = proof.clone();
        short.sumcheck.polynomial_evaluations.pop();
        assert!(matches!(
            layout.verify(&point, value, &short, &mut challenger()),
            Err(JaggedError::Sumcheck(
                SumcheckError::RoundCountMismatch { .. }
            ))
        ));

        // A zero-difficulty reduction admits no grinding witnesses at all.
        let mut ground = proof.clone();
        ground.sumcheck.pow_witnesses.push(F::ONE);
        assert_eq!(
            layout.verify(&point, value, &ground, &mut challenger()),
            Err(JaggedError::Sumcheck(
                SumcheckError::PowWitnessCountMismatch {
                    expected: 0,
                    actual: 1
                }
            ))
        );

        // A forged dense evaluation breaks the terminal product relation.
        let mut forged = proof;
        forged.dense_evaluation += EF::ONE;
        assert_eq!(
            layout.verify(&point, value, &forged, &mut challenger()),
            Err(JaggedError::TerminalMismatch)
        );
    }

    #[test]
    fn the_smallest_folding_shape_still_samples_a_challenge() {
        // Invariant: at the minimum non-degenerate shape the reduction draws a challenge, and
        // the challenge moves with the statement.
        //
        // This is the test that fails if the reduction ever contracts to a bare identity at
        // its smallest legal shape. A reduction that samples nothing cannot separate two
        // instances, so every separation test above it would pass for the wrong reason.
        //
        // Fixture state: two live cells in two columns, so
        //
        //     area 2  ->  dense_variables 1  ->  exactly one sumcheck round
        let heights = vec![1, 1];
        let layout = JaggedLayout::new(1, &heights).unwrap();
        assert_eq!(layout.dense_variables(), 1);

        let dense = vec![F::from_u64(5), F::from_u64(9)];
        let point = JaggedPoint::new(
            extension_point(&[2, 3, 5, 7]),
            extension_point(&[11, 13, 17, 19]),
        );
        let value = jagged_evaluation(&heights, &dense, &point);

        let output = layout
            .prove(&dense, &point, value, &mut challenger())
            .unwrap();
        let (proof, _) = output.into_parts();
        let claim = layout
            .verify(&point, value, &proof, &mut challenger())
            .unwrap();

        // One round means one challenge. Zero would make the dense point empty.
        assert_eq!(claim.point().num_variables(), 1);

        // The surviving claim is an honest evaluation of the committed vector at that challenge.
        assert_eq!(*claim.value(), committed_evaluation(&dense, claim.point()));

        // Mutation: swap the two live cells between columns. The layout is unchanged, the
        // statement is not, and the drawn challenge must follow it.
        let other_dense = vec![F::from_u64(9), F::from_u64(5)];
        let other_value = jagged_evaluation(&heights, &other_dense, &point);
        let other_proof = layout
            .prove(&other_dense, &point, other_value, &mut challenger())
            .unwrap()
            .into_parts()
            .0;
        let other_claim = layout
            .verify(&point, other_value, &other_proof, &mut challenger())
            .unwrap();
        assert_ne!(claim.point(), other_claim.point());
    }

    #[test]
    fn empty_trace_proves_only_the_zero_sparse_polynomial() {
        // Zero live cells leave a one-cell envelope that is entirely padding, and no sumcheck rounds.
        let layout = JaggedLayout::new(2, &[0, 0]).unwrap();
        let point = JaggedPoint::new(
            extension_point(&[2, 3, 5, 7, 11, 13, 17, 19]),
            extension_point(&[23, 29, 31, 37]),
        );
        let output = layout
            .prove(&[F::ONE], &point, EF::ZERO, &mut challenger())
            .unwrap();
        let (proof, claim) = output.into_parts();

        // The sparse table is empty, yet the claim still names the nonzero cell that was committed.
        assert_eq!(*claim.value(), EF::ONE);

        // No witness can make the empty table take a nonzero value.
        assert_eq!(
            layout
                .prove(&[F::ONE], &point, EF::ONE, &mut challenger())
                .err(),
            Some(JaggedError::ClaimMismatch)
        );
        assert!(
            layout
                .verify(&point, EF::ZERO, &proof, &mut challenger())
                .is_ok()
        );
    }

    #[test]
    fn padding_is_free_but_the_claim_names_the_committed_vector() {
        // Two witnesses agreeing on every live cell state the same sparse claim.
        let (layout, heights, dense, point) = fixture();
        let mut other = dense.clone();
        other[layout.area()] += F::ONE;
        let value = jagged_evaluation(&heights, &dense, &point);
        assert_eq!(value, jagged_evaluation(&heights, &other, &point));

        // Both proofs are honest, and neither verifier learns which padding was used.
        let (proof, claim) = layout
            .prove(&dense, &point, value, &mut challenger())
            .unwrap()
            .into_parts();
        let (other_proof, other_claim) = layout
            .prove(&other, &point, value, &mut challenger())
            .unwrap()
            .into_parts();
        assert!(
            layout
                .verify(&point, value, &proof, &mut challenger())
                .is_ok()
        );
        assert!(
            layout
                .verify(&point, value, &other_proof, &mut challenger())
                .is_ok()
        );

        // Each delegated claim names its own committed vector, so a prover handed only the live cells would open against the wrong one.
        assert_eq!(*claim.value(), committed_evaluation(&dense, claim.point()));
        assert_eq!(
            *other_claim.value(),
            committed_evaluation(&other, other_claim.point())
        );
        assert_ne!(
            *other_claim.value(),
            committed_evaluation(&dense, other_claim.point())
        );
    }

    proptest! {
        #[test]
        fn the_reduction_proves_the_jagged_table_evaluation(
            heights in prop::collection::vec(0usize..=8, 4),
            row in prop::collection::vec(any::<u32>(), 12),
            column in prop::collection::vec(any::<u32>(), 8),
            cells in prop::collection::vec(any::<u32>(), 32),
        ) {
            // Heights up to the row bound sweep empty columns, a full column and non-power-of-two areas.
            let layout = JaggedLayout::new(3, &heights).unwrap();
            let point = JaggedPoint::new(extension_point(&row), extension_point(&column));
            let witness = cells[..layout.dense_capacity()]
                .iter()
                .map(|&cell| F::from_u32(cell))
                .collect::<Vec<_>>();

            // An addressing convention that drifted on both sides of the module still misses this.
            let value = jagged_evaluation(&heights, &witness, &point);
            let (proof, prover_claim) = layout
                .prove(&witness, &point, value, &mut challenger())
                .unwrap()
                .into_parts();
            let verifier_claim = layout
                .verify(&point, value, &proof, &mut challenger())
                .unwrap();

            prop_assert_eq!(&prover_claim, &verifier_claim);
            prop_assert_eq!(
                *verifier_claim.value(),
                committed_evaluation(&witness, verifier_claim.point())
            );
            prop_assert_eq!(
                layout.prove(&witness, &point, value + EF::ONE, &mut challenger()).err(),
                Some(JaggedError::ClaimMismatch)
            );
        }

        #[test]
        fn a_row_bound_far_above_the_dense_arity_round_trips(
            heights in prop::collection::vec(0usize..=3, 4),
            row in prop::collection::vec(any::<u32>(), 80),
            column in prop::collection::vec(any::<u32>(), 8),
            cells in prop::collection::vec(any::<u32>(), 16),
        ) {
            // Twenty row variables over twelve live cells is the capacity-free shape this exists for.
            let layout = JaggedLayout::new(20, &heights).unwrap();
            prop_assert!(layout.dense_variables() <= 4);
            let point = JaggedPoint::new(extension_point(&row), extension_point(&column));
            let witness = cells[..layout.dense_capacity()]
                .iter()
                .map(|&cell| F::from_u32(cell))
                .collect::<Vec<_>>();

            let value = jagged_evaluation(&heights, &witness, &point);
            let (proof, prover_claim) = layout
                .prove(&witness, &point, value, &mut challenger())
                .unwrap()
                .into_parts();
            let verifier_claim = layout
                .verify(&point, value, &proof, &mut challenger())
                .unwrap();

            prop_assert_eq!(&prover_claim, &verifier_claim);
            prop_assert_eq!(
                *verifier_claim.value(),
                committed_evaluation(&witness, verifier_claim.point())
            );
        }
    }
}
