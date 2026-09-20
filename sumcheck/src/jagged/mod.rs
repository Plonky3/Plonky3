//! Capacity-free commitments to columns of unequal height.
//!
//! This module implements the basic sparse-to-dense reduction from ePrint 2025/917.
//!
//! A downstream multilinear PCS authenticates only the live cells concatenated column by column.
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
    #[must_use]
    pub const fn row(&self) -> &Point<F> {
        // Return the borrowed point without re-encoding its coordinates.
        &self.row
    }

    /// Returns the column coordinates.
    #[must_use]
    pub const fn column(&self) -> &Point<F> {
        // Return the borrowed point without re-encoding its coordinates.
        &self.column
    }
}

/// A claim on the dense multilinear authenticated by the underlying PCS.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JaggedDenseClaim<F> {
    /// Point produced by the quadratic sumcheck challenges.
    point: Point<F>,
    /// Dense multilinear evaluation at the produced point.
    value: F,
}

impl<F> JaggedDenseClaim<F> {
    /// Returns the dense evaluation point.
    #[must_use]
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
#[derive(Clone, Debug)]
pub struct JaggedProverOutput<F, EF> {
    /// Proof consumed by the sparse verifier.
    proof: JaggedProof<F, EF>,
    /// Evaluation of the virtual sparse polynomial.
    sparse_value: EF,
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

    /// Returns the claimed sparse evaluation.
    #[must_use]
    pub const fn sparse_value(&self) -> &EF {
        // This is the evaluation asserted by the sparse statement.
        &self.sparse_value
    }

    /// Returns the dense claim to authenticate through the underlying PCS.
    #[must_use]
    pub const fn dense_claim(&self) -> &JaggedDenseClaim<EF> {
        // The claim is the only value delegated to the commitment scheme.
        &self.dense_claim
    }

    /// Splits the output into its proof, sparse value and dense claim.
    #[must_use]
    pub fn into_parts(self) -> (JaggedProof<F, EF>, EF, JaggedDenseClaim<EF>) {
        // Ownership passes through without copying any proof or point data.
        (self.proof, self.sparse_value, self.dense_claim)
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use p3_challenger::FieldChallenger;
    use p3_field::PrimeCharacteristicRing;
    use p3_multilinear_util::poly::Poly;

    use super::transcript::JaggedProverTranscript;
    use super::*;
    use crate::SumcheckError;
    use crate::tests::{EF, F, MyChallenger, challenger};

    fn fixture() -> (JaggedLayout, Vec<F>, JaggedPoint<EF>) {
        // Four unequal columns concatenate to nine live dense cells.
        let layout = JaggedLayout::new(3, &[3, 0, 5, 1]).unwrap();
        let dense = (1..=layout.area())
            .map(|value| F::from_u64(value as u64))
            .collect();
        let point = JaggedPoint::new(
            Point::new(vec![EF::from_u64(2), EF::from_u64(3), EF::from_u64(5)]),
            Point::new(vec![EF::from_u64(7), EF::from_u64(11)]),
        );
        (layout, dense, point)
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
    fn the_column_geometry_separates_two_transcript_seeds() {
        // Moving one live row between columns keeps the point widths and the round count.
        // Only the seed can tell the two statements apart, so the challenge must move with it.
        let (layout, _, point) = fixture();
        let moved = JaggedLayout::new(3, &[2, 1, 5, 1]).unwrap();
        let value = EF::from_u64(42);
        assert_eq!(moved.dense_variables(), layout.dense_variables());

        assert_eq!(
            seeded_challenge(&layout, &point, value),
            seeded_challenge(&layout, &point, value)
        );
        assert_ne!(
            seeded_challenge(&layout, &point, value),
            seeded_challenge(&moved, &point, value)
        );
    }

    #[test]
    fn honest_sparse_evaluation_reduces_to_the_dense_multilinear() {
        // Prover and verifier begin from identical transcript states.
        let (layout, dense, point) = fixture();
        let mut prover_challenger = challenger();
        let output = layout
            .prove(&dense, &point, &mut prover_challenger)
            .unwrap();
        let (proof, sparse_value, prover_claim) = output.into_parts();
        let mut verifier_challenger = challenger();
        let verifier_claim = layout
            .verify(&point, sparse_value, &proof, &mut verifier_challenger)
            .unwrap();

        // The surviving claim is exactly an evaluation of the zero-padded dense witness.
        let mut padded = dense.into_iter().map(EF::from).collect::<Vec<_>>();
        padded.resize(layout.dense_capacity(), EF::ZERO);
        let expected = Poly::new(padded).eval_base(verifier_claim.point());

        assert_eq!(prover_claim, verifier_claim);
        assert_eq!(*verifier_claim.value(), expected);
    }

    #[test]
    fn changing_any_public_statement_component_rejects() {
        // The proof is generated for one layout, point and claimed value.
        let (layout, dense, point) = fixture();
        let mut prover_challenger = challenger();
        let output = layout
            .prove(&dense, &point, &mut prover_challenger)
            .unwrap();
        let (proof, sparse_value, _) = output.into_parts();

        // Mutation: move one live row from the first column to the second.
        let other_layout = JaggedLayout::new(3, &[2, 1, 5, 1]).unwrap();
        let mut verifier_challenger = challenger();
        assert!(
            other_layout
                .verify(&point, sparse_value, &proof, &mut verifier_challenger,)
                .is_err()
        );

        // Mutation: change one coordinate while preserving the point width.
        let mut other_row = point.row().as_slice().to_vec();
        other_row[0] += EF::ONE;
        let other_point = JaggedPoint::new(Point::new(other_row), point.column().clone());
        let mut verifier_challenger = challenger();
        assert!(
            layout
                .verify(&other_point, sparse_value, &proof, &mut verifier_challenger,)
                .is_err()
        );

        // Mutation: change only the claimed sparse evaluation.
        let mut verifier_challenger = challenger();
        assert!(
            layout
                .verify(
                    &point,
                    sparse_value + EF::ONE,
                    &proof,
                    &mut verifier_challenger,
                )
                .is_err()
        );
    }

    #[test]
    fn malformed_proof_shapes_and_terminal_values_reject() {
        // Start from an honest proof, then mutate one independently checked component at a time.
        let (layout, dense, point) = fixture();
        let mut prover_challenger = challenger();
        let output = layout
            .prove(&dense, &point, &mut prover_challenger)
            .unwrap();
        let (proof, sparse_value, _) = output.into_parts();

        // One missing round cannot redefine the transcript shape.
        let mut short = proof.clone();
        short.sumcheck.polynomial_evaluations.pop();
        let mut verifier_challenger = challenger();
        assert!(matches!(
            layout.verify(&point, sparse_value, &short, &mut verifier_challenger,),
            Err(JaggedError::Sumcheck(
                SumcheckError::RoundCountMismatch { .. }
            ))
        ));

        // A forged dense evaluation breaks the terminal product relation.
        let mut forged = proof;
        forged.dense_evaluation += EF::ONE;
        let mut verifier_challenger = challenger();
        assert_eq!(
            layout.verify(&point, sparse_value, &forged, &mut verifier_challenger,),
            Err(JaggedError::TerminalMismatch)
        );
    }

    #[test]
    fn empty_trace_proves_only_the_zero_sparse_polynomial() {
        // Zero live cells produce a constant dense zero polynomial and no sumcheck rounds.
        let layout = JaggedLayout::new(2, &[0, 0]).unwrap();
        let point = JaggedPoint::new(
            Point::new(vec![EF::from_u64(2), EF::from_u64(3)]),
            Point::new(vec![EF::from_u64(5)]),
        );
        let mut prover_challenger = challenger();
        let output = layout.prove(&[], &point, &mut prover_challenger).unwrap();
        let (proof, sparse_value, _) = output.into_parts();
        assert_eq!(sparse_value, EF::ZERO);

        let mut verifier_challenger = challenger();
        assert!(
            layout
                .verify(&point, EF::ZERO, &proof, &mut verifier_challenger)
                .is_ok()
        );
    }
}
