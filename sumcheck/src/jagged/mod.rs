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
use p3_challenger::fs::TranscriptField;
use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::ExtensionField;
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;
use serde::{Deserialize, Serialize};

use self::selector::{selector_evaluation, selector_table, validate_point};
use self::transcript::{JaggedProverTranscript, JaggedVerifierTranscript};
use crate::product_polynomial::ProductPolynomial;
use crate::strategy::{Basis, SumcheckProver, VariableOrder};
use crate::{SumcheckData, SumcheckError};

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

impl JaggedLayout {
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
        validate_point(self, point)?;
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
        let selector = selector_table(self, point);
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
        validate_point(self, point)?;

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
        let selector = selector_evaluation(self, point, &dense_point);
        if terminal != proof.dense_evaluation * selector {
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
    use alloc::vec::Vec;

    use p3_field::PrimeCharacteristicRing;
    use p3_multilinear_util::poly::Poly;

    use super::*;
    use crate::tests::{EF, F, challenger};

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
