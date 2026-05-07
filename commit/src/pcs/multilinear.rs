//! Polynomial commitment scheme trait for multilinear polynomials.

use alloc::vec::Vec;
use core::fmt::Debug;

use p3_field::{ExtensionField, Field};
use p3_matrix::dense::RowMajorMatrix;
use p3_multilinear_util::point::Point;
use serde::Serialize;
use serde::de::DeserializeOwned;

/// Claimed evaluation values from an opening.
///
/// `values[i][j]` = evaluation of polynomial i at its j-th opening point.
pub type MultilinearOpenedValues<F> = Vec<Vec<F>>;

/// Polynomial commitment scheme for multilinear polynomials over the Boolean hypercube.
///
/// A multilinear polynomial in m variables is defined by its 2^m evaluations
/// on {0,1}^m. This trait abstracts the three phases of a PCS:
///
/// - **Commit**: bind to one or more polynomials and register the evaluation
///   points where they will be opened.
/// - **Open**: produce a proof that the committed polynomials evaluate to the
///   claimed values at the registered points.
/// - **Verify**: check the proof against the commitment and claimed values.
pub trait MultilinearPcs<Challenge, Challenger>
where
    Challenge: ExtensionField<Self::Val>,
{
    /// Base field of the committed polynomials.
    type Val: Field;

    /// Succinct binding commitment sent to the verifier.
    type Commitment: Clone + Serialize + DeserializeOwned;

    /// Prover-side auxiliary data retained between commit and open.
    /// Never sent to the verifier.
    type ProverData;

    /// Opening proof checked by the verifier.
    type Proof: Clone + Serialize + DeserializeOwned;

    /// Verification failure type.
    type Error: Debug;

    /// Number of variables m of the committed polynomials.
    /// Every polynomial has 2^m evaluations.
    fn num_vars(&self) -> usize;

    /// Commit to a batch of multilinear polynomials and register opening points.
    ///
    /// Each column of the evaluation matrix holds one polynomial's 2^m values
    /// on the Boolean hypercube in lexicographic order.
    ///
    /// `opening_points[i]` lists the points at which the i-th polynomial
    /// will later be opened.
    ///
    /// # Returns
    ///
    /// - A succinct commitment (e.g. a Merkle root).
    /// - Opaque prover data consumed by `open`.
    fn commit(
        &self,
        evaluations: RowMajorMatrix<Self::Val>,
        opening_points: &[Vec<Point<Challenge>>],
        challenger: &mut Challenger,
    ) -> (Self::Commitment, Self::ProverData);

    /// Produce an opening proof for the points registered during commit.
    ///
    /// Consumes the prover data.
    ///
    /// # Returns
    ///
    /// - Claimed evaluation values: `values[i][j]` is the evaluation of the
    ///   i-th polynomial at its j-th registered point.
    /// - The opening proof.
    fn open(
        &self,
        prover_data: Self::ProverData,
        challenger: &mut Challenger,
    ) -> (MultilinearOpenedValues<Challenge>, Self::Proof);

    /// Verify an opening proof against a commitment and claimed evaluations.
    ///
    /// `opening_claims[i]` contains (point, claimed_value) pairs for the
    /// i-th committed polynomial.
    ///
    /// The challenger must be in the same transcript state as the prover's
    /// challenger was at the corresponding protocol step.
    fn verify(
        &self,
        commitment: &Self::Commitment,
        opening_claims: &[Vec<(Point<Challenge>, Challenge)>],
        proof: &Self::Proof,
        challenger: &mut Challenger,
    ) -> Result<(), Self::Error>;
}
