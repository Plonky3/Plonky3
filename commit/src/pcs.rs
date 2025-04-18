//! Traits for polynomial commitment schemes.

use alloc::vec::Vec;
use core::fmt::Debug;

use p3_field::ExtensionField;
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::PolynomialSpace;

pub type Val<D> = <D as PolynomialSpace>::Val;

/// A polynomial commitment scheme, for committing to batches of polynomials defined by their evaluations
/// over some domain.
///
/// In general this does not have to be a hiding commitment scheme but it might be for some implementations.
// TODO: Should we have a super-trait for weakly-binding PCSs, like FRI outside unique decoding radius?
pub trait Pcs<Challenge, Challenger>
where
    Challenge: ExtensionField<Val<Self::Domain>>,
{
    /// The types of domain that the polynomials can be evaluated on.
    type Domain: PolynomialSpace;

    /// The commitment that's sent to the verifier.
    type Commitment: Clone + Serialize + DeserializeOwned;

    /// Data that the prover stores for committed polynomials, to help the prover with opening.
    type ProverData;

    /// Type of the output of `get_evaluations_on_domain`.
    type EvaluationsOnDomain<'a>: Matrix<Val<Self::Domain>> + 'a;

    /// The opening argument.
    type Proof: Clone + Serialize + DeserializeOwned;

    /// Possible verification errors.
    type Error: Debug;

    /// This should return a coset domain (s.t. Domain::next_point returns Some)
    fn natural_domain_for_degree(&self, degree: usize) -> Self::Domain;

    /// Given a collection of evaluation matrices, produce a binding commitment to
    /// the polynomials defined by those evaluations.
    ///
    /// Returns both the commitment which should be sent to the verifier
    /// and the prover data which can be used to produce opening proofs.
    #[allow(clippy::type_complexity)]
    fn commit(
        &self,
        evaluations: Vec<(Self::Domain, RowMajorMatrix<Val<Self::Domain>>)>,
    ) -> (Self::Commitment, Self::ProverData);

    /// Given prover data corresponding to a commitment of a collection of evaluation matrices,
    /// return the evaluations of those matrices on the given domain.
    ///
    /// Ideally this should only be called when the domain is a subset of the evaluation domain
    /// on which the evaluation matrices are defined. In these cases, this should be essentially
    /// a no-op.
    fn get_evaluations_on_domain<'a>(
        &self,
        prover_data: &'a Self::ProverData,
        idx: usize,
        domain: Self::Domain,
    ) -> Self::EvaluationsOnDomain<'a>;

    /// Open a collection of polynomial commitments at a set of points. Produce the values at those points along with a proof
    /// of correctness.
    ///
    /// Arguments:
    /// - `commitments_with_opening_points`: A vector whose elements are a pair:
    ///     - `data`: The prover data corresponding to a multi-matrix commitment.
    ///     - `opening_points`: A vector containing, for each matrix committed to, a vector of opening points.
    /// - `fiat_shamir_challenger`: The challenger that will be used to generate the proof.
    ///
    /// Unwrapping the arguments further, each `data` contains a vector of the committed matrices `matrices = Vec<M>`.
    /// If the length of `matrices` is not equal to the length of `opening_points` the function will error. Otherwise, for
    /// each index `i`, the matrix `M = matrices[i]` will be opened at the points `M_points = opening_points[i]`.
    ///
    /// This means that each column of `M` will be interpreted as the evaluation vector of some polynomial
    /// and we will compute the value of all of those polynomials at `M_points`.
    ///
    /// The domains on which the evaluation vectors are defined is not part of the arguments here
    /// but should be essentially public information known to both the prover and verifier.
    fn open(
        &self,
        // For each multi-matrix commitment,
        commitments_with_opening_points: Vec<(
            // The matrices and auxiliary prover data
            &Self::ProverData,
            // for each matrix,
            Vec<
                // the points to open
                Vec<Challenge>,
            >,
        )>,
        fiat_shamir_challenger: &mut Challenger,
    ) -> (OpenedValues<Challenge>, Self::Proof);

    /// Verify that a collection of opened values is correct.
    ///
    /// Arguments:
    /// - `commitments_with_opening_points`: A vector whose elements are a pair:
    ///     - `commitment`: A multi matrix commitment.
    ///     - `opening_points`: A vector containing, for each matrix committed to, a vector of opening points and claimed evaluations.
    /// - `proof`: A claimed proof of correctness for the opened values.
    /// - `fiat_shamir_challenger`: The challenger that will be used to generate the proof.
    #[allow(clippy::type_complexity)]
    fn verify(
        &self,
        // For each commitment:
        commitments_with_opening_points: Vec<(
            // The commitment
            Self::Commitment,
            // for each claimed matrix in the commitment:
            Vec<(
                // its domain,
                Self::Domain,
                // A vector of (point, claimed_evaluation) pairs
                Vec<(
                    // the point the matrix was opened at,
                    Challenge,
                    // the claimed evaluations at that point
                    Vec<Challenge>,
                )>,
            )>,
        )>,
        // The opening proof for all claimed evaluations.
        proof: &Self::Proof,
        fiat_shamir_challenger: &mut Challenger,
    ) -> Result<(), Self::Error>;
}

pub type OpenedValues<F> = Vec<OpenedValuesForRound<F>>;
pub type OpenedValuesForRound<F> = Vec<OpenedValuesForMatrix<F>>;
pub type OpenedValuesForMatrix<F> = Vec<OpenedValuesForPoint<F>>;
pub type OpenedValuesForPoint<F> = Vec<F>;
