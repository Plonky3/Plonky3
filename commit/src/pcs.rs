//! Traits for polynomial commitment schemes.

use alloc::vec::Vec;
use core::fmt::Debug;

use p3_field::ExtensionField;
use p3_matrix::dense::RowMajorMatrix;
use serde::de::DeserializeOwned;
use serde::Serialize;

use crate::PolynomialDomain;

pub type Val<D> = <D as PolynomialDomain>::Val;

/// A (not necessarily hiding) polynomial commitment scheme, for committing to (batches of) polynomials
// TODO: Should we have a super-trait for weakly-binding PCSs, like FRI outside unique decoding radius?
pub trait Pcs<Challenge, Challenger>
where
    Challenge: ExtensionField<Val<Self::Domain>>,
{
    type Domain: PolynomialDomain;

    /// The commitment that's sent to the verifier.
    type Commitment: Clone + Serialize + DeserializeOwned;

    /// Data that the prover stores for committed polynomials, to help the prover with opening.
    type ProverData;

    /// The opening argument.
    type Proof: Serialize + DeserializeOwned;

    type Error: Debug;

    fn natural_domain_for_degree(&self, degree: usize) -> Self::Domain;

    #[allow(clippy::type_complexity)]
    fn commit(
        &self,
        evaluations: Vec<(Self::Domain, RowMajorMatrix<Val<Self::Domain>>)>,
    ) -> (Self::Commitment, Self::ProverData);

    fn get_evaluations_on_domain(
        &self,
        prover_data: &Self::ProverData,
        idx: usize,
        domain: Self::Domain,
    ) -> RowMajorMatrix<Val<Self::Domain>>;

    fn open(
        &self,
        // For each round,
        rounds: Vec<(
            &Self::ProverData,
            // for each matrix,
            Vec<
                // points to open
                Vec<Challenge>,
            >,
        )>,
        challenger: &mut Challenger,
    ) -> (OpenedValues<Challenge>, Self::Proof);

    #[allow(clippy::type_complexity)]
    fn verify(
        &self,
        // For each round:
        rounds: Vec<(
            Self::Commitment,
            // for each matrix:
            Vec<(
                // its domain,
                Self::Domain,
                // for each point:
                Vec<(
                    // the point,
                    Challenge,
                    // values at the point
                    Vec<Challenge>,
                )>,
            )>,
        )>,
        proof: &Self::Proof,
        challenger: &mut Challenger,
    ) -> Result<(), Self::Error>;
}

pub type OpenedValues<F> = Vec<OpenedValuesForRound<F>>;
pub type OpenedValuesForRound<F> = Vec<OpenedValuesForMatrix<F>>;
pub type OpenedValuesForMatrix<F> = Vec<OpenedValuesForPoint<F>>;
pub type OpenedValuesForPoint<F> = Vec<F>;
