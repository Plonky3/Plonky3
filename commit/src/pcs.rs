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

/// A (not necessarily hiding) polynomial commitment scheme, for committing to (batches of) polynomials
// TODO: Should we have a super-trait for weakly-binding PCSs, like FRI outside unique decoding radius?
pub trait Pcs<Challenge, Challenger>
where
    Challenge: ExtensionField<Val<Self::Domain>>,
{
    type Domain: PolynomialSpace;

    /// The commitment that's sent to the verifier.
    type Commitment: Clone + Serialize + DeserializeOwned;

    /// Data that the prover stores for committed polynomials, to help the prover with opening.
    type ProverData;

    /// Type of the output of `get_evaluations_on_domain`.
    type EvaluationsOnDomain<'a>: Matrix<Val<Self::Domain>> + 'a;

    /// The opening argument.
    type Proof: Clone + Serialize + DeserializeOwned;

    type Error: Debug;

    /// Set to true to activate randomization and achieve zero-knowledge.
    const ZK: bool;

    /// This should return a coset domain (s.t. Domain::next_point returns Some)
    fn natural_domain_for_degree(&self, degree: usize) -> Self::Domain;

    /// Commit to the batch of `evaluations`, without any randomization: used when `zk` is disabled.
    #[allow(clippy::type_complexity)]
    fn commit(
        &self,
        evaluations: Vec<(Self::Domain, RowMajorMatrix<Val<Self::Domain>>)>,
    ) -> (Self::Commitment, Self::ProverData);

    /// Commit to the batch of `evaluations`. If `zk` is enabled and `is_random_poly` is false, the evaluations are
    /// first randomized as explained in Section 3 of https://github.com/IrreducibleOSS/binius.git .
    ///
    /// *** Arguments
    /// - `evaluations` are the evaluations of the polynomials we need to commit to.
    /// - `use_randomization` is set to `true` when we are randomizing the evaluations before committing to them.
    #[allow(clippy::type_complexity)]
    fn commit_zk(
        &self,
        evaluations: Vec<(Self::Domain, RowMajorMatrix<Val<Self::Domain>>)>,
        _use_randomization: bool,
    ) -> (Self::Commitment, Self::ProverData) {
        self.commit(evaluations)
    }

    /// Commit to the quotient polynomials. If `zk` is not enabled, this is the same as `commit`.
    /// If `zk` is enabled, the quotient polynomials are randomized as explained in Section 4.2 of
    /// https://github.com/IrreducibleOSS/binius.git .
    ///
    /// *** Arguments
    /// - `evaluations` are the evaluations of the quotient polynomial chunks we need to commit to.
    /// - `cis[i]` is the normailizing constant for the Lagrange selector over coset `i`.
    #[allow(clippy::type_complexity)]
    fn commit_quotient(
        &self,
        evaluations: Vec<(Self::Domain, RowMajorMatrix<Val<Self::Domain>>)>,
        _cis: Vec<Val<Self::Domain>>,
    ) -> (Self::Commitment, Self::ProverData) {
        self.commit(evaluations)
    }

    fn generate_random_vals(&self, _random_len: usize) -> RowMajorMatrix<Val<Self::Domain>> {
        RowMajorMatrix::default(0, 0)
    }

    fn get_evaluations_on_domain<'a>(
        &self,
        prover_data: &'a Self::ProverData,
        idx: usize,
        domain: Self::Domain,
    ) -> Self::EvaluationsOnDomain<'a>;

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
