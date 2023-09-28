//! Traits for polynomial commitment schemes.

use alloc::vec;
use alloc::vec::Vec;

use p3_challenger::FieldChallenger;
use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_matrix::dense::RowMajorMatrixView;
use p3_matrix::MatrixRows;

/// A (not necessarily hiding) polynomial commitment scheme, for committing to (batches of)
/// polynomials defined over the field `F`.
///
/// This high-level trait is agnostic with respect to the structure of a point; see `UnivariatePCS`
/// and `MultivariatePcs` for more specific subtraits.
// TODO: Should we have a super-trait for weakly-binding PCSs, like FRI outside unique decoding radius?
pub trait Pcs<Val: Field, In: MatrixRows<Val>> {
    /// The commitment that's sent to the verifier.
    type Commitment: Clone;

    /// Data that the prover stores for committed polynomials, to help the prover with opening.
    type ProverData;

    /// The opening argument.
    type Proof;

    type Error;

    fn commit_batches(&self, polynomials: Vec<In>) -> (Self::Commitment, Self::ProverData);

    fn commit_batch(&self, polynomials: In) -> (Self::Commitment, Self::ProverData) {
        self.commit_batches(vec![polynomials])
    }
}

pub type OpenedValues<F> = Vec<OpenedValuesForRound<F>>;
pub type OpenedValuesForRound<F> = Vec<OpenedValuesForPoint<F>>;
pub type OpenedValuesForPoint<F> = Vec<OpenedValuesForMatrix<F>>;
pub type OpenedValuesForMatrix<F> = Vec<F>;

pub trait UnivariatePcs<Val, Domain, EF, In, Challenger>: Pcs<Val, In>
where
    Val: Field,
    Domain: ExtensionField<Val> + TwoAdicField,
    EF: ExtensionField<Domain>,
    In: MatrixRows<Val>,
    Challenger: FieldChallenger<Val>,
{
    fn open_multi_batches(
        &self,
        prover_data_and_points: &[(&Self::ProverData, &[EF])],
        challenger: &mut Challenger,
    ) -> (OpenedValues<EF>, Self::Proof);

    fn verify_multi_batches(
        &self,
        commits_and_points: &[(Self::Commitment, &[EF])],
        values: OpenedValues<EF>,
        proof: &Self::Proof,
        challenger: &mut Challenger,
    ) -> Result<(), Self::Error>;
}

/// A `UnivariatePcs` where the commitment process involves computing a low-degree extension (LDE)
/// of each polynomial. These LDEs can be reused in other prover work.
pub trait UnivariatePcsWithLde<Val, Domain, EF, In, Challenger>:
    UnivariatePcs<Val, Domain, EF, In, Challenger>
where
    Val: Field,
    Domain: ExtensionField<Val> + TwoAdicField,
    EF: ExtensionField<Domain>,
    In: MatrixRows<Val>,
    Challenger: FieldChallenger<Val>,
{
    fn coset_shift(&self) -> Domain;

    fn get_ldes<'a, 'b>(
        &'a self,
        _prover_data: &'b Self::ProverData,
    ) -> Vec<RowMajorMatrixView<'b, Domain>>
    where
        'a: 'b;
}

pub trait MultivariatePcs<Val, Domain, EF, In, Challenger>: Pcs<Val, In>
where
    Val: Field,
    Domain: ExtensionField<Val>,
    EF: ExtensionField<Domain>,
    In: MatrixRows<Val>,
    Challenger: FieldChallenger<Val>,
{
    fn open_multi_batches(
        &self,
        prover_data_and_points: &[(&Self::ProverData, &[Vec<EF>])],
        challenger: &mut Challenger,
    ) -> (OpenedValues<EF>, Self::Proof);

    fn verify_multi_batches(
        &self,
        commits_and_points: &[(Self::Commitment, &[Vec<EF>])],
        values: OpenedValues<EF>,
        proof: &Self::Proof,
    ) -> Result<(), Self::Error>;
}
