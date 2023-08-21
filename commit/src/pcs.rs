//! Traits for polynomial commitment schemes.

use alloc::vec;
use alloc::vec::Vec;

use p3_challenger::FieldChallenger;
use p3_field::{ExtensionField, Field, TwoAdicField};
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
pub type OpenedValuesForRound<F> = Vec<OpenedValuesForMatrix<F>>;
pub type OpenedValuesForMatrix<F> = Vec<OpenedValuesForPoint<F>>;
pub type OpenedValuesForPoint<F> = Vec<F>;

pub trait UnivariatePcs<Val, Domain, In, Challenger>: Pcs<Val, In>
where
    Val: Field,
    Domain: ExtensionField<Val> + TwoAdicField,
    In: MatrixRows<Val>,
    Challenger: FieldChallenger<Val>,
{
    fn open_multi_batches<EF>(
        &self,
        prover_data_and_points: &[(&Self::ProverData, &[EF])],
        challenger: &mut Challenger,
    ) -> (OpenedValues<EF>, Self::Proof)
    where
        EF: ExtensionField<Domain> + TwoAdicField;

    fn verify_multi_batches<EF>(
        &self,
        commits_and_points: &[(Self::Commitment, &[EF])],
        values: OpenedValues<EF>,
        proof: &Self::Proof,
    ) -> Result<(), Self::Error>
    where
        EF: ExtensionField<Domain> + TwoAdicField;
}

pub trait MultivariatePcs<Val, In, Challenger>: Pcs<Val, In>
where
    Val: Field,
    In: MatrixRows<Val>,
    Challenger: FieldChallenger<Val>,
{
    fn open_multi_batches<EF>(
        &self,
        prover_data_and_points: &[(&Self::ProverData, &[Vec<EF>])],
        challenger: &mut Challenger,
    ) -> (OpenedValues<EF>, Self::Proof)
    where
        EF: ExtensionField<Val>,
        Challenger: FieldChallenger<Val>;

    fn verify_multi_batches<EF>(
        &self,
        commits_and_points: &[(Self::Commitment, &[Vec<EF>])],
        values: OpenedValues<EF>,
        proof: &Self::Proof,
    ) -> Result<(), Self::Error>
    where
        EF: ExtensionField<Val>,
        Challenger: FieldChallenger<Val>;
}
