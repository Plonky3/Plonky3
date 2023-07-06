//! Traits for polynomial commitment schemes.

use alloc::vec;
use p3_field::{ExtensionField, Field};

use alloc::vec::Vec;
use p3_challenger::Challenger;
use p3_matrix::MatrixRows;

/// A (not necessarily hiding) polynomial commitment scheme, for committing to (batches of)
/// polynomials defined over the field `F`.
///
/// This high-level trait is agnostic with respect to the structure of a point; see `UnivariatePCS`
/// and `MultivariatePCS` for more specific subtraits.
// TODO: Should we have a super-trait for weakly-binding PCSs, like FRI outside unique decoding radius?
pub trait PCS<F: Field, In: for<'a> MatrixRows<'a, F>> {
    /// The commitment that's sent to the verifier.
    type Commitment;

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

pub trait UnivariatePCS<F: Field, In: for<'a> MatrixRows<'a, F>>: PCS<F, In> {
    fn open_multi_batches<EF, Chal>(
        &self,
        prover_data: &[&Self::ProverData],
        points: &[EF],
        challenger: &mut Chal,
    ) -> (Vec<Vec<Vec<EF>>>, Self::Proof)
    where
        EF: ExtensionField<F>,
        Chal: Challenger<F>;

    fn verify_multi_batches<EF, Chal>(
        &self,
        commits: &[Self::Commitment],
        points: &[EF],
        values: &[Vec<Vec<EF>>],
        proof: &Self::Proof,
    ) -> Result<(), Self::Error>
    where
        EF: ExtensionField<F>,
        Chal: Challenger<F>;
}

pub trait MultivariatePCS<F: Field, In: for<'a> MatrixRows<'a, F>>: PCS<F, In> {
    fn open_multi_batches<EF, Chal>(
        &self,
        prover_data: &[&Self::ProverData],
        points: &[Vec<EF>],
        challenger: &mut Chal,
    ) -> (Vec<Vec<Vec<EF>>>, Self::Proof)
    where
        EF: ExtensionField<F>,
        Chal: Challenger<F>;

    fn verify_multi_batches<EF, Chal>(
        &self,
        commits: &[Self::Commitment],
        points: &[Vec<EF>],
        values: &[Vec<Vec<EF>>],
        proof: &Self::Proof,
    ) -> Result<(), Self::Error>
    where
        EF: ExtensionField<F>,
        Chal: Challenger<F>;
}
