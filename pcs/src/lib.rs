#![no_std]

extern crate alloc;

pub mod multi_from_uni;
pub mod uni_from_multi;

use p3_field::field::{Field, FieldExtension};
use p3_field::matrix::dense::DenseMatrix;

use alloc::vec::Vec;

/// A polynomial commitment scheme, for committing to (batches of) polynomials defined over the
/// field `F`.
pub trait PCS<F: Field>: 'static {
    /// The commitment that's sent to the verifier.
    type Commitment;

    /// Data that the prover stores for committed polynomials, to help the prover with opening.
    type ProverData: ProverData<F>;

    /// The opening argument.
    type Proof;

    type Error;

    fn commit_batches(polynomials: Vec<DenseMatrix<F>>) -> (Self::Commitment, Self::ProverData);
}

pub trait UnivariatePCS<F: Field>: PCS<F> {
    // type UnivariateProverData: UnivariateProverData;

    fn open_batches<FE: FieldExtension<Base = F>>(
        points: &[FE],
        prover_data: &[Self::ProverData],
    ) -> (Vec<Vec<Vec<FE>>>, Self::Proof);

    fn verify_batches<FE: FieldExtension<Base = F>>(
        commit: &Self::Commitment,
        points: &[FE],
        values: &[Vec<Vec<FE>>],
        proof: &Self::Proof,
    ) -> Result<(), Self::Error>;
}

pub trait MultivariatePCS<F: Field>: PCS<F> {
    // type MultivariateProverData: MultivariateProverData;

    fn open_batches<FE: FieldExtension<Base = F>>(
        points: &[FE],
        prover_data: &[Self::ProverData],
    ) -> (Vec<Vec<Vec<FE>>>, Self::Proof);

    fn verify_batches<FE: FieldExtension<Base = F>>(
        commit: &Self::Commitment,
        points: &[FE],
        values: &[Vec<Vec<FE>>],
        proof: &Self::Proof,
    );
}

/// Data that the prover stores for committed polynomials, to help the prover with opening.
pub trait ProverData<F: Field> {
    fn get_original_value(&self, batch: usize, poly: usize, value: usize) -> F;
}

// pub trait UnivariateProverData<F: Field>: ProverData<F> {}
//
// pub trait MultivariateProverData<F: Field>: ProverData<F> {}
