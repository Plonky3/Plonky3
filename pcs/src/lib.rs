#![no_std]

extern crate alloc;

use hyperfield::field::{Field, FieldExtension};
use hyperfield::matrix::dense::DenseMatrixView;

use alloc::vec::Vec;

// TODO: Multivariate also
// TODO: Batch verify

pub trait PCS<F: Field>: 'static {
    /// The commitment that's sent to the verifier.
    type Commitment;
    /// A hint which helps the prover with opening.
    type Hint;
    /// The opening argument.
    type Proof;

    fn commit_batches(polynomials: Vec<DenseMatrixView<F>>) -> (Self::Commitment, Self::Hint);

    fn open_batches<FE: FieldExtension<Base = F>>(
        points: &[FE],
        hints: &[Self::Hint],
    ) -> (Vec<FE>, Self::Proof);

    fn verify_batches<FE: FieldExtension<Base = F>>(
        commit: &Self::Commitment,
        points: &[FE],
        values: &Vec<Vec<Vec<FE>>>,
        proof: &Self::Proof,
    );
}

pub trait MultivariatePCS<F: Field>: 'static {
    // TODO
}

// struct UnivariateFromMultivariate<M: MultivariatePCS<>> {}
//
// impl UnivariatePCS for UnivariateFromMultivariate {}
