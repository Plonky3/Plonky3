use alloc::borrow::Cow;
use alloc::vec::Vec;
use core::fmt::Debug;

use p3_field::{ExtensionField, Field};
use p3_matrix::Dimensions;
use p3_verifier_arith::{Lift, Native};

use crate::Mmcs;

/// A commitment scheme whose openings are checked within an arithmetic.
///
/// Rows hold values of type `T`, the arithmetic's view of the committed elements.
pub trait MmcsArith<A: Lift, T: Clone> {
    /// Native commitment.
    type Commitment;

    /// Native opening proof of several rows.
    type MultiProof;

    /// A commitment within the arithmetic.
    type Target: Clone;

    /// Why an opening failed.
    type Error: Debug;

    /// Lift a commitment the prover sent.
    fn load_commitment(&self, arith: &mut A, commitment: &Self::Commitment) -> Self::Target;

    /// Lift a commitment the verifier holds, such as its preprocessed columns.
    fn load_key_commitment(&self, arith: &mut A, commitment: &Self::Commitment) -> Self::Target;

    /// Announce a commitment the native transcript absorbs next.
    fn will_observe(&self, arith: &mut A, commitment: &Self::Target);

    /// Check opened rows against a commitment.
    ///
    /// `rows[q][m]` is the row of matrix `m` opened at `indices[q]`.
    ///
    /// # Errors
    ///
    /// When a row does not open against the commitment.
    fn verify_rows(
        &self,
        arith: &mut A,
        commitment: &Self::Target,
        dimensions: &[Dimensions],
        indices: &[A::Index],
        rows: &[Vec<Cow<'_, [T]>>],
        proof: &Self::MultiProof,
    ) -> Result<(), Self::Error>;
}

impl<Val, Challenge, T, M> MmcsArith<Native<Val, Challenge>, T> for M
where
    Val: Field,
    Challenge: ExtensionField<Val>,
    T: Send + Sync + Clone + PartialEq,
    M: Mmcs<T>,
{
    type Commitment = M::Commitment;
    type MultiProof = M::MultiProof;
    type Target = M::Commitment;
    type Error = M::Error;

    #[inline]
    fn load_commitment(
        &self,
        _arith: &mut Native<Val, Challenge>,
        commitment: &M::Commitment,
    ) -> M::Commitment {
        commitment.clone()
    }

    #[inline]
    fn load_key_commitment(
        &self,
        _arith: &mut Native<Val, Challenge>,
        commitment: &M::Commitment,
    ) -> M::Commitment {
        commitment.clone()
    }

    #[inline]
    fn will_observe(&self, _arith: &mut Native<Val, Challenge>, _commitment: &M::Commitment) {}

    #[inline]
    fn verify_rows(
        &self,
        _arith: &mut Native<Val, Challenge>,
        commitment: &M::Commitment,
        dimensions: &[Dimensions],
        indices: &[usize],
        rows: &[Vec<Cow<'_, [T]>>],
        proof: &M::MultiProof,
    ) -> Result<(), M::Error> {
        self.verify_multi_batch(commitment, dimensions, indices, rows, proof)
    }
}
