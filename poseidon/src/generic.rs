//! Generic implementations of Poseidon linear layers.
//!
//! This module provides two things:
//!
//! 1. [`GenericPoseidonLinearLayers`] — a trait with default dense/sparse linear
//!    layer methods, used by field-specific implementations that only need to
//!    override individual operations.
//!
//! 2. [`PoseidonExternalLayerGeneric`] and [`PoseidonInternalLayerGeneric`] —
//!    ready-to-use layer structs that implement [`FullRoundLayer`] and
//!    [`PartialRoundLayer`] for **any** field with a conforming MDS permutation.
//!    These are suitable for fields without hand-tuned SIMD implementations
//!    (e.g., Goldilocks, Mersenne-31).

use core::marker::PhantomData;

use p3_field::{Algebra, Field, InjectiveMonomial};
use p3_symmetric::Permutation;

use crate::external::{
    FullRoundConstants, FullRoundLayer, FullRoundLayerConstructor,
    full_round_initial_permute_state, full_round_terminal_permute_state, mds_multiply,
};
use crate::internal::{
    PartialRoundConstants, PartialRoundLayer, PartialRoundLayerConstructor, cheap_matmul,
    partial_permute_state,
};

/// Generic round-constant addition followed by S-box evaluation: `val <- (val + rc)^D`.
#[inline(always)]
pub fn add_rc_and_sbox_generic<F: Field, A: Algebra<F> + InjectiveMonomial<D>, const D: u64>(
    val: &mut A,
    rc: F,
) {
    *val += rc;
    *val = val.injective_exp_n();
}

/// Generic linear layer trait for Poseidon.
///
/// Provides default dense MDS multiplication and sparse matrix multiplication
/// for use when no field-specific optimized implementation is available.
pub trait GenericPoseidonLinearLayers<F: Field, const WIDTH: usize>: Sync {
    /// Dense MDS matrix-vector multiplication in O(t^2).
    ///
    /// Used for the dense transition matrix in partial rounds.
    fn mds_multiply<R: Algebra<F>>(state: &mut [R; WIDTH], mds: &[[F; WIDTH]; WIDTH]) {
        mds_multiply(state, mds);
    }

    /// MDS multiplication dispatched via a permutation trait.
    ///
    /// Used in full rounds. Concrete fields can provide sub-O(t^2)
    /// implementations (e.g., Karatsuba convolution for circulant matrices).
    fn mds_permute<R: Algebra<F>>(state: &mut [R; WIDTH], mds: &impl Permutation<[R; WIDTH]>) {
        mds.permute_mut(state);
    }

    /// Sparse matrix-vector multiplication for partial rounds in O(t).
    fn cheap_matmul<R: Algebra<F>>(state: &mut [R; WIDTH], first_row: &[F; WIDTH], v: &[F; WIDTH]) {
        cheap_matmul(state, first_row, v);
    }
}

/// Generic external (full round) layer for the Poseidon permutation.
///
/// Dispatches MDS multiplication through a [`Permutation`] trait instance,
/// allowing any field with a conforming MDS implementation to use the
/// optimized Poseidon without a field-specific layer.
///
/// The `Mds` type must implement [`Default`] (typically a zero-sized struct)
/// and [`Permutation<[A; WIDTH]>`] for each ring type `A` used.
#[derive(Debug, Clone)]
pub struct PoseidonExternalLayerGeneric<F, Mds, const WIDTH: usize> {
    constants: FullRoundConstants<F, WIDTH>,
    _mds: PhantomData<Mds>,
}

impl<F: Field, Mds, const WIDTH: usize> FullRoundLayerConstructor<F, WIDTH>
    for PoseidonExternalLayerGeneric<F, Mds, WIDTH>
{
    fn new_from_constants(constants: FullRoundConstants<F, WIDTH>) -> Self {
        Self {
            constants,
            _mds: PhantomData,
        }
    }
}

impl<F, A, Mds, const WIDTH: usize, const D: u64> FullRoundLayer<A, WIDTH, D>
    for PoseidonExternalLayerGeneric<F, Mds, WIDTH>
where
    F: Field + InjectiveMonomial<D>,
    A: Algebra<F> + InjectiveMonomial<D>,
    Mds: Permutation<[A; WIDTH]> + Default + Sync + Clone,
{
    fn permute_state_initial(&self, state: &mut [A; WIDTH]) {
        let mds = Mds::default();
        full_round_initial_permute_state::<F, A, _, WIDTH, D>(state, &self.constants, &mds);
    }

    fn permute_state_terminal(&self, state: &mut [A; WIDTH]) {
        let mds = Mds::default();
        full_round_terminal_permute_state::<F, A, _, WIDTH, D>(state, &self.constants, &mds);
    }
}

/// Generic internal (partial round) layer for the Poseidon permutation.
///
/// Uses the sparse matrix decomposition for O(t) partial rounds.
///
/// Works with any field that supports the required algebra operations.
#[derive(Debug, Clone)]
pub struct PoseidonInternalLayerGeneric<F, const WIDTH: usize> {
    constants: PartialRoundConstants<F, WIDTH>,
}

impl<F: Field, const WIDTH: usize> PartialRoundLayerConstructor<F, WIDTH>
    for PoseidonInternalLayerGeneric<F, WIDTH>
{
    fn new_from_constants(constants: PartialRoundConstants<F, WIDTH>) -> Self {
        Self { constants }
    }
}

impl<F, A, const WIDTH: usize, const D: u64> PartialRoundLayer<A, WIDTH, D>
    for PoseidonInternalLayerGeneric<F, WIDTH>
where
    F: Field + InjectiveMonomial<D>,
    A: Algebra<F> + InjectiveMonomial<D>,
{
    fn permute_state(&self, state: &mut [A; WIDTH]) {
        partial_permute_state::<F, A, WIDTH, D>(state, &self.constants);
    }
}
