//! Generic implementations of Poseidon linear layers.

use p3_field::{Algebra, Field, InjectiveMonomial};

use crate::external::mds_multiply;
use crate::internal::cheap_matmul;

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
    /// Dense MDS matrix-vector multiplication (O(t^2)).
    fn mds_multiply<R: Algebra<F>>(state: &mut [R; WIDTH], mds: &[[F; WIDTH]; WIDTH]) {
        mds_multiply(state, mds);
    }

    /// Sparse matrix-vector multiplication for partial rounds (O(t)).
    fn cheap_matmul<R: Algebra<F>>(state: &mut [R; WIDTH], mds_0_0: F, v: &[F], w_hat: &[F]) {
        cheap_matmul(state, mds_0_0, v, w_hat);
    }
}
