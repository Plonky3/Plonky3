//! Inside the Posedion paper, they describe that the internal layers of the hash
//! function do not require the full properties of MDS matrices.
//!
//! > For the partial rounds, the MDS property is not required anymore, and
//! > we can set up the matrix MI focusing only on providing full diffusion, breaking
//! > arbitrarily long subspace trails, and ensuring that the polynomial representation
//! > of the scheme is dense." (Section 5.2)
//!
//! This file implements a trait for linear layers that satisfy these three properties.

use p3_field::AbstractField;
use p3_symmetric::Permutation;

pub trait DiffusionPermutation<T: Clone, const WIDTH: usize>: Permutation<[T; WIDTH]> {}

pub fn matmul_internal<AF: AbstractField, const WIDTH: usize>(
    state: &mut [AF; WIDTH],
    mat_internal_diag_m_1: [u64; WIDTH],
) {
    let sum: AF = state.iter().cloned().sum();
    for i in 0..WIDTH {
        state[i] *= AF::from_canonical_u64(mat_internal_diag_m_1[i]);
        state[i] += sum.clone();
    }
}
