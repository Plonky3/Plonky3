//! Inside the Posedion paper, they describe that the internal layers of the hash
//! function do not require the full properties of MDS matrices.
//!
//! > For the partial rounds, the MDS property is not required anymore, and
//! > we can set up the matrix MI focusing only on providing full diffusion, breaking
//! > arbitrarily long subspace trails, and ensuring that the polynomial representation
//! > of the scheme is dense." (Section 5.2)
//!
//! > These properties can be ensured by checking the following two conditions:
//! > Every entry of the Matrix is non 0.
//! > The characteristic polynomial of the matrix is irreducible.
//!
//! This file implements a trait for linear layers that satisfy these three properties.

use p3_field::{AbstractField, Field};
use p3_symmetric::Permutation;

pub trait DiffusionPermutation<T: Clone, const WIDTH: usize>: Permutation<[T; WIDTH]> {}

pub fn matmul_internal<F: Field, AF: AbstractField<F = F>, const WIDTH: usize>(
    state: &mut [AF; WIDTH],
    mat_internal_diag_m_1: [F; WIDTH],
) {
    let sum: AF = state.iter().cloned().sum();
    for i in 0..WIDTH {
        state[i] = state[i].clone() * AF::from_f(mat_internal_diag_m_1[i]);
        state[i] = state[i].clone() + sum.clone();
    }
}
