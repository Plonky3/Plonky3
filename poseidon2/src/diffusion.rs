//! Inside the Posedion paper, they describe that the internal layers of the hash
//! function do not require the full properties of MDS matrices.
//!
//! > For the partial rounds, the MDS property is not required anymore, and
//! > we can set up the matrix MI focusing only on providing full diffusion, breaking
//! > arbitrarily long subspace trails, and ensuring that the polynomial representation
//! > of the scheme is dense." (Section 5.2)
//!
//! This file implements a trait for linear layers that satisfy these three properties.

use p3_field::Field;
use p3_symmetric::permutation::ArrayPermutation;

pub trait DiffusionPermutation<T, const WIDTH: usize>: ArrayPermutation<T, WIDTH> {}

pub fn matmul_internal<F: Field, const WIDTH: usize>(
    state: &mut [F; WIDTH],
    mat_internal_diag_m_1: [u64; WIDTH],
) {
    match WIDTH {
        2 => {
            let mut sum = state[0];
            sum += state[1];
            state[0] += sum;
            state[1] += state[1];
            state[1] += sum;
        }
        3 => {
            let mut sum = state[0];
            sum += state[1];
            sum += state[2];
            state[0] += sum;
            state[1] += sum;
            state[2] += state[2];
            state[2] += sum;
        }
        4 | 8 | 12 | 16 | 20 | 24 => {
            let mut sum = state[0];
            #[allow(clippy::needless_range_loop)]
            for i in 1..WIDTH {
                sum += state[i];
            }
            for i in 0..WIDTH {
                state[i] *= F::from_canonical_u64(mat_internal_diag_m_1[i]);
                state[i] += sum;
            }
        }
        _ => panic!("WIDTH not supported"),
    }
}
