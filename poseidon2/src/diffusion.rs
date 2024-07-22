//! Inside the Posedion paper, they describe that the internal layers of the hash
//! function do not require the full properties of MDS matrices.
//!
//! > For the partial rounds, the MDS property is not required anymore, and
//! > we can set up the matrix MI focusing only on providing full diffusion, breaking
//! > arbitrarily long subspace trails, and ensuring that the polynomial representation
//! > of the scheme is dense." (Section 5.2)
//!
//! This file implements a trait for linear layers that satisfy these three properties.

// The requirements translate to the following 3 properties:
// 1: All entries are non 0.
// 2: No Subspace Trails.
// 3: For a matrix of the form 1 + D, the diagonal D should also be non 0.
//
// Properties 1 and 3 are essentially immediate to check and a sufficient condition for property 2
// is that the minimal polynomial of the matrix M and all its powers M^2, ..., M^{2WIDTH} are maximal and irreducible.
// This is equivalent to all the characteristic polynomials being irreducible.
//
// These can be verified by the following sage code (Changing field/vector/length as desired):
//
// field = GF(2^31 - 1);
// length = 16;
// vector = [-2,  1,   2,   4,   8,  16,  32,  64, 128, 256, 1024, 4096, 8192, 16384, 32768, 65536];
// const_mat = matrix.ones(field, length);
// diag_mat  = diagonal_matrix(field, vector);
// for i in range(1, 2 * length + 1)
//      assert ((const_mat + diag_mat)^i).characteristic_polynomial().is_irreducible()

use p3_field::{AbstractField, Field};

/// Given a vector v compute the matrix vector product (1 + diag(v))state with 1 denoting the constant matrix of ones.
#[inline]
pub fn matmul_internal<F: Field, AF: AbstractField<F = F>, const WIDTH: usize>(
    state: &mut [AF; WIDTH],
    mat_internal_diag_m_1: [F; WIDTH],
) {
    let sum: AF = state.iter().cloned().sum();
    for i in 0..WIDTH {
        state[i] *= AF::from_f(mat_internal_diag_m_1[i]);
        state[i] += sum.clone();
    }
}

#[inline]
pub fn internal_permute_state<AF: AbstractField, const WIDTH: usize, const D: u64>(
    state: &mut [AF; WIDTH],
    diffusion_mat: fn(&mut [AF; WIDTH]),
    internal_constants: &[AF::F],
) {
    for elem in internal_constants.iter() {
        state[0] += AF::from_f(*elem);
        state[0] = state[0].exp_const_u64::<D>();
        diffusion_mat(state);
    }
}

pub trait InternalLayer<AF: AbstractField, const WIDTH: usize, const D: u64>: Sync + Clone {
    // In the basic case, InternalState = [AF; WIDTH] but we will use delayed reduction states in other cases.
    type InternalState;

    // In the basic case this is AF::F but more generally we should save constants in a way to avoid unneeded manipulations.
    type InternalConstantsType;

    fn permute_state(
        &self,
        state: &mut Self::InternalState,
        internal_constants: &[Self::InternalConstantsType],
    );
}
