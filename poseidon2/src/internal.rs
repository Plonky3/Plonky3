//! Inside the Poseidon2 paper, they describe that the internal layers of the hash
//! function do not require the full properties of MDS matrices.
//!
//! > For the partial rounds, the MDS property is not required anymore, and
//! > we can set up the matrix MI focusing only on providing full diffusion, breaking
//! > arbitrarily long subspace trails, and ensuring that the polynomial representation
//! > of the scheme is dense. (Section 5.2)
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

use alloc::vec::Vec;

use p3_field::{Field, FieldAlgebra, InjectiveMonomial, PrimeCharacteristicRing};

use crate::add_rc_and_sbox_generic;

/// Initialize an internal layer from a set of constants.
pub trait InternalLayerConstructor<F>
where
    F: Field,
{
    /// A constructor which internally will convert the supplied
    /// constants into the appropriate form for the implementation.
    fn new_from_constants(internal_constants: Vec<F>) -> Self;
}

/// Given a vector v compute the matrix vector product (1 + diag(v))state with 1 denoting the constant matrix of ones.
pub fn matmul_internal<
    F: Field,
    FA: FieldAlgebra<F> + PrimeCharacteristicRing,
    const WIDTH: usize,
>(
    state: &mut [FA; WIDTH],
    mat_internal_diag_m_1: [F; WIDTH],
) {
    let sum: FA = state.iter().cloned().sum();
    for i in 0..WIDTH {
        state[i] *= mat_internal_diag_m_1[i];
        state[i] += sum.clone();
    }
}

/// A trait containing all data needed to implement the internal layers of Poseidon2.
pub trait InternalLayer<FA, const WIDTH: usize, const D: u64>: Sync + Clone
where
    FA: PrimeCharacteristicRing,
{
    /// Perform the internal layers of the Poseidon2 permutation on the given state.
    fn permute_state(&self, state: &mut [FA; WIDTH]);
}

/// A helper method which allows any field to easily implement Internal Layer.
/// This should only be used in places where performance is not critical.
#[inline]
pub fn internal_permute_state<
    F: Field,
    FA: FieldAlgebra<F> + PrimeCharacteristicRing + InjectiveMonomial<D>,
    const WIDTH: usize,
    const D: u64,
>(
    state: &mut [FA; WIDTH],
    diffusion_mat: fn(&mut [FA; WIDTH]),
    internal_constants: &[F],
) {
    for elem in internal_constants.iter() {
        add_rc_and_sbox_generic(&mut state[0], *elem);
        diffusion_mat(state);
    }
}

/// The compiler doesn't realize that add is associative
/// so we help it out and minimize the dependency chains by hand.
#[inline(always)]
fn sum_7<FA: PrimeCharacteristicRing + Copy>(state: &[FA]) -> FA {
    assert_eq!(state.len(), 7);

    let s01 = state[0] + state[1];
    let s23 = state[2] + state[3];
    let s45 = state[4] + state[5];

    let s0123 = s01 + s23;
    let s456 = s45 + state[6];
    s0123 + s456
}

/// The compiler doesn't realize that add is associative
/// so we help it out and minimize the dependency chains by hand.
#[inline(always)]
fn sum_8<FA: PrimeCharacteristicRing + Copy>(state: &[FA]) -> FA {
    assert_eq!(state.len(), 8);

    let s01 = state[0] + state[1];
    let s23 = state[2] + state[3];
    let s45 = state[4] + state[5];
    let s67 = state[6] + state[7];

    let s0123 = s01 + s23;
    let s4567 = s45 + s67;
    s0123 + s4567
}

/// The compiler doesn't realize that add is associative
/// so we help it out and minimize the dependency chains by hand.
#[inline(always)]
pub fn sum_15<FA: PrimeCharacteristicRing + Copy>(state: &[FA]) -> FA {
    assert_eq!(state.len(), 15);

    let bot_sum = sum_8(&state[..8]);
    let top_sum = sum_7(&state[8..]);

    bot_sum + top_sum
}

/// The compiler doesn't realize that add is associative
/// so we help it out and minimize the dependency chains by hand.
#[inline(always)]
pub fn sum_23<FA: PrimeCharacteristicRing + Copy>(state: &[FA]) -> FA {
    assert_eq!(state.len(), 23);

    let bot_sum = sum_8(&state[..8]);
    let mid_sum = sum_8(&state[8..16]);
    let top_sum = sum_7(&state[16..]);

    bot_sum + mid_sum + top_sum
}
