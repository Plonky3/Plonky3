//! Poseidon2 round logic shared by the x86_64 AVX2 and AVX512 Goldilocks backends.
//!
//! Both backends batch the S-box and internal-round diffusion the same way over their own
//! packed vector type (see each backend's `poseidon2` module for the design rationale); only
//! the concrete packed type and its lane width differ, and neither shows up in the arithmetic
//! below, so it lives here once instead of once per backend.

use p3_field::{Algebra, InjectiveMonomial};
use p3_poseidon2::{MDSMat4, mds_light_permutation};

use crate::Goldilocks;
use crate::poseidon1::GOLDILOCKS_S_BOX_DEGREE;

/// Add a round constant then apply the S-box (`x^7`) to a single packed element. Used by the
/// internal rounds, where only `state[0]` is S-box'd each round; see [`sbox_array`] for the
/// external rounds, where every state element is S-box'd and batching by stage matters.
#[inline(always)]
pub(crate) fn add_rc_and_sbox<P>(val: &mut P, rc: P)
where
    P: Algebra<Goldilocks> + InjectiveMonomial<GOLDILOCKS_S_BOX_DEGREE> + Copy,
{
    *val = (*val + rc).injective_exp_n();
}

/// Apply the Poseidon2 S-box (`x^7`) to every element of a packed state.
///
/// `p3_poseidon2`'s generic external-round driver applies the S-box one element at a time,
/// fully completing one element's multiply/reduce chain (`x^2`, `x^3`, `x^4`, `x^7`) before
/// starting the next. Since every state element's S-box is independent of every other, this
/// instead batches the computation by stage across the whole state: all `WIDTH` squarings,
/// then all `WIDTH` pairs of the next two (mutually independent) products, then all `WIDTH`
/// final products. This gives the compiler `WIDTH`-many independent multiply/reduce chains to
/// interleave at each stage instead of one at a time, mirroring `PrimeCharacteristicRing`'s own
/// `exp_const_u64::<7>` addition chain (`x2 = x^2`, `x3 = x2*x`, `x4 = x2^2`, `x7 = x3*x4`),
/// just computed array-wise rather than per-element.
#[inline(always)]
pub(crate) fn sbox_array<P, const WIDTH: usize>(state: &mut [P; WIDTH])
where
    P: Algebra<Goldilocks> + Copy,
{
    let x2: [P; WIDTH] = core::array::from_fn(|i| state[i].square());
    // `x3`/`x4` start as copies of `x2` (a plain register/array copy, no computation) so that
    // computing `x3[i] = x2[i] * state[i]` and `x4[i] = x2[i]^2` can share a single loop over
    // `i`, instead of two separate full-width passes.
    let mut x3 = x2;
    let mut x4 = x2;
    for i in 0..WIDTH {
        x3[i] *= state[i];
        x4[i] = x4[i].square();
    }
    for i in 0..WIDTH {
        state[i] = x3[i] * x4[i];
    }
}

/// Apply one external round (add round constants, S-box every element, apply the `4x4`-MDS
/// light permutation) to a packed state.
#[inline(always)]
pub(crate) fn external_round<P, const WIDTH: usize>(state: &mut [P; WIDTH], rc: &[P; WIDTH])
where
    P: Algebra<Goldilocks> + Copy,
{
    for i in 0..WIDTH {
        state[i] += rc[i];
    }
    sbox_array(state);
    mds_light_permutation(state, &MDSMat4);
}

/// Apply one internal round (add round constant, S-box `state[0]`, diffuse) of the width-8
/// Goldilocks Poseidon2 internal linear layer to a packed state.
///
/// Mirrors `internal_layer_mat_mul_goldilocks_8` in the crate root `poseidon2` module, with
/// the lane sum split into a part independent of the S-box (`sum_tail`, covering
/// `state[1..8]`) and the S-box'd `state[0]`, combined only once both are ready.
#[inline(always)]
pub(crate) fn internal_round_goldilocks_8<P>(state: &mut [P; 8], rc: P)
where
    P: Algebra<Goldilocks> + InjectiveMonomial<GOLDILOCKS_S_BOX_DEGREE> + Copy,
{
    let s1 = state[1];
    let s2 = state[2];
    let s3 = state[3];
    let s4 = state[4];
    let s5 = state[5];
    let s6 = state[6];
    let s7 = state[7];
    let sum_tail = s1 + s2 + s3 + s4 + s5 + s6 + s7;

    add_rc_and_sbox(&mut state[0], rc);
    let s0 = state[0];
    let sum = sum_tail + s0;

    // V[0] = -2
    state[0] = sum - (s0 + s0);
    // V[1] = 1
    state[1] = sum + s1;
    // V[2] = 2
    state[2] = sum + (s2 + s2);
    // V[3] = 1/2
    state[3] = sum + s3.halve();
    // V[4] = 3
    state[4] = sum + (s4 + s4 + s4);
    // V[5] = -1/2
    state[5] = sum - s5.halve();
    // V[6] = -3
    state[6] = sum - (s6 + s6 + s6);
    // V[7] = -4
    let two_s7 = s7 + s7;
    state[7] = sum - (two_s7 + two_s7);
}

/// Apply one internal round of the width-12 Goldilocks Poseidon2 internal linear layer to a
/// packed state. See [`internal_round_goldilocks_8`] and `internal_layer_mat_mul_goldilocks_12`
/// in the crate root `poseidon2` module.
#[inline(always)]
pub(crate) fn internal_round_goldilocks_12<P>(state: &mut [P; 12], rc: P)
where
    P: Algebra<Goldilocks> + InjectiveMonomial<GOLDILOCKS_S_BOX_DEGREE> + Copy,
{
    let s1 = state[1];
    let s2 = state[2];
    let s3 = state[3];
    let s4 = state[4];
    let s5 = state[5];
    let s6 = state[6];
    let s7 = state[7];
    let s8 = state[8];
    let s9 = state[9];
    let s10 = state[10];
    let s11 = state[11];
    let sum_tail = s1 + s2 + s3 + s4 + s5 + s6 + s7 + s8 + s9 + s10 + s11;

    add_rc_and_sbox(&mut state[0], rc);
    let s0 = state[0];
    let sum = sum_tail + s0;

    // V[0] = -2
    state[0] = sum - (s0 + s0);
    // V[1] = 1
    state[1] = sum + s1;
    // V[2] = 2
    state[2] = sum + (s2 + s2);
    // V[3] = 1/2
    state[3] = sum + s3.halve();
    // V[4] = 3
    state[4] = sum + (s4 + s4 + s4);
    // V[5] = 4
    let two_s5 = s5 + s5;
    state[5] = sum + (two_s5 + two_s5);
    // V[6] = -1/2
    state[6] = sum - s6.halve();
    // V[7] = -3
    state[7] = sum - (s7 + s7 + s7);
    // V[8] = -4
    let two_s8 = s8 + s8;
    state[8] = sum - (two_s8 + two_s8);
    // V[9] = 1/2^2
    state[9] = sum + s9.halve().halve();
    // V[10] = -1/2^2
    state[10] = sum - s10.halve().halve();
    // V[11] = 1/2^3
    state[11] = sum + s11.halve().halve().halve();
}

/// Apply one internal round of the width-16 Goldilocks Poseidon2 internal linear layer to a
/// packed state. See [`internal_round_goldilocks_8`] and `internal_layer_mat_mul_goldilocks_16`
/// in the crate root `poseidon2` module.
#[inline(always)]
pub(crate) fn internal_round_goldilocks_16<P>(state: &mut [P; 16], rc: P)
where
    P: Algebra<Goldilocks> + InjectiveMonomial<GOLDILOCKS_S_BOX_DEGREE> + Copy,
{
    let s1 = state[1];
    let s2 = state[2];
    let s3 = state[3];
    let s4 = state[4];
    let s5 = state[5];
    let s6 = state[6];
    let s7 = state[7];
    let s8 = state[8];
    let s9 = state[9];
    let s10 = state[10];
    let s11 = state[11];
    let s12 = state[12];
    let s13 = state[13];
    let s14 = state[14];
    let s15 = state[15];
    let sum_tail = s1 + s2 + s3 + s4 + s5 + s6 + s7 + s8 + s9 + s10 + s11 + s12 + s13 + s14 + s15;

    add_rc_and_sbox(&mut state[0], rc);
    let s0 = state[0];
    let sum = sum_tail + s0;

    // V[0] = -2
    state[0] = sum - (s0 + s0);
    // V[1] = 1
    state[1] = sum + s1;
    // V[2] = 2
    state[2] = sum + (s2 + s2);
    // V[3] = 1/2
    state[3] = sum + s3.halve();
    // V[4] = 3
    state[4] = sum + (s4 + s4 + s4);
    // V[5] = 4
    let two_s5 = s5 + s5;
    state[5] = sum + (two_s5 + two_s5);
    // V[6] = -1/2
    state[6] = sum - s6.halve();
    // V[7] = -3
    state[7] = sum - (s7 + s7 + s7);
    // V[8] = -4
    let two_s8 = s8 + s8;
    state[8] = sum - (two_s8 + two_s8);
    // V[9] = 1/2^3
    state[9] = sum + s9.halve().halve().halve();
    // V[10] = 1/2^4
    state[10] = sum + s10.halve().halve().halve().halve();
    // V[11] = 1/2^5
    state[11] = sum + s11.halve().halve().halve().halve().halve();
    // V[12] = -1/2^3
    state[12] = sum - s12.halve().halve().halve();
    // V[13] = -1/2^4
    state[13] = sum - s13.halve().halve().halve().halve();
    // V[14] = -1/2^5
    state[14] = sum - s14.halve().halve().halve().halve().halve();
    // V[15] = 1/2^32
    let inv_2_32 = crate::MATRIX_DIAG_16_GOLDILOCKS[15];
    state[15] = sum + s15 * inv_2_32;
}
