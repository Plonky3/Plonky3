//! AVX2-optimized Poseidon1 permutation layers for Mersenne31.

use alloc::vec::Vec;
use core::arch::x86_64::__m256i;

use p3_field::InjectiveMonomial;
use p3_poseidon1::external::{
    FullRoundConstants, FullRoundLayer, FullRoundLayerConstructor, mds_multiply,
};
use p3_poseidon1::internal::{
    PartialRoundConstants, PartialRoundLayer, PartialRoundLayerConstructor, cheap_matmul,
    partial_permute_state,
};
use p3_symmetric::Permutation;

use super::utils::{add_rc_and_sbox, convert_to_vec_neg_form};
use crate::{MdsMatrixMersenne31, Mersenne31, PackedMersenne31AVX2};

/// AVX2-optimized external (full-round) layer for Mersenne31 Poseidon1.
///
/// Stores round constants in two forms:
/// - **Scalar** (`FullRoundConstants<Mersenne31, WIDTH>`) — used by the
///   scalar fallback path (`FullRoundLayer<Mersenne31, …>`).
/// - **Packed** (`Vec<[__m256i; WIDTH]>`) — each constant broadcast to
///   all eight AVX2 lanes in negative form, enabling the fused [`add_rc_and_sbox`] path.
#[derive(Clone)]
pub struct Poseidon1ExternalLayerMersenne31<const WIDTH: usize> {
    constants: FullRoundConstants<Mersenne31, WIDTH>,
    packed_initial_constants: Vec<[__m256i; WIDTH]>,
    packed_terminal_constants: Vec<[__m256i; WIDTH]>,
}

impl<const WIDTH: usize> FullRoundLayerConstructor<Mersenne31, WIDTH>
    for Poseidon1ExternalLayerMersenne31<WIDTH>
{
    fn new_from_constants(constants: FullRoundConstants<Mersenne31, WIDTH>) -> Self {
        let pack_rc = |rcs: &[[Mersenne31; WIDTH]]| -> Vec<[__m256i; WIDTH]> {
            rcs.iter()
                .map(|rc| rc.map(|c| convert_to_vec_neg_form(c.value as i32)))
                .collect()
        };
        let packed_initial_constants = pack_rc(&constants.initial);
        let packed_terminal_constants = pack_rc(&constants.terminal);
        Self {
            constants,
            packed_initial_constants,
            packed_terminal_constants,
        }
    }
}

/// Apply a sequence of full rounds using AVX2-packed negative-form constants.
///
/// For each round: fuse `add_rc + x^5` via [`add_rc_and_sbox`], then
/// apply the MDS via the permutation trait.
#[inline]
fn full_rounds_packed<const WIDTH: usize>(
    state: &mut [PackedMersenne31AVX2; WIDTH],
    packed_constants: &[[__m256i; WIDTH]],
) where
    MdsMatrixMersenne31: Permutation<[PackedMersenne31AVX2; WIDTH]>,
{
    let mds = MdsMatrixMersenne31;
    for rc in packed_constants {
        for (s, &c) in state.iter_mut().zip(rc.iter()) {
            add_rc_and_sbox(s, c);
        }
        mds.permute_mut(state);
    }
}

/// Packed AVX2 path: fused AddRC + S-box with pre-packed negative-form constants.
impl<const WIDTH: usize> FullRoundLayer<PackedMersenne31AVX2, WIDTH, 5>
    for Poseidon1ExternalLayerMersenne31<WIDTH>
where
    MdsMatrixMersenne31: Permutation<[PackedMersenne31AVX2; WIDTH]>,
{
    fn permute_state_initial(&self, state: &mut [PackedMersenne31AVX2; WIDTH]) {
        full_rounds_packed(state, &self.packed_initial_constants);
    }

    fn permute_state_terminal(&self, state: &mut [PackedMersenne31AVX2; WIDTH]) {
        full_rounds_packed(state, &self.packed_terminal_constants);
    }
}

impl<const WIDTH: usize> FullRoundLayer<Mersenne31, WIDTH, 5>
    for Poseidon1ExternalLayerMersenne31<WIDTH>
where
    MdsMatrixMersenne31: Permutation<[Mersenne31; WIDTH]>,
{
    fn permute_state_initial(&self, state: &mut [Mersenne31; WIDTH]) {
        let mds = MdsMatrixMersenne31;
        for round_constants in &self.constants.initial {
            for (s, &rc) in state.iter_mut().zip(round_constants.iter()) {
                *s += rc;
            }
            for s in state.iter_mut() {
                *s = s.injective_exp_n();
            }
            mds.permute_mut(state);
        }
    }

    fn permute_state_terminal(&self, state: &mut [Mersenne31; WIDTH]) {
        let mds = MdsMatrixMersenne31;
        for round_constants in &self.constants.terminal {
            for (s, &rc) in state.iter_mut().zip(round_constants.iter()) {
                *s += rc;
            }
            for s in state.iter_mut() {
                *s = s.injective_exp_n();
            }
            mds.permute_mut(state);
        }
    }
}

/// AVX2-optimized internal (partial-round) layer for Mersenne31 Poseidon1.
///
/// Uses the sparse matrix decomposition from the Poseidon paper (Appendix B).
/// Constants are stored as scalar `Mersenne31` values; the `Algebra<Mersenne31>`
/// impl on `PackedMersenne31AVX2` handles broadcasting during multiplication.
///
/// Each partial round applies the S-box only to `state[0]`, then performs
/// a cheap sparse matrix-vector product via [`cheap_matmul`].
#[derive(Clone)]
pub struct Poseidon1InternalLayerMersenne31<const WIDTH: usize> {
    constants: PartialRoundConstants<Mersenne31, WIDTH>,
}

impl<const WIDTH: usize> PartialRoundLayerConstructor<Mersenne31, WIDTH>
    for Poseidon1InternalLayerMersenne31<WIDTH>
{
    fn new_from_constants(constants: PartialRoundConstants<Mersenne31, WIDTH>) -> Self {
        Self { constants }
    }
}

/// Packed AVX2 path: S-box on `state[0]` only, sparse matmul via scalar constants.
impl<const WIDTH: usize> PartialRoundLayer<PackedMersenne31AVX2, WIDTH, 5>
    for Poseidon1InternalLayerMersenne31<WIDTH>
{
    fn permute_state(&self, state: &mut [PackedMersenne31AVX2; WIDTH]) {
        for (s, &rc) in state
            .iter_mut()
            .zip(self.constants.first_round_constants.iter())
        {
            *s += rc;
        }

        mds_multiply(state, &self.constants.m_i);

        let rounds_p = self.constants.sparse_first_row.len();

        for r in 0..rounds_p - 1 {
            state[0] = state[0].injective_exp_n();
            state[0] += self.constants.round_constants[r];
            cheap_matmul(
                state,
                &self.constants.sparse_first_row[r],
                &self.constants.v[r],
            );
        }

        state[0] = state[0].injective_exp_n();
        cheap_matmul(
            state,
            &self.constants.sparse_first_row[rounds_p - 1],
            &self.constants.v[rounds_p - 1],
        );
    }
}

/// Scalar fallback: delegates to the generic partial-round implementation.
impl<const WIDTH: usize> PartialRoundLayer<Mersenne31, WIDTH, 5>
    for Poseidon1InternalLayerMersenne31<WIDTH>
{
    fn permute_state(&self, state: &mut [Mersenne31; WIDTH]) {
        partial_permute_state::<Mersenne31, Mersenne31, WIDTH, 5>(state, &self.constants);
    }
}

#[cfg(test)]
mod tests {
    use p3_field::PrimeCharacteristicRing;
    use p3_symmetric::Permutation;
    use proptest::prelude::*;

    use crate::poseidon1::{default_mersenne31_poseidon1_16, default_mersenne31_poseidon1_32};
    use crate::{Mersenne31, PackedMersenne31AVX2};

    type F = Mersenne31;

    /// Known-answer test for width 16 through the AVX2 packed path.
    #[test]
    fn test_avx2_poseidon1_width_16() {
        let perm = default_mersenne31_poseidon1_16();

        let input: [F; 16] = F::new_array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);

        let expected: [F; 16] = F::new_array([
            763678880, 1665665156, 138326798, 2029009038, 523315643, 1240724959, 799985579,
            1533764468, 1851415257, 580298256, 158301910, 1486286674, 1604442932, 919070942,
            791307160, 922090452,
        ]);

        let mut avx2_input = input.map(Into::<PackedMersenne31AVX2>::into);
        perm.permute_mut(&mut avx2_input);
        let avx2_output = avx2_input.map(|x| x.0[0]);

        assert_eq!(avx2_output, expected);
    }

    /// Known-answer test for width 32 through the AVX2 packed path.
    #[test]
    fn test_avx2_poseidon1_width_32() {
        let perm = default_mersenne31_poseidon1_32();

        let input: [F; 32] = F::new_array([
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31,
        ]);

        let expected: [F; 32] = F::new_array([
            493956664, 1487097341, 1986839634, 1423871566, 183460581, 827438159, 7996988,
            141241897, 1403482130, 847367286, 2077667889, 1108646476, 1352254530, 1822401306,
            809224972, 1606586582, 1039326136, 622010047, 1526365331, 1585000638, 1938294847,
            559133752, 570966981, 1111956911, 1758188893, 1919461707, 940683889, 1707731554,
            1949319314, 1540753789, 1964681567, 229242586,
        ]);

        let mut avx2_input = input.map(Into::<PackedMersenne31AVX2>::into);
        perm.permute_mut(&mut avx2_input);
        let avx2_output = avx2_input.map(|x| x.0[0]);

        assert_eq!(avx2_output, expected);
    }

    fn arb_f() -> impl Strategy<Value = F> {
        prop::num::u32::ANY.prop_map(F::from_u32)
    }

    proptest! {
        #[test]
        fn poseidon1_avx2_matches_scalar_width_16(
            input in prop::array::uniform16(arb_f())
        ) {
            let perm = default_mersenne31_poseidon1_16();

            let mut expected = input;
            perm.permute_mut(&mut expected);

            let mut avx2_input = input.map(Into::<PackedMersenne31AVX2>::into);
            perm.permute_mut(&mut avx2_input);
            let avx2_output = avx2_input.map(|x| x.0[0]);

            prop_assert_eq!(avx2_output, expected);
        }

        #[test]
        fn poseidon1_avx2_matches_scalar_width_32(
            input in prop::array::uniform32(arb_f())
        ) {
            let perm = default_mersenne31_poseidon1_32();

            let mut expected = input;
            perm.permute_mut(&mut expected);

            let mut avx2_input = input.map(Into::<PackedMersenne31AVX2>::into);
            perm.permute_mut(&mut avx2_input);
            let avx2_output = avx2_input.map(|x| x.0[0]);

            prop_assert_eq!(avx2_output, expected);
        }
    }
}
