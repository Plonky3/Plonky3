//! Cross-permute batched Poseidon2 for Goldilocks.
//!
//! Inlines the full round structure across two states inside one function
//! scope, exposing cross-permute ILP that the single-permute trait method
//! hides behind a function-call boundary. Widths 8 and 12.

use alloc::vec::Vec;

use p3_field::{Algebra, InjectiveMonomial};
use p3_poseidon2::{ExternalLayerConstants, MDSMat4, mds_light_permutation};

use crate::Goldilocks;
use crate::poseidon1::GOLDILOCKS_S_BOX_DEGREE;
use crate::poseidon2::{
    GOLDILOCKS_POSEIDON2_RC_8_EXTERNAL_FINAL, GOLDILOCKS_POSEIDON2_RC_8_EXTERNAL_INITIAL,
    GOLDILOCKS_POSEIDON2_RC_8_INTERNAL, GOLDILOCKS_POSEIDON2_RC_12_EXTERNAL_FINAL,
    GOLDILOCKS_POSEIDON2_RC_12_EXTERNAL_INITIAL, GOLDILOCKS_POSEIDON2_RC_12_INTERNAL,
    internal_layer_mat_mul_goldilocks_8, internal_layer_mat_mul_goldilocks_12,
};

/// Pre-computed Poseidon2 constants for the batched paths.
#[derive(Clone, Debug)]
pub struct Poseidon2BatchedConstants<const W: usize> {
    pub external: ExternalLayerConstants<Goldilocks, W>,
    pub internal: Vec<Goldilocks>,
}

pub fn default_goldilocks_poseidon2_batched_8() -> Poseidon2BatchedConstants<8> {
    Poseidon2BatchedConstants {
        external: ExternalLayerConstants::new(
            GOLDILOCKS_POSEIDON2_RC_8_EXTERNAL_INITIAL.to_vec(),
            GOLDILOCKS_POSEIDON2_RC_8_EXTERNAL_FINAL.to_vec(),
        ),
        internal: GOLDILOCKS_POSEIDON2_RC_8_INTERNAL.to_vec(),
    }
}

pub fn default_goldilocks_poseidon2_batched_12() -> Poseidon2BatchedConstants<12> {
    Poseidon2BatchedConstants {
        external: ExternalLayerConstants::new(
            GOLDILOCKS_POSEIDON2_RC_12_EXTERNAL_INITIAL.to_vec(),
            GOLDILOCKS_POSEIDON2_RC_12_EXTERNAL_FINAL.to_vec(),
        ),
        internal: GOLDILOCKS_POSEIDON2_RC_12_INTERNAL.to_vec(),
    }
}

#[inline(always)]
fn external_round_b2<A, const W: usize>(
    state_a: &mut [A; W],
    state_b: &mut [A; W],
    rc: &[Goldilocks; W],
) where
    A: Algebra<Goldilocks> + InjectiveMonomial<GOLDILOCKS_S_BOX_DEGREE>,
{
    for (s, &c) in state_a.iter_mut().zip(rc.iter()) {
        *s += c;
    }
    for (s, &c) in state_b.iter_mut().zip(rc.iter()) {
        *s += c;
    }
    for s in state_a.iter_mut() {
        *s = s.injective_exp_n();
    }
    for s in state_b.iter_mut() {
        *s = s.injective_exp_n();
    }
    mds_light_permutation(state_a, &MDSMat4);
    mds_light_permutation(state_b, &MDSMat4);
}

/// Apply two Poseidon2 (width 8) permutations in lock-step.
///
/// Bitwise-equivalent to two calls of `Poseidon2Goldilocks<8>::permute_mut`
/// on the platform-generic path.
pub fn permute_batch_b2_p2_w8<A>(
    state_a: &mut [A; 8],
    state_b: &mut [A; 8],
    constants: &Poseidon2BatchedConstants<8>,
) where
    A: Algebra<Goldilocks> + InjectiveMonomial<GOLDILOCKS_S_BOX_DEGREE>,
{
    mds_light_permutation(state_a, &MDSMat4);
    mds_light_permutation(state_b, &MDSMat4);
    for rc in constants.external.get_initial_constants() {
        external_round_b2(state_a, state_b, rc);
    }

    for &rc in constants.internal.iter() {
        state_a[0] += rc;
        state_b[0] += rc;
        state_a[0] = state_a[0].injective_exp_n();
        state_b[0] = state_b[0].injective_exp_n();
        internal_layer_mat_mul_goldilocks_8(state_a);
        internal_layer_mat_mul_goldilocks_8(state_b);
    }

    for rc in constants.external.get_terminal_constants() {
        external_round_b2(state_a, state_b, rc);
    }
}

/// Apply two Poseidon2 (width 12) permutations in lock-step.
pub fn permute_batch_b2_p2_w12<A>(
    state_a: &mut [A; 12],
    state_b: &mut [A; 12],
    constants: &Poseidon2BatchedConstants<12>,
) where
    A: Algebra<Goldilocks> + InjectiveMonomial<GOLDILOCKS_S_BOX_DEGREE>,
{
    mds_light_permutation(state_a, &MDSMat4);
    mds_light_permutation(state_b, &MDSMat4);
    for rc in constants.external.get_initial_constants() {
        external_round_b2(state_a, state_b, rc);
    }

    for &rc in constants.internal.iter() {
        state_a[0] += rc;
        state_b[0] += rc;
        state_a[0] = state_a[0].injective_exp_n();
        state_b[0] = state_b[0].injective_exp_n();
        internal_layer_mat_mul_goldilocks_12(state_a);
        internal_layer_mat_mul_goldilocks_12(state_b);
    }

    for rc in constants.external.get_terminal_constants() {
        external_round_b2(state_a, state_b, rc);
    }
}

#[cfg(test)]
mod tests {
    use p3_poseidon2::Poseidon2;
    use p3_symmetric::Permutation;

    use super::*;
    use crate::poseidon2::{Poseidon2ExternalLayerGoldilocks, Poseidon2InternalLayerGoldilocks};

    type F = Goldilocks;
    type Poseidon2GoldilocksGeneric8 = Poseidon2<
        F,
        Poseidon2ExternalLayerGoldilocks<8>,
        Poseidon2InternalLayerGoldilocks,
        8,
        GOLDILOCKS_S_BOX_DEGREE,
    >;

    fn build_generic_p2_w8() -> Poseidon2GoldilocksGeneric8 {
        Poseidon2::new(
            ExternalLayerConstants::new(
                GOLDILOCKS_POSEIDON2_RC_8_EXTERNAL_INITIAL.to_vec(),
                GOLDILOCKS_POSEIDON2_RC_8_EXTERNAL_FINAL.to_vec(),
            ),
            GOLDILOCKS_POSEIDON2_RC_8_INTERNAL.to_vec(),
        )
    }

    type Poseidon2GoldilocksGeneric12 = Poseidon2<
        F,
        Poseidon2ExternalLayerGoldilocks<12>,
        Poseidon2InternalLayerGoldilocks,
        12,
        GOLDILOCKS_S_BOX_DEGREE,
    >;

    fn build_generic_p2_w12() -> Poseidon2GoldilocksGeneric12 {
        Poseidon2::new(
            ExternalLayerConstants::new(
                GOLDILOCKS_POSEIDON2_RC_12_EXTERNAL_INITIAL.to_vec(),
                GOLDILOCKS_POSEIDON2_RC_12_EXTERNAL_FINAL.to_vec(),
            ),
            GOLDILOCKS_POSEIDON2_RC_12_INTERNAL.to_vec(),
        )
    }

    #[test]
    fn batched_b2_matches_single_w8_scalar() {
        let perm = build_generic_p2_w8();
        let constants = default_goldilocks_poseidon2_batched_8();

        let in_a: [F; 8] = F::new_array([0, 1, 2, 3, 4, 5, 6, 7]);
        let in_b: [F; 8] = F::new_array([100, 101, 102, 103, 104, 105, 106, 107]);

        let mut exp_a = in_a;
        let mut exp_b = in_b;
        perm.permute_mut(&mut exp_a);
        perm.permute_mut(&mut exp_b);

        let mut got_a = in_a;
        let mut got_b = in_b;
        permute_batch_b2_p2_w8(&mut got_a, &mut got_b, &constants);

        assert_eq!(got_a, exp_a, "p2 w8 scalar a mismatch");
        assert_eq!(got_b, exp_b, "p2 w8 scalar b mismatch");
    }

    #[test]
    fn batched_b2_matches_single_w12_scalar() {
        let perm = build_generic_p2_w12();
        let constants = default_goldilocks_poseidon2_batched_12();

        let in_a: [F; 12] = F::new_array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]);
        let in_b: [F; 12] =
            F::new_array([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111]);

        let mut exp_a = in_a;
        let mut exp_b = in_b;
        perm.permute_mut(&mut exp_a);
        perm.permute_mut(&mut exp_b);

        let mut got_a = in_a;
        let mut got_b = in_b;
        permute_batch_b2_p2_w12(&mut got_a, &mut got_b, &constants);

        assert_eq!(got_a, exp_a, "p2 w12 scalar a mismatch");
        assert_eq!(got_b, exp_b, "p2 w12 scalar b mismatch");
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    mod avx512 {
        use super::*;
        use crate::PackedGoldilocksAVX512;

        #[test]
        fn batched_b2_matches_single_w8_avx512() {
            let perm = build_generic_p2_w8();
            let constants = default_goldilocks_poseidon2_batched_8();

            let in_a: [F; 8] = F::new_array([0, 1, 2, 3, 4, 5, 6, 7]);
            let in_b: [F; 8] = F::new_array([100, 101, 102, 103, 104, 105, 106, 107]);

            let mut exp_a: [PackedGoldilocksAVX512; 8] = in_a.map(Into::into);
            let mut exp_b: [PackedGoldilocksAVX512; 8] = in_b.map(Into::into);
            perm.permute_mut(&mut exp_a);
            perm.permute_mut(&mut exp_b);

            let mut got_a: [PackedGoldilocksAVX512; 8] = in_a.map(Into::into);
            let mut got_b: [PackedGoldilocksAVX512; 8] = in_b.map(Into::into);
            permute_batch_b2_p2_w8(&mut got_a, &mut got_b, &constants);

            assert_eq!(got_a, exp_a, "p2 w8 avx512 a mismatch");
            assert_eq!(got_b, exp_b, "p2 w8 avx512 b mismatch");
        }

        #[test]
        fn batched_b2_matches_single_w12_avx512() {
            let perm = build_generic_p2_w12();
            let constants = default_goldilocks_poseidon2_batched_12();

            let in_a: [F; 12] = F::new_array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]);
            let in_b: [F; 12] =
                F::new_array([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111]);

            let mut exp_a: [PackedGoldilocksAVX512; 12] = in_a.map(Into::into);
            let mut exp_b: [PackedGoldilocksAVX512; 12] = in_b.map(Into::into);
            perm.permute_mut(&mut exp_a);
            perm.permute_mut(&mut exp_b);

            let mut got_a: [PackedGoldilocksAVX512; 12] = in_a.map(Into::into);
            let mut got_b: [PackedGoldilocksAVX512; 12] = in_b.map(Into::into);
            permute_batch_b2_p2_w12(&mut got_a, &mut got_b, &constants);

            assert_eq!(got_a, exp_a, "p2 w12 avx512 a mismatch");
            assert_eq!(got_b, exp_b, "p2 w12 avx512 b mismatch");
        }
    }

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        not(target_feature = "avx512f")
    ))]
    mod avx2 {
        use super::*;
        use crate::PackedGoldilocksAVX2;

        #[test]
        fn batched_b2_matches_single_w8_avx2() {
            let perm = build_generic_p2_w8();
            let constants = default_goldilocks_poseidon2_batched_8();

            let in_a: [F; 8] = F::new_array([0, 1, 2, 3, 4, 5, 6, 7]);
            let in_b: [F; 8] = F::new_array([100, 101, 102, 103, 104, 105, 106, 107]);

            let mut exp_a: [PackedGoldilocksAVX2; 8] = in_a.map(Into::into);
            let mut exp_b: [PackedGoldilocksAVX2; 8] = in_b.map(Into::into);
            perm.permute_mut(&mut exp_a);
            perm.permute_mut(&mut exp_b);

            let mut got_a: [PackedGoldilocksAVX2; 8] = in_a.map(Into::into);
            let mut got_b: [PackedGoldilocksAVX2; 8] = in_b.map(Into::into);
            permute_batch_b2_p2_w8(&mut got_a, &mut got_b, &constants);

            assert_eq!(got_a, exp_a, "p2 w8 avx2 a mismatch");
            assert_eq!(got_b, exp_b, "p2 w8 avx2 b mismatch");
        }

        #[test]
        fn batched_b2_matches_single_w12_avx2() {
            let perm = build_generic_p2_w12();
            let constants = default_goldilocks_poseidon2_batched_12();

            let in_a: [F; 12] = F::new_array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]);
            let in_b: [F; 12] =
                F::new_array([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111]);

            let mut exp_a: [PackedGoldilocksAVX2; 12] = in_a.map(Into::into);
            let mut exp_b: [PackedGoldilocksAVX2; 12] = in_b.map(Into::into);
            perm.permute_mut(&mut exp_a);
            perm.permute_mut(&mut exp_b);

            let mut got_a: [PackedGoldilocksAVX2; 12] = in_a.map(Into::into);
            let mut got_b: [PackedGoldilocksAVX2; 12] = in_b.map(Into::into);
            permute_batch_b2_p2_w12(&mut got_a, &mut got_b, &constants);

            assert_eq!(got_a, exp_a, "p2 w12 avx2 a mismatch");
            assert_eq!(got_b, exp_b, "p2 w12 avx2 b mismatch");
        }
    }

    #[cfg(target_arch = "aarch64")]
    mod neon {
        use super::*;
        use crate::PackedGoldilocksNeon;

        #[test]
        fn batched_b2_matches_single_w8_neon() {
            let perm = build_generic_p2_w8();
            let constants = default_goldilocks_poseidon2_batched_8();

            let in_a: [F; 8] = F::new_array([0, 1, 2, 3, 4, 5, 6, 7]);
            let in_b: [F; 8] = F::new_array([100, 101, 102, 103, 104, 105, 106, 107]);

            let mut exp_a: [PackedGoldilocksNeon; 8] = in_a.map(Into::into);
            let mut exp_b: [PackedGoldilocksNeon; 8] = in_b.map(Into::into);
            perm.permute_mut(&mut exp_a);
            perm.permute_mut(&mut exp_b);

            let mut got_a: [PackedGoldilocksNeon; 8] = in_a.map(Into::into);
            let mut got_b: [PackedGoldilocksNeon; 8] = in_b.map(Into::into);
            permute_batch_b2_p2_w8(&mut got_a, &mut got_b, &constants);

            assert_eq!(got_a, exp_a, "p2 w8 neon a mismatch");
            assert_eq!(got_b, exp_b, "p2 w8 neon b mismatch");
        }

        #[test]
        fn batched_b2_matches_single_w12_neon() {
            let perm = build_generic_p2_w12();
            let constants = default_goldilocks_poseidon2_batched_12();

            let in_a: [F; 12] = F::new_array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]);
            let in_b: [F; 12] =
                F::new_array([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111]);

            let mut exp_a: [PackedGoldilocksNeon; 12] = in_a.map(Into::into);
            let mut exp_b: [PackedGoldilocksNeon; 12] = in_b.map(Into::into);
            perm.permute_mut(&mut exp_a);
            perm.permute_mut(&mut exp_b);

            let mut got_a: [PackedGoldilocksNeon; 12] = in_a.map(Into::into);
            let mut got_b: [PackedGoldilocksNeon; 12] = in_b.map(Into::into);
            permute_batch_b2_p2_w12(&mut got_a, &mut got_b, &constants);

            assert_eq!(got_a, exp_a, "p2 w12 neon a mismatch");
            assert_eq!(got_b, exp_b, "p2 w12 neon b mismatch");
        }
    }
}
