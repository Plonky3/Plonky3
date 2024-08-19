use p3_field::PrimeField32;
use p3_poseidon2::{
    external_final_permute_state, external_initial_permute_state, internal_permute_state,
    ExternalLayer, InternalLayer, MDSMat4, Poseidon2ExternalPackedConstants,
    Poseidon2InternalPackedConstants,
};

use crate::{from_u62, to_mersenne31_array, Mersenne31};

// See poseidon2\src\diffusion.rs for information on how to double check these matrices in Sage.
// Optimised diffusion matrices for Mersenne31/16:
// Small entries: [-2, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 17]
// Power of 2 entries: [-2,  1,   2,   4,   8,  16,  32,  64, 128, 256, 1024, 4096, 8192, 16384, 32768, 65536]
//                   = [?, 2^0, 2^1, 2^2, 2^3, 2^4, 2^5, 2^6, 2^7, 2^8, 2^10, 2^12, 2^13,  2^14,  2^15, 2^16]
//
// Optimised diffusion matrices for Mersenne31/24:
// Small entries: [-2, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22, 23, 24]
// Power of 2 entries: [-2,  1,   2,   4,   8,  16,  32,  64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304]
//                   = [?, 2^0, 2^1, 2^2, 2^3, 2^4, 2^5, 2^6, 2^7, 2^8, 2^9, 2^10, 2^11, 2^12, 2^13,  2^14,  2^15,  2^16,   2^17,   2^18,   2^19,    2^20,    2^21,    2^22]
//
// Long term, POSEIDON2_INTERNAL_MATRIX_DIAG_16, POSEIDON2_INTERNAL_MATRIX_DIAG_24 can be removed.
// Currently they are needed for packed field implementations.
// They need to be pub and not pub(crate) as otherwise clippy gets annoyed if no vector intrinsics are available.
pub const POSEIDON2_INTERNAL_MATRIX_DIAG_16: [Mersenne31; 16] = to_mersenne31_array([
    Mersenne31::ORDER_U32 - 2,
    1,
    1 << 1,
    1 << 2,
    1 << 3,
    1 << 4,
    1 << 5,
    1 << 6,
    1 << 7,
    1 << 8,
    1 << 10,
    1 << 12,
    1 << 13,
    1 << 14,
    1 << 15,
    1 << 16,
]);

const POSEIDON2_INTERNAL_MATRIX_DIAG_16_SHIFTS: [u8; 15] =
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 13, 14, 15, 16];

pub const POSEIDON2_INTERNAL_MATRIX_DIAG_24: [Mersenne31; 24] = to_mersenne31_array([
    Mersenne31::ORDER_U32 - 2,
    1,
    1 << 1,
    1 << 2,
    1 << 3,
    1 << 4,
    1 << 5,
    1 << 6,
    1 << 7,
    1 << 8,
    1 << 9,
    1 << 10,
    1 << 11,
    1 << 12,
    1 << 13,
    1 << 14,
    1 << 15,
    1 << 16,
    1 << 17,
    1 << 18,
    1 << 19,
    1 << 20,
    1 << 21,
    1 << 22,
]);

const POSEIDON2_INTERNAL_MATRIX_DIAG_24_SHIFTS: [u8; 23] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
];

fn permute_mut<const N: usize>(state: &mut [Mersenne31; N], shifts: &[u8]) {
    let part_sum: u64 = state.iter().skip(1).map(|x| x.value as u64).sum();
    let full_sum = part_sum + (state[0].value as u64);
    let s0 = part_sum + (-state[0]).value as u64;
    state[0] = from_u62(s0);
    for i in 1..N {
        let si = full_sum + ((state[i].value as u64) << shifts[i - 1]);
        state[i] = from_u62(si);
    }
}

#[derive(Debug, Clone, Default)]
pub struct Poseidon2InternalLayerMersenne31;

impl InternalLayer<Mersenne31, 16, 5> for Poseidon2InternalLayerMersenne31 {
    type InternalState = [Mersenne31; 16];

    fn permute_state(
        &self,
        state: &mut Self::InternalState,
        internal_constants: &[Mersenne31],
        _packed_internal_constants: &[<Poseidon2InternalLayerMersenne31 as Poseidon2InternalPackedConstants::<Mersenne31>>::ConstantsType],
    ) {
        internal_permute_state::<Mersenne31, 16, 5>(
            state,
            |x| permute_mut(x, &POSEIDON2_INTERNAL_MATRIX_DIAG_16_SHIFTS),
            internal_constants,
        )
    }
}

impl InternalLayer<Mersenne31, 24, 5> for Poseidon2InternalLayerMersenne31 {
    type InternalState = [Mersenne31; 24];

    fn permute_state(
        &self,
        state: &mut Self::InternalState,
        internal_constants: &[Mersenne31],
        _packed_internal_constants: &[<Poseidon2InternalLayerMersenne31 as Poseidon2InternalPackedConstants::<Mersenne31>>::ConstantsType],
    ) {
        internal_permute_state::<Mersenne31, 24, 5>(
            state,
            |x| permute_mut(x, &POSEIDON2_INTERNAL_MATRIX_DIAG_24_SHIFTS),
            internal_constants,
        )
    }
}

#[derive(Default, Clone)]
pub struct Poseidon2ExternalLayerMersenne31;

impl<const WIDTH: usize> ExternalLayer<Mersenne31, WIDTH, 5> for Poseidon2ExternalLayerMersenne31 {
    type InternalState = [Mersenne31; WIDTH];

    fn permute_state_initial(
        &self,
        mut state: [Mersenne31; WIDTH],
        initial_external_constants: &[[Mersenne31; WIDTH]],
        _packed_initial_external_constants: &[<Poseidon2ExternalLayerMersenne31 as Poseidon2ExternalPackedConstants::<Mersenne31, WIDTH>>::ConstantsType],
    ) -> [Mersenne31; WIDTH] {
        external_initial_permute_state::<Mersenne31, MDSMat4, WIDTH, 5>(
            &mut state,
            initial_external_constants,
            &MDSMat4,
        );
        state
    }

    fn permute_state_final(
        &self,
        mut state: Self::InternalState,
        final_external_constants: &[[Mersenne31; WIDTH]],
        _packed_final_external_constants: &[<Poseidon2ExternalLayerMersenne31 as Poseidon2ExternalPackedConstants::<Mersenne31, WIDTH>>::ConstantsType],
    ) -> [Mersenne31; WIDTH] {
        external_final_permute_state::<Mersenne31, MDSMat4, WIDTH, 5>(
            &mut state,
            final_external_constants,
            &MDSMat4,
        );
        state
    }
}

#[cfg(test)]
mod tests {
    use p3_field::AbstractField;
    use p3_poseidon2::Poseidon2;
    use p3_symmetric::Permutation;
    use rand::SeedableRng;
    use rand_xoshiro::Xoroshiro128Plus;

    use super::*;

    type F = Mersenne31;

    // We need to make some round constants. We use Xoroshiro128Plus for this as we can easily match this PRNG in sage.
    // See: https://github.com/0xPolygonZero/hash-constants for the sage code used to create all these tests.

    // Our Poseidon2 Implementation for Mersenne31
    fn poseidon2_mersenne31<const WIDTH: usize, const D: u64>(input: &mut [F; WIDTH])
    where
        Poseidon2ExternalLayerMersenne31: ExternalLayer<Mersenne31, WIDTH, D>,
        Poseidon2InternalLayerMersenne31: InternalLayer<
            Mersenne31,
            WIDTH,
            D,
            InternalState = <Poseidon2ExternalLayerMersenne31 as ExternalLayer<
                Mersenne31,
                WIDTH,
                D,
            >>::InternalState,
        >,
    {
        let mut rng = Xoroshiro128Plus::seed_from_u64(1);

        // Our Poseidon2 implementation.
        let poseidon2: Poseidon2<
            F,
            Poseidon2ExternalLayerMersenne31,
            Poseidon2InternalLayerMersenne31,
            WIDTH,
            D,
        > = Poseidon2::new_from_rng_128(
            Poseidon2ExternalLayerMersenne31,
            Poseidon2InternalLayerMersenne31,
            &mut rng,
        );

        poseidon2.permute_mut(input);
    }

    /// Test on a roughly random input.
    /// This random input is generated by the following sage code:
    /// set_random_seed(16)
    /// vector([M31.random_element() for t in range(16)]).
    #[test]
    fn test_poseidon2_width_16_random() {
        let mut input: [F; 16] = [
            894848333, 1437655012, 1200606629, 1690012884, 71131202, 1749206695, 1717947831,
            120589055, 19776022, 42382981, 1831865506, 724844064, 171220207, 1299207443, 227047920,
            1783754913,
        ]
        .map(F::from_canonical_u32);

        let expected: [F; 16] = [
            1124552602, 2127602268, 1834113265, 1207687593, 1891161485, 245915620, 981277919,
            627265710, 1534924153, 1580826924, 887997842, 1526280482, 547791593, 1028672510,
            1803086471, 323071277,
        ]
        .map(F::from_canonical_u32);

        poseidon2_mersenne31::<16, 5>(&mut input);
        assert_eq!(input, expected);
    }

    /// Test on a roughly random input.
    /// This random input is generated by the following sage code:
    /// set_random_seed(24)
    /// vector([M31.random_element() for t in range(24)]).
    #[test]
    fn test_poseidon2_width_24_random() {
        let mut input: [F; 24] = [
            886409618, 1327899896, 1902407911, 591953491, 648428576, 1844789031, 1198336108,
            355597330, 1799586834, 59617783, 790334801, 1968791836, 559272107, 31054313,
            1042221543, 474748436, 135686258, 263665994, 1962340735, 1741539604, 2026927696,
            449439011, 1131357108, 50869465,
        ]
        .map(F::from_canonical_u32);

        let expected: [F; 24] = [
            87189408, 212775836, 954807335, 1424761838, 1222521810, 1264950009, 1891204592,
            710452896, 957091834, 1776630156, 1091081383, 786687731, 1101902149, 1281649821,
            436070674, 313565599, 1961711763, 2002894460, 2040173120, 854107426, 25198245,
            1967213543, 604802266, 2086190331,
        ]
        .map(F::from_canonical_u32);

        poseidon2_mersenne31::<24, 5>(&mut input);
        assert_eq!(input, expected);
    }
}
