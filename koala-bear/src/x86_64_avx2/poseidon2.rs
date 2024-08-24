use core::{
    arch::x86_64::{self, __m256i},
    mem::transmute,
};

use p3_monty_31::InternalLayerParametersAVX2;

use crate::KoalaBearInternalLayerParameters;

const PACKED_2P: __m256i = unsafe { transmute([0x7f0000010_u64; 4]) };

// [00, 01, 02, 03, 04, 05, 06, 07, 08, 09, 10, 11, 12, 13, 15]
// [32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 17]
// [00, 01, 02, 03, 04, 05, 06, 07, 08, 09, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23]
// [32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 9]

impl InternalLayerParametersAVX2<16> for KoalaBearInternalLayerParameters {
    fn diagonal_mul_shifted_input(state: &mut [__m256i; 16]) {
        unsafe {
            // State[0] lies in {-P, ... P}.
            // We need -2 State[0]
            state[0] = x86_64::_mm256_sub_epi64(PACKED_2P, state[0]);
            state[0] = x86_64::_mm256_add_epi64(state[0], state[0]);

            state[1] = x86_64::_mm256_srli_epi64::<32>(state[1]);
            state[2] = x86_64::_mm256_srli_epi64::<31>(state[2]);
            state[3] = x86_64::_mm256_srli_epi64::<30>(state[3]);
            state[4] = x86_64::_mm256_srli_epi64::<29>(state[4]);
            state[5] = x86_64::_mm256_srli_epi64::<28>(state[5]);
            state[6] = x86_64::_mm256_srli_epi64::<27>(state[6]);
            state[7] = x86_64::_mm256_srli_epi64::<26>(state[7]);
            state[8] = x86_64::_mm256_srli_epi64::<25>(state[8]);
            state[9] = x86_64::_mm256_srli_epi64::<24>(state[9]);
            state[10] = x86_64::_mm256_srli_epi64::<23>(state[10]);
            state[11] = x86_64::_mm256_srli_epi64::<22>(state[11]);
            state[12] = x86_64::_mm256_srli_epi64::<21>(state[12]);
            state[13] = x86_64::_mm256_srli_epi64::<20>(state[13]);
            state[14] = x86_64::_mm256_srli_epi64::<19>(state[14]);
            state[15] = x86_64::_mm256_srli_epi64::<17>(state[15]);
        }
    }

    fn diagonal_mul_standard_input(state: &mut [__m256i; 16]) {
        unsafe {
            // State[0] lies in {-P, ... P}.
            // We need -2 State[0]
            state[0] = x86_64::_mm256_sub_epi64(PACKED_2P, state[0]);
            state[0] = x86_64::_mm256_add_epi64(state[0], state[0]);

            state[2] = x86_64::_mm256_slli_epi64::<1>(state[2]);
            state[3] = x86_64::_mm256_slli_epi64::<2>(state[3]);
            state[4] = x86_64::_mm256_slli_epi64::<3>(state[4]);
            state[5] = x86_64::_mm256_slli_epi64::<4>(state[5]);
            state[6] = x86_64::_mm256_slli_epi64::<5>(state[6]);
            state[7] = x86_64::_mm256_slli_epi64::<6>(state[7]);
            state[8] = x86_64::_mm256_slli_epi64::<7>(state[8]);
            state[9] = x86_64::_mm256_slli_epi64::<8>(state[9]);
            state[10] = x86_64::_mm256_slli_epi64::<9>(state[10]);
            state[11] = x86_64::_mm256_slli_epi64::<10>(state[11]);
            state[12] = x86_64::_mm256_slli_epi64::<11>(state[12]);
            state[13] = x86_64::_mm256_slli_epi64::<12>(state[13]);
            state[14] = x86_64::_mm256_slli_epi64::<13>(state[14]);
            state[15] = x86_64::_mm256_slli_epi64::<15>(state[15]);
        }
    }

    /// Reduce input elements from [0, 127P] to [0, 2P).
    fn reduce(input: __m256i) -> __m256i {
        unsafe {
            // We approach this using Crandall reduction:
            // As input < 127P < 2^7 * 2^31 = 2^38 all bits above the 38'th bit are 0.
            // Hence writing input = x0 + 2^31 x1 with x0 < 2^31 this function returns
            // output = x0 + (2^24 - 1)x1.

            // We quickly prove that output < 2P:
            // As input <= 127P < 127 2^31, x1 < 127.
            // As x1 is an integer we must have x1 <= 126 and so:
            // output < 2^31 + (2^7 - 2)(2^24 - 1)
            //        = 2 * 2^31 - 2 * 2^24 - 2^7 + 2
            //        = 2P - 2^7
            //        < 2P

            let top_7 = x86_64::_mm256_srli_epi64::<31>(input); // Get the top 7 bits.

            let top_7_x_2exp24 = x86_64::_mm256_slli_epi64::<24>(top_7); //
            const LOW_31: __m256i = unsafe { transmute([0x7fffffff_i64; 4]) };
            let bottom_31 = x86_64::_mm256_and_si256(input, LOW_31);

            let sub = x86_64::_mm256_sub_epi64(bottom_31, top_7);
            x86_64::_mm256_add_epi64(sub, top_7_x_2exp24)
        }
    }
}

impl InternalLayerParametersAVX2<24> for KoalaBearInternalLayerParameters {
    fn diagonal_mul_shifted_input(state: &mut [__m256i; 24]) {
        unsafe {
            // State[0] lies in {-P, ... P}.
            // We need -2 State[0]
            state[0] = x86_64::_mm256_sub_epi64(PACKED_2P, state[0]);
            state[0] = x86_64::_mm256_add_epi64(state[0], state[0]);

            state[1] = x86_64::_mm256_srli_epi64::<32>(state[1]);
            state[2] = x86_64::_mm256_srli_epi64::<31>(state[2]);
            state[3] = x86_64::_mm256_srli_epi64::<30>(state[3]);
            state[4] = x86_64::_mm256_srli_epi64::<29>(state[4]);
            state[5] = x86_64::_mm256_srli_epi64::<28>(state[5]);
            state[6] = x86_64::_mm256_srli_epi64::<27>(state[6]);
            state[7] = x86_64::_mm256_srli_epi64::<26>(state[7]);
            state[8] = x86_64::_mm256_srli_epi64::<25>(state[8]);
            state[9] = x86_64::_mm256_srli_epi64::<24>(state[9]);
            state[10] = x86_64::_mm256_srli_epi64::<23>(state[10]);
            state[11] = x86_64::_mm256_srli_epi64::<22>(state[11]);
            state[12] = x86_64::_mm256_srli_epi64::<21>(state[12]);
            state[13] = x86_64::_mm256_srli_epi64::<20>(state[13]);
            state[14] = x86_64::_mm256_srli_epi64::<19>(state[14]);
            state[15] = x86_64::_mm256_srli_epi64::<18>(state[15]);
            state[16] = x86_64::_mm256_srli_epi64::<17>(state[16]);
            state[17] = x86_64::_mm256_srli_epi64::<16>(state[17]);
            state[18] = x86_64::_mm256_srli_epi64::<15>(state[18]);
            state[19] = x86_64::_mm256_srli_epi64::<14>(state[19]);
            state[20] = x86_64::_mm256_srli_epi64::<13>(state[20]);
            state[21] = x86_64::_mm256_srli_epi64::<12>(state[21]);
            state[22] = x86_64::_mm256_srli_epi64::<11>(state[22]);
            state[23] = x86_64::_mm256_srli_epi64::<9>(state[23]);
        }
    }

    /// Reduce elements from [0, ..., 127P] to [0, ..., 2P].
    fn reduce(input: __m256i) -> __m256i {
        unsafe {
            let top_7 = x86_64::_mm256_srli_epi64::<31>(input);
            let top_7_x_2exp24 = x86_64::_mm256_slli_epi64::<24>(top_7);
            const LOW_31: __m256i = unsafe { transmute([0x7fffffff_i64; 4]) };
            let bottom_31 = x86_64::_mm256_and_si256(input, LOW_31);

            let sub = x86_64::_mm256_sub_epi64(bottom_31, top_7);
            x86_64::_mm256_add_epi64(sub, top_7_x_2exp24)
        }
    }

    fn diagonal_mul_standard_input(state: &mut [__m256i; 24]) {
        unsafe {
            // State[0] lies in {-P, ... P}.
            // We need -2 State[0]
            state[0] = x86_64::_mm256_sub_epi64(PACKED_2P, state[0]);
            state[0] = x86_64::_mm256_add_epi64(state[0], state[0]);

            state[2] = x86_64::_mm256_slli_epi64::<1>(state[2]);
            state[3] = x86_64::_mm256_slli_epi64::<2>(state[3]);
            state[4] = x86_64::_mm256_slli_epi64::<3>(state[4]);
            state[5] = x86_64::_mm256_slli_epi64::<4>(state[5]);
            state[6] = x86_64::_mm256_slli_epi64::<5>(state[6]);
            state[7] = x86_64::_mm256_slli_epi64::<6>(state[7]);
            state[8] = x86_64::_mm256_slli_epi64::<7>(state[8]);
            state[9] = x86_64::_mm256_slli_epi64::<8>(state[9]);
            state[10] = x86_64::_mm256_slli_epi64::<9>(state[10]);
            state[11] = x86_64::_mm256_slli_epi64::<10>(state[11]);
            state[12] = x86_64::_mm256_slli_epi64::<11>(state[12]);
            state[13] = x86_64::_mm256_slli_epi64::<12>(state[13]);
            state[14] = x86_64::_mm256_slli_epi64::<13>(state[14]);
            state[15] = x86_64::_mm256_slli_epi64::<14>(state[15]);
            state[16] = x86_64::_mm256_slli_epi64::<15>(state[16]);
            state[17] = x86_64::_mm256_slli_epi64::<16>(state[17]);
            state[18] = x86_64::_mm256_slli_epi64::<17>(state[18]);
            state[19] = x86_64::_mm256_slli_epi64::<18>(state[19]);
            state[20] = x86_64::_mm256_slli_epi64::<19>(state[20]);
            state[21] = x86_64::_mm256_slli_epi64::<20>(state[21]);
            state[22] = x86_64::_mm256_slli_epi64::<21>(state[22]);
            state[23] = x86_64::_mm256_slli_epi64::<23>(state[23]);
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use p3_field::AbstractField;
    use p3_poseidon2::Poseidon2;
    use p3_symmetric::Permutation;
    use rand::Rng;

    use crate::{
        KoalaBear, PackedKoalaBearAVX2, Poseidon2ExternalLayerKoalaBear,
        Poseidon2InternalLayerKoalaBear,
    };

    type F = KoalaBear;
    const D: u64 = 3;
    type Perm16 = Poseidon2<
        F,
        Poseidon2ExternalLayerKoalaBear<16>,
        Poseidon2InternalLayerKoalaBear<16>,
        16,
        D,
    >;
    type Perm24 = Poseidon2<
        F,
        Poseidon2ExternalLayerKoalaBear<24>,
        Poseidon2InternalLayerKoalaBear<24>,
        24,
        D,
    >;

    /// Test that the output is the same as the scalar version on a random input.
    #[test]
    fn test_avx2_poseidon2_width_16_0_constants() {
        let mut rng = rand::thread_rng();

        // Our Poseidon2 implementation.
        let poseidon2 = Perm16::new(
            [
                vec![[KoalaBear::zero(); 16]; 1],
                vec![[KoalaBear::zero(); 16]; 1],
            ],
            Poseidon2ExternalLayerKoalaBear::default(),
            vec![KoalaBear::zero(); 1],
            Poseidon2InternalLayerKoalaBear::default(),
        );

        let input: [F; 16] = rng.gen();

        let mut expected = input;
        poseidon2.permute_mut(&mut expected);

        let mut avx2_input = input.map(PackedKoalaBearAVX2::from_f);
        poseidon2.permute_mut(&mut avx2_input);

        let avx2_output = avx2_input.map(|x| x.0[0]);

        assert_eq!(avx2_output, expected);
    }

    /// Test that the output is the same as the scalar version on a random input.
    #[test]
    fn test_avx2_poseidon2_width_16() {
        let mut rng = rand::thread_rng();

        // Our Poseidon2 implementation.
        let poseidon2 = Perm16::new_from_rng_128(
            Poseidon2ExternalLayerKoalaBear::default(),
            Poseidon2InternalLayerKoalaBear::default(),
            &mut rng,
        );

        let input: [F; 16] = rng.gen();

        let mut expected = input;
        poseidon2.permute_mut(&mut expected);

        let mut avx2_input = input.map(PackedKoalaBearAVX2::from_f);
        poseidon2.permute_mut(&mut avx2_input);

        let avx2_output = avx2_input.map(|x| x.0[0]);

        assert_eq!(avx2_output, expected);
    }

    /// Test that the output is the same as the scalar version on a random input.
    #[test]
    fn test_avx2_poseidon2_width_24() {
        let mut rng = rand::thread_rng();

        // Our Poseidon2 implementation.
        let poseidon2 = Perm24::new_from_rng_128(
            Poseidon2ExternalLayerKoalaBear::default(),
            Poseidon2InternalLayerKoalaBear::default(),
            &mut rng,
        );

        let input: [F; 24] = rng.gen();

        let mut expected = input;
        poseidon2.permute_mut(&mut expected);

        let mut avx2_input = input.map(PackedKoalaBearAVX2::from_f);
        poseidon2.permute_mut(&mut avx2_input);

        let avx2_output = avx2_input.map(|x| x.0[0]);

        assert_eq!(avx2_output, expected);
    }
}
