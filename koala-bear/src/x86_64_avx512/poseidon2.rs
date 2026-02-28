use core::arch::x86_64::{self, __m512i};

use p3_monty_31::{
    InternalLayerParametersAVX512, mul_neg_2exp_neg_8_avx512, mul_neg_2exp_neg_n_avx512,
    mul_neg_2exp_neg_two_adicity_avx512,
};

use crate::{KoalaBearInternalLayerParameters, KoalaBearParameters};

/// Multiply a vector of KoalaBear field elements in canonical form by -2^{-16}.
///
/// The generic `mul_neg_2exp_neg_n_avx512` cannot be used with N=16 because `_mm512_madd_epi16`
/// interprets inputs as signed 16-bit integers, and the low 16 bits of the input can exceed
/// the signed i16 range (max 32767). We use `_mm512_mullo_epi32` (32-bit multiply) instead.
///
/// # Safety
///
/// Input must be given in canonical form.
/// Output may not be in canonical form but will lie in [0, P].
#[inline(always)]
unsafe fn mul_neg_2exp_neg_16_avx512(input: __m512i) -> __m512i {
    unsafe {
        // KoalaBear prime: P = 127 * 2^24 + 1 = 0x7F000001
        let p = x86_64::_mm512_set1_epi32(0x7F000001_u32 as i32);
        let odd_factor = x86_64::_mm512_set1_epi32(127);
        let mask = x86_64::_mm512_set1_epi32(0xFFFF);

        let hi = x86_64::_mm512_srli_epi32::<16>(input);
        let lo = x86_64::_mm512_and_si512(input, mask);

        // Determine the non-zero values of lo.
        let lo_mask = x86_64::_mm512_test_epi32_mask(input, mask);

        // Use mullo_epi32 instead of madd_epi16 since lo can be up to 2^16-1
        let lo_x_r = x86_64::_mm512_mullo_epi32(lo, odd_factor);

        // When lo = 0, lo_shft = P
        // When lo > 0, lo_shft = 127 * 2^8 * x_lo
        let lo_shft = x86_64::_mm512_mask_slli_epi32::<8>(p, lo_mask, lo_x_r);

        x86_64::_mm512_sub_epi32(lo_shft, hi)
    }
}

impl InternalLayerParametersAVX512<KoalaBearParameters, 16> for KoalaBearInternalLayerParameters {
    type ArrayLike = [__m512i; 15];

    /// For the KoalaBear field and width 16 we multiply by the diagonal matrix:
    /// D = [-2, 1, 2, 1/2, 3, 4, -1/2, -3, -4, 1/2^8, 1/8, 1/2^24, -1/2^8, -1/8, -1/16, -1/2^24].
    /// The inputs must be in canonical form, otherwise the result is undefined.
    /// Even when the inputs are in canonical form, we make no guarantees on the output except that, provided
    /// the output is piped directly into add_sum the vector will be modified such that x[i] = D[i]*x[i] + sum.
    #[inline(always)]
    unsafe fn diagonal_mul_remainder(input: &mut [__m512i; 15]) {
        unsafe {
            // As far as we know this is optimal in that it need the fewest instructions to perform all of these
            // multiplications. (Note that -1, 0 are not allowed on the diagonal for technical reasons).
            // If there exist other number b for which x*b mod P can be computed quickly this diagonal can be updated.

            // This following 3 muls (from input[8] to input[10]) output the negative of what we want.
            // This will be handled in add_sum.

            // input[8]-> sum + input[8]/2^8
            input[8] = mul_neg_2exp_neg_8_avx512::<KoalaBearParameters, 16>(input[8]);

            // input[9] -> sum + input[9]/2^3
            input[9] = mul_neg_2exp_neg_n_avx512::<KoalaBearParameters, 3, 21>(input[9]);

            // input[10] -> sum + input[10]/2^24
            input[10] =
                mul_neg_2exp_neg_two_adicity_avx512::<KoalaBearParameters, 24, 7>(input[10]);

            // The remaining muls output the correct value again.

            // input[11] -> sum - input[11]/2^8
            input[11] = mul_neg_2exp_neg_8_avx512::<KoalaBearParameters, 16>(input[11]);

            // input[12] -> sum - input[12]/2^3
            input[12] = mul_neg_2exp_neg_n_avx512::<KoalaBearParameters, 3, 21>(input[12]);

            // input[13] -> sum - input[13]/2^4
            input[13] = mul_neg_2exp_neg_n_avx512::<KoalaBearParameters, 4, 20>(input[13]);

            // input[14] -> sum - input[14]/2^24
            input[14] =
                mul_neg_2exp_neg_two_adicity_avx512::<KoalaBearParameters, 24, 7>(input[14]);
        }
    }

    /// There are 3 positive inverse powers of two after the 4: 1/2^8, 1/8, 1/2^24,
    const NUM_POS: usize = 3;
}

impl InternalLayerParametersAVX512<KoalaBearParameters, 24> for KoalaBearInternalLayerParameters {
    type ArrayLike = [__m512i; 23];

    /// For the KoalaBear field and width 24 we multiply by the diagonal matrix:
    /// D = [-2, 1, 2, 1/2, 3, 4, -1/2, -3, -4, 1/2^8, 1/4, 1/8, 1/16, 1/32, 1/64, 1/2^24, -1/2^8, -1/8, -1/16, -1/32, -1/64, -1/2^7, -1/2^9, -1/2^24]
    /// The inputs must be in canonical form, otherwise the result is undefined.
    /// Even when the inputs are in canonical form, we make no guarantees on the output except that, provided
    /// the output is piped directly into add_sum, the vector will be modified such that x[i] = D[i]*x[i] + sum.
    #[inline(always)]
    unsafe fn diagonal_mul_remainder(input: &mut [__m512i; 23]) {
        unsafe {
            // As far as we know this is optimal in that it need the fewest instructions to perform all of these
            // multiplications. (Note that -1, 0 are not allowed on the diagonal for technical reasons).
            // If there exist other number b for which x*b mod P can be computed quickly this diagonal can be updated.

            // This following 7 muls (from input[8] to input[14]) output the negative of what we want.
            // This will be handled in add_sum.

            // input[8] -> sum + input[8]/2^8
            input[8] = mul_neg_2exp_neg_8_avx512::<KoalaBearParameters, 16>(input[8]);

            // input[9] -> sum + input[9]/2^2
            input[9] = mul_neg_2exp_neg_n_avx512::<KoalaBearParameters, 2, 22>(input[9]);

            // input[10] -> sum + input[10]/2^3
            input[10] = mul_neg_2exp_neg_n_avx512::<KoalaBearParameters, 3, 21>(input[10]);

            // input[11] -> sum + input[11]/2^4
            input[11] = mul_neg_2exp_neg_n_avx512::<KoalaBearParameters, 4, 20>(input[11]);

            // input[12] -> sum + input[12]/2^5
            input[12] = mul_neg_2exp_neg_n_avx512::<KoalaBearParameters, 5, 19>(input[12]);

            // input[13] -> sum + input[13]/2^6
            input[13] = mul_neg_2exp_neg_n_avx512::<KoalaBearParameters, 6, 18>(input[13]);

            // input[14] -> sum + input[14]/2^24
            input[14] =
                mul_neg_2exp_neg_two_adicity_avx512::<KoalaBearParameters, 24, 7>(input[14]);

            // The remaining muls output the correct value again.

            // input[15] -> sum - input[15]/2^8
            input[15] = mul_neg_2exp_neg_8_avx512::<KoalaBearParameters, 16>(input[15]);

            // input[16] -> sum - input[16]/2^3
            input[16] = mul_neg_2exp_neg_n_avx512::<KoalaBearParameters, 3, 21>(input[16]);

            // input[17] -> sum - input[17]/2^4
            input[17] = mul_neg_2exp_neg_n_avx512::<KoalaBearParameters, 4, 20>(input[17]);

            // input[18] -> sum - input[18]/2^5
            input[18] = mul_neg_2exp_neg_n_avx512::<KoalaBearParameters, 5, 19>(input[18]);

            // input[19] -> sum - input[19]/2^6
            input[19] = mul_neg_2exp_neg_n_avx512::<KoalaBearParameters, 6, 18>(input[19]);

            // input[20] -> sum - input[20]/2^7
            input[20] = mul_neg_2exp_neg_n_avx512::<KoalaBearParameters, 7, 17>(input[20]);

            // input[21] -> sum - input[21]/2^9
            input[21] = mul_neg_2exp_neg_n_avx512::<KoalaBearParameters, 9, 15>(input[21]);

            // input[22] -> sum - input[22]/2^24
            input[22] =
                mul_neg_2exp_neg_two_adicity_avx512::<KoalaBearParameters, 24, 7>(input[22]);
        }
    }

    /// There are 7 positive inverse powers of two after the 4: 1/2^8, 1/4, 1/8, 1/16, 1/32, 1/64, 1/2^24;
    const NUM_POS: usize = 7;
}

impl InternalLayerParametersAVX512<KoalaBearParameters, 32> for KoalaBearInternalLayerParameters {
    type ArrayLike = [__m512i; 31];

    /// For the KoalaBear field and width 32 we multiply by the diagonal matrix:
    /// D = [-2, 1, 2, 1/2, 3, 4, -1/2, -3, -4,
    ///      1/2^8, 1/4, 1/8, 1/16, 1/32, 1/64, 1/2^10, 1/2^12, 1/2^14, 1/2^16, 1/2^24,
    ///      -1/2^8, -1/8, -1/16, -1/32, -1/64, -1/2^7, -1/2^9, -1/2^10, -1/2^12, -1/2^14, -1/2^16, -1/2^24]
    /// The inputs must be in canonical form, otherwise the result is undefined.
    /// Even when the inputs are in canonical form, we make no guarantees on the output except that, provided
    /// the output is piped directly into add_sum, the vector will be modified such that x[i] = D[i]*x[i] + sum.
    #[inline(always)]
    unsafe fn diagonal_mul_remainder(input: &mut [__m512i; 31]) {
        unsafe {
            // The following 11 muls (input[8] to input[18]) output the negative of what we want.
            // This will be handled in add_sum.

            // input[8] -> input[8] / 2^8
            input[8] = mul_neg_2exp_neg_8_avx512::<KoalaBearParameters, 16>(input[8]);
            // input[9] -> input[9] / 2^2
            input[9] = mul_neg_2exp_neg_n_avx512::<KoalaBearParameters, 2, 22>(input[9]);
            // input[10] -> input[10] / 2^3
            input[10] = mul_neg_2exp_neg_n_avx512::<KoalaBearParameters, 3, 21>(input[10]);
            // input[11] -> input[11] / 2^4
            input[11] = mul_neg_2exp_neg_n_avx512::<KoalaBearParameters, 4, 20>(input[11]);
            // input[12] -> input[12] / 2^5
            input[12] = mul_neg_2exp_neg_n_avx512::<KoalaBearParameters, 5, 19>(input[12]);
            // input[13] -> input[13] / 2^6
            input[13] = mul_neg_2exp_neg_n_avx512::<KoalaBearParameters, 6, 18>(input[13]);
            // input[14] -> input[14] / 2^10
            input[14] = mul_neg_2exp_neg_n_avx512::<KoalaBearParameters, 10, 14>(input[14]);
            // input[15] -> input[15] / 2^12
            input[15] = mul_neg_2exp_neg_n_avx512::<KoalaBearParameters, 12, 12>(input[15]);
            // input[16] -> input[16] / 2^14
            input[16] = mul_neg_2exp_neg_n_avx512::<KoalaBearParameters, 14, 10>(input[16]);
            // input[17] -> input[17] / 2^16
            input[17] = mul_neg_2exp_neg_16_avx512(input[17]);
            // input[18] -> input[18] / 2^24
            input[18] =
                mul_neg_2exp_neg_two_adicity_avx512::<KoalaBearParameters, 24, 7>(input[18]);

            // The remaining muls output the correct value again.

            // input[19] -> -input[19] / 2^8
            input[19] = mul_neg_2exp_neg_8_avx512::<KoalaBearParameters, 16>(input[19]);
            // input[20] -> -input[20] / 2^3
            input[20] = mul_neg_2exp_neg_n_avx512::<KoalaBearParameters, 3, 21>(input[20]);
            // input[21] -> -input[21] / 2^4
            input[21] = mul_neg_2exp_neg_n_avx512::<KoalaBearParameters, 4, 20>(input[21]);
            // input[22] -> -input[22] / 2^5
            input[22] = mul_neg_2exp_neg_n_avx512::<KoalaBearParameters, 5, 19>(input[22]);
            // input[23] -> -input[23] / 2^6
            input[23] = mul_neg_2exp_neg_n_avx512::<KoalaBearParameters, 6, 18>(input[23]);
            // input[24] -> -input[24] / 2^7
            input[24] = mul_neg_2exp_neg_n_avx512::<KoalaBearParameters, 7, 17>(input[24]);
            // input[25] -> -input[25] / 2^9
            input[25] = mul_neg_2exp_neg_n_avx512::<KoalaBearParameters, 9, 15>(input[25]);
            // input[26] -> -input[26] / 2^10
            input[26] = mul_neg_2exp_neg_n_avx512::<KoalaBearParameters, 10, 14>(input[26]);
            // input[27] -> -input[27] / 2^12
            input[27] = mul_neg_2exp_neg_n_avx512::<KoalaBearParameters, 12, 12>(input[27]);
            // input[28] -> -input[28] / 2^14
            input[28] = mul_neg_2exp_neg_n_avx512::<KoalaBearParameters, 14, 10>(input[28]);
            // input[29] -> -input[29] / 2^16
            input[29] = mul_neg_2exp_neg_16_avx512(input[29]);
            // input[30] -> -input[30] / 2^24
            input[30] =
                mul_neg_2exp_neg_two_adicity_avx512::<KoalaBearParameters, 24, 7>(input[30]);
        }
    }

    /// There are 11 positive inverse powers of two after the -4.
    const NUM_POS: usize = 11;
}

#[cfg(test)]
mod tests {
    use p3_symmetric::Permutation;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use crate::{KoalaBear, PackedKoalaBearAVX512, Poseidon2KoalaBear};

    type F = KoalaBear;
    type Perm16 = Poseidon2KoalaBear<16>;
    type Perm24 = Poseidon2KoalaBear<24>;

    /// Test that the output is the same as the scalar version on a random input.
    #[test]
    fn test_avx512_poseidon2_width_16() {
        let mut rng = SmallRng::seed_from_u64(1);

        // Our Poseidon2 implementation.
        let poseidon2 = Perm16::new_from_rng_128(&mut rng);

        let input: [F; 16] = rng.random();

        let mut expected = input;
        poseidon2.permute_mut(&mut expected);

        let mut avx512_input = input.map(Into::<PackedKoalaBearAVX512>::into);
        poseidon2.permute_mut(&mut avx512_input);

        let avx512_output = avx512_input.map(|x| x.0[0]);

        assert_eq!(avx512_output, expected);
    }

    /// Test that the output is the same as the scalar version on a random input.
    #[test]
    fn test_avx512_poseidon2_width_24() {
        let mut rng = SmallRng::seed_from_u64(1);

        // Our Poseidon2 implementation.
        let poseidon2 = Perm24::new_from_rng_128(&mut rng);

        let input: [F; 24] = rng.random();

        let mut expected = input;
        poseidon2.permute_mut(&mut expected);

        let mut avx512_input = input.map(Into::<PackedKoalaBearAVX512>::into);
        poseidon2.permute_mut(&mut avx512_input);

        let avx512_output = avx512_input.map(|x| x.0[0]);

        assert_eq!(avx512_output, expected);
    }

    /// Test that the output is the same as the scalar version on a random input.
    #[test]
    fn test_avx512_poseidon2_width_32() {
        let mut rng = SmallRng::seed_from_u64(1);

        // Our Poseidon2 implementation.
        let poseidon2 = Poseidon2KoalaBear::<32>::new_from_rng_128(&mut rng);

        let input: [F; 32] = rng.random();

        let mut expected = input;
        poseidon2.permute_mut(&mut expected);

        let mut avx512_input = input.map(Into::<PackedKoalaBearAVX512>::into);
        poseidon2.permute_mut(&mut avx512_input);

        let avx512_output = avx512_input.map(|x| x.0[0]);

        assert_eq!(avx512_output, expected);
    }
}
