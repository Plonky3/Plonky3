use core::arch::aarch64::uint32x4_t;

use p3_monty_31::{
    InternalLayerParametersNeon, mul_neg_2exp_neg_8_neon, mul_neg_2exp_neg_n_neon,
    mul_neg_2exp_neg_two_adicity_neon,
};

use crate::{BabyBearInternalLayerParameters, BabyBearParameters};

impl InternalLayerParametersNeon<BabyBearParameters, 16> for BabyBearInternalLayerParameters {
    type ArrayLike = [uint32x4_t; 15];

    /// For the BabyBear field and width 16 we multiply by the diagonal matrix:
    /// D = [-2, 1, 2, 1/2, 3, 4, -1/2, -3, -4, 1/2^8, 1/4, 1/8, 1/2^27, -1/2^8, -1/16, -1/2^27].
    ///
    /// The inputs must be in canonical form, otherwise the result is undefined.
    ///
    /// Even when the inputs are in canonical form, we make no guarantees on the output except that, provided
    /// the output is piped directly into add_sum the vector will be modified such that x[i] = D[i]*x[i] + sum.
    #[inline(always)]
    unsafe fn diagonal_mul_remainder(input: &mut [uint32x4_t; 15]) {
        unsafe {
            // These first 4 muls output the negative of the desired value and are corrected in `add_sum`.

            // input[8] -> 1/2^8
            input[8] = mul_neg_2exp_neg_8_neon::<BabyBearParameters, 19>(input[8]);
            // input[9] -> 1/4
            input[9] = mul_neg_2exp_neg_n_neon::<BabyBearParameters, 2, 25>(input[9]);
            // input[10] -> 1/8
            input[10] = mul_neg_2exp_neg_n_neon::<BabyBearParameters, 3, 24>(input[10]);
            // input[11] -> 1/2^27
            input[11] = mul_neg_2exp_neg_two_adicity_neon::<BabyBearParameters, 27, 4>(input[11]);

            // These remaining muls are for negative coefficients and are correct as is.

            // input[12] -> -1/2^8
            input[12] = mul_neg_2exp_neg_8_neon::<BabyBearParameters, 19>(input[12]);
            // input[13] -> -1/16
            input[13] = mul_neg_2exp_neg_n_neon::<BabyBearParameters, 4, 23>(input[13]);
            // input[14] -> -1/2^27
            input[14] = mul_neg_2exp_neg_two_adicity_neon::<BabyBearParameters, 27, 4>(input[14]);
        }
    }

    /// There are 4 positive inverse powers of two after the 4: 1/2^8, 1/4, 1/8, 1/2^27.
    const NUM_POS: usize = 4;
}

impl InternalLayerParametersNeon<BabyBearParameters, 24> for BabyBearInternalLayerParameters {
    type ArrayLike = [uint32x4_t; 23];

    /// For the BabyBear field and width 24 we multiply by the diagonal matrix:
    /// D = [-2, 1, 2, 1/2, 3, 4, -1/2, -3, -4, 1/2^8, 1/4, 1/8, 1/16, 1/2^7, 1/2^9, 1/2^27, -1/2^8, -1/4, -1/8, -1/16, -1/32, -1/64, -1/2^7, -1/2^27]
    ///
    /// The inputs must be in canonical form, otherwise the result is undefined.
    ///
    /// Even when the inputs are in canonical form, we make no guarantees on the output except that, provided
    /// the output is piped directly into add_sum, the vector will be modified such that x[i] = D[i]*x[i] + sum.
    #[inline(always)]
    unsafe fn diagonal_mul_remainder(input: &mut [uint32x4_t; 23]) {
        unsafe {
            // Positive coefficients (corrected in add_sum)

            // input[8] -> sum + input[8]/2**8
            input[8] = mul_neg_2exp_neg_8_neon::<BabyBearParameters, 19>(input[8]);
            // input[9] -> sum + input[9]/2**2
            input[9] = mul_neg_2exp_neg_n_neon::<BabyBearParameters, 2, 25>(input[9]);
            // input[10] -> sum + input[10]/2**3
            input[10] = mul_neg_2exp_neg_n_neon::<BabyBearParameters, 3, 24>(input[10]);
            // input[11] -> sum + input[11]/2**4
            input[11] = mul_neg_2exp_neg_n_neon::<BabyBearParameters, 4, 23>(input[11]);
            // input[12] -> sum + input[12]/2**7
            input[12] = mul_neg_2exp_neg_n_neon::<BabyBearParameters, 7, 20>(input[12]);
            // input[13] -> sum + input[13]/2**9
            input[13] = mul_neg_2exp_neg_n_neon::<BabyBearParameters, 9, 18>(input[13]);
            // input[14] -> sum + input[14]/2**27
            input[14] = mul_neg_2exp_neg_two_adicity_neon::<BabyBearParameters, 27, 4>(input[14]);

            // Negative coefficients

            // input[15] -> sum - input[15]/2**8
            input[15] = mul_neg_2exp_neg_8_neon::<BabyBearParameters, 19>(input[15]);
            // input[16] -> sum - input[16]/2**2
            input[16] = mul_neg_2exp_neg_n_neon::<BabyBearParameters, 2, 25>(input[16]);
            // input[17] -> sum - input[17]/2**3
            input[17] = mul_neg_2exp_neg_n_neon::<BabyBearParameters, 3, 24>(input[17]);
            // input[18] -> sum - input[18]/2**4
            input[18] = mul_neg_2exp_neg_n_neon::<BabyBearParameters, 4, 23>(input[18]);
            // input[19] -> sum - input[19]/2**5
            input[19] = mul_neg_2exp_neg_n_neon::<BabyBearParameters, 5, 22>(input[19]);
            // input[20] -> sum - input[20]/2**6
            input[20] = mul_neg_2exp_neg_n_neon::<BabyBearParameters, 6, 21>(input[20]);
            // input[21] -> sum - input[21]/2**7
            input[21] = mul_neg_2exp_neg_n_neon::<BabyBearParameters, 7, 20>(input[21]);
            // input[22] -> sum - input[22]/2**27
            input[22] = mul_neg_2exp_neg_two_adicity_neon::<BabyBearParameters, 27, 4>(input[22]);
        }
    }

    /// There are 7 positive inverse powers of two after the 4: 1/2^8, 1/4, 1/8, 1/16, 1/2^7, 1/2^9, 1/2^27,.
    const NUM_POS: usize = 7;
}

#[cfg(test)]
mod tests {
    use p3_symmetric::Permutation;
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    use crate::{BabyBear, PackedBabyBearNeon, Poseidon2BabyBear};

    type F = BabyBear;
    type Perm16 = Poseidon2BabyBear<16>;
    type Perm24 = Poseidon2BabyBear<24>;

    /// Test that the output is the same as the scalar version on a random input.
    #[test]
    fn test_avx2_poseidon2_width_16() {
        let mut rng = SmallRng::seed_from_u64(1);

        // Our Poseidon2 implementation.
        let poseidon2 = Perm16::new_from_rng_128(&mut rng);

        let input: [F; 16] = rng.random();

        let mut expected = input;
        poseidon2.permute_mut(&mut expected);

        let mut avx2_input = input.map(Into::<PackedBabyBearNeon>::into);
        poseidon2.permute_mut(&mut avx2_input);

        let avx2_output = avx2_input.map(|x| x.0[0]);

        assert_eq!(avx2_output, expected);
    }

    /// Test that the output is the same as the scalar version on a random input.
    #[test]
    fn test_avx2_poseidon2_width_24() {
        let mut rng = SmallRng::seed_from_u64(1);

        // Our Poseidon2 implementation.
        let poseidon2 = Perm24::new_from_rng_128(&mut rng);

        let input: [F; 24] = rng.random();

        let mut expected = input;
        poseidon2.permute_mut(&mut expected);

        let mut avx2_input = input.map(Into::<PackedBabyBearNeon>::into);
        poseidon2.permute_mut(&mut avx2_input);

        let avx2_output = avx2_input.map(|x| x.0[0]);

        assert_eq!(avx2_output, expected);
    }
}
