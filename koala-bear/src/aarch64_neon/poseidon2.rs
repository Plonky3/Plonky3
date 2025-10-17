use core::arch::aarch64::uint32x4_t;

use p3_monty_31::{
    InternalLayerParametersNeon, mul_2exp_neg_n_neon, mul_2exp_neg_two_adicity_neon,
};

use crate::{KoalaBearInternalLayerParameters, KoalaBearParameters};

impl InternalLayerParametersNeon<KoalaBearParameters, 16> for KoalaBearInternalLayerParameters {
    type ArrayLike = [uint32x4_t; 15];

    /// For the KoalaBear field and width 16 we multiply by the diagonal matrix:
    /// D = [-2, 1, 2, 1/2, 3, 4, -1/2, -3, -4, 1/2^8, 1/8, 1/2^24, -1/2^8, -1/8, -1/16, -1/2^24].
    ///
    /// The inputs must be in canonical form, otherwise the result is undefined.
    ///
    /// Even when the inputs are in canonical form, we make no guarantees on the output except that, provided
    /// the output is piped directly into add_sum the vector will be modified such that x[i] = D[i]*x[i] + sum.
    #[inline(always)]
    unsafe fn diagonal_mul_remainder(input: &mut [uint32x4_t; 15]) {
        unsafe {
            // These multiplications are for positive coefficients. The results are added to the
            // sum in the `add_sum` function.
            // input[8] -> input[8] / 2^8
            input[8] = mul_2exp_neg_n_neon::<KoalaBearParameters, 8>(input[8]);
            // input[9] -> input[9] / 8
            input[9] = mul_2exp_neg_n_neon::<KoalaBearParameters, 3>(input[9]);
            // input[10] -> input[10] / 2^24
            input[10] = mul_2exp_neg_two_adicity_neon::<KoalaBearParameters, 24, 7>(input[10]);

            // These multiplications are for negative coefficients. We compute the multiplication
            // by the positive value, and the result is later subtracted from the sum in `add_sum`.
            // input[11] -> input[11] / 2^8
            input[11] = mul_2exp_neg_n_neon::<KoalaBearParameters, 8>(input[11]);
            // input[12] -> input[12] / 8
            input[12] = mul_2exp_neg_n_neon::<KoalaBearParameters, 3>(input[12]);
            // input[13] -> input[13] / 16
            input[13] = mul_2exp_neg_n_neon::<KoalaBearParameters, 4>(input[13]);
            // input[14] -> input[14] / 2^24
            input[14] = mul_2exp_neg_two_adicity_neon::<KoalaBearParameters, 24, 7>(input[14]);
        }
    }

    /// There are 3 positive inverse powers of two after the -4: 1/2^8, 1/8, 1/2^24,
    const NUM_POS: usize = 3;
}

impl InternalLayerParametersNeon<KoalaBearParameters, 24> for KoalaBearInternalLayerParameters {
    type ArrayLike = [uint32x4_t; 23];

    /// For the KoalaBear field and width 24 we multiply by the diagonal matrix:
    /// D = [-2, 1, 2, 1/2, 3, 4, -1/2, -3, -4, 1/2^8, 1/4, 1/8, 1/16, 1/32, 1/64, 1/2^24, -1/2^8, -1/8, -1/16, -1/32, -1/64, -1/2^7, -1/2^9, -1/2^24]
    ///
    /// The inputs must be in canonical form, otherwise the result is undefined.
    ///
    /// Even when the inputs are in canonical form, we make no guarantees on the output except that, provided
    /// the output is piped directly into add_sum, the vector will be modified such that x[i] = D[i]*x[i] + sum.
    #[inline(always)]
    unsafe fn diagonal_mul_remainder(input: &mut [uint32x4_t; 23]) {
        unsafe {
            // Positive coefficients
            // input[8] -> sum + input[8]/2**8
            input[8] = mul_2exp_neg_n_neon::<KoalaBearParameters, 8>(input[8]);
            // input[9] -> sum + input[9]/2**2
            input[9] = mul_2exp_neg_n_neon::<KoalaBearParameters, 2>(input[9]);
            // input[10] -> sum + input[10]/2**3
            input[10] = mul_2exp_neg_n_neon::<KoalaBearParameters, 3>(input[10]);
            // input[11] -> sum + input[11]/2**4
            input[11] = mul_2exp_neg_n_neon::<KoalaBearParameters, 4>(input[11]);
            // input[12] -> sum + input[12]/2**5
            input[12] = mul_2exp_neg_n_neon::<KoalaBearParameters, 5>(input[12]);
            // input[13] -> sum + input[13]/2**6
            input[13] = mul_2exp_neg_n_neon::<KoalaBearParameters, 6>(input[13]);
            // input[14] -> sum + input[14]/2**24
            input[14] = mul_2exp_neg_two_adicity_neon::<KoalaBearParameters, 24, 7>(input[14]);

            // Negative coefficients
            // input[15] -> sum - input[15]/2**8
            input[15] = mul_2exp_neg_n_neon::<KoalaBearParameters, 8>(input[15]);
            // input[16] -> sum - input[16]/2**3
            input[16] = mul_2exp_neg_n_neon::<KoalaBearParameters, 3>(input[16]);
            // input[17] -> sum - input[17]/2**4
            input[17] = mul_2exp_neg_n_neon::<KoalaBearParameters, 4>(input[17]);
            // input[18] -> sum - input[18]/2**5
            input[18] = mul_2exp_neg_n_neon::<KoalaBearParameters, 5>(input[18]);
            // input[19] -> sum - input[19]/2**6
            input[19] = mul_2exp_neg_n_neon::<KoalaBearParameters, 6>(input[19]);
            // input[20] -> sum - input[20]/2**7
            input[20] = mul_2exp_neg_n_neon::<KoalaBearParameters, 7>(input[20]);
            // input[21] -> sum - input[21]/2**9
            input[21] = mul_2exp_neg_n_neon::<KoalaBearParameters, 9>(input[21]);
            // input[22] -> sum - input[22]/2**24
            input[22] = mul_2exp_neg_two_adicity_neon::<KoalaBearParameters, 24, 7>(input[22]);
        }
    }

    /// There are 7 positive inverse powers of two after the -4: 1/2^8, 1/4, 1/8, 1/16, 1/32, 1/64, 1/2^24;
    const NUM_POS: usize = 7;
}

#[cfg(test)]
mod tests {
    use p3_field::PrimeCharacteristicRing;
    use p3_symmetric::Permutation;
    use proptest::prelude::*;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    use crate::{KoalaBear, PackedKoalaBearNeon, Poseidon2KoalaBear};

    type F = KoalaBear;
    type Perm16 = Poseidon2KoalaBear<16>;
    type Perm24 = Poseidon2KoalaBear<24>;

    /// A proptest strategy to generate an arbitrary field element.
    fn arb_f() -> impl Strategy<Value = F> {
        prop::num::u32::ANY.prop_map(F::from_u32)
    }

    proptest! {

        #[test]
        fn vectorized_permutation_matches_scalar_for_width_16(
            input in prop::array::uniform16(arb_f())
        ) {
            // Use a fixed seed for the Poseidon2 constants for reproducibility.
            let mut rng = SmallRng::seed_from_u64(1);
            let poseidon2 = Perm16::new_from_rng_128(&mut rng);

            // Calculate the expected result using the scalar implementation.
            let mut expected = input;
            poseidon2.permute_mut(&mut expected);

            // Calculate the actual result using the NEON implementation.
            // First, map the scalar inputs into packed NEON vectors.
            let mut neon_input = input.map(Into::<PackedKoalaBearNeon>::into);
            poseidon2.permute_mut(&mut neon_input);

            // Finally, unpack the NEON vectors back into scalar values.
            let neon_output = neon_input.map(|x| x.0[0]);

            // 3. Assert that the results are identical.
            prop_assert_eq!(neon_output, expected, "NEON implementation did not match scalar reference");
        }

        #[test]
        fn vectorized_permutation_matches_scalar_for_width_24(
            input in prop::array::uniform24(arb_f())
        ) {
            let mut rng = SmallRng::seed_from_u64(1);
            let poseidon2 = Perm24::new_from_rng_128(&mut rng);

            // Calculate expected (scalar) result.
            let mut expected = input;
            poseidon2.permute_mut(&mut expected);

            // Calculate actual (NEON) result.
            let mut neon_input = input.map(Into::<PackedKoalaBearNeon>::into);
            poseidon2.permute_mut(&mut neon_input);
            let neon_output = neon_input.map(|x| x.0[0]);

            // Assert equality.
            prop_assert_eq!(neon_output, expected, "NEON implementation did not match scalar reference");
        }
    }
}
