use core::arch::x86_64::__m512i;

use p3_monty_31::{
    mul_neg_2exp_neg_8_avx512, mul_neg_2exp_neg_n_avx512, mul_neg_2exp_neg_two_adicity_avx512,
    InternalLayerParametersAVX512,
};

use crate::{BabyBearInternalLayerParameters, BabyBearParameters};

impl InternalLayerParametersAVX512<BabyBearParameters, 16> for BabyBearInternalLayerParameters {
    type ArrayLike = [__m512i; 15];

    /// For the BabyBear field and width 16 we multiply by the diagonal matrix:
    /// D = [-2, 1, 2, 1/2, 3, 4, -1/2, -3, -4, 1/2^8, 1/4, 1/8, 1/2^27, -1/2^8, -1/16, -1/2^27].
    /// The inputs must be in canonical form, otherwise the result is undefined.
    /// Even when the inputs are in canonical form, we make no guarantees on the output except that, provided
    /// the output is piped directly into add_sum the vector will be modified such that x[i] = D[i]*x[i] + sum.
    #[inline(always)]
    unsafe fn diagonal_mul_remainder(input: &mut [__m512i; 15]) {
        // As far as we know this is optimal in that it need the fewest instructions to perform all of these
        // multiplications. (Note that -1, 0 are not allowed on the diagonal for technical reasons).
        // If there exist other numbers b for which x*b mod P can be computed quickly this diagonal can be updated.

        // This following 4 muls (from input[8] to input[11]) output the negative of what we want.
        // This will be handled in add_sum.

        // input[8] -> sum + input[8]/2**8
        input[8] = mul_neg_2exp_neg_8_avx512::<BabyBearParameters, 19>(input[8]);

        // input[9] -> sum + input[9]/2**2
        input[9] = mul_neg_2exp_neg_n_avx512::<BabyBearParameters, 2, 25>(input[9]);

        // input[10] -> sum + input[10]/2**3
        input[10] = mul_neg_2exp_neg_n_avx512::<BabyBearParameters, 3, 24>(input[10]);

        // input[11] -> sum + input[11]/2**27
        input[11] = mul_neg_2exp_neg_two_adicity_avx512::<BabyBearParameters, 27, 4>(input[11]);

        // The remaining muls output the correct value again.

        // input[12] -> sum - input[12]/2**8
        input[12] = mul_neg_2exp_neg_8_avx512::<BabyBearParameters, 19>(input[12]);

        // input[13] -> sum - input[13]/2**4
        input[13] = mul_neg_2exp_neg_n_avx512::<BabyBearParameters, 4, 23>(input[13]);

        // input[14] -> sum - input[14]/2**27
        input[14] = mul_neg_2exp_neg_two_adicity_avx512::<BabyBearParameters, 27, 4>(input[14]);
    }

    /// There are 4 positive inverse powers of two after the 4: 1/2^8, 1/4, 1/8, 1/2^27.
    const NUM_POS: usize = 4;
}

impl InternalLayerParametersAVX512<BabyBearParameters, 24> for BabyBearInternalLayerParameters {
    type ArrayLike = [__m512i; 23];

    /// For the BabyBear field and width 24 we multiply by the diagonal matrix:
    /// D = [-2, 1, 2, 1/2, 3, 4, -1/2, -3, -4, 1/2^8, 1/4, 1/8, 1/16, 1/2^7, 1/2^9, 1/2^27, -1/2^8, -1/4, -1/8, -1/16, -1/32, -1/64, -1/2^7, -1/2^27]
    /// The inputs must be in canonical form, otherwise the result is undefined.
    /// Even when the inputs are in canonical form, we make no guarantees on the output except that, provided
    /// the output is piped directly into add_sum, the vector will be modified such that x[i] = D[i]*x[i] + sum.
    #[inline(always)]
    unsafe fn diagonal_mul_remainder(input: &mut [__m512i; 23]) {
        // As far as we know this is optimal in that it need the fewest instructions to perform all of these
        // multiplications. (Note that -1, 0 are not allowed on the diagonal for technical reasons).
        // If there exist other number b for which x*b mod P can be computed quickly this diagonal can be updated.

        // This following 7 muls (from input[8] to input[14]) output the negative of what we want.
        // This will be handled in add_sum.

        // input[8] -> sum + input[8]/2**8
        input[8] = mul_neg_2exp_neg_8_avx512::<BabyBearParameters, 19>(input[8]);

        // input[9] -> sum + input[9]/2**2
        input[9] = mul_neg_2exp_neg_n_avx512::<BabyBearParameters, 2, 25>(input[9]);

        // input[10] -> sum + input[10]/2**3
        input[10] = mul_neg_2exp_neg_n_avx512::<BabyBearParameters, 3, 24>(input[10]);

        // input[11] -> sum + input[11]/2**4
        input[11] = mul_neg_2exp_neg_n_avx512::<BabyBearParameters, 4, 23>(input[11]);

        // input[12] -> sum + input[12]/2**7
        input[12] = mul_neg_2exp_neg_n_avx512::<BabyBearParameters, 7, 20>(input[12]);

        // input[13] -> sum + input[13]/2**9
        input[13] = mul_neg_2exp_neg_n_avx512::<BabyBearParameters, 9, 18>(input[13]);

        // input[14] -> sum + input[14]/2**27
        input[14] = mul_neg_2exp_neg_two_adicity_avx512::<BabyBearParameters, 27, 4>(input[14]);

        // The remaining muls output the correct value again.

        // input[15] -> sum - input[15]/2**8
        input[15] = mul_neg_2exp_neg_8_avx512::<BabyBearParameters, 19>(input[15]);

        // input[16] -> sum - input[16]/2**2
        input[16] = mul_neg_2exp_neg_n_avx512::<BabyBearParameters, 2, 25>(input[16]);

        // input[17] -> sum - input[17]/2**3
        input[17] = mul_neg_2exp_neg_n_avx512::<BabyBearParameters, 3, 24>(input[17]);

        // input[18] -> sum - input[18]/2**4
        input[18] = mul_neg_2exp_neg_n_avx512::<BabyBearParameters, 4, 23>(input[18]);

        // input[19] -> sum - input[19]/2**5
        input[19] = mul_neg_2exp_neg_n_avx512::<BabyBearParameters, 5, 22>(input[19]);

        // input[20] -> sum - input[20]/2**6
        input[20] = mul_neg_2exp_neg_n_avx512::<BabyBearParameters, 6, 21>(input[20]);

        // input[21] -> sum - input[21]/2**7
        input[21] = mul_neg_2exp_neg_n_avx512::<BabyBearParameters, 7, 20>(input[21]);

        // input[22] -> sum - input[22]/2**27
        input[22] = mul_neg_2exp_neg_two_adicity_avx512::<BabyBearParameters, 27, 4>(input[22]);
    }

    /// There are 7 positive inverse powers of two after the 4: 1/2^8, 1/4, 1/8, 1/16, 1/2^7, 1/2^9, 1/2^27,.
    const NUM_POS: usize = 7;
}

#[cfg(test)]
mod tests {
    use p3_field::FieldAlgebra;
    use p3_symmetric::Permutation;
    use rand::Rng;

    use crate::{BabyBear, PackedBabyBearAVX512, Poseidon2BabyBear};

    type F = BabyBear;
    type Perm16 = Poseidon2BabyBear<16>;
    type Perm24 = Poseidon2BabyBear<24>;

    /// Test that the output is the same as the scalar version on a random input.
    #[test]
    fn test_avx512_poseidon2_width_16() {
        let mut rng = rand::thread_rng();

        // Our Poseidon2 implementation.
        let poseidon2 = Perm16::new_from_rng_128(&mut rng);

        let input: [F; 16] = rng.gen();

        let mut expected = input;
        poseidon2.permute_mut(&mut expected);

        let mut avx512_input = input.map(Into::<PackedBabyBearAVX512>::into);
        poseidon2.permute_mut(&mut avx512_input);

        let avx512_output = avx512_input.map(|x| x.0[0]);

        assert_eq!(avx512_output, expected);
    }

    /// Test that the output is the same as the scalar version on a random input.
    #[test]
    fn test_avx512_poseidon2_width_24() {
        let mut rng = rand::thread_rng();

        // Our Poseidon2 implementation.
        let poseidon2 = Perm24::new_from_rng_128(&mut rng);

        let input: [F; 24] = rng.gen();

        let mut expected = input;
        poseidon2.permute_mut(&mut expected);

        let mut avx512_input = input.map(Into::<PackedBabyBearAVX512>::into);
        poseidon2.permute_mut(&mut avx512_input);

        let avx512_output = avx512_input.map(|x| x.0[0]);

        assert_eq!(avx512_output, expected);
    }
}
