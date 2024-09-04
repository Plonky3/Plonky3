use core::{
    arch::x86_64::{self, __m256i},
    mem::transmute,
};

use p3_monty_31::{add, sub, InternalLayerParametersAVX2, MontyParameters};

use crate::{KoalaBearInternalLayerParameters, KoalaBearParameters};

// mul_inv2:
//     vpand    least_bit, val, ONE
//     vpsrld   t, val, 1
//     vpsignd  maybe_half, HALF, least_bit
//     vpaddd   res, t, maybe_half
//     TP: 1.33
#[inline(always)]
fn halve(input: __m256i) -> __m256i {
    unsafe {
        const ONE: __m256i = unsafe { transmute([1; 8]) };
        const HALF: __m256i = unsafe { transmute([(KoalaBearParameters::PRIME + 1) / 2; 8]) };

        let least_bit = x86_64::_mm256_and_si256(input, ONE);
        let t = x86_64::_mm256_srli_epi32::<1>(input);
        let maybe_half = x86_64::_mm256_sign_epi32(HALF, least_bit);
        x86_64::_mm256_add_epi32(t, maybe_half)
    }
}

// lhs > 0, -P < rhs < P
// signed_add:
//     vpsignd  pos_neg_P,  neg_P, rhs
//     vpaddd   sum,        lhs,   rhs
//     vpaddd   sum_corr,   sum,   pos_neg_P
//     vpminud  res,        sum,   sum_corr
#[inline(always)]
fn signed_add(lhs: __m256i, rhs: __m256i) -> __m256i {
    unsafe {
        const NEG_P: __m256i = unsafe { transmute([-(KoalaBearParameters::PRIME as i32); 8]) };

        let pos_neg_p = x86_64::_mm256_sign_epi32(NEG_P, rhs);
        let sum = x86_64::_mm256_add_epi32(lhs, rhs);
        let sum_corr = x86_64::_mm256_add_epi32(sum, pos_neg_p);
        x86_64::_mm256_min_epu32(sum, sum_corr)
    }
}

// mul_2powneg8:
//     vpsrld      hi, val, 8
//     vpmaddubsw  lo, val, bcast32(7fh)
//     vpslld      lo, lo, 16
//     vpsubd      t, hi, lo
#[inline(always)]
fn mul_2_exp_neg_8(input: __m256i) -> __m256i {
    unsafe {
        const ONE_TWENTY_SEVEN: __m256i = unsafe { transmute([127; 8]) };
        let hi = x86_64::_mm256_srli_epi32::<8>(input);
        let lo = x86_64::_mm256_maddubs_epi16(input, ONE_TWENTY_SEVEN);
        let lo_shft = x86_64::_mm256_slli_epi32::<16>(lo);
        x86_64::_mm256_sub_epi32(hi, lo_shft)
    }
}

// mul_neg2powneg8:
//     vpsrld      hi, val, 8
//     vpmaddubsw  lo, val, bcast32(7fh)
//     vpslld      lo, lo, 16
//     vpsubd      t, lo, hi
#[inline(always)]
fn mul_neg_2_exp_neg_8(input: __m256i) -> __m256i {
    unsafe {
        const ONE_TWENTY_SEVEN: __m256i = unsafe { transmute([127; 8]) };
        let hi = x86_64::_mm256_srli_epi32::<8>(input);
        let lo = x86_64::_mm256_maddubs_epi16(input, ONE_TWENTY_SEVEN);
        let lo_shft = x86_64::_mm256_slli_epi32::<16>(lo);
        x86_64::_mm256_sub_epi32(lo_shft, hi)
    }
}

// mul_2pownegn:
//     vpslld      val_hi,     val,        n
//     vpand       val_lo,     val,        2^{n} - 1
//     vpmaddwd    lo_x_127    val_lo,     [[0, 127]; 4]
//     vpslld      lo          lo_x_127    24 - n
//     vpsubd      res         val_hi      lo
// N + N_PRIME = 24
#[inline(always)]
fn mul_2_exp_neg_n<const N: i32, const N_PRIME: i32>(input: __m256i) -> __m256i {
    unsafe {
        const ONE_TWENTY_SEVEN: __m256i = unsafe { transmute([127; 8]) };

        // Compiler should realize this is a constant.
        // Check this.
        let mask: __m256i = transmute([(1 << N) - 1; 8]);

        let hi = x86_64::_mm256_srli_epi32::<N>(input);
        let val_lo = x86_64::_mm256_and_si256(input, mask);
        let lo = x86_64::_mm256_madd_epi16(val_lo, ONE_TWENTY_SEVEN);
        let lo_shft = x86_64::_mm256_slli_epi32::<N_PRIME>(lo);
        x86_64::_mm256_sub_epi32(hi, lo_shft)
    }
}

// mul_neg2pownegn:
//     vpslld      val_hi,     val,        n
//     vpand       val_lo,     val,        2^{n} - 1
//     vpmaddwd    lo_x_127    val_lo,     [[0, 127]; 4]
//     vpslld      lo          lo_x_127    24 - n
//     vpsubd      res         lo          val_hi
#[inline(always)]
fn mul_neg_2_exp_neg_n<const N: i32, const N_PRIME: i32>(input: __m256i) -> __m256i {
    unsafe {
        const ONE_TWENTY_SEVEN: __m256i = unsafe { transmute([127; 8]) };

        // Compiler should realise this is a constant.
        // Check this.
        let mask: __m256i = transmute([(1 << N) - 1; 8]);

        let hi = x86_64::_mm256_srli_epi32::<N>(input);
        let val_lo = x86_64::_mm256_and_si256(input, mask);
        let lo = x86_64::_mm256_madd_epi16(val_lo, ONE_TWENTY_SEVEN);
        let lo_shft = x86_64::_mm256_slli_epi32::<N_PRIME>(lo);
        x86_64::_mm256_sub_epi32(lo_shft, hi)
    }
}

// mul_2powneg24:
//     vpslld     	val_hi, 	 	val,            24
//     vpand     	val_lo, 	 	val,     	    2^{24} - 1
//     vpslrd  	    val_lo_hi,   	val_lo,         7
//     vpaddd  	    val_hi_plus_lo, val_lo,         val_hi
//     vpsubd		res 		 	val_hi_plus_lo, val_lo_hi,
#[inline(always)]
fn mul_2_exp_neg_24(input: __m256i) -> __m256i {
    unsafe {
        const MASK: __m256i = unsafe { transmute([(1 << 24) - 1; 8]) };

        let hi = x86_64::_mm256_srli_epi32::<24>(input);
        let lo = x86_64::_mm256_and_si256(input, MASK);
        let lo_shft = x86_64::_mm256_slli_epi32::<7>(lo);
        let lo_plus_hi = x86_64::_mm256_add_epi32(lo, hi);
        x86_64::_mm256_sub_epi32(lo_plus_hi, lo_shft)
    }
}

// mul_neg2powneg24:
//     vpslld     	val_hi, 	 	val,        24
//     vpand     	val_lo, 	 	val,     	2^{24} - 1
//     vpslrd  	    val_lo_hi,   	val_lo,     7
//     vpaddd  	    val_hi_plus_lo, val_lo,     val_hi
//     vpsubd		res 		 	val_lo_hi, 	val_hi_plus_lo
#[inline(always)]
fn mul_neg_2_exp_neg_24(input: __m256i) -> __m256i {
    unsafe {
        const MASK: __m256i = unsafe { transmute([(1 << 24) - 1; 8]) };

        let hi = x86_64::_mm256_srli_epi32::<24>(input);
        let lo = x86_64::_mm256_and_si256(input, MASK);
        let lo_shft = x86_64::_mm256_slli_epi32::<7>(lo);
        let lo_plus_hi = x86_64::_mm256_add_epi32(lo, hi);
        x86_64::_mm256_sub_epi32(lo_shft, lo_plus_hi)
    }
}

// [-2, 1, 2, 1/2, 3, 4, -1/2, -3, -4, 1/(2**8), -1/(2**8), 1/8, -1/8, -1/16, 1/2**24, -1/2**24]
impl InternalLayerParametersAVX2<16> for KoalaBearInternalLayerParameters {
    type ArrayLike = [__m256i; 15];

    #[inline(always)]
    fn diagonal_mul(input: &mut [__m256i; 15]) {
        // x0 -> sum - 2*x0
        // x1 -> sum + x1

        // x2 -> sum + 2*x2
        input[1] = add::<KoalaBearParameters>(input[1], input[1]);

        // x3 -> sum + x3/2
        input[2] = halve(input[2]);

        // x4 -> sum + 3*x4
        let acc3 = add::<KoalaBearParameters>(input[3], input[3]);
        input[3] = add::<KoalaBearParameters>(acc3, input[3]);

        // x5 -> sum + 4*x5
        let acc4 = add::<KoalaBearParameters>(input[4], input[4]);
        input[4] = add::<KoalaBearParameters>(acc4, acc4);

        // x6 -> sum - x6/2
        input[5] = halve(input[5]);

        // x7 -> sum - 3*x7
        let acc6 = add::<KoalaBearParameters>(input[6], input[6]);
        input[6] = add::<KoalaBearParameters>(acc6, input[6]);

        // x8 -> sum - 4*x8
        let acc7 = add::<KoalaBearParameters>(input[7], input[7]);
        input[7] = add::<KoalaBearParameters>(acc7, acc7);

        // x9 -> sum + x9/2**8
        input[8] = mul_2_exp_neg_8(input[8]);

        // x10 -> sum - x10/2**8
        input[9] = mul_neg_2_exp_neg_8(input[9]);

        // x11 -> sum + x11/2**3
        input[10] = mul_2_exp_neg_n::<3, 21>(input[10]);

        // x12 -> sum - x12/2**3
        input[11] = mul_neg_2_exp_neg_n::<3, 21>(input[11]);

        // x13 -> sum - x13/2**4
        input[12] = mul_neg_2_exp_neg_n::<4, 20>(input[12]);

        // x14 -> sum + x14/2**24
        input[13] = mul_2_exp_neg_24(input[13]);

        // x15 -> sum - x15/2**24
        input[14] = mul_neg_2_exp_neg_24(input[14]);
    }

    fn add_sum(input: &mut [__m256i; 15], sum: __m256i) {
        input[..5]
            .iter_mut()
            .for_each(|x| *x = add::<KoalaBearParameters>(sum, *x));

        input[5..8]
            .iter_mut()
            .for_each(|x| *x = sub::<KoalaBearParameters>(sum, *x));

        input[8..].iter_mut().for_each(|x| *x = signed_add(sum, *x));
    }
}

// [-2, 1, 2, 1/2, 3, 4, -1/2, -3, -4, 1/(2**8), -1/(2**8), 1/2**2, 1/(2**3), -1/(2**3), 1/(2**4), -1/(2**4), 1/(2**5), -1/(2**5), 1/(2**6), -1/(2**6), -1/(2**7), -1/(2**9), 1/2**24, -1/2**24]
impl InternalLayerParametersAVX2<24> for KoalaBearInternalLayerParameters {
    type ArrayLike = [__m256i; 23];

    #[inline(always)]
    fn diagonal_mul(input: &mut [__m256i; 23]) {
        // x0 -> sum - 2*x0
        // x1 -> sum + x1

        // x2 -> sum + 2*x2
        input[1] = add::<KoalaBearParameters>(input[1], input[1]);

        // x3 -> sum + x3/2
        input[2] = halve(input[2]);

        // x4 -> sum + 3*x4
        let acc3 = add::<KoalaBearParameters>(input[3], input[3]);
        input[3] = add::<KoalaBearParameters>(acc3, input[3]);

        // x5 -> sum + 4*x5
        let acc4 = add::<KoalaBearParameters>(input[4], input[4]);
        input[4] = add::<KoalaBearParameters>(acc4, acc4);

        // x6 -> sum - x6/2
        input[5] = halve(input[5]);

        // x7 -> sum - 3*x7
        let acc6 = add::<KoalaBearParameters>(input[6], input[6]);
        input[6] = add::<KoalaBearParameters>(acc6, input[6]);

        // x8 -> sum - 4*x8
        let acc7 = add::<KoalaBearParameters>(input[7], input[7]);
        input[7] = add::<KoalaBearParameters>(acc7, acc7);

        // x9 -> sum + x9/2**8
        input[8] = mul_2_exp_neg_8(input[8]);

        // x10 -> sum - x10/2**8
        input[9] = mul_neg_2_exp_neg_8(input[9]);

        // x11 -> sum + x11/2**2
        input[10] = mul_2_exp_neg_n::<2, 22>(input[10]);

        // x12 -> sum + x12/2**3
        input[11] = mul_2_exp_neg_n::<3, 21>(input[11]);

        // x13 -> sum - x13/2**3
        input[12] = mul_neg_2_exp_neg_n::<3, 21>(input[12]);

        // x14 -> sum + x14/2**4
        input[13] = mul_2_exp_neg_n::<4, 20>(input[13]);

        // x15 -> sum - x15/2**4
        input[14] = mul_neg_2_exp_neg_n::<4, 20>(input[14]);

        // x16 -> sum + x16/2**5
        input[15] = mul_2_exp_neg_n::<5, 19>(input[15]);

        // x17 -> sum - x17/2**5
        input[16] = mul_neg_2_exp_neg_n::<5, 19>(input[16]);

        // x18 -> sum + x18/2**6
        input[17] = mul_2_exp_neg_n::<6, 18>(input[17]);

        // x19 -> sum - x19/2**6
        input[18] = mul_neg_2_exp_neg_n::<6, 18>(input[18]);

        // x20 -> sum - x20/2**7
        input[19] = mul_neg_2_exp_neg_n::<7, 17>(input[19]);

        // x21 -> sum - x21/2**9
        input[20] = mul_neg_2_exp_neg_n::<9, 15>(input[20]);

        // x22 -> sum - x22/2**24
        input[21] = mul_2_exp_neg_24(input[21]);

        // x23 -> sum - x23/2**24
        input[22] = mul_neg_2_exp_neg_24(input[22]);
    }

    #[inline(always)]
    fn add_sum(input: &mut [__m256i; 23], sum: __m256i) {
        input[..5]
            .iter_mut()
            .for_each(|x| *x = add::<KoalaBearParameters>(sum, *x));

        input[5..8]
            .iter_mut()
            .for_each(|x| *x = sub::<KoalaBearParameters>(sum, *x));

        input[8..].iter_mut().for_each(|x| *x = signed_add(sum, *x));
    }
}

#[cfg(test)]
mod tests {
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
