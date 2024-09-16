use core::arch::x86_64::{self, __m256i};
use core::mem::transmute;

use p3_monty_31::{add, sub, InternalLayerParametersAVX2, MontyParameters, MontyParametersAVX2};

use crate::{BabyBearInternalLayerParameters, BabyBearParameters};

// Godbolt file showing that these all compile to the expected instructions. (Potentially plus a few memory ops):
// https://godbolt.org/#z:OYLghAFBqd5QCxAYwPYBMCmBRdBLAF1QCcAaPECAMzwBtMA7AQwFtMQByAcQwCNVaBANQFMAZwJihVEkJaoGBAJ5DimdAFdkBPArEA6YmI6kZigMrIm9EAEYALKYUEAqmMwAFAB6cADCYArO1JaJgZQYg0JUgAHMMwAOVZ2bj4BYVEJKQBpVGsmACFMJmIhD1R3PHQFACYTd3ptXQZKRnQAYQENFhaAVhrSNoAZPAZE7t5MYhAAdliKwmbO2m6%2BgZiFnQURsYSJqdnSBswmhXMCEoJl1ZB%2Bo8xGrYZzy53xlknpuYlL657bgY/YgEN57D4HGYASiOqA0xGQKSimCEaDUIBAAFIZgUMb4AIJCQlCErIJAgLwADgAbAB9Kn2dFYgoNKikIQ0mksGq9Kl4LEAEVIuIJRLYLHRBGIYTELA0oiF%2BIFGIAzDjFfi0AwJGUAEoASQAstgQEINMqakIVfyhDVbMrfDNfPTlcqVWq8ZrtR48e1sth%2BTSPCaOVyeXhLcrrRotUwqMimSIpVrZaIIBjegUPPqjW6hBT0/zIZaZvy3cLhTENLxpAwhAhrAA3TAQUaVgjBznc3lFgC0Kuw7M7YeL7qJppjcZHwrHY89wgA8gljYPQ7yI1GJ/HsYnpSnm%2BmCrZc/neoXi6XVdOZ4S50IABJ4oYAMQ7q/DVvHYljW4KO%2BTcv3DMICzQ0BwxGpf1sIsAHobWPAsiyVS91RFa96GEegmAkGleEIdchEpWl6XRTlVxpMJ0BpMQ8C7FsGDbNlF2wSEy3xa9CXQkR8MIukGRAUiuyo4haDwGlMBiPBzUZZV2iPZVsDotsWOQ1CZ04lgmCUSYaXrWgqG46leJIlgyOo4AGDEiTzQgB9nzZTDsNwghlNHdieOI/iTMEph0Eo8TJJqCACDZDStMwHTrCoFi2KJJVFRLct8UrasqFrMyxkonz0AgWgEDEV8uzwNliDygqw17fsV0KqcYsJaMv0nJkrzQzBhA2MQaTGYAaRiAyiL4gSeSovBzMsgLgJ9P0Aw8Yq8pc5q1NaoQxG6PqjM8sisrG6zcrEWaxHm2rFuEFaWBpVFSg/dyBq8oaVt4bbAtOtl2s6zBupiQ7VLHa7jLIlhRkss0nu6NlTvOkhiGi1S4rxWGKyrGs5A0WgaRqMSvBiN7uopRS5TK7shD7eSquHJqjvq78au%2BolbyYmkABUAHVsASBmAE0aXMbAADVWYJ99I0/KmE0lXcALTDNbBqGZ4NPRCEpU9ihE4hBBetX6NsEsRhNE/ypMxGT83kvHnNYmmOKW2hUDWjzBtpDTfKrDr/NsKlTcYpdGZZtnOe5vmEi%2B5XOOtqiECoYQrsMu3btpMRaBEx7pNkql%2Bxy1Ag7c6ObtMqtHogNX7NQMOI%2Bhsd4YSlDkqR2VUa6tGMax%2BvcdbfHScJ4mBxDarydUynGuxBbZz0BcveZ1mOa53n%2Bfb9XhYH38xf/VMD2l2XVTzBDz3N4OlrV22c%2B13Wk8N9pjYU1uzaV9iQ5tqP%2Br%2BwTHc0XgXYkt2PaEenx99qeA8zlqGFi5iHDpHIWmt7ZUQTnrKyNRk5uzTtbABM4IGxyonnfWgVQ4gIjmyNWZdYqVzhkQ8sNRXQZlGCJMYEBrAAHdNIHQLMKVKyNUbo0wJjbGnUVTtFvAkE0AU2R8MDNmZcAU06XwFhVEm3cyaDwppuamyssLuGBGJAAjmASACQ2Q1HsETIQCQRGgWQcPLUo9sDewnn7ae/DZ74X7j%2BP8MoJarxlnLM8SF3RDyJOpLCABrAW%2BEl4uJXkBWwEZ2g8MMb2IQclfwnkLObHxlthD73vutSBOtE6YOTgkCR9E5SmN8UtBs1gaTWwPo/IaFFhq0UviFAJxTUkqzvuA7O1SHZbVdu7MpqNrae0sT/Se/tWbNJVlbYBoCqlazutAk%2BPCjEgRzCbJBO8s4P1mXHDBsCC5FVaSXM2R0K6lhQuBchBRKGjGbHQhhLFTzMNrLXLh7DOH1wYDwvhAjzRCJHoY4xRpvk1AKW2KRRNKqyLXL3McjilHsRUVMAgGitEQB0TafRPZ/nLOYus68dMx4%2BxGbYoJH5YWiyTKEwCh53Eb0SQrC83ijpjj8WIQJ9iPwhL3JLAoEAIk8OiYHAx8TN7y2SUykpaS56oNMsfXJp98km0vuMzifSKltI1h0rZ5EGCUWovUwpwU5BNNxcdVpMzIHPzGh/VVAyv4EusX/MZJrmWTMOeatB8ccmwLyQC7AiCM7OqJNK7WOzxrYNAXgvABDCQnMSniauLDnmvKbu9GkvRTZgs7uy%2BRqkk2Ny4S0U%2BypdFyQvga6G8MkqI0TSjBuHCU3dRmBm2e0iu5DihTmsceb60FuTgATjZLYXoIKinlhIVWlKTza312TVw92kiW3gpke298nbRTTtTbO95yc5hxPXmWpSY7TlwwnTXDd3Ut2pqbQuyFUal1trfHCwkzyZ35u3afBwg7fAjqOcQ49CNJ2sLrW8zd9hm23tbdm1ydVFHQvYreA0eJzDZBJULMl25OWuPCZE6JejYnCrpdvFSKSJmSvdTKr1AVk56J/cqyZ5HvI6rqTyT%2BiHkN0aAW6jJMdTLzLlTw/d6cOMHJiCsDq6T2mbMgd03Ztr8GBsJMGu6oadrF1E1EHS%2Bzw2lwWrGqu1ap1103W%2B0D4GV2QdvU%2B%2BeP4SMIaQyh9laHYMYYpVyg8vKcMyXRfhjx9KxUW1I3WKVmqsmyu9afGjiry0KcC5U7jh8alMb1SxhpQg2PZGE9psBGqpMer4%2BFgT/rMtqbE5phjNTfL5zk1GmLSntkPUwenQ5Rceqlfk8ckhJYODQloJwXoJgWgcFsCYVAnAeDoH4IIEQ4hJBCFyPkIoJQygVEwFUWoJh2wcH8JCaECBihYGmBAaEgSeT6EdBSPt5o9EzCpA4ZUtgQicHsCYcUthfC%2BFIIN/wpBRscHqCAD7m3tukDgLAJAaAWASXoGQCgEAIdQ4OMQWwFJ3umDoKIIwlBeB%2BBMLhZgxAlCcGG6QPHJQlDzl4AEE4m3icQ7YIoecDBaCE62yYLAvANDAHaNYWg%2BUhts8wBpcINhWekHwGoJoTY%2BffY4ScACROTCjFEL10XIleBSgJ50LAOPSCSjwOKfnpAmzEH4O4fkguwjACoaAVn0IqChGAGIHmeBMC0PnDERgCunAWCsDYT9ZhXDuG8DroID3QjhBAJEaIav8qQGhKgGITw%2Bc9nnETdoIgSjAFaj2ZAlYrRSkTyQUI/jkQ9nTwnggPZ6BNloFaZUI3jfECqOIeAEBmBsBAOoRYCgjfWA0CkGovhbTdehMcU4LQIBtD%2BC0B7wxrlgk%2BMEdq3eGDT6X5sZooJ9jTAe2Pp4LxgRr93w8anzQD8gnn9v4IQIrhdH%2BLvi4wIt/gh36P2E8J2BQUexwfrn2de/dUCiAIGQDiX0BR30F8CEAgFwEIFkHOSgg2xxx21IECWVApH0BqD0SpGVF6D7TIQuxmBu16G/2ez/1F1%2B3%2B0ByQJBxgEQBAFhAIAYlh3hzoCmCSA7w4Cj2ANAPAJl3wCICb3QGCAD0sGsE/0cADzcE8B8FF1D1IFoQLwVx6z6wG3/04HnDlDbCEFQH0i4JANsDAN8AgKgM6Eh1YMujIQQN1yQOOxAHsH0D7Xe1wPsGVHsDwNsBu2wJINUPIM4EoOsNt2UI4DqDIO%2BwoICO22hGN2ogUDsKAA%3D%3D

/// Halve a vector of Monty31 field elements in canonical form.
/// If the inputs are not in canonical form, the result is undefined.
#[inline(always)]
fn halve(input: __m256i) -> __m256i {
    // We want this to compile to:
    //      vpand    least_bit, val, ONE
    //      vpsrld   t, val, 1
    //      vpsignd  maybe_half, HALF, least_bit
    //      vpaddd   res, t, maybe_half
    // throughput: 1.33 cyc/vec
    // latency: 3 cyc

    // Given an element val in [0, P), we want to compute val/2 mod P.
    // If val is even: val/2 mod P = val/2 = val >> 1.
    // If val is odd: val/2 mod P = (val + P)/2 = (val >> 1) + (P + 1)/2
    unsafe {
        // Safety: If this code got compiled then AVX2 intrinsics are available.
        const ONE: __m256i = unsafe { transmute([1; 8]) };
        const HALF: __m256i = unsafe { transmute([(BabyBearParameters::PRIME + 1) / 2; 8]) };

        let least_bit = x86_64::_mm256_and_si256(input, ONE); // Determine the parity of val.
        let t = x86_64::_mm256_srli_epi32::<1>(input);
        let maybe_half = x86_64::_mm256_sign_epi32(HALF, least_bit); // This does nothing when least_bit = 1 and sets the corresponding entry to 0 when least_bit = 0
        x86_64::_mm256_add_epi32(t, maybe_half)
    }
}

/// Add two vectors of Monty31 field elements with lhs in canonical form and rhs in (-P, P).
/// To reiterate, the two inputs are not symmetric, lhs must be positive. Return a value in canonical form.
/// If the inputs do not conform to these restrictions, the result is undefined.
#[inline(always)]
fn signed_add(lhs: __m256i, rhs: __m256i) -> __m256i {
    // We want this to compile to:
    //      vpsignd  pos_neg_P,  P, rhs
    //      vpaddd   sum,        lhs,   rhs
    //      vpsubd   sum_corr,   sum,   pos_neg_P
    //      vpminud  res,        sum,   sum_corr
    // throughput: 1.33 cyc/vec
    // latency: 3 cyc

    // Let t = lhs + rhs mod 2^32, we want to return t mod P while correcting for any possible wraparound.
    // We make use of the fact wrapping addition acts identically on signed and unsigned inputs.

    // If rhs is positive, lhs + rhs < 2P < 2^32 and so we interpret t as a unsigned 32 bit integer.
    //      In this case, t mod P = min_{u32}(t, t - P) where min_{u32} takes the min treating both inputs as unsigned 32 bit integers.
    //      This works as if t >= P then t - P < t and if t < P then, due to wraparound, t - P outputs t - P + 2^32 > t.
    // If rhs is negative, -2^31 < -P < lhs + rhs < P < 2^31 so we interpret t as a signed 32 bit integer.
    //      In this case t mod P = min_{u32}(t, t + P)
    //      This works as if t > 0 then t < t + P and if t < 0 then due to wraparound when we interpret t as an unsigned integer it becomes
    //      2^32 + t > t + P.
    // if rhs = 0 then we can just return t = lhs as it is already in the desired range.
    unsafe {
        // If rhs > 0 set the value to P, if rhs < 0 set it to -P and if rhs = 0 set it to 0.
        let pos_neg_p = x86_64::_mm256_sign_epi32(BabyBearParameters::PACKED_P, rhs);

        // Compute t = lhs + rhs
        let sum = x86_64::_mm256_add_epi32(lhs, rhs);

        // sum_corr = (t - P) if rhs > 0, t + P if rhs < 0 and t if rhs = 0 as desired.
        let sum_corr = x86_64::_mm256_sub_epi32(sum, pos_neg_p);

        x86_64::_mm256_min_epu32(sum, sum_corr)
    }
}

// The following functions all implement x -> +/- 2^{-n} x and output a value in (-P, P).
// The methods work provided n < 15 and our prime is of the form P = r * 2^j + 1 with r < 2^15.
// If, r < 2^8, there is a faster method for n = 8 which we also implement.
// Finally, if r = 2^i - 1, there is a related but slightly different method for n = j.

// The strategy for all these products is to observe that -2^{-n} = r2^{j - n} mod P.
// Hence given a field element x write it as x = x_lo + 2^n x_hi where x_lo < 2^n.
// Then -2^{-n} x = -x_hi + r2^{j - n} x_lo.
// Clearly x_hi < P and, as x_lo < 2^n, r2^{j - n} x_lo < r2^j < P so
// -P < r2^{j - n} x_lo - x_hi < P

// When r, n < 2^16, r2^{j - n} x_lo can be computed efficiently in AVX2.
// Additionally when n = j, rx_lo can be computed efficiently for some r.

// For 2^{-n} we do an identical thing but instead return x_hi - r2^{j - n} x_lo

/// Multiply a vector of Monty31 field elements in canonical form by 2**{-8}.
/// Output is returned as a vector of field elements in (-P, P).
/// If the inputs are not in canonical form, the result is undefined.
#[inline(always)]
fn mul_2_exp_neg_8(input: __m256i) -> __m256i {
    // We want this to compile to:
    //      vpsrld      hi, val, 8
    //      vpmaddubsw  lo, val, bcast32(7fh)
    //      vpslldq     lo, lo, 2
    //      vpsubd      t, hi, lo
    // throughput: 1.333
    // latency: 7
    unsafe {
        const FIFTEEN: __m256i = unsafe { transmute([15; 8]) };
        let hi = x86_64::_mm256_srli_epi32::<8>(input);

        // This returns 127 the bottom 8 bits of input which is exactly 127*x_lo.
        let lo = x86_64::_mm256_maddubs_epi16(input, FIFTEEN);

        // As the high bits 16 bits of each 32 bit word are all 0
        // we don't need to worry about shifting the high bits of one
        // word into the low bits of another. Thus we can use
        // _mm256_bslli_epi128 which can run on Port 5 as it is classed as
        // a swizzle operation.
        let lo_shft = x86_64::_mm256_slli_epi32::<19>(lo);
        x86_64::_mm256_sub_epi32(hi, lo_shft)
    }
}

/// Multiply a vector of Monty31 field elements in canonical form by -2**{-8}.
/// Output is returned as a vector of field elements in (-P, P).
/// If the inputs are not in canonical form, the result is undefined.
#[inline(always)]
fn mul_neg_2_exp_neg_8(input: __m256i) -> __m256i {
    // We want this to compile to:
    //      vpsrld      hi, val, 8
    //      vpmaddubsw  lo, val, bcast32(7fh)
    //      vpslldq     lo, lo, 2
    //      vpsubd      t, lo, hi
    // throughput: 1.333
    // latency: 7
    unsafe {
        const FIFTEEN: __m256i = unsafe { transmute([15; 8]) };
        let hi = x86_64::_mm256_srli_epi32::<8>(input);

        // This returns 127 the bottom 8 bits of input which is exactly 127*x_lo.
        let lo = x86_64::_mm256_maddubs_epi16(input, FIFTEEN);

        // As the high bits 16 bits of each 32 bit word are all 0
        // we don't need to worry about shifting the high bits of one
        // word into the low bits of another. Thus we can use
        // _mm256_bslli_epi128 which can run on Port 5 as it is classed as
        // a swizzle operation.
        let lo_shft = x86_64::_mm256_slli_epi32::<19>(lo);
        x86_64::_mm256_sub_epi32(lo_shft, hi)
    }
}

/// Multiply a vector of Monty31 field elements in canonical form by 2**{-N}.
/// N must be between 0 and 15, N_PRIME must be 27 - N.
/// Output is returned as a vector of field elements in (-P, P).
/// If the inputs are not in canonical form or N is outside the bounds, the result is undefined.
#[inline(always)]
fn mul_2_exp_neg_n<const N: i32, const N_PRIME: i32>(input: __m256i) -> __m256i {
    // We want this to compile to:
    //      vpslld      val_hi,     val,        n
    //      vpand       val_lo,     val,        2^{n} - 1
    //      vpmaddwd    lo_x_127    val_lo,     [[0, 127]; 4]
    //      vpslld      lo          lo_x_127    24 - n
    //      vpsubd      res         val_hi      lo
    // throughput: 1.67
    // latency: 8
    unsafe {
        assert_eq!(N, 27 - N_PRIME); // Compiler removes this provided it is satisfied.
        const FIFTEEN: __m256i = unsafe { transmute([15; 8]) };

        let mask: __m256i = transmute([(1 << N) - 1; 8]); // Compiler realises this is a constant.

        let hi = x86_64::_mm256_srli_epi32::<N>(input);
        let val_lo = x86_64::_mm256_and_si256(input, mask);

        // This returns 127 the bottom 16 bits of val_lo which is exactly 127*x_lo.
        let lo = x86_64::_mm256_madd_epi16(val_lo, FIFTEEN);
        let lo_shft = x86_64::_mm256_slli_epi32::<N_PRIME>(lo);
        x86_64::_mm256_sub_epi32(hi, lo_shft)
    }
}

/// Multiply a vector of Monty31 field elements in canonical form by -2**{-N}.
/// N must be between 0 and 15, N_PRIME must be 24 - N.
/// Output is returned as a vector of field elements in (-P, P).
/// If the inputs are not in canonical form or N is outside the bounds, the result is undefined.
#[inline(always)]
fn mul_neg_2_exp_neg_n<const N: i32, const N_PRIME: i32>(input: __m256i) -> __m256i {
    // We want this to compile to:
    //      vpslld      val_hi,     val,        n
    //      vpand       val_lo,     val,        2^{n} - 1
    //      vpmaddwd    lo_x_127    val_lo,     [[0, 127]; 4]
    //      vpslld      lo          lo_x_127    24 - n
    //      vpsubd      res         lo          val_hi
    // throughput: 1.67
    // latency: 8
    unsafe {
        assert_eq!(N, 27 - N_PRIME); // Compiler removes this provided it is satisfied.
        const FIFTEEN: __m256i = unsafe { transmute([15; 8]) };

        let mask: __m256i = transmute([(1 << N) - 1; 8]); // Compiler realises this is a constant.

        let hi = x86_64::_mm256_srli_epi32::<N>(input);
        let val_lo = x86_64::_mm256_and_si256(input, mask);

        // This returns 127 the bottom 16 bits of val_lo which is exactly 127*x_lo.
        let lo = x86_64::_mm256_madd_epi16(val_lo, FIFTEEN);
        let lo_shft = x86_64::_mm256_slli_epi32::<N_PRIME>(lo);
        x86_64::_mm256_sub_epi32(lo_shft, hi)
    }
}

/// Multiply a vector of Monty31 field elements in canonical form by 2**{-27}.
/// Output is returned as a vector of field elements in (-P, P).
/// If the inputs are not in canonical form, the result is undefined.
#[inline(always)]
fn mul_2_exp_neg_27(input: __m256i) -> __m256i {
    // We want this to compile to:
    //      vpslld  val_hi, 	 	val,            24
    //      vpand   val_lo, 	 	val,     	    2^{24} - 1
    //      vpslrd  val_lo_hi,   	val_lo,         7
    //      vpaddd  val_hi_plus_lo, val_lo,         val_hi
    //      vpsubd  res 		 	val_hi_plus_lo, val_lo_hi,
    unsafe {
        const MASK: __m256i = unsafe { transmute([(1 << 27) - 1; 8]) };
        let hi = x86_64::_mm256_srli_epi32::<27>(input);

        // As 15 = 2^4 - 1, provided overflow is impossible 127*x = (x << 7) - 1.
        // lo < 2^24 => (lo << 7) < 2^31 and (lo << 7) - lo < P.
        let lo = x86_64::_mm256_and_si256(input, MASK);
        let lo_shft = x86_64::_mm256_slli_epi32::<4>(lo);
        let lo_plus_hi = x86_64::_mm256_add_epi32(lo, hi);
        x86_64::_mm256_sub_epi32(lo_plus_hi, lo_shft)
    }
}

/// Multiply a vector of Monty31 field elements in canonical form by 2**{-27}.
/// Output is returned as a vector of field elements in (-P, P).
/// If the inputs are not in canonical form, the result is undefined.
#[inline(always)]
fn mul_neg_2_exp_neg_27(input: __m256i) -> __m256i {
    // We want this to compile to:
    //      vpslld  val_hi, 	 	val,        24
    //      vpand   val_lo, 	 	val,        2^{24} - 1
    //      vpslrd  val_lo_hi,   	val_lo,     7
    //      vpaddd  val_hi_plus_lo, val_lo,     val_hi
    //      vpsubd  res 		 	val_lo_hi,  val_hi_plus_lo
    unsafe {
        const MASK: __m256i = unsafe { transmute([(1 << 27) - 1; 8]) };
        let hi = x86_64::_mm256_srli_epi32::<27>(input);

        // As 127 = 2^7 - 1, provided overflow is impossible 127*x = (x << 7) - 1.
        // lo < 2^24 => (lo << 7) < 2^31 and (lo << 7) - lo < P.
        let lo = x86_64::_mm256_and_si256(input, MASK);
        let lo_shft = x86_64::_mm256_slli_epi32::<4>(lo);
        let lo_plus_hi = x86_64::_mm256_add_epi32(lo, hi);
        x86_64::_mm256_sub_epi32(lo_shft, lo_plus_hi)
    }
}

impl InternalLayerParametersAVX2<16> for BabyBearInternalLayerParameters {
    type ArrayLike = [__m256i; 15];

    /// For the BabyBear field and width 16 we multiply by the diagonal matrix:
    /// D = [-2, 1, 2, 1/2, 3, 4, -1/2, -3, -4, 1/(2**8), -1/(2**8), 1/4, 1/8, -1/16, 1/2**27, -1/2**27].
    /// The inputs must be in canonical form, otherwise the result is undefined.
    /// Even when the inputs are in canonical form, we make no garuntees on the output except that, provided
    /// the output is piped directly into add_sum the vector will be modified such that x[i] = D[i]*x[i] + sum.
    #[inline(always)]
    unsafe fn diagonal_mul(input: &mut [__m256i; 15]) {
        // As far as we know this is optimal in that it need the fewest instructions to perform all of these
        // multiplications. (Note that -1, 0 are not allowed on the diagonal for technical reasons).
        // If there exist other number b for which x*b mod P can be computed quickly this diagonal can be updated.

        // The strategy is very simple. 2, 3, 4, -3, -4 are implemented using addition.
        //                              1/2, -1/2 using the custom half function.
        //                              and the remainder utilising the custom functions for multiplication by 2^{-n}.

        // Note that for -3, -4, -1/2 we actually output 3x, 4x, x/2 and the negative is dealt with in add_sum by subtracting
        // this from the summation instead of adding it.

        // x0 is handled seperately as we need to apply the s-box to it.
        // x1 is being multiplied by 1 so we can also ignore it.

        // x2 -> sum + 2*x2
        input[1] = add::<BabyBearParameters>(input[1], input[1]);

        // x3 -> sum + x3/2
        input[2] = halve(input[2]);

        // x4 -> sum + 3*x4
        let acc3 = add::<BabyBearParameters>(input[3], input[3]);
        input[3] = add::<BabyBearParameters>(acc3, input[3]);

        // x5 -> sum + 4*x5
        let acc4 = add::<BabyBearParameters>(input[4], input[4]);
        input[4] = add::<BabyBearParameters>(acc4, acc4);

        // x6 -> sum - x6/2
        input[5] = halve(input[5]);

        // x7 -> sum - 3*x7
        let acc6 = add::<BabyBearParameters>(input[6], input[6]);
        input[6] = add::<BabyBearParameters>(acc6, input[6]);

        // x8 -> sum - 4*x8
        let acc7 = add::<BabyBearParameters>(input[7], input[7]);
        input[7] = add::<BabyBearParameters>(acc7, acc7);

        // x9 -> sum + x9/2**8
        input[8] = mul_2_exp_neg_8(input[8]);

        // x10 -> sum - x10/2**8
        input[9] = mul_neg_2_exp_neg_8(input[9]);

        // x11 -> sum + x11/2**2
        input[10] = mul_2_exp_neg_n::<2, 25>(input[10]);

        // x12 -> sum + x12/2**3
        input[11] = mul_2_exp_neg_n::<3, 24>(input[11]);

        // x13 -> sum - x13/2**4
        input[12] = mul_neg_2_exp_neg_n::<4, 23>(input[12]);

        // x14 -> sum + x14/2**24
        input[13] = mul_2_exp_neg_27(input[13]);

        // x15 -> sum - x15/2**24
        input[14] = mul_neg_2_exp_neg_27(input[14]);
    }

    /// Add sum to every element of input.
    /// Sum must be in canonical form and input must be exactly the output of diagonal mul.
    /// If either of these does not hold, the result is undefined.
    unsafe fn add_sum(input: &mut [__m256i; 15], sum: __m256i) {
        input[..5]
            .iter_mut()
            .for_each(|x| *x = add::<BabyBearParameters>(sum, *x));

        // Diagonal mul multiplied these by 1/2, 3, 4 instead of -1/2, -3, -4 so we need to subtract instead of adding.
        input[5..8]
            .iter_mut()
            .for_each(|x| *x = sub::<BabyBearParameters>(sum, *x));

        // Diagonal mul output a signed value in (-P, P) so we need to do a signed add.
        // Note that signed add's parameters are not interchangable. The first parameter must be positive.
        input[8..].iter_mut().for_each(|x| *x = signed_add(sum, *x));
    }
}

impl InternalLayerParametersAVX2<24> for BabyBearInternalLayerParameters {
    type ArrayLike = [__m256i; 23];

    /// For the BabyBear field and width 16 we multiply by the diagonal matrix:
    /// D = [-2, 1, 2, 1/2, 3, 4, -1/2, -3, -4, 1/(2**8), -1/(2**8), 1/2**2, -1/2**2, 1/(2**3), -1/(2**3), 1/(2**4), -1/(2**4), -1/(2**5), -1/(2**6), 1/(2**7), -1/(2**7), 1/(2**9), 1/2**24, -1/2**24]
    /// The inputs must be in canonical form, otherwise the result is undefined.
    /// Even when the inputs are in canonical form, we make no garuntees on the output except that, provided
    /// the output is piped directly into add_sum, the vector will be modified such that x[i] = D[i]*x[i] + sum.
    #[inline(always)]
    unsafe fn diagonal_mul(input: &mut [__m256i; 23]) {
        // As far as we know this is optimal in that it need the fewest instructions to perform all of these
        // multiplications. (Note that -1, 0 are not allowed on the diagonal for technical reasons).
        // If there exist other number b for which x*b mod P can be computed quickly this diagonal can be updated.

        // The strategy is very simple. 2, 3, 4, -3, -4 are implemented using addition.
        //                              1/2, -1/2 using the custom half function.
        //                              and the remainder utilising the custom functions for multiplication by 2^{-n}.

        // Note that for -3, -4, -1/2 we actually output 3x, 4x, x/2 and the negative is dealt with in add_sum by subtracting
        // this from the summation instead of adding it.

        // x0 is handled seperately as we need to apply the s-box to it.
        // x1 is being multiplied by 1 so we can also ignore it.

        // x2 -> sum + 2*x2
        input[1] = add::<BabyBearParameters>(input[1], input[1]);

        // x3 -> sum + x3/2
        input[2] = halve(input[2]);

        // x4 -> sum + 3*x4
        let acc3 = add::<BabyBearParameters>(input[3], input[3]);
        input[3] = add::<BabyBearParameters>(acc3, input[3]);

        // x5 -> sum + 4*x5
        let acc4 = add::<BabyBearParameters>(input[4], input[4]);
        input[4] = add::<BabyBearParameters>(acc4, acc4);

        // x6 -> sum - x6/2
        input[5] = halve(input[5]);

        // x7 -> sum - 3*x7
        let acc6 = add::<BabyBearParameters>(input[6], input[6]);
        input[6] = add::<BabyBearParameters>(acc6, input[6]);

        // x8 -> sum - 4*x8
        let acc7 = add::<BabyBearParameters>(input[7], input[7]);
        input[7] = add::<BabyBearParameters>(acc7, acc7);

        // x9 -> sum + x9/2**8
        input[8] = mul_2_exp_neg_8(input[8]);

        // x10 -> sum - x10/2**8
        input[9] = mul_neg_2_exp_neg_8(input[9]);

        // x11 -> sum + x11/2**2
        input[10] = mul_2_exp_neg_n::<2, 25>(input[10]);

        // x12 -> sum - x12/2**2
        input[11] = mul_neg_2_exp_neg_n::<2, 25>(input[11]);

        // x13 -> sum + x13/2**3
        input[12] = mul_2_exp_neg_n::<3, 24>(input[12]);

        // x14 -> sum - x14/2**3
        input[13] = mul_neg_2_exp_neg_n::<3, 24>(input[13]);

        // x15 -> sum + x15/2**4
        input[14] = mul_2_exp_neg_n::<4, 23>(input[14]);

        // x16 -> sum - x16/2**4
        input[15] = mul_neg_2_exp_neg_n::<4, 23>(input[15]);

        // x17 -> sum - x17/2**5
        input[16] = mul_neg_2_exp_neg_n::<5, 22>(input[16]);

        // x18 -> sum - x18/2**6
        input[17] = mul_neg_2_exp_neg_n::<6, 21>(input[17]);

        // x19 -> sum + x19/2**7
        input[18] = mul_2_exp_neg_n::<7, 20>(input[18]);

        // x20 -> sum - x20/2**7
        input[19] = mul_neg_2_exp_neg_n::<7, 20>(input[19]);

        // x21 -> sum + x21/2**9
        input[20] = mul_2_exp_neg_n::<9, 18>(input[20]);

        // x22 -> sum - x22/2**24
        input[21] = mul_2_exp_neg_27(input[21]);

        // x23 -> sum - x23/2**24
        input[22] = mul_neg_2_exp_neg_27(input[22]);
    }

    /// Add sum to every element of input.
    /// Sum must be in canonical form and input must be exactly the output of diagonal mul.
    /// If either of these does not hold, the result is undefined.
    unsafe fn add_sum(input: &mut [__m256i; 23], sum: __m256i) {
        input[..5]
            .iter_mut()
            .for_each(|x| *x = add::<BabyBearParameters>(sum, *x));

        // Diagonal mul multiplied these by 1/2, 3, 4 instead of -1/2, -3, -4 so we need to subtract instead of adding.
        input[5..8]
            .iter_mut()
            .for_each(|x| *x = sub::<BabyBearParameters>(sum, *x));

        // Diagonal mul output a signed value in (-P, P) so we need to do a signed add.
        // Note that signed add's parameters are not interchangable. The first parameter must be positive.
        input[8..].iter_mut().for_each(|x| *x = signed_add(sum, *x));
    }
}

#[cfg(test)]
mod tests {
    use p3_field::AbstractField;
    use p3_symmetric::Permutation;
    use rand::Rng;

    use crate::{BabyBear, PackedBabyBearAVX2, Poseidon2BabyBear};

    type F = BabyBear;
    const D: u64 = 7;
    type Perm16 = Poseidon2BabyBear<16, D>;
    type Perm24 = Poseidon2BabyBear<24, D>;

    /// Test that the output is the same as the scalar version on a random input.
    #[test]
    fn test_avx2_poseidon2_width_16() {
        let mut rng = rand::thread_rng();

        // Our Poseidon2 implementation.
        let poseidon2 = Perm16::new_from_rng_128(&mut rng);

        let input: [F; 16] = rng.gen();

        let mut expected = input;
        poseidon2.permute_mut(&mut expected);

        let mut avx2_input = input.map(PackedBabyBearAVX2::from_f);
        poseidon2.permute_mut(&mut avx2_input);

        let avx2_output = avx2_input.map(|x| x.0[0]);

        assert_eq!(avx2_output, expected);
    }

    /// Test that the output is the same as the scalar version on a random input.
    #[test]
    fn test_avx2_poseidon2_width_24() {
        let mut rng = rand::thread_rng();

        // Our Poseidon2 implementation.
        let poseidon2 = Perm24::new_from_rng_128(&mut rng);

        let input: [F; 24] = rng.gen();

        let mut expected = input;
        poseidon2.permute_mut(&mut expected);

        let mut avx2_input = input.map(PackedBabyBearAVX2::from_f);
        poseidon2.permute_mut(&mut avx2_input);

        let avx2_output = avx2_input.map(|x| x.0[0]);

        assert_eq!(avx2_output, expected);
    }
}
