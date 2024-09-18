use core::arch::x86_64::{self, __m256i};
use core::mem::transmute;

use crate::{MontyParameters, PackedMontyParameters, TwoAdicData};

// Godbolt file showing that these all compile to the expected instructions. (Potentially plus a few memory ops):
// https://godbolt.org/#z:OYLghAFBqd5QCxAYwPYBMCmBRdBLAF1QCcAaPECAMzwBtMA7AQwFtMQByAcQwCNVaBANQFMAZwJihVEkJaoGBAJ5DimdAFdkBPArEA6YmI6kZigMrIm9EAEYALKYUEAqmMwAFAB6cADCYArO1JaJgZQYg0JUgAHMMwAOVZ2bj4BYVEJKQBpVGsmACFMJmIhD1R3PHQFACYTd3ptXQZKRnQAYQENFhb7ADZSNoAZPAZE7t5MYjsADgBOWIrCZs7abt6BmKWdBRGxhImp2YWGzCaFcwISglX1kH7SU/OGS%2Bu98ZZJ6dt5x6viG5dHr3AYSN6jD5fY4ASkeqA0xGQKSimCEaDUIBAAFIAOwFLG%2BACCQhJQhKyCQIC8Mz6AH0%2BvZMbiCg0qKQhLTaSwagBWPp4XEAEVIBOJpLYLExBGIYTELA0ohFRKFWIAzPjlUTpUxCGUSslREYhMzRaS0XphB4AEoASQAstgQEINKqamqNWKSWgGBIyoT2tlsILaR4nZzuXy8MbVYLnT6mFRUcyRDKffLRBAsTyCuZMLQqJjrfbsO6hDMs4LocacYL3abSd7fQAVADqAHlaYTBTb2jamwBNJ1RPAAL0wdaJZsbwjbguDADF/U221ah67o7GILn84XbQ7oyXVdghNuCyBWx2uz2%2B/3oRPCSrNYSJJFtEJcvkiiUPPq2IaxPeop4CwMS0HqMp/lMUgyKUH6hF%2BxA/hBmD/tWHpThaZR7o6zrrmqsY1LYqq%2BDivgMqqqr3hhPrCBenbdr2A5DmIo5JjGQg1PYgE1qKvE1JR2ajLQEIQNYADuTBKGId48rWRJUAwQgINYABu46qu0obgQaUFqtgECjDECphlyvL8lWAC0ekcqZkZofWJIaPGib2ZOZoNphbYJDh4ZmVG%2BFxmICZJniKayummCZtmtiluWslViq6oOe59DCMpO42RG/IbmFaYKpFWYFBAoYgEW%2B5YjUBRCLYVYAPQcbFFZ3klT7uaSqVCPQTASLSvC6gF1J0gymJclltJhOgtIsWZBkMEZBDsl52DNehbWdShIg5YN9KMiAo1mVNxDCbSmAxHgrpMhpMVHrN80rclZodSwkmTLS6VUFtNI7SNLBjSxwAMCdZ2uhA73sl1PV9QQ91uW123DXtv0HUw6CTad501BAC1yC9mBvdYVB3rD1ZyQ%2BPFPhVAkFEJIniZJ0kVqKClCP9YyTSj6BqppTpITpRh6RAtAIGIJlZXg7LEMLot%2BZZ1m%2BXZJrE05QUuYrnopRtWxiLSYzALSMSfUNu37XyU14ADQMY8Vhb%2BoGwYeBLwsw%2Brj0bWI3SG99iNjRzlsg0LYiOwzLUu%2B1bvdLS6KlANX0IybdLu7wfuY%2B7LDslrOuYHrMTO2tQjw8bSOmywoxAy6KfdOyqeRyQxBE%2Brj5k6TfFUzTYyibQElSTJpPM/KtC0jUJ1eDEmd6wwXNabzkFGOy05CAkToY3PmEJCG2FL66AuGcZmUy0IVlHnvCt4slyvBa5ockt17gAidACOYCQAkxqVQv6/FuyJV0VejH9mSUgMa5zWh1DAk0qBMG0CQaWCt2LajyhmQqJVZwLiXCuRq8VALE1dsIZ6YgADWMDsoBXgXKfKUUiq2GjO0LmC9LLVQwZWLBV91ppX8uxAuP0/pHTwMnS67QEjbzmgqYBbUOoqWsLSWgqBPZxyLnSCaZsZo72xng/BoiNbCGkbIwuY1nqo0trYPoEAJED2keyMBtIIFQLrlREBG1pFTQQFQYQMcjZcIOmIWgx10YXWxBpNeZVDz6WkRos0nDvaeI0EnXxmMEDi06qgJxLj65mkbuk5U/FCpt0inTbujN5KKX7mPQew9R66x1pPHmv4UJQRXjRBem8aj1N9IEjeQgMZCPmkQvAssj7y2IafJWzkQqrTajfKYBAH5PwgC/CqVU2mfzKJiH%2BDEbwAI6a6MJYdhCWOsUQaYx9iFwNTGQxB2ZkFzlpIudoy4rSMOdg9HZOMCE9JyqQiKFCIBUK5rQhI9DrpVTikwkOTySQdXiTojxpsxA8L4f4gRXSRF2LERtUxUiZFuK9vHcaDBJrTT5LdBU7I1HbPBQ4zFHDY66IOvotGZ0jEmMkeYoQezIEHLJawxJyTXFUvcZEmF3jeGxP4Ysh0AtQkovchEnFidk6CySWIZx2N4mpNJOk8mD5MmtwYMJdueSGaySZkUjQA8h6YBHiUggYkkkozwMgQgSgqnaRnoHc0DTF6bOae61pH8HRNKRQQHpfTjwDP8kM9W59VYRrzhMu%2BmBH7Py/is9s9FrwDg2UAqVZo41TITTMuZb8xXYHZKqGqzC85PW6oQo57DYwfPIYVb51C/kAoeRW%2BxbCoUCoTnCkVCLBE3RUZyjq2isVyJ9nipRhKVEkurSOilPLu2yqFfCrmxaJWoAXVopJoEohvTrfnal0KFEGNiQq9kqrs2khlfIqa0T5WOL3dreJ4NFXKrVSSDVzdtXZN1bTTu9Me7GrkKakp5rLUVOtba/ADrlDOunrU2ePrhCeuXih9%2BQSA1DuEUG2tIba2XzNFG0ZYKAG3zzYm2Zybzypt/us7qXrOW5umc/V%2BCy/UlqEGWx52DnlqLeSQ05nym0/I0q2g%2BDD1Rliah21FXbx00phX24GNRRWBu3Yk5dd7FEEuMbOl56jr3kp3UuxTJ6pqrv7euzjm7NNPrWC%2Bw9t6fZntUxepSvTjNHv5bKh957HFKpcW%2B/WjmD2fpJrxTVLc/16tyYB/JRrCmgbNWUkp5YNJTxqf%2BFpqHOPYf0io4NB85a2UGWMwKF81axrEBR1jEAZjscw9hGjqz03/0Y1m0FfGTOsoMfs6BhGhPhUbRczEKDrloPudJ4FmnIXmZ7YdHxqn%2BEZcK7h%2BzlLYwudpRzaJ2t0aMoM2ymxm2zN8uxXerxy2MaitszdSVIc847Zhf59zr7uVBehslb9vFf2CX/fqhLhre4moHhUiD5Ss60gy9zF1SG3Xz2LQVoleGw0EbDUR0kJGsfjNq5M%2BrjX5nNaWd/OjayM2da2XJzRfXwHssG5j4bCCCpjZABNm5dz21Pc7Z57T3Cbt%2BK5mt1Hm3%2Be7dRvtwx%2BncMWP6wz2xPP5OfeVeLwVgu1MDvuyErd3mXsJze1bQLyrL1eeJr9p8RleDSEUulNStJ8G8FR8Vw%2Boayvhoq3b9gCK4KFGKIhbLukcN3Siz%2BwkVubcs3NmzcaBjHeCylrWoOLvSti1x1HgG6hY/oH4b7hCiH/wSuFkHeuGTw/RMj8UyHJSeTO/wyV/p7v09V7SxUloPu8jwX9wXupQgeTslsHMDToe/vl%2Bt33MD1eKlzDr%2Bjhvbu0/VfFJP1v0P29czz93wPyGFjVR5MP5U0WiQR4n%2BD6HU/ocy%2B6fX13Q2Y3L7P3rC/49c%2Bd799%2BbfbqBjVRF8OkfluK9T9wNV89YHBZ93cMcm8l8SRikIcQCdZX9Pwt9kIctqpHBqpfAD8m5R8T8wdSkLUoc9ZoNY97VHUHcncisb9U8/Jm8V8CCrUbUSC4MlBECu8P8UDe9OJ2QcQsCy9cCUtgD6CoNGC7VmDyDwCxZIDF978YCwM4ChDodiDRDHVWD38A8ODkMuChAeDg9kVD8w9%2BCW8FC9YZhxDKC59b8mcZCBDn8YdVD89P8B8%2BheCj8x9K85Dz94DTD49zCID5878KtYDPDjC7CO8kD2C%2BY3UjEXCDDAC8DbDTCxAxJRwRx6AJD95LCoDrCcdoD3J54lpaRWxsAEgBxaRzBsAAA1YowTdiHI0KBtc5AoWwGoHQoFJqSLJXWnebC7CdTxFTW7BFX/DbbzUdLbHzS7PRPbXgA7BlK/YlIQAoooko/sMoyo4os7L7NXOkaYqzBlGoGYfhN0B7XXTo8JY9RbOVc9D7Y3FJH7aLVwwwjwp/LwqaZIkcVIyKXwyQ/wqwirOoiraiX0RYlsYo0o8oqoz1JnWokZNCXKM5VnJoloh5DogE55bo7bc42VfooXDSIYkPU455MdHopTOkOlKXQ7OY7GYE0ElY8E9YkYxdTYhbHFHYm7Zog4hFI4nXTlfXe9GJdzG4lVM3Bue4wUDgWEWgTgHkEwFoDgWwEwVATgHgdAfgQQEQcQSQd8N/BCMoCoTAKoWoEwINDgfwaEWEBAYoLAaYCAWEQhPkfQUieYV0TiHEPoBwMtEITgewEwSUWwXwXwUgWU/wUgRUjgeoEAAM40000gOAWAJANAECOgKYcgSgBMs6egaYYgH4f00wOgf8SgXgPwEwPqZgYgFguU4s0YEoJQNsXgAIM4Y0%2BU0gBMtgRQNsXVcs4MrAXgDQYAdoawWgEWCs0gLAZ6cIGwE0kwfANQJoNSIc4Mi1M4fKTgJs0YUQSUyckIPAXgGUMszoLAIs0gaUYCFckwNSYgfgdwQUTAMc4APVUASc2EKgUIYAMQCovATAMSNsGIRgU8pwCwKwGwBwf81wdwbwQ8oIWwEIMICIKIBaYSaY%2BAWEVAGIHYH0TgCyNsA%2BdoEQEoYAFCCyZAIyfCGUVCkgUIfBVECyHClCggCyegNSWgfCVUBU884gKocQeACAZgNgEAdQZYBQUgUxDQFIGoXwQicU2EJ4NC1oPFW4YEZowYPFd4A4T4I4ewGYAMrWAShgeSloRS7StClSw4aYDSgM6S5oV4AEPSuwZpCyi4f4AgYytS0yzSv4a4GyxSsEAEZyqEMyqS%2BEREdgGqT0jgaUwMw80M1QOC5AaqfQTS/QXwIQCAXAQgWQSmGqI0oss00gQhewXwB05ovoGoOYOYHEGYWwWwHkOYPkUK70iKzc0M8MyM7KmMmARAEAeEAgeaFMiANMpM4gJIXijgSICQWK2weKgqhc/AA5KoYIMwAgSwawYKxwBatwTwHwTcyC0gMSUi08iUqUmUyKzgNsBUeaVlD6UaggcayaxK5KzoRMjM1%2BHjLKx820kAGofQewIq/0kq%2BwUquYfoRwDc%2BqoMhUzgZqo87Kg6jgOoBq4MpqqGt6oSqCZoe4IAA%3D%3D%3D

/// Halve a vector of Monty31 field elements in canonical form.
/// If the inputs are not in canonical form, the result is undefined.
#[inline(always)]
pub fn halve_avx2<MP: MontyParameters>(input: __m256i) -> __m256i {
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
        let half: __m256i = transmute([(MP::PRIME + 1) / 2; 8]); // Compiler realises this is constant.

        let least_bit = x86_64::_mm256_and_si256(input, ONE); // Determine the parity of val.
        let t = x86_64::_mm256_srli_epi32::<1>(input);
        let maybe_half = x86_64::_mm256_sign_epi32(half, least_bit); // This does nothing when least_bit = 1 and sets the corresponding entry to 0 when least_bit = 0
        x86_64::_mm256_add_epi32(t, maybe_half)
    }
}

/// Add two vectors of Monty31 field elements with lhs in canonical form and rhs in (-P, P).
///
/// # Safety
///
/// This function is not symmetric in the inputs. The caller must ensure that inputs
/// conform to teh expected representation. Each element of lhs must lie in [0, P) and
/// each element of rhs in (-P, P).
#[inline(always)]
pub unsafe fn signed_add_avx2<PMP: PackedMontyParameters>(lhs: __m256i, rhs: __m256i) -> __m256i {
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
        let pos_neg_p = x86_64::_mm256_sign_epi32(PMP::PACKED_P, rhs);

        // Compute t = lhs + rhs
        let sum = x86_64::_mm256_add_epi32(lhs, rhs);

        // sum_corr = (t - P) if rhs > 0, t + P if rhs < 0 and t if rhs = 0 as desired.
        let sum_corr = x86_64::_mm256_sub_epi32(sum, pos_neg_p);

        x86_64::_mm256_min_epu32(sum, sum_corr)
    }
}

// The following functions implement x -> +/- 2^{-N} x and output a value in (-P, P).
// The method works provided N < 15 and our prime is of the form P = r * 2^j + 1 with r < 2^15.
// Additionally, when r = 2^i - 1, there is a related method for N = j which we implement below.

// We present the method for multiplication by -2^{-N} here.
// The method for 2^{-N} is essentially identical, we simply return the negative.
// The strategy for these products is to observe that -2^{-N} = r2^{j - N} mod P.
// Hence given a field element x write it as x = x_lo + 2^N x_hi where x_lo < 2^N.
// Then -2^{-N} x = -x_hi + r2^{j - N} x_lo.
// Clearly x_hi < P and, as x_lo < 2^N, r2^{j - N} x_lo < r2^j < P so
// -P < r2^{j - N} x_lo - x_hi < P

// When r < 2^16, N < 15, r2^{j - N} x_lo can be computed efficiently in AVX2 using _mm256_madd_epi16.
// This avoids having to split the input in two and doing multiple multiplications. It also
// lets us avoid needing any monty reductions.
// There is a further improvment possible when if r < 2^7 and N = 8 using _mm256_maddubs_epi16.
// This lets us avoid a mask and an and.

// When n = j and r = 2^i - 1, rx_lo can also be computed efficiently using a shift and subtraction.

/// Multiply a vector of Monty31 field elements in canonical form by 2**{-N}.
///
/// # Safety
///
/// The prime P must be of the form P = r * 2^j + 1 with r odd and r < 2^15.
/// N must be between 0 and 15.
/// Input must be given in canonical form.
/// Output is not in canonical form, outputs are only guaranteed to lie in(-P, P).
#[inline(always)]
pub unsafe fn mul_2_exp_neg_n_avx2<TAD: TwoAdicData, const N: i32, const N_PRIME: i32>(
    input: __m256i,
) -> __m256i {
    // We want this to compile to:
    //      vpslld      val_hi,     val,        n
    //      vpand       val_lo,     val,        2^{n} - 1
    //      vpmaddwd    lo_x_127    val_lo,     [[0, 127]; 4]
    //      vpslld      lo          lo_x_127    24 - n
    //      vpsubd      res         val_hi      lo
    // throughput: 1.67
    // latency: 8
    unsafe {
        assert_eq!(N + N_PRIME, TAD::TWO_ADICITY as i32); // Compiler removes this provided it is satisfied.

        let odd_factor: __m256i = transmute([TAD::ODD_FACTOR; 8]);
        let mask: __m256i = transmute([(1 << N) - 1; 8]); // Compiler realises this is a constant.

        let hi = x86_64::_mm256_srli_epi32::<N>(input);
        let val_lo = x86_64::_mm256_and_si256(input, mask);

        // This returns 127 the bottom 16 bits of val_lo which is exactly 127*x_lo.
        let lo = x86_64::_mm256_madd_epi16(val_lo, odd_factor);
        let lo_shft = x86_64::_mm256_slli_epi32::<N_PRIME>(lo);
        x86_64::_mm256_sub_epi32(hi, lo_shft)
    }
}

/// Multiply a vector of Monty31 field elements in canonical form by -2**{-N}.
/// # Safety
///
/// The prime P must be of the form P = r * 2^j + 1 with r odd and r < 2^15.
/// N must be between 0 and 15.
/// Input must be given in canonical form.
/// Output is not in canonical form, outputs are only guaranteed to lie in(-P, P).
#[inline(always)]
pub unsafe fn mul_neg_2_exp_neg_n_avx2<TAD: TwoAdicData, const N: i32, const N_PRIME: i32>(
    input: __m256i,
) -> __m256i {
    // We want this to compile to:
    //      vpslld      val_hi,     val,        n
    //      vpand       val_lo,     val,        2^{n} - 1
    //      vpmaddwd    lo_x_127    val_lo,     [[0, 127]; 4]
    //      vpslld      lo          lo_x_127    24 - n
    //      vpsubd      res         lo          val_hi
    // throughput: 1.67
    // latency: 8
    unsafe {
        assert_eq!(N + N_PRIME, TAD::TWO_ADICITY as i32); // Compiler removes this provided it is satisfied.

        let odd_factor: __m256i = transmute([TAD::ODD_FACTOR; 8]); // Compiler realises this is a constant.
        let mask: __m256i = transmute([(1 << N) - 1; 8]); // Compiler realises this is a constant.

        let hi = x86_64::_mm256_srli_epi32::<N>(input);
        let val_lo = x86_64::_mm256_and_si256(input, mask);

        // This returns 127 the bottom 16 bits of val_lo which is exactly 127*x_lo.
        let lo = x86_64::_mm256_madd_epi16(val_lo, odd_factor);
        let lo_shft = x86_64::_mm256_slli_epi32::<N_PRIME>(lo);
        x86_64::_mm256_sub_epi32(lo_shft, hi)
    }
}

/// Multiply a vector of Monty31 field elements in canonical form by 2**{-8}.
/// # Safety
///
/// The prime P must be of the form P = r * 2^j + 1 with r odd and r < 2^7.
/// Input must be given in canonical form.
/// Output is not in canonical form, outputs are only guaranteed to lie in(-P, P).
#[inline(always)]
pub unsafe fn mul_2_exp_neg_8_avx2<TAD: TwoAdicData, const N_PRIME: i32>(
    input: __m256i,
) -> __m256i {
    // We want this to compile to:
    //      vpsrld      hi, val, 8
    //      vpmaddubsw  lo, val, bcast32(7fh)
    //      vpslldq     lo, lo, 2
    //      vpsubd      t, hi, lo
    // throughput: 1.333
    // latency: 7
    unsafe {
        assert_eq!(8 + N_PRIME, TAD::TWO_ADICITY as i32); // Compiler removes this provided it is satisfied.

        let odd_factor: __m256i = transmute([TAD::ODD_FACTOR; 8]);
        let hi = x86_64::_mm256_srli_epi32::<8>(input);

        // This returns 127 the bottom 8 bits of input which is exactly 127*x_lo.
        let lo = x86_64::_mm256_maddubs_epi16(input, odd_factor);

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
/// # Safety
///
/// The prime P must be of the form P = r * 2^j + 1 with r odd and r < 2^7.
/// Input must be given in canonical form.
/// Output is not in canonical form, outputs are only guaranteed to lie in(-P, P).
#[inline(always)]
pub unsafe fn mul_neg_2_exp_neg_8_avx2<TAD: TwoAdicData, const N_PRIME: i32>(
    input: __m256i,
) -> __m256i {
    // We want this to compile to:
    //      vpsrld      hi, val, 8
    //      vpmaddubsw  lo, val, bcast32(7fh)
    //      vpslldq     lo, lo, 2
    //      vpsubd      t, lo, hi
    // throughput: 1.333
    // latency: 7
    unsafe {
        assert_eq!(8 + N_PRIME, TAD::TWO_ADICITY as i32); // Compiler removes this provided it is satisfied.

        let odd_factor: __m256i = transmute([TAD::ODD_FACTOR; 8]);
        let hi = x86_64::_mm256_srli_epi32::<8>(input);

        // This returns 127 the bottom 8 bits of input which is exactly 127*x_lo.
        let lo = x86_64::_mm256_maddubs_epi16(input, odd_factor);

        // As the high bits 16 bits of each 32 bit word are all 0
        // we don't need to worry about shifting the high bits of one
        // word into the low bits of another. Thus we can use
        // _mm256_bslli_epi128 which can run on Port 5 as it is classed as
        // a swizzle operation.
        let lo_shft = x86_64::_mm256_slli_epi32::<19>(lo);
        x86_64::_mm256_sub_epi32(lo_shft, hi)
    }
}

/// Multiply a vector of Monty31 field elements in canonical form by 2**{-N} where p = 2^31 - 2^N + 1.
/// # Safety
///
/// The prime P must have the form P = 2^31 - 2^N + 1.
/// Input must be given in canonical form.
/// Output is not in canonical form, outputs are only guaranteed to lie in(-P, P).
#[inline(always)]
pub unsafe fn mul_2_exp_neg_two_adicity_avx2<TAD: TwoAdicData, const N: i32, const N_PRIME: i32>(
    input: __m256i,
) -> __m256i {
    // We want this to compile to:
    //      vpslld  val_hi, 	 	val,            24
    //      vpand   val_lo, 	 	val,     	    2^{24} - 1
    //      vpslrd  val_lo_hi,   	val_lo,         7
    //      vpaddd  val_hi_plus_lo, val_lo,         val_hi
    //      vpsubd  res 		 	val_hi_plus_lo, val_lo_hi,
    unsafe {
        assert_eq!(N, (TAD::TWO_ADICITY as i32)); // Compiler removes this provided it is satisfied.
        assert_eq!(N + N_PRIME, 31); // Compiler removes this provided it is satisfied.

        let mask: __m256i = transmute([(1 << N) - 1; 8]); // Compiler realises this is a constant.
        let hi = x86_64::_mm256_srli_epi32::<N>(input);

        // As 127 = 2^7 - 1, provided overflow is impossible 127*x = (x << 7) - 1.
        // lo < 2^24 => (lo << 7) < 2^31 and (lo << 7) - lo < P.
        let lo = x86_64::_mm256_and_si256(input, mask);
        let lo_shft = x86_64::_mm256_slli_epi32::<N_PRIME>(lo);
        let lo_plus_hi = x86_64::_mm256_add_epi32(lo, hi);
        x86_64::_mm256_sub_epi32(lo_plus_hi, lo_shft)
    }
}

/// Multiply a vector of Monty31 field elements in canonical form by -2**{-N} where p = 2^31 - 2^N + 1.
/// # Safety
///
/// The prime P must have the form P = 2^31 - 2^N + 1.
/// Input must be given in canonical form.
/// Output is not in canonical form, outputs are only guaranteed to lie in(-P, P).
#[inline(always)]
pub unsafe fn mul_neg_2_exp_neg_two_adicity_avx2<
    TAD: TwoAdicData,
    const N: i32,
    const N_PRIME: i32,
>(
    input: __m256i,
) -> __m256i {
    // We want this to compile to:
    //      vpslld  val_hi, 	 	val,        24
    //      vpand   val_lo, 	 	val,        2^{24} - 1
    //      vpslrd  val_lo_hi,   	val_lo,     7
    //      vpaddd  val_hi_plus_lo, val_lo,     val_hi
    //      vpsubd  res 		 	val_lo_hi,  val_hi_plus_lo
    unsafe {
        assert_eq!(N, (TAD::TWO_ADICITY as i32)); // Compiler removes this provided it is satisfied.
        assert_eq!(N + N_PRIME, 31); // Compiler removes this provided it is satisfied.

        let mask: __m256i = transmute([(1 << N) - 1; 8]); // Compiler realises this is a constant.
        let hi = x86_64::_mm256_srli_epi32::<N>(input);

        // As 127 = 2^7 - 1, provided overflow is impossible 127*x = (x << 7) - 1.
        // lo < 2^24 => (lo << 7) < 2^31 and (lo << 7) - lo < P.
        let lo = x86_64::_mm256_and_si256(input, mask);
        let lo_shft = x86_64::_mm256_slli_epi32::<N_PRIME>(lo);
        let lo_plus_hi = x86_64::_mm256_add_epi32(lo, hi);
        x86_64::_mm256_sub_epi32(lo_shft, lo_plus_hi)
    }
}
