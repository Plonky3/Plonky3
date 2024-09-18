use core::arch::x86_64::__m256i;

use p3_monty_31::{
    add, halve_avx2, mul_2_exp_neg_8_avx2, mul_2_exp_neg_n_avx2, mul_2_exp_neg_two_adicity_avx2,
    mul_neg_2_exp_neg_8_avx2, mul_neg_2_exp_neg_n_avx2, mul_neg_2_exp_neg_two_adicity_avx2,
    signed_add_avx2, sub, InternalLayerParametersAVX2,
};

use crate::{BabyBearInternalLayerParameters, BabyBearParameters};

// Godbolt file showing that these all compile to the expected instructions. (Potentially plus a few memory ops):
// https://godbolt.org/#z:OYLghAFBqd5QCxAYwPYBMCmBRdBLAF1QCcAaPECAMzwBtMA7AQwFtMQByAcQwCNVaBANQFMAZwJihVEkJaoGBAJ5DimdAFdkBPArEA6YmI6kZigMrIm9EAEYALKYUEAqmMwAFAB6cADCYArO1JaJgZQYg0JUgAHMMwAOVZ2bj4BYVEJKQBpVGsmACFMJmIhD1R3PHQFACYTd3ptXQZKRnQAYQENFhb7ADZSNoAZPAZE7t5MYjsADgBOWIrCZs7abt6BmKWdBRGxhImp2YWGzCaFcwISglX1kH7SU/OGS%2Bu98ZZJ6dt5x6viG5dHr3AYSN6jD5fY4ASkeqA0xGQKSimCEaDUIBAAFIAOwFLG%2BACCQhJQhKyCQIC8Mz6AH0%2BvZMbiCg0qKQhLTaSwagBWPp4XEAEVIBOJpLYLExBGIYTELA0ohFRKFWIAzPjlUTpUxCGUSslREYhMzRaS0XphB4AEoASQAstgQEINKqamqNWKSWgGBIyoT2tlsILaR4nZzuXy8MbVYLnT6mFRUcyRDKffLRBAsTyCuZMLQqJjrfbsO6hDMs4LocacYL3abSd7fQAVADqAHlaYTBTb2jamwBNJ1RPAAL0wdaJZsbwjbguDADF/U221ah67o7GILn84XbQ7oyXVdghNuCyBWx2uz2%2B/3oRPCSrNYSJJFtEJcvkiiUPPq2IaxPeop4CwMS0HqMp/lMUgyKUH6hF%2BxA/hBmD/tWHpThaZR7o6zrrmqsY1LYqq%2BDivgMqqqr3hhPrCBenbdr2A5DmIo5JjGQg1PYgE1qKvE1JR2ajLQEIQNYADuTBKGId48rWRJUAwQgINYABu46qu0obgQaUFqtgECjDECphlyvL8lWAC0ekcqZkZofWJIaPGib2ZOZoNphbYJDh4ZmVG%2BFxmICZJniKayummCZtmtiluWslViq6oOe59DCMpO42RG/IbmFaYKpFWYFBAoYgEW%2B5YjUBRCLYVYAPQcbFFZ3klT7uaSqVCPQTASLSvC6gF1J0gymJclltJhOgtIsWZBkMEZBDsl52DNehbWdShIg5YN9KMiAo1mVNxDCbSmAxHgrpMhpMVHrN80rclZodSwkmTLS6VUFtNI7SNLBjSxwAMCdZ2uhA73sl1PV9QQ91uW123DXtv0HUw6CTad501BAC1yC9mBvdYVB3rD1ZyQ%2BPFPhVAkFEJIniZJ0kVqKClCP9YyTSj6BqppTpITpRh6RAtAIGIJlZXg7LEMLot%2BZZ1m%2BXZJrE05QUuYrnopRtWxiLSYzALSMSfUNu37XyU14ADQMY8Vhb%2BoGwYeBLwsw%2Brj0bWI3SG99iNjRzlsg0LYiOwzLUu%2B1bvdLS6KlANX0IybdLu7wfuY%2B7LDslrOuYHrMTO2tQjw8bSOmywoxAy6KfdOyqeRyQxBE%2Brj5k6TfFUzTYyibQElSTJpPM/KtC0jUJ1eDEmd6wwXNabzkFGOy05CAkToY3PmEJCG2FL66AuGcZmUy0IVlHnvCt4slyvBa5ockt17gAidACOYCQAkxqVQv6/FuyJV0VejH9mSUgMa5zWh1DAk0qBMG0CQaWCt2LajyhmQqJVZwLiXCuRq8VALE1dsIZ6YgADWMDsoBXgXKfKUUiq2GjO0LmC9LLVQwZWLBV91ppX8uxAuP0/pHTwMnS67QEjbzmgqYBbUOoqWsLSWgqBPZxyLnSCaZsZo72xng/BoiNbCGkbIwuY1nqo0trYPoEAJED2keyMBtIIFQLrlREBG1pFTQQFQYQMcjZcIOmIWgx10YXWxBpNeZVDz6WkRos0nDvaeI0EnXxmMEDi06qgJxLj65mkbuk5U/FCpt0inTbujN5KKX7mPQew9R66x1pPHmv4UJQRXjRBem8aj1N9IEjeQgMZCPmkQvAssj7y2IafJWzkQqrTajfKYBAH5PwgC/CqVU2mfzKJiH%2BDEbwAI6a6MJYdhCWOsUQaYx9iFwNTGQxB2ZkFzlpIudoy4rSMOdg9HZOMCE9JyqQiKFCIBUK5rQhI9DrpVTikwkOTySQdXiTojxpsxA8L4f4gRXSRF2LERtUxUiZFuK9vHcaDBJrTT5LdBU7I1HbPBQ4zFHDY66IOvotGZ0jEmMkeYoQezIEHLJawxJyTXFUvcZEmF3jeGxP4Ysh0AtQkovchEnFidk6CySWIZx2N4mpNJOk8mD5MmtwYMJdueSGaySZkUjQA8h6YBHiUggYkkkozwMgQgSgqnaRnoHc0DTF6bOae61pH8HRNKRQQHpfTjwDP8kM9W59VYRrzhMu%2BmBH7Py/is9s9FrwDg2UAqVZo41TITTMuZb8xXYHZKqGqzC85PW6oQo57DYwfPIYVb51C/kAoeRW%2BxbCoUCoTnCkVCLBE3RUZyjq2isVyJ9nipRhKVEkurSOilPLu2yqFfCrmxaJWoAXVopJoEohvTrfnal0KFEGNiQq9kqrs2khlfIqa0T5WOL3dreJ4NFXKrVSSDVzdtXZN1bTTu9Me7GrkKakp5rLUVOtba/ADrlDOunrU2ePrhCeuXih9%2BQSA1DuEUG2tIba2XzNFG0ZYKAG3zzYm2Zybzypt/us7qXrOW5umc/V%2BCy/UlqEGWx52DnlqLeSQ05nym0/I0q2g%2BDD1Rliah21FXbx00phX24GNRRWBu3Yk5dd7FEEuMbOl56jr3kp3UuxTJ6pqrv7euzjm7NNPrWC%2Bw9t6fZntUxepSvTjNHv5bKh957HFKpcW%2B/WjmD2fpJrxTVLc/16tyYB/JRrCmgbNWUkp5YNJTxqf%2BFpqHOPYf0io4NB85a2UGWMwKF81axrEBR1jEAZjscw9hGjqz03/0Y1m0FfGTOsoMfs6BhGhPhUbRczEKDrloPudJ4FmnIXmZ7YdHxqn%2BEZcK7h%2BzlLYwudpRzaJ2t0aMoM2ymxm2zN8uxXerxy2MaitszdSVIc847Zhf59zr7uVBehslb9vFf2CX/fqhLhre4moHhUiD5Ss60gy9zF1SG3Xz2LQVoleGw0EbDUR0kJGsfjNq5M%2BrjX5nNaWd/OjayM2da2XJzRfXwHssG5j4bCCCpjZABNm5dz21Pc7Z57T3Cbt%2BK5mt1Hm3%2Be7dRvtwx%2BncMWP6wz2xPP5OfeVeLwVgu1MDvuyErd3mXsJze1bQLyrL1eeJr9p8RleDSEUulNStJ8G8FR8Vw%2Boayvhoq3b9gCK4KFGKIhbLukcN3Siz%2BwkVubcs3NmzcaBjHeCylrWoOLvSti1x1HgG6hY/oH4b7hCiH/wSuFkHeuGTw/RMj8UyHJSeTO/wyV/p7v09V7SxUloPu8jwX9wXupQgeTslsHMDToe/vl%2Bt33MD1eKlzDr%2Bjhvbu0/VfFJP1v0P29czz93wPyGFjVR5MP5U0WiQR4n%2BD6HU/ocy%2B6fX13Q2Y3L7P3rC/49c%2Bd799%2BbfbqBjVRF8OkfluK9T9wNV89YHBZ93cMcm8l8SRikIcQCdZX9Pwt9kIctqpHBqpfAD8m5R8T8wdSkLUoc9ZoNY97VHUHcncisb9U8/Jm8V8CCrUbUSC4MlBECu8P8UDe9OJ2QcQsCy9cCUtgD6CoNGC7VmDyDwCxZIDF978YCwM4ChDodiDRDHVWD38A8ODkMuChAeDg9kVD8w9%2BCW8FC9YZhxDKC59b8mcZCBDn8YdVD89P8B8%2BheCj8x9K85Dz94DTD49zCID5878KtYDPDjC7CO8kD2C%2BY3UjEXCDDAC8DbDTCxAxJRwRx6AJD95LCoDrCcdoD3J54lpaRWxsAEgBxaRzBsAAA1YowTdiHI0KBtc5AoWwGoHQoFJqSLJXWnebC7CdTxFTW7BFX/DbbzUdLbHzS7PRPbXgA7BlK/YlIQAoooko/sMoyo4os7L7NXOkaYqzBlGoGYfhN0B7XXTo8JY9RbOVc9D7Y3FJH7aLVwwwjwp/LwqaZIkcVIyKXwyQ/wqwirOoiraiX0RYlsYo0o8oqoz1JnWokZNCXKM5VnJoloh5DogE55bo7bc42VfooXDSIYkPU455MdHopTOkOlKXQ7OY7GYE0ElY8E9YkYxdTYhbHFHYm7Zog4hFI4nXTlfXe9GJdzG4lVM3Bue4wUDgWEWgTgHkEwFoDgWwEwVATgHgdAfgQQEQcQSQd8N/BCMoCoTAKoWoEwINDgfwaEWEBAYoLAaYCAWEQhPkfQUieYV0TiHEPoBwMtEITgewEwSUWwXwXwUgWU/wUgRUjgeoEAAM40000gOAWAJANAECOgKYcgSgBMs6egaYYgH4f00wOgf8SgXgPwEwPqZgYgFguU4s0YEoJQNsXgAIM4Y0%2BU0gBMtgRQNsXVcs4MrAXgDQYAdoawWgEWCs0gLAZ6cIGwE0kwfANQJoNSIc4Mi1M4fKTgJs0YUQSUyckIPAXgGUMszoLAIs0gaUYCFckwNSYgfgdwQUTAMc4APVUASc2EKgUIYAMQCovATAMSNsGIRgU8pwCwKwGwBwf81wdwbwQ8oIWwEIMICIKIBaYSaY%2BAWEVAGIHYH0TgCyNsA%2BdoEQEoYAFCCyZAIyfCGUVCkgUIfBVECyHClCggCyegNSWgfCVUBU884gKocQeACAZgNgEAdQZYBQUgUxDQFIGoXwQicU2EJ4NC1oPFW4YEZowYPFd4A4T4I4ewGYAMrWAShgeSloRS7StClSw4aYDSgM6S5oV4AEPSuwZpCyi4f4AgYytS0yzSv4a4GyxSsEAEZyqEMyqS%2BEREdgGqT0jgaUwMw80M1QOC5AaqfQTS/QXwIQCAXAQgWQSmGqI0oss00gQhewXwB05ovoGoOYOYHEGYWwWwHkOYPkUK70iKzc0M8MyM7KmMmARAEAeEAgeaFMiANMpM4gJIXijgSICQWK2weKgqhc/AA5KoYIMwAgSwawYKxwBatwTwHwTcyC0gMSUi08iUqUmUyKzgNsBUeaVlD6UaggcayaxK5KzoRMjM1%2BHjLKx820kAGofQewIq/0kq%2BwUquYfoRwDc%2BqoMhUzgZqo87Kg6jgOoBq4MpqqGt6oSqCZoe4IAA%3D%3D%3D

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
        // If there exist other numbers b for which x*b mod P can be computed quickly this diagonal can be updated.

        // The strategy is very simple. 2, 3, 4, -3, -4 are implemented using addition.
        //                              1/2, -1/2 using the custom half function.
        //                              and the remainder utilising the custom functions for multiplication by 2^{-n}.

        // Note that for -3, -4, -1/2 we actually output 3x, 4x, x/2 and the negative is dealt with in add_sum by subtracting
        // this from the summation instead of adding it.

        // Note that input only contains the last 15 elements of the state.
        // The first element is handled seperately as we need to apply the s-box to it.

        // x1 is being multiplied by 1 so we can also ignore it.

        // x2 -> sum + 2*x2
        input[1] = add::<BabyBearParameters>(input[1], input[1]);

        // x3 -> sum + x3/2
        input[2] = halve_avx2::<BabyBearParameters>(input[2]);

        // x4 -> sum + 3*x4
        let acc3 = add::<BabyBearParameters>(input[3], input[3]);
        input[3] = add::<BabyBearParameters>(acc3, input[3]);

        // x5 -> sum + 4*x5
        let acc4 = add::<BabyBearParameters>(input[4], input[4]);
        input[4] = add::<BabyBearParameters>(acc4, acc4);

        // x6 -> sum - x6/2
        input[5] = halve_avx2::<BabyBearParameters>(input[5]);

        // x7 -> sum - 3*x7
        let acc6 = add::<BabyBearParameters>(input[6], input[6]);
        input[6] = add::<BabyBearParameters>(acc6, input[6]);

        // x8 -> sum - 4*x8
        let acc7 = add::<BabyBearParameters>(input[7], input[7]);
        input[7] = add::<BabyBearParameters>(acc7, acc7);

        // x9 -> sum + x9/2**8
        input[8] = mul_2_exp_neg_8_avx2::<BabyBearParameters, 19>(input[8]);

        // x10 -> sum - x10/2**8
        input[9] = mul_neg_2_exp_neg_8_avx2::<BabyBearParameters, 19>(input[9]);

        // x11 -> sum + x11/2**2
        input[10] = mul_2_exp_neg_n_avx2::<BabyBearParameters, 2, 25>(input[10]);

        // x12 -> sum + x12/2**3
        input[11] = mul_2_exp_neg_n_avx2::<BabyBearParameters, 3, 24>(input[11]);

        // x13 -> sum - x13/2**4
        input[12] = mul_neg_2_exp_neg_n_avx2::<BabyBearParameters, 4, 23>(input[12]);

        // x14 -> sum + x14/2**27
        input[13] = mul_2_exp_neg_two_adicity_avx2::<BabyBearParameters, 27, 4>(input[13]);

        // x15 -> sum - x15/2**27
        input[14] = mul_neg_2_exp_neg_two_adicity_avx2::<BabyBearParameters, 27, 4>(input[14]);
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
        input[8..]
            .iter_mut()
            .for_each(|x| *x = signed_add_avx2::<BabyBearParameters>(sum, *x));
    }
}

impl InternalLayerParametersAVX2<24> for BabyBearInternalLayerParameters {
    type ArrayLike = [__m256i; 23];

    /// For the BabyBear field and width 16 we multiply by the diagonal matrix:
    /// D = [-2, 1, 2, 1/2, 3, 4, -1/2, -3, -4, 1/(2**8), -1/(2**8), 1/2**2, -1/2**2, 1/(2**3), -1/(2**3), 1/(2**4), -1/(2**4), -1/(2**5), -1/(2**6), 1/(2**7), -1/(2**7), 1/(2**9), 1/2**27, -1/2**27]
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

        // Note that input only contains the last 23 elements of the state.
        // The first element is handled seperately as we need to apply the s-box to it.

        // x1 is being multiplied by 1 so we can also ignore it.

        // x2 -> sum + 2*x2
        input[1] = add::<BabyBearParameters>(input[1], input[1]);

        // x3 -> sum + x3/2
        input[2] = halve_avx2::<BabyBearParameters>(input[2]);

        // x4 -> sum + 3*x4
        let acc3 = add::<BabyBearParameters>(input[3], input[3]);
        input[3] = add::<BabyBearParameters>(acc3, input[3]);

        // x5 -> sum + 4*x5
        let acc4 = add::<BabyBearParameters>(input[4], input[4]);
        input[4] = add::<BabyBearParameters>(acc4, acc4);

        // x6 -> sum - x6/2
        input[5] = halve_avx2::<BabyBearParameters>(input[5]);

        // x7 -> sum - 3*x7
        let acc6 = add::<BabyBearParameters>(input[6], input[6]);
        input[6] = add::<BabyBearParameters>(acc6, input[6]);

        // x8 -> sum - 4*x8
        let acc7 = add::<BabyBearParameters>(input[7], input[7]);
        input[7] = add::<BabyBearParameters>(acc7, acc7);

        // x9 -> sum + x9/2**8
        input[8] = mul_2_exp_neg_8_avx2::<BabyBearParameters, 19>(input[8]);

        // x10 -> sum - x10/2**8
        input[9] = mul_neg_2_exp_neg_8_avx2::<BabyBearParameters, 19>(input[9]);

        // x11 -> sum + x11/2**2
        input[10] = mul_2_exp_neg_n_avx2::<BabyBearParameters, 2, 25>(input[10]);

        // x12 -> sum - x12/2**2
        input[11] = mul_neg_2_exp_neg_n_avx2::<BabyBearParameters, 2, 25>(input[11]);

        // x13 -> sum + x13/2**3
        input[12] = mul_2_exp_neg_n_avx2::<BabyBearParameters, 3, 24>(input[12]);

        // x14 -> sum - x14/2**3
        input[13] = mul_neg_2_exp_neg_n_avx2::<BabyBearParameters, 3, 24>(input[13]);

        // x15 -> sum + x15/2**4
        input[14] = mul_2_exp_neg_n_avx2::<BabyBearParameters, 4, 23>(input[14]);

        // x16 -> sum - x16/2**4
        input[15] = mul_neg_2_exp_neg_n_avx2::<BabyBearParameters, 4, 23>(input[15]);

        // x17 -> sum - x17/2**5
        input[16] = mul_neg_2_exp_neg_n_avx2::<BabyBearParameters, 5, 22>(input[16]);

        // x18 -> sum - x18/2**6
        input[17] = mul_neg_2_exp_neg_n_avx2::<BabyBearParameters, 6, 21>(input[17]);

        // x19 -> sum + x19/2**7
        input[18] = mul_2_exp_neg_n_avx2::<BabyBearParameters, 7, 20>(input[18]);

        // x20 -> sum - x20/2**7
        input[19] = mul_neg_2_exp_neg_n_avx2::<BabyBearParameters, 7, 20>(input[19]);

        // x21 -> sum + x21/2**9
        input[20] = mul_2_exp_neg_n_avx2::<BabyBearParameters, 9, 18>(input[20]);

        // x22 -> sum - x22/2**27
        input[21] = mul_2_exp_neg_two_adicity_avx2::<BabyBearParameters, 27, 4>(input[21]);

        // x23 -> sum - x23/2**27
        input[22] = mul_neg_2_exp_neg_two_adicity_avx2::<BabyBearParameters, 27, 4>(input[22]);
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
        input[8..]
            .iter_mut()
            .for_each(|x| *x = signed_add_avx2::<BabyBearParameters>(sum, *x));
    }
}

#[cfg(test)]
mod tests {
    use p3_field::AbstractField;
    use p3_symmetric::Permutation;
    use rand::Rng;

    use crate::{BabyBear, PackedBabyBearAVX2, Poseidon2BabyBear};

    type F = BabyBear;
    type Perm16 = Poseidon2BabyBear<16>;
    type Perm24 = Poseidon2BabyBear<24>;

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
