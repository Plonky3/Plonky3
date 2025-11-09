use crate::{FieldParameters, MontyParameters};

/// Convert a u32 into MONTY form.
/// There are no constraints on the input.
/// The output will be a u32 in range `[0, P)`.
#[inline]
pub(crate) const fn to_monty<MP: MontyParameters>(x: u32) -> u32 {
    (((x as u64) << MP::MONTY_BITS) % MP::PRIME as u64) as u32
}

/// Convert an i32 into MONTY form.
/// There are no constraints on the input.
/// The output will be a u32 in range `[0, P)`.
#[inline]
pub(crate) const fn to_monty_signed<MP: MontyParameters>(x: i32) -> u32 {
    let red = (((x as i64) << MP::MONTY_BITS) % MP::PRIME as i64) as i32;
    if red >= 0 {
        red as u32
    } else {
        MP::PRIME.wrapping_add_signed(red)
    }
}

/// Convert a u64 into MONTY form.
/// There are no constraints on the input.
/// The output will be a u32 in range `[0, P)`.
#[inline]
pub(crate) const fn to_monty_64<MP: MontyParameters>(x: u64) -> u32 {
    (((x as u128) << MP::MONTY_BITS) % MP::PRIME as u128) as u32
}

/// Convert an i64 into MONTY form.
/// There are no constraints on the input.
/// The output will be a u32 in range `[0, P)`.
#[inline]
pub(crate) const fn to_monty_64_signed<MP: MontyParameters>(x: i64) -> u32 {
    let red = (((x as i128) << MP::MONTY_BITS) % MP::PRIME as i128) as i32;
    if red >= 0 {
        red as u32
    } else {
        MP::PRIME.wrapping_add_signed(red)
    }
}

/// Convert a u32 out of MONTY form.
/// There are no constraints on the input.
/// The output will be a u32 in range `[0, P)`.
#[inline]
#[must_use]
pub(crate) const fn from_monty<MP: MontyParameters>(x: u32) -> u32 {
    monty_reduce::<MP>(x as u64)
}

/// Add two integers modulo `P = MP::PRIME`.
///
/// Assumes that `P` is less than `2^31` and `a + b <= 2P` for all array pairs `a, b`.
/// If the inputs are not in this range, the result may be incorrect.
/// The result will be in the range `[0, P]` and equal to `(a + b) mod P`.
/// It will be equal to `P` if and only if `a + b = 2P` so provided `a + b < 2P`
/// the result is guaranteed to be less than `P`.
#[inline]
#[must_use]
pub(crate) const fn add<MP: MontyParameters>(lhs: u32, rhs: u32) -> u32 {
    let mut sum = lhs + rhs;
    let (corr_sum, over) = sum.overflowing_sub(MP::PRIME);
    if !over {
        sum = corr_sum;
    }
    sum
}

/// Subtract two integers modulo `P = MP::PRIME`.
///
/// Assumes that `P` is less than `2^31` and `|a - b| <= P` for all array pairs `a, b`.
/// If the inputs are not in this range, the result may be incorrect.
/// The result will be in the range `[0, P]` and equal to `(a - b) mod P`.
/// It will be equal to `P` if and only if `a - b = P` so provided `a - b < P`
/// the result is guaranteed to be less than `P`.
#[inline]
#[must_use]
pub(crate) const fn sub<MP: MontyParameters>(lhs: u32, rhs: u32) -> u32 {
    let (mut diff, over) = lhs.overflowing_sub(rhs);
    let corr = if over { MP::PRIME } else { 0 };
    diff = diff.wrapping_add(corr);
    diff
}

/// Given an element `x` from a 31 bit field `F` compute `x/2`.
/// The input must be in `[0, P)`.
/// The output will also be in `[0, P)`.
#[inline]
pub(crate) const fn halve_u32<FP: FieldParameters>(input: u32) -> u32 {
    let shr = input >> 1;
    let lo_bit = input & 1;
    let shr_corr = shr + FP::HALF_P_PLUS_1;
    if lo_bit == 0 { shr } else { shr_corr }
}

/// Montgomery reduction of a value in `0..P << MONTY_BITS`.
///
/// The input must be in `[0, MONTY * P)`.
/// The output will be in `[0, P)`.
#[inline]
#[must_use]
pub(crate) const fn monty_reduce<MP: MontyParameters>(x: u64) -> u32 {
    // t = x * MONTY_MU mod MONTY
    let t = x.wrapping_mul(MP::MONTY_MU as u64) & (MP::MONTY_MASK as u64);

    // u = t * P
    let u = t * (MP::PRIME as u64);

    // Thus:
    // 1. x - u = x - t * P = x mod P
    // 2. x - u = x - x * MONTY_MU * P mod MONTY = 0 mod MONTY
    // For the second point note that MONTY_MU = P^{-1} mod MONTY.

    // Additionally, u < MONTY * P so: - MONTY * P < x - u < MONTY * P
    // Thus after dividing by MONTY, -P < (x - u)/MONTY < P.
    // So we can just add P to the result if it is negative.

    let (x_sub_u, over) = x.overflowing_sub(u);
    let x_sub_u_hi = (x_sub_u >> MP::MONTY_BITS) as u32;
    let corr = if over { MP::PRIME } else { 0 };
    x_sub_u_hi.wrapping_add(corr)
}

/// Montgomery reduction of a value in `0..P << MONTY_BITS`.
/// The input must be in [0, 2 * MONTY * P).
/// The output will be in [0, P).
///
/// This is slower than `monty_reduce` but has a larger input range.
#[inline]
#[must_use]
pub(crate) const fn large_monty_reduce<MP: MontyParameters>(x: u64) -> u32 {
    // t = x * MONTY_MU mod MONTY
    let t = x.wrapping_mul(MP::MONTY_MU as u64) & (MP::MONTY_MASK as u64);

    // u = t * P
    let u = t * (MP::PRIME as u64);

    // Thus:
    // 1. x - u = x - t * P = x mod P
    // 2. x - u = x - x * MONTY_MU * P mod MONTY = 0 mod MONTY
    // For the second point note that MONTY_MU = P^{-1} mod MONTY.

    // This time, - MONTY * P < x - u < 2 * MONTY * P so we need to be
    // more careful with our reduction.
    // The trick is just to first reduce x to lie in [0, MONTY * P).
    let (x_prime, over) = x.overflowing_sub((MP::PRIME as u64) << MP::MONTY_BITS);
    let x_corr = if over { x } else { x_prime };

    // Now we can do the same as before.

    let (x_sub_u, over) = x_corr.overflowing_sub(u);
    let x_sub_u_hi = (x_sub_u >> MP::MONTY_BITS) as u32;
    let corr = if over { MP::PRIME } else { 0 };
    x_sub_u_hi.wrapping_add(corr)
}

/// Perform a monty reduction on a u128 in the range `[0, 2^96)`
///
/// The input will be in `[0, P)` and be equal to `x * MONTY^{-1} mod P`.
pub(crate) const fn monty_reduce_u128<MP: MontyParameters>(x: u128) -> u32 {
    // TODO: There is probably a way to do this faster than using %.

    // Need to find MONTY^{-1} mod P.
    // As P * MONTY_MU = 1 mod MONTY, we know that P * MONTY_MU = 1 + k * MONTY for some k.
    // Thus k * MONTY = -1 mod P.
    // Rearranging, we get k = (P * MONTY_MU - 1) / MONTY.
    // Thus we want -k = P - k = P - (P * MONTY_MU - 1) / MONTY.

    // Compiler should realize that this is a constant.
    let monty_inv_mod_p =
        MP::PRIME - ((((MP::PRIME as u64) * (MP::MONTY_MU as u64)) - 1) >> MP::MONTY_BITS) as u32;

    // As monty_inv_mod_p < 2^32, x * monty_inv_mod_p < 2^128 so the product below will not overflow.
    ((x * (monty_inv_mod_p as u128)) % (MP::PRIME as u128)) as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FieldParameters, MontyParameters};

    // Test type implementing MontyParameters
    // Using the same prime as BabyBear: 2^31 - 2^27 + 1 = 0x78000001 = 2013265921
    // MONTY_MU = PRIME^-1 (mod 2^32) = 0x88000001
    #[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
    struct TestMontyParams;

    impl MontyParameters for TestMontyParams {
        // Using the same prime as BabyBear: 2^31 - 2^27 + 1 = 0x78000001 = 2013265921
        const PRIME: u32 = 0x78000001;
        const MONTY_BITS: u32 = 32;
        // MONTY_MU = PRIME^-1 mod 2^32
        // For PRIME = 0x78000001, the modular inverse mod 2^32 is 0x88000001
        // This is the same value used in BabyBear field
        const MONTY_MU: u32 = 0x88000001;
    }

    // PackedMontyParameters is just MontyParameters on non-SIMD platforms
    impl crate::PackedMontyParameters for TestMontyParams {}

    impl FieldParameters for TestMontyParams {
        const MONTY_GEN: crate::MontyField31<Self> = crate::MontyField31::new(7);
    }

    type TestMP = TestMontyParams;
    type TestFP = TestMontyParams;

    #[test]
    fn test_to_monty_basic() {
        let x = 12345u32;
        let monty = to_monty::<TestMP>(x);
        // Check that monty is in range [0, P)
        assert!(monty < TestMP::PRIME);
        // Check that monty = x * 2^MONTY_BITS mod P
        let expected = ((x as u64) << TestMP::MONTY_BITS) % (TestMP::PRIME as u64);
        assert_eq!(monty as u64, expected);
    }

    #[test]
    fn test_to_monty_zero() {
        let monty = to_monty::<TestMP>(0);
        assert_eq!(monty, 0);
    }

    #[test]
    fn test_to_monty_roundtrip() {
        for &x in &[0, 1, 100, 1000, TestMP::PRIME - 1] {
            let monty = to_monty::<TestMP>(x);
            let back = from_monty::<TestMP>(monty);
            assert_eq!(x, back, "Roundtrip failed for x = {}", x);
        }
    }

    #[test]
    fn test_to_monty_signed_positive() {
        let x = 12345i32;
        let monty = to_monty_signed::<TestMP>(x);
        assert!(monty < TestMP::PRIME);
        // For positive values, should match unsigned version
        let monty_unsigned = to_monty::<TestMP>(x as u32);
        assert_eq!(monty, monty_unsigned);
    }

    #[test]
    fn test_to_monty_signed_negative() {
        let x = -12345i32;
        let monty = to_monty_signed::<TestMP>(x);
        assert!(monty < TestMP::PRIME);
        // Negative value should be converted to positive mod P
        let expected = (TestMP::PRIME as i64 + x as i64) % TestMP::PRIME as i64;
        let back = from_monty::<TestMP>(monty);
        assert_eq!(back as i64, expected as u32 as i64);
    }

    #[test]
    fn test_to_monty_signed_zero() {
        let monty = to_monty_signed::<TestMP>(0);
        assert_eq!(monty, 0);
    }

    #[test]
    fn test_to_monty_64() {
        let x = 123456789u64;
        let monty = to_monty_64::<TestMP>(x);
        assert!(monty < TestMP::PRIME);
        // Check that monty = x * 2^MONTY_BITS mod P
        let expected = ((x as u128) << TestMP::MONTY_BITS) % (TestMP::PRIME as u128);
        assert_eq!(monty as u128, expected);
    }

    #[test]
    fn test_to_monty_64_large() {
        let x = u64::MAX;
        let monty = to_monty_64::<TestMP>(x);
        assert!(monty < TestMP::PRIME);
    }

    #[test]
    fn test_to_monty_64_signed_positive() {
        let x = 123456789i64;
        let monty = to_monty_64_signed::<TestMP>(x);
        assert!(monty < TestMP::PRIME);
    }

    #[test]
    fn test_to_monty_64_signed_negative() {
        let x = -123456789i64;
        let monty = to_monty_64_signed::<TestMP>(x);
        assert!(monty < TestMP::PRIME);
    }

    #[test]
    fn test_from_monty() {
        // Test that from_monty correctly converts back
        for &x in &[0, 1, 100, 1000, TestMP::PRIME - 1] {
            let monty = to_monty::<TestMP>(x);
            let back = from_monty::<TestMP>(monty);
            assert_eq!(x, back, "from_monty failed for x = {}", x);
        }
    }

    #[test]
    fn test_add_modular() {
        let a = 100u32;
        let b = 200u32;
        let result = add::<TestMP>(a, b);
        let expected = (a + b) % TestMP::PRIME;
        assert_eq!(result, expected);
    }

    #[test]
    fn test_add_with_overflow() {
        // Test case where a + b >= P
        let a = TestMP::PRIME - 100;
        let b = 200u32;
        let result = add::<TestMP>(a, b);
        let expected = (a + b) % TestMP::PRIME;
        assert_eq!(result, expected);
    }

    #[test]
    fn test_add_zero() {
        let a = 12345u32;
        assert_eq!(add::<TestMP>(a, 0), a);
        assert_eq!(add::<TestMP>(0, a), a);
        assert_eq!(add::<TestMP>(0, 0), 0);
    }

    #[test]
    fn test_sub_modular() {
        let a = 200u32;
        let b = 100u32;
        let result = sub::<TestMP>(a, b);
        let expected = (a + TestMP::PRIME - b) % TestMP::PRIME;
        assert_eq!(result, expected);
    }

    #[test]
    fn test_sub_with_underflow() {
        // Test case where a < b
        let a = 100u32;
        let b = 200u32;
        let result = sub::<TestMP>(a, b);
        let expected = (a + TestMP::PRIME - b) % TestMP::PRIME;
        assert_eq!(result, expected);
    }

    #[test]
    fn test_sub_zero() {
        let a = 12345u32;
        assert_eq!(sub::<TestMP>(a, 0), a);
        assert_eq!(sub::<TestMP>(0, 0), 0);
    }

    #[test]
    fn test_sub_identity() {
        let a = 12345u32;
        let result = sub::<TestMP>(a, a);
        assert_eq!(result, 0);
    }

    #[test]
    fn test_halve_u32_even() {
        let x = 100u32;
        let half = halve_u32::<TestFP>(x);
        // Check that (half * 2) % P == x
        let doubled = (half * 2) % TestFP::PRIME;
        assert_eq!(doubled, x);
    }

    #[test]
    fn test_halve_u32_odd() {
        let x = 101u32;
        let half = halve_u32::<TestFP>(x);
        // Check that (half * 2) % P == x
        let doubled = (half * 2) % TestFP::PRIME;
        assert_eq!(doubled, x);
    }

    #[test]
    fn test_halve_u32_zero() {
        let half = halve_u32::<TestFP>(0);
        assert_eq!(half, 0);
    }

    #[test]
    fn test_halve_u32_one() {
        let half = halve_u32::<TestFP>(1);
        // For odd numbers, halve should add (P+1)/2
        let expected = (TestFP::PRIME + 1) >> 1;
        assert_eq!(half, expected);
    }

    #[test]
    fn test_monty_reduce() {
        // Test monty_reduce with a value in monty form
        let x_monty = to_monty::<TestMP>(12345);
        let x_u64 = x_monty as u64;
        let reduced = monty_reduce::<TestMP>(x_u64);
        // The result should be the original value
        assert_eq!(reduced, 12345);
    }

    #[test]
    fn test_monty_reduce_zero() {
        let reduced = monty_reduce::<TestMP>(0);
        assert_eq!(reduced, 0);
    }

    #[test]
    fn test_monty_reduce_roundtrip() {
        for &x in &[0, 1, 100, 1000, TestMP::PRIME - 1] {
            let monty = to_monty::<TestMP>(x);
            let monty_u64 = monty as u64;
            let reduced = monty_reduce::<TestMP>(monty_u64);
            assert_eq!(reduced, x, "monty_reduce roundtrip failed for x = {}", x);
        }
    }

    #[test]
    fn test_large_monty_reduce() {
        // Test large_monty_reduce with a value that might be >= MONTY * P
        let x = 12345u32;
        let monty = to_monty::<TestMP>(x);
        // Create a value that's in the range [MONTY * P, 2 * MONTY * P)
        let large_value = (monty as u64) + ((TestMP::PRIME as u64) << TestMP::MONTY_BITS);
        let reduced = large_monty_reduce::<TestMP>(large_value);
        assert_eq!(reduced, x);
    }

    #[test]
    fn test_large_monty_reduce_zero() {
        let reduced = large_monty_reduce::<TestMP>(0);
        assert_eq!(reduced, 0);
    }

    #[test]
    fn test_large_monty_reduce_roundtrip() {
        for &x in &[0, 1, 100, 1000, TestMP::PRIME - 1] {
            let monty = to_monty::<TestMP>(x);
            let monty_u64 = monty as u64;
            // Test with value in normal range
            let reduced = large_monty_reduce::<TestMP>(monty_u64);
            assert_eq!(
                reduced, x,
                "large_monty_reduce roundtrip failed for x = {}",
                x
            );
        }
    }

    #[test]
    fn test_monty_reduce_u128() {
        // Test monty_reduce_u128
        let x = 12345u32;
        let monty = to_monty::<TestMP>(x);
        let monty_u128 = monty as u128;
        let reduced = monty_reduce_u128::<TestMP>(monty_u128);
        assert_eq!(reduced, x);
    }

    #[test]
    fn test_monty_reduce_u128_zero() {
        let reduced = monty_reduce_u128::<TestMP>(0);
        assert_eq!(reduced, 0);
    }

    #[test]
    fn test_monty_reduce_u128_large() {
        // Test with a large u128 value
        let x = 12345u32;
        let monty = to_monty::<TestMP>(x);
        // Create a large u128 value
        let large_value = (monty as u128) << 64;
        let reduced = monty_reduce_u128::<TestMP>(large_value);
        // Should still reduce correctly
        assert!(reduced < TestMP::PRIME);
    }

    #[test]
    fn test_monty_arithmetic_consistency() {
        // Test that monty arithmetic is consistent
        let a = 12345u32;
        let b = 67890u32;

        let a_monty = to_monty::<TestMP>(a);
        let b_monty = to_monty::<TestMP>(b);

        // Test addition: (a + b) mod P
        let sum_monty = add::<TestMP>(a_monty, b_monty);
        let sum_expected = (a + b) % TestMP::PRIME;
        let sum_back = from_monty::<TestMP>(sum_monty);
        assert_eq!(sum_back, sum_expected);

        // Test subtraction: (a - b) mod P
        let diff_monty = sub::<TestMP>(a_monty, b_monty);
        let diff_expected = (a + TestMP::PRIME - b) % TestMP::PRIME;
        let diff_back = from_monty::<TestMP>(diff_monty);
        assert_eq!(diff_back, diff_expected);
    }

    #[test]
    fn test_monty_multiplication_consistency() {
        // Test that monty multiplication works correctly
        let a = 123u32;
        let b = 456u32;

        let a_monty = to_monty::<TestMP>(a);
        let b_monty = to_monty::<TestMP>(b);

        // Multiply in monty form: (a_monty * b_monty) = (a * R) * (b * R) = a * b * R^2 mod P
        let prod_monty_u64 = (a_monty as u64) * (b_monty as u64);
        // After monty_reduce, we get: (a * b * R^2) / R = a * b * R mod P (still in monty form)
        let prod_reduced_monty = monty_reduce::<TestMP>(prod_monty_u64);
        // Convert back to normal form to get (a * b) mod P
        let prod_normal = from_monty::<TestMP>(prod_reduced_monty);

        // Expected result: (a * b) mod P
        let prod_expected = ((a as u64) * (b as u64)) % (TestMP::PRIME as u64);
        assert_eq!(prod_normal as u64, prod_expected);
    }
}
