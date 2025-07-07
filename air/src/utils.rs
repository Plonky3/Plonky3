//! A collection of utility functions helpful in defining AIRs.

use core::array;

use p3_field::integers::QuotientMap;
use p3_field::{Field, PrimeCharacteristicRing};

use crate::AirBuilder;

/// Pack a collection of bits into a number.
///
/// Given `vec = [v_0, v_1, ..., v_n]` returns `v_0 + 2v_1 + ... + 2^n v_n`
#[inline]
pub fn pack_bits_le<R, Var, I>(iter: I) -> R
where
    R: PrimeCharacteristicRing,
    Var: Into<R> + Clone,
    I: DoubleEndedIterator<Item = Var>,
{
    iter.rev()
        .map(Into::into)
        .reduce(|acc, elem| acc.double() + elem)
        .unwrap_or(R::ZERO)
}

/// Compute `xor` on a list of boolean field elements.
///
/// Verifies at debug time that all inputs are boolean.
#[inline(always)]
pub fn checked_xor<F: Field, const N: usize>(xs: &[F]) -> F {
    xs.iter().fold(F::ZERO, |acc, x| {
        debug_assert!(x.is_zero() || x.is_one());
        acc.xor(x)
    })
}

/// Compute `andnot` on a pair of boolean field elements.
///
/// Verifies at debug time that both inputs are boolean.
#[inline(always)]
pub fn checked_andn<F: Field>(x: F, y: F) -> F {
    debug_assert!(x.is_zero() || x.is_one());
    debug_assert!(y.is_zero() || y.is_one());
    x.andn(&y)
}

/// Convert a 32-bit integer into an array of 32 0 or 1 field elements.
///
/// The output array is in little-endian order.
#[inline]
pub fn u32_to_bits_le<R: PrimeCharacteristicRing>(val: u32) -> [R; 32] {
    array::from_fn(|i| R::from_bool(val & (1 << i) != 0))
}

/// Convert a 64-bit integer into an array of 64 0 or 1 field elements.
///
/// The output array is in little-endian order.
#[inline]
pub fn u64_to_bits_le<R: PrimeCharacteristicRing>(val: u64) -> [R; 64] {
    array::from_fn(|i| R::from_bool(val & (1 << i) != 0))
}

/// Convert a 64-bit integer into an array of four field elements representing the 16 bit limb decomposition.
///
/// The output array is in little-endian order.
#[inline]
pub fn u64_to_16_bit_limbs<R: PrimeCharacteristicRing>(val: u64) -> [R; 4] {
    array::from_fn(|i| R::from_u16((val >> (16 * i)) as u16))
}

/// Verify that `a = b + c + d mod 2^32`
///
/// We assume that a, b, c, d are all given as `2, 16` bit limbs (e.g. `a = a[0] + 2^16 a[1]`) and
/// each `16` bit limb has been range checked to ensure it contains a value in `[0, 2^16)`.
///
/// This function assumes we are working over a field with characteristic `P > 3*2^16`.
///
/// # Panics
///
/// The function will panic if the characteristic of the field is less than or equal to 2^16.
#[inline]
pub fn add3<AB: AirBuilder>(
    builder: &mut AB,
    a: &[AB::Var; 2],
    b: &[AB::Var; 2],
    c: &[AB::Expr; 2],
    d: &[AB::Expr; 2],
) {
    // Define:
    //  acc    = a - b - c - d (mod P)
    //  acc_16 = a[0] - b[0] - c[0] - d[0] (mod P)
    //
    // We perform 2 checks:
    //
    // (1) acc*(acc + 2^32)*(acc + 2*2^32) = 0.
    // (2) acc_16*(acc_16 + 2^16)*(acc_16 + 2*2^16) = 0.
    //
    // We give a short proof for why this lets us conclude that a = b + c + d mod 2^32:
    //
    // As all 16 bit limbs have been range checked, we know that a, b, c, d lie in [0, 2^32) and hence
    // a = b + c + d mod 2^32 if and only if, over the integers, a - b - c - d = 0, -2^32 or -2*2^32.
    //
    // Equation (1) verifies that a - b - c - d mod P = 0, -2^32 or -2*2^32.
    //
    // Field overflow cannot occur when computing acc_16 as our characteristic is larger than 3*2^16.
    // Hence, equation (2) verifies that, over the integers, a[0] - b[0] - c[0] - d[0] = 0, -2^16 or -2*2^16.
    // Either way we can immediately conclude that a - b - c - d = 0 mod 2^16.
    //
    // Now we can use the chinese remainder theorem to combine these results to conclude that
    // a - b - c - d mod 2^16P = 0, -2^32 or -2*2^32.
    //
    // No overflow can occur mod 2^16 P as 2^16 P > 3*2^32 and a, b, c, d < 2^32. Hence we conclude that
    // over the integers a - b - c - d = 0, -2^32 or -2*2^32 which implies a = b + c + d mod 2^32.

    // By assumption P > 3*2^16 so 1 << 16 will be less than P. We use the checked version just to be safe.
    // The compiler should optimize it away.
    let two_16 =
        <AB::Expr as PrimeCharacteristicRing>::PrimeSubfield::from_canonical_checked(1 << 16)
            .unwrap();
    let two_32 = two_16.square();

    let acc_16 = a[0] - b[0] - c[0].clone() - d[0].clone();
    let acc_32 = a[1] - b[1] - c[1].clone() - d[1].clone();
    let acc = acc_16.clone() + acc_32.mul_2exp_u64(16);

    builder.assert_zeros([
        acc.clone()
            * (acc.clone() + AB::Expr::from_prime_subfield(two_32))
            * (acc + AB::Expr::from_prime_subfield(two_32.double())),
        acc_16.clone()
            * (acc_16.clone() + AB::Expr::from_prime_subfield(two_16))
            * (acc_16 + AB::Expr::from_prime_subfield(two_16.double())),
    ]);
}

/// Verify that `a = b + c mod 2^32`
///
/// We assume that a, b, c are all given as `2, 16` bit limbs (e.g. `a = a[0] + 2^16 a[1]`) and
/// each `16` bit limb has been range checked to ensure it contains a value in `[0, 2^16)`.
///
/// This function assumes we are working over a field with characteristic `P > 2^17`.
///
/// # Panics
///
/// The function will panic if the characteristic of the field is less than or equal to 2^16.
#[inline]
pub fn add2<AB: AirBuilder>(
    builder: &mut AB,
    a: &[AB::Var; 2],
    b: &[AB::Var; 2],
    c: &[AB::Expr; 2],
) {
    // Define:
    //  acc    = a - b - c (mod P)
    //  acc_16 = a[0] - b[0] - c[0] (mod P)
    //
    // We perform 2 checks:
    //
    // (1) acc*(acc + 2^32) = 0.
    // (2) acc_16*(acc_16 + 2^16) = 0.
    //
    // We give a short proof for why this lets us conclude that a = b + c mod 2^32:
    //
    // As all 16 bit limbs have been range checked, we know that a, b, c lie in [0, 2^32) and hence
    // a = b + c mod 2^32 if and only if, over the integers, a - b - c = 0 or -2^32.
    //
    // Equation (1) verifies that either a - b - c = 0 mod P or a - b - c = -2^32 mod P.
    //
    // Field overflow cannot occur when computing acc_16 as our characteristic is larger than 2^17.
    // Hence, equation (2) verifies that, over the integers, a[0] - b[0] - c[0] = 0 or -2^16.
    // Either way we can immediately conclude that a - b - c = 0 mod 2^16.
    //
    // Now we can use the chinese remainder theorem to combine these results to conclude that
    // either a - b - c = 0 mod 2^16 P or a - b - c = -2^32 mod 2^16 P.
    //
    // No overflow can occur mod 2^16 P as 2^16 P > 2^33 and a, b, c < 2^32. Hence we conclude that
    // over the integers a - b - c = 0 or a - b - c = -2^32 which is equivalent to a = b + c mod 2^32.

    // By assumption P > 2^17 so 1 << 16 will be less than P. We use the checked version just to be safe.
    // The compiler should optimize it away.
    let two_16 =
        <AB::Expr as PrimeCharacteristicRing>::PrimeSubfield::from_canonical_checked(1 << 16)
            .unwrap();
    let two_32 = two_16.square();

    let acc_16 = a[0] - b[0] - c[0].clone();
    let acc_32 = a[1] - b[1] - c[1].clone();
    let acc = acc_16.clone() + acc_32.mul_2exp_u64(16);

    builder.assert_zeros([
        acc.clone() * (acc + AB::Expr::from_prime_subfield(two_32)),
        acc_16.clone() * (acc_16 + AB::Expr::from_prime_subfield(two_16)),
    ]);
}

/// Verify that `a = (b ^ (c << shift))`
///
/// We assume that a is given as `2 16` bit limbs and both b and c are unpacked into 32 individual bits.
/// We assume that the bits of b have been range checked but not the inputs in c or a. Both of these are
/// range checked as part of this function.
#[inline]
pub fn xor_32_shift<AB: AirBuilder>(
    builder: &mut AB,
    a: &[AB::Var; 2],
    b: &[AB::Var; 32],
    c: &[AB::Var; 32],
    shift: usize,
) {
    // First we range check all elements of c.
    builder.assert_bools(*c);

    // Next we compute (b ^ (c << shift)) and pack the result into two 16-bit integers.
    let xor_shift_c_0_16 = b[..16]
        .iter()
        .enumerate()
        .map(|(i, elem)| (*elem).into().xor(&c[(32 + i - shift) % 32].into()));
    let sum_0_16: AB::Expr = pack_bits_le(xor_shift_c_0_16);

    let xor_shift_c_16_32 = b[16..]
        .iter()
        .enumerate()
        .map(|(i, elem)| (*elem).into().xor(&c[(32 + (i + 16) - shift) % 32].into()));
    let sum_16_32: AB::Expr = pack_bits_le(xor_shift_c_16_32);

    // As both b and c have been range checked to be boolean, all the (b ^ (c << shift))
    // are also boolean and so this final check additionally has the effect of range checking a[0], a[1].
    builder.assert_zeros([a[0] - sum_0_16, a[1] - sum_16_32]);
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_baby_bear::BabyBear;

    use super::*;

    type F = BabyBear;

    #[test]
    fn test_pack_bits_le_various_patterns() {
        // Pattern: [1, 0, 1] as little-endian => 1 + 2*0 + 4*1 = 5
        let bits = [F::ONE, F::ZERO, F::ONE];
        let packed = pack_bits_le::<F, _, _>(bits.iter().cloned());
        assert_eq!(packed, F::from_u8(5));

        // Pattern: [1, 1, 0, 1] => 1 + 2*1 + 4*0 + 8*1 = 1 + 2 + 8 = 11
        let bits = [F::ONE, F::ONE, F::ZERO, F::ONE];
        let packed = pack_bits_le::<F, _, _>(bits.iter().cloned());
        assert_eq!(packed, F::from_u8(11));

        // Pattern: all zeros
        let bits = [F::ZERO; 5];
        let packed = pack_bits_le::<F, _, _>(bits.iter().cloned());
        assert_eq!(packed, F::ZERO);

        // Pattern: single one at the highest place
        let bits = [F::ZERO, F::ZERO, F::ZERO, F::ZERO, F::ONE];
        let packed = pack_bits_le::<F, _, _>(bits.iter().cloned());
        assert_eq!(packed, F::from_u8(16));
    }

    #[test]
    fn test_checked_xor_multiple_cases() {
        // Input: [1, 0, 1] => XOR(1 ^ 0 ^ 1) = 0
        let bits = vec![F::ONE, F::ZERO, F::ONE];
        let result = checked_xor::<F, 3>(&bits);
        assert_eq!(result, F::ZERO);

        // [1, 1, 1] => XOR = 1 ^ 1 ^ 1 = 1
        let bits = vec![F::ONE, F::ONE, F::ONE];
        let result = checked_xor::<F, 3>(&bits);
        assert_eq!(result, F::ONE);

        // [0, 0, 0] => XOR = 0
        let bits = vec![F::ZERO, F::ZERO, F::ZERO];
        let result = checked_xor::<F, 3>(&bits);
        assert_eq!(result, F::ZERO);

        // [1, 0, 1, 0] => XOR = 1 ^ 0 ^ 1 ^ 0 = 0
        let bits = vec![F::ONE, F::ZERO, F::ONE, F::ZERO];
        let result = checked_xor::<F, 4>(&bits);
        assert_eq!(result, F::ZERO);
    }

    #[test]
    fn test_checked_andn() {
        // x = 1, y = 0 => 1 & !0 = 0
        let result = checked_andn(F::ONE, F::ZERO);
        assert_eq!(result, F::ZERO);

        // x = 0, y = 1 => 0 & !1 = 1
        let result = checked_andn(F::ZERO, F::ONE);
        assert_eq!(result, F::ONE);

        // x = 0, y = 0 => 0 & !0 = 0
        let result = checked_andn(F::ZERO, F::ZERO);
        assert_eq!(result, F::ZERO);

        // x = 1, y = 1 => 1 & !1 = 0
        let result = checked_andn(F::ONE, F::ONE);
        assert_eq!(result, F::ZERO);
    }

    #[test]
    fn test_u32_to_bits_le() {
        // Convert 0b1010 (decimal 10) => [0, 1, 0, 1, ...]
        let bits = u32_to_bits_le::<F>(10);
        assert_eq!(bits[0], F::ZERO); // LSB first
        assert_eq!(bits[1], F::ONE);
        assert_eq!(bits[2], F::ZERO);
        assert_eq!(bits[3], F::ONE);

        for &bit in &bits[4..] {
            assert_eq!(bit, F::ZERO);
        }

        // Check 0 => all zeros
        let bits = u32_to_bits_le::<F>(0);
        assert!(bits.iter().all(|b| *b == F::ZERO));

        // Check max => all ones
        let bits = u32_to_bits_le::<F>(u32::MAX);
        assert!(bits.iter().all(|b| *b == F::ONE));
    }

    #[test]
    fn test_u64_to_bits_le() {
        // Convert 0b11 (decimal 3) => [1, 1, 0, ...]
        let bits = u64_to_bits_le::<F>(3);
        assert_eq!(bits[0], F::ONE);
        assert_eq!(bits[1], F::ONE);
        assert_eq!(bits[2], F::ZERO);

        for &bit in &bits[3..] {
            assert_eq!(bit, F::ZERO);
        }

        // Check 0 => all zeros
        let bits = u64_to_bits_le::<F>(0);
        assert!(bits.iter().all(|b| *b == F::ZERO));

        // Check max => all ones
        let bits = u64_to_bits_le::<F>(u64::MAX);
        assert!(bits.iter().all(|b| *b == F::ONE));
    }

    #[test]
    fn test_u64_to_16_bit_limbs() {
        // Convert 0x123456789ABCDEF0
        let val: u64 = 0x123456789ABCDEF0;
        let limbs = u64_to_16_bit_limbs::<F>(val);

        // Expected limbs (little endian): [0xDEF0, 0x9ABC, 0x5678, 0x1234]
        assert_eq!(limbs[0], F::from_u16(0xDEF0));
        assert_eq!(limbs[1], F::from_u16(0x9ABC));
        assert_eq!(limbs[2], F::from_u16(0x5678));
        assert_eq!(limbs[3], F::from_u16(0x1234));

        assert_eq!(
            limbs[0]
                + limbs[1].mul_2exp_u64(16)
                + limbs[2].mul_2exp_u64(32)
                + limbs[3].mul_2exp_u64(48),
            F::from_u64(val)
        );

        // Check zero
        let limbs = u64_to_16_bit_limbs::<F>(0);
        assert!(limbs.iter().all(|l| *l == F::ZERO));

        // Check max
        let limbs = u64_to_16_bit_limbs::<F>(u64::MAX);
        for l in limbs {
            assert_eq!(l, F::from_u64(0xFFFF));
        }

        // Check small value
        let val: u64 = 0x1234;
        let limbs = u64_to_16_bit_limbs::<F>(val);
        assert_eq!(limbs[0], F::from_u64(0x1234));
        assert!(limbs[1..].iter().all(|l| *l == F::ZERO));
    }
}
