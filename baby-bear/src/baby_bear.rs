use core::fmt::{self, Debug, Display, Formatter};
use core::iter::{Product, Sum};
use core::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

use p3_field::{
    exp_1725656503, exp_u64_by_squaring, halve_u32, AbstractField, Field, Packable, PrimeField,
    PrimeField32, PrimeField64, TwoAdicField,
};
use rand::distributions::{Distribution, Standard};
use rand::Rng;
use serde::{Deserialize, Deserializer, Serialize};

/// The Baby Bear prime
const P: u32 = 0x78000001;
const MONTY_BITS: u32 = 32;
// We are defining MU = P^-1 (mod 2^MONTY_BITS). This is different from the usual convention
// (MU = -P^-1 (mod 2^MONTY_BITS)) but it avoids a carry.
const MONTY_MU: u32 = 0x88000001;

// This is derived from above.
const MONTY_MASK: u32 = ((1u64 << MONTY_BITS) - 1) as u32;

/// The prime field `2^31 - 2^27 + 1`, a.k.a. the Baby Bear field.
#[derive(Copy, Clone, Default, Eq, Hash, PartialEq)]
#[repr(transparent)] // `PackedBabyBearNeon` relies on this!
pub struct BabyBear {
    // This is `pub(crate)` just for tests. If you're accessing `value` outside of those, you're
    // likely doing something fishy.
    pub(crate) value: u32,
}

impl BabyBear {
    /// create a new `BabyBear` from a canonical `u32`.
    #[inline]
    pub(crate) const fn new(n: u32) -> Self {
        Self { value: to_monty(n) }
    }
}

impl Ord for BabyBear {
    #[inline]
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.as_canonical_u32().cmp(&other.as_canonical_u32())
    }
}

impl PartialOrd for BabyBear {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Display for BabyBear {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Display::fmt(&self.as_canonical_u32(), f)
    }
}

impl Debug for BabyBear {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Debug::fmt(&self.as_canonical_u32(), f)
    }
}

impl Distribution<BabyBear> for Standard {
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> BabyBear {
        loop {
            let next_u31 = rng.next_u32() & 0x7ffffff;
            let is_canonical = next_u31 < P;
            if is_canonical {
                return BabyBear { value: next_u31 };
            }
        }
    }
}

impl Serialize for BabyBear {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_u32(self.as_canonical_u32())
    }
}

impl<'de> Deserialize<'de> for BabyBear {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let val = u32::deserialize(d)?;
        Ok(BabyBear::from_canonical_u32(val))
    }
}

const MONTY_ZERO: u32 = to_monty(0);
const MONTY_ONE: u32 = to_monty(1);
const MONTY_TWO: u32 = to_monty(2);
const MONTY_NEG_ONE: u32 = to_monty(P - 1);

impl Packable for BabyBear {}

impl AbstractField for BabyBear {
    type F = Self;

    fn zero() -> Self {
        Self { value: MONTY_ZERO }
    }
    fn one() -> Self {
        Self { value: MONTY_ONE }
    }
    fn two() -> Self {
        Self { value: MONTY_TWO }
    }
    fn neg_one() -> Self {
        Self {
            value: MONTY_NEG_ONE,
        }
    }

    #[inline]
    fn from_f(f: Self::F) -> Self {
        f
    }

    #[inline]
    fn from_bool(b: bool) -> Self {
        Self::from_canonical_u32(b as u32)
    }

    #[inline]
    fn from_canonical_u8(n: u8) -> Self {
        Self::from_canonical_u32(n as u32)
    }

    #[inline]
    fn from_canonical_u16(n: u16) -> Self {
        Self::from_canonical_u32(n as u32)
    }

    #[inline]
    fn from_canonical_u32(n: u32) -> Self {
        debug_assert!(n < P);
        Self::from_wrapped_u32(n)
    }

    #[inline]
    fn from_canonical_u64(n: u64) -> Self {
        debug_assert!(n < P as u64);
        Self::from_canonical_u32(n as u32)
    }

    #[inline]
    fn from_canonical_usize(n: usize) -> Self {
        debug_assert!(n < P as usize);
        Self::from_canonical_u32(n as u32)
    }

    #[inline]
    fn from_wrapped_u32(n: u32) -> Self {
        Self { value: to_monty(n) }
    }

    #[inline]
    fn from_wrapped_u64(n: u64) -> Self {
        Self {
            value: to_monty_64(n),
        }
    }

    #[inline]
    fn generator() -> Self {
        Self::from_canonical_u32(0x1f)
    }
}

impl Field for BabyBear {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    type Packing = crate::PackedBabyBearNeon;
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        not(all(feature = "nightly-features", target_feature = "avx512f"))
    ))]
    type Packing = crate::PackedBabyBearAVX2;
    #[cfg(all(
        feature = "nightly-features",
        target_arch = "x86_64",
        target_feature = "avx512f"
    ))]
    type Packing = crate::PackedBabyBearAVX512;
    #[cfg(not(any(
        all(target_arch = "aarch64", target_feature = "neon"),
        all(
            target_arch = "x86_64",
            target_feature = "avx2",
            not(all(feature = "nightly-features", target_feature = "avx512f"))
        ),
        all(
            feature = "nightly-features",
            target_arch = "x86_64",
            target_feature = "avx512f"
        ),
    )))]
    type Packing = Self;

    #[inline]
    fn mul_2exp_u64(&self, exp: u64) -> Self {
        let product = (self.value as u64) << exp;
        let value = (product % (P as u64)) as u32;
        Self { value }
    }

    #[inline]
    fn exp_u64_generic<AF: AbstractField<F = Self>>(val: AF, power: u64) -> AF {
        match power {
            1725656503 => exp_1725656503(val), // used to compute x^{1/7}
            _ => exp_u64_by_squaring(val, power),
        }
    }

    fn try_inverse(&self) -> Option<Self> {
        if self.is_zero() {
            return None;
        }

        // From Fermat's little theorem, in a prime field `F_p`, the inverse of `a` is `a^(p-2)`.
        // Here p-2 = 2013265919 = 1110111111111111111111111111111_2.
        // Uses 30 Squares + 7 Multiplications => 37 Operations total.

        let p1 = *self;
        let p100000000 = p1.exp_power_of_2(8);
        let p100000001 = p100000000 * p1;
        let p10000000000000000 = p100000000.exp_power_of_2(8);
        let p10000000100000001 = p10000000000000000 * p100000001;
        let p10000000100000001000 = p10000000100000001.exp_power_of_2(3);
        let p1000000010000000100000000 = p10000000100000001000.exp_power_of_2(5);
        let p1000000010000000100000001 = p1000000010000000100000000 * p1;
        let p1000010010000100100001001 = p1000000010000000100000001 * p10000000100000001000;
        let p10000000100000001000000010 = p1000000010000000100000001.square();
        let p11000010110000101100001011 = p10000000100000001000000010 * p1000010010000100100001001;
        let p100000001000000010000000100 = p10000000100000001000000010.square();
        let p111000011110000111100001111 =
            p100000001000000010000000100 * p11000010110000101100001011;
        let p1110000111100001111000011110000 = p111000011110000111100001111.exp_power_of_2(4);
        let p1110111111111111111111111111111 =
            p1110000111100001111000011110000 * p111000011110000111100001111;

        Some(p1110111111111111111111111111111)
    }

    #[inline]
    fn halve(&self) -> Self {
        BabyBear {
            value: halve_u32::<P>(self.value),
        }
    }
}

impl PrimeField for BabyBear {}

impl PrimeField64 for BabyBear {
    const ORDER_U64: u64 = <Self as PrimeField32>::ORDER_U32 as u64;

    #[inline]
    fn as_canonical_u64(&self) -> u64 {
        u64::from(self.as_canonical_u32())
    }
}

impl PrimeField32 for BabyBear {
    const ORDER_U32: u32 = P;

    #[inline]
    fn as_canonical_u32(&self) -> u32 {
        from_monty(self.value)
    }
}

impl TwoAdicField for BabyBear {
    const TWO_ADICITY: usize = 27;

    fn two_adic_generator(bits: usize) -> Self {
        assert!(bits <= Self::TWO_ADICITY);
        match bits {
            0 => Self::one(),
            1 => Self::from_canonical_u32(0x78000000),
            2 => Self::from_canonical_u32(0x67055c21),
            3 => Self::from_canonical_u32(0x5ee99486),
            4 => Self::from_canonical_u32(0xbb4c4e4),
            5 => Self::from_canonical_u32(0x2d4cc4da),
            6 => Self::from_canonical_u32(0x669d6090),
            7 => Self::from_canonical_u32(0x17b56c64),
            8 => Self::from_canonical_u32(0x67456167),
            9 => Self::from_canonical_u32(0x688442f9),
            10 => Self::from_canonical_u32(0x145e952d),
            11 => Self::from_canonical_u32(0x4fe61226),
            12 => Self::from_canonical_u32(0x4c734715),
            13 => Self::from_canonical_u32(0x11c33e2a),
            14 => Self::from_canonical_u32(0x62c3d2b1),
            15 => Self::from_canonical_u32(0x77cad399),
            16 => Self::from_canonical_u32(0x54c131f4),
            17 => Self::from_canonical_u32(0x4cabd6a6),
            18 => Self::from_canonical_u32(0x5cf5713f),
            19 => Self::from_canonical_u32(0x3e9430e8),
            20 => Self::from_canonical_u32(0xba067a3),
            21 => Self::from_canonical_u32(0x18adc27d),
            22 => Self::from_canonical_u32(0x21fd55bc),
            23 => Self::from_canonical_u32(0x4b859b3d),
            24 => Self::from_canonical_u32(0x3bd57996),
            25 => Self::from_canonical_u32(0x4483d85a),
            26 => Self::from_canonical_u32(0x3a26eef8),
            27 => Self::from_canonical_u32(0x1a427a41),
            _ => unreachable!("Already asserted that bits <= Self::TWO_ADICITY"),
        }
    }
}

impl Add for BabyBear {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        let mut sum = self.value + rhs.value;
        let (corr_sum, over) = sum.overflowing_sub(P);
        if !over {
            sum = corr_sum;
        }
        Self { value: sum }
    }
}

impl AddAssign for BabyBear {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Sum for BabyBear {
    #[inline]
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        // This is faster than iter.reduce(|x, y| x + y).unwrap_or(Self::zero()) for iterators of length > 2.
        // There might be a faster reduction method possible for lengths <= 16 which avoids %.

        // This sum will not overflow so long as iter.len() < 2^33.
        let sum = iter.map(|x| (x.value as u64)).sum::<u64>();
        BabyBear {
            value: (sum % P as u64) as u32,
        }
    }
}

impl Sub for BabyBear {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        let (mut diff, over) = self.value.overflowing_sub(rhs.value);
        let corr = if over { P } else { 0 };
        diff = diff.wrapping_add(corr);
        BabyBear { value: diff }
    }
}

impl SubAssign for BabyBear {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl Neg for BabyBear {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        Self::zero() - self
    }
}

impl Mul for BabyBear {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let long_prod = self.value as u64 * rhs.value as u64;
        Self {
            value: monty_reduce(long_prod),
        }
    }
}

impl MulAssign for BabyBear {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl Product for BabyBear {
    #[inline]
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|x, y| x * y).unwrap_or(Self::one())
    }
}

impl Div for BabyBear {
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)]
    #[inline]
    fn div(self, rhs: Self) -> Self {
        self * rhs.inverse()
    }
}

#[inline]
#[must_use]
const fn to_monty(x: u32) -> u32 {
    (((x as u64) << MONTY_BITS) % P as u64) as u32
}

/// Convert a constant u32 array into a constant Babybear array.
/// Saves every element in Monty Form
#[inline]
#[must_use]
pub(crate) const fn to_babybear_array<const N: usize>(input: [u32; N]) -> [BabyBear; N] {
    let mut output = [BabyBear { value: 0 }; N];
    let mut i = 0;
    loop {
        if i == N {
            break;
        }
        output[i].value = to_monty(input[i]);
        i += 1;
    }
    output
}

#[inline]
#[must_use]
fn to_monty_64(x: u64) -> u32 {
    (((x as u128) << MONTY_BITS) % P as u128) as u32
}

#[inline]
#[must_use]
fn from_monty(x: u32) -> u32 {
    monty_reduce(x as u64)
}

/// Montgomery reduction of a value in `0..P << MONTY_BITS`.
#[inline]
#[must_use]
fn monty_reduce(x: u64) -> u32 {
    let t = x.wrapping_mul(MONTY_MU as u64) & (MONTY_MASK as u64);
    let u = t * (P as u64);

    let (x_sub_u, over) = x.overflowing_sub(u);
    let x_sub_u_hi = (x_sub_u >> MONTY_BITS) as u32;
    let corr = if over { P } else { 0 };
    x_sub_u_hi.wrapping_add(corr)
}

#[cfg(test)]
mod tests {
    use p3_field_testing::{test_field, test_two_adic_field};

    use super::*;

    type F = BabyBear;

    #[test]
    fn test_baby_bear_two_adicity_generators() {
        let base = BabyBear::from_canonical_u32(0x1a427a41);
        for bits in 0..=BabyBear::TWO_ADICITY {
            assert_eq!(
                BabyBear::two_adic_generator(bits),
                base.exp_power_of_2(BabyBear::TWO_ADICITY - bits)
            );
        }
    }

    #[test]
    fn test_baby_bear() {
        let f = F::from_canonical_u32(100);
        assert_eq!(f.as_canonical_u64(), 100);

        let f = F::from_canonical_u32(0);
        assert!(f.is_zero());

        let f = F::from_wrapped_u32(F::ORDER_U32);
        assert!(f.is_zero());

        let f_1 = F::one();
        let f_1_copy = F::from_canonical_u32(1);

        let expected_result = F::zero();
        assert_eq!(f_1 - f_1_copy, expected_result);

        let expected_result = F::two();
        assert_eq!(f_1 + f_1_copy, expected_result);

        let f_2 = F::from_canonical_u32(2);
        let expected_result = F::from_canonical_u32(3);
        assert_eq!(f_1 + f_1_copy * f_2, expected_result);

        let expected_result = F::from_canonical_u32(5);
        assert_eq!(f_1 + f_2 * f_2, expected_result);

        let f_p_minus_1 = F::from_canonical_u32(F::ORDER_U32 - 1);
        let expected_result = F::zero();
        assert_eq!(f_1 + f_p_minus_1, expected_result);

        let f_p_minus_2 = F::from_canonical_u32(F::ORDER_U32 - 2);
        let expected_result = F::from_canonical_u32(F::ORDER_U32 - 3);
        assert_eq!(f_p_minus_1 + f_p_minus_2, expected_result);

        let expected_result = F::from_canonical_u32(1);
        assert_eq!(f_p_minus_1 - f_p_minus_2, expected_result);

        let expected_result = f_p_minus_1;
        assert_eq!(f_p_minus_2 - f_p_minus_1, expected_result);

        let expected_result = f_p_minus_2;
        assert_eq!(f_p_minus_1 - f_1, expected_result);

        let m1 = F::from_canonical_u32(0x34167c58);
        let m2 = F::from_canonical_u32(0x61f3207b);
        let expected_prod = F::from_canonical_u32(0x1b5c8046);
        assert_eq!(m1 * m2, expected_prod);

        assert_eq!(m1.exp_u64(1725656503).exp_const_u64::<7>(), m1);
        assert_eq!(m2.exp_u64(1725656503).exp_const_u64::<7>(), m2);
        assert_eq!(f_2.exp_u64(1725656503).exp_const_u64::<7>(), f_2);

        let f_serialized = serde_json::to_string(&f).unwrap();
        let f_deserialized: F = serde_json::from_str(&f_serialized).unwrap();
        assert_eq!(f, f_deserialized);

        let f_1_serialized = serde_json::to_string(&f_1).unwrap();
        let f_1_deserialized: F = serde_json::from_str(&f_1_serialized).unwrap();
        let f_1_serialized_again = serde_json::to_string(&f_1_deserialized).unwrap();
        let f_1_deserialized_again: F = serde_json::from_str(&f_1_serialized_again).unwrap();
        assert_eq!(f_1, f_1_deserialized);
        assert_eq!(f_1, f_1_deserialized_again);

        let f_2_serialized = serde_json::to_string(&f_2).unwrap();
        let f_2_deserialized: F = serde_json::from_str(&f_2_serialized).unwrap();
        assert_eq!(f_2, f_2_deserialized);

        let f_p_minus_1_serialized = serde_json::to_string(&f_p_minus_1).unwrap();
        let f_p_minus_1_deserialized: F = serde_json::from_str(&f_p_minus_1_serialized).unwrap();
        assert_eq!(f_p_minus_1, f_p_minus_1_deserialized);

        let f_p_minus_2_serialized = serde_json::to_string(&f_p_minus_2).unwrap();
        let f_p_minus_2_deserialized: F = serde_json::from_str(&f_p_minus_2_serialized).unwrap();
        assert_eq!(f_p_minus_2, f_p_minus_2_deserialized);

        let m1_serialized = serde_json::to_string(&m1).unwrap();
        let m1_deserialized: F = serde_json::from_str(&m1_serialized).unwrap();
        assert_eq!(m1, m1_deserialized);

        let m2_serialized = serde_json::to_string(&m2).unwrap();
        let m2_deserialized: F = serde_json::from_str(&m2_serialized).unwrap();
        assert_eq!(m2, m2_deserialized);
    }

    test_field!(crate::BabyBear);
    test_two_adic_field!(crate::BabyBear);
}
