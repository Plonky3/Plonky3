use core::fmt::{self, Debug, Display, Formatter};
use core::iter::{Product, Sum};
use core::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

use num_bigint::BigUint;
use p3_field::{
    exp_1420470955, exp_u64_by_squaring, halve_u32, AbstractField, Field, Packable, PrimeField,
    PrimeField32, PrimeField64, TwoAdicField, PrimeField31
};
use rand::distributions::{Distribution, Standard};
use rand::Rng;
use serde::{Deserialize, Deserializer, Serialize};

/// The KoalaBear prime: 2^31 - 2^24 + 1
/// This is a 31-bit prime with the highest possible two adicity if we additionally demand that
/// the cube map (x -> x^3) is an automorphism of the multiplicative group.
/// Its not unique, as there is one other option with equal 2 adicity: 2^30 + 2^27 + 2^24 + 1.
/// There is also one 29-bit prime with higher two adicity which might be appropriate for some applications: 2^29 - 2^26 + 1.
const P: u32 = 0x7f000001;

const MONTY_BITS: u32 = 32;

// We are defining MU = P^-1 (mod 2^MONTY_BITS). This is different from the usual convention
// (MU = -P^-1 (mod 2^MONTY_BITS)) but it avoids a carry.
const MONTY_MU: u32 = 0x81000001;

// This is derived from above.
const MONTY_MASK: u32 = ((1u64 << MONTY_BITS) - 1) as u32;

/// The prime field `2^31 - 2^24 + 1`, a.k.a. the Koala Bear field.
#[derive(Copy, Clone, Default, Eq, Hash, PartialEq)]
#[repr(transparent)] // `PackedKoalaBearNeon` relies on this!
pub struct KoalaBear {
    // This is `pub(crate)` for tests and delayed reduction strategies. If you're accessing `value` outside of those, you're
    // likely doing something fishy.
    pub(crate) value: u32,
}

impl KoalaBear {
    /// create a new `KoalaBear` from a canonical `u32`.
    #[inline]
    pub(crate) const fn new(n: u32) -> Self {
        Self { value: to_monty(n) }
    }
}

impl Ord for KoalaBear {
    #[inline]
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.as_canonical_u32().cmp(&other.as_canonical_u32())
    }
}

impl PartialOrd for KoalaBear {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Display for KoalaBear {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Display::fmt(&self.as_canonical_u32(), f)
    }
}

impl Debug for KoalaBear {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Debug::fmt(&self.as_canonical_u32(), f)
    }
}

impl Distribution<KoalaBear> for Standard {
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> KoalaBear {
        loop {
            let next_u31 = rng.next_u32() >> 1;
            let is_canonical = next_u31 < P;
            if is_canonical {
                return KoalaBear { value: next_u31 };
            }
        }
    }
}

impl Serialize for KoalaBear {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_u32(self.as_canonical_u32())
    }
}

impl<'de> Deserialize<'de> for KoalaBear {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let val = u32::deserialize(d)?;
        Ok(KoalaBear::from_canonical_u32(val))
    }
}

const MONTY_ZERO: u32 = to_monty(0);
const MONTY_ONE: u32 = to_monty(1);
const MONTY_TWO: u32 = to_monty(2);
const MONTY_NEG_ONE: u32 = to_monty(P - 1);

impl Packable for KoalaBear {}

impl AbstractField for KoalaBear {
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
        Self::from_canonical_u32(0x3)
    }
}

impl Field for KoalaBear {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    type Packing = crate::PackedKoalaBearNeon;
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        not(all(feature = "nightly-features", target_feature = "avx512f"))
    ))]
    type Packing = crate::PackedKoalaBearAVX2;
    #[cfg(all(
        feature = "nightly-features",
        target_arch = "x86_64",
        target_feature = "avx512f"
    ))]
    type Packing = crate::PackedKoalaBearAVX512;
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
            1420470955 => exp_1420470955(val), // used to compute x^{1/3}
            _ => exp_u64_by_squaring(val, power),
        }
    }

    fn try_inverse(&self) -> Option<Self> {
        if self.is_zero() {
            return None;
        }

        // From Fermat's little theorem, in a prime field `F_p`, the inverse of `a` is `a^(p-2)`.
        // Here p-2 = 2130706431 = 1111110111111111111111111111111_2
        // Uses 29 Squares + 7 Multiplications => 36 Operations total.

        let p1 = *self;
        let p10 = p1.square();
        let p11 = p10 * p1;
        let p1100 = p11.exp_power_of_2(2);
        let p1111 = p1100 * p11;
        let p110000 = p1100.exp_power_of_2(2);
        let p111111 = p110000 * p1111;
        let p1111110000 = p111111.exp_power_of_2(4);
        let p1111111111 = p1111110000 * p1111;
        let p11111101111 = p1111111111 * p1111110000;
        let p111111011110000000000 = p11111101111.exp_power_of_2(10);
        let p111111011111111111111 = p111111011110000000000 * p1111111111;
        let p1111110111111111111110000000000 = p111111011111111111111.exp_power_of_2(10);
        let p1111110111111111111111111111111 = p1111110111111111111110000000000 * p1111111111;

        Some(p1111110111111111111111111111111)
    }

    #[inline]
    fn halve(&self) -> Self {
        KoalaBear {
            value: halve_u32::<P>(self.value),
        }
    }

    #[inline]
    fn order() -> BigUint {
        P.into()
    }
}

impl PrimeField for KoalaBear {
    fn as_canonical_biguint(&self) -> BigUint {
        <Self as PrimeField32>::as_canonical_u32(self).into()
    }
}

impl PrimeField64 for KoalaBear {
    const ORDER_U64: u64 = <Self as PrimeField32>::ORDER_U32 as u64;

    #[inline]
    fn as_canonical_u64(&self) -> u64 {
        u64::from(self.as_canonical_u32())
    }
}

impl PrimeField32 for KoalaBear {
    const ORDER_U32: u32 = P;

    #[inline]
    fn as_canonical_u32(&self) -> u32 {
        from_monty(self.value)
    }
}

impl PrimeField31 for KoalaBear {}

impl TwoAdicField for KoalaBear {
    const TWO_ADICITY: usize = 24;

    fn two_adic_generator(bits: usize) -> Self {
        assert!(bits <= Self::TWO_ADICITY);
        match bits {
            0 => Self::one(),
            1 => Self::from_canonical_u32(0x7f000000),
            2 => Self::from_canonical_u32(0x7e010002),
            3 => Self::from_canonical_u32(0x6832fe4a),
            4 => Self::from_canonical_u32(0x8dbd69c),
            5 => Self::from_canonical_u32(0xa28f031),
            6 => Self::from_canonical_u32(0x5c4a5b99),
            7 => Self::from_canonical_u32(0x29b75a80),
            8 => Self::from_canonical_u32(0x17668b8a),
            9 => Self::from_canonical_u32(0x27ad539b),
            10 => Self::from_canonical_u32(0x334d48c7),
            11 => Self::from_canonical_u32(0x7744959c),
            12 => Self::from_canonical_u32(0x768fc6fa),
            13 => Self::from_canonical_u32(0x303964b2),
            14 => Self::from_canonical_u32(0x3e687d4d),
            15 => Self::from_canonical_u32(0x45a60e61),
            16 => Self::from_canonical_u32(0x6e2f4d7a),
            17 => Self::from_canonical_u32(0x163bd499),
            18 => Self::from_canonical_u32(0x6c4a8a45),
            19 => Self::from_canonical_u32(0x143ef899),
            20 => Self::from_canonical_u32(0x514ddcad),
            21 => Self::from_canonical_u32(0x484ef19b),
            22 => Self::from_canonical_u32(0x205d63c3),
            23 => Self::from_canonical_u32(0x68e7dd49),
            24 => Self::from_canonical_u32(0x6ac49f88),
            _ => unreachable!("Already asserted that bits <= Self::TWO_ADICITY"),
        }
    }
}

impl Add for KoalaBear {
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

impl AddAssign for KoalaBear {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Sum for KoalaBear {
    #[inline]
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        // This is faster than iter.reduce(|x, y| x + y).unwrap_or(Self::zero()) for iterators of length > 2.
        // There might be a faster reduction method possible for lengths <= 16 which avoids %.

        // This sum will not overflow so long as iter.len() < 2^33.
        let sum = iter.map(|x| (x.value as u64)).sum::<u64>();
        Self {
            value: (sum % P as u64) as u32,
        }
    }
}

impl Sub for KoalaBear {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        let (mut diff, over) = self.value.overflowing_sub(rhs.value);
        let corr = if over { P } else { 0 };
        diff = diff.wrapping_add(corr);
        Self { value: diff }
    }
}

impl SubAssign for KoalaBear {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl Neg for KoalaBear {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        Self::zero() - self
    }
}

impl Mul for KoalaBear {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let long_prod = self.value as u64 * rhs.value as u64;
        Self {
            value: monty_reduce(long_prod),
        }
    }
}

impl MulAssign for KoalaBear {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl Product for KoalaBear {
    #[inline]
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|x, y| x * y).unwrap_or(Self::one())
    }
}

impl Div for KoalaBear {
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

/// Convert a constant u32 array into a constant KoalaBear array.
/// Saves every element in Monty Form
#[inline]
#[must_use]
pub(crate) const fn to_koalabear_array<const N: usize>(input: [u32; N]) -> [KoalaBear; N] {
    let mut output = [KoalaBear { value: 0 }; N];
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
const fn to_monty_64(x: u64) -> u32 {
    (((x as u128) << MONTY_BITS) % P as u128) as u32
}

#[inline]
#[must_use]
const fn from_monty(x: u32) -> u32 {
    monty_reduce(x as u64)
}

/// Montgomery reduction of a value in `0..P << MONTY_BITS`.
#[inline]
#[must_use]
pub(crate) const fn monty_reduce(x: u64) -> u32 {
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

    type F = KoalaBear;

    #[test]
    fn test_koala_bear_two_adicity_generators() {
        let base = KoalaBear::from_canonical_u32(0x6ac49f88);
        for bits in 0..=KoalaBear::TWO_ADICITY {
            assert_eq!(
                KoalaBear::two_adic_generator(bits),
                base.exp_power_of_2(KoalaBear::TWO_ADICITY - bits)
            );
        }
    }

    #[test]
    fn test_koala_bear() {
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
        let expected_prod = F::from_canonical_u32(0x54b46b81);
        assert_eq!(m1 * m2, expected_prod);

        assert_eq!(m1.exp_u64(1420470955).exp_const_u64::<3>(), m1);
        assert_eq!(m2.exp_u64(1420470955).exp_const_u64::<3>(), m2);
        assert_eq!(f_2.exp_u64(1420470955).exp_const_u64::<3>(), f_2);

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

    test_field!(crate::KoalaBear);
    test_two_adic_field!(crate::KoalaBear);
}
