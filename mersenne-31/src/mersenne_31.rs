use alloc::vec;
use alloc::vec::Vec;
use core::fmt::{Debug, Display, Formatter};
use core::hash::{Hash, Hasher};
use core::iter::{Product, Sum};
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use core::{array, fmt, iter};

use num_bigint::BigUint;
use p3_challenger::UniformSamplingField;
use p3_field::exponentiation::exp_1717986917;
use p3_field::integers::QuotientMap;
use p3_field::op_assign_macros::{
    impl_add_assign, impl_div_methods, impl_mul_methods, impl_sub_assign,
};
use p3_field::{
    Field, InjectiveMonomial, Packable, PermutationMonomial, PrimeCharacteristicRing, PrimeField,
    PrimeField32, PrimeField64, RawDataSerializable, halve_u32, impl_raw_serializable_primefield32,
    quotient_map_large_iint, quotient_map_large_uint, quotient_map_small_int,
};
use p3_util::{flatten_to_base, gcd_inversion_prime_field_32};
use rand::Rng;
use rand::distr::{Distribution, StandardUniform};
use serde::de::Error;
use serde::{Deserialize, Deserializer, Serialize};

/// The Mersenne31 prime
const P: u32 = (1 << 31) - 1;

/// The prime field `F_p` where `p = 2^31 - 1`.
#[derive(Copy, Clone, Default)]
#[repr(transparent)] // Important for reasoning about memory layout.
#[must_use]
pub struct Mersenne31 {
    /// Not necessarily canonical, but must fit in 31 bits.
    pub(crate) value: u32,
}

impl Mersenne31 {
    /// Create a new field element from any `u32`.
    ///
    /// Any `u32` value is accepted and automatically reduced modulo P.
    #[inline]
    pub const fn new(value: u32) -> Self {
        Self { value: value % P }
    }

    /// Create a field element from a value assumed to be < 2^31.
    ///
    /// # Safety
    /// The element must lie in the range: `[0, 2^31 - 1]`.
    #[inline]
    pub(crate) const fn new_reduced(value: u32) -> Self {
        debug_assert!((value >> 31) == 0);
        Self { value }
    }

    /// Convert a u32 element into a Mersenne31 element.
    ///
    /// Returns `None` if the element does not lie in the range: `[0, 2^31 - 1]`.
    #[inline]
    pub const fn new_checked(value: u32) -> Option<Self> {
        if (value >> 31) == 0 {
            Some(Self { value })
        } else {
            None
        }
    }

    /// Convert a `[u32; N]` array to an array of field elements.
    ///
    /// Const version of `input.map(Mersenne31::new)`.
    #[inline]
    pub const fn new_array<const N: usize>(input: [u32; N]) -> [Self; N] {
        let mut output = [Self::ZERO; N];
        let mut i = 0;
        while i < N {
            output[i].value = input[i] % P;
            i += 1;
        }
        output
    }

    /// Precomputed table of generators for two-adic subgroups of the degree two extension field over Mersenne31.
    /// The `i`'th element is a generator of the subgroup of order `2^i`.
    pub const EXT_TWO_ADIC_GENERATORS: [[Self; 2]; 33] = [
        [Self::ONE, Self::ZERO],
        [Self::new(2_147_483_646), Self::new(0)],
        [Self::new(0), Self::new(2_147_483_646)],
        [Self::new(32_768), Self::new(2_147_450_879)],
        [Self::new(590_768_354), Self::new(978_592_373)],
        [Self::new(1_179_735_656), Self::new(1_241_207_368)],
        [Self::new(1_567_857_810), Self::new(456_695_729)],
        [Self::new(1_774_253_895), Self::new(1_309_288_441)],
        [Self::new(736_262_640), Self::new(1_553_669_210)],
        [Self::new(1_819_216_575), Self::new(1_662_816_114)],
        [Self::new(1_323_191_254), Self::new(1_936_974_060)],
        [Self::new(605_622_498), Self::new(1_964_232_216)],
        [Self::new(343_674_985), Self::new(501_786_993)],
        [Self::new(1_995_316_534), Self::new(149_306_621)],
        [Self::new(2_107_600_913), Self::new(1_378_821_388)],
        [Self::new(541_476_169), Self::new(2_101_081_972)],
        [Self::new(2_135_874_973), Self::new(483_411_332)],
        [Self::new(2_097_144_245), Self::new(1_684_033_590)],
        [Self::new(1_662_322_247), Self::new(670_236_780)],
        [Self::new(1_172_215_635), Self::new(595_888_646)],
        [Self::new(241_940_101), Self::new(323_856_519)],
        [Self::new(1_957_194_259), Self::new(2_139_647_100)],
        [Self::new(1_957_419_629), Self::new(1_541_039_442)],
        [Self::new(1_062_045_235), Self::new(1_824_580_421)],
        [Self::new(1_929_382_196), Self::new(1_664_698_822)],
        [Self::new(1_889_294_251), Self::new(331_248_939)],
        [Self::new(1_214_231_414), Self::new(1_646_302_518)],
        [Self::new(1_765_392_370), Self::new(461_136_547)],
        [Self::new(1_629_751_483), Self::new(66_485_474)],
        [Self::new(1_501_355_827), Self::new(1_439_063_420)],
        [Self::new(509_778_402), Self::new(800_467_507)],
        [Self::new(311_014_874), Self::new(1_584_694_829)],
        [Self::new(1_166_849_849), Self::new(1_117_296_306)],
    ];
}

impl PartialEq for Mersenne31 {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.as_canonical_u32() == other.as_canonical_u32()
    }
}

impl Eq for Mersenne31 {}

impl Packable for Mersenne31 {}

impl Hash for Mersenne31 {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u32(self.to_unique_u32());
    }
}

impl Ord for Mersenne31 {
    #[inline]
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.as_canonical_u32().cmp(&other.as_canonical_u32())
    }
}

impl PartialOrd for Mersenne31 {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Display for Mersenne31 {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Display::fmt(&self.value, f)
    }
}

impl Debug for Mersenne31 {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Debug::fmt(&self.value, f)
    }
}

impl Distribution<Mersenne31> for StandardUniform {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Mersenne31 {
        loop {
            let next_u31 = rng.next_u32() >> 1;
            let is_canonical = next_u31 != Mersenne31::ORDER_U32;
            if is_canonical {
                return Mersenne31::new_reduced(next_u31);
            }
        }
    }
}

impl Serialize for Mersenne31 {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        // No need to convert to canonical.
        serializer.serialize_u32(self.value)
    }
}

impl<'a> Deserialize<'a> for Mersenne31 {
    fn deserialize<D: Deserializer<'a>>(d: D) -> Result<Self, D::Error> {
        let val = u32::deserialize(d)?;
        // Ensure that `val` satisfies our invariant. i.e. Not necessarily canonical, but must fit in 31 bits.
        if val <= P {
            Ok(Self::new_reduced(val))
        } else {
            Err(D::Error::custom("Value is out of range"))
        }
    }
}

impl RawDataSerializable for Mersenne31 {
    impl_raw_serializable_primefield32!();
}

impl PrimeCharacteristicRing for Mersenne31 {
    type PrimeSubfield = Self;

    const ZERO: Self = Self { value: 0 };
    const ONE: Self = Self { value: 1 };
    const TWO: Self = Self { value: 2 };
    const NEG_ONE: Self = Self {
        value: Self::ORDER_U32 - 1,
    };

    #[inline]
    fn from_prime_subfield(f: Self::PrimeSubfield) -> Self {
        f
    }

    #[inline]
    fn from_bool(b: bool) -> Self {
        Self::new_reduced(b as u32)
    }

    #[inline]
    fn halve(&self) -> Self {
        Self::new_reduced(halve_u32::<P>(self.value))
    }

    #[inline]
    fn mul_2exp_u64(&self, exp: u64) -> Self {
        // In a Mersenne field, multiplication by 2^k is just a left rotation by k bits.
        let exp = exp % 31;
        let left = (self.value << exp) & ((1 << 31) - 1);
        let right = self.value >> (31 - exp);
        let rotated = left | right;
        Self::new_reduced(rotated)
    }

    #[inline]
    fn div_2exp_u64(&self, exp: u64) -> Self {
        // In a Mersenne field, division by 2^k is just a right rotation by k bits.
        let exp = (exp % 31) as u8;
        let left = self.value >> exp;
        let right = (self.value << (31 - exp)) & ((1 << 31) - 1);
        let rotated = left | right;
        Self::new_reduced(rotated)
    }

    #[inline]
    fn sum_array<const N: usize>(input: &[Self]) -> Self {
        assert_eq!(N, input.len());
        // Benchmarking shows that for N <= 5 it's faster to sum the elements directly
        // but for N > 5 it's faster to use the .sum() methods which passes through u64's
        // allowing for delayed reductions.
        match N {
            0 => Self::ZERO,
            1 => input[0],
            2 => input[0] + input[1],
            3 => input[0] + input[1] + input[2],
            4 => (input[0] + input[1]) + (input[2] + input[3]),
            5 => {
                let lhs = input[0] + input[1];
                let rhs = input[2] + input[3];
                lhs + rhs + input[4]
            }
            _ => input.iter().copied().sum(),
        }
    }

    #[inline]
    fn zero_vec(len: usize) -> Vec<Self> {
        // SAFETY:
        // Due to `#[repr(transparent)]`, Mersenne31 and u32 have the same size, alignment
        // and memory layout making `flatten_to_base` safe. This this will create
        // a vector Mersenne31 elements with value set to 0.
        unsafe { flatten_to_base(vec![0u32; len]) }
    }
}

// Degree of the smallest permutation polynomial for Mersenne31.
//
// As p - 1 = 2×3^2×7×11×... the smallest choice for a degree D satisfying gcd(p - 1, D) = 1 is 5.
impl InjectiveMonomial<5> for Mersenne31 {}

impl PermutationMonomial<5> for Mersenne31 {
    /// In the field `Mersenne31`, `a^{1/5}` is equal to a^{1717986917}.
    ///
    /// This follows from the calculation `5 * 1717986917 = 4*(2^31 - 2) + 1 = 1 mod p - 1`.
    fn injective_exp_root_n(&self) -> Self {
        exp_1717986917(*self)
    }
}

impl Field for Mersenne31 {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    type Packing = crate::PackedMersenne31Neon;
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        not(target_feature = "avx512f")
    ))]
    type Packing = crate::PackedMersenne31AVX2;
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    type Packing = crate::PackedMersenne31AVX512;
    #[cfg(not(any(
        all(target_arch = "aarch64", target_feature = "neon"),
        all(
            target_arch = "x86_64",
            target_feature = "avx2",
            not(target_feature = "avx512f")
        ),
        all(target_arch = "x86_64", target_feature = "avx512f"),
    )))]
    type Packing = Self;

    // Sage: GF(2^31 - 1).multiplicative_generator()
    const GENERATOR: Self = Self::new(7);

    #[inline]
    fn is_zero(&self) -> bool {
        self.value == 0 || self.value == Self::ORDER_U32
    }

    fn try_inverse(&self) -> Option<Self> {
        if self.is_zero() {
            return None;
        }

        // Number of bits in the Mersenne31 prime.
        const NUM_PRIME_BITS: u32 = 31;

        // gcd_inversion returns the inverse multiplied by 2^60 so we need to correct for that.
        let inverse_i64 = gcd_inversion_prime_field_32::<NUM_PRIME_BITS>(self.value, P);
        Some(Self::from_int(inverse_i64).div_2exp_u64(60))
    }

    #[inline]
    fn order() -> BigUint {
        P.into()
    }
}

// We can use some macros to implement QuotientMap<Int> for all integer types except for u32 and i32's.
quotient_map_small_int!(Mersenne31, u32, [u8, u16]);
quotient_map_small_int!(Mersenne31, i32, [i8, i16]);
quotient_map_large_uint!(
    Mersenne31,
    u32,
    Mersenne31::ORDER_U32,
    "`[0, 2^31 - 2]`",
    "`[0, 2^31 - 1]`",
    [u64, u128]
);
quotient_map_large_iint!(
    Mersenne31,
    i32,
    "`[-2^30, 2^30]`",
    "`[1 - 2^31, 2^31 - 1]`",
    [(i64, u64), (i128, u128)]
);

// We simple need to prove custom Mersenne31 impls for QuotientMap<u32> and QuotientMap<i32>
impl QuotientMap<u32> for Mersenne31 {
    /// Convert a given `u32` integer into an element of the `Mersenne31` field.
    #[inline]
    fn from_int(int: u32) -> Self {
        // To reduce `n` to 31 bits, we clear its MSB, then add it back in its reduced form.
        let msb = int & (1 << 31);
        let msb_reduced = msb >> 31;
        Self::new_reduced(int ^ msb) + Self::new_reduced(msb_reduced)
    }

    /// Convert a given `u32` integer into an element of the `Mersenne31` field.
    ///
    /// Returns none if the input does not lie in the range `[0, 2^31 - 1]`.
    #[inline]
    fn from_canonical_checked(int: u32) -> Option<Self> {
        (int < Self::ORDER_U32).then(|| Self::new_reduced(int))
    }

    /// Convert a given `u32` integer into an element of the `Mersenne31` field.
    ///
    /// # Safety
    /// The input must lie in the range: `[0, 2^31 - 1]`.
    #[inline(always)]
    unsafe fn from_canonical_unchecked(int: u32) -> Self {
        debug_assert!(int < Self::ORDER_U32);
        Self::new_reduced(int)
    }
}

impl QuotientMap<i32> for Mersenne31 {
    /// Convert a given `i32` integer into an element of the `Mersenne31` field.
    #[inline]
    fn from_int(int: i32) -> Self {
        if int >= 0 {
            Self::new_reduced(int as u32)
        } else if int > (-1 << 31) {
            Self::new_reduced(Self::ORDER_U32.wrapping_add_signed(int))
        } else {
            // The only other option is int = -(2^31) = -1 mod p.
            Self::NEG_ONE
        }
    }

    /// Convert a given `i32` integer into an element of the `Mersenne31` field.
    ///
    /// Returns none if the input does not lie in the range `(-2^30, 2^30)`.
    #[inline]
    fn from_canonical_checked(int: i32) -> Option<Self> {
        const TWO_EXP_30: i32 = 1 << 30;
        const NEG_TWO_EXP_30_PLUS_1: i32 = (-1 << 30) + 1;
        match int {
            0..TWO_EXP_30 => Some(Self::new_reduced(int as u32)),
            NEG_TWO_EXP_30_PLUS_1..0 => {
                Some(Self::new_reduced(Self::ORDER_U32.wrapping_add_signed(int)))
            }
            _ => None,
        }
    }

    /// Convert a given `i32` integer into an element of the `Mersenne31` field.
    ///
    /// # Safety
    /// The input must lie in the range: `[1 - 2^31, 2^31 - 1]`.
    #[inline(always)]
    unsafe fn from_canonical_unchecked(int: i32) -> Self {
        if int >= 0 {
            Self::new_reduced(int as u32)
        } else {
            Self::new_reduced(Self::ORDER_U32.wrapping_add_signed(int))
        }
    }
}

impl PrimeField for Mersenne31 {
    fn as_canonical_biguint(&self) -> BigUint {
        <Self as PrimeField32>::as_canonical_u32(self).into()
    }
}

impl PrimeField32 for Mersenne31 {
    const ORDER_U32: u32 = P;

    #[inline]
    fn as_canonical_u32(&self) -> u32 {
        // Since our invariant guarantees that `value` fits in 31 bits, there is only one possible
        // `value` that is not canonical, namely 2^31 - 1 = p = 0.
        if self.value == Self::ORDER_U32 {
            0
        } else {
            self.value
        }
    }
}

impl PrimeField64 for Mersenne31 {
    const ORDER_U64: u64 = <Self as PrimeField32>::ORDER_U32 as u64;

    #[inline]
    fn as_canonical_u64(&self) -> u64 {
        self.as_canonical_u32().into()
    }
}

impl Add for Mersenne31 {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        // See the following for a way to compute the sum that avoids
        // the conditional which may be preferable on some
        // architectures.
        // https://github.com/Plonky3/Plonky3/blob/6049a30c3b1f5351c3eb0f7c994dc97e8f68d10d/mersenne-31/src/lib.rs#L249

        // Working with i32 means we get a flag which informs us if overflow happened.
        let (sum_i32, over) = (self.value as i32).overflowing_add(rhs.value as i32);
        let sum_u32 = sum_i32 as u32;
        let sum_corr = sum_u32.wrapping_sub(Self::ORDER_U32);

        // If self + rhs did not overflow, return it.
        // If self + rhs overflowed, sum_corr = self + rhs - (2**31 - 1).
        Self::new_reduced(if over { sum_corr } else { sum_u32 })
    }
}

impl Sub for Mersenne31 {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        let (mut sub, over) = self.value.overflowing_sub(rhs.value);

        // If we didn't overflow we have the correct value.
        // Otherwise we have added 2**32 = 2**31 + 1 mod 2**31 - 1.
        // Hence we need to remove the most significant bit and subtract 1.
        sub -= over as u32;
        Self::new_reduced(sub & Self::ORDER_U32)
    }
}

impl Neg for Mersenne31 {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        // Can't underflow, since self.value is 31-bits and thus can't exceed ORDER.
        Self::new_reduced(Self::ORDER_U32 - self.value)
    }
}

impl Mul for Mersenne31 {
    type Output = Self;

    #[inline]
    #[allow(clippy::cast_possible_truncation)]
    fn mul(self, rhs: Self) -> Self {
        let prod = u64::from(self.value) * u64::from(rhs.value);
        from_u62(prod)
    }
}

impl_add_assign!(Mersenne31);
impl_sub_assign!(Mersenne31);
impl_mul_methods!(Mersenne31);
impl_div_methods!(Mersenne31, Mersenne31);

impl Sum for Mersenne31 {
    #[inline]
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        // This is faster than iter.reduce(|x, y| x + y).unwrap_or(Self::ZERO) for iterators of length >= 6.
        // It assumes that iter.len() < 2^31.

        // This sum will not overflow so long as iter.len() < 2^33.
        let sum = iter.map(|x| x.value as u64).sum::<u64>();

        // sum is < 2^62 provided iter.len() < 2^31.
        from_u62(sum)
    }
}

#[inline(always)]
pub(crate) fn from_u62(input: u64) -> Mersenne31 {
    debug_assert!(input < (1 << 62));
    let input_lo = (input & ((1 << 31) - 1)) as u32;
    let input_high = (input >> 31) as u32;
    Mersenne31::new_reduced(input_lo) + Mersenne31::new_reduced(input_high)
}

impl UniformSamplingField for Mersenne31 {
    const MAX_SINGLE_SAMPLE_BITS: usize = 16;
    // For Mersenne31 uniform sampling really only makes sense if we allow rejection sampling.
    // Sampling 16 bits already has a chance of 3e-5 to require a resample!
    const SAMPLING_BITS_M: [u64; 64] = {
        let prime: u64 = P as u64;
        let mut a = [0u64; 64];
        let mut k = 0;
        while k < 64 {
            if k == 0 {
                a[k] = prime; // This value is irrelevant in practice. `bits = 0` returns 0 always.
            } else {
                // Create a mask to zero out the last k bits
                let mask = !((1u64 << k) - 1);
                a[k] = prime & mask;
            }
            k += 1;
        }
        a
    };
}

#[cfg(test)]
mod tests {
    use num_bigint::BigUint;
    use p3_field::{InjectiveMonomial, PermutationMonomial, PrimeCharacteristicRing};
    use p3_field_testing::{
        test_field, test_prime_field, test_prime_field_32, test_prime_field_64,
    };

    use crate::Mersenne31;

    type F = Mersenne31;

    #[test]
    fn exp_root() {
        // Confirm that (x^{1/5})^5 = x

        let m1 = F::from_u32(0x34167c58);
        let m2 = F::from_u32(0x61f3207b);

        assert_eq!(m1.injective_exp_n().injective_exp_root_n(), m1);
        assert_eq!(m2.injective_exp_n().injective_exp_root_n(), m2);
        assert_eq!(F::TWO.injective_exp_n().injective_exp_root_n(), F::TWO);
    }

    // Mersenne31 has a redundant representation of Zero but no redundant representation of One.
    const ZEROS: [Mersenne31; 2] = [Mersenne31::ZERO, Mersenne31::new((1_u32 << 31) - 1)];
    const ONES: [Mersenne31; 1] = [Mersenne31::ONE];

    // Get the prime factorization of the order of the multiplicative group.
    // i.e. the prime factorization of P - 1.
    fn multiplicative_group_prime_factorization() -> [(BigUint, u32); 7] {
        [
            (BigUint::from(2u8), 1),
            (BigUint::from(3u8), 2),
            (BigUint::from(7u8), 1),
            (BigUint::from(11u8), 1),
            (BigUint::from(31u8), 1),
            (BigUint::from(151u8), 1),
            (BigUint::from(331u16), 1),
        ]
    }

    test_field!(
        crate::Mersenne31,
        &super::ZEROS,
        &super::ONES,
        &super::multiplicative_group_prime_factorization()
    );
    test_prime_field!(crate::Mersenne31);
    test_prime_field_64!(crate::Mersenne31, &super::ZEROS, &super::ONES);
    test_prime_field_32!(crate::Mersenne31, &super::ZEROS, &super::ONES);
}
