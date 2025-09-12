//! Goldilocks field implementation using Montgomery arithmetic.

use core::fmt::{Debug, Display};
use core::hash::Hash;
use core::iter::{Product, Sum};
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use num_bigint::BigUint;
use p3_field::exponentiation::exp_10540996611094048183;
use p3_field::integers::QuotientMap;
use p3_field::{
    Field, InjectiveMonomial, Packable, PermutationMonomial, PrimeCharacteristicRing, PrimeField,
    PrimeField64, RawDataSerializable, TwoAdicField,
};
use p3_monty_64::{MontyField64, MontyParameters64};
use rand::Rng;
use rand::distr::{Distribution, StandardUniform};
use serde::{Deserialize, Deserializer, Serialize};

/// The Goldilocks prime: 2^64 - 2^32 + 1
pub const GOLDILOCKS_PRIME: u64 = 0xffffffff00000001;

/// Montgomery parameters for the Goldilocks field
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq, Hash)]
pub struct GoldilocksMontyParameters;

impl MontyParameters64 for GoldilocksMontyParameters {
    /// The Goldilocks prime: 2^64 - 2^32 + 1
    const PRIME: u64 = GOLDILOCKS_PRIME;

    /// R = 2^64 mod P
    /// Since P = 2^64 - 2^32 + 1, we have 2^64 ≡ 2^32 - 1 (mod P)
    const MONTY_R: u64 = 0x00000000ffffffff;

    /// R^2 mod P used for conversion to Montgomery form
    /// We need to compute (2^32 - 1)^2 mod P
    /// (2^32 - 1)^2 = 2^64 - 2^33 + 1 ≡ (2^32 - 1) - 2^33 + 1 = 2^32 - 2^33 = 2^32(1 - 2) = -2^32 ≡ P - 2^32 (mod P)
    const MONTY_R2: u64 = 0xfffffffe00000001; // P - 2^32 = (2^64 - 2^32 + 1) - 2^32

    /// -P^{-1} mod 2^64, used in Montgomery reduction
    /// Need to find x such that P * x ≡ -1 (mod 2^64)
    /// From calculation: P^{-1} = 0x100000001, so -P^{-1} = 0xfffffffeffffffff
    const MONTY_INV: u64 = 0xfffffffeffffffff;

    /// Montgomery form of 0
    const MONTY_ZERO: MontyField64<Self> = MontyField64::new_monty(0);

    /// Montgomery form of 1  
    /// 1 in Montgomery form = R mod P = 2^32 - 1
    const MONTY_ONE: MontyField64<Self> = MontyField64::new_monty(0x00000000ffffffff);

    /// Montgomery form of 2
    /// 2 in Montgomery form = 2 * R mod P = 2 * (2^32 - 1) = 2^33 - 2 = 0x1fffffffe
    const MONTY_TWO: MontyField64<Self> = MontyField64::new_monty(0x00000001fffffffe);

    /// Montgomery form of -1 (P-1)
    /// (P-1) in Montgomery form = (P-1) * R mod P = 0xfffffffe00000002
    const MONTY_NEG_ONE: MontyField64<Self> = MontyField64::new_monty(0xfffffffe00000002);
}

/// The Goldilocks field element using Montgomery representation
/// This is a newtype wrapper to allow implementing traits
#[derive(Clone, Copy, Default, Eq, Hash, PartialEq, PartialOrd, Ord)]
#[repr(transparent)]
pub struct Goldilocks(MontyField64<GoldilocksMontyParameters>);

impl Goldilocks {
    /// Create a new Goldilocks field element
    pub const fn new(value: u64) -> Self {
        Self(MontyField64::new(value))
    }

    /// Create from Montgomery form
    pub const fn new_monty(value: u64) -> Self {
        Self(MontyField64::new_monty(value))
    }

    /// Create an array of Goldilocks field elements from u64 values
    pub const fn new_array<const N: usize>(input: [u64; N]) -> [Self; N] {
        // We can't use generic const fn yet, so we'll use unsafe to cast the array
        // This is safe because Goldilocks is #[repr(transparent)] over MontyField64
        unsafe {
            let mut result = core::mem::MaybeUninit::<[Self; N]>::uninit();
            let result_ptr = result.as_mut_ptr() as *mut Self;
            let mut i = 0;
            while i < N {
                core::ptr::write(result_ptr.add(i), Self::new(input[i]));
                i += 1;
            }
            result.assume_init()
        }
    }

    /// Get the inner MontyField64
    pub const fn inner(&self) -> MontyField64<GoldilocksMontyParameters> {
        self.0
    }
}

impl Display for Goldilocks {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        Display::fmt(&self.0, f)
    }
}

impl Debug for Goldilocks {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        Debug::fmt(&self.0, f)
    }
}

impl Distribution<Goldilocks> for StandardUniform {
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Goldilocks {
        Goldilocks(StandardUniform.sample(rng))
    }
}

impl Serialize for Goldilocks {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.0.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Goldilocks {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        Ok(Goldilocks(MontyField64::deserialize(d)?))
    }
}

impl Packable for Goldilocks {}

impl RawDataSerializable for Goldilocks {
    const NUM_BYTES: usize = 8;

    fn into_bytes(self) -> impl IntoIterator<Item = u8> {
        self.0.into_bytes()
    }
}

impl PrimeCharacteristicRing for Goldilocks {
    type PrimeSubfield = Self;

    const ZERO: Self = Goldilocks(GoldilocksMontyParameters::MONTY_ZERO);
    const ONE: Self = Goldilocks(GoldilocksMontyParameters::MONTY_ONE);
    const TWO: Self = Goldilocks(GoldilocksMontyParameters::MONTY_TWO);
    const NEG_ONE: Self = Goldilocks(GoldilocksMontyParameters::MONTY_NEG_ONE);

    #[inline(always)]
    fn from_prime_subfield(f: Self) -> Self {
        f
    }

    #[inline]
    fn halve(&self) -> Self {
        Goldilocks(self.0.halve())
    }

    #[inline]
    fn zero_vec(len: usize) -> alloc::vec::Vec<Self> {
        // SAFETY: Goldilocks is repr(transparent) over MontyField64
        unsafe { core::mem::transmute(MontyField64::<GoldilocksMontyParameters>::zero_vec(len)) }
    }
}

impl Field for Goldilocks {
    type Packing = Self;

    const GENERATOR: Self = Goldilocks::new(7); // Standard Goldilocks generator

    fn try_inverse(&self) -> Option<Self> {
        self.0.try_inverse().map(Goldilocks)
    }

    #[inline]
    fn order() -> BigUint {
        GOLDILOCKS_PRIME.into()
    }
}

// Delegate QuotientMap implementations
impl QuotientMap<u8> for Goldilocks {
    #[inline]
    fn from_int(int: u8) -> Self {
        Goldilocks(MontyField64::from_int(int))
    }
    #[inline]
    fn from_canonical_checked(int: u8) -> Option<Self> {
        MontyField64::from_canonical_checked(int).map(Goldilocks)
    }
    #[inline]
    unsafe fn from_canonical_unchecked(int: u8) -> Self {
        unsafe { Goldilocks(MontyField64::from_canonical_unchecked(int)) }
    }
}

impl QuotientMap<u16> for Goldilocks {
    #[inline]
    fn from_int(int: u16) -> Self {
        Goldilocks(MontyField64::from_int(int))
    }
    #[inline]
    fn from_canonical_checked(int: u16) -> Option<Self> {
        MontyField64::from_canonical_checked(int).map(Goldilocks)
    }
    #[inline]
    unsafe fn from_canonical_unchecked(int: u16) -> Self {
        unsafe { Goldilocks(MontyField64::from_canonical_unchecked(int)) }
    }
}

impl QuotientMap<u32> for Goldilocks {
    #[inline]
    fn from_int(int: u32) -> Self {
        Goldilocks(MontyField64::from_int(int))
    }
    #[inline]
    fn from_canonical_checked(int: u32) -> Option<Self> {
        MontyField64::from_canonical_checked(int).map(Goldilocks)
    }
    #[inline]
    unsafe fn from_canonical_unchecked(int: u32) -> Self {
        unsafe { Goldilocks(MontyField64::from_canonical_unchecked(int)) }
    }
}

impl QuotientMap<u64> for Goldilocks {
    #[inline]
    fn from_int(int: u64) -> Self {
        Goldilocks(MontyField64::from_int(int))
    }
    #[inline]
    fn from_canonical_checked(int: u64) -> Option<Self> {
        MontyField64::from_canonical_checked(int).map(Goldilocks)
    }
    #[inline]
    unsafe fn from_canonical_unchecked(int: u64) -> Self {
        unsafe { Goldilocks(MontyField64::from_canonical_unchecked(int)) }
    }
}

impl QuotientMap<u128> for Goldilocks {
    #[inline]
    fn from_int(int: u128) -> Self {
        Goldilocks(MontyField64::from_int(int))
    }
    #[inline]
    fn from_canonical_checked(int: u128) -> Option<Self> {
        MontyField64::from_canonical_checked(int).map(Goldilocks)
    }
    #[inline]
    unsafe fn from_canonical_unchecked(int: u128) -> Self {
        unsafe { Goldilocks(MontyField64::from_canonical_unchecked(int)) }
    }
}

impl QuotientMap<i8> for Goldilocks {
    #[inline]
    fn from_int(int: i8) -> Self {
        Goldilocks(MontyField64::from_int(int))
    }
    #[inline]
    fn from_canonical_checked(int: i8) -> Option<Self> {
        MontyField64::from_canonical_checked(int).map(Goldilocks)
    }
    #[inline]
    unsafe fn from_canonical_unchecked(int: i8) -> Self {
        unsafe { Goldilocks(MontyField64::from_canonical_unchecked(int)) }
    }
}

impl QuotientMap<i16> for Goldilocks {
    #[inline]
    fn from_int(int: i16) -> Self {
        Goldilocks(MontyField64::from_int(int))
    }
    #[inline]
    fn from_canonical_checked(int: i16) -> Option<Self> {
        MontyField64::from_canonical_checked(int).map(Goldilocks)
    }
    #[inline]
    unsafe fn from_canonical_unchecked(int: i16) -> Self {
        unsafe { Goldilocks(MontyField64::from_canonical_unchecked(int)) }
    }
}

impl QuotientMap<i32> for Goldilocks {
    #[inline]
    fn from_int(int: i32) -> Self {
        Goldilocks(MontyField64::from_int(int))
    }
    #[inline]
    fn from_canonical_checked(int: i32) -> Option<Self> {
        MontyField64::from_canonical_checked(int).map(Goldilocks)
    }
    #[inline]
    unsafe fn from_canonical_unchecked(int: i32) -> Self {
        unsafe { Goldilocks(MontyField64::from_canonical_unchecked(int)) }
    }
}

impl QuotientMap<i64> for Goldilocks {
    #[inline]
    fn from_int(int: i64) -> Self {
        Goldilocks(MontyField64::from_int(int))
    }
    #[inline]
    fn from_canonical_checked(int: i64) -> Option<Self> {
        MontyField64::from_canonical_checked(int).map(Goldilocks)
    }
    #[inline]
    unsafe fn from_canonical_unchecked(int: i64) -> Self {
        unsafe { Goldilocks(MontyField64::from_canonical_unchecked(int)) }
    }
}

impl QuotientMap<i128> for Goldilocks {
    #[inline]
    fn from_int(int: i128) -> Self {
        Goldilocks(MontyField64::from_int(int))
    }
    #[inline]
    fn from_canonical_checked(int: i128) -> Option<Self> {
        MontyField64::from_canonical_checked(int).map(Goldilocks)
    }
    #[inline]
    unsafe fn from_canonical_unchecked(int: i128) -> Self {
        unsafe { Goldilocks(MontyField64::from_canonical_unchecked(int)) }
    }
}

impl PrimeField for Goldilocks {
    fn as_canonical_biguint(&self) -> BigUint {
        self.0.as_canonical_biguint()
    }
}

impl PrimeField64 for Goldilocks {
    const ORDER_U64: u64 = GOLDILOCKS_PRIME;

    #[inline]
    fn as_canonical_u64(&self) -> u64 {
        self.0.as_canonical_u64()
    }

    #[inline]
    fn to_unique_u64(&self) -> u64 {
        self.0.to_unique_u64()
    }
}

impl TwoAdicField for Goldilocks {
    /// Goldilocks has 2-adicity of 32
    const TWO_ADICITY: usize = 32;

    fn two_adic_generator(bits: usize) -> Self {
        assert!(bits <= Self::TWO_ADICITY);

        // Precomputed two-adic generators for Goldilocks field
        // These are the same as in the original Goldilocks implementation
        const TWO_ADIC_GENERATORS: [u64; 33] = [
            0x0000000000000001,
            0xffffffff00000000,
            0x0001000000000000,
            0xfffffffeff000001,
            0xefffffff00000001,
            0x00003fffffffc000,
            0x0000008000000000,
            0xf80007ff08000001,
            0xbf79143ce60ca966,
            0x1905d02a5c411f4e,
            0x9d8f2ad78bfed972,
            0x0653b4801da1c8cf,
            0xf2c35199959dfcb6,
            0x1544ef2335d17997,
            0xe0ee099310bba1e2,
            0xf6b2cffe2306baac,
            0x54df9630bf79450e,
            0xabd0a6e8aa3d8a0e,
            0x81281a7b05f9beac,
            0xfbd41c6b8caa3302,
            0x30ba2ecd5e93e76d,
            0xf502aef532322654,
            0x4b2a18ade67246b5,
            0xea9d5a1336fbc98b,
            0x86cdcc31c307e171,
            0x4bbaf5976ecfefd8,
            0xed41d05b78d6e286,
            0x10d78dd8915a171d,
            0x59049500004a4485,
            0xdfa8c93ba46d2666,
            0x7e9bd009b86a0845,
            0x400a7f755588e659,
            0x185629dcda58878c,
        ];

        Goldilocks::new(TWO_ADIC_GENERATORS[bits])
    }
}

// Arithmetic operations - delegate to inner MontyField64

impl Add for Goldilocks {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Goldilocks(self.0 + rhs.0)
    }
}

impl Sub for Goldilocks {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Goldilocks(self.0 - rhs.0)
    }
}

impl Mul for Goldilocks {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Goldilocks(self.0 * rhs.0)
    }
}

impl Div for Goldilocks {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self {
        Goldilocks(self.0 / rhs.0)
    }
}

impl Neg for Goldilocks {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Goldilocks(-self.0)
    }
}

// Assignment operators

impl AddAssign for Goldilocks {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
    }
}

impl SubAssign for Goldilocks {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.0 -= rhs.0;
    }
}

impl MulAssign for Goldilocks {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        self.0 *= rhs.0;
    }
}

impl DivAssign for Goldilocks {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        self.0 /= rhs.0;
    }
}

impl Sum for Goldilocks {
    #[inline]
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        Goldilocks(iter.map(|x| x.0).sum())
    }
}

impl Product for Goldilocks {
    #[inline]
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        Goldilocks(iter.map(|x| x.0).product())
    }
}

impl InjectiveMonomial<7> for Goldilocks {}

impl PermutationMonomial<7> for Goldilocks {
    /// In the field `Goldilocks`, `a^{1/7}` is equal to a^{10540996611094048183}.
    ///
    /// This follows from the calculation `7*10540996611094048183 = 4*(2^64 - 2**32) + 1 = 1 mod (p - 1)`.
    fn injective_exp_root_n(&self) -> Self {
        exp_10540996611094048183(*self)
    }
}

#[cfg(test)]
mod tests {
    use p3_field::Field;

    use super::*;

    #[test]
    fn test_basic_arithmetic() {
        let a_u64 = (2u128.pow(64) - 2u128.pow(32)) as u64;
        let b_u64 = (a_u64 - 1) as u64;
        let a = Goldilocks::new(a_u64);
        let b = Goldilocks::new(b_u64);

        // Test addition
        let sum = a + b;
        assert_eq!(
            sum.as_canonical_u64(),
            ((a_u64 as u128 + b_u64 as u128) % GOLDILOCKS_PRIME as u128) as u64
        );

        // Test multiplication
        let product = a * b;
        assert_eq!(
            product.as_canonical_u64(),
            ((a_u64 as u128 * b_u64 as u128) % GOLDILOCKS_PRIME as u128) as u64
        );

        // Test subtraction
        let diff = a - b;
        assert_eq!(diff.as_canonical_u64(), a_u64 - b_u64);

        // Test multiplication with ONE should give the same value
        let a_times_one = a * Goldilocks::ONE;
        if a_times_one.as_canonical_u64() != a.as_canonical_u64() {
            panic!(
                "a * ONE = {} != a = {}",
                a_times_one.as_canonical_u64(),
                a.as_canonical_u64()
            );
        }
    }

    #[test]
    fn test_field_properties() {
        let zero = Goldilocks::ZERO;
        let one = Goldilocks::ONE;
        let two = Goldilocks::TWO;

        assert_eq!(zero.as_canonical_u64(), 0);
        assert_eq!(one.as_canonical_u64(), 1);
        assert_eq!(two.as_canonical_u64(), 2);

        // Test multiplicative identity
        let a = Goldilocks::new(42);
        assert_eq!(a * one, a);
        assert_eq!(one * a, a);

        // Test additive identity
        assert_eq!(a + zero, a);
        assert_eq!(zero + a, a);
    }

    #[test]
    fn test_inverse() {
        let a = Goldilocks::new(123456);

        let inv_a = a.inverse();

        // Check if inverse is 0 (which would be wrong)
        if inv_a.as_canonical_u64() == 0 {
            panic!("Inverse is 0! This is wrong");
        }

        let product = a * inv_a;
        let expected = Goldilocks::ONE;

        // a * a^{-1} should equal 1
        if product != expected {
            panic!(
                "a={}, inv_a={}, product={}, expected={}",
                a.as_canonical_u64(),
                inv_a.as_canonical_u64(),
                product.as_canonical_u64(),
                expected.as_canonical_u64()
            );
        }
        assert_eq!(inv_a * a, Goldilocks::ONE);
    }

    #[test]
    fn test_two_adic_generator() {
        // Test that the generator has the correct order
        let generator = Goldilocks::two_adic_generator(32);
        let order = 1u64 << 32;

        // generator^{2^32} should equal 1
        assert_eq!(generator.exp_u64(order), Goldilocks::ONE);

        // generator^{2^31} should not equal 1 (primitive root property)
        assert_ne!(generator.exp_u64(order / 2), Goldilocks::ONE);
    }
}
