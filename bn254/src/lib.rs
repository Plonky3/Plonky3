//! The prime field known as BN254, defined as `F_p` where `p = 21888242871839275222246405745257275088696311157297823662689037894645226208583`.

#![no_std]

extern crate alloc;

use alloc::vec::Vec;

use core::fmt;
use core::fmt::{Debug, Display, Formatter};
use core::hash::{Hash, Hasher};
use core::iter::{Product, Sum};
use core::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

use serde::ser::SerializeSeq;

use ark_ff::{Fp256, Field as af, MontBackend, MontConfig, One, Zero};
use p3_field::{AbstractField, Field, Packable, PrimeField};
use serde::{Deserialize, Deserializer, Serialize};

const NUM_BIGINT_LIMBS: usize = 4;

#[derive(MontConfig)]
#[modulus = "21888242871839275222246405745257275088548364400416034343698204186575808495617"]
#[generator = "7"]
struct FqConfig;
type FpBN256 = Fp256<MontBackend<FqConfig, NUM_BIGINT_LIMBS>>;

/// The BN254 curve base field prime, defined as `F_p` where `p = 21888242871839275222246405745257275088696311157297823662689037894645226208583`.
#[derive(Copy, Clone, Default)]
pub struct BN256 {
    value: FpBN256,
}

impl Serialize for BN256 {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let limbs = self.value.0.0;

        let mut seq = serializer.serialize_seq(Some(limbs.len()))?;
        for e in limbs {
            seq.serialize_element(&e)?;
        }
        seq.end()
    }
}

impl<'de> Deserialize<'de> for BN256 {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let limbs: Vec<u64> = Deserialize::deserialize(d)?;
        let mut value = FpBN256::zero();
        for (i, limb) in limbs.iter().enumerate() {
            value.0.0[i] = *limb;
        }
        Ok(Self { value })
    }
}


impl PartialEq for BN256 {
    fn eq(&self, other: &Self) -> bool {
        self == other
    }
}

impl Eq for BN256 {}

impl Packable for BN256 {}

impl Hash for BN256 {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.value.hash(state);
    }
}

impl Ord for BN256 {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.value.cmp(&other.value)
    }
}

impl PartialOrd for BN256 {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Display for BN256 {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Display::fmt(&self.value, f)
    }
}

impl Debug for BN256 {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Debug::fmt(&self.value, f)
    }
}

impl AbstractField for BN256 {
    type F = Self;

    fn zero() -> Self {
        Self { value: FpBN256::zero() }
    }
    fn one() -> Self {
        Self { value: FpBN256::one() }
    }
    fn two() -> Self {
        let two = FpBN256::from(2u32);
        Self { value: two }
    }

    fn neg_one() -> Self {
        let neg_one = FpBN256::from(-1i32);
        Self { value: neg_one }
    }

    #[inline]
    fn from_f(f: Self::F) -> Self {
        f
    }

    fn from_bool(b: bool) -> Self {
        Self { value: FpBN256::from(b) }
    }

    fn from_canonical_u8(n: u8) -> Self {
        Self { value: FpBN256::from(n) }
    }

    fn from_canonical_u16(n: u16) -> Self {
        Self { value: FpBN256::from(n) }
    }

    fn from_canonical_u32(n: u32) -> Self {
        Self { value: FpBN256::from(n) }
    }

    fn from_canonical_u64(n: u64) -> Self {
        Self { value: FpBN256::from(n) }
    }

    fn from_canonical_usize(n: usize) -> Self {
        Self { value: FpBN256::from(n as u64) }
    }

    fn from_wrapped_u32(n: u32) -> Self {
        Self { value: FpBN256::from(n) }
    }

    fn from_wrapped_u64(n: u64) -> Self {
        Self { value: FpBN256::from(n) }
    }

    // Sage: GF(2^64 - 2^32 + 1).multiplicative_generator()
    fn generator() -> Self {
        let seven = FpBN256::from(7u32);
        Self { value: seven }
    }
}

impl Field for BN256 {
    // TODO: Add cfg-guarded Packing for AVX2, NEON, etc.
    type Packing = Self;

    fn is_zero(&self) -> bool {
        self.value.is_zero()
    }

    fn try_inverse(&self) -> Option<Self> {
        let inverse = self.value.inverse();

        inverse.map(|inverse| Self { value: inverse })
    }
}

impl PrimeField for BN256 {}

impl Add for BN256 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self { value: self.value.add(rhs.value) }
    }
}

impl AddAssign for BN256 {
    fn add_assign(&mut self, rhs: Self) {
        self.value += rhs.value;
    }
}

impl Sum for BN256 {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|x, y| x + y).unwrap_or(Self::zero())
    }
}

impl Sub for BN256 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Self { value: self.value.sub(rhs.value) }
    }
}

impl SubAssign for BN256 {
    fn sub_assign(&mut self, rhs: Self) {
        self.value -= rhs.value;
    }
}

impl Neg for BN256 {
    type Output = Self;

    fn neg(self) -> Self::Output {
        self * Self::neg_one()
    }
}

impl Mul for BN256 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        Self { value: self.value * rhs.value }
    }
}

impl MulAssign for BN256 {
    fn mul_assign(&mut self, rhs: Self) {
        self.value *= rhs.value;
    }
}

impl Product for BN256 {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|x, y| x * y).unwrap_or(Self::one())
    }
}

impl Div for BN256 {
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, rhs: Self) -> Self {
        self * rhs.inverse()
    }
}
