//! The prime field known as BN254, defined as `F_p` where `p = 21888242871839275222246405745257275088696311157297823662689037894645226208583`.

#![no_std]

extern crate alloc;

mod poseidon2;

use alloc::vec::Vec;
use zkhash::fields::bn256::FpBN256;

use core::fmt;
use core::fmt::{Debug, Display, Formatter};
use core::hash::{Hash, Hasher};
use core::iter::{Product, Sum};
use core::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

use serde::ser::SerializeSeq;

use ark_ff::{Field as af, Zero, One};
use p3_field::{AbstractField, Field, Packable, PrimeField};
use serde::{Deserialize, Deserializer, Serialize};

/// The BN254 curve base field prime, defined as `F_p` where `p = 21888242871839275222246405745257275088696311157297823662689037894645226208583`.
#[derive(Copy, Clone, Default)]
pub struct BN254 {
    value: FpBN256,
}

impl Serialize for BN254 {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let limbs = self.value.0.0;

        let mut seq = serializer.serialize_seq(Some(limbs.len()))?;
        for e in limbs {
            seq.serialize_element(&e)?;
        }
        seq.end()
    }
}

impl<'de> Deserialize<'de> for BN254 {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let limbs: Vec<u64> = Deserialize::deserialize(d)?;
        let mut value = FpBN256::zero();
        for (i, limb) in limbs.iter().enumerate() {
            value.0.0[i] = *limb;
        }
        Ok(Self { value })
    }
}


impl PartialEq for BN254 {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl Eq for BN254 {}

impl Packable for BN254 {}

impl Hash for BN254 {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.value.hash(state);
    }
}

impl Ord for BN254 {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.value.cmp(&other.value)
    }
}

impl PartialOrd for BN254 {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Display for BN254 {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Display::fmt(&self.value, f)
    }
}

impl Debug for BN254 {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Debug::fmt(&self.value, f)
    }
}

impl AbstractField for BN254 {
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

impl Field for BN254 {
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

impl PrimeField for BN254 {}

impl Add for BN254 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self { value: self.value.add(rhs.value) }
    }
}

impl AddAssign for BN254 {
    fn add_assign(&mut self, rhs: Self) {
        self.value += rhs.value;
    }
}

impl Sum for BN254 {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|x, y| x + y).unwrap_or(Self::zero())
    }
}

impl Sub for BN254 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Self { value: self.value.sub(rhs.value) }
    }
}

impl SubAssign for BN254 {
    fn sub_assign(&mut self, rhs: Self) {
        self.value -= rhs.value;
    }
}

impl Neg for BN254 {
    type Output = Self;

    fn neg(self) -> Self::Output {
        self * Self::neg_one()
    }
}

impl Mul for BN254 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        Self { value: self.value * rhs.value }
    }
}

impl MulAssign for BN254 {
    fn mul_assign(&mut self, rhs: Self) {
        self.value *= rhs.value;
    }
}

impl Product for BN254 {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|x, y| x * y).unwrap_or(Self::one())
    }
}

impl Div for BN254 {
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, rhs: Self) -> Self {
        self * rhs.inverse()
    }
}
