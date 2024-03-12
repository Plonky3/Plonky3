//! The prime field known as BN254, defined as `F_p` where `p = 21888242871839275222246405745257275088696311157297823662689037894645226208583`.

// #![no_std]

extern crate alloc;

mod poseidon2;

use alloc::vec::Vec;
use ff::{Field as ff_Field, PrimeField as ff_PrimeField};
use num::bigint::BigUint;
use p3_baby_bear::BabyBear;
use p3_field::PrimeField32;
use rand::distributions::{Distribution, Standard};
use rand::Rng;

use core::fmt;
use core::fmt::{Debug, Display, Formatter};
use core::hash::{Hash, Hasher};
use core::iter::{Product, Sum};
use core::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

use serde::ser::SerializeSeq;

use p3_field::{AbstractField, Field, Packable, PrimeField};
use serde::{Deserialize, Deserializer, Serialize};

pub use poseidon2::DiffusionMatrixBN254;

#[derive(ff_PrimeField)]
#[PrimeFieldModulus = "21888242871839275222246405745257275088548364400416034343698204186575808495617"]
#[PrimeFieldGenerator = "7"]
#[PrimeFieldReprEndianness = "little"]
struct FpBN256([u64; 4]);

/// The BN254 curve base field prime, defined as `F_p` where `p = 21888242871839275222246405745257275088696311157297823662689037894645226208583`.
#[derive(Copy, Clone, Default, Eq, PartialEq)]
pub struct BN254 {
    value: FpBN256,
}

impl BN254 {
    /// Returns the value of the field element.
    pub fn to_bytes_le(&self) -> Vec<u8> {
        self.value.to_repr().as_ref().to_vec()
    }
}

impl Serialize for BN254 {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let repr = self.value.to_repr();
        let bytes = repr.as_ref();

        let mut seq = serializer.serialize_seq(Some(bytes.len()))?;
        for e in bytes {
            seq.serialize_element(&e)?;
        }
        seq.end()
    }
}

impl<'de> Deserialize<'de> for BN254 {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let bytes: Vec<u8> = Deserialize::deserialize(d)?;

        let mut res = <FpBN256 as ff::PrimeField>::Repr::default();

        for (i, digit) in res.0.as_mut().iter_mut().enumerate() {
            *digit = bytes[i];
        }

        let value = FpBN256::from_repr(res);

        if value.is_some().into() {
            Ok(Self { value: value.unwrap() })
        } else {
            Err(serde::de::Error::custom("Invalid field element"))
        }
    }
}

impl Packable for BN254 {}

impl Hash for BN254 {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for byte in self.value.to_repr().as_ref().iter() {
            state.write_u8(*byte);
        }
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
        <FpBN256 as Debug>::fmt(&self.value, f)
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
        Self { value: FpBN256::ZERO }
    }
    fn one() -> Self {
        Self { value: FpBN256::ONE }
    }
    fn two() -> Self {
        let two = FpBN256::from(2u64);
        Self { value: two }
    }

    fn neg_one() -> Self {
        let neg_one = FpBN256::ZERO - FpBN256::ONE;
        Self { value: neg_one }
    }

    #[inline]
    fn from_f(f: Self::F) -> Self {
        f
    }

    fn from_bool(b: bool) -> Self {
        Self { value: FpBN256::from(b as u64) }
    }

    fn from_canonical_u8(n: u8) -> Self {
        Self { value: FpBN256::from(n as u64) }
    }

    fn from_canonical_u16(n: u16) -> Self {
        Self { value: FpBN256::from(n as u64) }
    }

    fn from_canonical_u32(n: u32) -> Self {
        Self { value: FpBN256::from(n as u64) }
    }

    fn from_canonical_u64(n: u64) -> Self {
        Self { value: FpBN256::from(n) }
    }

    fn from_canonical_usize(n: usize) -> Self {
        Self { value: FpBN256::from(n as u64) }
    }

    fn from_wrapped_u32(n: u32) -> Self {
        Self { value: FpBN256::from(n as u64) }
    }

    fn from_wrapped_u64(n: u64) -> Self {
        Self { value: FpBN256::from(n) }
    }

    fn generator() -> Self {
        let seven = FpBN256::from(7u64);
        Self { value: seven }
    }
}

impl Field for BN254 {
    // TODO: Add cfg-guarded Packing for AVX2, NEON, etc.
    type Packing = Self;

    fn is_zero(&self) -> bool {
        self.value.is_zero().into()
    }

    fn try_inverse(&self) -> Option<Self> {
        let inverse = self.value.invert();

        if inverse.is_some().into() {
            Some(Self { value: inverse.unwrap() })
        } else {
            None
        }
    }
}

impl PrimeField for BN254 {}

impl Add for BN254 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self { value: self.value + rhs.value }
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

impl Distribution<BN254> for Standard {
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> BN254 {
        let value = FpBN256::random(rng);

        BN254 { value }
    }
}


pub fn convert_bn254_element_to_babybear_elements(element: BN254) -> [BabyBear; 8] {
    let mut val = BigUint::from_bytes_le(&element.to_bytes_le());
    let mut ret: [BabyBear; 8] = [BabyBear::zero(); 8];
    for i in 0..8 {
        let rem: BigUint = val.clone() % 0x78000001u32;
        val /= 0x78000001u32;
        ret[i] = BabyBear::from_canonical_u32(rem.to_u32_digits()[0]);
    }
    ret
}

pub fn convert_babybear_elements_to_bn254_element(elements: &[BabyBear]) -> BN254 {
    assert!(elements.len() <= 8);

    // TODO: This should be a const
    let alpha = BN254::from_canonical_u32(0x78000001);

    let mut sum = BN254::zero();

    for &term in elements.iter().rev() {
        sum = sum * alpha + BN254::from_canonical_u32(term.as_canonical_u32());
    }

    sum
}