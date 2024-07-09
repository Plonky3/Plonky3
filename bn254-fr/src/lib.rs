//! The scalar field of the BN254 curve, defined as `F_r` where `r = 21888242871839275222246405745257275088548364400416034343698204186575808495617`.

mod poseidon2;

use core::fmt;
use core::fmt::{Debug, Display, Formatter};
use core::hash::{Hash, Hasher};
use core::iter::{Product, Sum};
use core::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

use ff::{Field as FFField, PrimeField as FFPrimeField};
pub use halo2curves::bn256::Fr as FFBn254Fr;
use halo2curves::serde::SerdeObject;
use num_bigint::BigUint;
use p3_field::{AbstractField, Field, Packable, PrimeField, TwoAdicField};
pub use poseidon2::DiffusionMatrixBN254;
use rand::distributions::{Distribution, Standard};
use rand::Rng;
use serde::{Deserialize, Deserializer, Serialize};

/// The BN254 curve scalar field prime, defined as `F_r` where `r = 21888242871839275222246405745257275088548364400416034343698204186575808495617`.
#[derive(Copy, Clone, Default, Eq, PartialEq)]
pub struct Bn254Fr {
    pub value: FFBn254Fr,
}

impl Bn254Fr {
    pub(crate) const fn new(value: FFBn254Fr) -> Self {
        Self { value }
    }
}

impl Serialize for Bn254Fr {
    /// Serializes to raw bytes, which are typically of the Montgomery representation of the field element.
    // See https://github.com/privacy-scaling-explorations/halo2curves/blob/d34e9e46f7daacd194739455de3b356ca6c03206/derive/src/field/mod.rs#L493
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let bytes = self.value.to_raw_bytes();
        serializer.serialize_bytes(&bytes)
    }
}

impl<'de> Deserialize<'de> for Bn254Fr {
    /// Deserializes from raw bytes, which are typically of the Montgomery representation of the field element.
    /// Performs a check that the deserialized field element corresponds to a value less than the field modulus, and
    /// returns error otherwise.
    // See https://github.com/privacy-scaling-explorations/halo2curves/blob/d34e9e46f7daacd194739455de3b356ca6c03206/derive/src/field/mod.rs#L485
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let bytes: Vec<u8> = Deserialize::deserialize(d)?;

        let value = FFBn254Fr::from_raw_bytes(&bytes);

        value
            .map(Self::new)
            .ok_or(serde::de::Error::custom("Invalid field element"))
    }
}

impl Packable for Bn254Fr {}

impl Hash for Bn254Fr {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for byte in self.value.to_repr().as_ref().iter() {
            state.write_u8(*byte);
        }
    }
}

impl Ord for Bn254Fr {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.value.cmp(&other.value)
    }
}

impl PartialOrd for Bn254Fr {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Display for Bn254Fr {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        <FFBn254Fr as Debug>::fmt(&self.value, f)
    }
}

impl Debug for Bn254Fr {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Debug::fmt(&self.value, f)
    }
}

impl AbstractField for Bn254Fr {
    type F = Self;

    fn zero() -> Self {
        Self::new(FFBn254Fr::ZERO)
    }
    fn one() -> Self {
        Self::new(FFBn254Fr::ONE)
    }
    fn two() -> Self {
        Self::new(FFBn254Fr::from(2u64))
    }

    fn neg_one() -> Self {
        Self::new(-FFBn254Fr::ONE)
    }

    #[inline]
    fn from_f(f: Self::F) -> Self {
        f
    }

    fn from_bool(b: bool) -> Self {
        Self::new(FFBn254Fr::from(b as u64))
    }

    fn from_canonical_u8(n: u8) -> Self {
        Self::new(FFBn254Fr::from(n as u64))
    }

    fn from_canonical_u16(n: u16) -> Self {
        Self::new(FFBn254Fr::from(n as u64))
    }

    fn from_canonical_u32(n: u32) -> Self {
        Self::new(FFBn254Fr::from(n as u64))
    }

    fn from_canonical_u64(n: u64) -> Self {
        Self::new(FFBn254Fr::from(n))
    }

    fn from_canonical_usize(n: usize) -> Self {
        Self::new(FFBn254Fr::from(n as u64))
    }

    fn from_wrapped_u32(n: u32) -> Self {
        Self::new(FFBn254Fr::from(n as u64))
    }

    fn from_wrapped_u64(n: u64) -> Self {
        Self::new(FFBn254Fr::from(n))
    }

    fn generator() -> Self {
        Self::new(FFBn254Fr::from(5u64))
    }
}

impl Field for Bn254Fr {
    type Packing = Self;

    fn is_zero(&self) -> bool {
        self.value.is_zero().into()
    }

    fn try_inverse(&self) -> Option<Self> {
        let inverse = self.value.invert();

        if inverse.is_some().into() {
            Some(Self::new(inverse.unwrap()))
        } else {
            None
        }
    }

    /// r = 0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001
    fn order() -> BigUint {
        BigUint::new(vec![
            0xf0000001, 0x43e1f593, 0x79b97091, 0x2833e848, 0x8181585d, 0xb85045b6, 0xe131a029,
            0x30644e72,
        ])
    }

    fn multiplicative_group_factors() -> Vec<(BigUint, usize)> {
        vec![
            (BigUint::from(2u8), 28),
            (BigUint::from(3u8), 2),
            (BigUint::from(13u8), 1),
            (BigUint::from(29u8), 1),
            (BigUint::from(983u16), 1),
            (BigUint::from(11003u16), 1),
            (BigUint::from(237073u32), 1),
            (BigUint::from(405928799u32), 1),
            (BigUint::from(1670836401704629u64), 1),
            (BigUint::from(13818364434197438864469338081u128), 1),
        ]
    }
}

impl PrimeField for Bn254Fr {
    fn as_canonical_biguint(&self) -> BigUint {
        let repr = self.value.to_repr();
        let le_bytes = repr.as_ref();
        BigUint::from_bytes_le(le_bytes)
    }
}

impl Add for Bn254Fr {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self::new(self.value + rhs.value)
    }
}

impl AddAssign for Bn254Fr {
    fn add_assign(&mut self, rhs: Self) {
        self.value += rhs.value;
    }
}

impl Sum for Bn254Fr {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|x, y| x + y).unwrap_or(Self::zero())
    }
}

impl Sub for Bn254Fr {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Self::new(self.value.sub(rhs.value))
    }
}

impl SubAssign for Bn254Fr {
    fn sub_assign(&mut self, rhs: Self) {
        self.value -= rhs.value;
    }
}

impl Neg for Bn254Fr {
    type Output = Self;

    fn neg(self) -> Self::Output {
        self * Self::neg_one()
    }
}

impl Mul for Bn254Fr {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        Self::new(self.value * rhs.value)
    }
}

impl MulAssign for Bn254Fr {
    fn mul_assign(&mut self, rhs: Self) {
        self.value *= rhs.value;
    }
}

impl Product for Bn254Fr {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|x, y| x * y).unwrap_or(Self::one())
    }
}

impl Div for Bn254Fr {
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, rhs: Self) -> Self {
        self * rhs.inverse()
    }
}

impl Distribution<Bn254Fr> for Standard {
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Bn254Fr {
        Bn254Fr::new(FFBn254Fr::random(rng))
    }
}

impl TwoAdicField for Bn254Fr {
    const TWO_ADICITY: usize = FFBn254Fr::S as usize;

    fn two_adic_generator(bits: usize) -> Self {
        let mut omega = FFBn254Fr::ROOT_OF_UNITY;
        for _ in bits..Self::TWO_ADICITY {
            omega = omega.square();
        }
        Self::new(omega)
    }
}

#[cfg(test)]
mod tests {
    use num_traits::One;
    use p3_field_testing::test_field;

    use super::*;

    type F = Bn254Fr;

    #[test]
    fn test_bn254fr() {
        let f = F::new(FFBn254Fr::from_u128(100));
        assert_eq!(f.as_canonical_biguint(), BigUint::new(vec![100]));

        let f = F::from_canonical_u64(0);
        assert!(f.is_zero());

        let f = F::new(FFBn254Fr::from_str_vartime(&F::order().to_str_radix(10)).unwrap());
        assert!(f.is_zero());

        assert_eq!(F::generator().as_canonical_biguint(), BigUint::new(vec![5]));

        let f_1 = F::new(FFBn254Fr::from_u128(1));
        let f_1_copy = F::new(FFBn254Fr::from_u128(1));

        let expected_result = F::zero();
        assert_eq!(f_1 - f_1_copy, expected_result);

        let expected_result = F::new(FFBn254Fr::from_u128(2));
        assert_eq!(f_1 + f_1_copy, expected_result);

        let f_2 = F::new(FFBn254Fr::from_u128(2));
        let expected_result = F::new(FFBn254Fr::from_u128(3));
        assert_eq!(f_1 + f_1_copy * f_2, expected_result);

        let expected_result = F::new(FFBn254Fr::from_u128(5));
        assert_eq!(f_1 + f_2 * f_2, expected_result);

        let f_r_minus_1 = F::new(
            FFBn254Fr::from_str_vartime(&(F::order() - BigUint::one()).to_str_radix(10)).unwrap(),
        );
        let expected_result = F::zero();
        assert_eq!(f_1 + f_r_minus_1, expected_result);

        let f_r_minus_2 = F::new(
            FFBn254Fr::from_str_vartime(&(F::order() - BigUint::new(vec![2])).to_str_radix(10))
                .unwrap(),
        );
        let expected_result = F::new(
            FFBn254Fr::from_str_vartime(&(F::order() - BigUint::new(vec![3])).to_str_radix(10))
                .unwrap(),
        );
        assert_eq!(f_r_minus_1 + f_r_minus_2, expected_result);

        let expected_result = F::new(FFBn254Fr::from_u128(1));
        assert_eq!(f_r_minus_1 - f_r_minus_2, expected_result);

        let expected_result = f_r_minus_1;
        assert_eq!(f_r_minus_2 - f_r_minus_1, expected_result);

        let expected_result = f_r_minus_2;
        assert_eq!(f_r_minus_1 - f_1, expected_result);

        let expected_result = F::new(FFBn254Fr::from_u128(3));
        assert_eq!(f_2 * f_2 - f_1, expected_result);

        // Generator check
        let expected_multiplicative_group_generator = F::new(FFBn254Fr::from_u128(5));
        assert_eq!(F::generator(), expected_multiplicative_group_generator);

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

        let f_r_minus_1_serialized = serde_json::to_string(&f_r_minus_1).unwrap();
        let f_r_minus_1_deserialized: F = serde_json::from_str(&f_r_minus_1_serialized).unwrap();
        assert_eq!(f_r_minus_1, f_r_minus_1_deserialized);

        let f_r_minus_2_serialized = serde_json::to_string(&f_r_minus_2).unwrap();
        let f_r_minus_2_deserialized: F = serde_json::from_str(&f_r_minus_2_serialized).unwrap();
        assert_eq!(f_r_minus_2, f_r_minus_2_deserialized);
    }

    test_field!(crate::Bn254Fr);
}
