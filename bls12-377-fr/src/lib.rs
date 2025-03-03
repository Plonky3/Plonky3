//! The scalar field of the BLS12-377 curve, defined as `F_r` where `r = 8444461749428370424248824938781546531375899335154063827935233455917409239041`
mod poseidon2;
pub mod rc;

use core::fmt;
use core::fmt::{Debug, Display, Formatter};
use core::hash::{Hash, Hasher};
use core::iter::{Product, Sum};
use core::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

pub use ark_bls12_377::Fr as FF_Bls12_377Fr;
use ark_ff::{
    AdditiveGroup, BigInteger, FftField, Field as ArkField, PrimeField as ArkPrimeField,
    UniformRand, Zero,
};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use num_bigint::BigUint;
use p3_field::{Field, FieldAlgebra, Packable, PrimeField, TwoAdicField};
pub use poseidon2::Poseidon2Bls12337;
use rand::distributions::{Distribution, Standard};
use rand::Rng;
use serde::{Deserialize, Deserializer, Serialize};

/// The BLS12-377 curve scalar field prime, defined as `F_r` where `r = 8444461749428370424248824938781546531375899335154063827935233455917409239041`.
#[derive(Copy, Clone, Eq)]
pub struct Bls12_377Fr {
    pub value: FF_Bls12_377Fr,
}

impl Bls12_377Fr {
    pub const fn new(value: FF_Bls12_377Fr) -> Self {
        Self { value }
    }
}

impl Serialize for Bls12_377Fr {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut bytes = Vec::new();
        self.value
            .serialize_compressed(&mut bytes)
            .map_err(|err| serde::ser::Error::custom(err.to_string()))?;
        serializer.serialize_bytes(&bytes)
    }
}

impl<'de> Deserialize<'de> for Bls12_377Fr {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let bytes: Vec<u8> = Deserialize::deserialize(d)?;

        let value = FF_Bls12_377Fr::deserialize_compressed(&bytes[..]);

        value
            .map(Self::new)
            .map_err(|_err| serde::de::Error::custom("Invalid field element"))
    }
}

impl Default for Bls12_377Fr {
    fn default() -> Self {
        Self::new(FF_Bls12_377Fr::default())
    }
}

impl PartialEq for Bls12_377Fr {
    fn eq(&self, other: &Self) -> bool {
        self.value.eq(&other.value)
    }
}

impl Packable for Bls12_377Fr {}

impl Hash for Bls12_377Fr {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let mut serialized_bytes = Vec::new();
        self.value
            .serialize_compressed(&mut serialized_bytes)
            .unwrap();

        serialized_bytes.hash(state);
    }
}

impl Ord for Bls12_377Fr {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.value.cmp(&other.value)
    }
}

impl PartialOrd for Bls12_377Fr {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Display for Bls12_377Fr {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        <Bls12_377Fr as Debug>::fmt(self, f)
    }
}

impl Debug for Bls12_377Fr {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Debug::fmt(&self.value, f)
    }
}

impl FieldAlgebra for Bls12_377Fr {
    type F = Self;

    const ZERO: Self = Self::new(FF_Bls12_377Fr::ZERO);
    const ONE: Self = Self::new(FF_Bls12_377Fr::ONE);

    const TWO: Self = Self::new(FF_Bls12_377Fr::new(ark_ff::biginteger::BigInt::new([
        2_u64, 0, 0, 0,
    ])));

    // r - 1 = 0x12ab655e9a2ca55660b44d1e5c37b00159aa76fed00000010a11800000000000
    const NEG_ONE: Self = Self::new(FF_Bls12_377Fr::new(ark_ff::biginteger::BigInt::new([
        0x0a11800000000000,
        0x59aa76fed0000001,
        0x60b44d1e5c37b001,
        0x12ab655e9a2ca556,
    ])));

    #[inline]
    fn from_f(f: Self::F) -> Self {
        f
    }

    fn from_bool(b: bool) -> Self {
        Self::new(FF_Bls12_377Fr::from(b as u64))
    }

    fn from_canonical_u8(n: u8) -> Self {
        Self::new(FF_Bls12_377Fr::from(n as u64))
    }

    fn from_canonical_u16(n: u16) -> Self {
        Self::new(FF_Bls12_377Fr::from(n as u64))
    }

    fn from_canonical_u32(n: u32) -> Self {
        Self::new(FF_Bls12_377Fr::from(n as u64))
    }

    fn from_canonical_u64(n: u64) -> Self {
        Self::new(FF_Bls12_377Fr::from(n))
    }

    fn from_canonical_usize(n: usize) -> Self {
        Self::new(FF_Bls12_377Fr::from(n as u64))
    }

    fn from_wrapped_u32(n: u32) -> Self {
        Self::new(FF_Bls12_377Fr::from(n as u64))
    }

    fn from_wrapped_u64(n: u64) -> Self {
        Self::new(FF_Bls12_377Fr::from(n))
    }
}

impl Field for Bls12_377Fr {
    type Packing = Self;

    // generator is 22
    const GENERATOR: Self = Self::new(FF_Bls12_377Fr::new(ark_ff::biginteger::BigInt::new([
        22_u64, 0, 0, 0,
    ])));

    fn is_zero(&self) -> bool {
        self.value.is_zero()
    }

    fn try_inverse(&self) -> Option<Self> {
        self.value.inverse().map(Self::new)
    }

    /// r = 0x12ab655e9a2ca55660b44d1e5c37b00159aa76fed00000010a11800000000001
    fn order() -> BigUint {
        BigUint::new(vec![
            0x00000001, 0x0a118000, 0xd0000001, 0x59aa76fe, 0x5c37b001, 0x60b44d1e, 0x9a2ca556,
            0x12ab655e,
        ])
    }

    fn multiplicative_group_factors() -> Vec<(BigUint, usize)> {
        vec![
            (BigUint::from(2u8), 47),
            (BigUint::from(3u8), 1),
            (BigUint::from(5u8), 1),
            (BigUint::from(7u8), 1),
            (BigUint::from(13u16), 1),
            (BigUint::from(499u16), 1),
            (BigUint::from(958612291309063373u128), 1),
            (BigUint::from(9586122913090633729u128), 2),
        ]
    }
}

impl PrimeField for Bls12_377Fr {
    fn as_canonical_biguint(&self) -> BigUint {
        BigUint::from_bytes_le(self.value.into_bigint().to_bytes_le().as_slice())
    }
}

impl Add for Bls12_377Fr {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self::new(self.value + rhs.value)
    }
}

impl AddAssign for Bls12_377Fr {
    fn add_assign(&mut self, rhs: Self) {
        self.value += rhs.value;
    }
}

impl Sum for Bls12_377Fr {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|x, y| x + y).unwrap_or(Self::ZERO)
    }
}

impl Sub for Bls12_377Fr {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Self::new(self.value.sub(rhs.value))
    }
}

impl SubAssign for Bls12_377Fr {
    fn sub_assign(&mut self, rhs: Self) {
        self.value -= rhs.value;
    }
}

impl Neg for Bls12_377Fr {
    type Output = Self;

    fn neg(self) -> Self::Output {
        self * Self::NEG_ONE
    }
}

impl Mul for Bls12_377Fr {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        Self::new(self.value * rhs.value)
    }
}

impl MulAssign for Bls12_377Fr {
    fn mul_assign(&mut self, rhs: Self) {
        self.value *= rhs.value;
    }
}

impl Product for Bls12_377Fr {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|x, y| x * y).unwrap_or(Self::ONE)
    }
}

impl Div for Bls12_377Fr {
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, rhs: Self) -> Self {
        self * rhs.inverse()
    }
}

impl Distribution<Bls12_377Fr> for Standard {
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Bls12_377Fr {
        Bls12_377Fr::new(FF_Bls12_377Fr::rand(rng))
    }
}

impl TwoAdicField for Bls12_377Fr {
    const TWO_ADICITY: usize = FF_Bls12_377Fr::TWO_ADICITY as usize;

    fn two_adic_generator(bits: usize) -> Self {
        let mut omega = FF_Bls12_377Fr::TWO_ADIC_ROOT_OF_UNITY;
        for _ in bits..Self::TWO_ADICITY {
            omega = omega.square();
        }
        Self::new(omega)
    }
}

#[cfg(test)]
mod tests {
    use std::str::FromStr;

    use num_traits::One;
    use p3_field_testing::{test_field, test_two_adic_field};

    use super::*;

    type F = Bls12_377Fr;

    #[test]
    fn test_bls12_337fr() {
        let f = F::new(FF_Bls12_377Fr::from(100));
        assert_eq!(f.as_canonical_biguint(), BigUint::new(vec![100]));

        let f = F::from_canonical_u64(0);
        assert!(f.is_zero());

        let f = F::new(FF_Bls12_377Fr::from_str(&F::order().to_str_radix(10)).unwrap());
        assert!(f.is_zero());

        assert_eq!(F::GENERATOR.as_canonical_biguint(), BigUint::new(vec![22]));

        let f_1 = F::new(FF_Bls12_377Fr::from(1_u128));
        let f_1_copy = F::new(FF_Bls12_377Fr::from(1_u128));

        let expected_result = F::ZERO;
        assert_eq!(f_1 - f_1_copy, expected_result);

        let expected_result = F::new(FF_Bls12_377Fr::from(2_u128));
        assert_eq!(f_1 + f_1_copy, expected_result);

        let f_2 = F::new(FF_Bls12_377Fr::from(2_u128));
        let expected_result = F::new(FF_Bls12_377Fr::from(3_u128));
        assert_eq!(f_1 + f_1_copy * f_2, expected_result);

        let expected_result = F::new(FF_Bls12_377Fr::from(5_u128));
        assert_eq!(f_1 + f_2 * f_2, expected_result);

        let f_r_minus_1 = F::new(
            FF_Bls12_377Fr::from_str(&(F::order() - BigUint::one()).to_str_radix(10)).unwrap(),
        );
        let expected_result = F::ZERO;
        assert_eq!(f_1 + f_r_minus_1, expected_result);

        let f_r_minus_2 = F::new(
            FF_Bls12_377Fr::from_str(&(F::order() - BigUint::new(vec![2])).to_str_radix(10))
                .unwrap(),
        );
        let expected_result = F::new(
            FF_Bls12_377Fr::from_str(&(F::order() - BigUint::new(vec![3])).to_str_radix(10))
                .unwrap(),
        );
        assert_eq!(f_r_minus_1 + f_r_minus_2, expected_result);

        let expected_result = F::new(FF_Bls12_377Fr::from(1_u128));
        assert_eq!(f_r_minus_1 - f_r_minus_2, expected_result);

        let expected_result = f_r_minus_1;
        assert_eq!(f_r_minus_2 - f_r_minus_1, expected_result);

        let expected_result = f_r_minus_2;
        assert_eq!(f_r_minus_1 - f_1, expected_result);

        let expected_result = F::new(FF_Bls12_377Fr::from(3_u128));
        assert_eq!(f_2 * f_2 - f_1, expected_result);

        // Generator check
        let expected_multiplicative_group_generator = F::new(FF_Bls12_377Fr::from(22_u128));
        assert_eq!(F::GENERATOR, expected_multiplicative_group_generator);

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

    test_field!(crate::Bls12_377Fr);
    test_two_adic_field!(crate::Bls12_377Fr);
}
