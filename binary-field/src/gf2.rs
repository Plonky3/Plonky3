use core::fmt::{self, Debug, Display, Formatter};
use core::iter::{Product, Sum};
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use num_bigint::BigUint;
use p3_field::integers::QuotientMap;
use p3_field::op_assign_macros::{
    impl_add_assign, impl_div_methods, impl_mul_methods, impl_sub_assign, ring_sum,
};
use p3_field::{
    Field, Packable, PrimeCharacteristicRing, PrimeField, PrimeField32, PrimeField64,
    RawDataSerializable, quotient_map_large_iint, quotient_map_large_uint,
};
use rand::Rng;
use rand::distr::{Distribution, StandardUniform};
use serde::de::Error;
use serde::{Deserialize, Deserializer, Serialize};

/// The prime field `GF(2) = {0, 1}`, with addition given by `XOR` and multiplication by `AND`.
///
/// This is the base case of the characteristic-2 tower `GF(2) ⊂ GF(4) ⊂ … ⊂ GF(2^128)`.
///
/// The serde encoding is canonical: every field element has exactly one valid byte
/// representation (see the manual [`Deserialize`] impl below).
#[derive(Copy, Clone, Default, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
#[repr(transparent)]
#[must_use]
pub struct Gf2(u8);

impl Gf2 {
    /// Create a new field element from a bit.
    ///
    /// Only the least significant bit of `bit` is used.
    #[inline]
    const fn new(bit: u8) -> Self {
        Self(bit & 1)
    }

    /// Construct a field element from its little-endian byte representation.
    ///
    /// Every byte is a valid input; only its least significant bit is used.
    #[inline]
    pub const fn from_le_bytes(bytes: [u8; 1]) -> Self {
        Self::new(bytes[0])
    }
}

impl Packable for Gf2 {}

impl<'de> Deserialize<'de> for Gf2 {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let val = u8::deserialize(d)?;
        // Reject non-canonical encodings so a proof cannot be re-encoded without the witness.
        // Only `0` and `1` are canonical for `GF(2)`.
        if val < 2 {
            Ok(Self(val))
        } else {
            Err(D::Error::custom("Value is out of range"))
        }
    }
}

impl Display for Gf2 {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Display::fmt(&self.0, f)
    }
}

impl Debug for Gf2 {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Debug::fmt(&self.0, f)
    }
}

impl Distribution<Gf2> for StandardUniform {
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Gf2 {
        Gf2::new((rng.next_u32() & 1) as u8)
    }
}

impl PrimeCharacteristicRing for Gf2 {
    type PrimeSubfield = Self;

    const ZERO: Self = Self(0);
    const ONE: Self = Self(1);
    // The characteristic is 2, so `TWO = ONE + ONE = ZERO`.
    const TWO: Self = Self(0);
    // The characteristic is 2, so `NEG_ONE = ONE`.
    const NEG_ONE: Self = Self(1);

    #[inline]
    fn from_prime_subfield(f: Self::PrimeSubfield) -> Self {
        f
    }

    #[inline]
    fn from_bool(b: bool) -> Self {
        Self(b as u8)
    }

    #[inline]
    fn double(&self) -> Self {
        // `a + a = 0` in characteristic 2.
        Self::ZERO
    }

    /// # Panics
    /// Always panics: `2` is not invertible in characteristic 2.
    #[inline]
    fn halve(&self) -> Self {
        panic!("halve is undefined in characteristic 2")
    }

    #[inline]
    fn square(&self) -> Self {
        // `0^2 = 0` and `1^2 = 1`.
        *self
    }

    #[inline]
    fn xor(&self, y: &Self) -> Self {
        *self + *y
    }

    #[inline]
    fn mul_2exp_u64(&self, exp: u64) -> Self {
        if exp == 0 { *self } else { Self::ZERO }
    }

    /// # Panics
    /// Always panics: `2` is not invertible in characteristic 2.
    #[inline]
    fn div_2exp_u64(&self, _exp: u64) -> Self {
        panic!("div_2exp_u64 is undefined in characteristic 2")
    }
}

impl Field for Gf2 {
    type Packing = Self;

    // The only nonzero element, `1`, trivially generates the (trivial) multiplicative group.
    const GENERATOR: Self = Self::ONE;

    #[inline]
    fn try_inverse(&self) -> Option<Self> {
        (self.0 == 1).then_some(Self::ONE)
    }

    #[inline]
    fn try_sqrt(&self) -> Option<Self> {
        // `0^2 = 0` and `1^2 = 1`, so every element is its own (unique) square root.
        Some(*self)
    }

    #[inline]
    fn order() -> BigUint {
        BigUint::from(2u8)
    }

    /// The enumeration `interpolation_node(0) = 0, interpolation_node(1) = 1` is the only
    /// injective enumeration of the two elements of `GF(2)` satisfying `Field::interpolation_node`'s
    /// contract. It is not injective beyond index `1`.
    #[inline]
    fn interpolation_node(i: usize) -> Self {
        Self::new(i as u8)
    }
}

impl QuotientMap<u8> for Gf2 {
    /// Convert a given `u8` integer into an element of the `Gf2` field.
    #[inline]
    fn from_int(int: u8) -> Self {
        Self::new(int)
    }

    /// Convert a given `u8` integer into an element of the `Gf2` field.
    ///
    /// Returns `None` if the input does not lie in the range `[0, 1]`.
    #[inline]
    fn from_canonical_checked(int: u8) -> Option<Self> {
        (int < 2).then(|| Self::new(int))
    }

    /// Convert a given `u8` integer into an element of the `Gf2` field.
    ///
    /// # Safety
    /// The caller must guarantee that `int < 2`.
    #[inline]
    unsafe fn from_canonical_unchecked(int: u8) -> Self {
        // SAFETY: the caller guarantees `int < 2`, so `int` is already the canonical
        // representative and masking to its least significant bit is a no-op.
        debug_assert!(int < 2);
        Self::new(int)
    }
}

impl QuotientMap<i8> for Gf2 {
    /// Convert a given `i8` integer into an element of the `Gf2` field.
    #[inline]
    fn from_int(int: i8) -> Self {
        // Two's-complement `& 1` extracts the parity bit regardless of sign.
        Self::new((int & 1) as u8)
    }

    /// Convert a given `i8` integer into an element of the `Gf2` field.
    ///
    /// Returns `None` if the input does not lie in the range `[0, 1]`.
    #[inline]
    fn from_canonical_checked(int: i8) -> Option<Self> {
        (0..2).contains(&int).then(|| Self::new(int as u8))
    }

    /// Convert a given `i8` integer into an element of the `Gf2` field.
    ///
    /// # Safety
    /// The caller must guarantee that `int` lies in `[0, 1]`.
    #[inline]
    unsafe fn from_canonical_unchecked(int: i8) -> Self {
        // SAFETY: the caller guarantees `0 <= int < 2`, so `int` is already the
        // canonical representative and casting to `u8` is exact.
        debug_assert!((0..2).contains(&int));
        Self::new(int as u8)
    }
}

// `u8` is the smallest integer type, so `QuotientMap<u8>` is implemented by hand above.
quotient_map_large_uint!(Gf2, u8, 2u8, "`[0, 1]`", "`[0, 1]`", [u16, u32, u64, u128]);

// `i8` is the smallest signed integer type, so `QuotientMap<i8>` is implemented by hand above.
quotient_map_large_iint!(
    Gf2,
    i8,
    "`[0, 1]`",
    "`[0, 1]`",
    [(i16, u16), (i32, u32), (i64, u64), (i128, u128)]
);

impl PrimeField for Gf2 {
    #[inline]
    fn as_canonical_biguint(&self) -> BigUint {
        BigUint::from(self.0)
    }
}

impl PrimeField64 for Gf2 {
    const ORDER_U64: u64 = 2;

    #[inline]
    fn as_canonical_u64(&self) -> u64 {
        self.0 as u64
    }
}

impl PrimeField32 for Gf2 {
    const ORDER_U32: u32 = 2;

    #[inline]
    fn as_canonical_u32(&self) -> u32 {
        self.0 as u32
    }
}

impl RawDataSerializable for Gf2 {
    const NUM_BYTES: usize = 1;

    #[inline]
    fn into_bytes(self) -> impl IntoIterator<Item = u8> {
        [self.0]
    }
}

impl Add for Gf2 {
    type Output = Self;

    #[inline]
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn add(self, rhs: Self) -> Self {
        // Addition in `GF(2)` is `XOR`.
        Self(self.0 ^ rhs.0)
    }
}

impl Sub for Gf2 {
    type Output = Self;

    #[inline]
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn sub(self, rhs: Self) -> Self {
        // Subtraction coincides with addition in characteristic 2.
        self + rhs
    }
}

impl Neg for Gf2 {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        // `-x = x` in characteristic 2.
        self
    }
}

impl Mul for Gf2 {
    type Output = Self;

    #[inline]
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn mul(self, rhs: Self) -> Self {
        // Multiplication in `GF(2)` is `AND`.
        Self(self.0 & rhs.0)
    }
}

impl_add_assign!(Gf2);
impl_sub_assign!(Gf2);
impl_mul_methods!(Gf2);
impl_div_methods!(Gf2, Gf2);
ring_sum!(Gf2);

#[cfg(test)]
mod tests {
    use p3_field::integers::QuotientMap;
    use p3_field::{
        Field, PrimeCharacteristicRing, PrimeField, PrimeField32, PrimeField64, RawDataSerializable,
    };

    use super::Gf2;

    #[test]
    fn arithmetic_is_boolean() {
        assert_eq!(Gf2::ONE + Gf2::ONE, Gf2::ZERO);
        assert_eq!(Gf2::TWO, Gf2::ZERO);
        assert_eq!(Gf2::NEG_ONE, Gf2::ONE);
        assert_eq!(Gf2::ONE * Gf2::ONE, Gf2::ONE);
        assert_eq!(Gf2::ONE.inverse(), Gf2::ONE);
        assert_eq!(Gf2::ZERO.try_inverse(), None);
        assert_eq!(Gf2::from_u64(7), Gf2::ONE);
        assert_eq!(Gf2::from_u64(6), Gf2::ZERO);
    }

    // `p3_field_testing`'s `test_prime_field!`/`test_prime_field_64!`/`test_prime_field_32!`
    // are not usable against `Gf2`: they hard-code assumptions that only hold for fields with
    // a "reasonably large" order (e.g. `generate_from_small_int_tests!` asserts
    // `from_canonical_checked` is `Some` for literal test values up to `108`;
    // `generate_from_large_u_int_tests!`/`generate_from_large_i_int_tests!` unconditionally
    // call `.halve()`, which is undefined in characteristic 2; and
    // `test_prime_field_32!`'s raw-data-serializable/JSON-boundary sub-tests assume a 4-byte,
    // ~31-bit representative). The tests below cover the same ground by hand, adapted to
    // `GF(2)`'s actual canonical range of `{0, 1}`.

    /// Every integer type required by `PrimeField`'s `QuotientMap<Int>` bound reduces mod 2,
    /// and `from_canonical_checked`/`from_canonical_unchecked` agree with `from_int` exactly on
    /// `{0, 1}` and reject (or are only guaranteed correct for) everything else.
    #[test]
    fn quotient_map_covers_every_integer_type() {
        macro_rules! check_unsigned {
            ($int:ty) => {
                assert_eq!(<Gf2 as QuotientMap<$int>>::from_int(0), Gf2::ZERO);
                assert_eq!(<Gf2 as QuotientMap<$int>>::from_int(1), Gf2::ONE);
                assert_eq!(<Gf2 as QuotientMap<$int>>::from_int(2), Gf2::ZERO);
                assert_eq!(<Gf2 as QuotientMap<$int>>::from_int(3), Gf2::ONE);
                assert_eq!(<Gf2 as QuotientMap<$int>>::from_int(<$int>::MAX), Gf2::ONE);
                assert_eq!(
                    <Gf2 as QuotientMap<$int>>::from_canonical_checked(0),
                    Some(Gf2::ZERO)
                );
                assert_eq!(
                    <Gf2 as QuotientMap<$int>>::from_canonical_checked(1),
                    Some(Gf2::ONE)
                );
                assert_eq!(<Gf2 as QuotientMap<$int>>::from_canonical_checked(2), None);
                assert_eq!(
                    <Gf2 as QuotientMap<$int>>::from_canonical_checked(<$int>::MAX),
                    None
                );
                unsafe {
                    assert_eq!(
                        <Gf2 as QuotientMap<$int>>::from_canonical_unchecked(0),
                        Gf2::ZERO
                    );
                    assert_eq!(
                        <Gf2 as QuotientMap<$int>>::from_canonical_unchecked(1),
                        Gf2::ONE
                    );
                }
            };
        }
        macro_rules! check_signed {
            ($int:ty) => {
                assert_eq!(<Gf2 as QuotientMap<$int>>::from_int(0), Gf2::ZERO);
                assert_eq!(<Gf2 as QuotientMap<$int>>::from_int(1), Gf2::ONE);
                assert_eq!(<Gf2 as QuotientMap<$int>>::from_int(-1), Gf2::ONE);
                assert_eq!(<Gf2 as QuotientMap<$int>>::from_int(-2), Gf2::ZERO);
                assert_eq!(<Gf2 as QuotientMap<$int>>::from_int(-3), Gf2::ONE);
                assert_eq!(<Gf2 as QuotientMap<$int>>::from_int(<$int>::MIN), Gf2::ZERO);
                assert_eq!(<Gf2 as QuotientMap<$int>>::from_int(<$int>::MAX), Gf2::ONE);
                assert_eq!(
                    <Gf2 as QuotientMap<$int>>::from_canonical_checked(0),
                    Some(Gf2::ZERO)
                );
                assert_eq!(
                    <Gf2 as QuotientMap<$int>>::from_canonical_checked(1),
                    Some(Gf2::ONE)
                );
                assert_eq!(<Gf2 as QuotientMap<$int>>::from_canonical_checked(2), None);
                assert_eq!(<Gf2 as QuotientMap<$int>>::from_canonical_checked(-1), None);
                assert_eq!(
                    <Gf2 as QuotientMap<$int>>::from_canonical_checked(<$int>::MIN),
                    None
                );
                unsafe {
                    assert_eq!(
                        <Gf2 as QuotientMap<$int>>::from_canonical_unchecked(0),
                        Gf2::ZERO
                    );
                    assert_eq!(
                        <Gf2 as QuotientMap<$int>>::from_canonical_unchecked(1),
                        Gf2::ONE
                    );
                }
            };
        }

        check_unsigned!(u8);
        check_unsigned!(u16);
        check_unsigned!(u32);
        check_unsigned!(u64);
        check_unsigned!(u128);
        check_unsigned!(usize);
        check_signed!(i8);
        check_signed!(i16);
        check_signed!(i32);
        check_signed!(i64);
        check_signed!(i128);
        check_signed!(isize);
    }

    /// Large values reduce mod 2 rather than saturating or panicking, on both sides of a
    /// `u8`-internal-representation boundary (256) and near the top of wider integer ranges.
    #[test]
    fn quotient_map_large_value_reduction() {
        assert_eq!(<Gf2 as QuotientMap<u16>>::from_int(60_000), Gf2::ZERO);
        assert_eq!(<Gf2 as QuotientMap<u16>>::from_int(60_001), Gf2::ONE);
        assert_eq!(
            <Gf2 as QuotientMap<u32>>::from_int(4_000_000_000),
            Gf2::ZERO
        );
        assert_eq!(<Gf2 as QuotientMap<u32>>::from_int(4_000_000_001), Gf2::ONE);
        assert_eq!(
            <Gf2 as QuotientMap<u64>>::from_int(18_000_000_000_000_000_002),
            Gf2::ZERO
        );
        assert_eq!(
            <Gf2 as QuotientMap<u128>>::from_int(u128::MAX - 1),
            Gf2::ZERO
        );
        assert_eq!(
            <Gf2 as QuotientMap<i32>>::from_int(-2_000_000_001),
            Gf2::ONE
        );
        assert_eq!(
            <Gf2 as QuotientMap<i64>>::from_int(-9_000_000_000_000_000_002),
            Gf2::ZERO
        );
        assert_eq!(
            <Gf2 as QuotientMap<i128>>::from_int(i128::MIN + 1),
            Gf2::ONE
        );
    }

    #[test]
    fn as_canonical_and_raw_bytes() {
        assert_eq!(Gf2::ORDER_U32, 2);
        assert_eq!(Gf2::ORDER_U64, 2);
        assert_eq!(Gf2::ZERO.as_canonical_u32(), 0);
        assert_eq!(Gf2::ONE.as_canonical_u32(), 1);
        assert_eq!(Gf2::ZERO.as_canonical_u64(), 0);
        assert_eq!(Gf2::ONE.as_canonical_u64(), 1);
        assert_eq!(
            Gf2::ONE.as_canonical_biguint(),
            num_bigint::BigUint::from(1u8)
        );
        assert_eq!(
            Gf2::ZERO.as_canonical_biguint(),
            num_bigint::BigUint::from(0u8)
        );

        assert_eq!(Gf2::NUM_BYTES, 1);
        let mut ones_bytes = Gf2::ONE.into_bytes().into_iter();
        assert_eq!(ones_bytes.next(), Some(1u8));
        assert_eq!(ones_bytes.next(), None);
        let mut zeros_bytes = Gf2::ZERO.into_bytes().into_iter();
        assert_eq!(zeros_bytes.next(), Some(0u8));
        assert_eq!(zeros_bytes.next(), None);
    }

    #[test]
    fn field_order_and_interpolation_nodes() {
        assert_eq!(Gf2::order(), num_bigint::BigUint::from(2u8));
        assert_eq!(Gf2::bits(), 2);
        assert_eq!(Gf2::interpolation_node(0), Gf2::ZERO);
        assert_eq!(Gf2::interpolation_node(1), Gf2::ONE);
    }

    /// Serialization always emits the canonical byte, and deserialization rejects any byte
    /// other than `0`/`1` — this is what makes finding #1's manual `Deserialize` impl (in
    /// place of the derive) actually necessary: without it, deserializing untrusted or
    /// corrupted bytes could silently construct a `Gf2` whose invariant (`.0 in {0, 1}`) is
    /// broken, and that would propagate wrong results through arithmetic.
    #[test]
    fn serde_round_trip_rejects_non_canonical_encodings() {
        assert_eq!(serde_json::to_string(&Gf2::ZERO).unwrap(), "0");
        assert_eq!(serde_json::to_string(&Gf2::ONE).unwrap(), "1");

        let decoded_zero: Gf2 = serde_json::from_str("0").unwrap();
        let decoded_one: Gf2 = serde_json::from_str("1").unwrap();
        assert_eq!(decoded_zero, Gf2::ZERO);
        assert_eq!(decoded_one, Gf2::ONE);

        // Every byte other than 0/1 must be rejected, not silently truncated into range.
        assert!(serde_json::from_str::<Gf2>("2").is_err());
        assert!(serde_json::from_str::<Gf2>("255").is_err());
    }
}
