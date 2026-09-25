//! `GF(2^64)` in the polynomial basis of `x^64 + x^4 + x^3 + x + 1`.

use alloc::vec::Vec;
use core::fmt::{self, Debug, Display, Formatter};
use core::iter::{Product, Sum};
use core::mem::ManuallyDrop;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use core::slice;

use num_bigint::BigUint;
use p3_field::op_assign_macros::{
    impl_add_assign, impl_add_base_field, impl_div_methods, impl_mul_base_field, impl_mul_methods,
    impl_sub_assign, impl_sub_base_field, ring_sum,
};
use p3_field::{Algebra, Field, Packable, PrimeCharacteristicRing, RawDataSerializable};
use p3_maybe_rayon::prelude::*;
use rand::Rng;
use rand::distr::{Distribution, StandardUniform};
use serde::{Deserialize, Serialize};

use crate::cantor::CANTOR_BASIS_128;
use crate::gf2::characteristic_two_methods;
use crate::tower::TowerLevel;
use crate::{BinaryField8, BinaryField16, BinaryField32, BinaryField64, Gf2, clmul};

/// Elements one parallel task converts between the two 64-bit bases.
const TABLE_CHUNK: usize = 1 << 16;

/// The Cantor basis in this representation, carried over from the tower's own.
///
/// Both representations then span the same additive transform domain.
const CANTOR_BASIS: [u64; 64] = {
    let mut basis = [0u64; 64];
    let mut i = 0;
    while i < 64 {
        basis[i] = clmul::tower_image_64(CANTOR_BASIS_128[i] as u64);
        i += 1;
    }
    basis
};

/// The bit pattern of the multiplicative generator of the tower representation.
///
/// Its image here is the generator, so the change of basis carries one onto the other.
const TOWER_GENERATOR: u64 = 0x1_0000_0004;

/// The tower's GF(4) generator in this polynomial basis.
pub(crate) const GF4_GENERATOR: u64 = clmul::tower_image_64(2);

/// The generator of this level over the one below, in this representation.
///
/// The tower carries that element as a single basis vector, at bit 32.
const ALPHA: u64 = clmul::tower_image_64(1 << 32);

/// The binary field `GF(2^64)` modulo `x^64 + x^4 + x^3 + x + 1`.
///
/// Bit `i` stores the coefficient of `x^i`, and every 64-bit pattern is a distinct element.
///
/// One carryless product covers a whole multiplication, against four at 128 bits.
///
/// A committed column here is also half the volume of a 128-bit one.
///
/// Multiplication, squaring, the square root and inversion here are all table-free.
///
/// The change of basis to and from the tower is table-driven, and so is not.
#[derive(Copy, Clone, Default, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
#[repr(transparent)]
#[must_use]
pub struct Poly64(u64);

impl Poly64 {
    /// The number of bits of an element.
    pub(crate) const BITS: usize = 64;

    /// Converts a whole tower-basis table in its existing allocation.
    ///
    /// Every entry is the image of the corresponding tower element under the field isomorphism.
    pub fn from_tower_vec(values: Vec<BinaryField64>) -> Vec<Self> {
        let mut values = ManuallyDrop::new(values);
        let (ptr, len, capacity) = (values.as_mut_ptr(), values.len(), values.capacity());

        // Both fields are transparent over a 64-bit word.
        // Reusing the allocation avoids copying the residual tables before sumcheck.
        let words = unsafe { slice::from_raw_parts_mut(ptr.cast::<u64>(), len) };
        words.par_chunks_mut(TABLE_CHUNK).for_each(|chunk| {
            chunk
                .iter_mut()
                .for_each(|word| *word = clmul::tower_to_poly_64(*word));
        });

        // SAFETY: both element types have the size and alignment of `u64`.
        // Every 64-bit pattern is canonical in both fields.
        unsafe { Vec::from_raw_parts(ptr.cast::<Self>(), len, capacity) }
    }

    /// The element with the given polynomial-basis coordinates.
    ///
    /// Bit `i` is the coefficient of `x^i`, and every pattern is a valid element.
    #[inline]
    pub const fn new(coordinates: u64) -> Self {
        Self(coordinates)
    }

    /// The polynomial-basis coordinates of this element.
    #[must_use]
    #[inline]
    pub const fn to_bits(self) -> u64 {
        self.0
    }

    /// Construct a field element from its little-endian byte representation.
    ///
    /// Every byte string of this length is a valid element.
    #[inline]
    pub const fn from_le_bytes(bytes: [u8; 8]) -> Self {
        Self(u64::from_le_bytes(bytes))
    }

    /// The inverse of this element, with zero sent to zero.
    ///
    /// The chain runs whatever the operand is, so its cost says nothing about the value.
    #[inline]
    pub fn invert_or_zero(self) -> Self {
        Self(clmul::poly_inverse_64(self.0))
    }
}

impl Packable for Poly64 {}

impl Display for Poly64 {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Display::fmt(&self.0, f)
    }
}

impl Debug for Poly64 {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Debug::fmt(&self.0, f)
    }
}

impl Distribution<Poly64> for StandardUniform {
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Poly64 {
        let mut bytes = [0u8; 8];
        rng.fill_bytes(&mut bytes);
        Poly64::from_le_bytes(bytes)
    }
}

impl PrimeCharacteristicRing for Poly64 {
    #[inline]
    fn zero_vec(len: usize) -> Vec<Self> {
        let mut values = ManuallyDrop::new(alloc::vec![0u64; len]);
        // SAFETY: the transparent wrapper has exactly the integer's layout, and zero is
        // canonical. The allocation retains its original size and alignment.
        unsafe { Vec::from_raw_parts(values.as_mut_ptr().cast(), values.len(), values.capacity()) }
    }

    type PrimeSubfield = Gf2;

    const ZERO: Self = Self(0);
    const ONE: Self = Self(1);
    // The characteristic is 2, so `TWO = ONE + ONE = ZERO`.
    const TWO: Self = Self(0);
    // The characteristic is 2, so `NEG_ONE = ONE`.
    const NEG_ONE: Self = Self(1);

    #[inline]
    fn from_prime_subfield(f: Self::PrimeSubfield) -> Self {
        Self::from_bool(f.is_one())
    }

    #[inline]
    fn from_bool(b: bool) -> Self {
        Self(u64::from(b))
    }

    characteristic_two_methods!();

    #[inline]
    fn square(&self) -> Self {
        Self(clmul::poly_square_64(self.0))
    }

    #[inline]
    fn dot_product<const N: usize>(u: &[Self; N], v: &[Self; N]) -> Self {
        // Reduction is linear, so an entire sum pays for it only once.
        Self(clmul::poly_dot_64(u.iter().zip(v).map(|(a, b)| (a.0, b.0))))
    }
}

impl Field for Poly64 {
    type Packing = Self;

    const GENERATOR: Self = Self(clmul::tower_image_64(TOWER_GENERATOR));

    #[inline]
    fn try_inverse(&self) -> Option<Self> {
        // The chain runs whatever the operand, so its cost says nothing about the value.
        //
        // Zero is outside the multiplicative group and the chain already sends it to itself.
        let inverse = self.invert_or_zero();
        (self.0 != 0).then_some(inverse)
    }

    #[inline]
    fn try_sqrt(&self) -> Option<Self> {
        // Separate even and odd coefficients to invert the squaring map directly.
        Some(Self(clmul::poly_sqrt_64(self.0)))
    }

    #[inline]
    fn order() -> BigUint {
        BigUint::from(1u8) << Self::BITS
    }

    /// An element of `GF(2^n)` is exactly `n` bits wide.
    #[inline]
    fn bits() -> usize {
        Self::BITS
    }

    /// An injective enumeration that puts the embedded GF(4) first.
    ///
    /// Swapping two pairs of indices preserves the usual polynomial-basis enumeration
    /// everywhere else and makes the zerocheck's first interpolation nodes slicable.
    #[inline]
    fn interpolation_node(i: usize) -> Self {
        let bits = i as u64;
        Self(match bits {
            2 => GF4_GENERATOR,
            3 => GF4_GENERATOR ^ 1,
            value if value == GF4_GENERATOR => 2,
            value if value == (GF4_GENERATOR ^ 1) => 3,
            value => value,
        })
    }

    #[inline]
    fn position_bit_node(bit: usize) -> Self {
        Self::new(bit as u64)
    }
}

impl RawDataSerializable for Poly64 {
    const NUM_BYTES: usize = 8;

    #[inline]
    fn into_bytes(self) -> impl IntoIterator<Item = u8> {
        self.0.to_le_bytes()
    }
}

impl crate::tower::private::Sealed for Poly64 {}

impl TowerLevel for Poly64 {
    type Repr = u64;

    const LOG_BITS: usize = 6;

    #[inline]
    fn from_repr(r: Self::Repr) -> Self {
        Self(r)
    }

    #[inline]
    fn to_repr(self) -> Self::Repr {
        self.0
    }

    /// The tower scales by a basis vector, which costs it a shift.
    ///
    /// Here the same element is an arbitrary one, so it costs a full product.
    #[inline]
    fn mul_alpha(self) -> Self {
        self * Self(ALPHA)
    }

    /// # Panics
    /// Panics if the stream ends before a whole element has been read.
    #[inline]
    fn from_le_byte_iter(mut bytes: impl Iterator<Item = u8>) -> Self {
        let mut buffer = [0u8; 8];
        for byte in &mut buffer {
            *byte = bytes
                .next()
                .expect("byte stream ended before a whole element was read");
        }
        Self::from_le_bytes(buffer)
    }

    /// # Panics
    /// Panics if the index is at least the bit width of the field.
    #[inline]
    fn cantor_basis(i: usize) -> Self {
        assert!(i < Self::BITS, "Cantor basis index out of range");
        Self(CANTOR_BASIS[i])
    }
}

impl From<BinaryField64> for Poly64 {
    /// The same field element, seen in the polynomial basis.
    #[inline]
    fn from(x: BinaryField64) -> Self {
        Self(clmul::tower_to_poly_64(x.to_repr()))
    }
}

impl From<Poly64> for BinaryField64 {
    /// The same field element, seen in the tower basis.
    #[inline]
    fn from(x: Poly64) -> Self {
        Self::from_repr(clmul::poly_to_tower_64(x.0))
    }
}

impl Add<BinaryField64> for Poly64 {
    type Output = Self;

    #[inline]
    fn add(self, rhs: BinaryField64) -> Self {
        self + Self::from(rhs)
    }
}

impl Sub<BinaryField64> for Poly64 {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: BinaryField64) -> Self {
        self - Self::from(rhs)
    }
}

impl Mul<BinaryField64> for Poly64 {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: BinaryField64) -> Self {
        self * Self::from(rhs)
    }
}

impl Algebra<BinaryField64> for Poly64 {}

macro_rules! impl_narrow_algebra {
    ($($field:ty),* $(,)?) => {$(
        impl From<$field> for Poly64 {
            #[inline]
            fn from(x: $field) -> Self {
                Self::from(BinaryField64::from(x))
            }
        }

        impl_add_base_field!(Poly64, $field);
        impl_sub_base_field!(Poly64, $field);
        impl_mul_base_field!(Poly64, $field);

        impl Algebra<$field> for Poly64 {}
    )*};
}

impl_narrow_algebra!(BinaryField8, BinaryField16, BinaryField32);

impl Add for Poly64 {
    type Output = Self;

    #[inline]
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn add(self, rhs: Self) -> Self {
        // Addition in characteristic 2 is `XOR`.
        Self(self.0 ^ rhs.0)
    }
}

impl Sub for Poly64 {
    type Output = Self;

    #[inline]
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn sub(self, rhs: Self) -> Self {
        // Subtraction coincides with addition in characteristic 2.
        self + rhs
    }
}

impl Neg for Poly64 {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        // `-x = x` in characteristic 2.
        self
    }
}

impl Mul for Poly64 {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self(clmul::poly_mul_64(self.0, rhs.0))
    }
}

impl_add_assign!(Poly64);
impl_sub_assign!(Poly64);
impl_mul_methods!(Poly64);
impl_div_methods!(Poly64, Poly64);
ring_sum!(Poly64);

impl From<Gf2> for Poly64 {
    /// `GF(2)` is the prime subfield, embedded as `{ZERO, ONE}`.
    #[inline]
    fn from(x: Gf2) -> Self {
        Self::from_prime_subfield(x)
    }
}

impl_add_base_field!(Poly64, Gf2);
impl_sub_base_field!(Poly64, Gf2);
impl_mul_base_field!(Poly64, Gf2);

impl Algebra<Gf2> for Poly64 {}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use p3_field::{Algebra, Field, PrimeCharacteristicRing};
    use proptest::prelude::*;

    use super::Poly64;
    use crate::tower::TowerLevel;
    use crate::{BinaryField8, BinaryField16, BinaryField32, BinaryField64, Gf2};

    #[test]
    fn tower_table_conversion_and_narrow_algebras_match_the_tower() {
        // The bulk conversion must be the scalar field isomorphism without another allocation.
        let table: Vec<BinaryField64> = (0..257)
            .map(|i| BinaryField64::from_repr(i * 0x9e37_79b9))
            .collect();
        let expected: Vec<Poly64> = table.iter().copied().map(Poly64::from).collect();
        let allocation = table.as_ptr();
        let converted = Poly64::from_tower_vec(table);
        assert_eq!(converted, expected);
        assert_eq!(converted.as_ptr().cast::<BinaryField64>(), allocation);

        // Each supported subfield must use the same embedding as tower multiplication.
        const fn assert_algebra<F: PrimeCharacteristicRing, A: Algebra<F>>() {}
        assert_algebra::<BinaryField8, Poly64>();
        assert_algebra::<BinaryField16, Poly64>();
        assert_algebra::<BinaryField32, Poly64>();
        assert_algebra::<BinaryField64, Poly64>();

        let x = BinaryField64::from_repr(0xd6e8_feb8_6659_fd93);
        let x_poly = Poly64::from(x);
        let scalar = BinaryField32::from_repr(0xa5c3_19e7);
        assert_eq!(
            BinaryField64::from(x_poly * scalar),
            x * BinaryField64::from(scalar)
        );
    }

    #[test]
    fn the_change_of_basis_fixes_the_constants() {
        // Any field isomorphism fixes zero and one.
        assert_eq!(Poly64::from(BinaryField64::ZERO), Poly64::ZERO);
        assert_eq!(Poly64::from(BinaryField64::ONE), Poly64::ONE);
    }

    #[test]
    fn the_generator_is_the_image_of_the_tower_generator() {
        // Both representations are the same field, so one generator maps onto the other.
        //
        // This is what pins the transcribed bit pattern the constant is built from.
        assert_eq!(Poly64::GENERATOR, Poly64::from(BinaryField64::GENERATOR));
    }

    #[test]
    fn the_generator_has_the_full_order() {
        // Invariant: 2^64 - 1 = 3 * 5 * 17 * 257 * 641 * 65537 * 6700417.
        //
        // A generator survives every proper-divisor power.
        let order = u64::MAX;
        for prime in [3u64, 5, 17, 257, 641, 65537, 6_700_417] {
            assert_ne!(
                Poly64::GENERATOR.exp_u64(order / prime),
                Poly64::ONE,
                "{prime}"
            );
        }
        assert_eq!(Poly64::GENERATOR.exp_u64(order), Poly64::ONE);
    }

    #[test]
    fn the_cantor_basis_satisfies_its_recurrence() {
        // Invariant: v_0 = 1 and v_i^2 + v_i = v_{i-1}.
        assert_eq!(Poly64::cantor_basis(0), Poly64::ONE);
        for i in 1..Poly64::BITS {
            let v = Poly64::cantor_basis(i);
            assert_eq!(v.square() + v, Poly64::cantor_basis(i - 1), "vector {i}");
        }

        // And it is the image of the tower's own, so both span the same transform domain.
        for i in 0..Poly64::BITS {
            assert_eq!(
                Poly64::cantor_basis(i),
                Poly64::from(BinaryField64::cantor_basis(i)),
                "vector {i}"
            );
        }
    }

    #[test]
    #[should_panic = "Cantor basis index out of range"]
    fn the_cantor_basis_rejects_an_index_beyond_the_field() {
        let _vector = Poly64::cantor_basis(64);
    }

    #[test]
    #[should_panic = "byte stream ended before a whole element was read"]
    fn a_truncated_byte_stream_is_rejected() {
        // Seven bytes is one short of an element.
        let _element = Poly64::from_le_byte_iter([0u8; 7].into_iter());
    }

    #[test]
    fn zero_vectors_preserve_layout_and_support_growth() {
        for len in [0, 1, 33, 1024] {
            let mut values = Poly64::zero_vec(len);
            assert_eq!(values.len(), len);
            assert!(values.iter().all(|x| *x == Poly64::ZERO));
            values.push(Poly64::ONE);
            values.reserve(100);
            assert_eq!(values.pop(), Some(Poly64::ONE));
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(2000))]

        #[test]
        fn the_change_of_basis_round_trips(bits: u64) {
            let x = Poly64::from_repr(bits);
            prop_assert_eq!(Poly64::from(BinaryField64::from(x)), x);
        }

        /// The change of basis must carry products, not merely sums.
        #[test]
        fn the_change_of_basis_is_a_ring_isomorphism(a: u64, b: u64) {
            let (x, y) = (BinaryField64::from_repr(a), BinaryField64::from_repr(b));
            prop_assert_eq!(Poly64::from(x + y), Poly64::from(x) + Poly64::from(y));
            prop_assert_eq!(
                Poly64::from(x.reference_mul(y)),
                Poly64::from(x) * Poly64::from(y)
            );
        }

        /// Scaling by the level's own generator must agree across the two representations.
        #[test]
        fn scaling_by_the_generator_agrees_with_the_tower(bits: u64) {
            let x = BinaryField64::from_repr(bits);
            prop_assert_eq!(Poly64::from(x).mul_alpha(), Poly64::from(x.mul_alpha()));
        }

        #[test]
        fn the_prime_subfield_embeds_as_zero_and_one(bit: bool) {
            prop_assert_eq!(Poly64::from(Gf2::from_bool(bit)), Poly64::from_bool(bit));
        }

        #[test]
        fn the_byte_stream_reads_back_what_the_element_wrote(bits: u64) {
            use p3_field::RawDataSerializable;
            let x = Poly64::from_repr(bits);
            prop_assert_eq!(Poly64::from_le_byte_iter(x.into_bytes().into_iter()), x);
        }
    }

    #[test]
    fn serde_accepts_every_bit_pattern() {
        // No bit is masked off, so nothing is out of range.
        for bits in [0u64, 1, u64::MAX, 1 << 63] {
            let x = Poly64::from_repr(bits);
            let encoded = serde_json::to_string(&x).unwrap();
            assert_eq!(serde_json::from_str::<Poly64>(&encoded).unwrap(), x);
        }
    }
}
