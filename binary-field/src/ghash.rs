//! `GF(2^128)` in the polynomial basis of `x^128 + x^7 + x^2 + x + 1`.

use core::fmt::{self, Debug, Display, Formatter};
use core::iter::{Product, Sum};
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use num_bigint::BigUint;
use p3_field::op_assign_macros::{
    impl_add_assign, impl_add_base_field, impl_div_methods, impl_mul_base_field, impl_mul_methods,
    impl_sub_assign, impl_sub_base_field, ring_sum,
};
use p3_field::{Algebra, Field, Packable, PrimeCharacteristicRing, RawDataSerializable};
use rand::Rng;
use rand::distr::{Distribution, StandardUniform};
use serde::{Deserialize, Serialize};

use crate::cantor::CANTOR_BASIS_128;
use crate::tower::TowerLevel;
use crate::{BinaryField128, Gf2, clmul};

/// The bit pattern of the multiplicative generator of the tower representation.
///
/// Its image is the generator here, so the conversion carries one onto the other.
const TOWER_GENERATOR: u128 = 0x1_0000_0000_0000_0005;

/// The tower's generator of `GF(2^128)` over `GF(2^64)`, in this representation.
///
/// The tower carries that element as a single basis vector, at bit 64.
const ALPHA: u128 = clmul::tower_image_128(1 << 64);

/// The inverse of an element known to be nonzero, by the addition chain over Frobenius maps.
#[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
#[inline]
fn invert_nonzero(x: Ghash128) -> Ghash128 {
    Ghash128(clmul::poly_inverse_128(x.0))
}

/// The inverse of an element known to be nonzero, through the tower norm.
///
/// The tower recurses through the norm down to a `GF(2^8)` lookup table.
/// Everywhere the addition chain is not faster, that beats it for no table at all.
#[cfg(not(all(target_arch = "x86_64", target_feature = "pclmulqdq")))]
#[inline]
fn invert_nonzero(x: Ghash128) -> Ghash128 {
    Ghash128::from(BinaryField128::from(x).inverse())
}

/// The Cantor basis in this representation.
///
/// These are the images of the tower's own basis vectors.
/// Both representations therefore span the same additive NTT domain.
const CANTOR_BASIS: [u128; 128] = {
    let mut basis = [0u128; 128];
    let mut i = 0;
    while i < 128 {
        basis[i] = clmul::tower_image_128(CANTOR_BASIS_128[i]);
        i += 1;
    }
    basis
};

/// The binary field GF(2^128) modulo x^128 + x^7 + x^2 + x + 1.
///
/// Bit i stores the coefficient of x^i.
/// Every 128-bit pattern is a distinct field element.
///
/// Hardware carryless multiplication operates directly on this representation.
/// The tower representation instead provides byte-aligned subfields.
///
/// Inversion and conversions to or from the tower use operand-indexed tables.
/// They are not constant-time for secret inputs.
///
/// NIST GCM blocks use the opposite coefficient order.
/// Reverse all 128 bits of a big-endian block integer before interpreting it here.
#[derive(Copy, Clone, Default, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
#[repr(transparent)]
#[must_use]
pub struct Ghash128(u128);

impl Ghash128 {
    /// The number of bits of an element.
    pub(crate) const BITS: usize = 128;

    /// Construct a field element from its little-endian byte representation.
    ///
    /// Every byte string of this length is a valid element.
    #[inline]
    pub const fn from_le_bytes(bytes: [u8; 16]) -> Self {
        Self(u128::from_le_bytes(bytes))
    }
}

impl Packable for Ghash128 {}

impl Display for Ghash128 {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Display::fmt(&self.0, f)
    }
}

impl Debug for Ghash128 {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Debug::fmt(&self.0, f)
    }
}

impl Distribution<Ghash128> for StandardUniform {
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Ghash128 {
        let mut bytes = [0u8; 16];
        rng.fill_bytes(&mut bytes);
        Ghash128::from_le_bytes(bytes)
    }
}

impl PrimeCharacteristicRing for Ghash128 {
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
        Self(u128::from(b))
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
        Self(clmul::poly_square_128(self.0))
    }

    #[inline]
    fn dot_product<const N: usize>(u: &[Self; N], v: &[Self; N]) -> Self {
        // Reduction is linear, so an entire sum pays for it only once.
        Self(clmul::poly_dot_128(
            u.iter().zip(v).map(|(a, b)| (a.0, b.0)),
        ))
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

impl Field for Ghash128 {
    // One element is one 128-bit lane, so a wide carryless multiply packs several of them.
    // Which register that is, and whether there is one at all, is settled in `packed`.
    //
    // Without a packing the alias resolves to this type itself, which is why the lint is off.
    #[allow(clippy::use_self)]
    type Packing = crate::packed::Packing;

    const GENERATOR: Self = Self(clmul::tower_image_128(TOWER_GENERATOR));

    /// Invert through precomputed squaring maps on carryless-multiply targets.
    /// The software backend instead uses the recursive tower norm.
    ///
    /// The operand-indexed tables make this operation variable-time.
    #[inline]
    fn try_inverse(&self) -> Option<Self> {
        // Zero has no multiplicative inverse.
        (self.0 != 0).then(|| invert_nonzero(*self))
    }

    #[inline]
    fn try_sqrt(&self) -> Option<Self> {
        // Separate even and odd coefficients to invert the squaring map directly.
        Some(Self(clmul::poly_sqrt_128(self.0)))
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

    /// The enumeration by bit pattern.
    ///
    /// The coordinates of the returned element over `GF(2)` are the bits of the index.
    /// A pointer is never 128 bits wide, so every index is in range.
    #[inline]
    fn interpolation_node(i: usize) -> Self {
        Self(i as u128)
    }
}

impl RawDataSerializable for Ghash128 {
    const NUM_BYTES: usize = 16;

    #[inline]
    fn into_bytes(self) -> impl IntoIterator<Item = u8> {
        self.0.to_le_bytes()
    }
}

impl crate::tower::private::Sealed for Ghash128 {}

impl TowerLevel for Ghash128 {
    type Repr = u128;

    const LOG_BITS: usize = 7;

    #[inline]
    fn from_repr(r: Self::Repr) -> Self {
        Self(r)
    }

    #[inline]
    fn to_repr(self) -> Self::Repr {
        self.0
    }

    /// That generator is a basis element in the tower, which gets this for a shift.
    /// Here it is an arbitrary element, so it costs a full product.
    #[inline]
    fn mul_alpha(self) -> Self {
        self * Self(ALPHA)
    }

    /// # Panics
    /// Panics if the stream ends before a whole element has been read.
    #[inline]
    fn from_le_byte_iter(mut bytes: impl Iterator<Item = u8>) -> Self {
        let mut buffer = [0u8; 16];
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

impl From<BinaryField128> for Ghash128 {
    /// The same field element, seen in the polynomial basis.
    #[inline]
    fn from(x: BinaryField128) -> Self {
        Self(clmul::tower_to_poly_128(x.to_repr()))
    }
}

impl From<Ghash128> for BinaryField128 {
    /// The same field element, seen in the tower basis.
    #[inline]
    fn from(x: Ghash128) -> Self {
        Self::from_repr(clmul::poly_to_tower_128(x.0))
    }
}

impl Add for Ghash128 {
    type Output = Self;

    #[inline]
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn add(self, rhs: Self) -> Self {
        // Addition in characteristic 2 is `XOR`.
        Self(self.0 ^ rhs.0)
    }
}

impl Sub for Ghash128 {
    type Output = Self;

    #[inline]
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn sub(self, rhs: Self) -> Self {
        // Subtraction coincides with addition in characteristic 2.
        self + rhs
    }
}

impl Neg for Ghash128 {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        // `-x = x` in characteristic 2.
        self
    }
}

impl Mul for Ghash128 {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self(clmul::poly_mul_128(self.0, rhs.0))
    }
}

impl_add_assign!(Ghash128);
impl_sub_assign!(Ghash128);
impl_mul_methods!(Ghash128);
impl_div_methods!(Ghash128, Ghash128);
ring_sum!(Ghash128);

impl From<Gf2> for Ghash128 {
    /// `GF(2)` is the prime subfield, embedded as `{ZERO, ONE}`.
    #[inline]
    fn from(x: Gf2) -> Self {
        Self::from_prime_subfield(x)
    }
}

impl_add_base_field!(Ghash128, Gf2);
impl_sub_base_field!(Ghash128, Gf2);
impl_mul_base_field!(Ghash128, Gf2);

impl Algebra<Gf2> for Ghash128 {}

#[cfg(test)]
mod tests {
    extern crate std;

    use std::vec::Vec;

    use p3_field::{Field, PrimeCharacteristicRing, RawDataSerializable};
    use proptest::prelude::*;

    use super::{CANTOR_BASIS, Ghash128};
    use crate::tower::TowerLevel;
    use crate::{BinaryField128, Gf2};

    /// The tower element with the given bit pattern.
    fn tower(bits: u128) -> BinaryField128 {
        BinaryField128::from_repr(bits)
    }

    #[test]
    fn the_change_of_basis_fixes_the_constants() {
        // Any field isomorphism fixes zero and one.
        assert_eq!(Ghash128::from(BinaryField128::ZERO), Ghash128::ZERO);
        assert_eq!(Ghash128::from(BinaryField128::ONE), Ghash128::ONE);
    }

    #[test]
    fn the_generator_is_the_image_of_the_tower_generator() {
        // Both representations are the same field, so one generator maps onto the other.
        // This is what pins the transcribed bit pattern the constant is built from.
        assert_eq!(
            Ghash128::GENERATOR,
            Ghash128::from(BinaryField128::GENERATOR)
        );
    }

    #[test]
    fn the_modulus_reduces_the_way_the_polynomial_says() {
        // x^127 * x = x^128 = x^7 + x^2 + x + 1, the tail spelled 0x87.
        let x = Ghash128::from_repr(2);
        let top = Ghash128::from_repr(1 << 127);
        assert_eq!(top * x, Ghash128::from_repr(0x87));

        // x * x = x^2, nowhere near the modulus.
        assert_eq!(x * x, Ghash128::from_repr(4));

        // (x + 1)^2 = x^2 + 1, since the cross term doubles to zero.
        assert_eq!(Ghash128::from_repr(3).square(), Ghash128::from_repr(5));
    }

    #[test]
    fn the_cantor_basis_satisfies_its_recurrence() {
        // Invariant: v_0 = 1 and v_i^2 + v_i = v_{i-1}.
        // This is what the additive NTT domain is built on.
        assert_eq!(Ghash128::cantor_basis(0), Ghash128::ONE);

        for i in 1..Ghash128::BITS {
            let v = Ghash128::cantor_basis(i);
            assert_eq!(v.square() + v, Ghash128::cantor_basis(i - 1), "vector {i}");
        }
    }

    #[test]
    fn the_cantor_basis_is_the_image_of_the_tower_one() {
        // The two representations must span the same additive NTT domain, vector by vector.
        for i in 0..Ghash128::BITS {
            assert_eq!(
                Ghash128::cantor_basis(i),
                Ghash128::from(BinaryField128::cantor_basis(i)),
                "vector {i}"
            );
        }
    }

    #[test]
    fn the_cantor_basis_is_linearly_independent() {
        // Row-reduce the vectors over GF(2): 128 independent vectors leave 128 pivots.
        let mut rows: Vec<u128> = CANTOR_BASIS.to_vec();
        let mut pivots = 0;

        for bit in 0..128 {
            // Find a remaining row with this bit set and move it into pivot position.
            if let Some(k) = (pivots..rows.len()).find(|&k| (rows[k] >> bit) & 1 == 1) {
                rows.swap(pivots, k);

                // Clear the bit from every other row.
                for k in 0..rows.len() {
                    if k != pivots && (rows[k] >> bit) & 1 == 1 {
                        rows[k] ^= rows[pivots];
                    }
                }
                pivots += 1;
            }
        }

        assert_eq!(pivots, 128, "the Cantor basis is not a basis");
    }

    #[test]
    #[should_panic = "Cantor basis index out of range"]
    fn the_cantor_basis_rejects_an_index_beyond_the_field() {
        let _vector = Ghash128::cantor_basis(128);
    }

    #[test]
    fn scaling_by_alpha_agrees_with_the_tower() {
        // The tower scales by a basis element; here the same element is an arbitrary one.
        for bits in [0, 1, 2, 0x87, 1 << 127, u128::MAX] {
            let x = tower(bits);
            assert_eq!(
                Ghash128::from(x).mul_alpha(),
                Ghash128::from(x.mul_alpha()),
                "{bits:#x}"
            );
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(2000))]

        #[test]
        fn the_change_of_basis_round_trips(bits: u128) {
            let x = Ghash128::from_repr(bits);
            prop_assert_eq!(Ghash128::from(BinaryField128::from(x)), x);
        }

        #[test]
        fn the_change_of_basis_is_a_ring_isomorphism(a: u128, b: u128) {
            let (x, y) = (tower(a), tower(b));

            // Additive: both representations add by exclusive or, over the same coordinates.
            prop_assert_eq!(Ghash128::from(x + y), Ghash128::from(x) + Ghash128::from(y));

            // Multiplicative: this is the part a mere change of coordinates would not give.
            prop_assert_eq!(Ghash128::from(x * y), Ghash128::from(x) * Ghash128::from(y));
        }

        #[test]
        fn squaring_agrees_with_multiplying_by_self(bits: u128) {
            let x = Ghash128::from_repr(bits);
            prop_assert_eq!(x.square(), x * x);
        }

        #[test]
        fn a_nonzero_element_times_its_inverse_is_one(bits: u128) {
            let x = Ghash128::from_repr(bits);
            match x.try_inverse() {
                Some(inverse) => prop_assert_eq!(x * inverse, Ghash128::ONE),
                None => prop_assert_eq!(x, Ghash128::ZERO),
            }
        }

        #[test]
        fn the_square_root_squares_back(bits: u128) {
            let x = Ghash128::from_repr(bits);
            let root = x.try_sqrt().expect("every element of a binary field is a square");
            prop_assert_eq!(root.square(), x);
        }

        #[test]
        fn the_prime_subfield_embeds_as_zero_and_one(bit: bool) {
            let embedded = Ghash128::from(Gf2::from_bool(bit));
            prop_assert_eq!(embedded, Ghash128::from_bool(bit));
        }

        #[test]
        fn the_byte_stream_reads_back_what_the_element_wrote(bits: u128) {
            let x = Ghash128::from_repr(bits);
            prop_assert_eq!(Ghash128::from_le_byte_iter(x.into_bytes().into_iter()), x);
        }
    }

    #[test]
    #[should_panic = "byte stream ended before a whole element was read"]
    fn a_truncated_byte_stream_is_rejected() {
        // Fifteen bytes is one short of an element.
        let _element = Ghash128::from_le_byte_iter([0u8; 15].into_iter());
    }

    #[test]
    fn serde_accepts_every_bit_pattern() {
        // Unlike the narrower tower levels, no bit is masked off, so nothing is out of range.
        for bits in [0u128, 1, u128::MAX, 1 << 127] {
            let x = Ghash128::from_repr(bits);
            let encoded = serde_json::to_string(&x).unwrap();
            assert_eq!(serde_json::from_str::<Ghash128>(&encoded).unwrap(), x);
        }
    }
}
