//! `GF(2^128)` in the polynomial basis of `x^128 + x^7 + x^2 + x + 1`.

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
use crate::tower::TowerLevel;
use crate::{
    BinaryField8, BinaryField16, BinaryField32, BinaryField64, BinaryField128, Gf2, clmul,
};

/// The bit pattern of the multiplicative generator of the tower representation.
///
/// Its image is the generator here, so the conversion carries one onto the other.
const TOWER_GENERATOR: u128 = 0x1_0000_0000_0000_0005;

/// The tower's generator of `GF(2^128)` over `GF(2^64)`, in this representation.
///
/// The tower carries that element as a single basis vector, at bit 64.
const ALPHA: u128 = clmul::tower_image_128(1 << 64);

/// The inverse of an element, with zero sent to zero, by the addition chain over Frobenius maps.
///
/// The chain runs whatever the operand is, so its cost says nothing about the value.
#[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
#[inline]
fn invert_or_zero(x: Ghash128) -> Ghash128 {
    Ghash128(clmul::poly_inverse_128(x.0))
}

/// The inverse of an element, with zero sent to zero, through the tower norm.
///
/// The tower recurses through the norm down to a `GF(2^8)` lookup table.
/// Everywhere the addition chain is not faster, that beats it for no table at all.
/// That recursion returns early on a zero operand at every level, so this route is not
/// branch-free.
#[cfg(not(all(target_arch = "x86_64", target_feature = "pclmulqdq")))]
#[inline]
fn invert_or_zero(x: Ghash128) -> Ghash128 {
    BinaryField128::from(x)
        .try_inverse()
        .map_or(Ghash128::ZERO, Ghash128::from)
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

/// Elements one parallel task converts in [`Ghash128::from_tower_vec`].
///
/// A power of two, so a whole number of blocks of the blocked basis-change kernel.
const TABLE_CHUNK: usize = 1 << 16;

impl Ghash128 {
    /// The number of bits of an element.
    pub(crate) const BITS: usize = 128;

    /// A whole table of tower elements, seen in the polynomial basis, in the table's own buffer.
    ///
    /// Each entry becomes what [`From`] makes of it.
    ///
    /// Chunks of the table convert in parallel, a block at a time where the build has the
    /// blocked kernel [`crate::poly_basis::from_tower_slice`] describes.
    pub fn from_tower_vec(values: Vec<BinaryField128>) -> Vec<Self> {
        let mut values = ManuallyDrop::new(values);
        let (ptr, len, capacity) = (values.as_mut_ptr(), values.len(), values.capacity());

        // SAFETY: `BinaryField128` is transparent over `u128`, so the initialized entries are
        // `len` valid `u128`s. The vector is never used again, so this is the only reference.
        let words = unsafe { slice::from_raw_parts_mut(ptr.cast::<u128>(), len) };
        words.par_chunks_mut(TABLE_CHUNK).for_each(|chunk| {
            clmul::tower_to_poly_128_slice(chunk);
        });

        // SAFETY: `Ghash128` is transparent over `u128` as well, so the allocation has its size
        // and alignment, and every `u128` is a valid element. This also relies on taking
        // `BinaryField128` specifically: its `MASK` is `u128::MAX`, so every bit pattern left in
        // `words` is already canonical. A smaller tower field's mask clears high bits, so its
        // buffer could hold non-canonical entries and this reinterpretation would not be sound.
        unsafe { Vec::from_raw_parts(ptr.cast::<Self>(), len, capacity) }
    }

    /// Construct a field element from its little-endian byte representation.
    ///
    /// Every byte string of this length is a valid element.
    #[inline]
    pub const fn from_le_bytes(bytes: [u8; 16]) -> Self {
        Self(u128::from_le_bytes(bytes))
    }

    /// Combine values against the successive powers of the indeterminate.
    ///
    /// ```text
    ///     sum_k values_k * x^k
    /// ```
    ///
    /// In this representation a power of the indeterminate shifts the coefficients up.
    ///
    /// So the combination can be shifts and exclusive ors, with no multiplication at all.
    ///
    /// Either way the modulus is folded in once at the end, not once per term.
    ///
    /// Which route is faster depends on the target.
    ///
    /// A hardware carryless multiply turns each term into one cheap product, and beats the
    /// shifts.
    ///
    /// Without one a product costs sixteen integer multiplies, and the shifts win by an
    /// order of magnitude.
    ///
    /// The choice is made at compile time from the same flag the rest of the crate reads.
    ///
    /// # Panics
    ///
    /// Panics on more values than the field has bits.
    ///
    /// That ceiling is where the last shift would leave the word.
    #[inline]
    pub fn dot_powers_of_x(values: &[Self]) -> Self {
        if clmul::HAS_HARDWARE_CLMUL {
            // The powers are bare bit patterns, so the deferred dot product needs no table.
            //
            // Each one is the previous shifted up, which is cheaper than shifting by the index.
            let mut power = 1u128;
            let terms = values.iter().map(|v| {
                let term = (v.0, power);
                power <<= 1;
                term
            });
            Self(clmul::poly_dot_128(terms))
        } else {
            Self(clmul::poly_dot_powers_128(values.iter().map(|v| v.0)))
        }
    }

    /// The inverse of this element, with zero sent to zero.
    ///
    /// On a carryless-multiply target the addition chain runs whatever the operand is, so its
    /// cost says nothing about the value. The operand-indexed tables it uses still make it
    /// variable-time.
    #[inline]
    pub fn invert_or_zero(self) -> Self {
        invert_or_zero(self)
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
    #[inline]
    fn zero_vec(len: usize) -> Vec<Self> {
        let mut values = ManuallyDrop::new(alloc::vec![0u128; len]);
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

    /// `x·(x - 1) = x² - x = x² + x` in characteristic 2, and `poly_square_128` skips the
    /// cross-term carryless multiplies a general product pays for.
    #[inline]
    fn bool_check(&self) -> Self {
        self.square() + *self
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
        // The chain runs first; only the answer depends on whether the operand was zero.
        let inverse = self.invert_or_zero();
        (self.0 != 0).then_some(inverse)
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

// The change of basis is a field isomorphism, so it makes this field an algebra over the tower.
// A mixed operation takes the polynomial-basis operand on the left and returns this basis.

impl Add<BinaryField128> for Ghash128 {
    type Output = Self;

    /// The sum, in the polynomial basis.
    ///
    /// The tower operand is converted first, which costs sixteen table lookups.
    #[inline]
    fn add(self, rhs: BinaryField128) -> Self {
        self + Self::from(rhs)
    }
}

impl Sub<BinaryField128> for Ghash128 {
    type Output = Self;

    /// The difference, in the polynomial basis.
    ///
    /// The tower operand is converted first, which costs sixteen table lookups.
    #[inline]
    fn sub(self, rhs: BinaryField128) -> Self {
        self - Self::from(rhs)
    }
}

impl Mul<BinaryField128> for Ghash128 {
    type Output = Self;

    /// The product, in the polynomial basis.
    ///
    /// The tower operand is converted first, which costs sixteen table lookups.
    #[inline]
    fn mul(self, rhs: BinaryField128) -> Self {
        self * Self::from(rhs)
    }
}

impl Algebra<BinaryField128> for Ghash128 {}

macro_rules! impl_narrow_algebra {
    ($($field:ty),* $(,)?) => {$(
        impl From<$field> for Ghash128 {
            #[inline]
            fn from(x: $field) -> Self {
                Self::from(BinaryField128::from(x))
            }
        }

        impl_add_base_field!(Ghash128, $field);
        impl_sub_base_field!(Ghash128, $field);
        impl_mul_base_field!(Ghash128, $field);

        impl Algebra<$field> for Ghash128 {}
    )*};
}

impl_narrow_algebra!(BinaryField8, BinaryField16, BinaryField32, BinaryField64);

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

    use p3_field::{Algebra, Field, PrimeCharacteristicRing, RawDataSerializable};
    use proptest::prelude::*;

    use super::{CANTOR_BASIS, Ghash128};
    use crate::tower::TowerLevel;
    use crate::{BinaryField8, BinaryField16, BinaryField32, BinaryField64, BinaryField128, Gf2};

    #[test]
    fn narrow_algebras_match_multiplication_in_the_tower() {
        // The static checks cover every committed alphabet admitted under 128-bit challenges.
        const fn assert_algebra<F: PrimeCharacteristicRing, A: Algebra<F>>() {}
        assert_algebra::<BinaryField8, Ghash128>();
        assert_algebra::<BinaryField16, Ghash128>();
        assert_algebra::<BinaryField32, Ghash128>();
        assert_algebra::<BinaryField64, Ghash128>();
        assert_algebra::<BinaryField128, Ghash128>();

        // A nontrivial 32-bit scalar pins the embedding against the tower-field reference.
        let x = BinaryField128::from_repr(0x3141_5926_5358_9793_2384_6264_3383_2795);
        let x_poly = Ghash128::from(x);
        let scalar = BinaryField32::from_repr(0xa5c3_19e7);
        assert_eq!(
            BinaryField128::from(x_poly * scalar),
            x * BinaryField128::from(scalar)
        );
    }

    /// The tower element with the given bit pattern.
    fn tower(bits: u128) -> BinaryField128 {
        BinaryField128::from_repr(bits)
    }

    #[test]
    fn a_tower_table_converts_entry_by_entry_in_its_own_buffer() {
        // Invariant: the table map is the element map, and the buffer is reused.
        //
        // Fixture state: lengths around one block of the blocked kernel and across several
        // parallel chunks, with a partial last chunk and a partial last block.
        for len in [0, 1, 63, 64, 65, 1000, 2 * super::TABLE_CHUNK + 67] {
            let table: Vec<BinaryField128> = (0..len as u128)
                .map(|i| tower(i.wrapping_mul(0x9e37_79b9_7f4a_7c15_f39c_c060_5ced_c835) ^ i))
                .collect();
            let want: Vec<Ghash128> = table.iter().map(|&x| Ghash128::from(x)).collect();

            let at = table.as_ptr() as usize;
            let got = Ghash128::from_tower_vec(table);

            assert_eq!(got, want, "length {len}");
            if len > 0 {
                assert_eq!(got.as_ptr() as usize, at, "length {len}");
            }
        }
    }

    #[test]
    fn the_change_of_basis_fixes_the_constants() {
        // Any field isomorphism fixes zero and one.
        assert_eq!(Ghash128::from(BinaryField128::ZERO), Ghash128::ZERO);
        assert_eq!(Ghash128::from(BinaryField128::ONE), Ghash128::ONE);
    }

    #[test]
    fn the_algebra_over_the_tower_fixes_the_prime_subfield() {
        /// Statically require the full `Algebra<BinaryField128>` bound, not merely the operators.
        const fn assert_algebra_over_the_tower<T: Algebra<BinaryField128>>() {}
        assert_algebra_over_the_tower::<Ghash128>();

        // Both representations embed `GF(2)` as zero and one, and the isomorphism agrees.
        for bit in [Gf2::ZERO, Gf2::ONE] {
            assert_eq!(
                Ghash128::from(BinaryField128::from(bit)),
                Ghash128::from(bit)
            );
        }
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
    fn bool_check_matches_the_vanishing_polynomial_at_zero_and_one() {
        for x in [Ghash128::ZERO, Ghash128::ONE] {
            assert_eq!(x.bool_check(), x * (x - Ghash128::ONE));
        }
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

    #[test]
    fn zero_vectors_preserve_layout_and_support_growth() {
        for len in [0, 1, 33, 1024] {
            let mut values = Ghash128::zero_vec(len);
            assert_eq!(values.len(), len);
            assert!(values.iter().all(|x| *x == Ghash128::ZERO));
            values.push(Ghash128::from_repr(7));
            values.reserve(100);
            assert_eq!(values.pop(), Some(Ghash128::from_repr(7)));
        }
    }

    #[test]
    fn zero_inverts_to_zero() {
        assert_eq!(Ghash128::ZERO.invert_or_zero(), Ghash128::ZERO);
        assert_eq!(Ghash128::ZERO.try_inverse(), None);
        assert_eq!(Ghash128::ONE.invert_or_zero(), Ghash128::ONE);
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
        fn the_tower_acts_through_the_change_of_basis(a: u128, b: u128) {
            let (g, t) = (Ghash128::from_repr(a), tower(b));
            let converted = Ghash128::from(t);

            // Each mixed operation agrees with converting first.
            prop_assert_eq!(g + t, g + converted);
            prop_assert_eq!(g - t, g - converted);
            prop_assert_eq!(g * t, g * converted);

            let mut acc = g;
            acc += t;
            prop_assert_eq!(acc, g + converted);
            let mut acc = g;
            acc -= t;
            prop_assert_eq!(acc, g - converted);
            let mut acc = g;
            acc *= t;
            prop_assert_eq!(acc, g * converted);

            // Carried back, each result is the tower's own arithmetic on the same elements.
            let tower_g = BinaryField128::from(g);
            prop_assert_eq!(BinaryField128::from(g + t), tower_g + t);
            prop_assert_eq!(BinaryField128::from(g - t), tower_g - t);
            prop_assert_eq!(BinaryField128::from(g * t), tower_g * t);
        }

        #[test]
        fn squaring_agrees_with_multiplying_by_self(bits: u128) {
            let x = Ghash128::from_repr(bits);
            prop_assert_eq!(x.square(), x * x);
        }

        #[test]
        fn bool_check_agrees_with_the_vanishing_polynomial(bits: u128) {
            let x = Ghash128::from_repr(bits);
            prop_assert_eq!(x.bool_check(), x * (x - Ghash128::ONE));
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
        fn invert_or_zero_agrees_with_try_inverse(bits: u128) {
            let x = Ghash128::from_repr(bits);
            prop_assert_eq!(
                x.invert_or_zero(),
                x.try_inverse().unwrap_or(Ghash128::ZERO)
            );
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
