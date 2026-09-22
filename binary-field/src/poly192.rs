//! `GF(2^192)` as the cubic extension `y^3 + y + 1` of the 64-bit polynomial-basis field.
//!
//! A cubic over a finite field is irreducible exactly when it has no root there.
//!
//! A root of this one would generate `GF(2^3)` inside `GF(2^64)`, and `3` does not divide `64`.
//!
//! The pairing matters for soundness budgeting.
//!
//! A 64-bit column field halves committed volume, and 192 bits of challenge leave headroom.

use alloc::vec::Vec;
use core::fmt::{self, Debug, Display, Formatter};
use core::iter::{Product, Sum};
use core::mem::ManuallyDrop;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use num_bigint::BigUint;
use p3_field::extension::HasFrobenius;
use p3_field::op_assign_macros::{
    impl_add_assign, impl_add_base_field, impl_div_methods, impl_mul_methods, impl_sub_assign,
    impl_sub_base_field, ring_sum,
};
use p3_field::{
    Algebra, AlgebraIdentity, BasedVectorSpace, ExtensionField, Field, Packable,
    PackedFieldExtension, Powers, PrimeCharacteristicRing, RawDataSerializable,
};
use rand::Rng;
use rand::distr::{Distribution, StandardUniform};
use serde::{Deserialize, Serialize};

use crate::gf2::characteristic_two_methods;
use crate::{Gf2, Poly64};

/// The number of coordinates over the coefficient field.
const DEGREE: usize = 3;

/// `GF(2^192)`, as three coordinates over `GF(2^64)` in the basis `1, y, y^2`.
///
/// The defining relation is `y^3 = y + 1`, which also gives `y^4 = y^2 + y`.
///
/// Everything below reduces with those two rewrites and nothing else.
#[derive(Copy, Clone, Default, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
#[repr(transparent)]
#[must_use]
pub struct Poly192([Poly64; DEGREE]);

impl Poly192 {
    /// The number of bits of an element.
    pub(crate) const BITS: usize = DEGREE * 64;

    /// The element with the given coordinates in the basis `1, y, y^2`.
    #[inline]
    pub const fn new(coefficients: [Poly64; DEGREE]) -> Self {
        Self(coefficients)
    }

    /// The coordinates of this element in the basis `1, y, y^2`.
    #[inline]
    pub const fn coefficients(self) -> [Poly64; DEGREE] {
        self.0
    }

    /// The element with all its weight on one coordinate.
    #[inline]
    const fn embed(value: Poly64) -> Self {
        Self([value, Poly64::ZERO, Poly64::ZERO])
    }

    /// The product assembled from the coefficient field's own reduced products.
    ///
    /// Compiled everywhere, so its tests and its benchmark run even where a backend wins.
    #[doc(hidden)]
    #[inline]
    pub fn composed_mul(self, rhs: Self) -> Self {
        let ([a0, a1, a2], [b0, b1, b2]) = (self.0, rhs.0);

        // The three diagonal products.
        let c0 = a0 * b0;
        let c1 = a1 * b1;
        let c2 = a2 * b2;

        // One product per off-diagonal pair, each carrying both of that pair's cross terms.
        let d01 = (a0 + a1) * (b0 + b1);
        let d02 = (a0 + a2) * (b0 + b2);
        let d12 = (a1 + a2) * (b1 + b2);

        // The three middle coefficients of the unreduced product.
        let p1 = d01 + c0 + c1;
        let p2 = d02 + c0 + c1 + c2;
        let p3 = d12 + c1 + c2;

        Self([c0 + p3, p1 + p3 + c2, p2 + c2])
    }

    /// The inverse of this element, with zero sent to zero.
    ///
    /// Every step runs whatever the operand is, so the cost says nothing about the value.
    #[inline]
    pub fn invert_or_zero(self) -> Self {
        self.try_inverse().unwrap_or(Self::ZERO)
    }
}

impl Packable for Poly192 {}

impl Display for Poly192 {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{} + {}*y + {}*y^2", self.0[0], self.0[1], self.0[2])
    }
}

impl Debug for Poly192 {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Debug::fmt(&self.0, f)
    }
}

impl Distribution<Poly192> for StandardUniform {
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Poly192 {
        Poly192(core::array::from_fn(|_| self.sample(rng)))
    }
}

impl Add for Poly192 {
    type Output = Self;

    #[inline]
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn add(self, rhs: Self) -> Self {
        // Addition in characteristic 2 is `XOR`, coordinate by coordinate.
        Self(core::array::from_fn(|i| self.0[i] + rhs.0[i]))
    }
}

impl Sub for Poly192 {
    type Output = Self;

    #[inline]
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn sub(self, rhs: Self) -> Self {
        // Subtraction coincides with addition in characteristic 2.
        self + rhs
    }
}

impl Neg for Poly192 {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        // `-x = x` in characteristic 2.
        self
    }
}

impl Mul for Poly192 {
    type Output = Self;

    /// # Algorithm
    ///
    /// Karatsuba over three limbs: six products instead of the schoolbook nine.
    ///
    /// ```text
    ///     c_i    = a_i b_i
    ///     d_ij   = (a_i + a_j) (b_i + b_j)
    ///     p_1    = d_01 + c_0 + c_1
    ///     p_2    = d_02 + c_0 + c_1 + c_2
    ///     p_3    = d_12 + c_1 + c_2
    /// ```
    ///
    /// The unreduced product spans degrees 0 to 4, and the modulus folds the top two down:
    ///
    /// ```text
    ///     y^3    = y + 1
    ///     y^4    = y^2 + y
    /// ```
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        #[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
        {
            let limbs = |x: Self| x.0.map(Poly64::to_bits);
            Self(crate::clmul::poly_mul_192(limbs(self), limbs(rhs)).map(Poly64::new))
        }
        #[cfg(not(all(target_arch = "x86_64", target_feature = "pclmulqdq")))]
        {
            self.composed_mul(rhs)
        }
    }
}

impl Mul<Poly64> for Poly192 {
    type Output = Self;

    /// Scaling stays inside each coordinate, so it costs three products rather than six.
    #[inline]
    fn mul(self, rhs: Poly64) -> Self {
        Self(self.0.map(|c| c * rhs))
    }
}

impl Mul<Poly192> for Poly64 {
    type Output = Poly192;

    #[inline]
    fn mul(self, rhs: Poly192) -> Poly192 {
        rhs * self
    }
}

impl PrimeCharacteristicRing for Poly192 {
    #[inline]
    fn zero_vec(len: usize) -> Vec<Self> {
        let mut values = ManuallyDrop::new(alloc::vec![[0u64; DEGREE]; len]);
        // SAFETY: the transparent wrapper has exactly the integer's layout, and zero is
        // canonical. The allocation retains its original size and alignment.
        unsafe { Vec::from_raw_parts(values.as_mut_ptr().cast(), values.len(), values.capacity()) }
    }

    type PrimeSubfield = Gf2;

    const ZERO: Self = Self::embed(Poly64::ZERO);
    const ONE: Self = Self::embed(Poly64::ONE);
    // The characteristic is 2, so `TWO = ONE + ONE = ZERO`.
    const TWO: Self = Self::embed(Poly64::ZERO);
    // The characteristic is 2, so `NEG_ONE = ONE`.
    const NEG_ONE: Self = Self::embed(Poly64::ONE);

    #[inline]
    fn from_prime_subfield(f: Self::PrimeSubfield) -> Self {
        Self::embed(Poly64::from_prime_subfield(f))
    }

    /// Squaring is `F_2`-linear, so the cross terms vanish and only the fold survives.
    ///
    /// ```text
    ///     (a_0 + a_1 y + a_2 y^2)^2  = a_0^2 + a_1^2 y^2 + a_2^2 y^4
    ///                                = a_0^2 + a_2^2 y + (a_1^2 + a_2^2) y^2
    /// ```
    #[inline]
    fn square(&self) -> Self {
        let [a0, a1, a2] = self.0;
        let (s0, s1, s2) = (a0.square(), a1.square(), a2.square());
        Self([s0, s2, s1 + s2])
    }

    characteristic_two_methods!();
}

impl Field for Poly192 {
    type Packing = Self;

    // An element of the multiplicative group whose order is the whole group.
    const GENERATOR: Self = Self([Poly64::new(5), Poly64::new(3), Poly64::new(1)]);

    /// # Algorithm
    ///
    /// Multiplication by a fixed element is a `3 x 3` matrix over the coefficient field.
    ///
    /// Its columns are where that multiplication sends the three basis elements:
    ///
    /// ```text
    ///     a * 1   = ( a_0,        a_1,        a_2       )
    ///     a * y   = ( a_2,        a_0 + a_2,  a_1       )
    ///     a * y^2 = ( a_1,        a_1 + a_2,  a_0 + a_2 )
    /// ```
    ///
    /// The inverse is the first column of that matrix inverted.
    ///
    /// The adjugate gives it as the top row of cofactors over the determinant.
    ///
    /// The determinant is the norm down to the coefficient field, and it vanishes only at zero.
    #[inline]
    fn try_inverse(&self) -> Option<Self> {
        let [a0, a1, a2] = self.0;

        // The entry the modulus makes appear three times in the matrix.
        let diagonal = a0 + a2;

        // The cofactors along the top row.
        let c0 = diagonal.square() + a1.square() + a1 * a2;
        let c1 = a1 * diagonal + (a1 + a2) * a2;
        let c2 = a1.square() + diagonal * a2;

        // Expanding along that row gives the determinant.
        let norm = a0 * c0 + a2 * c1 + a1 * c2;

        // Every step runs whatever the operand is, so the cost says nothing about the value.
        let candidate = Self([c0, c1, c2]) * norm.invert_or_zero();
        (!norm.is_zero()).then_some(candidate)
    }

    /// Squaring is a triangular map on the coordinates, so its inverse is one too.
    ///
    /// ```text
    ///     b_0 = a_0^2,  b_1 = a_2^2,  b_2 = a_1^2 + a_2^2
    /// ```
    ///
    /// Reading that backwards costs three coefficient-field square roots and one addition.
    #[inline]
    fn try_sqrt(&self) -> Option<Self> {
        let [b0, b1, b2] = self.0;
        let root = |x: Poly64| {
            x.try_sqrt()
                .expect("every element of a binary field is a square")
        };
        Some(Self([root(b0), root(b1 + b2), root(b1)]))
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
    /// The coordinates over `GF(2)` run through the coefficient fields in order.
    ///
    /// An index narrower than 64 bits therefore lands entirely in the first of them.
    #[inline]
    fn interpolation_node(i: usize) -> Self {
        Self::embed(Poly64::interpolation_node(i))
    }
}

impl RawDataSerializable for Poly192 {
    const NUM_BYTES: usize = DEGREE * Poly64::NUM_BYTES;

    #[inline]
    fn into_bytes(self) -> impl IntoIterator<Item = u8> {
        self.0.into_iter().flat_map(RawDataSerializable::into_bytes)
    }
}

impl From<Poly64> for Poly192 {
    /// The coefficient field sits at the constant coordinate.
    #[inline]
    fn from(x: Poly64) -> Self {
        Self::embed(x)
    }
}

impl From<Gf2> for Poly192 {
    /// `GF(2)` is the prime subfield, embedded as `{ZERO, ONE}`.
    #[inline]
    fn from(x: Gf2) -> Self {
        Self::from_prime_subfield(x)
    }
}

impl Mul<Gf2> for Poly192 {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Gf2) -> Self {
        // The prime subfield has two elements, so this keeps the element or clears it.
        if rhs.is_one() { self } else { Self::ZERO }
    }
}

impl Mul<Poly192> for Gf2 {
    type Output = Poly192;

    #[inline]
    fn mul(self, rhs: Poly192) -> Poly192 {
        rhs * self
    }
}

impl_add_assign!(Poly192);
impl_sub_assign!(Poly192);
impl_mul_methods!(Poly192);
impl_div_methods!(Poly192, Poly192);
impl_div_methods!(Poly192, Poly64);
ring_sum!(Poly192);

impl_add_base_field!(Poly192, Poly64);
impl_sub_base_field!(Poly192, Poly64);
impl_add_base_field!(Poly192, Gf2);
impl_sub_base_field!(Poly192, Gf2);

impl Algebra<Poly64> for Poly192 {}

impl Algebra<Gf2> for Poly192 {}

impl AlgebraIdentity<Poly64> for Poly192 {
    fn algebra_id() -> Vec<u8> {
        b"p3-binary-cubic-v1:y^3+y+1:64:192".to_vec()
    }
}

impl BasedVectorSpace<Poly64> for Poly192 {
    const DIMENSION: usize = DEGREE;

    #[inline]
    fn as_basis_coefficients_slice(&self) -> &[Poly64] {
        &self.0
    }

    #[inline]
    fn from_basis_coefficients_fn<F: FnMut(usize) -> Poly64>(f: F) -> Self {
        Self(core::array::from_fn(f))
    }

    #[inline]
    fn from_basis_coefficients_iter<I: ExactSizeIterator<Item = Poly64>>(iter: I) -> Option<Self> {
        (iter.len() == DEGREE).then(|| {
            // The reported length is not a guarantee, so zipping bounds the fill either way.
            let mut coefficients = [Poly64::ZERO; DEGREE];
            for (slot, value) in coefficients.iter_mut().zip(iter) {
                *slot = value;
            }
            Self(coefficients)
        })
    }

    /// A whole vector of elements is one contiguous run of coordinates, so this is a copy.
    #[inline]
    fn flatten_to_base(vec: Vec<Self>) -> Vec<Poly64> {
        let mut out = Vec::with_capacity(vec.len() * DEGREE);
        for element in vec {
            out.extend_from_slice(&element.0);
        }
        out
    }
}

impl ExtensionField<Poly64> for Poly192 {
    type ExtensionPacking = Self;

    #[inline]
    fn is_in_basefield(&self) -> bool {
        // The coefficient field is exactly the elements with no weight above the constant.
        self.0[1].is_zero() && self.0[2].is_zero()
    }

    #[inline]
    fn as_base(&self) -> Option<Poly64> {
        <Self as ExtensionField<Poly64>>::is_in_basefield(self).then_some(self.0[0])
    }
}

/// One element per vector, so every lane index is zero.
impl PackedFieldExtension<Poly64, Self> for Poly192 {
    #[inline]
    fn from_ext_fn(f: impl Fn(usize) -> Self) -> Self {
        f(0)
    }

    #[inline]
    fn from_ext_slice(slice: &[Self]) -> Self {
        assert_eq!(slice.len(), 1);
        slice[0]
    }

    #[inline]
    fn extract(&self, lane: usize) -> Self {
        assert_eq!(lane, 0, "lane index out of range");
        *self
    }

    #[inline]
    fn add_assign_lane(&mut self, lane: usize, value: Self) {
        assert_eq!(lane, 0, "lane index out of range");
        *self += value;
    }

    #[inline]
    fn packed_ext_powers(base: Self) -> Powers<Self> {
        base.powers()
    }
}

impl HasFrobenius<Poly64> for Poly192 {
    /// The map raising to the size of the coefficient field.
    ///
    /// It fixes every coefficient and permutes the basis.
    ///
    /// The three roots of the modulus are `y`, `y^2` and `y^2 + y`, so:
    ///
    /// ```text
    ///     y^(2^64)                 =  y^2
    ///     a_0 + a_1 y + a_2 y^2    ->  a_0 + a_2 y + (a_1 + a_2) y^2
    /// ```
    ///
    /// So one application is one addition and a permutation, with no products at all.
    #[inline]
    fn frobenius(&self) -> Self {
        let [a0, a1, a2] = self.0;
        Self([a0, a2, a1 + a2])
    }

    #[inline]
    fn repeated_frobenius(&self, count: usize) -> Self {
        // The Galois group is cyclic of order three, so the count reduces modulo it.
        match count % DEGREE {
            0 => *self,
            1 => self.frobenius(),
            _ => self.frobenius().frobenius(),
        }
    }

    #[inline]
    fn pseudo_inv(&self) -> Self {
        // Inversion through the adjugate beats the exponentiation the contract is phrased in.
        self.invert_or_zero()
    }
}

#[cfg(test)]
mod tests {
    use p3_field::extension::HasFrobenius;
    use p3_field::{BasedVectorSpace, ExtensionField, Field, PrimeCharacteristicRing};
    use proptest::prelude::*;

    use super::{DEGREE, Poly192};
    use crate::Poly64;

    /// An element from three raw bit patterns.
    fn element(a: [u64; 3]) -> Poly192 {
        Poly192::new(a.map(Poly64::new))
    }

    /// The `j`-th vector of the basis over the coefficient field.
    fn basis_element(j: usize) -> Poly192 {
        <Poly192 as BasedVectorSpace<Poly64>>::ith_basis_element(j).unwrap()
    }

    /// Schoolbook multiplication, nine products and the modulus applied term by term.
    ///
    /// The routine under test recombines a Karatsuba schedule and folds the top two.
    ///
    /// This shares neither of those steps.
    fn schoolbook(a: Poly192, b: Poly192) -> Poly192 {
        let (x, y) = (a.coefficients(), b.coefficients());

        // The nine term products, placed by total degree.
        let mut raw = [Poly64::ZERO; 5];
        for (i, &xi) in x.iter().enumerate() {
            for (j, &yj) in y.iter().enumerate() {
                raw[i + j] += xi * yj;
            }
        }

        // Reduce from the top down, rewriting each overflowing power once.
        //
        //     y^4  =  y^3 * y  =  (y + 1) * y  =  y^2 + y
        //     y^3  =  y + 1
        raw[2] += raw[4];
        raw[1] += raw[4];
        raw[1] += raw[3];
        raw[0] += raw[3];

        Poly192::new([raw[0], raw[1], raw[2]])
    }

    /// The inverse by Gaussian elimination on the multiplication matrix.
    ///
    /// The routine under test uses closed-form cofactors instead, so the two share no algebra.
    fn inverse_by_elimination(a: Poly192) -> Option<Poly192> {
        // Column `j` is where multiplication by the element sends basis vector `j`.
        let columns: [Poly192; DEGREE] = core::array::from_fn(|j| a * basis_element(j));

        // The augmented system, solving for the coordinates that multiply back to one.
        let mut rows: [[Poly64; DEGREE + 1]; DEGREE] = core::array::from_fn(|i| {
            core::array::from_fn(|j| {
                if j < DEGREE {
                    columns[j].coefficients()[i]
                } else {
                    Poly64::from_bool(i == 0)
                }
            })
        });

        for pivot in 0..DEGREE {
            // Bring a row with a nonzero pivot entry into place, or report singularity.
            let found = (pivot..DEGREE).find(|&r| !rows[r][pivot].is_zero())?;
            rows.swap(pivot, found);

            // Normalize the pivot row, then clear the column everywhere else.
            let scale = rows[pivot][pivot].inverse();
            for entry in &mut rows[pivot][pivot..] {
                *entry *= scale;
            }
            let pivot_row = rows[pivot];
            for (r, row) in rows.iter_mut().enumerate() {
                if r == pivot || row[pivot].is_zero() {
                    continue;
                }
                let factor = row[pivot];
                for (entry, &above) in row[pivot..].iter_mut().zip(&pivot_row[pivot..]) {
                    *entry += above * factor;
                }
            }
        }

        Some(Poly192::new(core::array::from_fn(|i| rows[i][DEGREE])))
    }

    #[test]
    fn the_modulus_reduces_the_way_the_relation_says() {
        let y = element([0, 1, 0]);

        // y^2 is still a basis element.
        assert_eq!(y * y, element([0, 0, 1]));

        // y^3 = y + 1, which is the defining relation.
        assert_eq!(y * y * y, element([1, 1, 0]));

        // y^4 = y^2 + y, one more step of the same rewrite.
        assert_eq!(y * y * y * y, element([0, 1, 1]));
    }

    #[test]
    fn the_modulus_has_no_root_in_the_coefficient_field() {
        // Invariant: a cubic over a finite field factors only if it has a root there.
        //
        // A root would generate GF(2^3), and 3 does not divide 64, so none exists.
        //
        // The argument is checked where a search can reach it.
        //
        // That is the 256 elements whose coordinates fit in a byte.
        for c in 0..=u8::MAX {
            let x = Poly64::new(u64::from(c));
            assert_ne!(x * x * x + x + Poly64::ONE, Poly64::ZERO, "{c}");
        }
    }

    #[test]
    fn the_frobenius_fixes_the_coefficient_field_and_has_order_three() {
        // Fixture state: the coefficient field embedded at the constant coordinate.
        for bits in [0u64, 1, 0x1b, u64::MAX] {
            let embedded = Poly192::from(Poly64::new(bits));
            assert_eq!(embedded.frobenius(), embedded, "{bits:#x}");
        }

        // It moves everything outside that field, and cycles after three steps.
        let y = element([0, 1, 0]);
        assert_ne!(y.frobenius(), y);
        assert_eq!(y.frobenius(), element([0, 0, 1]));
        assert_eq!(y.frobenius().frobenius().frobenius(), y);
    }

    #[test]
    fn zero_vectors_preserve_layout_and_support_growth() {
        for len in [0, 1, 33, 1024] {
            let mut values = Poly192::zero_vec(len);
            assert_eq!(values.len(), len);
            assert!(values.iter().all(|x| *x == Poly192::ZERO));
            values.push(Poly192::ONE);
            values.reserve(100);
            assert_eq!(values.pop(), Some(Poly192::ONE));
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(1000))]

        #[test]
        fn the_karatsuba_product_matches_the_schoolbook_one(a: [u64; 3], b: [u64; 3]) {
            let (x, y) = (element(a), element(b));
            prop_assert_eq!(x * y, schoolbook(x, y));
        }

        #[test]
        fn the_square_matches_the_product_with_itself(a: [u64; 3]) {
            let x = element(a);
            prop_assert_eq!(x.square(), schoolbook(x, x));
        }

        #[test]
        fn the_cofactor_inverse_matches_the_eliminated_one(a: [u64; 3]) {
            let x = element(a);
            prop_assert_eq!(x.try_inverse(), inverse_by_elimination(x));
        }

        #[test]
        fn a_nonzero_element_times_its_inverse_is_one(a: [u64; 3]) {
            let x = element(a);
            match x.try_inverse() {
                Some(inverse) => prop_assert_eq!(x * inverse, Poly192::ONE),
                None => prop_assert_eq!(x, Poly192::ZERO),
            }
        }

        #[test]
        fn the_square_root_squares_back(a: [u64; 3]) {
            let x = element(a);
            let root = x.try_sqrt().expect("every element of a binary field is a square");
            prop_assert_eq!(root.square(), x);
        }

        /// The Frobenius must be a ring homomorphism, not merely an additive one.
        #[test]
        fn the_frobenius_is_a_ring_homomorphism(a: [u64; 3], b: [u64; 3]) {
            let (x, y) = (element(a), element(b));
            prop_assert_eq!((x + y).frobenius(), x.frobenius() + y.frobenius());
            prop_assert_eq!((x * y).frobenius(), x.frobenius() * y.frobenius());
        }

        /// Raising to the coefficient field's size must be what the coordinate shuffle does.
        #[test]
        fn the_frobenius_is_the_power_map_it_claims_to_be(a: [u64; 3]) {
            let x = element(a);
            prop_assert_eq!(x.frobenius(), x.exp_power_of_2(64));
        }

        /// Scaling by a coefficient must agree with embedding the scalar first.
        #[test]
        fn scaling_by_a_coefficient_agrees_with_the_full_product(a: [u64; 3], s: u64) {
            let (x, scalar) = (element(a), Poly64::new(s));
            prop_assert_eq!(x * scalar, x * Poly192::from(scalar));
        }

        #[test]
        fn the_coefficients_round_trip(a: [u64; 3]) {
            let x = element(a);
            let coefficients = BasedVectorSpace::<Poly64>::as_basis_coefficients_slice(&x);
            prop_assert_eq!(
                <Poly192 as BasedVectorSpace<Poly64>>::from_basis_coefficients_slice(coefficients),
                Some(x)
            );
            prop_assert_eq!(
                ExtensionField::<Poly64>::is_in_basefield(&x),
                a[1] == 0 && a[2] == 0
            );
        }
    }
}
