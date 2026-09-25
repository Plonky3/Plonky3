//! The packing of `GF(2^192)` over the packing of `GF(2^64)`, one register per coordinate.
//!
//! Lane `k` of coordinate register `i` holds coordinate `i` of element `k`:
//!
//! ```text
//!     coordinates[0]  =  [ e_0.0  e_1.0  e_2.0  e_3.0 ]
//!     coordinates[1]  =  [ e_0.1  e_1.1  e_2.1  e_3.1 ]
//!     coordinates[2]  =  [ e_0.2  e_1.2  e_2.2  e_3.2 ]
//! ```
//!
//! Every register is then a packing of the coefficient field.
//!
//! So the cubic algebra runs on whole registers, and every carryless multiply fills every lane.
//!
//! A product costs twelve 256-bit carryless multiplies for four elements.
//!
//! The scalar route pays six 128-bit ones per element, twenty-four for the same four.

use alloc::vec::Vec;
use core::array;
use core::iter::{Product, Sum};
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use p3_field::{
    Algebra, BasedVectorSpace, Dup, Field, PackedFieldExtension, PackedValue, Powers,
    PrimeCharacteristicRing,
};
use rand::Rng;
use rand::distr::{Distribution, StandardUniform};

use super::lanes::gf64::{self as lanes, Reg, WIDTH_64};
use super::poly64::PackedPoly64;
use crate::clmul::wide::{Wide, cubic_mul, cubic_mul_base, cubic_square};
use crate::{Gf2, Poly64, Poly192};

/// The number of coordinates over the coefficient field.
const DEGREE: usize = 3;

/// Several elements of `GF(2^192)`, as one packing of `GF(2^64)` per coordinate.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(transparent)]
#[must_use]
pub struct PackedPoly192([PackedPoly64; DEGREE]);

impl PackedPoly192 {
    /// The three coordinate registers.
    #[inline(always)]
    fn to_vectors(self) -> [Reg; DEGREE] {
        // Each coordinate is already one register of lanes.
        self.0.map(PackedPoly64::to_vector)
    }

    /// The elements held in three coordinate registers.
    #[inline(always)]
    fn from_vectors(vectors: [Reg; DEGREE]) -> Self {
        Self(vectors.map(PackedPoly64::from_vector))
    }

    /// The reduced elements of an unreduced product.
    #[inline(always)]
    fn reduce(wide: [Wide<Reg>; DEGREE]) -> Self {
        // One base-field reduction per coordinate register.
        Self::from_vectors(wide.map(Wide::reduce))
    }

    /// The element with all its weight on the constant coordinate.
    #[inline]
    const fn embed(value: PackedPoly64) -> Self {
        Self([value, PackedPoly64::ZERO, PackedPoly64::ZERO])
    }
}

impl From<Poly192> for PackedPoly192 {
    /// The same element in every lane.
    #[inline]
    fn from(value: Poly192) -> Self {
        // Broadcast each coordinate across its own register.
        Self(value.coefficients().map(PackedPoly64::from))
    }
}

impl From<PackedPoly64> for PackedPoly192 {
    /// The coefficient field sits at the constant coordinate, lane by lane.
    #[inline]
    fn from(value: PackedPoly64) -> Self {
        Self::embed(value)
    }
}

impl From<Poly64> for PackedPoly192 {
    #[inline]
    fn from(value: Poly64) -> Self {
        // Broadcast, then place at the constant coordinate.
        Self::embed(value.into())
    }
}

impl From<Gf2> for PackedPoly192 {
    #[inline]
    fn from(value: Gf2) -> Self {
        // The prime subfield embeds as zero or one in every lane.
        Self::from_prime_subfield(value)
    }
}

impl Distribution<PackedPoly192> for StandardUniform {
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> PackedPoly192 {
        // Every lane of every coordinate independently uniform.
        PackedPoly192(array::from_fn(|_| self.sample(rng)))
    }
}

impl Add for PackedPoly192 {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        // Addition is exclusive or, coordinate register by coordinate register.
        Self(array::from_fn(|i| self.0[i] + rhs.0[i]))
    }
}

impl Sub for PackedPoly192 {
    type Output = Self;

    #[inline]
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn sub(self, rhs: Self) -> Self {
        // Subtraction coincides with addition in characteristic 2.
        self + rhs
    }
}

impl Neg for PackedPoly192 {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        // `-x = x` in characteristic 2.
        self
    }
}

impl Mul for PackedPoly192 {
    type Output = Self;

    /// Six unreduced coordinate products, the fold of `y`, then three reductions.
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        // Twelve carryless multiplies for the register, two per Karatsuba term.
        Self::reduce(cubic_mul(self.to_vectors(), rhs.to_vectors()))
    }
}

impl Mul<PackedPoly64> for PackedPoly192 {
    type Output = Self;

    /// The scalar stays inside each coordinate: three products, no fold in `y`.
    #[inline]
    fn mul(self, rhs: PackedPoly64) -> Self {
        // Six carryless multiplies for the register, two per coordinate.
        Self::reduce(cubic_mul_base(self.to_vectors(), rhs.to_vector()))
    }
}

/// Mixed arithmetic against one operand type, each operand converted and then combined.
macro_rules! impl_mixed_ops {
    ($($rhs:ty),*) => {$(
        impl Add<$rhs> for PackedPoly192 {
            type Output = Self;

            #[inline]
            fn add(self, rhs: $rhs) -> Self {
                // Lift the operand into every lane, then add coordinate-wise.
                self + Self::from(rhs)
            }
        }

        impl AddAssign<$rhs> for PackedPoly192 {
            #[inline]
            fn add_assign(&mut self, rhs: $rhs) {
                // In place, through the lifted addition above.
                *self = *self + rhs;
            }
        }

        impl Sub<$rhs> for PackedPoly192 {
            type Output = Self;

            #[inline]
            fn sub(self, rhs: $rhs) -> Self {
                // Subtraction is addition in characteristic 2, after the same lift.
                self - Self::from(rhs)
            }
        }

        impl SubAssign<$rhs> for PackedPoly192 {
            #[inline]
            fn sub_assign(&mut self, rhs: $rhs) {
                // In place, through the lifted subtraction above.
                *self = *self - rhs;
            }
        }

        impl MulAssign<$rhs> for PackedPoly192 {
            #[inline]
            fn mul_assign(&mut self, rhs: $rhs) {
                // In place, through the product with that operand type.
                *self = *self * rhs;
            }
        }
    )*};
}

impl_mixed_ops!(Poly192, PackedPoly64, Poly64, Gf2);

impl AddAssign for PackedPoly192 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        // In place, through the coordinate-wise addition.
        *self = *self + rhs;
    }
}

impl SubAssign for PackedPoly192 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        // In place, through the coordinate-wise subtraction.
        *self = *self - rhs;
    }
}

impl MulAssign for PackedPoly192 {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        // In place, through the Karatsuba product.
        *self = *self * rhs;
    }
}

impl Mul<Poly192> for PackedPoly192 {
    type Output = Self;

    /// Broadcasting first lets a loop-invariant multiplier hoist its three Karatsuba sums.
    #[inline]
    fn mul(self, rhs: Poly192) -> Self {
        // The same element in every lane, then the full packed product.
        self * Self::from(rhs)
    }
}

impl Mul<Poly64> for PackedPoly192 {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Poly64) -> Self {
        // A broadcast coefficient scales every coordinate: three products, no fold.
        self * PackedPoly64::from(rhs)
    }
}

impl Mul<Gf2> for PackedPoly192 {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Gf2) -> Self {
        // The prime subfield has two elements, so this keeps the element or clears it.
        if rhs.is_one() { self } else { Self::ZERO }
    }
}

impl Sum for PackedPoly192 {
    #[inline]
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        // Exclusive or of every term, starting from zero.
        iter.fold(Self::ZERO, Add::add)
    }
}

impl Product for PackedPoly192 {
    #[inline]
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        // Product of every term, starting from one.
        iter.fold(Self::ONE, Mul::mul)
    }
}

impl PrimeCharacteristicRing for PackedPoly192 {
    type PrimeSubfield = Gf2;

    const ZERO: Self = Self::embed(PackedPoly64::ZERO);
    const ONE: Self = Self::embed(PackedPoly64::ONE);
    // The characteristic is 2, so `TWO = ONE + ONE = ZERO`.
    const TWO: Self = Self::embed(PackedPoly64::ZERO);
    // The characteristic is 2, so `NEG_ONE = ONE`.
    const NEG_ONE: Self = Self::embed(PackedPoly64::ONE);

    #[inline]
    fn from_prime_subfield(f: Self::PrimeSubfield) -> Self {
        // Zero or one in every lane, at the constant coordinate.
        Self::embed(PackedPoly64::from_prime_subfield(f))
    }

    #[inline]
    fn double(&self) -> Self {
        // `x + x = 0` in characteristic 2.
        Self::ZERO
    }

    /// Three coordinate squares and the fold of `y`: six carryless multiplies for the register.
    #[inline]
    fn square(&self) -> Self {
        // The cross terms vanish in characteristic 2, so only the coordinate squares remain.
        Self::reduce(cubic_square(self.to_vectors()))
    }

    /// Reduction is linear, so the whole sum reduces its three coordinates once.
    #[inline]
    fn dot_product<const N: usize>(u: &[Self; N], v: &[Self; N]) -> Self {
        // Accumulate the folded, unreduced coordinates of every term.
        let sum = u.iter().zip(v).fold([Wide::zero(); DEGREE], |sum, (a, b)| {
            let product = cubic_mul(a.to_vectors(), b.to_vectors());
            array::from_fn(|i| sum[i].xor(product[i]))
        });

        // Three reductions for the whole sum, not three per term.
        Self::reduce(sum)
    }

    #[inline]
    fn zero_vec(len: usize) -> Vec<Self> {
        // Zero is all-zero bits, so the allocator's zeroed path applies.
        alloc::vec![Self::ZERO; len]
    }
}

impl Algebra<Gf2> for PackedPoly192 {}

impl Algebra<Poly64> for PackedPoly192 {}

impl Algebra<Poly192> for PackedPoly192 {}

impl Algebra<PackedPoly64> for PackedPoly192 {
    /// Three products per term and one reduction per coordinate for the whole sum.
    #[inline]
    fn mixed_dot_product<const N: usize>(a: &[Self; N], f: &[PackedPoly64; N]) -> Self
    where
        PackedPoly64: Dup,
    {
        // Accumulate three unreduced coordinate products per term.
        let sum = a.iter().zip(f).fold([Wide::zero(); DEGREE], |sum, (x, k)| {
            let product = cubic_mul_base(x.to_vectors(), k.to_vector());
            array::from_fn(|i| sum[i].xor(product[i]))
        });

        // One reduction per coordinate for the whole sum.
        Self::reduce(sum)
    }
}

impl BasedVectorSpace<PackedPoly64> for PackedPoly192 {
    const DIMENSION: usize = DEGREE;

    #[inline]
    fn as_basis_coefficients_slice(&self) -> &[PackedPoly64] {
        // The coordinate registers are the coefficients, stored in basis order.
        &self.0
    }

    #[inline]
    fn from_basis_coefficients_fn<F: FnMut(usize) -> PackedPoly64>(f: F) -> Self {
        // One call per basis element, in order.
        Self(array::from_fn(f))
    }

    #[inline]
    fn from_basis_coefficients_iter<I: ExactSizeIterator<Item = PackedPoly64>>(
        iter: I,
    ) -> Option<Self> {
        (iter.len() == DEGREE).then(|| {
            // The reported length is not a guarantee, so zipping bounds the fill either way.
            let mut coordinates = [PackedPoly64::ZERO; DEGREE];
            for (slot, value) in coordinates.iter_mut().zip(iter) {
                *slot = value;
            }
            Self(coordinates)
        })
    }
}

impl PackedFieldExtension<Poly64, Poly192> for PackedPoly192 {
    /// Calls the closure once per lane, then transposes the elements into coordinate registers.
    #[inline]
    fn from_ext_fn(f: impl Fn(usize) -> Poly192) -> Self {
        // Why: a column-by-column gather would call the closure once per coordinate, thrice a lane.
        //
        // The caller's closure is often a fold or a product, so that would triple its cost.
        let rows: [Poly192; WIDTH_64] = array::from_fn(f);
        Self::from_ext_slice(&rows)
    }

    /// Three registers of consecutive elements, transposed into coordinate registers.
    ///
    /// # Panics
    ///
    /// Panics if the slice does not hold exactly one packing's worth of elements.
    #[inline]
    fn from_ext_slice(slice: &[Poly192]) -> Self {
        let rows: &[Poly192; WIDTH_64] = slice.try_into().expect("slice length is not the width");

        // SAFETY: an element is `repr(transparent)` over three quadwords.
        //
        // One packing's worth of elements is then exactly three registers of quadwords.
        //
        // The loads are the unaligned form.
        let interleaved = unsafe {
            let base = rows.as_ptr().cast::<u128>();
            array::from_fn(|r| lanes::load(base.add(r * lanes::WIDTH)))
        };
        // Six blends and three permutes gather each coordinate into its own register.
        Self::from_vectors(lanes::deinterleave_3(interleaved))
    }

    /// The inverse transpose, written straight into the slice.
    ///
    /// # Panics
    ///
    /// Panics if the slice does not hold exactly one packing's worth of elements.
    #[inline]
    fn to_ext_slice(&self, out: &mut [Poly192]) {
        let rows: &mut [Poly192; WIDTH_64] = out.try_into().expect("slice length is not the width");

        // Back to consecutive elements, three registers of them.
        let interleaved = lanes::interleave_3(self.to_vectors());

        // SAFETY: the destination is one packing's worth of elements, exactly three registers.
        unsafe {
            let base = rows.as_mut_ptr().cast::<u128>();
            for (r, register) in interleaved.into_iter().enumerate() {
                lanes::store(base.add(r * lanes::WIDTH), register);
            }
        }
    }

    #[inline]
    fn extract(&self, lane: usize) -> Poly192 {
        // Read the lane out of each coordinate register.
        Poly192::new(self.0.map(|c| c.as_slice()[lane]))
    }

    #[inline]
    fn add_assign_lane(&mut self, lane: usize, value: Poly192) {
        // Add each coordinate into its register at the shared lane.
        for (coordinate, v) in self.0.iter_mut().zip(value.coefficients()) {
            coordinate.as_slice_mut()[lane] += v;
        }
    }

    #[inline]
    fn packed_ext_powers(base: Poly192) -> Powers<Self> {
        // The first W powers transposed into lanes, then stepped by base^W.
        //
        //     lanes   = [ 1, b, b^2, ..., b^(W - 1) ]
        //     step    = b^W
        let powers = base.powers().collect_n(WIDTH_64 + 1);
        Powers {
            base: Self::from(powers[WIDTH_64]),
            current: Self::from_ext_slice(&powers[..WIDTH_64]),
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use p3_field::{PackedFieldExtension, PrimeCharacteristicRing};
    use p3_field_testing::{
        test_add_assign_lane_ext, test_batched_linear_combination_ext, test_packed_extension,
        test_ring_axioms_proptest_char2, test_ring_with_eq_char2,
    };
    use proptest::prelude::*;

    use super::super::lanes::gf64::WIDTH_64;
    use crate::{PackedPoly64, PackedPoly192, Poly64, Poly192};

    /// A packing of the given elements.
    fn packed(values: &[Poly192]) -> PackedPoly192 {
        // Through the transposing load, so every test also exercises it.
        PackedPoly192::from_ext_slice(values)
    }

    /// The elements a packing holds, lane by lane.
    fn unpacked(value: PackedPoly192) -> Vec<Poly192> {
        // Through the transposing store.
        let mut out = alloc::vec![Poly192::ZERO; WIDTH_64];
        value.to_ext_slice(&mut out);
        out
    }

    /// Elements from raw coordinates.
    fn elements(raw: [[u64; 3]; WIDTH_64]) -> Vec<Poly192> {
        // Every bit pattern is an element, so no rejection is needed.
        raw.iter()
            .map(|c| Poly192::new(c.map(Poly64::new)))
            .collect()
    }

    /// The corner coordinates of the base field.
    const CORNERS: [u64; 5] = [0, 1, u64::MAX, 1 << 63, 0xf << 60];

    #[test]
    fn every_lane_matches_the_scalar_field_at_the_corners() {
        // Invariant: every lane multiplies independently, even at the extremes.
        //
        // Fixture state: coordinate i of lane l is corner (shift + l + 2i) mod 5.
        //
        // Neighbouring lanes then differ, so a result that crossed a lane shows as a mismatch.
        for shift in 0..CORNERS.len() {
            let raw: [[u64; 3]; WIDTH_64] = core::array::from_fn(|lane| {
                core::array::from_fn(|i| CORNERS[(shift + lane + 2 * i) % CORNERS.len()])
            });
            let a = elements(raw);
            // The second operand is the first reversed, pairing different corners per lane.
            let b: Vec<Poly192> = a.iter().rev().copied().collect();
            let got = unpacked(packed(&a) * packed(&b));
            let want: Vec<Poly192> = a.iter().zip(&b).map(|(x, y)| *x * *y).collect();
            assert_eq!(got, want, "offset {shift}");
            assert_eq!(
                unpacked(packed(&a).square()),
                a.iter().map(Poly192::square).collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn the_shared_packed_extension_suite_passes() {
        // The transposes, the lane injection, the powers and the batched combination.
        test_packed_extension::<Poly64, Poly192>();
        test_add_assign_lane_ext::<Poly64, Poly192, PackedPoly192>();
        test_batched_linear_combination_ext::<Poly64, Poly192, PackedPoly192>();
    }

    #[test]
    fn the_shared_ring_suite_passes() {
        // The packing is a ring of characteristic 2 in its own right.
        test_ring_with_eq_char2::<PackedPoly192>(&[PackedPoly192::ZERO], &[PackedPoly192::ONE]);
        test_ring_axioms_proptest_char2::<PackedPoly192>();
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(1000))]

        #[test]
        fn the_transpose_round_trips(raw in any::<[[u64; 3]; WIDTH_64]>()) {
            // Invariant: the transposing load and store are inverse permutations.
            let a = elements(raw);
            let p = packed(&a);
            prop_assert_eq!(&unpacked(p), &a);

            // The fast transpose against the lane-by-lane constructor.
            prop_assert_eq!(p, PackedPoly192::from_ext_fn(|lane| a[lane]));
            for (lane, &value) in a.iter().enumerate() {
                prop_assert_eq!(PackedFieldExtension::<Poly64, Poly192>::extract(&p, lane), value);
            }
        }

        #[test]
        fn every_lane_matches_the_scalar_field(
            a in any::<[[u64; 3]; WIDTH_64]>(),
            b in any::<[[u64; 3]; WIDTH_64]>(),
            k in any::<[u64; WIDTH_64]>(),
            s in any::<[u64; 3]>(),
        ) {
            // Invariant: every packed operation is the scalar one, lane by lane.
            //
            // Fixture state: two packed elements, one packed base scalar, one broadcast element.
            let (x, y) = (elements(a), elements(b));
            let (px, py) = (packed(&x), packed(&y));
            let scalars: Vec<Poly64> = k.iter().copied().map(Poly64::new).collect();
            let pk = *<PackedPoly64 as p3_field::PackedValue>::from_slice(&scalars);
            let broadcast = Poly192::new(s.map(Poly64::new));

            // The scalar expectation, one lane at a time.
            let each = |f: &dyn Fn(usize) -> Poly192| (0..WIDTH_64).map(f).collect::<Vec<_>>();

            prop_assert_eq!(unpacked(px * py), each(&|l| x[l] * y[l]));
            prop_assert_eq!(unpacked(px.square()), each(&|l| x[l].square()));
            prop_assert_eq!(unpacked(px * pk), each(&|l| x[l] * scalars[l]));
            prop_assert_eq!(unpacked(px * broadcast), each(&|l| x[l] * broadcast));

            // Both deferred-reduction sums against one reduction per term.
            prop_assert_eq!(
                unpacked(PackedPoly192::dot_product(&[px, py], &[py, px * py])),
                each(&|l| x[l] * y[l] + y[l] * (x[l] * y[l])),
            );
            prop_assert_eq!(
                unpacked(<PackedPoly192 as p3_field::Algebra<PackedPoly64>>::mixed_dot_product(
                    &[px, py],
                    &[pk, pk],
                )),
                each(&|l| x[l] * scalars[l] + y[l] * scalars[l]),
            );
        }
    }
}
