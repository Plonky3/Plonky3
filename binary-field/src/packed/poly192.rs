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
//! A product costs six carryless multiplies per element, all of them filling every lane.
//!
//! Only three reductions follow, one per coordinate register.

use core::array;
use core::iter::{Product, Sum};
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use p3_field::{
    Algebra, BasedVectorSpace, Dup, Field, PackedFieldExtension, PackedValue, Powers,
    PrimeCharacteristicRing,
};
use rand::Rng;
use rand::distr::{Distribution, StandardUniform};

use super::gf64::{self as lanes, Reg, WIDTH_64};
use super::poly64::PackedPoly64;
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
use super::x86_64::pairs;
use crate::clmul::wide::{Lanes64, Wide, cubic_mul, cubic_mul_base, cubic_square};
use crate::clmul::{poly_dot_192_by_64, raw_product_64, reduce_64};
use crate::{Gf2, Poly64, Poly192, Poly192Unreduced};

/// Multiply exactly four scalar pairs with the 256-bit coordinate backend.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "vpclmulqdq",
    target_feature = "avx2",
    not(all(
        feature = "wide-poly",
        target_feature = "avx512f",
        target_feature = "avx512bw"
    ))
))]
#[inline]
pub(crate) fn mul4(a: [Poly192; 4], b: [Poly192; 4]) -> [Poly192; 4] {
    use super::x86_64::lanes::gf64 as short;
    // SAFETY: each array is twelve quadwords, exactly three 256-bit registers.
    // Poly192 and Poly64 are transparent over their coordinate arrays and words.
    let (a, b) = unsafe {
        (
            short::gather_3(a.as_ptr().cast()),
            short::gather_3(b.as_ptr().cast()),
        )
    };
    let products = cubic_mul(a, b).map(Wide::reduce);
    let mut out = [Poly192::ZERO; 4];
    // SAFETY: the destination is four complete extension elements; no alignment is required.
    unsafe { short::scatter_3(out.as_mut_ptr().cast(), products) };
    out
}

/// Multiply four pairs using one 128-bit polynomial lane per pair.
#[cfg(all(
    feature = "wide-poly",
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512bw",
    target_feature = "vpclmulqdq",
    target_feature = "avx2"
))]
#[inline]
pub(crate) fn mul4(a: [Poly192; 4], b: [Poly192; 4]) -> [Poly192; 4] {
    use core::arch::x86_64::{__m512i, _mm512_set_epi64};

    use crate::clmul::wide::fold_cubic;

    // Each 128-bit lane holds one coefficient in its low half. One carryless
    // multiplication therefore computes all four products of that coefficient.
    let pack = |values: &[Poly192; 4], coordinate| {
        // SAFETY: this function is compiled only with AVX-512F enabled.
        unsafe {
            _mm512_set_epi64(
                0,
                values[3].limbs()[coordinate] as i64,
                0,
                values[2].limbs()[coordinate] as i64,
                0,
                values[1].limbs()[coordinate] as i64,
                0,
                values[0].limbs()[coordinate] as i64,
            )
        }
    };
    let [a0, a1, a2] = array::from_fn(|i| pack(&a, i));
    let [b0, b1, b2] = array::from_fn(|i| pack(&b, i));
    let terms = [
        a0.clmul::<0>(b0),
        a1.clmul::<0>(b1),
        a2.clmul::<0>(b2),
        a0.xor(a1).clmul::<0>(b0.xor(b1)),
        a0.xor(a2).clmul::<0>(b0.xor(b2)),
        a1.xor(a2).clmul::<0>(b1.xor(b2)),
    ];
    let [r0, r1, r2] = fold_cubic(terms, Lanes64::xor, Lanes64::xor3);
    let pair = Wide { even: r0, odd: r1 }.reduce();
    let last = r2.reduce_lane();
    // SAFETY: both registers contain eight unrestricted u64 words. The pair
    // has the first two reduced coefficients together; the last has each third
    // coefficient in the low half of its original 128-bit lane.
    let (pair, last): ([u64; 8], [u64; 8]) = unsafe {
        (
            core::mem::transmute::<__m512i, [u64; 8]>(pair),
            core::mem::transmute::<__m512i, [u64; 8]>(last),
        )
    };
    array::from_fn(|i| Poly192::from_limbs([pair[2 * i], pair[2 * i + 1], last[2 * i]]))
}

/// The number of coordinates over the coefficient field.
const DEGREE: usize = 3;

/// Several elements of `GF(2^192)`, as one packing of `GF(2^64)` per coordinate.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(transparent)]
#[must_use]
pub struct PackedPoly192([PackedPoly64; DEGREE]);

/// Lane-wise extension products whose coefficient reductions remain deferred.
#[derive(Clone, Copy, Debug)]
#[must_use]
pub struct PackedPoly192Unreduced(
    /// Even and odd polynomial products of each coordinate.
    [Wide<Reg>; DEGREE],
);

impl Default for PackedPoly192Unreduced {
    fn default() -> Self {
        Self([Wide::zero(); DEGREE])
    }
}

impl Add for PackedPoly192Unreduced {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self(core::array::from_fn(|i| self.0[i].xor(rhs.0[i])))
    }
}

impl AddAssign for PackedPoly192Unreduced {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl PackedPoly192Unreduced {
    /// Reduce each lane, preserving the selected packing's lane order.
    #[inline]
    pub fn reduce(self) -> PackedPoly192 {
        PackedPoly192::reduce(self.0)
    }

    /// Sum every lane without reducing, so partial sums can still be combined.
    #[inline]
    pub fn sum_lanes(self) -> Poly192Unreduced {
        Poly192Unreduced(self.0.map(|sum| {
            let both = sum.even.xor(sum.odd);
            // SAFETY: a register is WIDTH_64/2 unrestricted 128-bit polynomial lanes.
            let lanes: [u128; WIDTH_64 / 2] = unsafe { core::mem::transmute(both) };
            lanes.into_iter().fold(0, |sum, lane| sum ^ lane)
        }))
    }
}

/// Coordinate sums whose 128-bit polynomial lanes remain in the selected vector registers.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PackedMixedAccumulator(
    /// One register of polynomial sums per extension coordinate.
    [Reg; DEGREE],
);

impl Default for PackedMixedAccumulator {
    fn default() -> Self {
        Self([Reg::zero(); DEGREE])
    }
}

impl PackedMixedAccumulator {
    /// Accumulate one complete packing before any horizontal sum or field reduction.
    #[inline]
    pub(crate) fn add_packed(&mut self, values: PackedPoly192, weights: PackedPoly64) {
        let products = cubic_mul_base(values.to_vectors(), weights.to_vector());
        for (sum, product) in self.0.iter_mut().zip(products) {
            *sum = sum.xor3(product.even, product.odd);
        }
    }

    /// Accumulate whole packings, then place a scalar tail in the first polynomial lane.
    #[inline]
    pub(crate) fn add_dot(&mut self, values: &[Poly192], weights: &[Poly64]) {
        let done = values.len() / WIDTH_64 * WIDTH_64;
        for start in (0..done).step_by(WIDTH_64) {
            let values = PackedPoly192::from_ext_slice(&values[start..start + WIDTH_64]);
            let weights = PackedPoly64::from_fn(|lane| weights[start + lane]);
            self.add_packed(values, weights);
        }
        if done < values.len() {
            let tail = values[done..].iter().zip(&weights[done..]).fold(
                [0u128; DEGREE],
                |sum, (value, weight)| {
                    core::array::from_fn(|i| {
                        sum[i] ^ raw_product_64(value.limbs()[i], weight.to_bits())
                    })
                },
            );
            for (sum, tail) in self.0.iter_mut().zip(tail) {
                let mut lanes = [0u128; WIDTH_64 / 2];
                lanes[0] = tail;
                // SAFETY: the lane array and register have equal sizes and unrestricted bit patterns.
                let tail: Reg = unsafe { core::mem::transmute(lanes) };
                *sum = sum.xor(tail);
            }
        }
    }

    /// Combine partial sums in their vector representation.
    #[inline]
    pub(crate) fn merge(&mut self, other: Self) {
        for (sum, term) in self.0.iter_mut().zip(other.0) {
            *sum = sum.xor(term);
        }
    }

    /// Sum the polynomial lanes, leaving the field reduction to the caller.
    #[inline]
    pub(crate) fn coordinates(self) -> [u128; DEGREE] {
        self.0.map(|sum| {
            // SAFETY: each register holds WIDTH_64/2 unrestricted 128-bit polynomial sums.
            let products: [u128; WIDTH_64 / 2] = unsafe { core::mem::transmute(sum) };
            products.into_iter().fold(0, |sum, product| sum ^ product)
        })
    }
}

impl PackedPoly192 {
    /// Multiply each lane without reducing its coefficient-field coordinates.
    #[inline]
    pub fn mul_unreduced(self, rhs: Self) -> PackedPoly192Unreduced {
        PackedPoly192Unreduced(cubic_mul(self.to_vectors(), rhs.to_vectors()))
    }

    /// Multiply each lane by its coefficient-field weight without reducing.
    #[inline]
    pub fn mul_base_unreduced(self, rhs: PackedPoly64) -> PackedPoly192Unreduced {
        PackedPoly192Unreduced(cubic_mul_base(self.to_vectors(), rhs.to_vector()))
    }

    /// Sum scalar extension-by-base products across packed lanes before reducing.
    #[inline]
    pub(crate) fn mixed_dot_scalar(a: &[Poly192], f: &[Poly64]) -> Poly192 {
        let mut sums = [Wide::<Reg>::zero(); DEGREE];
        let done = a.len() / WIDTH_64 * WIDTH_64;
        for start in (0..done).step_by(WIDTH_64) {
            // Full groups load one extension coordinate per register lane.
            let x = Self::from_ext_slice(&a[start..start + WIDTH_64]);
            let k = PackedPoly64::from_fn(|lane| f[start + lane]);
            let products = cubic_mul_base(x.to_vectors(), k.to_vector());
            for (sum, product) in sums.iter_mut().zip(products) {
                *sum = sum.xor(product);
            }
        }
        let coordinates = sums.map(|sum| {
            // Even and odd products share the same 128-bit polynomial representation.
            let both = sum.even.xor(sum.odd);
            // SAFETY: a coordinate register holds WIDTH_64/2 unrestricted 128-bit products.
            let products: [u128; WIDTH_64 / 2] = unsafe { core::mem::transmute(both) };
            let product = products.into_iter().fold(0, |sum, product| sum ^ product);
            Poly64::new(reduce_64(product))
        });
        let tail = poly_dot_192_by_64(
            a[done..]
                .iter()
                .zip(&f[done..])
                .map(|(x, k)| (x.limbs(), k.as_bits())),
        );
        Poly192::new(coordinates) + Poly192::from_limbs(tail)
    }

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

    /// `x (x - 1) = x^2 + x` in characteristic 2, and squaring is the cheaper product.
    #[inline]
    fn bool_check(&self) -> Self {
        // Zero exactly on the two roots of x^2 + x, which are zero and one.
        self.square() + *self
    }

    /// Reduction is linear, so the whole sum reduces its three coordinates once.
    #[inline]
    fn dot_product<const N: usize>(u: &[Self; N], v: &[Self; N]) -> Self {
        // Two terms per 512-bit multiply, where the target has one.
        #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
        let sum = pairs::sum_of_products(
            u,
            v,
            |a, b| {
                let join = |x: &[Self; 2]| pairs::join_all(x[0].to_vectors(), x[1].to_vectors());
                cubic_mul(join(a), join(b))
            },
            |a, b| cubic_mul(a.to_vectors(), b.to_vectors()),
        );

        // Otherwise accumulate the folded, unreduced coordinates of every term.
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
        let sum = u.iter().zip(v).fold([Wide::zero(); DEGREE], |sum, (a, b)| {
            let product = cubic_mul(a.to_vectors(), b.to_vectors());
            array::from_fn(|i| sum[i].xor(product[i]))
        });

        // Three reductions for the whole sum, not three per term.
        Self::reduce(sum)
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
        // Two terms per 512-bit multiply, where the target has one.
        #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
        let sum = pairs::sum_of_products(
            a,
            f,
            |x, k| {
                let x = pairs::join_all(x[0].to_vectors(), x[1].to_vectors());
                cubic_mul_base(x, pairs::join(k[0].to_vector(), k[1].to_vector()))
            },
            |x, k| cubic_mul_base(x.to_vectors(), k.to_vector()),
        );

        // Otherwise accumulate three unreduced coordinate products per term.
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
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
        let coordinates = unsafe { lanes::gather_3(rows.as_ptr().cast()) };
        Self::from_vectors(coordinates)
    }

    /// The inverse transpose, written straight into the slice.
    ///
    /// # Panics
    ///
    /// Panics if the slice does not hold exactly one packing's worth of elements.
    #[inline]
    fn to_ext_slice(&self, out: &mut [Poly192]) {
        let rows: &mut [Poly192; WIDTH_64] = out.try_into().expect("slice length is not the width");

        // SAFETY: the destination is one packing's worth of elements, exactly three registers.
        unsafe { lanes::scatter_3(rows.as_mut_ptr().cast(), self.to_vectors()) }
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

    use p3_field::{PackedFieldExtension, PackedValue, PrimeCharacteristicRing};
    use p3_field_testing::{
        test_add_assign_lane_ext, test_batched_linear_combination_ext, test_packed_extension,
        test_ring_axioms_proptest_char2, test_ring_with_eq_char2,
    };
    use proptest::prelude::*;

    use crate::packed::gf64::WIDTH_64;
    use crate::{PackedPoly64, PackedPoly192, PackedPoly192Unreduced, Poly64, Poly192};

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
            // The booleanity shortcut against the general product it replaces.
            assert_eq!(
                unpacked(packed(&a).bool_check()),
                a.iter()
                    .map(|&x| x * (x - Poly192::ONE))
                    .collect::<Vec<_>>()
            );
        }
    }

    /// Both dot products of the first `N` corner terms, lane by lane against the scalar sums.
    fn check_dot_products<const N: usize>() {
        // Term t takes corner (step t + lane + 2i + offset) in coordinate i.
        //
        // So every term, every lane and every coordinate differs from its neighbours.
        let corner = |t: usize, step: usize, offset: usize, lane: usize, i: usize| {
            CORNERS[(step * t + lane + 2 * i + offset) % CORNERS.len()]
        };
        let term = |t: usize, step: usize, offset: usize| {
            elements(core::array::from_fn(|lane| {
                core::array::from_fn(|i| corner(t, step, offset, lane, i))
            }))
        };
        let x: [Vec<Poly192>; N] = core::array::from_fn(|t| term(t, 1, 0));
        let y: [Vec<Poly192>; N] = core::array::from_fn(|t| term(t, 2, 1));
        let k: [Vec<Poly64>; N] = core::array::from_fn(|t| {
            (0..WIDTH_64)
                .map(|lane| Poly64::new(corner(t, 3, 2, lane, 0)))
                .collect()
        });

        let px: [PackedPoly192; N] = core::array::from_fn(|t| packed(&x[t]));
        let py: [PackedPoly192; N] = core::array::from_fn(|t| packed(&y[t]));
        let pk: [PackedPoly64; N] =
            core::array::from_fn(|t| *<PackedPoly64 as p3_field::PackedValue>::from_slice(&k[t]));

        // One product per term, each reduced on its own, lane by lane.
        let dot: Vec<Poly192> = (0..WIDTH_64)
            .map(|l| (0..N).map(|t| x[t][l] * y[t][l]).sum())
            .collect();
        let mixed: Vec<Poly192> = (0..WIDTH_64)
            .map(|l| (0..N).map(|t| x[t][l] * k[t][l]).sum())
            .collect();

        assert_eq!(
            unpacked(PackedPoly192::dot_product(&px, &py)),
            dot,
            "N = {N}"
        );
        assert_eq!(
            unpacked(
                <PackedPoly192 as p3_field::Algebra<PackedPoly64>>::mixed_dot_product(&px, &pk)
            ),
            mixed,
            "N = {N}"
        );
    }

    #[test]
    fn every_dot_product_length_matches_the_sum_of_products() {
        // Wide builds pair the terms, so the lengths cover every split:
        //
        // - 0: the empty sum;
        // - 1: no pair, only the leftover term;
        // - 2 and 4: pairs only;
        // - 3 and 5: pairs and a leftover term.
        check_dot_products::<0>();
        check_dot_products::<1>();
        check_dot_products::<2>();
        check_dot_products::<3>();
        check_dot_products::<4>();
        check_dot_products::<5>();
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
        fn deferred_products_keep_lanes_and_horizontal_sums(
            a in any::<[[[u64; 3]; WIDTH_64]; 3]>(),
            b in any::<[[[u64; 3]; WIDTH_64]; 3]>(),
            k in any::<[[u64; WIDTH_64]; 3]>(),
        ) {
            let mut sum = PackedPoly192Unreduced::default();
            let mut expected = alloc::vec![Poly192::ZERO; WIDTH_64];
            for group in 0..3 {
                let (a, b) = (elements(a[group]), elements(b[group]));
                let k = PackedPoly64::from_fn(|lane| Poly64::new(k[group][lane]));
                sum += packed(&a).mul_unreduced(packed(&b));
                sum += packed(&a).mul_base_unreduced(k);
                for lane in 0..WIDTH_64 {
                    expected[lane] += a[lane] * b[lane] + a[lane] * k.as_slice()[lane];
                }
            }
            prop_assert_eq!(&unpacked(sum.reduce()), &expected);
            prop_assert_eq!(sum.sum_lanes().reduce(), expected.into_iter().sum::<Poly192>());
            prop_assert_eq!(unpacked((sum + sum).reduce()), alloc::vec![Poly192::ZERO; WIDTH_64]);
        }

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
            prop_assert_eq!(unpacked(px.bool_check()), each(&|l| x[l] * (x[l] - Poly192::ONE)));
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
