//! The packing of the polynomial-basis `GF(2^128)` over the wide carryless multiply.
//!
//! `VPCLMULQDQ` applies the carryless multiply to every 128-bit lane of a wide register.
//! One field element is exactly one lane, so the scalar kernel becomes the packed one.
//!
//! Every other operation is exclusive or, a lane-local shift, or a permutation of lanes.
//!
//! Only the widest register the target supports is compiled.
//! One width per build means everything below the lane wrappers is written once.

use core::iter::{Product, Sum};
use core::mem::transmute;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use p3_field::op_assign_macros::{
    impl_add_assign, impl_add_base_field, impl_div_methods, impl_mul_base_field, impl_mul_methods,
    impl_packed_field_div, impl_packed_value, impl_rng, impl_sub_assign, impl_sub_base_field,
    impl_sum_prod_base_field, ring_sum,
};
use p3_field::{
    Algebra, Field, PackedField, PackedFieldPow2, PackedValue, PrimeCharacteristicRing,
};
use rand::distr::{Distribution, StandardUniform};
use rand::{Rng, RngExt};

use super::split::{HIGH_BY_HIGH, LOW_BY_LOW, Lanes, fold_shifted};
use crate::{Gf2, Ghash128};

/// Swaps the two quadwords of every lane, so `x ^ swap(x)` holds `x_lo ^ x_hi` in both halves.
const SWAP_QUADWORDS: i32 = 0x4e;

/// The 256-bit register, holding two field elements.
#[cfg(all(
    target_feature = "avx2",
    target_feature = "vpclmulqdq",
    not(target_feature = "avx512f")
))]
pub(crate) mod lanes {
    use core::arch::x86_64::{
        __m256i, _mm_set_epi64x, _mm256_broadcastsi128_si256, _mm256_clmulepi64_epi128,
        _mm256_loadu_si256, _mm256_setzero_si256, _mm256_shuffle_epi32, _mm256_storeu_si256,
        _mm256_unpacklo_epi64, _mm256_xor_si256,
    };

    use p3_field::interleave::interleave_u128;

    /// The register one packed value occupies.
    pub(crate) type Reg = __m256i;

    /// Field elements per register.
    pub(crate) const WIDTH: usize = 2;

    // SAFETY for every wrapper below: this module is compiled only when the target features
    // its intrinsics require are enabled for the crate, which is what makes each call sound.

    /// All lanes zero.
    #[inline(always)]
    pub(crate) fn zero() -> Reg {
        unsafe { _mm256_setzero_si256() }
    }

    /// Bitwise exclusive or.
    #[inline(always)]
    pub(crate) fn xor(a: Reg, b: Reg) -> Reg {
        unsafe { _mm256_xor_si256(a, b) }
    }

    /// The low quadword of each operand, paired within each lane.
    #[inline(always)]
    pub(crate) fn unpack_low_64(a: Reg, b: Reg) -> Reg {
        unsafe { _mm256_unpacklo_epi64(a, b) }
    }

    /// The carryless product of one quadword of each operand, in every lane.
    #[inline(always)]
    pub(crate) fn clmul<const IMM: i32>(a: Reg, b: Reg) -> Reg {
        unsafe { _mm256_clmulepi64_epi128::<IMM>(a, b) }
    }

    /// Exchanges the two halves of every lane.
    #[inline(always)]
    pub(crate) fn swap_halves(a: Reg) -> Reg {
        unsafe { _mm256_shuffle_epi32::<{ super::SWAP_QUADWORDS }>(a) }
    }

    /// The same field element in every lane.
    #[inline(always)]
    pub(crate) fn broadcast(value: u128) -> Reg {
        unsafe {
            // Arguments run from the highest quadword down.
            let lane = _mm_set_epi64x((value >> 64) as i64, value as i64);
            _mm256_broadcastsi128_si256(lane)
        }
    }

    /// Reads one register of consecutive field elements.
    ///
    /// # Safety
    ///
    /// The address must be readable for as many elements as one register holds.
    /// No alignment is required.
    #[inline(always)]
    pub(crate) unsafe fn load(from: *const u128) -> Reg {
        // SAFETY: the readability of the address is the caller's obligation.
        unsafe { _mm256_loadu_si256(from.cast()) }
    }

    /// Writes one register of consecutive field elements.
    ///
    /// # Safety
    ///
    /// The address must be writable for as many elements as one register holds.
    /// No alignment is required.
    #[inline(always)]
    pub(crate) unsafe fn store(to: *mut u128, value: Reg) {
        // SAFETY: the writability of the address is the caller's obligation.
        unsafe { _mm256_storeu_si256(to.cast(), value) }
    }

    /// Interleaves whole field elements between two registers.
    ///
    /// # Panics
    /// Panics if the block length does not divide the width.
    #[inline(always)]
    pub(crate) fn interleave(a: Reg, b: Reg, block_len: usize) -> (Reg, Reg) {
        match block_len {
            1 => interleave_u128(a, b),
            WIDTH => (a, b),
            _ => panic!("unsupported block_len"),
        }
    }
}

/// The 512-bit register, holding four field elements.
#[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
pub(crate) mod lanes {
    use core::arch::x86_64::{
        __m512i, _mm_set_epi64x, _mm512_broadcast_i32x4, _mm512_clmulepi64_epi128,
        _mm512_loadu_si512, _mm512_setzero_si512, _mm512_shuffle_epi32, _mm512_storeu_si512,
        _mm512_unpacklo_epi64, _mm512_xor_si512,
    };

    use p3_field::interleave::{interleave_u128, interleave_u256};

    /// The register one packed value occupies.
    pub(crate) type Reg = __m512i;

    /// Field elements per register.
    pub(crate) const WIDTH: usize = 4;

    // SAFETY for every wrapper below: this module is compiled only when the target features
    // its intrinsics require are enabled for the crate, which is what makes each call sound.

    /// All lanes zero.
    #[inline(always)]
    pub(crate) fn zero() -> Reg {
        unsafe { _mm512_setzero_si512() }
    }

    /// Bitwise exclusive or.
    #[inline(always)]
    pub(crate) fn xor(a: Reg, b: Reg) -> Reg {
        unsafe { _mm512_xor_si512(a, b) }
    }

    /// The low quadword of each operand, paired within each lane.
    #[inline(always)]
    pub(crate) fn unpack_low_64(a: Reg, b: Reg) -> Reg {
        unsafe { _mm512_unpacklo_epi64(a, b) }
    }

    /// The carryless product of one quadword of each operand, in every lane.
    #[inline(always)]
    pub(crate) fn clmul<const IMM: i32>(a: Reg, b: Reg) -> Reg {
        unsafe { _mm512_clmulepi64_epi128::<IMM>(a, b) }
    }

    /// Exchanges the two halves of every lane.
    #[inline(always)]
    pub(crate) fn swap_halves(a: Reg) -> Reg {
        unsafe { _mm512_shuffle_epi32::<{ super::SWAP_QUADWORDS }>(a) }
    }

    /// The same field element in every lane.
    #[inline(always)]
    pub(crate) fn broadcast(value: u128) -> Reg {
        unsafe {
            // Arguments run from the highest quadword down.
            let lane = _mm_set_epi64x((value >> 64) as i64, value as i64);
            _mm512_broadcast_i32x4(lane)
        }
    }

    /// Reads one register of consecutive field elements.
    ///
    /// # Safety
    ///
    /// The address must be readable for as many elements as one register holds.
    /// No alignment is required.
    #[inline(always)]
    pub(crate) unsafe fn load(from: *const u128) -> Reg {
        // SAFETY: the readability of the address is the caller's obligation.
        unsafe { _mm512_loadu_si512(from.cast()) }
    }

    /// Writes one register of consecutive field elements.
    ///
    /// # Safety
    ///
    /// The address must be writable for as many elements as one register holds.
    /// No alignment is required.
    #[inline(always)]
    pub(crate) unsafe fn store(to: *mut u128, value: Reg) {
        // SAFETY: the writability of the address is the caller's obligation.
        unsafe { _mm512_storeu_si512(to.cast(), value) }
    }

    /// Interleaves whole field elements between two registers.
    ///
    /// # Panics
    /// Panics if the block length does not divide the width.
    #[inline(always)]
    pub(crate) fn interleave(a: Reg, b: Reg, block_len: usize) -> (Reg, Reg) {
        match block_len {
            1 => interleave_u128(a, b),
            2 => interleave_u256(a, b),
            WIDTH => (a, b),
            _ => panic!("unsupported block_len"),
        }
    }
}

use lanes::WIDTH;

// The register is the production backend of the shared split-multiplier algebra.
//
// Every method is one intrinsic.
//
// So the generic code monomorphizes to the instructions a hand-written kernel would emit.
impl Lanes for lanes::Reg {
    #[inline(always)]
    fn zero() -> Self {
        lanes::zero()
    }

    #[inline(always)]
    fn broadcast(value: u128) -> Self {
        lanes::broadcast(value)
    }

    #[inline(always)]
    fn xor(self, other: Self) -> Self {
        lanes::xor(self, other)
    }

    #[inline(always)]
    fn unpack_low_64(self, other: Self) -> Self {
        lanes::unpack_low_64(self, other)
    }

    #[inline(always)]
    fn clmul<const IMM: i32>(self, other: Self) -> Self {
        lanes::clmul::<IMM>(self, other)
    }
}

/// Several elements of the polynomial-basis `GF(2^128)`, one per 128-bit lane of a register.
///
/// Two under `AVX2`, four under `AVX-512`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
// Needed to make the transmutes below sound.
#[repr(transparent)]
#[must_use]
pub struct PackedGhash128([Ghash128; WIDTH]);

impl PackedGhash128 {
    /// The register holding these elements.
    #[inline]
    #[must_use]
    fn to_vector(self) -> lanes::Reg {
        // SAFETY: the scalar is `repr(transparent)` over `u128`, so the array is `WIDTH`
        // contiguous `u128` values, which is the register's own layout.
        // This type is `repr(transparent)` over that array.
        unsafe { transmute(self) }
    }

    /// The elements held in a register.
    #[inline]
    fn from_vector(vector: lanes::Reg) -> Self {
        // SAFETY: the inverse of the transmute above.
        // Every bit pattern is a valid element, so no value can be out of range.
        unsafe { transmute(vector) }
    }

    /// The same element in every lane.
    #[inline]
    const fn broadcast(value: Ghash128) -> Self {
        Self([value; WIDTH])
    }
}

impl From<Ghash128> for PackedGhash128 {
    #[inline]
    fn from(value: Ghash128) -> Self {
        Self::broadcast(value)
    }
}

impl Add for PackedGhash128 {
    type Output = Self;

    #[inline]
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn add(self, rhs: Self) -> Self {
        // Addition in characteristic 2 is `XOR`, lane by lane.
        Self::from_vector(lanes::xor(self.to_vector(), rhs.to_vector()))
    }
}

impl Sub for PackedGhash128 {
    type Output = Self;

    #[inline]
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn sub(self, rhs: Self) -> Self {
        // Subtraction coincides with addition in characteristic 2.
        self + rhs
    }
}

impl Neg for PackedGhash128 {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        // `-x = x` in characteristic 2.
        self
    }
}

impl Mul for PackedGhash128 {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let (x, y) = (self.to_vector(), rhs.to_vector());

        // The two diagonal half products.
        let low = lanes::clmul::<LOW_BY_LOW>(x, y);
        let high = lanes::clmul::<HIGH_BY_HIGH>(x, y);

        // Karatsuba reaches the middle coefficient with one product instead of two.
        //
        //     middle = (a0 + a1)(b0 + b1) + a0 b0 + a1 b1
        //
        // The scalar kernel takes the schoolbook form instead.
        // A wide carryless multiply has half the throughput of the 128-bit one on Zen 4 and
        // Zen 5, so trading a product for two shuffles and three exclusive ors pays only here.
        //
        // Measured on Zen 5, four lanes: 0.47 ns per element against 0.55 for schoolbook.
        let mixed_x = lanes::xor(x, lanes::swap_halves(x));
        let mixed_y = lanes::xor(y, lanes::swap_halves(y));
        let middle = lanes::xor(
            lanes::xor(low, high),
            lanes::clmul::<LOW_BY_LOW>(mixed_x, mixed_y),
        );

        // Inner fold brings the top limb down, outer fold finishes the reduction.
        Self::from_vector(fold_shifted(low, fold_shifted(middle, high)))
    }
}

impl PrimeCharacteristicRing for PackedGhash128 {
    type PrimeSubfield = Gf2;

    const ZERO: Self = Self::broadcast(Ghash128::ZERO);
    const ONE: Self = Self::broadcast(Ghash128::ONE);
    // The characteristic is 2, so `TWO = ONE + ONE = ZERO`.
    const TWO: Self = Self::broadcast(Ghash128::ZERO);
    // The characteristic is 2, so `NEG_ONE = ONE`.
    const NEG_ONE: Self = Self::broadcast(Ghash128::ONE);

    #[inline]
    fn from_prime_subfield(f: Self::PrimeSubfield) -> Self {
        Self::broadcast(Ghash128::from_prime_subfield(f))
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
        let x = self.to_vector();

        // The cross term of `(p0 + p1 x^64)^2` doubles to zero, so the square has no middle
        // coefficient and the inner fold has nothing to add to.
        let low = lanes::clmul::<LOW_BY_LOW>(x, x);
        let high = lanes::clmul::<HIGH_BY_HIGH>(x, x);

        Self::from_vector(fold_shifted(low, fold_shifted(lanes::zero(), high)))
    }

    #[inline]
    fn dot_product<const N: usize>(u: &[Self; N], v: &[Self; N]) -> Self {
        let (mut low, mut high, mut middle) = (lanes::zero(), lanes::zero(), lanes::zero());

        for (a, b) in u.iter().zip(v) {
            let (x, y) = (a.to_vector(), b.to_vector());

            // Accumulate the three polynomial coefficients without reducing.
            low = lanes::xor(low, lanes::clmul::<LOW_BY_LOW>(x, y));
            high = lanes::xor(high, lanes::clmul::<HIGH_BY_HIGH>(x, y));

            // Karatsuba's third product, from the sum of each operand's halves.
            let mixed_x = lanes::xor(x, lanes::swap_halves(x));
            let mixed_y = lanes::xor(y, lanes::swap_halves(y));
            middle = lanes::xor(middle, lanes::clmul::<LOW_BY_LOW>(mixed_x, mixed_y));
        }

        // Remove the diagonal contributions from the accumulated third product.
        middle = lanes::xor(middle, lanes::xor(low, high));

        // Reduction is linear, so the whole sum folds the modulus once.
        Self::from_vector(fold_shifted(low, fold_shifted(middle, high)))
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

impl_add_assign!(PackedGhash128);
impl_sub_assign!(PackedGhash128);
impl_mul_methods!(PackedGhash128);
ring_sum!(PackedGhash128);
impl_rng!(PackedGhash128);

impl_add_base_field!(PackedGhash128, Ghash128);
impl_sub_base_field!(PackedGhash128, Ghash128);
impl_mul_base_field!(PackedGhash128, Ghash128);
impl_div_methods!(PackedGhash128, Ghash128);
impl_packed_field_div!(PackedGhash128);
impl_sum_prod_base_field!(PackedGhash128, Ghash128);

impl Algebra<Ghash128> for PackedGhash128 {}

impl From<Gf2> for PackedGhash128 {
    /// `GF(2)` is the prime subfield, embedded as `{ZERO, ONE}` in every lane.
    #[inline]
    fn from(x: Gf2) -> Self {
        Self::from_prime_subfield(x)
    }
}

impl_add_base_field!(PackedGhash128, Gf2);
impl_sub_base_field!(PackedGhash128, Gf2);
impl_mul_base_field!(PackedGhash128, Gf2);

impl Algebra<Gf2> for PackedGhash128 {}

impl_packed_value!(PackedGhash128, Ghash128, WIDTH);

// SAFETY: the transparent array satisfies the packed layout contract.
// Arithmetic acts independently on each 128-bit field element.
unsafe impl PackedField for PackedGhash128 {
    type Scalar = Ghash128;
}

// SAFETY: the width is two or four, both powers of two.
unsafe impl PackedFieldPow2 for PackedGhash128 {
    /// # Panics
    /// Panics if the block length does not divide the width.
    #[inline]
    fn interleave(&self, other: Self, block_len: usize) -> (Self, Self) {
        let (a, b) = lanes::interleave(self.to_vector(), other.to_vector(), block_len);
        (Self::from_vector(a), Self::from_vector(b))
    }
}

#[cfg(test)]
mod tests {
    use p3_field::PackedValue;
    use p3_field_testing::test_packed_binary_field;
    use proptest::prelude::*;

    use super::lanes::{self, WIDTH};
    use crate::packed::split::model::Model;
    use crate::packed::split::{
        HIGH_BY_HIGH, HIGH_BY_LOW, LOW_BY_HIGH, LOW_BY_LOW, Lanes, SplitScalar, fold_shifted,
    };
    use crate::{Ghash128, PackedGhash128};

    /// The bit patterns a random search is unlikely to reach.
    ///
    /// Each one drives the fold of the modulus to an extreme:
    ///
    /// ```text
    ///     0            nothing to reduce
    ///     all ones     every coefficient of the product is live
    ///     x^127        the highest degree, so the fold spills furthest
    ///     0x87         the modulus tail itself
    /// ```
    const SPECIAL: [u128; 4] = [0, u128::MAX, 1 << 127, 0x87];

    /// One extreme bit pattern per lane.
    fn specials() -> PackedGhash128 {
        PackedValue::from_fn(|i| Ghash128::from_le_bytes(SPECIAL[i].to_le_bytes()))
    }

    /// The elements a register holds, read back one per lane.
    fn lanes_of(register: lanes::Reg) -> [u128; WIDTH] {
        let mut out = [0u128; WIDTH];

        // SAFETY: the destination is exactly one register of contiguous 128-bit integers.
        //
        // The store is the unaligned form, so the array's own alignment is irrelevant.
        unsafe { lanes::store(out.as_mut_ptr(), register) };

        out
    }

    /// A register holding the given elements, one per lane.
    fn register_of(values: [u128; WIDTH]) -> lanes::Reg {
        // SAFETY: the source is exactly one register of contiguous 128-bit integers.
        unsafe { lanes::load(values.as_ptr()) }
    }

    /// The two 64-bit halves of every lane, exchanged.
    fn swapped_halves(values: [u128; WIDTH]) -> [u128; WIDTH] {
        // Rotating a 128-bit value by half its width is exactly the exchange.
        core::array::from_fn(|i| values[i].rotate_left(64))
    }

    /// Each lane-local operation on the register, against the same operation on the model.
    ///
    /// One assertion per operation, so a mismatch names the intrinsic that disagreed.
    ///
    /// The whole-element interleave is the one left out, since it crosses lanes by design.
    fn lanes_conform(
        a: [u128; WIDTH],
        b: [u128; WIDTH],
        scalar: u128,
    ) -> Result<(), TestCaseError> {
        let (x, y) = (register_of(a), register_of(b));
        let (mx, my) = (Model(a), Model(b));

        prop_assert_eq!(lanes_of(lanes::Reg::zero()), Model::<WIDTH>::zero().0);
        prop_assert_eq!(
            lanes_of(lanes::Reg::broadcast(scalar)),
            Model::<WIDTH>::broadcast(scalar).0
        );
        prop_assert_eq!(lanes_of(lanes::Reg::tail()), Model::<WIDTH>::tail().0);
        prop_assert_eq!(lanes_of(x.xor(y)), mx.xor(my).0);
        prop_assert_eq!(lanes_of(x.unpack_low_64(y)), mx.unpack_low_64(my).0);
        prop_assert_eq!(
            lanes_of(x.clmul::<LOW_BY_LOW>(y)),
            mx.clmul::<LOW_BY_LOW>(my).0
        );
        prop_assert_eq!(
            lanes_of(x.clmul::<HIGH_BY_LOW>(y)),
            mx.clmul::<HIGH_BY_LOW>(my).0
        );
        prop_assert_eq!(
            lanes_of(x.clmul::<LOW_BY_HIGH>(y)),
            mx.clmul::<LOW_BY_HIGH>(my).0
        );
        prop_assert_eq!(
            lanes_of(x.clmul::<HIGH_BY_HIGH>(y)),
            mx.clmul::<HIGH_BY_HIGH>(my).0
        );

        // Not a trait method, so the model cannot supply the expectation.
        //
        // It carries the only hand-written shuffle immediate here, which is why it is pinned.
        prop_assert_eq!(lanes_of(lanes::swap_halves(x)), swapped_halves(a));

        // The two composites built from those operations.
        //
        // A lane-crossing slip therefore shows up here as well as in the primitives above.
        prop_assert_eq!(lanes_of(fold_shifted(x, y)), fold_shifted(mx, my).0);
        prop_assert_eq!(
            lanes_of(SplitScalar::new(scalar).apply(x)),
            SplitScalar::new(scalar).apply(mx).0
        );

        Ok(())
    }

    #[test]
    fn the_register_matches_the_scalar_model_at_the_corners() {
        // Invariant: each operation is lane-local, so distinct lanes must stay distinct.
        //
        // Lane 0 carries the pair under test, so all sixteen combinations are reached.
        //
        // That includes the squaring-shaped case where both operands are the same value.
        //
        // The other lanes rotate through the corners, so a value that crosses a lane
        // boundary lands on a different extreme and shows as a mismatch.
        for (i, &x) in SPECIAL.iter().enumerate() {
            for (j, &y) in SPECIAL.iter().enumerate() {
                let a = core::array::from_fn(|lane| match lane {
                    0 => x,
                    _ => SPECIAL[(i + lane) % SPECIAL.len()],
                });
                let b = core::array::from_fn(|lane| match lane {
                    0 => y,
                    _ => SPECIAL[(j + lane) % SPECIAL.len()],
                });

                // The multiplier walks the corners too, so the split runs from each extreme.
                for scalar in SPECIAL {
                    lanes_conform(a, b, scalar)
                        .unwrap_or_else(|e| panic!("a {x:#x}, b {y:#x}, scalar {scalar:#x}: {e}"));
                }
            }
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(256))]

        #[test]
        fn the_register_matches_the_scalar_model(
            a in prop::array::uniform(any::<u128>()),
            b in prop::array::uniform(any::<u128>()),
            scalar in any::<u128>(),
        ) {
            // The seam the scalar model alone cannot reach.
            //
            // Each intrinsic must be the one the algebra was written against, lane by lane.
            lanes_conform(a, b, scalar)?;
        }
    }

    test_packed_binary_field!(
        crate::PackedGhash128,
        &[crate::PackedGhash128::ZERO],
        &[crate::PackedGhash128::ONE],
        super::specials()
    );
}
