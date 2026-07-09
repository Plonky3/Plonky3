//! Fixed-length (VLS) SVE packing for `Goldilocks`.
//!
//! Where the NEON backend cannot vectorise the multiply — AArch64 NEON has no 64×64→128 integer
//! multiply, so `PackedGoldilocksNeon` extracts its two lanes and runs a dual-lane *scalar* asm
//! reduction — base SVE provides `UMULH`. This backend therefore computes the full 128-bit product
//! across all `WIDTH` lanes at once (`svmul` for the low word, `svmulh` for the high word) and
//! reduces with the standard Goldilocks fold, mirroring the scalar `reduce128`.
//!
//! `Goldilocks` stores a *non-canonical* `u64` (any value; `PartialEq` compares canonical
//! representatives), so `mul`/`square` follow the scalar contract: inputs are arbitrary `u64`, the
//! output is the correct residue reduced to `< 2^64` but not necessarily `< P`.
//!
//! Following the NEON backend, `add`/`sub`/`neg`/`halve` stay scalar per lane; only the multiply is
//! vectorised. `WIDTH` is fixed at compile time (256-bit vector length); the governing predicate
//! covers exactly the first `WIDTH` lanes so loads/stores never run past the backing array.
//!
//! SVE intrinsic names follow ACLE (`core::arch::aarch64`); any not yet in `stdarch` can be
//! expressed with an inline `asm!` block over the same registers.

use alloc::vec::Vec;
use core::arch::aarch64::{self, svbool_t, svuint64_t};
use core::iter::{Product, Sum};
use core::mem::MaybeUninit;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use p3_field::exponentiation::exp_10540996611094048183;
use p3_field::op_assign_macros::{
    impl_add_assign, impl_add_base_field, impl_div_methods, impl_mul_base_field, impl_mul_methods,
    impl_packed_field_div, impl_packed_value, impl_rng, impl_sub_assign, impl_sub_base_field,
    impl_sum_prod_base_field, ring_sum,
};
use p3_field::{
    Algebra, Field, InjectiveMonomial, PackedField, PackedFieldPow2, PackedValue,
    PermutationMonomial, PrimeCharacteristicRing, PrimeField64,
};
use p3_util::reconstitute_from_base;
use rand::distr::{Distribution, StandardUniform};
use rand::{Rng, RngExt};

use crate::Goldilocks;

/// Number of `Goldilocks` elements per packed vector (256-bit vector length).
const WIDTH: usize = 4;

/// `2^64 mod P = 2^32 - 1`. The fold constant for Goldilocks reduction.
const EPSILON: u64 = Goldilocks::ORDER_U64.wrapping_neg();

/// The governing predicate: active on exactly the first `WIDTH` 64-bit lanes.
#[inline(always)]
fn pg() -> svbool_t {
    unsafe { aarch64::svwhilelt_b64_u64(0, WIDTH as u64) }
}

/// Fixed-length SVE implementation of `Goldilocks` arithmetic.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(transparent)]
#[must_use]
pub struct PackedGoldilocksSve(pub [Goldilocks; WIDTH]);

impl PackedGoldilocksSve {
    /// Load the backing array into an SVE vector, reading only the first `WIDTH` lanes.
    #[inline]
    fn to_vec(self) -> svuint64_t {
        debug_assert!(unsafe { aarch64::svcntd() } as usize >= WIDTH);
        // Safety: `Goldilocks` is `repr(transparent)` over `u64`; the predicate activates exactly
        // `WIDTH` lanes so `svld1` reads only within the array.
        unsafe { aarch64::svld1_u64(pg(), self.0.as_ptr().cast::<u64>()) }
    }

    /// Store an SVE vector into a fresh backing array, writing only the first `WIDTH` lanes.
    ///
    /// Any `u64` is a valid (non-canonical) `Goldilocks`, so this needs no range precondition.
    #[inline]
    fn from_vec(vec: svuint64_t) -> Self {
        let mut out = MaybeUninit::<[Goldilocks; WIDTH]>::uninit();
        // Safety: the predicate activates exactly `WIDTH` lanes and `out` is `WIDTH` `u64`s wide, so
        // `svst1` initialises every lane and stays within `out`.
        unsafe {
            aarch64::svst1_u64(pg(), out.as_mut_ptr().cast::<u64>(), vec);
            Self(out.assume_init())
        }
    }

    #[inline]
    const fn broadcast(value: Goldilocks) -> Self {
        Self([value; WIDTH])
    }
}

impl From<Goldilocks> for PackedGoldilocksSve {
    #[inline]
    fn from(value: Goldilocks) -> Self {
        Self::broadcast(value)
    }
}

/// Reduce a per-lane 128-bit product `(hi, lo)` modulo `P`, using `2^64 ≡ 2^32 - 1 (mod P)`.
///
/// Vector form of the scalar `reduce128`. Output is the correct residue in `[0, 2^64)`, not
/// necessarily canonical.
#[inline]
#[must_use]
fn vec_reduce128(hi: svuint64_t, lo: svuint64_t) -> svuint64_t {
    unsafe {
        let epsilon = aarch64::svdup_n_u64(EPSILON);

        let x_hi_hi = aarch64::svlsr_n_u64_x(pg(), hi, 32); // hi >> 32
        let x_hi_lo = aarch64::svand_u64_x(pg(), hi, epsilon); // hi & (2^32 - 1)

        // t0 = lo - x_hi_hi, then subtract EPSILON on the lanes that borrowed. Cannot underflow.
        let t0 = aarch64::svsub_u64_x(pg(), lo, x_hi_hi);
        let borrow = aarch64::svcmplt_u64(pg(), lo, x_hi_hi);
        let t0 = aarch64::svsub_u64_m(borrow, t0, epsilon);

        // t1 = x_hi_lo * EPSILON. Both factors are `< 2^32`, so the low 64 bits are exact.
        let t1 = aarch64::svmul_u64_x(pg(), x_hi_lo, epsilon);

        // t2 = t0 + t1, adding EPSILON back on overflow.
        let t2 = aarch64::svadd_u64_x(pg(), t0, t1);
        let carry = aarch64::svcmplt_u64(pg(), t2, t0);
        aarch64::svadd_u64_m(carry, t2, epsilon)
    }
}

/// Full-width Goldilocks multiplication: `a * b mod P` per lane.
///
/// `svmul` yields the low 64 bits and `svmulh` (UMULH) the high 64 bits of each lane's 128-bit
/// product; `vec_reduce128` folds them. Inputs may be arbitrary `u64`.
#[inline]
#[must_use]
fn vec_mul(a: svuint64_t, b: svuint64_t) -> svuint64_t {
    unsafe {
        let lo = aarch64::svmul_u64_x(pg(), a, b);
        let hi = aarch64::svmulh_u64_x(pg(), a, b);
        vec_reduce128(hi, lo)
    }
}

/// Lane-wise `a + b mod P` for arbitrary (possibly non-canonical) `u64` lanes.
///
/// Vector form of the scalar `Goldilocks::add`: an overflow folds in `EPSILON = 2^64 mod P`, and a
/// second overflow from that fold (only when both inputs exceed `P`) folds again. Both corrections
/// are predicated, so the rare double-overflow lanes cost no branch.
#[inline]
#[must_use]
fn vec_add(a: svuint64_t, b: svuint64_t) -> svuint64_t {
    unsafe {
        let eps = aarch64::svdup_n_u64(EPSILON);
        let sum1 = aarch64::svadd_u64_x(pg(), a, b);
        let over1 = aarch64::svcmplt_u64(pg(), sum1, a);
        let sum2 = aarch64::svadd_u64_m(over1, sum1, eps);
        let over2 = aarch64::svcmplt_u64(pg(), sum2, sum1);
        aarch64::svadd_u64_m(over2, sum2, eps)
    }
}

/// Lane-wise `a - b mod P` for arbitrary `u64` lanes. Mirror of `vec_add` with borrows.
#[inline]
#[must_use]
fn vec_sub(a: svuint64_t, b: svuint64_t) -> svuint64_t {
    unsafe {
        let eps = aarch64::svdup_n_u64(EPSILON);
        let diff1 = aarch64::svsub_u64_x(pg(), a, b);
        let under1 = aarch64::svcmplt_u64(pg(), a, b);
        let diff2 = aarch64::svsub_u64_m(under1, diff1, eps);
        let under2 = aarch64::svcmpgt_u64(pg(), diff2, diff1);
        aarch64::svsub_u64_m(under2, diff2, eps)
    }
}

impl Add for PackedGoldilocksSve {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self::from_vec(vec_add(self.to_vec(), rhs.to_vec()))
    }
}

impl Sub for PackedGoldilocksSve {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self::from_vec(vec_sub(self.to_vec(), rhs.to_vec()))
    }
}

impl Neg for PackedGoldilocksSve {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        // `-a = 0 - a mod P`, reusing the arbitrary-input subtraction.
        let zero = unsafe { aarch64::svdup_n_u64(0) };
        Self::from_vec(vec_sub(zero, self.to_vec()))
    }
}

impl Mul for PackedGoldilocksSve {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self::from_vec(vec_mul(self.to_vec(), rhs.to_vec()))
    }
}

impl_add_assign!(PackedGoldilocksSve);
impl_sub_assign!(PackedGoldilocksSve);
impl_mul_methods!(PackedGoldilocksSve);
ring_sum!(PackedGoldilocksSve);
impl_rng!(PackedGoldilocksSve);

impl PrimeCharacteristicRing for PackedGoldilocksSve {
    type PrimeSubfield = Goldilocks;

    const ZERO: Self = Self::broadcast(Goldilocks::ZERO);
    const ONE: Self = Self::broadcast(Goldilocks::ONE);
    const TWO: Self = Self::broadcast(Goldilocks::TWO);
    const NEG_ONE: Self = Self::broadcast(Goldilocks::NEG_ONE);

    #[inline]
    fn from_prime_subfield(f: Self::PrimeSubfield) -> Self {
        f.into()
    }

    #[inline]
    fn square(&self) -> Self {
        let x = self.to_vec();
        Self::from_vec(vec_mul(x, x))
    }

    #[inline]
    fn halve(&self) -> Self {
        Self(self.0.map(|x| x.halve()))
    }

    #[inline]
    fn zero_vec(len: usize) -> Vec<Self> {
        // Safety: this is a `repr(transparent)` wrapper around `[Goldilocks; WIDTH]`.
        unsafe { reconstitute_from_base(Goldilocks::zero_vec(len * WIDTH)) }
    }
}

impl InjectiveMonomial<7> for PackedGoldilocksSve {}

impl PermutationMonomial<7> for PackedGoldilocksSve {
    fn injective_exp_root_n(&self) -> Self {
        exp_10540996611094048183(*self)
    }
}

impl_add_base_field!(PackedGoldilocksSve, Goldilocks);
impl_sub_base_field!(PackedGoldilocksSve, Goldilocks);
impl_mul_base_field!(PackedGoldilocksSve, Goldilocks);
impl_div_methods!(PackedGoldilocksSve, Goldilocks);
impl_packed_field_div!(PackedGoldilocksSve);
impl_sum_prod_base_field!(PackedGoldilocksSve, Goldilocks);

impl Algebra<Goldilocks> for PackedGoldilocksSve {}

impl_packed_value!(PackedGoldilocksSve, Goldilocks, WIDTH);

unsafe impl PackedField for PackedGoldilocksSve {
    type Scalar = Goldilocks;
}

unsafe impl PackedFieldPow2 for PackedGoldilocksSve {
    #[inline]
    fn interleave(&self, other: Self, block_len: usize) -> (Self, Self) {
        // Interleaving `block_len`-sized chunks of 64-bit lanes is a transpose of `64*block_len`-bit
        // elements: `TRN1`/`TRN2` at the matching element (or quadword) granularity.
        assert!(block_len.is_power_of_two() && block_len <= WIDTH);
        if block_len == WIDTH {
            return (*self, other);
        }

        let a = self.to_vec();
        let b = other.to_vec();
        let (r0, r1) = unsafe {
            match block_len {
                1 => (aarch64::svtrn1_u64(a, b), aarch64::svtrn2_u64(a, b)),
                // `block_len == 2` transposes 128-bit blocks (quadwords).
                _ => (aarch64::svtrn1q_u64(a, b), aarch64::svtrn2q_u64(a, b)),
            }
        };
        (Self::from_vec(r0), Self::from_vec(r1))
    }
}

#[cfg(test)]
mod tests {
    use p3_field_testing::test_packed_field;

    use super::{Goldilocks, PackedGoldilocksSve, WIDTH};

    const SPECIAL_VALS: [Goldilocks; WIDTH] = Goldilocks::new_array([
        0xFFFF_FFFF_0000_0000,
        0xFFFF_FFFF_FFFF_FFFF,
        0x0000_0000_0000_0000,
        0xFFFF_FFFF_0000_0001, // = P, canonicalizes to 0
    ]);

    const ZEROS: PackedGoldilocksSve = PackedGoldilocksSve(Goldilocks::new_array([
        0x0000_0000_0000_0000,
        0xFFFF_FFFF_0000_0001, // = P, canonicalizes to 0
        0x0000_0000_0000_0000,
        0xFFFF_FFFF_0000_0001,
    ]));

    const ONES: PackedGoldilocksSve = PackedGoldilocksSve(Goldilocks::new_array([
        0x0000_0000_0000_0001,
        0xFFFF_FFFF_0000_0002, // = P + 1, canonicalizes to 1
        0x0000_0000_0000_0001,
        0xFFFF_FFFF_0000_0002,
    ]));

    test_packed_field!(
        crate::PackedGoldilocksSve,
        &[super::ZEROS],
        &[super::ONES],
        crate::PackedGoldilocksSve(super::SPECIAL_VALS)
    );
}
