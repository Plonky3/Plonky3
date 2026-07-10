//! Fixed-length (VLS) SVE packing for `Goldilocks`, built on inline `asm!`.
//!
//! Where the NEON backend cannot vectorise the multiply — AArch64 NEON has no 64×64→128 integer
//! multiply, so `PackedGoldilocksNeon` extracts its two lanes and runs a dual-lane *scalar* asm
//! reduction — base SVE provides `UMULH`. This backend therefore computes the full 128-bit product
//! across all `WIDTH` lanes at once (`mul` for the low word, `umulh` for the high word) and reduces
//! with the standard Goldilocks fold, mirroring the scalar `reduce128`.
//!
//! Each kernel is a self-contained `asm!` block using only base **SVE (SVE1)** instructions, so it
//! runs on Neoverse V1 (Graviton3) as well as SVE2 parts and requires no nightly feature flag: the
//! backend builds on stable Rust whenever `-C target-feature=+sve` is set.
//!
//! `Goldilocks` stores a *non-canonical* `u64` (any value; `PartialEq` compares canonical
//! representatives), so `mul`/`square` follow the scalar contract: inputs are arbitrary `u64`, the
//! output is the correct residue reduced to `< 2^64` but not necessarily `< P`.
//!
//! `WIDTH` matches the target vector length (256-bit by default, 128-bit under SVE2); the governing
//! predicate covers exactly the first `WIDTH` lanes so loads/stores never run past the backing array.

use alloc::vec::Vec;
use core::arch::asm;
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

/// Number of `Goldilocks` elements per packed vector.
///
/// 4 lanes (256-bit, Graviton3 / Neoverse V1) by default, or 2 lanes (128-bit, Graviton4 /
/// Neoverse V2) when SVE2 is enabled.
#[cfg(not(target_feature = "sve2"))]
const WIDTH: usize = 4;
#[cfg(target_feature = "sve2")]
const WIDTH: usize = 2;

/// `2^64 mod P = 2^32 - 1`. The fold constant for Goldilocks reduction (and the low-32-bit mask).
const EPSILON: u64 = Goldilocks::ORDER_U64.wrapping_neg();

/// Hardware vector length expressed as a count of 64-bit doublewords (`VL / 64`).
#[inline(always)]
fn hw_vl_doublewords() -> u64 {
    let n: u64;
    // SAFETY: `cntd` reads the (process-constant) vector length and has no other effect.
    unsafe {
        asm!("cntd {n}", n = out(reg) n, options(pure, nomem, nostack, preserves_flags));
    }
    n
}

/// Debug-only assertion that the hardware vector length covers `WIDTH` 64-bit lanes.
#[inline(always)]
fn debug_assert_vl() {
    debug_assert!(
        hw_vl_doublewords() >= WIDTH as u64,
        "SVE vector length is below WIDTH * 64 bits"
    );
}

/// Fixed-length SVE implementation of `Goldilocks` arithmetic.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(transparent)]
#[must_use]
pub struct PackedGoldilocksSve(pub [Goldilocks; WIDTH]);

impl PackedGoldilocksSve {
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

// -----------------------------------------------------------------------------
// Vector kernels. Each loads its operands, computes over the first `WIDTH` 64-bit
// lanes (predicate `whilelt p0.d, xzr, WIDTH`), and stores the result. Only base
// SVE1 instructions are used, so `mul`/`umulh` take the predicated, destructive
// form and `movprfx` is used where a non-destructive result is needed.
// -----------------------------------------------------------------------------

/// Full-width Goldilocks multiplication: `a * b mod P` per lane, folded with `2^64 ≡ 2^32 - 1`.
///
/// `mul` yields the low 64 bits and `umulh` the high 64 bits of each lane's 128-bit product; the
/// remainder of the block is the vector form of the scalar `reduce128`. Inputs may be arbitrary
/// `u64`; the output is the correct residue in `[0, 2^64)`, not necessarily canonical.
#[inline]
fn vec_mul(a: &PackedGoldilocksSve, b: &PackedGoldilocksSve) -> PackedGoldilocksSve {
    debug_assert_vl();
    let mut out = MaybeUninit::<[Goldilocks; WIDTH]>::uninit();
    // SAFETY: the operand pointers span `WIDTH` contiguous `u64`s (`Goldilocks` is
    // `repr(transparent)` over `u64`), and the predicate activates exactly `WIDTH` lanes, so the
    // load reads and the store writes only within the arrays. Every lane of `out` is written, so
    // `assume_init` is sound.
    unsafe {
        asm!(
            "whilelt p0.d, xzr, {w}",
            "dup z2.d, {e}",
            "ld1d {{z0.d}}, p0/z, [{a}]",
            "ld1d {{z1.d}}, p0/z, [{b}]",
            "movprfx z3, z0",
            "umulh z3.d, p0/m, z3.d, z1.d", // hi = high(a * b)
            "mul z0.d, p0/m, z0.d, z1.d",   // lo = low(a * b)
            "lsr z4.d, z3.d, #32",          // x_hi_hi = hi >> 32
            "and z5.d, z3.d, z2.d",         // x_hi_lo = hi & (2^32 - 1)
            "cmphi p1.d, p0/z, z4.d, z0.d", // borrow = x_hi_hi > lo
            "sub z0.d, z0.d, z4.d",         // t0 = lo - x_hi_hi
            "sub z0.d, p1/m, z0.d, z2.d",   // t0 -= EPSILON on borrow
            "mul z5.d, p0/m, z5.d, z2.d",   // t1 = x_hi_lo * EPSILON
            "add z6.d, z0.d, z5.d",         // t2 = t0 + t1
            "cmphi p2.d, p0/z, z0.d, z6.d", // carry = t0 > t2
            "add z6.d, p2/m, z6.d, z2.d",   // + EPSILON on carry
            "st1d {{z6.d}}, p0, [{o}]",
            w = in(reg) WIDTH as u64,
            e = in(reg) EPSILON,
            a = in(reg) a.0.as_ptr().cast::<u64>(),
            b = in(reg) b.0.as_ptr().cast::<u64>(),
            o = in(reg) out.as_mut_ptr().cast::<u64>(),
            out("z0") _, out("z1") _, out("z2") _, out("z3") _, out("z4") _, out("z5") _, out("z6") _,
            out("p0") _, out("p1") _, out("p2") _,
            options(nostack),
        );
        PackedGoldilocksSve(out.assume_init())
    }
}

/// Lane-wise `a + b mod P` for arbitrary (possibly non-canonical) `u64` lanes.
///
/// Vector form of the scalar `Goldilocks::add`: an overflow folds in `EPSILON = 2^64 mod P`, and a
/// second overflow from that fold (only when both inputs exceed `P`) folds again. Both corrections
/// are predicated, so the rare double-overflow lanes cost no branch.
#[inline]
fn vec_add(a: &PackedGoldilocksSve, b: &PackedGoldilocksSve) -> PackedGoldilocksSve {
    debug_assert_vl();
    let mut out = MaybeUninit::<[Goldilocks; WIDTH]>::uninit();
    // SAFETY: the predicate bounds every access to the `WIDTH`-lane arrays and all lanes of `out`
    // are written.
    unsafe {
        asm!(
            "whilelt p0.d, xzr, {w}",
            "dup z2.d, {e}",
            "ld1d {{z0.d}}, p0/z, [{a}]",
            "ld1d {{z1.d}}, p0/z, [{b}]",
            "add z3.d, z0.d, z1.d",         // sum1 = a + b
            "cmphi p1.d, p0/z, z0.d, z3.d", // over1 = sum1 < a
            "movprfx z4, z3",
            "add z4.d, p1/m, z4.d, z2.d",   // sum2 = sum1 + EPSILON on over1
            "cmphi p2.d, p0/z, z3.d, z4.d", // over2 = sum2 < sum1
            "add z4.d, p2/m, z4.d, z2.d",   // + EPSILON on over2
            "st1d {{z4.d}}, p0, [{o}]",
            w = in(reg) WIDTH as u64,
            e = in(reg) EPSILON,
            a = in(reg) a.0.as_ptr().cast::<u64>(),
            b = in(reg) b.0.as_ptr().cast::<u64>(),
            o = in(reg) out.as_mut_ptr().cast::<u64>(),
            out("z0") _, out("z1") _, out("z2") _, out("z3") _, out("z4") _,
            out("p0") _, out("p1") _, out("p2") _,
            options(nostack),
        );
        PackedGoldilocksSve(out.assume_init())
    }
}

/// Lane-wise `a - b mod P` for arbitrary `u64` lanes. Mirror of `vec_add` with borrows.
#[inline]
fn vec_sub(a: &PackedGoldilocksSve, b: &PackedGoldilocksSve) -> PackedGoldilocksSve {
    debug_assert_vl();
    let mut out = MaybeUninit::<[Goldilocks; WIDTH]>::uninit();
    // SAFETY: the predicate bounds every access to the `WIDTH`-lane arrays and all lanes of `out`
    // are written.
    unsafe {
        asm!(
            "whilelt p0.d, xzr, {w}",
            "dup z2.d, {e}",
            "ld1d {{z0.d}}, p0/z, [{a}]",
            "ld1d {{z1.d}}, p0/z, [{b}]",
            "sub z3.d, z0.d, z1.d",         // diff1 = a - b
            "cmphi p1.d, p0/z, z1.d, z0.d", // under1 = a < b
            "movprfx z4, z3",
            "sub z4.d, p1/m, z4.d, z2.d",   // diff2 = diff1 - EPSILON on under1
            "cmphi p2.d, p0/z, z4.d, z3.d", // under2 = diff2 > diff1
            "sub z4.d, p2/m, z4.d, z2.d",   // - EPSILON on under2
            "st1d {{z4.d}}, p0, [{o}]",
            w = in(reg) WIDTH as u64,
            e = in(reg) EPSILON,
            a = in(reg) a.0.as_ptr().cast::<u64>(),
            b = in(reg) b.0.as_ptr().cast::<u64>(),
            o = in(reg) out.as_mut_ptr().cast::<u64>(),
            out("z0") _, out("z1") _, out("z2") _, out("z3") _, out("z4") _,
            out("p0") _, out("p1") _, out("p2") _,
            options(nostack),
        );
        PackedGoldilocksSve(out.assume_init())
    }
}

impl Add for PackedGoldilocksSve {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        vec_add(&self, &rhs)
    }
}

impl Sub for PackedGoldilocksSve {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        vec_sub(&self, &rhs)
    }
}

impl Neg for PackedGoldilocksSve {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        // `-a = 0 - a mod P`, reusing the arbitrary-input subtraction.
        vec_sub(&Self::ZERO, &self)
    }
}

impl Mul for PackedGoldilocksSve {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        vec_mul(&self, &rhs)
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
        vec_mul(self, self)
    }

    #[inline]
    fn halve(&self) -> Self {
        Self(self.0.map(|x| x.halve()))
    }

    #[inline]
    fn zero_vec(len: usize) -> Vec<Self> {
        // SAFETY: this is a `repr(transparent)` wrapper around `[Goldilocks; WIDTH]`.
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

/// Interleave `a` and `b` at 64-bit element granularity via `TRN1`/`TRN2` (base SVE); the 128-bit
/// granule would need F64MM and is handled by the scalar path in `interleave`.
#[inline]
fn interleave_trn_d(
    a: &PackedGoldilocksSve,
    b: &PackedGoldilocksSve,
) -> (PackedGoldilocksSve, PackedGoldilocksSve) {
    debug_assert_vl();
    let mut r0 = MaybeUninit::<[Goldilocks; WIDTH]>::uninit();
    let mut r1 = MaybeUninit::<[Goldilocks; WIDTH]>::uninit();
    // SAFETY: the predicate bounds every access to the `WIDTH`-lane arrays; `trn` only permutes the
    // active lanes, and both outputs are fully written.
    unsafe {
        asm!(
            "whilelt p0.d, xzr, {w}",
            "ld1d {{z0.d}}, p0/z, [{a}]",
            "ld1d {{z1.d}}, p0/z, [{b}]",
            "trn1 z2.d, z0.d, z1.d",
            "trn2 z3.d, z0.d, z1.d",
            "st1d {{z2.d}}, p0, [{r0}]",
            "st1d {{z3.d}}, p0, [{r1}]",
            w = in(reg) WIDTH as u64,
            a = in(reg) a.0.as_ptr().cast::<u64>(),
            b = in(reg) b.0.as_ptr().cast::<u64>(),
            r0 = in(reg) r0.as_mut_ptr().cast::<u64>(),
            r1 = in(reg) r1.as_mut_ptr().cast::<u64>(),
            out("z0") _, out("z1") _, out("z2") _, out("z3") _, out("p0") _,
            options(nostack),
        );
        (
            PackedGoldilocksSve(r0.assume_init()),
            PackedGoldilocksSve(r1.assume_init()),
        )
    }
}

unsafe impl PackedFieldPow2 for PackedGoldilocksSve {
    #[inline]
    fn interleave(&self, other: Self, block_len: usize) -> (Self, Self) {
        // Interleaving `block_len`-sized chunks of lanes is a `TRN1`/`TRN2` of `64*block_len`-bit
        // elements. The 64-bit granule uses base-SVE `trn`; the 128-bit granule (`block_len == 2`)
        // would need the optional F64MM extension, so it is permuted directly.
        assert!(block_len.is_power_of_two() && block_len <= WIDTH);
        if block_len == WIDTH {
            return (*self, other);
        }
        match block_len {
            1 => interleave_trn_d(self, &other),
            _ => {
                let a = self.0;
                let b = other.0;
                let mut r0 = [Goldilocks::ZERO; WIDTH];
                let mut r1 = [Goldilocks::ZERO; WIDTH];
                // `TRN1` gathers even-indexed blocks, `TRN2` odd-indexed blocks, taking each pair.
                let blocks = WIDTH / block_len;
                for i in 0..blocks / 2 {
                    let even = 2 * i;
                    let odd = 2 * i + 1;
                    let lo = 2 * i * block_len;
                    let hi = (2 * i + 1) * block_len;
                    r0[lo..lo + block_len]
                        .copy_from_slice(&a[even * block_len..(even + 1) * block_len]);
                    r0[hi..hi + block_len]
                        .copy_from_slice(&b[even * block_len..(even + 1) * block_len]);
                    r1[lo..lo + block_len]
                        .copy_from_slice(&a[odd * block_len..(odd + 1) * block_len]);
                    r1[hi..hi + block_len]
                        .copy_from_slice(&b[odd * block_len..(odd + 1) * block_len]);
                }
                (Self(r0), Self(r1))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use p3_field_testing::test_packed_field;

    use super::{Goldilocks, PackedGoldilocksSve, WIDTH};

    #[cfg(not(target_feature = "sve2"))]
    const SPECIAL_VALS: [Goldilocks; WIDTH] = Goldilocks::new_array([
        0xFFFF_FFFF_0000_0000,
        0xFFFF_FFFF_FFFF_FFFF,
        0x0000_0000_0000_0000,
        0xFFFF_FFFF_0000_0001, // = P, canonicalizes to 0
    ]);
    #[cfg(target_feature = "sve2")]
    const SPECIAL_VALS: [Goldilocks; WIDTH] =
        Goldilocks::new_array([0xFFFF_FFFF_0000_0000, 0xFFFF_FFFF_FFFF_FFFF]);

    #[cfg(not(target_feature = "sve2"))]
    const ZEROS: PackedGoldilocksSve = PackedGoldilocksSve(Goldilocks::new_array([
        0x0000_0000_0000_0000,
        0xFFFF_FFFF_0000_0001, // = P, canonicalizes to 0
        0x0000_0000_0000_0000,
        0xFFFF_FFFF_0000_0001,
    ]));
    #[cfg(target_feature = "sve2")]
    const ZEROS: PackedGoldilocksSve = PackedGoldilocksSve(Goldilocks::new_array([
        0x0000_0000_0000_0000,
        0xFFFF_FFFF_0000_0001, // = P, canonicalizes to 0
    ]));

    #[cfg(not(target_feature = "sve2"))]
    const ONES: PackedGoldilocksSve = PackedGoldilocksSve(Goldilocks::new_array([
        0x0000_0000_0000_0001,
        0xFFFF_FFFF_0000_0002, // = P + 1, canonicalizes to 1
        0x0000_0000_0000_0001,
        0xFFFF_FFFF_0000_0002,
    ]));
    #[cfg(target_feature = "sve2")]
    const ONES: PackedGoldilocksSve = PackedGoldilocksSve(Goldilocks::new_array([
        0x0000_0000_0000_0001,
        0xFFFF_FFFF_0000_0002, // = P + 1, canonicalizes to 1
    ]));

    test_packed_field!(
        crate::PackedGoldilocksSve,
        &[super::ZEROS],
        &[super::ONES],
        crate::PackedGoldilocksSve(super::SPECIAL_VALS)
    );
}
