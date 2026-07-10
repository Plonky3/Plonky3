//! Fixed-length (VLS) SVE packing for `MontyField31`, built on inline `asm!`.
//!
//! SVE vector registers are *sizeless*: they cannot be struct fields, array elements, or `Sized`.
//! The packed type therefore keeps its data in a plain `[MontyField31; WIDTH]` array (exactly like
//! the NEON and WASM backends), and each arithmetic kernel is a self-contained `asm!` block that
//! loads the operands into `z` registers (`ld1w`), computes, and stores the result (`st1w`).
//!
//! The kernels use only base **SVE (SVE1)** instructions, so they run on Neoverse V1 (Graviton3) as
//! well as SVE2 parts. Because the intrinsics are expressed as assembly rather than
//! `core::arch::aarch64` intrinsics, no nightly feature flag is required: the backend builds on
//! stable Rust whenever `-C target-feature=+sve` is set.
//!
//! `WIDTH` is fixed at compile time. The governing predicate covers exactly the first `WIDTH` lanes
//! (`whilelt p, xzr, WIDTH`), so loads and stores never touch memory past the backing array even
//! when the hardware vector length exceeds `WIDTH * 32` bits. Correctness still requires the
//! hardware vector length to be *at least* `WIDTH * 32` bits; this is checked once per kernel via
//! `debug_assert` (see [`debug_assert_vl`]).

use alloc::vec::Vec;
use core::arch::asm;
use core::iter::{Product, Sum};
use core::mem::MaybeUninit;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use p3_field::op_assign_macros::{
    impl_add_assign, impl_add_base_field, impl_div_methods, impl_mul_base_field, impl_mul_methods,
    impl_packed_field_div, impl_packed_value, impl_rng, impl_sub_assign, impl_sub_base_field,
    impl_sum_prod_base_field, ring_sum,
};
use p3_field::{
    Algebra, Field, InjectiveMonomial, PackedField, PackedFieldPow2, PackedValue,
    PermutationMonomial, PrimeCharacteristicRing,
};
use p3_util::reconstitute_from_base;
use rand::distr::{Distribution, StandardUniform};
use rand::{Rng, RngExt};

use crate::{FieldParameters, MontyField31, PackedMontyParameters, RelativelyPrimePower};

/// Number of `MontyField31` elements per packed vector.
///
/// Fixed at 8, i.e. a 256-bit vector length (Graviton3 / Neoverse V1). For a 512-bit part set this
/// to 16; the runtime guard in [`debug_assert_vl`] ensures the hardware vector length is large
/// enough.
const WIDTH: usize = 8;

/// Hardware vector length expressed as a count of 32-bit words (`VL / 32`).
#[inline(always)]
fn hw_vl_words() -> u64 {
    let n: u64;
    // SAFETY: `cntw` reads the (process-constant) vector length and has no other effect.
    unsafe {
        asm!("cntw {n}", n = out(reg) n, options(pure, nomem, nostack, preserves_flags));
    }
    n
}

/// Debug-only assertion that the hardware vector length covers `WIDTH` 32-bit lanes.
///
/// A machine whose vector length is below `WIDTH * 32` bits would leave the high predicate lanes
/// outside the register and silently drop elements; this catches that in debug builds.
#[inline(always)]
fn debug_assert_vl() {
    debug_assert!(
        hw_vl_words() >= WIDTH as u64,
        "SVE vector length is below WIDTH * 32 bits"
    );
}

/// Fixed-length SVE implementation of `MontyField31` arithmetic.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(transparent)]
#[must_use]
pub struct PackedMontyField31Sve<PMP: PackedMontyParameters>(pub [MontyField31<PMP>; WIDTH]);

impl<PMP: PackedMontyParameters> PackedMontyField31Sve<PMP> {
    /// Copy `value` to all positions in a packed vector.
    #[inline]
    const fn broadcast(value: MontyField31<PMP>) -> Self {
        Self([value; WIDTH])
    }

    /// Fused DIF butterfly for the forward FFT: computes `(x + y, (x - y) * roots)`.
    ///
    /// The `x - y` term is left unreduced: as `x, y` are in `[0, P)`, the raw wrapping subtraction
    /// reinterpreted as signed lies in `(-P, P)`, which is exactly the input range accepted by the
    /// signed Montgomery multiply. This skips the modular reduction on `x - y`.
    #[inline]
    pub(crate) fn forward_butterfly(self, y: Self, roots: Self) -> (Self, Self) {
        let sum = vec_add(&self, &y);
        let diff = vec_sub_raw(&self, &y);
        let product = vec_mul_signed(&diff, &roots);
        (sum, product)
    }
}

impl<PMP: PackedMontyParameters> From<MontyField31<PMP>> for PackedMontyField31Sve<PMP> {
    #[inline]
    fn from(value: MontyField31<PMP>) -> Self {
        Self::broadcast(value)
    }
}

// -----------------------------------------------------------------------------
// Vector kernels. Each loads its operands, computes over the first `WIDTH` lanes
// (predicate `whilelt p0, xzr, WIDTH`), and stores the result. Only base SVE1
// instructions are used, so `mul`/`umulh`/`smulh` take the predicated,
// destructive form and `movprfx` is used where a non-destructive result is
// needed.
// -----------------------------------------------------------------------------

/// Canonical modular addition: given `a, b` in `[0, P)` returns `(a + b) mod P` in `[0, P)`.
///
/// `t = a + b`, `u = t - P`; `min(t, u)` picks `t` when `t < P` (then `u` wrapped high) and `u`
/// otherwise. Mirrors `uint32x4_mod_add`.
#[inline]
fn vec_add<PMP: PackedMontyParameters>(
    a: &PackedMontyField31Sve<PMP>,
    b: &PackedMontyField31Sve<PMP>,
) -> PackedMontyField31Sve<PMP> {
    debug_assert_vl();
    let mut out = MaybeUninit::<[MontyField31<PMP>; WIDTH]>::uninit();
    // SAFETY: the operand pointers span `WIDTH` contiguous `u32`s (`MontyField31` is
    // `repr(transparent)` over `u32`), and the predicate activates exactly `WIDTH` lanes, so the
    // load reads and the store writes only within the arrays. Every lane of `out` is written, so
    // `assume_init` is sound; canonical inputs yield a canonical result.
    unsafe {
        asm!(
            "whilelt p0.s, xzr, {w}",
            "dup z2.s, {p:w}",
            "ld1w {{z0.s}}, p0/z, [{a}]",
            "ld1w {{z1.s}}, p0/z, [{b}]",
            "add z0.s, z0.s, z1.s",
            "sub z1.s, z0.s, z2.s",
            "umin z0.s, p0/m, z0.s, z1.s",
            "st1w {{z0.s}}, p0, [{o}]",
            w = in(reg) WIDTH as u64,
            p = in(reg) PMP::PRIME,
            a = in(reg) a.0.as_ptr().cast::<u32>(),
            b = in(reg) b.0.as_ptr().cast::<u32>(),
            o = in(reg) out.as_mut_ptr().cast::<u32>(),
            out("z0") _, out("z1") _, out("z2") _, out("p0") _,
            options(nostack),
        );
        PackedMontyField31Sve(out.assume_init())
    }
}

/// Canonical modular subtraction: given `a, b` in `[0, P)` returns `(a - b) mod P` in `[0, P)`.
///
/// Mirrors `uint32x4_mod_sub`.
#[inline]
fn vec_sub<PMP: PackedMontyParameters>(
    a: &PackedMontyField31Sve<PMP>,
    b: &PackedMontyField31Sve<PMP>,
) -> PackedMontyField31Sve<PMP> {
    debug_assert_vl();
    let mut out = MaybeUninit::<[MontyField31<PMP>; WIDTH]>::uninit();
    // SAFETY: as in `vec_add`; the predicate bounds every access to the `WIDTH`-lane arrays and all
    // lanes of `out` are written.
    unsafe {
        asm!(
            "whilelt p0.s, xzr, {w}",
            "dup z2.s, {p:w}",
            "ld1w {{z0.s}}, p0/z, [{a}]",
            "ld1w {{z1.s}}, p0/z, [{b}]",
            "sub z0.s, z0.s, z1.s",
            "add z1.s, z0.s, z2.s",
            "umin z0.s, p0/m, z0.s, z1.s",
            "st1w {{z0.s}}, p0, [{o}]",
            w = in(reg) WIDTH as u64,
            p = in(reg) PMP::PRIME,
            a = in(reg) a.0.as_ptr().cast::<u32>(),
            b = in(reg) b.0.as_ptr().cast::<u32>(),
            o = in(reg) out.as_mut_ptr().cast::<u32>(),
            out("z0") _, out("z1") _, out("z2") _, out("p0") _,
            options(nostack),
        );
        PackedMontyField31Sve(out.assume_init())
    }
}

/// Unreduced signed subtraction used by the fused butterfly: returns the raw wrapping `a - b` bit
/// pattern, which for canonical `a, b` is the two's-complement of a value in `(-P, P)`.
#[inline]
#[must_use]
fn vec_sub_raw<PMP: PackedMontyParameters>(
    a: &PackedMontyField31Sve<PMP>,
    b: &PackedMontyField31Sve<PMP>,
) -> [u32; WIDTH] {
    debug_assert_vl();
    let mut out = MaybeUninit::<[u32; WIDTH]>::uninit();
    // SAFETY: the predicate bounds every access to the `WIDTH`-lane arrays and all lanes of `out`
    // are written.
    unsafe {
        asm!(
            "whilelt p0.s, xzr, {w}",
            "ld1w {{z0.s}}, p0/z, [{a}]",
            "ld1w {{z1.s}}, p0/z, [{b}]",
            "sub z0.s, z0.s, z1.s",
            "st1w {{z0.s}}, p0, [{o}]",
            w = in(reg) WIDTH as u64,
            a = in(reg) a.0.as_ptr().cast::<u32>(),
            b = in(reg) b.0.as_ptr().cast::<u32>(),
            o = in(reg) out.as_mut_ptr().cast::<u32>(),
            out("z0") _, out("z1") _, out("p0") _,
            options(nostack),
        );
        out.assume_init()
    }
}

/// Negate a vector in canonical form: returns `0` where `a == 0` and `P - a` otherwise.
#[inline]
fn vec_neg<PMP: PackedMontyParameters>(
    a: &PackedMontyField31Sve<PMP>,
) -> PackedMontyField31Sve<PMP> {
    debug_assert_vl();
    let mut out = MaybeUninit::<[MontyField31<PMP>; WIDTH]>::uninit();
    // SAFETY: the predicate bounds every access to the `WIDTH`-lane arrays and all lanes of `out`
    // are written; `P - a` and the zeroed lanes are canonical.
    unsafe {
        asm!(
            "whilelt p0.s, xzr, {w}",
            "dup z2.s, {p:w}",
            "ld1w {{z0.s}}, p0/z, [{a}]",
            "sub z1.s, z2.s, z0.s",
            "cmpeq p1.s, p0/z, z0.s, #0",
            "mov z1.s, p1/m, #0",
            "st1w {{z1.s}}, p0, [{o}]",
            w = in(reg) WIDTH as u64,
            p = in(reg) PMP::PRIME,
            a = in(reg) a.0.as_ptr().cast::<u32>(),
            o = in(reg) out.as_mut_ptr().cast::<u32>(),
            out("z0") _, out("z1") _, out("z2") _, out("p0") _, out("p1") _,
            options(nostack),
        );
        PackedMontyField31Sve(out.assume_init())
    }
}

/// Montgomery multiplication of two canonical `MontyField31` vectors.
///
/// Given `a, b` in `[0, P)`, returns `a * b * R^{-1} mod P` in `[0, P)`, where `R = 2^32`. This is
/// the vector form of the scalar `monty_reduce` applied to the 64-bit product `x = a * b`:
///
/// ```text
///   t      = (x mod 2^32) * MONTY_MU     mod 2^32
///   x_hi   = x >> 32                                   (via UMULH)
///   u_hi   = (t * P) >> 32                             (via UMULH)
///   result = x_hi - u_hi  (+ P if it borrowed)
/// ```
///
/// The low words of `x` and `t * P` are congruent mod `2^32` by definition of `MONTY_MU`, so they
/// cancel and only the high words are needed. `x = a * b < P^2 < 2^32 * P`, satisfying the
/// `monty_reduce` precondition.
#[inline]
fn vec_mul<PMP: PackedMontyParameters>(
    a: &PackedMontyField31Sve<PMP>,
    b: &PackedMontyField31Sve<PMP>,
) -> PackedMontyField31Sve<PMP> {
    debug_assert_vl();
    let mut out = MaybeUninit::<[MontyField31<PMP>; WIDTH]>::uninit();
    // SAFETY: the predicate bounds every access to the `WIDTH`-lane arrays and all lanes of `out`
    // are written; canonical inputs yield a canonical result.
    unsafe {
        asm!(
            "whilelt p0.s, xzr, {w}",
            "dup z2.s, {p:w}",
            "dup z3.s, {mu:w}",
            "ld1w {{z0.s}}, p0/z, [{a}]",
            "ld1w {{z1.s}}, p0/z, [{b}]",
            "movprfx z4, z0",
            "umulh z4.s, p0/m, z4.s, z1.s", // x_hi = high(a * b)
            "mul z0.s, p0/m, z0.s, z1.s",   // x_lo = low(a * b)
            "mul z0.s, p0/m, z0.s, z3.s",   // t = low(x_lo * MONTY_MU)
            "umulh z0.s, p0/m, z0.s, z2.s", // u_hi = high(t * P)
            "cmphi p1.s, p0/z, z0.s, z4.s", // borrow = u_hi > x_hi
            "sub z4.s, z4.s, z0.s",         // x_hi - u_hi
            "add z4.s, p1/m, z4.s, z2.s",   // + P on the lanes that borrowed
            "st1w {{z4.s}}, p0, [{o}]",
            w = in(reg) WIDTH as u64,
            p = in(reg) PMP::PRIME,
            mu = in(reg) PMP::MONTY_MU,
            a = in(reg) a.0.as_ptr().cast::<u32>(),
            b = in(reg) b.0.as_ptr().cast::<u32>(),
            o = in(reg) out.as_mut_ptr().cast::<u32>(),
            out("z0") _, out("z1") _, out("z2") _, out("z3") _, out("z4") _,
            out("p0") _, out("p1") _,
            options(nostack),
        );
        PackedMontyField31Sve(out.assume_init())
    }
}

/// Signed Montgomery multiplication: `lhs, rhs` are signed values in `[-P, P]`, output is canonical
/// `[0, P)`. Used by the fused forward butterfly, which feeds an unreduced `x - y`.
///
/// `SMULH` gives the true high word of each signed product, so `D = ⌊C/2³²⌋ − ⌊Q·P/2³²⌋` needs no
/// doubling/halving correction (the low words of `C = lhs·rhs` and `Q·P` are congruent mod `2³²`, so
/// they cancel exactly). `Q = (lhs·rhs·MONTY_MU) mod 2³²`.
#[inline]
fn vec_mul_signed<PMP: PackedMontyParameters>(
    lhs: &[u32; WIDTH],
    rhs: &PackedMontyField31Sve<PMP>,
) -> PackedMontyField31Sve<PMP> {
    debug_assert_vl();
    let mut out = MaybeUninit::<[MontyField31<PMP>; WIDTH]>::uninit();
    // SAFETY: the predicate bounds every access to the `WIDTH`-lane arrays and all lanes of `out`
    // are written; the reduced result lands in `[0, P)`.
    unsafe {
        asm!(
            "whilelt p0.s, xzr, {w}",
            "dup z2.s, {p:w}",
            "dup z3.s, {mu:w}",
            "ld1w {{z0.s}}, p0/z, [{a}]", // lhs
            "ld1w {{z1.s}}, p0/z, [{b}]", // rhs
            "movprfx z4, z0",
            "smulh z4.s, p0/m, z4.s, z1.s", // c_hi = high(lhs * rhs), signed
            "mul z1.s, p0/m, z1.s, z3.s",   // mu_rhs = low(rhs * MONTY_MU)
            "mul z0.s, p0/m, z0.s, z1.s",   // q = low(lhs * mu_rhs)
            "smulh z0.s, p0/m, z0.s, z2.s", // qp_hi = high(q * P), signed
            "cmpgt p1.s, p0/z, z0.s, z4.s", // underflow = qp_hi > c_hi (signed)
            "sub z4.s, z4.s, z0.s",         // c_hi - qp_hi
            "add z4.s, p1/m, z4.s, z2.s",   // + P on the lanes that underflowed
            "st1w {{z4.s}}, p0, [{o}]",
            w = in(reg) WIDTH as u64,
            p = in(reg) PMP::PRIME,
            mu = in(reg) PMP::MONTY_MU,
            a = in(reg) lhs.as_ptr(),
            b = in(reg) rhs.0.as_ptr().cast::<u32>(),
            o = in(reg) out.as_mut_ptr().cast::<u32>(),
            out("z0") _, out("z1") _, out("z2") _, out("z3") _, out("z4") _,
            out("p0") _, out("p1") _,
            options(nostack),
        );
        PackedMontyField31Sve(out.assume_init())
    }
}

impl<PMP: PackedMontyParameters> Add for PackedMontyField31Sve<PMP> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        vec_add(&self, &rhs)
    }
}

impl<PMP: PackedMontyParameters> Sub for PackedMontyField31Sve<PMP> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        vec_sub(&self, &rhs)
    }
}

impl<PMP: PackedMontyParameters> Neg for PackedMontyField31Sve<PMP> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        vec_neg(&self)
    }
}

impl<PMP: PackedMontyParameters> Mul for PackedMontyField31Sve<PMP> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        vec_mul(&self, &rhs)
    }
}

impl_add_assign!(PackedMontyField31Sve, (PackedMontyParameters, PMP));
impl_sub_assign!(PackedMontyField31Sve, (PackedMontyParameters, PMP));
impl_mul_methods!(PackedMontyField31Sve, (FieldParameters, FP));
ring_sum!(PackedMontyField31Sve, (FieldParameters, FP));
impl_rng!(PackedMontyField31Sve, (PackedMontyParameters, PMP));

impl<FP: FieldParameters> PrimeCharacteristicRing for PackedMontyField31Sve<FP> {
    type PrimeSubfield = MontyField31<FP>;

    const ZERO: Self = Self::broadcast(MontyField31::ZERO);
    const ONE: Self = Self::broadcast(MontyField31::ONE);
    const TWO: Self = Self::broadcast(MontyField31::TWO);
    const NEG_ONE: Self = Self::broadcast(MontyField31::NEG_ONE);

    #[inline]
    fn from_prime_subfield(f: Self::PrimeSubfield) -> Self {
        f.into()
    }

    #[inline]
    fn cube(&self) -> Self {
        let x2 = vec_mul(self, self);
        vec_mul(&x2, self)
    }

    #[inline]
    fn exp_const_u64<const POWER: u64>(&self) -> Self {
        // Specialise the small powers that dominate Poseidon2 / Rescue S-boxes (D = 3, 5, 7). Each
        // chains `vec_mul`; larger powers fall back to the generic square-and-multiply.
        match POWER {
            0 => Self::ONE,
            1 => *self,
            2 => vec_mul(self, self),
            3 => self.cube(),
            4 => {
                let x2 = vec_mul(self, self);
                vec_mul(&x2, &x2)
            }
            5 => {
                let x2 = vec_mul(self, self);
                let x4 = vec_mul(&x2, &x2);
                vec_mul(&x4, self)
            }
            6 => self.square().cube(),
            7 => {
                let x2 = vec_mul(self, self);
                let x3 = vec_mul(&x2, self);
                let x4 = vec_mul(&x2, &x2);
                vec_mul(&x4, &x3)
            }
            _ => self.exp_u64(POWER),
        }
    }

    #[inline]
    fn halve(&self) -> Self {
        Self(self.0.map(|x| x.halve()))
    }

    #[inline(always)]
    fn zero_vec(len: usize) -> Vec<Self> {
        // SAFETY: this is a `repr(transparent)` wrapper around `[MontyField31; WIDTH]`.
        unsafe { reconstitute_from_base(MontyField31::<FP>::zero_vec(len * WIDTH)) }
    }
}

impl_add_base_field!(
    PackedMontyField31Sve,
    MontyField31,
    (PackedMontyParameters, PMP)
);
impl_sub_base_field!(
    PackedMontyField31Sve,
    MontyField31,
    (PackedMontyParameters, PMP)
);
impl_mul_base_field!(
    PackedMontyField31Sve,
    MontyField31,
    (PackedMontyParameters, PMP)
);
impl_div_methods!(PackedMontyField31Sve, MontyField31, (FieldParameters, FP));
impl_packed_field_div!(PackedMontyField31Sve, (FieldParameters, FP));
impl_sum_prod_base_field!(PackedMontyField31Sve, MontyField31, (FieldParameters, FP));

impl<FP: FieldParameters> Algebra<MontyField31<FP>> for PackedMontyField31Sve<FP> {}

impl<FP: FieldParameters + RelativelyPrimePower<D>, const D: u64> InjectiveMonomial<D>
    for PackedMontyField31Sve<FP>
{
}

impl<FP: FieldParameters + RelativelyPrimePower<D>, const D: u64> PermutationMonomial<D>
    for PackedMontyField31Sve<FP>
{
    fn injective_exp_root_n(&self) -> Self {
        FP::exp_root_d(*self)
    }
}

impl_packed_value!(
    PackedMontyField31Sve,
    MontyField31,
    WIDTH,
    (PackedMontyParameters, PMP)
);

unsafe impl<FP: FieldParameters> PackedField for PackedMontyField31Sve<FP> {
    type Scalar = MontyField31<FP>;
}

unsafe impl<FP: FieldParameters> PackedFieldPow2 for PackedMontyField31Sve<FP> {
    #[inline]
    fn interleave(&self, other: Self, block_len: usize) -> (Self, Self) {
        // Interleaving `block_len`-sized chunks of lanes is a `TRN1`/`TRN2` of `32*block_len`-bit
        // elements. Since the vector length is fixed at `WIDTH`, this is a fixed permutation of the
        // backing arrays, computed directly (the 128-bit-granule case would otherwise need the
        // optional F64MM extension).
        assert!(block_len.is_power_of_two() && block_len <= WIDTH);
        if block_len == WIDTH {
            return (*self, other);
        }

        let a = self.0;
        let b = other.0;
        let mut r0 = [MontyField31::ZERO; WIDTH];
        let mut r1 = [MontyField31::ZERO; WIDTH];
        // `TRN1` gathers even-indexed blocks, `TRN2` odd-indexed blocks, taking each pair `(a, b)`.
        let blocks = WIDTH / block_len;
        for i in 0..blocks / 2 {
            let even = 2 * i;
            let odd = 2 * i + 1;
            let lo = 2 * i * block_len;
            let hi = (2 * i + 1) * block_len;
            r0[lo..lo + block_len].copy_from_slice(&a[even * block_len..(even + 1) * block_len]);
            r0[hi..hi + block_len].copy_from_slice(&b[even * block_len..(even + 1) * block_len]);
            r1[lo..lo + block_len].copy_from_slice(&a[odd * block_len..(odd + 1) * block_len]);
            r1[hi..hi + block_len].copy_from_slice(&b[odd * block_len..(odd + 1) * block_len]);
        }
        (Self(r0), Self(r1))
    }
}
