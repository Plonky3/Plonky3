//! Fixed-length (VLS) SVE packing for `MontyField31`.
//!
//! Unlike the NEON backend, which stores its lanes in a `uint32x4_t` register type, SVE vector
//! registers are *sizeless*: they cannot be struct fields, array elements, or `Sized`. The packed
//! type therefore keeps its data in a plain `[MontyField31; WIDTH]` array (exactly like the NEON and
//! WASM backends) and only materialises an `svuint32_t` as a transient function local, loading with
//! `svld1` and storing with `svst1`.
//!
//! `WIDTH` is fixed at compile time. The governing predicate covers exactly the first `WIDTH` lanes
//! (`svwhilelt`), so loads and stores never touch memory past the backing array even when the
//! hardware vector length exceeds `WIDTH * 32` bits. Correctness still requires the hardware vector
//! length to be *at least* `WIDTH * 32` bits; this is checked once via `debug_assert` in
//! [`PackedMontyField31Sve::to_vec`].
//!
//! The SVE intrinsic names below follow ACLE and target `core::arch::aarch64`. Coverage of the SVE
//! surface in `stdarch` evolves; any intrinsic that is not yet available on the pinned nightly can
//! be expressed with an inline `asm!` block over the same registers.

use alloc::vec::Vec;
use core::arch::aarch64::{self, svbool_t, svint32_t, svuint32_t};
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
/// to 16; the runtime guard in [`PackedMontyField31Sve::to_vec`] ensures the hardware vector length
/// is large enough.
const WIDTH: usize = 8;

/// The governing predicate: active on exactly the first `WIDTH` 32-bit lanes.
#[inline(always)]
fn pg() -> svbool_t {
    unsafe { aarch64::svwhilelt_b32_u32(0, WIDTH as u32) }
}

/// Fixed-length SVE implementation of `MontyField31` arithmetic.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(transparent)]
#[must_use]
pub struct PackedMontyField31Sve<PMP: PackedMontyParameters>(pub [MontyField31<PMP>; WIDTH]);

impl<PMP: PackedMontyParameters> PackedMontyField31Sve<PMP> {
    /// Load the backing array into an SVE vector, reading only the first `WIDTH` lanes.
    #[inline]
    #[must_use]
    fn to_vec(self) -> svuint32_t {
        // The hardware vector length (in 32-bit words) must cover our fixed `WIDTH`; otherwise the
        // predicate's high lanes fall outside the register and elements are silently dropped.
        debug_assert!(unsafe { aarch64::svcntw() } as usize >= WIDTH);

        // Safety: `MontyField31` is `repr(transparent)` over `u32`, so the array is `WIDTH`
        // contiguous `u32`s. The predicate activates exactly `WIDTH` lanes, so `svld1` reads only
        // within the array.
        unsafe { aarch64::svld1_u32(pg(), self.0.as_ptr().cast::<u32>()) }
    }

    /// Store an SVE vector into a fresh backing array, writing only the first `WIDTH` lanes.
    ///
    /// SAFETY: every active lane of `vec` must be a canonical `MontyField31`, i.e. in `[0, P)`.
    #[inline]
    unsafe fn from_vec(vec: svuint32_t) -> Self {
        let mut out = MaybeUninit::<[MontyField31<PMP>; WIDTH]>::uninit();
        // Safety: the predicate activates exactly `WIDTH` lanes and `out` is `WIDTH` `u32`s wide, so
        // `svst1` initialises every lane and stays within `out`. The caller guarantees canonical
        // lane values.
        unsafe {
            aarch64::svst1_u32(pg(), out.as_mut_ptr().cast::<u32>(), vec);
            Self(out.assume_init())
        }
    }

    /// Copy `value` to all positions in a packed vector.
    #[inline]
    const fn broadcast(value: MontyField31<PMP>) -> Self {
        Self([value; WIDTH])
    }

    /// Fused DIF butterfly for the forward FFT: computes `(x + y, (x - y) * roots)`.
    ///
    /// The `x - y` term is left unreduced: as `x, y` are in `[0, P)`, the raw `svsub` reinterpreted
    /// as signed lies in `(-P, P)`, which is exactly the input range accepted by the signed
    /// Montgomery multiply. This skips the modular reduction on `x - y`.
    #[inline]
    pub(crate) fn forward_butterfly(self, y: Self, roots: Self) -> (Self, Self) {
        let (sum, product);
        unsafe {
            let x_vec = self.to_vec();
            let y_vec = y.to_vec();

            sum = vec_add::<PMP>(x_vec, y_vec);

            let diff = aarch64::svreinterpret_s32_u32(aarch64::svsub_u32_x(pg(), x_vec, y_vec));
            let roots_s = aarch64::svreinterpret_s32_u32(roots.to_vec());
            product = vec_mul_signed::<PMP>(diff, roots_s);
        }
        // Safety: `vec_add` and `vec_mul_signed` return canonical values.
        unsafe { (Self::from_vec(sum), Self::from_vec(product)) }
    }
}

impl<PMP: PackedMontyParameters> From<MontyField31<PMP>> for PackedMontyField31Sve<PMP> {
    #[inline]
    fn from(value: MontyField31<PMP>) -> Self {
        Self::broadcast(value)
    }
}

// -----------------------------------------------------------------------------
// Vector kernels. Each takes and returns a transient `svuint32_t`; the governing
// predicate is `pg()` throughout. Inactive lanes are don't-care (`_x` forms).
// -----------------------------------------------------------------------------

/// Canonical modular addition: given `a, b` in `[0, P)` returns `(a + b) mod P` in `[0, P)`.
///
/// `t = a + b`, `u = t - P`; `min(t, u)` picks `t` when `t < P` (then `u` wrapped high) and `u`
/// otherwise. Mirrors `uint32x4_mod_add`.
#[inline]
#[must_use]
fn vec_add<PMP: PackedMontyParameters>(a: svuint32_t, b: svuint32_t) -> svuint32_t {
    unsafe {
        let p = aarch64::svdup_n_u32(PMP::PRIME);
        let t = aarch64::svadd_u32_x(pg(), a, b);
        let u = aarch64::svsub_u32_x(pg(), t, p);
        aarch64::svmin_u32_x(pg(), t, u)
    }
}

/// Canonical modular subtraction: given `a, b` in `[0, P)` returns `(a - b) mod P` in `[0, P)`.
///
/// Mirrors `uint32x4_mod_sub`.
#[inline]
#[must_use]
fn vec_sub<PMP: PackedMontyParameters>(a: svuint32_t, b: svuint32_t) -> svuint32_t {
    unsafe {
        let p = aarch64::svdup_n_u32(PMP::PRIME);
        let t = aarch64::svsub_u32_x(pg(), a, b);
        let u = aarch64::svadd_u32_x(pg(), t, p);
        aarch64::svmin_u32_x(pg(), t, u)
    }
}

/// Negate a vector in canonical form: returns `0` where `a == 0` and `P - a` otherwise.
#[inline]
#[must_use]
fn vec_neg<PMP: PackedMontyParameters>(a: svuint32_t) -> svuint32_t {
    unsafe {
        let p = aarch64::svdup_n_u32(PMP::PRIME);
        let zero = aarch64::svdup_n_u32(0);
        let t = aarch64::svsub_u32_x(pg(), p, a);
        let is_zero = aarch64::svcmpeq_n_u32(pg(), a, 0);
        aarch64::svsel_u32(is_zero, zero, t)
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
#[must_use]
fn vec_mul<PMP: PackedMontyParameters>(a: svuint32_t, b: svuint32_t) -> svuint32_t {
    unsafe {
        let p = aarch64::svdup_n_u32(PMP::PRIME);
        let mu = aarch64::svdup_n_u32(PMP::MONTY_MU);

        let x_hi = aarch64::svmulh_u32_x(pg(), a, b); // UMULH: high 32 bits of a * b.
        let x_lo = aarch64::svmul_u32_x(pg(), a, b); // low 32 bits of a * b.
        let t = aarch64::svmul_u32_x(pg(), x_lo, mu); // (x_lo * MONTY_MU) mod 2^32.
        let u_hi = aarch64::svmulh_u32_x(pg(), t, p); // high 32 bits of t * P.

        let diff = aarch64::svsub_u32_x(pg(), x_hi, u_hi);
        let borrow = aarch64::svcmplt_u32(pg(), x_hi, u_hi);
        // Add P back on the lanes that borrowed.
        aarch64::svadd_u32_m(borrow, diff, p)
    }
}

/// Signed Montgomery multiplication: `lhs, rhs` are signed values in `[-P, P]`, output is canonical
/// `[0, P)`. Used by the fused forward butterfly, which feeds an unreduced `x - y`.
///
/// `SMULH` gives the true high word of each signed product, so `D = ⌊C/2³²⌋ − ⌊Q·P/2³²⌋` needs no
/// doubling/halving correction (the low words of `C = lhs·rhs` and `Q·P` are congruent mod `2³²`, so
/// they cancel exactly). `Q = (lhs·rhs·MONTY_MU) mod 2³²`.
#[inline]
#[must_use]
fn vec_mul_signed<PMP: PackedMontyParameters>(lhs: svint32_t, rhs: svint32_t) -> svuint32_t {
    unsafe {
        let p = aarch64::svdup_n_s32(PMP::PRIME as i32);
        let mu = aarch64::svdup_n_s32(PMP::MONTY_MU as i32);

        let c_hi = aarch64::svmulh_s32_x(pg(), lhs, rhs); // SMULH: ⌊lhs·rhs / 2³²⌋.
        let mu_rhs = aarch64::svmul_s32_x(pg(), rhs, mu); // (rhs·MONTY_MU) mod 2³².
        let q = aarch64::svmul_s32_x(pg(), lhs, mu_rhs); // (lhs·rhs·MONTY_MU) mod 2³².
        let qp_hi = aarch64::svmulh_s32_x(pg(), q, p); // SMULH: ⌊Q·P / 2³²⌋.

        let d = aarch64::svsub_s32_x(pg(), c_hi, qp_hi); // D in (-P, P).
        // D is negative iff `c_hi < qp_hi`; add P there to land in [0, P).
        let underflow = aarch64::svcmplt_s32(pg(), c_hi, qp_hi);
        let reduced = aarch64::svadd_s32_m(underflow, d, p);
        aarch64::svreinterpret_u32_s32(reduced)
    }
}

impl<PMP: PackedMontyParameters> Add for PackedMontyField31Sve<PMP> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        let res = vec_add::<PMP>(self.to_vec(), rhs.to_vec());
        // Safety: `vec_add` returns canonical values given canonical inputs.
        unsafe { Self::from_vec(res) }
    }
}

impl<PMP: PackedMontyParameters> Sub for PackedMontyField31Sve<PMP> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        let res = vec_sub::<PMP>(self.to_vec(), rhs.to_vec());
        // Safety: `vec_sub` returns canonical values given canonical inputs.
        unsafe { Self::from_vec(res) }
    }
}

impl<PMP: PackedMontyParameters> Neg for PackedMontyField31Sve<PMP> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        let res = vec_neg::<PMP>(self.to_vec());
        // Safety: `vec_neg` returns canonical values given canonical inputs.
        unsafe { Self::from_vec(res) }
    }
}

impl<PMP: PackedMontyParameters> Mul for PackedMontyField31Sve<PMP> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let res = vec_mul::<PMP>(self.to_vec(), rhs.to_vec());
        // Safety: `vec_mul` returns canonical values given canonical inputs.
        unsafe { Self::from_vec(res) }
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
        let x = self.to_vec();
        let x2 = vec_mul::<FP>(x, x);
        let x3 = vec_mul::<FP>(x2, x);
        // Safety: `vec_mul` returns canonical values given canonical inputs.
        unsafe { Self::from_vec(x3) }
    }

    #[inline]
    fn exp_const_u64<const POWER: u64>(&self) -> Self {
        // Specialise the small powers that dominate Poseidon2 / Rescue S-boxes (D = 3, 5, 7). Each
        // loads once, chains `vec_mul`, and stores once. Larger powers fall back to the generic
        // square-and-multiply.
        let x = self.to_vec();
        // Safety: every `vec_mul` result is canonical given canonical inputs.
        unsafe {
            match POWER {
                0 => Self::ONE,
                1 => *self,
                2 => Self::from_vec(vec_mul::<FP>(x, x)),
                3 => self.cube(),
                4 => {
                    let x2 = vec_mul::<FP>(x, x);
                    Self::from_vec(vec_mul::<FP>(x2, x2))
                }
                5 => {
                    let x2 = vec_mul::<FP>(x, x);
                    let x4 = vec_mul::<FP>(x2, x2);
                    Self::from_vec(vec_mul::<FP>(x4, x))
                }
                6 => self.square().cube(),
                7 => {
                    let x2 = vec_mul::<FP>(x, x);
                    let x3 = vec_mul::<FP>(x2, x);
                    let x4 = vec_mul::<FP>(x2, x2);
                    Self::from_vec(vec_mul::<FP>(x4, x3))
                }
                _ => self.exp_u64(POWER),
            }
        }
    }

    #[inline]
    fn halve(&self) -> Self {
        // Scalar per-lane fallback. A native version would fold the odd/even correction into a
        // predicated shift, mirroring `halve_neon`.
        Self(self.0.map(|x| x.halve()))
    }

    #[inline(always)]
    fn zero_vec(len: usize) -> Vec<Self> {
        // Safety: this is a `repr(transparent)` wrapper around `[MontyField31; WIDTH]`.
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
        // Interleaving `block_len`-sized chunks of 32-bit lanes is a transpose of `32*block_len`-bit
        // elements: `TRN1`/`TRN2` at the matching element (or quadword) granularity.
        assert!(block_len.is_power_of_two() && block_len <= WIDTH);
        if block_len == WIDTH {
            return (*self, other);
        }

        let a = self.to_vec();
        let b = other.to_vec();
        let (r0, r1) = unsafe {
            match block_len {
                1 => (aarch64::svtrn1_u32(a, b), aarch64::svtrn2_u32(a, b)),
                2 => {
                    let a64 = aarch64::svreinterpret_u64_u32(a);
                    let b64 = aarch64::svreinterpret_u64_u32(b);
                    (
                        aarch64::svreinterpret_u32_u64(aarch64::svtrn1_u64(a64, b64)),
                        aarch64::svreinterpret_u32_u64(aarch64::svtrn2_u64(a64, b64)),
                    )
                }
                // `block_len == 4` transposes 128-bit blocks (quadwords).
                _ => (aarch64::svtrn1q_u32(a, b), aarch64::svtrn2q_u32(a, b)),
            }
        };
        // Safety: interleaving only permutes canonical lanes, so the results stay canonical.
        unsafe { (Self::from_vec(r0), Self::from_vec(r1)) }
    }
}
