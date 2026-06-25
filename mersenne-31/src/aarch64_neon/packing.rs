use alloc::vec::Vec;
use core::arch::aarch64::{self, uint32x4_t, uint64x2_t};
use core::iter::{Product, Sum};
use core::mem::transmute;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use p3_field::exponentiation::exp_1717986917;
use p3_field::interleave::{interleave_u32, interleave_u64};
use p3_field::op_assign_macros::{
    impl_add_assign, impl_add_base_field, impl_div_methods, impl_mul_base_field, impl_mul_methods,
    impl_packed_field_div, impl_packed_value, impl_rng, impl_sub_assign, impl_sub_base_field,
    impl_sum_prod_base_field, ring_sum,
};
use p3_field::{
    Algebra, BasedVectorSpace, ExtensionField, Field, InjectiveMonomial, PackedField,
    PackedFieldPow2, PackedValue, PermutationMonomial, PrimeCharacteristicRing,
    dispatch_chunked_mixed_dot_product, generic_batched_columnwise_dot_product,
    impl_packed_field_pow_2, uint32x4_mod_add, uint32x4_mod_sub,
};
use p3_util::reconstitute_from_base;
use rand::distr::{Distribution, StandardUniform};
use rand::{Rng, RngExt};

use crate::Mersenne31;

const WIDTH: usize = 4;
const P: uint32x4_t = unsafe { transmute::<[u32; WIDTH], _>([0x7fffffff; WIDTH]) };

/// Vectorized NEON implementation of `Mersenne31` arithmetic.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(transparent)] // Needed to make `transmute`s safe.
#[must_use]
pub struct PackedMersenne31Neon(pub [Mersenne31; WIDTH]);

impl PackedMersenne31Neon {
    #[inline]
    #[must_use]
    /// Get an arch-specific vector representing the packed values.
    pub(crate) fn to_vector(self) -> uint32x4_t {
        unsafe {
            // Safety: `Mersenne31` is `repr(transparent)` so it can be transmuted to `u32`. It
            // follows that `[Mersenne31; WIDTH]` can be transmuted to `[u32; WIDTH]`, which can be
            // transmuted to `uint32x4_t`, since arrays are guaranteed to be contiguous in memory.
            // Finally `PackedMersenne31Neon` is `repr(transparent)` so it can be transmuted to
            // `[Mersenne31; WIDTH]`.
            transmute(self)
        }
    }

    #[inline]
    /// Make a packed field vector from an arch-specific vector.
    ///
    /// SAFETY: The caller must ensure that each element of `vector` represents a valid
    /// `Mersenne31`.  In particular, each element of vector must be in `0..=P` (i.e. it fits in 31
    /// bits).
    pub(crate) unsafe fn from_vector(vector: uint32x4_t) -> Self {
        // Safety: It is up to the user to ensure that elements of `vector` represent valid
        // `Mersenne31` values. We must only reason about memory representations. `uint32x4_t` can
        // be transmuted to `[u32; WIDTH]` (since arrays elements are contiguous in memory), which
        // can be transmuted to `[Mersenne31; WIDTH]` (since `Mersenne31` is `repr(transparent)`),
        // which in turn can be transmuted to `PackedMersenne31Neon` (since `PackedMersenne31Neon`
        // is also `repr(transparent)`).
        unsafe { transmute(vector) }
    }

    /// Copy `value` to all positions in a packed vector. This is the same as
    /// `From<Mersenne31>::from`, but `const`.
    #[inline]
    const fn broadcast(value: Mersenne31) -> Self {
        Self([value; WIDTH])
    }
}

impl From<Mersenne31> for PackedMersenne31Neon {
    #[inline]
    fn from(value: Mersenne31) -> Self {
        Self::broadcast(value)
    }
}

impl Add for PackedMersenne31Neon {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        let lhs = self.to_vector();
        let rhs = rhs.to_vector();
        let res = uint32x4_mod_add(lhs, rhs, P);
        unsafe {
            // Safety: `uint32x4_mod_add` returns valid values when given valid values.
            Self::from_vector(res)
        }
    }
}

impl Sub for PackedMersenne31Neon {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        let lhs = self.to_vector();
        let rhs = rhs.to_vector();
        let res = uint32x4_mod_sub(lhs, rhs, P);
        unsafe {
            // Safety: `uint32x4_mod_sub` returns valid values when given valid values.
            Self::from_vector(res)
        }
    }
}

impl Neg for PackedMersenne31Neon {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        let val = self.to_vector();
        let res = neg(val);
        unsafe {
            // Safety: `neg` returns valid values when given valid values.
            Self::from_vector(res)
        }
    }
}

impl Mul for PackedMersenne31Neon {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let lhs = self.to_vector();
        let rhs = rhs.to_vector();
        let res = mul(lhs, rhs);
        unsafe {
            // Safety: `mul` returns valid values when given valid values.
            Self::from_vector(res)
        }
    }
}

impl_add_assign!(PackedMersenne31Neon);
impl_sub_assign!(PackedMersenne31Neon);
impl_mul_methods!(PackedMersenne31Neon);
ring_sum!(PackedMersenne31Neon);
impl_rng!(PackedMersenne31Neon);

impl PrimeCharacteristicRing for PackedMersenne31Neon {
    type PrimeSubfield = Mersenne31;

    const ZERO: Self = Self::broadcast(Mersenne31::ZERO);
    const ONE: Self = Self::broadcast(Mersenne31::ONE);
    const TWO: Self = Self::broadcast(Mersenne31::TWO);
    const NEG_ONE: Self = Self::broadcast(Mersenne31::NEG_ONE);

    #[inline]
    fn from_prime_subfield(f: Self::PrimeSubfield) -> Self {
        f.into()
    }

    #[inline]
    fn halve(&self) -> Self {
        // Compute (val + (val & 1) * P) >> 1 per lane.
        // This equals val * 2^{-1} mod P, staying in the 0..=P representation.
        let val = (*self).to_vector();
        unsafe {
            // Safety: If this code got compiled then NEON intrinsics are available.
            let one = aarch64::vdupq_n_u32(1);
            // is_odd_mask = 0xFFFF_FFFF when LSB set, else 0.
            let is_odd_mask = aarch64::vtstq_u32(val, one);
            // Select P for odd lanes, 0 for even lanes.
            let to_add = aarch64::vandq_u32(P, is_odd_mask);
            // Halving add: (val + to_add) >> 1
            let halved = aarch64::vhaddq_u32(val, to_add);
            Self::from_vector(halved)
        }
    }

    #[inline(always)]
    fn exp_const_u64<const POWER: u64>(&self) -> Self {
        // We provide specialised code for power 5 as this turns up regularly.
        //
        // The other powers could be specialised similarly but we ignore this for now.
        match POWER {
            0 => Self::ONE,
            1 => *self,
            2 => self.square(),
            3 => self.cube(),
            4 => self.square().square(),
            5 => unsafe {
                let val = self.to_vector();
                Self::from_vector(exp5(val))
            },
            6 => self.square().cube(),
            7 => {
                let x2 = self.square();
                let x3 = x2 * *self;
                let x4 = x2.square();
                x3 * x4
            }
            _ => self.exp_u64(POWER),
        }
    }

    #[inline(always)]
    fn zero_vec(len: usize) -> Vec<Self> {
        // SAFETY: this is a repr(transparent) wrapper around an array.
        unsafe { reconstitute_from_base(Mersenne31::zero_vec(len * WIDTH)) }
    }

    #[inline]
    fn dot_product<const N: usize>(u: &[Self; N], v: &[Self; N]) -> Self {
        // For small `N` the deferred-reduction setup (two final folds plus the narrowing)
        // is not amortized, and NEON's `sqdmulh`-based reduced multiply is cheap enough that
        // a plain reduced-multiply sum wins. Measured on NEON: deferral regresses ~10% at
        // `N = 2` but improves 10-32% from `N = 4` upward.
        if N < 4 {
            return u.iter().zip(v).map(|(&x, &y)| x * y).sum();
        }

        // Single-accumulator form of the `coeffwise_dot_product` scheme: widen each
        // 32x32 -> 64-bit product into `lo` (lanes 0-1) and `hi` (lanes 2-3) with
        // multiply-accumulates, deferring the Mersenne reduction. Inputs are in `0..=P`,
        // so a product is at most `P^2 < 2^62`. `N == 4` accumulates all four without an
        // intermediate fold (`4 * P^2 < 2^64`); larger `N` fold every 3 products, keeping
        // `2^33 + 3 * P^2 < 2^64`, so the u64 lanes never overflow.
        unsafe {
            // Safety: If this code got compiled then NEON intrinsics are available.
            let zero = aarch64::vdupq_n_u64(0);
            let mut lo = zero;
            let mut hi = zero;
            let mut unreduced = 0;
            for i in 0..N {
                let a = u[i].to_vector();
                let b = v[i].to_vector();
                lo = aarch64::vmlal_u32(lo, aarch64::vget_low_u32(a), aarch64::vget_low_u32(b));
                hi = aarch64::vmlal_high_u32(hi, a, b);
                unreduced += 1;
                // `N == 4` is the only size here whose products all fit without an
                // intermediate fold, so skip it; larger `N` fold every 3.
                if unreduced == 3 && N != 4 {
                    unreduced = 0;
                    lo = partial_reduce_u64(lo);
                    hi = partial_reduce_u64(hi);
                }
            }
            // At most 2 unreduced products on top of a folded value (< 2^33 + 2 * P^2 < 2^64);
            // two folds bring each lane to <= 2 P, then `reduce_sum` canonicalizes to 0..=P.
            let l = partial_reduce_u64(partial_reduce_u64(lo));
            let h = partial_reduce_u64(partial_reduce_u64(hi));
            let narrowed = aarch64::vuzp1q_u32(
                aarch64::vreinterpretq_u32_u64(l),
                aarch64::vreinterpretq_u32_u64(h),
            );
            Self::from_vector(reduce_sum(narrowed))
        }
    }
}

// Degree of the smallest permutation polynomial for Mersenne31.
//
// As p - 1 = 2×3^2×7×11×... the smallest choice for a degree D satisfying gcd(p - 1, D) = 1 is 5.
impl InjectiveMonomial<5> for PackedMersenne31Neon {}

impl PermutationMonomial<5> for PackedMersenne31Neon {
    /// In the field `Mersenne31`, `a^{1/5}` is equal to a^{1717986917}.
    ///
    /// This follows from the calculation `5 * 1717986917 = 4*(2^31 - 2) + 1 = 1 mod p - 1`.
    fn injective_exp_root_n(&self) -> Self {
        exp_1717986917(*self)
    }
}

impl_add_base_field!(PackedMersenne31Neon, Mersenne31);
impl_sub_base_field!(PackedMersenne31Neon, Mersenne31);
impl_mul_base_field!(PackedMersenne31Neon, Mersenne31);
impl_div_methods!(PackedMersenne31Neon, Mersenne31);
impl_packed_field_div!(PackedMersenne31Neon);
impl_sum_prod_base_field!(PackedMersenne31Neon, Mersenne31);

impl Algebra<Mersenne31> for PackedMersenne31Neon {
    // Benchmarked on AArch64 NEON: chunk=16 ≈ 51ns, chunk=8 ≈ 54ns, chunk=4 ≈ 59ns.
    const BATCHED_LC_CHUNK: usize = 16;

    #[inline(always)]
    fn mixed_dot_product<const N: usize>(a: &[Self; N], f: &[Mersenne31; N]) -> Self {
        dispatch_chunked_mixed_dot_product::<Self, Mersenne31, N>(
            a,
            f,
            <Self as Algebra<Mersenne31>>::BATCHED_LC_CHUNK,
        )
    }
}

/// Given a `val` in `0, ..., 2 P`, return a `res` in `0, ..., P` such that `res = val (mod P)`
#[inline]
#[must_use]
fn reduce_sum(val: uint32x4_t) -> uint32x4_t {
    // val is in 0, ..., 2 P. If val is in 0, ..., P - 1 then it is valid and
    // u := (val - P) mod 2^32 is in P <u 2^32 - P, ..., 2^32 - 1 and unsigned_min(val, u) = val as
    // desired. If val is in P + 1, ..., 2 P, then u is in 1, ..., P < P + 1 so u is valid, and
    // unsigned_min(val, u) = u as desired. The remaining case of val = P, u = 0 is trivial.

    unsafe {
        // Safety: If this code got compiled then NEON intrinsics are available.
        let u = aarch64::vsubq_u32(val, P);
        aarch64::vminq_u32(val, u)
    }
}

/// Multiply two 31-bit numbers to obtain a 62-bit immediate result, and return the high 31 bits of
/// that result. Results are arbitrary if the inputs do not fit in 31 bits.
#[inline]
#[must_use]
fn mul_31x31_to_hi_31(lhs: uint32x4_t, rhs: uint32x4_t) -> uint32x4_t {
    // This is just a wrapper around `aarch64::vqdmulhq_s32`, so we don't have to worry about the
    // casting elsewhere.
    unsafe {
        // Safety: If this code got compiled then NEON intrinsics are available.
        aarch64::vreinterpretq_u32_s32(aarch64::vqdmulhq_s32(
            aarch64::vreinterpretq_s32_u32(lhs),
            aarch64::vreinterpretq_s32_u32(rhs),
        ))
    }
}

/// Multiply vectors of Mersenne-31 field elements that fit in 31 bits.
/// If the inputs do not fit in 31 bits, the result is undefined.
#[inline]
#[must_use]
fn mul(lhs: uint32x4_t, rhs: uint32x4_t) -> uint32x4_t {
    // We want this to compile to:
    //      sqdmulh  prod_hi31.4s, lhs.4s, rhs.4s
    //      mul      t.4s, lhs.4s, rhs.4s
    //      mla      t.4s, prod_hi31.4s, P.4s
    //      sub      u.4s, t.4s, P.4s
    //      umin     res.4s, t.4s, u.4s
    // throughput: 1.25 cyc/vec (3.2 els/cyc)
    // latency: 10 cyc

    // We want to return res in 0, ..., P such that res = lhs * rhs (mod P).
    // Let prod := lhs * rhs. Break it up into prod = 2^31 prod_hi31 + prod_lo31, where both limbs
    // are in 0, ..., 2^31 - 1. Then prod = prod_hi31 + prod_lo31 (mod P), so let
    // t := prod_hi31 + prod_lo31.
    // Define prod_lo32 = prod mod 2^32 and observe that
    //   prod_lo32 = prod_lo31 + 2^31 (prod_hi31 mod 2)
    //             = prod_lo31 + 2^31 prod_hi31                                          (mod 2^32)
    // Then
    //   t = prod_lo32 - 2^31 prod_hi31 + prod_hi31                                      (mod 2^32)
    //     = prod_lo32 - (2^31 - 1) prod_hi31                                            (mod 2^32)
    //     = prod_lo32 - prod_hi31 * P                                                   (mod 2^32)
    //
    // t is in 0, ..., 2 P, so we apply reduce_sum to get the result.

    unsafe {
        // Safety: If this code got compiled then NEON intrinsics are available.
        let prod_hi31 = mul_31x31_to_hi_31(lhs, rhs);
        let prod_lo32 = aarch64::vmulq_u32(lhs, rhs);
        let t = aarch64::vmlsq_u32(prod_lo32, prod_hi31, P);
        reduce_sum(t)
    }
}

/// Fold a vector of 64-bit accumulators once: given `val`, return `res = val (mod P)`
/// with `res <= (val >> 31) + P`.
///
/// Uses `2^31 = 1 (mod P)`: writing `val = hi * 2^31 + lo` with `lo <= P`, we have
/// `val = hi + lo (mod P)`. Two applications bring any `val < 2^64` to `0..=P + 3`.
#[inline]
#[must_use]
fn partial_reduce_u64(val: uint64x2_t) -> uint64x2_t {
    unsafe {
        // Safety: If this code got compiled then NEON intrinsics are available.
        let p64 = aarch64::vdupq_n_u64(0x7fffffff);
        aarch64::vaddq_u64(aarch64::vandq_u64(val, p64), aarch64::vshrq_n_u64(val, 31))
    }
}

/// Negate a vector of Mersenne-31 field elements that fit in 31 bits.
/// If the inputs do not fit in 31 bits, the result is undefined.
#[inline]
#[must_use]
fn neg(val: uint32x4_t) -> uint32x4_t {
    // We want this to compile to:
    //      eor  res.16b, val.16b, P.16b
    // throughput: .25 cyc/vec (16 els/cyc)
    // latency: 2 cyc

    // val is in 0, ..., P, so res := P - val is also in 0, ..., P.

    unsafe {
        // Safety: If this code got compiled then NEON intrinsics are available.
        aarch64::vsubq_u32(P, val)
    }
}

impl_packed_value!(PackedMersenne31Neon, Mersenne31, WIDTH);

unsafe impl PackedField for PackedMersenne31Neon {
    type Scalar = Mersenne31;

    #[inline]
    fn coeffwise_dot_product<'a, I>(d: usize, pairs: I) -> [Self; 8]
    where
        Self: 'a,
        I: Iterator<Item = (&'a [Self], Self)>,
    {
        // Accumulate the raw 32x32 -> 64-bit products with widening multiply-accumulates
        // (2 instructions per product), deferring the Mersenne reduction, instead of
        // paying the 8-instruction reduced multiply-add per product.
        //
        // Each coefficient accumulator is a pair of u64x2 vectors: `lo` holds lanes 0-1
        // and `hi` lanes 2-3. Inputs are in `0..=P`, so a product is at most
        // `P^2 < 2^62`; folding the accumulators below `2^33` every 3 iterations keeps
        // `2^33 + 3 * P^2 < 2^64`, so the u64 lanes never overflow.
        debug_assert!(d <= 8, "Extension degree > 8 not supported");
        unsafe {
            // Safety: If this code got compiled then NEON intrinsics are available.
            let zero = aarch64::vdupq_n_u64(0);
            let mut lo = [zero; 8];
            let mut hi = [zero; 8];
            let mut unreduced = 0;
            for (coeffs, base) in pairs {
                let b = base.to_vector();
                for (k, coeff) in coeffs.iter().take(d).enumerate() {
                    let c = coeff.to_vector();
                    lo[k] = aarch64::vmlal_u32(
                        lo[k],
                        aarch64::vget_low_u32(c),
                        aarch64::vget_low_u32(b),
                    );
                    hi[k] = aarch64::vmlal_high_u32(hi[k], c, b);
                }
                unreduced += 1;
                if unreduced == 3 {
                    unreduced = 0;
                    for k in 0..d {
                        lo[k] = partial_reduce_u64(lo[k]);
                        hi[k] = partial_reduce_u64(hi[k]);
                    }
                }
            }
            core::array::from_fn(|k| {
                if k < d {
                    // At most 2 unreduced products on top of a folded value: the lanes
                    // are below 2^33 + 2 * P^2 < 2^64, so two folds bring them to
                    // P + 3 <= 2 P, and `reduce_sum` yields a canonical value in 0..=P.
                    let l = partial_reduce_u64(partial_reduce_u64(lo[k]));
                    let h = partial_reduce_u64(partial_reduce_u64(hi[k]));
                    let narrowed = aarch64::vuzp1q_u32(
                        aarch64::vreinterpretq_u32_u64(l),
                        aarch64::vreinterpretq_u32_u64(h),
                    );
                    Self::from_vector(reduce_sum(narrowed))
                } else {
                    Self::ZERO
                }
            })
        }
    }
}

/// Columnwise dot-product kernel with deferred Mersenne reductions, implementing
/// [`Field::batched_columnwise_dot_product`] for `Mersenne31`.
///
/// Dispatches the extension degree to a monomorphized kernel so the per-word loops
/// fully unroll; degrees without a kernel fall back to the generic accumulation.
pub(crate) fn batched_columnwise_dot_product<EF, R, I, const N: usize>(
    acc: &mut [EF::ExtensionPacking],
    items: I,
) where
    EF: ExtensionField<Mersenne31>,
    R: Iterator<Item = PackedMersenne31Neon>,
    I: Iterator<Item = (R, [EF; N])>,
{
    match EF::DIMENSION {
        1 => columnwise_kernel::<EF, R, I, N, 1>(acc, items),
        2 => columnwise_kernel::<EF, R, I, N, 2>(acc, items),
        4 => columnwise_kernel::<EF, R, I, N, 4>(acc, items),
        8 => columnwise_kernel::<EF, R, I, N, 8>(acc, items),
        _ => generic_batched_columnwise_dot_product::<Mersenne31, EF, R, I, N>(acc, items),
    }
}

/// Accumulate up to 3 products into a fresh (lanes 0-1, lanes 2-3) pair of u64x2
/// vectors and fold it once, leaving both results below `2^33` (`3 * P^2 < 2^64`).
#[inline(always)]
fn mac_up_to_3(vs: &[uint32x4_t], svs: [uint32x4_t; 3]) -> (uint64x2_t, uint64x2_t) {
    unsafe {
        // Safety: If this code got compiled then NEON intrinsics are available.
        let mut lo =
            aarch64::vmull_u32(aarch64::vget_low_u32(vs[0]), aarch64::vget_low_u32(svs[0]));
        let mut hi = aarch64::vmull_high_u32(vs[0], svs[0]);
        for (&v, &s) in vs[1..].iter().zip(&svs[1..]) {
            lo = aarch64::vmlal_u32(lo, aarch64::vget_low_u32(v), aarch64::vget_low_u32(s));
            hi = aarch64::vmlal_high_u32(hi, v, s);
        }
        (partial_reduce_u64(lo), partial_reduce_u64(hi))
    }
}

/// Monomorphized body of [`batched_columnwise_dot_product`] for extension degree `D`.
///
/// Rows are consumed three at a time: each triple of raw 32x32 -> 64-bit products is
/// accumulated in registers with widening MACs, folded once below `2^33`, and added
/// into u64 lane accumulators. A safety fold of the accumulators every `2^28` row
/// triples keeps them below `2^62` for any stream length, so the final two folds
/// plus `reduce_sum` always produce canonical values.
fn columnwise_kernel<EF, R, I, const N: usize, const D: usize>(
    out: &mut [EF::ExtensionPacking],
    mut items: I,
) where
    EF: ExtensionField<Mersenne31>,
    R: Iterator<Item = PackedMersenne31Neon>,
    I: Iterator<Item = (R, [EF; N])>,
{
    debug_assert_eq!(EF::DIMENSION, D);
    debug_assert_eq!(out.len() % N, 0);
    let packed_width = out.len() / N;
    let zero64 = unsafe { aarch64::vdupq_n_u64(0) };
    let zero32 = unsafe { aarch64::vdupq_n_u32(0) };

    // Word accumulator layout: word-major, with the `(weight j, coefficient k)` pair of
    // `(lanes 0-1, lanes 2-3)` u64x2 vectors of word `c` at `c * N * D * 2 + (j * D + k) * 2`.
    let mut words = alloc::vec![zero64; packed_width * N * D * 2];
    let broadcast = |scales: &[EF; N]| -> [[uint32x4_t; D]; N] {
        let mut out = [[zero32; D]; N];
        for (row, scale) in out.iter_mut().zip(scales) {
            let coeffs = scale.as_basis_coefficients_slice();
            for (v, &coeff) in row.iter_mut().zip(coeffs) {
                *v = PackedMersenne31Neon::from(coeff).to_vector();
            }
        }
        out
    };

    let mut triples = 0u32;
    while let Some((r0, s0)) = items.next() {
        let sv0 = broadcast(&s0);
        let Some((r1, s1)) = items.next() else {
            // Single trailing row.
            for (aw, m0) in words.chunks_exact_mut(N * D * 2).zip(r0) {
                let v0 = m0.to_vector();
                for (j, sv0_j) in sv0.iter().enumerate() {
                    for (k, &s) in sv0_j.iter().enumerate() {
                        let (lo, hi) = mac_up_to_3(&[v0], [s; 3]);
                        let idx = (j * D + k) * 2;
                        unsafe {
                            aw[idx] = aarch64::vaddq_u64(aw[idx], lo);
                            aw[idx + 1] = aarch64::vaddq_u64(aw[idx + 1], hi);
                        }
                    }
                }
            }
            break;
        };
        let sv1 = broadcast(&s1);
        let Some((r2, s2)) = items.next() else {
            // Trailing row pair.
            for (aw, (m0, m1)) in words.chunks_exact_mut(N * D * 2).zip(r0.zip(r1)) {
                let (v0, v1) = (m0.to_vector(), m1.to_vector());
                for j in 0..N {
                    for k in 0..D {
                        let (lo, hi) = mac_up_to_3(&[v0, v1], [sv0[j][k], sv1[j][k], sv1[j][k]]);
                        let idx = (j * D + k) * 2;
                        unsafe {
                            aw[idx] = aarch64::vaddq_u64(aw[idx], lo);
                            aw[idx + 1] = aarch64::vaddq_u64(aw[idx + 1], hi);
                        }
                    }
                }
            }
            break;
        };
        let sv2 = broadcast(&s2);

        for (aw, ((m0, m1), m2)) in words.chunks_exact_mut(N * D * 2).zip(r0.zip(r1).zip(r2)) {
            let (v0, v1, v2) = (m0.to_vector(), m1.to_vector(), m2.to_vector());
            for j in 0..N {
                for k in 0..D {
                    let (lo, hi) = mac_up_to_3(&[v0, v1, v2], [sv0[j][k], sv1[j][k], sv2[j][k]]);
                    let idx = (j * D + k) * 2;
                    unsafe {
                        aw[idx] = aarch64::vaddq_u64(aw[idx], lo);
                        aw[idx + 1] = aarch64::vaddq_u64(aw[idx + 1], hi);
                    }
                }
            }
        }

        triples += 1;
        if triples == 1 << 28 {
            triples = 0;
            for w in &mut words {
                *w = partial_reduce_u64(*w);
            }
        }
    }

    for (out_cj, words_c) in out.iter_mut().zip(words.chunks_exact(D * 2)) {
        let reduced = EF::ExtensionPacking::from_basis_coefficients_fn(|k| {
            let l = partial_reduce_u64(partial_reduce_u64(words_c[k * 2]));
            let h = partial_reduce_u64(partial_reduce_u64(words_c[k * 2 + 1]));
            unsafe {
                let narrowed = aarch64::vuzp1q_u32(
                    aarch64::vreinterpretq_u32_u64(l),
                    aarch64::vreinterpretq_u32_u64(h),
                );
                PackedMersenne31Neon::from_vector(reduce_sum(narrowed))
            }
        });
        *out_cj += reduced;
    }
}

/// Compute the permutation x -> x^5 on Mersenne-31 field elements.
///
/// # Safety
/// `x` must be represented as a value in `{0, ..., P}`.
/// If the input does not conform to this representation, the result is undefined.
/// The output will be represented as a value in `{0, ..., P}`.
///
/// # TODO
/// This could be further improved with a specialized function.
#[inline(always)]
pub(crate) fn exp5(x: uint32x4_t) -> uint32x4_t {
    // For Mersenne31, x^5 = x * x^4 = x * (x^2)^2
    //
    // We compute:
    //   x2 = x * x
    //   x4 = x2 * x2
    //   x5 = x4 * x
    //
    // throughput: ~4 cyc/vec
    // latency: ~30 cyc (3 dependent multiplications)

    // x is guaranteed to be in [0, P]
    let x2 = mul(x, x);
    let x4 = mul(x2, x2);
    mul(x4, x)
}

impl_packed_field_pow_2!(
    PackedMersenne31Neon;
    [
        (1, interleave_u32),
        (2, interleave_u64)
    ],
    WIDTH
);

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use p3_field_testing::{test_packed_field, test_packed_field_dot_product_boundary};

    use super::{Mersenne31, PackedMersenne31Neon};

    /// Zero has a redundant representation, so let's test both.
    const ZEROS: PackedMersenne31Neon = PackedMersenne31Neon(Mersenne31::new_array([
        0x00000000, 0x7fffffff, 0x00000000, 0x7fffffff,
    ]));

    const SPECIAL_VALS: PackedMersenne31Neon = PackedMersenne31Neon(Mersenne31::new_array([
        0x00000000, 0x00000001, 0x00000002, 0x7ffffffe,
    ]));

    test_packed_field!(
        crate::PackedMersenne31Neon,
        &[super::ZEROS],
        &[crate::PackedMersenne31Neon::ONE],
        super::SPECIAL_VALS
    );

    test_packed_field_dot_product_boundary!(crate::PackedMersenne31Neon);

    /// The NEON `coeffwise_dot_product` must agree with the generic coefficient-wise
    /// accumulation, including on boundary values (0, 1, P - 1 and the redundant
    /// representation P of zero) and on stream lengths exercising every deferred
    /// reduction phase.
    #[test]
    fn coeffwise_dot_product_matches_generic() {
        use p3_field::{PackedField, PrimeCharacteristicRing};

        const P: u32 = 0x7fffffff;
        let val = |i: u32| -> u32 {
            match i % 7 {
                0 => 0,
                1 => 1,
                2 => P - 1,
                3 => P,
                _ => i.wrapping_mul(0x9e3779b9) % P,
            }
        };
        let packed = |i: u32| {
            PackedMersenne31Neon(Mersenne31::new_array([
                val(i),
                val(i.wrapping_add(1)),
                val(i.wrapping_add(2)),
                val(i.wrapping_add(3)),
            ]))
        };

        for d in 1..=8usize {
            for len in [0usize, 1, 2, 3, 4, 6, 7, 100] {
                let coeffs: Vec<[PackedMersenne31Neon; 8]> = (0..len)
                    .map(|i| core::array::from_fn(|k| packed((i * 8 + k) as u32)))
                    .collect();
                let bases: Vec<PackedMersenne31Neon> =
                    (0..len).map(|i| packed((i + 1000) as u32)).collect();

                let mut expected = [PackedMersenne31Neon::ZERO; 8];
                for (c, &b) in coeffs.iter().zip(&bases) {
                    for k in 0..d {
                        expected[k] += c[k] * b;
                    }
                }

                let got = PackedMersenne31Neon::coeffwise_dot_product(
                    d,
                    coeffs.iter().zip(&bases).map(|(c, &b)| (&c[..], b)),
                );
                assert_eq!(expected, got, "d = {d}, len = {len}");
            }
        }
    }

    /// The NEON columnwise kernel must agree with the generic accumulation for every
    /// row-count phase of the 3-row blocking (0, 1 and 2 trailing rows), on boundary
    /// values, for both the trivial and a degree-4 extension.
    #[test]
    fn batched_columnwise_dot_product_matches_generic() {
        use p3_field::extension::Complex;
        use p3_field::{
            BasedVectorSpace, ExtensionField, PackedFieldExtension, PrimeCharacteristicRing,
        };

        use crate::QM31;

        const P: u32 = 0x7fffffff;
        let val = |i: u32| -> u32 {
            match i % 7 {
                0 => 0,
                1 => 1,
                2 => P - 1,
                3 => P,
                _ => i.wrapping_mul(0x9e3779b9) % P,
            }
        };
        let packed = |i: u32| {
            PackedMersenne31Neon(Mersenne31::new_array([
                val(i),
                val(i.wrapping_add(1)),
                val(i.wrapping_add(2)),
                val(i.wrapping_add(3)),
            ]))
        };
        let scalar = |i: u32| Mersenne31::new_checked(val(i) % P).unwrap();

        fn check<EF: ExtensionField<Mersenne31>, const N: usize>(
            packed_width: usize,
            rows: &[Vec<PackedMersenne31Neon>],
            scales: &[[EF; N]],
        ) {
            let items = || {
                rows.iter()
                    .zip(scales)
                    .map(|(row, &s)| (row.iter().copied(), s))
            };

            let mut expected = EF::ExtensionPacking::zero_vec(packed_width * N);
            for (row, s) in items() {
                let packed_scales = s.map(EF::ExtensionPacking::from);
                for (acc_c, r) in expected.chunks_exact_mut(N).zip(row) {
                    for (a, &ps) in acc_c.iter_mut().zip(&packed_scales) {
                        *a += ps * r;
                    }
                }
            }

            let mut got = EF::ExtensionPacking::zero_vec(packed_width * N);
            super::batched_columnwise_dot_product::<EF, _, _, N>(&mut got, items());
            let expected: Vec<EF> = EF::ExtensionPacking::to_ext_iter(expected).collect();
            let got: Vec<EF> = EF::ExtensionPacking::to_ext_iter(got).collect();
            assert_eq!(expected, got, "rows = {}, N = {N}", rows.len());
        }

        let packed_width = 5;
        for height in [0usize, 1, 2, 3, 4, 5, 6, 100] {
            let rows: Vec<Vec<PackedMersenne31Neon>> = (0..height)
                .map(|r| {
                    (0..packed_width)
                        .map(|c| packed((r * 64 + c * 4) as u32))
                        .collect()
                })
                .collect();

            let scales_m31: Vec<[Mersenne31; 2]> = (0..height)
                .map(|r| core::array::from_fn(|j| scalar((r * 2 + j) as u32)))
                .collect();
            check::<Mersenne31, 2>(packed_width, &rows, &scales_m31);

            let scales_cm31: Vec<[Complex<Mersenne31>; 2]> = (0..height)
                .map(|r| {
                    core::array::from_fn(|j| {
                        Complex::from_basis_coefficients_fn(|k| scalar((r * 4 + j * 2 + k) as u32))
                    })
                })
                .collect();
            check::<Complex<Mersenne31>, 2>(packed_width, &rows, &scales_cm31);

            let scales_qm31: Vec<[QM31; 2]> = (0..height)
                .map(|r| {
                    core::array::from_fn(|j| {
                        QM31::from_basis_coefficients_fn(|k| scalar((r * 8 + j * 4 + k) as u32))
                    })
                })
                .collect();
            check::<QM31, 2>(packed_width, &rows, &scales_qm31);
        }
    }
}
