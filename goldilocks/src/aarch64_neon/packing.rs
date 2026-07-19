use alloc::vec::Vec;
use core::arch::aarch64::{
    uint64x2_t, vaddq_u64, vandq_u64, vdupq_n_u64, vgetq_lane_u64, vsetq_lane_u64, vshrq_n_u64,
    vsubq_u64,
};
use core::fmt::Debug;
use core::iter::{Product, Sum};
use core::mem::transmute;
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

use crate::{Goldilocks, P};

const WIDTH: usize = 2;

/// Equal to `2^32 - 1 = 2^64 mod P`.
const EPSILON: u64 = Goldilocks::ORDER_U64.wrapping_neg();

/// Width-2 packed `Goldilocks` for aarch64.
///
/// `mul`, `square`, and the cubic-extension helpers use a dual-lane interleaved
/// scalar ASM block (`mul_reduce_dual_asm`); `add`, `sub`, and `neg` operate on
/// the underlying `[Goldilocks; 2]` storage directly in scalar `u64` space.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
#[repr(transparent)]
#[must_use]
pub struct PackedGoldilocksNeon(pub [Goldilocks; WIDTH]);

impl PackedGoldilocksNeon {
    #[inline]
    #[must_use]
    pub(crate) fn to_vector(self) -> uint64x2_t {
        unsafe { transmute(self) }
    }

    #[inline]
    pub(crate) fn from_vector(vector: uint64x2_t) -> Self {
        unsafe { transmute(vector) }
    }

    #[inline]
    const fn broadcast(value: Goldilocks) -> Self {
        Self([value; WIDTH])
    }
}

impl From<Goldilocks> for PackedGoldilocksNeon {
    fn from(x: Goldilocks) -> Self {
        Self::broadcast(x)
    }
}

impl Add for PackedGoldilocksNeon {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self([self.0[0] + rhs.0[0], self.0[1] + rhs.0[1]])
    }
}

impl Sub for PackedGoldilocksNeon {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self([self.0[0] - rhs.0[0], self.0[1] - rhs.0[1]])
    }
}

impl Neg for PackedGoldilocksNeon {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self([-self.0[0], -self.0[1]])
    }
}

impl Mul for PackedGoldilocksNeon {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self::from_vector(mul(self.to_vector(), rhs.to_vector()))
    }
}

impl_add_assign!(PackedGoldilocksNeon);
impl_sub_assign!(PackedGoldilocksNeon);
impl_mul_methods!(PackedGoldilocksNeon);
ring_sum!(PackedGoldilocksNeon);
impl_rng!(PackedGoldilocksNeon);

impl PrimeCharacteristicRing for PackedGoldilocksNeon {
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
    fn halve(&self) -> Self {
        Self::from_vector(halve(self.to_vector()))
    }

    #[inline]
    fn dot_product<const N: usize>(lhs: &[Self; N], rhs: &[Self; N]) -> Self {
        Self::from_fn(|lane| {
            let lhs_lane: [Goldilocks; N] = core::array::from_fn(|i| lhs[i].as_slice()[lane]);
            let rhs_lane: [Goldilocks; N] = core::array::from_fn(|i| rhs[i].as_slice()[lane]);
            Goldilocks::dot_product(&lhs_lane, &rhs_lane)
        })
    }

    #[inline]
    fn square(&self) -> Self {
        Self::from_vector(square(self.to_vector()))
    }

    #[inline]
    fn zero_vec(len: usize) -> Vec<Self> {
        unsafe { reconstitute_from_base(Goldilocks::zero_vec(len * WIDTH)) }
    }
}

impl InjectiveMonomial<7> for PackedGoldilocksNeon {}

impl PermutationMonomial<7> for PackedGoldilocksNeon {
    fn injective_exp_root_n(&self) -> Self {
        exp_10540996611094048183(*self)
    }
}

impl_add_base_field!(PackedGoldilocksNeon, Goldilocks);
impl_sub_base_field!(PackedGoldilocksNeon, Goldilocks);
impl_mul_base_field!(PackedGoldilocksNeon, Goldilocks);
impl_div_methods!(PackedGoldilocksNeon, Goldilocks);
impl_packed_field_div!(PackedGoldilocksNeon);
impl_sum_prod_base_field!(PackedGoldilocksNeon, Goldilocks);

impl Algebra<Goldilocks> for PackedGoldilocksNeon {
    // With the delayed-reduction dot product below, one 192-bit reduction is
    // amortized over the whole chunk, so larger chunks win.
    #[cfg(target_feature = "sve2")]
    const BATCHED_LC_CHUNK: usize = 64;
    // Benchmarked on AArch64 NEON: chunk=2 ≈ 182ns, chunk=4 ≈ 198ns, chunk=8 ≈ 221ns.
    #[cfg(not(target_feature = "sve2"))]
    const BATCHED_LC_CHUNK: usize = 2;

    #[inline]
    fn mixed_dot_product<const N: usize>(a: &[Self; N], f: &[Goldilocks; N]) -> Self {
        #[cfg(target_feature = "sve2")]
        {
            sve2_mixed_dot_delayed(a, f)
        }
        #[cfg(not(target_feature = "sve2"))]
        Self::from_fn(|lane| {
            let a_lane: [Goldilocks; N] = core::array::from_fn(|i| a[i].as_slice()[lane]);
            Goldilocks::dot_product(&a_lane, f)
        })
    }
}

/// `Σ a[i]·f[i]` with delayed reduction: 128-bit products accumulate unreduced
/// per lane, and a single 192-bit reduction runs at the end. Coefficients
/// broadcast via `ld1rd`; products via SVE2 vector 64-bit `mul`/`umulh`.
///
/// The loop keeps four per-lane accumulators: `lo` (Σ products mod 2^64), `hi`
/// (Σ high halves mod 2^64), and exact wrap counts for each (`lo_w`, `hi_w`;
/// both ≤ N, so they cannot themselves wrap). Folding `lo_w` into `hi` with
/// checked scalar arithmetic afterwards makes the accumulation exact for any
/// `N` and any inputs — no probabilistic carry argument.
///
/// All loads, stores, and pointer steps are fixed to the low two lanes
/// (`ptrue vl2`, 16-byte steps), so the routine is correct at any SVE vector
/// length; wider lanes are loaded as zero and never stored.
#[cfg(target_feature = "sve2")]
#[inline]
fn sve2_mixed_dot_delayed<const N: usize>(
    a: &[PackedGoldilocksNeon; N],
    f: &[Goldilocks; N],
) -> PackedGoldilocksNeon {
    if N == 0 {
        return PackedGoldilocksNeon::ZERO;
    }
    let mut acc = [0u64; 8]; // [lo, hi, hi_wraps, lo_wraps] × 2 lanes
    unsafe {
        core::arch::asm!(
            "ptrue p7.d, vl2",
            "dup   z0.d, #0",
            "dup   z1.d, #0",
            "dup   z2.d, #0",
            "dup   z3.d, #0",
            "dup   z31.d, #1",
            "2:",
            "ld1d  {{ z4.d }}, p7/z, [{ap}]",
            "ld1rd {{ z5.d }}, p7/z, [{fp}]",
            "mul   z6.d, z4.d, z5.d",
            "umulh z7.d, z4.d, z5.d",
            "add   z0.d, z0.d, z6.d",
            "cmplo p1.d, p7/z, z0.d, z6.d",
            "add   z3.d, p1/m, z3.d, z31.d",
            "add   z1.d, z1.d, z7.d",
            "cmplo p2.d, p7/z, z1.d, z7.d",
            "add   z2.d, p2/m, z2.d, z31.d",
            "add   {ap}, {ap}, #16",
            "add   {fp}, {fp}, #8",
            "subs  {cnt}, {cnt}, #1",
            "b.ne  2b",
            "st1d  {{ z0.d }}, p7, [{op}]",
            "add   {op}, {op}, #16",
            "st1d  {{ z1.d }}, p7, [{op}]",
            "add   {op}, {op}, #16",
            "st1d  {{ z2.d }}, p7, [{op}]",
            "add   {op}, {op}, #16",
            "st1d  {{ z3.d }}, p7, [{op}]",
            ap = inout(reg) a.as_ptr() as *const u64 => _,
            fp = inout(reg) f.as_ptr() as *const u64 => _,
            op = inout(reg) acc.as_mut_ptr() => _,
            cnt = inout(reg) N => _,
            out("v0") _, out("v1") _, out("v2") _, out("v3") _, out("v4") _,
            out("v5") _, out("v6") _, out("v7") _, out("v31") _,
            out("p1") _, out("p2") _, out("p7") _,
            options(nostack),
        );
    }
    PackedGoldilocksNeon::from_fn(|lane| {
        let (lo, hi_raw) = (acc[lane], acc[2 + lane]);
        let (hi_wraps, lo_wraps) = (acc[4 + lane], acc[6 + lane]);
        let (hi, c) = hi_raw.overflowing_add(lo_wraps);
        reduce192(lo, hi, hi_wraps + c as u64)
    })
}

/// Reduce `carry·2^128 + hi·2^64 + lo` to a Goldilocks element, using
/// `2^64 ≡ ε` and `2^128 ≡ ε² ≡ P − 2^32 (mod P)` — two shift-epsilon folds,
/// no 128-bit division.
#[cfg(target_feature = "sve2")]
#[inline]
fn reduce192(lo: u64, hi: u64, carry: u64) -> Goldilocks {
    const EPS2_MOD_P: u64 = P - (1 << 32); // ε² mod P
    // Fold hi: t ≡ hi·2^64 + lo (mod P), t < 2^96.
    let t = (hi as u128) * (EPSILON as u128) + lo as u128;
    let (t_lo, t_hi) = (t as u64, (t >> 64) as u64); // t_hi ≤ ε
    // r ≡ t_hi·2^64 + t_lo (mod P); t_hi·ε ≤ ε², so the wrap fix cannot re-wrap.
    let (mut r, c) = t_lo.overflowing_add(t_hi * EPSILON);
    if c {
        r = r.wrapping_add(EPSILON);
    }
    // Fold carry the same way via carry·2^128 ≡ carry·(ε² mod P). Here
    // u_hi ≤ carry, so the wrap fix cannot re-wrap while carry < 2^32 − 1
    // (callers pass carry ≤ N + 1, a slice length).
    debug_assert!(carry < (1 << 32) - 1, "reduce192 carry bound violated");
    let u = (carry as u128) * (EPS2_MOD_P as u128) + r as u128;
    let (u_lo, u_hi) = (u as u64, (u >> 64) as u64);
    let (mut v, c2) = u_lo.overflowing_add(u_hi * EPSILON);
    if c2 {
        v = v.wrapping_add(EPSILON);
    }
    // v is a valid, not necessarily canonical, representative.
    Goldilocks::new(v)
}

impl_packed_value!(PackedGoldilocksNeon, Goldilocks, WIDTH);

unsafe impl PackedField for PackedGoldilocksNeon {
    type Scalar = Goldilocks;
}

/// Interleave two 64-bit vectors at the element level.
/// For block_len=1: [a0, a1] x [b0, b1] -> [a0, b0], [a1, b1]
#[inline]
pub fn interleave_u64(v0: uint64x2_t, v1: uint64x2_t) -> (uint64x2_t, uint64x2_t) {
    unsafe {
        let a0 = vgetq_lane_u64::<0>(v0);
        let a1 = vgetq_lane_u64::<1>(v0);
        let b0 = vgetq_lane_u64::<0>(v1);
        let b1 = vgetq_lane_u64::<1>(v1);

        // r0 = [a0, b0], r1 = [a1, b1]
        let r0 = vsetq_lane_u64::<1>(b0, vsetq_lane_u64::<0>(a0, vdupq_n_u64(0)));
        let r1 = vsetq_lane_u64::<1>(b1, vsetq_lane_u64::<0>(a1, vdupq_n_u64(0)));

        (r0, r1)
    }
}

unsafe impl PackedFieldPow2 for PackedGoldilocksNeon {
    fn interleave(&self, other: Self, block_len: usize) -> (Self, Self) {
        let (v0, v1) = (self.to_vector(), other.to_vector());
        let (res0, res1) = match block_len {
            1 => interleave_u64(v0, v1),
            2 => (v0, v1),
            _ => panic!("unsupported block length"),
        };
        (Self::from_vector(res0), Self::from_vector(res1))
    }
}

/// Halve a vector of Goldilocks field elements.
#[inline(always)]
pub(crate) fn halve(input: uint64x2_t) -> uint64x2_t {
    unsafe {
        let one = vdupq_n_u64(1);
        let zero = vdupq_n_u64(0);
        let half = vdupq_n_u64(P.div_ceil(2));

        let least_bit = vandq_u64(input, one);
        let t = vshrq_n_u64::<1>(input);
        // neg_least_bit is 0 or -1 (all bits 1)
        let neg_least_bit = vsubq_u64(zero, least_bit);
        let maybe_half = vandq_u64(half, neg_least_bit);
        vaddq_u64(t, maybe_half)
    }
}

/// Goldilocks modular multiplication staying in the vector domain (SVE2).
///
/// The 128-bit products come from SVE2 unpredicated `MUL.d`/`UMULH.d`, which
/// NEON lacks; the low 128 bits of the `z` registers alias the `v` registers,
/// so no lane extraction is needed. The reduction runs on plain NEON
/// intrinsics, letting LLVM schedule it across neighboring multiplications.
#[cfg(target_feature = "sve2")]
#[inline]
fn mul(x: uint64x2_t, y: uint64x2_t) -> uint64x2_t {
    unsafe {
        use core::arch::aarch64::{vcgtq_u64, vmlal_n_u32, vmovn_u64, vsraq_n_u64};
        use core::arch::asm;
        let lo: uint64x2_t;
        let hi: uint64x2_t;
        asm!(
            "mul   z2.d, z0.d, z1.d",
            "umulh z3.d, z0.d, z1.d",
            in("v0") x,
            in("v1") y,
            out("v2") lo,
            out("v3") hi,
            options(pure, nomem, nostack),
        );

        // Reduction with the plonky2-NEON idioms: `vmlal_n_u32` folds hi_lo·ε into
        // the accumulator in one op; `vsraq` (mask ≫ 32 = ε) applies each rare
        // wraparound correction in one op. ~9 vector ops vs 13 for the naive form.
        // t1 = lo − hi_hi, minus ε on per-lane borrow (2^64 ≡ ε mod P).
        let hi_hi = vshrq_n_u64::<32>(hi);
        let borrow = vcgtq_u64(hi_hi, lo);
        let t1 = vsubq_u64(vsubq_u64(lo, hi_hi), vshrq_n_u64::<32>(borrow));
        // res = t1 + hi_lo·ε via widening multiply-accumulate.
        let hi_lo32 = vmovn_u64(hi);
        let res = vmlal_n_u32(t1, hi_lo32, EPSILON as u32);
        // += ε on per-lane overflow (res wrapped iff res < t1).
        let ovf = vcgtq_u64(t1, res);
        vsraq_n_u64::<32>(res, ovf)
    }
}

/// Goldilocks modular multiplication using interleaved dual-lane ASM.
#[cfg(not(target_feature = "sve2"))]
#[inline]
fn mul(x: uint64x2_t, y: uint64x2_t) -> uint64x2_t {
    unsafe {
        let x0 = vgetq_lane_u64::<0>(x);
        let x1 = vgetq_lane_u64::<1>(x);
        let y0 = vgetq_lane_u64::<0>(y);
        let y1 = vgetq_lane_u64::<1>(y);

        let (res_0, res_1) = mul_reduce_dual_asm(x0, y0, x1, y1);

        transmute([res_0, res_1])
    }
}

/// Interleaved dual-lane multiplication and reduction using scalar ASM.
/// Uses shift-based EPSILON multiplication: hi_lo * EPSILON = (hi_lo << 32) - hi_lo
#[cfg(not(target_feature = "sve2"))]
#[inline(always)]
unsafe fn mul_reduce_dual_asm(a0: u64, b0: u64, a1: u64, b1: u64) -> (u64, u64) {
    use core::arch::asm;
    let result0: u64;
    let result1: u64;

    unsafe {
        asm!(
            // Compute both 128-bit products (interleaved for ILP)
            "mul   {lo0}, {a0}, {b0}",
            "mul   {lo1}, {a1}, {b1}",
            "umulh {hi0}, {a0}, {b0}",
            "umulh {hi1}, {a1}, {b1}",

            // hi_hi = hi >> 32
            "lsr   {hi_hi0}, {hi0}, #32",
            "lsr   {hi_hi1}, {hi1}, #32",

            // tmp = lo - hi_hi (with borrow handling)
            "subs  {tmp0}, {lo0}, {hi_hi0}",
            "csetm {adj0:w}, cc",
            "subs  {tmp1}, {lo1}, {hi_hi1}",
            "csetm {adj1:w}, cc",
            "sub   {tmp0}, {tmp0}, {adj0}",
            "sub   {tmp1}, {tmp1}, {adj1}",

            // hi_lo = hi & EPSILON
            "and   {hi_lo0}, {hi0}, {epsilon}",
            "and   {hi_lo1}, {hi1}, {epsilon}",

            // hi_lo_eps = (hi_lo << 32) - hi_lo (avoids multiply)
            "lsl   {t0}, {hi_lo0}, #32",
            "lsl   {t1}, {hi_lo1}, #32",
            "sub   {hi_lo_eps0}, {t0}, {hi_lo0}",
            "sub   {hi_lo_eps1}, {t1}, {hi_lo1}",

            // result = tmp + hi_lo_eps (with overflow handling)
            "adds  {result0}, {tmp0}, {hi_lo_eps0}",
            "csetm {adj0:w}, cs",
            "adds  {result1}, {tmp1}, {hi_lo_eps1}",
            "csetm {adj1:w}, cs",
            "add   {result0}, {result0}, {adj0}",
            "add   {result1}, {result1}, {adj1}",

            a0 = in(reg) a0,
            b0 = in(reg) b0,
            a1 = in(reg) a1,
            b1 = in(reg) b1,
            epsilon = in(reg) EPSILON,
            lo0 = out(reg) _,
            lo1 = out(reg) _,
            hi0 = out(reg) _,
            hi1 = out(reg) _,
            hi_hi0 = out(reg) _,
            hi_hi1 = out(reg) _,
            tmp0 = out(reg) _,
            tmp1 = out(reg) _,
            hi_lo0 = out(reg) _,
            hi_lo1 = out(reg) _,
            t0 = out(reg) _,
            t1 = out(reg) _,
            hi_lo_eps0 = out(reg) _,
            hi_lo_eps1 = out(reg) _,
            adj0 = out(reg) _,
            adj1 = out(reg) _,
            result0 = out(reg) result0,
            result1 = out(reg) result1,
            options(pure, nomem, nostack),
        );
    }

    (result0, result1)
}

/// Goldilocks modular square — SVE2 path shares the vector-domain mul.
#[cfg(target_feature = "sve2")]
#[inline]
fn square(x: uint64x2_t) -> uint64x2_t {
    mul(x, x)
}

/// Goldilocks modular square using interleaved dual-lane ASM.
#[cfg(not(target_feature = "sve2"))]
#[inline]
fn square(x: uint64x2_t) -> uint64x2_t {
    unsafe {
        let x0 = vgetq_lane_u64::<0>(x);
        let x1 = vgetq_lane_u64::<1>(x);

        let (res_0, res_1) = mul_reduce_dual_asm(x0, x0, x1, x1);

        transmute([res_0, res_1])
    }
}

#[cfg(test)]
mod tests {
    use p3_field_testing::test_packed_field;

    use super::{Goldilocks, PackedGoldilocksNeon, WIDTH};

    const SPECIAL_VALS: [Goldilocks; WIDTH] =
        Goldilocks::new_array([0xFFFF_FFFF_0000_0000, 0xFFFF_FFFF_FFFF_FFFF]);

    const ZEROS: PackedGoldilocksNeon = PackedGoldilocksNeon(Goldilocks::new_array([
        0x0000_0000_0000_0000,
        0xFFFF_FFFF_0000_0001, // = P, canonicalizes to 0
    ]));

    const ONES: PackedGoldilocksNeon = PackedGoldilocksNeon(Goldilocks::new_array([
        0x0000_0000_0000_0001,
        0xFFFF_FFFF_0000_0002, // = P + 1, canonicalizes to 1
    ]));

    test_packed_field!(
        crate::PackedGoldilocksNeon,
        &[super::ZEROS],
        &[super::ONES],
        crate::PackedGoldilocksNeon(super::SPECIAL_VALS)
    );
}

#[cfg(test)]
mod mixed_dot_tests {
    use p3_field::PrimeField64;
    use proptest::prelude::*;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::*;

    /// Raw value patterns stressing the accumulator and the final reduction:
    /// field extremes, the epsilon window, and non-canonical values up to
    /// u64::MAX. `mixed_dot_product` accepts any u64 residues.
    const EDGE: [u64; 8] = [0, 1, (1 << 32) - 1, 1 << 32, 1 << 63, P - 1, P, u64::MAX];

    /// Reference: canonicalize inputs, accumulate mod P in u128.
    fn dot_ref<const N: usize>(
        a: &[PackedGoldilocksNeon; N],
        f: &[Goldilocks; N],
        lane: usize,
    ) -> u64 {
        let mut acc: u128 = 0;
        for i in 0..N {
            let ai = a[i].as_slice()[lane].as_canonical_u64() as u128;
            let fi = f[i].as_canonical_u64() as u128;
            acc = (acc + ai * fi) % (P as u128);
        }
        acc as u64
    }

    fn check_mixed_dot<const N: usize>(
        a_raw: &dyn Fn(usize, usize) -> u64,
        f_raw: &dyn Fn(usize) -> u64,
    ) {
        let a: [PackedGoldilocksNeon; N] = core::array::from_fn(|i| {
            PackedGoldilocksNeon(Goldilocks::new_array([a_raw(i, 0), a_raw(i, 1)]))
        });
        let f: [Goldilocks; N] = core::array::from_fn(|i| Goldilocks::new(f_raw(i)));
        let got = PackedGoldilocksNeon::mixed_dot_product(&a, &f);
        for lane in 0..WIDTH {
            assert_eq!(
                got.as_slice()[lane].as_canonical_u64(),
                dot_ref(&a, &f, lane),
                "lane {lane}, N={N}"
            );
        }
    }

    #[test]
    fn mixed_dot_product_edge_values() {
        // All 64 (a, f) edge pairs in lane 0; lane 1 pinned to u64::MAX.
        check_mixed_dot::<64>(
            &|i, lane| if lane == 0 { EDGE[i / 8] } else { u64::MAX },
            &|i| EDGE[i % 8],
        );
    }

    #[test]
    fn mixed_dot_product_max_values() {
        // Maximum-magnitude accumulation: stresses both wrap counters and the
        // carry fold of the final reduction.
        check_mixed_dot::<200>(&|_, _| u64::MAX, &|_| u64::MAX);
    }

    #[test]
    fn mixed_dot_product_lengths() {
        fn case<const N: usize>(seed: u64) {
            let mut rng = SmallRng::seed_from_u64(seed);
            let a_vals: [[u64; 2]; N] = core::array::from_fn(|_| [rng.random(), rng.random()]);
            let f_vals: [u64; N] = core::array::from_fn(|_| rng.random());
            check_mixed_dot::<N>(&|i, lane| a_vals[i][lane], &|i| f_vals[i]);
        }
        // Around the chunk boundary and both parities.
        case::<0>(0);
        case::<1>(1);
        case::<2>(2);
        case::<3>(3);
        case::<5>(5);
        case::<63>(63);
        case::<64>(64);
        case::<65>(65);
        case::<129>(129);
    }

    proptest! {
        #[test]
        fn mixed_dot_product_prop(
            a in prop::array::uniform16(any::<u64>()),
            f in prop::array::uniform8(any::<u64>()),
        ) {
            let packed: [PackedGoldilocksNeon; 8] = core::array::from_fn(|i| {
                PackedGoldilocksNeon(Goldilocks::new_array([a[2 * i], a[2 * i + 1]]))
            });
            let coeffs: [Goldilocks; 8] = core::array::from_fn(|i| Goldilocks::new(f[i]));
            let got = PackedGoldilocksNeon::mixed_dot_product(&packed, &coeffs);
            for lane in 0..WIDTH {
                prop_assert_eq!(
                    got.as_slice()[lane].as_canonical_u64(),
                    dot_ref(&packed, &coeffs, lane)
                );
            }
        }
    }

    #[cfg(target_feature = "sve2")]
    #[test]
    fn reduce192_matches_reference() {
        fn reference(lo: u64, hi: u64, carry: u64) -> u64 {
            const P128: u128 = P as u128;
            let two64 = (u64::MAX as u128 + 1) % P128;
            let two128 = two64 * two64 % P128;
            let total =
                lo as u128 % P128 + (hi as u128) * two64 % P128 + (carry as u128) * two128 % P128;
            (total % P128) as u64
        }

        let mut rng = SmallRng::seed_from_u64(0x192);
        for _ in 0..100_000 {
            let (lo, hi): (u64, u64) = (rng.random(), rng.random());
            // Callers pass carry ≤ N + 1; test the documented domain boundary.
            let carry = rng.random::<u64>() % ((1 << 32) - 1);
            assert_eq!(
                reduce192(lo, hi, carry).as_canonical_u64(),
                reference(lo, hi, carry),
                "lo={lo:#x} hi={hi:#x} carry={carry:#x}"
            );
        }
        for &(lo, hi, carry) in &[
            (0, 0, 0),
            (u64::MAX, u64::MAX, (1 << 32) - 2),
            (P, P, 1),
            (u64::MAX, 0, 0),
        ] {
            assert_eq!(
                reduce192(lo, hi, carry).as_canonical_u64(),
                reference(lo, hi, carry)
            );
        }
    }
}
