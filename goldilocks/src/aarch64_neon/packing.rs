use alloc::vec::Vec;
use core::arch::aarch64::{
    uint64x2_t, vaddq_u64, vandq_u64, vcltq_u64, vdupq_n_s64, vdupq_n_u64, vgetq_lane_u64,
    vsetq_lane_u64, vshlq_u64, vshrq_n_u64, vsubq_u64,
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
    PermutationMonomial, PrimeCharacteristicRing,
};
use p3_util::reconstitute_from_base;
use rand::distr::{Distribution, StandardUniform};
use rand::{Rng, RngExt};

use super::utils::EPSILON;
use crate::{Goldilocks, P};

const WIDTH: usize = 2;

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
    fn mul_2exp_u64(&self, mut exp: u64) -> Self {
        exp %= 192;
        match exp {
            0 => *self,
            1 => self.double(),
            _ => *self * Self::broadcast(Goldilocks::power_of_two(exp)),
        }
    }

    #[inline]
    fn div_2exp_u64(&self, mut exp: u64) -> Self {
        exp %= 192;
        match exp {
            0 => *self,
            1 => self.halve(),
            2..=32 => unsafe {
                let x = self.to_vector();
                let lo = vandq_u64(x, vdupq_n_u64((1u64 << exp) - 1));
                let hi = vshlq_u64(x, vdupq_n_s64(-(exp as i64)));
                let a = vaddq_u64(hi, vshlq_u64(lo, vdupq_n_s64((32 - exp) as i64)));
                let b = vshlq_u64(lo, vdupq_n_s64((64 - exp) as i64));
                // 2^-exp = 2^(32-exp) - 2^(64-exp) mod P. Both a and b are
                // below P, so a - b > -P and one borrow correction suffices.
                let borrow = vcltq_u64(a, b);
                let correction = vandq_u64(borrow, vdupq_n_u64(EPSILON));
                Self::from_vector(vsubq_u64(vsubq_u64(a, b), correction))
            },
            _ => *self * Self::broadcast(Goldilocks::power_of_two(192 - exp)),
        }
    }

    #[inline]
    fn sum_array<const N: usize>(input: &[Self]) -> Self {
        assert_eq!(N, input.len());
        match N {
            0 => Self::ZERO,
            1 => input[0],
            2 => input[0] + input[1],
            3 => input[0] + input[1] + input[2],
            4 => (input[0] + input[1]) + (input[2] + input[3]),
            5 => Self::sum_array::<4>(&input[..4]) + Self::sum_array::<1>(&input[4..]),
            6 => Self::sum_array::<4>(&input[..4]) + Self::sum_array::<2>(&input[4..]),
            7 => Self::sum_array::<4>(&input[..4]) + Self::sum_array::<3>(&input[4..]),
            8 => Self::sum_array::<4>(&input[..4]) + Self::sum_array::<4>(&input[4..]),
            9..=63 => {
                // Keep the existing eight-term trees for short sums: carry bookkeeping
                // adds latency until enough independent work amortizes the final fold.
                let mut acc = Self::sum_array::<8>(&input[..8]);
                for i in (16..=N).step_by(8) {
                    acc += Self::sum_array::<8>(&input[(i - 8)..i]);
                }
                let tail = &input[(8 * (N / 8))..];
                match N & 7 {
                    0 => acc,
                    1 => acc + Self::sum_array::<1>(tail),
                    2 => acc + Self::sum_array::<2>(tail),
                    3 => acc + Self::sum_array::<3>(tail),
                    4 => acc + Self::sum_array::<4>(tail),
                    5 => acc + Self::sum_array::<5>(tail),
                    6 => acc + Self::sum_array::<6>(tail),
                    7 => acc + Self::sum_array::<7>(tail),
                    _ => unreachable!(),
                }
            }
            _ => {
                let mut chunks = input.chunks(64);
                let mut sum = sum_chunk(chunks.next().expect("N is nonzero"));
                for chunk in chunks {
                    sum += sum_chunk(chunk);
                }
                sum
            }
        }
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

/// Sum at most 64 raw u64 values per lane, delaying the Goldilocks fold.
#[inline]
fn sum_chunk(input: &[PackedGoldilocksNeon]) -> PackedGoldilocksNeon {
    debug_assert!((1..=64).contains(&input.len()));
    unsafe {
        use core::arch::aarch64::{vcgtq_u64, vshlq_n_u64, vsraq_n_u64};

        let zero = vdupq_n_u64(0);
        let (lo, carries) = if input.len() < 4 {
            // A short final chunk only needs one dependency chain.
            let mut lo = input[0].to_vector();
            let mut carries = zero;
            for term in &input[1..] {
                let next = vaddq_u64(lo, term.to_vector());
                carries = vsubq_u64(carries, vcgtq_u64(lo, next));
                lo = next;
            }
            (lo, carries)
        } else {
            let (groups, remainder) = input.as_chunks::<4>();
            let (first, groups) = groups.split_first().unwrap();
            let mut lo0 = first[0].to_vector();
            let mut lo1 = first[1].to_vector();
            let mut lo2 = first[2].to_vector();
            let mut lo3 = first[3].to_vector();
            let mut carries0 = zero;
            let mut carries1 = zero;
            let mut carries2 = zero;
            let mut carries3 = zero;

            // Keep four independent chains explicit, without an indexed accumulator array.
            // Subtracting each all-ones overflow mask increments its exact wrap count.
            for values in groups {
                let next0 = vaddq_u64(lo0, values[0].to_vector());
                carries0 = vsubq_u64(carries0, vcgtq_u64(lo0, next0));
                lo0 = next0;

                let next1 = vaddq_u64(lo1, values[1].to_vector());
                carries1 = vsubq_u64(carries1, vcgtq_u64(lo1, next1));
                lo1 = next1;

                let next2 = vaddq_u64(lo2, values[2].to_vector());
                carries2 = vsubq_u64(carries2, vcgtq_u64(lo2, next2));
                lo2 = next2;

                let next3 = vaddq_u64(lo3, values[3].to_vector());
                carries3 = vsubq_u64(carries3, vcgtq_u64(lo3, next3));
                lo3 = next3;
            }

            // Distribute the one-to-three remainder terms across distinct chains.
            if let Some(value) = remainder.first() {
                let next = vaddq_u64(lo0, value.to_vector());
                carries0 = vsubq_u64(carries0, vcgtq_u64(lo0, next));
                lo0 = next;
            }
            if let Some(value) = remainder.get(1) {
                let next = vaddq_u64(lo1, value.to_vector());
                carries1 = vsubq_u64(carries1, vcgtq_u64(lo1, next));
                lo1 = next;
            }
            if let Some(value) = remainder.get(2) {
                let next = vaddq_u64(lo2, value.to_vector());
                carries2 = vsubq_u64(carries2, vcgtq_u64(lo2, next));
                lo2 = next;
            }

            // Merge the exact states in a balanced tree. Include each low-word overflow
            // alongside both input carry counts, so the final count is at most len - 1.
            let lo01 = vaddq_u64(lo0, lo1);
            let carries01 = vsubq_u64(vaddq_u64(carries0, carries1), vcgtq_u64(lo0, lo01));
            let lo23 = vaddq_u64(lo2, lo3);
            let carries23 = vsubq_u64(vaddq_u64(carries2, carries3), vcgtq_u64(lo2, lo23));
            let lo = vaddq_u64(lo01, lo23);
            let carries = vsubq_u64(vaddq_u64(carries01, carries23), vcgtq_u64(lo01, lo));
            (lo, carries)
        };

        // The exact sum is lo + carries * 2^64, including noncanonical inputs.
        // There are at most 63 carries, so correction = carries * EPSILON < P.
        let correction = vsubq_u64(vshlq_n_u64::<32>(carries), carries);
        let sum = vaddq_u64(lo, correction);
        let overflow = vcgtq_u64(lo, sum);
        // On overflow, sum <= correction - 1, so sum + EPSILON <= 64 * EPSILON - 1 < P.
        // The all-ones overflow mask shifted right by 32 is exactly EPSILON.
        PackedGoldilocksNeon::from_vector(vsraq_n_u64::<32>(sum, overflow))
    }
}

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
/// per lane, and a single vector reduction runs at the end. Coefficients
/// broadcast via `ld1rd`; products via SVE2 vector 64-bit `mul`/`umulh`.
///
/// The loop keeps four per-lane accumulators: `lo` (Σ products mod 2^64), `hi`
/// (Σ high halves mod 2^64), and exact wrap counts for each (`lo_w`, `hi_w`;
/// both < the chunk length). Chunks contain at most `u32::MAX` products, so the
/// carry above bit 127 is at most `u32::MAX - 1` and its final `c << 32` fold
/// cannot overflow.
///
/// All loads and pointer steps are fixed to the low two lanes (`ptrue vl2`,
/// 16-byte packed steps and 8-byte scalar steps), so the routine is correct at
/// any SVE vector length; wider lanes are loaded as zero and ignored.
#[cfg(target_feature = "sve2")]
#[inline]
fn sve2_mixed_dot_delayed<const N: usize>(
    a: &[PackedGoldilocksNeon; N],
    f: &[Goldilocks; N],
) -> PackedGoldilocksNeon {
    sve2_mixed_dot_delayed_with_chunk_limit(a, f, u32::MAX as usize)
}

#[cfg(target_feature = "sve2")]
#[inline]
fn sve2_mixed_dot_delayed_with_chunk_limit<const N: usize>(
    a: &[PackedGoldilocksNeon; N],
    f: &[Goldilocks; N],
    chunk_limit: usize,
) -> PackedGoldilocksNeon {
    if N == 0 {
        return PackedGoldilocksNeon::ZERO;
    }
    assert!((1..=u32::MAX as usize).contains(&chunk_limit));

    if N <= chunk_limit {
        // SAFETY: Both arrays contain N readable elements and N is nonzero and
        // bounded by u32::MAX.
        return unsafe { sve2_mixed_dot_chunk(a, f) };
    }

    let mut chunks = a.chunks(chunk_limit).zip(f.chunks(chunk_limit));
    let (first_a, first_f) = chunks.next().expect("N is nonzero");
    // SAFETY: `chunks` produces equally sized, nonempty chunks no longer than
    // `chunk_limit`, which is bounded by u32::MAX above.
    let mut sum = unsafe { sve2_mixed_dot_chunk(first_a, first_f) };
    for (a_chunk, f_chunk) in chunks {
        // SAFETY: Same invariants as the first chunk.
        sum += unsafe { sve2_mixed_dot_chunk(a_chunk, f_chunk) };
    }
    sum
}

/// Accumulate one chunk of products without intermediate field reductions.
///
/// # Safety
///
/// Both slices must have the same nonzero length, at most `u32::MAX`.
#[cfg(target_feature = "sve2")]
#[inline]
unsafe fn sve2_mixed_dot_chunk(
    a: &[PackedGoldilocksNeon],
    f: &[Goldilocks],
) -> PackedGoldilocksNeon {
    let lo: uint64x2_t;
    let hi_raw: uint64x2_t;
    let hi_wraps: uint64x2_t;
    let lo_wraps: uint64x2_t;
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
            ap = inout(reg) a.as_ptr() as *const u64 => _,
            fp = inout(reg) f.as_ptr() as *const u64 => _,
            cnt = inout(reg) a.len() => _,
            out("v0") lo,
            out("v1") hi_raw,
            out("v2") hi_wraps,
            out("v3") lo_wraps,
            out("v4") _,
            out("v5") _, out("v6") _, out("v7") _, out("v31") _,
            out("p1") _, out("p2") _, out("p7") _,
            options(readonly, nostack),
        );
    }
    PackedGoldilocksNeon::from_vector(reduce_sve2_dot_accumulators(lo, hi_raw, hi_wraps, lo_wraps))
}

/// Reduce the four exact accumulator limbs for one bounded SVE2 dot chunk.
///
/// The caller must ensure the resulting top carry is at most
/// `u32::MAX - 1`; `sve2_mixed_dot_chunk` establishes this via its length cap.
#[cfg(any(target_feature = "sve2", test))]
#[inline]
fn reduce_sve2_dot_accumulators(
    lo: uint64x2_t,
    hi_raw: uint64x2_t,
    hi_wraps: uint64x2_t,
    lo_wraps: uint64x2_t,
) -> uint64x2_t {
    unsafe {
        use core::arch::aarch64::vcgtq_u64;

        // Fold the low-word wraps into the high word. A true comparison mask
        // is all ones, so subtracting it adds the possible extra top carry.
        let hi = vaddq_u64(hi_raw, lo_wraps);
        let extra = vcgtq_u64(hi_raw, hi);
        let carry = vsubq_u64(hi_wraps, extra);

        let reduced = reduce128_vector(lo, hi);
        // 2^128 ≡ -2^32 (mod P). The chunk bound makes this correction < P.
        let correction = core::arch::aarch64::vshlq_n_u64::<32>(carry);
        let borrow = vcgtq_u64(correction, reduced);
        vsubq_u64(vsubq_u64(reduced, correction), vshrq_n_u64::<32>(borrow))
    }
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

        reduce128_vector(lo, hi)
    }
}

/// Reduce `hi·2^64 + lo` using the established vector Goldilocks fold.
#[cfg(any(target_feature = "sve2", test))]
#[inline(always)]
fn reduce128_vector(lo: uint64x2_t, hi: uint64x2_t) -> uint64x2_t {
    unsafe {
        use core::arch::aarch64::{vcgtq_u64, vmlal_n_u32, vmovn_u64, vsraq_n_u64};

        // `vmlal_n_u32` folds hi_lo·ε into the accumulator in one op; `vsraq`
        // (mask ≫ 32 = ε) applies each rare wraparound correction in one op.
        let hi_hi = vshrq_n_u64::<32>(hi);
        let borrow = vcgtq_u64(hi_hi, lo);
        let t1 = vsubq_u64(vsubq_u64(lo, hi_hi), vshrq_n_u64::<32>(borrow));
        let hi_lo32 = vmovn_u64(hi);
        let res = vmlal_n_u32(t1, hi_lo32, EPSILON as u32);
        let overflow = vcgtq_u64(t1, res);
        vsraq_n_u64::<32>(res, overflow)
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

            // lo -= hi >> 32, minus EPSILON on borrow.
            "subs  {lo0}, {lo0}, {hi0}, lsr #32",
            "csetm {adj0:w}, cc",
            "subs  {lo1}, {lo1}, {hi1}, lsr #32",
            "csetm {adj1:w}, cc",
            "sub   {lo0}, {lo0}, {adj0}",
            "sub   {lo1}, {lo1}, {adj1}",

            // Zero-extend hi_lo without another input register for EPSILON.
            "mov   {hi0:w}, {hi0:w}",
            "mov   {hi1:w}, {hi1:w}",

            // hi_lo_eps = (hi_lo << 32) - hi_lo (avoids multiply)
            "lsl   {adj0}, {hi0}, #32",
            "lsl   {adj1}, {hi1}, #32",
            "sub   {hi0}, {adj0}, {hi0}",
            "sub   {hi1}, {adj1}, {hi1}",

            // result = lo + hi_lo_eps (with overflow handling)
            "adds  {lo0}, {lo0}, {hi0}",
            "csetm {adj0:w}, cs",
            "adds  {lo1}, {lo1}, {hi1}",
            "csetm {adj1:w}, cs",
            "add   {lo0}, {lo0}, {adj0}",
            "add   {lo1}, {lo1}, {adj1}",

            a0 = in(reg) a0,
            b0 = in(reg) b0,
            a1 = in(reg) a1,
            b1 = in(reg) b1,
            lo0 = out(reg) result0,
            lo1 = out(reg) result1,
            hi0 = out(reg) _,
            hi1 = out(reg) _,
            adj0 = out(reg) _,
            adj1 = out(reg) _,
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
    use p3_field::PrimeCharacteristicRing;
    use p3_field_testing::test_packed_field;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::{Goldilocks, PackedGoldilocksNeon, WIDTH};

    const MUL_EDGE: [u64; 11] = [
        0,
        1,
        2,
        0xFFFF_FFFE,
        0xFFFF_FFFF,
        0x1_0000_0000,
        0x8000_0000_0000_0000,
        super::P - 1,
        super::P,
        super::P + 1,
        u64::MAX,
    ];

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

    fn mul_reduce_reference(a: u64, b: u64) -> u64 {
        let product = (a as u128) * (b as u128);
        let lo = product as u64;
        let hi = (product >> 64) as u64;

        let (lo, borrow) = lo.overflowing_sub(hi >> 32);
        let lo = lo.wrapping_sub(u64::from(borrow) * super::EPSILON);
        let hi_lo = hi & 0xFFFF_FFFF;
        let hi_lo_epsilon = (hi_lo << 32) - hi_lo;
        let (result, overflow) = lo.overflowing_add(hi_lo_epsilon);
        result.wrapping_add(u64::from(overflow) * super::EPSILON)
    }

    fn check_mul_and_square(a: [u64; WIDTH], b: [u64; WIDTH]) {
        let lhs = PackedGoldilocksNeon(Goldilocks::new_array(a));
        let rhs = PackedGoldilocksNeon(Goldilocks::new_array(b));

        let mul = lhs * rhs;
        let square = lhs.square();
        for lane in 0..WIDTH {
            let mul_raw = mul.0[lane].value;
            let square_raw = square.0[lane].value;
            let expected_mul = mul_reduce_reference(a[lane], b[lane]);
            let expected_square = mul_reduce_reference(a[lane], a[lane]);
            assert_eq!(
                mul_raw, expected_mul,
                "mul lane {lane}: a={:#x}, b={:#x}",
                a[lane], b[lane]
            );
            assert_eq!(
                square_raw, expected_square,
                "square lane {lane}: a={:#x}",
                a[lane]
            );
            assert_eq!(
                mul_raw % super::P,
                ((a[lane] as u128 * b[lane] as u128) % super::P as u128) as u64,
                "mul residue lane {lane}: a={:#x}, b={:#x}",
                a[lane],
                b[lane]
            );
            assert_eq!(
                square_raw % super::P,
                ((a[lane] as u128 * a[lane] as u128) % super::P as u128) as u64,
                "square residue lane {lane}: a={:#x}",
                a[lane]
            );
        }
    }

    #[test]
    fn sum_chunk_carry_boundaries() {
        for len in [1, 2, 3, 4, 9, 10, 11, 15, 16, 31, 32, 63, 64] {
            for full_terms in 0..=len {
                let input = (0..len)
                    .map(|i| {
                        PackedGoldilocksNeon(Goldilocks::new_array([
                            if i < full_terms { u64::MAX } else { 0 },
                            if i < full_terms { 1 } else { u64::MAX },
                        ]))
                    })
                    .collect::<alloc::vec::Vec<_>>();
                let actual = super::sum_chunk(&input);
                for lane in 0..WIDTH {
                    let expected = input.iter().fold(0u128, |sum, term| {
                        (sum + u128::from(term.0[lane].value)) % u128::from(super::P)
                    }) as u64;
                    assert_eq!(
                        actual.0[lane].value % super::P,
                        expected,
                        "len={len}, full_terms={full_terms}, lane={lane}"
                    );
                }
            }
        }
    }

    #[test]
    fn mul_and_square_full_range_edges() {
        for &a0 in &MUL_EDGE {
            for &b0 in &MUL_EDGE {
                for &a1 in &MUL_EDGE {
                    for &b1 in &MUL_EDGE {
                        check_mul_and_square([a0, a1], [b0, b1]);
                    }
                }
            }
        }
    }

    #[test]
    fn mul_and_square_full_range_random() {
        let mut rng = SmallRng::seed_from_u64(0xD0A1_1A0E);
        for _ in 0..100_000 {
            check_mul_and_square(
                [rng.random::<u64>(), rng.random::<u64>()],
                [rng.random::<u64>(), rng.random::<u64>()],
            );
        }
    }
}

#[cfg(test)]
mod mixed_dot_tests {
    use p3_field::PrimeField64;
    use proptest::prelude::*;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::super::utils::tests::EDGE;
    use super::*;

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
        case::<4>(4);
        case::<5>(5);
        case::<6>(6);
        case::<16>(16);
        case::<63>(63);
        case::<64>(64);
        case::<65>(65);
        case::<129>(129);
    }

    fn reduce_accumulators_ref(lo: u64, hi_raw: u64, hi_wraps: u64, lo_wraps: u64) -> u64 {
        const P128: u128 = P as u128;
        let two64 = (u64::MAX as u128 + 1) % P128;
        let two128 = two64 * two64 % P128;
        let limbs = [
            lo as u128 % P128,
            (hi_raw as u128 % P128) * two64 % P128,
            (lo_wraps as u128 % P128) * two64 % P128,
            (hi_wraps as u128 % P128) * two128 % P128,
        ];
        limbs.into_iter().fold(0, |sum, limb| (sum + limb) % P128) as u64
    }

    fn check_vector_reducer(
        lo: [u64; WIDTH],
        hi_raw: [u64; WIDTH],
        hi_wraps: [u64; WIDTH],
        lo_wraps: [u64; WIDTH],
    ) {
        let got = PackedGoldilocksNeon::from_vector(reduce_sve2_dot_accumulators(
            unsafe { core::mem::transmute::<[u64; WIDTH], uint64x2_t>(lo) },
            unsafe { core::mem::transmute::<[u64; WIDTH], uint64x2_t>(hi_raw) },
            unsafe { core::mem::transmute::<[u64; WIDTH], uint64x2_t>(hi_wraps) },
            unsafe { core::mem::transmute::<[u64; WIDTH], uint64x2_t>(lo_wraps) },
        ));
        for lane in 0..WIDTH {
            assert_eq!(
                got.as_slice()[lane].as_canonical_u64(),
                reduce_accumulators_ref(lo[lane], hi_raw[lane], hi_wraps[lane], lo_wraps[lane]),
                "lane {lane}: lo={:#x} hi_raw={:#x} hi_wraps={:#x} lo_wraps={:#x}",
                lo[lane],
                hi_raw[lane],
                hi_wraps[lane],
                lo_wraps[lane]
            );
        }
    }

    #[test]
    fn sve2_dot_reducer_matches_weighted_limb_reference() {
        const MAX_CARRY: u64 = (1 << 32) - 2;
        for &lo in &EDGE {
            for &hi in &EDGE {
                for &carry in &[0, 1, MAX_CARRY] {
                    check_vector_reducer([lo, hi], [hi, lo], [carry, carry], [0, 0]);
                }
            }
        }

        let mut rng = SmallRng::seed_from_u64(0x5E2_D07);
        for _ in 0..100_000 {
            check_vector_reducer(
                [rng.random(), rng.random()],
                [rng.random(), rng.random()],
                [
                    (rng.random::<u32>() as u64) % MAX_CARRY,
                    (rng.random::<u32>() as u64) % MAX_CARRY,
                ],
                [
                    (rng.random::<u32>() as u64) % (MAX_CARRY + 1),
                    (rng.random::<u32>() as u64) % (MAX_CARRY + 1),
                ],
            );
        }
    }

    #[test]
    fn sve2_dot_reducer_handles_hi_fold_overflow() {
        const MAX_CARRY: u64 = (1 << 32) - 2;
        check_vector_reducer(
            [0, u64::MAX],
            [u64::MAX, u64::MAX - 5],
            [0, MAX_CARRY - 1],
            [1, 10],
        );
    }

    #[cfg(target_feature = "sve2")]
    #[test]
    fn sve2_mixed_dot_forced_small_chunks_match_reference() {
        fn case<const N: usize>(seed: u64) {
            let mut rng = SmallRng::seed_from_u64(seed);
            let a: [PackedGoldilocksNeon; N] = core::array::from_fn(|_| {
                PackedGoldilocksNeon(Goldilocks::new_array([rng.random(), rng.random()]))
            });
            let f: [Goldilocks; N] = core::array::from_fn(|_| Goldilocks::new(rng.random()));
            let got = sve2_mixed_dot_delayed_with_chunk_limit(&a, &f, 3);
            for lane in 0..WIDTH {
                assert_eq!(
                    got.as_slice()[lane].as_canonical_u64(),
                    dot_ref(&a, &f, lane),
                    "lane {lane}, N={N}"
                );
            }
        }

        case::<0>(0);
        case::<1>(1);
        case::<2>(2);
        case::<3>(3);
        case::<4>(4);
        case::<7>(7);
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
}
