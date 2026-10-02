//! Sums of products in the cubic extension `y^3 + y + 1` of `GF(2^64)`, one term per wide register.
//!
//! On a core with a full 512-bit datapath, a carryless multiply costs the same at every width.
//!
//! So a term packs all six Karatsuba products into one 512-bit register pair: two multiplies, not six.
//!
//! A multiple by a coefficient packs its three products into 256 bits: two multiplies, not three.
//!
//! Moving the coordinates across 128-bit lanes lengthens the latency of one product.
//!
//! A sum pays that once, so only sums take this route.

use core::arch::x86_64::{
    __m256i, __m512i, _mm_loadl_epi64, _mm_loadu_si128, _mm_storel_epi64, _mm_storeu_si128,
    _mm256_castsi128_si256, _mm256_castsi256_si128, _mm256_extracti128_si256,
    _mm256_inserti128_si256, _mm256_set1_epi64x, _mm512_castsi256_si512, _mm512_castsi512_si256,
    _mm512_extracti64x4_epi64, _mm512_maskz_permutexvar_epi64, _mm512_permutex2var_epi64,
    _mm512_permutexvar_epi64, _mm512_set_epi64,
};

use crate::clmul::wide::{HIGH_BY_LOW, LOW_BY_LOW, Lanes64, Wide};

/// Reads `[a_0, a_1, a_2, 0]` in the 16 + 8 byte shape a stored result takes.
///
/// So a result forwards straight into the next product.
#[inline(always)]
fn load(a: &[u64; 3]) -> __m256i {
    // SAFETY: the array is 24 readable bytes, and both loads are the unaligned forms.
    //
    // The first reads bytes 0 to 15, the second bytes 16 to 23 and zeroes the quadword above.
    unsafe {
        let pair = _mm_loadu_si128(a.as_ptr().cast());
        let last = _mm_loadl_epi64(a[2..].as_ptr().cast());
        _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(pair), last)
    }
}

/// Writes the three low quadwords back in the same 16 + 8 byte shape.
#[inline(always)]
fn store(r: __m256i) -> [u64; 3] {
    let mut out = [0u64; 3];
    // SAFETY: the array is 24 writable bytes, and both stores are the unaligned forms.
    unsafe {
        _mm_storeu_si128(out.as_mut_ptr().cast(), _mm256_castsi256_si128(r));
        _mm_storel_epi64(
            out[2..].as_mut_ptr().cast(),
            _mm256_extracti128_si256::<1>(r),
        );
    }
    out
}

/// The eight quadwords one operand feeds the two multiplies, in slot order.
///
/// Slot `2k` meets its partner in the even multiply, slot `2k + 1` in the odd one.
///
/// ```text
///     slot      0      1        2     3     4        5        6        7
///     operand   a_2    a_0+a_1  a_1   a_0   a_1+a_2  a_1+a_2  a_0+a_2  a_1
///     product   c_2    d_01     c_1   c_0   d_12     d_12     d_02     c_1
/// ```
///
/// With `c_i = a_i b_i` and `d_ij = (a_i + a_j)(b_i + b_j)`, the halves then line up:
///
/// ```text
///     low half + high half  =  [ c_2 + d_12,  d_01 + d_12,  c_1 + d_02,  c_0 + c_1 ]
///     r                     =  [ c_0 + c_1 + c_2 + d_12,  c_0 + d_01 + d_12,  c_0 + c_1 + d_02 ]
/// ```
///
/// So the output is that sum plus `[ slot 3 of the sum, c_0, c_0 ]`, one two-source permute.
#[inline(always)]
fn operand(a: __m256i) -> __m512i {
    // SAFETY: `avx512f` is enabled for this module.
    //
    // The permutes read only the four defined quadwords of the widened register.
    unsafe {
        let a = _mm512_castsi256_si512(a);
        // The single coordinates, in slot order.
        //
        // The index vector lists the highest slot first.
        let base = _mm512_permutexvar_epi64(_mm512_set_epi64(1, 0, 1, 1, 0, 1, 0, 2), a);
        // The second summand of slots 1, 4, 5 and 6, zero elsewhere.
        let summand = _mm512_maskz_permutexvar_epi64(
            0b0111_0010,
            _mm512_set_epi64(0, 2, 2, 2, 0, 0, 1, 0),
            a,
        );
        base.xor(summand)
    }
}

/// The three coordinates from the eight reduced slot products.
#[inline(always)]
fn combine(slots: __m512i) -> __m256i {
    // SAFETY: `avx512f` is enabled for this module.
    unsafe {
        // Each output coordinate gets one product from each half.
        let halves = _mm512_castsi512_si256(slots).xor(_mm512_extracti64x4_epi64::<1>(slots));
        // The rest: slot 3 of that sum into coordinate 0, and c_0 (slot 3) into coordinates 1 and 2.
        let rest = _mm512_permutex2var_epi64(
            _mm512_castsi256_si512(halves),
            _mm512_set_epi64(0, 0, 0, 0, 0, 11, 11, 3),
            slots,
        );
        halves.xor(_mm512_castsi512_si256(rest))
    }
}

/// Sum unreduced products before paying for one reduction.
#[inline]
pub(crate) fn poly_dot_192<'a>(
    pairs: impl Iterator<Item = (&'a [u64; 3], &'a [u64; 3])>,
) -> [u64; 3] {
    // Every term adds into the same eight slots, so the whole sum reduces and combines once.
    let sum = pairs.fold(Wide::<__m512i>::zero(), |sum, (a, b)| {
        sum.xor(Wide::mul(operand(load(a)), operand(load(b))))
    });

    // Reduction is linear, so the slots reduce before they combine.
    store(combine(sum.reduce()))
}

/// The unreduced multiple by a coefficient-field element, in element order once reduced.
///
/// ```text
///     even  =  [ a_0 k | a_2 k ]       odd  =  [ a_1 k | 0 ]
/// ```
#[inline(always)]
fn mul_by_64_unreduced(a: &[u64; 3], k: u64) -> Wide<__m256i> {
    // The coordinates as [a_0, a_1 | a_2, 0].
    let x = load(a);
    // SAFETY: `avx2` is implied by `avx512f`, which is enabled for this module.
    let k = unsafe { _mm256_set1_epi64x(k as i64) };
    // The low quadword of each lane, then the high one, each times the broadcast scalar.
    Wide {
        even: x.clmul::<LOW_BY_LOW>(k),
        odd: x.clmul::<HIGH_BY_LOW>(k),
    }
}

/// Sum coefficient-field multiples before paying for one reduction.
#[inline]
pub(crate) fn poly_dot_192_by_64<'a>(
    pairs: impl Iterator<Item = (&'a [u64; 3], &'a u64)>,
) -> [u64; 3] {
    // Two multiplies per term into the same accumulators.
    let sum = pairs.fold(Wide::zero(), |sum, (a, k)| {
        sum.xor(mul_by_64_unreduced(a, *k))
    });

    // The reduced products already sit in coordinate order.
    store(sum.reduce())
}
