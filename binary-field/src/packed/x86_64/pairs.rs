//! Two 256-bit packings side by side in one 512-bit register.
//!
//! On a core with a full 512-bit datapath, a carryless multiply costs the same at either width.
//!
//! So a sum pairs its terms, runs each pair at 512 bits, and folds the halves once at the end:
//!
//! ```text
//!     term 2i      term 2i+1
//!     [ 256 bits | 256 bits ]      one register, one multiply for both
//! ```
//!
//! A square, or a product by a base element, pairs two of its own coordinates the same way.
//!
//! The packings themselves stay 256 bits wide, so their footprint does not change.

use core::arch::x86_64::{
    __m256i, __m512i, _mm512_castsi256_si512, _mm512_castsi512_si256, _mm512_extracti64x4_epi64,
    _mm512_inserti64x4,
};
use core::array;

use crate::clmul::wide::{Lanes64, Wide};

/// Two 256-bit registers as the low and high halves of one 512-bit register.
#[inline(always)]
pub(super) fn join(low: __m256i, high: __m256i) -> __m512i {
    // SAFETY: this module compiles only with `avx512f`.
    unsafe { _mm512_inserti64x4::<1>(_mm512_castsi256_si512(low), high) }
}

/// Two elements, each held as coordinate registers, joined coordinate by coordinate.
#[inline(always)]
pub(super) fn join_all<const D: usize>(low: [__m256i; D], high: [__m256i; D]) -> [__m512i; D] {
    // Coordinate i of the low element under coordinate i of the high one.
    array::from_fn(|i| join(low[i], high[i]))
}

/// The sum of the two halves.
#[inline(always)]
fn fold_halves(x: __m512i) -> __m256i {
    // SAFETY: this module compiles only with `avx512f`.
    unsafe { _mm512_castsi512_si256(x).xor(_mm512_extracti64x4_epi64::<1>(x)) }
}

/// The unreduced sum of the products of `u_i` and `v_i`, two terms per 512-bit multiply.
///
/// # Arguments
///
/// - `pair`: the unreduced products of two consecutive terms, one per half.
/// - `single`: the unreduced product of one term, for an odd count's last one.
#[inline(always)]
pub(super) fn sum_of_products<U, V, const D: usize>(
    u: &[U],
    v: &[V],
    pair: impl Fn(&[U; 2], &[V; 2]) -> [Wide<__m512i>; D],
    single: impl Fn(&U, &V) -> [Wide<__m256i>; D],
) -> [Wide<__m256i>; D] {
    // Whole pairs first.
    //
    // An odd count leaves one term over.
    let (u_pairs, u_last) = u.as_chunks::<2>();
    let (v_pairs, v_last) = v.as_chunks::<2>();

    // Every pair adds into the same wide accumulators: the halves never mix until the end.
    let wide = u_pairs
        .iter()
        .zip(v_pairs)
        .fold([Wide::zero(); D], |sum, (a, b)| {
            let product = pair(a, b);
            array::from_fn(|i| sum[i].xor(product[i]))
        });

    // Reduction is linear, so the two halves sum before it.
    let mut sum = wide.map(|w| Wide {
        even: fold_halves(w.even),
        odd: fold_halves(w.odd),
    });

    // The term left over takes a 256-bit multiply.
    if let (Some(a), Some(b)) = (u_last.first(), v_last.first()) {
        let product = single(a, b);
        sum = array::from_fn(|i| sum[i].xor(product[i]));
    }
    sum
}

/// The low half.
#[inline(always)]
fn low(x: __m512i) -> __m256i {
    // SAFETY: this module compiles only with `avx512f`.
    unsafe { _mm512_castsi512_si256(x) }
}

/// The high half.
#[inline(always)]
fn high(x: __m512i) -> __m256i {
    // SAFETY: this module compiles only with `avx512f`.
    unsafe { _mm512_extracti64x4_epi64::<1>(x) }
}

/// One square in the cubic extension, the first two coordinate squares in one register.
///
/// ```text
///     (a_0 + a_1 y + a_2 y^2)^2  =  a_0^2 + a_2^2 y + (a_1^2 + a_2^2) y^2
/// ```
#[inline(always)]
pub(super) fn cubic_square(a: [__m256i; 3]) -> [__m256i; 3] {
    let [x0, x1, x2] = a;

    // [a_0^2 | a_1^2] at 512 bits, a_2^2 at 256: four multiplies, not six.
    let x01 = join(x0, x1);
    let s01 = Wide::mul(x01, x01).reduce();
    let s2 = Wide::mul(x2, x2).reduce();

    // y^4 = y^2 + y moves the top square onto the two coordinates above the constant.
    [low(s01), s2, high(s01).xor(s2)]
}

/// One product by a coefficient-field element, the first two coordinates in one register.
#[inline(always)]
pub(super) fn cubic_mul_base(a: [__m256i; 3], k: __m256i) -> [__m256i; 3] {
    let [x0, x1, x2] = a;

    // [a_0 k | a_1 k] at 512 bits, a_2 k at 256: four multiplies, not six.
    let p01 = Wide::mul(join(x0, x1), join(k, k)).reduce();
    let p2 = Wide::mul(x2, k).reduce();

    // The degree in y never grows, so the products are already the coordinates.
    [low(p01), high(p01), p2]
}
