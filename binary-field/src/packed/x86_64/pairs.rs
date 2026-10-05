//! Sums of products over two 256-bit packings side by side in one 512-bit register.
//!
//! The packings stay 256 bits wide, so their footprint does not change.
//!
//! A sum of products still has terms to spare, so it pairs them:
//!
//! - term `2i` takes the low 256 bits of a register, term `2i + 1` the high ones;
//! - each carryless multiply then serves both terms;
//! - the halves never mix inside the loop, and fold into one 256-bit sum at the end.
//!
//! A lone product gains nothing this way: the joins and the fold sit on its critical path.
//!
//! So only sums take this route.

use core::arch::x86_64::{
    __m256i, __m512i, _mm512_castsi256_si512, _mm512_castsi512_si256, _mm512_extracti64x4_epi64,
    _mm512_inserti64x4,
};
use core::array;

use crate::clmul::wide::{Lanes64, Wide};

/// Two 256-bit registers as the low and high halves of one 512-bit register.
#[inline(always)]
pub(crate) fn join(low: __m256i, high: __m256i) -> __m512i {
    // SAFETY: this module compiles only with `avx512f`.
    unsafe { _mm512_inserti64x4::<1>(_mm512_castsi256_si512(low), high) }
}

/// Two elements, each held as coordinate registers, joined coordinate by coordinate.
#[inline(always)]
pub(crate) fn join_all<const D: usize>(low: [__m256i; D], high: [__m256i; D]) -> [__m512i; D] {
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
pub(crate) fn sum_of_products<U, V, const D: usize>(
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

    // Every pair adds into the same wide accumulators.
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
