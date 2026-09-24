//! The cubic extension `y^3 + y + 1` of `GF(2^64)`, two coordinates per register.
//!
//! A carryless multiply picks one quadword of each operand by its immediate.
//!
//! So one register pair feeds two products:
//!
//! ```text
//!     pair   = [ a_0        , a_1       ]      the first two coordinates
//!     tail   = [ a_2        , a_0 + a_1 ]      the third, beside one Karatsuba sum
//!     sums   = [ a_0 + a_2  , a_1 + a_2 ]      the other two Karatsuba sums
//! ```
//!
//! The low and the high products of those three pairs are the six Karatsuba terms.

use core::arch::x86_64::{
    __m128i, _mm_loadl_epi64, _mm_loadu_si128, _mm_shuffle_epi32, _mm_storel_epi64,
    _mm_storeu_si128,
};
use core::ptr;

use crate::clmul::wide::{HIGH_BY_HIGH, HIGH_BY_LOW, LOW_BY_LOW, Lanes64, Wide, fold};

/// Swaps the two quadwords of a register.
const SWAP_QUADWORDS: i32 = 0x4e;

/// The first two coordinates as one register, and the third alone.
///
/// Each register is one load, in the shape a stored result takes.
///
/// So a result forwards straight into the next product.
#[inline(always)]
fn load(a: &[u64; 3]) -> (__m128i, __m128i) {
    // SAFETY: the array is 24 readable bytes, and both loads are the unaligned forms.
    //
    // The first reads bytes 0 to 15, the second bytes 16 to 23.
    unsafe {
        (
            _mm_loadu_si128(a.as_ptr().cast()),
            _mm_loadl_epi64(a[2..].as_ptr().cast()),
        )
    }
}

/// The three coordinates, from a register holding the first two and one holding the third.
#[inline(always)]
fn store(pair: __m128i, last: __m128i) -> [u64; 3] {
    let mut out = [0u64; 3];

    // SAFETY: the array is 24 writable bytes, and both stores are the unaligned forms.
    unsafe {
        _mm_storeu_si128(out.as_mut_ptr().cast(), pair);
        _mm_storel_epi64(out[2..].as_mut_ptr().cast(), last);
    }
    out
}

/// One element as the three registers the products read.
#[derive(Clone, Copy)]
struct Operand {
    /// `[a_0, a_1]`.
    pair: __m128i,
    /// `[a_2, a_0 + a_1]`.
    tail: __m128i,
    /// `[a_0 + a_2, a_1 + a_2]`.
    sums: __m128i,
}

impl Operand {
    /// Lays out the coordinates and the three Karatsuba sums.
    #[inline(always)]
    fn new(a: &[u64; 3]) -> Self {
        let (pair, last) = load(a);

        // `a_2` in both quadwords.
        let third = last.unpack_low(last);

        // `a_0 + a_1` in both quadwords, from the pair and its own swap.
        //
        // SAFETY: `sse2` is part of the `x86_64` baseline.
        let crossed = pair.xor(unsafe { _mm_shuffle_epi32::<SWAP_QUADWORDS>(pair) });

        // The three product-ready registers of the module header.
        Self {
            pair,
            tail: third.unpack_low(crossed),
            sums: pair.xor(third),
        }
    }
}

/// The three unreduced output coordinates, each a 128-bit product.
#[derive(Clone, Copy)]
struct Unreduced([__m128i; 3]);

impl Unreduced {
    /// The empty sum.
    #[inline(always)]
    fn zero() -> Self {
        Self([__m128i::zero(); 3])
    }

    /// Coordinate-wise sum.
    #[inline(always)]
    fn xor(self, other: Self) -> Self {
        Self(core::array::from_fn(|i| self.0[i].xor(other.0[i])))
    }

    /// The reduced coordinates.
    #[inline(always)]
    fn reduce(self) -> [u64; 3] {
        let [r0, r1, r2] = self.0;

        // The first two coordinates reduce together, one per quadword.
        let pair = Wide { even: r0, odd: r1 }.reduce();

        // The third reduces alone, reading its high half from its own upper quadword.
        let last = fold(r2, r2.unpack_high(r2));

        // Back to memory in the same 16 + 8 byte shape the loads read.
        store(pair, last)
    }
}

/// The unreduced product, with the top two powers of `y` already folded.
///
/// ```text
///     r_0  =  c_0 + c_1 + c_2 + d_12
///     r_1  =  c_0 + d_01 + d_12
///     r_2  =  c_0 + c_1 + d_02
/// ```
#[inline(always)]
fn mul_unreduced(a: Operand, b: Operand) -> Unreduced {
    // The six Karatsuba terms, two per register pair.
    let c0 = a.pair.clmul::<LOW_BY_LOW>(b.pair);
    let c1 = a.pair.clmul::<HIGH_BY_HIGH>(b.pair);
    let c2 = a.tail.clmul::<LOW_BY_LOW>(b.tail);
    let d01 = a.tail.clmul::<HIGH_BY_HIGH>(b.tail);
    let d02 = a.sums.clmul::<LOW_BY_LOW>(b.sums);
    let d12 = a.sums.clmul::<HIGH_BY_HIGH>(b.sums);

    // The one sum two coordinates share.
    let shared = c0.xor(d12);

    // The folded coordinates r_0, r_1, r_2, still unreduced in the base field.
    Unreduced([shared.xor3(c1, c2), shared.xor(d01), c0.xor3(c1, d02)])
}

/// Multiplication: six carryless products and two reductions.
#[inline]
pub(crate) fn poly_mul_192(a: &[u64; 3], b: &[u64; 3]) -> [u64; 3] {
    // One register fold for the first two coordinates, one for the third.
    mul_unreduced(Operand::new(a), Operand::new(b)).reduce()
}

/// Squaring: three carryless products.
///
/// ```text
///     (a_0 + a_1 y + a_2 y^2)^2  =  a_0^2 + a_2^2 y + (a_1^2 + a_2^2) y^2
/// ```
#[inline]
pub(crate) fn poly_square_192(a: &[u64; 3]) -> [u64; 3] {
    let (pair, third) = load(a);

    // The three coordinate squares, unreduced.
    let s0 = pair.clmul::<LOW_BY_LOW>(pair);
    let s1 = pair.clmul::<HIGH_BY_HIGH>(pair);
    let s2 = third.clmul::<LOW_BY_LOW>(third);

    // y^4 = y^2 + y moves the top square onto the two coordinates above the constant.
    Unreduced([s0, s2, s1.xor(s2)]).reduce()
}

/// The unreduced multiple by a coefficient-field element: three products, no fold in `y`.
#[inline(always)]
fn mul_by_64_unreduced(a: &[u64; 3], k: &u64) -> Unreduced {
    let (pair, third) = load(a);

    // SAFETY: the reference is 8 readable bytes, and the load is the unaligned form.
    let scalar = unsafe { _mm_loadl_epi64(ptr::from_ref(k).cast()) };

    // a_0 k and a_1 k from the pair, a_2 k from the third register.
    Unreduced([
        pair.clmul::<LOW_BY_LOW>(scalar),
        pair.clmul::<HIGH_BY_LOW>(scalar),
        third.clmul::<LOW_BY_LOW>(scalar),
    ])
}

/// Multiplication by a coefficient-field element.
#[inline]
pub(crate) fn poly_mul_192_by_64(a: &[u64; 3], k: &u64) -> [u64; 3] {
    // Three products, then the same two reductions as a full product.
    mul_by_64_unreduced(a, k).reduce()
}

/// Sum unreduced products before paying for one reduction.
#[inline]
pub(crate) fn poly_dot_192<'a>(
    pairs: impl Iterator<Item = (&'a [u64; 3], &'a [u64; 3])>,
) -> [u64; 3] {
    // Reduction is linear, so the whole sum reduces once.
    pairs
        .fold(Unreduced::zero(), |sum, (a, b)| {
            sum.xor(mul_unreduced(Operand::new(a), Operand::new(b)))
        })
        .reduce()
}

/// Sum coefficient-field multiples before paying for one reduction.
#[inline]
pub(crate) fn poly_dot_192_by_64<'a>(
    pairs: impl Iterator<Item = (&'a [u64; 3], &'a u64)>,
) -> [u64; 3] {
    // Reduction is linear, so the whole sum reduces once.
    pairs
        .fold(Unreduced::zero(), |sum, (a, k)| {
            sum.xor(mul_by_64_unreduced(a, k))
        })
        .reduce()
}
