//! Sums of scalar `GF(2^192)` elements times `GF(2^64)` weights, read in place by 512-bit registers.
//!
//! Eight elements are 24 quadwords, so three registers load them as they lie in memory:
//!
//! ```text
//!     quadword      0  1 |  2  3 |  4  5 |  6  7     8  9 | 10 11 | ...
//!     element       0  0 |  0  1 |  1  1 |  2  2     2  3 |  3  3 | ...
//!     coordinate    0  1 |  2  0 |  1  2 |  0  1     2  0 |  1  2 | ...
//! ```
//!
//! One permute per register places each element's weight under its three coordinates.
//!
//! A low and a high carryless multiply then give every quadword its product.
//!
//! The data itself never moves.
//!
//! Permutes share a port with the multiplier on some Intel cores, so only the weights pay one.
//!
//! The coordinate in each 128-bit lane repeats from block to block.
//!
//! So the sums stay in their lanes until one final regrouping.

use core::arch::x86_64::{
    __m512i, _mm512_clmulepi64_epi128, _mm512_loadu_si512, _mm512_maskz_loadu_epi64,
    _mm512_permutex2var_epi64, _mm512_permutexvar_epi64, _mm512_set_epi64, _mm512_setzero_si512,
    _mm512_shuffle_i64x2, _mm512_xor_si512,
};

use crate::clmul::wide::Lanes64;
use crate::{Poly64, Poly192};

/// Elements per block: three registers of quadwords.
const BLOCK: usize = 8;

/// The quadwords of one block.
const WORDS: usize = 3 * BLOCK;

/// The sum of `values[i] * weights[i]` over every index, reduced once per coordinate.
///
/// # Panics
///
/// Panics if the slices differ in length.
#[inline]
pub(crate) fn dot(values: &[Poly192], weights: &[Poly64]) -> Poly192 {
    assert_eq!(values.len(), weights.len());

    // SAFETY: `Poly192` is three transparent quadwords and `Poly64` one.
    let (words, scalars) = unsafe {
        (
            core::slice::from_raw_parts(values.as_ptr().cast::<u64>(), 3 * values.len()),
            core::slice::from_raw_parts(weights.as_ptr().cast::<u64>(), weights.len()),
        )
    };

    Sums::default().add_all(words, scalars).finish()
}

/// Three registers of unreduced 128-bit sums.
///
/// Each 128-bit lane holds one coordinate, fixed by the register and the lane:
///
/// ```text
///     lane                 0  1  2  3
///     first register       0  2  1  0
///     second register      1  0  2  1
///     third register       2  1  0  2
/// ```
#[derive(Clone, Copy)]
struct Sums([__m512i; 3]);

impl Default for Sums {
    #[inline(always)]
    fn default() -> Self {
        // SAFETY: this module compiles only with `avx512f`.
        Self([unsafe { _mm512_setzero_si512() }; 3])
    }
}

impl Sums {
    /// Add every term of whole and ragged blocks, three quadwords per weight.
    #[inline(always)]
    fn add_all(mut self, words: &[u64], weights: &[u64]) -> Self {
        // Whole blocks load straight from memory.
        let (blocks, tail) = words.as_chunks::<WORDS>();
        let (weight_blocks, weight_tail) = weights.as_chunks::<BLOCK>();
        for (block, weights) in blocks.iter().zip(weight_blocks) {
            // SAFETY: a block is three registers of quadwords and eight weights one register.
            let (data, weights) = unsafe {
                (
                    core::array::from_fn(|r| {
                        _mm512_loadu_si512(block[r * BLOCK..].as_ptr().cast())
                    }),
                    _mm512_loadu_si512(weights.as_ptr().cast()),
                )
            };
            self.add_block(data, weights);
        }

        // The ragged block loads under masks, so its missing quadwords are zero and add nothing.
        if !weight_tail.is_empty() {
            let mask = |len: usize| ((1u16 << len.min(BLOCK)) - 1) as u8;
            // SAFETY: every enabled quadword lies inside the tail.
            //
            // Masked-off quadwords are never read, so the offsets past the tail are not dereferenced.
            let (data, weights) = unsafe {
                (
                    core::array::from_fn(|r| {
                        let len = tail.len().saturating_sub(r * BLOCK);
                        _mm512_maskz_loadu_epi64(
                            mask(len),
                            tail.as_ptr().wrapping_add(r * BLOCK).cast(),
                        )
                    }),
                    _mm512_maskz_loadu_epi64(mask(weight_tail.len()), weight_tail.as_ptr().cast()),
                )
            };
            self.add_block(data, weights);
        }
        self
    }

    /// Add the eight products of one block.
    #[inline(always)]
    fn add_block(&mut self, data: [__m512i; 3], weights: __m512i) {
        // The element of each quadword, as the register of quadwords 8r .. 8r + 8 needs it.
        //
        // Quadword `q` belongs to element `q / 3`.
        let spread = |indices: [i64; 8]| {
            // SAFETY: this module compiles only with `avx512f`.
            unsafe {
                let [i0, i1, i2, i3, i4, i5, i6, i7] = indices;
                _mm512_permutexvar_epi64(_mm512_set_epi64(i7, i6, i5, i4, i3, i2, i1, i0), weights)
            }
        };
        let k = [
            spread([0, 0, 0, 1, 1, 1, 2, 2]),
            spread([2, 3, 3, 3, 4, 4, 4, 5]),
            spread([5, 5, 6, 6, 6, 7, 7, 7]),
        ];

        // The low and high quadword products of each register.
        //
        // SAFETY: this module compiles only with `avx512f` and `vpclmulqdq`.
        let [low, high]: [[__m512i; 3]; 2] = unsafe {
            [
                core::array::from_fn(|r| _mm512_clmulepi64_epi128::<0x00>(data[r], k[r])),
                core::array::from_fn(|r| _mm512_clmulepi64_epi128::<0x11>(data[r], k[r])),
            ]
        };

        // Products of one lane pattern share a sum.
        //
        //     first:   low of register 0,  high of register 1
        //     second:  high of register 0, low of register 2
        //     third:   low of register 1,  high of register 2
        //
        // The compiler fuses each pair of exclusive ors into one ternary logic op.
        let [first, second, third] = self.0;
        self.0 = [
            xor3(first, low[0], high[1]),
            xor3(second, high[0], low[2]),
            xor3(third, low[1], high[2]),
        ];
    }

    /// Gather each coordinate's four lanes, then reduce all three at once.
    #[inline(always)]
    fn finish(self) -> Poly192 {
        let [p, q, s] = self.0;

        // Lane `c` of the result collects coordinate `c` from the lane table above:
        //
        //     coordinate 0:   p0  p3  q1  s2
        //     coordinate 1:   p2  q0  q3  s1
        //     coordinate 2:   p1  q2  s0  s3
        //
        // SAFETY: this module compiles only with `avx512f`.
        let sum = unsafe {
            // Lanes [p0, p2, p1, p3] and [s2, s1, s3, s0].
            let p_own = _mm512_shuffle_i64x2::<0b11_01_10_00>(p, p);
            let s_own = _mm512_shuffle_i64x2::<0b00_11_01_10>(s, s);
            // Lanes [p3, q0, q2, p0] and [q1, q3, s0, q0], as quadword indices into two sources.
            let p_q = _mm512_permutex2var_epi64(p, _mm512_set_epi64(1, 0, 13, 12, 9, 8, 7, 6), q);
            let q_s = _mm512_permutex2var_epi64(q, _mm512_set_epi64(1, 0, 9, 8, 7, 6, 3, 2), s);
            _mm512_xor_si512(xor3(p_own, s_own, p_q), q_s)
        };

        // One reduction per lane leaves each coordinate in its lane's low quadword.
        //
        // SAFETY: a register is eight unrestricted quadwords.
        let words: [u64; 8] = unsafe { core::mem::transmute(sum.reduce_lane()) };
        Poly192::from_limbs([words[0], words[2], words[4]])
    }
}

/// Three-way exclusive or.
#[inline(always)]
fn xor3(a: __m512i, b: __m512i, c: __m512i) -> __m512i {
    // SAFETY: this module compiles only with `avx512f`.
    unsafe { _mm512_xor_si512(_mm512_xor_si512(a, b), c) }
}
