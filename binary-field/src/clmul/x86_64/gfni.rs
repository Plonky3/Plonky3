//! Repeated squaring in `GF(2^64)` as one bit-matrix product, on the `GFNI` affine instruction.
//!
//! Squaring is `F_2`-linear, so squaring `K` times is a fixed `64 x 64` bit matrix `M`.
//!
//! Split `M` into `8 x 8` blocks, one per pair of output and input bytes:
//!
//! ```text
//!     y_q  =  sum_j  M_qj x_j                   q, j   byte indices
//! ```
//!
//! `vgf2p8affineqb` applies one `8 x 8` matrix per quadword to every byte of that quadword.
//!
//! Broadcasting `x` and rotating it by `t` bytes puts `x_(q - t)` in byte `q` of every quadword.
//!
//! So quadword `q` of the `t`-th product holds `M_q,(q - t) x_(q - t)` in its byte `q`:
//!
//! ```text
//!     sum_t  (byte q of quadword q of product t)  =  sum_j  M_qj x_j  =  y_q
//! ```
//!
//! Eight products, one exclusive-or tree, and one byte permute to collect the diagonal.
//!
//! The cost is the same for every `K`, and nothing is indexed by the operand.

#[cfg(not(miri))]
use core::arch::x86_64::_mm512_ternarylogic_epi64;
use core::arch::x86_64::{
    __m128i, __m512i, _mm_cvtsi64_si128, _mm_cvtsi128_si64, _mm512_broadcastq_epi64,
    _mm512_castsi512_si128, _mm512_gf2p8affine_epi64_epi8, _mm512_loadu_si512,
    _mm512_permutexvar_epi8, _mm512_rol_epi64, _mm512_xor_si512,
};

use crate::clmul::reduce_64;

/// The truth table of `a ^ b ^ c` for a ternary logic instruction.
#[cfg(not(miri))]
const XOR3: i32 = 0x96;

/// Bits `0 .. 32` of `v`, each moved to twice its position: the carryless square of `v`.
const fn spread_32(v: u64) -> u64 {
    let mut out = 0;
    let mut i = 0;
    while i < 32 {
        // Bit i of v becomes the coefficient of x^(2i), since (x^i)^2 = x^(2i).
        out |= ((v >> i) & 1) << (2 * i);
        i += 1;
    }
    out
}

/// `x^(2^K)`, one squaring at a time.
const fn square_times_slow(mut x: u64, k: usize) -> u64 {
    let mut i = 0;
    while i < k {
        // The carryless square of a 64-bit value is its two halves, each spread.
        let square = (spread_32(x >> 32) as u128) << 64 | spread_32(x) as u128;
        x = reduce_64(square);
        i += 1;
    }
    x
}

/// The affine matrices of `x -> x^(2^K)`, product `t` then quadword `q`.
///
/// The instruction takes the row of output bit `i` from byte `7 - i` of the matrix quadword.
struct Matrices<const K: usize>([[u64; 8]; 8]);

impl<const K: usize> Matrices<K> {
    /// The matrices, evaluated at compile time.
    const NEW: Self = Self::new();

    /// Derives every block from the images of the 64 basis vectors.
    const fn new() -> Self {
        // Column `c` of the map is the image of `x^c`.
        let mut columns = [0u64; 64];
        let mut c = 0;
        while c < 64 {
            columns[c] = square_times_slow(1 << c, K);
            c += 1;
        }

        // Cut the 64 x 64 map into the 8 x 8 blocks each product applies.
        let mut matrices = [[0u64; 8]; 8];
        let mut t = 0;
        while t < 8 {
            let mut q = 0;
            while q < 8 {
                // Product `t` reads input byte `j = q - t` in byte `q`.
                let j = (q + 8 - t) % 8;
                let mut block = 0u64;
                let mut i = 0;
                while i < 8 {
                    // Row `i` of block `M_qj`: which bits of input byte `j` reach bit `i` of `y_q`.
                    let mut row = 0u64;
                    let mut s = 0;
                    while s < 8 {
                        row |= ((columns[8 * j + s] >> (8 * q + i)) & 1) << s;
                        s += 1;
                    }
                    // The instruction reads the row of output bit i from byte 7 - i.
                    block |= row << (8 * (7 - i));
                    i += 1;
                }
                matrices[t][q] = block;
                q += 1;
            }
            t += 1;
        }
        Self(matrices)
    }

    /// Product `t`'s matrices as one register.
    #[inline(always)]
    fn register(t: usize) -> __m512i {
        // A reference in a constant is promoted to a static, so the table is never copied.
        let table: &'static [[u64; 8]; 8] = const { &Self::NEW.0 };

        // SAFETY: each row of the table is exactly one register of quadwords.
        unsafe { _mm512_loadu_si512(table[t].as_ptr().cast()) }
    }
}

/// The byte permute that gathers byte `q` of quadword `q` into byte `q`.
const DIAGONAL: [u8; 64] = {
    let mut index = [0u8; 64];
    let mut q = 0;
    while q < 8 {
        // Byte q of quadword q is byte 8q + q = 9q of the register.
        index[q] = (9 * q) as u8;
        q += 1;
    }
    index
};

/// `x^(2^K)` in one bit-matrix product.
#[inline]
pub(crate) fn square_times<const K: usize>(x: u64) -> u64 {
    // SAFETY: this module compiles only with `gfni`, `avx512f`, `avx512bw` and `avx512vbmi`.
    //
    // Those are exactly the features every intrinsic below requires.
    unsafe {
        // Product t applies the eight blocks M_q,(q - t), one per quadword q.
        let product = |t: usize, rotated: __m512i| {
            _mm512_gf2p8affine_epi64_epi8::<0>(rotated, Matrices::<K>::register(t))
        };

        // The same eight input bytes in every quadword.
        let x: __m128i = _mm_cvtsi64_si128(x as i64);
        let b = _mm512_broadcastq_epi64(x);

        // Rotating by `t` bytes aligns input byte `q - t` with output byte `q`.
        let terms = [
            product(0, b),
            product(1, _mm512_rol_epi64::<8>(b)),
            product(2, _mm512_rol_epi64::<16>(b)),
            product(3, _mm512_rol_epi64::<24>(b)),
            product(4, _mm512_rol_epi64::<32>(b)),
            product(5, _mm512_rol_epi64::<40>(b)),
            product(6, _mm512_rol_epi64::<48>(b)),
            product(7, _mm512_rol_epi64::<56>(b)),
        ];

        // A two-level tree keeps the sum at two instructions of latency.
        #[cfg(not(miri))]
        let sum = _mm512_ternarylogic_epi64::<XOR3>(
            _mm512_ternarylogic_epi64::<XOR3>(terms[0], terms[1], terms[2]),
            _mm512_ternarylogic_epi64::<XOR3>(terms[3], terms[4], terms[5]),
            _mm512_xor_si512(terms[6], terms[7]),
        );

        // Miri has no shim for the 512-bit ternary logic instruction.
        //
        // A plain exclusive-or chain computes the same sum.
        //
        // With it, the interpreter runs every other step of the kernel.
        #[cfg(miri)]
        let sum = terms[1..]
            .iter()
            .fold(terms[0], |acc, &term| _mm512_xor_si512(acc, term));

        // Only byte q of quadword q is wanted.
        //
        // The permute packs those eight bytes together.
        //
        //     sum     = [ y_0 . . . . . . . | . y_1 . . . . . . | ... | . . . . . . . y_7 ]
        //     result  = [ y_0 y_1 y_2 y_3 y_4 y_5 y_6 y_7 ]
        let diagonal = _mm512_loadu_si512(DIAGONAL.as_ptr().cast());
        _mm_cvtsi128_si64(_mm512_castsi512_si128(_mm512_permutexvar_epi8(
            diagonal, sum,
        ))) as u64
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::{square_times, square_times_slow};
    use crate::clmul::poly_square_64;

    /// The operands whose images are extreme: the identities, all ones, the top bits.
    const CORNERS: [u64; 5] = [0, 1, u64::MAX, 1 << 63, 0xf << 60];

    #[test]
    fn the_reference_squaring_matches_the_field() {
        // Invariant: the compile-time squaring that builds the matrices is the field's own.
        for x in CORNERS {
            assert_eq!(square_times_slow(x, 1), poly_square_64(x), "{x:#x}");
        }
    }

    #[test]
    fn the_matrix_product_is_exact_on_the_basis() {
        // Invariant: the map is linear, so agreeing on all 64 basis vectors settles every input.
        //
        // Fixture state: the shortest and the longest run the inversion chain takes, 3 and 24.
        for c in 0..64 {
            let x = 1u64 << c;
            assert_eq!(square_times::<3>(x), square_times_slow(x, 3), "x^{c}");
            assert_eq!(square_times::<24>(x), square_times_slow(x, 24), "x^{c}");
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(2000))]

        #[test]
        fn the_matrix_product_matches_repeated_squaring(x: u64) {
            // The powers the inversion chain takes, against the field's own squaring.
            let repeated = |k: usize| (0..k).fold(x, |y, _| poly_square_64(y));
            prop_assert_eq!(square_times::<3>(x), repeated(3));
            prop_assert_eq!(square_times::<6>(x), repeated(6));
            prop_assert_eq!(square_times::<12>(x), repeated(12));
            prop_assert_eq!(square_times::<24>(x), repeated(24));
        }
    }
}
