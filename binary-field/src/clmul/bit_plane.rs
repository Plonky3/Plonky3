//! GFNI expansion of 64 Boolean corner words into 64 polynomial-basis field values.

use alloc::vec::Vec;
use core::arch::x86_64::{
    __m512i, _mm512_gf2p8affine_epi64_epi8, _mm512_loadu_si512, _mm512_permutexvar_epi8,
    _mm512_set1_epi64, _mm512_storeu_si512, _mm512_ternarylogic_epi64, _mm512_unpackhi_epi8,
    _mm512_unpackhi_epi16, _mm512_unpackhi_epi32, _mm512_unpackhi_epi64, _mm512_unpacklo_epi8,
    _mm512_unpacklo_epi16, _mm512_unpacklo_epi32, _mm512_unpacklo_epi64, _mm512_xor_si512,
};

use crate::Ghash128;

/// The immediate of a three-input exclusive-or, as a truth table of its three operands.
const XOR3: i32 = 0x96;

/// Quadword whose byte `j` is `1 << j`.
const UNIT: u64 = 0x8040_2010_0804_0201;

/// Moves byte `k` of word `i` to byte `7 - i` of quadword `k`.
const GATHER: __m512i = {
    let mut index = [0_u8; 64];
    let mut word = 0;
    while word < 8 {
        let mut byte = 0;
        while byte < 8 {
            index[8 * byte + 7 - word] = (8 * word + byte) as u8;
            byte += 1;
        }
        word += 1;
    }
    // SAFETY: a 512-bit register is 64 bytes, and every bit pattern is valid.
    unsafe { core::mem::transmute::<[u8; 64], __m512i>(index) }
};

/// Reorders natural row lanes for the lane-local 16 × 16 byte transpose.
///
/// After this permutation and the transpose below, register `r` holds rows `4r..4r+4`.
const TRANSPOSE_ORDER: __m512i = {
    let mut index = [0_u8; 64];
    let mut i = 0;
    while i < 64 {
        index[i] = (4 * (i % 16) + i / 16) as u8;
        i += 1;
    }
    // SAFETY: a 512-bit register is 64 bytes, and every bit pattern is valid.
    unsafe { core::mem::transmute::<[u8; 64], __m512i>(index) }
};

/// Transposes the bits of an 8 × 8 byte matrix packed row-major in a u64.
#[inline]
const fn transpose8(mut value: u64) -> u64 {
    let mut exchange = (value ^ (value >> 7)) & 0x00aa_00aa_00aa_00aa;
    value ^= exchange ^ (exchange << 7);
    exchange = (value ^ (value >> 14)) & 0x0000_cccc_0000_cccc;
    value ^= exchange ^ (exchange << 14);
    exchange = (value ^ (value >> 28)) & 0x0000_0000_f0f0_f0f0;
    value ^ exchange ^ (exchange << 28)
}

/// The 16 output-byte rows of a 64 × 128 linear map, eight input-byte blocks each.
#[inline]
const fn affine_blocks(columns: &[u128; 64]) -> [[u64; 8]; 16] {
    let mut blocks = [[0u64; 8]; 16];
    let mut j = 0;
    while j < 8 {
        let c0 = columns[8 * j].to_le_bytes();
        let c1 = columns[8 * j + 1].to_le_bytes();
        let c2 = columns[8 * j + 2].to_le_bytes();
        let c3 = columns[8 * j + 3].to_le_bytes();
        let c4 = columns[8 * j + 4].to_le_bytes();
        let c5 = columns[8 * j + 5].to_le_bytes();
        let c6 = columns[8 * j + 6].to_le_bytes();
        let c7 = columns[8 * j + 7].to_le_bytes();
        let mut k = 0;
        while k < 16 {
            let packed =
                u64::from_le_bytes([c0[k], c1[k], c2[k], c3[k], c4[k], c5[k], c6[k], c7[k]]);
            blocks[k][j] = transpose8(packed).swap_bytes();
            k += 1;
        }
        j += 1;
    }
    blocks
}

/// One lane-mask register for eight consecutive corner words.
///
/// # Safety
///
/// `words` must address eight readable `u64`s. The build target supplies every named intrinsic.
#[inline(always)]
unsafe fn masks(words: *const u64) -> __m512i {
    // SAFETY: guaranteed by the caller; the load accepts any alignment and the build target
    // supplies every intrinsic used here.
    unsafe {
        let words = _mm512_loadu_si512(words.cast());
        let gathered = _mm512_permutexvar_epi8(GATHER, words);
        let natural = _mm512_gf2p8affine_epi64_epi8::<0>(_mm512_set1_epi64(UNIT as i64), gathered);
        _mm512_permutexvar_epi8(TRANSPOSE_ORDER, natural)
    }
}

/// One output byte plane from the eight input mask planes.
#[inline(always)]
unsafe fn plane(input: &[__m512i; 8], row: &[u64; 8]) -> __m512i {
    // SAFETY: the build target supplies GFNI, AVX-512F, and AVX-512BW.
    unsafe {
        let image =
            |j| _mm512_gf2p8affine_epi64_epi8::<0>(input[j], _mm512_set1_epi64(row[j] as i64));
        let a = _mm512_ternarylogic_epi64::<XOR3>(image(0), image(1), image(2));
        let b = _mm512_ternarylogic_epi64::<XOR3>(image(3), image(4), image(5));
        let c = _mm512_xor_si512(image(6), image(7));
        _mm512_ternarylogic_epi64::<XOR3>(a, b, c)
    }
}

/// A prepared 64-corner Boolean-to-`Ghash128` expansion.
pub(crate) struct PreparedBitPlaneExpansion {
    blocks: [[u64; 8]; 16],
}

impl PreparedBitPlaneExpansion {
    /// Prepares the eight GFNI blocks for each output byte.
    #[inline]
    pub(crate) const fn new(weights: [u128; 64]) -> Self {
        Self {
            blocks: affine_blocks(&weights),
        }
    }

    /// Appends the 64 lane sums selected by `words`.
    #[inline(never)]
    pub(crate) fn append(&self, words: &[u64; 64], output: &mut Vec<Ghash128>) {
        output.reserve(64);
        let old_len = output.len();

        // SAFETY: each input load reads one of the eight disjoint eight-word groups in `words`.
        // The output has reserved space for 64 more `Ghash128` values. The sixteen stores write
        // exactly those 64 entries, four per register, without assuming alignment. `Ghash128` is
        // transparent over `u128` and every 128-bit pattern is valid. No operation below can
        // panic, so the vector length is raised only after every entry has been initialized.
        unsafe {
            let input: [__m512i; 8] =
                core::array::from_fn(|group| masks(words.as_ptr().add(8 * group)));
            let p0 = plane(&input, &self.blocks[0]);
            let p1 = plane(&input, &self.blocks[1]);
            let p2 = plane(&input, &self.blocks[2]);
            let p3 = plane(&input, &self.blocks[3]);
            let p4 = plane(&input, &self.blocks[4]);
            let p5 = plane(&input, &self.blocks[5]);
            let p6 = plane(&input, &self.blocks[6]);
            let p7 = plane(&input, &self.blocks[7]);
            let p8 = plane(&input, &self.blocks[8]);
            let p9 = plane(&input, &self.blocks[9]);
            let p10 = plane(&input, &self.blocks[10]);
            let p11 = plane(&input, &self.blocks[11]);
            let p12 = plane(&input, &self.blocks[12]);
            let p13 = plane(&input, &self.blocks[13]);
            let p14 = plane(&input, &self.blocks[14]);
            let p15 = plane(&input, &self.blocks[15]);

            macro_rules! unpack {
                ($low:ident, $high:ident, $x:ident, $y:ident, $low_op:path, $high_op:path) => {
                    let $low = $low_op($x, $y);
                    let $high = $high_op($x, $y);
                };
            }

            unpack!(a0, a1, p0, p1, _mm512_unpacklo_epi8, _mm512_unpackhi_epi8);
            unpack!(a2, a3, p2, p3, _mm512_unpacklo_epi8, _mm512_unpackhi_epi8);
            unpack!(a4, a5, p4, p5, _mm512_unpacklo_epi8, _mm512_unpackhi_epi8);
            unpack!(a6, a7, p6, p7, _mm512_unpacklo_epi8, _mm512_unpackhi_epi8);
            unpack!(a8, a9, p8, p9, _mm512_unpacklo_epi8, _mm512_unpackhi_epi8);
            unpack!(
                a10,
                a11,
                p10,
                p11,
                _mm512_unpacklo_epi8,
                _mm512_unpackhi_epi8
            );
            unpack!(
                a12,
                a13,
                p12,
                p13,
                _mm512_unpacklo_epi8,
                _mm512_unpackhi_epi8
            );
            unpack!(
                a14,
                a15,
                p14,
                p15,
                _mm512_unpacklo_epi8,
                _mm512_unpackhi_epi8
            );

            unpack!(b0, b2, a0, a2, _mm512_unpacklo_epi16, _mm512_unpackhi_epi16);
            unpack!(b1, b3, a1, a3, _mm512_unpacklo_epi16, _mm512_unpackhi_epi16);
            unpack!(b4, b6, a4, a6, _mm512_unpacklo_epi16, _mm512_unpackhi_epi16);
            unpack!(b5, b7, a5, a7, _mm512_unpacklo_epi16, _mm512_unpackhi_epi16);
            unpack!(
                b8,
                b10,
                a8,
                a10,
                _mm512_unpacklo_epi16,
                _mm512_unpackhi_epi16
            );
            unpack!(
                b9,
                b11,
                a9,
                a11,
                _mm512_unpacklo_epi16,
                _mm512_unpackhi_epi16
            );
            unpack!(
                b12,
                b14,
                a12,
                a14,
                _mm512_unpacklo_epi16,
                _mm512_unpackhi_epi16
            );
            unpack!(
                b13,
                b15,
                a13,
                a15,
                _mm512_unpacklo_epi16,
                _mm512_unpackhi_epi16
            );

            unpack!(c0, c4, b0, b4, _mm512_unpacklo_epi32, _mm512_unpackhi_epi32);
            unpack!(c1, c5, b1, b5, _mm512_unpacklo_epi32, _mm512_unpackhi_epi32);
            unpack!(c2, c6, b2, b6, _mm512_unpacklo_epi32, _mm512_unpackhi_epi32);
            unpack!(c3, c7, b3, b7, _mm512_unpacklo_epi32, _mm512_unpackhi_epi32);
            unpack!(
                c8,
                c12,
                b8,
                b12,
                _mm512_unpacklo_epi32,
                _mm512_unpackhi_epi32
            );
            unpack!(
                c9,
                c13,
                b9,
                b13,
                _mm512_unpacklo_epi32,
                _mm512_unpackhi_epi32
            );
            unpack!(
                c10,
                c14,
                b10,
                b14,
                _mm512_unpacklo_epi32,
                _mm512_unpackhi_epi32
            );
            unpack!(
                c11,
                c15,
                b11,
                b15,
                _mm512_unpacklo_epi32,
                _mm512_unpackhi_epi32
            );

            unpack!(d0, d8, c0, c8, _mm512_unpacklo_epi64, _mm512_unpackhi_epi64);
            unpack!(d1, d9, c1, c9, _mm512_unpacklo_epi64, _mm512_unpackhi_epi64);
            unpack!(
                d2,
                d10,
                c2,
                c10,
                _mm512_unpacklo_epi64,
                _mm512_unpackhi_epi64
            );
            unpack!(
                d3,
                d11,
                c3,
                c11,
                _mm512_unpacklo_epi64,
                _mm512_unpackhi_epi64
            );
            unpack!(
                d4,
                d12,
                c4,
                c12,
                _mm512_unpacklo_epi64,
                _mm512_unpackhi_epi64
            );
            unpack!(
                d5,
                d13,
                c5,
                c13,
                _mm512_unpacklo_epi64,
                _mm512_unpackhi_epi64
            );
            unpack!(
                d6,
                d14,
                c6,
                c14,
                _mm512_unpacklo_epi64,
                _mm512_unpackhi_epi64
            );
            unpack!(
                d7,
                d15,
                c7,
                c15,
                _mm512_unpacklo_epi64,
                _mm512_unpackhi_epi64
            );

            let destination = output.as_mut_ptr().add(old_len);
            _mm512_storeu_si512(destination.cast(), d0);
            _mm512_storeu_si512(destination.add(4).cast(), d8);
            _mm512_storeu_si512(destination.add(8).cast(), d4);
            _mm512_storeu_si512(destination.add(12).cast(), d12);
            _mm512_storeu_si512(destination.add(16).cast(), d2);
            _mm512_storeu_si512(destination.add(20).cast(), d10);
            _mm512_storeu_si512(destination.add(24).cast(), d6);
            _mm512_storeu_si512(destination.add(28).cast(), d14);
            _mm512_storeu_si512(destination.add(32).cast(), d1);
            _mm512_storeu_si512(destination.add(36).cast(), d9);
            _mm512_storeu_si512(destination.add(40).cast(), d5);
            _mm512_storeu_si512(destination.add(44).cast(), d13);
            _mm512_storeu_si512(destination.add(48).cast(), d3);
            _mm512_storeu_si512(destination.add(52).cast(), d11);
            _mm512_storeu_si512(destination.add(56).cast(), d7);
            _mm512_storeu_si512(destination.add(60).cast(), d15);
            output.set_len(old_len + 64);
        }
    }
}
