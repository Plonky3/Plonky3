//! Sixteen lanes per 512-bit AVX-512 register.

use core::arch::x86_64::*;

use super::Word;
use crate::DIGEST_BYTES;
use crate::batch::compress::{BLOCK_BYTES, BLOCK_WORDS, STATE_WORDS};
use crate::params::IV;

/// One state or message word for sixteen lanes.
pub(super) type Vector = __m512i;

/// Lanes in one register.
pub(super) const WIDTH: usize = 16;

/// Independent register groups hashed together.
///
/// Two beat one and four here: one leaves the pipeline waiting on the dependency chains of
/// G, and four spill the working vectors out of the 32 registers.
pub(super) const GROUPS: usize = 2;

// SAFETY (every block below): this module only compiles when the target enables AVX-512F.
impl Word for __m512i {
    #[inline(always)]
    fn splat(value: u32) -> Self {
        unsafe { _mm512_set1_epi32(value as i32) }
    }

    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        unsafe { _mm512_add_epi32(self, rhs) }
    }

    #[inline(always)]
    fn and(self, rhs: Self) -> Self {
        unsafe { _mm512_and_si512(self, rhs) }
    }

    #[inline(always)]
    fn xor(self, rhs: Self) -> Self {
        unsafe { _mm512_xor_si512(self, rhs) }
    }

    #[inline(always)]
    fn xor3(self, b: Self, c: Self) -> Self {
        // 0x96 is the truth table of a ^ b ^ c.
        unsafe { _mm512_ternarylogic_epi32::<0x96>(self, b, c) }
    }

    #[inline(always)]
    fn rotr_16(self) -> Self {
        unsafe { _mm512_ror_epi32::<16>(self) }
    }

    #[inline(always)]
    fn rotr_12(self) -> Self {
        unsafe { _mm512_ror_epi32::<12>(self) }
    }

    #[inline(always)]
    fn rotr_8(self) -> Self {
        unsafe { _mm512_ror_epi32::<8>(self) }
    }

    #[inline(always)]
    fn rotr_7(self) -> Self {
        unsafe { _mm512_ror_epi32::<7>(self) }
    }
}

/// Transpose a 16 x 16 matrix of words held one row per register.
///
/// ```text
///     phase 1: unpack 32-bit pairs, then 64-bit pairs, inside each 128-bit block
///     phase 2: shuffle whole 128-bit blocks across the four rows of each quad
/// ```
#[inline(always)]
fn transpose(r: [__m512i; 16]) -> [__m512i; 16] {
    // SAFETY: this module only compiles when the target enables AVX-512F.
    unsafe {
        // Phase 1: a 4 x 4 transpose inside every 128-bit block of every quad of rows.
        //
        //     u[q][j], block k = word 4k + j of rows 4q .. 4q + 3
        let u: [[__m512i; 4]; 4] = core::array::from_fn(|q| {
            let [a, b, c, d] = [r[4 * q], r[4 * q + 1], r[4 * q + 2], r[4 * q + 3]];
            let ab_lo = _mm512_unpacklo_epi32(a, b);
            let ab_hi = _mm512_unpackhi_epi32(a, b);
            let cd_lo = _mm512_unpacklo_epi32(c, d);
            let cd_hi = _mm512_unpackhi_epi32(c, d);
            [
                _mm512_unpacklo_epi64(ab_lo, cd_lo),
                _mm512_unpackhi_epi64(ab_lo, cd_lo),
                _mm512_unpacklo_epi64(ab_hi, cd_hi),
                _mm512_unpackhi_epi64(ab_hi, cd_hi),
            ]
        });

        // Phase 2: word 4k + j gathers block k of u[0][j] .. u[3][j], in quad order.
        let mut out = [_mm512_setzero_si512(); 16];
        for j in 0..4 {
            // Blocks 0 and 1, then blocks 2 and 3, of each pair of quads.
            let q01_lo = _mm512_shuffle_i32x4::<0x44>(u[0][j], u[1][j]);
            let q01_hi = _mm512_shuffle_i32x4::<0xEE>(u[0][j], u[1][j]);
            let q23_lo = _mm512_shuffle_i32x4::<0x44>(u[2][j], u[3][j]);
            let q23_hi = _mm512_shuffle_i32x4::<0xEE>(u[2][j], u[3][j]);

            // Even blocks, then odd blocks, of each half.
            out[j] = _mm512_shuffle_i32x4::<0x88>(q01_lo, q23_lo);
            out[4 + j] = _mm512_shuffle_i32x4::<0xDD>(q01_lo, q23_lo);
            out[8 + j] = _mm512_shuffle_i32x4::<0x88>(q01_hi, q23_hi);
            out[12 + j] = _mm512_shuffle_i32x4::<0xDD>(q01_hi, q23_hi);
        }
        out
    }
}

/// Load one block from each of sixteen lanes as sixteen message words.
#[inline(always)]
pub(super) fn load_block(rows: &[&[u8; BLOCK_BYTES]; WIDTH]) -> [__m512i; BLOCK_WORDS] {
    // SAFETY: each row is 64 readable bytes, and the load has no alignment requirement.
    transpose(rows.map(|row| unsafe { _mm512_loadu_si512(row.as_ptr().cast()) }))
}

/// Write the digests of sixteen lanes.
#[inline(always)]
pub(super) fn store_digests(state: &[__m512i; 8], out: &mut [[u8; DIGEST_BYTES]; WIDTH]) {
    // SAFETY: this module only compiles when the target enables AVX-512F.
    let zero = unsafe { _mm512_setzero_si512() };

    // The eight state words fill the top half of a square, so row l comes back as lane l.
    let rows = transpose(core::array::from_fn(
        |w| if w < 8 { state[w] } else { zero },
    ));
    for (digest, row) in out.iter_mut().zip(rows) {
        // SAFETY: a digest is 32 writable bytes, and the store has no alignment requirement.
        unsafe { _mm256_storeu_si256(digest.as_mut_ptr().cast(), _mm512_castsi512_si256(row)) };
    }
}

/// `op zmmD, zmmD, zmmS`.
macro_rules! rr {
    ($op:literal, $d:literal, $s:literal) => {
        concat!($op, " zmm", $d, ", zmm", $d, ", zmm", $s, "\n")
    };
}

/// `zmmD += m[w]`, the message word read straight from memory.
macro_rules! addm {
    ($d:literal, $m:literal, $w:literal) => {
        concat!("vpaddd zmm", $d, ", zmm", $d, ", zmmword ptr [", $m, " + 64*", $w, "]\n")
    };
}

/// `zmmD >>>= n`.
macro_rules! ror {
    ($d:literal, $n:literal) => {
        concat!("vprord zmm", $d, ", zmm", $d, ", ", $n, "\n")
    };
}

/// Independent G functions, emitted one step at a time across all of them.
///
/// Each step is then a run of independent instructions.
/// The second message add is hoisted off the dependency chain.
macro_rules! g {
    ($(($a:literal, $b:literal, $c:literal, $d:literal, $m:literal, $x:literal, $y:literal))*) => {
        concat!(
            $(addm!($a, $m, $x),)*
            $(rr!("vpaddd", $a, $b),)*
            $(rr!("vpxord", $d, $a),)*
            $(addm!($a, $m, $y),)*
            $(ror!($d, 16),)*
            $(rr!("vpaddd", $c, $d),)*
            $(rr!("vpxord", $b, $c),)*
            $(ror!($b, 12),)*
            $(rr!("vpaddd", $a, $b),)*
            $(rr!("vpxord", $d, $a),)*
            $(ror!($d, 8),)*
            $(rr!("vpaddd", $c, $d),)*
            $(rr!("vpxord", $b, $c),)*
            $(ror!($b, 7),)*
        )
    };
}

/// One round of both groups with the message schedule `s`: columns, then diagonals.
#[rustfmt::skip]
macro_rules! round {
    ([$s0:literal, $s1:literal, $s2:literal, $s3:literal, $s4:literal, $s5:literal, $s6:literal, $s7:literal,
      $s8:literal, $s9:literal, $s10:literal, $s11:literal, $s12:literal, $s13:literal, $s14:literal, $s15:literal]) => {
        concat!(
            g!(
                (0, 4, 8, 12, "{ma}", $s0, $s1)       (1, 5, 9, 13, "{ma}", $s2, $s3)
                (2, 6, 10, 14, "{ma}", $s4, $s5)      (3, 7, 11, 15, "{ma}", $s6, $s7)
                (16, 20, 24, 28, "{mb}", $s0, $s1)    (17, 21, 25, 29, "{mb}", $s2, $s3)
                (18, 22, 26, 30, "{mb}", $s4, $s5)    (19, 23, 27, 31, "{mb}", $s6, $s7)
            ),
            g!(
                (0, 5, 10, 15, "{ma}", $s8, $s9)      (1, 6, 11, 12, "{ma}", $s10, $s11)
                (2, 7, 8, 13, "{ma}", $s12, $s13)     (3, 4, 9, 14, "{ma}", $s14, $s15)
                (16, 21, 26, 31, "{mb}", $s8, $s9)    (17, 22, 27, 28, "{mb}", $s10, $s11)
                (18, 23, 24, 29, "{mb}", $s12, $s13)  (19, 20, 25, 30, "{mb}", $s14, $s15)
            ),
        )
    };
}

/// Load `v[i] = h[i]` and broadcast `v[8 + i]` from `src`, for both groups.
macro_rules! init {
    ($($i:literal $j:literal $vi:literal $vj:literal $src:literal),*) => {
        concat!($(
            "vmovdqu64 zmm", $i, ", zmmword ptr [{h} + 64*", $i, "]\n",
            "vmovdqu64 zmm", $j, ", zmmword ptr [{h} + 512 + 64*", $i, "]\n",
            "vpbroadcastd zmm", $vi, ", ", $src, "\n",
            "vmovdqa64 zmm", $vj, ", zmm", $vi, "\n",
        )*)
    };
}

/// `h[i] ^= v[i] ^ v[i + 8]` for both groups, written back in place.
///
/// One `vpternlogd` each, since `0x96` is the truth table of a three-way xor.
macro_rules! feed_forward {
    ($($i:literal $j:literal $vi:literal $vj:literal),*) => {
        concat!($(
            "vpternlogd zmm", $i, ", zmm", $vi, ", zmmword ptr [{h} + 64*", $i, "], 0x96\n",
            "vmovdqu64 zmmword ptr [{h} + 64*", $i, "], zmm", $i, "\n",
            "vpternlogd zmm", $j, ", zmm", $vj, ", zmmword ptr [{h} + 512 + 64*", $i, "], 0x96\n",
            "vmovdqu64 zmmword ptr [{h} + 512 + 64*", $i, "], zmm", $j, "\n",
        )*)
    };
}

/// Advance both groups by one block, in one hand-scheduled kernel.
///
/// Both working vectors fill the whole register file:
///
/// ```text
///     zmm0  .. zmm15    group 0
///     zmm16 .. zmm31    group 1
/// ```
///
/// Every update is then in place, and every message word is a memory operand.
/// The compiler spills at this pressure, hence assembly.
///
/// Equal to the generic compression, which the known answers pin it to.
#[inline(always)]
pub(in crate::batch) fn compress(
    h: &mut [[__m512i; STATE_WORDS]; GROUPS],
    m: &[[__m512i; BLOCK_WORDS]; GROUPS],
    counter: u64,
    last: bool,
) {
    let iv: &'static [u32; STATE_WORDS] = &IV;
    // SAFETY: AVX-512F is enabled, as this module requires.
    // `h` is 2 * 8 vectors read and written, each block of `m` is 16 vectors read, `IV` is 8 words read.
    // The kernel touches no other memory and no stack.
    unsafe {
        core::arch::asm!(
            // v[0..8] = h, and v[8..16] = the IV with the counter and final flag folded in.
            init!(
                0 16 8 24 "dword ptr [{iv}]",
                1 17 9 25 "dword ptr [{iv} + 4]",
                2 18 10 26 "dword ptr [{iv} + 8]",
                3 19 11 27 "dword ptr [{iv} + 12]",
                4 20 12 28 "{t0:e}",
                5 21 13 29 "{t1:e}",
                6 22 14 30 "{f:e}",
                7 23 15 31 "dword ptr [{iv} + 28]"
            ),
            // The ten rounds of RFC 7693, both groups in lockstep.
            round!([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
            round!([14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3]),
            round!([11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4]),
            round!([7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8]),
            round!([9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13]),
            round!([2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9]),
            round!([12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11]),
            round!([13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10]),
            round!([6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5]),
            round!([10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0]),
            feed_forward!(
                0 16 8 24, 1 17 9 25, 2 18 10 26, 3 19 11 27, 4 20 12 28, 5 21 13 29, 6 22 14 30, 7 23 15 31
            ),
            h = in(reg) h.as_mut_ptr(),
            ma = in(reg) m[0].as_ptr(),
            mb = in(reg) m[1].as_ptr(),
            iv = in(reg) iv.as_ptr(),
            t0 = in(reg) iv[4] ^ counter as u32,
            t1 = in(reg) iv[5] ^ (counter >> 32) as u32,
            f = in(reg) if last { !iv[6] } else { iv[6] },
            out("zmm0") _, out("zmm1") _, out("zmm2") _, out("zmm3") _,
            out("zmm4") _, out("zmm5") _, out("zmm6") _, out("zmm7") _,
            out("zmm8") _, out("zmm9") _, out("zmm10") _, out("zmm11") _,
            out("zmm12") _, out("zmm13") _, out("zmm14") _, out("zmm15") _,
            out("zmm16") _, out("zmm17") _, out("zmm18") _, out("zmm19") _,
            out("zmm20") _, out("zmm21") _, out("zmm22") _, out("zmm23") _,
            out("zmm24") _, out("zmm25") _, out("zmm26") _, out("zmm27") _,
            out("zmm28") _, out("zmm29") _, out("zmm30") _, out("zmm31") _,
            options(nostack),
        );
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;
    use crate::batch::compress;
    use crate::batch::lanes::to_lanes;

    /// Vectors filled lane by lane from a flat list of words.
    fn vectors<const N: usize>(words: &[u32]) -> [__m512i; N] {
        // SAFETY: a vector is exactly sixteen packed words, lane 0 first.
        core::array::from_fn(|i| unsafe { _mm512_loadu_si512(words[16 * i..].as_ptr().cast()) })
    }

    proptest! {
        #[test]
        fn the_kernel_is_the_generic_compression(
            h in prop::collection::vec(any::<u32>(), 2 * 8 * 16),
            m in prop::collection::vec(any::<u32>(), 2 * 16 * 16),
            counter in any::<u64>(),
            last in any::<bool>(),
        ) {
            // Every lane of both groups gets its own chaining value and block.
            let h: [[__m512i; STATE_WORDS]; GROUPS] =
                core::array::from_fn(|g| vectors(&h[g * 8 * 16..]));
            let m: [[__m512i; BLOCK_WORDS]; GROUPS] =
                core::array::from_fn(|g| vectors(&m[g * 16 * 16..]));

            // Kernel and generic rounds advance copies of the same state.
            let (mut kernel, mut generic) = (h, h);
            compress(&mut kernel, &m, counter, last);
            compress::compress(&mut generic, &m, counter, last);
            for (kernel, generic) in kernel.iter().flatten().zip(generic.iter().flatten()) {
                prop_assert_eq!(to_lanes(*kernel), to_lanes(*generic));
            }
        }
    }
}
