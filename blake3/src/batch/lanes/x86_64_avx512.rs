//! Sixteen lanes per 512-bit AVX-512 register.

use core::arch::x86_64::*;

use blake3::{BLOCK_LEN, OUT_LEN};

use super::{Backend, Kernel, Word};
use crate::batch::compress::{BLOCK_WORDS, STATE_WORDS};

/// Lanes in one register.
const WIDTH: usize = 16;

/// Independent register groups hashed together.
///
/// Two groups fill all 32 registers, driven by the hand-scheduled kernel below.
///
/// Four only spill their transposes.
const GROUPS: usize = 2;

// SAFETY (every block below): the driver only runs this backend on a CPU with AVX-512F.
impl Word for __m512i {
    #[inline(always)]
    fn compress_scheduled<const G: usize>(
        h: &mut [[Self; STATE_WORDS]; G],
        m: &[[Self; BLOCK_WORDS]; G],
        params: &[u32; STATE_WORDS],
    ) -> bool {
        // The pair kernel needs exactly two groups, and `G` is a constant, so this folds away.
        let (Ok(h), Ok(m)) = (h.as_mut_slice().try_into(), m.as_slice().try_into()) else {
            return false;
        };
        compress_pair(h, m, params);
        true
    }

    #[inline(always)]
    fn compress_scheduled_counters<const G: usize>(
        h: &mut [[Self; STATE_WORDS]; G],
        m: &[[Self; BLOCK_WORDS]; G],
        params: &[u32; STATE_WORDS],
        counters: &[[Self; 2]; G],
    ) -> bool {
        // As above, only the two-group shape has a hand-scheduled kernel.
        let (Ok(h), Ok(m), Ok(counters)) = (
            h.as_mut_slice().try_into(),
            m.as_slice().try_into(),
            counters.as_slice().try_into(),
        ) else {
            return false;
        };
        compress_pair_counters(h, m, params, counters);
        true
    }

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

/// A 4 x 4 transpose inside every 128-bit block of four rows.
///
/// Block `k` of output `j` holds word `4k + j` of rows `a`, `b`, `c` and `d`, in that order.
#[inline]
#[target_feature(enable = "avx512f")]
fn transpose_blocks(a: __m512i, b: __m512i, c: __m512i, d: __m512i) -> [__m512i; 4] {
    // Interleave 32-bit words of row pairs, then 64-bit pairs of those.
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
}

/// Eight G functions, four per group, issued one step at a time across all eight.
///
/// Each entry is `[a b c d x y base]`:
///
/// - `a b c d`: the registers holding the four working words;
/// - `x y`: the two message words, as indices into the block;
/// - `base`: the byte offset of that group's block.
macro_rules! half_round {
    ($([$a:literal $b:literal $c:literal $d:literal $x:literal $y:literal $base:literal])*) => {
        concat!(
            $(
                "vpaddd zmm", $a, ", zmm", $a,
                ", zmmword ptr [{m} + ", $base, " + 64 * ", $x, "]\n",
            )*
            $("vpaddd zmm", $a, ", zmm", $a, ", zmm", $b, "\n",)*
            $("vpxord zmm", $d, ", zmm", $d, ", zmm", $a, "\n",)*
            $("vprord zmm", $d, ", zmm", $d, ", 16\n",)*
            $("vpaddd zmm", $c, ", zmm", $c, ", zmm", $d, "\n",)*
            $("vpxord zmm", $b, ", zmm", $b, ", zmm", $c, "\n",)*
            $("vprord zmm", $b, ", zmm", $b, ", 12\n",)*
            $(
                "vpaddd zmm", $a, ", zmm", $a,
                ", zmmword ptr [{m} + ", $base, " + 64 * ", $y, "]\n",
            )*
            $("vpaddd zmm", $a, ", zmm", $a, ", zmm", $b, "\n",)*
            $("vpxord zmm", $d, ", zmm", $d, ", zmm", $a, "\n",)*
            $("vprord zmm", $d, ", zmm", $d, ", 8\n",)*
            $("vpaddd zmm", $c, ", zmm", $c, ", zmm", $d, "\n",)*
            $("vpxord zmm", $b, ", zmm", $b, ", zmm", $c, "\n",)*
            $("vprord zmm", $b, ", zmm", $b, ", 7\n",)*
        )
    };
}

/// One round of both groups, given the message schedule of that round.
///
/// Group A holds `v` in zmm0 to zmm15 and group B in zmm16 to zmm31.
macro_rules! round {
    (
        $s0:literal $s1:literal $s2:literal $s3:literal
        $s4:literal $s5:literal $s6:literal $s7:literal
        $s8:literal $s9:literal $s10:literal $s11:literal
        $s12:literal $s13:literal $s14:literal $s15:literal
    ) => {
        concat!(
            // Columns.
            half_round!(
                [0 4 8 12 $s0 $s1 0] [1 5 9 13 $s2 $s3 0]
                [2 6 10 14 $s4 $s5 0] [3 7 11 15 $s6 $s7 0]
                [16 20 24 28 $s0 $s1 1024] [17 21 25 29 $s2 $s3 1024]
                [18 22 26 30 $s4 $s5 1024] [19 23 27 31 $s6 $s7 1024]
            ),
            // Diagonals.
            half_round!(
                [0 5 10 15 $s8 $s9 0] [1 6 11 12 $s10 $s11 0]
                [2 7 8 13 $s12 $s13 0] [3 4 9 14 $s14 $s15 0]
                [16 21 26 31 $s8 $s9 1024] [17 22 27 28 $s10 $s11 1024]
                [18 23 24 29 $s12 $s13 1024] [19 20 25 30 $s14 $s15 1024]
            ),
        )
    };
}

/// The two-group compression, with optional template lines run just before the rounds.
///
/// Those lines may overwrite any word of the working vector, such as the counter words.
///
/// Their operands follow the fixed ones.
macro_rules! compress_pair_asm {
    ($h:ident, $m:ident, $params:ident, [$($extra:literal),*], $($operands:tt)*) => {
        core::arch::asm!(
            // Chaining values into v[0..8] of both groups.
            "vmovdqu64 zmm0, zmmword ptr [{h} + 64 * 0]",
            "vmovdqu64 zmm1, zmmword ptr [{h} + 64 * 1]",
            "vmovdqu64 zmm2, zmmword ptr [{h} + 64 * 2]",
            "vmovdqu64 zmm3, zmmword ptr [{h} + 64 * 3]",
            "vmovdqu64 zmm4, zmmword ptr [{h} + 64 * 4]",
            "vmovdqu64 zmm5, zmmword ptr [{h} + 64 * 5]",
            "vmovdqu64 zmm6, zmmword ptr [{h} + 64 * 6]",
            "vmovdqu64 zmm7, zmmword ptr [{h} + 64 * 7]",
            "vmovdqu64 zmm16, zmmword ptr [{h} + 512 + 64 * 0]",
            "vmovdqu64 zmm17, zmmword ptr [{h} + 512 + 64 * 1]",
            "vmovdqu64 zmm18, zmmword ptr [{h} + 512 + 64 * 2]",
            "vmovdqu64 zmm19, zmmword ptr [{h} + 512 + 64 * 3]",
            "vmovdqu64 zmm20, zmmword ptr [{h} + 512 + 64 * 4]",
            "vmovdqu64 zmm21, zmmword ptr [{h} + 512 + 64 * 5]",
            "vmovdqu64 zmm22, zmmword ptr [{h} + 512 + 64 * 6]",
            "vmovdqu64 zmm23, zmmword ptr [{h} + 512 + 64 * 7]",
            // The parameters into v[8..16] of both groups.
            "vpbroadcastd zmm8, dword ptr [{p} + 4 * 0]",
            "vpbroadcastd zmm9, dword ptr [{p} + 4 * 1]",
            "vpbroadcastd zmm10, dword ptr [{p} + 4 * 2]",
            "vpbroadcastd zmm11, dword ptr [{p} + 4 * 3]",
            "vpbroadcastd zmm12, dword ptr [{p} + 4 * 4]",
            "vpbroadcastd zmm13, dword ptr [{p} + 4 * 5]",
            "vpbroadcastd zmm14, dword ptr [{p} + 4 * 6]",
            "vpbroadcastd zmm15, dword ptr [{p} + 4 * 7]",
            "vmovdqa64 zmm24, zmm8",
            "vmovdqa64 zmm25, zmm9",
            "vmovdqa64 zmm26, zmm10",
            "vmovdqa64 zmm27, zmm11",
            "vmovdqa64 zmm28, zmm12",
            "vmovdqa64 zmm29, zmm13",
            "vmovdqa64 zmm30, zmm14",
            "vmovdqa64 zmm31, zmm15",
            // The caller's lines, such as a counter per lane.
            $($extra,)*
            // The rows of `SCHEDULE`, one per round.
            round!(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15),
            round!(2 6 3 10 7 0 4 13 1 11 12 5 9 14 15 8),
            round!(3 4 10 12 13 2 7 14 6 5 9 0 11 15 8 1),
            round!(10 7 12 9 14 3 13 15 4 0 11 2 5 8 1 6),
            round!(12 13 9 11 15 10 14 8 7 2 5 3 0 1 6 4),
            round!(9 14 11 5 8 12 15 1 13 3 0 10 2 6 4 7),
            round!(11 15 5 0 1 9 8 6 14 10 2 12 3 4 7 13),
            // Feed-forward: h'[i] = v[i] ^ v[i + 8].
            "vpxord zmm0, zmm0, zmm8",
            "vpxord zmm1, zmm1, zmm9",
            "vpxord zmm2, zmm2, zmm10",
            "vpxord zmm3, zmm3, zmm11",
            "vpxord zmm4, zmm4, zmm12",
            "vpxord zmm5, zmm5, zmm13",
            "vpxord zmm6, zmm6, zmm14",
            "vpxord zmm7, zmm7, zmm15",
            "vpxord zmm16, zmm16, zmm24",
            "vpxord zmm17, zmm17, zmm25",
            "vpxord zmm18, zmm18, zmm26",
            "vpxord zmm19, zmm19, zmm27",
            "vpxord zmm20, zmm20, zmm28",
            "vpxord zmm21, zmm21, zmm29",
            "vpxord zmm22, zmm22, zmm30",
            "vpxord zmm23, zmm23, zmm31",
            "vmovdqu64 zmmword ptr [{h} + 64 * 0], zmm0",
            "vmovdqu64 zmmword ptr [{h} + 64 * 1], zmm1",
            "vmovdqu64 zmmword ptr [{h} + 64 * 2], zmm2",
            "vmovdqu64 zmmword ptr [{h} + 64 * 3], zmm3",
            "vmovdqu64 zmmword ptr [{h} + 64 * 4], zmm4",
            "vmovdqu64 zmmword ptr [{h} + 64 * 5], zmm5",
            "vmovdqu64 zmmword ptr [{h} + 64 * 6], zmm6",
            "vmovdqu64 zmmword ptr [{h} + 64 * 7], zmm7",
            "vmovdqu64 zmmword ptr [{h} + 512 + 64 * 0], zmm16",
            "vmovdqu64 zmmword ptr [{h} + 512 + 64 * 1], zmm17",
            "vmovdqu64 zmmword ptr [{h} + 512 + 64 * 2], zmm18",
            "vmovdqu64 zmmword ptr [{h} + 512 + 64 * 3], zmm19",
            "vmovdqu64 zmmword ptr [{h} + 512 + 64 * 4], zmm20",
            "vmovdqu64 zmmword ptr [{h} + 512 + 64 * 5], zmm21",
            "vmovdqu64 zmmword ptr [{h} + 512 + 64 * 6], zmm22",
            "vmovdqu64 zmmword ptr [{h} + 512 + 64 * 7], zmm23",
            h = in(reg) $h.as_mut_ptr(),
            m = in(reg) $m.as_ptr(),
            p = in(reg) $params.as_ptr(),
            $($operands)*
            out("zmm0") _, out("zmm1") _, out("zmm2") _, out("zmm3") _,
            out("zmm4") _, out("zmm5") _, out("zmm6") _, out("zmm7") _,
            out("zmm8") _, out("zmm9") _, out("zmm10") _, out("zmm11") _,
            out("zmm12") _, out("zmm13") _, out("zmm14") _, out("zmm15") _,
            out("zmm16") _, out("zmm17") _, out("zmm18") _, out("zmm19") _,
            out("zmm20") _, out("zmm21") _, out("zmm22") _, out("zmm23") _,
            out("zmm24") _, out("zmm25") _, out("zmm26") _, out("zmm27") _,
            out("zmm28") _, out("zmm29") _, out("zmm30") _, out("zmm31") _,
            options(nostack, preserves_flags),
        );
    };
}

/// Advance two register groups by one block, entirely inside the 32 vector registers.
///
/// On cores with two-cycle vector latency, one group stalls on the four G chains of a half round.
///
/// Two groups double the chains, but their working vectors take every register.
///
/// The compiler spills under that pressure, so the rounds are written out by hand.
///
/// Message words stay in memory and enter as operands of the adds.
///
/// `params` is the second half of the working vector: IV[0..4], counter, block length, flags.
#[inline(always)]
fn compress_pair(
    h: &mut [[__m512i; STATE_WORDS]; 2],
    m: &[[__m512i; BLOCK_WORDS]; 2],
    params: &[u32; STATE_WORDS],
) {
    // SAFETY:
    // - the driver only runs this backend on a CPU with AVX-512F;
    // - `h` is 1024 bytes of chaining values, read and then written in place;
    // - `m` is 2048 bytes of message words, only read;
    // - `params` is 32 bytes, only read;
    // - every vector register is declared clobbered, and no flags are touched.
    unsafe {
        compress_pair_asm!(h, m, params, [],);
    }
}

/// The two-group compression with a counter per lane.
///
/// `counters` holds, in order, the low and high counter words of group A, then of group B.
///
/// They replace words 12 and 13 of each group's working vector before the first round.
#[inline(always)]
fn compress_pair_counters(
    h: &mut [[__m512i; STATE_WORDS]; 2],
    m: &[[__m512i; BLOCK_WORDS]; 2],
    params: &[u32; STATE_WORDS],
    counters: &[[__m512i; 2]; 2],
) {
    // SAFETY: as for the shared-counter kernel, plus `counters` is 256 bytes, only read.
    unsafe {
        compress_pair_asm!(
            h,
            m,
            params,
            [
                "vmovdqu64 zmm12, zmmword ptr [{c} + 64 * 0]",
                "vmovdqu64 zmm13, zmmword ptr [{c} + 64 * 1]",
                "vmovdqu64 zmm28, zmmword ptr [{c} + 64 * 2]",
                "vmovdqu64 zmm29, zmmword ptr [{c} + 64 * 3]"
            ],
            c = in(reg) counters.as_ptr(),
        );
    }
}

/// Load one block from each of sixteen lanes as sixteen message words.
#[inline]
#[target_feature(enable = "avx512f")]
fn load(rows: &[&[u8; BLOCK_LEN]; WIDTH]) -> [__m512i; BLOCK_WORDS] {
    // SAFETY: each row is 64 readable bytes, and the load has no alignment requirement.
    let r: [__m512i; WIDTH] =
        core::array::from_fn(|l| unsafe { _mm512_loadu_si512(rows[l].as_ptr().cast()) });

    // Phase 1: block k of u[q][j] is word 4k + j of rows 4q to 4q + 3.
    let u: [[__m512i; 4]; 4] = core::array::from_fn(|q| {
        transpose_blocks(r[4 * q], r[4 * q + 1], r[4 * q + 2], r[4 * q + 3])
    });

    // Phase 2: word 4k + j gathers block k of u[0][j] to u[3][j], in quad order.
    let mut out = [r[0]; BLOCK_WORDS];
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

/// Write the digests of sixteen lanes.
///
/// Two adjacent digests fill one 64-byte store, so eight stores cover all sixteen lanes.
#[inline]
#[target_feature(enable = "avx512f")]
fn store(state: &[__m512i; STATE_WORDS], out: &mut [[u8; OUT_LEN]; WIDTH]) {
    // Block k of lo[j] is words 0 to 3 of lane 4k + j, and hi[j] holds words 4 to 7.
    let lo = transpose_blocks(state[0], state[1], state[2], state[3]);
    let hi = transpose_blocks(state[4], state[5], state[6], state[7]);

    // SAFETY: the driver only runs this backend on a CPU with AVX-512F.
    unsafe {
        for j in [0, 2] {
            // Blocks 0 and 1 of each half, then blocks 2 and 3.
            let pairs = [
                _mm512_shuffle_i32x4::<0x44>(lo[j], hi[j]),
                _mm512_shuffle_i32x4::<0x44>(lo[j + 1], hi[j + 1]),
                _mm512_shuffle_i32x4::<0xEE>(lo[j], hi[j]),
                _mm512_shuffle_i32x4::<0xEE>(lo[j + 1], hi[j + 1]),
            ];

            // The digests of lanes 4k + j and 4k + j + 1, back to back.
            let digests = [
                _mm512_shuffle_i32x4::<0x88>(pairs[0], pairs[1]),
                _mm512_shuffle_i32x4::<0xDD>(pairs[0], pairs[1]),
                _mm512_shuffle_i32x4::<0x88>(pairs[2], pairs[3]),
                _mm512_shuffle_i32x4::<0xDD>(pairs[2], pairs[3]),
            ];
            for (k, digests) in digests.into_iter().enumerate() {
                // SAFETY: lanes 4k + j and 4k + j + 1 exist, and their digests are adjacent.
                let pair = out[4 * k + j..][..2].as_flattened_mut();
                _mm512_storeu_si512(pair.as_mut_ptr().cast(), digests);
            }
        }
    }
}

/// The batched driver on this backend.
pub(super) const KERNEL: Kernel = Kernel::new::<__m512i, WIDTH, GROUPS>("AVX-512");

impl Backend<WIDTH> for __m512i {
    #[inline]
    fn supported() -> bool {
        cpufeatures::new!(has_avx512f, "avx512f");
        has_avx512f::get()
    }

    /// Load one block from each of sixteen lanes as sixteen message words.
    #[inline(always)]
    fn load_block(rows: &[&[u8; BLOCK_LEN]; WIDTH]) -> [Self; BLOCK_WORDS] {
        // SAFETY: the driver only runs this backend on a CPU with AVX-512F.
        unsafe { load(rows) }
    }

    /// Write the digests of sixteen lanes.
    ///
    /// Two adjacent digests fill one 64-byte store, so eight stores cover all sixteen lanes.
    #[inline(always)]
    fn store_digests(state: &[Self; STATE_WORDS], out: &mut [[u8; OUT_LEN]; WIDTH]) {
        // SAFETY: the driver only runs this backend on a CPU with AVX-512F.
        unsafe { store(state, out) }
    }

    out_of_line_steps!(WIDTH, "avx512f");
}
