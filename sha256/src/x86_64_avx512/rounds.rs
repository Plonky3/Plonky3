//! The 64 rounds of SHA-256 on two groups of sixteen lanes, written in `asm!`.
//!
//! The eight working words of both groups take 16 of the 32 vector registers.
//! The other 16 are scratch, so no round ever spills.
//!
//! One group alone runs the same rounds in its half of the registers.
//!
//! The message schedule lives in a stack buffer, and enters the rounds as memory operands.
//!
//! # Performance
//!
//! Each big sigma costs three rotations, and only two pipes can rotate.
//!
//! A single group leaves them idle while it waits on its own dependency chain.
//! A second group fills those gaps, but then the compiler spills the working words.
//!
//! So the rounds are written out by hand.

use core::arch::x86_64::__m512i;
use core::mem::MaybeUninit;

use super::{BLOCK_WORDS, GROUPS, ROUNDS, STATE_WORDS};

/// The round constants of FIPS 180-4 section 4.2.2.
///
/// They are the first 32 bits of the fractional parts of the cube roots of the first 64 primes.
#[rustfmt::skip]
pub(super) static K: [u32; ROUNDS] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

/// Add `K_t + W_t` into `h` of both groups, for round `j` of an eight-round pass.
///
/// Each entry is `[h t0 t1 t2 t3 p wo]`, as in the round below.
macro_rules! round_input {
    // Rounds 0 to 15: W_t is a word of the block.
    (block $j:literal $([$h:literal $t0:literal $t1:literal $t2:literal $t3:literal $p:literal $wo:literal])*) => {
        concat!(
            $("vpaddd zmm", $h, ", zmm", $h, ", zmmword ptr [{w} + ", $wo, " + 128 * ", $j, "]\n",)*
            $("vpaddd zmm", $h, ", zmm", $h, ", dword ptr [{k} + 4 * ", $j, "]{{1to16}}\n",)*
        )
    };
    // Rounds 16 to 63: W_t = s1(W_{t-2}) + W_{t-7} + s0(W_{t-15}) + W_{t-16}.
    //
    // W_{t-2} waits in register p, which then takes W_t.
    // The older words come back from the schedule buffer.
    (expand $j:literal $([$h:literal $t0:literal $t1:literal $t2:literal $t3:literal $p:literal $wo:literal])*) => {
        concat!(
            // s1(W_{t-2}) = ROTR^17 ^ ROTR^19 ^ SHR^10, into t3.
            $("vprord zmm", $t1, ", zmm", $p, ", 17\n",)*
            $("vprord zmm", $t2, ", zmm", $p, ", 19\n",)*
            $("vpsrld zmm", $t3, ", zmm", $p, ", 10\n",)*
            $("vpternlogd zmm", $t3, ", zmm", $t1, ", zmm", $t2, ", 0x96\n",)*
            // s0(W_{t-15}) = ROTR^7 ^ ROTR^18 ^ SHR^3, into t0.
            $("vmovdqa64 zmm", $t0, ", zmmword ptr [{w} + ", $wo, " + 128 * (", $j, " - 15)]\n",)*
            $("vprord zmm", $t1, ", zmm", $t0, ", 7\n",)*
            $("vprord zmm", $t2, ", zmm", $t0, ", 18\n",)*
            $("vpsrld zmm", $t0, ", zmm", $t0, ", 3\n",)*
            $("vpternlogd zmm", $t0, ", zmm", $t1, ", zmm", $t2, ", 0x96\n",)*
            // W_t into p, and into the buffer for the rounds that reach further back.
            $("vpaddd zmm", $p, ", zmm", $t0, ", zmmword ptr [{w} + ", $wo, " + 128 * (", $j, " - 16)]\n",)*
            $("vpaddd zmm", $p, ", zmm", $p, ", zmmword ptr [{w} + ", $wo, " + 128 * (", $j, " - 7)]\n",)*
            $("vpaddd zmm", $p, ", zmm", $p, ", zmm", $t3, "\n",)*
            $("vmovdqa64 zmmword ptr [{w} + ", $wo, " + 128 * ", $j, "], zmm", $p, "\n",)*
            $("vpaddd zmm", $h, ", zmm", $h, ", zmm", $p, "\n",)*
            $("vpaddd zmm", $h, ", zmm", $h, ", dword ptr [{k} + 4 * ", $j, "]{{1to16}}\n",)*
        )
    };
    // A block every lane shares: K_t + W_t is one precomputed word, broadcast.
    (shared $j:literal $([$h:literal $t0:literal $t1:literal $t2:literal $t3:literal $p:literal $wo:literal])*) => {
        concat!(
            $("vpaddd zmm", $h, ", zmm", $h, ", dword ptr [{k} + 4 * ", $j, "]{{1to16}}\n",)*
        )
    };
}

/// One round of both groups, issued one instruction at a time across the two.
///
/// Each entry is `[a b c d e f g h t0 t1 t2 t3 p wo]`:
///
/// - `a` to `h`: the registers holding the eight working words;
/// - `t0` to `t3`: scratch registers;
/// - `p`: the schedule register of this round's parity;
/// - `wo`: the byte offset of this group's words in the schedule buffer.
///
/// The ternary-logic immediates are truth tables:
///
/// - `0x96`: `x ^ y ^ z`;
/// - `0xCA`: `x ? y : z`, which is Ch;
/// - `0xE8`: the bitwise majority, which is Maj.
macro_rules! round {
    ($input:ident $j:literal $([
        $a:literal $b:literal $c:literal $d:literal $e:literal $f:literal $g:literal $h:literal
        $t0:literal $t1:literal $t2:literal $t3:literal $p:literal $wo:literal
    ])*) => {
        concat!(
            // h + K_t + W_t, off the dependency chain.
            round_input!($input $j $([$h $t0 $t1 $t2 $t3 $p $wo])*),
            // S1(e) = ROTR^6 ^ ROTR^11 ^ ROTR^25, into t1.
            //
            // Ch(e, f, g) into t0.
            // A ternary op overwrites its first operand, and e lives on as the next f, so t0 takes a copy.
            //
            // Both join h, which is then T1.
            $("vprord zmm", $t1, ", zmm", $e, ", 6\n",)*
            $("vprord zmm", $t2, ", zmm", $e, ", 11\n",)*
            $("vprord zmm", $t3, ", zmm", $e, ", 25\n",)*
            $("vmovdqa64 zmm", $t0, ", zmm", $e, "\n",)*
            $("vpternlogd zmm", $t0, ", zmm", $f, ", zmm", $g, ", 0xCA\n",)*
            $("vpternlogd zmm", $t1, ", zmm", $t2, ", zmm", $t3, ", 0x96\n",)*
            $("vpaddd zmm", $h, ", zmm", $h, ", zmm", $t0, "\n",)*
            $("vpaddd zmm", $h, ", zmm", $h, ", zmm", $t1, "\n",)*
            // d + T1 is the next e.
            $("vpaddd zmm", $d, ", zmm", $d, ", zmm", $h, "\n",)*
            // S0(a) = ROTR^2 ^ ROTR^13 ^ ROTR^22 and Maj(a, b, c), both joining T1: the next a.
            $("vprord zmm", $t1, ", zmm", $a, ", 2\n",)*
            $("vprord zmm", $t2, ", zmm", $a, ", 13\n",)*
            $("vprord zmm", $t3, ", zmm", $a, ", 22\n",)*
            $("vmovdqa64 zmm", $t0, ", zmm", $a, "\n",)*
            $("vpternlogd zmm", $t0, ", zmm", $b, ", zmm", $c, ", 0xE8\n",)*
            $("vpternlogd zmm", $t1, ", zmm", $t2, ", zmm", $t3, ", 0x96\n",)*
            $("vpaddd zmm", $h, ", zmm", $h, ", zmm", $t0, "\n",)*
            $("vpaddd zmm", $h, ", zmm", $h, ", zmm", $t1, "\n",)*
        )
    };
}

/// Eight rounds of both groups.
///
/// The words never move between registers.
/// Instead the names rotate by one register per round, so the new a lands where h was.
///
/// After eight rounds every name is back home, so the kernels loop over this body.
///
/// Registers of each group:
///
/// - group A: words in zmm0 to zmm7, scratch in zmm16 to zmm19, schedule in zmm24 and zmm25;
/// - group B: words in zmm8 to zmm15, scratch in zmm20 to zmm23, schedule in zmm26 and zmm27.
macro_rules! eight_rounds {
    ($input:ident) => {
        concat!(
            round!($input 0 [0 1 2 3 4 5 6 7 16 17 18 19 24 0] [8 9 10 11 12 13 14 15 20 21 22 23 26 64]),
            round!($input 1 [7 0 1 2 3 4 5 6 16 17 18 19 25 0] [15 8 9 10 11 12 13 14 20 21 22 23 27 64]),
            round!($input 2 [6 7 0 1 2 3 4 5 16 17 18 19 24 0] [14 15 8 9 10 11 12 13 20 21 22 23 26 64]),
            round!($input 3 [5 6 7 0 1 2 3 4 16 17 18 19 25 0] [13 14 15 8 9 10 11 12 20 21 22 23 27 64]),
            round!($input 4 [4 5 6 7 0 1 2 3 16 17 18 19 24 0] [12 13 14 15 8 9 10 11 20 21 22 23 26 64]),
            round!($input 5 [3 4 5 6 7 0 1 2 16 17 18 19 25 0] [11 12 13 14 15 8 9 10 20 21 22 23 27 64]),
            round!($input 6 [2 3 4 5 6 7 0 1 16 17 18 19 24 0] [10 11 12 13 14 15 8 9 20 21 22 23 26 64]),
            round!($input 7 [1 2 3 4 5 6 7 0 16 17 18 19 25 0] [9 10 11 12 13 14 15 8 20 21 22 23 27 64]),
        )
    };
}

/// Eight rounds of group A alone, in the registers group A uses above.
macro_rules! eight_rounds_one {
    ($input:ident) => {
        concat!(
            round!($input 0 [0 1 2 3 4 5 6 7 16 17 18 19 24 0]),
            round!($input 1 [7 0 1 2 3 4 5 6 16 17 18 19 25 0]),
            round!($input 2 [6 7 0 1 2 3 4 5 16 17 18 19 24 0]),
            round!($input 3 [5 6 7 0 1 2 3 4 16 17 18 19 25 0]),
            round!($input 4 [4 5 6 7 0 1 2 3 16 17 18 19 24 0]),
            round!($input 5 [3 4 5 6 7 0 1 2 16 17 18 19 25 0]),
            round!($input 6 [2 3 4 5 6 7 0 1 16 17 18 19 24 0]),
            round!($input 7 [1 2 3 4 5 6 7 0 16 17 18 19 25 0]),
        )
    };
}

/// W_14 and W_15 of both groups, the inputs of the first two extensions.
macro_rules! seed_schedule {
    () => {
        concat!(
            "vmovdqa64 zmm24, zmmword ptr [{w} - 128 * 2]\n",
            "vmovdqa64 zmm25, zmmword ptr [{w} - 128 * 1]\n",
            "vmovdqa64 zmm26, zmmword ptr [{w} + 64 - 128 * 2]\n",
            "vmovdqa64 zmm27, zmmword ptr [{w} + 64 - 128 * 1]\n",
        )
    };
}

/// W_14 and W_15 of group A alone.
macro_rules! seed_schedule_one {
    () => {
        concat!(
            "vmovdqa64 zmm24, zmmword ptr [{w} - 128 * 2]\n",
            "vmovdqa64 zmm25, zmmword ptr [{w} - 128 * 1]\n",
        )
    };
}

/// Load the chaining values of group A into zmm0 to zmm7.
macro_rules! load_state_one {
    () => {
        concat!(
            "vmovdqu64 zmm0, zmmword ptr [{h} + 64 * 0]\n",
            "vmovdqu64 zmm1, zmmword ptr [{h} + 64 * 1]\n",
            "vmovdqu64 zmm2, zmmword ptr [{h} + 64 * 2]\n",
            "vmovdqu64 zmm3, zmmword ptr [{h} + 64 * 3]\n",
            "vmovdqu64 zmm4, zmmword ptr [{h} + 64 * 4]\n",
            "vmovdqu64 zmm5, zmmword ptr [{h} + 64 * 5]\n",
            "vmovdqu64 zmm6, zmmword ptr [{h} + 64 * 6]\n",
            "vmovdqu64 zmm7, zmmword ptr [{h} + 64 * 7]\n",
        )
    };
}

/// Feed-forward of group A alone.
macro_rules! feed_forward_one {
    () => {
        concat!(
            "vpaddd zmm0, zmm0, zmmword ptr [{h} + 64 * 0]\n",
            "vpaddd zmm1, zmm1, zmmword ptr [{h} + 64 * 1]\n",
            "vpaddd zmm2, zmm2, zmmword ptr [{h} + 64 * 2]\n",
            "vpaddd zmm3, zmm3, zmmword ptr [{h} + 64 * 3]\n",
            "vpaddd zmm4, zmm4, zmmword ptr [{h} + 64 * 4]\n",
            "vpaddd zmm5, zmm5, zmmword ptr [{h} + 64 * 5]\n",
            "vpaddd zmm6, zmm6, zmmword ptr [{h} + 64 * 6]\n",
            "vpaddd zmm7, zmm7, zmmword ptr [{h} + 64 * 7]\n",
            "vmovdqu64 zmmword ptr [{h} + 64 * 0], zmm0\n",
            "vmovdqu64 zmmword ptr [{h} + 64 * 1], zmm1\n",
            "vmovdqu64 zmmword ptr [{h} + 64 * 2], zmm2\n",
            "vmovdqu64 zmmword ptr [{h} + 64 * 3], zmm3\n",
            "vmovdqu64 zmmword ptr [{h} + 64 * 4], zmm4\n",
            "vmovdqu64 zmmword ptr [{h} + 64 * 5], zmm5\n",
            "vmovdqu64 zmmword ptr [{h} + 64 * 6], zmm6\n",
            "vmovdqu64 zmmword ptr [{h} + 64 * 7], zmm7\n",
        )
    };
}

/// Load the chaining values of both groups into zmm0 to zmm15.
macro_rules! load_state {
    () => {
        concat!(
            "vmovdqu64 zmm0, zmmword ptr [{h} + 64 * 0]\n",
            "vmovdqu64 zmm1, zmmword ptr [{h} + 64 * 1]\n",
            "vmovdqu64 zmm2, zmmword ptr [{h} + 64 * 2]\n",
            "vmovdqu64 zmm3, zmmword ptr [{h} + 64 * 3]\n",
            "vmovdqu64 zmm4, zmmword ptr [{h} + 64 * 4]\n",
            "vmovdqu64 zmm5, zmmword ptr [{h} + 64 * 5]\n",
            "vmovdqu64 zmm6, zmmword ptr [{h} + 64 * 6]\n",
            "vmovdqu64 zmm7, zmmword ptr [{h} + 64 * 7]\n",
            "vmovdqu64 zmm8, zmmword ptr [{h} + 64 * 8]\n",
            "vmovdqu64 zmm9, zmmword ptr [{h} + 64 * 9]\n",
            "vmovdqu64 zmm10, zmmword ptr [{h} + 64 * 10]\n",
            "vmovdqu64 zmm11, zmmword ptr [{h} + 64 * 11]\n",
            "vmovdqu64 zmm12, zmmword ptr [{h} + 64 * 12]\n",
            "vmovdqu64 zmm13, zmmword ptr [{h} + 64 * 13]\n",
            "vmovdqu64 zmm14, zmmword ptr [{h} + 64 * 14]\n",
            "vmovdqu64 zmm15, zmmword ptr [{h} + 64 * 15]\n",
        )
    };
}

/// Feed-forward: add the chaining values into the working words, and store the sums back.
macro_rules! feed_forward {
    () => {
        concat!(
            "vpaddd zmm0, zmm0, zmmword ptr [{h} + 64 * 0]\n",
            "vpaddd zmm1, zmm1, zmmword ptr [{h} + 64 * 1]\n",
            "vpaddd zmm2, zmm2, zmmword ptr [{h} + 64 * 2]\n",
            "vpaddd zmm3, zmm3, zmmword ptr [{h} + 64 * 3]\n",
            "vpaddd zmm4, zmm4, zmmword ptr [{h} + 64 * 4]\n",
            "vpaddd zmm5, zmm5, zmmword ptr [{h} + 64 * 5]\n",
            "vpaddd zmm6, zmm6, zmmword ptr [{h} + 64 * 6]\n",
            "vpaddd zmm7, zmm7, zmmword ptr [{h} + 64 * 7]\n",
            "vpaddd zmm8, zmm8, zmmword ptr [{h} + 64 * 8]\n",
            "vpaddd zmm9, zmm9, zmmword ptr [{h} + 64 * 9]\n",
            "vpaddd zmm10, zmm10, zmmword ptr [{h} + 64 * 10]\n",
            "vpaddd zmm11, zmm11, zmmword ptr [{h} + 64 * 11]\n",
            "vpaddd zmm12, zmm12, zmmword ptr [{h} + 64 * 12]\n",
            "vpaddd zmm13, zmm13, zmmword ptr [{h} + 64 * 13]\n",
            "vpaddd zmm14, zmm14, zmmword ptr [{h} + 64 * 14]\n",
            "vpaddd zmm15, zmm15, zmmword ptr [{h} + 64 * 15]\n",
            "vmovdqu64 zmmword ptr [{h} + 64 * 0], zmm0\n",
            "vmovdqu64 zmmword ptr [{h} + 64 * 1], zmm1\n",
            "vmovdqu64 zmmword ptr [{h} + 64 * 2], zmm2\n",
            "vmovdqu64 zmmword ptr [{h} + 64 * 3], zmm3\n",
            "vmovdqu64 zmmword ptr [{h} + 64 * 4], zmm4\n",
            "vmovdqu64 zmmword ptr [{h} + 64 * 5], zmm5\n",
            "vmovdqu64 zmmword ptr [{h} + 64 * 6], zmm6\n",
            "vmovdqu64 zmmword ptr [{h} + 64 * 7], zmm7\n",
            "vmovdqu64 zmmword ptr [{h} + 64 * 8], zmm8\n",
            "vmovdqu64 zmmword ptr [{h} + 64 * 9], zmm9\n",
            "vmovdqu64 zmmword ptr [{h} + 64 * 10], zmm10\n",
            "vmovdqu64 zmmword ptr [{h} + 64 * 11], zmm11\n",
            "vmovdqu64 zmmword ptr [{h} + 64 * 12], zmm12\n",
            "vmovdqu64 zmmword ptr [{h} + 64 * 13], zmm13\n",
            "vmovdqu64 zmmword ptr [{h} + 64 * 14], zmm14\n",
            "vmovdqu64 zmmword ptr [{h} + 64 * 15], zmm15\n",
        )
    };
}

/// The block kernel, for the group count its three macros cover.
///
/// Rounds 0 to 15 read the block, in two passes of eight.
/// Rounds 16 to 63 extend the schedule as they go, in six passes of eight.
macro_rules! block_kernel {
    ($load:ident, $seed:ident, $rounds:ident, $feed_forward:ident, $h:expr, $w:expr) => {
        core::arch::asm!(
            $load!(),
            "mov {n:e}, 2",
            ".p2align 6",
            "2:",
            $rounds!(block),
            "add {w}, 1024",
            "add {k}, 32",
            "dec {n:e}",
            "jnz 2b",
            $seed!(),
            "mov {n:e}, 6",
            ".p2align 6",
            "3:",
            $rounds!(expand),
            "add {w}, 1024",
            "add {k}, 32",
            "dec {n:e}",
            "jnz 3b",
            $feed_forward!(),
            h = in(reg) $h,
            w = inout(reg) $w => _,
            k = inout(reg) K.as_ptr() => _,
            n = out(reg) _,
            out("zmm0") _, out("zmm1") _, out("zmm2") _, out("zmm3") _,
            out("zmm4") _, out("zmm5") _, out("zmm6") _, out("zmm7") _,
            out("zmm8") _, out("zmm9") _, out("zmm10") _, out("zmm11") _,
            out("zmm12") _, out("zmm13") _, out("zmm14") _, out("zmm15") _,
            out("zmm16") _, out("zmm17") _, out("zmm18") _, out("zmm19") _,
            out("zmm20") _, out("zmm21") _, out("zmm22") _, out("zmm23") _,
            out("zmm24") _, out("zmm25") _, out("zmm26") _, out("zmm27") _,
            out("zmm28") _, out("zmm29") _, out("zmm30") _, out("zmm31") _,
            options(nostack),
        )
    };
}

/// The shared-block kernel: all 64 rounds, in eight passes of eight.
macro_rules! shared_kernel {
    ($load:ident, $rounds:ident, $feed_forward:ident, $h:expr, $kw:expr) => {
        core::arch::asm!(
            $load!(),
            "mov {n:e}, 8",
            ".p2align 6",
            "2:",
            $rounds!(shared),
            "add {k}, 32",
            "dec {n:e}",
            "jnz 2b",
            $feed_forward!(),
            h = in(reg) $h,
            k = inout(reg) $kw => _,
            n = out(reg) _,
            out("zmm0") _, out("zmm1") _, out("zmm2") _, out("zmm3") _,
            out("zmm4") _, out("zmm5") _, out("zmm6") _, out("zmm7") _,
            out("zmm8") _, out("zmm9") _, out("zmm10") _, out("zmm11") _,
            out("zmm12") _, out("zmm13") _, out("zmm14") _, out("zmm15") _,
            out("zmm16") _, out("zmm17") _, out("zmm18") _, out("zmm19") _,
            out("zmm20") _, out("zmm21") _, out("zmm22") _, out("zmm23") _,
            out("zmm24") _, out("zmm25") _, out("zmm26") _, out("zmm27") _,
            out("zmm28") _, out("zmm29") _, out("zmm30") _, out("zmm31") _,
            options(nostack),
        )
    };
}

/// Advance `G` groups by one block each, for `G` of one or two.
///
/// `block[g][w]` holds message word `w` of every lane of group `g`.
#[inline]
#[target_feature(enable = "avx512f")]
pub(super) fn compress_blocks<const G: usize>(
    state: &mut [[__m512i; STATE_WORDS]; G],
    block: &[[__m512i; BLOCK_WORDS]; G],
) {
    const { assert!(G == 1 || G == 2) };

    // The schedule buffer: slot t holds W_t of group A, then W_t of group B.
    //
    // The block opens it, and the kernel writes the other 48 slots before reading them.
    //
    // A lone group uses only the first half of every slot.
    let mut w = [[MaybeUninit::<__m512i>::uninit(); GROUPS]; ROUNDS];
    for (t, w) in w[..BLOCK_WORDS].iter_mut().enumerate() {
        for (w, block) in w.iter_mut().zip(block) {
            w.write(block[t]);
        }
    }

    // SAFETY:
    // - the function enables AVX-512F, which every instruction below needs;
    // - `state` is `G` times 512 bytes, read and then written in place;
    // - `w` is 8192 bytes: the block's slots written above, and the rest written before any read;
    // - `K` is 256 bytes, only read;
    // - every vector register is declared clobbered, and the stack is never touched.
    unsafe {
        if G == 2 {
            block_kernel!(
                load_state,
                seed_schedule,
                eight_rounds,
                feed_forward,
                state.as_mut_ptr(),
                w.as_mut_ptr()
            );
        } else {
            block_kernel!(
                load_state_one,
                seed_schedule_one,
                eight_rounds_one,
                feed_forward_one,
                state.as_mut_ptr(),
                w.as_mut_ptr()
            );
        }
    }
}

/// Advance `G` groups by one block that every lane shares, for `G` of one or two.
///
/// `kw[t]` holds `K_t + W_t`, so the kernel computes no schedule.
#[inline]
#[target_feature(enable = "avx512f")]
pub(super) fn compress_shared<const G: usize>(
    state: &mut [[__m512i; STATE_WORDS]; G],
    kw: &[u32; ROUNDS],
) {
    const { assert!(G == 1 || G == 2) };

    // SAFETY:
    // - the function enables AVX-512F, which every instruction below needs;
    // - `state` is `G` times 512 bytes, read and then written in place;
    // - `kw` is 256 bytes, only read;
    // - every vector register is declared clobbered, and the stack is never touched.
    unsafe {
        if G == 2 {
            shared_kernel!(
                load_state,
                eight_rounds,
                feed_forward,
                state.as_mut_ptr(),
                kw.as_ptr()
            );
        } else {
            shared_kernel!(
                load_state_one,
                eight_rounds_one,
                feed_forward_one,
                state.as_mut_ptr(),
                kw.as_ptr()
            );
        }
    }
}
