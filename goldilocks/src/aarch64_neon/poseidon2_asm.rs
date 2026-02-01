//! ARM assembly primitives for Poseidon2 on Goldilocks.
//!
//! Latency hiding: ARM mul/umulh have ~4-5 cycle latency. By interleaving
//! S-box computation with MDS operations, we hide much of this latency.

use core::arch::asm;

use crate::{Goldilocks, P};

const EPSILON: u64 = P.wrapping_neg(); // 2^32 - 1

/// Multiply two Goldilocks elements using inline assembly.
/// Returns the product reduced modulo P.
#[inline(always)]
unsafe fn mul_asm(a: u64, b: u64) -> u64 {
    let _lo: u64;
    let _hi: u64;
    let _t0: u64;
    let _t1: u64;
    let _t2: u64;
    let result: u64;

    // SAFETY: Inline assembly performs Goldilocks multiplication with proper reduction
    unsafe {
        asm!(
            // Compute 128-bit product: hi:lo = a * b
            "mul   {lo}, {a}, {b}",
            "umulh {hi}, {a}, {b}",

            // Reduce: result = lo - hi_hi + hi_lo * EPSILON
            // where hi = hi_hi * 2^32 + hi_lo

            // t0 = lo - (hi >> 32), with borrow detection
            "lsr   {t0}, {hi}, #32",          // t0 = hi >> 32
            "subs  {t1}, {lo}, {t0}",         // t1 = lo - t0, set flags
            "csetm {t2:w}, cc",               // t2 = -1 if borrow, 0 otherwise
            "sub   {t1}, {t1}, {t2}",         // Adjust for borrow (subtract EPSILON)

            // t0 = (hi & EPSILON) * EPSILON
            "and   {t0}, {hi}, {epsilon}",    // t0 = hi & EPSILON
            "mul   {t0}, {t0}, {epsilon}",    // t0 = t0 * EPSILON

            // result = t1 + t0, with overflow detection
            "adds  {result}, {t1}, {t0}",     // result = t1 + t0, set flags
            "csetm {t2:w}, cs",               // t2 = -1 if carry, 0 otherwise
            "add   {result}, {result}, {t2}", // Add EPSILON on overflow

            a = in(reg) a,
            b = in(reg) b,
            epsilon = in(reg) EPSILON,
            lo = out(reg) _lo,
            hi = out(reg) _hi,
            t0 = out(reg) _t0,
            t1 = out(reg) _t1,
            t2 = out(reg) _t2,
            result = out(reg) result,
            options(pure, nomem, nostack),
        );
    }

    result
}

/// Add two Goldilocks elements with overflow handling using inline assembly.
#[inline(always)]
unsafe fn add_asm(a: u64, b: u64) -> u64 {
    let result: u64;
    let _adj: u64;

    // SAFETY: Inline assembly performs Goldilocks addition with overflow handling
    unsafe {
        asm!(
            "adds  {result}, {a}, {b}",
            "csetm {adj:w}, cs",
            "add   {result}, {result}, {adj}",
            a = in(reg) a,
            b = in(reg) b,
            result = out(reg) result,
            adj = out(reg) _adj,
            options(pure, nomem, nostack),
        );
    }

    result
}

/// Optimized internal round that interleaves S-box with MDS.
///
/// The internal MDS for Poseidon2 is:
///   sum = state[0] + state[1] + ... + state[WIDTH-1]
///   state[i] = state[i] * diag[i] + sum
///
/// We interleave this with the S-box computation (x^7 = x * x^2 * x^4) to hide
/// multiplication latency.
#[inline(always)]
#[allow(clippy::needless_range_loop)]
pub unsafe fn internal_round_asm<const WIDTH: usize>(
    state: &mut [u64; WIDTH],
    diag: &[u64; WIDTH],
    rc: u64,
) {
    unsafe {
        // Add round constant to state[0]
        let s0 = add_asm(state[0], rc);

        // Start S-box computation - compute s0^2
        let s0_2 = mul_asm(s0, s0);

        // While s0^2 is computing, start summing state[1..WIDTH]
        // This can execute in parallel with the multiplication
        let mut sum_hi: u64 = 0;
        for i in 1..WIDTH {
            sum_hi = add_asm(sum_hi, state[i]);
        }

        // Continue S-box - compute s0^3 and s0^4
        let s0_3 = mul_asm(s0_2, s0);
        let s0_4 = mul_asm(s0_2, s0_2);

        // While s0^3 and s0^4 are computing, start diagonal multiplies
        // for state[1..WIDTH]. Store results temporarily.
        let mut diag_muls: [u64; WIDTH] = [0; WIDTH];
        for i in 1..WIDTH {
            diag_muls[i] = mul_asm(state[i], diag[i]);
        }

        // Finish S-box - compute s0^7
        let s0_7 = mul_asm(s0_3, s0_4);

        // Complete the sum with s0^7
        let sum = add_asm(sum_hi, s0_7);

        // Compute state[0] = s0^7 * diag[0] + sum
        let s0_diag = mul_asm(s0_7, diag[0]);
        state[0] = add_asm(s0_diag, sum);

        // Finalize state[1..WIDTH] = diag_muls[i] + sum
        for i in 1..WIDTH {
            state[i] = add_asm(diag_muls[i], sum);
        }
    }
}

/// Interleaved dual-lane internal round for better ILP.
/// Processes two independent states simultaneously, interleaving operations to hide latency.
#[inline(always)]
pub unsafe fn internal_round_dual_asm<const WIDTH: usize>(
    state0: &mut [u64; WIDTH],
    state1: &mut [u64; WIDTH],
    diag: &[u64; WIDTH],
    rc: u64,
) {
    unsafe {
        // Add round constant to state[0] for both lanes
        let s0_a = add_asm(state0[0], rc);
        let s0_b = add_asm(state1[0], rc);

        // Start S-box - compute s0^2 for both lanes (independent muls)
        let s0_2_a = mul_asm(s0_a, s0_a);
        let s0_2_b = mul_asm(s0_b, s0_b);

        // While s0^2 is computing, sum state[1..WIDTH] for both lanes
        let mut sum_hi_a: u64 = 0;
        let mut sum_hi_b: u64 = 0;
        for i in 1..WIDTH {
            sum_hi_a = add_asm(sum_hi_a, state0[i]);
            sum_hi_b = add_asm(sum_hi_b, state1[i]);
        }

        // Continue S-box - compute s0^3 and s0^4 for both lanes (interleaved)
        let s0_3_a = mul_asm(s0_2_a, s0_a);
        let s0_3_b = mul_asm(s0_2_b, s0_b);
        let s0_4_a = mul_asm(s0_2_a, s0_2_a);
        let s0_4_b = mul_asm(s0_2_b, s0_2_b);

        // Start diagonal multiplies for both lanes (interleaved)
        let mut diag_muls_a: [u64; WIDTH] = [0; WIDTH];
        let mut diag_muls_b: [u64; WIDTH] = [0; WIDTH];
        for i in 1..WIDTH {
            diag_muls_a[i] = mul_asm(state0[i], diag[i]);
            diag_muls_b[i] = mul_asm(state1[i], diag[i]);
        }

        // Finish S-box - compute s0^7 for both lanes
        let s0_7_a = mul_asm(s0_3_a, s0_4_a);
        let s0_7_b = mul_asm(s0_3_b, s0_4_b);

        // Complete the sums
        let sum_a = add_asm(sum_hi_a, s0_7_a);
        let sum_b = add_asm(sum_hi_b, s0_7_b);

        // Compute state[0] = s0^7 * diag[0] + sum for both lanes
        let s0_diag_a = mul_asm(s0_7_a, diag[0]);
        let s0_diag_b = mul_asm(s0_7_b, diag[0]);
        state0[0] = add_asm(s0_diag_a, sum_a);
        state1[0] = add_asm(s0_diag_b, sum_b);

        // Finalize state[1..WIDTH] for both lanes
        for i in 1..WIDTH {
            state0[i] = add_asm(diag_muls_a[i], sum_a);
            state1[i] = add_asm(diag_muls_b[i], sum_b);
        }
    }
}

/// Fully unrolled dual-lane internal round for WIDTH=12.
/// Hand-unrolled for maximum performance on the most common width.
#[inline(always)]
pub unsafe fn internal_round_dual_asm_w12(
    state0: &mut [u64; 12],
    state1: &mut [u64; 12],
    diag: &[u64; 12],
    rc: u64,
) {
    unsafe {
        // Add round constant and start S-box
        let s0_a = add_asm(state0[0], rc);
        let s0_b = add_asm(state1[0], rc);

        let s0_2_a = mul_asm(s0_a, s0_a);
        let s0_2_b = mul_asm(s0_b, s0_b);

        // Unrolled sum computation - interleaved between lanes
        let sum1_a = add_asm(state0[1], state0[2]);
        let sum1_b = add_asm(state1[1], state1[2]);
        let sum2_a = add_asm(state0[3], state0[4]);
        let sum2_b = add_asm(state1[3], state1[4]);
        let sum3_a = add_asm(state0[5], state0[6]);
        let sum3_b = add_asm(state1[5], state1[6]);
        let sum4_a = add_asm(state0[7], state0[8]);
        let sum4_b = add_asm(state1[7], state1[8]);
        let sum5_a = add_asm(state0[9], state0[10]);
        let sum5_b = add_asm(state1[9], state1[10]);

        // Continue S-box
        let s0_3_a = mul_asm(s0_2_a, s0_a);
        let s0_3_b = mul_asm(s0_2_b, s0_b);
        let s0_4_a = mul_asm(s0_2_a, s0_2_a);
        let s0_4_b = mul_asm(s0_2_b, s0_2_b);

        // Combine partial sums
        let sum12_a = add_asm(sum1_a, sum2_a);
        let sum12_b = add_asm(sum1_b, sum2_b);
        let sum34_a = add_asm(sum3_a, sum4_a);
        let sum34_b = add_asm(sum3_b, sum4_b);
        let sum511_a = add_asm(sum5_a, state0[11]);
        let sum511_b = add_asm(sum5_b, state1[11]);

        // Diagonal multiplies - unrolled and interleaved
        let d1_a = mul_asm(state0[1], diag[1]);
        let d1_b = mul_asm(state1[1], diag[1]);
        let d2_a = mul_asm(state0[2], diag[2]);
        let d2_b = mul_asm(state1[2], diag[2]);
        let d3_a = mul_asm(state0[3], diag[3]);
        let d3_b = mul_asm(state1[3], diag[3]);
        let d4_a = mul_asm(state0[4], diag[4]);
        let d4_b = mul_asm(state1[4], diag[4]);
        let d5_a = mul_asm(state0[5], diag[5]);
        let d5_b = mul_asm(state1[5], diag[5]);
        let d6_a = mul_asm(state0[6], diag[6]);
        let d6_b = mul_asm(state1[6], diag[6]);

        // More partial sum combining
        let sum1234_a = add_asm(sum12_a, sum34_a);
        let sum1234_b = add_asm(sum12_b, sum34_b);
        let sum_hi_a = add_asm(sum1234_a, sum511_a);
        let sum_hi_b = add_asm(sum1234_b, sum511_b);

        // More diagonal multiplies
        let d7_a = mul_asm(state0[7], diag[7]);
        let d7_b = mul_asm(state1[7], diag[7]);
        let d8_a = mul_asm(state0[8], diag[8]);
        let d8_b = mul_asm(state1[8], diag[8]);
        let d9_a = mul_asm(state0[9], diag[9]);
        let d9_b = mul_asm(state1[9], diag[9]);
        let d10_a = mul_asm(state0[10], diag[10]);
        let d10_b = mul_asm(state1[10], diag[10]);
        let d11_a = mul_asm(state0[11], diag[11]);
        let d11_b = mul_asm(state1[11], diag[11]);

        // Finish S-box
        let s0_7_a = mul_asm(s0_3_a, s0_4_a);
        let s0_7_b = mul_asm(s0_3_b, s0_4_b);

        // Complete sum
        let sum_a = add_asm(sum_hi_a, s0_7_a);
        let sum_b = add_asm(sum_hi_b, s0_7_b);

        // Compute state[0]
        let s0_diag_a = mul_asm(s0_7_a, diag[0]);
        let s0_diag_b = mul_asm(s0_7_b, diag[0]);
        state0[0] = add_asm(s0_diag_a, sum_a);
        state1[0] = add_asm(s0_diag_b, sum_b);

        // Finalize all other states - unrolled
        state0[1] = add_asm(d1_a, sum_a);
        state1[1] = add_asm(d1_b, sum_b);
        state0[2] = add_asm(d2_a, sum_a);
        state1[2] = add_asm(d2_b, sum_b);
        state0[3] = add_asm(d3_a, sum_a);
        state1[3] = add_asm(d3_b, sum_b);
        state0[4] = add_asm(d4_a, sum_a);
        state1[4] = add_asm(d4_b, sum_b);
        state0[5] = add_asm(d5_a, sum_a);
        state1[5] = add_asm(d5_b, sum_b);
        state0[6] = add_asm(d6_a, sum_a);
        state1[6] = add_asm(d6_b, sum_b);
        state0[7] = add_asm(d7_a, sum_a);
        state1[7] = add_asm(d7_b, sum_b);
        state0[8] = add_asm(d8_a, sum_a);
        state1[8] = add_asm(d8_b, sum_b);
        state0[9] = add_asm(d9_a, sum_a);
        state1[9] = add_asm(d9_b, sum_b);
        state0[10] = add_asm(d10_a, sum_a);
        state1[10] = add_asm(d10_b, sum_b);
        state0[11] = add_asm(d11_a, sum_a);
        state1[11] = add_asm(d11_b, sum_b);
    }
}

/// Run all internal rounds with the optimized assembly implementation.
#[inline]
pub fn internal_permute_state_asm<const WIDTH: usize>(
    state: &mut [Goldilocks; WIDTH],
    diag: [Goldilocks; WIDTH],
    internal_constants: &[Goldilocks],
) {
    // Convert to raw u64 arrays for assembly processing
    let state_raw: &mut [u64; WIDTH] =
        unsafe { &mut *(state as *mut [Goldilocks; WIDTH] as *mut [u64; WIDTH]) };
    let diag_raw: [u64; WIDTH] = unsafe { core::mem::transmute_copy(&diag) };

    for &rc in internal_constants {
        unsafe {
            internal_round_asm(state_raw, &diag_raw, rc.value);
        }
    }
}

/// Fully unrolled internal round for WIDTH=12 (scalar version).
#[inline(always)]
pub unsafe fn internal_round_asm_w12(state: &mut [u64; 12], diag: &[u64; 12], rc: u64) {
    unsafe {
        // Add round constant and start S-box
        let s0 = add_asm(state[0], rc);
        let s0_2 = mul_asm(s0, s0);

        // Unrolled sum computation with tree reduction
        let sum1 = add_asm(state[1], state[2]);
        let sum2 = add_asm(state[3], state[4]);
        let sum3 = add_asm(state[5], state[6]);
        let sum4 = add_asm(state[7], state[8]);
        let sum5 = add_asm(state[9], state[10]);

        let s0_3 = mul_asm(s0_2, s0);
        let s0_4 = mul_asm(s0_2, s0_2);

        let sum12 = add_asm(sum1, sum2);
        let sum34 = add_asm(sum3, sum4);
        let sum511 = add_asm(sum5, state[11]);

        // Diagonal multiplies - fully unrolled
        let d1 = mul_asm(state[1], diag[1]);
        let d2 = mul_asm(state[2], diag[2]);
        let d3 = mul_asm(state[3], diag[3]);
        let d4 = mul_asm(state[4], diag[4]);
        let d5 = mul_asm(state[5], diag[5]);
        let d6 = mul_asm(state[6], diag[6]);

        let sum1234 = add_asm(sum12, sum34);
        let sum_hi = add_asm(sum1234, sum511);

        let d7 = mul_asm(state[7], diag[7]);
        let d8 = mul_asm(state[8], diag[8]);
        let d9 = mul_asm(state[9], diag[9]);
        let d10 = mul_asm(state[10], diag[10]);
        let d11 = mul_asm(state[11], diag[11]);

        // Finish S-box
        let s0_7 = mul_asm(s0_3, s0_4);
        let sum = add_asm(sum_hi, s0_7);

        // Compute state[0]
        let s0_diag = mul_asm(s0_7, diag[0]);
        state[0] = add_asm(s0_diag, sum);

        // Finalize all other states - unrolled
        state[1] = add_asm(d1, sum);
        state[2] = add_asm(d2, sum);
        state[3] = add_asm(d3, sum);
        state[4] = add_asm(d4, sum);
        state[5] = add_asm(d5, sum);
        state[6] = add_asm(d6, sum);
        state[7] = add_asm(d7, sum);
        state[8] = add_asm(d8, sum);
        state[9] = add_asm(d9, sum);
        state[10] = add_asm(d10, sum);
        state[11] = add_asm(d11, sum);
    }
}

/// Fully unrolled internal round for WIDTH=8 (scalar version).
#[inline(always)]
pub unsafe fn internal_round_asm_w8(state: &mut [u64; 8], diag: &[u64; 8], rc: u64) {
    unsafe {
        // Add round constant and start S-box
        let s0 = add_asm(state[0], rc);
        let s0_2 = mul_asm(s0, s0);

        // Unrolled sum computation with tree reduction
        let sum1 = add_asm(state[1], state[2]);
        let sum2 = add_asm(state[3], state[4]);
        let sum3 = add_asm(state[5], state[6]);

        let s0_3 = mul_asm(s0_2, s0);
        let s0_4 = mul_asm(s0_2, s0_2);

        let sum12 = add_asm(sum1, sum2);
        let sum37 = add_asm(sum3, state[7]);

        // Diagonal multiplies - fully unrolled
        let d1 = mul_asm(state[1], diag[1]);
        let d2 = mul_asm(state[2], diag[2]);
        let d3 = mul_asm(state[3], diag[3]);
        let d4 = mul_asm(state[4], diag[4]);

        let sum_hi = add_asm(sum12, sum37);

        let d5 = mul_asm(state[5], diag[5]);
        let d6 = mul_asm(state[6], diag[6]);
        let d7 = mul_asm(state[7], diag[7]);

        // Finish S-box
        let s0_7 = mul_asm(s0_3, s0_4);
        let sum = add_asm(sum_hi, s0_7);

        // Compute state[0]
        let s0_diag = mul_asm(s0_7, diag[0]);
        state[0] = add_asm(s0_diag, sum);

        // Finalize all other states - unrolled
        state[1] = add_asm(d1, sum);
        state[2] = add_asm(d2, sum);
        state[3] = add_asm(d3, sum);
        state[4] = add_asm(d4, sum);
        state[5] = add_asm(d5, sum);
        state[6] = add_asm(d6, sum);
        state[7] = add_asm(d7, sum);
    }
}

/// Fully unrolled dual-lane internal round for WIDTH=8.
#[inline(always)]
pub unsafe fn internal_round_dual_asm_w8(
    state0: &mut [u64; 8],
    state1: &mut [u64; 8],
    diag: &[u64; 8],
    rc: u64,
) {
    unsafe {
        // Add round constant and start S-box
        let s0_a = add_asm(state0[0], rc);
        let s0_b = add_asm(state1[0], rc);

        let s0_2_a = mul_asm(s0_a, s0_a);
        let s0_2_b = mul_asm(s0_b, s0_b);

        // Unrolled sum computation - interleaved between lanes
        let sum1_a = add_asm(state0[1], state0[2]);
        let sum1_b = add_asm(state1[1], state1[2]);
        let sum2_a = add_asm(state0[3], state0[4]);
        let sum2_b = add_asm(state1[3], state1[4]);
        let sum3_a = add_asm(state0[5], state0[6]);
        let sum3_b = add_asm(state1[5], state1[6]);

        // Continue S-box
        let s0_3_a = mul_asm(s0_2_a, s0_a);
        let s0_3_b = mul_asm(s0_2_b, s0_b);
        let s0_4_a = mul_asm(s0_2_a, s0_2_a);
        let s0_4_b = mul_asm(s0_2_b, s0_2_b);

        // Combine partial sums
        let sum12_a = add_asm(sum1_a, sum2_a);
        let sum12_b = add_asm(sum1_b, sum2_b);
        let sum37_a = add_asm(sum3_a, state0[7]);
        let sum37_b = add_asm(sum3_b, state1[7]);

        // Diagonal multiplies - unrolled and interleaved
        let d1_a = mul_asm(state0[1], diag[1]);
        let d1_b = mul_asm(state1[1], diag[1]);
        let d2_a = mul_asm(state0[2], diag[2]);
        let d2_b = mul_asm(state1[2], diag[2]);
        let d3_a = mul_asm(state0[3], diag[3]);
        let d3_b = mul_asm(state1[3], diag[3]);
        let d4_a = mul_asm(state0[4], diag[4]);
        let d4_b = mul_asm(state1[4], diag[4]);

        let sum_hi_a = add_asm(sum12_a, sum37_a);
        let sum_hi_b = add_asm(sum12_b, sum37_b);

        let d5_a = mul_asm(state0[5], diag[5]);
        let d5_b = mul_asm(state1[5], diag[5]);
        let d6_a = mul_asm(state0[6], diag[6]);
        let d6_b = mul_asm(state1[6], diag[6]);
        let d7_a = mul_asm(state0[7], diag[7]);
        let d7_b = mul_asm(state1[7], diag[7]);

        // Finish S-box
        let s0_7_a = mul_asm(s0_3_a, s0_4_a);
        let s0_7_b = mul_asm(s0_3_b, s0_4_b);

        // Complete sum
        let sum_a = add_asm(sum_hi_a, s0_7_a);
        let sum_b = add_asm(sum_hi_b, s0_7_b);

        // Compute state[0]
        let s0_diag_a = mul_asm(s0_7_a, diag[0]);
        let s0_diag_b = mul_asm(s0_7_b, diag[0]);
        state0[0] = add_asm(s0_diag_a, sum_a);
        state1[0] = add_asm(s0_diag_b, sum_b);

        // Finalize all other states - unrolled
        state0[1] = add_asm(d1_a, sum_a);
        state1[1] = add_asm(d1_b, sum_b);
        state0[2] = add_asm(d2_a, sum_a);
        state1[2] = add_asm(d2_b, sum_b);
        state0[3] = add_asm(d3_a, sum_a);
        state1[3] = add_asm(d3_b, sum_b);
        state0[4] = add_asm(d4_a, sum_a);
        state1[4] = add_asm(d4_b, sum_b);
        state0[5] = add_asm(d5_a, sum_a);
        state1[5] = add_asm(d5_b, sum_b);
        state0[6] = add_asm(d6_a, sum_a);
        state1[6] = add_asm(d6_b, sum_b);
        state0[7] = add_asm(d7_a, sum_a);
        state1[7] = add_asm(d7_b, sum_b);
    }
}

/// Specialized W8 internal permute using fully unrolled rounds.
#[inline]
pub fn internal_permute_state_asm_w8(
    state: &mut [Goldilocks; 8],
    diag: [Goldilocks; 8],
    internal_constants: &[Goldilocks],
) {
    let state_raw: &mut [u64; 8] =
        unsafe { &mut *(state as *mut [Goldilocks; 8] as *mut [u64; 8]) };
    let diag_raw: [u64; 8] = unsafe { core::mem::transmute_copy(&diag) };

    for &rc in internal_constants {
        unsafe {
            internal_round_asm_w8(state_raw, &diag_raw, rc.value);
        }
    }
}

/// Specialized W12 internal permute using fully unrolled rounds.
#[inline]
pub fn internal_permute_state_asm_w12(
    state: &mut [Goldilocks; 12],
    diag: [Goldilocks; 12],
    internal_constants: &[Goldilocks],
) {
    let state_raw: &mut [u64; 12] =
        unsafe { &mut *(state as *mut [Goldilocks; 12] as *mut [u64; 12]) };
    let diag_raw: [u64; 12] = unsafe { core::mem::transmute_copy(&diag) };

    for &rc in internal_constants {
        unsafe {
            internal_round_asm_w12(state_raw, &diag_raw, rc.value);
        }
    }
}

/// Fully unrolled internal round for WIDTH=16 (scalar version).
#[inline(always)]
pub unsafe fn internal_round_asm_w16(state: &mut [u64; 16], diag: &[u64; 16], rc: u64) {
    unsafe {
        // Add round constant and start S-box
        let s0 = add_asm(state[0], rc);
        let s0_2 = mul_asm(s0, s0);

        // Unrolled sum computation with tree reduction
        let sum1 = add_asm(state[1], state[2]);
        let sum2 = add_asm(state[3], state[4]);
        let sum3 = add_asm(state[5], state[6]);
        let sum4 = add_asm(state[7], state[8]);
        let sum5 = add_asm(state[9], state[10]);
        let sum6 = add_asm(state[11], state[12]);
        let sum7 = add_asm(state[13], state[14]);

        let s0_3 = mul_asm(s0_2, s0);
        let s0_4 = mul_asm(s0_2, s0_2);

        let sum12 = add_asm(sum1, sum2);
        let sum34 = add_asm(sum3, sum4);
        let sum56 = add_asm(sum5, sum6);
        let sum715 = add_asm(sum7, state[15]);

        // Diagonal multiplies - fully unrolled (first batch)
        let d1 = mul_asm(state[1], diag[1]);
        let d2 = mul_asm(state[2], diag[2]);
        let d3 = mul_asm(state[3], diag[3]);
        let d4 = mul_asm(state[4], diag[4]);
        let d5 = mul_asm(state[5], diag[5]);
        let d6 = mul_asm(state[6], diag[6]);
        let d7 = mul_asm(state[7], diag[7]);
        let d8 = mul_asm(state[8], diag[8]);

        let sum1234 = add_asm(sum12, sum34);
        let sum56715 = add_asm(sum56, sum715);
        let sum_hi = add_asm(sum1234, sum56715);

        // Diagonal multiplies - second batch
        let d9 = mul_asm(state[9], diag[9]);
        let d10 = mul_asm(state[10], diag[10]);
        let d11 = mul_asm(state[11], diag[11]);
        let d12 = mul_asm(state[12], diag[12]);
        let d13 = mul_asm(state[13], diag[13]);
        let d14 = mul_asm(state[14], diag[14]);
        let d15 = mul_asm(state[15], diag[15]);

        // Finish S-box
        let s0_7 = mul_asm(s0_3, s0_4);
        let sum = add_asm(sum_hi, s0_7);

        // Compute state[0]
        let s0_diag = mul_asm(s0_7, diag[0]);
        state[0] = add_asm(s0_diag, sum);

        // Finalize all other states - unrolled
        state[1] = add_asm(d1, sum);
        state[2] = add_asm(d2, sum);
        state[3] = add_asm(d3, sum);
        state[4] = add_asm(d4, sum);
        state[5] = add_asm(d5, sum);
        state[6] = add_asm(d6, sum);
        state[7] = add_asm(d7, sum);
        state[8] = add_asm(d8, sum);
        state[9] = add_asm(d9, sum);
        state[10] = add_asm(d10, sum);
        state[11] = add_asm(d11, sum);
        state[12] = add_asm(d12, sum);
        state[13] = add_asm(d13, sum);
        state[14] = add_asm(d14, sum);
        state[15] = add_asm(d15, sum);
    }
}

/// Fully unrolled dual-lane internal round for WIDTH=16.
#[inline(always)]
pub unsafe fn internal_round_dual_asm_w16(
    state0: &mut [u64; 16],
    state1: &mut [u64; 16],
    diag: &[u64; 16],
    rc: u64,
) {
    unsafe {
        // Add round constant and start S-box
        let s0_a = add_asm(state0[0], rc);
        let s0_b = add_asm(state1[0], rc);

        let s0_2_a = mul_asm(s0_a, s0_a);
        let s0_2_b = mul_asm(s0_b, s0_b);

        // Unrolled sum computation - interleaved between lanes
        let sum1_a = add_asm(state0[1], state0[2]);
        let sum1_b = add_asm(state1[1], state1[2]);
        let sum2_a = add_asm(state0[3], state0[4]);
        let sum2_b = add_asm(state1[3], state1[4]);
        let sum3_a = add_asm(state0[5], state0[6]);
        let sum3_b = add_asm(state1[5], state1[6]);
        let sum4_a = add_asm(state0[7], state0[8]);
        let sum4_b = add_asm(state1[7], state1[8]);
        let sum5_a = add_asm(state0[9], state0[10]);
        let sum5_b = add_asm(state1[9], state1[10]);
        let sum6_a = add_asm(state0[11], state0[12]);
        let sum6_b = add_asm(state1[11], state1[12]);
        let sum7_a = add_asm(state0[13], state0[14]);
        let sum7_b = add_asm(state1[13], state1[14]);

        // Continue S-box
        let s0_3_a = mul_asm(s0_2_a, s0_a);
        let s0_3_b = mul_asm(s0_2_b, s0_b);
        let s0_4_a = mul_asm(s0_2_a, s0_2_a);
        let s0_4_b = mul_asm(s0_2_b, s0_2_b);

        // Combine partial sums
        let sum12_a = add_asm(sum1_a, sum2_a);
        let sum12_b = add_asm(sum1_b, sum2_b);
        let sum34_a = add_asm(sum3_a, sum4_a);
        let sum34_b = add_asm(sum3_b, sum4_b);
        let sum56_a = add_asm(sum5_a, sum6_a);
        let sum56_b = add_asm(sum5_b, sum6_b);
        let sum715_a = add_asm(sum7_a, state0[15]);
        let sum715_b = add_asm(sum7_b, state1[15]);

        // Diagonal multiplies - first batch, interleaved
        let d1_a = mul_asm(state0[1], diag[1]);
        let d1_b = mul_asm(state1[1], diag[1]);
        let d2_a = mul_asm(state0[2], diag[2]);
        let d2_b = mul_asm(state1[2], diag[2]);
        let d3_a = mul_asm(state0[3], diag[3]);
        let d3_b = mul_asm(state1[3], diag[3]);
        let d4_a = mul_asm(state0[4], diag[4]);
        let d4_b = mul_asm(state1[4], diag[4]);
        let d5_a = mul_asm(state0[5], diag[5]);
        let d5_b = mul_asm(state1[5], diag[5]);
        let d6_a = mul_asm(state0[6], diag[6]);
        let d6_b = mul_asm(state1[6], diag[6]);
        let d7_a = mul_asm(state0[7], diag[7]);
        let d7_b = mul_asm(state1[7], diag[7]);
        let d8_a = mul_asm(state0[8], diag[8]);
        let d8_b = mul_asm(state1[8], diag[8]);

        // More partial sum combining
        let sum1234_a = add_asm(sum12_a, sum34_a);
        let sum1234_b = add_asm(sum12_b, sum34_b);
        let sum56715_a = add_asm(sum56_a, sum715_a);
        let sum56715_b = add_asm(sum56_b, sum715_b);
        let sum_hi_a = add_asm(sum1234_a, sum56715_a);
        let sum_hi_b = add_asm(sum1234_b, sum56715_b);

        // Diagonal multiplies - second batch
        let d9_a = mul_asm(state0[9], diag[9]);
        let d9_b = mul_asm(state1[9], diag[9]);
        let d10_a = mul_asm(state0[10], diag[10]);
        let d10_b = mul_asm(state1[10], diag[10]);
        let d11_a = mul_asm(state0[11], diag[11]);
        let d11_b = mul_asm(state1[11], diag[11]);
        let d12_a = mul_asm(state0[12], diag[12]);
        let d12_b = mul_asm(state1[12], diag[12]);
        let d13_a = mul_asm(state0[13], diag[13]);
        let d13_b = mul_asm(state1[13], diag[13]);
        let d14_a = mul_asm(state0[14], diag[14]);
        let d14_b = mul_asm(state1[14], diag[14]);
        let d15_a = mul_asm(state0[15], diag[15]);
        let d15_b = mul_asm(state1[15], diag[15]);

        // Finish S-box
        let s0_7_a = mul_asm(s0_3_a, s0_4_a);
        let s0_7_b = mul_asm(s0_3_b, s0_4_b);

        // Complete sum
        let sum_a = add_asm(sum_hi_a, s0_7_a);
        let sum_b = add_asm(sum_hi_b, s0_7_b);

        // Compute state[0]
        let s0_diag_a = mul_asm(s0_7_a, diag[0]);
        let s0_diag_b = mul_asm(s0_7_b, diag[0]);
        state0[0] = add_asm(s0_diag_a, sum_a);
        state1[0] = add_asm(s0_diag_b, sum_b);

        // Finalize all other states - unrolled
        state0[1] = add_asm(d1_a, sum_a);
        state1[1] = add_asm(d1_b, sum_b);
        state0[2] = add_asm(d2_a, sum_a);
        state1[2] = add_asm(d2_b, sum_b);
        state0[3] = add_asm(d3_a, sum_a);
        state1[3] = add_asm(d3_b, sum_b);
        state0[4] = add_asm(d4_a, sum_a);
        state1[4] = add_asm(d4_b, sum_b);
        state0[5] = add_asm(d5_a, sum_a);
        state1[5] = add_asm(d5_b, sum_b);
        state0[6] = add_asm(d6_a, sum_a);
        state1[6] = add_asm(d6_b, sum_b);
        state0[7] = add_asm(d7_a, sum_a);
        state1[7] = add_asm(d7_b, sum_b);
        state0[8] = add_asm(d8_a, sum_a);
        state1[8] = add_asm(d8_b, sum_b);
        state0[9] = add_asm(d9_a, sum_a);
        state1[9] = add_asm(d9_b, sum_b);
        state0[10] = add_asm(d10_a, sum_a);
        state1[10] = add_asm(d10_b, sum_b);
        state0[11] = add_asm(d11_a, sum_a);
        state1[11] = add_asm(d11_b, sum_b);
        state0[12] = add_asm(d12_a, sum_a);
        state1[12] = add_asm(d12_b, sum_b);
        state0[13] = add_asm(d13_a, sum_a);
        state1[13] = add_asm(d13_b, sum_b);
        state0[14] = add_asm(d14_a, sum_a);
        state1[14] = add_asm(d14_b, sum_b);
        state0[15] = add_asm(d15_a, sum_a);
        state1[15] = add_asm(d15_b, sum_b);
    }
}

/// Specialized W16 internal permute using fully unrolled rounds.
#[inline]
pub fn internal_permute_state_asm_w16(
    state: &mut [Goldilocks; 16],
    diag: [Goldilocks; 16],
    internal_constants: &[Goldilocks],
) {
    let state_raw: &mut [u64; 16] =
        unsafe { &mut *(state as *mut [Goldilocks; 16] as *mut [u64; 16]) };
    let diag_raw: [u64; 16] = unsafe { core::mem::transmute_copy(&diag) };

    for &rc in internal_constants {
        unsafe {
            internal_round_asm_w16(state_raw, &diag_raw, rc.value);
        }
    }
}

// External layer: S-box on all elements, then MDS. Pipelined for latency hiding.

/// Double a Goldilocks element.
#[inline(always)]
unsafe fn double_asm(a: u64) -> u64 {
    // SAFETY: add_asm is safe with valid Goldilocks field elements
    unsafe { add_asm(a, a) }
}

/// 4x4 circulant MDS with coefficients [2,3,1,1].
#[inline(always)]
unsafe fn apply_mat4_asm(x: &mut [u64; 4]) {
    unsafe {
        let t01 = add_asm(x[0], x[1]);
        let t23 = add_asm(x[2], x[3]);
        let t0123 = add_asm(t01, t23);
        let t01123 = add_asm(t0123, x[1]);
        let t01233 = add_asm(t0123, x[3]);

        let y3 = add_asm(t01233, double_asm(x[0]));
        let y1 = add_asm(t01123, double_asm(x[2]));
        let y0 = add_asm(t01123, t01);
        let y2 = add_asm(t01233, t23);

        x[0] = y0;
        x[1] = y1;
        x[2] = y2;
        x[3] = y3;
    }
}

/// Poseidon2 MDS light permutation: 4x4 blocks + outer sums.
#[inline(always)]
pub unsafe fn mds_light_permutation_asm<const WIDTH: usize>(state: &mut [u64; WIDTH]) {
    unsafe {
        // Apply M_4 to each consecutive four elements
        let mut i = 0;
        while i < WIDTH {
            let chunk: &mut [u64; 4] = (&mut state[i..i + 4]).try_into().unwrap();
            apply_mat4_asm(chunk);
            i += 4;
        }

        // Compute the four sums of every 4th element
        let mut sums = [0u64; 4];
        for j in (0..WIDTH).step_by(4) {
            sums[0] = add_asm(sums[0], state[j]);
            sums[1] = add_asm(sums[1], state[j + 1]);
            sums[2] = add_asm(sums[2], state[j + 2]);
            sums[3] = add_asm(sums[3], state[j + 3]);
        }

        // Add sums back to state
        for (i, elem) in state.iter_mut().enumerate() {
            *elem = add_asm(*elem, sums[i % 4]);
        }
    }
}

/// Pipelined S-box computation for all elements.
/// Computes x^7 for all elements by interleaving stages to hide latency.
#[inline(always)]
pub unsafe fn sbox_layer_asm<const WIDTH: usize>(state: &mut [u64; WIDTH]) {
    unsafe {
        // Stage 1: Compute all x^2 values
        let mut x2 = [0u64; WIDTH];
        for i in 0..WIDTH {
            x2[i] = mul_asm(state[i], state[i]);
        }

        // Stage 2: Compute x^3 and x^4 values interleaved
        // x^3 = x^2 * x, x^4 = x^2 * x^2
        let mut x3 = [0u64; WIDTH];
        let mut x4 = [0u64; WIDTH];
        for i in 0..WIDTH {
            x3[i] = mul_asm(x2[i], state[i]);
            x4[i] = mul_asm(x2[i], x2[i]);
        }

        // Stage 3: Compute x^7 = x^3 * x^4
        for i in 0..WIDTH {
            state[i] = mul_asm(x3[i], x4[i]);
        }
    }
}

/// Optimized external round: add RC, S-box, MDS.
#[inline(always)]
pub unsafe fn external_round_asm<const WIDTH: usize>(state: &mut [u64; WIDTH], rc: &[u64; WIDTH]) {
    unsafe {
        // Add round constants
        for i in 0..WIDTH {
            state[i] = add_asm(state[i], rc[i]);
        }

        // Apply S-box (x^7) to all elements
        sbox_layer_asm(state);

        // Apply MDS light permutation
        mds_light_permutation_asm(state);
    }
}

/// Interleaved dual-lane S-box layer for better ILP.
#[inline(always)]
pub unsafe fn sbox_layer_dual_asm<const WIDTH: usize>(
    state0: &mut [u64; WIDTH],
    state1: &mut [u64; WIDTH],
) {
    unsafe {
        // Stage 1: Compute all x^2 values for both lanes (interleaved)
        let mut x2_a = [0u64; WIDTH];
        let mut x2_b = [0u64; WIDTH];
        for i in 0..WIDTH {
            x2_a[i] = mul_asm(state0[i], state0[i]);
            x2_b[i] = mul_asm(state1[i], state1[i]);
        }

        // Stage 2: Compute x^3 and x^4 for both lanes (interleaved)
        let mut x3_a = [0u64; WIDTH];
        let mut x3_b = [0u64; WIDTH];
        let mut x4_a = [0u64; WIDTH];
        let mut x4_b = [0u64; WIDTH];
        for i in 0..WIDTH {
            x3_a[i] = mul_asm(x2_a[i], state0[i]);
            x3_b[i] = mul_asm(x2_b[i], state1[i]);
            x4_a[i] = mul_asm(x2_a[i], x2_a[i]);
            x4_b[i] = mul_asm(x2_b[i], x2_b[i]);
        }

        // Stage 3: Compute x^7 = x^3 * x^4 for both lanes
        for i in 0..WIDTH {
            state0[i] = mul_asm(x3_a[i], x4_a[i]);
            state1[i] = mul_asm(x3_b[i], x4_b[i]);
        }
    }
}

/// Interleaved dual-lane external round for better ILP.
#[inline(always)]
pub unsafe fn external_round_dual_asm<const WIDTH: usize>(
    state0: &mut [u64; WIDTH],
    state1: &mut [u64; WIDTH],
    rc: &[u64; WIDTH],
) {
    unsafe {
        // Add round constants (interleaved)
        for i in 0..WIDTH {
            state0[i] = add_asm(state0[i], rc[i]);
            state1[i] = add_asm(state1[i], rc[i]);
        }

        // Apply S-box (interleaved dual-lane)
        sbox_layer_dual_asm(state0, state1);

        // Apply MDS (sequential - MDS is mostly additions which are fast)
        mds_light_permutation_asm(state0);
        mds_light_permutation_asm(state1);
    }
}

/// Run initial external rounds with ASM optimization.
#[inline]
pub fn external_initial_permute_state_asm<const WIDTH: usize>(
    state: &mut [Goldilocks; WIDTH],
    initial_constants: &[[Goldilocks; WIDTH]],
) {
    let state_raw: &mut [u64; WIDTH] =
        unsafe { &mut *(state as *mut [Goldilocks; WIDTH] as *mut [u64; WIDTH]) };

    // Initial MDS before rounds
    unsafe {
        mds_light_permutation_asm(state_raw);
    }

    // Run initial rounds
    for rc in initial_constants {
        let rc_raw: [u64; WIDTH] = unsafe { core::mem::transmute_copy(rc) };
        unsafe {
            external_round_asm(state_raw, &rc_raw);
        }
    }
}

/// Run terminal external rounds with ASM optimization.
#[inline]
pub fn external_terminal_permute_state_asm<const WIDTH: usize>(
    state: &mut [Goldilocks; WIDTH],
    terminal_constants: &[[Goldilocks; WIDTH]],
) {
    let state_raw: &mut [u64; WIDTH] =
        unsafe { &mut *(state as *mut [Goldilocks; WIDTH] as *mut [u64; WIDTH]) };

    for rc in terminal_constants {
        let rc_raw: [u64; WIDTH] = unsafe { core::mem::transmute_copy(rc) };
        unsafe {
            external_round_asm(state_raw, &rc_raw);
        }
    }
}

#[cfg(test)]
mod tests {
    use p3_field::PrimeField64;
    use p3_poseidon2::matmul_internal;

    use super::*;
    use crate::{
        MATRIX_DIAG_8_GOLDILOCKS, MATRIX_DIAG_12_GOLDILOCKS, MATRIX_DIAG_16_GOLDILOCKS,
        MATRIX_DIAG_20_GOLDILOCKS,
    };

    fn test_internal_round_matches<const WIDTH: usize>(diag: [Goldilocks; WIDTH]) {
        let mut rng_state = 12345u64;
        let mut next_rand = || {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            rng_state
        };

        // Generate random state
        let mut state_asm: [Goldilocks; WIDTH] =
            core::array::from_fn(|_| Goldilocks::new(next_rand()));
        let mut state_generic = state_asm;

        // Generate random internal constants
        let internal_constants: [Goldilocks; 22] =
            core::array::from_fn(|_| Goldilocks::new(next_rand()));

        // Run ASM version
        internal_permute_state_asm(&mut state_asm, diag, &internal_constants);

        // Run generic version - manually implement the internal permute
        for &rc in internal_constants.iter() {
            // Add round constant and apply S-box to state[0]
            state_generic[0] += rc;
            let s = state_generic[0];
            let s2 = s * s;
            let s3 = s2 * s;
            let s4 = s2 * s2;
            state_generic[0] = s3 * s4; // s^7

            // Apply internal MDS: sum + diagonal multiply
            matmul_internal(&mut state_generic, diag);
        }

        // Compare results
        for i in 0..WIDTH {
            assert_eq!(
                state_asm[i].as_canonical_u64(),
                state_generic[i].as_canonical_u64(),
                "Mismatch at index {i}"
            );
        }
    }

    #[test]
    fn test_internal_round_width_8() {
        test_internal_round_matches(MATRIX_DIAG_8_GOLDILOCKS);
    }

    #[test]
    fn test_internal_round_width_12() {
        test_internal_round_matches(MATRIX_DIAG_12_GOLDILOCKS);
    }

    #[test]
    fn test_internal_round_width_16() {
        test_internal_round_matches(MATRIX_DIAG_16_GOLDILOCKS);
    }

    #[test]
    fn test_internal_round_width_20() {
        test_internal_round_matches(MATRIX_DIAG_20_GOLDILOCKS);
    }

    #[test]
    fn test_add_asm() {
        // Test addition against the standard implementation
        let test_cases: [(u64, u64); 8] = [
            (0, 0),
            (1, 1),
            (P - 1, 1),         // Should wrap to 0
            (P - 1, P - 1),     // Should wrap
            (P / 2, P / 2),     // No wrap
            (P / 2, P / 2 + 1), // Just wraps
            (0xFFFFFFFF00000000, 0xFFFFFFFF00000000),
            (0x123456789ABCDEF0, 0xFEDCBA9876543210),
        ];

        for (a, b) in test_cases {
            let expected = (Goldilocks::new(a) + Goldilocks::new(b)).as_canonical_u64();
            let result = unsafe { add_asm(a, b) };
            let result_canonical = Goldilocks::new(result).as_canonical_u64();
            assert_eq!(
                result_canonical, expected,
                "add_asm({a:#x}, {b:#x}) = {result:#x}, expected {expected:#x}"
            );
        }
    }

    #[test]
    fn test_mul_asm() {
        // Test multiplication against the standard implementation
        let test_cases: [(u64, u64); 8] = [
            (0, 0),
            (1, 1),
            (P - 1, P - 1),
            (P - 1, 2), // Near-max times small
            (2, P - 1), // Commutative check
            (0x123456789ABCDEF0, 0xFEDCBA9876543210),
            (0xFFFFFFFF00000000, 0xFFFFFFFF00000000),
            (EPSILON, EPSILON), // Edge case with epsilon
        ];

        for (a, b) in test_cases {
            let expected = (Goldilocks::new(a) * Goldilocks::new(b)).as_canonical_u64();
            let result = unsafe { mul_asm(a, b) };
            let result_canonical = Goldilocks::new(result).as_canonical_u64();
            assert_eq!(
                result_canonical, expected,
                "mul_asm({a:#x}, {b:#x}) = {result:#x}, expected {expected:#x}"
            );
        }
    }

    #[test]
    fn test_mul_asm_random() {
        // Test with many random values
        let mut rng_state = 0xDEADBEEFCAFEBABEu64;
        let mut next_rand = || {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            rng_state
        };

        for _ in 0..1000 {
            let a = next_rand();
            let b = next_rand();
            let expected = (Goldilocks::new(a) * Goldilocks::new(b)).as_canonical_u64();
            let result = unsafe { mul_asm(a, b) };
            let result_canonical = Goldilocks::new(result).as_canonical_u64();
            assert_eq!(
                result_canonical, expected,
                "mul_asm({a:#x}, {b:#x}) = {result:#x}, expected {expected:#x}"
            );
        }
    }

    #[test]
    fn test_sbox_layer() {
        let mut rng_state = 98765u64;
        let mut next_rand = || {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            rng_state
        };

        // Test S-box layer against generic implementation
        let mut state_asm: [Goldilocks; 8] = core::array::from_fn(|_| Goldilocks::new(next_rand()));
        let state_generic = state_asm;

        // Convert to raw and apply ASM S-box
        let state_raw: &mut [u64; 8] =
            unsafe { &mut *(&mut state_asm as *mut [Goldilocks; 8] as *mut [u64; 8]) };
        unsafe {
            sbox_layer_asm(state_raw);
        }

        // Compute expected results
        for i in 0..8 {
            let x = state_generic[i];
            let x2 = x * x;
            let x3 = x2 * x;
            let x4 = x2 * x2;
            let x7 = x3 * x4;
            assert_eq!(
                state_asm[i].as_canonical_u64(),
                x7.as_canonical_u64(),
                "S-box mismatch at index {i}"
            );
        }
    }

    #[test]
    fn test_mds_light_permutation() {
        use p3_poseidon2::{MDSMat4, mds_light_permutation};

        let mut rng_state = 54321u64;
        let mut next_rand = || {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            rng_state
        };

        // Test MDS against generic implementation
        let mut state_asm: [Goldilocks; 12] =
            core::array::from_fn(|_| Goldilocks::new(next_rand()));
        let mut state_generic = state_asm;

        // Apply ASM MDS
        let state_raw: &mut [u64; 12] =
            unsafe { &mut *(&mut state_asm as *mut [Goldilocks; 12] as *mut [u64; 12]) };
        unsafe {
            mds_light_permutation_asm(state_raw);
        }

        // Apply generic MDS
        mds_light_permutation(&mut state_generic, &MDSMat4);

        // Compare results
        for i in 0..12 {
            assert_eq!(
                state_asm[i].as_canonical_u64(),
                state_generic[i].as_canonical_u64(),
                "MDS mismatch at index {i}"
            );
        }
    }

    // ===================== Boundary Value Tests =====================
    // Test edge cases that stress reduction logic and overflow handling

    const P_MINUS_1: u64 = P - 1;
    const TWO_POW_32: u64 = 0x1_0000_0000;

    /// Boundary values for arithmetic tests
    const BOUNDARY_VALS: [u64; 8] = [
        0,
        1,
        P_MINUS_1,
        EPSILON,
        TWO_POW_32,
        TWO_POW_32 - 1,
        0x8000_0000_0000_0000,
        0x123456789ABCDEF0,
    ];

    fn check_binary_op<F>(op_name: &str, asm_op: F)
    where
        F: Fn(u64, u64) -> u64,
    {
        for &a in &BOUNDARY_VALS {
            for &b in &BOUNDARY_VALS {
                let expected = match op_name {
                    "add" => (Goldilocks::new(a) + Goldilocks::new(b)).as_canonical_u64(),
                    "mul" => (Goldilocks::new(a) * Goldilocks::new(b)).as_canonical_u64(),
                    _ => panic!("unknown op"),
                };
                let result = Goldilocks::new(asm_op(a, b)).as_canonical_u64();
                assert_eq!(result, expected, "{op_name}({a:#x}, {b:#x}) mismatch");
            }
        }
    }

    #[test]
    fn test_arithmetic_boundary_values() {
        // Test add_asm and mul_asm with all boundary value combinations
        check_binary_op("add", |a, b| unsafe { add_asm(a, b) });
        check_binary_op("mul", |a, b| unsafe { mul_asm(a, b) });

        // Verify key identities
        assert_eq!(
            Goldilocks::new(unsafe { mul_asm(P_MINUS_1, P_MINUS_1) }).as_canonical_u64(),
            1,
            "(-1)^2 should equal 1"
        );
    }

    #[test]
    fn test_sbox_mds_boundary_values() {
        use p3_poseidon2::{MDSMat4, mds_light_permutation};

        // Test S-box with boundary values
        for &val in &BOUNDARY_VALS {
            let g = Goldilocks::new(val);
            let expected = {
                let x2 = g * g;
                let x4 = x2 * x2;
                x2 * x4 * g
            };

            let mut state: [u64; 8] = [val, 1, 1, 1, 1, 1, 1, 1];
            unsafe {
                sbox_layer_asm(&mut state);
            }

            assert_eq!(
                Goldilocks::new(state[0]).as_canonical_u64(),
                expected.as_canonical_u64(),
                "sbox({val:#x}) mismatch"
            );
        }

        // Test MDS with mixed boundary patterns
        let mds_vals: [u64; 4] = [0, 1, EPSILON, TWO_POW_32];
        for &v0 in &mds_vals {
            for &v1 in &mds_vals {
                let mut state_generic: [Goldilocks; 8] =
                    core::array::from_fn(|i| Goldilocks::new(if i % 2 == 0 { v0 } else { v1 }));
                let mut state_raw: [u64; 8] =
                    core::array::from_fn(|i| state_generic[i].as_canonical_u64());

                unsafe {
                    mds_light_permutation_asm(&mut state_raw);
                }
                mds_light_permutation(&mut state_generic, &MDSMat4);

                for i in 0..8 {
                    assert_eq!(
                        Goldilocks::new(state_raw[i]).as_canonical_u64(),
                        state_generic[i].as_canonical_u64(),
                        "MDS v0={v0:#x} v1={v1:#x} mismatch at {i}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_internal_round_correctness() {
        // Verify internal round formula: s0=state[0]+rc, s0_7=s0^7,
        // sum=s0_7+sum(state[1..]), state[i]=state[i]*diag[i]+sum
        let diag: [u64; 8] = [
            0xc3b6c08e23ba9300,
            0xd84b5de94a324fb6,
            0x0d0c371c5b35b84f,
            0x7964f570a8f648d,
            0x5daf18bbd996604b,
            0x6743bc47b9595257,
            0x5528b9362c59bb70,
            0xac45e25b7127b68b,
        ];
        let rc = 0x123456789ABCDEF0u64;

        let mut state: [u64; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
        let orig = state;

        // Compute reference
        let s0 = Goldilocks::new(orig[0]) + Goldilocks::new(rc);
        let s0_7 = {
            let x2 = s0 * s0;
            let x4 = x2 * x2;
            x2 * x4 * s0
        };
        let sum_hi: Goldilocks = orig[1..].iter().map(|&x| Goldilocks::new(x)).sum();
        let sum = sum_hi + s0_7;

        let mut expected: [Goldilocks; 8] = core::array::from_fn(|i| Goldilocks::new(orig[i]));
        expected[0] = s0_7 * Goldilocks::new(diag[0]) + sum;
        for i in 1..8 {
            expected[i] = Goldilocks::new(orig[i]) * Goldilocks::new(diag[i]) + sum;
        }

        unsafe {
            internal_round_asm_w8(&mut state, &diag, rc);
        }

        for i in 0..8 {
            assert_eq!(
                Goldilocks::new(state[i]).as_canonical_u64(),
                expected[i].as_canonical_u64(),
                "internal_round mismatch at {i}"
            );
        }
    }
}
