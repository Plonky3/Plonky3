//! ARM assembly primitives for Poseidon2 on Goldilocks.
//!
//! Latency hiding: ARM mul/umulh have ~4-5 cycle latency. By interleaving
//! S-box computation with MDS operations, we hide much of this latency.

use core::arch::aarch64::*;
use core::arch::asm;

use crate::P;

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

/// Compute a * b + c in the Goldilocks field using inline assembly.
/// All inputs must be valid field elements; the final result is reduced modulo P.
#[inline(always)]
unsafe fn mul_add_asm(a: u64, b: u64, c: u64) -> u64 {
    let _lo: u64;
    let _hi: u64;
    let _t0: u64;
    let _t1: u64;
    let _t2: u64;
    let result: u64;

    unsafe {
        asm!(
            // Compute 128-bit product: hi:lo = a * b
            "mul   {lo}, {a}, {b}",
            "umulh {hi}, {a}, {b}",

            // Accumulate c into the 128-bit product: hi:lo = hi:lo + c
            "adds  {lo}, {lo}, {c}",
            "adc   {hi}, {hi}, xzr",

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
            c = in(reg) c,
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

/// Compute x / 2 in the Goldilocks field, matching `halve_u64::<P>`.
#[inline(always)]
unsafe fn div2_asm(x: u64) -> u64 {
    let shift = (P + 1) >> 1;
    let result: u64;
    let _tmp: u64;

    unsafe {
        asm!(
            // result = x >> 1
            "lsr   {result}, {x}, #1",
            // tmp = x & 1
            "and   {tmp}, {x}, #1",
            // if tmp != 0 (x odd), tmp := shift, else tmp := 0
            "cmp   {tmp}, #0",
            "csel  {tmp}, {shift}, xzr, ne",
            // result += tmp
            "add   {result}, {result}, {tmp}",
            x      = in(reg) x,
            shift  = in(reg) shift,
            tmp    = lateout(reg) _tmp,
            result = out(reg) result,
            options(pure, nomem, nostack),
        );
    }

    result
}

#[inline(always)]
unsafe fn div4_asm(x: u64) -> u64 {
    unsafe { div2_asm(div2_asm(x)) }
}

#[inline(always)]
unsafe fn div8_asm(x: u64) -> u64 {
    unsafe { div2_asm(div4_asm(x)) }
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

/// Subtract two Goldilocks elements with borrow handling using inline assembly.
#[inline(always)]
unsafe fn sub_asm(a: u64, b: u64) -> u64 {
    let result: u64;
    let _adj: u64;

    unsafe {
        asm!(
            "subs  {result}, {a}, {b}",
            "csetm {adj:w}, cc",
            "sub   {result}, {result}, {adj}",
            a = in(reg) a,
            b = in(reg) b,
            result = out(reg) result,
            adj = out(reg) _adj,
            options(pure, nomem, nostack),
        );
    }

    result
}

/// Reference implementation of internal round for testing.
#[cfg(test)]
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

        // Complete the sum with s0^7 and apply the diagonal for state[0] in one fused op.
        let sum = add_asm(sum_hi, s0_7);
        state[0] = mul_add_asm(s0_7, diag[0], sum);

        // Finalize state[1..WIDTH] = diag_muls[i] + sum
        for i in 1..WIDTH {
            state[i] = add_asm(diag_muls[i], sum);
        }
    }
}

#[cfg(test)]
#[inline(always)]
#[allow(dead_code)]
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

        // Compute state[0] = s0^7 * diag[0] + sum for both lanes (fused).
        state0[0] = mul_add_asm(s0_7_a, diag[0], sum_a);
        state1[0] = mul_add_asm(s0_7_b, diag[0], sum_b);

        // Finalize state[1..WIDTH] for both lanes
        for i in 1..WIDTH {
            state0[i] = add_asm(diag_muls_a[i], sum_a);
            state1[i] = add_asm(diag_muls_b[i], sum_b);
        }
    }
}

#[cfg(test)]
#[inline(always)]
#[allow(dead_code)]
pub unsafe fn internal_round_dual_asm_w12(
    state0: &mut [u64; 12],
    state1: &mut [u64; 12],
    _diag: &[u64; 12],
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

        // Diagonal multiplies using cheap ops
        // V = [-2, 1, 2, 1/2, 3, 4, -1/2, -3, -4, 1/4, -1/4, 1/8]

        let d1_a = state0[1];
        let d1_b = state1[1];
        let d2_a = double_asm(state0[2]);
        let d2_b = double_asm(state1[2]);
        let d3_a = div2_asm(state0[3]);
        let d3_b = div2_asm(state1[3]);
        let d4_a = add_asm(double_asm(state0[4]), state0[4]);
        let d4_b = add_asm(double_asm(state1[4]), state1[4]);

        let sum1234_a = add_asm(sum12_a, sum34_a);
        let sum1234_b = add_asm(sum12_b, sum34_b);

        let d5_a = double_asm(double_asm(state0[5]));
        let d5_b = double_asm(double_asm(state1[5]));
        let d6_a = div2_asm(state0[6]);
        let d6_b = div2_asm(state1[6]);
        let d7_a = add_asm(double_asm(state0[7]), state0[7]);
        let d7_b = add_asm(double_asm(state1[7]), state1[7]);
        let d8_a = double_asm(double_asm(state0[8]));
        let d8_b = double_asm(double_asm(state1[8]));

        let sum_hi_a = add_asm(sum1234_a, sum511_a);
        let sum_hi_b = add_asm(sum1234_b, sum511_b);

        let d9_a = div4_asm(state0[9]);
        let d9_b = div4_asm(state1[9]);
        let d10_a = div4_asm(state0[10]);
        let d10_b = div4_asm(state1[10]);
        let d11_a = div8_asm(state0[11]);
        let d11_b = div8_asm(state1[11]);

        // Finish S-box
        let s0_7_a = mul_asm(s0_3_a, s0_4_a);
        let s0_7_b = mul_asm(s0_3_b, s0_4_b);

        let sum_a = add_asm(sum_hi_a, s0_7_a);
        let sum_b = add_asm(sum_hi_b, s0_7_b);

        // V[0] = -2
        state0[0] = sub_asm(sum_a, double_asm(s0_7_a));
        state1[0] = sub_asm(sum_b, double_asm(s0_7_b));

        // Positive coefficients
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
        // Negative coefficients
        state0[6] = sub_asm(sum_a, d6_a);
        state1[6] = sub_asm(sum_b, d6_b);
        state0[7] = sub_asm(sum_a, d7_a);
        state1[7] = sub_asm(sum_b, d7_b);
        state0[8] = sub_asm(sum_a, d8_a);
        state1[8] = sub_asm(sum_b, d8_b);
        // Positive fractional
        state0[9] = add_asm(d9_a, sum_a);
        state1[9] = add_asm(d9_b, sum_b);
        // Negative fractional
        state0[10] = sub_asm(sum_a, d10_a);
        state1[10] = sub_asm(sum_b, d10_b);
        // Positive fractional
        state0[11] = add_asm(d11_a, sum_a);
        state1[11] = add_asm(d11_b, sum_b);
    }
}

/// Split-state generic internal permute: s0 stays in a register across all rounds.
/// Uses diag[0] for the general diagonal formula (not the V[0]=-2 shortcut).
#[inline]
#[allow(clippy::needless_range_loop)]
pub fn internal_permute_state_asm<const WIDTH: usize>(
    state: &mut [u64; WIDTH],
    diag: &[u64; WIDTH],
    constants: &[u64],
) {
    let mut s0 = state[0];
    for &rc in constants {
        unsafe {
            s0 = add_asm(s0, rc);
            let s0_2 = mul_asm(s0, s0);
            let s0_3 = mul_asm(s0_2, s0);
            let s0_4 = mul_asm(s0_2, s0_2);
            s0 = mul_asm(s0_3, s0_4);

            let mut sum_hi: u64 = 0;
            for i in 1..WIDTH {
                sum_hi = add_asm(sum_hi, state[i]);
            }

            let mut diag_muls: [u64; WIDTH] = [0; WIDTH];
            for i in 1..WIDTH {
                diag_muls[i] = mul_asm(state[i], diag[i]);
            }

            let sum = add_asm(sum_hi, s0);
            s0 = mul_add_asm(s0, diag[0], sum);

            for i in 1..WIDTH {
                state[i] = add_asm(diag_muls[i], sum);
            }
        }
    }
    state[0] = s0;
}

/// Split-state generic dual-lane internal permute for packed processing.
/// Uses diag[0] for the general diagonal formula (not the V[0]=-2 shortcut).
#[inline]
#[allow(clippy::needless_range_loop)]
pub fn internal_permute_split_dual<const WIDTH: usize>(
    lane0: &mut [u64; WIDTH],
    lane1: &mut [u64; WIDTH],
    diag: &[u64; WIDTH],
    constants: &[u64],
) {
    let mut s0_a = lane0[0];
    let mut s0_b = lane1[0];
    for &rc in constants {
        unsafe {
            s0_a = add_asm(s0_a, rc);
            s0_b = add_asm(s0_b, rc);
            let s0_2_a = mul_asm(s0_a, s0_a);
            let s0_2_b = mul_asm(s0_b, s0_b);
            let s0_3_a = mul_asm(s0_2_a, s0_a);
            let s0_3_b = mul_asm(s0_2_b, s0_b);
            let s0_4_a = mul_asm(s0_2_a, s0_2_a);
            let s0_4_b = mul_asm(s0_2_b, s0_2_b);
            s0_a = mul_asm(s0_3_a, s0_4_a);
            s0_b = mul_asm(s0_3_b, s0_4_b);

            let mut sum_hi_a: u64 = 0;
            let mut sum_hi_b: u64 = 0;
            for i in 1..WIDTH {
                sum_hi_a = add_asm(sum_hi_a, lane0[i]);
                sum_hi_b = add_asm(sum_hi_b, lane1[i]);
            }

            let mut diag_muls_a: [u64; WIDTH] = [0; WIDTH];
            let mut diag_muls_b: [u64; WIDTH] = [0; WIDTH];
            for i in 1..WIDTH {
                diag_muls_a[i] = mul_asm(lane0[i], diag[i]);
                diag_muls_b[i] = mul_asm(lane1[i], diag[i]);
            }

            let sum_a = add_asm(sum_hi_a, s0_a);
            let sum_b = add_asm(sum_hi_b, s0_b);
            s0_a = mul_add_asm(s0_a, diag[0], sum_a);
            s0_b = mul_add_asm(s0_b, diag[0], sum_b);

            for i in 1..WIDTH {
                lane0[i] = add_asm(diag_muls_a[i], sum_a);
                lane1[i] = add_asm(diag_muls_b[i], sum_b);
            }
        }
    }
    lane0[0] = s0_a;
    lane1[0] = s0_b;
}

#[cfg(test)]
#[inline(always)]
pub unsafe fn internal_round_asm_w12(state: &mut [u64; 12], _diag: &[u64; 12], rc: u64) {
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

        // Diagonal multiplies using cheap ops
        // V = [-2, 1, 2, 1/2, 3, 4, -1/2, -3, -4, 1/4, -1/4, 1/8]
        let d1 = state[1];
        let d2 = double_asm(state[2]);
        let d3 = div2_asm(state[3]);
        let d4 = add_asm(double_asm(state[4]), state[4]);
        let d5 = double_asm(double_asm(state[5]));

        let sum1234 = add_asm(sum12, sum34);
        let sum_hi = add_asm(sum1234, sum511);

        let d6 = div2_asm(state[6]);
        let d7 = add_asm(double_asm(state[7]), state[7]);
        let d8 = double_asm(double_asm(state[8]));
        let d9 = div4_asm(state[9]);
        let d10 = div4_asm(state[10]);
        let d11 = div8_asm(state[11]);

        // Finish S-box
        let s0_7 = mul_asm(s0_3, s0_4);
        let sum = add_asm(sum_hi, s0_7);

        // V[0] = -2
        state[0] = sub_asm(sum, double_asm(s0_7));

        // Positive coefficients
        state[1] = add_asm(d1, sum);
        state[2] = add_asm(d2, sum);
        state[3] = add_asm(d3, sum);
        state[4] = add_asm(d4, sum);
        state[5] = add_asm(d5, sum);
        // Negative coefficients
        state[6] = sub_asm(sum, d6);
        state[7] = sub_asm(sum, d7);
        state[8] = sub_asm(sum, d8);
        // Positive fractional
        state[9] = add_asm(d9, sum);
        // Negative fractional
        state[10] = sub_asm(sum, d10);
        // Positive fractional
        state[11] = add_asm(d11, sum);
    }
}

#[cfg(test)]
#[inline(always)]
pub unsafe fn internal_round_asm_w8(state: &mut [u64; 8], _diag: &[u64; 8], rc: u64) {
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

        // Diagonal multiplies.
        // V[1]=1: identity
        let d1 = state[1];
        // V[2]=2
        let d2 = double_asm(state[2]);
        // V[3]=1/2
        let d3 = div2_asm(state[3]);
        // V[4]=3
        let d4 = add_asm(double_asm(state[4]), state[4]);

        let sum_hi = add_asm(sum12, sum37);

        // V[5]=1/2 (absolute value; sign applied at final step)
        let d5 = div2_asm(state[5]);
        // V[6]=3 (absolute value; sign applied at final step)
        let d6 = add_asm(double_asm(state[6]), state[6]);
        // V[7]=4 (absolute value; sign applied at final step)
        let d7 = double_asm(double_asm(state[7]));

        // Finish S-box
        let s0_7 = mul_asm(s0_3, s0_4);
        let sum = add_asm(sum_hi, s0_7);

        // V[0]=-2: state[0] = sum - 2*s0_7
        state[0] = sub_asm(sum, double_asm(s0_7));

        // Positive coefficients: state[i] = d_i + sum
        state[1] = add_asm(d1, sum);
        state[2] = add_asm(d2, sum);
        state[3] = add_asm(d3, sum);
        state[4] = add_asm(d4, sum);
        // Negative coefficients: state[i] = sum - |d_i|
        state[5] = sub_asm(sum, d5);
        state[6] = sub_asm(sum, d6);
        state[7] = sub_asm(sum, d7);
    }
}

#[cfg(test)]
#[inline(always)]
#[allow(dead_code)]
pub unsafe fn internal_round_dual_asm_w8(
    state0: &mut [u64; 8],
    state1: &mut [u64; 8],
    _diag: &[u64; 8],
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

        // Diagonal multiplies using cheap ops (V = [-2, 1, 2, 1/2, 3, -1/2, -3, -4])

        // V[1] = 1
        let d1_a = state0[1];
        let d1_b = state1[1];

        // V[2] = 2
        let d2_a = double_asm(state0[2]);
        let d2_b = double_asm(state1[2]);

        // V[3] = 1/2
        let d3_a = div2_asm(state0[3]);
        let d3_b = div2_asm(state1[3]);

        // V[4] = 3
        let d4_a = add_asm(double_asm(state0[4]), state0[4]);
        let d4_b = add_asm(double_asm(state1[4]), state1[4]);

        let sum_hi_a = add_asm(sum12_a, sum37_a);
        let sum_hi_b = add_asm(sum12_b, sum37_b);

        // V[5] = |1/2| (sign applied at final step)
        let d5_a = div2_asm(state0[5]);
        let d5_b = div2_asm(state1[5]);

        // V[6] = |3| (sign applied at final step)
        let d6_a = add_asm(double_asm(state0[6]), state0[6]);
        let d6_b = add_asm(double_asm(state1[6]), state1[6]);

        // V[7] = |4| (sign applied at final step)
        let d7_a = double_asm(double_asm(state0[7]));
        let d7_b = double_asm(double_asm(state1[7]));

        // Finish S-box
        let s0_7_a = mul_asm(s0_3_a, s0_4_a);
        let s0_7_b = mul_asm(s0_3_b, s0_4_b);

        let sum_a = add_asm(sum_hi_a, s0_7_a);
        let sum_b = add_asm(sum_hi_b, s0_7_b);

        // V[0] = -2: state[0] = sum - 2*s0_7
        state0[0] = sub_asm(sum_a, double_asm(s0_7_a));
        state1[0] = sub_asm(sum_b, double_asm(s0_7_b));

        // Positive coefficients: state[i] = d_i + sum
        state0[1] = add_asm(d1_a, sum_a);
        state1[1] = add_asm(d1_b, sum_b);
        state0[2] = add_asm(d2_a, sum_a);
        state1[2] = add_asm(d2_b, sum_b);
        state0[3] = add_asm(d3_a, sum_a);
        state1[3] = add_asm(d3_b, sum_b);
        state0[4] = add_asm(d4_a, sum_a);
        state1[4] = add_asm(d4_b, sum_b);
        // Negative coefficients: state[i] = sum - |d_i|
        state0[5] = sub_asm(sum_a, d5_a);
        state1[5] = sub_asm(sum_b, d5_b);
        state0[6] = sub_asm(sum_a, d6_a);
        state1[6] = sub_asm(sum_b, d6_b);
        state0[7] = sub_asm(sum_a, d7_a);
        state1[7] = sub_asm(sum_b, d7_b);
    }
}

/// Split-state W8 internal permute: s0 stays in a register across all rounds,
/// avoiding 22 loads + 22 stores of state[0] per permutation.
#[inline]
pub fn internal_permute_state_asm_w8(state: &mut [u64; 8], constants: &[u64]) {
    let mut s0 = state[0];
    for &rc in constants {
        unsafe {
            s0 = add_asm(s0, rc);
            let s0_2 = mul_asm(s0, s0);

            let sum1 = add_asm(state[1], state[2]);
            let sum2 = add_asm(state[3], state[4]);
            let sum3 = add_asm(state[5], state[6]);

            let s0_3 = mul_asm(s0_2, s0);
            let s0_4 = mul_asm(s0_2, s0_2);

            let sum12 = add_asm(sum1, sum2);
            let sum37 = add_asm(sum3, state[7]);

            let d1 = state[1];
            let d2 = double_asm(state[2]);
            let d3 = div2_asm(state[3]);
            let d4 = add_asm(double_asm(state[4]), state[4]);

            let sum_hi = add_asm(sum12, sum37);

            let d5 = div2_asm(state[5]);
            let d6 = add_asm(double_asm(state[6]), state[6]);
            let d7 = double_asm(double_asm(state[7]));

            s0 = mul_asm(s0_3, s0_4);
            let sum = add_asm(sum_hi, s0);
            // V[0]=-2: new_s0 = sum + (-2)*s0 = sum_hi + s0 - 2*s0 = sum_hi - s0
            s0 = sub_asm(sum_hi, s0);

            state[1] = add_asm(d1, sum);
            state[2] = add_asm(d2, sum);
            state[3] = add_asm(d3, sum);
            state[4] = add_asm(d4, sum);
            state[5] = sub_asm(sum, d5);
            state[6] = sub_asm(sum, d6);
            state[7] = sub_asm(sum, d7);
        }
    }
    state[0] = s0;
}

/// Split-state dual-lane W8 internal permute for packed processing.
#[inline]
pub fn internal_permute_split_dual_w8(
    lane0: &mut [u64; 8],
    lane1: &mut [u64; 8],
    constants: &[u64],
) {
    let mut s0_a = lane0[0];
    let mut s0_b = lane1[0];
    for &rc in constants {
        unsafe {
            s0_a = add_asm(s0_a, rc);
            s0_b = add_asm(s0_b, rc);

            let s0_2_a = mul_asm(s0_a, s0_a);
            let s0_2_b = mul_asm(s0_b, s0_b);

            let sum1_a = add_asm(lane0[1], lane0[2]);
            let sum1_b = add_asm(lane1[1], lane1[2]);
            let sum2_a = add_asm(lane0[3], lane0[4]);
            let sum2_b = add_asm(lane1[3], lane1[4]);
            let sum3_a = add_asm(lane0[5], lane0[6]);
            let sum3_b = add_asm(lane1[5], lane1[6]);

            let s0_3_a = mul_asm(s0_2_a, s0_a);
            let s0_3_b = mul_asm(s0_2_b, s0_b);
            let s0_4_a = mul_asm(s0_2_a, s0_2_a);
            let s0_4_b = mul_asm(s0_2_b, s0_2_b);

            let sum12_a = add_asm(sum1_a, sum2_a);
            let sum12_b = add_asm(sum1_b, sum2_b);
            let sum37_a = add_asm(sum3_a, lane0[7]);
            let sum37_b = add_asm(sum3_b, lane1[7]);

            let d1_a = lane0[1];
            let d1_b = lane1[1];
            let d2_a = double_asm(lane0[2]);
            let d2_b = double_asm(lane1[2]);
            let d3_a = div2_asm(lane0[3]);
            let d3_b = div2_asm(lane1[3]);
            let d4_a = add_asm(double_asm(lane0[4]), lane0[4]);
            let d4_b = add_asm(double_asm(lane1[4]), lane1[4]);

            let sum_hi_a = add_asm(sum12_a, sum37_a);
            let sum_hi_b = add_asm(sum12_b, sum37_b);

            let d5_a = div2_asm(lane0[5]);
            let d5_b = div2_asm(lane1[5]);
            let d6_a = add_asm(double_asm(lane0[6]), lane0[6]);
            let d6_b = add_asm(double_asm(lane1[6]), lane1[6]);
            let d7_a = double_asm(double_asm(lane0[7]));
            let d7_b = double_asm(double_asm(lane1[7]));

            s0_a = mul_asm(s0_3_a, s0_4_a);
            s0_b = mul_asm(s0_3_b, s0_4_b);

            let sum_a = add_asm(sum_hi_a, s0_a);
            let sum_b = add_asm(sum_hi_b, s0_b);
            s0_a = sub_asm(sum_hi_a, s0_a);
            s0_b = sub_asm(sum_hi_b, s0_b);

            lane0[1] = add_asm(d1_a, sum_a);
            lane1[1] = add_asm(d1_b, sum_b);
            lane0[2] = add_asm(d2_a, sum_a);
            lane1[2] = add_asm(d2_b, sum_b);
            lane0[3] = add_asm(d3_a, sum_a);
            lane1[3] = add_asm(d3_b, sum_b);
            lane0[4] = add_asm(d4_a, sum_a);
            lane1[4] = add_asm(d4_b, sum_b);
            lane0[5] = sub_asm(sum_a, d5_a);
            lane1[5] = sub_asm(sum_b, d5_b);
            lane0[6] = sub_asm(sum_a, d6_a);
            lane1[6] = sub_asm(sum_b, d6_b);
            lane0[7] = sub_asm(sum_a, d7_a);
            lane1[7] = sub_asm(sum_b, d7_b);
        }
    }
    lane0[0] = s0_a;
    lane1[0] = s0_b;
}

/// Split-state W12 internal permute: s0 stays in a register across all rounds.
#[inline]
pub fn internal_permute_state_asm_w12(state: &mut [u64; 12], constants: &[u64]) {
    let mut s0 = state[0];
    for &rc in constants {
        unsafe {
            s0 = add_asm(s0, rc);
            let s0_2 = mul_asm(s0, s0);

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

            let d1 = state[1];
            let d2 = double_asm(state[2]);
            let d3 = div2_asm(state[3]);
            let d4 = add_asm(double_asm(state[4]), state[4]);

            let sum1234 = add_asm(sum12, sum34);

            let d5 = double_asm(double_asm(state[5]));
            let d6 = div2_asm(state[6]);
            let d7 = add_asm(double_asm(state[7]), state[7]);
            let d8 = double_asm(double_asm(state[8]));

            let sum_hi = add_asm(sum1234, sum511);

            let d9 = div4_asm(state[9]);
            let d10 = div4_asm(state[10]);
            let d11 = div8_asm(state[11]);

            s0 = mul_asm(s0_3, s0_4);
            let sum = add_asm(sum_hi, s0);
            s0 = sub_asm(sum_hi, s0);

            state[1] = add_asm(d1, sum);
            state[2] = add_asm(d2, sum);
            state[3] = add_asm(d3, sum);
            state[4] = add_asm(d4, sum);
            state[5] = add_asm(d5, sum);
            state[6] = sub_asm(sum, d6);
            state[7] = sub_asm(sum, d7);
            state[8] = sub_asm(sum, d8);
            state[9] = add_asm(d9, sum);
            state[10] = sub_asm(sum, d10);
            state[11] = add_asm(d11, sum);
        }
    }
    state[0] = s0;
}

/// Split-state dual-lane W12 internal permute for packed processing.
#[inline]
pub fn internal_permute_split_dual_w12(
    lane0: &mut [u64; 12],
    lane1: &mut [u64; 12],
    constants: &[u64],
) {
    let mut s0_a = lane0[0];
    let mut s0_b = lane1[0];
    for &rc in constants {
        unsafe {
            s0_a = add_asm(s0_a, rc);
            s0_b = add_asm(s0_b, rc);

            let s0_2_a = mul_asm(s0_a, s0_a);
            let s0_2_b = mul_asm(s0_b, s0_b);

            let sum1_a = add_asm(lane0[1], lane0[2]);
            let sum1_b = add_asm(lane1[1], lane1[2]);
            let sum2_a = add_asm(lane0[3], lane0[4]);
            let sum2_b = add_asm(lane1[3], lane1[4]);
            let sum3_a = add_asm(lane0[5], lane0[6]);
            let sum3_b = add_asm(lane1[5], lane1[6]);
            let sum4_a = add_asm(lane0[7], lane0[8]);
            let sum4_b = add_asm(lane1[7], lane1[8]);
            let sum5_a = add_asm(lane0[9], lane0[10]);
            let sum5_b = add_asm(lane1[9], lane1[10]);

            let s0_3_a = mul_asm(s0_2_a, s0_a);
            let s0_3_b = mul_asm(s0_2_b, s0_b);
            let s0_4_a = mul_asm(s0_2_a, s0_2_a);
            let s0_4_b = mul_asm(s0_2_b, s0_2_b);

            let sum12_a = add_asm(sum1_a, sum2_a);
            let sum12_b = add_asm(sum1_b, sum2_b);
            let sum34_a = add_asm(sum3_a, sum4_a);
            let sum34_b = add_asm(sum3_b, sum4_b);
            let sum511_a = add_asm(sum5_a, lane0[11]);
            let sum511_b = add_asm(sum5_b, lane1[11]);

            let d1_a = lane0[1];
            let d1_b = lane1[1];
            let d2_a = double_asm(lane0[2]);
            let d2_b = double_asm(lane1[2]);
            let d3_a = div2_asm(lane0[3]);
            let d3_b = div2_asm(lane1[3]);
            let d4_a = add_asm(double_asm(lane0[4]), lane0[4]);
            let d4_b = add_asm(double_asm(lane1[4]), lane1[4]);

            let sum1234_a = add_asm(sum12_a, sum34_a);
            let sum1234_b = add_asm(sum12_b, sum34_b);

            let d5_a = double_asm(double_asm(lane0[5]));
            let d5_b = double_asm(double_asm(lane1[5]));
            let d6_a = div2_asm(lane0[6]);
            let d6_b = div2_asm(lane1[6]);
            let d7_a = add_asm(double_asm(lane0[7]), lane0[7]);
            let d7_b = add_asm(double_asm(lane1[7]), lane1[7]);
            let d8_a = double_asm(double_asm(lane0[8]));
            let d8_b = double_asm(double_asm(lane1[8]));

            let sum_hi_a = add_asm(sum1234_a, sum511_a);
            let sum_hi_b = add_asm(sum1234_b, sum511_b);

            let d9_a = div4_asm(lane0[9]);
            let d9_b = div4_asm(lane1[9]);
            let d10_a = div4_asm(lane0[10]);
            let d10_b = div4_asm(lane1[10]);
            let d11_a = div8_asm(lane0[11]);
            let d11_b = div8_asm(lane1[11]);

            s0_a = mul_asm(s0_3_a, s0_4_a);
            s0_b = mul_asm(s0_3_b, s0_4_b);

            let sum_a = add_asm(sum_hi_a, s0_a);
            let sum_b = add_asm(sum_hi_b, s0_b);
            s0_a = sub_asm(sum_hi_a, s0_a);
            s0_b = sub_asm(sum_hi_b, s0_b);

            lane0[1] = add_asm(d1_a, sum_a);
            lane1[1] = add_asm(d1_b, sum_b);
            lane0[2] = add_asm(d2_a, sum_a);
            lane1[2] = add_asm(d2_b, sum_b);
            lane0[3] = add_asm(d3_a, sum_a);
            lane1[3] = add_asm(d3_b, sum_b);
            lane0[4] = add_asm(d4_a, sum_a);
            lane1[4] = add_asm(d4_b, sum_b);
            lane0[5] = add_asm(d5_a, sum_a);
            lane1[5] = add_asm(d5_b, sum_b);
            lane0[6] = sub_asm(sum_a, d6_a);
            lane1[6] = sub_asm(sum_b, d6_b);
            lane0[7] = sub_asm(sum_a, d7_a);
            lane1[7] = sub_asm(sum_b, d7_b);
            lane0[8] = sub_asm(sum_a, d8_a);
            lane1[8] = sub_asm(sum_b, d8_b);
            lane0[9] = add_asm(d9_a, sum_a);
            lane1[9] = add_asm(d9_b, sum_b);
            lane0[10] = sub_asm(sum_a, d10_a);
            lane1[10] = sub_asm(sum_b, d10_b);
            lane0[11] = add_asm(d11_a, sum_a);
            lane1[11] = add_asm(d11_b, sum_b);
        }
    }
    lane0[0] = s0_a;
    lane1[0] = s0_b;
}

#[cfg(test)]
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

        let sum1234 = add_asm(sum12, sum34);
        let sum56715 = add_asm(sum56, sum715);
        let sum_hi = add_asm(sum1234, sum56715);

        // Diagonal multiplies using cheap ops where possible.
        // V = [-2, 1, 2, 1/2, 3, 4, -1/2, -3, -4,
        //      1/2^3, 1/2^4, 1/2^5, -1/2^3, -1/2^4, -1/2^5, 1/2^32]
        let d1 = state[1];
        let d2 = double_asm(state[2]);
        let d3 = div2_asm(state[3]);
        let d4 = add_asm(double_asm(state[4]), state[4]);
        let d5 = double_asm(double_asm(state[5]));
        let d6 = div2_asm(state[6]);
        let d7 = add_asm(double_asm(state[7]), state[7]);
        let d8 = double_asm(double_asm(state[8]));

        // Finish S-box
        let s0_7 = mul_asm(s0_3, s0_4);
        let sum = add_asm(sum_hi, s0_7);

        // V[0] = -2
        state[0] = sub_asm(sum, double_asm(s0_7));

        // Positive coefficients
        state[1] = add_asm(d1, sum);
        state[2] = add_asm(d2, sum);
        state[3] = add_asm(d3, sum);
        state[4] = add_asm(d4, sum);
        state[5] = add_asm(d5, sum);
        // Negative coefficients
        state[6] = sub_asm(sum, d6);
        state[7] = sub_asm(sum, d7);
        state[8] = sub_asm(sum, d8);

        // V[9..15]: higher powers of 1/2 -- keep mul_add_asm
        state[9] = mul_add_asm(state[9], diag[9], sum);
        state[10] = mul_add_asm(state[10], diag[10], sum);
        state[11] = mul_add_asm(state[11], diag[11], sum);
        state[12] = mul_add_asm(state[12], diag[12], sum);
        state[13] = mul_add_asm(state[13], diag[13], sum);
        state[14] = mul_add_asm(state[14], diag[14], sum);
        state[15] = mul_add_asm(state[15], diag[15], sum);
    }
}

#[cfg(test)]
#[inline(always)]
#[allow(dead_code)]
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

        // Diagonal multiplies - first batch, interleaved.
        // These lanes ultimately appear only as diag[i] * state[i] + sum, so we
        // will compute them with mul_add_asm in the finalize step.

        // More partial sum combining
        let sum1234_a = add_asm(sum12_a, sum34_a);
        let sum1234_b = add_asm(sum12_b, sum34_b);
        let sum56715_a = add_asm(sum56_a, sum715_a);
        let sum56715_b = add_asm(sum56_b, sum715_b);
        let sum_hi_a = add_asm(sum1234_a, sum56715_a);
        let sum_hi_b = add_asm(sum1234_b, sum56715_b);

        // Diagonal multiplies using cheap ops (overlap with s0_7 multiply).
        // V = [-2, 1, 2, 1/2, 3, 4, -1/2, -3, -4,
        //      1/2^3, 1/2^4, 1/2^5, -1/2^3, -1/2^4, -1/2^5, 1/2^32]
        let d1_a = state0[1];
        let d1_b = state1[1];
        let d2_a = double_asm(state0[2]);
        let d2_b = double_asm(state1[2]);
        let d3_a = div2_asm(state0[3]);
        let d3_b = div2_asm(state1[3]);
        let d4_a = add_asm(double_asm(state0[4]), state0[4]);
        let d4_b = add_asm(double_asm(state1[4]), state1[4]);
        let d5_a = double_asm(double_asm(state0[5]));
        let d5_b = double_asm(double_asm(state1[5]));
        let d6_a = div2_asm(state0[6]);
        let d6_b = div2_asm(state1[6]);
        let d7_a = add_asm(double_asm(state0[7]), state0[7]);
        let d7_b = add_asm(double_asm(state1[7]), state1[7]);
        let d8_a = double_asm(double_asm(state0[8]));
        let d8_b = double_asm(double_asm(state1[8]));

        // Finish S-box
        let s0_7_a = mul_asm(s0_3_a, s0_4_a);
        let s0_7_b = mul_asm(s0_3_b, s0_4_b);

        let sum_a = add_asm(sum_hi_a, s0_7_a);
        let sum_b = add_asm(sum_hi_b, s0_7_b);

        // V[0] = -2
        state0[0] = sub_asm(sum_a, double_asm(s0_7_a));
        state1[0] = sub_asm(sum_b, double_asm(s0_7_b));

        // Positive coefficients
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
        // Negative coefficients
        state0[6] = sub_asm(sum_a, d6_a);
        state1[6] = sub_asm(sum_b, d6_b);
        state0[7] = sub_asm(sum_a, d7_a);
        state1[7] = sub_asm(sum_b, d7_b);
        state0[8] = sub_asm(sum_a, d8_a);
        state1[8] = sub_asm(sum_b, d8_b);

        // V[9..15]: higher powers of 1/2 -- keep mul_add_asm
        state0[9] = mul_add_asm(state0[9], diag[9], sum_a);
        state1[9] = mul_add_asm(state1[9], diag[9], sum_b);
        state0[10] = mul_add_asm(state0[10], diag[10], sum_a);
        state1[10] = mul_add_asm(state1[10], diag[10], sum_b);
        state0[11] = mul_add_asm(state0[11], diag[11], sum_a);
        state1[11] = mul_add_asm(state1[11], diag[11], sum_b);
        state0[12] = mul_add_asm(state0[12], diag[12], sum_a);
        state1[12] = mul_add_asm(state1[12], diag[12], sum_b);
        state0[13] = mul_add_asm(state0[13], diag[13], sum_a);
        state1[13] = mul_add_asm(state1[13], diag[13], sum_b);
        state0[14] = mul_add_asm(state0[14], diag[14], sum_a);
        state1[14] = mul_add_asm(state1[14], diag[14], sum_b);
        state0[15] = mul_add_asm(state0[15], diag[15], sum_a);
        state1[15] = mul_add_asm(state1[15], diag[15], sum_b);
    }
}

/// Split-state W16 internal permute: s0 stays in a register across all rounds.
#[inline]
pub fn internal_permute_state_asm_w16(state: &mut [u64; 16], diag: &[u64; 16], constants: &[u64]) {
    let mut s0 = state[0];
    for &rc in constants {
        unsafe {
            s0 = add_asm(s0, rc);
            let s0_2 = mul_asm(s0, s0);

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

            let sum1234 = add_asm(sum12, sum34);
            let sum56715 = add_asm(sum56, sum715);
            let sum_hi = add_asm(sum1234, sum56715);

            let d1 = state[1];
            let d2 = double_asm(state[2]);
            let d3 = div2_asm(state[3]);
            let d4 = add_asm(double_asm(state[4]), state[4]);
            let d5 = double_asm(double_asm(state[5]));
            let d6 = div2_asm(state[6]);
            let d7 = add_asm(double_asm(state[7]), state[7]);
            let d8 = double_asm(double_asm(state[8]));

            s0 = mul_asm(s0_3, s0_4);
            let sum = add_asm(sum_hi, s0);
            s0 = sub_asm(sum_hi, s0);

            state[1] = add_asm(d1, sum);
            state[2] = add_asm(d2, sum);
            state[3] = add_asm(d3, sum);
            state[4] = add_asm(d4, sum);
            state[5] = add_asm(d5, sum);
            state[6] = sub_asm(sum, d6);
            state[7] = sub_asm(sum, d7);
            state[8] = sub_asm(sum, d8);

            state[9] = mul_add_asm(state[9], diag[9], sum);
            state[10] = mul_add_asm(state[10], diag[10], sum);
            state[11] = mul_add_asm(state[11], diag[11], sum);
            state[12] = mul_add_asm(state[12], diag[12], sum);
            state[13] = mul_add_asm(state[13], diag[13], sum);
            state[14] = mul_add_asm(state[14], diag[14], sum);
            state[15] = mul_add_asm(state[15], diag[15], sum);
        }
    }
    state[0] = s0;
}

/// Split-state dual-lane W16 internal permute for packed processing.
#[inline]
pub fn internal_permute_split_dual_w16(
    lane0: &mut [u64; 16],
    lane1: &mut [u64; 16],
    diag: &[u64; 16],
    constants: &[u64],
) {
    let mut s0_a = lane0[0];
    let mut s0_b = lane1[0];
    for &rc in constants {
        unsafe {
            s0_a = add_asm(s0_a, rc);
            s0_b = add_asm(s0_b, rc);

            let s0_2_a = mul_asm(s0_a, s0_a);
            let s0_2_b = mul_asm(s0_b, s0_b);

            let sum1_a = add_asm(lane0[1], lane0[2]);
            let sum1_b = add_asm(lane1[1], lane1[2]);
            let sum2_a = add_asm(lane0[3], lane0[4]);
            let sum2_b = add_asm(lane1[3], lane1[4]);
            let sum3_a = add_asm(lane0[5], lane0[6]);
            let sum3_b = add_asm(lane1[5], lane1[6]);
            let sum4_a = add_asm(lane0[7], lane0[8]);
            let sum4_b = add_asm(lane1[7], lane1[8]);
            let sum5_a = add_asm(lane0[9], lane0[10]);
            let sum5_b = add_asm(lane1[9], lane1[10]);
            let sum6_a = add_asm(lane0[11], lane0[12]);
            let sum6_b = add_asm(lane1[11], lane1[12]);
            let sum7_a = add_asm(lane0[13], lane0[14]);
            let sum7_b = add_asm(lane1[13], lane1[14]);

            let s0_3_a = mul_asm(s0_2_a, s0_a);
            let s0_3_b = mul_asm(s0_2_b, s0_b);
            let s0_4_a = mul_asm(s0_2_a, s0_2_a);
            let s0_4_b = mul_asm(s0_2_b, s0_2_b);

            let sum12_a = add_asm(sum1_a, sum2_a);
            let sum12_b = add_asm(sum1_b, sum2_b);
            let sum34_a = add_asm(sum3_a, sum4_a);
            let sum34_b = add_asm(sum3_b, sum4_b);
            let sum56_a = add_asm(sum5_a, sum6_a);
            let sum56_b = add_asm(sum5_b, sum6_b);
            let sum715_a = add_asm(sum7_a, lane0[15]);
            let sum715_b = add_asm(sum7_b, lane1[15]);

            let sum1234_a = add_asm(sum12_a, sum34_a);
            let sum1234_b = add_asm(sum12_b, sum34_b);
            let sum56715_a = add_asm(sum56_a, sum715_a);
            let sum56715_b = add_asm(sum56_b, sum715_b);
            let sum_hi_a = add_asm(sum1234_a, sum56715_a);
            let sum_hi_b = add_asm(sum1234_b, sum56715_b);

            let d1_a = lane0[1];
            let d1_b = lane1[1];
            let d2_a = double_asm(lane0[2]);
            let d2_b = double_asm(lane1[2]);
            let d3_a = div2_asm(lane0[3]);
            let d3_b = div2_asm(lane1[3]);
            let d4_a = add_asm(double_asm(lane0[4]), lane0[4]);
            let d4_b = add_asm(double_asm(lane1[4]), lane1[4]);
            let d5_a = double_asm(double_asm(lane0[5]));
            let d5_b = double_asm(double_asm(lane1[5]));
            let d6_a = div2_asm(lane0[6]);
            let d6_b = div2_asm(lane1[6]);
            let d7_a = add_asm(double_asm(lane0[7]), lane0[7]);
            let d7_b = add_asm(double_asm(lane1[7]), lane1[7]);
            let d8_a = double_asm(double_asm(lane0[8]));
            let d8_b = double_asm(double_asm(lane1[8]));

            s0_a = mul_asm(s0_3_a, s0_4_a);
            s0_b = mul_asm(s0_3_b, s0_4_b);

            let sum_a = add_asm(sum_hi_a, s0_a);
            let sum_b = add_asm(sum_hi_b, s0_b);
            s0_a = sub_asm(sum_hi_a, s0_a);
            s0_b = sub_asm(sum_hi_b, s0_b);

            lane0[1] = add_asm(d1_a, sum_a);
            lane1[1] = add_asm(d1_b, sum_b);
            lane0[2] = add_asm(d2_a, sum_a);
            lane1[2] = add_asm(d2_b, sum_b);
            lane0[3] = add_asm(d3_a, sum_a);
            lane1[3] = add_asm(d3_b, sum_b);
            lane0[4] = add_asm(d4_a, sum_a);
            lane1[4] = add_asm(d4_b, sum_b);
            lane0[5] = add_asm(d5_a, sum_a);
            lane1[5] = add_asm(d5_b, sum_b);
            lane0[6] = sub_asm(sum_a, d6_a);
            lane1[6] = sub_asm(sum_b, d6_b);
            lane0[7] = sub_asm(sum_a, d7_a);
            lane1[7] = sub_asm(sum_b, d7_b);
            lane0[8] = sub_asm(sum_a, d8_a);
            lane1[8] = sub_asm(sum_b, d8_b);

            lane0[9] = mul_add_asm(lane0[9], diag[9], sum_a);
            lane1[9] = mul_add_asm(lane1[9], diag[9], sum_b);
            lane0[10] = mul_add_asm(lane0[10], diag[10], sum_a);
            lane1[10] = mul_add_asm(lane1[10], diag[10], sum_b);
            lane0[11] = mul_add_asm(lane0[11], diag[11], sum_a);
            lane1[11] = mul_add_asm(lane1[11], diag[11], sum_b);
            lane0[12] = mul_add_asm(lane0[12], diag[12], sum_a);
            lane1[12] = mul_add_asm(lane1[12], diag[12], sum_b);
            lane0[13] = mul_add_asm(lane0[13], diag[13], sum_a);
            lane1[13] = mul_add_asm(lane1[13], diag[13], sum_b);
            lane0[14] = mul_add_asm(lane0[14], diag[14], sum_a);
            lane1[14] = mul_add_asm(lane1[14], diag[14], sum_b);
            lane0[15] = mul_add_asm(lane0[15], diag[15], sum_a);
            lane1[15] = mul_add_asm(lane1[15], diag[15], sum_b);
        }
    }
    lane0[0] = s0_a;
    lane1[0] = s0_b;
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

/// Fully unrolled and fused external round for W8: RC add + S-box + MDS in one pass.
/// Keeps intermediate values in registers instead of staging through memory.
#[inline(always)]
pub unsafe fn external_round_fused_w8(state: &mut [u64; 8], rc: &[u64; 8]) {
    unsafe {
        let s0 = add_asm(state[0], rc[0]);
        let s1 = add_asm(state[1], rc[1]);
        let x2_0 = mul_asm(s0, s0);
        let x2_1 = mul_asm(s1, s1);

        let s2 = add_asm(state[2], rc[2]);
        let s3 = add_asm(state[3], rc[3]);
        let x2_2 = mul_asm(s2, s2);
        let x2_3 = mul_asm(s3, s3);

        let s4 = add_asm(state[4], rc[4]);
        let s5 = add_asm(state[5], rc[5]);
        let x2_4 = mul_asm(s4, s4);
        let x2_5 = mul_asm(s5, s5);

        let s6 = add_asm(state[6], rc[6]);
        let s7 = add_asm(state[7], rc[7]);
        let x2_6 = mul_asm(s6, s6);
        let x2_7 = mul_asm(s7, s7);

        let x3_0 = mul_asm(x2_0, s0);
        let x3_1 = mul_asm(x2_1, s1);
        let x4_0 = mul_asm(x2_0, x2_0);
        let x4_1 = mul_asm(x2_1, x2_1);
        let x3_2 = mul_asm(x2_2, s2);
        let x3_3 = mul_asm(x2_3, s3);
        let x4_2 = mul_asm(x2_2, x2_2);
        let x4_3 = mul_asm(x2_3, x2_3);
        let x3_4 = mul_asm(x2_4, s4);
        let x3_5 = mul_asm(x2_5, s5);
        let x4_4 = mul_asm(x2_4, x2_4);
        let x4_5 = mul_asm(x2_5, x2_5);
        let x3_6 = mul_asm(x2_6, s6);
        let x3_7 = mul_asm(x2_7, s7);
        let x4_6 = mul_asm(x2_6, x2_6);
        let x4_7 = mul_asm(x2_7, x2_7);

        state[0] = mul_asm(x3_0, x4_0);
        state[1] = mul_asm(x3_1, x4_1);
        state[2] = mul_asm(x3_2, x4_2);
        state[3] = mul_asm(x3_3, x4_3);
        state[4] = mul_asm(x3_4, x4_4);
        state[5] = mul_asm(x3_5, x4_5);
        state[6] = mul_asm(x3_6, x4_6);
        state[7] = mul_asm(x3_7, x4_7);

        mds_light_permutation_asm(state);
    }
}

/// Fully unrolled and fused dual-lane external round for W8.
/// Processes elements in two halves to manage register pressure.
#[inline(always)]
pub unsafe fn external_round_fused_dual_w8(
    state0: &mut [u64; 8],
    state1: &mut [u64; 8],
    rc: &[u64; 8],
) {
    unsafe {
        // Half 1: elements 0-3 across both lanes
        let s0_a = add_asm(state0[0], rc[0]);
        let s0_b = add_asm(state1[0], rc[0]);
        let s1_a = add_asm(state0[1], rc[1]);
        let s1_b = add_asm(state1[1], rc[1]);
        let s2_a = add_asm(state0[2], rc[2]);
        let s2_b = add_asm(state1[2], rc[2]);
        let s3_a = add_asm(state0[3], rc[3]);
        let s3_b = add_asm(state1[3], rc[3]);

        let x2_0a = mul_asm(s0_a, s0_a);
        let x2_0b = mul_asm(s0_b, s0_b);
        let x2_1a = mul_asm(s1_a, s1_a);
        let x2_1b = mul_asm(s1_b, s1_b);
        let x2_2a = mul_asm(s2_a, s2_a);
        let x2_2b = mul_asm(s2_b, s2_b);
        let x2_3a = mul_asm(s3_a, s3_a);
        let x2_3b = mul_asm(s3_b, s3_b);

        let x3_0a = mul_asm(x2_0a, s0_a);
        let x3_0b = mul_asm(x2_0b, s0_b);
        let x4_0a = mul_asm(x2_0a, x2_0a);
        let x4_0b = mul_asm(x2_0b, x2_0b);
        let x3_1a = mul_asm(x2_1a, s1_a);
        let x3_1b = mul_asm(x2_1b, s1_b);
        let x4_1a = mul_asm(x2_1a, x2_1a);
        let x4_1b = mul_asm(x2_1b, x2_1b);
        let x3_2a = mul_asm(x2_2a, s2_a);
        let x3_2b = mul_asm(x2_2b, s2_b);
        let x4_2a = mul_asm(x2_2a, x2_2a);
        let x4_2b = mul_asm(x2_2b, x2_2b);
        let x3_3a = mul_asm(x2_3a, s3_a);
        let x3_3b = mul_asm(x2_3b, s3_b);
        let x4_3a = mul_asm(x2_3a, x2_3a);
        let x4_3b = mul_asm(x2_3b, x2_3b);

        state0[0] = mul_asm(x3_0a, x4_0a);
        state1[0] = mul_asm(x3_0b, x4_0b);
        state0[1] = mul_asm(x3_1a, x4_1a);
        state1[1] = mul_asm(x3_1b, x4_1b);
        state0[2] = mul_asm(x3_2a, x4_2a);
        state1[2] = mul_asm(x3_2b, x4_2b);
        state0[3] = mul_asm(x3_3a, x4_3a);
        state1[3] = mul_asm(x3_3b, x4_3b);

        // Half 2: elements 4-7 across both lanes
        let s4_a = add_asm(state0[4], rc[4]);
        let s4_b = add_asm(state1[4], rc[4]);
        let s5_a = add_asm(state0[5], rc[5]);
        let s5_b = add_asm(state1[5], rc[5]);
        let s6_a = add_asm(state0[6], rc[6]);
        let s6_b = add_asm(state1[6], rc[6]);
        let s7_a = add_asm(state0[7], rc[7]);
        let s7_b = add_asm(state1[7], rc[7]);

        let x2_4a = mul_asm(s4_a, s4_a);
        let x2_4b = mul_asm(s4_b, s4_b);
        let x2_5a = mul_asm(s5_a, s5_a);
        let x2_5b = mul_asm(s5_b, s5_b);
        let x2_6a = mul_asm(s6_a, s6_a);
        let x2_6b = mul_asm(s6_b, s6_b);
        let x2_7a = mul_asm(s7_a, s7_a);
        let x2_7b = mul_asm(s7_b, s7_b);

        let x3_4a = mul_asm(x2_4a, s4_a);
        let x3_4b = mul_asm(x2_4b, s4_b);
        let x4_4a = mul_asm(x2_4a, x2_4a);
        let x4_4b = mul_asm(x2_4b, x2_4b);
        let x3_5a = mul_asm(x2_5a, s5_a);
        let x3_5b = mul_asm(x2_5b, s5_b);
        let x4_5a = mul_asm(x2_5a, x2_5a);
        let x4_5b = mul_asm(x2_5b, x2_5b);
        let x3_6a = mul_asm(x2_6a, s6_a);
        let x3_6b = mul_asm(x2_6b, s6_b);
        let x4_6a = mul_asm(x2_6a, x2_6a);
        let x4_6b = mul_asm(x2_6b, x2_6b);
        let x3_7a = mul_asm(x2_7a, s7_a);
        let x3_7b = mul_asm(x2_7b, s7_b);
        let x4_7a = mul_asm(x2_7a, x2_7a);
        let x4_7b = mul_asm(x2_7b, x2_7b);

        state0[4] = mul_asm(x3_4a, x4_4a);
        state1[4] = mul_asm(x3_4b, x4_4b);
        state0[5] = mul_asm(x3_5a, x4_5a);
        state1[5] = mul_asm(x3_5b, x4_5b);
        state0[6] = mul_asm(x3_6a, x4_6a);
        state1[6] = mul_asm(x3_6b, x4_6b);
        state0[7] = mul_asm(x3_7a, x4_7a);
        state1[7] = mul_asm(x3_7b, x4_7b);

        mds_light_permutation_asm(state0);
        mds_light_permutation_asm(state1);
    }
}

/// Run initial external rounds with pre-converted raw u64 constants.
#[inline]
pub fn external_initial_permute_state_asm<const WIDTH: usize>(
    state: &mut [u64; WIDTH],
    initial_constants: &[[u64; WIDTH]],
) {
    unsafe {
        mds_light_permutation_asm(state);
    }
    for rc in initial_constants {
        unsafe {
            external_round_asm(state, rc);
        }
    }
}

/// Run terminal external rounds with pre-converted raw u64 constants.
#[inline]
pub fn external_terminal_permute_state_asm<const WIDTH: usize>(
    state: &mut [u64; WIDTH],
    terminal_constants: &[[u64; WIDTH]],
) {
    for rc in terminal_constants {
        unsafe {
            external_round_asm(state, rc);
        }
    }
}

/// W8-specialized initial external permute using fused rounds.
#[inline]
pub fn external_initial_permute_w8(state: &mut [u64; 8], initial_constants: &[[u64; 8]]) {
    unsafe {
        mds_light_permutation_asm(state);
    }
    for rc in initial_constants {
        unsafe {
            external_round_fused_w8(state, rc);
        }
    }
}

/// W8-specialized terminal external permute using fused rounds.
#[inline]
pub fn external_terminal_permute_w8(state: &mut [u64; 8], terminal_constants: &[[u64; 8]]) {
    for rc in terminal_constants {
        unsafe {
            external_round_fused_w8(state, rc);
        }
    }
}

/// Dual-lane initial external permute with pre-converted constants.
#[inline]
pub fn external_initial_permute_dual<const WIDTH: usize>(
    lane0: &mut [u64; WIDTH],
    lane1: &mut [u64; WIDTH],
    constants: &[[u64; WIDTH]],
) {
    unsafe {
        mds_light_permutation_asm(lane0);
        mds_light_permutation_asm(lane1);
    }
    for rc in constants {
        unsafe {
            external_round_dual_asm(lane0, lane1, rc);
        }
    }
}

/// Dual-lane terminal external permute with pre-converted constants.
#[inline]
pub fn external_terminal_permute_dual<const WIDTH: usize>(
    lane0: &mut [u64; WIDTH],
    lane1: &mut [u64; WIDTH],
    constants: &[[u64; WIDTH]],
) {
    for rc in constants {
        unsafe {
            external_round_dual_asm(lane0, lane1, rc);
        }
    }
}

/// W8-specialized dual-lane initial external permute using fused rounds.
#[inline]
pub fn external_initial_permute_dual_w8(
    lane0: &mut [u64; 8],
    lane1: &mut [u64; 8],
    constants: &[[u64; 8]],
) {
    unsafe {
        mds_light_permutation_asm(lane0);
        mds_light_permutation_asm(lane1);
    }
    for rc in constants {
        unsafe {
            external_round_fused_dual_w8(lane0, lane1, rc);
        }
    }
}

/// W8-specialized dual-lane terminal external permute using fused rounds.
#[inline]
pub fn external_terminal_permute_dual_w8(
    lane0: &mut [u64; 8],
    lane1: &mut [u64; 8],
    constants: &[[u64; 8]],
) {
    for rc in constants {
        unsafe {
            external_round_fused_dual_w8(lane0, lane1, rc);
        }
    }
}

// NEON 2-wide Goldilocks field primitives.
// Each operates on both packed lanes simultaneously using uint64x2_t.

#[inline(always)]
unsafe fn add_neon(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
    unsafe {
        let res = vaddq_u64(a, b);
        let overflow = vcgtq_u64(a, res);
        let adj = vshrq_n_u64::<32>(overflow);
        vaddq_u64(res, adj)
    }
}

#[inline(always)]
unsafe fn sub_neon(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
    unsafe {
        let res = vsubq_u64(a, b);
        let underflow = vcgtq_u64(b, a);
        let adj = vshrq_n_u64::<32>(underflow);
        vsubq_u64(res, adj)
    }
}

#[inline(always)]
unsafe fn double_neon(a: uint64x2_t) -> uint64x2_t {
    unsafe { add_neon(a, a) }
}

#[inline(always)]
unsafe fn div2_neon(x: uint64x2_t) -> uint64x2_t {
    unsafe {
        let half_p_plus_1 = vdupq_n_u64((P + 1) >> 1);
        let one = vdupq_n_u64(1);
        let is_odd = vandq_u64(x, one);
        let half = vshrq_n_u64::<1>(x);
        let mask = vtstq_u64(is_odd, is_odd);
        let adj = vandq_u64(mask, half_p_plus_1);
        vaddq_u64(half, adj)
    }
}

#[inline(always)]
unsafe fn div4_neon(x: uint64x2_t) -> uint64x2_t {
    unsafe { div2_neon(div2_neon(x)) }
}

#[inline(always)]
unsafe fn div8_neon(x: uint64x2_t) -> uint64x2_t {
    unsafe { div2_neon(div4_neon(x)) }
}

#[inline(always)]
unsafe fn apply_mat4_neon(x: &mut [uint64x2_t; 4]) {
    unsafe {
        let t01 = add_neon(x[0], x[1]);
        let t23 = add_neon(x[2], x[3]);
        let t0123 = add_neon(t01, t23);
        let t01123 = add_neon(t0123, x[1]);
        let t01233 = add_neon(t0123, x[3]);
        x[3] = add_neon(t01233, double_neon(x[0]));
        x[1] = add_neon(t01123, double_neon(x[2]));
        x[0] = add_neon(t01123, t01);
        x[2] = add_neon(t01233, t23);
    }
}

#[inline(always)]
unsafe fn mds_light_neon<const WIDTH: usize>(state: &mut [uint64x2_t; WIDTH]) {
    unsafe {
        let mut i = 0;
        while i < WIDTH {
            let chunk: &mut [uint64x2_t; 4] = (&mut state[i..i + 4]).try_into().unwrap();
            apply_mat4_neon(chunk);
            i += 4;
        }
        let zero = vdupq_n_u64(0);
        let mut sums = [zero; 4];
        for j in (0..WIDTH).step_by(4) {
            sums[0] = add_neon(sums[0], state[j]);
            sums[1] = add_neon(sums[1], state[j + 1]);
            sums[2] = add_neon(sums[2], state[j + 2]);
            sums[3] = add_neon(sums[3], state[j + 3]);
        }
        for (i, elem) in state.iter_mut().enumerate() {
            *elem = add_neon(*elem, sums[i % 4]);
        }
    }
}

/// Convert separate lane arrays into NEON vector array.
#[inline]
pub fn lanes_to_neon<const WIDTH: usize>(
    lane0: &[u64; WIDTH],
    lane1: &[u64; WIDTH],
) -> [uint64x2_t; WIDTH] {
    core::array::from_fn(|i| unsafe {
        let lo = vcreate_u64(lane0[i]);
        let hi = vcreate_u64(lane1[i]);
        vcombine_u64(lo, hi)
    })
}

/// Convert NEON vector array back to separate lane arrays.
#[inline]
pub fn neon_to_lanes<const WIDTH: usize>(
    state_v: &[uint64x2_t; WIDTH],
    lane0: &mut [u64; WIDTH],
    lane1: &mut [u64; WIDTH],
) {
    for i in 0..WIDTH {
        unsafe {
            lane0[i] = vgetq_lane_u64::<0>(state_v[i]);
            lane1[i] = vgetq_lane_u64::<1>(state_v[i]);
        }
    }
}

// NEON-based internal permutation: both packed lanes processed
// simultaneously via uint64x2_t for sum tree, diagonal, and writeback.

#[inline]
pub fn internal_permute_neon_w12(state: &mut [uint64x2_t; 12], constants: &[u64]) {
    let mut s0 = state[0];
    for &rc in constants {
        unsafe {
            let rc_vec = vdupq_n_u64(rc);
            s0 = add_neon(s0, rc_vec);

            let s0_0 = vgetq_lane_u64::<0>(s0);
            let s0_1 = vgetq_lane_u64::<1>(s0);
            let s0_2_0 = mul_asm(s0_0, s0_0);
            let s0_2_1 = mul_asm(s0_1, s0_1);

            let sum1 = add_neon(state[1], state[2]);
            let sum2 = add_neon(state[3], state[4]);
            let sum3 = add_neon(state[5], state[6]);
            let sum4 = add_neon(state[7], state[8]);
            let sum5 = add_neon(state[9], state[10]);

            let s0_3_0 = mul_asm(s0_2_0, s0_0);
            let s0_3_1 = mul_asm(s0_2_1, s0_1);
            let s0_4_0 = mul_asm(s0_2_0, s0_2_0);
            let s0_4_1 = mul_asm(s0_2_1, s0_2_1);

            let sum12 = add_neon(sum1, sum2);
            let sum34 = add_neon(sum3, sum4);
            let sum511 = add_neon(sum5, state[11]);

            let d1 = state[1];
            let d2 = double_neon(state[2]);
            let d3 = div2_neon(state[3]);
            let d4 = add_neon(double_neon(state[4]), state[4]);

            let sum1234 = add_neon(sum12, sum34);

            let d5 = double_neon(double_neon(state[5]));
            let d6 = div2_neon(state[6]);
            let d7 = add_neon(double_neon(state[7]), state[7]);
            let d8 = double_neon(double_neon(state[8]));

            let sum_hi = add_neon(sum1234, sum511);

            let d9 = div4_neon(state[9]);
            let d10 = div4_neon(state[10]);
            let d11 = div8_neon(state[11]);

            let s0_7_0 = mul_asm(s0_3_0, s0_4_0);
            let s0_7_1 = mul_asm(s0_3_1, s0_4_1);
            let s0_7 = vcombine_u64(vcreate_u64(s0_7_0), vcreate_u64(s0_7_1));

            let sum = add_neon(sum_hi, s0_7);
            s0 = sub_neon(sum_hi, s0_7);

            state[1] = add_neon(d1, sum);
            state[2] = add_neon(d2, sum);
            state[3] = add_neon(d3, sum);
            state[4] = add_neon(d4, sum);
            state[5] = add_neon(d5, sum);
            state[6] = sub_neon(sum, d6);
            state[7] = sub_neon(sum, d7);
            state[8] = sub_neon(sum, d8);
            state[9] = add_neon(d9, sum);
            state[10] = sub_neon(sum, d10);
            state[11] = add_neon(d11, sum);
        }
    }
    state[0] = s0;
}

#[inline]
pub fn internal_permute_neon_w16(
    state: &mut [uint64x2_t; 16],
    diag: &[u64; 16],
    constants: &[u64],
) {
    let mut s0 = state[0];
    for &rc in constants {
        unsafe {
            let rc_vec = vdupq_n_u64(rc);
            s0 = add_neon(s0, rc_vec);

            let s0_0 = vgetq_lane_u64::<0>(s0);
            let s0_1 = vgetq_lane_u64::<1>(s0);
            let s0_2_0 = mul_asm(s0_0, s0_0);
            let s0_2_1 = mul_asm(s0_1, s0_1);

            let sum1 = add_neon(state[1], state[2]);
            let sum2 = add_neon(state[3], state[4]);
            let sum3 = add_neon(state[5], state[6]);
            let sum4 = add_neon(state[7], state[8]);
            let sum5 = add_neon(state[9], state[10]);
            let sum6 = add_neon(state[11], state[12]);
            let sum7 = add_neon(state[13], state[14]);

            let s0_3_0 = mul_asm(s0_2_0, s0_0);
            let s0_3_1 = mul_asm(s0_2_1, s0_1);
            let s0_4_0 = mul_asm(s0_2_0, s0_2_0);
            let s0_4_1 = mul_asm(s0_2_1, s0_2_1);

            let sum12 = add_neon(sum1, sum2);
            let sum34 = add_neon(sum3, sum4);
            let sum56 = add_neon(sum5, sum6);
            let sum715 = add_neon(sum7, state[15]);

            let sum1234 = add_neon(sum12, sum34);
            let sum56715 = add_neon(sum56, sum715);
            let sum_hi = add_neon(sum1234, sum56715);

            let d1 = state[1];
            let d2 = double_neon(state[2]);
            let d3 = div2_neon(state[3]);
            let d4 = add_neon(double_neon(state[4]), state[4]);
            let d5 = double_neon(double_neon(state[5]));
            let d6 = div2_neon(state[6]);
            let d7 = add_neon(double_neon(state[7]), state[7]);
            let d8 = double_neon(double_neon(state[8]));

            let s0_7_0 = mul_asm(s0_3_0, s0_4_0);
            let s0_7_1 = mul_asm(s0_3_1, s0_4_1);
            let s0_7 = vcombine_u64(vcreate_u64(s0_7_0), vcreate_u64(s0_7_1));

            let sum = add_neon(sum_hi, s0_7);
            s0 = sub_neon(sum_hi, s0_7);

            state[1] = add_neon(d1, sum);
            state[2] = add_neon(d2, sum);
            state[3] = add_neon(d3, sum);
            state[4] = add_neon(d4, sum);
            state[5] = add_neon(d5, sum);
            state[6] = sub_neon(sum, d6);
            state[7] = sub_neon(sum, d7);
            state[8] = sub_neon(sum, d8);

            for i in 9..16 {
                let s_0 = mul_add_asm(
                    vgetq_lane_u64::<0>(state[i]),
                    diag[i],
                    vgetq_lane_u64::<0>(sum),
                );
                let s_1 = mul_add_asm(
                    vgetq_lane_u64::<1>(state[i]),
                    diag[i],
                    vgetq_lane_u64::<1>(sum),
                );
                state[i] = vcombine_u64(vcreate_u64(s_0), vcreate_u64(s_1));
            }
        }
    }
    state[0] = s0;
}

#[inline]
pub fn internal_permute_neon<const WIDTH: usize>(
    state: &mut [uint64x2_t; WIDTH],
    diag: &[u64; WIDTH],
    constants: &[u64],
) {
    let mut s0 = state[0];
    for &rc in constants {
        unsafe {
            let rc_vec = vdupq_n_u64(rc);
            s0 = add_neon(s0, rc_vec);

            let s0_0 = vgetq_lane_u64::<0>(s0);
            let s0_1 = vgetq_lane_u64::<1>(s0);
            let s0_2_0 = mul_asm(s0_0, s0_0);
            let s0_2_1 = mul_asm(s0_1, s0_1);
            let s0_3_0 = mul_asm(s0_2_0, s0_0);
            let s0_3_1 = mul_asm(s0_2_1, s0_1);
            let s0_4_0 = mul_asm(s0_2_0, s0_2_0);
            let s0_4_1 = mul_asm(s0_2_1, s0_2_1);
            let s0_7_0 = mul_asm(s0_3_0, s0_4_0);
            let s0_7_1 = mul_asm(s0_3_1, s0_4_1);
            let s0_7 = vcombine_u64(vcreate_u64(s0_7_0), vcreate_u64(s0_7_1));

            let zero = vdupq_n_u64(0);
            let mut sum_hi = zero;
            for &s in state.iter().skip(1) {
                sum_hi = add_neon(sum_hi, s);
            }

            let sum = add_neon(sum_hi, s0_7);
            s0 = vcombine_u64(
                vcreate_u64(mul_add_asm(s0_7_0, diag[0], vgetq_lane_u64::<0>(sum))),
                vcreate_u64(mul_add_asm(s0_7_1, diag[0], vgetq_lane_u64::<1>(sum))),
            );

            for i in 1..WIDTH {
                let s_0 = mul_add_asm(
                    vgetq_lane_u64::<0>(state[i]),
                    diag[i],
                    vgetq_lane_u64::<0>(sum),
                );
                let s_1 = mul_add_asm(
                    vgetq_lane_u64::<1>(state[i]),
                    diag[i],
                    vgetq_lane_u64::<1>(sum),
                );
                state[i] = vcombine_u64(vcreate_u64(s_0), vcreate_u64(s_1));
            }
        }
    }
    state[0] = s0;
}

// NEON-based external round: S-box stays scalar, MDS uses NEON.

#[inline(always)]
unsafe fn sbox_neon<const WIDTH: usize>(state: &mut [uint64x2_t; WIDTH]) {
    unsafe {
        let mut x2_0 = [0u64; WIDTH];
        let mut x2_1 = [0u64; WIDTH];
        for i in 0..WIDTH {
            let a = vgetq_lane_u64::<0>(state[i]);
            let b = vgetq_lane_u64::<1>(state[i]);
            x2_0[i] = mul_asm(a, a);
            x2_1[i] = mul_asm(b, b);
        }
        let mut x3_0 = [0u64; WIDTH];
        let mut x3_1 = [0u64; WIDTH];
        let mut x4_0 = [0u64; WIDTH];
        let mut x4_1 = [0u64; WIDTH];
        for i in 0..WIDTH {
            let a = vgetq_lane_u64::<0>(state[i]);
            let b = vgetq_lane_u64::<1>(state[i]);
            x3_0[i] = mul_asm(x2_0[i], a);
            x3_1[i] = mul_asm(x2_1[i], b);
            x4_0[i] = mul_asm(x2_0[i], x2_0[i]);
            x4_1[i] = mul_asm(x2_1[i], x2_1[i]);
        }
        for i in 0..WIDTH {
            let r0 = mul_asm(x3_0[i], x4_0[i]);
            let r1 = mul_asm(x3_1[i], x4_1[i]);
            state[i] = vcombine_u64(vcreate_u64(r0), vcreate_u64(r1));
        }
    }
}

#[inline(always)]
unsafe fn external_round_neon<const WIDTH: usize>(
    state: &mut [uint64x2_t; WIDTH],
    rc: &[u64; WIDTH],
) {
    unsafe {
        for i in 0..WIDTH {
            let rc_vec = vdupq_n_u64(rc[i]);
            state[i] = add_neon(state[i], rc_vec);
        }
        sbox_neon(state);
        mds_light_neon(state);
    }
}

/// NEON initial external permute.
#[inline]
pub fn external_initial_neon<const WIDTH: usize>(
    state: &mut [uint64x2_t; WIDTH],
    constants: &[[u64; WIDTH]],
) {
    unsafe {
        mds_light_neon(state);
    }
    for rc in constants {
        unsafe {
            external_round_neon(state, rc);
        }
    }
}

/// NEON terminal external permute.
#[inline]
pub fn external_terminal_neon<const WIDTH: usize>(
    state: &mut [uint64x2_t; WIDTH],
    constants: &[[u64; WIDTH]],
) {
    for rc in constants {
        unsafe {
            external_round_neon(state, rc);
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use p3_field::PrimeField64;
    use p3_poseidon2::matmul_internal;

    use super::*;
    use crate::{
        Goldilocks, MATRIX_DIAG_8_GOLDILOCKS, MATRIX_DIAG_12_GOLDILOCKS, MATRIX_DIAG_16_GOLDILOCKS,
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

        let constants_raw: Vec<u64> = internal_constants.iter().map(|c| c.value).collect();
        let diag_raw: [u64; WIDTH] = core::array::from_fn(|i| diag[i].value);

        let state_raw: &mut [u64; WIDTH] =
            unsafe { &mut *(&mut state_asm as *mut [Goldilocks; WIDTH] as *mut [u64; WIDTH]) };
        internal_permute_state_asm(state_raw, &diag_raw, &constants_raw);

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
    fn test_div2_asm_vs_field_halve() {
        use p3_field::PrimeCharacteristicRing;
        use rand::rngs::SmallRng;
        use rand::{RngExt, SeedableRng};

        let mut rng = SmallRng::seed_from_u64(999);

        for _ in 0..256 {
            let x: u64 = rng.random();
            let via_asm = Goldilocks::new(unsafe { div2_asm(x) }).as_canonical_u64();
            let via_field = Goldilocks::new(x).halve().as_canonical_u64();
            assert_eq!(via_asm, via_field, "div2 vs halve mismatch for x={x:#x}");
        }
    }

    #[test]
    fn test_internal_round_w8_unrolled_matches_generic_round() {
        use rand::rngs::SmallRng;
        use rand::{RngExt, SeedableRng};

        let mut rng = SmallRng::seed_from_u64(12345);

        let diag_field = MATRIX_DIAG_8_GOLDILOCKS;
        let diag_raw: [u64; 8] = core::array::from_fn(|i| diag_field[i].value);

        for iter in 0..64 {
            let mut state_generic: [u64; 8] = core::array::from_fn(|_| rng.random());
            let mut state_unrolled = state_generic;
            let rc: u64 = rng.random();

            unsafe {
                internal_round_asm::<8>(&mut state_generic, &diag_raw, rc);
                internal_round_asm_w8(&mut state_unrolled, &diag_raw, rc);
            }

            for i in 0..8 {
                assert_eq!(
                    Goldilocks::new(state_generic[i]).as_canonical_u64(),
                    Goldilocks::new(state_unrolled[i]).as_canonical_u64(),
                    "W8 round mismatch at index {i}, iteration {iter}"
                );
            }
        }
    }

    #[test]
    fn test_internal_round_w12_unrolled_matches_generic_round() {
        use rand::rngs::SmallRng;
        use rand::{RngExt, SeedableRng};

        let mut rng = SmallRng::seed_from_u64(12345);

        let diag_field = MATRIX_DIAG_12_GOLDILOCKS;
        let diag_raw: [u64; 12] = core::array::from_fn(|i| diag_field[i].value);

        for iter in 0..64 {
            let mut state_generic: [u64; 12] = core::array::from_fn(|_| rng.random());
            let mut state_unrolled = state_generic;
            let rc: u64 = rng.random();

            unsafe {
                internal_round_asm::<12>(&mut state_generic, &diag_raw, rc);
                internal_round_asm_w12(&mut state_unrolled, &diag_raw, rc);
            }

            for i in 0..12 {
                assert_eq!(
                    Goldilocks::new(state_generic[i]).as_canonical_u64(),
                    Goldilocks::new(state_unrolled[i]).as_canonical_u64(),
                    "W12 round mismatch at index {i}, iteration {iter}"
                );
            }
        }
    }

    #[test]
    fn test_internal_round_w16_unrolled_matches_generic_round() {
        use rand::rngs::SmallRng;
        use rand::{RngExt, SeedableRng};

        let mut rng = SmallRng::seed_from_u64(12345);

        let diag_field = MATRIX_DIAG_16_GOLDILOCKS;
        let diag_raw: [u64; 16] = core::array::from_fn(|i| diag_field[i].value);

        for iter in 0..64 {
            let mut state_generic: [u64; 16] = core::array::from_fn(|_| rng.random());
            let mut state_unrolled = state_generic;
            let rc: u64 = rng.random();

            unsafe {
                internal_round_asm::<16>(&mut state_generic, &diag_raw, rc);
                internal_round_asm_w16(&mut state_unrolled, &diag_raw, rc);
            }

            for i in 0..16 {
                assert_eq!(
                    Goldilocks::new(state_generic[i]).as_canonical_u64(),
                    Goldilocks::new(state_unrolled[i]).as_canonical_u64(),
                    "W16 round mismatch at index {i}, iteration {iter}"
                );
            }
        }
    }

    #[test]
    #[ignore]
    fn test_div2_asm_matches_mul_by_inv2() {
        use rand::rngs::SmallRng;
        use rand::{RngExt, SeedableRng};

        let mut rng = SmallRng::seed_from_u64(777);
        // MATRIX_DIAG_16_GOLDILOCKS[3] is 1/2 in the Goldilocks field.
        let inv2 = MATRIX_DIAG_16_GOLDILOCKS[3].value;

        for _ in 0..128 {
            let x: u64 = rng.random();
            let via_div2 = unsafe { div2_asm(x) };
            let via_mul = unsafe { mul_asm(x, inv2) };
            assert_eq!(via_div2, via_mul, "div2_asm(x) != x * (1/2) for x={x}");
        }
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
                    "sub" => (Goldilocks::new(a) - Goldilocks::new(b)).as_canonical_u64(),
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
        check_binary_op("sub", |a, b| unsafe { sub_asm(a, b) });
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
        let diag_field = MATRIX_DIAG_8_GOLDILOCKS;
        let diag: [u64; 8] = core::array::from_fn(|i| diag_field[i].value);
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
