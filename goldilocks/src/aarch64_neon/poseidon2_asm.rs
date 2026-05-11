//! ARM assembly primitives for Poseidon2 on Goldilocks.
//!
//! Latency hiding: ARM mul/umulh have ~4-5 cycle latency. By interleaving
//! S-box computation with MDS operations, we hide much of this latency.

use core::arch::aarch64::*;
use core::arch::asm;

use super::utils::{add_asm, mul_add_asm, mul_asm};
use crate::P;

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
            tmp    = out(reg) _tmp,
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

#[inline(always)]
unsafe fn div16_asm(x: u64) -> u64 {
    unsafe { div2_asm(div8_asm(x)) }
}

#[inline(always)]
unsafe fn div32_asm(x: u64) -> u64 {
    unsafe { div4_asm(div8_asm(x)) }
}

/// Compute x * 2^{-32} mod P using the Goldilocks structure.
///
/// Since P = 2^64 - 2^32 + 1, we have 2^{-32} ≡ 1 - 2^{32} (mod P).
/// So x * 2^{-32} = x_hi + x_lo - (x_lo << 32) mod P,
/// where x_hi = x >> 32, x_lo = x & 0xFFFFFFFF.
#[inline(always)]
unsafe fn div_2_32_asm(x: u64) -> u64 {
    let result: u64;
    let _hi: u64;
    let _lo: u64;
    let _t: u64;
    let _sum: u64;
    let _adj: u64;

    unsafe {
        asm!(
            "lsr   {hi}, {x}, #32",
            "and   {lo}, {x}, #0xFFFFFFFF",
            "add   {sum}, {hi}, {lo}",
            "lsl   {t}, {lo}, #32",
            "subs  {result}, {sum}, {t}",
            "csetm {adj:w}, cc",
            "sub   {result}, {result}, {adj}",
            x      = in(reg) x,
            hi     = out(reg) _hi,
            lo     = out(reg) _lo,
            t      = out(reg) _t,
            sum    = out(reg) _sum,
            result = out(reg) result,
            adj    = lateout(reg) _adj,
            options(pure, nomem, nostack),
        );
    }

    result
}

/// Subtract two Goldilocks elements, accepting non-canonical inputs.
#[inline(always)]
unsafe fn sub_asm(a: u64, b: u64) -> u64 {
    let result: u64;
    let _t0: u64;
    let _t1: u64;
    let _adj: u64;

    unsafe {
        asm!(
            // Canonicalize one input: if b >= P, subtract P.
            "subs  {t0}, {b}, {p}",
            "csel  {b_canon}, {t0}, {b}, cs",

            // Subtract, folding 2^64 underflow via EPSILON.
            "subs  {result}, {a}, {b_canon}",
            "csetm {adj:w}, cc",
            "sub   {result}, {result}, {adj}",

            // Final reduction: if result >= P, subtract P.
            "subs  {t1}, {result}, {p}",
            "csel  {result}, {t1}, {result}, cs",
            a = in(reg) a,
            b = in(reg) b,
            b_canon = out(reg) _,
            p = in(reg) P,
            result = out(reg) result,
            t0 = out(reg) _t0,
            t1 = out(reg) _t1,
            adj = out(reg) _adj,
            options(pure, nomem, nostack),
        );
    }

    result
}

/// Split-state generic internal permute: s0 stays in a register across all rounds.
#[inline]
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
            for &s in &state[1..] {
                sum_hi = add_asm(sum_hi, s);
            }

            let mut diag_muls: [u64; WIDTH] = [0; WIDTH];
            for (m, (&s, &d)) in diag_muls[1..]
                .iter_mut()
                .zip(state[1..].iter().zip(&diag[1..]))
            {
                *m = mul_asm(s, d);
            }

            let sum = add_asm(sum_hi, s0);
            s0 = mul_add_asm(s0, diag[0], sum);

            for (s, &m) in state[1..].iter_mut().zip(&diag_muls[1..]) {
                *s = add_asm(m, sum);
            }
        }
    }
    state[0] = s0;
}

/// Split-state generic dual-lane internal permute for packed processing.
#[inline]
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
            for (&a, &b) in lane0[1..].iter().zip(&lane1[1..]) {
                sum_hi_a = add_asm(sum_hi_a, a);
                sum_hi_b = add_asm(sum_hi_b, b);
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

/// Split-state W8 internal permute: s0 stays in a register across all rounds.
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

/// Split-state W16 internal permute: s0 stays in a register across all rounds.
#[inline]
pub fn internal_permute_state_asm_w16(state: &mut [u64; 16], constants: &[u64]) {
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

            let d9 = div8_asm(state[9]);
            let d10 = div16_asm(state[10]);
            let d11 = div32_asm(state[11]);
            let d12 = div8_asm(state[12]);
            let d13 = div16_asm(state[13]);
            let d14 = div32_asm(state[14]);
            let d15 = div_2_32_asm(state[15]);

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
            state[10] = add_asm(d10, sum);
            state[11] = add_asm(d11, sum);
            state[12] = sub_asm(sum, d12);
            state[13] = sub_asm(sum, d13);
            state[14] = sub_asm(sum, d14);
            state[15] = add_asm(d15, sum);
        }
    }
    state[0] = s0;
}

/// Split-state dual-lane W16 internal permute for packed processing.
#[inline]
pub fn internal_permute_split_dual_w16(
    lane0: &mut [u64; 16],
    lane1: &mut [u64; 16],
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

            let d9_a = div8_asm(lane0[9]);
            let d9_b = div8_asm(lane1[9]);
            let d10_a = div16_asm(lane0[10]);
            let d10_b = div16_asm(lane1[10]);
            let d11_a = div32_asm(lane0[11]);
            let d11_b = div32_asm(lane1[11]);
            let d12_a = div8_asm(lane0[12]);
            let d12_b = div8_asm(lane1[12]);
            let d13_a = div16_asm(lane0[13]);
            let d13_b = div16_asm(lane1[13]);
            let d14_a = div32_asm(lane0[14]);
            let d14_b = div32_asm(lane1[14]);
            let d15_a = div_2_32_asm(lane0[15]);
            let d15_b = div_2_32_asm(lane1[15]);

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
            lane0[10] = add_asm(d10_a, sum_a);
            lane1[10] = add_asm(d10_b, sum_b);
            lane0[11] = add_asm(d11_a, sum_a);
            lane1[11] = add_asm(d11_b, sum_b);
            lane0[12] = sub_asm(sum_a, d12_a);
            lane1[12] = sub_asm(sum_b, d12_b);
            lane0[13] = sub_asm(sum_a, d13_a);
            lane1[13] = sub_asm(sum_b, d13_b);
            lane0[14] = sub_asm(sum_a, d14_a);
            lane1[14] = sub_asm(sum_b, d14_b);
            lane0[15] = add_asm(d15_a, sum_a);
            lane1[15] = add_asm(d15_b, sum_b);
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

/// Fully unrolled and fused external round for W8.
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
unsafe fn div16_neon(x: uint64x2_t) -> uint64x2_t {
    unsafe { div2_neon(div8_neon(x)) }
}

#[inline(always)]
unsafe fn div32_neon(x: uint64x2_t) -> uint64x2_t {
    unsafe { div4_neon(div8_neon(x)) }
}

/// Compute x * 2^{-32} mod P for each lane using Goldilocks structure.
///
/// Since P = 2^64 - 2^32 + 1, we have 2^{-32} ≡ 1 - 2^{32} (mod P).
/// So x * 2^{-32} = x_hi + x_lo - (x_lo << 32) mod P.
#[inline(always)]
unsafe fn div_2_32_neon(x: uint64x2_t) -> uint64x2_t {
    unsafe {
        let mask_32 = vdupq_n_u64(0xFFFFFFFF);
        let hi = vshrq_n_u64::<32>(x);
        let lo = vandq_u64(x, mask_32);
        let sum = vaddq_u64(hi, lo);
        let t = vshlq_n_u64::<32>(lo);
        sub_neon(sum, t)
    }
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
pub fn internal_permute_neon_w16(state: &mut [uint64x2_t; 16], constants: &[u64]) {
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

            let d9 = div8_neon(state[9]);
            let d10 = div16_neon(state[10]);
            let d11 = div32_neon(state[11]);
            let d12 = div8_neon(state[12]);
            let d13 = div16_neon(state[13]);
            let d14 = div32_neon(state[14]);
            let d15 = div_2_32_neon(state[15]);

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
            state[10] = add_neon(d10, sum);
            state[11] = add_neon(d11, sum);
            state[12] = sub_neon(sum, d12);
            state[13] = sub_neon(sum, d13);
            state[14] = sub_neon(sum, d14);
            state[15] = add_neon(d15, sum);
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
    use alloc::vec;
    use alloc::vec::Vec;

    use p3_field::{PrimeCharacteristicRing, PrimeField64};
    use p3_poseidon2::{MDSMat4, matmul_internal, mds_light_permutation};
    use proptest::prelude::*;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::*;
    use crate::aarch64_neon::{EDGE_VALUES, danger_array, danger_u64};
    use crate::{
        Goldilocks, MATRIX_DIAG_8_GOLDILOCKS, MATRIX_DIAG_12_GOLDILOCKS, MATRIX_DIAG_16_GOLDILOCKS,
        MATRIX_DIAG_20_GOLDILOCKS,
    };

    type F = Goldilocks;

    /// Reduce a raw u64 to its canonical Goldilocks representative.
    fn canon(x: u64) -> u64 {
        F::new(x).as_canonical_u64()
    }

    /// Pack two u64 lanes into a single NEON vector.
    unsafe fn make_neon(a: u64, b: u64) -> uint64x2_t {
        unsafe { vcombine_u64(vcreate_u64(a), vcreate_u64(b)) }
    }

    /// Extract both u64 lanes from a NEON vector.
    unsafe fn read_neon(v: uint64x2_t) -> (u64, u64) {
        unsafe { (vgetq_lane_u64::<0>(v), vgetq_lane_u64::<1>(v)) }
    }

    // -------------------------------------------------------------------
    // Deterministic edge-value coverage for unary / binary scalar ops.
    // -------------------------------------------------------------------

    #[test]
    fn test_sub_asm_edge_pairs() {
        for &a in EDGE_VALUES {
            for &b in EDGE_VALUES {
                let expected = (F::new(a) - F::new(b)).as_canonical_u64();
                let got = canon(unsafe { sub_asm(a, b) });
                assert_eq!(got, expected, "sub({a}, {b})");
            }
        }
    }

    #[test]
    fn test_double_asm_edge_values() {
        for &a in EDGE_VALUES {
            let expected = (F::new(a) + F::new(a)).as_canonical_u64();
            let got = canon(unsafe { double_asm(a) });
            assert_eq!(got, expected, "double({a})");
        }
    }

    #[test]
    fn test_div2_asm_edge_values() {
        for &a in EDGE_VALUES {
            let expected = F::new(a).halve().as_canonical_u64();
            let got = canon(unsafe { div2_asm(a) });
            assert_eq!(got, expected, "div2({a})");
        }
    }

    #[test]
    fn test_div4_asm_edge_values() {
        for &a in EDGE_VALUES {
            let expected = F::new(a).halve().halve().as_canonical_u64();
            let got = canon(unsafe { div4_asm(a) });
            assert_eq!(got, expected, "div4({a})");
        }
    }

    #[test]
    fn test_div8_asm_edge_values() {
        for &a in EDGE_VALUES {
            let expected = F::new(a).halve().halve().halve().as_canonical_u64();
            let got = canon(unsafe { div8_asm(a) });
            assert_eq!(got, expected, "div8({a})");
        }
    }

    #[test]
    fn test_div16_asm_edge_values() {
        for &a in EDGE_VALUES {
            let expected = F::new(a).halve().halve().halve().halve().as_canonical_u64();
            let got = canon(unsafe { div16_asm(a) });
            assert_eq!(got, expected, "div16({a})");
        }
    }

    #[test]
    fn test_div32_asm_edge_values() {
        for &a in EDGE_VALUES {
            let expected = F::new(a)
                .halve()
                .halve()
                .halve()
                .halve()
                .halve()
                .as_canonical_u64();
            let got = canon(unsafe { div32_asm(a) });
            assert_eq!(got, expected, "div32({a})");
        }
    }

    #[test]
    fn test_div_2_32_asm_edge_values() {
        for &a in EDGE_VALUES {
            let mut v = F::new(a);
            for _ in 0..32 {
                v = v.halve();
            }
            let expected = v.as_canonical_u64();
            let got = canon(unsafe { div_2_32_asm(a) });
            assert_eq!(got, expected, "div_2_32({a})");
        }
    }

    proptest! {
        #[test]
        fn test_sub_asm(a: u64, b: u64) {
            // Compute a - b using the standard field implementation.
            let expected = (F::new(a) - F::new(b)).as_canonical_u64();

            // The ASM version should give the same canonical result.
            let got = canon(unsafe { sub_asm(a, b) });
            prop_assert_eq!(got, expected);
        }

        #[test]
        fn test_double_asm(a: u64) {
            // Doubling is just a + a in the field.
            let expected = (F::new(a) + F::new(a)).as_canonical_u64();

            // The ASM shortcut should match.
            let got = canon(unsafe { double_asm(a) });
            prop_assert_eq!(got, expected);
        }

        #[test]
        fn test_div2_asm(x: u64) {
            // Dividing by 2 is one halving in the field.
            let expected = F::new(x).halve().as_canonical_u64();

            let got = canon(unsafe { div2_asm(x) });
            prop_assert_eq!(got, expected);
        }

        #[test]
        fn test_div4_asm(x: u64) {
            // Dividing by 4 is two halvings.
            let expected = F::new(x).halve().halve().as_canonical_u64();

            let got = canon(unsafe { div4_asm(x) });
            prop_assert_eq!(got, expected);
        }

        #[test]
        fn test_div8_asm(x: u64) {
            // Dividing by 8 is three halvings.
            let expected = F::new(x).halve().halve().halve().as_canonical_u64();

            let got = canon(unsafe { div8_asm(x) });
            prop_assert_eq!(got, expected);
        }

        #[test]
        fn test_div16_asm(x: u64) {
            // Dividing by 16 is four halvings.
            let expected = F::new(x).halve().halve().halve().halve().as_canonical_u64();

            let got = canon(unsafe { div16_asm(x) });
            prop_assert_eq!(got, expected);
        }

        #[test]
        fn test_div32_asm(x: u64) {
            // Dividing by 32 is five halvings.
            let expected = F::new(x)
                .halve().halve().halve().halve().halve()
                .as_canonical_u64();

            let got = canon(unsafe { div32_asm(x) });
            prop_assert_eq!(got, expected);
        }

        #[test]
        fn test_div_2_32_asm(x: u64) {
            // Dividing by 2^32: apply halve 32 times as reference.
            let mut v = F::new(x);
            for _ in 0..32 {
                v = v.halve();
            }
            let expected = v.as_canonical_u64();

            let got = canon(unsafe { div_2_32_asm(x) });
            prop_assert_eq!(got, expected);
        }

        #[test]
        fn test_apply_mat4_asm(x0: u64, x1: u64, x2: u64, x3: u64) {
            // Build field elements from the raw inputs.
            let f = [F::new(x0), F::new(x1), F::new(x2), F::new(x3)];

            // The [2,3,1,1] circulant matrix rows.
            let two = F::TWO;
            let three = two + F::ONE;
            let e0 = two * f[0] + three * f[1] + f[2] + f[3];
            let e1 = f[0] + two * f[1] + three * f[2] + f[3];
            let e2 = f[0] + f[1] + two * f[2] + three * f[3];
            let e3 = three * f[0] + f[1] + f[2] + two * f[3];

            // Run the ASM version on raw u64s.
            let mut state = [x0, x1, x2, x3];
            unsafe { apply_mat4_asm(&mut state); }

            // Each slot must match the field-level reference.
            prop_assert_eq!(canon(state[0]), e0.as_canonical_u64());
            prop_assert_eq!(canon(state[1]), e1.as_canonical_u64());
            prop_assert_eq!(canon(state[2]), e2.as_canonical_u64());
            prop_assert_eq!(canon(state[3]), e3.as_canonical_u64());
        }

        #[test]
        fn test_mds_light_permutation_asm_w8(vals in prop::array::uniform8(any::<u64>())) {
            // Build field-level state and apply the generic MDS.
            let mut state_generic: [F; 8] = vals.map(F::new);
            mds_light_permutation(&mut state_generic, &MDSMat4);

            // Run the ASM version on the same raw values.
            let mut state_asm = vals;
            unsafe { mds_light_permutation_asm(&mut state_asm); }

            // Every element must agree.
            for i in 0..8 {
                prop_assert_eq!(canon(state_asm[i]), state_generic[i].as_canonical_u64());
            }
        }

        #[test]
        fn test_mds_light_permutation_asm_w12(vals in prop::array::uniform12(any::<u64>())) {
            let mut state_generic: [F; 12] = vals.map(F::new);
            mds_light_permutation(&mut state_generic, &MDSMat4);

            let mut state_asm = vals;
            unsafe { mds_light_permutation_asm(&mut state_asm); }

            for i in 0..12 {
                prop_assert_eq!(canon(state_asm[i]), state_generic[i].as_canonical_u64());
            }
        }

        #[test]
        fn test_mds_light_permutation_asm_w16(vals in prop::array::uniform16(any::<u64>())) {
            let mut state_generic: [F; 16] = vals.map(F::new);
            mds_light_permutation(&mut state_generic, &MDSMat4);

            let mut state_asm = vals;
            unsafe { mds_light_permutation_asm(&mut state_asm); }

            for i in 0..16 {
                prop_assert_eq!(canon(state_asm[i]), state_generic[i].as_canonical_u64());
            }
        }

        #[test]
        fn test_sbox_layer_asm(vals in prop::array::uniform8(any::<u64>())) {
            // Apply the ASM S-box to a copy of the input.
            let mut state = vals;
            unsafe { sbox_layer_asm(&mut state); }

            // Verify each element is x^7 = x^3 * x^4.
            for i in 0..8 {
                let x = F::new(vals[i]);
                let x2 = x * x;
                let x3 = x2 * x;
                let x4 = x2 * x2;
                let x7 = x3 * x4;
                prop_assert_eq!(canon(state[i]), x7.as_canonical_u64());
            }
        }

        #[test]
        fn test_external_round_asm(
            vals in prop::array::uniform8(any::<u64>()),
            rc in prop::array::uniform8(any::<u64>()),
        ) {
            // Build reference: add round constants, apply x^7, then MDS.
            let mut expected: [F; 8] = core::array::from_fn(|i| F::new(vals[i]) + F::new(rc[i]));
            for x in expected.iter_mut() {
                let x2 = *x * *x;
                let x3 = x2 * *x;
                let x4 = x2 * x2;
                *x = x3 * x4;
            }
            mds_light_permutation(&mut expected, &MDSMat4);

            // Run the ASM external round.
            let mut state = vals;
            unsafe { external_round_asm(&mut state, &rc); }

            for i in 0..8 {
                prop_assert_eq!(canon(state[i]), expected[i].as_canonical_u64());
            }
        }

        #[test]
        fn test_sbox_layer_dual_asm(
            vals0 in prop::array::uniform8(any::<u64>()),
            vals1 in prop::array::uniform8(any::<u64>()),
        ) {
            // Run sbox on each lane independently as reference.
            let mut ref0 = vals0;
            let mut ref1 = vals1;
            unsafe {
                sbox_layer_asm(&mut ref0);
                sbox_layer_asm(&mut ref1);
            }

            // The dual-lane version processes both at once.
            let mut s0 = vals0;
            let mut s1 = vals1;
            unsafe { sbox_layer_dual_asm(&mut s0, &mut s1); }

            // Both lanes must match their single-lane reference.
            for i in 0..8 {
                prop_assert_eq!(canon(s0[i]), canon(ref0[i]));
                prop_assert_eq!(canon(s1[i]), canon(ref1[i]));
            }
        }

        #[test]
        fn test_external_round_dual_asm(
            vals0 in prop::array::uniform8(any::<u64>()),
            vals1 in prop::array::uniform8(any::<u64>()),
            rc in prop::array::uniform8(any::<u64>()),
        ) {
            // Run external round on each lane independently as reference.
            let mut ref0 = vals0;
            let mut ref1 = vals1;
            unsafe {
                external_round_asm(&mut ref0, &rc);
                external_round_asm(&mut ref1, &rc);
            }

            // The dual-lane version processes both at once.
            let mut s0 = vals0;
            let mut s1 = vals1;
            unsafe { external_round_dual_asm(&mut s0, &mut s1, &rc); }

            for i in 0..8 {
                prop_assert_eq!(canon(s0[i]), canon(ref0[i]));
                prop_assert_eq!(canon(s1[i]), canon(ref1[i]));
            }
        }

        #[test]
        fn test_external_round_fused_w8(
            vals in prop::array::uniform8(any::<u64>()),
            rc in prop::array::uniform8(any::<u64>()),
        ) {
            // The generic external round is the reference.
            let mut ref_state = vals;
            unsafe { external_round_asm(&mut ref_state, &rc); }

            // The fused W8 version should produce the same output.
            let mut fused_state = vals;
            unsafe { external_round_fused_w8(&mut fused_state, &rc); }

            for i in 0..8 {
                prop_assert_eq!(canon(fused_state[i]), canon(ref_state[i]));
            }
        }

        #[test]
        fn test_external_round_fused_dual_w8(
            vals0 in prop::array::uniform8(any::<u64>()),
            vals1 in prop::array::uniform8(any::<u64>()),
            rc in prop::array::uniform8(any::<u64>()),
        ) {
            // Run the fused round on each lane independently as reference.
            let mut ref0 = vals0;
            let mut ref1 = vals1;
            unsafe {
                external_round_fused_w8(&mut ref0, &rc);
                external_round_fused_w8(&mut ref1, &rc);
            }

            // The dual version processes both at once.
            let mut s0 = vals0;
            let mut s1 = vals1;
            unsafe { external_round_fused_dual_w8(&mut s0, &mut s1, &rc); }

            for i in 0..8 {
                prop_assert_eq!(canon(s0[i]), canon(ref0[i]));
                prop_assert_eq!(canon(s1[i]), canon(ref1[i]));
            }
        }
    }

    fn test_internal_round_matches<const WIDTH: usize>(diag: [F; WIDTH]) {
        let mut rng = SmallRng::seed_from_u64(12345);

        // Build random state and constants.
        let mut state_asm: [F; WIDTH] = rng.random();
        let mut state_generic = state_asm;

        let internal_constants: [F; 22] = rng.random();
        let constants_raw: Vec<u64> = internal_constants.iter().map(|c| c.value).collect();
        let diag_raw: [u64; WIDTH] = core::array::from_fn(|i| diag[i].value);

        // Run the ASM internal permute on raw u64 representation.
        let state_raw: &mut [u64; WIDTH] =
            unsafe { &mut *(&mut state_asm as *mut [F; WIDTH] as *mut [u64; WIDTH]) };
        internal_permute_state_asm(state_raw, &diag_raw, &constants_raw);

        // Build the same result via field-level ops: add RC, S-box on s0, matmul.
        for &rc in internal_constants.iter() {
            state_generic[0] += rc;
            let s = state_generic[0];
            let s2 = s * s;
            let s3 = s2 * s;
            let s4 = s2 * s2;
            state_generic[0] = s3 * s4;
            matmul_internal(&mut state_generic, diag);
        }

        for i in 0..WIDTH {
            assert_eq!(
                state_asm[i].as_canonical_u64(),
                state_generic[i].as_canonical_u64(),
                "mismatch at index {i}"
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

    fn test_specialized_matches_generic<const WIDTH: usize>(
        diag: [F; WIDTH],
        specialized_fn: fn(&mut [u64; WIDTH], &[u64]),
    ) {
        let mut rng = SmallRng::seed_from_u64(42);

        let internal_constants: Vec<u64> = (0..22).map(|_| rng.random()).collect();
        let diag_raw: [u64; WIDTH] = core::array::from_fn(|i| diag[i].value);

        // Run both the specialized and generic versions on several random states.
        for _ in 0..8 {
            let mut state_specialized: [u64; WIDTH] = rng.random();
            let mut state_generic = state_specialized;

            specialized_fn(&mut state_specialized, &internal_constants);
            internal_permute_state_asm(&mut state_generic, &diag_raw, &internal_constants);

            for i in 0..WIDTH {
                assert_eq!(canon(state_specialized[i]), canon(state_generic[i]));
            }
        }
    }

    #[test]
    fn test_specialized_w8_matches_generic() {
        test_specialized_matches_generic(MATRIX_DIAG_8_GOLDILOCKS, internal_permute_state_asm_w8);
    }

    #[test]
    fn test_specialized_w12_matches_generic() {
        test_specialized_matches_generic(MATRIX_DIAG_12_GOLDILOCKS, internal_permute_state_asm_w12);
    }

    #[test]
    fn test_specialized_w16_matches_generic() {
        test_specialized_matches_generic(MATRIX_DIAG_16_GOLDILOCKS, internal_permute_state_asm_w16);
    }

    #[allow(clippy::type_complexity)]
    fn test_dual_matches_single<const WIDTH: usize>(
        diag: [F; WIDTH],
        single_fn: fn(&mut [u64; WIDTH], &[u64; WIDTH], &[u64]),
        dual_fn: fn(&mut [u64; WIDTH], &mut [u64; WIDTH], &[u64; WIDTH], &[u64]),
    ) {
        let mut rng = SmallRng::seed_from_u64(77);

        let diag_raw: [u64; WIDTH] = core::array::from_fn(|i| diag[i].value);
        let constants: Vec<u64> = (0..22).map(|_| rng.random()).collect();

        // Run single-lane on each lane independently.
        let mut lane0: [u64; WIDTH] = rng.random();
        let mut lane1: [u64; WIDTH] = rng.random();
        let mut ref0 = lane0;
        let mut ref1 = lane1;

        single_fn(&mut ref0, &diag_raw, &constants);
        single_fn(&mut ref1, &diag_raw, &constants);

        // Run dual-lane on both at once. Must match.
        dual_fn(&mut lane0, &mut lane1, &diag_raw, &constants);

        for i in 0..WIDTH {
            assert_eq!(canon(lane0[i]), canon(ref0[i]), "lane0 mismatch at {i}");
            assert_eq!(canon(lane1[i]), canon(ref1[i]), "lane1 mismatch at {i}");
        }
    }

    #[test]
    fn test_internal_permute_split_dual_w8() {
        test_dual_matches_single(
            MATRIX_DIAG_8_GOLDILOCKS,
            internal_permute_state_asm,
            internal_permute_split_dual,
        );
    }

    #[test]
    fn test_internal_permute_split_dual_w12() {
        test_dual_matches_single(
            MATRIX_DIAG_12_GOLDILOCKS,
            internal_permute_state_asm,
            internal_permute_split_dual,
        );
    }

    #[test]
    fn test_internal_permute_split_dual_w16() {
        test_dual_matches_single(
            MATRIX_DIAG_16_GOLDILOCKS,
            internal_permute_state_asm,
            internal_permute_split_dual,
        );
    }

    fn test_specialized_dual_matches_generic_dual<const WIDTH: usize>(
        diag: [F; WIDTH],
        specialized_dual_fn: fn(&mut [u64; WIDTH], &mut [u64; WIDTH], &[u64]),
    ) {
        let mut rng = SmallRng::seed_from_u64(99);

        let diag_raw: [u64; WIDTH] = core::array::from_fn(|i| diag[i].value);
        let constants: Vec<u64> = (0..22).map(|_| rng.random()).collect();

        // The generic dual-lane version is the reference.
        let mut lane0: [u64; WIDTH] = rng.random();
        let mut lane1: [u64; WIDTH] = rng.random();
        let mut ref0 = lane0;
        let mut ref1 = lane1;

        internal_permute_split_dual(&mut ref0, &mut ref1, &diag_raw, &constants);

        // The specialized version must match.
        specialized_dual_fn(&mut lane0, &mut lane1, &constants);

        for i in 0..WIDTH {
            assert_eq!(canon(lane0[i]), canon(ref0[i]), "lane0 mismatch at {i}");
            assert_eq!(canon(lane1[i]), canon(ref1[i]), "lane1 mismatch at {i}");
        }
    }

    #[test]
    fn test_specialized_dual_w8_matches_generic() {
        test_specialized_dual_matches_generic_dual(
            MATRIX_DIAG_8_GOLDILOCKS,
            internal_permute_split_dual_w8,
        );
    }

    #[test]
    fn test_specialized_dual_w12_matches_generic() {
        test_specialized_dual_matches_generic_dual(
            MATRIX_DIAG_12_GOLDILOCKS,
            internal_permute_split_dual_w12,
        );
    }

    #[test]
    fn test_specialized_dual_w16_matches_generic() {
        test_specialized_dual_matches_generic_dual(
            MATRIX_DIAG_16_GOLDILOCKS,
            internal_permute_split_dual_w16,
        );
    }

    fn make_round_constants<const WIDTH: usize>(seed: u64, num_rounds: usize) -> Vec<[u64; WIDTH]> {
        let mut rng = SmallRng::seed_from_u64(seed);
        (0..num_rounds).map(|_| rng.random()).collect()
    }

    proptest! {
        #[test]
        fn test_external_initial_permute_state_asm(
            vals in prop::array::uniform8(any::<u64>()),
        ) {
            let constants = make_round_constants::<8>(42, 4);

            // Reference: apply MDS once, then each external round manually.
            let mut expected = vals;
            unsafe { mds_light_permutation_asm(&mut expected); }
            for rc in &constants {
                unsafe { external_round_asm(&mut expected, rc); }
            }

            // The composed function should give the same result.
            let mut got = vals;
            external_initial_permute_state_asm(&mut got, &constants);

            for i in 0..8 {
                prop_assert_eq!(canon(got[i]), canon(expected[i]));
            }
        }

        #[test]
        fn test_external_terminal_permute_state_asm(
            vals in prop::array::uniform8(any::<u64>()),
        ) {
            let constants = make_round_constants::<8>(43, 4);

            // Reference: just the external rounds, no initial MDS.
            let mut expected = vals;
            for rc in &constants {
                unsafe { external_round_asm(&mut expected, rc); }
            }

            let mut got = vals;
            external_terminal_permute_state_asm(&mut got, &constants);

            for i in 0..8 {
                prop_assert_eq!(canon(got[i]), canon(expected[i]));
            }
        }

        #[test]
        fn test_external_initial_permute_w8(
            vals in prop::array::uniform8(any::<u64>()),
        ) {
            let constants = make_round_constants::<8>(44, 4);

            // The generic version is the reference.
            let mut expected = vals;
            external_initial_permute_state_asm(&mut expected, &constants);

            // The W8-specialized version must match.
            let mut got = vals;
            external_initial_permute_w8(&mut got, &constants);

            for i in 0..8 {
                prop_assert_eq!(canon(got[i]), canon(expected[i]));
            }
        }

        #[test]
        fn test_external_terminal_permute_w8(
            vals in prop::array::uniform8(any::<u64>()),
        ) {
            let constants = make_round_constants::<8>(45, 4);

            let mut expected = vals;
            external_terminal_permute_state_asm(&mut expected, &constants);

            let mut got = vals;
            external_terminal_permute_w8(&mut got, &constants);

            for i in 0..8 {
                prop_assert_eq!(canon(got[i]), canon(expected[i]));
            }
        }

        #[test]
        fn test_external_initial_permute_dual(
            vals0 in prop::array::uniform8(any::<u64>()),
            vals1 in prop::array::uniform8(any::<u64>()),
        ) {
            let constants = make_round_constants::<8>(46, 4);

            // Run single-lane on each independently as reference.
            let mut ref0 = vals0;
            let mut ref1 = vals1;
            external_initial_permute_state_asm(&mut ref0, &constants);
            external_initial_permute_state_asm(&mut ref1, &constants);

            // The dual version processes both at once.
            let mut l0 = vals0;
            let mut l1 = vals1;
            external_initial_permute_dual(&mut l0, &mut l1, &constants);

            for i in 0..8 {
                prop_assert_eq!(canon(l0[i]), canon(ref0[i]));
                prop_assert_eq!(canon(l1[i]), canon(ref1[i]));
            }
        }

        #[test]
        fn test_external_terminal_permute_dual(
            vals0 in prop::array::uniform8(any::<u64>()),
            vals1 in prop::array::uniform8(any::<u64>()),
        ) {
            let constants = make_round_constants::<8>(47, 4);

            let mut ref0 = vals0;
            let mut ref1 = vals1;
            external_terminal_permute_state_asm(&mut ref0, &constants);
            external_terminal_permute_state_asm(&mut ref1, &constants);

            let mut l0 = vals0;
            let mut l1 = vals1;
            external_terminal_permute_dual(&mut l0, &mut l1, &constants);

            for i in 0..8 {
                prop_assert_eq!(canon(l0[i]), canon(ref0[i]));
                prop_assert_eq!(canon(l1[i]), canon(ref1[i]));
            }
        }

        #[test]
        fn test_external_initial_permute_dual_w8(
            vals0 in prop::array::uniform8(any::<u64>()),
            vals1 in prop::array::uniform8(any::<u64>()),
        ) {
            let constants = make_round_constants::<8>(48, 4);

            // The generic dual version is the reference.
            let mut ref0 = vals0;
            let mut ref1 = vals1;
            external_initial_permute_dual(&mut ref0, &mut ref1, &constants);

            // The W8-specialized dual must match.
            let mut l0 = vals0;
            let mut l1 = vals1;
            external_initial_permute_dual_w8(&mut l0, &mut l1, &constants);

            for i in 0..8 {
                prop_assert_eq!(canon(l0[i]), canon(ref0[i]));
                prop_assert_eq!(canon(l1[i]), canon(ref1[i]));
            }
        }

        #[test]
        fn test_external_terminal_permute_dual_w8(
            vals0 in prop::array::uniform8(any::<u64>()),
            vals1 in prop::array::uniform8(any::<u64>()),
        ) {
            let constants = make_round_constants::<8>(49, 4);

            let mut ref0 = vals0;
            let mut ref1 = vals1;
            external_terminal_permute_dual(&mut ref0, &mut ref1, &constants);

            let mut l0 = vals0;
            let mut l1 = vals1;
            external_terminal_permute_dual_w8(&mut l0, &mut l1, &constants);

            for i in 0..8 {
                prop_assert_eq!(canon(l0[i]), canon(ref0[i]));
                prop_assert_eq!(canon(l1[i]), canon(ref1[i]));
            }
        }

        #[test]
        fn test_add_neon(a0: u64, a1: u64, b0: u64, b1: u64) {
            unsafe {
                // Pack two lanes into NEON vectors, add, then read back.
                let (r0, r1) = read_neon(add_neon(make_neon(a0, a1), make_neon(b0, b1)));

                // Each lane must match its scalar add_asm equivalent.
                prop_assert_eq!(canon(r0), canon(add_asm(a0, b0)));
                prop_assert_eq!(canon(r1), canon(add_asm(a1, b1)));
            }
        }

        #[test]
        fn test_sub_neon(a0: u64, a1: u64, b0: u64, b1: u64) {
            unsafe {
                let (r0, r1) = read_neon(sub_neon(make_neon(a0, a1), make_neon(b0, b1)));

                prop_assert_eq!(canon(r0), canon(sub_asm(a0, b0)));
                prop_assert_eq!(canon(r1), canon(sub_asm(a1, b1)));
            }
        }

        #[test]
        fn test_double_neon(a0: u64, a1: u64) {
            unsafe {
                let (r0, r1) = read_neon(double_neon(make_neon(a0, a1)));

                prop_assert_eq!(canon(r0), canon(double_asm(a0)));
                prop_assert_eq!(canon(r1), canon(double_asm(a1)));
            }
        }

        #[test]
        fn test_div2_neon(a0: u64, a1: u64) {
            unsafe {
                let (r0, r1) = read_neon(div2_neon(make_neon(a0, a1)));

                prop_assert_eq!(canon(r0), canon(div2_asm(a0)));
                prop_assert_eq!(canon(r1), canon(div2_asm(a1)));
            }
        }

        #[test]
        fn test_div4_neon(a0: u64, a1: u64) {
            unsafe {
                let (r0, r1) = read_neon(div4_neon(make_neon(a0, a1)));

                prop_assert_eq!(canon(r0), canon(div4_asm(a0)));
                prop_assert_eq!(canon(r1), canon(div4_asm(a1)));
            }
        }

        #[test]
        fn test_div8_neon(a0: u64, a1: u64) {
            unsafe {
                let (r0, r1) = read_neon(div8_neon(make_neon(a0, a1)));

                prop_assert_eq!(canon(r0), canon(div8_asm(a0)));
                prop_assert_eq!(canon(r1), canon(div8_asm(a1)));
            }
        }

        #[test]
        fn test_div16_neon(a0: u64, a1: u64) {
            unsafe {
                let (r0, r1) = read_neon(div16_neon(make_neon(a0, a1)));

                prop_assert_eq!(canon(r0), canon(div16_asm(a0)));
                prop_assert_eq!(canon(r1), canon(div16_asm(a1)));
            }
        }

        #[test]
        fn test_div32_neon(a0: u64, a1: u64) {
            unsafe {
                let (r0, r1) = read_neon(div32_neon(make_neon(a0, a1)));

                prop_assert_eq!(canon(r0), canon(div32_asm(a0)));
                prop_assert_eq!(canon(r1), canon(div32_asm(a1)));
            }
        }

        #[test]
        fn test_div_2_32_neon(a0: u64, a1: u64) {
            unsafe {
                let (r0, r1) = read_neon(div_2_32_neon(make_neon(a0, a1)));

                prop_assert_eq!(canon(r0), canon(div_2_32_asm(a0)));
                prop_assert_eq!(canon(r1), canon(div_2_32_asm(a1)));
            }
        }

        #[test]
        fn test_apply_mat4_neon(
            a0: u64, a1: u64, a2: u64, a3: u64,
            b0: u64, b1: u64, b2: u64, b3: u64,
        ) {
            unsafe {
                // Scalar reference: run apply_mat4_asm on each lane separately.
                let mut lane_a = [a0, a1, a2, a3];
                let mut lane_b = [b0, b1, b2, b3];
                apply_mat4_asm(&mut lane_a);
                apply_mat4_asm(&mut lane_b);

                // NEON version: pack both lanes into vectors, apply, read back.
                let mut neon_state = [
                    make_neon(a0, b0),
                    make_neon(a1, b1),
                    make_neon(a2, b2),
                    make_neon(a3, b3),
                ];
                apply_mat4_neon(&mut neon_state);

                for i in 0..4 {
                    let (r0, r1) = read_neon(neon_state[i]);
                    prop_assert_eq!(canon(r0), canon(lane_a[i]));
                    prop_assert_eq!(canon(r1), canon(lane_b[i]));
                }
            }
        }

        #[test]
        fn test_mds_light_neon_w8(
            lane_a in prop::array::uniform8(any::<u64>()),
            lane_b in prop::array::uniform8(any::<u64>()),
        ) {
            unsafe {
                // Run scalar MDS on each lane independently.
                let mut ref_a = lane_a;
                let mut ref_b = lane_b;
                mds_light_permutation_asm(&mut ref_a);
                mds_light_permutation_asm(&mut ref_b);

                // Pack both lanes into NEON vectors and run the NEON MDS.
                let mut neon_state: [uint64x2_t; 8] =
                    core::array::from_fn(|i| make_neon(lane_a[i], lane_b[i]));
                mds_light_neon(&mut neon_state);

                // Each lane of each vector must match the scalar reference.
                for i in 0..8 {
                    let (r0, r1) = read_neon(neon_state[i]);
                    prop_assert_eq!(canon(r0), canon(ref_a[i]));
                    prop_assert_eq!(canon(r1), canon(ref_b[i]));
                }
            }
        }

        #[test]
        fn test_sbox_neon(
            lane_a in prop::array::uniform8(any::<u64>()),
            lane_b in prop::array::uniform8(any::<u64>()),
        ) {
            unsafe {
                // Scalar reference on each lane.
                let mut ref_a = lane_a;
                let mut ref_b = lane_b;
                sbox_layer_asm(&mut ref_a);
                sbox_layer_asm(&mut ref_b);

                // NEON version on packed lanes.
                let mut neon_state: [uint64x2_t; 8] =
                    core::array::from_fn(|i| make_neon(lane_a[i], lane_b[i]));
                sbox_neon(&mut neon_state);

                for i in 0..8 {
                    let (r0, r1) = read_neon(neon_state[i]);
                    prop_assert_eq!(canon(r0), canon(ref_a[i]));
                    prop_assert_eq!(canon(r1), canon(ref_b[i]));
                }
            }
        }

        #[test]
        fn test_external_round_neon(
            lane_a in prop::array::uniform8(any::<u64>()),
            lane_b in prop::array::uniform8(any::<u64>()),
            rc in prop::array::uniform8(any::<u64>()),
        ) {
            unsafe {
                // Scalar reference on each lane.
                let mut ref_a = lane_a;
                let mut ref_b = lane_b;
                external_round_asm(&mut ref_a, &rc);
                external_round_asm(&mut ref_b, &rc);

                // NEON version on packed lanes.
                let mut neon_state: [uint64x2_t; 8] =
                    core::array::from_fn(|i| make_neon(lane_a[i], lane_b[i]));
                external_round_neon(&mut neon_state, &rc);

                for i in 0..8 {
                    let (r0, r1) = read_neon(neon_state[i]);
                    prop_assert_eq!(canon(r0), canon(ref_a[i]));
                    prop_assert_eq!(canon(r1), canon(ref_b[i]));
                }
            }
        }

        #[test]
        fn test_lanes_roundtrip(
            lane0 in prop::array::uniform8(any::<u64>()),
            lane1 in prop::array::uniform8(any::<u64>()),
        ) {
            // Pack two lane arrays into NEON vectors.
            let packed = lanes_to_neon(&lane0, &lane1);

            // Unpack back into separate arrays.
            let mut out0 = [0u64; 8];
            let mut out1 = [0u64; 8];
            neon_to_lanes(&packed, &mut out0, &mut out1);

            // Must recover the original values.
            prop_assert_eq!(out0, lane0);
            prop_assert_eq!(out1, lane1);
        }

        #[test]
        fn test_external_initial_neon(
            lane_a in prop::array::uniform8(any::<u64>()),
            lane_b in prop::array::uniform8(any::<u64>()),
        ) {
            let constants = make_round_constants::<8>(50, 4);

            // Scalar reference on each lane.
            let mut ref_a = lane_a;
            let mut ref_b = lane_b;
            external_initial_permute_state_asm(&mut ref_a, &constants);
            external_initial_permute_state_asm(&mut ref_b, &constants);

            // NEON version on packed lanes.
            let mut neon_state = lanes_to_neon(&lane_a, &lane_b);
            external_initial_neon(&mut neon_state, &constants);

            let mut out_a = [0u64; 8];
            let mut out_b = [0u64; 8];
            neon_to_lanes(&neon_state, &mut out_a, &mut out_b);

            for i in 0..8 {
                prop_assert_eq!(canon(out_a[i]), canon(ref_a[i]));
                prop_assert_eq!(canon(out_b[i]), canon(ref_b[i]));
            }
        }

        #[test]
        fn test_external_terminal_neon(
            lane_a in prop::array::uniform8(any::<u64>()),
            lane_b in prop::array::uniform8(any::<u64>()),
        ) {
            let constants = make_round_constants::<8>(51, 4);

            let mut ref_a = lane_a;
            let mut ref_b = lane_b;
            external_terminal_permute_state_asm(&mut ref_a, &constants);
            external_terminal_permute_state_asm(&mut ref_b, &constants);

            let mut neon_state = lanes_to_neon(&lane_a, &lane_b);
            external_terminal_neon(&mut neon_state, &constants);

            let mut out_a = [0u64; 8];
            let mut out_b = [0u64; 8];
            neon_to_lanes(&neon_state, &mut out_a, &mut out_b);

            for i in 0..8 {
                prop_assert_eq!(canon(out_a[i]), canon(ref_a[i]));
                prop_assert_eq!(canon(out_b[i]), canon(ref_b[i]));
            }
        }
    }

    fn test_internal_neon_matches_scalar<const WIDTH: usize>(
        diag: [F; WIDTH],
        neon_fn: fn(&mut [uint64x2_t; WIDTH], &[u64]),
        scalar_fn: fn(&mut [u64; WIDTH], &[u64; WIDTH], &[u64]),
    ) {
        let mut rng = SmallRng::seed_from_u64(55);

        let diag_raw: [u64; WIDTH] = core::array::from_fn(|i| diag[i].value);
        let constants: Vec<u64> = (0..22).map(|_| rng.random()).collect();

        let lane_a: [u64; WIDTH] = rng.random();
        let lane_b: [u64; WIDTH] = rng.random();

        // Scalar reference on each lane independently.
        let mut ref_a = lane_a;
        let mut ref_b = lane_b;
        scalar_fn(&mut ref_a, &diag_raw, &constants);
        scalar_fn(&mut ref_b, &diag_raw, &constants);

        // NEON version packs both lanes and processes them together.
        let mut neon_state = lanes_to_neon(&lane_a, &lane_b);
        neon_fn(&mut neon_state, &constants);

        let mut out_a = [0u64; WIDTH];
        let mut out_b = [0u64; WIDTH];
        neon_to_lanes(&neon_state, &mut out_a, &mut out_b);

        for i in 0..WIDTH {
            assert_eq!(canon(out_a[i]), canon(ref_a[i]), "lane0 mismatch at {i}");
            assert_eq!(canon(out_b[i]), canon(ref_b[i]), "lane1 mismatch at {i}");
        }
    }

    #[test]
    fn test_internal_permute_neon_w12() {
        test_internal_neon_matches_scalar(
            MATRIX_DIAG_12_GOLDILOCKS,
            internal_permute_neon_w12,
            internal_permute_state_asm,
        );
    }

    #[test]
    fn test_internal_permute_neon_w16() {
        test_internal_neon_matches_scalar(
            MATRIX_DIAG_16_GOLDILOCKS,
            internal_permute_neon_w16,
            internal_permute_state_asm,
        );
    }

    fn test_internal_neon_generic_matches_scalar<const WIDTH: usize>(diag: [F; WIDTH]) {
        let mut rng = SmallRng::seed_from_u64(66);

        let diag_raw: [u64; WIDTH] = core::array::from_fn(|i| diag[i].value);
        let constants: Vec<u64> = (0..22).map(|_| rng.random()).collect();

        let lane_a: [u64; WIDTH] = rng.random();
        let lane_b: [u64; WIDTH] = rng.random();

        // Scalar reference.
        let mut ref_a = lane_a;
        let mut ref_b = lane_b;
        internal_permute_state_asm(&mut ref_a, &diag_raw, &constants);
        internal_permute_state_asm(&mut ref_b, &diag_raw, &constants);

        // Generic NEON version.
        let mut neon_state = lanes_to_neon(&lane_a, &lane_b);
        internal_permute_neon(&mut neon_state, &diag_raw, &constants);

        let mut out_a = [0u64; WIDTH];
        let mut out_b = [0u64; WIDTH];
        neon_to_lanes(&neon_state, &mut out_a, &mut out_b);

        for i in 0..WIDTH {
            assert_eq!(canon(out_a[i]), canon(ref_a[i]), "lane0 mismatch at {i}");
            assert_eq!(canon(out_b[i]), canon(ref_b[i]), "lane1 mismatch at {i}");
        }
    }

    #[test]
    fn test_internal_permute_neon_generic_w8() {
        test_internal_neon_generic_matches_scalar(MATRIX_DIAG_8_GOLDILOCKS);
    }

    #[test]
    fn test_internal_permute_neon_generic_w12() {
        test_internal_neon_generic_matches_scalar(MATRIX_DIAG_12_GOLDILOCKS);
    }

    #[test]
    fn test_internal_permute_neon_generic_w16() {
        test_internal_neon_generic_matches_scalar(MATRIX_DIAG_16_GOLDILOCKS);
    }

    #[test]
    fn test_internal_permute_neon_generic_w20() {
        test_internal_neon_generic_matches_scalar(MATRIX_DIAG_20_GOLDILOCKS);
    }

    // -------------------------------------------------------------------
    // Danger-zone proptests:
    //
    // Same shape as the uniform variants, but inputs are concentrated in the non-canonical band.
    // -------------------------------------------------------------------

    proptest! {
        #[test]
        fn test_sub_asm_danger(a in danger_u64(), b in danger_u64()) {
            let expected = (F::new(a) - F::new(b)).as_canonical_u64();
            let got = canon(unsafe { sub_asm(a, b) });
            prop_assert_eq!(got, expected);
        }

        #[test]
        fn test_double_asm_danger(a in danger_u64()) {
            let expected = (F::new(a) + F::new(a)).as_canonical_u64();
            let got = canon(unsafe { double_asm(a) });
            prop_assert_eq!(got, expected);
        }

        #[test]
        fn test_div2_asm_danger(a in danger_u64()) {
            let expected = F::new(a).halve().as_canonical_u64();
            let got = canon(unsafe { div2_asm(a) });
            prop_assert_eq!(got, expected);
        }

        #[test]
        fn test_div_2_32_asm_danger(a in danger_u64()) {
            let mut v = F::new(a);
            for _ in 0..32 { v = v.halve(); }
            let got = canon(unsafe { div_2_32_asm(a) });
            prop_assert_eq!(got, v.as_canonical_u64());
        }

        #[test]
        fn test_apply_mat4_asm_danger(state in danger_array::<4>()) {
            let f: [F; 4] = state.map(F::new);
            let two = F::TWO;
            let three = two + F::ONE;
            let expected = [
                two * f[0] + three * f[1] + f[2] + f[3],
                f[0] + two * f[1] + three * f[2] + f[3],
                f[0] + f[1] + two * f[2] + three * f[3],
                three * f[0] + f[1] + f[2] + two * f[3],
            ];
            let mut got = state;
            unsafe { apply_mat4_asm(&mut got); }
            for i in 0..4 {
                prop_assert_eq!(canon(got[i]), expected[i].as_canonical_u64());
            }
        }

        #[test]
        fn test_sbox_layer_asm_danger(state in danger_array::<8>()) {
            let mut got = state;
            unsafe { sbox_layer_asm(&mut got); }
            for i in 0..8 {
                let x = F::new(state[i]);
                let x2 = x * x;
                let x3 = x2 * x;
                let x4 = x2 * x2;
                let expected = x3 * x4;
                prop_assert_eq!(canon(got[i]), expected.as_canonical_u64());
            }
        }

        #[test]
        fn test_mds_light_w8_danger(state in danger_array::<8>()) {
            let mut field_state: [F; 8] = state.map(F::new);
            mds_light_permutation(&mut field_state, &MDSMat4);
            let mut asm_state = state;
            unsafe { mds_light_permutation_asm(&mut asm_state); }
            for i in 0..8 {
                prop_assert_eq!(canon(asm_state[i]), field_state[i].as_canonical_u64());
            }
        }

        #[test]
        fn test_mds_light_w12_danger(state in danger_array::<12>()) {
            let mut field_state: [F; 12] = state.map(F::new);
            mds_light_permutation(&mut field_state, &MDSMat4);
            let mut asm_state = state;
            unsafe { mds_light_permutation_asm(&mut asm_state); }
            for i in 0..12 {
                prop_assert_eq!(canon(asm_state[i]), field_state[i].as_canonical_u64());
            }
        }

        #[test]
        fn test_mds_light_w16_danger(state in danger_array::<16>()) {
            let mut field_state: [F; 16] = state.map(F::new);
            mds_light_permutation(&mut field_state, &MDSMat4);
            let mut asm_state = state;
            unsafe { mds_light_permutation_asm(&mut asm_state); }
            for i in 0..16 {
                prop_assert_eq!(canon(asm_state[i]), field_state[i].as_canonical_u64());
            }
        }

        #[test]
        fn test_external_round_w8_danger(
            state in danger_array::<8>(),
            rc in danger_array::<8>(),
        ) {
            let mut expected: [F; 8] = core::array::from_fn(|i| F::new(state[i]) + F::new(rc[i]));
            for x in expected.iter_mut() {
                let x2 = *x * *x;
                let x3 = x2 * *x;
                let x4 = x2 * x2;
                *x = x3 * x4;
            }
            mds_light_permutation(&mut expected, &MDSMat4);
            let mut got = state;
            unsafe { external_round_asm(&mut got, &rc); }
            for i in 0..8 {
                prop_assert_eq!(canon(got[i]), expected[i].as_canonical_u64());
            }
        }

        #[test]
        fn test_external_round_fused_w8_danger(
            state in danger_array::<8>(),
            rc in danger_array::<8>(),
        ) {
            let mut ref_state = state;
            let mut got = state;
            unsafe { external_round_asm(&mut ref_state, &rc); }
            unsafe { external_round_fused_w8(&mut got, &rc); }
            for i in 0..8 {
                prop_assert_eq!(canon(got[i]), canon(ref_state[i]));
            }
        }
    }

    // -------------------------------------------------------------------
    // Adversarial-state stress tests for the full internal permute.
    //
    // Compares the ASM permute against a field-level reference.
    // -------------------------------------------------------------------

    fn field_internal_round<const WIDTH: usize>(state: &mut [F; WIDTH], diag: [F; WIDTH], rc: u64) {
        state[0] += F::new(rc);
        let s = state[0];
        let s2 = s * s;
        let s3 = s2 * s;
        let s4 = s2 * s2;
        state[0] = s3 * s4;
        matmul_internal(state, diag);
    }

    fn run_internal_stress<const WIDTH: usize>(
        diag: [F; WIDTH],
        state_init: [u64; WIDTH],
        constants: &[u64],
    ) {
        let mut state_field: [F; WIDTH] = state_init.map(F::new);
        for &rc in constants {
            field_internal_round(&mut state_field, diag, rc);
        }

        let mut state_asm = state_init;
        let diag_raw: [u64; WIDTH] = core::array::from_fn(|i| diag[i].value);
        internal_permute_state_asm(&mut state_asm, &diag_raw, constants);

        for i in 0..WIDTH {
            assert_eq!(
                canon(state_asm[i]),
                state_field[i].as_canonical_u64(),
                "i={i}, init={state_init:?}, constants={constants:?}",
            );
        }
    }

    /// State + constants designed to hit the non-canonical band hard:
    /// (a) all canonical max,
    /// (b) all non-canonical max,
    /// (c) alternating,
    /// (d) all-zero state with non-canonical constants.
    ///
    /// Repeated rounds compound any latent reduction bug.
    fn adversarial_states<const WIDTH: usize>() -> Vec<([u64; WIDTH], Vec<u64>)> {
        let max_canonical = [P - 1; WIDTH];
        let max_noncanonical = [u64::MAX; WIDTH];
        let alternating: [u64; WIDTH] =
            core::array::from_fn(|i| if i % 2 == 0 { P - 1 } else { u64::MAX });
        let near_p_plus: [u64; WIDTH] = core::array::from_fn(|i| P + (i as u64));
        let zero_state = [0u64; WIDTH];

        let canon_consts = vec![P - 1; 22];
        let noncanon_consts = vec![u64::MAX; 22];
        let mixed_consts: Vec<u64> = (0..22)
            .map(|i| if i % 2 == 0 { P } else { u64::MAX - i as u64 })
            .collect();

        vec![
            (max_canonical, canon_consts.clone()),
            (max_noncanonical, canon_consts),
            (max_noncanonical, noncanon_consts.clone()),
            (alternating, mixed_consts.clone()),
            (near_p_plus, mixed_consts),
            (zero_state, noncanon_consts),
        ]
    }

    #[test]
    fn test_internal_permute_w8_stress() {
        for (state, consts) in adversarial_states::<8>() {
            run_internal_stress(MATRIX_DIAG_8_GOLDILOCKS, state, &consts);
        }
    }

    #[test]
    fn test_internal_permute_w12_stress() {
        for (state, consts) in adversarial_states::<12>() {
            run_internal_stress(MATRIX_DIAG_12_GOLDILOCKS, state, &consts);
        }
    }

    #[test]
    fn test_internal_permute_w16_stress() {
        for (state, consts) in adversarial_states::<16>() {
            run_internal_stress(MATRIX_DIAG_16_GOLDILOCKS, state, &consts);
        }
    }

    #[test]
    fn test_internal_permute_w20_stress() {
        for (state, consts) in adversarial_states::<20>() {
            run_internal_stress(MATRIX_DIAG_20_GOLDILOCKS, state, &consts);
        }
    }

    #[test]
    fn test_internal_permute_specialized_w8_stress() {
        for (state, consts) in adversarial_states::<8>() {
            let mut got = state;
            internal_permute_state_asm_w8(&mut got, &consts);

            let mut expected = state;
            let diag: [u64; 8] = core::array::from_fn(|i| MATRIX_DIAG_8_GOLDILOCKS[i].value);
            internal_permute_state_asm(&mut expected, &diag, &consts);

            for i in 0..8 {
                assert_eq!(canon(got[i]), canon(expected[i]));
            }
        }
    }

    #[test]
    fn test_internal_permute_specialized_w12_stress() {
        for (state, consts) in adversarial_states::<12>() {
            let mut got = state;
            internal_permute_state_asm_w12(&mut got, &consts);

            let mut expected = state;
            let diag: [u64; 12] = core::array::from_fn(|i| MATRIX_DIAG_12_GOLDILOCKS[i].value);
            internal_permute_state_asm(&mut expected, &diag, &consts);

            for i in 0..12 {
                assert_eq!(canon(got[i]), canon(expected[i]));
            }
        }
    }

    #[test]
    fn test_internal_permute_specialized_w16_stress() {
        for (state, consts) in adversarial_states::<16>() {
            let mut got = state;
            internal_permute_state_asm_w16(&mut got, &consts);

            let mut expected = state;
            let diag: [u64; 16] = core::array::from_fn(|i| MATRIX_DIAG_16_GOLDILOCKS[i].value);
            internal_permute_state_asm(&mut expected, &diag, &consts);

            for i in 0..16 {
                assert_eq!(canon(got[i]), canon(expected[i]));
            }
        }
    }

    #[test]
    fn test_external_round_w8_stress() {
        for (state, _) in adversarial_states::<8>() {
            let rc = [P - 1; 8];

            let mut expected: [F; 8] = core::array::from_fn(|i| F::new(state[i]) + F::new(rc[i]));
            for x in expected.iter_mut() {
                let x2 = *x * *x;
                let x3 = x2 * *x;
                let x4 = x2 * x2;
                *x = x3 * x4;
            }
            mds_light_permutation(&mut expected, &MDSMat4);

            let mut got = state;
            unsafe {
                external_round_asm(&mut got, &rc);
            }

            for i in 0..8 {
                assert_eq!(canon(got[i]), expected[i].as_canonical_u64());
            }
        }
    }

    #[test]
    fn test_external_round_fused_w8_stress() {
        for (state, _) in adversarial_states::<8>() {
            let rc = [u64::MAX; 8];

            let mut expected = state;
            unsafe {
                external_round_asm(&mut expected, &rc);
            }

            let mut got = state;
            unsafe {
                external_round_fused_w8(&mut got, &rc);
            }

            for i in 0..8 {
                assert_eq!(canon(got[i]), canon(expected[i]));
            }
        }
    }
}
