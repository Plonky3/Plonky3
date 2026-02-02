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
pub(crate) unsafe fn mul_asm(a: u64, b: u64) -> u64 {
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
pub(crate) unsafe fn add_asm(a: u64, b: u64) -> u64 {
    let result: u64;
    let _adj: u64;

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

/// Interleaved dual-lane external round.
#[inline(always)]
pub unsafe fn external_round_dual_asm<const WIDTH: usize>(
    state0: &mut [u64; WIDTH],
    state1: &mut [u64; WIDTH],
    rc: &[u64; WIDTH],
) {
    unsafe {
        // Add round constants - interleaved
        for i in 0..WIDTH {
            state0[i] = add_asm(state0[i], rc[i]);
            state1[i] = add_asm(state1[i], rc[i]);
        }

        // S-box layer - interleaved x^2 computation
        let mut x2_a = [0u64; WIDTH];
        let mut x2_b = [0u64; WIDTH];
        for i in 0..WIDTH {
            x2_a[i] = mul_asm(state0[i], state0[i]);
            x2_b[i] = mul_asm(state1[i], state1[i]);
        }

        // x^3 and x^4 - interleaved
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

        // x^7 - interleaved
        for i in 0..WIDTH {
            state0[i] = mul_asm(x3_a[i], x4_a[i]);
            state1[i] = mul_asm(x3_b[i], x4_b[i]);
        }

        // MDS - sequential (MDS is mostly additions, less benefit from interleaving)
        mds_light_permutation_asm(state0);
        mds_light_permutation_asm(state1);
    }
}

// External layer: S-box on all elements, then MDS. Pipelined for latency hiding.

/// Double a Goldilocks element.
#[inline(always)]
unsafe fn double_asm(a: u64) -> u64 {
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

        // S-box layer
        sbox_layer_asm(state);

        // MDS layer
        mds_light_permutation_asm(state);
    }
}

/// Run all initial external rounds.
#[inline]
pub fn external_initial_permute_state_asm<const WIDTH: usize>(
    state: &mut [Goldilocks; WIDTH],
    initial_constants: &[[Goldilocks; WIDTH]],
) {
    let state_raw: &mut [u64; WIDTH] =
        unsafe { &mut *(state as *mut [Goldilocks; WIDTH] as *mut [u64; WIDTH]) };

    // Initial MDS
    unsafe {
        mds_light_permutation_asm(state_raw);
    }

    // External rounds
    for rc in initial_constants {
        let rc_raw: [u64; WIDTH] = unsafe { core::mem::transmute_copy(rc) };
        unsafe {
            external_round_asm(state_raw, &rc_raw);
        }
    }
}

/// Run all terminal external rounds.
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
