use p3_field::{AbstractField, TwoAdicField};
use p3_maybe_rayon::prelude::*;
use p3_util::{log2_strict_usize, reverse_slice_index_bits};

use super::{split_at_mut_unchecked, Real, P};
use crate::BabyBear;

// TODO: Consider following Hexl and storing the roots in a single
// array in bit-reversed order, but with duplicates for certain roots
// to avoid computing permutations in the inner loop.

const MONTY_ROOTS8: [u32; 3] = [1032137103, 473486609, 1964242958];

const MONTY_ROOTS16: [u32; 7] = [
    1594287233, 1032137103, 1173759574, 473486609, 1844575452, 1964242958, 270522423,
];

const MONTY_ROOTS32: [u32; 15] = [
    1063008748, 1594287233, 1648228672, 1032137103, 24220877, 1173759574, 1310027008, 473486609,
    518723214, 1844575452, 964210272, 1964242958, 48337049, 270522423, 434501889,
];

const MONTY_ROOTS64: [u32; 31] = [
    1427548538, 1063008748, 19319570, 1594287233, 292252822, 1648228672, 1754391076, 1032137103,
    1419020303, 24220877, 1848478141, 1173759574, 1270902541, 1310027008, 992470346, 473486609,
    690559708, 518723214, 1398247489, 1844575452, 1272476677, 964210272, 486600511, 1964242958,
    12128229, 48337049, 377028776, 270522423, 1626304099, 434501889, 741605237,
];

/// FIXME: The (i-1)th vector contains the roots...
pub fn roots_of_unity_table(n: usize) -> Vec<Vec<Real>> {
    let lg_n = log2_strict_usize(n);
    let half_n = 1 << (lg_n - 1);
    // nth_roots = [g, g^2, g^3, ..., g^{n/2 - 1}]
    let nth_roots: Vec<_> = BabyBear::two_adic_generator(lg_n)
        .powers()
        .take(half_n)
        .skip(1)
        .map(|x| x.value)
        .collect();

    (0..(lg_n - 1))
        .map(|i| {
            nth_roots
                .iter()
                .skip((1 << i) - 1)
                .step_by(1 << i)
                .copied()
                .collect::<Vec<_>>()
        })
        .collect()
}

const MONTY_BITS: u32 = 32;
const MONTY_MU: u32 = 0x88000001;

/// Montgomery reduction of a value in `0..P << MONTY_BITS` to a value in `0..P`.
#[inline(always)]
fn monty_reduce(x: u64) -> u32 {
    let t = x.wrapping_mul(MONTY_MU as u64) as u32 as u64;
    let u = t * (P as u64);

    let (x_sub_u, over) = x.overflowing_sub(u);
    let x_sub_u_hi = (x_sub_u >> MONTY_BITS) as u32;
    let corr = if over { P } else { 0 };
    x_sub_u_hi.wrapping_add(corr)
}

/// Given x in `0..P << MONTY_BITS`, return x mod P in [0, 2p).
/// TODO: Double-check the ranges above.
#[inline(always)]
pub fn partial_monty_reduce(x: u64) -> u32 {
    let q = MONTY_MU.wrapping_mul(x as u32);
    let h = ((q as u64 * P as u64) >> 32) as u32;
    let r = P - h + (x >> 32) as u32;
    r
}

/// Given x in [0, 2p), return the representative of x mod p in [0, p)
#[inline(always)]
fn reduce_2p(x: Real) -> Real {
    debug_assert!(x < 2 * P);

    if x < P {
        x
    } else {
        x - P
    }
}

/// Given x in [0, 4p), return the representative of x mod p in [0, p)
#[inline(always)]
fn reduce_4p(mut x: u64) -> Real {
    const PP: u64 = 0x78000001;
    debug_assert!(x < 4 * PP);

    if x > PP {
        x -= PP;
    }
    if x > PP {
        x -= PP;
    }
    if x > PP {
        x -= PP;
    }
    x as u32
}

#[inline(always)]
fn butterfly(x: Real, y: Real, w: Real) -> (Real, Real) {
    let t = P + x - y;
    (reduce_2p(x + y), monty_reduce(t as u64 * w as u64))
}

#[inline]
fn forward_pass(a: &mut [Real], roots: &[Real]) {
    let half_n = a.len() / 2;
    assert_eq!(roots.len(), half_n - 1);

    let (top, tail) = unsafe { split_at_mut_unchecked(a, half_n) };

    let x = top[0];
    let y = tail[0];

    top[0] = reduce_2p(x + y);
    tail[0] = reduce_2p(P + x - y);

    for i in 1..half_n {
        (top[i], tail[i]) = butterfly(top[i], tail[i], roots[i - 1]);
    }
}

#[inline(always)]
fn forward_2(a: &mut [Real]) {
    assert_eq!(a.len(), 2);

    let s = reduce_2p(a[0] + a[1]);
    let t = reduce_2p(P + a[0] - a[1]);
    a[0] = s;
    a[1] = t;
}

#[inline(always)]
fn forward_4(a: &mut [Real]) {
    assert_eq!(a.len(), 4);

    const ROOT: u64 = MONTY_ROOTS8[1] as u64;

    let t1 = (P + a[1] - a[3]) as u64;
    let t5 = (a[1] + a[3]) as u64;
    let t3 = partial_monty_reduce(t1 * ROOT) as u64;
    let t4 = (a[0] + a[2]) as u64;
    let t2 = (P + a[0] - a[2]) as u64;

    const TWO_P: u64 = 2 * P as u64;

    // Return in bit-reversed order
    a[0] = reduce_4p(t4 + t5); // b0
    a[2] = reduce_4p(t2 + t3); // b1
    a[1] = reduce_4p(TWO_P + t4 - t5); // b2
    a[3] = reduce_4p(TWO_P + t2 - t3); // b3
}

#[inline(always)]
pub fn forward_8(a: &mut [Real]) {
    assert_eq!(a.len(), 8);

    forward_pass(a, &MONTY_ROOTS8);

    let (a0, a1) = unsafe { split_at_mut_unchecked(a, a.len() / 2) };
    forward_4(a0);
    forward_4(a1);
}

#[inline(always)]
pub fn forward_16(a: &mut [Real]) {
    assert_eq!(a.len(), 16);

    forward_pass(a, &MONTY_ROOTS16);

    let (a0, a1) = unsafe { split_at_mut_unchecked(a, a.len() / 2) };
    forward_8(a0);
    forward_8(a1);
}

#[inline(always)]
pub fn forward_32(a: &mut [Real]) {
    assert_eq!(a.len(), 32);

    forward_pass(a, &MONTY_ROOTS32);

    let (a0, a1) = unsafe { split_at_mut_unchecked(a, a.len() / 2) };
    forward_16(a0);
    forward_16(a1);
}

#[inline(always)]
pub fn forward_64(a: &mut [Real]) {
    assert_eq!(a.len(), 64);

    forward_pass(a, &MONTY_ROOTS64);

    let (a0, a1) = unsafe { split_at_mut_unchecked(a, a.len() / 2) };
    forward_32(a0);
    forward_32(a1);
}

#[inline(always)]
pub fn forward_128(a: &mut [Real], roots: &[Real]) {
    assert_eq!(a.len(), 128);

    forward_pass(a, roots);

    let (a0, a1) = unsafe { split_at_mut_unchecked(a, a.len() / 2) };
    forward_64(a0);
    forward_64(a1);
}

#[inline(always)]
pub fn forward_256(a: &mut [Real], root_table: &[Vec<Real>]) {
    assert_eq!(a.len(), 256);

    forward_pass(a, &root_table[0]);

    let (a0, a1) = unsafe { split_at_mut_unchecked(a, a.len() / 2) };
    forward_128(a0, &root_table[1]);
    forward_128(a1, &root_table[1]);
}

#[inline]
pub fn forward_fft(a: &mut [Real], root_table: &[Vec<Real>]) {
    let n = a.len();
    assert!(1 << (root_table.len() + 1) == n);

    match n {
        256 => forward_256(a, &root_table),
        128 => forward_128(a, &root_table[0]),
        64 => forward_64(a),
        32 => forward_32(a),
        16 => forward_16(a),
        8 => forward_8(a),
        4 => forward_4(a),
        2 => forward_2(a),
        _ => {
            debug_assert!(n > 64);
            forward_pass(a, &root_table[0]);
            let (a0, a1) = unsafe { split_at_mut_unchecked(a, n / 2) };

            forward_fft(a0, &root_table[1..]);
            forward_fft(a1, &root_table[1..]);
        }
    }
}

pub fn batch_forward_fft(a: &mut [Vec<Real>], root_table: &[Vec<Real>]) {
    a.par_iter_mut().for_each(|v| forward_fft(v, root_table));
}

/// Square root of an integer
/// Source: https://en.wikipedia.org/wiki/Integer_square_root#Using_only_integer_division
#[inline]
fn _isqrt(s: usize) -> usize {
    // Zero yields zero
    // One yields one
    if s <= 1 {
        return s;
    }

    // Initial estimate (must be too high)
    let mut x0 = s / 2;

    // Update
    let mut x1 = (x0 + s / x0) / 2;

    while x1 < x0 {
        x0 = x1;
        x1 = (x0 + s / x0) / 2;
    }
    debug_assert_eq!(x0 * x0, s);
    x0
}

/// Parameter for tuning the transpose operation.
/// TODO: Should depend on the size of the matrix elements.
const TRANSPOSE_BLOCK_SIZE: usize = 64; // TODO: 64 is better for u32

#[inline(always)]
fn transpose_scalar_block(output: &mut [Real], input: &[Real], nrows: usize, ncols: usize) {
    for i in 0..TRANSPOSE_BLOCK_SIZE {
        for j in 0..TRANSPOSE_BLOCK_SIZE {
            // Ensure the generated code doesn't do bounds checks:
            unsafe {
                // Equivalent to: output[j * nrows + i] = input[i * ncols + j]
                *output.get_unchecked_mut(j * nrows + i) = *input.get_unchecked(i * ncols + j);
            }
        }
    }
}

#[inline]
fn transpose_block(output: &mut [Real], input: &[Real], nrows: usize, ncols: usize) {
    debug_assert_eq!(nrows % TRANSPOSE_BLOCK_SIZE, 0);
    debug_assert_eq!(ncols % TRANSPOSE_BLOCK_SIZE, 0);

    for i in (0..nrows).step_by(TRANSPOSE_BLOCK_SIZE) {
        for j in (0..ncols).step_by(TRANSPOSE_BLOCK_SIZE) {
            let in_begin = i * ncols + j;
            let out_begin = j * nrows + i;

            // Equivalent to:
            // transpose_scalar_block(
            //     &mut output[out_begin..],
            //     &input[in_begin..],
            //     nrows, ncols);
            let (out, inp) = unsafe {
                (
                    output.get_unchecked_mut(out_begin..),
                    input.get_unchecked(in_begin..),
                )
            };

            transpose_scalar_block(out, inp, nrows, ncols);
        }
    }
    // FIXME: Need to handle case where TRANSPOSE_BLOCK_SIZE doesn't
    // divide matrix dimensions.
}

fn _printmat(a: &[Real], nrows: usize, ncols: usize) {
    for i in 0..nrows {
        for j in 0..ncols {
            print!("{} ", a[i * ncols + j]);
        }
        println!("");
    }
    println!("");
}

// TODO: Write a proper out-of-place version
fn slow_forward_fft(output: &mut [Real], input: &[Real], root_table: &[Vec<Real>]) {
    output.copy_from_slice(input);
    forward_fft(output, root_table);
    reverse_slice_index_bits(output);
}

/// Size of FFT above which we parallelise the FFT.
const FFT_PARALLEL_THRESHOLD: usize = 2; //TODO: Use 1024 or something

fn four_step_fft_inner(output: &mut [Real], input: &mut [Real], root_table: &[Vec<Real>]) {
    let n = input.len();
    if n <= FFT_PARALLEL_THRESHOLD {
        slow_forward_fft(output, input, root_table);
        return;
    }
    assert_eq!(n, output.len());

    let lg_n = log2_strict_usize(n);
    let lg_sqrt_n = lg_n / 2;
    let sqrt_n = 1 << lg_sqrt_n;

    let ncols = sqrt_n;
    let nrows = n / sqrt_n;
    let lg_ncols = lg_sqrt_n;
    let lg_nrows = lg_n - lg_sqrt_n;

    debug_assert_eq!(n, nrows * ncols);
    debug_assert_eq!(nrows, 1 << lg_nrows);

    let buf1 = input;
    let buf2 = output;

    // buf1 is nrows x ncols
    transpose_block(buf2, buf1, nrows, ncols);

    // buf2 is ncols x nrows
    buf1.par_chunks_exact_mut(nrows)
        .zip(buf2.par_chunks_exact(nrows))
        .for_each(|(out, col)| {
            slow_forward_fft(out, col, &root_table[lg_ncols..]);
        });

    // buf1 is ncols x nrows, each row is fft(col of input)

    // TODO: parallelise
    // TODO: Store root_table[0] in an order that improves cache access
    for i in 1..ncols {
        for j in 1..nrows {
            let exp = i * j;
            let w = if exp < n / 2 {
                root_table[0][exp - 1]
            } else if exp == n / 2 {
                // TODO: Don't multiply in this case
                //P - 1
                1744830467
            } else {
                // exp > n / 2
                P - root_table[0][(exp - n / 2) - 1]
            };
            let s = buf1[i * nrows + j];
            let t = monty_reduce(s as u64 * w as u64);
            buf1[i * nrows + j] = t;
        }
    }

    // TODO: Consider combining this transpose and the twiddle adjustment above?
    transpose_block(buf2, buf1, ncols, nrows);

    // buf2 is nrows x ncols
    buf1.par_chunks_exact_mut(ncols)
        .zip(buf2.par_chunks_exact(ncols))
        .for_each(|(out, row)| {
            slow_forward_fft(out, row, &root_table[lg_nrows..]);
        });

    // TODO: If we're just doing convolution, then we can skip the last transpose
    transpose_block(buf2, buf1, nrows, ncols);
}

pub fn four_step_fft(input: &mut [Real], root_table: &[Vec<Real>]) {
    // TODO: Don't do this copy
    let mut output = Vec::with_capacity(input.len());
    unsafe {
        output.set_len(input.len());
    }
    four_step_fft_inner(&mut output, input, root_table);
    input.copy_from_slice(&output);
}
