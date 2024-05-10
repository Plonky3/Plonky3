use super::{split_at_mut_unchecked, Real, P};
use crate::BabyBear;
use p3_field::{AbstractField, TwoAdicField};
use p3_util::log2_strict_usize;

// TODO: Consider following Hexl and storing the roots in a single
// array in bit-reversed order, but with duplicates for certain roots
// to avoid computing permutations in the inner loop.

const MONTY_ROOTS8: [i64; 3] = [1032137103, 473486609, 1964242958];

const MONTY_ROOTS16: [i64; 7] = [
    1594287233, 1032137103, 1173759574, 473486609, 1844575452, 1964242958, 270522423,
];

const MONTY_ROOTS32: [i64; 15] = [
    1063008748, 1594287233, 1648228672, 1032137103, 24220877, 1173759574, 1310027008, 473486609,
    518723214, 1844575452, 964210272, 1964242958, 48337049, 270522423, 434501889,
];

const MONTY_ROOTS64: [i64; 31] = [
    1427548538, 1063008748, 19319570, 1594287233, 292252822, 1648228672, 1754391076, 1032137103,
    1419020303, 24220877, 1848478141, 1173759574, 1270902541, 1310027008, 992470346, 473486609,
    690559708, 518723214, 1398247489, 1844575452, 1272476677, 964210272, 486600511, 1964242958,
    12128229, 48337049, 377028776, 270522423, 1626304099, 434501889, 741605237,
];

pub fn roots_of_unity_table(n: usize) -> Vec<Vec<i64>> {
    let lg_n = log2_strict_usize(n);
    let half_n = 1 << (lg_n - 1);
    let nth_roots: Vec<_> = BabyBear::two_adic_generator(lg_n)
        .powers()
        .take(half_n)
        .skip(1)
        .map(|x| x.value as i64)
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

/*
#[inline]
fn forward_pass(a: &mut [Real], roots: &[Real]) {
    let half_n = a.len() / 2;
    assert_eq!(roots.len(), half_n - 1);

    let (top, tail) = unsafe { split_at_mut_unchecked(a, half_n) };

    let s = top[0];
    let t = tail[0];
    top[0] = reduce(s + t);
    tail[0] = reduce(s - t);

    for i in 1..half_n {
        let w = roots[i - 1];
        let s = top[i];
        let t = tail[i];
        top[i] = reduce(s + t);
        tail[i] = reduce((s - t) * w);
    }
}
*/

const TWO_P: i64 = 2 * P;

const MONTY_BITS: u32 = 32;
const MONTY_MU: u32 = 0x88000001;

/// Montgomery reduction of a value in `0..P << MONTY_BITS`.
#[inline(always)]
fn monty_reduce(x: u64) -> u32 {
    const PP: u32 = 0x78000001;
    let t = x.wrapping_mul(MONTY_MU as u64) as u32 as u64;
    let u = t * (P as u64);

    let (x_sub_u, over) = x.overflowing_sub(u);
    let x_sub_u_hi = (x_sub_u >> MONTY_BITS) as u32;
    let corr = if over { PP } else { 0 };
    x_sub_u_hi.wrapping_add(corr)
}

// FIXME: Use this!
#[inline(always)]
pub fn partial_monty_reduce(u: u64) -> u32 {
    const PP: u32 = 0x78000001;
    let q = MONTY_MU.wrapping_mul(u as u32);
    let h = ((q as u64 * P as u64) >> 32) as u32;
    let r = PP - h + (u >> 32) as u32;
    r as u32
}

/// Given x in [0, 2p), return the x mod p in [0, p)
#[inline(always)]
fn reduce_2p(x: i64) -> i64 {
    if x < P {
        x
    } else {
        x - P
    }
}

/// Given x in [0, 4p), return the x mod p in [0, p)
#[inline(always)]
fn reduce_4p(mut x: i64) -> i64 {
    if x > P {
        x -= P;
    }
    if x > P {
        x -= P;
    }
    if x > P {
        x -= P;
    }
    x
}

#[inline(always)]
fn partial_barrett_reduce(x: u32, w: u32) -> u32 {
    const P: u32 = 0x78000001;

    let w_pre = ((w as u64) << 32) / (P as u64);
    debug_assert!(w_pre <= u32::MAX as u64);
    let w_pre = w_pre as u32;

    let q = x as u64 * w_pre as u64;
    let q_hi = (q >> 32) as u32;
    let u = x.wrapping_mul(w);
    let qp = q_hi.wrapping_mul(P);
    let r = u.wrapping_sub(qp);
    debug_assert!(r < 2 * P);

    // Don't do final correction, so result is in [0, 2p)
    r
}

#[inline(always)]
fn barrett_reduce(x: u32, w: u32) -> u32 {
    let r = partial_barrett_reduce(x, w);
    reduce_2p(r as i64) as u32
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
        let w = roots[i - 1];
        let x = top[i];
        let y = tail[i];

        top[i] = reduce_2p(x + y);

        let t = P + x - y;
        tail[i] = monty_reduce((t * w) as u64) as i64;
    }
}

#[inline(always)]
fn forward_4(a: &mut [Real]) {
    assert_eq!(a.len(), 4);

    const ROOT: i64 = MONTY_ROOTS8[1];

    let t1 = P + a[1] - a[3];
    let t5 = a[1] + a[3];
    let t3 = monty_reduce((t1 * ROOT) as u64) as i64;
    let t4 = a[0] + a[2];
    let t2 = P + a[0] - a[2];

    // Return in bit-reversed order
    a[0] = reduce_4p(t4 + t5); // b0
    a[2] = reduce_4p(t2 + t3); // b1
    a[1] = reduce_4p(TWO_P + t4 - t5); // b2
    a[3] = reduce_4p(TWO_P + t2 - t3); // b3
}

// TODO: Maybe allow to expand into i64 and do a %P at the end?
/*
#[inline(always)]
pub fn forward_8(a: &mut [Real], roots: &[Real]) {
    assert_eq!(a.len(), 8);
    assert_eq!(roots.len(), 3);

    let e0 = a[0] + a[4];
    let e1 = a[1] + a[5];
    let e2 = a[2] + a[6];
    let e3 = a[3] + a[7];

    let f0 = P + a[0] - a[4];
    let f1 = P + a[1] - a[5];
    let f2 = P + a[2] - a[6];
    let f3 = P + a[3] - a[7];

    let e02 = e0 + e2;
    let e13 = e1 + e3;
    let g02 = e0 - e2;
    let g13 = e1 - e3;
    let t = g13 * roots[1]; // roots[i] holds g^{i+1}

    // Return result b = [b0, b1, .., b7] in bit-reversed order
    a[0] = reduce(e02 + e13); // b0
    a[2] = reduce(g02 + t); // b2
    a[1] = reduce(e02 - e13); // b4
    a[3] = reduce(g02 - t); // b6

    let t1 = f1 * roots[0];
    let t2 = f2 * roots[1];
    let t3 = f3 * roots[2];

    let u1 = f1 * roots[2];
    let u3 = f3 * roots[0];

    let v0 = f0 + t2; // (a0 - a4) + (a2 - a6)*r^2
    let v1 = f0 - t2; // (a0 - a4) - (a2 - a6)*r^2
    let w0 = t1 + t3; //
    let w1 = u1 + u3; // (a1 - a5)*r^3 + (a3 - a7)*r

    a[4] = reduce(v0 + w0); // f0 + t1 + t2 + t3; // b1
    a[6] = reduce(v1 + w1); // f0 + u1 - t2 + u3; // b3
    a[5] = reduce(v0 - w0); // f0 - t1 + t2 - t3; // b5
    a[7] = reduce(v1 - w1); // f0 - u1 - t2 - u3; // b7
}
*/

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

#[inline]
pub fn forward_fft(a: &mut [Real], root_table: &[Vec<i64>]) {
    let n = a.len();
    assert!(1 << (root_table.len() + 1) == n);

    match n {
        64 => forward_64(a),
        32 => forward_32(a),
        16 => forward_16(a),
        8 => forward_8(a),
        4 => forward_4(a),
        _ => {
            debug_assert!(n > 64);
            forward_pass(a, &root_table[0]);
            let (a0, a1) = unsafe { split_at_mut_unchecked(a, n / 2) };

            forward_fft(a0, &root_table[1..]);
            forward_fft(a1, &root_table[1..]);
        }
    }
}

// n = 4:
//  - one forward_pass on a[0..4]
//    -
//  - two ffts of size 2:
//    - n = 2: one forward_pass ==> transformzero
