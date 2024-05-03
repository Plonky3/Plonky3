use super::{split_at_mut_unchecked, Real, P};

// TODO: Consider following Hexl and storing the roots in a singler
// array in bit-reversed order, but with duplicates for certain roots
// to avoid computing permutations in the inner loop.

const ROOTS8: [i64; 3] = [1592366214, 1728404513, 211723194];
const ROOTS16: [i64; 7] = [
    196396260, 1592366214, 78945800, 1728404513, 1400279418, 211723194, 1446056615,
];
const ROOTS32: [i64; 15] = [
    760005850, 196396260, 1240658731, 1592366214, 177390144, 78945800, 1399190761, 1728404513,
    889310574, 1400279418, 1561292356, 211723194, 1424376889, 1446056615, 740045640,
];
const ROOTS64: [i64; 31] = [
    291676017, 760005850, 1141518129, 196396260, 1521113831, 1240658731, 1074029057, 1592366214,
    602251207, 177390144, 1684363409, 78945800, 406991886, 1399190761, 1094366075, 1728404513,
    72041623, 889310574, 1724976031, 1400279418, 1917679203, 1561292356, 1171812390, 211723194,
    1326890868, 1424376889, 680048719, 1446056615, 1957706687, 740045640, 662200255,
];

#[inline(always)]
fn reduce(x: Real) -> Real {
    (P + (x % P)) % P
}

const MONTY_BITS: u32 = 32;
const MONTY_MASK: u32 = ((1u64 << MONTY_BITS) - 1) as u32;
const MONTY_MU: u32 = 0x88000001;

/*
/// Montgomery reduction of a value in `0..P << MONTY_BITS`.
#[inline]
fn monty_reduce(x: u64) -> u32 {
    let t = x.wrapping_mul(MONTY_MU as u64) & (MONTY_MASK as u64);
    let u = t * (P as u64);

    let (x_sub_u, over) = x.overflowing_sub(u);
    let x_sub_u_hi = (x_sub_u >> 31) as u32;
    let corr = if over { P } else { 0 };
    x_sub_u_hi.wrapping_add(corr)
}
*/

#[inline]
fn forward_pass(a: &mut [Real], roots: &[Real]) {
    let half_n = a.len() / 2;
    assert_eq!(roots.len(), half_n - 1);

    let (top, tail) = unsafe { split_at_mut_unchecked(a, half_n) };

    let s = top[0];
    let t = tail[half_n];
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

#[inline(always)]
fn forward_4(a: &mut [Real], root: Real) {
    assert_eq!(a.len(), 4);

    let t1 = a[1] - a[3];
    let t5 = a[1] + a[3];
    let t3 = root * t1;
    let t4 = a[0] + a[2];
    let t2 = a[0] - a[2];

    // Return in bit-reversed order
    a[0] = reduce(t4 + t5); // b0
    a[2] = reduce(t2 + t3); // b1
    a[1] = reduce(t4 - t5); // b2
    a[3] = reduce(t2 - t3); // b3
}

#[inline(always)]
pub fn forward_8(a: &mut [Real], roots: &[Real]) {
    assert_eq!(a.len(), 8);
    assert_eq!(roots.len(), 3);

    let e0 = a[0] + a[4];
    let e1 = a[1] + a[5];
    let e2 = a[2] + a[6];
    let e3 = a[3] + a[7];

    let f0 = a[0] - a[4];
    let f1 = a[1] - a[5];
    let f2 = a[2] - a[6];
    let f3 = a[3] - a[7];

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

#[inline(always)]
pub fn forward_16(a: &mut [Real], roots: &[Real]) {
    assert_eq!(a.len(), 16);

    let half_n = a.len() / 2;

    forward_pass(a, roots);

    let (a0, a1) = unsafe { split_at_mut_unchecked(a, half_n) };
    forward_8(a0, &ROOTS8);
    forward_8(a1, &ROOTS8);
}

#[inline(always)]
pub fn forward_32(a: &mut [Real], roots: &[Real]) {
    assert_eq!(a.len(), 32);

    let half_n = a.len() / 2;

    forward_pass(a, roots);

    let (a0, a1) = unsafe { split_at_mut_unchecked(a, half_n) };
    forward_16(a0, &ROOTS16);
    forward_16(a1, &ROOTS16);
}

/*
#[inline]
pub fn forward_fft(a: &mut [Real], roots: &[Real]) {
    let n = a.len();

    if n > 8 {
        forward_pass(a, roots);
        let (a0, a1) = unsafe { split_at_mut_unchecked(a, n / 2) };

        forward_fft(a0, xxxroots);
        forward_fft(a1, xxxroots);
    } else if n > 4 {
        debug_assert_eq!(n, 8);
        forward_8(a, &ROOTS8);
    } else if n > 1 {
        debug_assert_eq!(n, 4);
        forward_4(a, ROOTS8[1]);
    }
}
*/

// n = 4:
//  - one forward_pass on a[0..4]
//    -
//  - two ffts of size 2:
//    - n = 2: one forward_pass ==> transformzero
