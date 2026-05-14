//! FFT-4 butterfly operations for real signals.
//!
//! These are pure integer arithmetic (i64/u64), completely field-independent.
//! Used by all MDS frequency-domain multiplications.

#[inline(always)]
pub const fn fft2_real(x: [u64; 2]) -> [i64; 2] {
    [(x[0] as i64 + x[1] as i64), (x[0] as i64 - x[1] as i64)]
}

#[inline(always)]
pub const fn ifft2_real(y: [i64; 2]) -> [u64; 2] {
    [(y[0] + y[1]) as u64, (y[0] - y[1]) as u64]
}

#[inline(always)]
pub const fn fft4_real(x: [u64; 4]) -> (i64, (i64, i64), i64) {
    let [z0, z2] = fft2_real([x[0], x[2]]);
    let [z1, z3] = fft2_real([x[1], x[3]]);
    (z0 + z1, (z2, -z3), z0 - z1)
}

#[inline(always)]
pub const fn ifft4_real(y: (i64, (i64, i64), i64)) -> [u64; 4] {
    let z0 = y.0 + y.2;
    let z1 = y.0 - y.2;
    let z2 = y.1 .0;
    let z3 = -y.1 .1;
    let [x0, x2] = ifft2_real([z0, z2]);
    let [x1, x3] = ifft2_real([z1, z3]);
    [x0, x1, x2, x3]
}
