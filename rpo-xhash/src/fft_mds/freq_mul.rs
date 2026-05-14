//! Frequency-domain MDS multiplication at size 12.
//!
//! Pure u64/i64 arithmetic. Circulant first row matches miden-crypto's
//! RPO/RPX MDS matrix, `[7, 23, 8, 26, 13, 10, 9, 7, 6, 22, 21, 8]`, expressed
//! in the frequency domain. The constants below are byte-compatible with
//! miden-crypto's `apply_mds` so RPO-Goldilocks and xHash/RPX-Goldilocks
//! produce the same digests as the canonical Rpo256/Rpx256 implementations.

use super::butterfly::{fft4_real, ifft4_real};

// ============================================================
// Block convolutions: size 3 (for N=12 = 4×3)
// ============================================================

#[inline(always)]
const fn block1_3(x: [i64; 3], y: [i64; 3]) -> [i64; 3] {
    [
        x[0] * y[0] + x[1] * y[2] + x[2] * y[1],
        x[0] * y[1] + x[1] * y[0] + x[2] * y[2],
        x[0] * y[2] + x[1] * y[1] + x[2] * y[0],
    ]
}

#[inline(always)]
fn block2_3(x: [(i64, i64); 3], y: [(i64, i64); 3]) -> [(i64, i64); 3] {
    let [(x0r, x0i), (x1r, x1i), (x2r, x2i)] = x;
    let [(y0r, y0i), (y1r, y1i), (y2r, y2i)] = y;
    let x0s = x0r + x0i;
    let x1s = x1r + x1i;
    let x2s = x2r + x2i;
    let y0s = y0r + y0i;
    let y1s = y1r + y1i;
    let y2s = y2r + y2i;

    let m0 = (x0r * y0r, x0i * y0i);
    let m1 = (x1r * y2r, x1i * y2i);
    let m2 = (x2r * y1r, x2i * y1i);
    let z0r = (m0.0 - m0.1) + (x1s * y2s - m1.0 - m1.1) + (x2s * y1s - m2.0 - m2.1);
    let z0i = (x0s * y0s - m0.0 - m0.1) + (-m1.0 + m1.1) + (-m2.0 + m2.1);

    let m0 = (x0r * y1r, x0i * y1i);
    let m1 = (x1r * y0r, x1i * y0i);
    let m2 = (x2r * y2r, x2i * y2i);
    let z1r = (m0.0 - m0.1) + (m1.0 - m1.1) + (x2s * y2s - m2.0 - m2.1);
    let z1i = (x0s * y1s - m0.0 - m0.1) + (x1s * y0s - m1.0 - m1.1) + (-m2.0 + m2.1);

    let m0 = (x0r * y2r, x0i * y2i);
    let m1 = (x1r * y1r, x1i * y1i);
    let m2 = (x2r * y0r, x2i * y0i);
    let z2r = (m0.0 - m0.1) + (m1.0 - m1.1) + (m2.0 - m2.1);
    let z2i = (x0s * y2s - m0.0 - m0.1) + (x1s * y1s - m1.0 - m1.1) + (x2s * y0s - m2.0 - m2.1);

    [(z0r, z0i), (z1r, z1i), (z2r, z2i)]
}

#[inline(always)]
const fn block3_3(x: [i64; 3], y: [i64; 3]) -> [i64; 3] {
    [
        x[0] * y[0] - x[1] * y[2] - x[2] * y[1],
        x[0] * y[1] + x[1] * y[0] - x[2] * y[2],
        x[0] * y[2] + x[1] * y[1] + x[2] * y[0],
    ]
}

// ============================================================
// N=12 frequency-domain constants and multiply
// ============================================================

// miden-crypto canonical RPO/RPX MDS in frequency domain.
const MDS12_BLOCK1: [i64; 3] = [16, 8, 16];
const MDS12_BLOCK2: [(i64, i64); 3] = [(-1, 2), (-1, 1), (4, 8)];
const MDS12_BLOCK3: [i64; 3] = [-8, 1, 1];

#[inline(always)]
pub fn mds12_multiply_freq(state: [u64; 12]) -> [u64; 12] {
    let [s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11] = state;
    let (u0, u1, u2) = fft4_real([s0, s3, s6, s9]);
    let (u4, u5, u6) = fft4_real([s1, s4, s7, s10]);
    let (u8, u9, u10) = fft4_real([s2, s5, s8, s11]);

    let [v0, v4, v8] = block1_3([u0, u4, u8], MDS12_BLOCK1);
    let [v1, v5, v9] = block2_3([u1, u5, u9], MDS12_BLOCK2);
    let [v2, v6, v10] = block3_3([u2, u6, u10], MDS12_BLOCK3);

    let [s0, s3, s6, s9] = ifft4_real((v0, v1, v2));
    let [s1, s4, s7, s10] = ifft4_real((v4, v5, v6));
    let [s2, s5, s8, s11] = ifft4_real((v8, v9, v10));
    [s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11]
}
