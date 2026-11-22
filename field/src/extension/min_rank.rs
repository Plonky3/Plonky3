//! Minimum-rank Winograd / Alder–Strassen `(2D - 1)` bilinear multiplication and
//! symmetric squaring kernels for extension algebras over a field `F`.
//!
//! When `A: Algebra<F>` represents symbolic or arithmetic AIR constraint expressions,
//! multiplying an expression by a constant `c: F` has polynomial degree `0`, whereas
//! multiplying two expressions `u * v` has degree `2` and emits a quadratic constraint
//! node. Evaluating polynomials of degree `< D` at `2D - 2` small integers together
//! with the point at infinity (`∞`) reduces extension multiplication from `D²`
//! quadratic `A * A` multiplications down to the theoretical Alder–Strassen minimum
//! `2D - 1` (`16 → 7` for `D = 4`, `25 → 9` for `D = 5`, `64 → 15` for `D = 8`).

use core::array;

use super::{
    binomial_square, cubic_square, quintic_square, trinomial_cubic_mul, trinomial_quintic_mul,
};
use crate::Algebra;
use crate::field::Field;

// ---------------------------------------------------------------------------
// Compile-time exact integer Lagrange + ∞ interpolation tables
// ---------------------------------------------------------------------------

/// Build the exact integer numerator matrix `T` of shape `[R; R]` (`R = M + 1 = 2D - 1`)
/// such that for any polynomial `C(X) = Σ_{m=0}^{M} c_m X^m` evaluated at the `M`
/// distinct integer points `xs` (`v_r = C(xs[r])` for `r < M`) and `∞` (`v_M = c_M`),
/// every coefficient satisfies `denom * c_m = Σ_{r=0}^{M} T[m][r] * v_r`.
///
/// Uses the exact identity:
/// `C(X) = v_∞ · Π_{r=0}^{M-1} (X - xs[r]) + Σ_{r=0}^{M-1} v_r · L_r(X)`,
/// and verifies `T · V == denom · I` over `i128` at compile time.
const fn build_unreduced_interp<const M: usize, const R: usize>(
    xs: [i64; M],
    denom: i64,
) -> [[i64; R]; R] {
    assert!(R == M + 1);
    assert!(denom > 0);

    // Compute monic P(X) = Π_{r=0}^{M-1} (X - xs[r]) of degree M.
    let mut p = [0i64; R];
    p[0] = 1;
    let mut deg = 0usize;
    while deg < M {
        let xr = xs[deg];
        let mut k = deg + 1;
        while k > 0 {
            p[k] = p[k - 1] - xr * p[k];
            k -= 1;
        }
        p[0] = -xr * p[0];
        deg += 1;
    }

    let mut table = [[0i64; R]; R];

    // Finite nodes r in 0..M: synthetic division Q_r(X) = P(X) / (X - xs[r]).
    let mut r = 0usize;
    while r < M {
        let xr = xs[r];
        let mut q = [0i64; R];
        let mut rem = p[M];
        let mut k = M;
        while k > 0 {
            q[k - 1] = rem;
            rem = p[k - 1] + xr * rem;
            k -= 1;
        }
        assert!(rem == 0);

        let mut prod_diff = 1i64;
        let mut j = 0usize;
        while j < M {
            if j != r {
                prod_diff *= xr - xs[j];
            }
            j += 1;
        }
        assert!(denom % prod_diff == 0);
        let scale = denom / prod_diff;

        let mut m = 0usize;
        while m < M {
            table[m][r] = scale * q[m];
            m += 1;
        }
        table[M][r] = 0;
        r += 1;
    }

    // Point at infinity r = M: c_m gets denom * [X^m] P(X).
    let mut m = 0usize;
    while m < R {
        table[m][M] = denom * p[m];
        m += 1;
    }

    // Compile-time exact Vandermonde verification: T * V == denom * I_{R x R}.
    let mut row = 0usize;
    while row < R {
        let mut col = 0usize;
        while col < R {
            let mut acc = 0i128;
            let mut node = 0usize;
            while node < M {
                let mut x_pow = 1i128;
                let mut e = 0usize;
                while e < col {
                    x_pow *= xs[node] as i128;
                    e += 1;
                }
                acc += (table[row][node] as i128) * x_pow;
                node += 1;
            }
            if col == M {
                acc += table[row][M] as i128;
            }
            let expected = if row == col { denom as i128 } else { 0 };
            assert!(acc == expected);
            col += 1;
        }
        row += 1;
    }

    table
}

/// Degree-3 evaluation nodes `[0, 1, -1, 2, ∞]`; common denominator `3! = 6`.
const DENOM_D3: u8 = 6;
const UNREDUCED_D3: [[i64; 5]; 5] = build_unreduced_interp::<4, 5>([0, 1, -1, 2], 6);

/// Degree-4 evaluation nodes `[0, 1, -1, 2, -2, 3, ∞]`; common denominator `5! = 120`.
const DENOM_D4: u8 = 120;
const UNREDUCED_D4: [[i64; 7]; 7] = build_unreduced_interp::<6, 7>([0, 1, -1, 2, -2, 3], 120);

/// Degree-5 evaluation nodes `[0, 1, -1, 2, -2, 3, -3, 4, ∞]`; common denominator `7! = 5040`.
const DENOM_D5: u16 = 5040;
const UNREDUCED_D5: [[i64; 9]; 9] =
    build_unreduced_interp::<8, 9>([0, 1, -1, 2, -2, 3, -3, 4], 5040);

/// Degree-8 evaluation nodes `[0, ±1, ±2, ±3, ±4, ±5, ±6, 7, ∞]`; common denominator `13!`.
const DENOM_D8: u64 = 6_227_020_800;
const UNREDUCED_D8: [[i64; 15]; 15] = build_unreduced_interp::<14, 15>(
    [0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6, 7],
    6_227_020_800,
);

/// Folded numerators for `X^3 - X - 1 = 0` (`X^3 = 1 + X`, `X^4 = X + X^2`), denom `6`.
const CUBIC_TRINOMIAL_NUM: [[i64; 5]; 3] = {
    let u = UNREDUCED_D3;
    let mut out = [[0i64; 5]; 3];
    let mut r = 0;
    while r < 5 {
        out[0][r] = u[0][r] + u[3][r];
        out[1][r] = u[1][r] + u[3][r] + u[4][r];
        out[2][r] = u[2][r] + u[4][r];
        r += 1;
    }
    out
};

/// Folded numerators for `X^5 + X^2 - 1 = 0` (`X^5 = 1 - X^2`, `X^6 = X - X^3`,
/// `X^7 = X^2 - X^4`, `X^8 = -1 + X^2 + X^3`), denom `5040`.
const QUINTIC_TRINOMIAL_NUM: [[i64; 9]; 5] = {
    let u = UNREDUCED_D5;
    let mut out = [[0i64; 9]; 5];
    let mut r = 0;
    while r < 9 {
        out[0][r] = u[0][r] + u[5][r] - u[8][r];
        out[1][r] = u[1][r] + u[6][r];
        out[2][r] = u[2][r] - u[5][r] + u[7][r] + u[8][r];
        out[3][r] = u[3][r] - u[6][r] + u[8][r];
        out[4][r] = u[4][r] - u[7][r];
        r += 1;
    }
    out
};

/// Folded numerators for `QM31` (`F[i][u] / (i^2 + 1, u^2 - (2 + i))`) in flattened
/// basis `[1, i, u, iu]`, obtained from `u^4 - 4u^2 + 5 = 0` mapped back via
/// `u^0 = 1`, `u^1 = u`, `u^2 = 2 + i`, `u^3 = 2u + iu`, `u^4 = 3 + 4i`,
/// `u^5 = 3u + 4iu`, `u^6 = 2 + 11i`; denom `120`.
const QM31_INTERP_NUM: [[i64; 7]; 4] = {
    let h = UNREDUCED_D4;
    let mut out = [[0i64; 7]; 4];
    let mut r = 0;
    while r < 7 {
        out[0][r] = h[0][r] + 2 * h[2][r] + 3 * h[4][r] + 2 * h[6][r];
        out[1][r] = h[2][r] + 4 * h[4][r] + 11 * h[6][r];
        out[2][r] = h[1][r] + 2 * h[3][r] + 3 * h[5][r];
        out[3][r] = h[3][r] + 4 * h[5][r];
        r += 1;
    }
    out
};

// ---------------------------------------------------------------------------
// Shared even/odd polynomial evaluation helpers
// ---------------------------------------------------------------------------

#[inline]
fn pm_pair<F: Field, A: Algebra<F>>(even: A, odd: A) -> [A; 2] {
    A::linear_combinations::<2, 2>(&[even, odd], &[[F::ONE, F::ONE], [F::ONE, F::NEG_ONE]])
}

#[inline]
fn eval_d3<F: Field, A: Algebra<F>>(a: &[A; 3]) -> [A; 5] {
    let [e0, e1, e4, a2_inf] = A::linear_combinations::<2, 4>(
        &[a[0].dup(), a[2].dup()],
        &[
            [F::ONE, F::ZERO],
            [F::ONE, F::ONE],
            [F::ONE, F::from_u8(4)],
            [F::ZERO, F::ONE],
        ],
    );
    let [o1, o2] = A::linear_combinations::<1, 2>(&[a[1].dup()], &[[F::ONE], [F::TWO]]);
    let [u1, u2] = pm_pair::<F, A>(e1, o1);
    [e0, u1, u2, e4 + o2, a2_inf]
}

#[inline]
fn eval_d4<F: Field, A: Algebra<F>>(a: &[A; 4]) -> [A; 7] {
    let [e0, e1, e4, e9] = A::linear_combinations::<2, 4>(
        &[a[0].dup(), a[2].dup()],
        &[
            [F::ONE, F::ZERO],
            [F::ONE, F::ONE],
            [F::ONE, F::from_u8(4)],
            [F::ONE, F::from_u8(9)],
        ],
    );
    let [o1, o2, o3, o_inf] = A::linear_combinations::<2, 4>(
        &[a[1].dup(), a[3].dup()],
        &[
            [F::ONE, F::ONE],
            [F::TWO, F::from_u8(8)],
            [F::from_u8(3), F::from_u8(27)],
            [F::ZERO, F::ONE],
        ],
    );
    let [u1, u2] = pm_pair::<F, A>(e1, o1);
    let [u3, u4] = pm_pair::<F, A>(e4, o2);
    [e0, u1, u2, u3, u4, e9 + o3, o_inf]
}

#[inline]
fn eval_d5<F: Field, A: Algebra<F>>(a: &[A; 5]) -> [A; 9] {
    let [e0, e1, e4, e9, e16, e_inf] = A::linear_combinations::<3, 6>(
        &[a[0].dup(), a[2].dup(), a[4].dup()],
        &[
            [F::ONE, F::ZERO, F::ZERO],
            [F::ONE, F::ONE, F::ONE],
            [F::ONE, F::from_u8(4), F::from_u8(16)],
            [F::ONE, F::from_u8(9), F::from_u8(81)],
            [F::ONE, F::from_u8(16), F::from_u16(256)],
            [F::ZERO, F::ZERO, F::ONE],
        ],
    );
    let [o1, o2, o3, o4] = A::linear_combinations::<2, 4>(
        &[a[1].dup(), a[3].dup()],
        &[
            [F::ONE, F::ONE],
            [F::TWO, F::from_u8(8)],
            [F::from_u8(3), F::from_u8(27)],
            [F::from_u8(4), F::from_u8(64)],
        ],
    );
    let [u1, u2] = pm_pair::<F, A>(e1, o1);
    let [u3, u4] = pm_pair::<F, A>(e4, o2);
    let [u5, u6] = pm_pair::<F, A>(e9, o3);
    [e0, u1, u2, u3, u4, u5, u6, e16 + o4, e_inf]
}

#[inline]
fn eval_d8<F: Field, A: Algebra<F>>(a: &[A; 8]) -> [A; 15] {
    let even_weights: [[F; 4]; 8] = array::from_fn(|k| {
        if k == 0 {
            [F::ONE, F::ZERO, F::ZERO, F::ZERO]
        } else {
            let k2 = (k * k) as u32;
            let k4 = k2 * k2;
            let k6 = k4 * k2;
            [F::ONE, F::from_u32(k2), F::from_u32(k4), F::from_u32(k6)]
        }
    });
    let odd_weights: [[F; 4]; 8] = array::from_fn(|idx| {
        if idx == 7 {
            [F::ZERO, F::ZERO, F::ZERO, F::ONE]
        } else {
            let k = (idx + 1) as u32;
            let k2 = k * k;
            let k3 = k2 * k;
            let k5 = k3 * k2;
            let k7 = k5 * k2;
            [
                F::from_u32(k),
                F::from_u32(k3),
                F::from_u32(k5),
                F::from_u32(k7),
            ]
        }
    });
    let [e0, e1, e2, e3, e4, e5, e6, e7] = A::linear_combinations::<4, 8>(
        &[a[0].dup(), a[2].dup(), a[4].dup(), a[6].dup()],
        &even_weights,
    );
    let [o1, o2, o3, o4, o5, o6, o7, o_inf] = A::linear_combinations::<4, 8>(
        &[a[1].dup(), a[3].dup(), a[5].dup(), a[7].dup()],
        &odd_weights,
    );
    let [u1, u2] = pm_pair::<F, A>(e1, o1);
    let [u3, u4] = pm_pair::<F, A>(e2, o2);
    let [u5, u6] = pm_pair::<F, A>(e3, o3);
    let [u7, u8] = pm_pair::<F, A>(e4, o4);
    let [u9, u10] = pm_pair::<F, A>(e5, o5);
    let [u11, u12] = pm_pair::<F, A>(e6, o6);
    [
        e0,
        u1,
        u2,
        u3,
        u4,
        u5,
        u6,
        u7,
        u8,
        u9,
        u10,
        u11,
        u12,
        e7 + o7,
        o_inf,
    ]
}

#[inline]
fn binomial_weights<F: Field, const D: usize, const R: usize>(
    unreduced: &[[i64; R]; R],
    denom: i64,
    inv_denom: F,
    w: F,
) -> [[F; R]; D] {
    array::from_fn(|k| {
        array::from_fn(|r| {
            let n0 = unreduced[k][r];
            let n1 = if k + 1 < D { unreduced[D + k][r] } else { 0 };
            if n0 == 0 && n1 == 0 {
                F::ZERO
            } else if n1 == 0 && n0 == denom {
                F::ONE
            } else if n1 == 0 && n0 == -denom {
                F::NEG_ONE
            } else if n1 == 0 {
                F::from_i64(n0) * inv_denom
            } else if n0 == 0 && n1 == denom {
                w
            } else {
                (F::from_i64(n0) + w * F::from_i64(n1)) * inv_denom
            }
        })
    })
}

#[inline]
fn scaled_int_weights<F: Field, const D: usize, const R: usize>(
    numerators: &[[i64; R]; D],
    denom: i64,
    inv_denom: F,
) -> [[F; R]; D] {
    array::from_fn(|k| {
        array::from_fn(|r| {
            let n = numerators[k][r];
            if n == 0 {
                F::ZERO
            } else if n == denom {
                F::ONE
            } else if n == -denom {
                F::NEG_ONE
            } else if n == 2 * denom {
                F::TWO
            } else {
                F::from_i64(n) * inv_denom
            }
        })
    })
}

#[inline]
fn binomial_mul_fallback<F: Field, A: Algebra<F>, const D: usize>(
    a: &[A; D],
    b: &[A; D],
    res: &mut [A; D],
    w: F,
) {
    let mut out: [A; D] = array::from_fn(|_| A::ZERO);
    for (i, a_i) in a.iter().enumerate() {
        for (j, b_j) in b.iter().enumerate() {
            let prod = a_i.dup() * b_j.dup();
            if i + j >= D {
                out[i + j - D] += prod * w;
            } else {
                out[i + j] += prod;
            }
        }
    }
    res.clone_from_slice(&out);
}

// ---------------------------------------------------------------------------
// Public Minimum-Rank Extension Kernels
// ---------------------------------------------------------------------------

/// Minimum-rank `(2D - 1)` multiplication in the binomial extension `A[X] / (X^D - w)`.
///
/// Uses exact Winograd / Alder–Strassen evaluation-interpolation with `2D - 1`
/// algebra multiplications for `D ∈ {2, 3, 4, 5, 8}` (`3, 5, 7, 9, 15` `A * A`
/// products respectively, versus `4, 6, 16, 25, 64` in [`super::binomial_mul`]).
/// Falls back cleanly to schoolbook multiplication for other degrees or if the small
/// interpolation denominator is not invertible in `F`.
#[inline]
pub fn binomial_mul_min_rank<F: Field, A: Algebra<F>, const D: usize>(
    a: &[A; D],
    b: &[A; D],
    res: &mut [A; D],
    w: F,
) {
    match D {
        2 => {
            let a2: &[A; 2] = a[..].try_into().unwrap();
            let b2: &[A; 2] = b[..].try_into().unwrap();
            let eval_w = [[F::ONE, F::ZERO], [F::ZERO, F::ONE], [F::ONE, F::ONE]];
            let u = A::linear_combinations::<2, 3>(a2, &eval_w);
            let y = A::linear_combinations::<2, 3>(b2, &eval_w);
            let v: [A; 3] = array::from_fn(|i| u[i].dup() * y[i].dup());
            let out = A::linear_combinations::<3, 2>(
                &v,
                &[[F::ONE, w, F::ZERO], [F::NEG_ONE, F::NEG_ONE, F::ONE]],
            );
            res.clone_from_slice(&out);
        }
        3 => {
            let Some(inv6) = F::from_u8(DENOM_D3).try_inverse() else {
                binomial_mul_fallback(a, b, res, w);
                return;
            };
            let u = eval_d3::<F, A>(a[..].try_into().unwrap());
            let y = eval_d3::<F, A>(b[..].try_into().unwrap());
            let v: [A; 5] = array::from_fn(|i| u[i].dup() * y[i].dup());
            let weights = binomial_weights::<F, 3, 5>(&UNREDUCED_D3, i64::from(DENOM_D3), inv6, w);
            let out = A::linear_combinations::<5, 3>(&v, &weights);
            res.clone_from_slice(&out);
        }
        4 => {
            let Some(inv120) = F::from_u8(DENOM_D4).try_inverse() else {
                binomial_mul_fallback(a, b, res, w);
                return;
            };
            let u = eval_d4::<F, A>(a[..].try_into().unwrap());
            let y = eval_d4::<F, A>(b[..].try_into().unwrap());
            let v: [A; 7] = array::from_fn(|i| u[i].dup() * y[i].dup());
            let weights =
                binomial_weights::<F, 4, 7>(&UNREDUCED_D4, i64::from(DENOM_D4), inv120, w);
            let out = A::linear_combinations::<7, 4>(&v, &weights);
            res.clone_from_slice(&out);
        }
        5 => {
            let Some(inv5040) = F::from_u16(DENOM_D5).try_inverse() else {
                binomial_mul_fallback(a, b, res, w);
                return;
            };
            let u = eval_d5::<F, A>(a[..].try_into().unwrap());
            let y = eval_d5::<F, A>(b[..].try_into().unwrap());
            let v: [A; 9] = array::from_fn(|i| u[i].dup() * y[i].dup());
            let weights =
                binomial_weights::<F, 5, 9>(&UNREDUCED_D5, i64::from(DENOM_D5), inv5040, w);
            let out = A::linear_combinations::<9, 5>(&v, &weights);
            res.clone_from_slice(&out);
        }
        8 => {
            let Some(inv_d8) = F::from_u64(DENOM_D8).try_inverse() else {
                binomial_mul_fallback(a, b, res, w);
                return;
            };
            let u = eval_d8::<F, A>(a[..].try_into().unwrap());
            let y = eval_d8::<F, A>(b[..].try_into().unwrap());
            let v: [A; 15] = array::from_fn(|i| u[i].dup() * y[i].dup());
            let weights = binomial_weights::<F, 8, 15>(&UNREDUCED_D8, DENOM_D8 as i64, inv_d8, w);
            let out = A::linear_combinations::<15, 8>(&v, &weights);
            res.clone_from_slice(&out);
        }
        _ => binomial_mul_fallback(a, b, res, w),
    }
}

/// Minimum-rank squaring in the binomial extension `A[X] / (X^D - w)`.
///
/// - For `D = 2`, uses **only 2** `A * A` multiplications via the identity
///   `(a_0 + a_1)(a_0 + w a_1) - (1 + w) a_0 a_1 = a_0^2 + w a_1^2`.
/// - For `D ∈ {3, 4, 5, 8}`, uses `2D - 1` (`5, 7, 9, 15`) calls to [`crate::PrimeCharacteristicRing::square`].
/// - Falls back cleanly to [`binomial_square`] for other degrees or non-invertible denominators.
#[inline]
pub fn binomial_square_min_rank<F: Field, A: Algebra<F>, const D: usize>(
    a: &[A; D],
    res: &mut [A; D],
    w: F,
) {
    match D {
        2 => {
            let a2: &[A; 2] = a[..].try_into().unwrap();
            let [s_sum, s_w, a0, a1] = A::linear_combinations::<2, 4>(
                a2,
                &[
                    [F::ONE, F::ONE],
                    [F::ONE, w],
                    [F::ONE, F::ZERO],
                    [F::ZERO, F::ONE],
                ],
            );
            let v = [s_sum * s_w, a0 * a1];
            let out =
                A::linear_combinations::<2, 2>(&v, &[[F::ONE, -(F::ONE + w)], [F::ZERO, F::TWO]]);
            res.clone_from_slice(&out);
        }
        3 => {
            let Some(inv6) = F::from_u8(DENOM_D3).try_inverse() else {
                binomial_square(a, res, w);
                return;
            };
            let u = eval_d3::<F, A>(a[..].try_into().unwrap());
            let v: [A; 5] = array::from_fn(|i| u[i].square());
            let weights = binomial_weights::<F, 3, 5>(&UNREDUCED_D3, i64::from(DENOM_D3), inv6, w);
            let out = A::linear_combinations::<5, 3>(&v, &weights);
            res.clone_from_slice(&out);
        }
        4 => {
            let Some(inv120) = F::from_u8(DENOM_D4).try_inverse() else {
                binomial_square(a, res, w);
                return;
            };
            let u = eval_d4::<F, A>(a[..].try_into().unwrap());
            let v: [A; 7] = array::from_fn(|i| u[i].square());
            let weights =
                binomial_weights::<F, 4, 7>(&UNREDUCED_D4, i64::from(DENOM_D4), inv120, w);
            let out = A::linear_combinations::<7, 4>(&v, &weights);
            res.clone_from_slice(&out);
        }
        5 => {
            let Some(inv5040) = F::from_u16(DENOM_D5).try_inverse() else {
                binomial_square(a, res, w);
                return;
            };
            let u = eval_d5::<F, A>(a[..].try_into().unwrap());
            let v: [A; 9] = array::from_fn(|i| u[i].square());
            let weights =
                binomial_weights::<F, 5, 9>(&UNREDUCED_D5, i64::from(DENOM_D5), inv5040, w);
            let out = A::linear_combinations::<9, 5>(&v, &weights);
            res.clone_from_slice(&out);
        }
        8 => {
            let Some(inv_d8) = F::from_u64(DENOM_D8).try_inverse() else {
                binomial_square(a, res, w);
                return;
            };
            let u = eval_d8::<F, A>(a[..].try_into().unwrap());
            let v: [A; 15] = array::from_fn(|i| u[i].square());
            let weights = binomial_weights::<F, 8, 15>(&UNREDUCED_D8, DENOM_D8 as i64, inv_d8, w);
            let out = A::linear_combinations::<15, 8>(&v, &weights);
            res.clone_from_slice(&out);
        }
        _ => binomial_square(a, res, w),
    }
}

/// Minimum-rank (`5`-multiplication) kernel in the cubic trinomial extension `A[X] / (X^3 - X - 1)`.
///
/// Evaluates at `{0, 1, -1, 2, ∞}` and folds modulo `X^3 = X + 1`, reducing quadratic
/// `A * A` multiplications from `9` (in [`trinomial_cubic_mul`]) to `5` (`-44.44%`).
#[inline]
pub fn cubic_trinomial_mul_min_rank<F: Field, A: Algebra<F>>(
    a: &[A; 3],
    b: &[A; 3],
    res: &mut [A; 3],
) {
    let Some(inv6) = F::from_u8(DENOM_D3).try_inverse() else {
        trinomial_cubic_mul(a, b, res);
        return;
    };
    let u = eval_d3::<F, A>(a);
    let y = eval_d3::<F, A>(b);
    let v: [A; 5] = array::from_fn(|i| u[i].dup() * y[i].dup());
    let weights = scaled_int_weights::<F, 3, 5>(&CUBIC_TRINOMIAL_NUM, i64::from(DENOM_D3), inv6);
    *res = A::linear_combinations::<5, 3>(&v, &weights);
}

/// Minimum-rank (`5`-squaring) kernel in the cubic trinomial extension `A[X] / (X^3 - X - 1)`.
#[inline]
pub fn cubic_trinomial_square_min_rank<F: Field, A: Algebra<F>>(a: &[A; 3], res: &mut [A; 3]) {
    let Some(inv6) = F::from_u8(DENOM_D3).try_inverse() else {
        cubic_square(a, res);
        return;
    };
    let u = eval_d3::<F, A>(a);
    let v: [A; 5] = array::from_fn(|i| u[i].square());
    let weights = scaled_int_weights::<F, 3, 5>(&CUBIC_TRINOMIAL_NUM, i64::from(DENOM_D3), inv6);
    *res = A::linear_combinations::<5, 3>(&v, &weights);
}

/// Minimum-rank (`9`-multiplication) kernel in the quintic trinomial extension `A[X] / (X^5 + X^2 - 1)`.
///
/// Evaluates at `{0, ±1, ±2, ±3, 4, ∞}` and folds modulo `X^5 = 1 - X^2`, reducing
/// quadratic `A * A` multiplications from `25` (in [`trinomial_quintic_mul`]) to `9` (`-64.00%`).
#[inline]
pub fn quintic_trinomial_mul_min_rank<F: Field, A: Algebra<F>>(
    a: &[A; 5],
    b: &[A; 5],
    res: &mut [A; 5],
) {
    let Some(inv5040) = F::from_u16(DENOM_D5).try_inverse() else {
        trinomial_quintic_mul(a, b, res);
        return;
    };
    let u = eval_d5::<F, A>(a);
    let y = eval_d5::<F, A>(b);
    let v: [A; 9] = array::from_fn(|i| u[i].dup() * y[i].dup());
    let weights =
        scaled_int_weights::<F, 5, 9>(&QUINTIC_TRINOMIAL_NUM, i64::from(DENOM_D5), inv5040);
    *res = A::linear_combinations::<9, 5>(&v, &weights);
}

/// Minimum-rank (`9`-squaring) kernel in the quintic trinomial extension `A[X] / (X^5 + X^2 - 1)`.
#[inline]
pub fn quintic_trinomial_square_min_rank<F: Field, A: Algebra<F>>(a: &[A; 5], res: &mut [A; 5]) {
    let Some(inv5040) = F::from_u16(DENOM_D5).try_inverse() else {
        quintic_square(a, res);
        return;
    };
    let u = eval_d5::<F, A>(a);
    let v: [A; 9] = array::from_fn(|i| u[i].square());
    let weights =
        scaled_int_weights::<F, 5, 9>(&QUINTIC_TRINOMIAL_NUM, i64::from(DENOM_D5), inv5040);
    *res = A::linear_combinations::<9, 5>(&v, &weights);
}

#[inline]
fn eval_qm31<F: Field, A: Algebra<F>>(a: &[A; 4]) -> [A; 7] {
    // Forward unimodular map (a0 + a1*i) + (a2 + a3*i)*u -> (a0 - 2*a1) + (a2 - 2*a3)*u + a1*u^2 + a3*u^3
    // evaluated at [0, 1, -1, 2, -2, 3, inf].
    let [e0, e1, e2, e3] = A::linear_combinations::<2, 4>(
        &[a[0].dup(), a[1].dup()],
        &[
            [F::ONE, -F::TWO],
            [F::ONE, F::NEG_ONE],
            [F::ONE, F::TWO],
            [F::ONE, F::from_u8(7)],
        ],
    );
    let [o1, o2, o3, o_inf] = A::linear_combinations::<2, 4>(
        &[a[2].dup(), a[3].dup()],
        &[
            [F::ONE, F::NEG_ONE],
            [F::TWO, F::from_u8(4)],
            [F::from_u8(3), F::from_u8(21)],
            [F::ZERO, F::ONE],
        ],
    );
    let [u1, u2] = pm_pair::<F, A>(e1, o1);
    let [u3, u4] = pm_pair::<F, A>(e2, o2);
    [e0, u1, u2, u3, u4, e3 + o3, o_inf]
}

/// Minimum-rank (`7`-multiplication) kernel for the degree-4 complex tower
/// `A[i][u] / (i^2 + 1, u^2 - (2 + i))` in flattened basis `[1, i, u, iu]` (e.g. `QM31`).
///
/// Uses the unimodular isomorphism `i = u^2 - 2` to `A[u] / (u^4 - 4u^2 + 5)` evaluated
/// at `{0, 1, -1, 2, -2, 3, ∞}`, slashing quadratic `A * A` multiplications from `16`
/// (schoolbook) / `9` (nested Karatsuba) down to the theoretical minimum `7`.
/// Falls back to `9`-mul nested Karatsuba if `120` is not invertible in `F`.
#[inline]
pub fn qm31_mul_min_rank<F: Field, A: Algebra<F>>(a: &[A; 4], b: &[A; 4], res: &mut [A; 4]) {
    let Some(inv120) = F::from_u8(DENOM_D4).try_inverse() else {
        // Characteristic 2, 3, or 5 fallback: 9-mul nested Karatsuba.
        let three = F::from_u8(3);
        let [a0, a1, a2, a3, sa0, sa1, a01, a23, sa01] = A::linear_combinations::<4, 9>(
            a,
            &[
                [F::ONE, F::ZERO, F::ZERO, F::ZERO],
                [F::ZERO, F::ONE, F::ZERO, F::ZERO],
                [F::ZERO, F::ZERO, F::ONE, F::ZERO],
                [F::ZERO, F::ZERO, F::ZERO, F::ONE],
                [F::ONE, F::ZERO, F::ONE, F::ZERO],
                [F::ZERO, F::ONE, F::ZERO, F::ONE],
                [F::ONE, F::ONE, F::ZERO, F::ZERO],
                [F::ZERO, F::ZERO, F::ONE, F::ONE],
                [F::ONE, F::ONE, F::ONE, F::ONE],
            ],
        );
        let [b0, b1, b2, b3, sb0, sb1, b01, b23, sb01] = A::linear_combinations::<4, 9>(
            b,
            &[
                [F::ONE, F::ZERO, F::ZERO, F::ZERO],
                [F::ZERO, F::ONE, F::ZERO, F::ZERO],
                [F::ZERO, F::ZERO, F::ONE, F::ZERO],
                [F::ZERO, F::ZERO, F::ZERO, F::ONE],
                [F::ONE, F::ZERO, F::ONE, F::ZERO],
                [F::ZERO, F::ONE, F::ZERO, F::ONE],
                [F::ONE, F::ONE, F::ZERO, F::ZERO],
                [F::ZERO, F::ZERO, F::ONE, F::ONE],
                [F::ONE, F::ONE, F::ONE, F::ONE],
            ],
        );
        let v = [
            a0 * b0,
            a1 * b1,
            a01 * b01,
            a2 * b2,
            a3 * b3,
            a23 * b23,
            sa0 * sb0,
            sa1 * sb1,
            sa01 * sb01,
        ];
        *res = A::linear_combinations::<9, 4>(
            &v,
            &[
                [
                    F::ONE,
                    F::NEG_ONE,
                    F::ZERO,
                    three,
                    -F::ONE,
                    F::NEG_ONE,
                    F::ZERO,
                    F::ZERO,
                    F::ZERO,
                ],
                [
                    F::NEG_ONE,
                    F::NEG_ONE,
                    F::ONE,
                    -F::ONE,
                    -three,
                    F::TWO,
                    F::ZERO,
                    F::ZERO,
                    F::ZERO,
                ],
                [
                    F::NEG_ONE,
                    F::ONE,
                    F::ZERO,
                    F::NEG_ONE,
                    F::ONE,
                    F::ZERO,
                    F::ONE,
                    F::NEG_ONE,
                    F::ZERO,
                ],
                [
                    F::ONE,
                    F::ONE,
                    F::NEG_ONE,
                    F::ONE,
                    F::ONE,
                    F::NEG_ONE,
                    F::NEG_ONE,
                    F::NEG_ONE,
                    F::ONE,
                ],
            ],
        );
        return;
    };

    let u = eval_qm31::<F, A>(a);
    let y = eval_qm31::<F, A>(b);
    let v: [A; 7] = array::from_fn(|i| u[i].dup() * y[i].dup());
    let weights = scaled_int_weights::<F, 4, 7>(&QM31_INTERP_NUM, i64::from(DENOM_D4), inv120);
    *res = A::linear_combinations::<7, 4>(&v, &weights);
}

/// Symmetry-exploiting (`6`-multiplication) squaring kernel for the degree-4 complex tower
/// `A[i][u] / (i^2 + 1, u^2 - (2 + i))` in flattened basis `[1, i, u, iu]` (e.g. `QM31`).
///
/// Combines quadratic binomial squaring `(A_0 + A_1)(A_0 + W A_1) - (1 + W) A_0 A_1`
/// with 3-mul Karatsuba complex multiplication on `M_0 = (A_0 + A_1)(A_0 + W A_1)` and
/// `M_1 = A_0 A_1`, using **only 6** `A * A` multiplications in any characteristic.
#[inline]
pub fn qm31_square_min_rank<F: Field, A: Algebra<F>>(a: &[A; 4], res: &mut [A; 4]) {
    let three = F::from_u8(3);
    let four = F::from_u8(4);
    let [x0, x1, x01, y0, y1, y01, a0, a1, a2, a3, a01, a23] = A::linear_combinations::<4, 12>(
        a,
        &[
            [F::ONE, F::ZERO, F::ONE, F::ZERO],
            [F::ZERO, F::ONE, F::ZERO, F::ONE],
            [F::ONE, F::ONE, F::ONE, F::ONE],
            [F::ONE, F::ZERO, F::TWO, F::NEG_ONE],
            [F::ZERO, F::ONE, F::ONE, F::TWO],
            [F::ONE, F::ONE, three, F::ONE],
            [F::ONE, F::ZERO, F::ZERO, F::ZERO],
            [F::ZERO, F::ONE, F::ZERO, F::ZERO],
            [F::ZERO, F::ZERO, F::ONE, F::ZERO],
            [F::ZERO, F::ZERO, F::ZERO, F::ONE],
            [F::ONE, F::ONE, F::ZERO, F::ZERO],
            [F::ZERO, F::ZERO, F::ONE, F::ONE],
        ],
    );
    let v = [x0 * y0, x1 * y1, x01 * y01, a0 * a2, a1 * a3, a01 * a23];
    *res = A::linear_combinations::<6, 4>(
        &v,
        &[
            [F::ONE, F::NEG_ONE, F::ZERO, -four, F::TWO, F::ONE],
            [F::NEG_ONE, F::NEG_ONE, F::ONE, F::TWO, four, -three],
            [F::ZERO, F::ZERO, F::ZERO, F::TWO, -F::TWO, F::ZERO],
            [F::ZERO, F::ZERO, F::ZERO, -F::TWO, -F::TWO, F::TWO],
        ],
    );
}
