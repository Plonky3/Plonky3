//! Scalar Montgomery kernels on 64-bit limbs.
//!
//! Every kernel assumes `2m < 2^(64 L)`, which keeps each intermediate below `2m` without an extra carry limb.

use crate::uint::{Uint, mac};

/// `t - m` when `t >= m`, for `t < 2m`.
#[inline(always)]
pub(crate) const fn sub_if_ge<const L: usize>(t: Uint<L>, m: &Uint<L>) -> Uint<L> {
    let (d, borrow) = t.sbb(m);
    if borrow { t } else { d }
}

/// `a b 2^(-64 L) mod m`, canonical, for `a < 2^(64 L)` and `b < m`.
///
/// The full product, then `L` REDC rounds; two limbs take a dedicated schedule.
#[inline(always)]
pub(crate) const fn mul<const L: usize>(
    a: &Uint<L>,
    b: &Uint<L>,
    m: &Uint<L>,
    m_neg_inv: u64,
) -> Uint<L> {
    if L == 2 {
        let r = mul2(
            [a.0[0], a.0[1]],
            [b.0[0], b.0[1]],
            [m.0[0], m.0[1]],
            m_neg_inv,
        );
        let mut out = [0; L];
        out[0] = r[0];
        out[1] = r[1];
        return Uint(out);
    }
    let (lo, hi) = a.mul_wide(b);
    let mut lo = lo.0;
    let mut hi = hi.0;
    // t + sum_i u_i m 2^(64 i) < m 2^(64 L) + m 2^(64 L) < 2^(128 L), so the rounds never carry out.
    let mut pending = 0;
    let mut i = 0;
    while i < L {
        let u = lo[i].wrapping_mul(m_neg_inv);
        let mut carry = 0;
        let mut j = 0;
        while j < L {
            let k = i + j;
            if k < L {
                (lo[k], carry) = mac(lo[k], u, m.0[j], carry);
            } else {
                (hi[k - L], carry) = mac(hi[k - L], u, m.0[j], carry);
            }
            j += 1;
        }
        let (s, c) = crate::uint::adc(hi[i], carry, pending);
        hi[i] = s;
        pending = c;
        i += 1;
    }
    sub_if_ge(Uint(hi), m)
}

/// The two-limb product: the full 256-bit product, then two REDC rounds.
#[inline(always)]
pub(crate) const fn mul2(a: [u64; 2], b: [u64; 2], m: [u64; 2], m_neg_inv: u64) -> [u64; 2] {
    let p00 = a[0] as u128 * b[0] as u128;
    let p01 = a[0] as u128 * b[1] as u128;
    let p10 = a[1] as u128 * b[0] as u128;
    let p11 = a[1] as u128 * b[1] as u128;
    let mid = (p00 >> 64) + (p01 as u64) as u128 + (p10 as u64) as u128;
    // The product is below 2^256, so the high half fits 128 bits.
    let high = (mid >> 64) + (p01 >> 64) + (p10 >> 64) + p11;
    redc2([p00 as u64, mid as u64], high, m, m_neg_inv)
}

/// `a^2 2^-128 mod m`, sharing the cross product.
#[inline(always)]
pub(crate) const fn square2(a: [u64; 2], m: [u64; 2], m_neg_inv: u64) -> [u64; 2] {
    let p00 = a[0] as u128 * a[0] as u128;
    let p01 = a[0] as u128 * a[1] as u128;
    let p11 = a[1] as u128 * a[1] as u128;
    let cross_lo = (p01 as u64) as u128;
    let mid = (p00 >> 64) + cross_lo + cross_lo;
    let high = (mid >> 64) + ((p01 >> 64) << 1) + p11;
    redc2([p00 as u64, mid as u64], high, m, m_neg_inv)
}

/// `(low + high 2^128) 2^-128 mod m`, canonical, for an input below `m 2^128`.
#[inline(always)]
const fn redc2(low: [u64; 2], high: u128, m: [u64; 2], m_neg_inv: u64) -> [u64; 2] {
    let [t0, t1] = low;
    let u0 = t0.wrapping_mul(m_neg_inv);
    let c = (u0 as u128 * m[0] as u128 + t0 as u128) >> 64;
    let y = u0 as u128 * m[1] as u128 + t1 as u128 + c;
    // t + u0 m < m 2^128 + m 2^64 < 2^256, so the high half does not overflow.
    let z = high + (y >> 64);
    let t1 = y as u64;
    let u1 = t1.wrapping_mul(m_neg_inv);
    let c = (u1 as u128 * m[0] as u128 + t1 as u128) >> 64;
    let y = u1 as u128 * m[1] as u128 + (z as u64) as u128 + c;
    // The quotient is below 2m < 2^128.
    let r = ((z >> 64) + (y >> 64)) << 64 | (y as u64) as u128;
    let q = m[0] as u128 | (m[1] as u128) << 64;
    let (d, borrow) = r.overflowing_sub(q);
    let r = if borrow { r } else { d };
    [r as u64, (r >> 64) as u64]
}

/// An exact sum of products, `low + high 2^(64 L) + top 2^(128 L)`.
///
/// The top limb counts carries, so fewer than `2^64` products never overflow.
#[derive(Clone, Copy)]
pub(crate) struct Wide<const L: usize> {
    /// Limbs `0..L`.
    pub(crate) low: Uint<L>,
    /// Limbs `L..2L`.
    pub(crate) high: Uint<L>,
    /// Limb `2L`.
    pub(crate) top: u64,
}

impl<const L: usize> Wide<L> {
    /// The empty sum.
    pub(crate) const ZERO: Self = Self {
        low: Uint::ZERO,
        high: Uint::ZERO,
        top: 0,
    };

    /// Adds `a b`.
    #[inline(always)]
    pub(crate) const fn add_product(&mut self, a: &Uint<L>, b: &Uint<L>) {
        if L == 2 {
            let mut w = [
                self.low.0[0],
                self.low.0[1],
                self.high.0[0],
                self.high.0[1],
                self.top,
            ];
            mac2(&mut w, [a.0[0], a.0[1]], [b.0[0], b.0[1]]);
            self.low.0[0] = w[0];
            self.low.0[1] = w[1];
            self.high.0[0] = w[2];
            self.high.0[1] = w[3];
            self.top = w[4];
            return;
        }
        let (lo, hi) = a.mul_wide(b);
        self.add(&Self {
            low: lo,
            high: hi,
            top: 0,
        });
    }

    /// Adds another sum.
    #[inline(always)]
    pub(crate) const fn add(&mut self, rhs: &Self) {
        let (low, c) = self.low.adc(&rhs.low);
        let (high, c1) = self.high.adc(&rhs.high);
        let (high, c2) = high.adc(&Uint::from_u64(c as u64));
        self.low = low;
        self.high = high;
        self.top = self
            .top
            .wrapping_add(rhs.top)
            .wrapping_add(c1 as u64 + c2 as u64);
    }
}

/// `w += a b` on five limbs, finishing both product rows before the carry walks the top.
#[inline(always)]
const fn mac2(w: &mut [u64; 5], a: [u64; 2], b: [u64; 2]) {
    let (w0, c00) = mac(w[0], a[0], b[0], 0);
    let (r01, c01) = mac(w[1], a[0], b[1], c00);
    let (w1, c10) = mac(r01, a[1], b[0], 0);
    let col2 = w[2] as u128 + c01 as u128;
    let (w2, c11) = mac(col2 as u64, a[1], b[1], c10);
    let col3 = w[3] as u128 + c11 as u128 + (col2 >> 64);
    w[0] = w0;
    w[1] = w1;
    w[2] = w2;
    w[3] = col3 as u64;
    w[4] = w[4].wrapping_add((col3 >> 64) as u64);
}

/// `m^-1 mod 2^64` negated, by Newton iteration from the odd low limb.
pub(crate) const fn neg_inv(m0: u64) -> u64 {
    let mut inv = 1u64;
    let mut i = 0;
    while i < 6 {
        inv = inv.wrapping_mul(2u64.wrapping_sub(m0.wrapping_mul(inv)));
        i += 1;
    }
    inv.wrapping_neg()
}

/// `2x mod m` for `x < m`.
pub(crate) const fn double<const L: usize>(x: &Uint<L>, m: &Uint<L>) -> Uint<L> {
    let (s, _) = x.adc(x);
    sub_if_ge(s, m)
}

/// `a + b mod m` for `a, b < m`.
#[inline(always)]
pub(crate) const fn add<const L: usize>(a: &Uint<L>, b: &Uint<L>, m: &Uint<L>) -> Uint<L> {
    // a + b < 2m < 2^(64 L), so the sum has no carry out.
    let (s, _) = a.adc(b);
    sub_if_ge(s, m)
}

/// `a - b mod m` for `a, b < m`.
#[inline(always)]
pub(crate) const fn sub<const L: usize>(a: &Uint<L>, b: &Uint<L>, m: &Uint<L>) -> Uint<L> {
    let (d, borrow) = a.sbb(b);
    if borrow { d.adc(m).0 } else { d }
}

/// `x / 2 mod m` for `x < m`.
pub(crate) const fn half<const L: usize>(x: &Uint<L>, m: &Uint<L>) -> Uint<L> {
    // x + m < 2^(64 L) by the spare bit.
    if x.is_odd() {
        x.adc(m).0.shr(1)
    } else {
        x.shr(1)
    }
}
