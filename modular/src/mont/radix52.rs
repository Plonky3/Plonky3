//! Two-limb residues in radix `2^52`, and a scalar model of the IFMA kernels, lane by lane.

/// `2^52 - 1`.
pub(crate) const MASK: u64 = (1 << 52) - 1;

/// The three digits of `x 2^shift`, for `x < 2^127` and `shift` 0 or 28.
pub(crate) const fn digits(x: [u64; 2], shift: u32) -> [u64; 3] {
    let [lo, hi] = x;
    if shift == 0 {
        [lo & MASK, (lo >> 52 | hi << 12) & MASK, hi >> 40]
    } else {
        [(lo << 28) & MASK, (lo >> 24 | hi << 40) & MASK, hi >> 12]
    }
}

#[cfg(test)]
pub(crate) mod model {
    use num_bigint::BigUint;

    use super::{MASK, digits};
    use crate::mont::MontModulus;
    use crate::mont::portable::Wide;
    use crate::ring::ModRing;
    use crate::uint::Uint;

    /// `acc + low 52 bits of (a b)`, on the low 52 bits of each source.
    fn madd_lo(acc: u64, a: u64, b: u64) -> u64 {
        acc.wrapping_add(((a & MASK) as u128 * (b & MASK) as u128) as u64 & MASK)
    }

    /// `acc + bits 52..104 of (a b)`, on the low 52 bits of each source.
    fn madd_hi(acc: u64, a: u64, b: u64) -> u64 {
        acc.wrapping_add((((a & MASK) as u128 * (b & MASK) as u128) >> 52) as u64)
    }

    /// The vector `mont_mul` on one lane: `a b 2^-156 mod q`, canonical.
    fn mont(a: [u64; 3], b: [u64; 3], ring: &MontModulus<2>) -> [u64; 2] {
        let q = digits(ring.modulus_uint().0, 0);
        let q_inv = ring.m_neg_inv & MASK;
        let mut t = [0u64; 6];
        for i in 0..3 {
            for j in 0..3 {
                t[i + j] = madd_lo(t[i + j], a[i], b[j]);
                t[i + j + 1] = madd_hi(t[i + j + 1], a[i], b[j]);
            }
        }
        for i in 0..3 {
            let u = madd_lo(0, t[i], q_inv);
            for j in 0..3 {
                t[i + j] = madd_lo(t[i + j], u, q[j]);
                t[i + j + 1] = madd_hi(t[i + j + 1], u, q[j]);
            }
            assert_eq!(t[i] & MASK, 0, "a REDC round leaves a low digit");
            t[i + 1] += t[i] >> 52;
        }
        let r1 = t[4] + (t[3] >> 52);
        let r = [t[3] & MASK, r1 & MASK, t[5] + (r1 >> 52)];
        sub_if_ge(r, q)
    }

    /// The vector `sub_if_ge` on one lane.
    fn sub_if_ge(r: [u64; 3], q: [u64; 3]) -> [u64; 2] {
        let d0 = r[0].wrapping_sub(q[0]);
        let d1 = r[1]
            .wrapping_sub(q[1])
            .wrapping_add(((d0 as i64) >> 52) as u64);
        let d2 = r[2]
            .wrapping_sub(q[2])
            .wrapping_add(((d1 as i64) >> 52) as u64);
        let s = if (d2 as i64) < 0 {
            r
        } else {
            [d0 & MASK, d1 & MASK, d2]
        };
        [s[0] | s[1] << 52, s[1] >> 12 | s[2] << 40]
    }

    /// The vector `mul_vec` on one lane.
    pub(crate) fn mul(a: [u64; 2], b: [u64; 2], ring: &MontModulus<2>) -> [u64; 2] {
        mont(digits(a, 0), digits(b, 28), ring)
    }

    /// The vector `scale_add` on one lane.
    pub(crate) fn scale_add(
        acc: [u64; 2],
        x: [u64; 2],
        s: [u64; 2],
        ring: &MontModulus<2>,
    ) -> [u64; 2] {
        let p = mont(digits(x, 0), digits(s, 28), ring);
        let (a, b) = (digits(acc, 0), digits(p, 0));
        let s0 = a[0] + b[0];
        let s1 = a[1] + b[1] + (s0 >> 52);
        let s2 = a[2] + b[2] + (s1 >> 52);
        sub_if_ge([s0 & MASK, s1 & MASK, s2], digits(ring.modulus_uint().0, 0))
    }

    /// The vector `eq_step` on one lane: `(lo - lo x, lo x)`.
    pub(crate) fn eq_step(
        lo: [u64; 2],
        x: [u64; 2],
        ring: &MontModulus<2>,
    ) -> ([u64; 2], [u64; 2]) {
        let p = mont(digits(lo, 0), digits(x, 28), ring);
        let (a, b, q) = (
            digits(lo, 0),
            digits(p, 0),
            digits(ring.modulus_uint().0, 0),
        );
        let s0 = (a[0] + q[0]).wrapping_sub(b[0]);
        let s1 = (a[1] + q[1])
            .wrapping_sub(b[1])
            .wrapping_add(((s0 as i64) >> 52) as u64);
        let s2 = (a[2] + q[2])
            .wrapping_sub(b[2])
            .wrapping_add(((s1 as i64) >> 52) as u64);
        (sub_if_ge([s0 & MASK, s1 & MASK, s2], q), p)
    }

    /// The vector `dot`: the exact column sums, recombined.
    pub(crate) fn dot(a: &[crate::Residue<2>], b: &[crate::Residue<2>]) -> Wide<2> {
        let mut cols = [0u128; 6];
        for (x, y) in a.iter().zip(b) {
            let (x, y) = (digits(x.0.0, 0), digits(y.0.0, 0));
            for i in 0..3 {
                for j in 0..3 {
                    let p = x[i] as u128 * y[j] as u128;
                    cols[i + j] += p & MASK as u128;
                    cols[i + j + 1] += p >> 52;
                }
            }
        }
        let total = cols.iter().enumerate().fold(BigUint::ZERO, |acc, (k, &c)| {
            acc + (BigUint::from(c) << (52 * k))
        });
        let mut w = total.to_u64_digits();
        w.resize(5, 0);
        Wide {
            low: Uint([w[0], w[1]]),
            high: Uint([w[2], w[3]]),
            top: w[4],
        }
    }

    mod tests {
        use alloc::vec::Vec;

        use proptest::prelude::*;

        use super::*;
        use crate::Residue;

        fn ring(m: [u64; 2], top: bool) -> MontModulus<2> {
            let mut m = Uint(m);
            m.0[1] &= u64::MAX >> 1;
            m.0[0] |= 1;
            if top {
                m.0[1] |= 1 << 62;
            }
            if m.bits() < 2 {
                m.0[0] = 3;
            }
            MontModulus::new(m).expect("odd")
        }

        proptest! {
            #[test]
            fn model_matches_the_scalar_kernels(m: [u64; 2], top: bool, xs: Vec<([u64; 2], [u64; 2])>) {
                let ring = ring(m, top);
                let mut top_minus_1 = *ring.modulus_uint();
                top_minus_1.0[0] -= 1;
                let mut a: Vec<Residue<2>> = xs.iter().map(|(x, _)| ring.reduce_limbs(x)).collect();
                let mut b: Vec<Residue<2>> = xs.iter().map(|(_, y)| ring.reduce_limbs(y)).collect();
                // The largest stored value loads every digit.
                a.push(Residue(top_minus_1));
                b.push(Residue(top_minus_1));
                for (&x, &y) in a.iter().zip(&b) {
                    prop_assert_eq!(mul(x.0.0, y.0.0, &ring), ring.mul(x, y).0.0);
                    prop_assert_eq!(scale_add(x.0.0, y.0.0, b[0].0.0, &ring), ring.add(x, ring.mul(b[0], y)).0.0);
                    let h = ring.mul(x, y);
                    prop_assert_eq!(eq_step(x.0.0, y.0.0, &ring), (ring.sub(x, h).0.0, h.0.0));
                }
                let want = a.iter().zip(&b).fold(Wide::ZERO, |mut acc, (x, y)| {
                    acc.add_product(&x.0, &y.0);
                    acc
                });
                let got = dot(&a, &b);
                prop_assert_eq!((got.low, got.high, got.top), (want.low, want.high, want.top));
            }
        }
    }
}
