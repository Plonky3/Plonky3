//! The product sumcheck `sum_i comb[i] z[i]`, binding the top variable first.

use alloc::vec::Vec;

use p3_field::PrimeCharacteristicRing;
use p3_maybe_rayon::prelude::*;

use crate::Challenge;

/// Quarter length from which a pass fans out to the pool.
const PAR_THRESHOLD: usize = 1 << 12;

/// `(q(1), q(inf))` of a round, summed over indices.
type Pair = (Challenge, Challenge);

/// One index of the two written quarters, weights then witness.
type Written<'a> = (
    (&'a mut Challenge, &'a mut Challenge),
    (&'a mut Challenge, &'a mut Challenge),
);

/// One index of the two read quarters, weights then witness.
type Read<'a> = (
    (&'a Challenge, &'a Challenge),
    (&'a Challenge, &'a Challenge),
);

/// Coefficient-wise sum of two round pairs.
fn add(x: Pair, y: Pair) -> Pair {
    (x.0 + y.0, x.1 + y.1)
}

/// The two tables of a product sumcheck, halving per bound variable.
#[derive(Clone, Debug)]
pub(crate) struct ProductSumcheck {
    /// The batched column weights.
    comb: Vec<Challenge>,
    /// The witness, folded over the instances.
    z: Vec<Challenge>,
}

impl ProductSumcheck {
    /// The sumcheck of `sum_i comb[i] z[i]`.
    ///
    /// # Panics
    ///
    /// - The tables differ in length, or are not a power of two.
    pub(crate) fn new(comb: Vec<Challenge>, z: Vec<Challenge>) -> Self {
        assert_eq!(comb.len(), z.len(), "one weight per witness value");
        assert!(z.len().is_power_of_two(), "a hypercube table");
        Self { comb, z }
    }

    /// The whole sum.
    pub(crate) fn claim(&self) -> Challenge {
        self.comb.iter().zip(&self.z).map(|(&c, &z)| c * z).sum()
    }

    /// The current round's `(q(1), q(inf))`, splitting both tables at their top variable.
    pub(crate) fn round(&self) -> Pair {
        let half = self.z.len() / 2;
        let (c_lo, c_hi) = self.comb.split_at(half);
        let (z_lo, z_hi) = self.z.split_at(half);
        let term =
            |(((&cl, &ch), &zl), &zh): (((&Challenge, &Challenge), &Challenge), &Challenge)| {
                (ch * zh, (cl + ch) * (zl + zh))
            };
        let zero = || (Challenge::ZERO, Challenge::ZERO);
        if half < PAR_THRESHOLD {
            return c_lo
                .iter()
                .zip(c_hi)
                .zip(z_lo)
                .zip(z_hi)
                .map(term)
                .fold(zero(), add);
        }
        c_lo.par_iter()
            .zip(c_hi.par_iter())
            .zip(z_lo.par_iter())
            .zip(z_hi.par_iter())
            .par_fold_reduce(zero, |acc, x| add(acc, term(x)), add)
    }

    /// Bind the top variable at `r`, and return the next round's `(q(1), q(inf))` from the same pass.
    ///
    /// Quarters `q0 q1 | q2 q3`: `(q0, q2)` bind to the next low half, `(q1, q3)` to the next high half.
    ///
    /// # Panics
    ///
    /// - Fewer than four values remain.
    pub(crate) fn bind_and_round(&mut self, r: Challenge) -> Pair {
        let len = self.z.len();
        let (half, quarter) = (len / 2, len / 4);
        assert!(quarter >= 1, "a next round needs two values after binding");

        // The low half is written, the high half only read.
        let (c_lo, c_hi) = self.comb.split_at_mut(half);
        let (c0, c1) = c_lo.split_at_mut(quarter);
        let (c2, c3) = c_hi.split_at(quarter);
        let (z_lo, z_hi) = self.z.split_at_mut(half);
        let (z0, z1) = z_lo.split_at_mut(quarter);
        let (z2, z3) = z_hi.split_at(quarter);

        // Bind one index of every quarter and return its two next-round terms.
        let step = |(((c0, c1), (z0, z1)), ((&c2, &c3), (&z2, &z3))): (Written<'_>, Read<'_>)| {
            let lo = *c0 + r * (c2 + *c0);
            let hi = *c1 + r * (c3 + *c1);
            let zl = *z0 + r * (z2 + *z0);
            let zh = *z1 + r * (z3 + *z1);
            (*c0, *c1, *z0, *z1) = (lo, hi, zl, zh);
            (hi * zh, (hi + lo) * (zh + zl))
        };
        let zero = || (Challenge::ZERO, Challenge::ZERO);
        let next = if quarter < PAR_THRESHOLD {
            c0.iter_mut()
                .zip(c1.iter_mut())
                .zip(z0.iter_mut().zip(z1.iter_mut()))
                .zip(c2.iter().zip(c3).zip(z2.iter().zip(z3)))
                .map(step)
                .fold(zero(), add)
        } else {
            c0.par_iter_mut()
                .zip(c1.par_iter_mut())
                .zip(z0.par_iter_mut().zip(z1.par_iter_mut()))
                .zip(
                    c2.par_iter()
                        .zip(c3.par_iter())
                        .zip(z2.par_iter().zip(z3.par_iter())),
                )
                .par_fold_reduce(zero, |acc, x| add(acc, step(x)), add)
        };
        self.comb.truncate(half);
        self.z.truncate(half);
        next
    }

    /// Bind the last variable at `r`, in the witness only, since the weights are never read again.
    pub(crate) fn bind_last(&mut self, r: Challenge) {
        let half = self.z.len() / 2;
        let (lo, hi) = self.z.split_at_mut(half);
        for (l, &h) in lo.iter_mut().zip(hi.iter()) {
            *l += r * (h + *l);
        }
        self.z.truncate(half);
        self.comb = Vec::new();
    }

    /// The witness as it stands.
    pub(crate) fn into_z(self) -> Vec<Challenge> {
        self.z
    }
}

#[cfg(test)]
mod tests {
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::*;

    /// One top-variable bind of a table, the definition.
    fn bind_top(v: &[Challenge], r: Challenge) -> Vec<Challenge> {
        let (lo, hi) = v.split_at(v.len() / 2);
        lo.iter().zip(hi).map(|(&l, &h)| l + r * (l + h)).collect()
    }

    #[test]
    fn the_fused_bind_is_a_bind_then_a_round() {
        // Invariant: binding and the next round in one pass equal a plain bind followed by a plain round.
        //
        // Fixture state: tables below and above the parallel threshold, down to the last round.
        let mut rng = SmallRng::seed_from_u64(0x5C0B);
        for log in [2, 5, 15] {
            let random = |rng: &mut SmallRng| {
                (0..1 << log)
                    .map(|_| rng.random())
                    .collect::<Vec<Challenge>>()
            };
            let (mut comb, mut z) = (random(&mut rng), random(&mut rng));
            let mut fused = ProductSumcheck::new(comb.clone(), z.clone());
            for _ in 0..log - 1 {
                let r: Challenge = rng.random();
                let next = fused.bind_and_round(r);
                (comb, z) = (bind_top(&comb, r), bind_top(&z, r));
                let plain = ProductSumcheck::new(comb.clone(), z.clone());
                assert_eq!(next, plain.round(), "log={log}, len={}", z.len());
                assert_eq!(fused.claim(), plain.claim(), "log={log}, len={}", z.len());
            }
            let r: Challenge = rng.random();
            fused.bind_last(r);
            assert_eq!(fused.into_z(), bind_top(&z, r), "log={log}");
        }
    }

    #[test]
    fn a_round_splits_the_claim() {
        // Invariant: q(0) + q(1) is the claim, with q(0) the low halves' product sum.
        let mut rng = SmallRng::seed_from_u64(0x5C0C);
        for log in [1, 13, 14] {
            let random = |rng: &mut SmallRng| {
                (0..1 << log)
                    .map(|_| rng.random())
                    .collect::<Vec<Challenge>>()
            };
            let s = ProductSumcheck::new(random(&mut rng), random(&mut rng));
            let (q1, qinf) = s.round();
            let half = 1 << (log - 1);
            let q0: Challenge = (0..half).map(|i| s.comb[i] * s.z[i]).sum();
            assert_eq!(q0 + q1, s.claim());
            let sums: Challenge = (0..half)
                .map(|i| (s.comb[i] + s.comb[half + i]) * (s.z[i] + s.z[half + i]))
                .sum();
            assert_eq!(qinf, sums);
        }
    }
}
