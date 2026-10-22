//! The multilinear rounds after the skip: `sum_x eq(r, x) (a(x) b(x) + c(x))`, `a, b, c` folded at `z`.
//!
//! Round `t` binds the lowest unbound variable: positions `2k` and `2k + 1` are its `X = 0` and `X = 1`.
//!
//! A kernel returns `(G(1), G(inf))` of the round's inner polynomial; the claim fixes `G(0)`.
//!
//! Every pass sends two rounds, the second as a quadratic in the first's challenge.
//!
//! - Bit passes fold the packed bits on the fly while a stored table would outweigh them.
//! - The last bit pass also stores the folded tables.
//! - Table passes fold the pending challenges into the stored tables, then send two rounds.
//!
//! Each round is the Gruen split of ePrint 2024/108: the eq factor of the bound variable leaves the sum.

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512bw",
    target_feature = "avx512vbmi",
    target_feature = "gfni",
    target_feature = "vpclmulqdq"
))]
mod x86_64_avx512;

use alloc::boxed::Box;
use core::mem::MaybeUninit;

use p3_binary_field::byteslice::BLOCK;
use p3_binary_field::{Poly192, Poly192Unreduced};
use p3_field::PrimeCharacteristicRing;
use p3_maybe_rayon::prelude::*;
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;
use p3_util::log2_strict_usize;

use super::fold::BitFold;
use super::{EqHalves, Padding};
use crate::stream::Stream;

/// Eq variables in the per-task half of a bit pass's split eq table.
const EQ_LO_VARS: usize = 10;

/// Most variables of a table pass's high eq table, one task per high index.
const EQ_HIGH_VARS: usize = 7;

/// Uninitialized output tables, one slot per position.
pub(crate) type Outs<'a> = [&'a mut [MaybeUninit<Poly192>]; 3];

/// The packed `a` and `b` bits; `c = a AND b` is derived, never read.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Bits<'a> {
    /// The `A z` bits.
    pub(crate) a: &'a [u8],
    /// The `B z` bits.
    pub(crate) b: &'a [u8],
}

impl<'a> Bits<'a> {
    /// Both tables cut into rows of `CHUNKS` bytes, one per position.
    fn rows<const CHUNKS: usize>(self) -> [&'a [[u8; CHUNKS]]; 2] {
        let rows = [self.a, self.b].map(|packed| {
            let (rows, rest) = packed.as_chunks::<CHUNKS>();
            assert!(rest.is_empty(), "packed bits are whole rows");
            rows
        });
        assert_eq!(rows[0].len(), rows[1].len(), "a and b have one length");
        assert!(
            rows[0].len().is_power_of_two(),
            "a power of two of positions"
        );
        rows
    }
}

/// The `c = a AND b` rows of up to 64 positions.
#[inline(always)]
fn and_rows<const CHUNKS: usize>(
    a: &[[u8; CHUNKS]],
    b: &[[u8; CHUNKS]],
    c: &mut [[u8; CHUNKS]; BLOCK],
) {
    for ((c, a), b) in c
        .as_flattened_mut()
        .iter_mut()
        .zip(a.as_flattened())
        .zip(b.as_flattened())
    {
        *c = a & b;
    }
}

/// The folded `a`, `b`, `c` values of up to 64 consecutive positions.
struct FoldedBlock {
    /// Folded `a`.
    a: [Poly192; BLOCK],
    /// Folded `b`.
    b: [Poly192; BLOCK],
    /// Folded `c`.
    c: [Poly192; BLOCK],
}

impl FoldedBlock {
    /// Zero values.
    const ZERO: Self = Self {
        a: [Poly192::ZERO; BLOCK],
        b: [Poly192::ZERO; BLOCK],
        c: [Poly192::ZERO; BLOCK],
    };

    /// Fold positions `first..first + len` into the first `len` values.
    #[inline(always)]
    fn fold<const CHUNKS: usize>(
        &mut self,
        fold: &BitFold,
        rows: [&[[u8; CHUNKS]]; 2],
        first: usize,
        len: usize,
    ) {
        let [a, b] = rows.map(|r| &r[first..first + len]);
        let mut c = [[0u8; CHUNKS]; BLOCK];
        and_rows(a, b, &mut c);
        fold.fold_block(a, &mut self.a);
        fold.fold_block(b, &mut self.b);
        fold.fold_block(&c[..len], &mut self.c);
    }
}

/// Which groups of positions can hold a nonzero bit, from the zero padding of every block.
///
/// Group `g` of `2^group_log` bits is wholly padding iff `(g & mask) >= live`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct LiveGroups {
    /// Masks a group index to its group within a block.
    mask: usize,
    /// Groups of a block before its padding.
    live: usize,
}

impl LiveGroups {
    /// The live groups of `2^group_log` bits.
    const fn new(padding: &Padding, group_log: usize) -> Self {
        let every = Self {
            mask: 0,
            live: usize::MAX,
        };
        // A group covering whole blocks holds data whenever its blocks do.
        if padding.k_log <= group_log {
            return every;
        }
        let per_block = 1usize << (padding.k_log - group_log);
        let live = padding.useful_bits.div_ceil(1 << group_log);
        if live >= per_block {
            return every;
        }
        Self {
            mask: per_block - 1,
            live,
        }
    }

    /// Whether group `g` may hold a nonzero bit.
    #[inline(always)]
    pub(crate) const fn contains(self, g: usize) -> bool {
        (g & self.mask) < self.live
    }
}

/// Two consecutive rounds from one pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RoundPair {
    /// Round `t`'s `(G(1), G(inf))`.
    pub(crate) first: (Poly192, Poly192),
    /// Round `t + 1` at `Y = 1` and `Y = inf`, each as `[S_0, S_1, S_2]` of `(1 + rho) S_0 + rho S_1 + rho (1 + rho) S_2`.
    second: [[Poly192; 3]; 2],
}

impl RoundPair {
    /// Round `t + 1`'s `(G(1), G(inf))`, once round `t`'s challenge `rho` is known.
    pub(crate) fn second(&self, rho: Poly192) -> (Poly192, Poly192) {
        let [one, inf] = self
            .second
            .map(|[s0, s1, s2]| s0 + rho * (s0 + s1) + rho * (Poly192::ONE + rho) * s2);
        (one, inf)
    }

    /// Both rounds from the eight eq-weighted quad sums, `r_v` the eq coordinate of the second round's variable.
    fn from_sums(sums: [Poly192; 8], r_v: Poly192) -> Self {
        // Slots 0, 1 are round t's G(1) at v = 0, 1; slots 2, 3 its G(inf).
        // Round t + 1 reads slots 4, 1, 3 at Y = 1 and 5, 6, 7 at Y = inf.
        let split_v = |v0: Poly192, v1: Poly192| v0 + r_v * (v0 + v1);
        Self {
            first: (split_v(sums[0], sums[1]), split_v(sums[2], sums[3])),
            second: [[sums[4], sums[1], sums[3]], [sums[5], sums[6], sums[7]]],
        }
    }
}

/// One quad's terms of two consecutive rounds, before its eq weight.
///
/// Position `4k + u + 2v` has `u` the first round's variable and `v` the second's.
///
/// Eight products cover both rounds, as many as two single rounds.
#[inline(always)]
fn quad_pair_terms(
    [a0, a1, a2, a3]: [Poly192; 4],
    [b0, b1, b2, b3]: [Poly192; 4],
    [_, c1, c2, c3]: [Poly192; 4],
) -> [[Poly192; 4]; 2] {
    // Leading coefficients along `u` (positions 0,1 and 2,3) and along `v` (0,2 and 1,3).
    let (du0, du1, dv0, dv1) = (a0 + a1, a2 + a3, a0 + a2, a1 + a3);
    let (eu0, eu1, ev0, ev1) = (b0 + b1, b2 + b3, b0 + b2, b1 + b3);
    let [p1, p2, p3, q0] = Poly192::mul4([a1, a2, a3, du0], [b1, b2, b3, eu0]);
    let [q1, r0, r1, r2] = Poly192::mul4([du1, dv0, dv1, du0 + du1], [eu1, ev0, ev1, eu0 + eu1]);
    [[p1 + c1, p3 + c3, q0, q1], [p2 + c2, r0, r1, r2]]
}

/// Add one quad's eq-weighted terms to the eight sums, unreduced.
#[inline(always)]
fn add_quad(acc: &mut [Poly192Unreduced; 8], eq: Poly192, terms: [[Poly192; 4]; 2]) {
    for (acc, t) in acc.iter_mut().zip(terms.as_flattened()) {
        *acc += eq.mul_unreduced(*t);
    }
}

/// Eight sums added.
fn add8(x: [Poly192; 8], y: [Poly192; 8]) -> [Poly192; 8] {
    core::array::from_fn(|i| x[i] + y[i])
}

/// Run one task per high index, each with its chunk of every output, and add their sums.
pub(crate) fn tasks(
    n: usize,
    outs: Option<Outs<'_>>,
    chunk: usize,
    task: impl Fn(usize, Option<Outs<'_>>) -> [Poly192; 8] + Sync,
) -> [Poly192; 8] {
    let zero = || [Poly192::ZERO; 8];
    match outs {
        None => (0..n)
            .into_par_iter()
            .map(|hi| task(hi, None))
            .par_fold_reduce(zero, add8, add8),
        Some([o0, o1, o2]) => o0
            .par_chunks_mut(chunk)
            .zip(o1.par_chunks_mut(chunk))
            .zip(o2.par_chunks_mut(chunk))
            .enumerate()
            .map(|(hi, ((x, y), z))| task(hi, Some([x, y, z])))
            .par_fold_reduce(zero, add8, add8),
    }
}

/// Rounds `t` and `t + 1` straight from the packed bits, `fold` weighting a position's `2^t` rows.
///
/// `r_eq` are the eq coordinates of the variables round `t` does not bind.
pub(crate) fn bit_pass(
    bits: Bits<'_>,
    fold: &BitFold,
    r_eq: &[Poly192],
    padding: &Padding,
) -> RoundPair {
    let sums = match fold.n_chunks() {
        8 => bit_pass_kernel::<8>(bits, fold, r_eq, padding, None),
        16 => bit_pass_kernel::<16>(bits, fold, r_eq, padding, None),
        32 => bit_pass_kernel::<32>(bits, fold, r_eq, padding, None),
        64 => bit_pass_kernel::<64>(bits, fold, r_eq, padding, None),
        128 => bit_pass_kernel::<128>(bits, fold, r_eq, padding, None),
        n => panic!("no bit pass for {n}-byte rows"),
    };
    RoundPair::from_sums(sums, r_eq[0])
}

/// The same two rounds, also storing the level-`t` folded `(a, b, c)` tables.
pub(crate) fn bit_pass_storing(
    bits: Bits<'_>,
    fold: &BitFold,
    r_eq: &[Poly192],
    padding: &Padding,
) -> (RoundPair, [Poly<Poly192>; 3]) {
    // Every slot is written by the kernel, so the tables start uninitialized.
    let n_pos = bits.a.len() / fold.n_chunks();
    let mut out: [Box<[MaybeUninit<Poly192>]>; 3] =
        core::array::from_fn(|_| Box::new_uninit_slice(n_pos));
    let outs = Some(out.each_mut().map(|o| &mut o[..]));
    let sums = match fold.n_chunks() {
        8 => bit_pass_kernel::<8>(bits, fold, r_eq, padding, outs),
        16 => bit_pass_kernel::<16>(bits, fold, r_eq, padding, outs),
        32 => bit_pass_kernel::<32>(bits, fold, r_eq, padding, outs),
        64 => bit_pass_kernel::<64>(bits, fold, r_eq, padding, outs),
        128 => bit_pass_kernel::<128>(bits, fold, r_eq, padding, outs),
        n => panic!("no bit pass for {n}-byte rows"),
    };
    // SAFETY: the kernel writes every slot, padding included.
    let tables = out.map(|o| Poly::new(unsafe { o.assume_init() }.into_vec()));
    (RoundPair::from_sums(sums, r_eq[0]), tables)
}

/// The two-round pass over rows of `CHUNKS` bytes: the eight sums the pair is read from.
fn bit_pass_kernel<const CHUNKS: usize>(
    bits: Bits<'_>,
    fold: &BitFold,
    r_eq: &[Poly192],
    padding: &Padding,
    outs: Option<Outs<'_>>,
) -> [Poly192; 8] {
    let rows = bits.rows::<CHUNKS>();
    let n_quads = rows[0].len() / 4;
    assert!(n_quads >= 1, "two rounds need four positions");
    assert_eq!(r_eq.len(), log2_strict_usize(n_quads) + 1);
    if let Some(outs) = &outs {
        assert!(
            outs.iter().all(|o| o.len() == 4 * n_quads),
            "one output per position"
        );
    }

    // `r_eq[0]` weights round `t`'s split by `v`; the rest weight the quads.
    let eq = EqHalves::new(&r_eq[1..], (r_eq.len() - 1).min(EQ_LO_VARS));
    let lo_size = eq.low.len();

    // A quad covers 64 skip bits times its 4 * 2^t rows.
    let live = LiveGroups::new(padding, log2_strict_usize(32 * CHUNKS));

    // A pass that stores nothing sums whole blocks in coefficient planes, where the target has them.
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512bw",
        target_feature = "avx512vbmi",
        target_feature = "gfni",
        target_feature = "vpclmulqdq"
    ))]
    if outs.is_none() && lo_size >= BLOCK / 4 {
        return x86_64_avx512::bit_pass_sums(fold, rows, &eq, live, n_quads);
    }

    let task = |hi: usize, mut outs: Option<Outs<'_>>| -> [Poly192; 8] {
        let stream = Stream::new();
        let mut acc = [Poly192Unreduced::ZERO; 8];
        let mut f = FoldedBlock::ZERO;
        // Sixteen quads per folded block.
        for lo_first in (0..lo_size).step_by(BLOCK / 4) {
            let n = (lo_size - lo_first).min(BLOCK / 4);
            let quad_first = hi * lo_size + lo_first;
            let dst = 4 * lo_first..4 * (lo_first + n);
            // A block wholly in padding folds to zero.
            if !(quad_first..quad_first + n).any(|q| live.contains(q)) {
                if let Some(outs) = &mut outs {
                    for o in outs.iter_mut() {
                        o[dst.clone()].fill(MaybeUninit::new(Poly192::ZERO));
                    }
                }
                continue;
            }
            f.fold(fold, rows, 4 * quad_first, 4 * n);
            let [qa, qb, qc] = [&f.a, &f.b, &f.c].map(|t| t.as_chunks::<4>().0);
            for i in 0..n {
                let terms = quad_pair_terms(qa[i], qb[i], qc[i]);
                add_quad(&mut acc, eq.low[lo_first + i], terms);
            }
            if let Some(outs) = &mut outs {
                for (o, t) in outs.iter_mut().zip([&f.a, &f.b, &f.c]) {
                    stream.write(&mut o[dst.clone()], &t[..4 * n]);
                }
            }
        }
        acc.map(|s| eq.high[hi] * s.reduce())
    };
    tasks(n_quads.div_ceil(lo_size), outs, 4 * lo_size, task)
}

/// One round straight from the folded tables, `(G(1), G(inf))`.
pub(crate) fn single_round([a, b, c]: [&[Poly192]; 3], r_eq: &[Poly192]) -> (Poly192, Poly192) {
    let n = a.len();
    assert!(n.is_power_of_two() && n >= 2);
    assert!(b.len() == n && c.len() == n, "a, b, c have one length");
    assert_eq!(r_eq.len(), log2_strict_usize(n) - 1);
    let eq = Point::new(r_eq.to_vec()).equality_weights_lsb();
    let (mut g1, mut g_inf) = (Poly192::ZERO, Poly192::ZERO);
    for (x, &e) in eq.iter().enumerate() {
        let (a0, a1, b0, b1, c1) = (a[2 * x], a[2 * x + 1], b[2 * x], b[2 * x + 1], c[2 * x + 1]);
        g1 += e * (a1 * b1 + c1);
        g_inf += e * (a0 + a1) * (b0 + b1);
    }
    (g1, g_inf)
}

/// `G(0)` of a round from the folded tables, for an eq coordinate of one, where the claim leaves it free.
pub(crate) fn round_at_zero([a, b, c]: [&[Poly192]; 3], r_eq: &[Poly192]) -> Poly192 {
    Point::new(r_eq.to_vec())
        .equality_weights_lsb()
        .iter()
        .enumerate()
        .map(|(x, &e)| e * (a[2 * x] * b[2 * x] + c[2 * x]))
        .sum()
}

/// Rounds `t` and `t + 1` from the stored tables, folding the one or two pending challenges first.
///
/// `outs` receive the level-`t` tables; each output covers `2^out_log` witness bits.
pub(crate) fn table_pass(
    ins: [&[Poly192]; 3],
    outs: Outs<'_>,
    pending: &[Poly192],
    r_eq: &[Poly192],
    padding: &Padding,
    out_log: usize,
) -> RoundPair {
    let sums = match *pending {
        [rho] => table_pass_kernel::<1>(ins, outs, [rho, Poly192::ZERO], r_eq, padding, out_log),
        [rho_0, rho_1] => table_pass_kernel::<2>(ins, outs, [rho_0, rho_1], r_eq, padding, out_log),
        _ => panic!("one or two pending challenges"),
    };
    RoundPair::from_sums(sums, r_eq[0])
}

/// The table pass for `K` pending challenges: the eight sums the pair is read from.
fn table_pass_kernel<const K: usize>(
    ins: [&[Poly192]; 3],
    outs: Outs<'_>,
    rhos: [Poly192; 2],
    r_eq: &[Poly192],
    padding: &Padding,
    out_log: usize,
) -> [Poly192; 8] {
    let n_out = ins[0].len() >> K;
    assert!(
        ins.iter().all(|t| t.len() == n_out << K),
        "a, b, c have one length"
    );
    assert!(
        outs.iter().all(|t| t.len() == n_out),
        "each output is the folded length"
    );
    let n_quads = n_out / 4;
    assert!(n_quads >= 1, "two rounds need four positions");
    assert_eq!(r_eq.len(), log2_strict_usize(n_quads) + 1);

    let eq = EqHalves::new(&r_eq[1..], (r_eq.len() - 1).saturating_sub(EQ_HIGH_VARS));
    let lo_size = eq.low.len();
    let live = LiveGroups::new(padding, out_log + 2);
    let chunk_in = (4 * lo_size) << K;

    // The weights of the composed fold, `rho_0`, `rho_1` and `rho_0 rho_1`.
    let [w0, w1, w01] = [rhos[0], rhos[1], rhos[0] * rhos[1]].map(|w| [w; 4]);

    // Four outputs of one table, each folded from its `2^K` inputs in one step:
    // one pending is `x_0 + rho_0 (x_0 + x_1)`; two add `rho_1 (x_0 + x_2) + rho_0 rho_1 (x_0 + x_1 + x_2 + x_3)`.
    let fold_quad = |g: &[Poly192]| -> [Poly192; 4] {
        let x = |j: usize, i: usize| g[(j << K) + i];
        let d = Poly192::mul4(core::array::from_fn(|j| x(j, 0) + x(j, 1)), w0);
        let z: [Poly192; 4] = core::array::from_fn(|j| x(j, 0) + d[j]);
        if K == 1 {
            return z;
        }
        let e = Poly192::mul4(core::array::from_fn(|j| x(j, 0) + x(j, 2)), w1);
        let f = Poly192::mul4(
            core::array::from_fn(|j| x(j, 0) + x(j, 1) + x(j, 2) + x(j, 3)),
            w01,
        );
        core::array::from_fn(|j| z[j] + e[j] + f[j])
    };

    let task = |hi: usize, outs: Option<Outs<'_>>| -> [Poly192; 8] {
        let mut outs = outs.expect("a table pass stores its tables");
        let ins = ins.map(|t| &t[hi * chunk_in..(hi + 1) * chunk_in]);
        let stream = Stream::new();
        let mut acc = [Poly192Unreduced::ZERO; 8];
        // Two quads of a table are three whole cache lines, streamed out at once.
        let mut staged = [[Poly192::ZERO; 8]; 3];
        for q in 0..lo_size {
            // A quad wholly in padding folds to zero.
            let folded = if live.contains(hi * lo_size + q) {
                let [a, b, c] = ins.map(|t| fold_quad(&t[(4 * q) << K..(4 * (q + 1)) << K]));
                add_quad(&mut acc, eq.low[q], quad_pair_terms(a, b, c));
                [a, b, c]
            } else {
                [[Poly192::ZERO; 4]; 3]
            };
            let half = 4 * (q % 2);
            for (stage, t) in staged.iter_mut().zip(folded) {
                stage[half..half + 4].copy_from_slice(&t);
            }
            if q % 2 == 1 {
                for (o, stage) in outs.iter_mut().zip(&staged) {
                    stream.write(&mut o[4 * (q - 1)..4 * (q + 1)], stage);
                }
            } else if q + 1 == lo_size {
                // An odd last quad is stored alone.
                for (o, stage) in outs.iter_mut().zip(&staged) {
                    o[4 * q..4 * q + 4].write_copy_of_slice(&stage[..4]);
                }
            }
        }
        acc.map(|s| eq.high[hi] * s.reduce())
    };
    tasks(n_quads / lo_size, Some(outs), 4 * lo_size, task)
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::*;
    use crate::skip::{SKIP_SIZE, lagrange_weights};

    /// Random packed bits of `2^m` bits.
    fn random_bits(rng: &mut SmallRng, m: usize) -> Vec<u8> {
        (0..(1usize << m) / 8).map(|_| rng.random()).collect()
    }

    /// The bits folded at the skip weights: one value per row of 64 bits.
    fn fold_rows(bits: &[u8], lagrange: &[Poly192]) -> Poly<Poly192> {
        Poly::new(
            bits.chunks(8)
                .map(|row| {
                    (0..SKIP_SIZE)
                        .filter(|&s| row[s / 8] >> (s % 8) & 1 == 1)
                        .map(|s| lagrange[s])
                        .sum()
                })
                .collect(),
        )
    }

    /// One round of the stored tables.
    fn round_of(t: &[Poly<Poly192>; 3], r_eq: &[Poly192]) -> (Poly192, Poly192) {
        single_round(t.each_ref().map(Poly::as_slice), r_eq)
    }

    #[test]
    fn bit_passes_match_the_naive_route() {
        // Invariant: both bit passes send rounds t and t + 1, and the storing one stores level t.
        for m in [13, 14, 15, 16] {
            let mut rng = SmallRng::seed_from_u64(0xB17 + m as u64);
            let (a, b) = (random_bits(&mut rng, m), random_bits(&mut rng, m));
            let c: Vec<u8> = a.iter().zip(&b).map(|(x, y)| x & y).collect();
            let r: Vec<Poly192> = (0..m - 6).map(|_| rng.random()).collect();
            let rho: Vec<Poly192> = (0..m - 6).map(|_| rng.random()).collect();
            let lagrange = lagrange_weights(&rng.random(), SKIP_SIZE);
            let dense = Padding::dense(m);
            let bits = Bits { a: &a, b: &b };

            let mut tables = [&a, &b, &c].map(|t| fold_rows(t, &lagrange));
            for t in 0..=4 {
                let fold = BitFold::at_level(&lagrange, &rho[..t]);
                let r_eq = &r[t + 1..];
                let expected = round_of(&tables, r_eq);
                let pair = bit_pass(bits, &fold, r_eq, &dense);
                let (stored_pair, stored) = bit_pass_storing(bits, &fold, r_eq, &dense);
                assert_eq!(stored, tables, "stored tables, m={m}, t={t}");
                for t_ in &mut tables {
                    t_.fix_suffix_var_mut(rho[t]);
                }
                let next = round_of(&tables, &r[t + 2..]);
                for (pair, name) in [(pair, "pass"), (stored_pair, "storing pass")] {
                    assert_eq!(pair.first, expected, "{name} round t, m={m}, t={t}");
                    assert_eq!(
                        pair.second(rho[t]),
                        next,
                        "{name} round t + 1, m={m}, t={t}"
                    );
                }
            }
        }
    }

    #[test]
    fn a_table_pass_is_one_round_at_a_time() {
        // Invariant: folding the pending challenges and sending two rounds in one pass is the one-round route.
        let mut rng = SmallRng::seed_from_u64(0x7AB1E);
        for k in [1, 2] {
            for log_out in [2, 3, 6, 12] {
                let tables: [Poly<Poly192>; 3] = core::array::from_fn(|_| {
                    Poly::new((0..1usize << (log_out + k)).map(|_| rng.random()).collect())
                });
                let pending: Vec<Poly192> = (0..k).map(|_| rng.random()).collect();
                let r_eq: Vec<Poly192> = (0..log_out - 1).map(|_| rng.random()).collect();
                let rho_t: Poly192 = rng.random();

                let mut naive = tables.clone();
                for &rho in &pending {
                    for t in &mut naive {
                        t.fix_suffix_var_mut(rho);
                    }
                }
                let level = naive.clone();
                let first = round_of(&naive, &r_eq);
                for t in &mut naive {
                    t.fix_suffix_var_mut(rho_t);
                }
                let second = round_of(&naive, &r_eq[1..]);

                let mut outs: [Box<[MaybeUninit<Poly192>]>; 3] =
                    core::array::from_fn(|_| Box::new_uninit_slice(1 << log_out));
                let pair = table_pass(
                    tables.each_ref().map(Poly::as_slice),
                    outs.each_mut().map(|o| &mut o[..]),
                    &pending,
                    &r_eq,
                    &Padding::dense(log_out),
                    0,
                );
                // SAFETY: the pass writes every slot of its outputs.
                let outs = outs.map(|o| Poly::new(unsafe { o.assume_init() }.into_vec()));
                assert_eq!(outs, level, "tables, k={k}, log_out={log_out}");
                assert_eq!(pair.first, first, "round t, k={k}, log_out={log_out}");
                assert_eq!(
                    pair.second(rho_t),
                    second,
                    "round t + 1, k={k}, log_out={log_out}"
                );
            }
        }
    }

    #[test]
    fn padding_is_skipped_exactly() {
        // Invariant: on blocks zero past their useful bits, skipping the padding changes no message or table.
        for (m, k_log, useful) in [(16usize, 13usize, 5000usize), (17, 14, 9_409), (16, 9, 300)] {
            let mut rng = SmallRng::seed_from_u64(0xFADE + useful as u64);
            let padded = |rng: &mut SmallRng| -> Vec<u8> {
                let mut bits = random_bits(rng, m);
                for i in 0..1usize << m {
                    if i % (1 << k_log) >= useful {
                        bits[i / 8] &= !(1 << (i % 8));
                    }
                }
                bits
            };
            let (a, b) = (padded(&mut rng), padded(&mut rng));
            let bits = Bits { a: &a, b: &b };
            let padding = Padding {
                k_log,
                useful_bits: useful,
            };
            let lagrange = lagrange_weights(&rng.random(), SKIP_SIZE);
            let r: Vec<Poly192> = (0..m - 6).map(|_| rng.random()).collect();
            let rho: Vec<Poly192> = (0..4).map(|_| rng.random()).collect();
            for t in 0..=4 {
                let fold = BitFold::at_level(&lagrange, &rho[..t]);
                let run = |p: &Padding| {
                    (
                        bit_pass(bits, &fold, &r[t + 1..], p),
                        bit_pass_storing(bits, &fold, &r[t + 1..], p),
                    )
                };
                let (dense, skipped) = (run(&Padding::dense(m)), run(&padding));
                assert_eq!(dense, skipped, "m={m}, useful={useful}, t={t}");

                // The stored tables then take a table pass the same way.
                let tables = dense.1.1;
                let n_out = tables[0].num_evals() / 4;
                if n_out >= 4 {
                    let pass = |p: &Padding| {
                        let mut outs: [Box<[MaybeUninit<Poly192>]>; 3] =
                            core::array::from_fn(|_| Box::new_uninit_slice(n_out));
                        let pair = table_pass(
                            tables.each_ref().map(Poly::as_slice),
                            outs.each_mut().map(|o| &mut o[..]),
                            &rho[..2],
                            &r[t + 3..],
                            p,
                            6 + t + 2,
                        );
                        // SAFETY: the pass writes every slot of its outputs.
                        (pair, outs.map(|o| unsafe { o.assume_init() }.into_vec()))
                    };
                    assert_eq!(
                        pass(&Padding::dense(m)),
                        pass(&padding),
                        "table pass, m={m}, t={t}"
                    );
                }
            }
        }
    }
}
