//! Folding a codeword over the Cantor additive domain.
//!
//! A codeword symbol at domain point `x` and its partner at `x + v_0 = x + 1` determine the
//! two halves of the novel-basis decomposition
//!
//! ```text
//!     f(x) = f_0(W_1(x)) + x * f_1(W_1(x))
//! ```
//!
//! so `f_1 = f(x) + f(x + 1)` and `f_0 = f(x) + x * f_1`. The fold at `beta` is the
//! evaluation-basis combination `(1 - beta) * f_0 + beta * f_1`, which over characteristic 2 is
//! `f_0 + beta * (f_0 + f_1)`. This combination is chosen because it is exactly what
//! `Poly::fix_suffix_var` computes on the message: folding the codeword and binding the
//! multilinear's lowest variable in the evaluation basis are the same operation, which is what
//! lets the sumcheck and the codeword move in lockstep.
//!
//! Because `W_1(x) = x^2 + x` is `F_2`-linear with `W_1(v_0) = 0` and `W_1(v_i) = v_{i-1}`, and
//! `domain_point` is `F_2`-linear too, `W_1(domain_point(i)) = domain_point(i >> 1)`: the folded
//! domain is the same function at a halved index, so no per-round domain state is carried. The
//! folding partners `domain_point(2j)` and `domain_point(2j + 1)` differ by `v_0 = 1`, so they
//! are adjacent rows in memory.
//!
//! Each output symbol costs two field multiplications.
//!
//! In the tower basis a multiplication is a polynomial-basis multiplication between two changes
//! of basis on the operands and one back on the result.
//! Each change of basis is sixteen dependent byte-table lookups, so the wrapper dominates.
//!
//! The fold therefore crosses into the polynomial basis once per loaded symbol and back once per
//! produced symbol, multiplying in between with the widest carryless-multiply register available.
//!
//! Input and output stay in the tower basis, so the folded codeword is unchanged bit for bit.

use alloc::vec::Vec;

use p3_binary_dft::domain_point;
use p3_binary_field::{BinaryField128, Ghash128, TowerLevel};
use p3_field::{Field, PackedValue, PrimeCharacteristicRing};
use p3_maybe_rayon::prelude::*;

/// The polynomial-basis type the fold multiplies with.
///
/// This is the widest carryless-multiply register the target offers.
/// Where the target has none it is a single element, and every loop below still holds.
type Packed = <Ghash128 as Field>::Packing;

/// How many output symbols one packed multiplication covers.
const WIDTH: usize = Packed::WIDTH;

/// Output symbols one parallel task owns.
///
/// Large enough that the task-opening domain evaluation is amortised over many folds.
///
/// Small enough to leave real parallelism at every codeword length this crate exercises.
const FOLD_GRAIN: usize = 1 << 10;

// A task's first output index is a multiple of the grain.
//
// The lane-offset identity below needs that index to be a multiple of the packing width.
//
// So the grain must cover whole packed groups.
const _: () = assert!(
    FOLD_GRAIN.is_multiple_of(WIDTH),
    "the fold grain must be a whole number of packed groups"
);

/// Fold one pair of the codeword.
///
/// `lo` is the symbol at `domain_point(2 * index)`, `hi` the one at `domain_point(2 * index + 1)`.
#[inline]
pub fn fold_pair(
    index: usize,
    beta: BinaryField128,
    lo: BinaryField128,
    hi: BinaryField128,
) -> BinaryField128 {
    let x: BinaryField128 = domain_point(index << 1);
    let f1 = lo + hi;
    let f0 = lo + x * f1;
    f0 + beta * (f0 + f1)
}

/// The domain point each lane adds on top of the one evaluated at its group's first index.
///
/// # Algorithm
///
/// Write `v_r` for the `r`-th Cantor basis vector, so that
///
/// ```text
///     domain_point(n) = sum_r bit_r(n) * v_r
/// ```
///
/// That map is `F_2`-linear in the *bits* of its index, hence additive over exclusive-or:
///
/// ```text
///     domain_point(m XOR n) = domain_point(m) + domain_point(n)
/// ```
///
/// It is not additive over integer addition, because a carry moves a bit to a place the sum of
/// the two basis vectors never reaches.
///
/// Carries are absent exactly when the two bit patterns are disjoint, and then addition and
/// exclusive-or agree:
///
/// ```text
///     m AND n == 0   =>   m + n == m XOR n
/// ```
///
/// A packed group's first output index `g` is a multiple of the width, which is a power of two,
/// so every bit of `g` below `log2(width)` is clear.
///
/// A lane index `k` is below the width, so every bit it sets is below `log2(width)`.
///
/// The two patterns are therefore disjoint, and doubling shifts both up one place without
/// disturbing that:
///
/// ```text
///     g AND k == 0   =>   2 * (g + k) == (2 * g) XOR (2 * k)
/// ```
///
/// Combining the two displays gives the identity the fold runs on:
///
/// ```text
///     domain_point(2 * (g + k)) = domain_point(2 * g) + domain_point(2 * k)
/// ```
///
/// The second term depends on the lane alone, so the whole fold needs one group base plus this
/// vector, built once.
///
/// The packing is a type parameter rather than `Packed` itself.
///
/// That lets the lane arithmetic be exercised at widths this target compiles no register for.
fn lane_offsets<P: PackedValue<Value = Ghash128>>() -> P {
    // Lane `k` carries `domain_point(2 * k)`, the offset from its group's first domain point.
    P::from_fn(|lane| domain_point(lane << 1))
}

/// The exclusive-or step from one packed group's first domain point to the next.
///
/// # Algorithm
///
/// Group `G` opens at output index `G * W`, so it opens at doubled index `G << L`:
///
/// ```text
///     L = log2(2 * W)
///     base(G) = domain_point(G << L) = sum_r bit_r(G) * v_{L + r}
/// ```
///
/// That map is `F_2`-linear in the bits of `G`, hence additive over exclusive-or.
///
/// Stepping from `G - 1` to `G` flips exactly bits `0 ..= k`, for `k = G.trailing_zeros()`:
///
/// ```text
///     (G - 1) XOR G = 2^(k + 1) - 1
/// ```
///
/// Consecutive group bases therefore differ by the basis vectors those bits select:
///
/// ```text
///     base(G) = base(G - 1) + sum_{r <= k} v_{L + r}
/// ```
///
/// Entry `k` is that sum, so a group past the first costs one exclusive-or, not a `domain_point`.
fn group_steps<const W: usize>(num_pairs: usize) -> Vec<Ghash128> {
    // A group strides `W` output indices, hence `2 * W` domain indices.
    let log_stride = (2 * W).trailing_zeros() as usize;

    // Group indices run below `num_pairs / W`, so one entry per bit of that count is enough.
    let levels = (num_pairs / W).next_power_of_two().trailing_zeros() as usize;

    let mut step = Ghash128::ZERO;
    (0..levels)
        .map(|level| {
            step += Ghash128::cantor_basis(log_stride + level);
            step
        })
        .collect()
}

/// Fold the pairs one parallel task owns.
///
/// `start` is the output index the task's first pair produces.
///
/// It is a multiple of the grain, hence of the packing width.
///
/// That alignment is what lets a group share one domain point across its lanes.
///
/// `group_steps` walks the group bases, so the task evaluates the domain exactly once.
///
/// # Panics
///
/// Panics unless the task holds exactly two input symbols per output slot.
///
/// Panics in debug builds unless `start` is a multiple of the packing width.
fn fold_task(
    start: usize,
    pairs: &[BinaryField128],
    out: &mut [BinaryField128],
    beta: Ghash128,
    lane_offsets: Packed,
    group_steps: &[Ghash128],
) {
    // Every output slot consumes one pair, so the task's two slices are locked together.
    assert_eq!(
        pairs.len(),
        2 * out.len(),
        "a fold task must hold one pair per output symbol"
    );

    // Invariant: `start` is a multiple of the packing width.
    //
    // The lane-offset identity rests on `g AND k == 0` for a group start `g` and a lane `k`.
    //
    // A lane index only sets bits below `log2(WIDTH)`, so `g` must carry none of them.
    //
    // Group starts are `start + group * WIDTH`, so they inherit that from `start` alone.
    debug_assert!(
        start.is_multiple_of(WIDTH),
        "a fold task must start on a packed group boundary"
    );

    // The challenge is the same in every lane, so it broadcasts once for the whole task.
    let beta_packed = Packed::broadcast(beta);

    // Split both sides into whole packed groups plus a shorter remainder.
    // The two remainders match because one output slot always consumes two input symbols.
    let (slot_groups, tail) = out.as_chunks_mut::<WIDTH>();
    let (symbol_blocks, tail_pairs) = pairs.as_chunks::<{ 2 * WIDTH }>();

    // Output index the remainder starts at, relative to the task.
    let tail_offset = slot_groups.len() * WIDTH;

    // Where the task's first group sits among all the codeword's groups.
    let first_group = start / WIDTH;

    // The task's only domain evaluation, opening the walk over its groups.
    let mut base: Ghash128 = domain_point(start << 1);

    // Phase 1: whole packed groups.
    //
    //     block  : [ lo_0 hi_0 | lo_1 hi_1 | ... | lo_{W-1} hi_{W-1} ]   2 * WIDTH symbols
    //     slots  : [ out_0     | out_1     | ... | out_{W-1}         ]       WIDTH symbols
    for (group, (slots, block)) in slot_groups.iter_mut().zip(symbol_blocks).enumerate() {
        // Advance the walk to this group's first domain point.
        if group != 0 {
            base += group_steps[(first_group + group).trailing_zeros() as usize];
        }

        // That base is lane 0's domain point, and the offsets carry the rest of the lanes.
        let x = Packed::broadcast(base) + lane_offsets;

        // Cross into the polynomial basis while gathering the lanes.
        // The change of basis is additive, so the sums formed below are the same either side.
        let lo = Packed::from_fn(|lane| Ghash128::from(block[2 * lane]));
        let hi = Packed::from_fn(|lane| Ghash128::from(block[2 * lane + 1]));

        // The novel-basis halves of each pair.
        let f1 = lo + hi;
        let f0 = lo + x * f1;

        // The evaluation-basis combination at the challenge.
        let folded = f0 + beta_packed * (f0 + f1);

        // Cross back so the caller sees a tower-basis codeword.
        for (slot, &value) in slots.iter_mut().zip(folded.as_slice()) {
            *slot = BinaryField128::from(value);
        }
    }

    // Phase 2: fewer output slots left than one packed group holds.
    //
    // A task's slot count is either a whole grain or the whole codeword's pair count.
    //
    // The grain covers whole groups, and a power-of-two pair count is a multiple of the width
    // unless it is below it, so this runs only for a codeword with fewer pairs than the width.
    //
    // Each leftover evaluates its own domain point, so it rests on no group alignment at all.
    //
    // Twice as many symbols are left as slots, so the leftovers pair up exactly and the
    // remainder discarded here is empty.
    let (tail_blocks, _) = tail_pairs.as_chunks::<2>();
    for (offset, (slot, pair)) in tail.iter_mut().zip(tail_blocks).enumerate() {
        let x: Ghash128 = domain_point((start + tail_offset + offset) << 1);
        let lo = Ghash128::from(pair[0]);
        let hi = Ghash128::from(pair[1]);
        let f1 = lo + hi;
        let f0 = lo + x * f1;
        *slot = BinaryField128::from(f0 + beta * (f0 + f1));
    }
}

/// Fold a whole codeword, halving its length.
///
/// Each parallel task owns a contiguous run of output symbols and the pairs feeding them, so no
/// two tasks touch the same symbol on either side.
///
/// A task evaluates the domain once, then advances it by one exclusive-or per packed group.
///
/// Evaluating per symbol instead is the right shape for a single query.
///
/// Over a whole codeword it would recompute that value once per output symbol.
///
/// # Panics
///
/// Panics unless the codeword is empty or has a power-of-two length.
/// That is the shape every round schedule produces.
///
/// The loops themselves cover any even length, so this is a deliberate narrowing of the
/// contract rather than a limit of the method.
#[must_use]
pub fn fold_codeword(codeword: &[BinaryField128], beta: BinaryField128) -> Vec<BinaryField128> {
    assert!(
        codeword.is_empty() || codeword.len().is_power_of_two(),
        "codeword length must be a power of two"
    );

    // Two input symbols make one output symbol.
    let num_pairs = codeword.len() / 2;

    // The challenge crosses into the polynomial basis once for the whole codeword.
    let beta_poly = Ghash128::from(beta);
    let offsets = lane_offsets::<Packed>();
    let steps = group_steps::<WIDTH>(num_pairs);

    // Zeroed here and fully overwritten below, so the allocation is handed out already blank
    // rather than assembled from one vector per task.
    let mut folded = BinaryField128::zero_vec(num_pairs);

    // Task `t` owns outputs `[t * GRAIN, (t + 1) * GRAIN)` and inputs `[2 * t * GRAIN, ...)`.
    // Output `j` reads inputs `2 * j` and `2 * j + 1`, both inside its own task's input chunk,
    // so the two chunk streams stay index-aligned and every task is disjoint from every other.
    folded
        .par_chunks_mut(FOLD_GRAIN)
        .zip(codeword.par_chunks(2 * FOLD_GRAIN))
        .enumerate()
        .for_each(|(task, (out, pairs))| {
            fold_task(task * FOLD_GRAIN, pairs, out, beta_poly, offsets, &steps);
        });

    folded
}

/// Fold one aligned coset, including its global offset in every virtual layer.
pub(crate) fn fold_coset(
    coset_index: usize,
    values: &mut [BinaryField128],
    challenges: &[BinaryField128],
) -> BinaryField128 {
    assert_eq!(values.len(), 1usize << challenges.len());
    let mut len = values.len();
    for &beta in challenges {
        len /= 2;
        for i in 0..len {
            values[i] = fold_pair(
                coset_index * len + i,
                beta,
                values[2 * i],
                values[2 * i + 1],
            );
        }
    }
    values[0]
}

/// Fold several sequential challenges without materializing full intermediate codewords.
///
/// Each task reuses one coset-sized scratch buffer in the polynomial basis. Symbols cross
/// bases only when loaded from the source or stored into the final output. The Cantor domain
/// bases advance by XOR, just as in the single-fold path. `challenges` stays in sumcheck order.
pub(crate) fn fold_codeword_batch(
    codeword: &[BinaryField128],
    challenges: &[BinaryField128],
) -> Vec<BinaryField128> {
    assert!(!challenges.is_empty());
    assert!(codeword.len().is_power_of_two());
    assert!(challenges.len() <= codeword.len().ilog2() as usize);
    if challenges.len() == 1 {
        return fold_codeword(codeword, challenges[0]);
    }
    let arity = challenges.len();
    let size = 1 << arity;
    let num_cosets = codeword.len() / size;
    let betas: Vec<_> = challenges.iter().copied().map(Ghash128::from).collect();
    let offsets: Vec<Ghash128> = (0..size / 2).map(|i| domain_point(i << 1)).collect();
    let steps: Vec<Vec<Ghash128>> = (0..arity)
        .map(|round| {
            let mut step = Ghash128::ZERO;
            (0..num_cosets.ilog2() as usize)
                .map(|bit| {
                    step += Ghash128::cantor_basis(arity - round + bit);
                    step
                })
                .collect()
        })
        .collect();
    let mut output = BinaryField128::zero_vec(num_cosets);
    output
        .par_chunks_mut(FOLD_GRAIN)
        .zip(codeword.par_chunks(FOLD_GRAIN.saturating_mul(size)))
        .enumerate()
        .for_each(|(task, (out, input))| {
            let start = task * FOLD_GRAIN;
            let mut bases: Vec<Ghash128> = (0..arity)
                .map(|r| domain_point(start << (arity - r)))
                .collect();
            let mut scratch = Ghash128::zero_vec(size);
            for (offset, (slot, coset)) in out.iter_mut().zip(input.chunks_exact(size)).enumerate()
            {
                let index = start + offset;
                if offset != 0 {
                    for (base, steps) in bases.iter_mut().zip(&steps) {
                        *base += steps[index.trailing_zeros() as usize];
                    }
                }
                for (dst, &src) in scratch.iter_mut().zip(coset) {
                    *dst = Ghash128::from(src);
                }
                let mut len = size;
                for (r, &beta) in betas.iter().enumerate() {
                    len /= 2;
                    let packed_end = len / WIDTH * WIDTH;
                    let beta_packed = Packed::broadcast(beta);
                    for i in (0..packed_end).step_by(WIDTH) {
                        let lo = Packed::from_fn(|lane| scratch[2 * (i + lane)]);
                        let hi = Packed::from_fn(|lane| scratch[2 * (i + lane) + 1]);
                        let x = Packed::from_fn(|lane| bases[r] + offsets[i + lane]);
                        let f1 = lo + hi;
                        let f0 = lo + x * f1;
                        let folded = f0 + beta_packed * (f0 + f1);
                        scratch[i..i + WIDTH].copy_from_slice(folded.as_slice());
                    }
                    for i in packed_end..len {
                        let lo = scratch[2 * i];
                        let f1 = lo + scratch[2 * i + 1];
                        let f0 = lo + (bases[r] + offsets[i]) * f1;
                        scratch[i] = f0 + beta * (f0 + f1);
                    }
                }
                *slot = BinaryField128::from(scratch[0]);
            }
        });
    output
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;
    use core::array;

    use p3_binary_dft::{AdditiveNtt, NaiveAdditiveNtt, domain_point};
    use p3_binary_field::{BinaryField128, Ghash128, TowerLevel};
    use p3_field::PrimeCharacteristicRing;
    use p3_matrix::dense::RowMajorMatrix;
    use p3_multilinear_util::poly::Poly;
    use proptest::prelude::*;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::{FOLD_GRAIN, WIDTH, fold_codeword, fold_pair, group_steps, lane_offsets};

    #[test]
    fn batched_folds_match_sequential_folds_at_every_coset_offset() {
        let mut rng = SmallRng::seed_from_u64(0xBA7C);
        for log_len in [4, 7, 12, 14] {
            let word: Vec<BinaryField128> = (0..1 << log_len).map(|_| rng.random()).collect();
            for arity in 1..=4.min(log_len) {
                let challenges: Vec<BinaryField128> = (0..arity).map(|_| rng.random()).collect();
                let mut expected = word.clone();
                for &beta in &challenges {
                    expected = fold_codeword(&expected, beta);
                }
                assert_eq!(super::fold_codeword_batch(&word, &challenges), expected);
                for (index, coset) in word.chunks(1 << arity).enumerate() {
                    assert_eq!(
                        super::fold_coset(index, &mut coset.to_vec(), &challenges),
                        expected[index]
                    );
                }
            }
        }
    }

    /// Encode novel-basis coefficients over the additive domain, through the oracle.
    ///
    /// `NaiveAdditiveNtt` evaluates the definition directly and depends on no identity the
    /// fold also relies on, so agreement between the two is real evidence rather than a
    /// restatement.
    fn encode(coeffs: &[BinaryField128]) -> Vec<BinaryField128> {
        NaiveAdditiveNtt::<BinaryField128>::default()
            .ntt_batch(RowMajorMatrix::new(coeffs.to_vec(), 1))
            .values
    }

    /// The XOR step from `domain_point(2 * j)` to `domain_point(2 * (j + 1))`, indexed by
    /// `(j + 1).trailing_zeros()`.
    ///
    /// The domain map is `F_2`-linear, and `j XOR (j + 1)` sets exactly the trailing ones of
    /// `j` together with the first zero bit above them.
    ///
    /// Consecutive doubled domain points therefore differ by the sum of Cantor basis vectors
    /// `1` through `level + 1`.
    fn gray_chain_steps(num_pairs: usize) -> Vec<BinaryField128> {
        let levels = num_pairs.next_power_of_two().trailing_zeros() as usize;
        let mut steps = Vec::with_capacity(levels);
        let mut acc = BinaryField128::ZERO;
        for level in 0..levels {
            acc += BinaryField128::cantor_basis(level + 1);
            steps.push(acc);
        }
        steps
    }

    /// The tower-basis fold, kept here as the bit-for-bit reference the packed form must match.
    ///
    /// It walks the domain by a chained `F_2`-linear step and multiplies in the tower basis, so
    /// it shares neither the loop shape nor the arithmetic representation of the code under
    /// test.
    fn fold_codeword_tower_reference(
        codeword: &[BinaryField128],
        beta: BinaryField128,
    ) -> Vec<BinaryField128> {
        let num_pairs = codeword.len() / 2;
        let steps = gray_chain_steps(num_pairs);

        codeword
            .chunks(2 * FOLD_GRAIN)
            .enumerate()
            .flat_map(|(block, chunk)| {
                let start = block * FOLD_GRAIN;
                let mut x: BinaryField128 = domain_point(start << 1);
                let steps = &steps;
                chunk
                    .chunks(2)
                    .enumerate()
                    .map(move |(offset, pair)| {
                        if offset != 0 {
                            x += steps[(start + offset).trailing_zeros() as usize];
                        }
                        let (lo, hi) = (pair[0], pair[1]);
                        let f1 = lo + hi;
                        let f0 = lo + x * f1;
                        f0 + beta * (f0 + f1)
                    })
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    #[test]
    fn the_folded_domain_is_domain_point_at_halved_index() {
        // W_1(x) = x^2 + x, and W_1(domain_point(i)) == domain_point(i >> 1).
        for i in 0..512usize {
            let x: BinaryField128 = domain_point(i);
            assert_eq!(
                x.square() + x,
                domain_point::<BinaryField128>(i >> 1),
                "i={i}"
            );
        }
    }

    #[test]
    fn folding_partners_are_adjacent_and_differ_by_one() {
        for j in 0..256usize {
            let lo: BinaryField128 = domain_point(2 * j);
            let hi: BinaryField128 = domain_point(2 * j + 1);
            assert_eq!(lo + BinaryField128::ONE, hi, "j={j}");
        }
    }

    #[test]
    fn a_lane_offset_shifts_a_group_start_to_its_own_domain_point() {
        // Invariant: a group's first index is a multiple of the width, so its low bits are
        // clear and adding a lane index is a disjoint exclusive-or.
        //
        //     group start 8, width 4:   1000
        //     lane 3:                   0011
        //     8 + 3 == 8 | 3 == 11:     1011   no carry crosses between the two
        //
        // The domain map is additive over exclusive-or, so the group's point plus the lane's
        // point is the point of the sum.
        for group in 0..64usize {
            let first = group * WIDTH;
            for lane in 0..WIDTH {
                // Disjointness is what the identity rests on, so pin it directly.
                assert_eq!(first & lane, 0, "group={group} lane={lane}");
                assert_eq!(
                    domain_point::<BinaryField128>((first + lane) << 1),
                    domain_point::<BinaryField128>(first << 1)
                        + domain_point::<BinaryField128>(lane << 1),
                    "group={group} lane={lane}"
                );
            }
        }
    }

    /// The task loop with `[Ghash128; W]` standing in for the packing.
    ///
    /// Every lane index, gather offset and step lookup is the one `fold_task` uses, and the two
    /// helpers it calls are the production ones.
    ///
    /// A build compiles exactly one packing width, so this is what reaches the others.
    fn fold_task_model<const W: usize>(
        start: usize,
        pairs: &[BinaryField128],
        out: &mut [BinaryField128],
        beta: Ghash128,
        lane_offsets: [Ghash128; W],
        group_steps: &[Ghash128],
    ) {
        // Whole groups, then the leftover slots, split where the packed form splits them.
        let (slot_groups, tail) = out.as_chunks_mut::<W>();
        let tail_offset = slot_groups.len() * W;

        // A block of `2 * W` symbols per group, so the remainder starts at twice that offset.
        let (symbol_blocks, tail_pairs) = pairs.split_at(2 * tail_offset);

        let first_group = start / W;
        let mut base: Ghash128 = domain_point(start << 1);

        for (group, slots) in slot_groups.iter_mut().enumerate() {
            if group != 0 {
                base += group_steps[(first_group + group).trailing_zeros() as usize];
            }

            // The `2 * W` symbols this group consumes.
            let block = &symbol_blocks[2 * W * group..][..2 * W];

            // Broadcasting the base and adding the offset vector is this, lane by lane.
            let x: [Ghash128; W] = array::from_fn(|lane| base + lane_offsets[lane]);

            // The gather a packed block performs: even symbols low, odd symbols high.
            let lo: [Ghash128; W] = array::from_fn(|lane| Ghash128::from(block[2 * lane]));
            let hi: [Ghash128; W] = array::from_fn(|lane| Ghash128::from(block[2 * lane + 1]));

            for (lane, slot) in slots.iter_mut().enumerate() {
                let f1 = lo[lane] + hi[lane];
                let f0 = lo[lane] + x[lane] * f1;
                *slot = BinaryField128::from(f0 + beta * (f0 + f1));
            }
        }

        let (tail_blocks, _) = tail_pairs.as_chunks::<2>();
        for (offset, (slot, pair)) in tail.iter_mut().zip(tail_blocks).enumerate() {
            let x: Ghash128 = domain_point((start + tail_offset + offset) << 1);
            let lo = Ghash128::from(pair[0]);
            let hi = Ghash128::from(pair[1]);
            let f1 = lo + hi;
            let f0 = lo + x * f1;
            *slot = BinaryField128::from(f0 + beta * (f0 + f1));
        }
    }

    /// `fold_codeword` driving the width-generic task loop, serially.
    fn fold_codeword_model<const W: usize>(
        codeword: &[BinaryField128],
        beta: BinaryField128,
    ) -> Vec<BinaryField128> {
        let num_pairs = codeword.len() / 2;
        let beta_poly = Ghash128::from(beta);
        let offsets = lane_offsets::<[Ghash128; W]>();
        let steps = group_steps::<W>(num_pairs);

        let mut folded = BinaryField128::zero_vec(num_pairs);
        for (task, (out, pairs)) in folded
            .chunks_mut(FOLD_GRAIN)
            .zip(codeword.chunks(2 * FOLD_GRAIN))
            .enumerate()
        {
            fold_task_model::<W>(task * FOLD_GRAIN, pairs, out, beta_poly, offsets, &steps);
        }
        folded
    }

    /// Fold every shape at one packing width and compare against the tower reference.
    fn check_width<const W: usize>() {
        let mut rng = SmallRng::seed_from_u64(0x1A5E_0FF5_E750_0000);

        // Shapes chosen so both phases run at all four widths:
        //
        //     0             no pairs at all
        //     2, 4          fewer pairs than a group holds at the wider widths
        //     8, 16, 1024   whole groups at every width, inside one task
        //     2048          pairs exactly one grain
        //     4096, 8192    several tasks, so more than one group base is evaluated
        for len in [0usize, 2, 4, 8, 16, 1024, 2048, 4096, 8192] {
            let codeword: Vec<BinaryField128> = (0..len).map(|_| rng.random()).collect();
            let beta: BinaryField128 = rng.random();

            assert_eq!(
                fold_codeword_model::<W>(&codeword, beta),
                fold_codeword_tower_reference(&codeword, beta),
                "W={W} len={len}"
            );
        }
    }

    #[test]
    fn the_lane_arithmetic_holds_at_every_packing_width() {
        // A target compiles one packing, so `the_packed_fold_matches_the_tower_reference` below
        // only ever exercises that one width.
        //
        // Widths swept here, none of them requiring the register that would carry them:
        //
        //     1   no packing
        //     2   the 256-bit carryless multiply
        //     4   the 512-bit carryless multiply
        //     8   past every packing this workspace defines
        //
        // The verifier folds pair by pair in the tower basis, so a lane-gather or step-table
        // mistake at a width this host cannot compile would split the prover from the verifier
        // on hosts that can.
        check_width::<1>();
        check_width::<2>();
        check_width::<4>();
        check_width::<8>();
    }

    #[test]
    fn the_packed_fold_matches_the_tower_reference() {
        let mut rng = SmallRng::seed_from_u64(0xF01D_0BA5_15C0_DE00);

        // Shapes worth separating, all of them power-of-two lengths the assertion admits:
        //
        //     0                  no pairs at all
        //     2                  one pair, fewer than one packed group
        //     4, 8               a partial group on a four-lane build, a whole one on two lanes
        //     16 .. 1024         several groups, still inside one parallel task
        //     2048               pairs exactly one grain, so exactly one task
        //     4096, 8192, 65536  several tasks, the last of them full
        for len in [0usize, 2, 4, 8, 16, 64, 1024, 2048, 4096, 8192, 65536] {
            let codeword: Vec<BinaryField128> = (0..len).map(|_| rng.random()).collect();
            let beta: BinaryField128 = rng.random();

            // Bit-for-bit, not merely equal up to the folded polynomial: the committed bytes
            // of every round after this one depend on the exact output.
            assert_eq!(
                fold_codeword(&codeword, beta),
                fold_codeword_tower_reference(&codeword, beta),
                "len={len}"
            );
        }
    }

    #[test]
    fn the_whole_fold_chain_matches_the_tower_reference() {
        let mut rng = SmallRng::seed_from_u64(0x0C0F_FEE0_DEAD_BEEF);

        // 8192 symbols is 13 rounds down to a single symbol, and 4096 pairs is four grains, so
        // the chain starts multi-task and shrinks through the single-task and sub-group shapes.
        let start: Vec<BinaryField128> = (0..8192).map(|_| rng.random()).collect();

        // One independent challenge per round, as the prover draws them.
        let betas: Vec<BinaryField128> = (0..13).map(|_| rng.random()).collect();

        let mut packed = start.clone();
        let mut reference = start;

        for (round, &beta) in betas.iter().enumerate() {
            packed = fold_codeword(&packed, beta);
            reference = fold_codeword_tower_reference(&reference, beta);

            // Diverging at any round would silently change every later commitment, so compare
            // the whole codeword each time rather than only the final symbol.
            assert_eq!(packed, reference, "round={round}");
        }

        // Thirteen halvings of 8192 leave one symbol, the value the proof carries in the clear.
        assert_eq!(packed.len(), 1);
        assert_eq!(reference.len(), 1);
    }

    proptest! {
        // Each case runs the naive oracle twice over a codeword of up to 512 symbols, so the
        // per-case cost is far above a typical property test's; 64 cases still explore every
        // (rate, length) pair many times over.
        #![proptest_config(ProptestConfig::with_cases(64))]

        /// The crate's central identity: folding the codeword equals binding the message's
        /// lowest variable in the evaluation basis (`Poly::fix_suffix_var`).
        ///
        /// The oracle is a direct evaluation of the additive-NTT definition, which shares no
        /// identity with the fold, so agreement is evidence rather than restatement.
        /// `log_n` and `log_inv_rate` are generated rather than swept because the identity is
        /// claimed for every message length and every blowup a prover can configure, not for a
        /// chosen list; every real prover fold runs on a blown-up codeword, so rate 0 is the
        /// degenerate end of the range rather than the case of interest.
        #[test]
        fn folding_the_codeword_equals_binding_the_lowest_variable(
            log_inv_rate in 0usize..=2,
            log_n in 1usize..=7,
            raw in prop::collection::vec(any::<u128>(), 128),
            beta_raw: u128,
        ) {
            let n = 1usize << log_n;
            let coeffs: Vec<BinaryField128> = raw[..n]
                .iter()
                .copied()
                .map(BinaryField128::from_repr)
                .collect();
            let beta = BinaryField128::from_repr(beta_raw);

            let mut message = coeffs.clone();
            message.resize(n << log_inv_rate, BinaryField128::ZERO);

            let bound = Poly::new(coeffs).fix_suffix_var(beta);
            let mut bound_message = bound.into_evals();
            bound_message.resize(bound_message.len() << log_inv_rate, BinaryField128::ZERO);

            prop_assert_eq!(
                fold_codeword(&encode(&message), beta),
                encode(&bound_message)
            );
        }
    }

    #[test]
    fn the_pair_form_agrees_with_the_vector_form() {
        // Invariant: the whole-codeword form agrees with the independent per-pair computation
        // at every position, including across a task boundary.
        //
        // Fixture state: 4096 symbols is 2048 pairs, past the 1024-symbol grain.
        //
        // So this spans two full parallel tasks and exercises the per-task index base that a
        // single-task codeword leaves silent.
        let mut rng = SmallRng::seed_from_u64(0x9E37_79B9_7F4A_7C15);
        let codeword: Vec<BinaryField128> = (0..4096).map(|_| rng.random()).collect();
        let beta: BinaryField128 = rng.random();
        let folded = fold_codeword(&codeword, beta);
        for (j, &value) in folded.iter().enumerate() {
            assert_eq!(
                value,
                fold_pair(j, beta, codeword[2 * j], codeword[2 * j + 1]),
                "j={j}"
            );
        }
    }

    #[test]
    #[should_panic(expected = "codeword length must be a power of two")]
    fn an_odd_length_codeword_is_rejected() {
        let odd = [BinaryField128::ONE; 3];
        let _ = fold_codeword(&odd, BinaryField128::ONE);
    }

    #[test]
    #[should_panic(expected = "codeword length must be a power of two")]
    fn an_even_non_power_of_two_codeword_is_rejected() {
        // Even is not enough: the contract is narrowed to the shape a round schedule produces.
        // Six symbols is three pairs, which no round schedule produces.
        let six = [BinaryField128::ONE; 6];
        let _ = fold_codeword(&six, BinaryField128::ONE);
    }

    #[test]
    fn an_empty_codeword_folds_to_an_empty_one() {
        // Zero pairs to fold, so the empty input is not a panic.
        assert!(fold_codeword(&[], BinaryField128::ONE).is_empty());
    }

    #[test]
    fn a_single_symbol_codeword_folds_to_an_empty_one() {
        // One symbol is half a pair, so there is nothing to fold and the output is empty.
        // A length of one is a power of two, so the length assertion admits it.
        assert!(fold_codeword(&[BinaryField128::ONE], BinaryField128::ONE).is_empty());
    }

    #[test]
    fn the_novel_basis_fold_is_a_different_operation() {
        // `f_0 + beta * f_1`, the novel-basis combination, is a different operation from the
        // evaluation-basis one this module folds with: the two differ by exactly `beta * f_0`,
        // so they agree only when that term vanishes.
        let mut rng = SmallRng::seed_from_u64(0x51DE);
        let j = 5usize;
        let x: BinaryField128 = domain_point(2 * j);
        let lo: BinaryField128 = rng.random();
        let hi: BinaryField128 = rng.random();
        let beta: BinaryField128 = rng.random();

        let f1 = lo + hi;
        let f0 = lo + x * f1;
        let novel_basis_fold = lo + (x + beta) * (lo + hi);

        assert_eq!(
            fold_pair(j, beta, lo, hi) + novel_basis_fold,
            beta * f0,
            "the evaluation-basis and novel-basis folds differ by beta * f_0"
        );
    }

    proptest! {
        #[test]
        fn folding_is_affine_in_the_challenge(
            lo_raw: u128, hi_raw: u128, b0_raw: u128, b1_raw: u128, j in 0usize..64,
        ) {
            let lo = BinaryField128::from_repr(lo_raw);
            let hi = BinaryField128::from_repr(hi_raw);
            let b0 = BinaryField128::from_repr(b0_raw);
            let b1 = BinaryField128::from_repr(b1_raw);
            let x: BinaryField128 = domain_point(2 * j);
            let f1 = lo + hi;
            let f0 = lo + x * f1;

            // fold(beta) = f0 + beta * (f0 + f1), so fold(b0) + fold(b1) == (b0 + b1) * (f0 + f1).
            prop_assert_eq!(
                fold_pair(j, b0, lo, hi) + fold_pair(j, b1, lo, hi),
                (b0 + b1) * (f0 + f1)
            );

            // At beta = 0 the fold is f0, pinning the internal domain index against one
            // computed independently here.
            prop_assert_eq!(fold_pair(j, BinaryField128::ZERO, lo, hi), f0);
        }
    }
}
