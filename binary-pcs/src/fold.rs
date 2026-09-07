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
use p3_binary_field::{BinaryField128, Ghash128};
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
// The lane-offset identity below needs that index to be a multiple of the packing width, so the
// grain must cover whole packed groups.
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
/// The second term depends on the lane alone, so the whole fold needs one evaluation per group
/// plus this vector, built once.
fn lane_offsets() -> Packed {
    // Lane `k` carries `domain_point(2 * k)`, the offset from its group's first domain point.
    Packed::from_fn(|lane| domain_point(lane << 1))
}

/// Fold the pairs one parallel task owns.
///
/// `start` is the output index the task's first pair produces.
/// It is a multiple of the grain, hence of the packing width, which is what lets whole groups
/// share a single domain evaluation.
///
/// # Panics
///
/// Panics unless the task holds exactly two input symbols per output slot.
fn fold_task(
    start: usize,
    pairs: &[BinaryField128],
    out: &mut [BinaryField128],
    beta: Ghash128,
    lane_offsets: Packed,
) {
    // Every output slot consumes one pair, so the task's two slices are locked together.
    assert_eq!(
        pairs.len(),
        2 * out.len(),
        "a fold task must hold one pair per output symbol"
    );

    // The challenge is the same in every lane, so it broadcasts once for the whole task.
    let beta_packed = Packed::broadcast(beta);

    // Split both sides into whole packed groups plus a shorter remainder.
    // The two remainders match because one output slot always consumes two input symbols.
    let (slot_groups, tail) = out.as_chunks_mut::<WIDTH>();
    let (symbol_blocks, tail_pairs) = pairs.as_chunks::<{ 2 * WIDTH }>();

    // Output index the remainder starts at, relative to the task.
    let tail_offset = slot_groups.len() * WIDTH;

    // Phase 1: whole packed groups.
    //
    //     block  : [ lo_0 hi_0 | lo_1 hi_1 | ... | lo_{W-1} hi_{W-1} ]   2 * WIDTH symbols
    //     slots  : [ out_0     | out_1     | ... | out_{W-1}         ]       WIDTH symbols
    for (group, (slots, block)) in slot_groups.iter_mut().zip(symbol_blocks).enumerate() {
        // Output index lane 0 of this group produces, a multiple of the width.
        let first = start + group * WIDTH;

        // One domain evaluation for the group, plus the constant per-lane offsets.
        let x = Packed::broadcast(domain_point(first << 1)) + lane_offsets;

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
/// Within a task one domain evaluation serves a whole packed group.
///
/// Evaluating per symbol instead is the right shape for a single query, but over a whole
/// codeword it would recompute that value once per output symbol.
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
    let offsets = lane_offsets();

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
            fold_task(task * FOLD_GRAIN, pairs, out, beta_poly, offsets);
        });

    folded
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use p3_binary_dft::{AdditiveNtt, NaiveAdditiveNtt, domain_point};
    use p3_binary_field::{BinaryField128, TowerLevel};
    use p3_field::PrimeCharacteristicRing;
    use p3_matrix::dense::RowMajorMatrix;
    use p3_multilinear_util::poly::Poly;
    use proptest::prelude::*;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::{FOLD_GRAIN, WIDTH, fold_codeword, fold_pair};

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
