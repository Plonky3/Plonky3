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

use alloc::vec::Vec;

use p3_binary_dft::domain_point;
use p3_binary_field::{BinaryField128, TowerLevel};
use p3_field::PrimeCharacteristicRing;
use p3_maybe_rayon::prelude::*;

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

/// Output pairs one parallel task chains together, opened by a single `domain_point` call.
///
/// Large enough that the task-opening `domain_point` call is amortised over many folds; small
/// enough to leave real parallelism at every codeword length this crate exercises.
const FOLD_GRAIN: usize = 1 << 10;

/// The XOR step from `domain_point(2 * j)` to `domain_point(2 * (j + 1))`, indexed by
/// `(j + 1).trailing_zeros()`.
///
/// `domain_point` is `F_2`-linear, and `j XOR (j + 1)` sets exactly the trailing ones of `j`
/// together with the first zero bit above them — bits `0..=level` for `level =
/// (j + 1).trailing_zeros()`. So `domain_point(2 * j) + domain_point(2 * (j + 1))` is
/// `domain_point` of that same run of bits shifted up by one: the sum of `cantor_basis(1)`
/// through `cantor_basis(level + 1)`. `steps[level]` is that sum, computed once and reused by
/// every transition that lands at the same level.
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

/// Fold a whole codeword, halving its length.
///
/// Each parallel task opens with one `domain_point` call, then chains it across the task's own
/// pairs by adding a precomputed per-level `F_2`-linear step at each transition: `fold_pair`'s
/// single per-call `domain_point` is the right form for one query's verifier-side check, but
/// here the same value would otherwise be recomputed once per output symbol of the whole
/// codeword.
///
/// # Panics
///
/// Panics unless the codeword is empty or has a power-of-two length.
/// That is the shape every round schedule produces.
/// It is also the shape the per-level step table is sized for.
#[must_use]
pub fn fold_codeword(codeword: &[BinaryField128], beta: BinaryField128) -> Vec<BinaryField128> {
    assert!(
        codeword.is_empty() || codeword.len().is_power_of_two(),
        "codeword length must be a power of two"
    );

    let num_pairs = codeword.len() / 2;
    let steps = gray_chain_steps(num_pairs);

    codeword
        .par_chunks(2 * FOLD_GRAIN)
        .enumerate()
        .flat_map_iter(|(block, chunk)| {
            let start = block * FOLD_GRAIN;
            let mut x: BinaryField128 = domain_point(start << 1);
            let steps = &steps;
            chunk.chunks(2).enumerate().map(move |(offset, pair)| {
                if offset != 0 {
                    x += steps[(start + offset).trailing_zeros() as usize];
                }
                let (lo, hi) = (pair[0], pair[1]);
                let f1 = lo + hi;
                let f0 = lo + x * f1;
                f0 + beta * (f0 + f1)
            })
        })
        .collect()
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

    use super::{fold_codeword, fold_pair};

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

    proptest! {
        // Each case runs the naive oracle twice over a codeword of up to 512 symbols, so the
        // per-case cost is far above a typical property test's; 64 cases still explore every
        // (rate, length) pair many times over.
        #![proptest_config(ProptestConfig::with_cases(64))]

        /// The crate's central identity: folding the codeword equals binding the message's
        /// lowest variable in the evaluation basis (`Poly::fix_suffix_var`).
        ///
        /// The oracle is [`NaiveAdditiveNtt`], which evaluates the definition directly and
        /// shares no identity with the fold, so agreement is evidence rather than restatement.
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

    /// The chained form must agree with `fold_pair`'s independent per-call computation at
    /// every position, including across `FOLD_GRAIN`'s block boundary: 4096 codeword symbols
    /// is 2048 pairs, past `FOLD_GRAIN`'s 1024, so this covers two full parallel tasks and
    /// exercises the cross-block chaining — block-start `domain_point(start << 1)` plus
    /// `(start + offset).trailing_zeros()` indexing — that a single-block codeword leaves
    /// silent.
    #[test]
    fn the_pair_form_agrees_with_the_vector_form() {
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
        // Even is not enough: the per-level step table is sized for a power of two.
        // Six symbols is three pairs, which no round schedule produces.
        let six = [BinaryField128::ONE; 6];
        let _ = fold_codeword(&six, BinaryField128::ONE);
    }

    #[test]
    fn an_empty_codeword_folds_to_an_empty_one() {
        // Zero pairs to fold, and zero levels to index, so the empty input is not a panic.
        assert!(fold_codeword(&[], BinaryField128::ONE).is_empty());
    }

    /// `f_0 + beta * f_1`, the novel-basis combination, is a different operation from
    /// `fold_pair`'s evaluation-basis combination: the two differ by exactly `beta * f_0`, so
    /// they agree only when that term vanishes.
    #[test]
    fn the_novel_basis_fold_is_a_different_operation() {
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

            // At beta = 0 the fold is f0, pinning fold_pair's internal domain index against one
            // computed independently here.
            prop_assert_eq!(fold_pair(j, BinaryField128::ZERO, lo, hi), f0);
        }
    }
}
