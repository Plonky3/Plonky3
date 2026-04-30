use alloc::vec::Vec;

use p3_challenger::{CanSampleUniformBits, FieldChallenger};
use p3_field::{ExtensionField, Field};
use p3_util::log2_strict_usize;

use crate::fiat_shamir::errors::FiatShamirError;

/// Sample distinct STIR query indices in `[0, domain_size >> folding_factor)`
/// from the Fiat–Shamir transcript.
///
/// Indices are drawn uniformly **without replacement** and returned sorted.
/// The output length is `min(num_queries, folded_domain_size)`: when the
/// requested count exceeds the folded domain (typical of WHIR's final round),
/// the verifier opens every position, which is the strongest possible check.
///
/// # Soundness
///
/// The WHIR shift-query bound (Arnon, Chiesa, Fenzi, Yogev 2024,
/// Theorem 5.2) is `ε^shift ≤ (1 - δ)^t`, where `t` is the number of
/// *distinct* query positions. A sample-with-replacement implementation
/// followed by `dedup()` leaks soundness in two ways:
///
/// 1. **Modular bias.** [`p3_challenger::CanSampleBits::sample_bits`]
///    bit-decomposes a uniformly drawn field element and is documented
///    as "reasonably close to" — not exactly — uniform. The per-draw
///    bias is bounded by `2^bits / |F|` but non-zero, and inflates the
///    effective `δ` in the bound.
/// 2. **Birthday-paradox shrinkage.** A `dedup()` pass on collisions
///    returns fewer than `num_queries` distinct positions, weakening
///    the effective `t`.
///
/// This implementation closes both gaps:
///
/// - [`CanSampleUniformBits::sample_uniform_bits`] with `RESAMPLE = true`
///   uses field-side rejection sampling so each draw is exactly uniform
///   on `[0, 2^bits)`; and
/// - duplicates are rejected so the returned vector has length exactly
///   `min(num_queries, folded_domain_size)`, matching the soundness
///   bound's `t` tightly.
///
/// # Performance
///
/// `sample_uniform_bits::<true>(bits)` is essentially free for
/// `bits ≤ MAX_SINGLE_SAMPLE_BITS` on small fields (single field draw,
/// resample probability `≈ 1 / |F|`). The duplicate-rejection loop runs
/// in `O(target)` iterations in expectation when `folded_domain_size`
/// is much larger than `target` (the typical WHIR regime), and in
/// `O(target · log target)` in the worst case when `target` approaches
/// `folded_domain_size` (coupon-collector behaviour).
///
/// # Panics
///
/// Panics if `domain_size >> folding_factor` is not a power of two
/// (precondition of [`log2_strict_usize`]).
pub fn get_challenge_stir_queries<Challenger, F, EF>(
    domain_size: usize,
    folding_factor: usize,
    num_queries: usize,
    challenger: &mut Challenger,
) -> Result<Vec<usize>, FiatShamirError>
where
    Challenger: FieldChallenger<F> + CanSampleUniformBits<F>,
    F: Field,
    EF: ExtensionField<F>,
{
    let folded_domain_size = domain_size >> folding_factor;
    let domain_size_bits = log2_strict_usize(folded_domain_size);

    // Cap requested queries at the domain size. When `num_queries` would
    // exceed `folded_domain_size`, the only set of distinct positions of
    // the requested size is the entire domain.
    let target = num_queries.min(folded_domain_size);

    let mut queries: Vec<usize> = Vec::with_capacity(target);
    while queries.len() < target {
        let q = challenger
            .sample_uniform_bits::<true>(domain_size_bits)
            .expect("Error impossible here due to resampling strategy");
        if !queries.contains(&q) {
            queries.push(q);
        }
    }
    queries.sort_unstable();
    Ok(queries)
}

#[cfg(test)]
mod tests {
    use alloc::collections::BTreeSet;
    use alloc::vec::Vec;

    use p3_challenger::{CanObserve, DuplexChallenger};
    use p3_field::extension::BinomialExtensionField;
    use p3_koala_bear::{KoalaBear, Poseidon2KoalaBear};
    use proptest::prelude::*;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::*;

    type F = KoalaBear;
    type EF = BinomialExtensionField<F, 4>;
    type Perm = Poseidon2KoalaBear<16>;
    type MyChallenger = DuplexChallenger<F, Perm, 16, 8>;

    /// Build a `DuplexChallenger` whose Poseidon2 instance is fixed (seed 42)
    /// and whose absorbed transcript is determined by `seed`.
    ///
    /// Two challengers built from the same `seed` are byte-identical and will
    /// therefore produce identical sample sequences — that is what the
    /// determinism property asserts.
    fn challenger_with_seed(seed: u64) -> MyChallenger {
        let mut perm_rng = SmallRng::seed_from_u64(42);
        let perm = Perm::new_from_rng_128(&mut perm_rng);
        let mut challenger = MyChallenger::new(perm);

        let mut transcript_rng = SmallRng::seed_from_u64(seed);
        let primer: Vec<F> = (0..8).map(|_| transcript_rng.random()).collect();
        challenger.observe_slice(&primer);
        challenger
    }

    /// Generate `(domain_size, folding_factor, num_queries)` such that
    /// `folded = domain_size >> folding_factor` is a positive power of two
    /// and `num_queries ∈ [1, folded]`. This keeps the strategy in the
    /// "common case" regime where the function returns exactly `num_queries`
    /// distinct positions; the saturation case is covered separately below.
    fn arb_query_params() -> impl Strategy<Value = (usize, usize, usize)> {
        (1usize..=8, 0usize..=4).prop_flat_map(|(log_folded, folding_factor)| {
            let folded = 1usize << log_folded;
            let domain_size = folded << folding_factor;
            (1usize..=folded)
                .prop_map(move |num_queries| (domain_size, folding_factor, num_queries))
        })
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        /// Length, range, sortedness, distinctness, and determinism in a
        /// single property so each case pays for one challenger setup.
        #[test]
        fn prop_get_challenge_stir_queries_invariants(
            (domain_size, folding_factor, num_queries) in arb_query_params(),
            seed in any::<u64>(),
        ) {
            let folded_domain_size = domain_size >> folding_factor;

            let mut challenger_a = challenger_with_seed(seed);
            let queries_a = get_challenge_stir_queries::<MyChallenger, F, EF>(
                domain_size,
                folding_factor,
                num_queries,
                &mut challenger_a,
            )
            .expect("sampling under RESAMPLE = true cannot fail");

            // 1. Length: exactly num_queries when num_queries ≤ folded_domain_size.
            prop_assert_eq!(queries_a.len(), num_queries);

            // 2. Range: every q in [0, folded_domain_size).
            for &q in &queries_a {
                prop_assert!(
                    q < folded_domain_size,
                    "query {} out of range [0, {})", q, folded_domain_size
                );
            }

            // 3. Sortedness (ascending).
            prop_assert!(
                queries_a.windows(2).all(|w| w[0] < w[1]),
                "queries not strictly sorted: {:?}", queries_a
            );

            // 4. Distinctness (already implied by strict sort, but assert
            //    explicitly via a set for clarity).
            let unique: BTreeSet<usize> = queries_a.iter().copied().collect();
            prop_assert_eq!(unique.len(), num_queries, "duplicates in {:?}", queries_a);

            // 5. Determinism: a fresh challenger from the same seed yields
            //    a byte-identical query vector.
            let mut challenger_b = challenger_with_seed(seed);
            let queries_b = get_challenge_stir_queries::<MyChallenger, F, EF>(
                domain_size,
                folding_factor,
                num_queries,
                &mut challenger_b,
            )
            .expect("sampling under RESAMPLE = true cannot fail");
            prop_assert_eq!(queries_a, queries_b);
        }
    }

    /// Saturation: when `num_queries > folded_domain_size`, return every
    /// position. This is the WHIR final-round regime where the folded domain
    /// is small (often 1–4) but `final_queries` can be much larger.
    #[test]
    fn saturates_when_num_queries_exceeds_domain() {
        // Folded domain of size 4 (= 16 >> 2); request 75 queries.
        let domain_size = 16usize;
        let folding_factor = 2usize;
        let folded_domain_size = domain_size >> folding_factor;
        let num_queries = 75usize;

        let mut challenger = challenger_with_seed(0xC0FFEE);
        let queries = get_challenge_stir_queries::<MyChallenger, F, EF>(
            domain_size,
            folding_factor,
            num_queries,
            &mut challenger,
        )
        .expect("sampling under RESAMPLE = true cannot fail");

        assert_eq!(queries.len(), folded_domain_size);
        assert_eq!(queries, (0..folded_domain_size).collect::<Vec<_>>());
    }

    /// Empirical uniformity check for single-query draws on a small folded
    /// domain.
    ///
    /// We draw one query per fresh challenger (seeded by a counter) and bin
    /// the results into `N = 16` buckets. With `M = 4096` independent draws
    /// and per-bucket expectation `M/N = 256`, Hoeffding on the indicator
    /// `1[q == k]` gives, for any fixed bucket `k`,
    ///
    ///     Pr[ |count_k − M/N| ≥ t ] ≤ 2 · exp(−2 t² / M).
    ///
    /// Tolerance `t = 160` gives `32 · exp(−160² / 2048) ≈ 1.2 · 10⁻⁴` after
    /// union-bounding both tails over 16 buckets — comfortably below CI's
    /// flake threshold.
    ///
    /// This is a plain `#[test]`, not a proptest: the challenger already
    /// supplies all the randomness needed; stacking proptest on top would
    /// only inflate the bound.
    #[test]
    fn empirical_uniformity_single_query() {
        const FOLDED_DOMAIN_SIZE: usize = 16;
        const NUM_DRAWS: usize = 4096;
        const TOLERANCE: usize = 160;

        let domain_size: usize = 64;
        let folding_factor: usize = 2;
        assert_eq!(domain_size >> folding_factor, FOLDED_DOMAIN_SIZE);

        let mut counts = [0usize; FOLDED_DOMAIN_SIZE];
        for seed in 0u64..NUM_DRAWS as u64 {
            let mut challenger = challenger_with_seed(seed);
            let q = get_challenge_stir_queries::<MyChallenger, F, EF>(
                domain_size,
                folding_factor,
                1,
                &mut challenger,
            )
            .expect("sampling under RESAMPLE = true cannot fail");
            assert_eq!(q.len(), 1);
            counts[q[0]] += 1;
        }

        let expected = NUM_DRAWS / FOLDED_DOMAIN_SIZE;
        for (bucket, &count) in counts.iter().enumerate() {
            let deviation = count.abs_diff(expected);
            assert!(
                deviation <= TOLERANCE,
                "bucket {bucket}: count {count} deviates from expected {expected} by {deviation} > {TOLERANCE}; full counts = {counts:?}"
            );
        }
    }
}
