use alloc::vec::Vec;

use p3_challenger::{CanSampleUniformBits, FieldChallenger};
use p3_field::{ExtensionField, Field};
use p3_util::log2_strict_usize;

/// Sample `t` distinct STIR query indices uniformly from the transcript.
///
/// # Pipeline
///
/// ```text
///   transcript --> uniform draws --> reject collisions --> sort
///                  [0, 2^k)          distinct              ascending
/// ```
///
/// with `k = log2(folded_domain_size)` and
/// `folded_domain_size = domain_size >> folding_factor`.
///
/// # Output
///
/// - length = `t = min(num_queries, folded_domain_size)`
/// - range  = `[0, folded_domain_size)`
/// - order  = strictly ascending, pairwise distinct, uniform per index
///
/// # Soundness
///
/// WHIR shift-query bound (Arnon-Chiesa-Fenzi-Yogev 2024, Thm 5.2):
///
/// ```text
///   eps_shift  <=  (1 - delta)^t
/// ```
///
/// `t` counts independent **uniformly-sampled** positions; distinctness
/// is not required (collisions waste opening work but do not weaken the
/// bound). Per-draw uniformity is the only leak closed here:
///
/// - **Biased draws** — bit-decomposing a uniform field element biases
///   each draw by `~ 2^bits / |F|`, which inflates `delta`. Routed
///   through `sample_uniform_bits` for exact uniformity.
///
/// Duplicate rejection is a cleanliness choice — it pins output length
/// at `t = min(num_queries, folded_domain_size)` and unifies the
/// common-case and saturation-case paths below.
///
/// # Saturation
///
/// `num_queries > folded_domain_size` returns the full domain. This is
/// WHIR's final round: 1-4 folded positions vs. `final_queries` up to 75.
///
/// # Cost
///
/// ```text
///   per draw         | 1-2 field samples, reject prob ~ 1 / |F|
///   loop, common     | O(t)
///   loop, saturated  | O(t log t)   (coupon collector)
/// ```
///
/// # Panics
///
/// `domain_size >> folding_factor` must be a power of two.
///
/// # TODO (possible recursion ideas)
///
/// - Prefer `n` independent draws (with duplicates allowed) over
///   `n` distinct indices for recursion-friendliness.
/// - The WHIR paper's `(1 - delta)^t` bound is over independent
///   draws; distinctness only lets `t` shrink slightly for the
///   same security, with negligible practical effect.
/// - Revisit when wiring this through a recursive verifier.
pub fn get_challenge_stir_queries<Challenger, F, EF>(
    domain_size: usize,
    folding_factor: usize,
    num_queries: usize,
    challenger: &mut Challenger,
) -> Vec<usize>
where
    Challenger: FieldChallenger<F> + CanSampleUniformBits<F>,
    F: Field,
    EF: ExtensionField<F>,
{
    // Phase 1: derive the addressable folded domain.
    //
    //   folded_domain_size = domain_size >> folding_factor
    //   k                  = log2(folded_domain_size)
    //
    // Each index fits in `k` bits.
    let folded_domain_size = domain_size >> folding_factor;
    let domain_size_bits = log2_strict_usize(folded_domain_size);

    // Phase 2: cap the request at the domain size.
    //
    //   num_queries <= folded_domain_size  ->  target = num_queries
    //   num_queries  > folded_domain_size  ->  target = folded_domain_size
    //                                          (open every position)
    let target = num_queries.min(folded_domain_size);

    // Phase 3: rejection-sample distinct indices.
    //
    //   loop:
    //     q <- uniform on [0, 2^k)
    //     append q if not already present
    //   until len == target
    let mut queries: Vec<usize> = Vec::with_capacity(target);
    while queries.len() < target {
        // RESAMPLE = true: the impl loops on field-side rejection internally.
        //
        // So the error arm is unreachable for every challenger in this workspace.
        let q = challenger
            .sample_uniform_bits::<true>(domain_size_bits)
            .expect("RESAMPLE = true: rejection loops internally, never errors");

        if !queries.contains(&q) {
            queries.push(q);
        }
    }

    // Phase 4: verifier and Merkle-proof code consume ascending indices.
    queries.sort_unstable();
    queries
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

    /// Build a deterministic duplex challenger from a seed.
    ///
    /// - Permutation: fixed across calls (seed-independent)
    /// - Transcript prefix: derived from `seed`
    ///
    /// Same seed -> byte-identical sample stream.
    fn challenger_with_seed(seed: u64) -> MyChallenger {
        // Permutation: deterministic across all test runs.
        let mut perm_rng = SmallRng::seed_from_u64(42);
        let perm = Perm::new_from_rng_128(&mut perm_rng);
        let mut challenger = MyChallenger::new(perm);

        // Transcript primer: 8 field elements derived from `seed`. This is
        // what makes two challengers with the same seed byte-identical.
        let mut transcript_rng = SmallRng::seed_from_u64(seed);
        let primer: Vec<F> = (0..8).map(|_| transcript_rng.random()).collect();
        challenger.observe_slice(&primer);
        challenger
    }

    /// Strategy for non-saturated `(domain_size, folding_factor, num_queries)`.
    ///
    /// ```text
    ///   log_folded     in [1, 8]   -> folded in [2, 256]
    ///   folding_factor in [0, 4]
    ///   num_queries    in [1, folded]
    /// ```
    ///
    /// Saturation (`num_queries > folded`) is covered by a dedicated test.
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

        #[test]
        fn prop_get_challenge_stir_queries_invariants(
            (domain_size, folding_factor, num_queries) in arb_query_params(),
            seed in any::<u64>(),
        ) {
            // Five invariants, one challenger setup per case:
            // (1) length
            // (2) range
            // (3) strict sort
            // (4) distinct
            // (5) deterministic replay
            let folded_domain_size = domain_size >> folding_factor;

            // First run: seed -> queries_a.
            let mut challenger_a = challenger_with_seed(seed);
            let queries_a = get_challenge_stir_queries::<MyChallenger, F, EF>(
                domain_size,
                folding_factor,
                num_queries,
                &mut challenger_a,
            );

            // (1) length == request.
            prop_assert_eq!(queries_a.len(), num_queries);

            // (2) every index in [0, folded_domain_size).
            for &q in &queries_a {
                prop_assert!(
                    q < folded_domain_size,
                    "out of range: {} not in [0, {})", q, folded_domain_size
                );
            }

            // (3) strictly ascending.
            prop_assert!(
                queries_a.windows(2).all(|w| w[0] < w[1]),
                "not sorted: {:?}", queries_a
            );

            // (4) pairwise distinct -- redundant given (3); set check
            //     guards against a future weakening of (3).
            let unique: BTreeSet<usize> = queries_a.iter().copied().collect();
            prop_assert_eq!(unique.len(), num_queries, "duplicates in {:?}", queries_a);

            // (5) determinism: same seed -> byte-identical output.
            //     This is the prover/verifier Fiat-Shamir replay property.
            let mut challenger_b = challenger_with_seed(seed);
            let queries_b = get_challenge_stir_queries::<MyChallenger, F, EF>(
                domain_size,
                folding_factor,
                num_queries,
                &mut challenger_b,
            );
            prop_assert_eq!(queries_a, queries_b);
        }
    }

    #[test]
    fn saturates_when_num_queries_exceeds_domain() {
        // WHIR final-round regime: ask for more queries than positions.
        //
        //   folded_domain_size = 16 >> 2 = 4
        //   num_queries        = 75            (>> 4)
        //   expected           = [0, 1, 2, 3]  (full domain, ascending)
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
        );

        // Length capped at the domain; output is the full domain ascending.
        assert_eq!(queries.len(), folded_domain_size);
        assert_eq!(queries, (0..folded_domain_size).collect::<Vec<_>>());
    }

    #[test]
    fn empirical_uniformity_single_query() {
        // Histogram-based regression detector for uniform sampling.
        //
        //   N = 16 buckets, M = 4096 draws, expected = M/N = 256
        //
        // Hoeffding on the bucket indicator 1[q == k]:
        //
        //   Pr[ |count_k - 256| >= t ]  <=  2 * exp(-2 t^2 / M)
        //
        // With t = 160:
        //
        //   per-bucket    : 2  * exp(-12.5) ~ 7.4e-6
        //   union 16 bins : 32 * exp(-12.5) ~ 1.2e-4
        //
        // Cryptographic uniformity rests on the field-side rejection
        // primitive; this test only catches gross regressions.
        const FOLDED_DOMAIN_SIZE: usize = 16;
        const NUM_DRAWS: usize = 4096;
        const TOLERANCE: usize = 160;

        // Pick parameters so the folded domain has exactly 16 positions.
        let domain_size: usize = 64;
        let folding_factor: usize = 2;
        assert_eq!(domain_size >> folding_factor, FOLDED_DOMAIN_SIZE);

        // Histogram one draw per challenger seed.
        let mut counts = [0usize; FOLDED_DOMAIN_SIZE];
        // Histogram: one draw per seed.
        for seed in 0u64..NUM_DRAWS as u64 {
            let mut challenger = challenger_with_seed(seed);
            let q = get_challenge_stir_queries::<MyChallenger, F, EF>(
                domain_size,
                folding_factor,
                1,
                &mut challenger,
            );
            assert_eq!(q.len(), 1);
            counts[q[0]] += 1;
        }

        // Each bucket within +/- TOLERANCE of expected count.
        let expected = NUM_DRAWS / FOLDED_DOMAIN_SIZE;
        for (bucket, &count) in counts.iter().enumerate() {
            let deviation = count.abs_diff(expected);
            assert!(
                deviation <= TOLERANCE,
                "bucket {bucket}: count {count} deviates from {expected} by {deviation} > {TOLERANCE}; counts = {counts:?}"
            );
        }
    }
}
