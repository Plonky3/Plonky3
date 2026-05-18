use alloc::vec::Vec;

use p3_challenger::{CanSampleUniformBits, FieldChallenger};
use p3_field::{ExtensionField, Field};
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;
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

/// Compute the r' correction to a STIR opening at position `challenge_idx`.
///
/// When the committed codeword is `Enc(f ∥ r')` (ZK padded, Suffix order),
/// the opened row at `challenge_idx` includes the r' contribution. This
/// function computes that contribution so the prover can subtract it before
/// adding the eval to the sumcheck constraint.
///
/// `dft_root` must be the primitive `height`-th root of unity used by
/// `dft_algebra_batch` when encoding the committed matrix. This is
/// `F::two_adic_generator(log2(height))`, NOT `folded_domain_gen`.
///
/// Layout (Suffix, `commit_extension_zk`): coefficient `i` goes to
/// row `i / width`, column `i % width`. DFT applied column-wise.
/// r' starts at global index `msg_len`.
pub(crate) fn zk_stir_correction<F, EF>(
    r_prime: &[EF],
    msg_len: usize,
    width: usize,
    height: usize,
    challenge_idx: usize,
    dft_root: F,
    query_randomness: &[EF],
) -> EF
where
    F: Field,
    EF: ExtensionField<F>,
{
    let omega = dft_root;
    let mut correction_row = alloc::vec![EF::ZERO; width];

    for (k, &r_k) in r_prime.iter().enumerate() {
        let global_idx = msg_len + k;
        let col = global_idx % width;
        let row = global_idx / width;
        if row >= height {
            break;
        }
        correction_row[col] += r_k * omega.exp_u64((challenge_idx * row) as u64);
    }

    Poly::new(correction_row).eval_ext::<F>(&Point::new(query_randomness.to_vec()))
}

#[cfg(test)]
mod tests {
    use alloc::collections::BTreeSet;
    use alloc::vec::Vec;

    use p3_challenger::{CanObserve, DuplexChallenger};
    use p3_field::extension::BinomialExtensionField;
    use p3_field::{PrimeCharacteristicRing, TwoAdicField};
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
    fn zk_stir_correction_matches_dft_ground_truth() {
        use p3_dft::{Radix2Dit, TwoAdicSubgroupDft};
        use p3_matrix::Matrix;
        use p3_matrix::dense::RowMajorMatrix;

        // Small polynomial: 4 variables = 16 coefficients.
        // Folding factor 2: width=4, msg rows = 4.
        // inv_rate 2: height = 2 * 4 = 8.
        let num_vars = 4usize;
        let folding = 2usize;
        let inv_rate = 2usize;
        let width = 1 << folding; // 4
        let msg_len = 1 << num_vars; // 16
        let height = inv_rate * (msg_len / width); // 8
        let total = height * width; // 32

        // Deterministic polynomial coefficients.
        let f_coeffs: Vec<EF> = (0..msg_len)
            .map(|i| EF::from(F::from_u64((i as u64 + 1) * 7)))
            .collect();

        // r' randomness: num_queries worth. Use 5 for a small test.
        let num_r_prime = 5;
        let r_prime: Vec<EF> = (0..num_r_prime)
            .map(|i| EF::from(F::from_u64((i as u64 + 1) * 13 + 3)))
            .collect();

        // Build padded coefficients: f ∥ r' ∥ zeros
        let mut padded_coeffs = Vec::with_capacity(total);
        padded_coeffs.extend_from_slice(&f_coeffs);
        padded_coeffs.extend_from_slice(&r_prime);
        padded_coeffs.resize(total, EF::ZERO);

        // Build unpadded coefficients: f ∥ zeros
        let mut plain_coeffs = Vec::with_capacity(total);
        plain_coeffs.extend_from_slice(&f_coeffs);
        plain_coeffs.resize(total, EF::ZERO);

        // DFT both column-wise.
        let dft: Radix2Dit<F> = Radix2Dit::default();
        let padded_mat = RowMajorMatrix::new(padded_coeffs, width);
        let plain_mat = RowMajorMatrix::new(plain_coeffs, width);
        let padded_encoded = dft.dft_algebra_batch(padded_mat);
        let plain_encoded = dft.dft_algebra_batch(plain_mat);

        // Query randomness (folding_factor many elements).
        let query_rand: Vec<EF> = (0..folding)
            .map(|i| EF::from(F::from_u64(i as u64 * 11 + 5)))
            .collect();

        let dft_root = F::two_adic_generator(p3_util::log2_strict_usize(height));

        // Test several challenge indices.
        for challenge_idx in [0, 1, 3, 7] {
            let padded_row: Vec<EF> = padded_encoded.row_slice(challenge_idx).unwrap().to_vec();
            let plain_row: Vec<EF> = plain_encoded.row_slice(challenge_idx).unwrap().to_vec();

            let padded_eval = Poly::new(padded_row).eval_ext::<F>(&Point::new(query_rand.clone()));
            let plain_eval = Poly::new(plain_row).eval_ext::<F>(&Point::new(query_rand.clone()));
            let ground_truth = padded_eval - plain_eval;

            let correction = super::zk_stir_correction(
                &r_prime,
                msg_len,
                width,
                height,
                challenge_idx,
                dft_root,
                &query_rand,
            );

            assert_eq!(
                correction, ground_truth,
                "correction mismatch at challenge_idx={challenge_idx}"
            );
        }
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
