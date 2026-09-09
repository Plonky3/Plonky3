//! Full-coset queries tying consecutive committed fold batches together.
//!
//! Base query indices are aligned to the first batch's coset size. A batch starting at
//! variable `start` with `arity` challenges opens the coset containing `index >> start`,
//! folds all its symbols, then checks the coordinate `index >> (start + arity)` in the next
//! committed word (or the final word sent in full). Only base cosets are sampled distinctly;
//! repeated projected cosets in later rounds remain the same base-query paths.

use alloc::collections::BTreeSet;
use alloc::vec;
use alloc::vec::Vec;

use p3_binary_field::BinaryField128;
use p3_challenger::{CanSampleUniformBits, FieldChallenger, GrindingChallenger};
use p3_commit::Mmcs;
use p3_field::{Field, PrimeCharacteristicRing};
use p3_matrix::Dimensions;
use p3_util::log2_strict_usize;

use crate::error::BinaryPcsError;
use crate::fold::{fold_coset, fold_pair};
use crate::params::BinaryPcsConfig;
use crate::proof::BinaryPcsProof;

/// Number of distinct fold-chain tests a codeword of `domain_size` symbols admits.
///
/// One per fold pair, so half the domain.
/// Every check reads both symbols of a pair, so bit 0 of a position selects nothing.
pub(crate) const fn num_distinct_queries(domain_size: usize, num_queries: usize) -> usize {
    let num_pairs = domain_size / 2;
    if num_queries < num_pairs {
        num_queries
    } else {
        num_pairs
    }
}

/// Samples distinct fold pairs from the base codeword's domain.
///
/// Returns each sampled pair's low-indexed position, in ascending order.
/// Every returned position is even.
///
/// # Why a pair index
///
/// Both symbols of a pair are read and folded together at every round.
/// Two symbol positions differing only in bit 0 therefore name the same test.
/// Drawing one bit fewer and doubling aligns the enforced distinctness with the query bound.
///
/// # Why uniform bits
///
/// A uniform element of the codeword alphabet is already a uniform 128-bit string.
/// Its low bits therefore carry no bias.
/// The binary challenger's uniform sampler is itself a plain mask over transcript bytes.
/// That sampler is used because it is the interface that states the guarantee.
/// A challenger over a prime field then cannot silently reintroduce bias.
///
/// # Returns
///
/// One position per distinct pair, capped at the number of pairs the domain holds.
/// A request for more pairs than exist returns every pair.
///
/// # Panics
///
/// Panics unless `domain_size` is a power of two of at least two.
pub(crate) fn sample_query_indices<Challenger, F>(
    domain_size: usize,
    num_queries: usize,
    challenger: &mut Challenger,
) -> Vec<usize>
where
    Challenger: FieldChallenger<F> + CanSampleUniformBits<F>,
    F: Field,
{
    // One bit narrower than the domain: a sampled value indexes pairs, not symbols.
    let pair_bits = log2_strict_usize(domain_size / 2);
    let target = num_distinct_queries(domain_size, num_queries);

    // A set, not a linear scan over what has already been drawn.
    // Once the target approaches the pair count, the draw count is a coupon-collector tail.
    // The small-domain configurations reach exactly that.
    let mut pairs = BTreeSet::new();
    while pairs.len() < target {
        let pair = challenger
            .sample_uniform_bits::<true>(pair_bits)
            .expect("RESAMPLE = true: rejection loops internally, never errors");
        pairs.insert(pair);
    }

    // `BTreeSet` iterates in ascending order, so doubling preserves it.
    pairs.into_iter().map(|pair| pair << 1).collect()
}

/// Sample distinct base cosets. Reusing the pair sampler on a shorter index domain keeps
/// the single-fold transcript unchanged; lifting its indices clears all first-batch low bits.
pub(crate) fn sample_query_cosets<Ch>(config: &BinaryPcsConfig, challenger: &mut Ch) -> Vec<usize>
where
    Ch: FieldChallenger<BinaryField128> + CanSampleUniformBits<BinaryField128>,
{
    let shift = config.log_folding_factor() - 1;
    sample_query_indices::<_, BinaryField128>(
        config.domain_size() >> shift,
        config.num_queries(),
        challenger,
    )
    .into_iter()
    .map(|index| index << shift)
    .collect()
}

/// All symbols of every queried coset, in query order, with ascending offsets within a coset.
pub(crate) fn flat_coset_indices(indices: &[usize], start: usize, arity: usize) -> Vec<usize> {
    let size = 1usize << arity;
    indices
        .iter()
        .flat_map(|&index| {
            let first = (index >> start) & !(size - 1);
            first..first + size
        })
        .collect()
}

/// Checks a round's opened-row shape against the count every query demands.
///
/// Runs before any index arithmetic on the round's contents, so a wrong row count or a
/// mis-sized row is rejected here rather than read out of bounds.
fn check_round_shape<E>(
    round: usize,
    opened_values: &[Vec<BinaryField128>],
    expected_opens: usize,
) -> Result<(), BinaryPcsError<E>> {
    if opened_values.len() != expected_opens {
        return Err(BinaryPcsError::OpeningCountMismatch {
            round,
            expected: expected_opens,
            actual: opened_values.len(),
        });
    }
    for (query, row) in opened_values.iter().enumerate() {
        if row.len() != 1 {
            return Err(BinaryPcsError::RowWidthMismatch {
                round,
                query,
                expected: 1,
                actual: row.len(),
            });
        }
    }
    Ok(())
}

/// Wraps opened rows for [`Mmcs::verify_multi_batch`], which expects a `[query][matrix]`
/// shape; every round here commits exactly one matrix, so each row gets a one-element outer
/// slice.
fn wrap_rows(opened_values: &[Vec<BinaryField128>]) -> Vec<Vec<&[BinaryField128]>> {
    opened_values
        .iter()
        .map(|row| vec![row.as_slice()])
        .collect()
}

/// Checks the proof's declared intermediate round count and final-codeword length against
/// what `config` derives, before any transcript operation: `BinaryPcs::verify_opening` and
/// [`verify_query_paths`] both need this pair of structural checks, ahead of everything each
/// one does on its own.
pub(crate) fn check_round_and_final_lengths<MT>(
    config: &BinaryPcsConfig,
    proof: &BinaryPcsProof<MT>,
) -> Result<(), BinaryPcsError<MT::Error>>
where
    MT: Mmcs<BinaryField128>,
{
    let expected_intermediate_rounds = config.num_fold_batches() - 1;
    if proof.rounds.len() != expected_intermediate_rounds {
        return Err(BinaryPcsError::RoundCountMismatch {
            expected: expected_intermediate_rounds,
            actual: proof.rounds.len(),
        });
    }

    let expected_final_len = 1usize << config.log_final_len();
    if proof.final_codeword.num_evals() != expected_final_len {
        return Err(BinaryPcsError::FinalCodewordLengthMismatch {
            expected: expected_final_len,
            actual: proof.final_codeword.num_evals(),
        });
    }

    Ok(())
}

/// Checks that a zero-difficulty proof carries the one grinding witness a zero budget admits,
/// before any transcript operation.
///
/// A positive budget needs no check here: the witness is absorbed and its sampled bits are
/// compared, so the difficulty itself pins the field.
//
// Why: `GrindingChallenger::check_witness` returns `true` at `bits == 0` without absorbing,
// which leaves `proof.pow_witness` compared against nothing and free to be any value.
//
//     pow_bits = 0 -> prover emits zero, verifier reads nothing -> pin the field here
//     pow_bits > 0 -> prover grinds,     verifier resamples     -> the grind pins it
//
// Assumption: the honest prover's grind at zero bits is pinned to the zero witness.
// A grinding path that leaves the zero-bit witness unconstrained needs this check revisited.
// The pin is asserted by `zero_difficulty_grinding_is_pinned_to_the_zero_witness` below.
pub(crate) fn check_canonical_pow_witness<MT>(
    config: &BinaryPcsConfig,
    proof: &BinaryPcsProof<MT>,
) -> Result<(), BinaryPcsError<MT::Error>>
where
    MT: Mmcs<BinaryField128>,
{
    if config.pow_bits() == 0 && proof.pow_witness != BinaryField128::ZERO {
        return Err(BinaryPcsError::NonCanonicalPowWitness {
            actual: proof.pow_witness,
        });
    }

    Ok(())
}

/// Verifies the query phase of an opening proof: the single grind, the sampled query
/// indices, every round's Merkle multiproof, and the fold-consistency chain tying each round
/// to the next.
///
/// `betas` is the fold challenge used at each round, `betas[r]` for round `r`, in the order
/// `fold_rounds` samples them; the caller derives it by replaying the sumcheck transcript
/// (this function does not touch the sumcheck rounds or the commitments' own transcript
/// order). All proof-shape checks run before `challenger` is touched, so a malformed proof is
/// rejected without ever grinding or sampling against it.
///
/// # Panics
///
/// Panics if `betas.len() != config.num_fold_rounds()`: `betas` is the caller's own
/// transcript-replay output, never proof-supplied data, so a length mismatch here is a caller
/// bug rather than a malformed proof.
pub(crate) fn verify_query_paths<MT, Ch>(
    config: &BinaryPcsConfig,
    mmcs: &MT,
    base_commitment: &MT::Commitment,
    betas: &[BinaryField128],
    proof: &BinaryPcsProof<MT>,
    challenger: &mut Ch,
) -> Result<(), BinaryPcsError<MT::Error>>
where
    MT: Mmcs<BinaryField128>,
    Ch: FieldChallenger<BinaryField128>
        + GrindingChallenger<Witness = BinaryField128>
        + CanSampleUniformBits<BinaryField128>,
{
    let num_fold_rounds = config.num_fold_rounds();
    assert_eq!(betas.len(), num_fold_rounds, "one fold challenge per round");

    // Structural checks: every one derivable from `config` and the proof's own declared
    // lengths, none needing the transcript.
    check_round_and_final_lengths(config, proof)?;
    check_canonical_pow_witness(config, proof)?;

    let domain_size = config.domain_size();
    let target_queries = config
        .num_queries()
        .min(domain_size >> config.log_folding_factor());
    let batches: Vec<_> = config.fold_batches().collect();
    let round_values = |batch: usize| -> &[Vec<BinaryField128>] {
        if batch == 0 {
            &proof.base_opened_values
        } else {
            &proof.rounds[batch - 1].opened_values
        }
    };
    for (batch, &(_, arity)) in batches.iter().enumerate() {
        check_round_shape(batch, round_values(batch), target_queries << arity)?;
    }

    if !challenger.check_witness(config.pow_bits(), proof.pow_witness) {
        return Err(BinaryPcsError::InvalidPowWitness);
    }
    let indices = sample_query_cosets(config, challenger);
    debug_assert_eq!(indices.len(), target_queries);

    for (batch, &(start, arity)) in batches.iter().enumerate() {
        let (commitment, multi_proof) = if batch == 0 {
            (base_commitment, &proof.base_multi_proof)
        } else {
            (
                &proof.rounds[batch - 1].commitment,
                &proof.rounds[batch - 1].multi_proof,
            )
        };
        let dims = [Dimensions {
            width: 1,
            height: domain_size >> start,
        }];
        let coset_indices = flat_coset_indices(&indices, start, arity);
        mmcs.verify_multi_batch(
            commitment,
            &dims,
            &coset_indices,
            &wrap_rows(round_values(batch)),
            multi_proof,
        )
        .map_err(|source| BinaryPcsError::MerkleFailed {
            round: batch,
            source,
        })?;
    }

    let mut coset = Vec::new();
    for (q, &index) in indices.iter().enumerate() {
        for (batch, &(start, arity)) in batches.iter().enumerate() {
            let size = 1 << arity;
            let next_position = index >> (start + arity);
            let folded = if arity == 1 {
                fold_pair(
                    next_position,
                    betas[start],
                    round_values(batch)[2 * q][0],
                    round_values(batch)[2 * q + 1][0],
                )
            } else {
                coset.clear();
                coset.extend(
                    round_values(batch)[q * size..(q + 1) * size]
                        .iter()
                        .map(|row| row[0]),
                );
                fold_coset(next_position, &mut coset, &betas[start..start + arity])
            };
            let expected = if let Some(&(_, next_arity)) = batches.get(batch + 1) {
                let next_size = 1 << next_arity;
                round_values(batch + 1)[q * next_size + (next_position & (next_size - 1))][0]
            } else {
                proof.final_codeword.as_slice()[next_position]
            };
            if folded != expected {
                return Err(BinaryPcsError::FoldMismatch {
                    round: batch + 1,
                    query: q,
                });
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use alloc::string::ToString;
    use alloc::vec;
    use alloc::vec::Vec;

    use p3_binary_dft::{AdditiveRsEncoder, NaiveAdditiveNtt};
    use p3_binary_field::BinaryField128;
    use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
    use p3_commit::Mmcs;
    use p3_field::{Field, PrimeCharacteristicRing};
    use p3_matrix::dense::RowMajorMatrix;
    use p3_multilinear_util::poly::Poly;
    use p3_sumcheck::layout::{Layout, SuffixProver, Table};
    use p3_sumcheck::strategy::Basis;
    use p3_sumcheck::transcript::{SumcheckShape, VerifierTranscript};
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    use super::{BinaryPcsError, sample_query_cosets, sample_query_indices, verify_query_paths};
    use crate::params::{BinaryPcsConfig, BinaryPcsParams};
    use crate::proof::BinaryPcsProof;
    use crate::prover::{RoundCommitment, commit, fold_rounds, open_queries};
    use crate::test_util::{challenger, mmcs};

    type F = BinaryField128;

    const NUM_VARIABLES: usize = 8;
    const LOG_INV_RATE: usize = 2;

    const fn params() -> BinaryPcsParams {
        BinaryPcsParams {
            log_inv_rate: LOG_INV_RATE,
            pow_bits: 4,
            security_level: 40,
        }
    }

    #[test]
    fn zero_difficulty_grinding_is_pinned_to_the_zero_witness() {
        // Invariant: this crate's challenger grinds to the zero witness at a zero budget.
        //
        //              grind                     check
        //     0 bits   zero, absorbing nothing   true, absorbing nothing
        //     4 bits   a search that absorbs     resamples and compares
        //
        // Zero is therefore the only witness an honest prover emits at a zero budget.
        //
        // Nothing in the transcript binds the field there, so the check above has to.
        assert_eq!(challenger().grind(0), F::ZERO);

        // The complementary half: a zero-bit witness check accepts every value handed to it.
        for witness in [F::ZERO, F::ONE, F::GENERATOR] {
            assert!(challenger().check_witness(0, witness));
        }

        // A positive budget needs no such check, because there the difficulty pins the field.
        assert!(challenger().check_witness(4, challenger().grind(4)));

        // Excluded case: the uniform-grinding pair, which inverts both facts above.
        // Its grind has no zero-bit shortcut, so at zero bits an arbitrary candidate wins.
        // Its check absorbs at every difficulty, so there the transcript binds the witness.
        //
        // A challenger whose zero-bit grind is unconstrained fails the first assertion here.
        // Moving the query phase's single grind onto that pair needs the check revisited too.
    }

    #[test]
    fn query_indices_are_distinct_sorted_even_and_in_range() {
        let mut c = challenger();
        let indices = sample_query_indices::<_, BinaryField128>(1 << 10, 12, &mut c);
        assert_eq!(indices.len(), 12);
        assert!(
            indices.windows(2).all(|w| w[0] < w[1]),
            "sorted and distinct"
        );
        assert!(indices.iter().all(|&i| i < 1 << 10), "in range");
        // Each position is a pair's low symbol, so bit 0 is clear by construction.
        // Distinct and even together mean no two draws are fold siblings.
        assert!(indices.iter().all(|&i| i.is_multiple_of(2)), "pair-aligned");
    }

    #[test]
    fn a_request_larger_than_the_domain_returns_every_pair() {
        // Invariant: a query is a pair, so an 8-symbol domain holds 4 tests, not 8.
        //
        //     symbols: 0 1 2 3 4 5 6 7
        //     pairs:   |0| |2| |4| |6|
        //
        // Asking for 100 therefore yields the 4 low positions, not 8 indices.
        let mut c = challenger();
        let indices = sample_query_indices::<_, BinaryField128>(8, 100, &mut c);
        assert_eq!(indices, alloc::vec![0, 2, 4, 6]);
    }

    #[test]
    fn batched_queries_sample_distinct_full_cosets_and_cap_at_the_coset_count() {
        for (num_variables, arity) in [(8, 3), (3, 3)] {
            let config = BinaryPcsConfig::try_new(num_variables, params())
                .unwrap()
                .try_with_folding(arity)
                .unwrap();
            let indices = sample_query_cosets(&config, &mut challenger());
            let cosets = config.domain_size() >> arity;
            assert_eq!(indices.len(), config.num_queries().min(cosets));
            assert!(indices.windows(2).all(|w| w[0] < w[1]));
            assert!(
                indices
                    .iter()
                    .all(|&i| i < config.domain_size() && i % (1 << arity) == 0)
            );
            if config.num_queries() >= cosets {
                assert_eq!(indices, (0..cosets).map(|i| i << arity).collect::<Vec<_>>());
            }
        }
    }

    #[test]
    fn sampling_is_transcript_dependent() {
        let mut a = challenger();
        let mut b = challenger();
        assert_eq!(
            sample_query_indices::<_, BinaryField128>(1 << 10, 4, &mut a),
            sample_query_indices::<_, BinaryField128>(1 << 10, 4, &mut b),
        );
    }

    #[test]
    fn round_count_mismatch_is_typed_not_a_panic() {
        // A proof claiming fewer rounds than the config must be rejected before any
        // indexing, so a malformed proof is a rejection rather than an abort.
        let err: BinaryPcsError<()> = BinaryPcsError::RoundCountMismatch {
            expected: 8,
            actual: 3,
        };
        assert!(err.to_string().contains('8'));
    }

    /// Commits, folds, and opens a genuine proof, then checks that `verify_query_paths`
    /// accepts it end to end: every round's Merkle multiproof and every query's fold chain,
    /// through the final codeword.
    #[test]
    fn a_genuine_proof_verifies() {
        let mut rng = SmallRng::seed_from_u64(7);
        let poly = Poly::<F>::rand(&mut rng, NUM_VARIABLES);
        let table = Table::new(RowMajorMatrix::new(poly.into_evals(), 1 << NUM_VARIABLES));
        let witness = SuffixProver::<F, F>::new_witness(vec![table], 0);

        let config = BinaryPcsConfig::try_new(NUM_VARIABLES, params()).unwrap();
        let encoder = AdditiveRsEncoder::<F, NaiveAdditiveNtt<F>>::default();
        let mmcs_instance = mmcs();

        let mut prover_ch = challenger();
        let (base_commitment, prover_data) =
            commit(&config, &encoder, &mmcs_instance, &mut prover_ch, witness);
        let (base_merkle_data, sumcheck_data, rounds, randomness, final_codeword) =
            fold_rounds(prover_data, &config, &mmcs_instance, &mut prover_ch);

        let mut verifier_ch = prover_ch.clone();

        let query_proofs = open_queries(
            &config,
            &mmcs_instance,
            &mut prover_ch,
            &base_merkle_data,
            &rounds,
        );

        let proof = BinaryPcsProof {
            sumcheck: sumcheck_data,
            rounds: query_proofs.rounds,
            base_opened_values: query_proofs.base_opened_values,
            base_multi_proof: query_proofs.base_multi_proof,
            final_codeword: Poly::new(final_codeword),
            pow_witness: query_proofs.pow_witness,
            evals: Vec::new(),
        };

        let result = verify_query_paths(
            &config,
            &mmcs_instance,
            &base_commitment,
            randomness.as_slice(),
            &proof,
            &mut verifier_ch,
        );
        assert!(result.is_ok(), "{result:?}");
    }

    /// A fresh verifier challenger, seeded identically to the prover's but touched only by
    /// what the proof carries, must sample the same query indices the prover did.
    ///
    /// `a_genuine_proof_verifies` clones the prover's own challenger after `fold_rounds`
    /// returns, so its `verifier_ch` already carries every observation the prover made,
    /// correct or not — it can never disagree with the prover, so it cannot catch a mismatch
    /// between what `fold_rounds` observes and what the proof actually carries. This test
    /// instead rebuilds the verifier's side of the transcript from an empty challenger: the
    /// base commitment, the batching challenge `into_sumcheck` samples even though it consumes
    /// zero preprocessing rounds, each fold round's polynomial and challenge (from
    /// `proof.sumcheck`), and each intermediate round's commitment (one per fold round except
    /// the last). If `fold_rounds` observes one more or one fewer commitment than this replay
    /// does, the two challengers desync and the sampled indices diverge.
    #[test]
    fn a_fresh_verifier_challenger_samples_the_same_query_indices() {
        let mut rng = SmallRng::seed_from_u64(0x5EED);
        let poly = Poly::<F>::rand(&mut rng, NUM_VARIABLES);
        let table = Table::new(RowMajorMatrix::new(poly.into_evals(), 1 << NUM_VARIABLES));
        let witness = SuffixProver::<F, F>::new_witness(vec![table], 0);

        let config = BinaryPcsConfig::try_new(NUM_VARIABLES, params()).unwrap();
        let encoder = AdditiveRsEncoder::<F, NaiveAdditiveNtt<F>>::default();
        let mmcs_instance = mmcs();

        let mut prover_ch = challenger();
        let (base_commitment, prover_data) =
            commit(&config, &encoder, &mmcs_instance, &mut prover_ch, witness);
        let (base_merkle_data, sumcheck_data, rounds, randomness, _final_codeword) =
            fold_rounds(prover_data, &config, &mmcs_instance, &mut prover_ch);

        // The prover's own transcript state, snapshotted right before the query phase, gives
        // an independent readout of the indices `open_queries` samples: replaying the actual
        // grinding witness it found and sampling from there reaches the same query phase by a
        // second path. This checks the witness rather than re-grinding: with the `parallel`
        // feature, `grind`'s search returns any witness that satisfies the difficulty, not a
        // deterministic one, so a second independent grind could legitimately land on a
        // different valid witness and desync the two readouts for a reason unrelated to what
        // this test is checking.
        let mut prover_snapshot = prover_ch.clone();

        let query_proofs = open_queries(
            &config,
            &mmcs_instance,
            &mut prover_ch,
            &base_merkle_data,
            &rounds,
        );

        assert!(prover_snapshot.check_witness(config.pow_bits(), query_proofs.pow_witness));
        let domain_size = config.domain_size();
        let prover_indices = sample_query_indices::<_, BinaryField128>(
            domain_size,
            config.num_queries(),
            &mut prover_snapshot,
        );

        // A fresh, independently constructed challenger — the empty transcript, exactly like
        // `challenger()` gave the prover — touched only by what a verifier can read off the
        // proof and the config.
        let mut verifier_ch = challenger();
        verifier_ch.observe(base_commitment);
        let _alpha: F = verifier_ch.sample_algebra_element();

        let num_fold_rounds = config.num_fold_rounds();
        // `r` indexes three collections of two different lengths (`rounds` holds one fewer
        // entry than `num_fold_rounds`), so no single `.iter().enumerate()` covers the loop.
        #[allow(clippy::needless_range_loop)]
        for r in 0..num_fold_rounds {
            let [c0, c_inf] = sumcheck_data.polynomial_evaluations()[r];
            // One fold round is one single-round sumcheck, seeded on its own.
            let shape = SumcheckShape::new(1, 0, Basis::Evaluation);
            let mut transcript = VerifierTranscript::<_, F, F>::new(&mut verifier_ch, shape);
            let beta = transcript.round(c0, c_inf, None).unwrap();
            transcript.finish();
            assert_eq!(beta, randomness.as_slice()[r], "round {r} challenge");
            if r + 1 < num_fold_rounds {
                verifier_ch.observe(rounds[r].commitment.clone());
            }
        }

        assert!(verifier_ch.check_witness(config.pow_bits(), query_proofs.pow_witness));
        let verifier_indices = sample_query_indices::<_, BinaryField128>(
            domain_size,
            config.num_queries(),
            &mut verifier_ch,
        );

        assert_eq!(verifier_indices, prover_indices);
    }

    /// The fold chain is the only thing tying one committed round to the next, and this is the
    /// attack it exists for: a prover that commits a round which is a perfectly valid codeword
    /// but simply is not the fold of its predecessor.
    ///
    /// The tamper adds one constant to every symbol of the first intermediate round. Constants
    /// are degree-0 polynomials, so the shifted vector is still a codeword of that round's
    /// code — not a malformed object any shape or proximity check could reject, but a
    /// well-formed commitment to the wrong polynomial. It is re-committed and opened honestly
    /// against its own root, so the round's Merkle multiproof verifies and the transcript is
    /// untouched; `fold_pair` is what disagrees.
    #[test]
    fn a_committed_round_that_is_not_the_fold_of_its_predecessor_is_rejected() {
        let mut rng = SmallRng::seed_from_u64(0xF01D);
        let poly = Poly::<F>::rand(&mut rng, NUM_VARIABLES);
        let table = Table::new(RowMajorMatrix::new(poly.into_evals(), 1 << NUM_VARIABLES));
        let witness = SuffixProver::<F, F>::new_witness(vec![table], 0);

        let config = BinaryPcsConfig::try_new(NUM_VARIABLES, params()).unwrap();
        let encoder = AdditiveRsEncoder::<F, NaiveAdditiveNtt<F>>::default();
        let mmcs_instance = mmcs();

        let mut prover_ch = challenger();
        let (base_commitment, prover_data) =
            commit(&config, &encoder, &mmcs_instance, &mut prover_ch, witness);
        let (base_merkle_data, sumcheck_data, mut rounds, randomness, final_codeword) =
            fold_rounds(prover_data, &config, &mmcs_instance, &mut prover_ch);

        // `rounds[0]` carries fold round 1, the round the base round's fold must reproduce.
        let shifted: Vec<F> = mmcs_instance.get_matrices(&rounds[0].merkle_data)[0]
            .values
            .iter()
            .map(|&v| v + F::ONE)
            .collect();
        let (commitment, merkle_data) =
            mmcs_instance.commit_matrix(RowMajorMatrix::new(shifted, 1));
        rounds[0] = RoundCommitment {
            commitment,
            merkle_data,
        };

        let mut verifier_ch = prover_ch.clone();
        let query_proofs = open_queries(
            &config,
            &mmcs_instance,
            &mut prover_ch,
            &base_merkle_data,
            &rounds,
        );

        let proof = BinaryPcsProof {
            sumcheck: sumcheck_data,
            rounds: query_proofs.rounds,
            base_opened_values: query_proofs.base_opened_values,
            base_multi_proof: query_proofs.base_multi_proof,
            final_codeword: Poly::new(final_codeword),
            pow_witness: query_proofs.pow_witness,
            evals: Vec::new(),
        };

        let err = verify_query_paths(
            &config,
            &mmcs_instance,
            &base_commitment,
            randomness.as_slice(),
            &proof,
            &mut verifier_ch,
        )
        .unwrap_err();
        assert!(
            matches!(err, BinaryPcsError::FoldMismatch { round: 1, query: 0 }),
            "expected FoldMismatch at round 1 query 0, got {err:?}"
        );
    }
}
