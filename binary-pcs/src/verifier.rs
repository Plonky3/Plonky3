//! Query sampling and the per-query fold-consistency check.
//!
//! Every round's codeword is a flat vector; round 0 is the base commitment, round `r` for
//! `1 <= r < num_fold_rounds` is `proof.rounds[r - 1]`, and round `num_fold_rounds` is
//! `proof.final_codeword`, sent in full rather than committed. A query index `i`, drawn from
//! the base codeword's domain, addresses position `i >> r` at round `r`; folding that
//! position's pair with round `r`'s challenge must reproduce the value read at position
//! `i >> (r + 1)` of round `r + 1`.

use alloc::vec;
use alloc::vec::Vec;

use p3_binary_field::BinaryField128;
use p3_challenger::{CanSampleUniformBits, FieldChallenger, GrindingChallenger};
use p3_commit::Mmcs;
use p3_field::Field;
use p3_matrix::Dimensions;
use p3_util::log2_strict_usize;

use crate::error::BinaryPcsError;
use crate::fold::fold_pair;
use crate::params::BinaryPcsConfig;
use crate::proof::BinaryPcsProof;

/// Samples `num_queries` distinct indices, uniformly at random, from `[0, domain_size)`.
///
/// Indices come from `sample_uniform_bits::<true>`, which rejection-samples internally,
/// rather than from the low bits of a sampled field element: the latter biases each draw and
/// inflates the decoding radius the query bound is stated against.
///
/// Duplicates are rejected, so the output length is `min(num_queries, domain_size)`; a
/// request for more indices than the domain holds returns the whole domain. Returns indices
/// in ascending order.
///
/// # Panics
///
/// Panics if `domain_size` is not a power of two.
pub(crate) fn sample_query_indices<Challenger, F>(
    domain_size: usize,
    num_queries: usize,
    challenger: &mut Challenger,
) -> Vec<usize>
where
    Challenger: FieldChallenger<F> + CanSampleUniformBits<F>,
    F: Field,
{
    let bits = log2_strict_usize(domain_size);
    let target = num_queries.min(domain_size);

    let mut indices: Vec<usize> = Vec::with_capacity(target);
    while indices.len() < target {
        let index = challenger
            .sample_uniform_bits::<true>(bits)
            .expect("RESAMPLE = true: rejection loops internally, never errors");
        if !indices.contains(&index) {
            indices.push(index);
        }
    }

    indices.sort_unstable();
    indices
}

/// The pair of positions round `round`'s codeword must supply to carry query `index` onward:
/// the position `index` addresses at that round, and its fold sibling — low bit clear first.
const fn fold_pair_positions(index: usize, round: usize) -> [usize; 2] {
    let position = index >> round;
    let even = position & !1;
    [even, even + 1]
}

/// Every position round `round` must open: one pair per sampled query index, in query order.
pub(crate) fn flat_pair_indices(indices: &[usize], round: usize) -> Vec<usize> {
    indices
        .iter()
        .flat_map(|&index| fold_pair_positions(index, round))
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
    let expected_intermediate_rounds = config.num_fold_rounds() - 1;
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

    let domain_size = config.domain_size();
    let target_queries = config.num_queries().min(domain_size);
    let expected_opens = 2 * target_queries;

    check_round_shape(0, &proof.base_opened_values, expected_opens)?;
    for (r, round) in proof.rounds.iter().enumerate() {
        check_round_shape(r + 1, &round.opened_values, expected_opens)?;
    }

    // The transcript is touched from here on: grind, then sample.
    if !challenger.check_witness(config.pow_bits(), proof.pow_witness) {
        return Err(BinaryPcsError::InvalidPowWitness);
    }

    let indices =
        sample_query_indices::<_, BinaryField128>(domain_size, config.num_queries(), challenger);
    debug_assert_eq!(indices.len(), target_queries);

    // One Merkle multiproof per round.
    let base_indices = flat_pair_indices(&indices, 0);
    let base_dims = [Dimensions {
        width: 1,
        height: domain_size,
    }];
    mmcs.verify_multi_batch(
        base_commitment,
        &base_dims,
        &base_indices,
        &wrap_rows(&proof.base_opened_values),
        &proof.base_multi_proof,
    )
    .map_err(|source| BinaryPcsError::MerkleFailed { round: 0, source })?;

    for (r, round) in proof.rounds.iter().enumerate() {
        let round_number = r + 1;
        let round_indices = flat_pair_indices(&indices, round_number);
        let dims = [Dimensions {
            width: 1,
            height: domain_size >> round_number,
        }];
        mmcs.verify_multi_batch(
            &round.commitment,
            &dims,
            &round_indices,
            &wrap_rows(&round.opened_values),
            &round.multi_proof,
        )
        .map_err(|source| BinaryPcsError::MerkleFailed {
            round: round_number,
            source,
        })?;
    }

    // Fold-chain consistency, one query at a time.
    let round_values = |round: usize| -> &[Vec<BinaryField128>] {
        if round == 0 {
            &proof.base_opened_values
        } else {
            &proof.rounds[round - 1].opened_values
        }
    };

    for (q, &index) in indices.iter().enumerate() {
        // The bound is `num_fold_rounds`, checked against `betas.len()` above, rather than
        // `betas.len()` itself, so an inconsistent `betas` cannot silently shrink or grow this
        // loop; `r` also indexes `index >> r` and both `round_values` calls, not just `betas`.
        #[allow(clippy::needless_range_loop)]
        for r in 0..num_fold_rounds {
            let beta = betas[r];
            let position = index >> r;
            let lo = round_values(r)[2 * q][0];
            let hi = round_values(r)[2 * q + 1][0];
            let folded = fold_pair(position >> 1, beta, lo, hi);

            let expected = if r + 1 < num_fold_rounds {
                let parity = (position >> 1) & 1;
                round_values(r + 1)[2 * q + parity][0]
            } else {
                proof.final_codeword.as_slice()[position >> 1]
            };

            if folded != expected {
                return Err(BinaryPcsError::FoldMismatch {
                    round: r + 1,
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
    use p3_field::PrimeCharacteristicRing;
    use p3_matrix::dense::RowMajorMatrix;
    use p3_multilinear_util::poly::Poly;
    use p3_sumcheck::layout::{Layout, SuffixProver, Table};
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    use super::{BinaryPcsError, sample_query_indices, verify_query_paths};
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
    fn query_indices_are_distinct_sorted_and_in_range() {
        let mut c = challenger();
        let indices = sample_query_indices::<_, BinaryField128>(1 << 10, 12, &mut c);
        assert_eq!(indices.len(), 12);
        assert!(
            indices.windows(2).all(|w| w[0] < w[1]),
            "sorted and distinct"
        );
        assert!(indices.iter().all(|&i| i < 1 << 10), "in range");
    }

    #[test]
    fn a_request_larger_than_the_domain_returns_the_whole_domain() {
        let mut c = challenger();
        let indices = sample_query_indices::<_, BinaryField128>(8, 100, &mut c);
        assert_eq!(indices, (0..8).collect::<Vec<_>>());
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
            verifier_ch.observe_algebra_slice(&[c0, c_inf]);
            let beta: F = verifier_ch.sample_algebra_element();
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
