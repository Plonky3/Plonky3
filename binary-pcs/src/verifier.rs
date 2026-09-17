//! Full-coset queries tying consecutive committed fold batches together.
//!
//! Base query indices are aligned to the first batch's coset size. A batch starting at
//! variable `start` with `arity` challenges opens the coset containing `index >> start`,
//! folds all its symbols, then checks the coordinate `index >> (start + arity)` in the next
//! committed word (or the final word sent in full). Only base cosets are sampled distinctly;
//! repeated projected cosets in later rounds remain the same base-query paths.
//!
//! The base word's symbols come from the committed alphabet.
//! Every folded word's come from the challenge field, so the first batch is what widens.

use alloc::vec;
use alloc::vec::Vec;

use p3_binary_field::TowerLevel;
use p3_challenger::fs::TranscriptField;
use p3_challenger::{CanSampleUniformBits, FieldChallenger, GrindingChallenger};
use p3_commit::Mmcs;
use p3_field::{ExtensionField, Field};
use p3_matrix::Dimensions;

use crate::error::BinaryPcsError;
use crate::fold::{fold_coset, fold_pair};
use crate::params::BinaryPcsConfig;
use crate::proof::BinaryPcsProof;
use crate::transcript::BinaryPcsVerifierTranscript;

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
fn check_round_shape<A, F, E>(
    round: usize,
    opened_values: &[Vec<A>],
    expected_opens: usize,
) -> Result<(), BinaryPcsError<F, E>> {
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

/// Wraps opened rows in the shape the batched-verification entry point expects.
/// That shape is indexed by query and then by matrix.
///
/// Every round here commits exactly one matrix, so each row gets a one-element slice.
fn wrap_rows<A>(opened_values: &[Vec<A>]) -> Vec<Vec<&[A]>> {
    opened_values
        .iter()
        .map(|row| vec![row.as_slice()])
        .collect()
}

/// Checks the proof's declared round count and final-codeword length against the schedule.
/// Both run before any transcript operation.
///
/// The opening verifier and the query path check each need this pair first.
/// Neither one's own work is reached until both have passed.
pub(crate) fn check_round_and_final_lengths<F, EF, MT, MX>(
    config: &BinaryPcsConfig,
    proof: &BinaryPcsProof<F, EF, MT, MX>,
) -> Result<(), BinaryPcsError<F, MT::Error>>
where
    F: Field,
    EF: Field,
    MT: Mmcs<F>,
    MX: Mmcs<EF>,
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
// Why: a zero-bit witness check returns true without absorbing anything.
// The proof's witness is then compared against nothing, and any value rides along.
//
//     pow_bits = 0 -> prover emits zero, verifier reads nothing -> pin the field here
//     pow_bits > 0 -> prover grinds,     verifier resamples     -> the grind pins it
//
// Assumption: the honest prover's grind at zero bits is pinned to the zero witness.
// A grinding path that leaves the zero-bit witness unconstrained needs this check revisited.
// The pin is asserted by `zero_difficulty_grinding_is_pinned_to_the_zero_witness` below.
pub(crate) fn check_canonical_pow_witness<F, EF, MT, MX>(
    config: &BinaryPcsConfig,
    proof: &BinaryPcsProof<F, EF, MT, MX>,
) -> Result<(), BinaryPcsError<F, MT::Error>>
where
    F: TowerLevel,
    EF: Field,
    MT: Mmcs<F>,
    MX: Mmcs<EF>,
{
    if config.pow_bits() == 0 && proof.pow_witness != F::ZERO {
        return Err(BinaryPcsError::NonCanonicalPowWitness {
            actual: proof.pow_witness,
        });
    }

    Ok(())
}

/// Folds one query's coset of a single batch, out of the rows the proof opened for it.
///
/// A single-challenge batch is one pair, so it needs neither a gather nor a scratch pass.
///
/// `gather` and `scratch` are working space the caller reuses across queries.
fn fold_query_coset<A, EF>(
    rows: &[Vec<A>],
    query: usize,
    next_position: usize,
    betas: &[EF],
    gather: &mut Vec<A>,
    scratch: &mut Vec<EF>,
) -> EF
where
    A: TowerLevel,
    EF: ExtensionField<A> + TowerLevel,
{
    if let [beta] = betas {
        return fold_pair(
            next_position,
            *beta,
            rows[2 * query][0],
            rows[2 * query + 1][0],
        );
    }

    // The coset's symbols are consecutive rows, one symbol each.
    let size = 1usize << betas.len();
    gather.clear();
    gather.extend(
        rows[query * size..(query + 1) * size]
            .iter()
            .map(|row| row[0]),
    );
    fold_coset(next_position, gather, betas, scratch)
}

/// Verifies the query phase of an opening proof: the single grind, the sampled query
/// indices, every round's Merkle multiproof, and the fold-consistency chain tying each round
/// to the next.
///
/// `betas` holds one fold challenge per round, in the order the fold phase sampled them.
/// The caller derives it by replaying the sumcheck transcript.
///
/// This function touches neither the sumcheck rounds nor the commitments' own order.
///
/// All proof-shape checks run before the transcript is touched.
///
/// A malformed proof is therefore rejected without ever grinding or sampling against it.
///
/// # Panics
///
/// Panics unless `betas` holds one challenge per fold round.
/// It is the caller's own transcript-replay output, never proof-supplied data.
///
/// A length mismatch is therefore a caller bug, not a malformed proof.
pub(crate) fn verify_query_paths<F, EF, MT, MX, Ch>(
    config: &BinaryPcsConfig,
    mmcs: &MT,
    round_mmcs: &MX,
    base_commitment: &MT::Commitment,
    betas: &[EF],
    proof: &BinaryPcsProof<F, EF, MT, MX>,
    transcript: &mut BinaryPcsVerifierTranscript<'_, F, EF, Ch>,
) -> Result<(), BinaryPcsError<F, MT::Error>>
where
    F: TranscriptField + TowerLevel,
    EF: ExtensionField<F> + TowerLevel,
    MT: Mmcs<F>,
    MX: Mmcs<EF, Error = MT::Error>,
    Ch: FieldChallenger<F> + GrindingChallenger<Witness = F> + CanSampleUniformBits<F>,
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

    // The base batch reads the committed alphabet, every later one the challenge field.
    check_round_shape(0, &proof.base_opened_values, target_queries << batches[0].1)?;
    for (batch, &(_, arity)) in batches.iter().enumerate().skip(1) {
        check_round_shape(
            batch,
            &proof.rounds[batch - 1].opened_values,
            target_queries << arity,
        )?;
    }

    // Match the prover: this uncommitted word precedes both grinding and sampling.
    // That holds even with no evaluation claim to constrain its value.
    transcript.final_codeword(proof.final_codeword.as_slice())?;
    transcript.query_pow(proof.pow_witness)?;

    // The draw indexes pairs, so each position is lifted back to a coset start.
    let shift = config.log_folding_factor() - 1;
    let indices: Vec<usize> = transcript
        .query_pairs()
        .into_iter()
        .map(|pair| pair << shift)
        .collect();
    debug_assert_eq!(indices.len(), target_queries);

    // Authenticate the base batch's rows against the base commitment.
    let base_dims = [Dimensions {
        width: 1,
        height: domain_size,
    }];
    let base_indices = flat_coset_indices(&indices, 0, batches[0].1);
    mmcs.verify_multi_batch(
        base_commitment,
        &base_dims,
        &base_indices,
        &wrap_rows(&proof.base_opened_values),
        &proof.base_multi_proof,
    )
    .map_err(|source| BinaryPcsError::MerkleFailed { round: 0, source })?;

    // Then every committed folded batch against its own root.
    for (batch, &(start, arity)) in batches.iter().enumerate().skip(1) {
        let round = &proof.rounds[batch - 1];
        let dims = [Dimensions {
            width: 1,
            height: domain_size >> start,
        }];
        let coset_indices = flat_coset_indices(&indices, start, arity);
        round_mmcs
            .verify_multi_batch(
                &round.commitment,
                &dims,
                &coset_indices,
                &wrap_rows(&round.opened_values),
                &round.multi_proof,
            )
            .map_err(|source| BinaryPcsError::MerkleFailed {
                round: batch,
                source,
            })?;
    }

    // The fold chain: each batch's coset must reproduce the symbol the next word carries.
    let mut base_gather: Vec<F> = Vec::new();
    let mut round_gather: Vec<EF> = Vec::new();
    let mut scratch: Vec<EF> = Vec::new();
    for (q, &index) in indices.iter().enumerate() {
        for (batch, &(start, arity)) in batches.iter().enumerate() {
            let next_position = index >> (start + arity);
            let betas = &betas[start..start + arity];

            // Only the base batch reads the narrow alphabet.
            let folded = if batch == 0 {
                fold_query_coset(
                    &proof.base_opened_values,
                    q,
                    next_position,
                    betas,
                    &mut base_gather,
                    &mut scratch,
                )
            } else {
                fold_query_coset(
                    &proof.rounds[batch - 1].opened_values,
                    q,
                    next_position,
                    betas,
                    &mut round_gather,
                    &mut scratch,
                )
            };

            // The next word is a committed round, or the final word sent in the clear.
            let expected = match batches.get(batch + 1) {
                Some(&(_, next_arity)) => {
                    let next_size = 1 << next_arity;
                    proof.rounds[batch].opened_values
                        [q * next_size + (next_position & (next_size - 1))][0]
                }
                None => proof.final_codeword.as_slice()[next_position],
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
    use p3_challenger::{CanObserve, GrindingChallenger};
    use p3_commit::Mmcs;
    use p3_field::{Field, PrimeCharacteristicRing};
    use p3_matrix::dense::RowMajorMatrix;
    use p3_multilinear_util::poly::Poly;
    use p3_sumcheck::SumcheckData;
    use p3_sumcheck::layout::{Layout, SuffixProver, Table, TableShape, Verifier};
    use p3_sumcheck::strategy::Basis;
    use p3_sumcheck::transcript::{SumcheckShape, VerifierTranscript};
    use p3_symmetric::MerkleCap;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    use super::{BinaryPcsError, verify_query_paths};
    use crate::params::{BinaryPcsConfig, BinaryPcsParams};
    use crate::proof::BinaryPcsProof;
    use crate::prover::{RoundCommitment, commit, fold_rounds_with, open_queries};
    use crate::test_util::{MyChallenger, MyMmcs, challenger, mmcs};
    use crate::transcript::{
        BinaryPcsProverTranscript, BinaryPcsShape, BinaryPcsVerifierTranscript,
    };

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

    /// Draw one run's query positions through the prover driver.
    ///
    /// The driver is the only sampler, so a property test asks it, not a copy of it.
    fn drawn_positions(config: &BinaryPcsConfig) -> Vec<usize> {
        drawn_positions_for(config, F::default())
    }

    /// Draw one run's query positions with `fill` as every final-codeword symbol.
    ///
    /// The codeword is the last thing bound before the positions are drawn.
    ///
    /// Varying it is therefore the cheapest way to move the transcript under them.
    fn drawn_positions_for(config: &BinaryPcsConfig, fill: F) -> Vec<usize> {
        let shape = BinaryPcsShape::new(config);
        let mut ch = challenger();
        let mut transcript = BinaryPcsProverTranscript::new(&mut ch, shape);

        // The positions are the last described step, so everything before it is played first.
        for _ in 0..shape.num_oracles {
            transcript.oracle_commitment(MerkleCap::<F, [u8; 32]>::new(vec![[0u8; 32]]));
        }
        transcript.final_codeword(&vec![fill; shape.final_codeword_len]);
        let _witness = transcript.query_pow();

        let shift = config.log_folding_factor() - 1;
        let positions: Vec<usize> = transcript
            .query_pairs()
            .into_iter()
            .map(|pair| pair << shift)
            .collect();
        transcript.finish();
        positions
    }

    #[test]
    fn query_indices_are_distinct_sorted_even_and_in_range() {
        // Invariant: a query names a fold pair, so positions are distinct and pair-aligned.
        //
        // Fixture state: a 2^10 domain folded by one, so pairs index 2^9 slots.
        let config = BinaryPcsConfig::try_new::<F, F>(10, params())
            .unwrap()
            .try_with_folding(1)
            .unwrap();
        let indices = drawn_positions(&config);
        assert_eq!(indices.len(), config.num_queries());
        assert!(
            indices.windows(2).all(|w| w[0] < w[1]),
            "sorted and distinct"
        );
        assert!(
            indices.iter().all(|&i| i < config.domain_size()),
            "in range"
        );
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
        let config = BinaryPcsConfig::try_new::<F, F>(3, params())
            .unwrap()
            .try_with_folding(1)
            .unwrap();
        let indices = drawn_positions(&config);
        let pairs = config.domain_size() / 2;
        assert_eq!(indices, (0..pairs).map(|p| p << 1).collect::<Vec<_>>());
    }

    #[test]
    fn batched_queries_sample_distinct_full_cosets_and_cap_at_the_coset_count() {
        for (num_variables, arity) in [(8, 3), (3, 3)] {
            let config = BinaryPcsConfig::try_new::<F, F>(num_variables, params())
                .unwrap()
                .try_with_folding(arity)
                .unwrap();
            let indices = drawn_positions(&config);
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
        // Invariant: the positions come out of the sponge, not out of the configuration.
        //
        // A prover that could fix them would choose which symbols it is asked for.
        //
        // Fixture state: one configuration, drawn twice.
        //
        // Mutation: change the last thing bound before the draw.
        //
        //     final codeword all-zero  ->  one position set
        //     final codeword all-one   ->  another
        //
        // The difficulty is zero here on purpose. A parallel grind returns whichever
        // valid witness a worker reaches first, so a ground run draws from a sponge
        // that is not a function of the transcript alone.
        let unground = BinaryPcsParams {
            pow_bits: 0,
            ..params()
        };
        let config = BinaryPcsConfig::try_new::<F, F>(10, unground)
            .unwrap()
            .try_with_folding(1)
            .unwrap();

        // The same transcript draws the same positions.
        assert_eq!(drawn_positions(&config), drawn_positions(&config));

        // A different transcript draws different ones.
        //
        // The draw covers only part of the domain here, so the two sets can differ.
        let shape = BinaryPcsShape::new(&config);
        assert!(
            shape.num_pairs < 1 << shape.pair_bits,
            "a saturated draw opens every position whatever the transcript says",
        );
        assert_ne!(
            drawn_positions_for(&config, F::ZERO),
            drawn_positions_for(&config, F::ONE),
        );
    }

    #[test]
    fn round_count_mismatch_is_typed_not_a_panic() {
        // A proof claiming fewer rounds than the config must be rejected before any
        // indexing, so a malformed proof is a rejection rather than an abort.
        let err: BinaryPcsError<F, ()> = BinaryPcsError::RoundCountMismatch {
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

        let config = BinaryPcsConfig::try_new::<F, F>(NUM_VARIABLES, params()).unwrap();
        let encoder = AdditiveRsEncoder::<F, NaiveAdditiveNtt<F>>::default();
        let mmcs_instance = mmcs();

        // Prover route: bind the root, then run the whole description.
        let mut prover_ch = challenger();
        let (base_commitment, prover_data) =
            commit::<F, F, _, _>(&config, &encoder, &mmcs_instance, witness);
        prover_ch.observe(base_commitment.clone());
        let mut prover_t =
            BinaryPcsProverTranscript::new(&mut prover_ch, BinaryPcsShape::new(&config));
        let (base_merkle_data, sumcheck_data, rounds, randomness, final_codeword) =
            fold_rounds_with::<false, F, F, _, _, _>(
                prover_data,
                &config,
                &mmcs_instance,
                &mmcs_instance,
                &mut prover_t,
            );
        let query_proofs = open_queries(
            &config,
            &mmcs_instance,
            &mmcs_instance,
            &mut prover_t,
            &base_merkle_data,
            &rounds,
            &final_codeword,
        );
        prover_t.finish();

        let proof = BinaryPcsProof {
            sumcheck: sumcheck_data,
            rounds: query_proofs.rounds,
            base_opened_values: query_proofs.base_opened_values,
            base_multi_proof: query_proofs.base_multi_proof,
            final_codeword: Poly::new(final_codeword),
            pow_witness: query_proofs.pow_witness,
            evals: Vec::new(),
        };

        // Verifier route: an empty challenger, replaying only what the proof carries.
        let oracles: Vec<_> = proof.rounds.iter().map(|r| r.commitment.clone()).collect();
        let mut verifier_ch = challenger();
        verifier_ch.observe(base_commitment.clone());
        let mut verifier_t =
            replay_fold_phase(&config, &mut verifier_ch, &proof.sumcheck, &oracles);

        let result = verify_query_paths(
            &config,
            &mmcs_instance,
            &mmcs_instance,
            &base_commitment,
            randomness.as_slice(),
            &proof,
            &mut verifier_t,
        );
        assert!(result.is_ok(), "{result:?}");
        verifier_t.finish();
    }

    /// Replay the fold phase on the verifier's side and hand back the open driver.
    ///
    /// The prover plays the batching challenge, then one sumcheck round per fold round.
    ///
    /// An oracle commitment follows every batch boundary but the last.
    ///
    /// A replay that plays one step too many or too few desyncs the two sponges.
    fn replay_fold_phase<'a>(
        config: &BinaryPcsConfig,
        challenger: &'a mut MyChallenger,
        sumcheck_data: &SumcheckData<F, F>,
        oracles: &[<MyMmcs as Mmcs<F>>::Commitment],
    ) -> BinaryPcsVerifierTranscript<'a, F, F, MyChallenger> {
        let mut transcript =
            BinaryPcsVerifierTranscript::new(challenger, BinaryPcsShape::new(config));

        // The layout draws its batching challenge before the first fold round.
        //
        // This replay records no claim, so both counts are zero, matching the prover.
        let layout_verifier = Verifier::<F, F>::new(
            &[TableShape::new(NUM_VARIABLES, 1)],
            SuffixProver::<F, F>::strategy(),
        );
        let _alpha: F = transcript.fold_batch(|ch| layout_verifier.batching_challenge(ch));

        let num_fold_rounds = config.num_fold_rounds();
        // `r` indexes collections of two different lengths, so no single zip covers the loop.
        #[allow(clippy::needless_range_loop)]
        for r in 0..num_fold_rounds {
            let [c0, c_inf] = sumcheck_data.polynomial_evaluations()[r];
            // One fold round is one single-round sumcheck, seeded on its own.
            let round_shape = SumcheckShape::new(1, 0, Basis::Evaluation);
            let _beta = transcript.fold_batch(|ch| {
                let mut t = VerifierTranscript::<_, F, F>::new(ch, round_shape);
                let beta = t.round(c0, c_inf, None).unwrap();
                t.finish();
                beta
            });
            if r + 1 < num_fold_rounds {
                transcript.oracle_commitment(oracles[r].clone());
            }
        }

        transcript
    }

    /// A fresh verifier challenger, seeded identically to the prover's but touched only by
    /// what the proof carries, must sample the same query positions the prover did.
    ///
    /// The round-trip test clones the prover's own challenger.
    ///
    /// It therefore carries every observation the prover made, right or wrong.
    ///
    /// It can never disagree with the prover.
    ///
    /// This one rebuilds the verifier's side from an empty challenger.
    ///
    /// It runs through the same driver the real verifier uses.
    ///
    /// A prover that plays one step too many desyncs the two, and the positions diverge.
    #[test]
    fn a_fresh_verifier_challenger_samples_the_same_query_indices() {
        let mut rng = SmallRng::seed_from_u64(0x5EED);
        let poly = Poly::<F>::rand(&mut rng, NUM_VARIABLES);
        let table = Table::new(RowMajorMatrix::new(poly.into_evals(), 1 << NUM_VARIABLES));
        let witness = SuffixProver::<F, F>::new_witness(vec![table], 0);

        let config = BinaryPcsConfig::try_new::<F, F>(NUM_VARIABLES, params()).unwrap();
        let encoder = AdditiveRsEncoder::<F, NaiveAdditiveNtt<F>>::default();
        let mmcs_instance = mmcs();
        let shape = BinaryPcsShape::new(&config);
        let shift = config.log_folding_factor() - 1;

        // Prover route: commit, bind the root, then run the whole description.
        let mut prover_ch = challenger();
        let (base_commitment, prover_data) =
            commit::<F, F, _, _>(&config, &encoder, &mmcs_instance, witness);
        prover_ch.observe(base_commitment.clone());
        let mut prover_t = BinaryPcsProverTranscript::new(&mut prover_ch, shape);
        let (_base_merkle_data, sumcheck_data, rounds, _randomness, final_codeword) =
            fold_rounds_with::<false, F, F, _, _, _>(
                prover_data,
                &config,
                &mmcs_instance,
                &mmcs_instance,
                &mut prover_t,
            );
        prover_t.final_codeword(&final_codeword);
        let pow_witness = prover_t.query_pow();
        let prover_positions: Vec<usize> = prover_t
            .query_pairs()
            .into_iter()
            .map(|pair| pair << shift)
            .collect();
        prover_t.finish();

        // Verifier route: an empty challenger, touched only by what the proof carries.
        let oracles: Vec<_> = rounds.iter().map(|r| r.commitment.clone()).collect();
        let mut verifier_ch = challenger();
        verifier_ch.observe(base_commitment);
        let mut verifier_t = replay_fold_phase(&config, &mut verifier_ch, &sumcheck_data, &oracles);

        verifier_t.final_codeword(&final_codeword).unwrap();
        verifier_t.query_pow(pow_witness).unwrap();
        let verifier_positions: Vec<usize> = verifier_t
            .query_pairs()
            .into_iter()
            .map(|pair| pair << shift)
            .collect();
        verifier_t.finish();

        assert_eq!(verifier_positions, prover_positions);
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

        let config = BinaryPcsConfig::try_new::<F, F>(NUM_VARIABLES, params()).unwrap();
        let encoder = AdditiveRsEncoder::<F, NaiveAdditiveNtt<F>>::default();
        let mmcs_instance = mmcs();

        let mut prover_ch = challenger();
        let (base_commitment, prover_data) =
            commit::<F, F, _, _>(&config, &encoder, &mmcs_instance, witness);
        // Mirror what the scheme's commit phase binds, so this replay walks the
        // same sponge stream production does.
        prover_ch.observe(base_commitment.clone());
        let mut prover_t =
            BinaryPcsProverTranscript::new(&mut prover_ch, BinaryPcsShape::new(&config));
        let (base_merkle_data, sumcheck_data, mut rounds, randomness, final_codeword) =
            fold_rounds_with::<false, F, F, _, _, _>(
                prover_data,
                &config,
                &mmcs_instance,
                &mmcs_instance,
                &mut prover_t,
            );

        // The prover bound this oracle before the tamper below replaces it.
        //
        // The verifier must replay that same value.
        //
        // Otherwise its query positions move.
        //
        // The Merkle check then fires before the fold check this test is about.
        let bound_oracles: Vec<_> = rounds.iter().map(|r| r.commitment.clone()).collect();

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

        let query_proofs = open_queries(
            &config,
            &mmcs_instance,
            &mmcs_instance,
            &mut prover_t,
            &base_merkle_data,
            &rounds,
            &final_codeword,
        );
        prover_t.finish();

        let proof = BinaryPcsProof {
            sumcheck: sumcheck_data,
            rounds: query_proofs.rounds,
            base_opened_values: query_proofs.base_opened_values,
            base_multi_proof: query_proofs.base_multi_proof,
            final_codeword: Poly::new(final_codeword),
            pow_witness: query_proofs.pow_witness,
            evals: Vec::new(),
        };

        let mut verifier_ch = challenger();
        verifier_ch.observe(base_commitment.clone());
        let mut verifier_t =
            replay_fold_phase(&config, &mut verifier_ch, &proof.sumcheck, &bound_oracles);

        let err = verify_query_paths(
            &config,
            &mmcs_instance,
            &mmcs_instance,
            &base_commitment,
            randomness.as_slice(),
            &proof,
            &mut verifier_t,
        )
        .unwrap_err();
        verifier_t.abort();
        assert!(
            matches!(err, BinaryPcsError::FoldMismatch { round: 1, query: 0 }),
            "expected FoldMismatch at round 1 query 0, got {err:?}"
        );
    }
}
