//! Prover side: commit the stacked polynomial as one codeword column, then fold that codeword
//! in lockstep with the residual sumcheck.
//!
//! [`PcsLayout`] commits with no preprocessing depth, so `Layout::commit` produces a width-1
//! codeword — one Reed-Solomon-encoded column, the whole committed polynomial — and
//! `Layout::into_sumcheck` consumes zero preprocessing rounds, leaving every one of the
//! `num_variables` residual sumcheck rounds a folding round. Each round's challenge is used
//! twice: it binds one multilinear variable, through [`PcsLayout`]'s evaluation-basis suffix
//! binding, and it folds the codeword, through [`fold_codeword_batch`], a Reed-Solomon codeword fold
//! in the same basis (see `fold.rs`). The two stay in correspondence throughout: see
//! `the_codeword_and_the_sumcheck_stay_in_lockstep` below, which drives the real `Layout`
//! machinery and checks it, and `prefix_layout_does_not_stay_in_lockstep`, which checks that
//! prefix-first binding does not share the property — the reason [`PcsLayout`] is fixed rather
//! than chosen by the caller.

use alloc::vec::Vec;

use p3_binary_field::BinaryField128;
use p3_challenger::{CanObserve, CanSampleUniformBits, FieldChallenger, GrindingChallenger};
use p3_commit::{Encoder, Mmcs};
use p3_matrix::dense::{DenseMatrix, RowMajorMatrix};
use p3_multilinear_util::point::Point;
use p3_sumcheck::SumcheckData;
use p3_sumcheck::layout::{Layout, Table, Witness};

use crate::PcsLayout;
use crate::fold::fold_codeword_batch;
use crate::params::BinaryPcsConfig;
use crate::proof::RoundProof;
use crate::verifier::{flat_coset_indices, sample_query_cosets};

/// Preprocessing depth the commit phase lays out inside a committed row.
///
/// Zero, so the codeword is a single column and every variable is folded rather than bound
/// across a row. Binding a prefix inside the row is the deferred head collapse, which needs
/// its own eq-weighted column combination before it composes with the folds below.
const FOLDING: usize = 0;

/// Data produced by committing the base codeword: the layout used to build the residual
/// sumcheck, and the base commitment's Merkle prover data.
///
/// Retains the source tables for AIR evaluation between `commit` and `open`.
/// Cloning reuses the encoding and Merkle tree for another opening of the commitment.
#[derive(Clone)]
pub struct BinaryPcsProverData<MT: Mmcs<BinaryField128>> {
    /// The layout that ran the commit phase, carried forward to build the residual sumcheck
    /// and, later, to evaluate opening claims against the committed polynomial.
    pub(crate) layout: PcsLayout,
    /// The base codeword's Merkle prover data, needed to open base-round queries once the
    /// query phase samples its indices.
    pub(crate) merkle_data: MT::ProverData<DenseMatrix<BinaryField128>>,
}

impl<MT: Mmcs<BinaryField128>> BinaryPcsProverData<MT> {
    /// Returns source table `id` retained by the committed layout.
    pub fn table(&self, id: usize) -> &Table<BinaryField128> {
        self.layout.table(id)
    }
}

/// One folding round's prover-side output.
pub(crate) struct RoundCommitment<MT: Mmcs<BinaryField128>> {
    /// Merkle root of this round's folded codeword.
    pub commitment: MT::Commitment,
    /// Merkle prover data for this round's folded codeword, needed to open its queries.
    pub merkle_data: MT::ProverData<DenseMatrix<BinaryField128>>,
}

/// Commits `witness`'s stacked polynomial and returns the base commitment alongside the data
/// needed to run the residual sumcheck and later open the base codeword's queries.
///
/// `witness` must have `config.num_variables()` variables and be built at [`FOLDING`].
#[tracing::instrument(name = "binary pcs commit", skip_all)]
pub(crate) fn commit<E, MT, Ch>(
    config: &BinaryPcsConfig,
    encoder: &E,
    mmcs: &MT,
    challenger: &mut Ch,
    witness: Witness<BinaryField128>,
) -> (MT::Commitment, BinaryPcsProverData<MT>)
where
    E: Encoder<BinaryField128>,
    MT: Mmcs<BinaryField128>,
    Ch: FieldChallenger<BinaryField128>
        + GrindingChallenger<Witness = BinaryField128>
        + CanObserve<MT::Commitment>,
{
    assert_eq!(
        witness.num_variables(),
        config.num_variables(),
        "witness arity must match the config it is committed against"
    );

    let (layout, commitment, merkle_data) = PcsLayout::commit(
        encoder,
        mmcs,
        challenger,
        witness,
        FOLDING,
        config.log_inv_rate(),
    );

    (
        commitment,
        BinaryPcsProverData {
            layout,
            merkle_data,
        },
    )
}

/// Runs each residual sumcheck round before consuming its challenge in a codeword fold.
/// Consecutive folds are fused into configured batches, with one commitment per batch except
/// the last, whose entire codeword travels in the clear.
///
/// The single grinding budget is spent once before the query phase, not here, so every round
/// runs with `pow_bits = 0`.
///
/// Returns the base commitment's Merkle prover data (handed back so the caller can still open
/// base-round queries against it), the sumcheck transcript, one [`RoundCommitment`] per fold
/// batch except the last, the folding randomness in round order — `randomness.as_slice()[r]` is
/// round `r`'s challenge, matching what [`Layout::into_sumcheck`] returns — and the final
/// folded codeword.
///
/// `BIND_EACH_ROUND` picks when each round's challenge is applied to the sumcheck tables:
///
/// ```text
///     false: left outstanding, so the next round's measuring pass absorbs it
///            one pass per round
///     true : applied on the spot, so the next round measures in a pass of its own
///            two passes per round
/// ```
///
/// The two produce the same round polynomials, so one can be pinned against the other.
///
/// The choice is a const parameter, so a build that never asks for the two-pass route
/// never compiles one.
///
/// This is the only seam between the two routes.
/// `BinaryPcs::finish_open_with` carries the same parameter one level up.
#[must_use]
#[allow(clippy::type_complexity)]
#[tracing::instrument(name = "binary pcs fold rounds", skip_all)]
pub(crate) fn fold_rounds_with<const BIND_EACH_ROUND: bool, MT, Ch>(
    prover_data: BinaryPcsProverData<MT>,
    config: &BinaryPcsConfig,
    mmcs: &MT,
    challenger: &mut Ch,
) -> (
    MT::ProverData<DenseMatrix<BinaryField128>>,
    SumcheckData<BinaryField128, BinaryField128>,
    Vec<RoundCommitment<MT>>,
    Point<BinaryField128>,
    Vec<BinaryField128>,
)
where
    MT: Mmcs<BinaryField128>,
    Ch: FieldChallenger<BinaryField128>
        + GrindingChallenger<Witness = BinaryField128>
        + CanObserve<MT::Commitment>,
{
    let BinaryPcsProverData {
        layout,
        merkle_data,
    } = prover_data;

    let mut sumcheck_data = SumcheckData::default();
    let (mut sumcheck, mut randomness) = layout.into_sumcheck(&mut sumcheck_data, 0, challenger);
    assert_eq!(
        randomness.num_variables(),
        0,
        "the commit phase runs at zero preprocessing depth, so the sumcheck consumes no head rounds"
    );

    assert_eq!(
        mmcs.get_matrices(&merkle_data)[0].width,
        1,
        "zero preprocessing depth commits a width-1 codeword"
    );

    // Sumcheck messages/challenges remain sequential. Only batch boundaries materialize
    // codewords and observe roots; the final batch is returned in the clear.
    let num_batches = config.num_fold_batches();
    let mut rounds: Vec<RoundCommitment<MT>> = Vec::with_capacity(num_batches - 1);
    let mut final_codeword = Vec::new();
    for (batch, (start, arity)) in config.fold_batches().enumerate() {
        let _batch_span = tracing::info_span!("fold batch", batch, start, arity).entered();
        let mut challenges = Vec::with_capacity(arity);
        for round in start..start + arity {
            let challenge = tracing::info_span!("sumcheck round", round).in_scope(|| {
                sumcheck.compute_sumcheck_polynomials(&mut sumcheck_data, challenger, 1, 0, None)
            });
            challenges.push(challenge.as_slice()[0]);
            randomness.extend(&challenge);

            // The reference route applies this round's binding now.
            //
            // The fused route instead lets the next round's measuring pass absorb it.
            if BIND_EACH_ROUND {
                sumcheck.settle();
            }
        }

        // Fold out of the previous batch's Merkle leaves, never out of a copy of them.
        // The commitment scheme already owns every codeword it committed.
        // The fold allocates its own output, so this borrow ends before the push below.
        let source = if batch == 0 {
            &merkle_data
        } else {
            &rounds[batch - 1].merkle_data
        };
        let folded = tracing::info_span!("fold codeword")
            .in_scope(|| fold_codeword_batch(&mmcs.get_matrices(source)[0].values, &challenges));
        if batch + 1 < num_batches {
            let (commitment, round_data) = tracing::info_span!("commit folded codeword")
                .in_scope(|| mmcs.commit_matrix(RowMajorMatrix::new(folded, 1)));
            challenger.observe(commitment.clone());
            rounds.push(RoundCommitment {
                commitment,
                merkle_data: round_data,
            });
        } else {
            final_codeword = folded;
        }
    }

    // The last round's challenge is discarded rather than applied.
    //
    // The codeword fold, not the sumcheck tables, carries the folded message forward.
    // The returned tuple holds no sumcheck state, so nothing downstream can read the tables.
    // A final binding pass would only produce a table nobody looks at.
    //
    // A debug build applies it anyway, purely to check the claim against the pair it binds.
    // That is the last held binding's only validation, in any profile.
    #[cfg(debug_assertions)]
    sumcheck.settle();

    (
        merkle_data,
        sumcheck_data,
        rounds,
        randomness,
        final_codeword,
    )
}

/// The query phase's prover-side output: every opening `verifier::verify_query_paths` needs,
/// plus the grinding witness.
pub(crate) struct QueryProofs<MT: Mmcs<BinaryField128>> {
    /// Every symbol of each queried base coset, one width-1 row per symbol.
    /// Cosets follow sampled query order; symbols inside a coset are ascending.
    pub base_opened_values: Vec<Vec<BinaryField128>>,
    /// Multiproof for the base commitment's queried rows.
    pub base_multi_proof: MT::MultiProof,
    /// One [`RoundProof`] per intermediate fold batch, excluding the final batch.
    pub rounds: Vec<RoundProof<MT>>,
    /// Witness for the single grind before the query phase.
    pub pow_witness: BinaryField128,
}

/// Runs the query phase: grinds the single proof-of-work witness, samples query indices from
/// the base codeword's domain, then opens the base commitment and every intermediate
/// fold-batch commitment at all coset indices each sampled query needs.
///
/// `rounds` is every [`RoundCommitment`] `fold_rounds_with` produced: one per fold batch except the
/// last, whose codeword is never committed — it travels in the clear as the proof's
/// `final_codeword` instead, so a Merkle path for it would only repeat what the verifier can
/// already read directly.
#[tracing::instrument(name = "binary pcs open queries", skip_all)]
pub(crate) fn open_queries<MT, Ch>(
    config: &BinaryPcsConfig,
    mmcs: &MT,
    challenger: &mut Ch,
    base_merkle_data: &MT::ProverData<DenseMatrix<BinaryField128>>,
    rounds: &[RoundCommitment<MT>],
) -> QueryProofs<MT>
where
    MT: Mmcs<BinaryField128>,
    Ch: FieldChallenger<BinaryField128>
        + GrindingChallenger<Witness = BinaryField128>
        + CanSampleUniformBits<BinaryField128>,
{
    assert_eq!(
        rounds.len(),
        config.num_fold_batches() - 1,
        "rounds is the caller's own fold_rounds_with output, never proof-supplied data"
    );

    let pow_witness = challenger.grind(config.pow_bits());

    let indices = sample_query_cosets(config, challenger);
    let base_indices = flat_coset_indices(&indices, 0, config.log_folding_factor());
    let (base_values, base_multi_proof) = mmcs.open_multi_batch(&base_indices, base_merkle_data);
    let base_opened_values = single_matrix_rows(base_values);

    let opened_rounds = rounds
        .iter()
        .zip(config.fold_batches().skip(1))
        .map(|(round, (start, arity))| {
            let round_indices = flat_coset_indices(&indices, start, arity);
            let (values, multi_proof) = mmcs.open_multi_batch(&round_indices, &round.merkle_data);
            RoundProof {
                commitment: round.commitment.clone(),
                opened_values: single_matrix_rows(values),
                multi_proof,
            }
        })
        .collect();

    QueryProofs {
        base_opened_values,
        base_multi_proof,
        rounds: opened_rounds,
        pow_witness,
    }
}

/// Strips the always-one-matrix middle index from an [`Mmcs::open_multi_batch`] result,
/// asserting the count instead of assuming it.
fn single_matrix_rows(values: Vec<Vec<Vec<BinaryField128>>>) -> Vec<Vec<BinaryField128>> {
    values
        .into_iter()
        .map(|mut per_matrix| {
            assert_eq!(per_matrix.len(), 1, "each round commits exactly one matrix");
            per_matrix.swap_remove(0)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use alloc::{format, vec};

    use p3_binary_dft::{AdditiveRsEncoder, NaiveAdditiveNtt};
    use p3_binary_field::BinaryField128;
    use p3_challenger::FieldChallenger;
    use p3_commit::Mmcs;
    use p3_matrix::dense::RowMajorMatrix;
    use p3_multilinear_util::poly::Poly;
    use p3_sumcheck::SumcheckData;
    use p3_sumcheck::layout::{Layout, PrefixProver, SuffixProver, Table};
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    use super::{commit, fold_rounds_with};
    use crate::fold::fold_codeword;
    use crate::params::{BinaryPcsConfig, BinaryPcsParams};
    use crate::test_util::{challenger, mmcs};

    type F = BinaryField128;

    const NUM_VARIABLES: usize = 8;
    const LOG_INV_RATE: usize = 2;

    /// After every round, the codeword and the sumcheck describe the same polynomial: the
    /// final codeword's symbols all equal the constant the sumcheck folds to.
    #[test]
    fn the_codeword_and_the_sumcheck_stay_in_lockstep() {
        let mut rng = SmallRng::seed_from_u64(21);
        let table = Table::rand(&mut rng, 1, NUM_VARIABLES);
        let witness = SuffixProver::<F, F>::new_witness(vec![table], 0);

        let mut ch = challenger();
        let (mut layout, _root, data) = SuffixProver::<F, F>::commit(
            &AdditiveRsEncoder::<F, NaiveAdditiveNtt<F>>::default(),
            &mmcs(),
            &mut ch,
            witness,
            0,
            LOG_INV_RATE,
        );
        let m = mmcs().get_matrices(&data)[0].clone();
        assert_eq!(m.width, 1);
        let mut codeword = m.values;

        let _virtual_eval = layout.add_virtual_eval(&mut ch);
        let mut sc = SumcheckData::<F, F>::default();
        let (mut prover, head) = layout.into_sumcheck(&mut sc, 0, &mut ch);
        assert_eq!(head.num_variables(), 0);

        for _ in 0..prover.num_variables() {
            let beta = prover.compute_sumcheck_polynomials(&mut sc, &mut ch, 1, 0, None);
            codeword = fold_codeword(&codeword, beta.as_slice()[0]);
        }

        let final_value = prover.evals().as_constant().expect("fully folded");
        assert_eq!(codeword.len(), 1 << LOG_INV_RATE);
        assert!(codeword.iter().all(|&v| v == final_value));
    }

    /// The lockstep property is specific to suffix binding's evaluation-basis order: driving
    /// the identical procedure through `PrefixProver` does not leave every symbol of the final
    /// codeword equal to the constant the sumcheck folds to.
    ///
    /// This is why `PcsLayout` is a fixed type rather than a caller-supplied parameter: the
    /// two orders are not interchangeable, and the difference is a property of the fold's
    /// pair-merging arithmetic, not a tuning choice.
    #[test]
    fn prefix_layout_does_not_stay_in_lockstep() {
        let mut rng = SmallRng::seed_from_u64(21);
        let table = Table::rand(&mut rng, 1, NUM_VARIABLES);
        let witness = PrefixProver::<F, F>::new_witness(vec![table], 0);

        let mut ch = challenger();
        let (mut layout, _root, data) = PrefixProver::<F, F>::commit(
            &AdditiveRsEncoder::<F, NaiveAdditiveNtt<F>>::default(),
            &mmcs(),
            &mut ch,
            witness,
            0,
            LOG_INV_RATE,
        );
        let m = mmcs().get_matrices(&data)[0].clone();
        assert_eq!(m.width, 1);
        let mut codeword = m.values;

        let _virtual_eval = layout.add_virtual_eval(&mut ch);
        let mut sc = SumcheckData::<F, F>::default();
        let (mut prover, head) = layout.into_sumcheck(&mut sc, 0, &mut ch);
        assert_eq!(head.num_variables(), 0);

        for _ in 0..prover.num_variables() {
            let beta = prover.compute_sumcheck_polynomials(&mut sc, &mut ch, 1, 0, None);
            codeword = fold_codeword(&codeword, beta.as_slice()[0]);
        }

        let final_value = prover.evals().as_constant().expect("fully folded");
        assert!(!codeword.iter().all(|&v| v == final_value));
    }

    /// Drives the crate's own [`commit`] and [`fold_rounds_with`] end to end and checks their
    /// output against an independent oracle: folding the original message variable by
    /// variable, via [`Poly::fix_suffix_var_mut`], over the randomness `fold_rounds_with` returns,
    /// in the order returned, must produce the constant every symbol of the final codeword
    /// equals.
    #[test]
    fn commit_and_fold_rounds_match_an_independent_message_fold() {
        let mut rng = SmallRng::seed_from_u64(0xC0FFEE);
        let poly = Poly::<F>::rand(&mut rng, NUM_VARIABLES);
        let mut message = poly.clone();
        let table = Table::new(RowMajorMatrix::new(poly.into_evals(), 1 << NUM_VARIABLES));
        let witness = SuffixProver::<F, F>::new_witness(vec![table], 0);

        let params = BinaryPcsParams {
            log_inv_rate: LOG_INV_RATE,
            pow_bits: 4,
            security_level: 40,
        };
        let config = BinaryPcsConfig::try_new(NUM_VARIABLES, params).unwrap();
        let encoder = AdditiveRsEncoder::<F, NaiveAdditiveNtt<F>>::default();
        let mmcs_instance = mmcs();

        let mut ch = challenger();
        let (_commitment, prover_data) =
            commit(&config, &encoder, &mmcs_instance, &mut ch, witness);
        let (_merkle_data, _sumcheck_data, rounds, randomness, final_codeword) =
            fold_rounds_with::<false, _, _>(prover_data, &config, &mmcs_instance, &mut ch);

        assert_eq!(rounds.len(), config.num_fold_rounds() - 1);
        assert_eq!(randomness.num_variables(), NUM_VARIABLES);
        assert_eq!(final_codeword.len(), 1 << LOG_INV_RATE);

        for &beta in randomness.as_slice() {
            message.fix_suffix_var_mut(beta);
        }
        let expected = message.as_constant().expect("fully folded");
        assert!(final_codeword.iter().all(|&v| v == expected));
    }

    /// Invariant: the fused route and the reference route agree on every fold round.
    ///
    ///     round messages       round commitments      folding randomness
    ///     final codeword       transcript state left behind
    ///
    /// The fold rounds are the only place the two routes differ.
    ///
    /// Everything they produce is a deterministic function of the transcript.
    ///
    /// So the comparison is reproducible under threaded execution.
    ///
    /// The query phase's grinding search is not, which is why it stays out of scope here.
    #[test]
    fn fold_rounds_agree_with_binding_each_round() {
        // Invariant: how many passes compute a round polynomial never changes its value.
        //
        //     fused    : round r measures and applies round r-1's binding in one pass
        //     reference: round r measures, then a second pass applies round r's binding
        //
        // Fixture shapes: driven twice from identically seeded challengers.
        //
        //     8  variables: every fused pass takes the serial branch
        //     15 variables: the early rounds take the threaded branch, where a fused pass
        //                   writing its own input in place would race another task's reads
        //
        // Both folding factors, because they exercise different held-challenge paths:
        //
        //     1: every round is its own batch, so every held challenge crosses a batch
        //     3: three rounds share a batch, so a held challenge also crosses a round
        //        boundary inside one — which is where the reference route's per-round
        //        `settle()` sits
        for (num_variables, log_folding_factor) in [(8usize, 1usize), (8, 3), (15, 1), (15, 3)] {
            let params = BinaryPcsParams {
                log_inv_rate: LOG_INV_RATE,
                pow_bits: 4,
                security_level: 40,
            };
            let config =
                BinaryPcsConfig::try_new_with_folding(num_variables, params, log_folding_factor)
                    .unwrap();

            // The shipped encoder, not the naive one.
            //
            // The larger arity is out of reach of a quadratic transform in a debug build.
            let encoder = AdditiveRsEncoder::<F>::default();
            let mmcs_instance = mmcs();

            let mut rng = SmallRng::seed_from_u64(0xF0FA + num_variables as u64);
            let table = Table::rand(&mut rng, 1, num_variables);
            let shape = format!("{num_variables} variables, arity {log_folding_factor}");

            // Fused route.
            let mut got_ch = challenger();
            let (_commitment, got_data) = commit(
                &config,
                &encoder,
                &mmcs_instance,
                &mut got_ch,
                SuffixProver::<F, F>::new_witness(vec![table.clone()], 0),
            );
            let (_, got_sumcheck, got_rounds, got_randomness, got_final) =
                fold_rounds_with::<false, _, _>(got_data, &config, &mmcs_instance, &mut got_ch);

            // Reference route, from an identically seeded challenger.
            let mut want_ch = challenger();
            let (_commitment, want_data) = commit(
                &config,
                &encoder,
                &mmcs_instance,
                &mut want_ch,
                SuffixProver::<F, F>::new_witness(vec![table], 0),
            );
            let (_, want_sumcheck, want_rounds, want_randomness, want_final) =
                fold_rounds_with::<true, _, _>(want_data, &config, &mmcs_instance, &mut want_ch);

            // Round by round first, so a discrepancy is localised to the round that drifted.
            assert_eq!(
                got_sumcheck.num_rounds(),
                want_sumcheck.num_rounds(),
                "{shape}: round count"
            );
            for (round, (got_msg, want_msg)) in got_sumcheck
                .polynomial_evaluations()
                .iter()
                .zip(want_sumcheck.polynomial_evaluations())
                .enumerate()
            {
                assert_eq!(got_msg, want_msg, "{shape}: round {round} message");
            }

            // The challenges follow the messages, and the codeword folds follow the challenges.
            assert_eq!(
                got_randomness.as_slice(),
                want_randomness.as_slice(),
                "{shape}: folding randomness"
            );
            assert_eq!(got_final, want_final, "{shape}: final codeword");
            for (round, (got_round, want_round)) in got_rounds.iter().zip(&want_rounds).enumerate()
            {
                assert_eq!(
                    got_round.commitment, want_round.commitment,
                    "{shape}: round {round} commitment"
                );
            }

            // The transcripts must also be left in the same state.
            //
            // Equal outputs do not show that on their own.
            //
            // Two challengers that diverged could still have produced the same outputs.
            assert_eq!(
                got_ch.sample_algebra_element::<F>(),
                want_ch.sample_algebra_element::<F>(),
                "{shape}: transcript state"
            );
        }
    }

    /// `commit` rejects a witness whose arity disagrees with the config in every build
    /// profile, not only a debug one: falling through would surface later as a confusing
    /// `FinalCodewordLengthMismatch`, or a panic inside `fold_codeword` on a length-1
    /// codeword, instead of at the actual precondition.
    #[test]
    #[should_panic(expected = "witness arity must match the config it is committed against")]
    fn commit_rejects_a_witness_arity_mismatch() {
        let params = BinaryPcsParams {
            log_inv_rate: LOG_INV_RATE,
            pow_bits: 4,
            security_level: 40,
        };
        let config = BinaryPcsConfig::try_new(NUM_VARIABLES, params).unwrap();
        let encoder = AdditiveRsEncoder::<F, NaiveAdditiveNtt<F>>::default();
        let mmcs_instance = mmcs();
        let mut ch = challenger();
        let mut rng = SmallRng::seed_from_u64(0);
        let table = Table::rand(&mut rng, 1, NUM_VARIABLES - 1);
        let witness = SuffixProver::<F, F>::new_witness(vec![table], 0);

        let _ = commit(&config, &encoder, &mmcs_instance, &mut ch, witness);
    }
}
