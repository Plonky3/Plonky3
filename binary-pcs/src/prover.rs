//! Prover side: commit the stacked polynomial as one codeword column, then fold that codeword
//! in lockstep with the residual sumcheck.
//!
//! [`PcsLayout`] commits with no preprocessing depth, so `Layout::commit` produces a width-1
//! codeword — one Reed-Solomon-encoded column, the whole committed polynomial — and
//! `Layout::into_sumcheck` consumes zero preprocessing rounds, leaving every one of the
//! `num_variables` residual sumcheck rounds a folding round. Each round's challenge is used
//! twice: it binds one multilinear variable, through [`PcsLayout`]'s evaluation-basis suffix
//! binding, and it folds the codeword, through [`fold_codeword`], a Reed-Solomon codeword fold
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
use crate::fold::fold_codeword;
use crate::params::BinaryPcsConfig;
use crate::proof::RoundProof;
use crate::verifier::{flat_pair_indices, sample_query_indices};

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

/// Runs the residual sumcheck in lockstep with the codeword fold: one sumcheck round, one
/// 2-to-1 fold, per iteration, with one Merkle commitment for every fold except the last —
/// that fold's codeword is returned directly as the final codeword rather than committed, since
/// its whole content already travels in the clear.
///
/// The single grinding budget is spent once before the query phase, not here, so every round
/// runs with `pow_bits = 0`.
///
/// Returns the base commitment's Merkle prover data (handed back so the caller can still open
/// base-round queries against it), the sumcheck transcript, one [`RoundCommitment`] per fold
/// round except the last, the folding randomness in round order — `randomness.as_slice()[r]` is
/// round `r`'s challenge, matching what [`Layout::into_sumcheck`] returns — and the final
/// folded codeword.
#[must_use]
#[allow(clippy::type_complexity)]
pub(crate) fn fold_rounds<MT, Ch>(
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
    fold_rounds_with(prover_data, config, mmcs, challenger, false)
}

/// Drives the fold rounds, optionally applying each round's binding on the spot.
///
/// On the spot means two passes per round.
/// The round reads its tables to measure, then a second pass applies the binding.
///
/// Left outstanding, the next round's measuring pass absorbs it.
/// That is one pass per round.
///
/// The two produce the same round polynomials, so the tests can pin one against the
/// other.
#[must_use]
#[allow(clippy::type_complexity)]
fn fold_rounds_with<MT, Ch>(
    prover_data: BinaryPcsProverData<MT>,
    config: &BinaryPcsConfig,
    mmcs: &MT,
    challenger: &mut Ch,
    bind_each_round: bool,
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

    // One sumcheck round, one codeword fold, per iteration.
    //
    //     round r < last:  challenge beta_r -> fold 2-to-1 -> commit -> observe root
    //     round r = last:  challenge beta_r -> fold 2-to-1 -> return in the clear
    //
    // The last fold's codeword is never committed.
    // It travels in the proof as the final codeword instead.
    let num_fold_rounds = config.num_fold_rounds();
    let mut rounds: Vec<RoundCommitment<MT>> = Vec::with_capacity(num_fold_rounds - 1);
    let mut final_codeword = Vec::new();
    for round in 0..num_fold_rounds {
        let challenge =
            sumcheck.compute_sumcheck_polynomials(&mut sumcheck_data, challenger, 1, 0, None);
        let beta = challenge.as_slice()[0];
        randomness.extend(&challenge);

        // The reference route: apply this round's binding now, rather than letting the
        // next round's measuring pass absorb it.
        if bind_each_round {
            sumcheck.settle();
        }

        // Fold out of the previous round's Merkle leaves, never out of a copy of them.
        // The commitment scheme already owns every codeword it committed.
        // The fold allocates its own output, so this borrow ends before the push below.
        let source = if round == 0 {
            &merkle_data
        } else {
            &rounds[round - 1].merkle_data
        };
        let folded = fold_codeword(&mmcs.get_matrices(source)[0].values, beta);

        if round + 1 < num_fold_rounds {
            let (commitment, round_data) = mmcs.commit_matrix(RowMajorMatrix::new(folded, 1));
            challenger.observe(commitment.clone());
            rounds.push(RoundCommitment {
                commitment,
                merkle_data: round_data,
            });
        } else {
            final_codeword = folded;
        }
    }

    // The last round's challenge stays unapplied, and is never applied.
    //
    // The codeword fold, not the sumcheck tables, is what carries the folded message
    // forward, and the sumcheck goes out of scope here without anything else reading it.
    // The final binding pass would therefore only compute a table nobody looks at.
    drop(sumcheck);

    (
        merkle_data,
        sumcheck_data,
        rounds,
        randomness,
        final_codeword,
    )
}

/// Reference route for the tests: applies each round's binding on the spot instead of
/// letting the next round's measuring pass absorb it.
///
/// Two passes per round rather than one, over the same round polynomials.
#[cfg(test)]
#[must_use]
#[allow(clippy::type_complexity)]
pub(crate) fn fold_rounds_binding_each_round<MT, Ch>(
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
    fold_rounds_with(prover_data, config, mmcs, challenger, true)
}

/// The query phase's prover-side output: every opening `verifier::verify_query_paths` needs,
/// plus the grinding witness.
pub(crate) struct QueryProofs<MT: Mmcs<BinaryField128>> {
    /// Openings of the base commitment: two width-1 rows per query — the pair's low- then
    /// high-indexed symbol — in the order query indices were sampled.
    pub base_opened_values: Vec<Vec<BinaryField128>>,
    /// Multiproof for the base commitment's queried rows.
    pub base_multi_proof: MT::MultiProof,
    /// One [`RoundProof`] per intermediate folding round, i.e. every round except the last.
    pub rounds: Vec<RoundProof<MT>>,
    /// Witness for the single grind before the query phase.
    pub pow_witness: BinaryField128,
}

/// Runs the query phase: grinds the single proof-of-work witness, samples query indices from
/// the base codeword's domain, then opens the base commitment and every intermediate
/// folding-round commitment at the paired indices each sampled query needs.
///
/// `rounds` is every [`RoundCommitment`] `fold_rounds` produced: one per fold round except the
/// last, whose codeword is never committed — it travels in the clear as the proof's
/// `final_codeword` instead, so a Merkle path for it would only repeat what the verifier can
/// already read directly.
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
        config.num_fold_rounds() - 1,
        "rounds is the caller's own fold_rounds output, never proof-supplied data"
    );

    let pow_witness = challenger.grind(config.pow_bits());

    let domain_size = config.domain_size();
    let indices =
        sample_query_indices::<_, BinaryField128>(domain_size, config.num_queries(), challenger);

    let base_indices = flat_pair_indices(&indices, 0);
    let (base_values, base_multi_proof) = mmcs.open_multi_batch(&base_indices, base_merkle_data);
    let base_opened_values = single_matrix_rows(base_values);

    let opened_rounds = rounds
        .iter()
        .enumerate()
        .map(|(r, round)| {
            let round_indices = flat_pair_indices(&indices, r + 1);
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
    use alloc::vec;

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

    use super::{commit, fold_codeword, fold_rounds, fold_rounds_binding_each_round};
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

    /// Drives the crate's own [`commit`] and [`fold_rounds`] end to end and checks their
    /// output against an independent oracle: folding the original message variable by
    /// variable, via [`Poly::fix_suffix_var_mut`], over the randomness `fold_rounds` returns,
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
            fold_rounds(prover_data, &config, &mmcs_instance, &mut ch);

        assert_eq!(rounds.len(), config.num_fold_rounds() - 1);
        assert_eq!(randomness.num_variables(), NUM_VARIABLES);
        assert_eq!(final_codeword.len(), 1 << LOG_INV_RATE);

        for &beta in randomness.as_slice() {
            message.fix_suffix_var_mut(beta);
        }
        let expected = message.as_constant().expect("fully folded");
        assert!(final_codeword.iter().all(|&v| v == expected));
    }

    /// The fused route and the reference route agree on every fold round: the round
    /// messages, the round commitments, the folding randomness, the final codeword, and the
    /// transcript state left behind.
    ///
    /// The fold rounds are the only place the two routes differ, and everything they produce
    /// is a deterministic function of the transcript, so this comparison is reproducible even
    /// under threaded execution — unlike the query phase's grinding search.
    #[test]
    fn fold_rounds_agree_with_binding_each_round() {
        // Invariant: how many passes compute a round polynomial never changes its value.
        //
        //     fused    : round r measures and applies round r-1's binding in one pass
        //     reference: round r measures, then a second pass applies round r's binding
        //
        // Fixture state: two arities, driven twice from identically seeded challengers.
        //
        //     8  variables: every fused pass takes the serial branch
        //     15 variables: the early rounds take the threaded branch, where a fused pass
        //                   writing its own input in place would race another task's reads
        for num_variables in [8usize, 15] {
            let params = BinaryPcsParams {
                log_inv_rate: LOG_INV_RATE,
                pow_bits: 4,
                security_level: 40,
            };
            let config = BinaryPcsConfig::try_new(num_variables, params).unwrap();

            // The shipped encoder, not the naive one: the larger arity is out of reach of a
            // quadratic transform in a debug build.
            let encoder = AdditiveRsEncoder::<F>::default();
            let mmcs_instance = mmcs();

            let mut rng = SmallRng::seed_from_u64(0xF0FA + num_variables as u64);
            let table = Table::rand(&mut rng, 1, num_variables);

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
                fold_rounds(got_data, &config, &mmcs_instance, &mut got_ch);

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
                fold_rounds_binding_each_round(want_data, &config, &mmcs_instance, &mut want_ch);

            // Round by round first, so a discrepancy is localised to the round that drifted.
            assert_eq!(
                got_sumcheck.num_rounds(),
                want_sumcheck.num_rounds(),
                "{num_variables} variables: round count"
            );
            for (round, (got_msg, want_msg)) in got_sumcheck
                .polynomial_evaluations()
                .iter()
                .zip(want_sumcheck.polynomial_evaluations())
                .enumerate()
            {
                assert_eq!(
                    got_msg, want_msg,
                    "{num_variables} variables: round {round} message"
                );
            }

            // The challenges follow the messages, and the codeword folds follow the challenges.
            assert_eq!(
                got_randomness.as_slice(),
                want_randomness.as_slice(),
                "{num_variables} variables: folding randomness"
            );
            assert_eq!(
                got_final, want_final,
                "{num_variables} variables: final codeword"
            );
            for (round, (got_round, want_round)) in got_rounds.iter().zip(&want_rounds).enumerate()
            {
                assert_eq!(
                    got_round.commitment, want_round.commitment,
                    "{num_variables} variables: round {round} commitment"
                );
            }

            // The transcripts must also be in the same state, which the outputs alone do not
            // show: two challengers that diverged could still have produced equal outputs.
            assert_eq!(
                got_ch.sample_algebra_element::<F>(),
                want_ch.sample_algebra_element::<F>(),
                "{num_variables} variables: transcript state"
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
