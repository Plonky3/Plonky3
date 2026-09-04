//! `MultilinearPcs` and `PrescribedPointPcs` over `BinaryField128`, tying the stacked-sumcheck
//! layout machinery to the commit/fold/query pipeline the rest of this crate builds.
//!
//! The opening claims an `OpeningProtocol` names are folded into the residual sumcheck exactly
//! as `p3_sumcheck::layout` already does for any other stacked-layout consumer: each claim
//! contributes an alpha-batched equality weight, and the sumcheck reduces the claim down to a
//! single scalar as it folds. What is specific to this crate is what that scalar is checked
//! against: the alpha-batched weight polynomial evaluated at the fold-derived point, times the
//! (uniform) value the final codeword carries in the clear — `verify_query_paths` ties every
//! sampled query to that same codeword, so together the two checks close the proximity and the
//! evaluation claim in one proof.

use alloc::vec;
use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_binary_dft::AdditiveRsEncoder;
use p3_binary_field::BinaryField128;
use p3_challenger::{CanObserve, CanSampleUniformBits, FieldChallenger, GrindingChallenger};
use p3_commit::{Mmcs, MultilinearPcs};
use p3_field::PrimeCharacteristicRing;
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;
use p3_sumcheck::layout::{Layout, Verifier, Witness};
use p3_sumcheck::strategy::{Basis, VariableOrder};
use p3_sumcheck::{OpeningEvals, OpeningProtocol, PrescribedPointPcs, SumcheckData, SumcheckError};

use crate::error::BinaryPcsError;
use crate::params::BinaryPcsConfig;
use crate::proof::BinaryPcsProof;
use crate::prover::{BinaryPcsProverData, commit, fold_rounds, open_queries};
use crate::verifier::{check_round_and_final_lengths, verify_query_paths};

/// A multilinear polynomial commitment scheme over `BinaryField128`: an additive-domain
/// Reed-Solomon codeword folded in lockstep with a residual sumcheck.
///
/// `L` selects the stacked-layout binding mode; the commit and fold phases reject any layout
/// other than [`SuffixProver`](p3_sumcheck::layout::SuffixProver) at run time (see `prover.rs`),
/// since the codeword fold only stays in correspondence with suffix-order binding.
pub struct BinaryPcs<MT, L> {
    config: BinaryPcsConfig,
    mmcs: MT,
    encoder: AdditiveRsEncoder<BinaryField128>,
    _marker: PhantomData<L>,
}

impl<MT, L> BinaryPcs<MT, L> {
    /// Builds a PCS instance from a derived configuration and a base-field MMCS.
    pub fn new(config: BinaryPcsConfig, mmcs: MT) -> Self {
        Self {
            config,
            mmcs,
            encoder: AdditiveRsEncoder::default(),
            _marker: PhantomData,
        }
    }
}

impl<MT, L> BinaryPcs<MT, L>
where
    MT: Mmcs<BinaryField128>,
    L: Layout<BinaryField128, BinaryField128>,
{
    /// Runs the fold-and-query pipeline shared by `open` and `open_at`, once every opening
    /// claim the protocol names has already been recorded against `prover_data.layout`.
    fn finish_open<Challenger>(
        &self,
        prover_data: BinaryPcsProverData<MT, L>,
        evals: Vec<OpeningEvals<BinaryField128>>,
        challenger: &mut Challenger,
    ) -> BinaryPcsProof<MT>
    where
        Challenger: FieldChallenger<BinaryField128>
            + GrindingChallenger<Witness = BinaryField128>
            + CanSampleUniformBits<BinaryField128>
            + CanObserve<MT::Commitment>,
    {
        let (base_merkle_data, sumcheck_data, rounds, _randomness, final_codeword) =
            fold_rounds(prover_data, &self.config, &self.mmcs, challenger);
        let query_proofs = open_queries(
            &self.config,
            &self.mmcs,
            challenger,
            &base_merkle_data,
            &rounds,
        );

        BinaryPcsProof {
            sumcheck: sumcheck_data,
            rounds: query_proofs.rounds,
            base_opened_values: query_proofs.base_opened_values,
            base_multi_proof: query_proofs.base_multi_proof,
            final_codeword: Poly::new(final_codeword),
            pow_witness: query_proofs.pow_witness,
            evals,
        }
    }

    /// Replays an opening proof's transcript against `protocol`'s claims and returns the
    /// claimed evaluations once the proof checks out.
    ///
    /// `points` selects prescribed-point mode: `Some` records each claim at its supplied point
    /// via [`Verifier::add_claim_at`], `None` samples the point from the transcript via
    /// [`Verifier::add_claim`], mirroring the prover's `eval_at`/`eval` choice.
    ///
    /// `OpeningBatchCountMismatch`, both round-count checks, `FinalCodewordLengthMismatch` and
    /// `NonEmptyPowWitnesses` run before this function performs any transcript operation of its
    /// own, so a malformed proof is rejected rather than indexed out of bounds or used to
    /// desync the replay. That does not mean the challenger itself is untouched: `verify`
    /// observes the commitment before calling here, and `verify_at`'s contract requires the
    /// caller to have done the same. `OpeningBatchSizeMismatch`, by contrast, is checked once
    /// per claim inside the claim-recording loop below, after every earlier claim in the same
    /// proof has already been absorbed — it is ordered only relative to its own claim, not to
    /// the transcript as a whole.
    ///
    /// The per-round sumcheck replay is interleaved with each intermediate round's commitment
    /// observation, one `SumcheckData::verify_rounds` call per fold round, because a single
    /// call covering every round would consume all of the proof's polynomial messages before
    /// any round commitment is observed, desyncing the transcript from what the prover produced.
    ///
    /// The claim closes by checking that the alpha-batched weight polynomial, evaluated at the
    /// point the fold challenges define, times the final codeword's (uniform) value equals the
    /// running sumcheck claim; `verify_query_paths` then ties every sampled query's fold chain
    /// to that same codeword.
    fn verify_opening<Challenger>(
        &self,
        commitment: &MT::Commitment,
        proof: &BinaryPcsProof<MT>,
        protocol: &OpeningProtocol,
        points: Option<&[Point<BinaryField128>]>,
        challenger: &mut Challenger,
    ) -> Result<Vec<OpeningEvals<BinaryField128>>, BinaryPcsError<MT::Error>>
    where
        Challenger: FieldChallenger<BinaryField128>
            + GrindingChallenger<Witness = BinaryField128>
            + CanSampleUniformBits<BinaryField128>
            + CanObserve<MT::Commitment>,
    {
        assert_eq!(
            L::strategy().variable_order,
            VariableOrder::Suffix,
            "the codeword folds adjacent pairs, which only suffix-order binding matches"
        );

        if protocol.num_openings() != proof.evals.len() {
            return Err(BinaryPcsError::OpeningBatchCountMismatch {
                expected: protocol.num_openings(),
                actual: proof.evals.len(),
            });
        }

        let num_fold_rounds = self.config.num_fold_rounds();
        if proof.sumcheck.num_rounds() != num_fold_rounds {
            return Err(SumcheckError::RoundCountMismatch {
                expected: num_fold_rounds,
                actual: proof.sumcheck.num_rounds(),
            }
            .into());
        }

        // Every fold round below replays with a freshly built `pow_witnesses: Vec::new()`
        // (see the round loop further down), so nothing ever reads `proof.sumcheck`'s own
        // vector; a non-empty one is unchecked, mutable data riding along with the proof.
        if !proof.sumcheck.pow_witnesses.is_empty() {
            return Err(BinaryPcsError::NonEmptyPowWitnesses {
                actual: proof.sumcheck.pow_witnesses.len(),
            });
        }

        check_round_and_final_lengths(&self.config, proof)?;

        // From here on the transcript is touched: every remaining check runs against the
        // replayed randomness, not the proof's raw bytes.
        let mut layout_verifier = Verifier::<BinaryField128, BinaryField128>::new(
            &protocol.table_shapes(),
            L::strategy(),
        );

        for (i, (table_idx, batch)) in protocol.iter_openings().enumerate() {
            let evals = &proof.evals[i];
            if !batch.has_same_shape(evals) {
                return Err(BinaryPcsError::OpeningBatchSizeMismatch {
                    table_idx,
                    expected: batch.len(),
                    actual: evals.len(),
                });
            }
            match points {
                Some(points) => {
                    layout_verifier
                        .add_claim_at(table_idx, batch, &points[i], evals, challenger)?;
                }
                None => {
                    layout_verifier.add_claim(table_idx, batch, evals, challenger)?;
                }
            }
        }

        // `into_sumcheck` samples this batching challenge unconditionally, even with no
        // recorded claims, and folds every claim's weight by its successive power.
        let alpha: BinaryField128 = challenger.sample_algebra_element();
        let constraint = layout_verifier.constraint(alpha);
        let mut claimed_sum = BinaryField128::ZERO;
        constraint.combine_evals(&mut claimed_sum);

        // One `verify_rounds` call per fold round: a single call spanning every round would
        // read all of the proof's polynomial messages before any intermediate commitment is
        // observed, which is not the order the prover produced them in.
        let mut betas = Vec::with_capacity(num_fold_rounds);
        for r in 0..num_fold_rounds {
            let round_data = SumcheckData {
                polynomial_evaluations: vec![proof.sumcheck.polynomial_evaluations()[r]],
                pow_witnesses: Vec::new(),
            };
            let round_point =
                round_data.verify_rounds(challenger, &mut claimed_sum, 1, 0, Basis::Evaluation)?;
            betas.push(round_point.as_slice()[0]);

            if r + 1 < num_fold_rounds {
                challenger.observe(proof.rounds[r].commitment.clone());
            }
        }

        // The codeword fold runs in `L`'s own variable order: suffix binding folds the last
        // variable first, so `betas` ends up in round order, and the committed polynomial's
        // variable-order point is its reverse — exactly what `eval_constraints_poly`
        // reconstructs internally when given that same order.
        let fold_point = Point::new(betas);
        let evaluation_of_weights = L::strategy()
            .variable_order
            .eval_constraints_poly(core::slice::from_ref(&constraint), &fold_point);
        let final_value = proof.final_codeword.as_slice()[0];
        let final_codeword_is_uniform = proof
            .final_codeword
            .as_slice()
            .iter()
            .all(|&v| v == final_value);
        if !final_codeword_is_uniform || claimed_sum != evaluation_of_weights * final_value {
            return Err(BinaryPcsError::FinalCheck);
        }

        verify_query_paths(
            &self.config,
            &self.mmcs,
            commitment,
            fold_point.as_slice(),
            proof,
            challenger,
        )?;

        Ok(proof.evals.clone())
    }
}

impl<MT, L, Challenger> MultilinearPcs<BinaryField128, Challenger> for BinaryPcs<MT, L>
where
    MT: Mmcs<BinaryField128>,
    L: Layout<BinaryField128, BinaryField128>,
    Challenger: FieldChallenger<BinaryField128>
        + GrindingChallenger<Witness = BinaryField128>
        + CanSampleUniformBits<BinaryField128>
        + CanObserve<MT::Commitment>,
{
    type Val = BinaryField128;
    type Commitment = MT::Commitment;
    type ProverData = BinaryPcsProverData<MT, L>;
    type Proof = BinaryPcsProof<MT>;
    type Error = BinaryPcsError<MT::Error>;
    type Witness = Witness<BinaryField128>;
    type OpeningProtocol = OpeningProtocol;

    fn num_vars(&self) -> usize {
        self.config.num_variables()
    }

    fn commit(
        &self,
        witness: Self::Witness,
        challenger: &mut Challenger,
    ) -> (Self::Commitment, Self::ProverData) {
        commit::<L, _, MT, _>(&self.config, &self.encoder, &self.mmcs, challenger, witness)
    }

    fn open(
        &self,
        mut prover_data: Self::ProverData,
        protocol: Self::OpeningProtocol,
        challenger: &mut Challenger,
    ) -> Self::Proof {
        let evals = protocol
            .iter_openings()
            .map(|(table_idx, batch)| prover_data.layout.eval(table_idx, batch, challenger))
            .collect();
        self.finish_open(prover_data, evals, challenger)
    }

    fn verify(
        &self,
        commitment: &Self::Commitment,
        proof: &Self::Proof,
        challenger: &mut Challenger,
        protocol: Self::OpeningProtocol,
    ) -> Result<(), Self::Error> {
        // `commit` absorbs the base commitment itself (via `Layout::commit` -> `commit_base`);
        // the verifier never calls `commit`, so it absorbs the same root here instead.
        challenger.observe(commitment.clone());
        self.verify_opening(commitment, proof, &protocol, None, challenger)
            .map(|_| ())
    }
}

impl<MT, L, Challenger> PrescribedPointPcs<BinaryField128, Challenger> for BinaryPcs<MT, L>
where
    MT: Mmcs<BinaryField128>,
    L: Layout<BinaryField128, BinaryField128>,
    Challenger: FieldChallenger<BinaryField128>
        + GrindingChallenger<Witness = BinaryField128>
        + CanSampleUniformBits<BinaryField128>
        + CanObserve<MT::Commitment>,
{
    /// Opens the columns `protocol` names at `points` instead of sampling each opening point
    /// from the transcript.
    ///
    /// This trait gives no Fiat-Shamir guarantee on its own: the caller must have bound
    /// `points` to the shared transcript (see [`PrescribedPointPcs`]'s own Fiat-Shamir /
    /// Soundness doc) before calling this method, exactly as it must before `verify_at`.
    fn open_at(
        &self,
        mut prover_data: Self::ProverData,
        protocol: &OpeningProtocol,
        points: &[Point<BinaryField128>],
        challenger: &mut Challenger,
    ) -> Self::Proof {
        assert_eq!(protocol.num_openings(), points.len());
        let evals = protocol
            .iter_openings()
            .zip(points)
            .map(|((table_idx, batch), point)| {
                prover_data
                    .layout
                    .eval_at(table_idx, batch, point, challenger)
            })
            .collect();
        self.finish_open(prover_data, evals, challenger)
    }

    /// Verifies an opening proof against `points` instead of sampling each opening point from
    /// the transcript.
    ///
    /// This trait gives no Fiat-Shamir guarantee on its own: the caller must have bound
    /// `points` to the shared transcript before calling, exactly as `open_at`'s prover side
    /// did (see [`PrescribedPointPcs`]'s own Fiat-Shamir / Soundness doc). This method also
    /// does not absorb `commitment` itself; the caller absorbs it once, before its own
    /// challenges — that absorption is what binding `points` to the transcript depends on in
    /// the first place.
    fn verify_at(
        &self,
        commitment: &Self::Commitment,
        proof: &Self::Proof,
        protocol: &OpeningProtocol,
        points: &[Point<BinaryField128>],
        challenger: &mut Challenger,
    ) -> Result<Vec<OpeningEvals<BinaryField128>>, Self::Error> {
        assert_eq!(protocol.num_openings(), points.len());
        self.verify_opening(commitment, proof, protocol, Some(points), challenger)
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use p3_binary_field::BinaryField128;
    use p3_challenger::{CanObserve, FieldChallenger};
    use p3_commit::{Mmcs, MultilinearPcs};
    use p3_multilinear_util::point::Point;
    use p3_sumcheck::layout::{Layout, PrefixProver, SuffixProver, Table};
    use p3_sumcheck::{OpeningBatch, OpeningProtocol, PrescribedPointPcs, TableShape, TableSpec};
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    use super::BinaryPcs;
    use crate::error::BinaryPcsError;
    use crate::params::{BinaryPcsConfig, BinaryPcsParams};
    use crate::proof::BinaryPcsProof;
    use crate::test_util::{MyMmcs, challenger, mmcs, run_lifecycle};

    type F = BinaryField128;

    const NUM_VARIABLES: usize = 8;

    /// Commit, open at a transcript-sampled point, verify. The prover and verifier run on
    /// independent challengers seeded identically, which is what makes a transcript desync
    /// show up as a failure rather than pass by sharing state.
    #[test]
    fn commit_open_verify_round_trips() {
        let (pcs, commitment, proof, protocol) = run_lifecycle(NUM_VARIABLES, 0);

        let mut verifier_challenger = challenger();
        pcs.verify(&commitment, &proof, &mut verifier_challenger, protocol)
            .unwrap();
    }

    /// A verifier instantiated with a `PrefixProver` layout is stopped before it inspects any
    /// proof content: `verify_opening` asserts `L::strategy()` against `VariableOrder::Suffix`,
    /// mirroring the same guard on the prover side (`prover::commit`, `prover::fold_rounds`).
    /// Without it, a `BinaryPcs<MT, PrefixProver<F, F>>` verifier would run the whole protocol
    /// against a genuine `SuffixProver` proof and reject it only at `FinalCheck`, with no
    /// indication that the layouts disagree.
    #[test]
    #[should_panic(expected = "only suffix-order binding matches")]
    fn verify_rejects_a_non_suffix_layout() {
        let (_pcs, commitment, proof, protocol) = run_lifecycle(NUM_VARIABLES, 0);

        let config = BinaryPcsConfig::try_new(NUM_VARIABLES, 0, params()).unwrap();
        let mismatched_pcs: BinaryPcs<MyMmcs, PrefixProver<F, F>> = BinaryPcs::new(config, mmcs());

        let mut verifier_challenger = challenger();
        let _ = mismatched_pcs.verify(&commitment, &proof, &mut verifier_challenger, protocol);
    }

    /// The workspace wire format is postcard, which is not self-describing; a proof type that
    /// only round-trips through a self-describing format is not actually shippable. Decoding
    /// also exercises the `Poly` deserialize path `FinalCodewordLengthMismatch` exists to guard.
    #[test]
    fn a_proof_round_trips_through_postcard() {
        let (pcs, commitment, proof, protocol) = run_lifecycle(NUM_VARIABLES, 0);

        let bytes = postcard::to_allocvec(&proof).unwrap();
        let decoded: BinaryPcsProof<MyMmcs> = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(decoded.rounds.len(), proof.rounds.len());

        let mut verifier_challenger = challenger();
        pcs.verify(&commitment, &decoded, &mut verifier_challenger, protocol)
            .unwrap();
    }

    const fn params() -> BinaryPcsParams {
        BinaryPcsParams {
            log_inv_rate: 2,
            pow_bits: 4,
            security_level: 40,
        }
    }

    /// Commits a random single-column table and opens it with [`PrescribedPointPcs::open_at`]
    /// at a point derived from the prover's own transcript, after the commitment `commit` has
    /// already absorbed — the same "sampled after the commitment" convention [`Layout::eval`]
    /// uses internally for the transcript-sampled path, just performed by the caller instead.
    ///
    /// Returns the point alongside everything a caller needs to replay `verify_at`, so a test
    /// can either re-derive a matching point on its own challenger, or reuse this exact one.
    #[allow(clippy::type_complexity)]
    fn open_at_fixture(
        seed: u64,
    ) -> (
        BinaryPcs<MyMmcs, SuffixProver<F, F>>,
        <MyMmcs as Mmcs<F>>::Commitment,
        BinaryPcsProof<MyMmcs>,
        OpeningProtocol,
        Point<F>,
    ) {
        let mut rng = SmallRng::seed_from_u64(seed);
        let table = Table::rand(&mut rng, 1, NUM_VARIABLES);
        let witness = SuffixProver::<F, F>::new_witness(vec![table], 0);

        let protocol = OpeningProtocol::new(vec![TableSpec::new(
            TableShape::new(NUM_VARIABLES, 1),
            vec![OpeningBatch::new(vec![0], Vec::new())],
        )]);

        let config = BinaryPcsConfig::try_new(NUM_VARIABLES, 0, params()).unwrap();
        let pcs = BinaryPcs::new(config, mmcs());

        let mut prover_challenger = challenger();
        let (commitment, prover_data) = pcs.commit(witness, &mut prover_challenger);
        let sample: F = prover_challenger.sample_algebra_element();
        let point = Point::expand_from_univariate(sample, NUM_VARIABLES);
        let proof = pcs.open_at(
            prover_data,
            &protocol,
            core::slice::from_ref(&point),
            &mut prover_challenger,
        );

        (pcs, commitment, proof, protocol, point)
    }

    /// Commit, open at a point the prover derives from its own transcript, verify against a
    /// point the verifier derives, independently, from its own — never a clone taken after
    /// proving. Both challengers are seeded identically and have, at the point each samples,
    /// observed exactly the commitment and nothing else, so the two derivations agree; that
    /// agreement is `PrescribedPointPcs`'s whole Fiat-Shamir contract, exercised here rather
    /// than assumed.
    #[test]
    fn verify_at_round_trips_with_transcript_derived_points() {
        let (pcs, commitment, proof, protocol, point) = open_at_fixture(0xFEED);

        // `verify_at` does not absorb the commitment; the caller does, exactly once, before
        // deriving the point it then hands to `verify_at`.
        let mut verifier_challenger = challenger();
        verifier_challenger.observe(commitment.clone());
        let sample: F = verifier_challenger.sample_algebra_element();
        let verifier_point = Point::expand_from_univariate(sample, NUM_VARIABLES);
        assert_eq!(
            verifier_point, point,
            "both sides derive the same point from identically-seeded transcripts"
        );

        pcs.verify_at(
            &commitment,
            &proof,
            &protocol,
            core::slice::from_ref(&verifier_point),
            &mut verifier_challenger,
        )
        .unwrap();
    }

    /// A verifier that skips absorbing the commitment before calling `verify_at` — the one
    /// responsibility `verify_at` leaves to its caller — must reject the proof, even though the
    /// point it supplies is the genuine one the proof was opened at. This is what would catch a
    /// future edit that "fixes" `verify_at` to absorb the commitment internally for symmetry
    /// with `verify`: such a fix could not repair a point already computed from an unabsorbed
    /// transcript, but it would make this exact scenario (genuine point, skipped absorption)
    /// verify anyway, since the point supplied here needs no repair — only the challenger does.
    #[test]
    fn verify_at_rejects_a_proof_when_the_caller_skips_absorbing_the_commitment() {
        let (pcs, commitment, proof, protocol, point) = open_at_fixture(0xFEED);

        let mut verifier_challenger = challenger();
        let err = pcs
            .verify_at(
                &commitment,
                &proof,
                &protocol,
                core::slice::from_ref(&point),
                &mut verifier_challenger,
            )
            .unwrap_err();
        assert!(
            matches!(err, BinaryPcsError::FinalCheck),
            "expected FinalCheck, got {err:?}"
        );
    }

    /// `run_lifecycle`'s single-table, single-column, `next = []` fixture stacks trivially:
    /// the committed polynomial's arity already equals the one table's own arity, so no
    /// selector bits are spent lifting a local claim into the stacked space, and the
    /// `Statements::Next` arm of `Verifier::constraint` never runs. A two-table layout forces
    /// real selector lifting, and a `next` opening exercises the repeat-last successor view.
    ///
    /// Table A costs one slot per column at its own arity — a multi-column table does not
    /// share a hypercube across its columns — so its two columns at arity 2 cost `2 * 2^2 = 8`
    /// cells; table B costs `1 * 2^3 = 8` more. The stacked arity is `log2_ceil` of that total,
    /// `4`, not a sum of the two tables' own arities.
    #[test]
    fn commit_open_verify_round_trips_with_a_stacked_multi_table_layout() {
        let table_a_arity = 2;
        let table_b_arity = 3;
        let stacked_arity = 4;

        let mut rng = SmallRng::seed_from_u64(0x57AC);
        let table_a = Table::rand(&mut rng, 2, table_a_arity);
        let table_b = Table::rand(&mut rng, 1, table_b_arity);
        let witness = SuffixProver::<F, F>::new_witness(vec![table_a, table_b], 0);

        let protocol = OpeningProtocol::new(vec![
            TableSpec::new(
                TableShape::new(table_a_arity, 2),
                vec![OpeningBatch::new(vec![0, 1], Vec::new())],
            ),
            TableSpec::new(
                TableShape::new(table_b_arity, 1),
                // Column 0 opened both directly and through the repeat-last successor view,
                // at the same sampled point, which is what puts a `Statements::Next` entry
                // into the constraint the final check evaluates.
                vec![OpeningBatch::new(vec![0], vec![0])],
            ),
        ]);

        let config = BinaryPcsConfig::try_new(stacked_arity, 0, params()).unwrap();
        let pcs: BinaryPcs<MyMmcs, SuffixProver<F, F>> = BinaryPcs::new(config, mmcs());

        let mut prover_challenger = challenger();
        let (commitment, prover_data) = pcs.commit(witness, &mut prover_challenger);
        let proof = pcs.open(prover_data, protocol.clone(), &mut prover_challenger);

        let mut verifier_challenger = challenger();
        pcs.verify(&commitment, &proof, &mut verifier_challenger, protocol)
            .unwrap();
    }
}
