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

use p3_binary_dft::AdditiveRsEncoder;
use p3_binary_field::BinaryField128;
use p3_challenger::{CanObserve, CanSampleUniformBits, FieldChallenger, GrindingChallenger};
use p3_commit::{Mmcs, MultilinearPcs};
use p3_field::PrimeCharacteristicRing;
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;
use p3_sumcheck::layout::{Layout, Verifier, Witness};
use p3_sumcheck::strategy::Basis;
use p3_sumcheck::{
    OpeningEvals, OpeningProtocol, PrescribedOpeningSecurity, PrescribedPointPcs, SumcheckData,
    SumcheckError,
};
use p3_util::log2_ceil_usize;

use crate::PcsLayout;
use crate::error::BinaryPcsError;
use crate::params::BinaryPcsConfig;
use crate::proof::BinaryPcsProof;
use crate::prover::{BinaryPcsProverData, commit, fold_rounds, open_queries};
use crate::verifier::{
    check_canonical_pow_witness, check_round_and_final_lengths, verify_query_paths,
};

/// A multilinear polynomial commitment scheme over `BinaryField128`: an additive-domain
/// Reed-Solomon codeword folded in lockstep with a residual sumcheck.
///
/// The stacked-layout binding mode is fixed rather than chosen: the codeword fold merges
/// adjacent pairs, which only the suffix-order binding of
/// [`SuffixProver`](p3_sumcheck::layout::SuffixProver) matches.
pub struct BinaryPcs<MT> {
    config: BinaryPcsConfig,
    mmcs: MT,
    encoder: AdditiveRsEncoder<BinaryField128>,
}

impl<MT> BinaryPcs<MT> {
    /// Builds a PCS instance from a derived configuration and a base-field MMCS.
    pub fn new(config: BinaryPcsConfig, mmcs: MT) -> Self {
        Self {
            config,
            mmcs,
            encoder: AdditiveRsEncoder::default(),
        }
    }
}

impl<MT> BinaryPcs<MT>
where
    MT: Mmcs<BinaryField128>,
{
    /// Check table dimensions and scalar claim capacity before committing or opening.
    ///
    /// Security rejection is a behavior change: protocols accepted by older releases can
    /// exceed the configured target. Use this preflight or [`Self::try_open`] /
    /// [`Self::try_open_at`] to handle rejection without the infallible traits' panic.
    pub fn validate_opening_protocol(
        &self,
        protocol: &OpeningProtocol,
    ) -> Result<(), BinaryPcsError<MT::Error>> {
        let total = protocol
            .table_shapes()
            .iter()
            .try_fold(0usize, |total, table| {
                let rows = 1usize.checked_shl(table.num_variables().try_into().ok()?)?;
                total.checked_add(rows.checked_mul(table.width())?)
            });
        if !matches!(total, Some(total) if total > 0 && log2_ceil_usize(total) == self.config.num_variables())
        {
            return Err(BinaryPcsError::InvalidOpeningProtocol);
        }
        let actual =
            Self::opening_claim_count(protocol).ok_or(BinaryPcsError::InvalidOpeningProtocol)?;
        let max = self.config.max_opening_claims();
        if actual > max {
            return Err(BinaryPcsError::OpeningClaimCountExceedsSecurityBudget {
                actual,
                max,
                security_level: self.config.security_level(),
            });
        }
        Ok(())
    }

    fn opening_claim_count(protocol: &OpeningProtocol) -> Option<usize> {
        protocol
            .iter_openings()
            .try_fold(0usize, |count, (_, batch)| count.checked_add(batch.len()))
    }

    /// Produce a sampled-point opening, returning an error for an invalid or over-budget
    /// protocol before touching the challenger. Prover data must match the committed tables.
    #[tracing::instrument(name = "binary pcs open", skip_all)]
    pub fn try_open<Challenger>(
        &self,
        mut prover_data: BinaryPcsProverData<MT>,
        protocol: &OpeningProtocol,
        challenger: &mut Challenger,
    ) -> Result<BinaryPcsProof<MT>, BinaryPcsError<MT::Error>>
    where
        Challenger: FieldChallenger<BinaryField128>
            + GrindingChallenger<Witness = BinaryField128>
            + CanSampleUniformBits<BinaryField128>
            + CanObserve<MT::Commitment>,
    {
        self.validate_opening_protocol(protocol)?;
        let evals = protocol
            .iter_openings()
            .map(|(table_idx, batch)| prover_data.layout.eval(table_idx, batch, challenger))
            .collect();
        Ok(self.finish_open(prover_data, evals, challenger))
    }

    /// Produce a prescribed-point opening with recoverable protocol/point-budget errors.
    /// Points must already be bound to the transcript as required by [`PrescribedPointPcs`].
    /// Rejection leaves the challenger unchanged. Prover data must match the committed tables.
    #[tracing::instrument(name = "binary pcs open", skip_all)]
    pub fn try_open_at<Challenger>(
        &self,
        mut prover_data: BinaryPcsProverData<MT>,
        protocol: &OpeningProtocol,
        points: &[Point<BinaryField128>],
        challenger: &mut Challenger,
    ) -> Result<BinaryPcsProof<MT>, BinaryPcsError<MT::Error>>
    where
        Challenger: FieldChallenger<BinaryField128>
            + GrindingChallenger<Witness = BinaryField128>
            + CanSampleUniformBits<BinaryField128>
            + CanObserve<MT::Commitment>,
    {
        self.validate_opening_protocol(protocol)?;
        Self::validate_points(protocol, points)?;
        let evals = protocol
            .iter_openings()
            .zip(points)
            .map(|((table_idx, batch), point)| {
                prover_data
                    .layout
                    .eval_at(table_idx, batch, point, challenger)
            })
            .collect();
        Ok(self.finish_open(prover_data, evals, challenger))
    }

    fn validate_points(
        protocol: &OpeningProtocol,
        points: &[Point<BinaryField128>],
    ) -> Result<(), BinaryPcsError<MT::Error>> {
        let shapes = protocol.table_shapes();
        if protocol.num_openings() != points.len()
            || protocol
                .iter_openings()
                .zip(points)
                .any(|((table, _), point)| point.num_variables() != shapes[table].num_variables())
        {
            return Err(BinaryPcsError::OpeningPointShapeMismatch);
        }
        Ok(())
    }

    /// Runs the fold-and-query pipeline shared by `open` and `open_at`, once every opening
    /// claim the protocol names has already been recorded against `prover_data.layout`.
    fn finish_open<Challenger>(
        &self,
        prover_data: BinaryPcsProverData<MT>,
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
    /// The proof-shape checks below all run before this function performs any transcript
    /// operation of its own:
    ///
    /// - `OpeningBatchCountMismatch`
    /// - both round-count checks
    /// - `FinalCodewordLengthMismatch`
    /// - `NonEmptyPowWitnesses`
    /// - `NonCanonicalPowWitness`
    ///
    /// A malformed proof is rejected there, rather than indexed out of bounds or used to
    /// desync the replay.
    ///
    /// The challenger is not untouched by then: `verify` observes the commitment before
    /// calling here, and `verify_at`'s contract requires the caller to have done the same.
    ///
    /// `OpeningBatchSizeMismatch` sits outside that group: it is checked once per claim inside
    /// the claim-recording loop below, ordered only against its own claim and not against the
    /// transcript as a whole.
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
    #[tracing::instrument(name = "binary pcs verify", skip_all)]
    fn verify_opening<'p, Challenger>(
        &self,
        commitment: &MT::Commitment,
        proof: &'p BinaryPcsProof<MT>,
        protocol: &OpeningProtocol,
        points: Option<&[Point<BinaryField128>]>,
        challenger: &mut Challenger,
    ) -> Result<&'p [OpeningEvals<BinaryField128>], BinaryPcsError<MT::Error>>
    where
        Challenger: FieldChallenger<BinaryField128>
            + GrindingChallenger<Witness = BinaryField128>
            + CanSampleUniformBits<BinaryField128>
            + CanObserve<MT::Commitment>,
    {
        self.validate_opening_protocol(protocol)?;
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

        // At a zero grinding budget the witness never reaches the sponge, so its value is
        // pinned here rather than by the grind.
        check_canonical_pow_witness(&self.config, proof)?;

        // From here on the transcript is touched: every remaining check runs against the
        // replayed randomness, not the proof's raw bytes.
        let mut layout_verifier = Verifier::<BinaryField128, BinaryField128>::new(
            &protocol.table_shapes(),
            PcsLayout::strategy(),
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

        // Replay each polynomial before its own challenge, observing roots only at the
        // batch boundaries chosen by the verifier's configuration.
        let mut betas = Vec::with_capacity(num_fold_rounds);
        for (batch, (start, arity)) in self.config.fold_batches().enumerate() {
            for r in start..start + arity {
                let round_data = SumcheckData {
                    polynomial_evaluations: vec![proof.sumcheck.polynomial_evaluations()[r]],
                    pow_witnesses: Vec::new(),
                };
                let round_point = round_data.verify_rounds(
                    challenger,
                    &mut claimed_sum,
                    1,
                    0,
                    Basis::Evaluation,
                )?;
                betas.push(round_point.as_slice()[0]);
            }
            if batch + 1 < self.config.num_fold_batches() {
                challenger.observe(proof.rounds[batch].commitment.clone());
            }
        }

        // The codeword fold runs in the layout's own variable order: suffix binding folds the
        // last variable first, so `betas` ends up in round order, and the committed
        // polynomial's variable-order point is its reverse — exactly what
        // `eval_constraints_poly` reconstructs internally when given that same order.
        let fold_point = Point::new(betas);
        let evaluation_of_weights = PcsLayout::strategy()
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

        Ok(&proof.evals)
    }
}

impl<MT, Challenger> MultilinearPcs<BinaryField128, Challenger> for BinaryPcs<MT>
where
    MT: Mmcs<BinaryField128>,
    Challenger: FieldChallenger<BinaryField128>
        + GrindingChallenger<Witness = BinaryField128>
        + CanSampleUniformBits<BinaryField128>
        + CanObserve<MT::Commitment>,
{
    type Val = BinaryField128;
    type Commitment = MT::Commitment;
    type ProverData = BinaryPcsProverData<MT>;
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
        commit(&self.config, &self.encoder, &self.mmcs, challenger, witness)
    }

    /// Panics if the opening protocol exceeds its security budget. This trait is infallible;
    /// use `BinaryPcs::try_open` to handle the security rejection as a typed error.
    fn open(
        &self,
        prover_data: Self::ProverData,
        protocol: Self::OpeningProtocol,
        challenger: &mut Challenger,
    ) -> Self::Proof {
        self.try_open(prover_data, &protocol, challenger)
            .unwrap_or_else(|e| panic!("invalid binary PCS opening protocol: {e}"))
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

impl<MT, Challenger> PrescribedPointPcs<BinaryField128, Challenger> for BinaryPcs<MT>
where
    MT: Mmcs<BinaryField128>,
    Challenger: FieldChallenger<BinaryField128>
        + GrindingChallenger<Witness = BinaryField128>
        + CanSampleUniformBits<BinaryField128>
        + CanObserve<MT::Commitment>,
{
    fn prescribed_security(&self, protocol: &OpeningProtocol) -> Option<PrescribedOpeningSecurity> {
        self.validate_opening_protocol(protocol).ok()?;
        Some(PrescribedOpeningSecurity {
            error: self
                .config
                .security_regime()
                .opening_error(Self::opening_claim_count(protocol)?),
            log2_max_candidates: 0.0,
        })
    }

    /// Panics on an invalid or over-budget protocol or mismatched points. Use
    /// `BinaryPcs::try_open_at` for typed errors. Points must already be transcript-bound.
    fn open_at(
        &self,
        prover_data: Self::ProverData,
        protocol: &OpeningProtocol,
        points: &[Point<BinaryField128>],
        challenger: &mut Challenger,
    ) -> Self::Proof {
        self.try_open_at(prover_data, protocol, points, challenger)
            .unwrap_or_else(|e| panic!("invalid binary PCS prescribed opening protocol: {e}"))
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
        Self::validate_points(protocol, points)?;
        self.verify_opening(commitment, proof, protocol, Some(points), challenger)
            .map(<[_]>::to_vec)
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
    use p3_sumcheck::layout::{Layout, SuffixProver, Table};
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
        let (pcs, commitment, proof, protocol) = run_lifecycle(NUM_VARIABLES);

        let mut verifier_challenger = challenger();
        pcs.verify(&commitment, &proof, &mut verifier_challenger, protocol)
            .unwrap();
    }

    /// The workspace wire format is postcard, which is not self-describing; a proof type that
    /// only round-trips through a self-describing format is not actually shippable. Decoding
    /// also exercises the `Poly` deserialize path `FinalCodewordLengthMismatch` exists to guard.
    #[test]
    fn a_proof_round_trips_through_postcard() {
        let (pcs, commitment, proof, protocol) = run_lifecycle(NUM_VARIABLES);

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
        log_folding_factor: usize,
    ) -> (
        BinaryPcs<MyMmcs>,
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

        let config = BinaryPcsConfig::try_new(NUM_VARIABLES, params())
            .unwrap()
            .try_with_folding(log_folding_factor)
            .unwrap();
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
        let (pcs, commitment, proof, protocol, point) = open_at_fixture(0xFEED, 1);

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

    #[test]
    fn batched_verify_at_round_trips_with_transcript_derived_points() {
        let (pcs, commitment, proof, protocol, point) = open_at_fixture(0xFEED, 3);
        let mut verifier_challenger = challenger();
        verifier_challenger.observe(commitment.clone());
        let sample: F = verifier_challenger.sample_algebra_element();
        let verifier_point = Point::expand_from_univariate(sample, NUM_VARIABLES);
        assert_eq!(verifier_point, point);
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
        let (pcs, commitment, proof, protocol, point) = open_at_fixture(0xFEED, 1);

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

        let config = BinaryPcsConfig::try_new(stacked_arity, params()).unwrap();
        let pcs: BinaryPcs<MyMmcs> = BinaryPcs::new(config, mmcs());

        let mut prover_challenger = challenger();
        let (commitment, prover_data) = pcs.commit(witness, &mut prover_challenger);
        let proof = pcs.open(prover_data, protocol.clone(), &mut prover_challenger);

        let mut verifier_challenger = challenger();
        pcs.verify(&commitment, &proof, &mut verifier_challenger, protocol)
            .unwrap();
    }
}
