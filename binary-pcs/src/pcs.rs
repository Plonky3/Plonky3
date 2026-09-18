//! The multilinear commitment scheme this crate exposes.
//! It ties the stacked-sumcheck layout machinery to the commit, fold and query pipeline.
//!
//! An opening protocol's claims fold into the residual sumcheck.
//! The layout already folds any other consumer's claims the same way.
//!
//! Each claim contributes one alpha-batched equality weight.
//! The sumcheck then reduces the whole batch to a single scalar as it folds.
//!
//! What is specific to this crate is what that scalar is checked against.
//!
//! It is the alpha-batched weight polynomial at the fold-derived point.
//! That value is multiplied by the uniform value the final codeword carries.
//!
//! The query paths tie every sampled position to that same codeword.
//! So together the two checks close the proximity and the evaluation claim in one proof.
//!
//! # The two fields
//!
//! ```text
//!     committed alphabet   the columns and the base codeword
//!     challenge field      every challenge, every folded codeword, every claimed value
//! ```
//!
//! A narrower alphabet shrinks the largest object in the proof without moving a challenge.

use alloc::vec;
use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_binary_dft::{AdditiveNtt, AdditiveRsEncoder, EncodableLevel};
use p3_binary_field::TowerLevel;
use p3_challenger::fs::TranscriptField;
use p3_challenger::{CanObserve, CanSampleUniformBits, FieldChallenger, GrindingChallenger};
use p3_commit::{Encoder, Mmcs, MultilinearPcs};
use p3_field::ExtensionField;
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;
use p3_sumcheck::layout::{Layout, Verifier, Witness, observe_commitment};
use p3_sumcheck::strategy::Basis;
use p3_sumcheck::{
    OpeningEvals, OpeningProtocol, PrescribedOpeningSecurity, PrescribedPointPcs, SumcheckData,
    SumcheckError,
};
use p3_util::log2_ceil_usize;

use crate::PcsLayout;
use crate::error::BinaryPcsError;
use crate::fold::{ChallengeField, FoldAlphabet};
use crate::params::{BinaryPcsConfig, BinaryPcsConfigError};
use crate::proof::BinaryPcsProof;
use crate::prover::{BinaryPcsProverData, commit, fold_rounds_with, open_queries};
use crate::transcript::{BinaryPcsProverTranscript, BinaryPcsShape, BinaryPcsVerifierTranscript};
use crate::verifier::{
    check_canonical_pow_witness, check_round_and_final_lengths, verify_query_paths,
};

/// Why an opening could not be produced or accepted, for one base commitment scheme.
type Failure<F, MT> = BinaryPcsError<F, <MT as Mmcs<F>>::Error>;

/// An opening proof, or the reason there is none.
type Opening<F, EF, MT, MX> = Result<BinaryPcsProof<F, EF, MT, MX>, Failure<F, MT>>;

/// A multilinear polynomial commitment scheme over a binary tower field: an additive-domain
/// Reed-Solomon codeword folded in lockstep with a residual sumcheck.
///
/// The stacked-layout binding mode is fixed rather than chosen.
/// The codeword fold merges adjacent pairs, which only suffix-order binding matches.
///
/// The base codeword and the folded codewords live over different fields.
/// Each therefore has its own commitment scheme.
///
/// The two schemes must report the same failure type.
/// One commitment family instantiated at two levels does.
pub struct BinaryPcs<F: EncodableLevel, EF, MT, MX, E = <F as EncodableLevel>::Encoder> {
    config: BinaryPcsConfig,
    mmcs: MT,
    round_mmcs: MX,
    encoder: E,
    _fields: PhantomData<(F, EF)>,
}

impl<F: EncodableLevel, EF: TowerLevel, MT, MX> BinaryPcs<F, EF, MT, MX> {
    /// Builds a PCS instance from a derived configuration and its two commitment schemes.
    ///
    /// The schedule carries the two widths it was derived for, and both are checked here.
    ///
    /// Every cap the derivation applied belongs to one of those two levels:
    ///
    /// ```text
    ///     committed alphabet   the additive domain the codeword lives in, and the grind
    ///     challenge field      the width every algebraic error is charged against
    /// ```
    ///
    /// A schedule derived elsewhere would otherwise report a bound it cannot deliver.
    ///
    /// # Arguments
    ///
    /// - `config`: the validated fold and query schedule both sides read.
    /// - `mmcs`: commits the base codeword over the committed alphabet.
    /// - `round_mmcs`: commits every folded codeword over the challenge field.
    ///
    /// # Errors
    ///
    /// Returns an error unless the schedule was derived for these two levels.
    pub fn new(
        config: BinaryPcsConfig,
        mmcs: MT,
        round_mmcs: MX,
    ) -> Result<Self, BinaryPcsConfigError> {
        config.check_alphabets::<F, EF>()?;
        Ok(Self {
            config,
            mmcs,
            round_mmcs,
            encoder: F::Encoder::default(),
            _fields: PhantomData,
        })
    }

    /// Variables of the committed stacked polynomial.
    ///
    /// The same number the commitment trait reports, reachable without naming a challenger.
    #[must_use]
    pub const fn num_variables(&self) -> usize {
        self.config.num_variables()
    }
}

impl<F, EF, MT, MX, Ntt> BinaryPcs<F, EF, MT, MX, AdditiveRsEncoder<F, Ntt>>
where
    F: EncodableLevel,
    EF: TowerLevel,
    Ntt: AdditiveNtt<F> + Sync,
{
    /// Builds an instance around an explicitly selected additive transform.
    ///
    /// # Errors
    ///
    /// Returns an error unless the schedule was derived for these two tower levels.
    pub fn with_ntt(
        config: BinaryPcsConfig,
        mmcs: MT,
        round_mmcs: MX,
        ntt: Ntt,
    ) -> Result<Self, BinaryPcsConfigError> {
        config.check_alphabets::<F, EF>()?;
        Ok(Self {
            config,
            mmcs,
            round_mmcs,
            encoder: AdditiveRsEncoder::new(ntt),
            _fields: PhantomData,
        })
    }
}

impl<F, EF, MT, MX, E> BinaryPcs<F, EF, MT, MX, E>
where
    F: EncodableLevel + TranscriptField + FoldAlphabet<EF>,
    EF: ChallengeField<F> + ExtensionField<F> + TowerLevel + FoldAlphabet<EF>,
    MT: Mmcs<F>,
    MX: Mmcs<EF, Error = MT::Error>,
    E: Encoder<F> + Sync,
{
    /// Check table dimensions and scalar claim capacity before committing or opening.
    ///
    /// Security rejection is a behavior change.
    /// A protocol an older release accepted can exceed the configured target.
    ///
    /// Use this preflight, or a fallible opening entry point, to validate one directly.
    /// The commitment traits propagate the same typed errors.
    pub fn validate_opening_protocol(
        &self,
        protocol: &OpeningProtocol,
    ) -> Result<(), BinaryPcsError<F, MT::Error>> {
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
        mut prover_data: BinaryPcsProverData<F, EF, MT>,
        protocol: &OpeningProtocol,
        challenger: &mut Challenger,
    ) -> Opening<F, EF, MT, MX>
    where
        Challenger: FieldChallenger<F>
            + GrindingChallenger<Witness = F>
            + CanSampleUniformBits<F>
            + CanObserve<MT::Commitment>
            + CanObserve<MX::Commitment>,
    {
        self.validate_opening_protocol(protocol)?;
        let evals = protocol
            .iter_openings()
            .map(|(table_idx, batch)| prover_data.layout.eval(table_idx, batch, challenger))
            .collect();
        Ok(self.finish_open(prover_data, evals, challenger))
    }

    /// Produce a prescribed-point opening with recoverable protocol and budget errors.
    ///
    /// Points must already be transcript-bound, as the prescribed-point contract requires.
    /// Rejection leaves the challenger unchanged.
    ///
    /// Prover data must match the committed tables.
    #[tracing::instrument(name = "binary pcs open", skip_all)]
    pub fn try_open_at<Challenger>(
        &self,
        mut prover_data: BinaryPcsProverData<F, EF, MT>,
        protocol: &OpeningProtocol,
        points: &[Point<EF>],
        challenger: &mut Challenger,
    ) -> Opening<F, EF, MT, MX>
    where
        Challenger: FieldChallenger<F>
            + GrindingChallenger<Witness = F>
            + CanSampleUniformBits<F>
            + CanObserve<MT::Commitment>
            + CanObserve<MX::Commitment>,
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
        points: &[Point<EF>],
    ) -> Result<(), BinaryPcsError<F, MT::Error>> {
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

    /// Runs the fold-and-query pipeline both opening modes share.
    /// Every claim the protocol names is already recorded against the layout by then.
    fn finish_open<Challenger>(
        &self,
        prover_data: BinaryPcsProverData<F, EF, MT>,
        evals: Vec<OpeningEvals<EF>>,
        challenger: &mut Challenger,
    ) -> BinaryPcsProof<F, EF, MT, MX>
    where
        Challenger: FieldChallenger<F>
            + GrindingChallenger<Witness = F>
            + CanSampleUniformBits<F>
            + CanObserve<MT::Commitment>
            + CanObserve<MX::Commitment>,
    {
        self.finish_open_with::<false, Challenger>(prover_data, evals, challenger)
    }

    /// The body of the opening pipeline, with the fold route selected by a const parameter.
    ///
    /// `BIND_EACH_ROUND` is the fold phase's own parameter, carried one level up:
    ///
    /// ```text
    ///     false: the shipped route, each held challenge absorbed by the next round's pass
    ///     true : the reference route, each round's binding applied on its own pass
    /// ```
    ///
    /// Both routes send the same transcript, so a test can pin one against the other
    /// without reproducing anything the shipped path does after the fold rounds.
    fn finish_open_with<const BIND_EACH_ROUND: bool, Challenger>(
        &self,
        prover_data: BinaryPcsProverData<F, EF, MT>,
        evals: Vec<OpeningEvals<EF>>,
        challenger: &mut Challenger,
    ) -> BinaryPcsProof<F, EF, MT, MX>
    where
        Challenger: FieldChallenger<F>
            + GrindingChallenger<Witness = F>
            + CanSampleUniformBits<F>
            + CanObserve<MT::Commitment>
            + CanObserve<MX::Commitment>,
    {
        // One driver spans the fold batches and the query phase.
        //
        // The description is therefore walked exactly once.
        let shape = BinaryPcsShape::new(&self.config);
        let mut transcript = BinaryPcsProverTranscript::new(challenger, shape);

        let (base_merkle_data, sumcheck_data, rounds, _randomness, final_codeword) =
            fold_rounds_with::<BIND_EACH_ROUND, F, EF, MT, MX, _>(
                prover_data,
                &self.config,
                &self.mmcs,
                &self.round_mmcs,
                &mut transcript,
            );
        let query_proofs = open_queries(
            &self.config,
            &self.mmcs,
            &self.round_mmcs,
            &mut transcript,
            &base_merkle_data,
            &rounds,
            &final_codeword,
        );

        // Require that every described step was played.
        transcript.finish();

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
    /// `points` selects prescribed-point mode.
    ///
    /// ```text
    ///     supplied  ->  each claim is recorded at its own point
    ///     absent    ->  each point is sampled from the transcript
    /// ```
    ///
    /// Either way the choice mirrors the one the prover made.
    ///
    /// The proof-shape checks below all run before this function performs any transcript
    /// operation of its own:
    ///
    /// - the opening-batch count
    /// - both round-count checks
    /// - the final codeword's length
    /// - the presence of sumcheck grinding witnesses
    /// - the canonical zero-difficulty grinding witness
    ///
    /// A malformed proof is rejected there, rather than indexed out of bounds or used to
    /// desync the replay.
    ///
    /// The challenger is not untouched by then.
    /// The sampled-point entry point observes the commitment before calling here.
    ///
    /// The prescribed-point one requires the caller to have done the same.
    ///
    /// The per-claim batch-size check sits outside that group.
    /// It runs once per claim, inside the claim-recording loop below.
    ///
    /// It is ordered against its own claim only, never against the transcript as a whole.
    ///
    /// The sumcheck replay is interleaved with each intermediate round's commitment.
    /// One round-verification call runs per fold round.
    ///
    /// A single call covering every round would read all the polynomial messages first.
    /// No round commitment would be observed until after them, desyncing the two sides.
    ///
    /// The claim closes on one equality.
    ///
    /// ```text
    ///     weights(fold point) * final value  ==  running sumcheck claim
    /// ```
    ///
    /// The weight polynomial is the alpha-batched one the recorded claims define.
    /// The fold point is what the fold challenges name, and the final value is uniform.
    ///
    /// The query paths then tie every sampled query's fold chain to that same codeword.
    #[tracing::instrument(name = "binary pcs verify", skip_all)]
    fn verify_opening<'p, Challenger>(
        &self,
        commitment: &MT::Commitment,
        proof: &'p BinaryPcsProof<F, EF, MT, MX>,
        protocol: &OpeningProtocol,
        points: Option<&[Point<EF>]>,
        challenger: &mut Challenger,
    ) -> Result<&'p [OpeningEvals<EF>], BinaryPcsError<F, MT::Error>>
    where
        Challenger: FieldChallenger<F>
            + GrindingChallenger<Witness = F>
            + CanSampleUniformBits<F>
            + CanObserve<MT::Commitment>
            + CanObserve<MX::Commitment>,
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

        // Every fold round below replays with a freshly built, always-empty witness vector.
        // Nothing ever reads the vector the proof carries.
        //
        // A non-empty one is therefore unchecked, mutable data riding along with the proof.
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
        let mut layout_verifier =
            Verifier::<F, EF>::new(&protocol.table_shapes(), PcsLayout::<F, EF>::strategy());

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

        // The layout draws this batching challenge unconditionally.
        //
        // It is drawn even when no claim was recorded at all.
        //
        // Both sides draw it through the layout.
        //
        // The recorded claim counts therefore reach the sponge first.
        // One driver spans the fold batches and the query phase.
        //
        // The prover seeds at the same point, just before the batching challenge.
        let shape = BinaryPcsShape::new(&self.config);
        let mut transcript = BinaryPcsVerifierTranscript::new(challenger, shape);

        let alpha = transcript.fold_batch(|ch| layout_verifier.batching_challenge(ch));
        let constraint = layout_verifier.constraint(alpha);
        let mut claimed_sum = EF::ZERO;
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
                // A rejection leaves the driver mid-description, so release it first.
                let round_point = match transcript.fold_batch(|ch| {
                    round_data.verify_rounds(ch, &mut claimed_sum, 1, 0, Basis::Evaluation)
                }) {
                    Ok(point) => point,
                    Err(error) => {
                        transcript.abort();
                        return Err(error.into());
                    }
                };
                betas.push(round_point.as_slice()[0]);
            }
            if batch + 1 < self.config.num_fold_batches() {
                transcript.oracle_commitment(proof.rounds[batch].commitment.clone());
            }
        }

        // The codeword fold runs in the layout's own variable order.
        // Suffix binding folds the last variable first, so the challenges end up in round order.
        //
        // The committed polynomial's variable-order point is that sequence reversed.
        // Reconstructing it is what the constraint evaluation does internally.
        let fold_point = Point::new(betas);
        let evaluation_of_weights = PcsLayout::<F, EF>::strategy()
            .variable_order
            .eval_constraints_poly(core::slice::from_ref(&constraint), &fold_point);
        let final_value = proof.final_codeword.as_slice()[0];
        let final_codeword_is_uniform = proof
            .final_codeword
            .as_slice()
            .iter()
            .all(|&v| v == final_value);
        if !final_codeword_is_uniform || claimed_sum != evaluation_of_weights * final_value {
            transcript.abort();
            return Err(BinaryPcsError::FinalCheck);
        }

        match verify_query_paths(
            &self.config,
            &self.mmcs,
            &self.round_mmcs,
            commitment,
            fold_point.as_slice(),
            proof,
            &mut transcript,
        ) {
            Ok(()) => transcript.finish(),
            Err(error) => {
                transcript.abort();
                return Err(error);
            }
        }

        Ok(&proof.evals)
    }
}

impl<F, EF, MT, MX, E, Challenger> MultilinearPcs<EF, Challenger> for BinaryPcs<F, EF, MT, MX, E>
where
    F: EncodableLevel + TranscriptField + FoldAlphabet<EF>,
    EF: ChallengeField<F> + ExtensionField<F> + TowerLevel + FoldAlphabet<EF>,
    MT: Mmcs<F>,
    MX: Mmcs<EF, Error = MT::Error>,
    E: Encoder<F> + Sync,
    Challenger: FieldChallenger<F>
        + GrindingChallenger<Witness = F>
        + CanSampleUniformBits<F>
        + CanObserve<MT::Commitment>
        + CanObserve<MX::Commitment>,
{
    type Val = F;
    type Commitment = MT::Commitment;
    type ProverData = BinaryPcsProverData<F, EF, MT>;
    type Proof = BinaryPcsProof<F, EF, MT, MX>;
    type Error = BinaryPcsError<F, MT::Error>;
    type ProverError = BinaryPcsError<F, MT::Error>;
    type Witness = Witness<F>;
    type OpeningProtocol = OpeningProtocol;

    fn num_vars(&self) -> usize {
        self.config.num_variables()
    }

    fn commit(
        &self,
        witness: Self::Witness,
        challenger: &mut Challenger,
    ) -> Result<(Self::Commitment, Self::ProverData), Self::ProverError> {
        let (commitment, prover_data) =
            commit::<F, EF, _, _>(&self.config, &self.encoder, &self.mmcs, witness);

        // The verifier reaches the same call, so neither side can bind differently.
        self.observe_commitment(&commitment, challenger);

        Ok((commitment, prover_data))
    }

    fn observe_commitment(&self, commitment: &Self::Commitment, challenger: &mut Challenger) {
        observe_commitment::<F, _, _>(challenger, commitment.clone());
    }

    /// Rejects an over-budget protocol before touching the challenger.
    fn open(
        &self,
        prover_data: Self::ProverData,
        protocol: Self::OpeningProtocol,
        challenger: &mut Challenger,
    ) -> Result<Self::Proof, Self::ProverError> {
        self.try_open(prover_data, &protocol, challenger)
    }

    fn verify(
        &self,
        commitment: &Self::Commitment,
        proof: &Self::Proof,
        challenger: &mut Challenger,
        protocol: Self::OpeningProtocol,
    ) -> Result<(), Self::Error> {
        // The prover binds the root while committing, so the verifier binds it here.
        self.observe_commitment(commitment, challenger);
        self.verify_opening(commitment, proof, &protocol, None, challenger)
            .map(|_| ())
    }
}

impl<F, EF, MT, MX, E, Challenger> PrescribedPointPcs<EF, Challenger>
    for BinaryPcs<F, EF, MT, MX, E>
where
    F: EncodableLevel + TranscriptField + FoldAlphabet<EF>,
    EF: ChallengeField<F> + ExtensionField<F> + TowerLevel + FoldAlphabet<EF>,
    MT: Mmcs<F>,
    MX: Mmcs<EF, Error = MT::Error>,
    E: Encoder<F> + Sync,
    Challenger: FieldChallenger<F>
        + GrindingChallenger<Witness = F>
        + CanSampleUniformBits<F>
        + CanObserve<MT::Commitment>
        + CanObserve<MX::Commitment>,
{
    fn prescribed_security(&self, protocol: &OpeningProtocol) -> Option<PrescribedOpeningSecurity> {
        self.validate_opening_protocol(protocol).ok()?;
        Some(PrescribedOpeningSecurity {
            terms: vec![
                self.config
                    .security_regime()
                    .opening_term(Self::opening_claim_count(protocol)?),
            ],
            log2_max_candidates: 0.0,
        })
    }

    /// Rejects invalid or over-budget protocols before touching the challenger.
    /// Points must already be transcript-bound.
    fn open_at(
        &self,
        prover_data: Self::ProverData,
        protocol: &OpeningProtocol,
        points: &[Point<EF>],
        challenger: &mut Challenger,
    ) -> Result<Self::Proof, Self::ProverError> {
        self.try_open_at(prover_data, protocol, points, challenger)
    }

    /// Verifies an opening proof against `points` instead of sampling each opening point from
    /// the transcript.
    ///
    /// This trait gives no Fiat-Shamir guarantee on its own.
    /// The caller must bind `points` to the shared transcript first, as the prover did.
    ///
    /// This method does not absorb `commitment` either.
    /// The caller absorbs it once, before drawing any challenge of its own.
    ///
    /// That absorption is what binding `points` to the transcript rests on.
    fn verify_at(
        &self,
        commitment: &Self::Commitment,
        proof: &Self::Proof,
        protocol: &OpeningProtocol,
        points: &[Point<EF>],
        challenger: &mut Challenger,
    ) -> Result<Vec<OpeningEvals<EF>>, Self::Error> {
        Self::validate_points(protocol, points)?;
        self.verify_opening(commitment, proof, protocol, Some(points), challenger)
            .map(<[_]>::to_vec)
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;
    use alloc::{format, vec};

    use p3_binary_field::{BinaryField8, BinaryField16, BinaryField64, BinaryField128};
    use p3_challenger::FieldChallenger;
    use p3_commit::{Mmcs, MultilinearPcs};
    use p3_multilinear_util::point::Point;
    use p3_sumcheck::layout::{Layout, SuffixProver, Table};
    use p3_sumcheck::{OpeningBatch, OpeningProtocol, PrescribedPointPcs, TableShape, TableSpec};
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    use super::BinaryPcs;
    use crate::error::BinaryPcsError;
    use crate::params::{BinaryPcsConfig, BinaryPcsConfigError, BinaryPcsParams};
    use crate::proof::BinaryPcsProof;
    use crate::prover::BinaryPcsProverData;
    use crate::test_util::{MyChallenger, MyMmcs, challenger, mmcs, run_lifecycle};

    type F = BinaryField128;

    const NUM_VARIABLES: usize = 8;

    #[test]
    fn constructor_rejects_schedules_derived_for_other_tower_levels() {
        // Invariant: the schedule's security bounds belong to both derivation fields.
        //
        // A wider alphabet can admit a larger domain and a wider grinding witness.
        // A wider challenge field can also report smaller algebraic errors.
        // Reusing either bound at a narrower level would overstate security.
        let high_security = BinaryPcsParams {
            log_inv_rate: 2,
            pow_bits: 0,
            security_level: 100,
        };

        // Fixture state: derive at (128, 128), then request (64, 64).
        // The alphabet mismatch is checked first and returned through the constructor.
        let wide =
            BinaryPcsConfig::try_new::<BinaryField128, BinaryField128>(20, high_security).unwrap();
        let committed = BinaryPcs::<BinaryField64, BinaryField64, (), ()>::new(wide, (), ()).err();
        assert_eq!(
            committed,
            Some(BinaryPcsConfigError::CommittedFieldMismatch {
                derived: 128,
                actual: 64,
            })
        );

        // Fixture state: derive at (16, 128), then narrow only the committed alphabet.
        let mixed = BinaryPcsConfig::try_new::<BinaryField16, BinaryField128>(
            8,
            BinaryPcsParams {
                log_inv_rate: 2,
                pow_bits: 0,
                security_level: 40,
            },
        )
        .unwrap();
        let committed = BinaryPcs::<BinaryField8, BinaryField128, (), ()>::new(mixed, (), ()).err();
        assert_eq!(
            committed,
            Some(BinaryPcsConfigError::CommittedFieldMismatch {
                derived: 16,
                actual: 8,
            })
        );

        // Fixture state: keep the 16-bit alphabet and narrow only the challenge field.
        let challenge = BinaryPcs::<BinaryField16, BinaryField64, (), ()>::new(mixed, (), ()).err();
        assert_eq!(
            challenge,
            Some(BinaryPcsConfigError::ChallengeFieldMismatch {
                derived: 128,
                actual: 64,
            })
        );
    }

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
        let decoded: BinaryPcsProof<F, F, MyMmcs, MyMmcs> = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(decoded.rounds.len(), proof.rounds.len());

        let mut verifier_challenger = challenger();
        pcs.verify(&commitment, &decoded, &mut verifier_challenger, protocol)
            .unwrap();
    }

    /// Opens through the reference fold route, and through the shipped path everywhere else.
    ///
    /// The claims are recorded exactly as [`BinaryPcs::try_open`] records them, then
    /// [`BinaryPcs::finish_open_with`] runs the rest with `BIND_EACH_ROUND = true`.
    ///
    /// Nothing the shipped path does after the fold rounds is reproduced here.
    ///
    /// So the two proofs can only differ if a round polynomial did.
    fn open_binding_each_round(
        pcs: &BinaryPcs<F, F, MyMmcs, MyMmcs>,
        mut prover_data: BinaryPcsProverData<F, F, MyMmcs>,
        protocol: &OpeningProtocol,
        challenger: &mut MyChallenger,
    ) -> BinaryPcsProof<F, F, MyMmcs, MyMmcs> {
        let evals = protocol
            .iter_openings()
            .map(|(table_idx, batch)| prover_data.layout.eval(table_idx, batch, challenger))
            .collect();

        pcs.finish_open_with::<true, MyChallenger>(prover_data, evals, challenger)
    }

    #[test]
    fn fusing_the_binding_into_the_measuring_pass_leaves_the_proof_byte_identical() {
        // Invariant: how many passes compute a round polynomial never changes its value.
        //
        //     shipped  : round r measures and applies round r-1's binding in one pass
        //     reference: round r measures, then a second pass applies round r's binding
        //
        // Both routes must send the same transcript, so the proof bytes must match.
        //
        // Fixture state: one random single-column table.
        //
        //     opened at : a transcript-sampled point
        //     driven    : twice, from identically seeded challengers
        //
        // The grinding budget is zero, which is what makes the whole proof reproducible.
        //
        // A non-zero budget searches its witness across threads.
        //
        // It keeps whichever witness a thread finds first.
        //
        // The witness, and every transcript draw after it, then varies run to run.
        //
        // Both folding factors, because a held challenge takes a different path in each:
        //
        //     1: every round is its own fold batch, so it only ever crosses a batch
        //     3: three rounds share a batch, so it also crosses a round boundary inside one
        let num_variables = NUM_VARIABLES;
        let params = BinaryPcsParams {
            log_inv_rate: 2,
            pow_bits: 0,
            security_level: 40,
        };

        for log_folding_factor in [1usize, 3] {
            let mut rng = SmallRng::seed_from_u64(0x50FA);
            let table = Table::rand(&mut rng, 1, num_variables);
            let shape = format!("arity {log_folding_factor}");

            let protocol = OpeningProtocol::new(vec![TableSpec::new(
                TableShape::new(num_variables, 1),
                vec![OpeningBatch::new(vec![0], Vec::new())],
            )]);

            let config = BinaryPcsConfig::try_new_with_folding::<F, F>(
                num_variables,
                params,
                log_folding_factor,
            )
            .unwrap();
            let pcs = BinaryPcs::new(config, mmcs(), mmcs()).unwrap();

            // Shipped route.
            let mut got_challenger = challenger();
            let (got_commitment, got_data) = pcs
                .commit(
                    SuffixProver::<F, F>::new_witness(vec![table.clone()], 0),
                    &mut got_challenger,
                )
                .unwrap();
            let got = pcs
                .open(got_data, protocol.clone(), &mut got_challenger)
                .unwrap();

            // Reference route, from an identically seeded challenger.
            let mut want_challenger = challenger();
            let (want_commitment, want_data) = pcs
                .commit(
                    SuffixProver::<F, F>::new_witness(vec![table], 0),
                    &mut want_challenger,
                )
                .unwrap();
            let want = open_binding_each_round(&pcs, want_data, &protocol, &mut want_challenger);

            // Round by round first, so a discrepancy is localised to the round that drifted.
            assert_eq!(
                got.sumcheck.num_rounds(),
                want.sumcheck.num_rounds(),
                "{shape}: round counts"
            );
            for (round, (got_msg, want_msg)) in got
                .sumcheck
                .polynomial_evaluations()
                .iter()
                .zip(want.sumcheck.polynomial_evaluations())
                .enumerate()
            {
                assert_eq!(got_msg, want_msg, "{shape}: round {round} message");
            }

            // Then the whole proof, on the wire.
            let got_bytes = postcard::to_allocvec(&got).unwrap();
            let want_bytes = postcard::to_allocvec(&want).unwrap();
            assert_eq!(got_bytes, want_bytes, "{shape}: proof bytes");

            // The transcripts must also be left in the same state.
            //
            // Equal proof bytes do not show that on their own.
            //
            // Two challengers that diverged could still have produced the same bytes.
            assert_eq!(
                got_challenger.sample_algebra_element::<F>(),
                want_challenger.sample_algebra_element::<F>(),
                "{shape}: transcript state after opening"
            );

            // And the proof both routes produced verifies.
            assert_eq!(got_commitment, want_commitment, "{shape}");
            pcs.verify(&got_commitment, &got, &mut challenger(), protocol)
                .unwrap();
        }
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
        BinaryPcs<F, F, MyMmcs, MyMmcs>,
        <MyMmcs as Mmcs<F>>::Commitment,
        BinaryPcsProof<F, F, MyMmcs, MyMmcs>,
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

        let config = BinaryPcsConfig::try_new::<F, F>(NUM_VARIABLES, params())
            .unwrap()
            .try_with_folding(log_folding_factor)
            .unwrap();
        let pcs = BinaryPcs::new(config, mmcs(), mmcs()).unwrap();

        let mut prover_challenger = challenger();
        let (commitment, prover_data) = pcs.commit(witness, &mut prover_challenger).unwrap();
        let sample: F = prover_challenger.sample_algebra_element();
        let point = Point::expand_from_univariate(sample, NUM_VARIABLES);
        let proof = pcs
            .open_at(
                prover_data,
                &protocol,
                core::slice::from_ref(&point),
                &mut prover_challenger,
            )
            .unwrap();

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
        pcs.observe_commitment(&commitment, &mut verifier_challenger);
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
        pcs.observe_commitment(&commitment, &mut verifier_challenger);
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

        let config = BinaryPcsConfig::try_new::<F, F>(stacked_arity, params()).unwrap();
        let pcs: BinaryPcs<F, F, MyMmcs, MyMmcs> = BinaryPcs::new(config, mmcs(), mmcs()).unwrap();

        let mut prover_challenger = challenger();
        let (commitment, prover_data) = pcs.commit(witness, &mut prover_challenger).unwrap();
        let proof = pcs
            .open(prover_data, protocol.clone(), &mut prover_challenger)
            .unwrap();

        let mut verifier_challenger = challenger();
        pcs.verify(&commitment, &proof, &mut verifier_challenger, protocol)
            .unwrap();
    }
}
