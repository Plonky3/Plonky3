use alloc::borrow::Cow;
use alloc::vec::Vec;
use alloc::{format, vec};
use core::fmt::Debug;
use core::ops::Deref;

use errors::VerifierError;
use p3_challenger::fs::TranscriptField;
use p3_challenger::{CanObserve, CanSampleUniformBits, FieldChallenger, GrindingChallenger};
use p3_commit::{ExtensionMmcs, Mmcs, MmcsArith};
use p3_field::{Dup, ExtensionField, Field, PrimeCharacteristicRing};
use p3_matrix::Dimensions;
use p3_multilinear_util::point::Point;
use p3_sumcheck::constraints::statement::SelectStatement;
use p3_sumcheck::constraints::{Constraint, Statements};
use p3_sumcheck::layout::Verifier as LayoutVerifier;
use p3_sumcheck::strategy::{Basis, VariableOrder};
use p3_sumcheck::verify_final_sumcheck_rounds_in;
use p3_verifier_arith::{Arith, Lift, Native};
use tracing::instrument;

use super::committer::reader::ParsedCommitment;
use crate::alloc::string::ToString;
use crate::domain::{WhirDomain, WhirQueryPoint};
use crate::parameters::{RoundConfig, WhirConfig};
use crate::pcs::proof::{QueryOpenings, WhirProof};
use crate::transcript::{WhirShape, WhirVerifierTranscript, lift_query_indices};

pub mod errors;

/// A weight's multilinear extension at a point, in natural variable order.
pub type WeightEval<'a, A> = dyn Fn(&mut A, &Point<<A as Arith>::EF>) -> <A as Arith>::EF + 'a;

/// An extra initial weight a verifier closes: its claimed sum, and its extension at a point.
#[allow(missing_debug_implementations)]
pub struct InitialWeightCheck<'a, A: Arith> {
    /// The weight's claimed inner product with the committed polynomial.
    pub sum: A::EF,
    /// The weight's multilinear extension at a point, in natural variable order.
    pub eval: &'a WeightEval<'a, A>,
}

/// Replays a WHIR opening proof against a public commitment and the
/// constraint built by the layout adapter.
///
/// # Borrowing
///
/// - Config and Merkle scheme are borrowed for the lifetime of the check.
/// - Nothing is cloned across `verify`.
/// - Construction only checks the config against the domain, so a fresh verifier per proof is cheap.
///
/// # Variable order
///
/// Tag declared by the prover at commit time. Selects which way folding
/// randomness is consumed in the final identity and STIR unfold:
///
/// ```text
///     Prefix:  fold(rs)         -> final eval, query unfold
///     Suffix:  fold(rs.rev())   -> same checks, reversed binding
/// ```
#[derive(Debug)]
pub struct WhirVerifier<'a, EF, F, Dft, MT, Challenger>
where
    F: Field,
    EF: ExtensionField<F>,
    MT: Mmcs<F>,
{
    /// Derived per-protocol parameters and per-round configuration.
    pub(crate) config: &'a WhirConfig<EF, F, Challenger>,
    /// Base-field Merkle commitment scheme used to authenticate STIR queries.
    pub(crate) mmcs: &'a MT,
    /// Domain implementation used to reconstruct queried evaluation points.
    pub(crate) domain: &'a Dft,
    /// Binding direction used to interpret folding randomness.
    pub(crate) variable_order: VariableOrder,
}

impl<'a, EF, F, Dft, MT, Challenger> WhirVerifier<'a, EF, F, Dft, MT, Challenger>
where
    F: Field + TranscriptField,
    EF: ExtensionField<F>,
    Dft: WhirDomain<F, EF>,
    MT: Mmcs<F>,
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F> + CanSampleUniformBits<F>,
{
    /// Wraps the verifier-side dependencies into a single replay context.
    ///
    /// # Arguments
    ///
    /// - `config`         — derived per-protocol parameters and per-round configuration.
    /// - `domain`         — code and evaluation-point map used by every query.
    /// - `mmcs`           — base-field Merkle commitment scheme used to authenticate STIR queries.
    /// - `variable_order` — binding direction the prover declared at commit time.
    ///
    /// # Panics
    ///
    /// Panics when the configuration was derived for another domain.
    pub fn new(
        config: &'a WhirConfig<EF, F, Challenger>,
        domain: &'a Dft,
        mmcs: &'a MT,
        variable_order: VariableOrder,
    ) -> Self {
        // The verifier must reconstruct the same code and query schedule as the prover.
        assert_eq!(
            config.max_log_domain_size,
            domain.max_log_domain_size(),
            "WHIR configuration and domain have different capacities"
        );
        assert_eq!(
            config.domain_id,
            domain.protocol_id(),
            "WHIR configuration and domain have different protocol identities"
        );
        assert_eq!(
            config.stratified_queries,
            domain.stratified_queries(),
            "WHIR configuration and domain disagree on query stratification"
        );
        assert!(
            domain.supports_security_assumption(config.soundness_type),
            "WHIR configuration uses a soundness regime the domain rejects"
        );

        Self {
            config,
            domain,
            mmcs,
            variable_order,
        }
    }

    /// Verify a WHIR proof against a commitment and statement.
    ///
    /// # Arguments
    ///
    /// - `proof`: the opening proof to replay.
    /// - `challenger`: the sponge the whole proof shares, borrowed for this run.
    /// - `parsed_commitment`: the initial commitment the run opens.
    /// - `num_opening_claims`: opening claims the caller bound before this run.
    /// - `layout`: the layout the caller recorded those claims against.
    ///
    /// The batching challenge belongs to the claims the caller recorded.
    ///
    /// It does not belong to this run.
    ///
    /// It is drawn inside the initial delegation bracket, from the layout.
    ///
    /// The layout is taken rather than a closure over the sponge.
    ///
    /// A bracket checks nothing about the draws inside it.
    ///
    /// A closure could therefore draw nothing, or twice, and only fail as a
    /// well-formed proof that does not verify.
    ///
    /// Taking the layout makes one draw the only expressible shape.
    ///
    /// # Returns
    ///
    /// The folding randomness point on success.
    ///
    /// # Errors
    ///
    /// Any rejection the replay raises, transcript failures included.
    #[instrument(skip_all)]
    pub fn verify(
        &self,
        proof: &WhirProof<F, EF, MT>,
        challenger: &mut Challenger,
        parsed_commitment: &MT::Commitment,
        num_opening_claims: usize,
        layout: &LayoutVerifier<F, EF>,
    ) -> Result<Point<EF>, VerifierError>
    where
        Challenger: CanObserve<MT::Commitment>,
    {
        self.verify_in(
            &mut Native::new(),
            proof,
            challenger,
            parsed_commitment,
            num_opening_claims,
            layout,
        )
    }

    /// As [`Self::verify`], the initial fold also proving an extra weight the caller describes.
    ///
    /// The weight takes the batching power after every claim the layout recorded.
    ///
    /// # Errors
    ///
    /// As [`Self::verify`].
    pub fn verify_with_initial_weight(
        &self,
        proof: &WhirProof<F, EF, MT>,
        challenger: &mut Challenger,
        parsed_commitment: &MT::Commitment,
        layout: &LayoutVerifier<F, EF>,
        weight: InitialWeightCheck<'_, Native<F, EF>>,
    ) -> Result<Point<EF>, VerifierError>
    where
        Challenger: CanObserve<MT::Commitment>,
    {
        self.verify_with_initial_weight_in(
            &mut Native::new(),
            proof,
            challenger,
            parsed_commitment,
            layout,
            weight,
        )
    }

    /// As [`Self::verify_with_initial_weight`], within an arithmetic.
    ///
    /// # Errors
    ///
    /// As [`Self::verify`].
    pub fn verify_with_initial_weight_in<A, Ch>(
        &self,
        arith: &mut A,
        proof: &WhirProof<F, EF, MT>,
        challenger: &mut Ch,
        commitment: &<MT as MmcsArith<A, A::F>>::Target,
        layout: &LayoutVerifier<A::F, A::EF>,
        weight: InitialWeightCheck<'_, A>,
    ) -> Result<Point<A::EF>, VerifierError>
    where
        A: Lift<Val = F, Challenge = EF>,
        Ch: FieldChallenger<F>
            + GrindingChallenger<Witness = F>
            + CanSampleUniformBits<F>
            + CanObserve<<MT as Mmcs<F>>::Commitment>,
        MT: MmcsArith<
                A,
                A::F,
                Commitment = <MT as Mmcs<F>>::Commitment,
                MultiProof = <MT as Mmcs<F>>::MultiProof,
            >,
        ExtensionMmcs<F, EF, &'a MT>: MmcsArith<
                A,
                A::EF,
                MultiProof = <MT as Mmcs<F>>::MultiProof,
                Target = <MT as MmcsArith<A, A::F>>::Target,
            >,
    {
        self.verify_inner(
            arith,
            proof,
            challenger,
            commitment,
            0,
            layout,
            Some(weight),
        )
    }

    /// As [`Self::verify`], within an arithmetic.
    ///
    /// The commitment and the layout's claims are values of the arithmetic.
    ///
    /// # Errors
    ///
    /// As [`Self::verify`].
    pub fn verify_in<A, Ch>(
        &self,
        arith: &mut A,
        proof: &WhirProof<F, EF, MT>,
        challenger: &mut Ch,
        commitment: &<MT as MmcsArith<A, A::F>>::Target,
        num_opening_claims: usize,
        layout: &LayoutVerifier<A::F, A::EF>,
    ) -> Result<Point<A::EF>, VerifierError>
    where
        A: Lift<Val = F, Challenge = EF>,
        Ch: FieldChallenger<F>
            + GrindingChallenger<Witness = F>
            + CanSampleUniformBits<F>
            + CanObserve<<MT as Mmcs<F>>::Commitment>,
        MT: MmcsArith<
                A,
                A::F,
                Commitment = <MT as Mmcs<F>>::Commitment,
                MultiProof = <MT as Mmcs<F>>::MultiProof,
            >,
        ExtensionMmcs<F, EF, &'a MT>: MmcsArith<
                A,
                A::EF,
                MultiProof = <MT as Mmcs<F>>::MultiProof,
                Target = <MT as MmcsArith<A, A::F>>::Target,
            >,
    {
        self.verify_inner(
            arith,
            proof,
            challenger,
            commitment,
            num_opening_claims,
            layout,
            None,
        )
    }

    /// The verification every entry point shares, with an optional extra initial weight.
    #[allow(clippy::too_many_arguments)]
    fn verify_inner<A, Ch>(
        &self,
        arith: &mut A,
        proof: &WhirProof<F, EF, MT>,
        challenger: &mut Ch,
        commitment: &<MT as MmcsArith<A, A::F>>::Target,
        num_opening_claims: usize,
        layout: &LayoutVerifier<A::F, A::EF>,
        weight: Option<InitialWeightCheck<'_, A>>,
    ) -> Result<Point<A::EF>, VerifierError>
    where
        A: Lift<Val = F, Challenge = EF>,
        Ch: FieldChallenger<F>
            + GrindingChallenger<Witness = F>
            + CanSampleUniformBits<F>
            + CanObserve<<MT as Mmcs<F>>::Commitment>,
        MT: MmcsArith<
                A,
                A::F,
                Commitment = <MT as Mmcs<F>>::Commitment,
                MultiProof = <MT as Mmcs<F>>::MultiProof,
            >,
        ExtensionMmcs<F, EF, &'a MT>: MmcsArith<
                A,
                A::EF,
                MultiProof = <MT as Mmcs<F>>::MultiProof,
                Target = <MT as MmcsArith<A, A::F>>::Target,
            >,
    {
        // Reject a proof that carries the wrong number of rounds before any
        // transcript work. The per-round commitment slot is checked further
        // down, where each round is parsed.
        let expected_rounds = self.n_rounds();
        if proof.rounds.len() != expected_rounds {
            return Err(VerifierError::RoundCountMismatch {
                expected: expected_rounds,
                actual: proof.rounds.len(),
            });
        }

        // A zero-difficulty site leaves its witness unread, so the value is pinned here
        // rather than by the grind.
        //
        // Why: `check_witness` returns `true` at zero bits without absorbing anything.
        //
        //     pow_bits = 0 -> prover emits zero, verifier reads nothing -> pin it here
        //     pow_bits > 0 -> prover grinds,     verifier resamples     -> the grind pins it
        //
        // Each round carries its own difficulty, so each is compared against its own.
        for (round, round_proof) in proof.rounds.iter().enumerate() {
            if self.round_parameters[round].pow_bits == 0 && round_proof.pow_witness != F::ZERO {
                return Err(VerifierError::NonCanonicalPowWitness { round });
            }
        }
        if self.final_round_config().pow_bits == 0 && proof.final_pow_witness != F::ZERO {
            return Err(VerifierError::NonCanonicalPowWitness {
                round: expected_rounds,
            });
        }

        // A schedule whose folds use up every variable plays no closing sumcheck.
        //
        //     final rounds = 0 -> prover writes None, verifier reads nothing -> pin it here
        //     final rounds > 0 -> the closing fold replays the data          -> the replay binds it
        if self.final_sumcheck_rounds == 0 && proof.final_sumcheck.is_some() {
            return Err(VerifierError::UnexpectedFinalSumcheck);
        }

        // One driver spans the whole run, so the description is walked exactly once.
        let shape = WhirShape::new(self.config, num_opening_claims);
        let mut transcript =
            arith.seeding(|| WhirVerifierTranscript::<Ch, F, EF>::new(challenger, shape));

        // A rejection releases the driver's completeness check on its way out.
        // Dropping an unfinished driver otherwise panics on top of the error.
        match self.replay(arith, proof, &mut transcript, commitment, layout, weight) {
            Ok(randomness) => {
                transcript.finish();
                Ok(randomness)
            }
            Err(err) => {
                transcript.abort();
                Err(err)
            }
        }
    }

    /// Replay every described step against the proof.
    ///
    /// # Errors
    ///
    /// Any rejection a step raises, or any consistency check that fails.
    #[allow(clippy::too_many_lines)]
    fn replay<A, Ch>(
        &self,
        arith: &mut A,
        proof: &WhirProof<F, EF, MT>,
        transcript: &mut WhirVerifierTranscript<'_, Ch, F, EF>,
        commitment: &<MT as MmcsArith<A, A::F>>::Target,
        layout: &LayoutVerifier<A::F, A::EF>,
        weight: Option<InitialWeightCheck<'_, A>>,
    ) -> Result<Point<A::EF>, VerifierError>
    where
        A: Lift<Val = F, Challenge = EF>,
        Ch: FieldChallenger<F>
            + GrindingChallenger<Witness = F>
            + CanSampleUniformBits<F>
            + CanObserve<<MT as Mmcs<F>>::Commitment>,
        MT: MmcsArith<
                A,
                A::F,
                Commitment = <MT as Mmcs<F>>::Commitment,
                MultiProof = <MT as Mmcs<F>>::MultiProof,
            >,
        ExtensionMmcs<F, EF, &'a MT>: MmcsArith<
                A,
                A::EF,
                MultiProof = <MT as Mmcs<F>>::MultiProof,
                Target = <MT as MmcsArith<A, A::F>>::Target,
            >,
    {
        let mut constraints = Vec::new();
        let mut round_folding_randomness = Vec::new();
        let mut prev_commitment = commitment.clone();

        // The delegate draws the claim-batching challenge, then replays its own rounds.
        //
        // The layout owns that draw, and the claims it batches are the caller's own.
        //
        // Both it and the rounds after it seed sub-transcripts inside this bracket.
        //
        // Initial sumcheck rounds == first-round folding factor.
        // `verify_rounds_in` rejects a proof that carries the wrong number of rounds.
        let mut weight_scale = A::EF::ZERO;
        let (constraint, mut claimed_eval, folding_randomness) =
            transcript.delegate_initial_fold(|challenger| {
                let alpha = layout.batching_challenge_in(arith, challenger);
                let constraint = layout.constraint(alpha.dup());
                let mut claimed_eval = A::EF::ZERO;
                constraint.combine_evals(&mut claimed_eval);
                // An extra weight takes the power after every recorded claim.
                if let Some(weight) = &weight {
                    weight_scale = layout.extra_weight_scale(&alpha);
                    claimed_eval += weight_scale.dup() * weight.sum.dup();
                }
                let randomness = proof.initial_sumcheck.verify_rounds_in(
                    arith,
                    challenger,
                    &mut claimed_eval,
                    self.round_folding_factor(0),
                    self.starting_folding_pow_bits,
                    Basis::Evaluation,
                );
                (constraint, claimed_eval, randomness)
            });
        constraints.push(constraint);
        round_folding_randomness.push(folding_randomness?);

        // Verify each intermediate round.
        for round_index in 0..self.n_rounds() {
            let round_params = &self.round_parameters[round_index];

            // Index is in bounds thanks to the length check at function entry,
            // so only a missing commitment slot can fail here.
            let ParsedCommitment {
                root,
                ood_statement,
            } = ParsedCommitment::parse_with_round(
                arith,
                self.mmcs,
                proof,
                transcript,
                round_params.num_variables,
                round_params.ood_samples,
                round_index,
            )?;

            // Verify STIR in-domain challenges against the previous commitment.
            let current_folding_randomness = round_folding_randomness
                .last()
                .ok_or(VerifierError::MissingFoldingRandomness { round: round_index })?;
            let stir_statement = self.verify_stir_challenges(
                arith,
                proof,
                transcript,
                round_params,
                &prev_commitment,
                current_folding_randomness,
                round_index,
            )?;

            // Rebuild the same batched constraint the prover formed for this round.
            //
            //     out-of-domain claims  ->  the equality group
            //     query openings        ->  the selection group
            //
            // The challenge drawn here weights the two groups by its successive powers.
            //
            // Both sides therefore combine the claims identically.
            //
            // The carried claim keeps the constant coefficient, so a fresh group
            // never shares a power with it.
            let constraint = Constraint::new_with_existing_claim(
                arith.sampled_ext(transcript.round_batching()),
                ood_statement.num_variables(),
                vec![
                    Statements::Eq(ood_statement),
                    Statements::Select(stir_statement),
                ],
            );
            constraint.combine_evals(&mut claimed_eval);
            constraints.push(constraint);

            // Intermediate-round sumcheck rounds == next-round folding factor.
            // `verify_rounds_in` rejects a wrong count instead of desyncing Fiat-Shamir.
            let folding_randomness = transcript.delegate_round_fold(|challenger| {
                proof.rounds[round_index].sumcheck.verify_rounds_in(
                    arith,
                    challenger,
                    &mut claimed_eval,
                    self.round_folding_factor(round_index + 1),
                    round_params.folding_pow_bits,
                    Basis::Evaluation,
                )
            })?;
            round_folding_randomness.push(folding_randomness);

            prev_commitment = root;
        }

        // Final round: receive the polynomial in the clear.
        //
        // The step records how many evaluations it accepts.
        // A proof carrying another count is rejected instead of desyncing.
        let final_evaluations = proof
            .final_poly
            .as_ref()
            .ok_or(VerifierError::MissingFinalPoly)?;
        let final_round_config = self.final_round_config();
        let final_poly = arith
            .load_ext_slice(final_evaluations.as_slice())
            .into_owned();
        for value in &final_poly {
            arith.will_observe_ext(value);
        }
        transcript.final_poly(final_evaluations.as_slice())?;

        // Verify final STIR challenges.
        let final_round_folding_randomness = round_folding_randomness.last().ok_or_else(|| {
            VerifierError::MissingFoldingRandomness {
                round: self.n_rounds(),
            }
        })?;
        let stir_statement = self.verify_stir_challenges(
            arith,
            proof,
            transcript,
            &final_round_config,
            &prev_commitment,
            final_round_folding_randomness,
            self.n_rounds(),
        )?;

        stir_statement.verify_in(arith, &final_poly, || VerifierError::StirChallengeFailed {
            challenge_id: 0,
            details: "STIR constraint verification failed on final polynomial".to_string(),
        })?;

        // A run with no closing rounds delegates nothing, so no bracket is played.
        let final_sumcheck_randomness = transcript
            .delegate_final_fold(|challenger| {
                verify_final_sumcheck_rounds_in(
                    arith,
                    proof.final_sumcheck.as_ref(),
                    challenger,
                    &mut claimed_eval,
                    self.final_sumcheck_rounds,
                    self.final_folding_pow_bits,
                    Basis::Evaluation,
                )
            })
            .transpose()?
            .unwrap_or_else(|| Point::new(Vec::new()));
        round_folding_randomness.push(final_sumcheck_randomness.clone());

        // Compute the full folding randomness across all rounds.
        let folding_randomness = Point::new(
            round_folding_randomness
                .into_iter()
                .flat_map(IntoIterator::into_iter)
                .collect(),
        );

        // Evaluate the constraint polynomial at the folding point.
        let mut evaluation_of_weights = self
            .variable_order
            .eval_constraints_poly(&constraints, &folding_randomness);
        if let Some(weight) = weight {
            // Each round binds its variables in the binding order, so the natural point is read back.
            let reversed = (self.variable_order == VariableOrder::Suffix)
                .then(|| folding_randomness.reversed());
            evaluation_of_weights += weight_scale
                * (weight.eval)(arith, reversed.as_ref().unwrap_or(&folding_randomness));
        }

        // Final consistency check: claimed_eval == weight * f(r).
        let final_value = match self.variable_order {
            VariableOrder::Prefix => {
                arith.eval_multilinear_ext(&final_poly, &final_sumcheck_randomness)
            }
            VariableOrder::Suffix => {
                arith.eval_multilinear_ext(&final_poly, &final_sumcheck_randomness.reversed())
            }
        };
        let expected = evaluation_of_weights * final_value;
        let (expected_value, actual_value) =
            (arith.value_ext(&expected), arith.value_ext(&claimed_eval));
        arith.assert_zero(claimed_eval - expected, || VerifierError::SumcheckFailed {
            round: self.final_sumcheck_rounds,
            expected: expected_value.to_string(),
            actual: actual_value.to_string(),
        })?;

        Ok(folding_randomness)
    }

    /// Verify STIR in-domain queries and produce associated constraints.
    ///
    /// Checks PoW witness, generates query indices, verifies Merkle proofs,
    /// and evaluates folded polynomials at the queried positions.
    #[allow(clippy::too_many_arguments)]
    fn verify_stir_challenges<A, Ch>(
        &self,
        arith: &mut A,
        proof: &WhirProof<F, EF, MT>,
        transcript: &mut WhirVerifierTranscript<'_, Ch, F, EF>,
        params: &RoundConfig,
        commitment: &<MT as MmcsArith<A, A::F>>::Target,
        folding_randomness: &Point<A::EF>,
        round_index: usize,
    ) -> Result<SelectStatement<A::F, A::EF>, VerifierError>
    where
        A: Lift<Val = F, Challenge = EF>,
        Ch: FieldChallenger<F>
            + GrindingChallenger<Witness = F>
            + CanSampleUniformBits<F>
            + CanObserve<<MT as Mmcs<F>>::Commitment>,
        MT: MmcsArith<
                A,
                A::F,
                Commitment = <MT as Mmcs<F>>::Commitment,
                MultiProof = <MT as Mmcs<F>>::MultiProof,
            >,
        ExtensionMmcs<F, EF, &'a MT>: MmcsArith<
                A,
                A::EF,
                MultiProof = <MT as Mmcs<F>>::MultiProof,
                Target = <MT as MmcsArith<A, A::F>>::Target,
            >,
    {
        // Verify PoW witness before generating challenges.
        let pow_witness = if round_index < self.n_rounds() {
            proof
                .get_pow_after_commitment(round_index)
                .ok_or(VerifierError::InvalidRoundIndex { index: round_index })?
        } else {
            proof.final_pow_witness
        };
        if transcript.shape().query_pow_site(round_index).1 > 0 {
            let witness = arith.load(pow_witness);
            arith.will_observe(vec![witness]);
        }
        transcript.query_pow(round_index, pow_witness)?;

        // Sample STIR query positions.
        let drawn = transcript.query_indices(round_index);
        let indices = lift_query_indices(arith, transcript.shape(), round_index, &drawn);

        let dimensions = vec![Dimensions {
            height: params.domain_size >> params.folding_factor,
            width: 1 << params.folding_factor,
        }];
        let query_randomness = match self.variable_order {
            VariableOrder::Prefix => folding_randomness.clone(),
            VariableOrder::Suffix => folding_randomness.reversed(),
        };
        let folds = self.verify_and_fold_leaves(
            arith,
            proof,
            commitment,
            &indices,
            &dimensions,
            round_index,
            &query_randomness,
        )?;

        let mut statement = SelectStatement::initialize(params.num_variables);
        for (index, evaluation) in indices.iter().zip(folds) {
            match self.domain.query_point_in(
                arith,
                params.log_folded_domain_size,
                params.num_variables,
                index,
            ) {
                WhirQueryPoint::Univariate(var) => {
                    statement.add_constraint(var, evaluation);
                }
                WhirQueryPoint::Multilinear(point) => {
                    statement.add_point_constraint(point, evaluation);
                }
            }
        }
        Ok(statement)
    }

    /// Verify the Merkle multi-opening and fold each opened row at the given point.
    ///
    /// Base-field rows use mixed-field evaluation without materializing a lift.
    #[allow(clippy::too_many_arguments)]
    fn verify_and_fold_leaves<A>(
        &self,
        arith: &mut A,
        proof: &WhirProof<F, EF, MT>,
        root: &<MT as MmcsArith<A, A::F>>::Target,
        indices: &[A::Index],
        dimensions: &[Dimensions],
        round_index: usize,
        randomness: &Point<A::EF>,
    ) -> Result<Vec<A::EF>, VerifierError>
    where
        A: Lift<Val = F, Challenge = EF>,
        MT: MmcsArith<
                A,
                A::F,
                Commitment = <MT as Mmcs<F>>::Commitment,
                MultiProof = <MT as Mmcs<F>>::MultiProof,
            >,
        ExtensionMmcs<F, EF, &'a MT>: MmcsArith<
                A,
                A::EF,
                MultiProof = <MT as Mmcs<F>>::MultiProof,
                Target = <MT as MmcsArith<A, A::F>>::Target,
            >,
    {
        let openings = if round_index == self.n_rounds() {
            &proof.final_openings
        } else {
            // `verify` pins `proof.rounds.len() == n_rounds` at entry and only
            // ever calls this with `round_index < n_rounds`, so this is always in bounds.
            &proof
                .rounds
                .get(round_index)
                .expect("round_index is in bounds: verify() pins proof.rounds.len() == n_rounds")
                .openings
        };

        // Round 0 queries the base-field initial commitment.
        // Every later round queries a folded extension-field commitment.
        // A variant that disagrees with the round is malformed.
        match (openings, round_index == 0) {
            (QueryOpenings::Base(opening), true) => {
                if opening.rows.len() != indices.len() {
                    return Err(VerifierError::StirQueryCountMismatch {
                        round_index,
                        expected: indices.len(),
                        actual: opening.rows.len(),
                    });
                }
                // WHIR commits one matrix per round, so each query opens one row.
                let rows: Vec<Vec<Cow<'_, [A::F]>>> = opening
                    .rows
                    .iter()
                    .map(|row| vec![arith.load_slice(row)])
                    .collect();
                self.mmcs
                    .verify_rows(arith, root, dimensions, indices, &rows, &opening.proof)
                    .map_err(|_| VerifierError::MerkleProofInvalid {
                        position: 0,
                        reason: "Base field Merkle multiproof verification failed".to_string(),
                    })?;
                Ok(rows
                    .iter()
                    .map(|row| arith.eval_multilinear(&row[0], randomness))
                    .collect())
            }
            (QueryOpenings::Extension(opening), false) => {
                if opening.rows.len() != indices.len() {
                    return Err(VerifierError::StirQueryCountMismatch {
                        round_index,
                        expected: indices.len(),
                        actual: opening.rows.len(),
                    });
                }
                let rows: Vec<Vec<Cow<'_, [A::EF]>>> = opening
                    .rows
                    .iter()
                    .map(|row| vec![arith.load_ext_slice(row)])
                    .collect();
                ExtensionMmcs::<F, EF, &'a MT>::new(self.mmcs)
                    .verify_rows(arith, root, dimensions, indices, &rows, &opening.proof)
                    .map_err(|_| VerifierError::MerkleProofInvalid {
                        position: 0,
                        reason: "Extension field Merkle multiproof verification failed".to_string(),
                    })?;
                Ok(rows
                    .iter()
                    .map(|row| arith.eval_multilinear_ext(&row[0], randomness))
                    .collect())
            }
            _ => Err(VerifierError::MerkleProofInvalid {
                position: 0,
                reason: format!("Query openings field does not match round {round_index}"),
            }),
        }
    }
}

impl<EF, F, Dft, MT, Challenger> Deref for WhirVerifier<'_, EF, F, Dft, MT, Challenger>
where
    F: Field,
    EF: ExtensionField<F>,
    MT: Mmcs<F>,
{
    type Target = WhirConfig<EF, F, Challenger>;

    fn deref(&self) -> &Self::Target {
        self.config
    }
}
