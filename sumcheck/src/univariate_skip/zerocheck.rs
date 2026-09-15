//! The whole binary zerocheck, from the committed witness to one opening point.

use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_challenger::fs::TranscriptField;
use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::{ExtensionField, Field};
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;

use super::composition::Composition;
use super::opening::{OPENING_DEGREE, SkipOpening};
use super::round::{MessageLenMismatch, SkipRound};
use super::zerocheck_transcript::{
    ZerocheckProverTranscript, ZerocheckShape, ZerocheckTranscriptError,
    ZerocheckVerifierTranscript,
};
use crate::generic_degree::{GenericDegreeError, GenericDegreeProof, RoundProver};

/// Everything a binary zerocheck sends.
#[derive(Debug, Clone)]
pub struct ZerocheckProof<EF> {
    /// The skip round's polynomial on the transmitted points.
    pub message: Vec<EF>,
    /// The grinding witness guarding the skip challenge, when grinding is enabled.
    pub skip_pow: Option<EF>,
    /// The sumcheck over the variables the skip round did not bind.
    pub residual: GenericDegreeProof<EF, EF>,
    /// Each operand's blended value, as the residual rounds left it.
    ///
    /// The residual rounds pin only their constraint.
    ///
    /// That is one equation in as many unknowns as there are operands.
    ///
    /// The blends cannot be recovered from it, so they are sent.
    pub blends: Vec<EF>,
    /// The sumcheck that turns the skip round's blend into an evaluation point.
    pub opening: GenericDegreeProof<EF, EF>,
}

/// What a verified zerocheck leaves for the commitment to discharge.
#[derive(Debug, Clone)]
pub struct ZerocheckClaim<EF> {
    /// The point every committed polynomial is claimed at.
    ///
    /// The residual variables come first, then the ones the skip round bound.
    pub point: Point<EF>,
    /// The batching challenge the operand claims were combined under.
    pub gamma: EF,
    /// The combined value the operands must open to.
    ///
    /// Opening each operand at the point and recombining under the challenge must match this.
    pub value: EF,
}

impl<EF: Field> ZerocheckClaim<EF> {
    /// Whether the committed openings at this claim's point recombine to its value.
    ///
    /// # Overview
    ///
    /// This is the last step of the chain, and the one a caller holding a commitment runs:
    ///
    /// ```text
    ///     open every operand at the point  ->  recombine under the challenge  ->  compare
    /// ```
    ///
    /// The recombination is the same batching the reduction proved over.
    ///
    /// It is taken from here rather than rebuilt by every caller.
    ///
    /// # Arguments
    ///
    /// The openings in operand order, as the constraint reads them.
    #[must_use]
    pub fn is_answered_by(&self, openings: &[EF]) -> bool {
        SkipOpening::batch_claims(openings, self.gamma) == self.value
    }
}

/// Reasons a binary zerocheck rejects.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ZerocheckError {
    /// The transcript replay refused the proof's shape.
    #[error(transparent)]
    Transcript(#[from] ZerocheckTranscriptError),
    /// The skip round's message is not the width the round transmits.
    #[error(transparent)]
    Message(#[from] MessageLenMismatch),
    /// The residual sumcheck rejected.
    #[error("residual sumcheck: {0}")]
    Residual(GenericDegreeError),
    /// The opening reduction's sumcheck rejected.
    #[error("opening sumcheck: {0}")]
    Opening(GenericDegreeError),
    /// The residual sumcheck claims a sum the skip round's message does not give.
    #[error("the residual claim does not match the round message")]
    ResidualClaimMismatch,
    /// The opening reduction claims a sum the residual rounds did not leave.
    #[error("the opening claim does not match the residual rounds")]
    OpeningClaimMismatch,
    /// The Lagrange weight vanished at the opening point, leaving the claim undetermined.
    ///
    /// The point is drawn after the weight is fixed.
    ///
    /// An honest run therefore reaches this with negligible probability.
    #[error("the Lagrange weight vanished at the opening point")]
    DegenerateOpening,
}

/// A binary zerocheck over a bit-valued witness.
///
/// # Overview
///
/// This owns the whole sequence, so nothing about the order is left to a caller:
///
/// ```text
///     draw the zerocheck point      over the kept variables, after the commitment
///     skip round                     one message, one challenge, k variables bound
///     residual sumcheck              m - k ordinary rounds
///     opening reduction              k degree-two rounds
///     -> one evaluation point
/// ```
///
/// The two obligations the skip round documents are discharged here rather than described.
///
/// - The point is drawn inside, from the transcript, so a prover cannot choose it and a caller
///   cannot draw it before committing.
/// - The run ends on a point a commitment opens, not on the blend the skip round leaves.
///
/// # What the caller still owes
///
/// The witness has to be committed before this runs.
///
/// Its commitment has to be absorbed into the transcript this borrows.
///
/// Discharging the claim at the end against that commitment is the caller's too.
#[derive(Debug, Clone)]
pub struct BinaryZerocheck<F, C> {
    /// The skip round this opens with.
    round: SkipRound<F>,
    /// The constraint being proved vanishing.
    composition: C,
    /// Grinding difficulty guarding each challenge, or zero to omit grinding.
    pow_bits: usize,
}

impl<F: p3_binary_field::TowerLevel, C> BinaryZerocheck<F, C> {
    /// Number of variables the skip round binds in one go.
    #[must_use]
    pub const fn log_skip(&self) -> usize {
        self.round.log_size()
    }

    /// The skip round this opens with.
    #[must_use]
    pub const fn round(&self) -> &SkipRound<F> {
        &self.round
    }
}

impl<F, C> BinaryZerocheck<F, C>
where
    F: p3_binary_field::TowerLevel + Send + Sync,
    C: Composition<F> + Sync,
{
    /// Set up a zerocheck skipping `log_skip` variables of the given constraint.
    ///
    /// # Errors
    ///
    /// Returns an error when no skip round of that shape and degree is realisable.
    pub fn new(
        log_skip: usize,
        composition: C,
        pow_bits: usize,
    ) -> Result<Self, super::round::SkipRoundError> {
        let round = SkipRound::new(log_skip, composition.degree())?;
        Ok(Self {
            round,
            composition,
            pow_bits,
        })
    }

    /// The description both sides seed from, for a witness of the given height.
    #[must_use]
    pub fn shape(&self, log_height: usize) -> ZerocheckShape {
        ZerocheckShape::new(
            log_height,
            self.round.log_size(),
            self.round.domain().log_extended(),
            self.composition.degree() + 1,
            self.composition.arity(),
            self.pow_bits,
        )
    }

    /// Prove that the constraint vanishes on every cell of the packed witness.
    ///
    /// # Arguments
    ///
    /// - `operands`: one packed witness per operand, rows back to back.
    /// - `log_height`: total number of variables, rows and skipped together.
    /// - `challenger`: the transcript, with the commitment already absorbed.
    ///
    /// # Panics
    ///
    /// Panics if the operand count disagrees with the constraint's arity.
    /// Panics if the height is not more than the skip width.
    pub fn prove<EF, Challenger>(
        &self,
        operands: &[&[u8]],
        log_height: usize,
        challenger: &mut Challenger,
    ) -> (ZerocheckProof<EF>, ZerocheckClaim<EF>)
    where
        EF: ExtensionField<F> + TranscriptField + Send + Sync,
        Challenger: FieldChallenger<EF> + GrindingChallenger<Witness = EF>,
    {
        assert_eq!(
            operands.len(),
            self.composition.arity(),
            "one packed witness per operand"
        );
        assert!(
            log_height > self.round.log_size(),
            "the skip round must leave at least one residual variable"
        );

        let log_rows = log_height - self.round.log_size();
        let shape = self.shape(log_height);
        let mut transcript =
            ZerocheckProverTranscript::<Challenger, EF, EF>::new(challenger, shape);

        // The point is drawn here, after the commitment, so the prover never chooses it.
        let zerocheck_point = transcript.zerocheck_point(log_rows);
        let eq = Poly::new_from_point(zerocheck_point.as_slice(), EF::ONE);

        // The message streams out of the packed witness, one row's scratch at a time.
        let message =
            self.round
                .stream_round_message::<EF, _>(operands, eq.as_slice(), &self.composition);
        let (lambda, skip_pow) = transcript.skip_round(&message);

        // Reading the message back at the challenge gives the residual claim.
        let residual_claim = self
            .round
            .evaluate(&message, lambda)
            .expect("the prover built the message at this round's width");

        // The residual rounds fold the rows read at that challenge.
        let selector = self.round.selector::<EF>(lambda);
        let bound = operands
            .iter()
            .map(|rows| selector.bind(rows))
            .collect::<Vec<_>>();
        let mut residual_prover = ResidualProver::<F, EF, C> {
            eq,
            operands: bound,
            composition: &self.composition,
            _f: PhantomData,
        };
        let (residual, rho) = transcript.residual_sumcheck(|challenger| {
            residual_prover.prove::<EF, _>(
                challenger,
                log_rows,
                self.composition.degree() + 1,
                self.pow_bits,
                residual_claim,
            )
        });

        // The residual rounds end on each operand's blended value.
        //
        // The final value pins only their constraint.
        //
        // One equation in three unknowns, so the blends themselves cross the wire.
        let blends = residual_prover
            .operands
            .iter()
            .map(|poly| poly.as_slice()[0])
            .collect::<Vec<_>>();
        transcript.operand_blends(&blends);

        // The opening reduction collapses those blends to one evaluation point.
        let opening = SkipOpening::new(selector.lagrange().clone());
        let folded = opening.partial_evaluations(operands.iter().copied(), &rho);
        let gamma = transcript.opening_batching();
        let opening_claim = SkipOpening::batch_claims(&blends, gamma);

        let mut opening_prover = opening.prover(SkipOpening::batch(&folded, gamma));
        let (opening_proof, tau) = transcript.opening_sumcheck(|challenger| {
            opening_prover.prove::<EF, _>(
                challenger,
                self.round.log_size(),
                OPENING_DEGREE,
                self.pow_bits,
                opening_claim,
            )
        });
        transcript.finish();

        // The residual variables come first, then the ones the skip round bound.
        let point = Point::new(
            rho.as_slice()
                .iter()
                .copied()
                .chain(tau.as_slice().iter().copied())
                .collect(),
        );

        (
            ZerocheckProof {
                message,
                skip_pow,
                residual,
                blends,
                opening: opening_proof,
            },
            ZerocheckClaim {
                point,
                gamma,
                value: opening_prover.surviving_claim(),
            },
        )
    }

    /// Replay a proof and return the claim a commitment has to discharge.
    ///
    /// # Errors
    ///
    /// Returns an error when any replayed step or algebraic check fails.
    ///
    /// # Panics
    ///
    /// Panics if the height is not more than the skip width.
    pub fn verify<EF, Challenger>(
        &self,
        proof: &ZerocheckProof<EF>,
        log_height: usize,
        challenger: &mut Challenger,
    ) -> Result<ZerocheckClaim<EF>, ZerocheckError>
    where
        EF: ExtensionField<F> + TranscriptField,
        Challenger: FieldChallenger<EF> + GrindingChallenger<Witness = EF>,
    {
        assert!(
            log_height > self.round.log_size(),
            "the skip round must leave at least one residual variable"
        );

        let log_rows = log_height - self.round.log_size();
        let shape = self.shape(log_height);
        let mut transcript =
            ZerocheckVerifierTranscript::<Challenger, EF, EF>::new(challenger, shape);

        // The point is drawn, never read from the proof.
        let zerocheck_point = transcript.zerocheck_point(log_rows);

        let lambda = transcript.skip_round(&proof.message, proof.skip_pow)?;

        // The round polynomial is defined to vanish on the subspace.
        //
        // Off it, it matches the message.
        //
        // Reading it at the challenge therefore needs nothing further from the prover.
        let residual_claim = match self.round.evaluate(&proof.message, lambda) {
            Ok(claim) => claim,
            Err(error) => {
                transcript.abort();
                return Err(error.into());
            }
        };
        if proof.residual.claimed_sum != residual_claim {
            transcript.abort();
            return Err(ZerocheckError::ResidualClaimMismatch);
        }

        let (rho, residual_final) = transcript
            .residual_sumcheck(|challenger| {
                proof.residual.verify(
                    challenger,
                    log_rows,
                    self.composition.degree() + 1,
                    self.pow_bits,
                )
            })
            .map_err(ZerocheckError::Residual)?;

        // The residual rounds end on the equality weight times the constraint of the blends.
        //
        // The verifier knows the weight and the constraint.
        //
        // The blends the proof carries are therefore checked here, not trusted.
        if let Err(error) = transcript.operand_blends(&proof.blends) {
            return Err(error.into());
        }
        let eq_at_rho = Poly::new_from_point(zerocheck_point.as_slice(), EF::ONE).eval_base(&rho);
        if residual_final != eq_at_rho * self.composition.eval(&proof.blends) {
            transcript.abort();
            return Err(ZerocheckError::ResidualClaimMismatch);
        }

        // Those blends are what the opening reduction has to start from.
        let gamma = transcript.opening_batching();
        if proof.opening.claimed_sum != SkipOpening::batch_claims(&proof.blends, gamma) {
            transcript.abort();
            return Err(ZerocheckError::OpeningClaimMismatch);
        }

        let (tau, opening_final) = transcript
            .opening_sumcheck(|challenger| {
                proof.opening.verify(
                    challenger,
                    self.round.log_size(),
                    OPENING_DEGREE,
                    self.pow_bits,
                )
            })
            .map_err(ZerocheckError::Opening)?;
        transcript.finish();

        // The reduction's final value still carries the Lagrange weight.
        //
        // The verifier reads that for itself and divides it out.
        //
        // Only the vector is needed here.
        //
        // The round hands that over without the byte table its own row reading uses.
        let opening = SkipOpening::new(self.round.lagrange::<EF>(lambda));
        let weight = opening.lagrange_at(&tau);
        if weight.is_zero() {
            return Err(ZerocheckError::DegenerateOpening);
        }

        let point = Point::new(
            rho.as_slice()
                .iter()
                .copied()
                .chain(tau.as_slice().iter().copied())
                .collect(),
        );

        Ok(ZerocheckClaim {
            point,
            gamma,
            value: opening_final * weight.inverse(),
        })
    }
}

/// Prover state of the residual rounds.
struct ResidualProver<'a, F, EF, C> {
    /// The zerocheck's equality weight over the rows.
    eq: Poly<EF>,
    /// The operands, read at the skip round's challenge.
    operands: Vec<Poly<EF>>,
    /// The constraint being proved vanishing.
    composition: &'a C,
    /// Marker for the witness alphabet the constraint is stated over.
    ///
    /// The rounds read it in the large field.
    ///
    /// The constraint itself belongs to the alphabet the witness was committed in.
    _f: PhantomData<F>,
}

impl<F, EF, C> RoundProver<EF> for ResidualProver<'_, F, EF, C>
where
    F: Field,
    EF: ExtensionField<F>,
    C: Composition<F>,
{
    fn fold(&mut self, r: EF) {
        // Binding a variable halves every factor in step.
        self.eq.fix_prefix_var_mut(r);
        for operand in &mut self.operands {
            operand.fix_prefix_var_mut(r);
        }
    }

    fn round_poly(&self) -> Vec<EF> {
        // The equality weight raises the constraint's degree by one.
        let degree = self.composition.degree() + 1;
        let half = self.eq.num_evals() / 2;
        let mut tuple = alloc::vec![EF::ZERO; self.operands.len()];

        // Nodes zero and two upward, the value at one being recoverable from the claim.
        core::iter::once(0)
            .chain(2..=degree)
            .map(|node| {
                let node = EF::interpolation_node(node);
                let mut total = EF::ZERO;
                for index in 0..half {
                    let at = |poly: &Poly<EF>| {
                        let values = poly.as_slice();
                        values[index] + (values[index + half] - values[index]) * node
                    };
                    for (slot, operand) in tuple.iter_mut().zip(&self.operands) {
                        *slot = at(operand);
                    }
                    total += at(&self.eq) * self.composition.eval(&tuple);
                }
                total
            })
            .collect()
    }
}
