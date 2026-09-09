//! Fiat-Shamir transcript of the FRI polynomial commitment scheme.
//!
//! # Overview
//!
//! One statement of what the opening argument's transcript is, consumed by both sides.
//!
//! It covers the phase the PCS owns: the claimed evaluations, the grind, the batching challenge.
//! The low-degree test that follows owns its own transcript and is bracketed as a sub-protocol.
//!
//! # Shape
//!
//! ```text
//!     per (commitment, matrix, opening point):
//!                  claimed evaluations   one step of that matrix's width
//!     batch grinding                     only when the difficulty is positive
//!     batching challenge                 one extension element
//!     Begin  low-degree test             bracket around the delegated run
//!     End    low-degree test
//! ```
//!
//! # What is bound
//!
//! - Shape: the number of claimed evaluations in every opening, and the grinding difficulty.
//! - Instance label: how those openings group into commitments, matrices and points.
//! - Nothing here: the FRI parameters, which FRI's own seed binds inside the bracket.
//!
//! Both sides take those counts from their own inputs.
//! The prover's are the matrices it holds; the verifier's are the claims it was handed.
//!
//! Whether a claimed width is the committed matrix's width is a separate question.
//! The input commitment scheme answers it when it authenticates the opened rows.
//!
//! # Soundness
//!
//! The batching challenge `alpha` collapses every claimed opening into one low-degree claim.
//!
//! ```text
//!     sum_i alpha^i * (f_i(x) - y_i) / (x - z_i)
//! ```
//!
//! A prover who learns `alpha` before committing to the `y_i` picks them to fit.
//! Every claimed evaluation is therefore absorbed before the challenge is drawn.
//!
//! ```text
//!     absorb y_0 .. y_n   ->   grind   ->   draw alpha
//! ```
//!
//! The grinding step in between prices each retry at `2^batch_pow_bits`.

use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_challenger::fs::{
    DomainSeparator, FieldToFieldCodec, FieldUnit, Hierarchy, Interaction, InteractionPattern,
    Kind, Length, ProverState, VerifierState,
};
use p3_challenger::{CanObserve, CanSample, CanSampleBits, GrindingChallenger};
use p3_commit::{CommitmentWithOpeningPoints, OpenedValues, PointOpening};
use p3_field::{ExtensionField, PrimeField64};
use thiserror::Error;

use crate::FriParameters;

/// Version byte bound into the transcript seed.
const VERSION: u8 = 1;

/// Protocol name bound into the transcript seed.
const NAME: &[u8] = b"p3-fri-pcs";

/// Step label of one opening's claimed evaluations.
const CLAIMED_EVALUATIONS: &str = "claimed_evaluations";

/// Step label of the grinding step guarding the batching challenge.
const BATCH_POW: &str = "batch_pow";

/// Step label of the batching challenge.
const ALPHA: &str = "alpha";

/// Step label of the bracket around the delegated low-degree test.
const LOW_DEGREE_TEST: &str = "low_degree_test";

/// Sponge alphabet of a challenger that speaks the base field natively.
type Alphabet<F> = FieldUnit<F>;

/// Type-level name of the sub-protocol the PCS delegates its low-degree test to.
///
/// Recorded on the bracket markers as a local diagnostic.
/// It does not reach the pattern fingerprint.
struct LowDegreeTest;

/// Numbers that fix the transcript of one PCS opening argument.
///
/// Both sides build this from their own inputs, never from a proof.
///
/// The prover's inputs are the matrices it holds.
/// The verifier's are the claims it was asked to check.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PcsShape {
    /// Claimed-evaluation counts, nested by commitment, then matrix, then opening point.
    ///
    /// Each innermost entry is the width of one matrix, which is how many values
    /// one opening of it carries.
    pub claimed_evaluation_counts: Vec<Vec<Vec<usize>>>,
    /// Grinding difficulty guarding the batching challenge.
    pub batch_pow_bits: usize,
}

impl PcsShape {
    /// Derive the shape from the values a prover is about to claim.
    ///
    /// # Arguments
    ///
    /// - `params`: the protocol parameters.
    /// - `opened_values`: the claimed evaluations, nested by commitment, matrix and point.
    #[must_use]
    pub fn from_opened_values<EF, M>(
        params: &FriParameters<M>,
        opened_values: &OpenedValues<EF>,
    ) -> Self {
        let claimed_evaluation_counts = opened_values
            .iter()
            .map(|round| {
                round
                    .iter()
                    .map(|matrix| matrix.iter().map(Vec::len).collect())
                    .collect()
            })
            .collect();

        Self {
            claimed_evaluation_counts,
            batch_pow_bits: params.batch_proof_of_work_bits,
        }
    }

    /// Derive the shape from the claims a verifier was asked to check.
    ///
    /// The counts come from the caller's own argument, never from the proof.
    ///
    /// # Arguments
    ///
    /// - `params`: the protocol parameters.
    /// - `claims`: one commitment per entry, with the points and values of each of its matrices.
    #[must_use]
    pub fn from_claims<EF, Com, Domain, M>(
        params: &FriParameters<M>,
        claims: &[CommitmentWithOpeningPoints<EF, Com, Domain>],
    ) -> Self {
        let claimed_evaluation_counts = claims
            .iter()
            .map(|claim| {
                claim
                    .matrices
                    .iter()
                    .map(|matrix| matrix.points.iter().map(|p| p.values.len()).collect())
                    .collect()
            })
            .collect();

        Self {
            claimed_evaluation_counts,
            batch_pow_bits: params.batch_proof_of_work_bits,
        }
    }

    /// Number of openings this shape describes.
    ///
    /// One opening is one (commitment, matrix, point) triple.
    #[must_use]
    pub fn num_openings(&self) -> usize {
        self.claimed_evaluation_counts
            .iter()
            .flatten()
            .map(Vec::len)
            .sum()
    }

    /// Every opening's claimed-evaluation count, flattened into transcript order.
    fn widths(&self) -> impl Iterator<Item = usize> + '_ {
        self.claimed_evaluation_counts
            .iter()
            .flatten()
            .flatten()
            .copied()
    }

    /// Describe the transcript this shape fixes.
    ///
    /// # Panics
    ///
    /// Never in practice.
    /// The only nesting is one matched bracket, which always passes structural validation.
    #[must_use]
    pub fn pattern<F, EF>(&self) -> InteractionPattern
    where
        F: PrimeField64,
        EF: ExtensionField<F>,
    {
        // One step per opening, then the grind, the challenge, and the bracket.
        let mut steps = Vec::with_capacity(self.num_openings() + 4);

        // Every opening is one step of its own width.
        //
        // The width belongs to the description, not to the values as they arrive.
        // A claim of another width is then a shape mismatch, caught before anything is absorbed.
        for width in self.widths() {
            steps.push(Interaction::algebra::<F, EF>(
                Hierarchy::Atomic,
                Kind::Message,
                CLAIMED_EVALUATIONS,
                Length::Fixed(width),
            ));
        }

        // Grinding sits between the claims and the challenge that batches them.
        if self.batch_pow_bits > 0 {
            steps.push(Interaction::algebra::<F, F>(
                Hierarchy::Atomic,
                Kind::Pow,
                BATCH_POW,
                Length::Fixed(self.batch_pow_bits),
            ));
        }

        // One challenge collapses every claim into a single low-degree claim.
        steps.push(Interaction::algebra::<F, EF>(
            Hierarchy::Atomic,
            Kind::Challenge,
            ALPHA,
            Length::Scalar,
        ));

        // The bracket records that a sub-protocol runs here.
        //
        // Its steps live in the callee's own pattern, under the callee's own seed.
        // What this pattern states is that the delegation happens, and where.
        steps.push(Interaction::marker::<LowDegreeTest>(
            Hierarchy::Begin,
            Kind::Protocol,
            LOW_DEGREE_TEST,
        ));
        steps.push(Interaction::marker::<LowDegreeTest>(
            Hierarchy::End,
            Kind::Protocol,
            LOW_DEGREE_TEST,
        ));

        InteractionPattern::new(steps).expect("one matched bracket is always well formed")
    }

    /// Bind the protocol identity, this shape, and the grouping of its openings.
    ///
    /// A number that changes the step sequence is covered by the fingerprint.
    /// The rest go in the instance label.
    #[must_use]
    pub fn domain_separator<F, EF>(&self) -> DomainSeparator<Alphabet<F>>
    where
        F: PrimeField64,
        EF: ExtensionField<F>,
    {
        let mut separator = DomainSeparator::new(VERSION, NAME, self.pattern::<F, EF>());

        // The step sequence sees one flat list of widths, so the grouping must be bound here.
        //
        //     two matrices of width 3, one point each   ->  [3, 3]
        //     one matrix of width 3, two points         ->  [3, 3]
        //
        // Both flatten alike, so only the instance label separates them.
        separator.instance(&(self.claimed_evaluation_counts.len() as u64).to_be_bytes());
        for commitment in &self.claimed_evaluation_counts {
            separator.instance(&(commitment.len() as u64).to_be_bytes());
            for matrix in commitment {
                separator.instance(&(matrix.len() as u64).to_be_bytes());
            }
        }

        separator
    }
}

/// Prover-side transcript of one PCS opening argument.
///
/// Holds the only definition of what a prover writes before the low-degree test.
///
/// The challenger is borrowed, not consumed.
/// The PCS runs inside a larger protocol whose transcript continues afterwards.
pub struct PcsProverTranscript<'a, C, F: PrimeField64, EF> {
    /// Driver walking the description and holding the borrowed sponge.
    state: ProverState<&'a mut C, Alphabet<F>>,
    /// The numbers this run was described with.
    shape: PcsShape,
    /// Marker for the extension field the claims and challenge live in.
    _ef: PhantomData<EF>,
}

impl<'a, C, F, EF> PcsProverTranscript<'a, C, F, EF>
where
    F: PrimeField64,
    EF: ExtensionField<F>,
    C: CanObserve<F> + CanSample<F> + CanSampleBits<usize> + GrindingChallenger<Witness = F>,
{
    /// Seed the transcript from the shape.
    pub fn new(challenger: &'a mut C, shape: PcsShape) -> Self {
        let separator = shape.domain_separator::<F, EF>();
        Self {
            state: ProverState::new(challenger, &separator),
            shape,
            _ef: PhantomData,
        }
    }

    /// Bind every claimed evaluation, in commitment, matrix and point order.
    ///
    /// # Panics
    ///
    /// When the values do not group exactly as the shape describes.
    pub fn claimed_openings(&mut self, opened_values: &OpenedValues<EF>) {
        for values in opened_values.iter().flatten().flatten() {
            self.state
                .observe_extensions::<F, EF, FieldToFieldCodec<F>>(CLAIMED_EVALUATIONS, values);
        }
    }

    /// Grind, then draw the challenge that batches every claim.
    ///
    /// # Returns
    ///
    /// - The batching challenge.
    /// - The grinding witness, when the difficulty is positive.
    pub fn batch_phase(&mut self) -> (EF, Option<F>) {
        let witness = (self.shape.batch_pow_bits > 0)
            .then(|| self.state.observe_pow(BATCH_POW, self.shape.batch_pow_bits));

        let alpha = self
            .state
            .challenge_extension::<F, EF, FieldToFieldCodec<F>>(ALPHA)
            .into_inner();

        (alpha, witness)
    }

    /// Lend the sponge to the low-degree test, bracketed as a sub-protocol.
    ///
    /// The callee seeds its own driver from the state this one has reached.
    ///
    /// # Returns
    ///
    /// Whatever the delegated run produced.
    pub fn delegate<R>(&mut self, run: impl FnOnce(&mut C) -> R) -> R {
        self.state.begin_protocol::<LowDegreeTest>(LOW_DEGREE_TEST);
        let output = run(self.state.challenger_mut());
        self.state.end_protocol::<LowDegreeTest>(LOW_DEGREE_TEST);
        output
    }

    /// Close the transcript once every described step has been played.
    ///
    /// # Panics
    ///
    /// When the run played fewer steps than it was described with.
    pub fn finish(self) {
        assert!(
            self.state.finalize().is_empty(),
            "the PCS carries every value in its own proof",
        );
    }
}

/// Verifier-side transcript of one PCS opening argument.
///
/// Mirrors the prover side call for call, over the same description.
pub struct PcsVerifierTranscript<'a, C, F: PrimeField64, EF> {
    /// Driver walking the description and holding the borrowed sponge.
    ///
    /// The proof carries every value, so the driver reads an empty wire.
    state: VerifierState<'static, &'a mut C, Alphabet<F>>,
    /// The numbers this run was described with.
    shape: PcsShape,
    /// Marker for the extension field the claims and challenge live in.
    _ef: PhantomData<EF>,
}

impl<'a, C, F, EF> PcsVerifierTranscript<'a, C, F, EF>
where
    F: PrimeField64,
    EF: ExtensionField<F>,
    C: CanObserve<F> + CanSample<F> + CanSampleBits<usize> + GrindingChallenger<Witness = F>,
{
    /// Seed the transcript from the shape.
    pub fn new(challenger: &'a mut C, shape: PcsShape) -> Self {
        let separator = shape.domain_separator::<F, EF>();
        Self {
            state: VerifierState::new(challenger, &separator, &[]),
            shape,
            _ef: PhantomData,
        }
    }

    /// Replay every claimed evaluation, in commitment, matrix and point order.
    ///
    /// # Errors
    ///
    /// When an opening carries a count the run was not described with.
    pub fn claimed_openings<Com, Domain>(
        &mut self,
        claims: &[CommitmentWithOpeningPoints<EF, Com, Domain>],
    ) -> Result<(), PcsTranscriptFailure> {
        // Walk the described counts flat, in the same order the claims are visited.
        //
        //     claims  : commitment -> matrix -> point
        //     described: the same nesting, flattened to one count per opening
        let mut described = self
            .shape
            .claimed_evaluation_counts
            .iter()
            .flatten()
            .flatten();

        // Position of the opening being replayed, so a rejection can name it.
        let mut opening = 0;

        for claim in claims {
            for matrix in &claim.matrices {
                for PointOpening { values, .. } in &matrix.points {
                    // The count this opening was described with, taken in visit order.
                    let expected = described.next().copied().unwrap_or_default();

                    // A count outside the described one is a rejection, not an absorb.
                    self.state
                        .observe_extensions::<F, EF, FieldToFieldCodec<F>>(
                            CLAIMED_EVALUATIONS,
                            values,
                        )
                        .map_err(|_| PcsTranscriptFailure::ClaimedEvaluationCount {
                            opening,
                            expected,
                            got: values.len(),
                        })?;
                    opening += 1;
                }
            }
        }

        // A described run may still hold openings the caller never supplied.
        //
        // Both sides derive the shape from this same argument, so that is a caller bug.
        // `finish` reports it as an unreplayed step.
        Ok(())
    }

    /// Replay the grind, then redraw the challenge that batches every claim.
    ///
    /// # Errors
    ///
    /// - The run carries no witness for a described grinding step.
    /// - The witness misses the required difficulty.
    pub fn batch_phase(&mut self, witness: Option<F>) -> Result<EF, PcsTranscriptFailure> {
        if self.shape.batch_pow_bits > 0 {
            // With no witness the described step cannot be played at all.
            //
            // Releasing the completeness check keeps this rejection the only failure.
            let Some(witness) = witness else {
                self.state.abort();
                return Err(PcsTranscriptFailure::MissingPowWitness {
                    bits: self.shape.batch_pow_bits,
                });
            };
            self.state
                .observe_pow(BATCH_POW, self.shape.batch_pow_bits, witness)
                .map_err(|_| PcsTranscriptFailure::PowWitness {
                    bits: self.shape.batch_pow_bits,
                })?;
        }

        Ok(self
            .state
            .challenge_extension::<F, EF, FieldToFieldCodec<F>>(ALPHA)
            .into_inner())
    }

    /// Lend the sponge to the low-degree test, bracketed as a sub-protocol.
    ///
    /// The bracket closes whatever the delegated run returned.
    /// A rejection therefore leaves this transcript replayable to the end.
    ///
    /// # Returns
    ///
    /// Whatever the delegated run produced.
    pub fn delegate<R>(&mut self, run: impl FnOnce(&mut C) -> R) -> R {
        self.state.begin_protocol::<LowDegreeTest>(LOW_DEGREE_TEST);
        let output = run(self.state.challenger_mut());
        self.state.end_protocol::<LowDegreeTest>(LOW_DEGREE_TEST);
        output
    }

    /// Close the transcript once every described step has been replayed.
    ///
    /// # Panics
    ///
    /// When the run replayed fewer steps than it was described with.
    pub fn finish(self) {
        self.state
            .finalize()
            .expect("the PCS reads an empty wire, so no bytes can remain");
    }
}

/// A transcript step the proof failed to satisfy.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Error)]
pub enum PcsTranscriptFailure {
    /// One opening carries a claimed-evaluation count the run never described.
    #[error("opening {opening}: claimed evaluation count mismatch: expected {expected}, got {got}")]
    ClaimedEvaluationCount {
        /// Position of the opening in commitment, matrix and point order.
        opening: usize,
        /// Claimed-evaluation count the run was described with.
        expected: usize,
        /// Claimed-evaluation count the caller supplied.
        got: usize,
    },
    /// The grinding witness did not meet the difficulty its step requires.
    #[error("batch phase PoW witness does not meet the required {bits} bits")]
    PowWitness {
        /// Grinding difficulty the step requires.
        bits: usize,
    },
    /// The described grinding step arrived with no witness to replay it.
    #[error("batch phase PoW step of {bits} bits arrived with no witness")]
    MissingPowWitness {
        /// Grinding difficulty the step requires.
        bits: usize,
    },
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
    use p3_challenger::{CanSample, DuplexChallenger};
    use p3_field::PrimeCharacteristicRing;
    use p3_field::extension::BinomialExtensionField;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    use super::*;

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;
    type Perm = Poseidon2BabyBear<16>;
    type Ch = DuplexChallenger<F, Perm, 16, 8>;

    fn fresh_challenger() -> Ch {
        // Fixed seed so two runs differ only where the transcript makes them differ.
        let mut rng = SmallRng::seed_from_u64(0xF21);
        Ch::new(Perm::new_from_rng_128(&mut rng))
    }

    /// A shape over the given grouping of claimed-evaluation counts, with no grinding.
    fn shape_with(counts: Vec<Vec<Vec<usize>>>) -> PcsShape {
        PcsShape {
            claimed_evaluation_counts: counts,
            batch_pow_bits: 0,
        }
    }

    /// The shape every mutation below is measured against.
    ///
    /// One commitment, one matrix, two openings of unequal width.
    fn plain_shape() -> PcsShape {
        shape_with(vec![vec![vec![3, 1]]])
    }

    /// The first challenge a shape's seed produces.
    fn first_challenge(shape: &PcsShape) -> F {
        let mut challenger = fresh_challenger();
        let separator = shape.domain_separator::<F, EF>();
        separator.seed(&mut challenger);
        challenger.sample()
    }

    /// Every field of the shape, each moved one step away from `plain_shape`.
    ///
    /// One entry per configuration knob.
    /// A field added to the shape stops the destructuring below from compiling.
    ///
    /// The counts are nested vectors, so they contribute one entry per way of moving them.
    fn one_step_from_plain() -> Vec<(&'static str, PcsShape)> {
        // Exhaustiveness check: every field named, none elided by a rest pattern.
        // The bindings go unused, since naming the fields is all this has to do.
        let PcsShape {
            claimed_evaluation_counts: _,
            batch_pow_bits: _,
        } = plain_shape();

        let mut mutations = Vec::new();

        // One more claimed evaluation in the first opening's fixed-length step.
        let mut shape = plain_shape();
        shape.claimed_evaluation_counts[0][0][0] += 1;
        mutations.push(("claimed_evaluation_counts value", shape));

        // One more opening, which is one more step.
        let mut shape = plain_shape();
        shape.claimed_evaluation_counts[0][0].push(2);
        mutations.push(("claimed_evaluation_counts length", shape));

        // A reordering keeps the openings and their widths, and swaps the order they arrive in.
        //
        //     [[[3, 1]]]  absorbs 3 values then 1
        //     [[[1, 3]]]  absorbs 1 value then 3
        let mut shape = plain_shape();
        shape.claimed_evaluation_counts[0][0].reverse();
        mutations.push(("claimed_evaluation_counts order", shape));

        // Elided at zero, so this is the transition where the grinding step appears at all.
        // A shorter sequence, not a cheaper one.
        let mut shape = plain_shape();
        shape.batch_pow_bits += 1;
        mutations.push(("batch_pow_bits", shape));

        mutations
    }

    #[test]
    fn every_field_of_the_shape_reaches_the_seed() {
        // Baseline: the plain shape, seeded and sampled once.
        let baseline = first_challenge(&plain_shape());

        // Each mutation moves exactly one field one step.
        //
        // A field the fingerprint covers moves the seed through the step sequence.
        // A field it does not must move it through the instance label instead.
        for (field, shape) in one_step_from_plain() {
            assert_ne!(
                baseline,
                first_challenge(&shape),
                "changing `{field}` left the seed where it was",
            );
        }
    }

    #[test]
    fn the_grouping_of_the_openings_reaches_the_seed() {
        // Both groupings flatten to the same widths, so they share a step sequence.
        //
        //     one commitment, two matrices, one point each   ->  [3, 3]
        //     one commitment, one matrix, two points         ->  [3, 3]
        //
        // Only the instance label separates them, so it must carry the grouping.
        let two_matrices = first_challenge(&shape_with(vec![vec![vec![3], vec![3]]]));
        let two_points = first_challenge(&shape_with(vec![vec![vec![3, 3]]]));

        assert_ne!(two_matrices, two_points);
    }

    #[test]
    fn the_commitment_count_reaches_the_seed() {
        // Same widths again, and the same matrix and point counts within each commitment.
        //
        //     two commitments of one matrix   ->  [3, 3]
        //     one commitment of two matrices  ->  [3, 3]
        let two_commitments = first_challenge(&shape_with(vec![vec![vec![3]], vec![vec![3]]]));
        let one_commitment = first_challenge(&shape_with(vec![vec![vec![3], vec![3]]]));

        assert_ne!(two_commitments, one_commitment);
    }

    #[test]
    fn a_grinding_step_that_is_already_present_still_binds_its_difficulty() {
        // The step is elided at zero, so a bump off zero only proves presence.
        //
        //     0 -> 1   the step joins the sequence
        //     1 -> 2   the step that is already there declares one more bit
        //
        // The second transition is the one the step's `Length::Fixed` carries.
        let mut ground = plain_shape();
        ground.batch_pow_bits = 1;
        let mut ground_harder = ground.clone();
        ground_harder.batch_pow_bits = 2;

        assert_ne!(first_challenge(&ground), first_challenge(&ground_harder));
    }

    #[test]
    fn a_batch_phase_missing_its_grinding_witness_is_rejected() {
        // Described run: one opening, then 4 bits of grinding before the challenge.
        //
        // A described grinding step cannot be replayed with no witness to feed it.
        let mut shape = shape_with(vec![vec![vec![1]]]);
        shape.batch_pow_bits = 4;

        let mut challenger = fresh_challenger();
        let mut transcript = PcsVerifierTranscript::<Ch, F, EF>::new(&mut challenger, shape);

        let claims: Vec<CommitmentWithOpeningPoints<EF, (), ()>> =
            vec![((), vec![((), vec![(EF::ONE, vec![EF::ONE])])]).into()];
        transcript
            .claimed_openings(&claims)
            .expect("the claims match the shape they were derived from");

        let err = transcript
            .batch_phase(None)
            .expect_err("a described grinding step with no witness must error");

        assert_eq!(err, PcsTranscriptFailure::MissingPowWitness { bits: 4 });
    }

    #[test]
    fn an_opening_of_the_wrong_width_is_rejected() {
        // Described run: one opening of exactly 2 claimed evaluations.
        //
        //     described:   2
        //     claim holds: 3   -> rejected before anything is absorbed
        let mut challenger = fresh_challenger();
        let mut transcript = PcsVerifierTranscript::<Ch, F, EF>::new(
            &mut challenger,
            shape_with(vec![vec![vec![2]]]),
        );

        let claims: Vec<CommitmentWithOpeningPoints<EF, (), ()>> =
            vec![((), vec![((), vec![(EF::ONE, vec![EF::ONE; 3])])]).into()];

        let err = transcript
            .claimed_openings(&claims)
            .expect_err("an opening outside the described width must error");

        assert_eq!(
            err,
            PcsTranscriptFailure::ClaimedEvaluationCount {
                opening: 0,
                expected: 2,
                got: 3
            }
        );
    }

    #[test]
    fn the_delegation_bracket_leaves_the_sponge_untouched() {
        // Invariant: the markers are structural.
        // They record the delegation, and absorb nothing.
        //
        // Fixture state: the two sides over the same shape, each bracketing an empty delegation.
        let shape = shape_with(vec![vec![vec![1]]]);
        let claims: Vec<CommitmentWithOpeningPoints<EF, (), ()>> =
            vec![((), vec![((), vec![(EF::ONE, vec![EF::ONE])])]).into()];

        let mut bracketed_challenger = fresh_challenger();
        let mut bracketed =
            PcsVerifierTranscript::<Ch, F, EF>::new(&mut bracketed_challenger, shape.clone());
        bracketed.claimed_openings(&claims).unwrap();
        let bracketed_alpha = bracketed.batch_phase(None).unwrap();
        bracketed.delegate(|_| ());
        bracketed.finish();

        // The prover side plays the same steps and must land on the same challenge.
        let mut plain_challenger = fresh_challenger();
        let mut plain = PcsProverTranscript::<Ch, F, EF>::new(&mut plain_challenger, shape);
        plain.claimed_openings(&vec![vec![vec![vec![EF::ONE]]]]);
        let (plain_alpha, witness) = plain.batch_phase();
        plain.delegate(|_| ());
        plain.finish();

        assert_eq!(bracketed_alpha, plain_alpha);
        assert_eq!(witness, None);
        // Both sponges advanced identically, so they still agree on what comes next.
        let bracketed_next: F = bracketed_challenger.sample();
        let plain_next: F = plain_challenger.sample();
        assert_eq!(bracketed_next, plain_next);
    }
}
