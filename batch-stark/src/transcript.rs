//! Fiat-Shamir transcript of the batch-STARK protocol.
//!
//! # Overview
//!
//! One statement of what a batch run's transcript is, consumed by both sides.
//!
//! It is built from the AIRs, the keygen-cached common data, and the configuration.
//!
//! All three are fixed before a run starts.
//! Neither side therefore reads the shape of the run from a proof.
//!
//! # Shape
//!
//! ```text
//!     per instance:  degree bits          one extension element
//!     main commitment                     one opaque value
//!     per instance:  public values        that AIR's public-value count
//!     preprocessed commitment             only when the batch has preprocessed columns
//!     when the batch declares a lookup:
//!         lookup grinding                 at the configured difficulty
//!         lookup alpha                    one extension element
//!         lookup beta                     one extension element
//!         permutation commitment          one opaque value
//!         lookup terminals                one per instance that declares a lookup
//!     constraint challenge                one extension element
//!     quotient commitment                 one opaque value
//!     randomization commitment            only when the PCS hides
//!     out-of-domain grinding              at the configured difficulty
//!     zeta                                one extension element
//!     Begin  opening argument             bracket around the delegated run
//!     End    opening argument
//! ```
//!
//! # What is bound
//!
//! - Shape: the instance count, and every instance's public-value count.
//! - Shape: both grinding difficulties, as the declared length of their steps.
//! - Shape: whether the batch has preprocessed columns, lookups, or randomization.
//! - Instance label: the main-trace width and the preprocessed width of every instance.
//! - Nothing here: the parameters of the opening argument, bound inside its own bracket.
//!
//! A commitment's width is not bound either, because this layer cannot see it.
//!
//! The opening check that recomputes such a commitment is what rejects a wrong one.
//!
//! # Soundness
//!
//! Two sampling sites decide the rest of the run, and a grind guards each one.
//!
//! ```text
//!     absorb main commitment, public values  ->  grind  ->  draw (alpha, beta)
//!     absorb every commitment and terminal   ->  grind  ->  draw zeta
//! ```
//!
//! A prover who learns `(alpha, beta)` first fits the trace to balance a lookup it never ran.
//!
//! A prover who learns `zeta` first fits the quotient to agree at that one point.
//!
//! Each grind prices one retry of such a search at `2^bits`.
//!
//! A grind of zero bits prices nothing, and absorbs nothing either.
//! Its witness is pinned to the one value a free search returns, so the field stays unique.

use alloc::vec::Vec;
use core::marker::PhantomData;

use hashbrown::HashMap;
use p3_challenger::fs::{
    DomainSeparator, FieldToFieldCodec, FieldUnit, Hierarchy, Interaction, InteractionPattern,
    Kind, Length, ProverState, VerifierState,
};
use p3_challenger::{CanObserve, CanSample, CanSampleBits, GrindingChallenger};
use p3_field::{ExtensionField, PrimeField64};
use p3_lookup::{
    Challenges, Kind as LookupKind, Lookup, LookupProtocol, assert_uniform_tuple_width,
};
use thiserror::Error;

/// Version byte bound into the transcript seed.
const VERSION: u8 = 1;

/// Protocol name bound into the transcript seed.
const NAME: &[u8] = b"p3-batch-stark";

/// Step label of one instance's extended-trace degree bits.
const DEGREE_BITS: &str = "degree_bits";

/// Step label of the batched main-trace commitment.
const MAIN_COMMITMENT: &str = "main_commitment";

/// Step label of one instance's public values.
const PUBLIC_VALUES: &str = "public_values";

/// Step label of the global preprocessed commitment.
const PREPROCESSED_COMMITMENT: &str = "preprocessed_commitment";

/// Step label of the grinding step guarding the lookup challenges.
const LOOKUP_POW: &str = "lookup_pow";

/// Step label of the lookup argument's base randomness.
const LOOKUP_ALPHA: &str = "lookup_alpha";

/// Step label of the lookup argument's payload combiner.
const LOOKUP_BETA: &str = "lookup_beta";

/// Step label of the batched permutation-trace commitment.
const PERMUTATION_COMMITMENT: &str = "permutation_commitment";

/// Step label of one AIR's lookup terminal.
const LOOKUP_TERMINAL: &str = "lookup_terminal";

/// Step label of the constraint-folding challenge.
const CONSTRAINT_CHALLENGE: &str = "constraint_challenge";

/// Step label of the batched quotient-chunk commitment.
const QUOTIENT_COMMITMENT: &str = "quotient_commitment";

/// Step label of the randomization-polynomial commitment.
const RANDOMIZATION_COMMITMENT: &str = "randomization_commitment";

/// Step label of the grinding step guarding the out-of-domain point.
const OOD_POW: &str = "ood_pow";

/// Step label of the out-of-domain point.
const ZETA: &str = "zeta";

/// Step label of the bracket around the delegated opening argument.
const OPENING_ARGUMENT: &str = "opening_argument";

/// Sponge alphabet of a challenger that speaks the base field natively.
type Alphabet<F> = FieldUnit<F>;

/// Type-level name of the sub-protocol the batch delegates its openings to.
///
/// Recorded on the bracket markers as a local diagnostic.
/// It does not reach the pattern fingerprint.
struct OpeningArgument;

/// A transcript step the proof failed to satisfy.
///
/// Only the two grinding steps carry a value the verifier can reject on its own.
///
/// Every other described step is replayed against data the caller validated first.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Error)]
pub enum BatchTranscriptFailure {
    /// The witness guarding the lookup challenges misses its difficulty.
    #[error("lookup phase PoW witness does not meet the required {bits} bits")]
    LookupPowWitness {
        /// Grinding difficulty the step requires.
        bits: usize,
    },
    /// The described lookup grinding step arrived with no witness to replay it.
    #[error("lookup phase PoW step of {bits} bits arrived with no witness")]
    MissingLookupPowWitness {
        /// Grinding difficulty the step requires.
        bits: usize,
    },
    /// A batch that samples no lookup challenge still carries a witness for one.
    #[error("a batch with no lookups carries a lookup PoW witness")]
    UnexpectedLookupPowWitness,
    /// The lookup witness is not the value a zero-difficulty step admits.
    ///
    /// A free search has one answer, and the proof carries a different one.
    //
    // Why: a zero-difficulty step absorbs nothing and compares nothing.
    //
    //     bits = 0 -> the witness is skipped, alpha and beta follow -> the value floats free
    //     bits > 0 -> absorbed, bits resampled                      -> the grind pins it
    //
    // The step is described whatever the difficulty, so the pattern pins the bit count.
    // A bit count is not a value, so any value rides along and the proof still verifies.
    #[error("lookup phase PoW witness is nonzero at zero difficulty, expected zero")]
    NonCanonicalLookupPowWitness,
    /// The witness guarding the out-of-domain point misses its difficulty.
    #[error("out-of-domain phase PoW witness does not meet the required {bits} bits")]
    OodPowWitness {
        /// Grinding difficulty the step requires.
        bits: usize,
    },
    /// The out-of-domain witness is not the value a zero-difficulty step admits.
    ///
    /// A free search has one answer, and the proof carries a different one.
    //
    // Why: a zero-difficulty step absorbs nothing and compares nothing.
    //
    //     bits = 0 -> the witness is skipped, zeta follows -> the value floats free
    //     bits > 0 -> absorbed, bits resampled             -> the grind pins it
    //
    // Left unpinned, the field is a second encoding of one and the same statement.
    #[error("out-of-domain phase PoW witness is nonzero at zero difficulty, expected zero")]
    NonCanonicalOodPowWitness,
}

/// Numbers that fix the transcript of one batch-STARK run.
///
/// Both sides build this from the AIRs, the common data, and the configuration.
///
/// None of it is read from a proof.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BatchShape {
    /// Main-trace width of every instance, in batch order.
    pub trace_widths: Vec<usize>,
    /// Public-value count of every instance, in batch order.
    pub public_value_counts: Vec<usize>,
    /// Preprocessed width of every instance, in batch order.
    ///
    /// Zero for an instance that declares no preprocessed columns.
    pub preprocessed_widths: Vec<usize>,
    /// Whether the batch carries a global preprocessed commitment.
    pub has_preprocessed_commitment: bool,
    /// Number of instances that declare at least one lookup.
    ///
    /// Zero means the batch runs no lookup phase at all.
    pub num_lookup_instances: usize,
    /// Grinding difficulty guarding the lookup challenges.
    pub lookup_pow_bits: usize,
    /// Whether the batch carries a randomization commitment.
    pub has_randomization_commitment: bool,
    /// Grinding difficulty guarding the out-of-domain point.
    pub ood_pow_bits: usize,
}

impl BatchShape {
    /// Number of instances in the batch.
    #[must_use]
    pub const fn num_instances(&self) -> usize {
        self.trace_widths.len()
    }

    /// Whether any instance in the batch declares a lookup.
    ///
    /// A batch with none samples no lookup challenge.
    ///
    /// It therefore grinds nothing, and carries no witness.
    #[must_use]
    pub const fn has_lookups(&self) -> bool {
        self.num_lookup_instances > 0
    }

    /// Describe the transcript this shape fixes.
    ///
    /// # Panics
    ///
    /// Never in practice.
    ///
    /// The only nesting is one matched bracket, which always passes validation.
    #[must_use]
    pub fn pattern<F, EF>(&self) -> InteractionPattern
    where
        F: PrimeField64,
        EF: ExtensionField<F>,
    {
        // Two steps per instance, one per lookup terminal, then at most nine closing steps.
        let mut steps =
            Vec::with_capacity(2 * self.num_instances() + self.num_lookup_instances + 9);

        // Each instance states the size of its extended trace domain.
        //
        // Both sides know how many such statements there are before the run.
        // The values themselves are the prover's, so they are absorbed rather than seeded.
        for _ in &self.trace_widths {
            steps.push(Interaction::algebra::<F, EF>(
                Hierarchy::Atomic,
                Kind::Message,
                DEGREE_BITS,
                Length::Scalar,
            ));
        }

        // One commitment covers every instance's main trace.
        steps.push(Interaction::opaque(
            Hierarchy::Atomic,
            Kind::Message,
            MAIN_COMMITMENT,
            Length::Scalar,
        ));

        // Public values are the caller's own input on both sides, so they never reach the wire.
        for &count in &self.public_value_counts {
            steps.push(Interaction::algebra::<F, F>(
                Hierarchy::Atomic,
                Kind::Public,
                PUBLIC_VALUES,
                Length::Fixed(count),
            ));
        }

        // Every preprocessed trace in the batch sits under one commitment.
        if self.has_preprocessed_commitment {
            steps.push(Interaction::opaque(
                Hierarchy::Atomic,
                Kind::Message,
                PREPROCESSED_COMMITMENT,
                Length::Scalar,
            ));
        }

        // A batch that declares no lookup runs none of the lookup phase.
        if self.has_lookups() {
            // Grinding sits between the trace and the pair the lookup argument runs on.
            steps.push(Interaction::algebra::<F, F>(
                Hierarchy::Atomic,
                Kind::Pow,
                LOOKUP_POW,
                Length::Fixed(self.lookup_pow_bits),
            ));

            // One pair is drawn for the whole batch, not one per bus.
            for label in [LOOKUP_ALPHA, LOOKUP_BETA] {
                steps.push(Interaction::algebra::<F, EF>(
                    Hierarchy::Atomic,
                    Kind::Challenge,
                    label,
                    Length::Scalar,
                ));
            }

            // One commitment covers every permutation trace the batch generated.
            steps.push(Interaction::opaque(
                Hierarchy::Atomic,
                Kind::Message,
                PERMUTATION_COMMITMENT,
                Length::Scalar,
            ));

            // Each participating AIR contributes the one terminal the cross-AIR sum reads.
            for _ in 0..self.num_lookup_instances {
                steps.push(Interaction::algebra::<F, EF>(
                    Hierarchy::Atomic,
                    Kind::Message,
                    LOOKUP_TERMINAL,
                    Length::Scalar,
                ));
            }
        }

        // One challenge folds every instance's constraints into one quotient claim.
        steps.push(Interaction::algebra::<F, EF>(
            Hierarchy::Atomic,
            Kind::Challenge,
            CONSTRAINT_CHALLENGE,
            Length::Scalar,
        ));

        // One commitment covers every instance's quotient chunks.
        steps.push(Interaction::opaque(
            Hierarchy::Atomic,
            Kind::Message,
            QUOTIENT_COMMITMENT,
            Length::Scalar,
        ));

        // A hiding PCS adds one randomization polynomial per instance, under one commitment.
        if self.has_randomization_commitment {
            steps.push(Interaction::opaque(
                Hierarchy::Atomic,
                Kind::Message,
                RANDOMIZATION_COMMITMENT,
                Length::Scalar,
            ));
        }

        // Grinding sits between the last commitment and the point every opening is taken at.
        //
        // Why: a witness for this step always travels in the proof.
        // The step is therefore described whatever the difficulty.
        // At zero bits the search is free and the witness is still absorbed.
        steps.push(Interaction::algebra::<F, F>(
            Hierarchy::Atomic,
            Kind::Pow,
            OOD_POW,
            Length::Fixed(self.ood_pow_bits),
        ));

        // The out-of-domain point is where the constraint identity is tested.
        steps.push(Interaction::algebra::<F, EF>(
            Hierarchy::Atomic,
            Kind::Challenge,
            ZETA,
            Length::Scalar,
        ));

        // The bracket records that a sub-protocol runs here.
        //
        // Its steps live in the callee's own pattern, under the callee's own seed.
        // What this pattern states is that the delegation happens, and where.
        steps.push(Interaction::marker::<OpeningArgument>(
            Hierarchy::Begin,
            Kind::Protocol,
            OPENING_ARGUMENT,
        ));
        steps.push(Interaction::marker::<OpeningArgument>(
            Hierarchy::End,
            Kind::Protocol,
            OPENING_ARGUMENT,
        ));

        InteractionPattern::new(steps).expect("one matched bracket is always well formed")
    }

    /// Bind the protocol identity, this shape, and the per-instance widths.
    ///
    /// A number that changes the step sequence is covered by the fingerprint.
    ///
    /// The rest go in the instance label.
    #[must_use]
    pub fn domain_separator<F, EF>(&self) -> DomainSeparator<Alphabet<F>>
    where
        F: PrimeField64,
        EF: ExtensionField<F>,
    {
        let mut separator = DomainSeparator::new(VERSION, NAME, self.pattern::<F, EF>());

        // The instance count already drives the step count, so this is defence in depth.
        separator.instance(&(self.num_instances() as u64).to_be_bytes());

        // Neither width shows up in the step sequence, so only the label can carry them.
        //
        //     trace widths [2, 3]  and  trace widths [3, 2]
        //
        // Both describe two instances, so both flatten to one step sequence.
        // Binding the values gives the two batches distinct seeds.
        for &width in &self.trace_widths {
            separator.instance(&(width as u64).to_be_bytes());
        }
        for &width in &self.preprocessed_widths {
            separator.instance(&(width as u64).to_be_bytes());
        }

        separator
    }
}

/// Lay a sampled pair out per instance, one bus offset and one combiner per lookup.
///
/// # Overview
///
/// - One pair is drawn for the whole batch, not one per bus.
/// - Each bus is separated by an additive offset from that pair.
/// - Local lookups get a unique bus, so they balance on their own.
/// - Global lookups sharing a name get one bus, so sends and receives cancel.
///
/// # Returns
///
/// - One challenge vector per instance.
/// - Each lookup contributes a pair: its bus offset, then the shared combiner.
/// - The gadget reads that pair exactly as it read its former per-lookup pair.
///
/// # Panics
///
/// - When the gadget does not read exactly two challenges per lookup.
/// - When two interactions sharing a global bus disagree on their payload width.
fn lay_out_lookup_challenges<F, EF, LG, L>(
    all_lookups: &[L],
    lookup_gadget: &LG,
    alpha: EF,
    beta: EF,
) -> Vec<Vec<EF>>
where
    F: PrimeField64,
    EF: ExtensionField<F>,
    LG: LookupProtocol,
    L: AsRef<[Lookup<F>]>,
{
    // The gadget reads two challenges per lookup: a denominator base and a combiner.
    // The single-pair scheme below relies on exactly that width.
    assert_eq!(
        lookup_gadget.num_challenges(),
        2,
        "single-pair domain separation expects a two-challenge gadget"
    );

    // Assign each bus a global index and measure the widest payload.
    //
    // - Global buses share an index by name, so cross-instance messages cancel.
    // - Local buses take a fresh index each, so nothing else can cancel them.
    // - The widest payload fixes where the bus offset sits, one power above it.
    let mut global_index: HashMap<&str, usize> = HashMap::new();
    // Every tuple folded onto a given global bus must agree on width.
    // A mismatch would let two differently-shaped payloads fingerprint identically.
    let mut global_width: HashMap<&str, usize> = HashMap::new();
    let mut next_bus = 0usize;
    let mut max_message_width = 1usize;
    let bus_ids: Vec<Vec<usize>> = all_lookups
        .iter()
        .map(|contexts| {
            contexts
                .as_ref()
                .iter()
                .map(|ctx| {
                    // A lookup's own tuples must already agree on width.
                    // That width also feeds the bus-offset power computed below.
                    let ctx_width = assert_uniform_tuple_width(&ctx.elements, "lookup");
                    max_message_width = max_message_width.max(ctx_width);

                    match &ctx.kind {
                        LookupKind::Global(name) => {
                            let id = *global_index.entry(name).or_insert_with(|| {
                                let id = next_bus;
                                next_bus += 1;
                                id
                            });
                            let expected = *global_width.entry(name).or_insert(ctx_width);
                            assert_eq!(
                                expected, ctx_width,
                                "bus {name:?}: tuple widths {expected} and {ctx_width} \
                                 differ; every interaction sharing a bus must use the \
                                 same payload width, or a shorter tuple can alias a \
                                 longer one",
                            );
                            id
                        }
                        LookupKind::Local => {
                            let id = next_bus;
                            next_bus += 1;
                            id
                        }
                    }
                })
                .collect()
        })
        .collect();

    // Precompute every bus offset once from the sampled pair.
    let challenges = Challenges::new(alpha, beta, max_message_width, next_bus);

    // Lay the challenges out per instance, one pair per lookup.
    //
    //     [ prefix[bus_0], beta, prefix[bus_1], beta, ... ]
    //
    // The gadget computes `base - combined`.
    // Passing `prefix[bus]` as the base yields the domain-separated denominator.
    bus_ids
        .iter()
        .map(|instance_buses| {
            instance_buses
                .iter()
                .flat_map(|&bus| [challenges.bus_prefix[bus], beta])
                .collect()
        })
        .collect()
}

/// Prover-side transcript of one batch-STARK run.
///
/// Holds the only definition of what a prover writes before the opening argument.
///
/// The challenger is borrowed, not consumed.
///
/// The opening argument the batch delegates to continues on the same sponge.
pub struct BatchProverTranscript<'a, C, F: PrimeField64, EF, Com> {
    /// Driver walking the description and holding the borrowed sponge.
    state: ProverState<&'a mut C, Alphabet<F>>,
    /// The numbers this run was described with.
    shape: BatchShape,
    /// Marker for the challenge field and the commitment type.
    _types: PhantomData<(EF, Com)>,
}

impl<'a, C, F, EF, Com> BatchProverTranscript<'a, C, F, EF, Com>
where
    F: PrimeField64,
    EF: ExtensionField<F>,
    Com: Clone,
    C: CanObserve<F>
        + CanObserve<Com>
        + CanSample<F>
        + CanSampleBits<usize>
        + GrindingChallenger<Witness = F>,
{
    /// Seed the transcript from the shape.
    pub fn new(challenger: &'a mut C, shape: BatchShape) -> Self {
        let separator = shape.domain_separator::<F, EF>();
        Self {
            state: ProverState::new(challenger, &separator),
            shape,
            _types: PhantomData,
        }
    }

    /// Bind the extended-trace degree bits of every instance, in batch order.
    ///
    /// # Panics
    ///
    /// When the batch does not hold one entry per described instance.
    pub fn instance_bindings(&mut self, degree_bits: &[usize]) {
        for &bits in degree_bits {
            self.state.observe_extension::<F, EF, FieldToFieldCodec<F>>(
                DEGREE_BITS,
                &EF::from(F::from_usize(bits)),
            );
        }
    }

    /// Bind the main-trace commitment, then every instance's public values.
    ///
    /// # Panics
    ///
    /// When an instance's public values do not have the described count.
    pub fn main_phase<PV>(&mut self, main_commitment: Com, public_values: &[PV])
    where
        PV: AsRef<[F]>,
    {
        // The commitment comes first, so the values are read against a fixed trace.
        self.state.observe_opaque(MAIN_COMMITMENT, main_commitment);

        for values in public_values {
            self.state
                .add_public_scalars::<F, FieldToFieldCodec<F>>(PUBLIC_VALUES, values.as_ref());
        }
    }

    /// Bind the global preprocessed commitment, when the batch has one.
    pub fn preprocessed_phase(&mut self, commitment: Option<Com>) {
        if let Some(commitment) = commitment {
            self.state
                .observe_opaque(PREPROCESSED_COMMITMENT, commitment);
        }
    }

    /// Grind, draw the lookup pair, and lay it out per instance.
    ///
    /// # Returns
    ///
    /// - One challenge vector per instance, empty for an instance with no lookup.
    /// - The grinding witness, when the batch declares a lookup.
    pub fn lookup_phase<LG, L>(
        &mut self,
        all_lookups: &[L],
        lookup_gadget: &LG,
    ) -> (Vec<Vec<EF>>, Option<F>)
    where
        LG: LookupProtocol,
        L: AsRef<[Lookup<F>]>,
    {
        // With no lookup there is no challenge to guard, so there is nothing to grind.
        if !self.shape.has_lookups() {
            return (all_lookups.iter().map(|_| Vec::new()).collect(), None);
        }

        let witness = self
            .state
            .observe_pow(LOOKUP_POW, self.shape.lookup_pow_bits);

        let alpha = self
            .state
            .challenge_extension::<F, EF, FieldToFieldCodec<F>>(LOOKUP_ALPHA)
            .into_inner();
        let beta = self
            .state
            .challenge_extension::<F, EF, FieldToFieldCodec<F>>(LOOKUP_BETA)
            .into_inner();

        (
            lay_out_lookup_challenges::<F, EF, LG, L>(all_lookups, lookup_gadget, alpha, beta),
            Some(witness),
        )
    }

    /// Bind the permutation commitment and every terminal, then draw the folding challenge.
    ///
    /// # Panics
    ///
    /// When the batch does not hold the described number of terminals.
    ///
    /// # Returns
    ///
    /// The challenge that folds every instance's constraints together.
    pub fn permutation_phase(&mut self, commitment: Option<Com>, terminals: &[EF]) -> EF {
        if let Some(commitment) = commitment {
            self.state
                .observe_opaque(PERMUTATION_COMMITMENT, commitment);
        }

        // The cross-AIR sum is checked over these, so each is bound before folding starts.
        for terminal in terminals {
            self.state
                .observe_extension::<F, EF, FieldToFieldCodec<F>>(LOOKUP_TERMINAL, terminal);
        }

        self.state
            .challenge_extension::<F, EF, FieldToFieldCodec<F>>(CONSTRAINT_CHALLENGE)
            .into_inner()
    }

    /// Bind the quotient-chunk commitment, then the randomization one when the PCS hides.
    pub fn quotient_phase(&mut self, quotient: Com, randomization: Option<Com>) {
        self.state.observe_opaque(QUOTIENT_COMMITMENT, quotient);

        if let Some(randomization) = randomization {
            self.state
                .observe_opaque(RANDOMIZATION_COMMITMENT, randomization);
        }
    }

    /// Grind, then draw the point every opening is taken at.
    ///
    /// # Returns
    ///
    /// - The out-of-domain point.
    /// - The grinding witness.
    pub fn ood_phase(&mut self) -> (EF, F) {
        let witness = self.state.observe_pow(OOD_POW, self.shape.ood_pow_bits);

        let zeta = self
            .state
            .challenge_extension::<F, EF, FieldToFieldCodec<F>>(ZETA)
            .into_inner();

        (zeta, witness)
    }

    /// Lend the sponge to the opening argument, bracketed as a sub-protocol.
    ///
    /// The callee seeds its own driver from the state this one has reached.
    ///
    /// # Returns
    ///
    /// Whatever the delegated run produced.
    pub fn delegate<R>(&mut self, run: impl FnOnce(&mut C) -> R) -> R {
        self.state
            .begin_protocol::<OpeningArgument>(OPENING_ARGUMENT);
        let output = run(self.state.challenger_mut());
        self.state.end_protocol::<OpeningArgument>(OPENING_ARGUMENT);
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
            "a batch proof carries every value in its own serde type",
        );
    }
}

/// Verifier-side transcript of one batch-STARK run.
///
/// Mirrors the prover side call for call, over the same description.
pub struct BatchVerifierTranscript<'a, C, F: PrimeField64, EF, Com> {
    /// Driver walking the description and holding the borrowed sponge.
    ///
    /// The proof carries every value, so the driver reads an empty wire.
    state: VerifierState<'static, &'a mut C, Alphabet<F>>,
    /// The numbers this run was described with.
    shape: BatchShape,
    /// Marker for the challenge field and the commitment type.
    _types: PhantomData<(EF, Com)>,
}

impl<'a, C, F, EF, Com> BatchVerifierTranscript<'a, C, F, EF, Com>
where
    F: PrimeField64,
    EF: ExtensionField<F>,
    Com: Clone,
    C: CanObserve<F>
        + CanObserve<Com>
        + CanSample<F>
        + CanSampleBits<usize>
        + GrindingChallenger<Witness = F>,
{
    /// Seed the transcript from the shape.
    pub fn new(challenger: &'a mut C, shape: BatchShape) -> Self {
        let separator = shape.domain_separator::<F, EF>();
        Self {
            state: VerifierState::new(challenger, &separator, &[]),
            shape,
            _types: PhantomData,
        }
    }

    /// Replay the extended-trace degree bits of every instance, in batch order.
    ///
    /// # Panics
    ///
    /// When the proof does not hold one entry per described instance.
    ///
    /// The caller rejects that mismatch before this driver is built.
    pub fn instance_bindings(&mut self, degree_bits: &[usize]) {
        for &bits in degree_bits {
            self.state.observe_extension::<F, EF, FieldToFieldCodec<F>>(
                DEGREE_BITS,
                &EF::from(F::from_usize(bits)),
            );
        }
    }

    /// Replay the main-trace commitment, then every instance's public values.
    ///
    /// # Panics
    ///
    /// When an instance's public values do not have the described count.
    ///
    /// The caller rejects that mismatch before this driver is built.
    pub fn main_phase<PV>(&mut self, main_commitment: Com, public_values: &[PV])
    where
        PV: AsRef<[F]>,
    {
        self.state.observe_opaque(MAIN_COMMITMENT, main_commitment);

        for values in public_values {
            self.state
                .observe_public_scalars::<F, FieldToFieldCodec<F>>(PUBLIC_VALUES, values.as_ref());
        }
    }

    /// Replay the global preprocessed commitment, when the batch has one.
    pub fn preprocessed_phase(&mut self, commitment: Option<Com>) {
        if let Some(commitment) = commitment {
            self.state
                .observe_opaque(PREPROCESSED_COMMITMENT, commitment);
        }
    }

    /// Replay the grind, redraw the lookup pair, and lay it out per instance.
    ///
    /// # Errors
    ///
    /// - The batch declares a lookup and the proof carries no witness.
    /// - The batch declares none and the proof carries one anyway.
    /// - The witness misses the difficulty its step requires.
    /// - The difficulty is zero and the witness is not the value a free search returns.
    pub fn lookup_phase<LG, L>(
        &mut self,
        all_lookups: &[L],
        lookup_gadget: &LG,
        witness: Option<F>,
    ) -> Result<Vec<Vec<EF>>, BatchTranscriptFailure>
    where
        LG: LookupProtocol,
        L: AsRef<[Lookup<F>]>,
    {
        let bits = self.shape.lookup_pow_bits;

        if !self.shape.has_lookups() {
            // No lookup challenge is drawn here, so no grinding step is described either.
            //
            // A witness therefore names a step that does not exist in this run.
            // Releasing the completeness check keeps this rejection the only failure.
            if witness.is_some() {
                self.state.abort();
                return Err(BatchTranscriptFailure::UnexpectedLookupPowWitness);
            }
            return Ok(all_lookups.iter().map(|_| Vec::new()).collect());
        }

        // With no witness the described step cannot be replayed at all.
        //
        // Releasing the completeness check keeps this rejection the only failure.
        let Some(witness) = witness else {
            self.state.abort();
            return Err(BatchTranscriptFailure::MissingLookupPowWitness { bits });
        };

        // A zero-difficulty step reads the witness without absorbing or comparing it.
        //
        //     bits = 0 -> prover emits zero, nothing is absorbed -> pin the value here
        //     bits > 0 -> prover grinds,     bits are resampled  -> the grind pins it
        //
        // The check sits ahead of the step, so nothing has entered the sponge yet.
        // Releasing the completeness check keeps this rejection the only failure.
        if bits == 0 && witness != F::ZERO {
            self.state.abort();
            return Err(BatchTranscriptFailure::NonCanonicalLookupPowWitness);
        }

        self.state
            .observe_pow(LOOKUP_POW, bits, witness)
            .map_err(|_| BatchTranscriptFailure::LookupPowWitness { bits })?;

        let alpha = self
            .state
            .challenge_extension::<F, EF, FieldToFieldCodec<F>>(LOOKUP_ALPHA)
            .into_inner();
        let beta = self
            .state
            .challenge_extension::<F, EF, FieldToFieldCodec<F>>(LOOKUP_BETA)
            .into_inner();

        Ok(lay_out_lookup_challenges::<F, EF, LG, L>(
            all_lookups,
            lookup_gadget,
            alpha,
            beta,
        ))
    }

    /// Replay the permutation commitment and every terminal, then redraw the folding challenge.
    ///
    /// # Panics
    ///
    /// When the proof does not hold the described number of terminals.
    ///
    /// The caller rejects that mismatch before this driver is built.
    ///
    /// # Returns
    ///
    /// The challenge that folds every instance's constraints together.
    pub fn permutation_phase(&mut self, commitment: Option<Com>, terminals: &[EF]) -> EF {
        if let Some(commitment) = commitment {
            self.state
                .observe_opaque(PERMUTATION_COMMITMENT, commitment);
        }

        for terminal in terminals {
            self.state
                .observe_extension::<F, EF, FieldToFieldCodec<F>>(LOOKUP_TERMINAL, terminal);
        }

        self.state
            .challenge_extension::<F, EF, FieldToFieldCodec<F>>(CONSTRAINT_CHALLENGE)
            .into_inner()
    }

    /// Replay the quotient-chunk commitment, then the randomization one when the PCS hides.
    pub fn quotient_phase(&mut self, quotient: Com, randomization: Option<Com>) {
        self.state.observe_opaque(QUOTIENT_COMMITMENT, quotient);

        if let Some(randomization) = randomization {
            self.state
                .observe_opaque(RANDOMIZATION_COMMITMENT, randomization);
        }
    }

    /// Replay the grind, then redraw the point every opening is taken at.
    ///
    /// # Errors
    ///
    /// - The witness misses the difficulty its step requires.
    /// - The difficulty is zero and the witness is not the value a free search returns.
    pub fn ood_phase(&mut self, witness: F) -> Result<EF, BatchTranscriptFailure> {
        let bits = self.shape.ood_pow_bits;

        // A zero-difficulty step reads the witness without absorbing or comparing it.
        //
        //     bits = 0 -> prover emits zero, nothing is absorbed -> pin the value here
        //     bits > 0 -> prover grinds,     bits are resampled  -> the grind pins it
        //
        // The check sits ahead of the step, so zeta is not drawn on a rejected proof.
        // Releasing the completeness check keeps this rejection the only failure.
        if bits == 0 && witness != F::ZERO {
            self.state.abort();
            return Err(BatchTranscriptFailure::NonCanonicalOodPowWitness);
        }

        self.state
            .observe_pow(OOD_POW, bits, witness)
            .map_err(|_| BatchTranscriptFailure::OodPowWitness { bits })?;

        Ok(self
            .state
            .challenge_extension::<F, EF, FieldToFieldCodec<F>>(ZETA)
            .into_inner())
    }

    /// Lend the sponge to the opening argument, bracketed as a sub-protocol.
    ///
    /// The bracket closes whatever the delegated run returned.
    ///
    /// A rejection therefore leaves this transcript replayable to the end.
    ///
    /// # Returns
    ///
    /// Whatever the delegated run produced.
    pub fn delegate<R>(&mut self, run: impl FnOnce(&mut C) -> R) -> R {
        self.state
            .begin_protocol::<OpeningArgument>(OPENING_ARGUMENT);
        let output = run(self.state.challenger_mut());
        self.state.end_protocol::<OpeningArgument>(OPENING_ARGUMENT);
        output
    }

    /// Release the completeness check because the proof is being rejected.
    ///
    /// A caller that bails between two described steps calls this before returning its error.
    ///
    /// Dropping an unfinished driver otherwise panics.
    ///
    /// That panic would land on top of an error already travelling to the caller.
    pub fn abort(mut self) {
        self.state.abort();
    }

    /// Close the transcript once every described step has been replayed.
    ///
    /// # Panics
    ///
    /// When the run replayed fewer steps than it was described with.
    pub fn finish(self) {
        self.state
            .finalize()
            .expect("a batch proof reads an empty wire, so no bytes can remain");
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
    use p3_challenger::testing::{SeedDigest, assert_seeds_pairwise_distinct, seed_digest};
    use p3_challenger::{CanSample, DuplexChallenger};
    use p3_field::PrimeCharacteristicRing;
    use p3_field::extension::BinomialExtensionField;
    use p3_lookup::logup::LogUpGadget;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    use super::*;

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;
    type Perm = Poseidon2BabyBear<16>;
    type Ch = DuplexChallenger<F, Perm, 16, 8>;

    /// A batch of two instances, neither of which declares a lookup.
    const NO_LOOKUPS: [&[Lookup<F>]; 2] = [&[], &[]];

    fn fresh_challenger() -> Ch {
        // Fixed seed so two runs differ only where the transcript makes them differ.
        let mut rng = SmallRng::seed_from_u64(0xF21);
        Ch::new(Perm::new_from_rng_128(&mut rng))
    }

    /// A two-instance batch with no preprocessed columns, no lookups, and no grinding.
    fn plain_shape() -> BatchShape {
        BatchShape {
            trace_widths: vec![2, 3],
            public_value_counts: vec![0, 1],
            preprocessed_widths: vec![0, 0],
            has_preprocessed_commitment: false,
            num_lookup_instances: 0,
            lookup_pow_bits: 0,
            has_randomization_commitment: false,
            ood_pow_bits: 0,
        }
    }

    /// The digest of the byte stream a shape seeds its sponge with.
    ///
    /// Comparing seed streams, rather than a sampled challenge, keeps the sponge out of it.
    fn seed_of(shape: &BatchShape) -> SeedDigest {
        seed_digest(&shape.domain_separator::<F, EF>())
    }

    /// One mutation of the plain shape per field a reader can set.
    fn one_mutation_per_field() -> Vec<(&'static str, BatchShape)> {
        let mut mutations = Vec::new();

        let mut wider_trace = plain_shape();
        wider_trace.trace_widths[0] += 1;
        mutations.push(("trace_widths", wider_trace));

        let mut more_public_values = plain_shape();
        more_public_values.public_value_counts[0] += 1;
        mutations.push(("public_value_counts", more_public_values));

        let mut wider_preprocessed = plain_shape();
        wider_preprocessed.preprocessed_widths[1] += 1;
        mutations.push(("preprocessed_widths", wider_preprocessed));

        let mut with_preprocessed = plain_shape();
        with_preprocessed.has_preprocessed_commitment = true;
        mutations.push(("has_preprocessed_commitment", with_preprocessed));

        let mut with_lookups = plain_shape();
        with_lookups.num_lookup_instances = 1;
        mutations.push(("num_lookup_instances", with_lookups));

        // The lookup grind is described only when the batch has lookups, so switch them on.
        let mut ground_lookups = plain_shape();
        ground_lookups.num_lookup_instances = 1;
        ground_lookups.lookup_pow_bits = 4;
        mutations.push(("lookup_pow_bits", ground_lookups));

        let mut with_randomization = plain_shape();
        with_randomization.has_randomization_commitment = true;
        mutations.push(("has_randomization_commitment", with_randomization));

        let mut ground_ood = plain_shape();
        ground_ood.ood_pow_bits = 4;
        mutations.push(("ood_pow_bits", ground_ood));

        let mut third_instance = plain_shape();
        third_instance.trace_widths.push(1);
        third_instance.public_value_counts.push(0);
        third_instance.preprocessed_widths.push(0);
        mutations.push(("num_instances", third_instance));

        mutations
    }

    #[test]
    fn no_two_configurations_of_the_shape_share_a_seed() {
        // A knob invisible to the seed is a knob the two sides can silently disagree on.
        //
        //     plain shape in the set  ->  every knob has to reach the seed
        //     pairwise over the set   ->  no two knobs may land on one seed
        //
        // Which of the fingerprint and the label carries a given knob is an implementation detail.
        // That every knob lands on a seed of its own is not.
        let mut seeds = vec![("plain", seed_of(&plain_shape()))];
        seeds.extend(
            one_mutation_per_field()
                .iter()
                .map(|(field, shape)| (*field, seed_of(shape))),
        );

        assert_seeds_pairwise_distinct(&seeds);
    }

    #[test]
    fn the_width_of_every_instance_reaches_the_seed() {
        // Reorderings share an instance count and every public-value count.
        //
        //     widths [2, 3]  and  widths [3, 2]
        //
        // Both describe two instances, so both flatten to one step sequence.
        // Only the instance label separates them, so it must carry the values.
        let ascending = plain_shape();
        let mut descending = ascending.clone();
        descending.trace_widths.reverse();

        assert_ne!(seed_of(&ascending), seed_of(&descending));
    }

    #[test]
    fn the_number_of_lookup_terminals_reaches_the_seed() {
        // Each participating AIR contributes one described terminal step.
        //
        //     one instance with lookups   ->  one terminal step
        //     two instances with lookups  ->  two terminal steps
        //
        // Nothing else in the shape changes between these two runs.
        let mut one = plain_shape();
        one.num_lookup_instances = 1;
        let mut two = one.clone();
        two.num_lookup_instances = 2;

        assert_ne!(seed_of(&one), seed_of(&two));
    }

    #[test]
    fn a_run_with_no_out_of_domain_grinding_differs_from_one_with_a_single_bit() {
        // The out-of-domain step is described at every difficulty, zero included.
        // What changes between these two runs is the difficulty the step declares.
        let ungrounded = plain_shape();
        let mut ground = ungrounded.clone();
        ground.ood_pow_bits = 1;

        assert_ne!(seed_of(&ungrounded), seed_of(&ground));
    }

    #[test]
    fn a_batch_with_no_lookups_rejects_a_smuggled_witness() {
        // Described run: two instances, neither declaring a lookup.
        //
        // No lookup challenge is drawn, so a witness names a step that does not exist.
        let mut challenger = fresh_challenger();
        let mut transcript =
            BatchVerifierTranscript::<Ch, F, EF, [F; 8]>::new(&mut challenger, plain_shape());

        let err = transcript
            .lookup_phase(&NO_LOOKUPS, &LogUpGadget::new(), Some(F::ONE))
            .expect_err("a witness for a step that is never described must error");

        assert_eq!(err, BatchTranscriptFailure::UnexpectedLookupPowWitness);
    }

    #[test]
    fn a_batch_with_lookups_rejects_a_missing_witness() {
        // Described run: two instances, one of which declares a lookup.
        //
        // A described grinding step cannot be replayed with no witness to feed it.
        let mut shape = plain_shape();
        shape.num_lookup_instances = 1;
        shape.lookup_pow_bits = 4;

        let mut challenger = fresh_challenger();
        let mut transcript =
            BatchVerifierTranscript::<Ch, F, EF, [F; 8]>::new(&mut challenger, shape);

        let err = transcript
            .lookup_phase(&NO_LOOKUPS, &LogUpGadget::new(), None)
            .expect_err("a described grinding step with no witness must error");

        assert_eq!(
            err,
            BatchTranscriptFailure::MissingLookupPowWitness { bits: 4 }
        );
    }

    #[test]
    fn a_lookup_witness_at_zero_difficulty_must_be_the_canonical_zero() {
        // Described run: two instances, one declaring a lookup, and a free grind.
        //
        //     required:  the one witness a search of zero bits returns
        //     supplied:  a value no search could have produced
        //
        // The step is described here whatever the difficulty, so the count is already bound.
        // A count is not a value, so the value is checked on its own.
        let mut shape = plain_shape();
        shape.num_lookup_instances = 1;
        assert_eq!(shape.lookup_pow_bits, 0);

        let mut challenger = fresh_challenger();
        let mut transcript =
            BatchVerifierTranscript::<Ch, F, EF, [F; 8]>::new(&mut challenger, shape);

        let err = transcript
            .lookup_phase(&NO_LOOKUPS, &LogUpGadget::new(), Some(F::ONE))
            .expect_err("a nonzero witness at zero difficulty must error");

        assert_eq!(err, BatchTranscriptFailure::NonCanonicalLookupPowWitness);
    }

    #[test]
    fn an_out_of_domain_witness_below_its_difficulty_is_rejected() {
        // Described run: two instances and a grind of 12 bits before the point is drawn.
        //
        //     required:  12 zero bits
        //     supplied:  a witness picked with no search at all
        let mut shape = plain_shape();
        shape.ood_pow_bits = 12;

        let mut challenger = fresh_challenger();
        let mut transcript =
            BatchVerifierTranscript::<Ch, F, EF, [F; 8]>::new(&mut challenger, shape);

        transcript.instance_bindings(&[4, 5]);
        transcript.main_phase([F::ONE; 8], &[&[] as &[F], &[F::ONE]]);
        transcript.preprocessed_phase(None);
        transcript
            .lookup_phase(&NO_LOOKUPS, &LogUpGadget::new(), None)
            .expect("a batch with no lookups replays with no witness");
        let _alpha = transcript.permutation_phase(None, &[]);
        transcript.quotient_phase([F::ZERO; 8], None);

        let err = transcript
            .ood_phase(F::ONE)
            .expect_err("a witness below the required difficulty must error");

        assert_eq!(err, BatchTranscriptFailure::OodPowWitness { bits: 12 });
    }

    #[test]
    fn an_out_of_domain_witness_at_zero_difficulty_must_be_the_canonical_zero() {
        // Described run: two instances and a free grind before the point is drawn.
        //
        //     required:  the one witness a search of zero bits returns
        //     supplied:  a value no search could have produced
        let shape = plain_shape();
        assert_eq!(shape.ood_pow_bits, 0);

        let mut challenger = fresh_challenger();
        let mut transcript =
            BatchVerifierTranscript::<Ch, F, EF, [F; 8]>::new(&mut challenger, shape);

        transcript.instance_bindings(&[4, 5]);
        transcript.main_phase([F::ONE; 8], &[&[] as &[F], &[F::ONE]]);
        transcript.preprocessed_phase(None);
        transcript
            .lookup_phase(&NO_LOOKUPS, &LogUpGadget::new(), None)
            .expect("a batch with no lookups replays with no witness");
        let _alpha = transcript.permutation_phase(None, &[]);
        transcript.quotient_phase([F::ZERO; 8], None);

        let err = transcript
            .ood_phase(F::ONE)
            .expect_err("a nonzero witness at zero difficulty must error");

        assert_eq!(err, BatchTranscriptFailure::NonCanonicalOodPowWitness);
    }

    #[test]
    fn the_delegation_bracket_leaves_the_sponge_untouched() {
        // Invariant: the markers are structural.
        // They record the delegation, and absorb nothing.
        //
        // Fixture state: two runs over the same shape, one bracketing an empty delegation.
        let shape = plain_shape();
        let degree_bits = [4usize, 5];
        let public_values: [&[F]; 2] = [&[], &[F::ONE]];

        let mut verifier_challenger = fresh_challenger();
        let mut verifier = BatchVerifierTranscript::<Ch, F, EF, [F; 8]>::new(
            &mut verifier_challenger,
            shape.clone(),
        );
        verifier.instance_bindings(&degree_bits);
        verifier.main_phase([F::ONE; 8], &public_values);
        verifier.preprocessed_phase(None);
        verifier
            .lookup_phase(&NO_LOOKUPS, &LogUpGadget::new(), None)
            .expect("a batch with no lookups replays with no witness");
        let verifier_alpha = verifier.permutation_phase(None, &[]);
        verifier.quotient_phase([F::ZERO; 8], None);
        let verifier_zeta = verifier
            .ood_phase(F::ZERO)
            .expect("a zero-bit grind accepts any witness");
        verifier.delegate(|_| ());
        verifier.finish();

        // The prover side plays the same steps and must land on the same challenges.
        let mut prover_challenger = fresh_challenger();
        let mut prover =
            BatchProverTranscript::<Ch, F, EF, [F; 8]>::new(&mut prover_challenger, shape);
        prover.instance_bindings(&degree_bits);
        prover.main_phase([F::ONE; 8], &public_values);
        prover.preprocessed_phase(None);
        let (_, lookup_witness) = prover.lookup_phase(&NO_LOOKUPS, &LogUpGadget::new());
        let prover_alpha = prover.permutation_phase(None, &[]);
        prover.quotient_phase([F::ZERO; 8], None);
        let (prover_zeta, ood_witness) = prover.ood_phase();
        prover.delegate(|_| ());
        prover.finish();

        assert_eq!(verifier_alpha, prover_alpha);
        assert_eq!(verifier_zeta, prover_zeta);
        assert_eq!(lookup_witness, None);
        assert_eq!(ood_witness, F::ZERO);

        // Both sponges advanced identically, so they still agree on what comes next.
        let verifier_next: F = verifier_challenger.sample();
        let prover_next: F = prover_challenger.sample();
        assert_eq!(verifier_next, prover_next);
    }
}
