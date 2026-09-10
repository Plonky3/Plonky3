//! Fiat-Shamir transcript of the batched multi-STARK statement.
//!
//! # Overview
//!
//! One statement of what a multi-STARK run binds at the top level, consumed by both sides.
//!
//! It is built from the batch: the AIRs, their trace arities, and the grinding difficulty.
//! All three exist before any proof does, so neither side reads the shape from one.
//!
//! # Shape
//!
//! ```text
//!     preprocessed commitment      only when some AIR declares preprocessed columns
//!     Begin  main commitment       bracket around the commitment scheme's own absorb
//!     End    main commitment
//!     public values                one step per instance, of that instance's own count
//!     Begin  lookup argument       bracket around the delegated run
//!     End    lookup argument
//!     Begin  zerocheck
//!     End    zerocheck
//!     Begin  main opening
//!     End    main opening
//!     Begin  preprocessed opening  only when some AIR declares preprocessed columns
//!     End    preprocessed opening
//! ```
//!
//! # What is bound
//!
//! - Shape: the instance count, every per-instance public-value count, the preprocessed presence.
//! - Instance label: every trace arity, every main width, every preprocessed width.
//! - Instance label: the grinding difficulty the delegated sumcheck rounds run at.
//! - Nothing here: the challenges, which belong entirely to the delegated sub-protocols.
//!
//! This transcript draws nothing.
//!
//! ```text
//!     statement level:  absorb, absorb, bracket, bracket, bracket, bracket
//!     inside brackets:  every challenge the run consumes
//! ```
//!
//! It therefore needs no challenge field, unlike the three sub-protocol transcripts.
//!
//! # Soundness
//!
//! Everything the whole batch is a statement about is absorbed before the first bracket.
//!
//! ```text
//!     preprocessed commitment  ->  main commitment  ->  public values  ->  first challenge
//! ```
//!
//! The first challenge any sub-protocol draws is therefore downstream of all three.
//! A prover who learned one of them first could pick a trace tuned to it.
//!
//! The public-value counts ride in the step lengths rather than in the label alone.
//!
//! ```text
//!     counts [1, 0]  ->  Fixed(1), Fixed(0)
//!     counts [0, 1]  ->  Fixed(0), Fixed(1)
//! ```
//!
//! Two batches that merely permute their arities are separated by the fingerprint alone.

use alloc::vec::Vec;

use p3_air::BaseAir;
use p3_challenger::fs::{
    DomainSeparator, FieldToFieldCodec, FieldUnit, Hierarchy, Interaction, InteractionPattern,
    Kind, Label, Length, ProverState, TranscriptField, VerifierState,
};
use p3_challenger::{CanObserve, CanSample};
use thiserror::Error;

/// Version byte bound into the transcript seed.
const VERSION: u8 = 2;

/// Protocol name bound into the transcript seed.
const NAME: &[u8] = b"p3-multi-stark";

/// Step label of the batched preprocessed commitment carried by the keys.
const PREPROCESSED_COMMITMENT: &str = "preprocessed_commitment";

/// Step label of the bracket around the main commitment.
const MAIN_COMMITMENT: &str = "main_commitment";

/// Step label of one instance's public values.
const PUBLIC_VALUES: &str = "public_values";

/// Step label of the bracket around the delegated lookup argument.
const LOOKUP_ARGUMENT: &str = "lookup_argument";

/// Step label of the bracket around the delegated AIR zerocheck.
const ZEROCHECK: &str = "zerocheck";

/// Step label of the bracket around the delegated main-trace opening.
const MAIN_OPENING: &str = "main_opening";

/// Step label of the bracket around the delegated preprocessed-trace opening.
const PREPROCESSED_OPENING: &str = "preprocessed_opening";

/// Sponge alphabet of a challenger that speaks the base field natively.
type Alphabet<F> = FieldUnit<F>;

/// Type-level name of the commitment scheme's own commit phase.
///
/// Recorded on the bracket markers as a local diagnostic.
/// It does not reach the pattern fingerprint.
struct MainCommitment;

/// Type-level name of the sub-protocol the statement delegates its lookups to.
struct LookupArgument;

/// Type-level name of the sub-protocol the statement delegates its constraints to.
struct Zerocheck;

/// Type-level name of the sub-protocol the statement delegates its main openings to.
struct MainOpening;

/// Type-level name of the sub-protocol the statement delegates its preprocessed openings to.
struct PreprocessedOpening;

/// The matched marker pair recording one delegation.
///
/// The pair absorbs nothing.
/// What it records is that a sub-protocol runs at this position, under its own seed.
///
/// The driver's opener and closer append this pair to their pattern record and touch nothing else, so a marker has no path to the sponge.
fn delegation<T: ?Sized>(label: Label) -> [Interaction; 2] {
    [
        Interaction::marker::<T>(Hierarchy::Begin, Kind::Protocol, label),
        Interaction::marker::<T>(Hierarchy::End, Kind::Protocol, label),
    ]
}

/// Numbers one AIR instance contributes to the statement's transcript.
///
/// Every one of them is read off the instance's AIR or off its declared trace arity.
/// None of them is read off a proof.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MultiStarkInstanceShape {
    /// Base-two logarithm of this instance's main trace height.
    pub num_variables: usize,
    /// Column count of this instance's main trace.
    pub main_width: usize,
    /// Column count of this instance's preprocessed trace, zero when the AIR declares none.
    pub preprocessed_width: usize,
    /// Number of public values this instance's AIR reads.
    pub num_public_values: usize,
    /// Main successor columns, in opening order.
    pub main_next_row_columns: Vec<usize>,
    /// Preprocessed successor columns, in opening order.
    pub preprocessed_next_row_columns: Vec<usize>,
}

/// Numbers that fix the transcript of one batched multi-STARK statement.
///
/// Both sides build this from their own AIRs, their own arities and their own difficulty.
///
/// The prover's arities come from the tables it holds.
/// The verifier's come from the arities its caller declared, alongside the verifying key.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MultiStarkShape {
    /// One entry per AIR instance, in batch order.
    pub instances: Vec<MultiStarkInstanceShape>,
    /// Grinding difficulty each delegated sumcheck round runs at.
    pub pow_bits: usize,
}

impl MultiStarkShape {
    /// Derive the shape of one batch from its AIRs and their trace arities.
    ///
    /// Three of the four per-instance numbers are read straight off the AIR.
    ///
    /// ```text
    ///     main_width          air.width()
    ///     preprocessed_width  air.preprocessed_width()
    ///     num_public_values   air.num_public_values()
    /// ```
    ///
    /// # Arguments
    ///
    /// - `airs`: the AIRs of the batch, in batch order.
    /// - `num_variables`: base-two logarithm of each instance's main trace height.
    /// - `pow_bits`: grinding difficulty each delegated sumcheck round runs at.
    ///
    /// # Panics
    ///
    /// When the two slices disagree on length.
    /// Both are keyed by the instance's position in batch order.
    #[must_use]
    pub fn new<F, A>(airs: &[&A], num_variables: &[usize], pow_bits: usize) -> Self
    where
        A: BaseAir<F> + ?Sized,
    {
        assert_eq!(
            airs.len(),
            num_variables.len(),
            "every AIR in the batch must declare one trace arity"
        );

        Self {
            instances: airs
                .iter()
                .zip(num_variables)
                .map(|(air, &num_variables)| MultiStarkInstanceShape {
                    num_variables,
                    main_width: air.width(),
                    preprocessed_width: air.preprocessed_width(),
                    num_public_values: air.num_public_values(),
                    main_next_row_columns: air.main_next_row_columns(),
                    preprocessed_next_row_columns: air.preprocessed_next_row_columns(),
                })
                .collect(),
            pow_bits,
        }
    }

    /// Number of preprocessed tables the batch stacks under one commitment.
    ///
    /// An AIR with no preprocessed columns contributes no table.
    #[must_use]
    pub fn num_preprocessed_tables(&self) -> usize {
        self.instances
            .iter()
            .filter(|instance| instance.preprocessed_width != 0)
            .count()
    }

    /// Whether the batch commits to a preprocessed trace at all.
    #[must_use]
    pub fn has_preprocessed(&self) -> bool {
        self.num_preprocessed_tables() > 0
    }

    /// Describe the transcript this shape fixes.
    ///
    /// # Panics
    ///
    /// Never in practice.
    /// Every bracket is emitted as a matched pair, which always passes structural validation.
    #[must_use]
    pub fn pattern<F>(&self) -> InteractionPattern
    where
        F: TranscriptField,
    {
        // One optional commitment, one step per instance, and up to five brackets.
        let mut steps = Vec::with_capacity(self.instances.len() + 11);

        // The keys carry this commitment, so it is bound before anything else.
        //
        // A batch where no AIR declares preprocessed columns commits to nothing.
        // The step is then absent rather than empty.
        if self.has_preprocessed() {
            steps.push(Interaction::opaque(
                Hierarchy::Atomic,
                Kind::Message,
                PREPROCESSED_COMMITMENT,
                Length::Scalar,
            ));
        }

        // The commitment scheme absorbs its own commitment when it commits.
        //
        // The bracket records that the absorb happens here, and that the scheme owns it.
        // The verifier never calls `commit`, so it replays the same absorb inside the bracket.
        steps.extend(delegation::<MainCommitment>(MAIN_COMMITMENT));

        // Public values are the verifier's own input, so they never travel on the wire.
        //
        // One step per instance keeps the arity vector in the step sequence.
        // A count of zero is still a step, so a permuted arity vector moves the fingerprint.
        for instance in &self.instances {
            steps.push(Interaction::algebra::<F, F>(
                Hierarchy::Atomic,
                Kind::Public,
                PUBLIC_VALUES,
                Length::Fixed(instance.num_public_values),
            ));
        }

        // Four delegations follow, in the order the run performs them.
        //
        // Each one's steps live in the callee's own pattern, under the callee's own seed.
        // What this pattern states is that the delegation happens, and where.
        steps.extend(delegation::<LookupArgument>(LOOKUP_ARGUMENT));
        steps.extend(delegation::<Zerocheck>(ZEROCHECK));
        steps.extend(delegation::<MainOpening>(MAIN_OPENING));

        // Nothing is opened against a commitment the batch never made.
        if self.has_preprocessed() {
            steps.extend(delegation::<PreprocessedOpening>(PREPROCESSED_OPENING));
        }

        InteractionPattern::new(steps).expect("matched brackets are always well formed")
    }

    /// Bind the protocol identity, this shape, and the rest of the batch's shape.
    ///
    /// A number that changes the step sequence is covered by the fingerprint.
    /// The rest go in the instance label.
    #[must_use]
    pub fn domain_separator<F>(&self) -> DomainSeparator<Alphabet<F>>
    where
        F: TranscriptField,
    {
        let mut separator = DomainSeparator::new(VERSION, NAME, self.pattern::<F>());

        // The instance count is already the public-value step count.
        // Binding it again costs one field element and keeps the label self-describing.
        separator
            .instance(&(self.instances.len() as u64).to_be_bytes())
            .instance(&(self.pow_bits as u64).to_be_bytes());

        // The step sequence sees a preprocessed commitment as present or absent.
        //
        //     widths [0, 0]  ->  no step
        //     widths [3, 0]  ->  one step
        //     widths [1, 2]  ->  one step
        //
        // Two batches that both commit therefore share a sequence, so the widths go here.
        // The arities and the main widths never touch the sequence at all.
        for instance in &self.instances {
            separator
                .instance(&(instance.num_variables as u64).to_be_bytes())
                .instance(&(instance.main_width as u64).to_be_bytes())
                .instance(&(instance.preprocessed_width as u64).to_be_bytes());
            for columns in [
                &instance.main_next_row_columns,
                &instance.preprocessed_next_row_columns,
            ] {
                separator.instance(&(columns.len() as u64).to_be_bytes());
                for &column in columns {
                    separator.instance(&(column as u64).to_be_bytes());
                }
            }
        }

        separator
    }
}

/// Prover-side transcript of one batched multi-STARK statement.
///
/// Holds the only definition of what a prover binds outside its four sub-protocols.
///
/// The challenger is borrowed, not consumed.
/// The caller keeps whatever it was doing with the sponge before and after.
pub struct MultiStarkProverTranscript<'a, C, F: TranscriptField> {
    /// Driver walking the description and holding the borrowed sponge.
    state: ProverState<&'a mut C, Alphabet<F>>,
    /// The numbers this run was described with.
    shape: MultiStarkShape,
}

impl<'a, C, F> MultiStarkProverTranscript<'a, C, F>
where
    F: TranscriptField,
    C: CanObserve<F> + CanSample<F>,
{
    /// Seed the transcript from the shape.
    pub fn new(challenger: &'a mut C, shape: MultiStarkShape) -> Self {
        let separator = shape.domain_separator::<F>();
        Self {
            state: ProverState::new(challenger, &separator),
            shape,
        }
    }

    /// Bind the batched preprocessed commitment the proving key carries.
    ///
    /// # Arguments
    ///
    /// - `commitment`: the key's commitment, absent when no AIR declares preprocessed columns.
    ///
    /// # Panics
    ///
    /// When the key and the AIRs disagree on whether a preprocessed commitment exists.
    /// Both are the caller's own inputs, so a mismatch is a caller bug.
    pub fn preprocessed_commitment<Com>(&mut self, commitment: Option<Com>)
    where
        Com: Clone,
        C: CanObserve<Com>,
    {
        match (self.shape.has_preprocessed(), commitment) {
            (true, Some(commitment)) => {
                self.state
                    .observe_opaque(PREPROCESSED_COMMITMENT, commitment);
            }
            (false, None) => {}
            (true, None) => panic!(
                "{} AIR(s) declare preprocessed columns but the proving key carries no commitment",
                self.shape.num_preprocessed_tables()
            ),
            (false, Some(_)) => panic!(
                "the proving key carries a preprocessed commitment no AIR declares columns for"
            ),
        }
    }

    /// Lend the sponge to the commitment scheme's commit phase, bracketed as a sub-protocol.
    ///
    /// The scheme absorbs the commitment it produces, so this transcript never sees it.
    ///
    /// # Returns
    ///
    /// Whatever the commit phase produced.
    pub fn main_commitment<R>(&mut self, run: impl FnOnce(&mut C) -> R) -> R {
        self.state.begin_protocol::<MainCommitment>(MAIN_COMMITMENT);
        let output = run(self.state.challenger_mut());
        self.state.end_protocol::<MainCommitment>(MAIN_COMMITMENT);
        output
    }

    /// Bind every instance's public values, one step each.
    ///
    /// # Arguments
    ///
    /// - `values`: the public values of each instance, in batch order.
    ///
    /// # Panics
    ///
    /// When the batch supplies a different instance count than the shape describes.
    /// When one instance supplies a public-value count its AIR does not declare.
    pub fn public_values(&mut self, values: &[&[F]]) {
        assert_eq!(
            values.len(),
            self.shape.instances.len(),
            "every instance in the batch must supply one public-value slice"
        );

        for (instance, slice) in self.shape.instances.iter().zip(values) {
            assert_eq!(
                slice.len(),
                instance.num_public_values,
                "an instance supplied a public-value count its AIR does not declare"
            );
            self.state
                .add_public_scalars::<F, FieldToFieldCodec<F>>(PUBLIC_VALUES, slice);
        }
    }

    /// Lend the sponge to the lookup argument, bracketed as a sub-protocol.
    ///
    /// The callee seeds its own driver from the state this one has reached.
    ///
    /// # Returns
    ///
    /// Whatever the delegated run produced.
    pub fn lookup_argument<R>(&mut self, run: impl FnOnce(&mut C) -> R) -> R {
        self.state.begin_protocol::<LookupArgument>(LOOKUP_ARGUMENT);
        let output = run(self.state.challenger_mut());
        self.state.end_protocol::<LookupArgument>(LOOKUP_ARGUMENT);
        output
    }

    /// Lend the sponge to the AIR zerocheck, bracketed as a sub-protocol.
    ///
    /// # Returns
    ///
    /// Whatever the delegated run produced.
    pub fn zerocheck<R>(&mut self, run: impl FnOnce(&mut C) -> R) -> R {
        self.state.begin_protocol::<Zerocheck>(ZEROCHECK);
        let output = run(self.state.challenger_mut());
        self.state.end_protocol::<Zerocheck>(ZEROCHECK);
        output
    }

    /// Lend the sponge to the main-trace opening, bracketed as a sub-protocol.
    ///
    /// # Returns
    ///
    /// Whatever the delegated run produced.
    pub fn main_opening<R>(&mut self, run: impl FnOnce(&mut C) -> R) -> R {
        self.state.begin_protocol::<MainOpening>(MAIN_OPENING);
        let output = run(self.state.challenger_mut());
        self.state.end_protocol::<MainOpening>(MAIN_OPENING);
        output
    }

    /// Lend the sponge to the preprocessed-trace opening, when the batch describes one.
    ///
    /// # Returns
    ///
    /// Whatever the delegated run produced, or nothing when no AIR declares columns.
    pub fn preprocessed_opening<R>(&mut self, run: impl FnOnce(&mut C) -> R) -> Option<R> {
        self.shape.has_preprocessed().then(|| {
            self.state
                .begin_protocol::<PreprocessedOpening>(PREPROCESSED_OPENING);
            let output = run(self.state.challenger_mut());
            self.state
                .end_protocol::<PreprocessedOpening>(PREPROCESSED_OPENING);
            output
        })
    }

    /// Abandon proof generation after a recoverable error, releasing the drop check.
    pub fn abort(&mut self) {
        self.state.abort();
    }

    /// Close the transcript once every described step has been played.
    ///
    /// # Panics
    ///
    /// When the run played fewer steps than it was described with.
    pub fn finish(self) {
        assert!(
            self.state.finalize().is_empty(),
            "a multi-STARK carries every value in its own proof",
        );
    }
}

/// Verifier-side transcript of one batched multi-STARK statement.
///
/// Mirrors the prover side call for call, over the same description.
pub struct MultiStarkVerifierTranscript<'a, C, F: TranscriptField> {
    /// Driver walking the description and holding the borrowed sponge.
    ///
    /// The proof carries every value, so the driver reads an empty wire.
    state: VerifierState<'static, &'a mut C, Alphabet<F>>,
    /// The numbers this run was described with.
    shape: MultiStarkShape,
}

impl<'a, C, F> MultiStarkVerifierTranscript<'a, C, F>
where
    F: TranscriptField,
    C: CanObserve<F> + CanSample<F>,
{
    /// Seed the transcript from the shape.
    pub fn new(challenger: &'a mut C, shape: MultiStarkShape) -> Self {
        let separator = shape.domain_separator::<F>();
        Self {
            state: VerifierState::new(challenger, &separator, &[]),
            shape,
        }
    }

    /// Replay the batched preprocessed commitment the verifying key carries.
    ///
    /// # Arguments
    ///
    /// - `commitment`: the key's commitment, absent when the key carries none.
    ///
    /// # Errors
    ///
    /// - The AIRs declare preprocessed columns and the key carries no commitment.
    /// - The key carries a commitment no AIR declares columns for.
    pub fn preprocessed_commitment<Com>(
        &mut self,
        commitment: Option<Com>,
    ) -> Result<(), MultiStarkTranscriptFailure>
    where
        Com: Clone,
        C: CanObserve<Com>,
    {
        match (self.shape.has_preprocessed(), commitment) {
            (true, Some(commitment)) => {
                self.state
                    .observe_opaque(PREPROCESSED_COMMITMENT, commitment);
                Ok(())
            }
            (false, None) => Ok(()),
            // Either way the run cannot continue, so the completeness check is released first.
            (true, None) => {
                self.state.abort();
                Err(MultiStarkTranscriptFailure::MissingPreprocessedCommitment {
                    tables: self.shape.num_preprocessed_tables(),
                })
            }
            (false, Some(_)) => {
                self.state.abort();
                Err(MultiStarkTranscriptFailure::UnexpectedPreprocessedCommitment)
            }
        }
    }

    /// Replay the absorb the commitment scheme performs when it commits.
    ///
    /// The scheme's own commit phase is not reachable from a verifier.
    /// The caller therefore absorbs the proof's commitment inside the same bracket.
    ///
    /// # Returns
    ///
    /// Whatever the replayed run produced.
    pub fn main_commitment<R>(&mut self, run: impl FnOnce(&mut C) -> R) -> R {
        self.state.begin_protocol::<MainCommitment>(MAIN_COMMITMENT);
        let output = run(self.state.challenger_mut());
        self.state.end_protocol::<MainCommitment>(MAIN_COMMITMENT);
        output
    }

    /// Replay every instance's public values, one step each.
    ///
    /// The values are this verifier's own input, so none of them travels on the wire.
    ///
    /// # Arguments
    ///
    /// - `values`: the public values of each instance, in batch order.
    ///
    /// # Errors
    ///
    /// - The batch supplies a different instance count than the shape describes.
    /// - One instance supplies a public-value count its AIR does not declare.
    pub fn public_values(&mut self, values: &[&[F]]) -> Result<(), MultiStarkTranscriptFailure> {
        if values.len() != self.shape.instances.len() {
            self.state.abort();
            return Err(MultiStarkTranscriptFailure::InstanceCount {
                expected: self.shape.instances.len(),
                actual: values.len(),
            });
        }

        for (index, (instance, slice)) in self.shape.instances.iter().zip(values).enumerate() {
            if slice.len() != instance.num_public_values {
                self.state.abort();
                return Err(MultiStarkTranscriptFailure::PublicValueCount {
                    instance: index,
                    expected: instance.num_public_values,
                    actual: slice.len(),
                });
            }
            self.state
                .observe_public_scalars::<F, FieldToFieldCodec<F>>(PUBLIC_VALUES, slice);
        }

        Ok(())
    }

    /// Lend the sponge to the lookup argument, bracketed as a sub-protocol.
    ///
    /// The bracket closes whatever the delegated run returned.
    /// A rejection therefore leaves this transcript replayable to the end.
    ///
    /// # Returns
    ///
    /// Whatever the delegated run produced.
    pub fn lookup_argument<R>(&mut self, run: impl FnOnce(&mut C) -> R) -> R {
        self.state.begin_protocol::<LookupArgument>(LOOKUP_ARGUMENT);
        let output = run(self.state.challenger_mut());
        self.state.end_protocol::<LookupArgument>(LOOKUP_ARGUMENT);
        output
    }

    /// Lend the sponge to the AIR zerocheck, bracketed as a sub-protocol.
    ///
    /// # Returns
    ///
    /// Whatever the delegated run produced.
    pub fn zerocheck<R>(&mut self, run: impl FnOnce(&mut C) -> R) -> R {
        self.state.begin_protocol::<Zerocheck>(ZEROCHECK);
        let output = run(self.state.challenger_mut());
        self.state.end_protocol::<Zerocheck>(ZEROCHECK);
        output
    }

    /// Lend the sponge to the main-trace opening, bracketed as a sub-protocol.
    ///
    /// # Returns
    ///
    /// Whatever the delegated run produced.
    pub fn main_opening<R>(&mut self, run: impl FnOnce(&mut C) -> R) -> R {
        self.state.begin_protocol::<MainOpening>(MAIN_OPENING);
        let output = run(self.state.challenger_mut());
        self.state.end_protocol::<MainOpening>(MAIN_OPENING);
        output
    }

    /// Lend the sponge to the preprocessed-trace opening, when the batch describes one.
    ///
    /// # Returns
    ///
    /// Whatever the delegated run produced, or nothing when no AIR declares columns.
    pub fn preprocessed_opening<R>(&mut self, run: impl FnOnce(&mut C) -> R) -> Option<R> {
        self.shape.has_preprocessed().then(|| {
            self.state
                .begin_protocol::<PreprocessedOpening>(PREPROCESSED_OPENING);
            let output = run(self.state.challenger_mut());
            self.state
                .end_protocol::<PreprocessedOpening>(PREPROCESSED_OPENING);
            output
        })
    }

    /// Release the completeness check because the proof is being rejected.
    ///
    /// A rejection that leaves steps unplayed would otherwise raise a drop-time panic.
    /// That panic would land on top of the error already travelling to the caller.
    pub fn abort(&mut self) {
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
            .expect("a multi-STARK reads an empty wire, so no bytes can remain");
    }
}

/// A statement-level transcript step the batch failed to satisfy.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum MultiStarkTranscriptFailure {
    /// The described preprocessed-commitment step arrived with no commitment.
    #[error("{tables} preprocessed table(s) described but the verifying key carries no commitment")]
    MissingPreprocessedCommitment {
        /// Number of preprocessed tables the AIRs declare.
        tables: usize,
    },
    /// A preprocessed commitment arrived for a batch described without one.
    #[error("verifying key carries a preprocessed commitment no AIR declares columns for")]
    UnexpectedPreprocessedCommitment,
    /// The batch supplied a different instance count than the shape describes.
    #[error("batch describes {expected} instance(s) but {actual} public-value slice(s) arrived")]
    InstanceCount {
        /// Instance count the shape was built from.
        expected: usize,
        /// Number of public-value slices the caller supplied.
        actual: usize,
    },
    /// One instance supplied a public-value count its AIR does not declare.
    #[error("instance {instance} declares {expected} public value(s) but {actual} arrived")]
    PublicValueCount {
        /// Position of the offending instance in batch order.
        instance: usize,
        /// Public-value count the instance's AIR declares.
        expected: usize,
        /// Public-value count the caller supplied.
        actual: usize,
    },
}

#[cfg(test)]
mod tests {
    extern crate std;

    use alloc::vec;
    use core::ops::Range;

    use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
    use p3_challenger::DuplexChallenger;
    use p3_field::PrimeCharacteristicRing;
    use proptest::prelude::*;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    use super::*;

    type F = BabyBear;
    type Perm = Poseidon2BabyBear<16>;
    type Ch = DuplexChallenger<F, Perm, 16, 8>;

    /// Width of the opaque commitments the fixtures absorb.
    const DIGEST: usize = 8;

    fn fresh_challenger() -> Ch {
        // Fixed seed so two runs differ only where the transcript makes them differ.
        let mut rng = SmallRng::seed_from_u64(0x4A15_7A7E);
        Ch::new(Perm::new_from_rng_128(&mut rng))
    }

    /// Baseline shape every walk below perturbs exactly one field of.
    ///
    /// Two instances, one of them preprocessed, both with public values.
    fn base_shape() -> MultiStarkShape {
        MultiStarkShape {
            instances: vec![
                MultiStarkInstanceShape {
                    num_variables: 8,
                    main_width: 4,
                    preprocessed_width: 2,
                    num_public_values: 3,
                    main_next_row_columns: vec![0, 1],
                    preprocessed_next_row_columns: vec![0],
                },
                MultiStarkInstanceShape {
                    num_variables: 6,
                    main_width: 5,
                    preprocessed_width: 0,
                    num_public_values: 1,
                    main_next_row_columns: vec![0],
                    preprocessed_next_row_columns: vec![],
                },
            ],
            pow_bits: 4,
        }
    }

    /// Whether two shapes seed a sponge identically, compared on the preimage not a sample.
    ///
    /// The preimage is the pattern fingerprint followed by the instance label.
    /// Comparing it rather than one sampled element keeps the comparison exact.
    fn seeds_agree(left: &MultiStarkShape, right: &MultiStarkShape) -> bool {
        let left = left.domain_separator::<F>();
        let right = right.domain_separator::<F>();

        left.pattern().pattern_hash() == right.pattern().pattern_hash()
            && left.instance_label() == right.instance_label()
    }

    #[test]
    fn ordered_successor_columns_change_the_statement_seed() {
        struct ColumnsAir(Vec<usize>, Vec<usize>);
        impl BaseAir<F> for ColumnsAir {
            fn width(&self) -> usize {
                3
            }
            fn preprocessed_width(&self) -> usize {
                3
            }
            fn main_next_row_columns(&self) -> Vec<usize> {
                self.0.clone()
            }
            fn preprocessed_next_row_columns(&self) -> Vec<usize> {
                self.1.clone()
            }
        }
        let baseline =
            MultiStarkShape::new::<F, _>(&[&ColumnsAir(vec![0, 1], vec![0, 1])], &[4], 0);
        for air in [
            ColumnsAir(vec![1, 0], vec![0, 1]),
            ColumnsAir(vec![0, 2], vec![0, 1]),
            ColumnsAir(vec![0, 1], vec![1, 0]),
            ColumnsAir(vec![0, 1], vec![0, 2]),
        ] {
            let changed = MultiStarkShape::new::<F, _>(&[&air], &[4], 0);
            assert!(!seeds_agree(&baseline, &changed));
        }
    }

    /// First challenge a shape's seed produces on a fresh sponge.
    fn first_challenge(shape: &MultiStarkShape) -> F {
        let mut challenger = fresh_challenger();
        shape.domain_separator::<F>().seed(&mut challenger);
        challenger.sample()
    }

    /// Every field of the shape, each moved one step away from `base_shape`.
    ///
    /// The batch and one of its instances are destructured below with no rest pattern.
    /// Both are destructured, since a new knob can land on either struct.
    /// Every binding then feeds the row that covers it, so an unread field is a build failure.
    ///
    /// The enforcement is weaker than it looks.
    /// A new field trips the missing-field error at the baseline's struct literals first.
    /// What is enforced is that the author must come through here, not that a row appears.
    fn one_step_from_base() -> Vec<(&'static str, MultiStarkShape)> {
        let MultiStarkShape {
            instances,
            pow_bits,
        } = base_shape();
        let num_instances = instances.len();
        let MultiStarkInstanceShape {
            num_variables,
            main_width,
            preprocessed_width,
            num_public_values,
            main_next_row_columns,
            preprocessed_next_row_columns,
        } = instances
            .into_iter()
            .next()
            .expect("the baseline batch is not empty");

        let mut mutations = Vec::new();

        let mut shape = base_shape();
        shape.pow_bits = pow_bits + 1;
        mutations.push(("pow_bits", shape));

        let mut shape = base_shape();
        shape.instances.truncate(num_instances - 1);
        mutations.push(("instances.len", shape));

        let mut shape = base_shape();
        shape.instances[0].num_variables = num_variables + 1;
        mutations.push(("instance.num_variables", shape));

        let mut shape = base_shape();
        shape.instances[0].main_width = main_width + 1;
        mutations.push(("instance.main_width", shape));

        let mut shape = base_shape();
        shape.instances[0].preprocessed_width = preprocessed_width + 1;
        mutations.push(("instance.preprocessed_width", shape));

        let mut shape = base_shape();
        shape.instances[0].num_public_values = num_public_values + 1;
        mutations.push(("instance.num_public_values", shape));

        let mut shape = base_shape();
        shape.instances[0].main_next_row_columns =
            main_next_row_columns.into_iter().rev().collect();
        mutations.push(("instance.main_next_row_columns", shape));
        let mut shape = base_shape();
        shape.instances[0].preprocessed_next_row_columns = preprocessed_next_row_columns
            .into_iter()
            .map(|c| c + 1)
            .collect();
        mutations.push(("instance.preprocessed_next_row_columns", shape));

        mutations
    }

    /// Public values matching a shape, filled with distinguishable elements.
    fn public_values_for(shape: &MultiStarkShape) -> Vec<Vec<F>> {
        shape
            .instances
            .iter()
            .enumerate()
            .map(|(index, instance)| {
                (0..instance.num_public_values)
                    .map(|slot| F::from_usize(1 + index * 16 + slot))
                    .collect()
            })
            .collect()
    }

    /// Reborrow owned public-value rows as the slice-of-slices both sides take.
    fn borrow_all(values: &[Vec<F>]) -> Vec<&[F]> {
        values.iter().map(Vec::as_slice).collect()
    }

    /// Play the whole prover side, returning the sponge state it leaves behind.
    ///
    /// The delegations absorb the commitment digests a real sub-protocol would absorb.
    fn play_prover(
        shape: &MultiStarkShape,
        preprocessed: Option<[F; DIGEST]>,
        main: [F; DIGEST],
        public_values: &[&[F]],
    ) -> F {
        let mut challenger = fresh_challenger();
        let mut transcript =
            MultiStarkProverTranscript::<Ch, F>::new(&mut challenger, shape.clone());

        transcript.preprocessed_commitment(preprocessed);
        transcript.main_commitment(|challenger| challenger.observe(main));
        transcript.public_values(public_values);
        transcript.lookup_argument(|_| ());
        transcript.zerocheck(|_| ());
        transcript.main_opening(|_| ());
        transcript.preprocessed_opening(|_| ());
        transcript.finish();

        challenger.sample()
    }

    /// Play the whole verifier side, returning the sponge state it leaves behind.
    fn play_verifier(
        shape: &MultiStarkShape,
        preprocessed: Option<[F; DIGEST]>,
        main: [F; DIGEST],
        public_values: &[&[F]],
    ) -> F {
        let mut challenger = fresh_challenger();
        let mut transcript =
            MultiStarkVerifierTranscript::<Ch, F>::new(&mut challenger, shape.clone());

        transcript
            .preprocessed_commitment(preprocessed)
            .expect("the described preprocessed commitment is supplied");
        transcript.main_commitment(|challenger| challenger.observe(main));
        transcript
            .public_values(public_values)
            .expect("the public values match the described counts");
        transcript.lookup_argument(|_| ());
        transcript.zerocheck(|_| ());
        transcript.main_opening(|_| ());
        transcript.preprocessed_opening(|_| ());
        transcript.finish();

        challenger.sample()
    }

    #[test]
    fn the_same_shape_seeds_the_same_stream_twice() {
        // Completeness: the seed is a pure function of the shape.
        let shape = base_shape();
        assert_eq!(first_challenge(&shape), first_challenge(&shape));
    }

    #[test]
    fn every_field_of_the_shape_reaches_the_seed() {
        // Baseline: the two-instance shape, seeded and sampled once.
        let baseline = first_challenge(&base_shape());

        // A field the fingerprint covers moves the seed through the step sequence.
        // A field it does not must move it through the instance label instead.
        for (field, shape) in one_step_from_base() {
            assert_ne!(
                baseline,
                first_challenge(&shape),
                "changing `{field}` left the seed where it was",
            );
        }
    }

    #[test]
    fn no_two_one_step_mutations_collide() {
        // Invariant: the knobs are separated from each other, not merely from the baseline.
        //
        // Two knobs bound as one number would agree here while both differing from the baseline.
        let mutations = one_step_from_base();

        for (i, (left_field, left)) in mutations.iter().enumerate() {
            for (right_field, right) in &mutations[i + 1..] {
                assert_ne!(
                    first_challenge(left),
                    first_challenge(right),
                    "`{left_field}` and `{right_field}` land on the same seed",
                );
            }
        }
    }

    #[test]
    fn the_shape_constructor_reads_the_airs_it_is_given() {
        // Fixture state: two AIRs whose self-reported numbers are all distinct.
        struct ShapeAir {
            width: usize,
            preprocessed_width: usize,
            num_public_values: usize,
        }

        impl BaseAir<F> for ShapeAir {
            fn width(&self) -> usize {
                self.width
            }

            fn preprocessed_width(&self) -> usize {
                self.preprocessed_width
            }

            fn num_public_values(&self) -> usize {
                self.num_public_values
            }
        }

        let first = ShapeAir {
            width: 7,
            preprocessed_width: 3,
            num_public_values: 5,
        };
        let second = ShapeAir {
            width: 2,
            preprocessed_width: 0,
            num_public_values: 1,
        };

        // The arities and the difficulty come from the caller's own configuration.
        let shape = MultiStarkShape::new::<F, _>(&[&first, &second], &[9, 4], 6);

        assert_eq!(
            shape,
            MultiStarkShape {
                instances: vec![
                    MultiStarkInstanceShape {
                        num_variables: 9,
                        main_width: 7,
                        preprocessed_width: 3,
                        num_public_values: 5,
                        main_next_row_columns: (0..7).collect(),
                        preprocessed_next_row_columns: (0..3).collect(),
                    },
                    MultiStarkInstanceShape {
                        num_variables: 4,
                        main_width: 2,
                        preprocessed_width: 0,
                        num_public_values: 1,
                        main_next_row_columns: (0..2).collect(),
                        preprocessed_next_row_columns: vec![],
                    },
                ],
                pow_bits: 6,
            }
        );
        assert_eq!(shape.num_preprocessed_tables(), 1);
        assert!(shape.has_preprocessed());
    }

    #[test]
    fn a_permuted_public_value_arity_vector_moves_the_fingerprint() {
        // Both batches read two public values in total, one per instance pair.
        //
        //     counts [3, 1]  ->  Fixed(3), Fixed(1)
        //     counts [1, 3]  ->  Fixed(1), Fixed(3)
        //
        // The step lengths carry the order, so the fingerprint alone separates them.
        let straight = base_shape();
        let mut swapped = base_shape();
        swapped.instances[0].num_public_values = 1;
        swapped.instances[1].num_public_values = 3;

        assert_ne!(
            straight.pattern::<F>().pattern_hash(),
            swapped.pattern::<F>().pattern_hash()
        );
    }

    #[test]
    fn a_batch_with_no_preprocessed_columns_describes_a_shorter_sequence() {
        // With no preprocessed columns there is nothing to commit and nothing to open.
        //
        //     with columns : commitment, 5 brackets, 2 public-value steps  -> 13 steps
        //     without them : 4 brackets, 2 public-value steps              -> 10 steps
        let with_preprocessed = base_shape();
        let mut without_preprocessed = base_shape();
        without_preprocessed.instances[0].preprocessed_width = 0;

        assert_eq!(with_preprocessed.pattern::<F>().len(), 13);
        assert_eq!(without_preprocessed.pattern::<F>().len(), 10);
        assert!(!without_preprocessed.has_preprocessed());
    }

    #[test]
    fn the_two_sides_land_on_the_same_sponge_state() {
        // Completeness: prover and verifier walk one description over one seed.
        let shape = base_shape();
        let public_values = public_values_for(&shape);
        let borrowed = borrow_all(&public_values);

        let written = play_prover(&shape, Some([F::ONE; DIGEST]), [F::TWO; DIGEST], &borrowed);
        let replayed = play_verifier(&shape, Some([F::ONE; DIGEST]), [F::TWO; DIGEST], &borrowed);

        assert_eq!(written, replayed);
    }

    #[test]
    fn a_perturbed_preprocessed_commitment_moves_the_stream() {
        // Soundness: the key's commitment is absorbed, so it reaches every later challenge.
        let shape = base_shape();
        let public_values = public_values_for(&shape);
        let borrowed = borrow_all(&public_values);

        let honest = play_verifier(&shape, Some([F::ONE; DIGEST]), [F::TWO; DIGEST], &borrowed);

        // Mutation: one limb of the preprocessed digest.
        let mut forged = [F::ONE; DIGEST];
        forged[DIGEST - 1] = F::from_u8(9);
        let tampered = play_verifier(&shape, Some(forged), [F::TWO; DIGEST], &borrowed);

        assert_ne!(honest, tampered);
    }

    #[test]
    fn a_perturbed_main_commitment_moves_the_stream() {
        // Soundness: the main commitment is absorbed inside its bracket, by both sides.
        let shape = base_shape();
        let public_values = public_values_for(&shape);
        let borrowed = borrow_all(&public_values);

        let honest = play_verifier(&shape, Some([F::ONE; DIGEST]), [F::TWO; DIGEST], &borrowed);

        // Mutation: one limb of the main digest.
        let mut forged = [F::TWO; DIGEST];
        forged[0] = F::from_u8(7);
        let tampered = play_verifier(&shape, Some([F::ONE; DIGEST]), forged, &borrowed);

        assert_ne!(honest, tampered);
    }

    #[test]
    fn a_perturbed_public_value_moves_the_stream() {
        // Public values never travel on the wire, so only the absorb binds them.
        //
        //     verifier holds [.., 3]  ->  one stream
        //     verifier holds [.., 4]  ->  another
        let shape = base_shape();
        let honest = public_values_for(&shape);

        for instance in 0..shape.instances.len() {
            for slot in 0..shape.instances[instance].num_public_values {
                let mut tampered = honest.clone();
                tampered[instance][slot] += F::ONE;

                let honest_borrowed = borrow_all(&honest);
                let tampered_borrowed = borrow_all(&tampered);

                assert_ne!(
                    play_verifier(
                        &shape,
                        Some([F::ONE; DIGEST]),
                        [F::TWO; DIGEST],
                        &honest_borrowed
                    ),
                    play_verifier(
                        &shape,
                        Some([F::ONE; DIGEST]),
                        [F::TWO; DIGEST],
                        &tampered_borrowed
                    ),
                    "public value {slot} of instance {instance} does not reach the stream",
                );
            }
        }
    }

    #[test]
    fn the_delegation_brackets_absorb_nothing() {
        // Invariant: the markers are structural.
        // They record the delegation, they absorb nothing.
        //
        // Fixture state: a batch with no preprocessed columns, so only three brackets run.
        let mut shape = base_shape();
        shape.instances[0].preprocessed_width = 0;
        let public_values = public_values_for(&shape);
        let borrowed = borrow_all(&public_values);

        // The bracketed run plays every marker and lends the sponge to empty callees.
        let bracketed = play_prover(&shape, None, [F::TWO; DIGEST], &borrowed);

        // The bare run absorbs the same two values with no driver at all.
        let mut bare = fresh_challenger();
        shape.domain_separator::<F>().seed(&mut bare);
        bare.observe([F::TWO; DIGEST]);
        for slice in &borrowed {
            bare.observe_slice(slice);
        }

        assert_eq!(bracketed, bare.sample());
    }

    #[test]
    fn a_described_preprocessed_commitment_that_never_arrives_is_rejected() {
        // Described batch: an AIR with two preprocessed columns, so the step exists.
        let mut challenger = fresh_challenger();
        let mut transcript =
            MultiStarkVerifierTranscript::<Ch, F>::new(&mut challenger, base_shape());

        let err = transcript
            .preprocessed_commitment::<[F; DIGEST]>(None)
            .expect_err("a described commitment step with no commitment must error");

        assert_eq!(
            err,
            MultiStarkTranscriptFailure::MissingPreprocessedCommitment { tables: 1 }
        );
    }

    #[test]
    fn a_preprocessed_commitment_nobody_described_is_rejected() {
        // Described batch: no AIR declares preprocessed columns, so there is no step.
        let mut shape = base_shape();
        shape.instances[0].preprocessed_width = 0;

        let mut challenger = fresh_challenger();
        let mut transcript = MultiStarkVerifierTranscript::<Ch, F>::new(&mut challenger, shape);

        let err = transcript
            .preprocessed_commitment(Some([F::ONE; DIGEST]))
            .expect_err("a commitment outside the described sequence must error");

        assert_eq!(
            err,
            MultiStarkTranscriptFailure::UnexpectedPreprocessedCommitment
        );
    }

    #[test]
    fn a_public_value_count_no_air_declares_is_rejected() {
        // Described batch: three public values for the first instance, one for the second.
        let mut challenger = fresh_challenger();
        let mut transcript =
            MultiStarkVerifierTranscript::<Ch, F>::new(&mut challenger, base_shape());

        transcript
            .preprocessed_commitment(Some([F::ONE; DIGEST]))
            .expect("the described preprocessed commitment is supplied");
        transcript.main_commitment(|challenger| challenger.observe([F::TWO; DIGEST]));

        // Mutation: the second instance arrives one value short.
        let err = transcript
            .public_values(&[&[F::ONE, F::TWO, F::ONE], &[]])
            .expect_err("a public-value count outside the description must error");

        assert_eq!(
            err,
            MultiStarkTranscriptFailure::PublicValueCount {
                instance: 1,
                expected: 1,
                actual: 0,
            }
        );
    }

    #[test]
    fn an_instance_count_the_shape_does_not_describe_is_rejected() {
        // Described batch: two instances, so two public-value steps.
        let mut challenger = fresh_challenger();
        let mut transcript =
            MultiStarkVerifierTranscript::<Ch, F>::new(&mut challenger, base_shape());

        transcript
            .preprocessed_commitment(Some([F::ONE; DIGEST]))
            .expect("the described preprocessed commitment is supplied");
        transcript.main_commitment(|challenger| challenger.observe([F::TWO; DIGEST]));

        // Mutation: one slice arrives for a batch described with two.
        let err = transcript
            .public_values(&[&[F::ONE, F::TWO, F::ONE]])
            .expect_err("a batch shorter than the description must error");

        assert_eq!(
            err,
            MultiStarkTranscriptFailure::InstanceCount {
                expected: 2,
                actual: 1,
            }
        );
    }

    #[test]
    fn an_aborted_verifier_transcript_does_not_panic_on_drop() {
        // A verifier that rejects mid-pattern must let the error reach its caller.
        //
        // Fixture state: both commitments replayed, every later step still unplayed.
        let shape = base_shape();
        let public_values = public_values_for(&shape);
        let borrowed = borrow_all(&public_values);

        let mut challenger = fresh_challenger();
        let mut transcript = MultiStarkVerifierTranscript::<Ch, F>::new(&mut challenger, shape);
        transcript
            .preprocessed_commitment(Some([F::ONE; DIGEST]))
            .expect("the described preprocessed commitment is supplied");
        transcript.main_commitment(|challenger| challenger.observe([F::TWO; DIGEST]));
        transcript
            .public_values(&borrowed)
            .expect("the public values match the described counts");

        // Mutation: the run is abandoned before the first bracket, as a rejection would.
        transcript.abort();
        drop(transcript);
    }

    /// A shape drawn from the same knobs the walk above perturbs, over a given batch size.
    ///
    /// The size is a parameter rather than fixed, so a property needing two instances draws two.
    /// Narrowing the draw beats filtering it, since a rejected case is a case that did no work.
    fn arbitrary_shape_sized(instances: Range<usize>) -> impl Strategy<Value = MultiStarkShape> {
        let instance = (1_usize..4, 1_usize..4, 0_usize..3, 0_usize..4).prop_map(
            |(num_variables, main_width, preprocessed_width, num_public_values)| {
                MultiStarkInstanceShape {
                    num_variables,
                    main_width,
                    preprocessed_width,
                    num_public_values,
                    main_next_row_columns: (0..main_width).collect(),
                    preprocessed_next_row_columns: (0..preprocessed_width).collect(),
                }
            },
        );

        (proptest::collection::vec(instance, instances), 0_usize..5).prop_map(
            |(instances, pow_bits)| MultiStarkShape {
                instances,
                pow_bits,
            },
        )
    }

    /// Every legal batch size, the single-instance batch included.
    fn arbitrary_shape() -> impl Strategy<Value = MultiStarkShape> {
        arbitrary_shape_sized(1..4)
    }

    /// Only the batch sizes a permutation can reorder.
    ///
    /// A one-instance batch has no permutation but the identity, so it is left out.
    fn arbitrary_reorderable_shape() -> impl Strategy<Value = MultiStarkShape> {
        arbitrary_shape_sized(2..4)
    }

    proptest! {
        #[test]
        fn both_sides_land_on_the_same_sponge_state_for_any_batch(
            shape in arbitrary_shape(),
        ) {
            // Completeness over random batches: one description, one seed, one stream.
            let public_values = public_values_for(&shape);
            let borrowed = borrow_all(&public_values);

            // The commitment exists exactly when the batch describes preprocessed columns.
            let preprocessed = shape.has_preprocessed().then_some([F::ONE; DIGEST]);

            let written = play_prover(&shape, preprocessed, [F::TWO; DIGEST], &borrowed);
            let replayed = play_verifier(&shape, preprocessed, [F::TWO; DIGEST], &borrowed);

            prop_assert_eq!(written, replayed);
        }

        #[test]
        fn two_distinct_batches_never_share_a_seed_preimage(
            left in arbitrary_shape(),
            right in arbitrary_shape(),
        ) {
            // Soundness over random batches: distinct descriptions land on distinct seeds.
            prop_assert_eq!(left == right, seeds_agree(&left, &right));
        }

        #[test]
        fn permuting_the_instances_of_a_batch_moves_the_seed(
            shape in arbitrary_reorderable_shape(),
        ) {
            // Soundness over random batches: the batch order is part of the statement.
            //
            //     public-value counts  ->  step lengths, so the fingerprint holds their order
            //     arities and widths   ->  instance label, written in the same order
            //
            // Every rotation and the reversal are tried, which is every order of a two-instance batch.
            //
            // Why the draw starts at two instances: a one-instance batch offers no rotation and a reversal equal to itself.
            // Every case then reaches at least one real comparison below.
            let rotations = (1..shape.instances.len()).map(|shift| {
                let mut rotated = shape.clone();
                rotated.instances.rotate_left(shift);
                rotated
            });
            let mut reversed = shape.clone();
            reversed.instances.reverse();

            for permuted in rotations.chain(core::iter::once(reversed)) {
                // A batch of interchangeable instances is left where it was by a permutation.
                if permuted == shape {
                    continue;
                }

                prop_assert!(
                    !seeds_agree(&shape, &permuted),
                    "{:?} and its permutation {:?} share a seed",
                    shape.instances,
                    permuted.instances,
                );
            }
        }
    }
}
