//! Fiat-Shamir transcript of one indexed-lookup reduction.
//!
//! # Shape
//!
//! ```text
//!     reader batching              one extension element
//!     pushforward                  one message per table, as wide as that table
//!     entry challenges             one extension element per table
//!     Begin  fraction reduction    bracket around the delegated run
//!     End    fraction reduction
//!     position claims              one extension element per reader
//!     column batching              one extension element
//!     Begin  product sumcheck      bracket around the delegated run
//!     End    product sumcheck
//!     column claims                one extension element per table column
//! ```
//!
//! # Soundness
//!
//! The reader batching challenge precedes the pushforwards, which are built from it.
//!
//! The pushforwards precede the entry challenges.
//!
//! No pushforward can then be tuned to the challenge that tests it.
//!
//! ```text
//!     draw batching  ->  build pushforwards  ->  send them  ->  draw entry challenges
//! ```
//!
//! One entry challenge is drawn per table rather than one for the whole reduction.
//!
//! Each table's contribution is then a rational function of its own challenge alone.
//!
//! A sum of such functions vanishes only when every term does.
//!
//! Under a shared challenge two tables could miscount opposite entries and cancel.
//!
//! The column batching challenge comes last.
//!
//! By then the reduction has fixed the point its claims are taken at.

use alloc::vec;
use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_challenger::fs::{
    DomainSeparator, FieldToFieldCodec, FieldUnit, Hierarchy, Interaction, InteractionPattern,
    Kind, Length, ProverState, TranscriptField, VerifierState,
};
use p3_challenger::{CanObserve, CanSample};
use p3_field::ExtensionField;

use super::plan::LogupStarPlan;

/// Version byte bound into the transcript seed.
const VERSION: u8 = 1;

/// Protocol name bound into the transcript seed.
const NAME: &[u8] = b"p3-multi-stark-logup-star";

/// Step label of the challenge weighting the readers of one table against each other.
const READER_BATCHING: &str = "reader_batching";

/// Step label of one table's pushforward.
const PUSHFORWARD: &str = "pushforward";

/// Step label of the per-table challenges the fraction identity is tested at.
const ENTRY_CHALLENGES: &str = "entry_challenges";

/// Step label of the bracket around the delegated fraction reduction.
const FRACTION_REDUCTION: &str = "fraction_reduction";

/// Step label of the position-column values at the reduction's output point.
const POSITION_CLAIMS: &str = "position_claims";

/// Step label of the challenge combining a table's columns into one product claim.
const COLUMN_BATCHING: &str = "column_batching";

/// Step label of the bracket around the delegated product sumcheck.
const PRODUCT_SUMCHECK: &str = "product_sumcheck";

/// Step label of the table-column values at the product sumcheck's output point.
const COLUMN_CLAIMS: &str = "column_claims";

/// Sponge alphabet of a challenger that speaks the base field natively.
type Alphabet<F> = FieldUnit<F>;

/// What a verifier-side width mismatch means.
///
/// Every count is settled against the shape before the sponge is touched.
///
/// A replay can only hit one of these if that check was skipped.
const MEASURED_FIRST: &str = "the proof was measured against this shape before the replay began";

/// Type-level name of the sub-protocol the fraction identity is delegated to.
struct FractionReduction;

/// Type-level name of the sub-protocol the product claims are delegated to.
struct ProductSumcheck;

/// Numbers that fix the transcript of one reduction.
///
/// Both sides read these off their own copy of the statement.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LogupStarShape {
    /// Base-two logarithm of each table's entry count, in statement order.
    table_variables: Vec<usize>,
    /// Number of columns each table's entries carry, in statement order.
    table_widths: Vec<usize>,
    /// Base-two logarithm of each reader's row count, tables in order and readers within them.
    reader_variables: Vec<usize>,
    /// Variable count of the padded leaf table the fraction reduction consumes.
    num_variables: usize,
}

impl LogupStarShape {
    /// Read the shape off a layout.
    pub fn new(plan: &LogupStarPlan) -> Self {
        Self {
            table_variables: plan
                .tables
                .iter()
                .map(|table| table.num_variables)
                .collect(),
            table_widths: plan.tables.iter().map(|table| table.width).collect(),
            reader_variables: plan
                .tables
                .iter()
                .flat_map(|table| table.readers.iter().copied())
                .collect(),
            num_variables: plan.num_variables,
        }
    }

    /// Total number of column claims the reduction closes on.
    fn num_column_claims(&self) -> usize {
        self.table_widths.iter().sum()
    }

    /// Describe the transcript this shape fixes.
    ///
    /// # Panics
    ///
    /// Never in practice.
    ///
    /// Two matched brackets always pass structural validation.
    #[must_use]
    pub fn pattern<F, EF>(&self) -> InteractionPattern
    where
        F: TranscriptField,
        EF: ExtensionField<F>,
    {
        let challenge = |label, length| {
            Interaction::algebra::<F, EF>(Hierarchy::Atomic, Kind::Challenge, label, length)
        };
        let message = |label, length| {
            Interaction::algebra::<F, EF>(Hierarchy::Atomic, Kind::Message, label, length)
        };

        // The readers of one table are weighted by powers of this, so it precedes their weights.
        let mut steps = vec![challenge(READER_BATCHING, Length::Scalar)];

        // One pushforward per table, each as wide as the table it summarizes.
        //
        // Widths differ between statements.
        //
        // They reach the fingerprint through the step lengths, not the instance label.
        steps.extend(
            self.table_variables
                .iter()
                .map(|&num_variables| message(PUSHFORWARD, Length::Fixed(1 << num_variables))),
        );

        // Every pushforward is bound before any challenge that tests one is drawn.
        steps.push(challenge(
            ENTRY_CHALLENGES,
            Length::Fixed(self.table_variables.len()),
        ));

        steps.push(Interaction::marker::<FractionReduction>(
            Hierarchy::Begin,
            Kind::Protocol,
            FRACTION_REDUCTION,
        ));
        steps.push(Interaction::marker::<FractionReduction>(
            Hierarchy::End,
            Kind::Protocol,
            FRACTION_REDUCTION,
        ));

        // The reduction closes on one position-column value per reader, which the caller opens.
        steps.push(message(
            POSITION_CLAIMS,
            Length::Fixed(self.reader_variables.len()),
        ));

        // Drawn once the reduction has fixed the point everything below is claimed at.
        steps.push(challenge(COLUMN_BATCHING, Length::Scalar));

        steps.push(Interaction::marker::<ProductSumcheck>(
            Hierarchy::Begin,
            Kind::Protocol,
            PRODUCT_SUMCHECK,
        ));
        steps.push(Interaction::marker::<ProductSumcheck>(
            Hierarchy::End,
            Kind::Protocol,
            PRODUCT_SUMCHECK,
        ));

        steps.push(message(
            COLUMN_CLAIMS,
            Length::Fixed(self.num_column_claims()),
        ));

        InteractionPattern::new(steps).expect("two matched brackets are always well formed")
    }

    /// Bind the protocol identity and every number the statement is shaped by.
    #[must_use]
    pub fn domain_separator<F, EF>(&self) -> DomainSeparator<Alphabet<F>>
    where
        F: TranscriptField,
        EF: ExtensionField<F>,
    {
        let mut separator = DomainSeparator::new(VERSION, NAME, self.pattern::<F, EF>());

        // How wide the padded leaf table is, and how many tables and readers share it.
        separator
            .instance(&(self.num_variables as u64).to_be_bytes())
            .instance(&(self.table_variables.len() as u64).to_be_bytes())
            .instance(&(self.reader_variables.len() as u64).to_be_bytes());

        // Two statements can agree on every total above and still differ below.
        //
        // They may split readers between tables differently, or pull different columns.
        for width in &self.table_widths {
            separator.instance(&(*width as u64).to_be_bytes());
        }
        for num_variables in &self.reader_variables {
            separator.instance(&(*num_variables as u64).to_be_bytes());
        }

        separator
    }
}

/// Prover-side transcript of one reduction.
///
/// The challenger is borrowed, not consumed.
///
/// The reduction runs inside a proof whose transcript continues afterwards.
pub struct LogupStarProverTranscript<'a, C, F: TranscriptField, EF> {
    /// Driver walking the description and holding the borrowed sponge.
    state: ProverState<&'a mut C, Alphabet<F>>,
    /// Marker for the extension field the reduction runs over.
    _ef: PhantomData<EF>,
}

impl<'a, C, F, EF> LogupStarProverTranscript<'a, C, F, EF>
where
    F: TranscriptField,
    EF: ExtensionField<F>,
    C: CanObserve<F> + CanSample<F>,
{
    /// Seed the transcript from the shape.
    pub fn new(challenger: &'a mut C, shape: &LogupStarShape) -> Self {
        let separator = shape.domain_separator::<F, EF>();
        Self {
            state: ProverState::new(challenger, &separator),
            _ef: PhantomData,
        }
    }

    /// Draw the challenge weighting the readers of one table against each other.
    pub fn reader_batching(&mut self) -> EF {
        self.state
            .challenge_extension::<F, EF, FieldToFieldCodec<F>>(READER_BATCHING)
            .into_inner()
    }

    /// Bind one table's pushforward, tables in statement order.
    pub fn pushforward(&mut self, values: &[EF]) {
        self.state
            .observe_extensions::<F, EF, FieldToFieldCodec<F>>(PUSHFORWARD, values);
    }

    /// Draw the challenge each table's fraction identity is tested at.
    ///
    /// Zero is rejected because the first table entry always embeds to zero.
    ///
    /// A zero challenge would therefore certainly put a zero in a denominator.
    ///
    /// Landing on a later entry has probability at most the table size over the field size.
    ///
    /// The reduction's own root-denominator check catches that.
    pub fn entry_challenges(&mut self, num_tables: usize) -> Vec<EF> {
        self.state
            .challenge_extensions_rejecting::<F, EF, FieldToFieldCodec<F>>(
                ENTRY_CHALLENGES,
                num_tables,
                |candidate, _| !candidate.is_zero(),
            )
            .into_iter()
            .map(|bound| bound.into_inner())
            .collect()
    }

    /// Lend the sponge to the fraction reduction, bracketed as a sub-protocol.
    pub fn fraction_reduction<R>(&mut self, run: impl FnOnce(&mut C) -> R) -> R {
        self.state
            .begin_protocol::<FractionReduction>(FRACTION_REDUCTION);
        let output = run(self.state.challenger_mut());
        self.state
            .end_protocol::<FractionReduction>(FRACTION_REDUCTION);
        output
    }

    /// Bind every reader's position-column value at the reduction's output point.
    pub fn position_claims(&mut self, claims: &[EF]) {
        self.state
            .observe_extensions::<F, EF, FieldToFieldCodec<F>>(POSITION_CLAIMS, claims);
    }

    /// Draw the challenge combining a table's columns into one product claim.
    pub fn column_batching(&mut self) -> EF {
        self.state
            .challenge_extension::<F, EF, FieldToFieldCodec<F>>(COLUMN_BATCHING)
            .into_inner()
    }

    /// Lend the sponge to the product sumcheck, bracketed as a sub-protocol.
    pub fn product_sumcheck<R>(&mut self, run: impl FnOnce(&mut C) -> R) -> R {
        self.state
            .begin_protocol::<ProductSumcheck>(PRODUCT_SUMCHECK);
        let output = run(self.state.challenger_mut());
        self.state.end_protocol::<ProductSumcheck>(PRODUCT_SUMCHECK);
        output
    }

    /// Bind every table column's value at the product sumcheck's output point.
    pub fn column_claims(&mut self, claims: &[EF]) {
        self.state
            .observe_extensions::<F, EF, FieldToFieldCodec<F>>(COLUMN_CLAIMS, claims);
    }

    /// Close the transcript once every described step has been played.
    ///
    /// # Panics
    ///
    /// When the run played fewer steps than it was described with.
    pub fn finish(self) {
        assert!(
            self.state.finalize().is_empty(),
            "the reduction carries every value in its own proof",
        );
    }
}

/// Verifier-side transcript of one reduction.
///
/// Mirrors the prover side call for call, over the same description.
///
/// Every value comes from the proof rather than from a wire.
pub struct LogupStarVerifierTranscript<'a, C, F: TranscriptField, EF> {
    /// Driver walking the description and holding the borrowed sponge.
    state: VerifierState<'static, &'a mut C, Alphabet<F>>,
    /// Marker for the extension field the reduction runs over.
    _ef: PhantomData<EF>,
}

impl<'a, C, F, EF> LogupStarVerifierTranscript<'a, C, F, EF>
where
    F: TranscriptField,
    EF: ExtensionField<F>,
    C: CanObserve<F> + CanSample<F>,
{
    /// Seed the transcript from the shape, exactly as the prover seeded it.
    pub fn new(challenger: &'a mut C, shape: &LogupStarShape) -> Self {
        let separator = shape.domain_separator::<F, EF>();
        Self {
            state: VerifierState::new(challenger, &separator, &[]),
            _ef: PhantomData,
        }
    }

    /// Redraw the challenge weighting the readers of one table against each other.
    pub fn reader_batching(&mut self) -> EF {
        self.state
            .challenge_extension::<F, EF, FieldToFieldCodec<F>>(READER_BATCHING)
            .into_inner()
    }

    /// Replay one table's pushforward, tables in statement order.
    ///
    /// # Panics
    ///
    /// Panics if the value count differs from the one the shape describes.
    ///
    /// A caller measures the proof against the shape before seeding this transcript.
    ///
    /// A mismatch here is therefore a broken invariant, not a malformed proof.
    pub fn pushforward(&mut self, values: &[EF]) {
        self.state
            .observe_extensions::<F, EF, FieldToFieldCodec<F>>(PUSHFORWARD, values)
            .expect(MEASURED_FIRST);
    }

    /// Redraw the challenge each table's fraction identity is tested at.
    pub fn entry_challenges(&mut self, num_tables: usize) -> Vec<EF> {
        self.state
            .challenge_extensions_rejecting::<F, EF, FieldToFieldCodec<F>>(
                ENTRY_CHALLENGES,
                num_tables,
                |candidate, _| !candidate.is_zero(),
            )
            .into_iter()
            .map(|bound| bound.into_inner())
            .collect()
    }

    /// Lend the sponge to the fraction reduction, bracketed as a sub-protocol.
    ///
    /// The bracket closes whatever the delegated run returned.
    ///
    /// A rejection therefore leaves this transcript replayable to the end.
    pub fn fraction_reduction<R>(&mut self, run: impl FnOnce(&mut C) -> R) -> R {
        self.state
            .begin_protocol::<FractionReduction>(FRACTION_REDUCTION);
        let output = run(self.state.challenger_mut());
        self.state
            .end_protocol::<FractionReduction>(FRACTION_REDUCTION);
        output
    }

    /// Replay every reader's position-column value at the reduction's output point.
    ///
    /// # Panics
    ///
    /// Panics if the value count differs from the one the shape describes.
    pub fn position_claims(&mut self, claims: &[EF]) {
        self.state
            .observe_extensions::<F, EF, FieldToFieldCodec<F>>(POSITION_CLAIMS, claims)
            .expect(MEASURED_FIRST);
    }

    /// Redraw the challenge combining a table's columns into one product claim.
    pub fn column_batching(&mut self) -> EF {
        self.state
            .challenge_extension::<F, EF, FieldToFieldCodec<F>>(COLUMN_BATCHING)
            .into_inner()
    }

    /// Lend the sponge to the product sumcheck, bracketed as a sub-protocol.
    pub fn product_sumcheck<R>(&mut self, run: impl FnOnce(&mut C) -> R) -> R {
        self.state
            .begin_protocol::<ProductSumcheck>(PRODUCT_SUMCHECK);
        let output = run(self.state.challenger_mut());
        self.state.end_protocol::<ProductSumcheck>(PRODUCT_SUMCHECK);
        output
    }

    /// Replay every table column's value at the product sumcheck's output point.
    ///
    /// # Panics
    ///
    /// Panics if the value count differs from the one the shape describes.
    pub fn column_claims(&mut self, claims: &[EF]) {
        self.state
            .observe_extensions::<F, EF, FieldToFieldCodec<F>>(COLUMN_CLAIMS, claims)
            .expect(MEASURED_FIRST);
    }

    /// Close the transcript once every described step has been replayed.
    ///
    /// # Panics
    ///
    /// When the run replayed fewer steps than it was described with.
    pub fn finish(self) {
        assert!(
            self.state.finalize().is_ok(),
            "the reduction carries every value in its own proof",
        );
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_binary_field::{BinaryChallenger, BinaryField128};
    use p3_challenger::{CanSample, HashChallenger};
    use p3_field::PrimeCharacteristicRing;
    use p3_keccak::Keccak256Hash;
    use p3_multilinear_util::point::Point;

    use super::*;
    use crate::logup_star::{Reader, TableLookup};

    type B = BinaryField128;
    type Chal = BinaryChallenger<B, HashChallenger<u8, Keccak256Hash, 32>>;

    fn challenger() -> Chal {
        Chal::from_hasher(b"p3-logup-star-transcript-test".to_vec(), Keccak256Hash)
    }

    /// A shape over one table of the given size, read by readers of the given heights.
    fn shape(table_variables: usize, width: usize, readers: &[usize]) -> LogupStarShape {
        let points = readers
            .iter()
            .map(|&height| Point::new(vec![B::ONE; height]))
            .collect::<Vec<_>>();
        let claims = vec![B::ONE; width];
        let readers = points
            .iter()
            .map(|point| Reader {
                point,
                claims: &claims,
            })
            .collect::<Vec<_>>();
        LogupStarShape::new(&LogupStarPlan::new(&[TableLookup {
            num_variables: table_variables,
            readers: &readers,
        }]))
    }

    /// Play every step of one shape, on whichever side the two closures pick.
    ///
    /// The values are arbitrary; what is being compared is the sponge they leave behind.
    fn play(shape: &LogupStarShape, prover: bool) -> B {
        let mut sponge = challenger();
        let pushforwards = shape
            .table_variables
            .iter()
            .map(|&num_variables| vec![B::ONE; 1 << num_variables])
            .collect::<Vec<_>>();
        let positions = vec![B::ONE; shape.reader_variables.len()];
        let columns = vec![B::ONE; shape.num_column_claims()];

        if prover {
            let mut transcript = LogupStarProverTranscript::<Chal, B, B>::new(&mut sponge, shape);
            let _ = transcript.reader_batching();
            for pushforward in &pushforwards {
                transcript.pushforward(pushforward);
            }
            let _ = transcript.entry_challenges(shape.table_variables.len());
            transcript.fraction_reduction(|_| ());
            transcript.position_claims(&positions);
            let _ = transcript.column_batching();
            transcript.product_sumcheck(|_| ());
            transcript.column_claims(&columns);
            transcript.finish();
        } else {
            let mut transcript = LogupStarVerifierTranscript::<Chal, B, B>::new(&mut sponge, shape);
            let _ = transcript.reader_batching();
            for pushforward in &pushforwards {
                transcript.pushforward(pushforward);
            }
            let _ = transcript.entry_challenges(shape.table_variables.len());
            transcript.fraction_reduction(|_| ());
            transcript.position_claims(&positions);
            let _ = transcript.column_batching();
            transcript.product_sumcheck(|_| ());
            transcript.column_claims(&columns);
            transcript.finish();
        }

        sponge.sample()
    }

    #[test]
    fn both_sides_leave_the_sponge_in_one_state() {
        // Whatever runs after the reduction draws from this sponge.
        //
        // The two sides must hand back the same one or everything downstream diverges.
        let shape = shape(3, 2, &[4, 2]);
        assert_eq!(play(&shape, true), play(&shape, false));
    }

    #[test]
    fn a_shape_is_bound_before_anything_is_drawn() {
        // Two statements that differ anywhere must not share a challenge stream.
        //
        // Every number below moves the seed, through a step length or the instance label.
        //
        //     baseline : one table of 2^3 entries, 2 columns, readers of 2^4 and 2^2
        let baseline = play(&shape(3, 2, &[4, 2]), true);

        // Table size rides in the pushforward's own step length.
        assert_ne!(baseline, play(&shape(2, 2, &[4, 2]), true));

        // Column count rides in the instance label.
        assert_ne!(baseline, play(&shape(3, 1, &[4, 2]), true));

        // So does each reader's height.
        assert_ne!(baseline, play(&shape(3, 2, &[4, 3]), true));

        // Swapping two readers of one table changes which weight each one earns.
        assert_ne!(baseline, play(&shape(3, 2, &[2, 4]), true));

        // And so does adding a reader.
        assert_ne!(baseline, play(&shape(3, 2, &[4, 2, 2]), true));
    }

    #[test]
    fn the_entry_challenges_are_never_zero() {
        // The first table entry embeds to zero.
        //
        // A zero challenge would then zero a denominator with certainty, not negligibly.
        //
        // The driver refuses to be dropped mid-pattern, so the run plays out in full.
        let shape = shape(3, 1, &[2]);
        let mut sponge = challenger();
        let mut transcript = LogupStarProverTranscript::<Chal, B, B>::new(&mut sponge, &shape);

        let _ = transcript.reader_batching();
        transcript.pushforward(&[B::ONE; 8]);
        let challenges = transcript.entry_challenges(1);

        transcript.fraction_reduction(|_| ());
        transcript.position_claims(&[B::ONE]);
        let _ = transcript.column_batching();
        transcript.product_sumcheck(|_| ());
        transcript.column_claims(&[B::ONE]);
        transcript.finish();

        assert_eq!(challenges.len(), 1);
        assert_ne!(challenges[0], B::ZERO);
    }
}
