//! Fiat-Shamir transcript of one indexed-lookup reduction.
//!
//! # Shape
//!
//! ```text
//!     statement                    every claim point and claimed value
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
//! What is being proved is bound before anything is drawn.
//!
//! Every challenge below weighs the claims.
//!
//! A prover that saw one first could choose claims the weighing cancels.
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
use p3_field::{BasedVectorSpace, ExtensionField};

use super::plan::LogupStarPlan;

/// Version byte bound into the transcript seed.
const VERSION: u8 = 1;

/// Protocol name bound into the transcript seed.
const NAME: &[u8] = b"p3-multi-stark-logup-star";

/// Step label of the claim points and claimed values the reduction is about.
const STATEMENT: &str = "statement";

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

/// Numbers that fix one table's part of the transcript.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LogupStarTableShape {
    /// Base-two logarithm of the number of table entries.
    pub num_variables: usize,
    /// Number of columns each entry carries.
    pub width: usize,
    /// Base-two logarithm of each reader's row count, in the table's reader order.
    pub readers: Vec<usize>,
}

/// Numbers that fix the transcript of one reduction.
///
/// Both sides read these off their own copy of the statement.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LogupStarShape {
    /// One entry per table, in statement order.
    pub tables: Vec<LogupStarTableShape>,
    /// Variable count of the padded leaf table the fraction reduction consumes.
    pub num_variables: usize,
}

impl LogupStarShape {
    /// Read the shape off a layout.
    #[must_use]
    pub fn new(plan: &LogupStarPlan) -> Self {
        Self {
            tables: plan
                .tables
                .iter()
                .map(|table| LogupStarTableShape {
                    num_variables: table.num_variables,
                    width: table.width,
                    readers: table.readers.clone(),
                })
                .collect(),
            num_variables: plan.num_variables,
        }
    }

    /// Total number of column claims the reduction closes on.
    fn num_column_claims(&self) -> usize {
        self.tables.iter().map(|table| table.width).sum()
    }

    /// Number of readers across every table.
    fn num_readers(&self) -> usize {
        self.tables.iter().map(|table| table.readers.len()).sum()
    }

    /// Number of extension values that fix what is being proved.
    ///
    /// A reader contributes its claim point and one claimed value per table column.
    fn num_statement_values(&self) -> usize {
        self.tables
            .iter()
            .flat_map(|table| {
                table
                    .readers
                    .iter()
                    .map(move |&height| height + table.width)
            })
            .sum()
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

        // What is being proved is bound before anything is drawn.
        //
        // Without it a prover could pick claims after seeing the challenges that weigh them.
        let mut steps = vec![
            Interaction::algebra::<F, F>(
                Hierarchy::Atomic,
                Kind::Public,
                STATEMENT,
                Length::Fixed(self.num_statement_values() * <EF as BasedVectorSpace<F>>::DIMENSION),
            ),
            challenge(READER_BATCHING, Length::Scalar),
        ];

        // One pushforward per table, each as wide as the table it summarizes.
        //
        // Widths differ between statements.
        //
        // They reach the fingerprint through the step lengths, not the instance label.
        steps.extend(
            self.tables
                .iter()
                .map(|table| message(PUSHFORWARD, Length::Fixed(1 << table.num_variables))),
        );

        // Every pushforward is bound before any challenge that tests one is drawn.
        steps.push(challenge(
            ENTRY_CHALLENGES,
            Length::Fixed(self.tables.len()),
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
        steps.push(message(POSITION_CLAIMS, Length::Fixed(self.num_readers())));

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
            .instance(&(self.tables.len() as u64).to_be_bytes())
            .instance(&(self.num_readers() as u64).to_be_bytes());

        // Two statements can agree on every total above and still differ below.
        //
        // They may split readers between tables differently, or pull different columns.
        // Each table's reader count precedes its own reader heights.
        //
        // Two statements that merely split the same readers differently part here.
        for table in &self.tables {
            separator
                .instance(&(table.width as u64).to_be_bytes())
                .instance(&(table.readers.len() as u64).to_be_bytes());
            for height in &table.readers {
                separator.instance(&(*height as u64).to_be_bytes());
            }
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

    /// Bind what is being proved, before anything is drawn from the sponge.
    ///
    /// Both sides hold these already, so they are absorbed rather than sent.
    ///
    /// # Arguments
    ///
    /// - `statement`: every reader's claim point and claimed values, in statement order.
    pub fn statement(&mut self, statement: &[EF]) {
        self.state.add_public_scalars::<F, FieldToFieldCodec<F>>(
            STATEMENT,
            &EF::flatten_to_base(statement.to_vec()),
        );
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

    /// Bind what is being proved, exactly as the prover bound it.
    pub fn statement(&mut self, statement: &[EF]) {
        self.state
            .observe_public_scalars::<F, FieldToFieldCodec<F>>(
                STATEMENT,
                &EF::flatten_to_base(statement.to_vec()),
            );
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

    /// One table of the given size and width, read by readers of the given heights.
    fn table(num_variables: usize, width: usize, readers: &[usize]) -> LogupStarTableShape {
        LogupStarTableShape {
            num_variables,
            width,
            readers: readers.to_vec(),
        }
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
        LogupStarShape::new(&LogupStarPlan::new::<B, B>(&[TableLookup {
            num_variables: table_variables,
            readers: &readers,
        }]))
    }

    /// Play every step of one shape, on whichever side the two closures pick.
    ///
    /// The values are arbitrary; what is being compared is the sponge they leave behind.
    fn play(shape: &LogupStarShape, prover: bool) -> B {
        play_with(shape, prover, B::ONE, B::ONE)
    }

    /// Play every step, with one statement value and one pushforward entry as given.
    ///
    /// Returns what the next draw off the shared sponge would be.
    fn play_with(shape: &LogupStarShape, prover: bool, claim: B, pushforward: B) -> B {
        let mut sponge = challenger();
        let mut pushforwards = shape
            .tables
            .iter()
            .map(|table| vec![B::ONE; 1 << table.num_variables])
            .collect::<Vec<_>>();
        pushforwards[0][0] = pushforward;
        let positions = vec![B::ONE; shape.num_readers()];
        let mut statement = vec![B::ONE; shape.num_statement_values()];
        statement[0] = claim;
        let columns = vec![B::ONE; shape.num_column_claims()];

        if prover {
            let mut transcript = LogupStarProverTranscript::<Chal, B, B>::new(&mut sponge, shape);
            transcript.statement(&statement);
            let _ = transcript.reader_batching();
            for pushforward in &pushforwards {
                transcript.pushforward(pushforward);
            }
            let _ = transcript.entry_challenges(shape.tables.len());
            transcript.fraction_reduction(|_| ());
            transcript.position_claims(&positions);
            let _ = transcript.column_batching();
            transcript.product_sumcheck(|_| ());
            transcript.column_claims(&columns);
            transcript.finish();
        } else {
            let mut transcript = LogupStarVerifierTranscript::<Chal, B, B>::new(&mut sponge, shape);
            transcript.statement(&statement);
            let _ = transcript.reader_batching();
            for pushforward in &pushforwards {
                transcript.pushforward(pushforward);
            }
            let _ = transcript.entry_challenges(shape.tables.len());
            transcript.fraction_reduction(|_| ());
            transcript.position_claims(&positions);
            let _ = transcript.column_batching();
            transcript.product_sumcheck(|_| ());
            transcript.column_claims(&columns);
            transcript.finish();
        }

        sponge.sample()
    }

    /// Play out the steps below the entry challenges, which the driver insists on.
    fn finish(mut transcript: LogupStarProverTranscript<'_, Chal, B, B>, shape: &LogupStarShape) {
        transcript.fraction_reduction(|_| ());
        transcript.position_claims(&vec![B::ONE; shape.num_readers()]);
        let _ = transcript.column_batching();
        transcript.product_sumcheck(|_| ());
        transcript.column_claims(&vec![B::ONE; shape.num_column_claims()]);
        transcript.finish();
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
    fn a_claim_reaches_the_sponge_before_anything_is_drawn() {
        // Every challenge below weighs the claims.
        //
        // A prover seeing one before fixing them could pick claims the weighing cancels.
        //
        // Binding the statement first is what stops that, and this is what binding means:
        // one different claimed value moves the whole stream.
        let shape = shape(3, 1, &[2, 2]);
        assert_ne!(
            play_with(&shape, true, B::ONE, B::ONE),
            play_with(&shape, true, B::ZERO, B::ONE)
        );
    }

    #[test]
    fn a_pushforward_reaches_the_sponge_before_the_entry_challenges() {
        // An unbound pushforward could be chosen after its own challenge.
        //
        // A prover would then face one linear constraint per table instead of a commitment.
        //
        // Fixture state: two runs differing only in entry zero of the first pushforward.
        let shape = shape(3, 1, &[2]);
        let mut sponge = challenger();
        let mut transcript = LogupStarProverTranscript::<Chal, B, B>::new(&mut sponge, &shape);
        transcript.statement(&vec![B::ONE; shape.num_statement_values()]);
        let _ = transcript.reader_batching();
        let mut pushforward = [B::ONE; 8];
        transcript.pushforward(&pushforward);
        let bound = transcript.entry_challenges(1);
        finish(transcript, &shape);

        let mut sponge = challenger();
        let mut transcript = LogupStarProverTranscript::<Chal, B, B>::new(&mut sponge, &shape);
        transcript.statement(&vec![B::ONE; shape.num_statement_values()]);
        let _ = transcript.reader_batching();
        pushforward[0] = B::ZERO;
        transcript.pushforward(&pushforward);
        let moved = transcript.entry_challenges(1);
        finish(transcript, &shape);

        assert_ne!(bound, moved);
    }

    #[test]
    fn each_table_draws_its_own_entry_challenge() {
        // One shared challenge would let two tables miscount opposite entries and cancel.
        //
        // The draw has to separate them.
        let shape = shape(3, 1, &[2]);
        let two_tables = LogupStarShape {
            tables: vec![table(3, 1, &[2]), table(3, 1, &[2])],
            num_variables: shape.num_variables,
        };

        let mut sponge = challenger();
        let mut transcript = LogupStarProverTranscript::<Chal, B, B>::new(&mut sponge, &two_tables);
        transcript.statement(&vec![B::ONE; two_tables.num_statement_values()]);
        let _ = transcript.reader_batching();
        transcript.pushforward(&[B::ONE; 8]);
        transcript.pushforward(&[B::ONE; 8]);
        let challenges = transcript.entry_challenges(2);
        finish(transcript, &two_tables);

        assert_eq!(challenges.len(), 2);
        assert_ne!(challenges[0], challenges[1]);
    }

    #[test]
    fn how_readers_split_between_tables_reaches_the_seed() {
        // Two tables of one size and width, read by equal readers, can still differ.
        //
        // Which table each reader belongs to is what differs.
        //
        //     [2 readers | 1 reader]   versus   [1 reader | 2 readers]
        //
        // Totals alone do not separate those.
        //
        // Each table's reader count is bound with its heights rather than after them.
        let left = LogupStarShape {
            tables: vec![table(3, 1, &[2, 2]), table(3, 1, &[2])],
            num_variables: 5,
        };
        let right = LogupStarShape {
            tables: vec![table(3, 1, &[2]), table(3, 1, &[2, 2])],
            num_variables: 5,
        };

        assert_ne!(left, right);
        assert_ne!(play(&left, true), play(&right, true));
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

        transcript.statement(&vec![B::ONE; shape.num_statement_values()]);
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
