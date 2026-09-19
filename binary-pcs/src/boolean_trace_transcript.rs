//! Typed transcript for batched Boolean trace column openings.

use alloc::vec::Vec;

use p3_challenger::fs::{
    DomainSeparator, FieldToFieldCodec, FieldUnit, Hierarchy, Interaction, InteractionPattern,
    Kind, Length, ProverState, TranscriptField, VerifierState,
};
use p3_challenger::{CanObserve, CanSample, CanSampleUniformBits, GrindingChallenger};
use p3_field::{AlgebraIdentity, Field};
use p3_multilinear_util::point::Point;

const VERSION: u8 = 1;
const NAME: &[u8] = b"p3-boolean-trace-column-batching";
const POINT: &str = "row_point";
const VALUES: &str = "column_values";
const NEXT_VALUES: &str = "next_column_values";
const COLUMN_POINT: &str = "column_point";
const SUCCESSOR_LABEL: &[u8] = b"successor";

type Alphabet<F> = FieldUnit<F>;

/// Public shape of one optimized column-batched opening run.
///
/// Every batch claims the table's whole width at its own row point, and, when `next`
/// holds, the same width one row ahead. Both value runs are bound before the batch
/// draws the column point that combines them.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ColumnBatchShape {
    pub(crate) table_variables: usize,
    pub(crate) width: usize,
    pub(crate) num_batches: usize,
    pub(crate) next: bool,
}

impl ColumnBatchShape {
    const fn column_variables(self) -> usize {
        self.width.next_power_of_two().trailing_zeros() as usize
    }

    fn pattern<F, EF>(self) -> InteractionPattern
    where
        F: TranscriptField,
        EF: Field + AlgebraIdentity<F>,
    {
        let per_batch = 3 + usize::from(self.next);
        let mut steps = Vec::with_capacity(self.num_batches * per_batch);
        for _ in 0..self.num_batches {
            steps.push(Interaction::algebra::<F, EF>(
                Hierarchy::Atomic,
                Kind::Message,
                POINT,
                Length::Fixed(self.table_variables),
            ));
            steps.push(Interaction::algebra::<F, EF>(
                Hierarchy::Atomic,
                Kind::Message,
                VALUES,
                Length::Fixed(self.width),
            ));
            if self.next {
                steps.push(Interaction::algebra::<F, EF>(
                    Hierarchy::Atomic,
                    Kind::Message,
                    NEXT_VALUES,
                    Length::Fixed(self.width),
                ));
            }
            steps.push(Interaction::algebra::<F, F>(
                Hierarchy::Atomic,
                Kind::Challenge,
                COLUMN_POINT,
                Length::Fixed(self.column_variables()),
            ));
        }
        InteractionPattern::new(steps).expect("flat column-batching transcript is valid")
    }

    fn separator<F, EF>(self) -> DomainSeparator<Alphabet<F>>
    where
        F: TranscriptField,
        EF: Field + AlgebraIdentity<F>,
    {
        let mut separator = DomainSeparator::new(VERSION, NAME, self.pattern::<F, EF>());
        separator
            .instance(&(self.table_variables as u64).to_be_bytes())
            .instance(&(self.width as u64).to_be_bytes())
            .instance(&(self.num_batches as u64).to_be_bytes());
        // A run reading one row ahead carries its own label, so the two seeds part.
        if self.next {
            separator.instance(SUCCESSOR_LABEL);
        }
        separator
    }
}

/// Prover-side driver for one optimized opening run.
pub(crate) struct ColumnBatchProverTranscript<'a, F: TranscriptField, C> {
    state: ProverState<&'a mut C, Alphabet<F>>,
    shape: ColumnBatchShape,
}

impl<'a, F, C> ColumnBatchProverTranscript<'a, F, C>
where
    F: Field + TranscriptField + AlgebraIdentity<F>,
    C: CanObserve<F> + CanSample<F> + CanSampleUniformBits<F> + GrindingChallenger<Witness = F>,
{
    pub(crate) fn new(challenger: &'a mut C, shape: ColumnBatchShape) -> Self {
        let separator = shape.separator::<F, F>();
        Self {
            state: ProverState::new(challenger, &separator),
            shape,
        }
    }

    pub(crate) fn batch(&mut self, point: &Point<F>, values: &[F], next: &[F]) -> Point<F> {
        debug_assert_eq!(point.num_variables(), self.shape.table_variables);
        debug_assert_eq!(values.len(), self.shape.width);
        debug_assert_eq!(next.len(), self.shape.width * usize::from(self.shape.next));
        self.state
            .observe_extensions::<F, F, FieldToFieldCodec<F>>(POINT, point.as_slice());
        self.state
            .observe_extensions::<F, F, FieldToFieldCodec<F>>(VALUES, values);
        if self.shape.next {
            self.state
                .observe_extensions::<F, F, FieldToFieldCodec<F>>(NEXT_VALUES, next);
        }
        let coordinates = self
            .state
            .challenge_scalars::<F, FieldToFieldCodec<F>>(
                COLUMN_POINT,
                self.shape.column_variables(),
            )
            .into_iter()
            .map(|coordinate| coordinate.into_inner());
        Point::new(coordinates.collect())
    }

    pub(crate) fn finish(self) {
        assert!(
            self.state.finalize().is_empty(),
            "column-batching transcript carries no wire values",
        );
    }
}

/// Verifier-side driver for one optimized opening run.
pub(crate) struct ColumnBatchVerifierTranscript<'a, F: TranscriptField, C> {
    state: VerifierState<'static, &'a mut C, Alphabet<F>>,
    shape: ColumnBatchShape,
}

impl<'a, F, C> ColumnBatchVerifierTranscript<'a, F, C>
where
    F: Field + TranscriptField + AlgebraIdentity<F>,
    C: CanObserve<F> + CanSample<F> + CanSampleUniformBits<F> + GrindingChallenger<Witness = F>,
{
    pub(crate) fn new(challenger: &'a mut C, shape: ColumnBatchShape) -> Self {
        let separator = shape.separator::<F, F>();
        Self {
            state: VerifierState::new(challenger, &separator, &[]),
            shape,
        }
    }

    pub(crate) fn batch(
        &mut self,
        point: &Point<F>,
        values: &[F],
        next: &[F],
    ) -> Result<Point<F>, p3_challenger::fs::TranscriptError> {
        debug_assert_eq!(point.num_variables(), self.shape.table_variables);
        debug_assert_eq!(values.len(), self.shape.width);
        debug_assert_eq!(next.len(), self.shape.width * usize::from(self.shape.next));
        self.state
            .observe_extensions::<F, F, FieldToFieldCodec<F>>(POINT, point.as_slice())?;
        self.state
            .observe_extensions::<F, F, FieldToFieldCodec<F>>(VALUES, values)?;
        if self.shape.next {
            self.state
                .observe_extensions::<F, F, FieldToFieldCodec<F>>(NEXT_VALUES, next)?;
        }
        let coordinates = self
            .state
            .challenge_scalars::<F, FieldToFieldCodec<F>>(
                COLUMN_POINT,
                self.shape.column_variables(),
            )
            .into_iter()
            .map(|coordinate| coordinate.into_inner());
        Ok(Point::new(coordinates.collect()))
    }

    pub(crate) fn finish(self) {
        self.state
            .finalize()
            .expect("column-batching transcript is complete");
    }
}

#[cfg(test)]
mod tests {
    use alloc::format;

    use p3_binary_field::BinaryField128;
    use p3_challenger::testing::{SeedDigest, assert_seeds_pairwise_distinct, seed_digest};

    use super::*;

    type EF = BinaryField128;

    /// Two batches over a table of 2^8 rows and three columns, read at the chosen views.
    const fn shape(next: bool) -> ColumnBatchShape {
        ColumnBatchShape {
            table_variables: 8,
            width: 3,
            num_batches: 2,
            next,
        }
    }

    /// One batch over a table of 2^9 rows and five columns, read at the chosen views.
    const fn wider_shape(next: bool) -> ColumnBatchShape {
        ColumnBatchShape {
            table_variables: 9,
            width: 5,
            num_batches: 1,
            next,
        }
    }

    /// The labels the batches of one shape play, in order.
    fn labels(shape: ColumnBatchShape) -> Vec<&'static str> {
        shape
            .separator::<EF, EF>()
            .pattern()
            .interactions()
            .iter()
            .map(Interaction::label)
            .collect()
    }

    /// The whole byte stream one shape seeds its sponge with, as a comparable digest.
    fn seed_of(shape: ColumnBatchShape) -> SeedDigest {
        seed_digest(&shape.separator::<EF, EF>())
    }

    #[test]
    fn a_current_row_run_is_labelled_by_its_three_numbers() {
        // Invariant: a run reading only the current row binds the shape numbers and nothing
        // else, and each of its batches sends a point and its values before drawing.
        let separator = shape(false).separator::<EF, EF>();
        let mut expected = Vec::new();
        for number in [8u64, 3, 2] {
            expected.extend_from_slice(&8u32.to_be_bytes());
            expected.extend_from_slice(&number.to_be_bytes());
        }
        assert_eq!(separator.instance_label(), expected.as_slice());
        assert_eq!(
            labels(shape(false)),
            [POINT, VALUES, COLUMN_POINT, POINT, VALUES, COLUMN_POINT]
        );
    }

    #[test]
    fn a_current_row_run_seeds_the_bytes_its_literals_name() {
        // Invariant: the whole seed of a current-row run is pinned, not only the instance
        // numbers and the step order.
        //
        //     2^8 rows, width 3, two batches
        //     2^9 rows, width 5, one batch
        //
        // Both halves of the seed reach the digest, so a literal moves exactly when the
        // seeded bytes move. Nothing that leaves the current-row protocol alone does that.
        assert_eq!(
            format!("{:?}", seed_of(shape(false))),
            "2043da134342cdc72a65bfcdb8cd9f7eff45bbeb127fbdd4793f39c23bea23dd",
        );
        assert_eq!(
            format!("{:?}", seed_of(wider_shape(false))),
            "90b5362da1b29b2acd186db9b8ad1cdea5f61298006f88e48aec7beb4ef06ce9",
        );
    }

    #[test]
    fn a_successor_run_seeds_a_transcript_of_its_own() {
        // Invariant: the two runs differ in both halves of the seed, so neither replays the
        // other's challenges, and the successor values are sent before the point is drawn.
        let current = shape(false).separator::<EF, EF>();
        let successor = shape(true).separator::<EF, EF>();
        assert_ne!(current.instance_label(), successor.instance_label());
        assert_ne!(
            current.pattern().pattern_hash(),
            successor.pattern().pattern_hash()
        );
        assert_eq!(
            labels(shape(true)),
            [
                POINT,
                VALUES,
                NEXT_VALUES,
                COLUMN_POINT,
                POINT,
                VALUES,
                NEXT_VALUES,
                COLUMN_POINT
            ]
        );

        // No two of the four runs share a seed, so none of them replays another's draws.
        assert_seeds_pairwise_distinct(&[
            ("current", seed_of(shape(false))),
            ("successor", seed_of(shape(true))),
            ("wider current", seed_of(wider_shape(false))),
            ("wider successor", seed_of(wider_shape(true))),
        ]);
    }
}
