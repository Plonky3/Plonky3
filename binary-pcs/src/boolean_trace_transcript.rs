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
const COLUMN_POINT: &str = "column_point";

type Alphabet<F> = FieldUnit<F>;

/// Public shape of one optimized column-batched opening run.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ColumnBatchShape {
    pub(crate) table_variables: usize,
    pub(crate) width: usize,
    pub(crate) num_batches: usize,
}

impl ColumnBatchShape {
    fn column_variables(self) -> usize {
        self.width.next_power_of_two().trailing_zeros() as usize
    }

    fn pattern<F, EF>(self) -> InteractionPattern
    where
        F: TranscriptField,
        EF: Field + AlgebraIdentity<F>,
    {
        let mut steps = Vec::with_capacity(self.num_batches * 3);
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

    pub(crate) fn batch(&mut self, point: &Point<F>, values: &[F]) -> Point<F> {
        debug_assert_eq!(point.num_variables(), self.shape.table_variables);
        debug_assert_eq!(values.len(), self.shape.width);
        self.state
            .observe_extensions::<F, F, FieldToFieldCodec<F>>(POINT, point.as_slice());
        self.state
            .observe_extensions::<F, F, FieldToFieldCodec<F>>(VALUES, values);
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
    ) -> Result<Point<F>, p3_challenger::fs::TranscriptError> {
        debug_assert_eq!(point.num_variables(), self.shape.table_variables);
        debug_assert_eq!(values.len(), self.shape.width);
        self.state
            .observe_extensions::<F, F, FieldToFieldCodec<F>>(POINT, point.as_slice())?;
        self.state
            .observe_extensions::<F, F, FieldToFieldCodec<F>>(VALUES, values)?;
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
