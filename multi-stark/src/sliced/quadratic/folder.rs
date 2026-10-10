//! The folder that evaluates AIR constraints as [`SlicedQuadratic`] values, and the lane sums it
//! weights them with.

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;

use p3_air::{Air, AirBuilder, RowWindow};
use p3_bus::{BusActivation, BusDirection, BusInteractionRecorder, BusName, RecordToken};
use p3_field::Field;
use p3_lookup::{Count, IndexedLookupBuilder, InteractionBuilder, TraceWindow};

use super::{CellWords, SLICED_CELLS, SlicedBit, SlicedQuadratic, xor, zip};
use crate::folder::eval_boundary_io;
use crate::selectors::BoundaryEvals;
use crate::sliced::{
    PreparedPowers, PreparedSums, SLICED_LANES, SlicedEvaluation, TABLE_BITS, TABLE_ENTRIES,
    TABLES_PER_PLANE,
};

/// Byte-indexed partial sums of sixty-four lane weights, for values with one bit per lane.
///
/// ```text
///     sum_lane w(lane) * bit(lane)  =  sum_byte table[byte][bits' byte]
///
///     table[byte][m] = sum of w(8 byte + i) over the set bits i of m
/// ```
#[derive(Clone, Debug)]
pub struct BitLaneSums<R> {
    /// One table per byte of a word.
    tables: Box<[[R; TABLE_ENTRIES]; TABLES_PER_PLANE]>,
}

impl<R: Field> BitLaneSums<R> {
    /// Tabulate the partial sums of `weights`.
    ///
    /// # Panics
    ///
    /// Panics if `weights` does not hold one weight per lane.
    #[must_use]
    pub fn new(weights: &[R]) -> Self {
        assert_eq!(weights.len(), SLICED_LANES, "one weight per lane");
        let mut tables = vec![[R::ZERO; TABLE_ENTRIES]; TABLES_PER_PLANE];
        for (table, weights) in tables.iter_mut().zip(weights.as_chunks::<TABLE_BITS>().0) {
            // Each mask adds its lowest set lane to the mask without it.
            for mask in 1..TABLE_ENTRIES {
                table[mask] = table[mask & (mask - 1)] + weights[mask.trailing_zeros() as usize];
            }
        }
        let tables = tables
            .into_boxed_slice()
            .try_into()
            .unwrap_or_else(|_| unreachable!("the table count is fixed"));
        Self { tables }
    }

    /// The lane-weighted sum of the lanes whose bit is set in `bits`.
    #[inline]
    #[must_use]
    pub fn sum(&self, bits: u64) -> R {
        let mut sum = R::ZERO;
        for (byte, table) in self.tables.iter().enumerate() {
            sum += table[usize::from((bits >> (byte * TABLE_BITS)) as u8)];
        }
        sum
    }
}

/// The kernel's running sums for every cell of one evaluation.
#[derive(Debug)]
struct KernelSums<'a, R> {
    /// The alpha powers laid out for the kernel.
    prepared: &'a PreparedPowers<R>,
    /// Per-cell sums of the constraints asserted so far.
    sums: [PreparedSums; SLICED_CELLS],
}

/// AIR folder over [`SLICED_CELLS`] evaluations of sixty-four rows, each constraint reduced to one
/// bit per lane.
///
/// In each evaluation every asserted constraint contributes its quadratic part, or its whole
/// value where the folder is built for whole values. The bits are summed across the lanes with
/// the weights of a [`BitLaneSums`], then weighted by the constraint's descending alpha power, one
/// sum per evaluation. Lookup declarations and bus tuples are dropped, and a bus declaration's
/// Booleanity check is asserted, as in [`SlicedFolder`](crate::sliced::SlicedFolder).
#[derive(Debug)]
pub struct SlicedQuadraticFolder<'a, F, R> {
    /// Two-row main window holding the current and shifted-by-one rows.
    main_window: RowWindow<'a, SlicedBit<F>>,
    /// Two-row preprocessed window; zero-width when the AIR has no preprocessed columns.
    preprocessed_window: RowWindow<'a, SlicedBit<F>>,
    /// Periodic column values, one per declared periodic column.
    periodic_values: &'a [SlicedBit<F>],
    /// Boundary-selector values shared by all selector accessors.
    boundary: BoundaryEvals<SlicedBit<F>>,
    /// Public inputs forwarded to the AIR, always in the trace field.
    public_values: &'a [F],
    /// Descending alpha powers, one per asserted constraint, boundary pins included.
    alpha_powers: &'a [R],
    /// The lane weights every constraint is summed with.
    lanes: &'a BitLaneSums<R>,
    /// Per evaluation, every bit set when a constraint contributes its whole value, clear for its
    /// quadratic part.
    whole: CellWords,
    /// Running lane-weighted, alpha-batched sums, one per evaluation.
    ///
    /// Beside the kernel only debug builds keep these, to check the kernel's sums against them.
    accumulators: [R; SLICED_CELLS],
    /// The kernel's sums, when the kernel sums the constraints instead.
    kernel: Option<KernelSums<'a, R>>,
    /// Number of constraints asserted so far, which is the next position in `alpha_powers`.
    constraint_index: usize,
    /// Whether any asserted value was poisoned.
    poisoned: bool,
}

impl<'a, F: Field, R: Field> SlicedQuadraticFolder<'a, F, R> {
    /// Build a folder for [`SLICED_CELLS`] evaluations over sixty-four rows each.
    ///
    /// # Arguments
    ///
    /// - `local`, `next`: main column inputs at the current and shifted-by-one rows.
    /// - `boundary`: selector inputs at the same rows.
    /// - `public_values`: public inputs forwarded to the AIR.
    /// - `alpha_powers`: `alpha^(n - 1 - i)` for each of the `n` constraints, pins included.
    /// - `lanes`: the weights summing each constraint across the lanes.
    /// - `whole`: per evaluation, whether each constraint contributes its whole value rather than
    ///   its quadratic part.
    ///
    /// # Panics
    ///
    /// Panics unless `F` has characteristic two, where the lanes' parts add as bits.
    #[inline]
    #[must_use]
    pub fn new(
        local: &'a [SlicedBit<F>],
        next: &'a [SlicedBit<F>],
        boundary: BoundaryEvals<SlicedBit<F>>,
        public_values: &'a [F],
        alpha_powers: &'a [R],
        lanes: &'a BitLaneSums<R>,
        whole: [bool; SLICED_CELLS],
    ) -> Self {
        assert!(
            F::TWO == F::ZERO,
            "sliced GF(2) parts add in characteristic two"
        );
        Self {
            main_window: RowWindow::from_two_rows(local, next),
            preprocessed_window: RowWindow::from_two_rows(&[], &[]),
            periodic_values: &[],
            boundary,
            public_values,
            alpha_powers,
            lanes,
            whole: whole.map(|whole| u64::from(whole).wrapping_neg()),
            accumulators: [R::ZERO; SLICED_CELLS],
            kernel: None,
            constraint_index: 0,
            poisoned: false,
        }
    }

    /// Sum the constraints with the kernel, from `prepared`, the layout of the alpha powers.
    ///
    /// # Panics
    ///
    /// Panics if `prepared` does not hold one power per alpha power.
    #[inline]
    #[must_use]
    pub(crate) fn with_prepared_powers(mut self, prepared: &'a PreparedPowers<R>) -> Self {
        assert_eq!(
            prepared.0.len(),
            self.alpha_powers.len(),
            "the prepared powers must be the attached alpha powers"
        );
        self.kernel = Some(KernelSums {
            prepared,
            sums: core::array::from_fn(|_| PreparedSums::new()),
        });
        self
    }

    /// Attach the two-row preprocessed window read by the AIR.
    #[inline]
    #[must_use]
    pub fn with_preprocessed(
        mut self,
        current: &'a [SlicedBit<F>],
        next: &'a [SlicedBit<F>],
    ) -> Self {
        self.preprocessed_window = RowWindow::from_two_rows(current, next);
        self
    }

    /// Attach the periodic column values read by the AIR.
    #[inline]
    #[must_use]
    pub const fn with_periodic(mut self, values: &'a [SlicedBit<F>]) -> Self {
        self.periodic_values = values;
        self
    }

    /// Run the AIR, then its public boundary pins, and return the batched sum of each evaluation.
    ///
    /// # Panics
    ///
    /// Panics if the alpha powers do not number one per asserted constraint.
    #[inline]
    #[must_use]
    pub fn eval_air<A>(mut self, air: &A) -> SlicedEvaluation<[R; SLICED_CELLS]>
    where
        A: Air<Self>,
        Self: AirBuilder,
    {
        air.eval(&mut self);
        eval_boundary_io(&mut self, air.public_boundary_io());
        assert_eq!(
            self.constraint_index,
            self.alpha_powers.len(),
            "attached alpha powers must match the number of asserted constraints"
        );
        let value = match &mut self.kernel {
            Some(kernel) => {
                let mut value = [R::ZERO; SLICED_CELLS];
                for (value, sums) in value.iter_mut().zip(&mut kernel.sums) {
                    *value = sums.finish_bits(kernel.prepared, self.lanes);
                }
                debug_assert_eq!(
                    value, self.accumulators,
                    "the kernel must sum what the lane tables sum"
                );
                value
            }
            None => self.accumulators,
        };
        SlicedEvaluation {
            value,
            poisoned: self.poisoned,
        }
    }

    /// Add the lane-weighted sum of each evaluation's bits, scaled by `power`, to its running sum.
    ///
    /// A word that vanishes on every lane adds nothing, as a selector-gated one mostly does.
    #[inline]
    fn accumulate(&mut self, power: R, bits: CellWords) {
        if self.kernel.is_none() || cfg!(debug_assertions) {
            for (accumulator, bits) in self.accumulators.iter_mut().zip(bits) {
                if bits != 0 {
                    *accumulator += power * self.lanes.sum(bits);
                }
            }
        }
        if let Some(kernel) = &mut self.kernel {
            for (sums, bits) in kernel.sums.iter_mut().zip(bits) {
                sums.add_bits(kernel.prepared, self.constraint_index, bits);
            }
        }
    }
}

impl<'a, F, R> AirBuilder for SlicedQuadraticFolder<'a, F, R>
where
    F: Field,
    R: Field,
{
    type F = F;
    type Expr = SlicedQuadratic<F>;
    type Var = SlicedBit<F>;
    type MainWindow = RowWindow<'a, SlicedBit<F>>;
    type PreprocessedWindow = RowWindow<'a, SlicedBit<F>>;
    // Public values stay in the trace field and narrow into the lanes on read.
    type PublicVar = F;
    type PeriodicVar = SlicedBit<F>;

    #[inline]
    fn main(&self) -> Self::MainWindow {
        self.main_window
    }

    #[inline]
    fn preprocessed(&self) -> &Self::PreprocessedWindow {
        &self.preprocessed_window
    }

    #[inline]
    fn is_first_row(&self) -> Self::Expr {
        self.boundary.first.into()
    }

    #[inline]
    fn is_last_row(&self) -> Self::Expr {
        self.boundary.last.into()
    }

    #[inline]
    fn is_transition(&self) -> Self::Expr {
        self.boundary.transition.into()
    }

    #[inline(always)]
    fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I) {
        let x = x.into();
        self.poisoned |= x.poisoned;
        let rest = zip(x.linear, self.whole, |linear, whole| {
            (linear ^ x.constant) & whole
        });
        let bits = xor(x.quadratic, rest);
        // A constraint past the last power is only counted; the count check rejects it.
        if let Some(&power) = self.alpha_powers.get(self.constraint_index) {
            self.accumulate(power, bits);
        }
        self.constraint_index += 1;
    }

    #[inline]
    fn public_values(&self) -> &[Self::PublicVar] {
        self.public_values
    }

    #[inline]
    fn periodic_values(&self) -> &[Self::PeriodicVar] {
        self.periodic_values
    }
}

/// Lookup declarations are dropped: a stage that declares lookups never runs this folder.
impl<'a, F, R> InteractionBuilder for SlicedQuadraticFolder<'a, F, R>
where
    F: Field,
    R: Field,
{
    fn push_interaction<E: Into<Self::Expr>>(
        &mut self,
        _bus_name: &str,
        _fields: impl IntoIterator<Item = E>,
        _count: impl Into<Count<Self::Expr>>,
    ) {
    }

    fn push_local_interaction(
        &mut self,
        _tuples: impl IntoIterator<Item = (Vec<Self::Expr>, Count<Self::Expr>)>,
    ) {
    }
}

/// Indexed reads are dropped, as every other lookup declaration is.
impl<'a, F, R> IndexedLookupBuilder for SlicedQuadraticFolder<'a, F, R>
where
    F: Field,
    R: Field,
{
    fn push_indexed_read(
        &mut self,
        _table: &str,
        _position: usize,
        payload: impl IntoIterator<Item = usize>,
    ) {
        payload.into_iter().for_each(drop);
    }

    fn push_indexed_table(
        &mut self,
        _name: &str,
        _window: TraceWindow,
        columns: impl IntoIterator<Item = usize>,
    ) {
        columns.into_iter().for_each(drop);
    }

    fn num_indexed_reads(&self) -> usize {
        0
    }

    fn num_indexed_tables(&self) -> usize {
        0
    }
}

/// Bus tuples are dropped: a declaration's Booleanity check is asserted before it is recorded.
impl<'a, F, R> BusInteractionRecorder for SlicedQuadraticFolder<'a, F, R>
where
    F: Field,
    R: Field,
{
    fn record_bus_interaction<E: Into<Self::Expr>>(
        &mut self,
        _token: RecordToken,
        _bus: BusName<'_>,
        _direction: BusDirection,
        fields: impl IntoIterator<Item = E>,
        _activation: BusActivation<Self::Expr>,
    ) {
        // The bus protocol evaluates its retained symbolic profile after commitment.
        fields.into_iter().for_each(|field| {
            let _ = field.into();
        });
    }
}
