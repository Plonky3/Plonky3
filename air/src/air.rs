use alloc::borrow::Cow;
use alloc::vec::Vec;

use p3_matrix::dense::RowMajorMatrix;

use crate::builder::AirBuilder;

/// Which end of the trace a boundary cell lives on.
///
/// These are the two rows a first-row and a last-row selector single out.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum BoundaryEnd {
    /// The first trace row, index `0`.
    First,
    /// The last trace row, index `height - 1`.
    Last,
}

/// One main-trace cell whose value is a public input, named by its position.
///
/// A cell pairs a trace position with one of the AIR's public values:
///
/// ```text
///     (column, end) holds public_values[public_value]
/// ```
///
/// This is a declaration, not a constraint.
/// Binding the cell to the value is the proving backend's job.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct BoundaryPublic {
    /// Main-trace column holding the cell.
    pub column: usize,
    /// Trace end the cell sits on.
    pub end: BoundaryEnd,
    /// Index into the AIR's public values supplying the cell's value.
    pub public_value: usize,
}

impl BoundaryPublic {
    /// Bundle a column, a trace end, and a public-value index into a boundary cell.
    pub const fn new(column: usize, end: BoundaryEnd, public_value: usize) -> Self {
        Self {
            column,
            end,
            public_value,
        }
    }

    /// Row index this cell sits on in a trace of `height` rows.
    ///
    /// ```text
    ///     first end -> 0
    ///     last  end -> height - 1
    /// ```
    ///
    /// # Panics
    ///
    /// Panics when `height` is zero.
    /// An empty trace has no boundary row to name.
    #[must_use]
    pub const fn row(&self, height: usize) -> usize {
        // A zero-height trace has no row to address at all.
        assert!(height > 0, "a boundary cell needs at least one trace row");

        // The two ends are the low and high rows of the trace.
        match self.end {
            BoundaryEnd::First => 0,
            BoundaryEnd::Last => height - 1,
        }
    }
}

/// The underlying structure of an AIR.
pub trait BaseAir<F>: Sync {
    /// The number of columns (a.k.a. registers) in this AIR.
    fn width(&self) -> usize;

    /// Return an optional preprocessed trace matrix to be included in the prover's trace.
    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        None
    }

    /// Width of the preprocessed trace, in columns.
    ///
    /// Defaults to `0`, matching the default [`Self::preprocessed_trace`] of
    /// `None`. Implementors that override [`Self::preprocessed_trace`] **must**
    /// also override this method to return a matching width — callers use this
    /// to size symbolic builders without materializing the preprocessed matrix.
    fn preprocessed_width(&self) -> usize {
        0
    }

    /// Return the number of periodic columns.
    ///
    /// Override when this AIR uses periodic columns; see [`Self::periodic_columns`].
    fn num_periodic_columns(&self) -> usize {
        0
    }

    /// Return the periodic table data.
    ///
    /// Periodic columns are columns whose values repeat with a fixed period that divides the
    /// trace length. They are derived from public parameters and are never committed as part
    /// of the trace — instead, both prover and verifier compute them from the data provided here.
    ///
    /// # Mathematical model
    ///
    /// For a trace of length n evaluated over a multiplicative subgroup H = {g⁰, g¹, ..., gⁿ⁻¹},
    /// a periodic column with period p (where p divides n, both powers of 2) is defined as follows:
    ///
    /// - Let r = n/p be the number of repetitions.
    /// - The p values are interpreted as evaluations of a polynomial f(x) of degree < p
    ///   over the subgroup Hʳ = {g⁰, gʳ, g²ʳ, ..., g⁽ᵖ⁻¹⁾ʳ} of order p.
    /// - The periodic extension f'(X) = f(Xʳ) has degree < p·r = n and satisfies
    ///   f'(gⁱ) = f(gⁱʳ), which cycles through the p values as i increases.
    ///
    /// # Commitment
    ///
    /// Periodic columns are public parameters and must be committed during initialization of
    /// the Fiat-Shamir transcript. The values returned are evaluations over a subgroup;
    /// callers may convert to coefficient form for efficient evaluation if needed.
    fn periodic_columns(&self) -> Cow<'_, [Vec<F>]>
    where
        F: Clone,
    {
        Cow::Borrowed(&[])
    }

    /// Return the periodic values for the given row index.
    fn periodic_values(&self, row_index: usize) -> Vec<F>
    where
        F: Clone,
    {
        self.periodic_columns()
            .iter()
            .map(|col| col[row_index % col.len()].clone())
            .collect()
    }

    /// Return a matrix with all periodic columns extended to a common height.
    ///
    /// The result is a row-major matrix where each row corresponds to a row index in the
    /// common extended domain (of size equal to the maximum period), and each column
    /// corresponds to one periodic column. Columns with smaller periods are repeated
    /// cyclically to fill the extended domain.
    ///
    /// Returns `None` if there are no periodic columns.
    fn periodic_columns_matrix(&self) -> Option<RowMajorMatrix<F>>
    where
        F: Clone + Send + Sync,
    {
        let cols = self.periodic_columns();
        if cols.is_empty() {
            return None;
        }

        let max_period = cols.iter().map(|c| c.len()).max()?;

        let values = (0..max_period)
            .flat_map(|row| cols.iter().map(move |col| col[row % col.len()].clone()))
            .collect();

        Some(RowMajorMatrix::new(values, cols.len()))
    }

    /// Which main trace columns have their next row accessed by this AIR's
    /// constraints.
    ///
    /// By default this returns every column index, which will require
    /// opening all main columns at both `zeta` and `zeta_next`.
    ///
    /// AIRs that only ever read the current main row (and never access an
    /// offset-1 main entry) can override this to return an empty vector to
    /// allow the prover and verifier to open only at `zeta`.
    ///
    /// # When to override
    ///
    /// - **Return empty**: single-row AIRs where all constraints are
    ///   evaluated within one row.
    /// - **Keep default** (all columns): AIRs with transition constraints
    ///   that reference `main.next_slice()`.
    /// - **Return a subset**: AIRs where only a few columns need next-row
    ///   access, enabling future per-column opening optimizations.
    ///
    /// # Correctness
    ///
    /// Must be consistent with [`Air::eval`]. Omitting a column index when
    /// the AIR actually reads its next row will cause verification failures
    /// or, in the worst case, a soundness gap.
    fn main_next_row_columns(&self) -> Vec<usize> {
        (0..self.width()).collect()
    }

    /// Which preprocessed trace columns have their next row accessed by this
    /// AIR's constraints.
    ///
    /// By default this returns every preprocessed column index, which will
    /// require opening preprocessed columns at both `zeta` and `zeta_next`.
    ///
    /// AIRs that only ever read the current preprocessed row (and never
    /// access an offset-1 preprocessed entry) can override this to return an
    /// empty vector to allow the prover and verifier to open only at `zeta`.
    fn preprocessed_next_row_columns(&self) -> Vec<usize> {
        (0..self.preprocessed_width()).collect()
    }

    /// Optional hint for the number of constraints in this AIR.
    ///
    /// Normally the prover runs a full symbolic evaluation just to count
    /// constraints. Overriding this method lets the prover skip that pass.
    ///
    /// The count must cover every constraint asserted during evaluation,
    /// including both transition and boundary constraints. It must **not**
    /// include lookup or permutation constraints, which are counted
    /// separately.
    ///
    /// # Correctness
    ///
    /// The returned value **must** exactly match the actual number of
    /// constraints. A wrong count will cause the prover to panic or
    /// produce an invalid proof.
    ///
    /// Returns `None` by default, which falls back to symbolic evaluation.
    fn num_constraints(&self) -> Option<usize> {
        None
    }

    /// Optional hint for the maximum constraint degree in this AIR.
    ///
    /// The constraint degree is the factor by which trace length N
    /// scales the constraint polynomial degree.
    ///
    /// For example, a constraint `x * y * z` where x, y, z are trace
    /// variables has degree multiple 3.
    ///
    /// Normally the prover runs a full symbolic evaluation to compute this.
    /// Overriding this method lets both the prover and verifier skip that
    /// pass when only the degree (not the full constraint list) is needed.
    ///
    /// The value must be an upper bound on the degree multiple of every
    /// constraint (base and extension). It does not need to be tight, but
    /// overestimating wastes prover work (larger quotient domain).
    ///
    /// # Correctness
    ///
    /// The returned value **must** be >= the actual max constraint degree.
    /// A value that is too small will cause the prover to produce an
    /// invalid proof.
    ///
    /// The hint covers only what this AIR asserts during its own evaluation.
    /// A backend that injects extra constraints scores their degree separately.
    ///
    /// Returns `None` by default, which falls back to symbolic evaluation.
    fn max_constraint_degree(&self) -> Option<usize> {
        None
    }

    /// Return the number of expected public values.
    fn num_public_values(&self) -> usize {
        0
    }

    /// Main-trace cells whose values are public inputs, named by position.
    ///
    /// A public input reaches a proof through one of two routes:
    ///
    /// ```text
    ///     boundary constraint : asserted by the AIR, honored by every backend
    ///     cell listed here    : bound by the backend, honored by some backends
    /// ```
    ///
    /// Listing a cell is therefore not by itself a binding.
    /// Support across this workspace:
    ///
    /// ```text
    ///     multilinear multi-STARK : binds every listed cell, needs no AIR constraint
    ///     univariate STARKs       : reject an AIR that lists any cell
    ///     debug constraint check  : compares each listed cell against the trace
    /// ```
    ///
    /// The default is the empty slice.
    /// An AIR that overrides nothing keeps binding its public inputs by constraint.
    ///
    /// # Correctness
    ///
    /// - Every column index is less than the main width.
    /// - Every public-value index is less than the declared public-value count.
    /// - No two cells name the same column and trace end.
    fn public_boundary_io(&self) -> &[BoundaryPublic] {
        &[]
    }
}

/// An algebraic intermediate representation (AIR) definition.
///
/// Contains an evaluation function for computing the constraints of the AIR.
/// This function can be applied to an evaluation trace in which case each
/// constraint will compute a particular value or it can be applied symbolically
/// with each constraint computing a symbolic expression.
pub trait Air<AB: AirBuilder>: BaseAir<AB::F> {
    /// Evaluate all AIR constraints using the provided builder.
    ///
    /// The builder provides both the trace on which the constraints
    /// are evaluated on as well as the method of accumulating the
    /// constraint evaluations.
    ///
    /// # Arguments
    /// - `builder`: Mutable reference to an `AirBuilder` for defining constraints.
    fn eval(&self, builder: &mut AB);
}
