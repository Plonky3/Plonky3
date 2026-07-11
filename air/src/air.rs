use alloc::borrow::Cow;
use alloc::vec::Vec;

use p3_matrix::dense::RowMajorMatrix;

use crate::builder::AirBuilder;

/// Which end of the trace a public boundary cell lives on.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum BoundaryEnd {
    /// The first trace row, the all-zeros hypercube corner (index `0`).
    First,
    /// The last trace row, the all-ones hypercube corner (index `2^num_variables - 1`).
    Last,
}

/// One main-trace cell exposed as a public input at a boundary corner.
///
/// This drives Borgeaud's boundary-IO handling of public inputs:
/// - the prover commits the column with this cell forced to zero,
/// - the verifier restores the true value from the public input by a Lagrange-at-corner correction.
///
/// A matching corner-zero constraint pins the committed cell to zero.
/// The restored value is then exactly the public input.
///
/// See <https://solvable.group/posts/super-air/> ("Handling public inputs").
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct BoundaryPublic {
    /// Main-trace column holding the public cell.
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
    /// Returns `None` by default, which falls back to symbolic evaluation.
    fn max_constraint_degree(&self) -> Option<usize> {
        None
    }

    /// Return the number of expected public values.
    fn num_public_values(&self) -> usize {
        0
    }

    /// Main-trace boundary cells bound as public inputs via boundary IO.
    ///
    /// Each returned cell is committed as zero.
    /// The verifier restores it from the public input by a Lagrange-at-corner correction.
    /// See [`BoundaryPublic`] for the mechanism and its soundness pin.
    ///
    /// Defaults to none, so public inputs are bound by ordinary constraints unless an AIR opts in.
    ///
    /// # Correctness
    ///
    /// The multilinear prover asserts a corner-zero pin for each cell.
    /// A listed cell therefore need not be pinned by the AIR's own evaluation.
    /// Every referenced column must be a real main column.
    /// Every public-value index must address a declared public value.
    fn public_boundary_io(&self) -> Vec<BoundaryPublic> {
        Vec::new()
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
