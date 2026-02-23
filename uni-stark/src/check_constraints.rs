pub use p3_air::DebugConstraintBuilder;
use p3_air::{Air, check_constraints as air_check_constraints};
use p3_field::Field;
use p3_matrix::dense::RowMajorMatrix;
use tracing::instrument;

/// Thin wrapper around the canonical constraint checker in `p3-air` that
/// adds a tracing span for profiling.
///
/// See [`p3_air::check_constraints`] for the full documentation.
#[instrument(skip_all)]
#[allow(unused)] // Suppresses warnings in release mode where this is dead code.
pub(crate) fn check_constraints<F, A>(air: &A, main: &RowMajorMatrix<F>, public_values: &[F])
where
    F: Field,
    A: for<'a> Air<DebugConstraintBuilder<'a, F>>,
{
    air_check_constraints(air, main, public_values);
}
