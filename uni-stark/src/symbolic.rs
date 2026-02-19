//! STARK-specific quotient degree computation functions.
//!
//! For symbolic constraint analysis utilities, see [`p3_air::SymbolicAirBuilder`]
//! and the functions in [`p3_air::symbolic`].

use p3_air::Air;
use p3_air::symbolic::{SymbolicAirBuilder, get_max_constraint_degree_extension};
use p3_field::{ExtensionField, Field};
use p3_util::log2_ceil_usize;
use tracing::instrument;

#[instrument(skip_all)]
pub fn get_log_num_quotient_chunks<F, A>(
    air: &A,
    preprocessed_width: usize,
    num_public_values: usize,
    is_zk: usize,
) -> usize
where
    F: Field,
    A: Air<SymbolicAirBuilder<F>>,
{
    get_log_quotient_degree_extension(air, preprocessed_width, num_public_values, 0, 0, is_zk)
}

#[instrument(name = "infer log of base and extension constraint degree", skip_all)]
pub fn get_log_quotient_degree_extension<F, EF, A>(
    air: &A,
    preprocessed_width: usize,
    num_public_values: usize,
    permutation_width: usize,
    num_permutation_challenges: usize,
    is_zk: usize,
) -> usize
where
    F: Field,
    EF: ExtensionField<F>,
    A: Air<SymbolicAirBuilder<F, EF>>,
{
    assert!(is_zk <= 1, "is_zk must be either 0 or 1");
    // We pad to at least degree 2, since a quotient argument doesn't make sense with smaller degrees.
    let constraint_degree = (get_max_constraint_degree_extension::<F, EF, A>(
        air,
        preprocessed_width,
        num_public_values,
        permutation_width,
        num_permutation_challenges,
        air.num_periodic_columns(),
    ) + is_zk)
        .max(2);

    // We bound the degree of the quotient polynomial by constraint_degree - 1,
    // then choose the number of quotient chunks as the smallest power of two
    // >= (constraint_degree - 1). This function returns log2(#chunks).
    log2_ceil_usize(constraint_degree - 1)
}
