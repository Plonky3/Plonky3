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
        0, // num_periodic_columns: uni-stark doesn't use periodic columns
    ) + is_zk)
        .max(2);

    // We bound the degree of the quotient polynomial by constraint_degree - 1,
    // then choose the number of quotient chunks as the smallest power of two
    // >= (constraint_degree - 1). This function returns log2(#chunks).
    log2_ceil_usize(constraint_degree - 1)
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use p3_air::{AirBuilder, BaseAir, Entry, SymbolicVariable};
    use p3_baby_bear::BabyBear;

    use super::*;

    #[derive(Debug)]
    struct MockAir {
        constraints: Vec<SymbolicVariable<BabyBear>>,
        width: usize,
    }

    impl BaseAir<BabyBear> for MockAir {
        fn width(&self) -> usize {
            self.width
        }
    }

    impl Air<SymbolicAirBuilder<BabyBear>> for MockAir {
        fn eval(&self, builder: &mut SymbolicAirBuilder<BabyBear>) {
            for constraint in &self.constraints {
                builder.assert_zero(*constraint);
            }
        }
    }

    #[test]
    fn test_get_log_num_quotient_chunks_no_constraints() {
        let air = MockAir {
            constraints: vec![],
            width: 4,
        };
        let log_degree = get_log_num_quotient_chunks(&air, 3, 2, 0);
        assert_eq!(log_degree, 0);
    }

    #[test]
    fn test_get_log_num_quotient_chunks_single_constraint() {
        let air = MockAir {
            constraints: vec![SymbolicVariable::new(Entry::Main { offset: 0 }, 0)],
            width: 4,
        };
        let log_degree = get_log_num_quotient_chunks(&air, 3, 2, 0);
        assert_eq!(log_degree, log2_ceil_usize(1));
    }

    #[test]
    fn test_get_log_num_quotient_chunks_multiple_constraints() {
        let air = MockAir {
            constraints: vec![
                SymbolicVariable::new(Entry::Main { offset: 0 }, 0),
                SymbolicVariable::new(Entry::Main { offset: 1 }, 1),
                SymbolicVariable::new(Entry::Main { offset: 2 }, 2),
            ],
            width: 4,
        };
        let log_degree = get_log_num_quotient_chunks(&air, 3, 2, 0);
        assert_eq!(log_degree, log2_ceil_usize(1));
    }
}
