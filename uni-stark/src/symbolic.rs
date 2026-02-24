//! STARK-specific quotient polynomial degree calculations.

use p3_air::Air;
use p3_air::symbolic::SymbolicAirBuilder;
use p3_field::{ExtensionField, Field};
use p3_util::log2_ceil_usize;
use tracing::instrument;

#[instrument(skip_all, level = "debug")]
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
    assert!(is_zk <= 1, "is_zk must be either 0 or 1");

    if let Some(degree_hint) = air.max_constraint_degree() {
        let constraint_degree = (degree_hint + is_zk).max(2);
        let result = log2_ceil_usize(constraint_degree - 1);

        debug_assert!(
            {
                let symbolic = get_log_quotient_degree_extension::<F, F, A>(
                    air,
                    preprocessed_width,
                    num_public_values,
                    0,
                    0,
                    is_zk,
                );
                result >= symbolic
            },
            "max_constraint_degree() hint {} is too small; actual log quotient degree is larger",
            degree_hint
        );

        return result;
    }

    get_log_quotient_degree_extension(air, preprocessed_width, num_public_values, 0, 0, is_zk)
}

#[instrument(
    name = "infer log of base and extension constraint degree",
    skip_all,
    level = "debug"
)]
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

    if let Some(degree_hint) = air.max_constraint_degree() {
        let constraint_degree = (degree_hint + is_zk).max(2);
        let result = log2_ceil_usize(constraint_degree - 1);

        debug_assert!(
            {
                let actual = p3_air::symbolic::get_max_constraint_degree_extension::<F, EF, A>(
                    air,
                    preprocessed_width,
                    num_public_values,
                    permutation_width,
                    num_permutation_challenges,
                );
                degree_hint >= actual
            },
            "max_constraint_degree() hint {} is too small; symbolic evaluation found a larger degree",
            degree_hint
        );

        return result;
    }

    // We pad to at least degree 2, since a quotient argument doesn't make sense with smaller degrees.
    let constraint_degree = (p3_air::symbolic::get_max_constraint_degree_extension::<F, EF, A>(
        air,
        preprocessed_width,
        num_public_values,
        permutation_width,
        num_permutation_challenges,
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

    use p3_air::symbolic::{SymbolicAirBuilder, SymbolicVariable};
    use p3_air::{AirBuilder, BaseAir, Entry};
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

    /// A mock AIR with a configurable `max_constraint_degree` hint.
    #[derive(Debug)]
    struct HintedMockAir {
        constraints: Vec<SymbolicVariable<BabyBear>>,
        width: usize,
        degree_hint: Option<usize>,
    }

    impl BaseAir<BabyBear> for HintedMockAir {
        fn width(&self) -> usize {
            self.width
        }

        fn max_constraint_degree(&self) -> Option<usize> {
            self.degree_hint
        }
    }

    impl Air<SymbolicAirBuilder<BabyBear>> for HintedMockAir {
        fn eval(&self, builder: &mut SymbolicAirBuilder<BabyBear>) {
            for constraint in &self.constraints {
                builder.assert_zero(*constraint);
            }
        }
    }

    #[test]
    fn test_max_constraint_degree_hint_is_used() {
        // Actual degree is 1 (single variable), hint says 3.
        // The hint should be used, giving log2_ceil(max(3, 2) - 1) = log2_ceil(2) = 1.
        let air = HintedMockAir {
            constraints: vec![SymbolicVariable::new(Entry::Main { offset: 0 }, 0)],
            width: 4,
            degree_hint: Some(3),
        };
        let log_chunks = get_log_num_quotient_chunks(&air, 0, 0, 0);
        assert_eq!(log_chunks, log2_ceil_usize(2));
    }

    #[test]
    fn test_max_constraint_degree_hint_none_falls_back() {
        // No hint provided — should fall back to symbolic evaluation.
        // Actual degree is 1, so log2_ceil(max(1, 2) - 1) = log2_ceil(1) = 0.
        let air = HintedMockAir {
            constraints: vec![SymbolicVariable::new(Entry::Main { offset: 0 }, 0)],
            width: 4,
            degree_hint: None,
        };
        let log_chunks = get_log_num_quotient_chunks(&air, 0, 0, 0);
        assert_eq!(log_chunks, 0);
    }

    #[test]
    fn test_max_constraint_degree_hint_exact_match() {
        // Hint matches actual degree exactly.
        let air = HintedMockAir {
            constraints: vec![SymbolicVariable::new(Entry::Main { offset: 0 }, 0)],
            width: 4,
            degree_hint: Some(1),
        };
        let with_hint = get_log_num_quotient_chunks(&air, 0, 0, 0);

        let air_no_hint = HintedMockAir {
            constraints: vec![SymbolicVariable::new(Entry::Main { offset: 0 }, 0)],
            width: 4,
            degree_hint: None,
        };
        let without_hint = get_log_num_quotient_chunks(&air_no_hint, 0, 0, 0);

        assert_eq!(with_hint, without_hint);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "max_constraint_degree() hint")]
    fn test_max_constraint_degree_hint_too_small_panics() {
        // Actual degree is 1, hint says 0 — debug_assert should fire.
        let air = HintedMockAir {
            constraints: vec![SymbolicVariable::new(Entry::Main { offset: 0 }, 0)],
            width: 4,
            degree_hint: Some(0),
        };
        let _ = get_log_num_quotient_chunks(&air, 0, 0, 0);
    }
}
