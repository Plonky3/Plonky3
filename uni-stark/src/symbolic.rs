//! STARK-specific quotient polynomial degree calculations.

use alloc::vec::Vec;

use p3_air::Air;
use p3_air::symbolic::{
    AirLayout, BaseEntry, BaseLeaf, ExtEntry, ExtLeaf, SymbolicAirBuilder, SymbolicExpr,
    SymbolicExpression, SymbolicExpressionExt, get_all_symbolic_constraints,
    get_max_constraint_degree_extension,
};
use p3_field::{ExtensionField, Field};
use p3_util::log2_ceil_usize;
use tracing::instrument;

#[instrument(skip_all, level = "debug")]
pub fn get_log_num_quotient_chunks<F, A>(
    air: &A,
    layout: AirLayout,
    trace_degree: usize,
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
                let hinted_degree = get_max_constraint_degree_extension::<F, F, A>(air, layout);
                let symbolic =
                    get_log_quotient_degree_extension::<F, F, A>(air, layout, trace_degree, is_zk);
                degree_hint >= hinted_degree && result >= symbolic
            },
            "max_constraint_degree() hint {} is too small; actual log quotient degree is larger",
            degree_hint
        );

        return result;
    }

    get_log_quotient_degree_extension(air, layout, trace_degree, is_zk)
}

#[instrument(
    name = "infer log of base and extension constraint degree",
    skip_all,
    level = "debug"
)]
pub fn get_log_quotient_degree_extension<F, EF, A>(
    air: &A,
    layout: AirLayout,
    trace_degree: usize,
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
                let hinted_degree = get_max_constraint_degree_extension::<F, EF, A>(air, layout);
                let actual = get_log_quotient_degree_extension_symbolic::<F, EF, A>(
                    air,
                    layout,
                    trace_degree,
                    is_zk,
                );
                degree_hint >= hinted_degree && result >= actual
            },
            "max_constraint_degree() hint {} is too small; actual log quotient degree is larger",
            degree_hint
        );

        return result;
    }

    get_log_quotient_degree_extension_symbolic(air, layout, trace_degree, is_zk)
}

fn get_log_quotient_degree_extension_symbolic<F, EF, A>(
    air: &A,
    layout: AirLayout,
    trace_degree: usize,
    is_zk: usize,
) -> usize
where
    F: Field,
    EF: ExtensionField<F>,
    A: Air<SymbolicAirBuilder<F, EF>>,
{
    let periodic_column_degrees = air
        .periodic_columns()
        .into_iter()
        .map(|col| {
            debug_assert!(
                trace_degree.is_multiple_of(col.len()),
                "periodic column period must divide the trace degree"
            );
            trace_degree - trace_degree / col.len()
        })
        .collect::<Vec<_>>();

    let (base_constraints, extension_constraints) =
        get_all_symbolic_constraints::<F, EF, _>(air, layout);

    let base_degree = base_constraints
        .iter()
        .map(|constraint| {
            symbolic_expression_degree(
                constraint,
                trace_degree,
                is_zk,
                periodic_column_degrees.as_slice(),
            )
        })
        .max()
        .unwrap_or(0);

    let extension_degree = extension_constraints
        .iter()
        .map(|constraint| {
            symbolic_expression_ext_degree(
                constraint,
                trace_degree,
                is_zk,
                periodic_column_degrees.as_slice(),
            )
        })
        .max()
        .unwrap_or(0);

    let constraint_degree = base_degree.max(extension_degree);
    log_chunks_from_constraint_degree(constraint_degree, trace_degree, is_zk)
}

fn symbolic_expression_degree<F: Field>(
    expr: &SymbolicExpression<F>,
    trace_degree: usize,
    is_zk: usize,
    periodic_column_degrees: &[usize],
) -> usize {
    match expr {
        SymbolicExpr::Leaf(leaf) => {
            base_leaf_degree(leaf, trace_degree, is_zk, periodic_column_degrees)
        }
        SymbolicExpr::Add { x, y, .. } | SymbolicExpr::Sub { x, y, .. } => {
            symbolic_expression_degree(x, trace_degree, is_zk, periodic_column_degrees).max(
                symbolic_expression_degree(y, trace_degree, is_zk, periodic_column_degrees),
            )
        }
        SymbolicExpr::Neg { x, .. } => {
            symbolic_expression_degree(x, trace_degree, is_zk, periodic_column_degrees)
        }
        SymbolicExpr::Mul { x, y, .. } => {
            symbolic_expression_degree(x, trace_degree, is_zk, periodic_column_degrees)
                + symbolic_expression_degree(y, trace_degree, is_zk, periodic_column_degrees)
        }
    }
}

fn base_leaf_degree<F: Field>(
    leaf: &BaseLeaf<F>,
    trace_degree: usize,
    is_zk: usize,
    periodic_column_degrees: &[usize],
) -> usize {
    let extended_trace_degree = trace_degree << is_zk;
    match leaf {
        BaseLeaf::Variable(variable) => match variable.entry {
            BaseEntry::Preprocessed { .. } | BaseEntry::Main { .. } => extended_trace_degree - 1,
            BaseEntry::Periodic => periodic_column_degrees[variable.index],
            BaseEntry::Public => 0,
        },
        BaseLeaf::IsFirstRow | BaseLeaf::IsLastRow => trace_degree - 1,
        BaseLeaf::IsTransition => 1,
        BaseLeaf::Constant(_) => 0,
    }
}

fn symbolic_expression_ext_degree<F: Field, EF: ExtensionField<F>>(
    expr: &SymbolicExpressionExt<F, EF>,
    trace_degree: usize,
    is_zk: usize,
    periodic_column_degrees: &[usize],
) -> usize {
    match expr {
        SymbolicExpr::Leaf(leaf) => {
            ext_leaf_degree(leaf, trace_degree, is_zk, periodic_column_degrees)
        }
        SymbolicExpr::Add { x, y, .. } | SymbolicExpr::Sub { x, y, .. } => {
            symbolic_expression_ext_degree(x, trace_degree, is_zk, periodic_column_degrees).max(
                symbolic_expression_ext_degree(y, trace_degree, is_zk, periodic_column_degrees),
            )
        }
        SymbolicExpr::Neg { x, .. } => {
            symbolic_expression_ext_degree(x, trace_degree, is_zk, periodic_column_degrees)
        }
        SymbolicExpr::Mul { x, y, .. } => {
            symbolic_expression_ext_degree(x, trace_degree, is_zk, periodic_column_degrees)
                + symbolic_expression_ext_degree(y, trace_degree, is_zk, periodic_column_degrees)
        }
    }
}

fn ext_leaf_degree<F: Field, EF: ExtensionField<F>>(
    leaf: &ExtLeaf<F, EF>,
    trace_degree: usize,
    is_zk: usize,
    periodic_column_degrees: &[usize],
) -> usize {
    let extended_trace_degree = trace_degree << is_zk;
    match leaf {
        ExtLeaf::Base(expr) => {
            symbolic_expression_degree(expr, trace_degree, is_zk, periodic_column_degrees)
        }
        ExtLeaf::ExtVariable(variable) => match variable.entry {
            ExtEntry::Permutation { .. } => extended_trace_degree - 1,
            ExtEntry::Challenge | ExtEntry::PermutationValue => 0,
        },
        ExtLeaf::ExtConstant(_) => 0,
    }
}

fn log_chunks_from_constraint_degree(
    constraint_degree: usize,
    trace_degree: usize,
    is_zk: usize,
) -> usize {
    if constraint_degree == 0 {
        return 0;
    }

    let quotient_chunk_degree = trace_degree << is_zk;
    let chunk_count = constraint_degree
        .saturating_sub(trace_degree)
        .saturating_add(1)
        .div_ceil(quotient_chunk_degree)
        .max(1);

    log2_ceil_usize(chunk_count)
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use p3_air::symbolic::{AirLayout, SymbolicAirBuilder, SymbolicExpression, SymbolicVariable};
    use p3_air::{AirBuilder, BaseAir, BaseEntry, PeriodicAirBuilder};
    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;

    use super::*;

    const TRACE_DEGREE: usize = 1 << 6;

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

    fn air_layout(air: &impl BaseAir<BabyBear>, preprocessed_width: usize) -> AirLayout {
        AirLayout {
            preprocessed_width,
            main_width: air.width(),
            num_public_values: air.num_public_values(),
            num_periodic_columns: air.num_periodic_columns(),
            ..Default::default()
        }
    }

    #[test]
    fn test_get_log_num_quotient_chunks_no_constraints() {
        let air = MockAir {
            constraints: vec![],
            width: 4,
        };
        let log_degree = get_log_num_quotient_chunks(&air, air_layout(&air, 3), TRACE_DEGREE, 0);
        assert_eq!(log_degree, 0);
    }

    #[test]
    fn test_get_log_num_quotient_chunks_single_constraint() {
        let air = MockAir {
            constraints: vec![SymbolicVariable::new(BaseEntry::Main { offset: 0 }, 0)],
            width: 4,
        };
        let log_degree = get_log_num_quotient_chunks(&air, air_layout(&air, 3), TRACE_DEGREE, 0);
        assert_eq!(log_degree, log2_ceil_usize(1));
    }

    #[test]
    fn test_get_log_num_quotient_chunks_multiple_constraints() {
        let air = MockAir {
            constraints: vec![
                SymbolicVariable::new(BaseEntry::Main { offset: 0 }, 0),
                SymbolicVariable::new(BaseEntry::Main { offset: 1 }, 1),
                SymbolicVariable::new(BaseEntry::Main { offset: 2 }, 2),
            ],
            width: 4,
        };
        let log_degree = get_log_num_quotient_chunks(&air, air_layout(&air, 3), TRACE_DEGREE, 0);
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
            constraints: vec![SymbolicVariable::new(BaseEntry::Main { offset: 0 }, 0)],
            width: 4,
            degree_hint: Some(3),
        };
        let log_chunks = get_log_num_quotient_chunks(&air, air_layout(&air, 0), TRACE_DEGREE, 0);
        assert_eq!(log_chunks, log2_ceil_usize(2));
    }

    #[test]
    fn test_max_constraint_degree_hint_none_falls_back() {
        // No hint provided — should fall back to symbolic evaluation.
        // Actual degree is 1, so log2_ceil(max(1, 2) - 1) = log2_ceil(1) = 0.
        let air = HintedMockAir {
            constraints: vec![SymbolicVariable::new(BaseEntry::Main { offset: 0 }, 0)],
            width: 4,
            degree_hint: None,
        };
        let log_chunks = get_log_num_quotient_chunks(&air, air_layout(&air, 0), TRACE_DEGREE, 0);
        assert_eq!(log_chunks, 0);
    }

    #[test]
    fn test_max_constraint_degree_hint_exact_match() {
        // Hint matches actual degree exactly.
        let air = HintedMockAir {
            constraints: vec![SymbolicVariable::new(BaseEntry::Main { offset: 0 }, 0)],
            width: 4,
            degree_hint: Some(1),
        };
        let with_hint = get_log_num_quotient_chunks(&air, air_layout(&air, 0), TRACE_DEGREE, 0);

        let air_no_hint = HintedMockAir {
            constraints: vec![SymbolicVariable::new(BaseEntry::Main { offset: 0 }, 0)],
            width: 4,
            degree_hint: None,
        };
        let without_hint =
            get_log_num_quotient_chunks(&air_no_hint, air_layout(&air_no_hint, 0), TRACE_DEGREE, 0);

        assert_eq!(with_hint, without_hint);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "max_constraint_degree() hint")]
    fn test_max_constraint_degree_hint_too_small_panics() {
        // Actual degree is 1, hint says 0 — debug_assert should fire.
        let air = HintedMockAir {
            constraints: vec![SymbolicVariable::new(BaseEntry::Main { offset: 0 }, 0)],
            width: 4,
            degree_hint: Some(0),
        };
        let _ = get_log_num_quotient_chunks(&air, air_layout(&air, 0), TRACE_DEGREE, 0);
    }

    #[derive(Clone, Debug)]
    struct PeriodicProductAir {
        periods: Vec<usize>,
    }

    impl BaseAir<BabyBear> for PeriodicProductAir {
        fn width(&self) -> usize {
            0
        }

        fn num_periodic_columns(&self) -> usize {
            self.periods.len()
        }

        fn periodic_columns(&self) -> Vec<Vec<BabyBear>> {
            self.periods
                .iter()
                .enumerate()
                .map(|(column_index, period)| {
                    (0..*period)
                        .map(|row| BabyBear::from_usize(column_index + row + 1))
                        .collect()
                })
                .collect()
        }
    }

    impl Air<SymbolicAirBuilder<BabyBear>> for PeriodicProductAir {
        fn eval(&self, builder: &mut SymbolicAirBuilder<BabyBear>) {
            let product = builder
                .periodic_values()
                .iter()
                .copied()
                .map(SymbolicExpression::from)
                .reduce(|acc, value| acc * value)
                .expect("test AIR must define at least one periodic column");
            builder.assert_zero(product);
        }
    }

    #[test]
    fn test_period_2_columns_reduce_inferred_chunk_count() {
        let air = PeriodicProductAir {
            periods: vec![2, 2, 2, 2, 2],
        };

        let log_chunks = get_log_num_quotient_chunks(&air, air_layout(&air, 0), TRACE_DEGREE, 0);

        assert_eq!(
            log_chunks, 1,
            "five period-2 columns should need only 2 quotient chunks at trace degree 64"
        );
    }

    #[test]
    fn test_period_4_columns_reduce_inferred_chunk_count() {
        let air = PeriodicProductAir {
            periods: vec![4, 4, 4, 4, 4, 4],
        };

        let log_chunks = get_log_num_quotient_chunks(&air, air_layout(&air, 0), TRACE_DEGREE, 0);

        assert_eq!(
            log_chunks, 2,
            "six period-4 columns should need only 4 quotient chunks at trace degree 64"
        );
    }
}
