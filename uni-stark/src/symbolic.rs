//! STARK-specific quotient polynomial degree calculations.

use alloc::vec::Vec;

use p3_air::symbolic::{
    AirLayout, NextRowColumns, SymbolicAirBuilder, SymbolicConstraints, UndeclaredNextRowRead,
    capture_symbolic_constraints, get_max_constraint_degree_extension,
};
use p3_air::{Air, BaseAir};
use p3_commit::PolynomialSpace;
use p3_field::{ExtensionField, Field};
use p3_util::log2_ceil_usize;
use tracing::instrument;

/// Size the quotient using the trace domain's transition-selector degree.
///
/// Two-adic domains retain the cached symbolic degree fast path. Circle uses
/// full trace-space degrees for selectors and periodic columns. Its symbolic
/// degree is checked even with a hint, since the hint does not encode how many
/// transition factors occur in a constraint.
pub fn get_log_num_quotient_chunks_for_domain<F, A>(
    air: &A,
    layout: AirLayout,
    domain: impl PolynomialSpace<Val = F>,
    is_zk: usize,
) -> usize
where
    F: Field,
    A: Air<SymbolicAirBuilder<F>>,
{
    AirConstraints::<F>::evaluate(air, layout).log_num_quotient_chunks(domain, is_zk)
}

/// The constraints of one AIR from a single symbolic evaluation.
///
/// It also keeps the AIR facts a STARK reads next to them:
///
/// - the declared next-row columns;
/// - the period of each periodic column;
/// - the optional constraint degree hint.
///
/// Provers and verifiers evaluate the AIR once and answer every question from this value.
#[derive(Debug)]
pub struct AirConstraints<F: Field, EF: ExtensionField<F> = F> {
    /// The evaluated constraints.
    constraints: SymbolicConstraints<F, EF>,
    /// The next-row columns the AIR declares.
    next_row: NextRowColumns,
    /// The period of each periodic column, empty when the AIR has none.
    periodic_periods: Vec<usize>,
    /// The constraint degree the AIR states, if any.
    degree_hint: Option<usize>,
}

impl<F: Field, EF: ExtensionField<F>> AirConstraints<F, EF> {
    /// Pair constraints already evaluated from an AIR with that AIR's facts.
    ///
    /// The constraints may include more than the AIR's own, such as lookup constraints.
    pub fn new(air: &impl BaseAir<F>, constraints: SymbolicConstraints<F, EF>) -> Self {
        // Materialize the periods only when the AIR has periodic columns.
        let periodic_periods = if air.num_periodic_columns() == 0 {
            Vec::new()
        } else {
            air.periodic_periods()
        };
        Self {
            constraints,
            next_row: NextRowColumns::from_air(air),
            periodic_periods,
            degree_hint: air.max_constraint_degree(),
        }
    }

    /// Evaluate an AIR symbolically.
    ///
    /// # Panics
    ///
    /// Panics if a layout width that the AIR itself determines disagrees with the AIR.
    pub fn evaluate<A>(air: &A, layout: AirLayout) -> Self
    where
        A: Air<SymbolicAirBuilder<F, EF>>,
    {
        // A mismatched width would silently point the AIR at the wrong variables.
        layout.validate_against_air(air);
        let constraints = capture_symbolic_constraints(layout, |builder| air.eval(builder));
        Self::new(air, constraints)
    }

    /// Log of the number of quotient chunks, before the zero-knowledge doubling.
    ///
    /// # Arguments
    ///
    /// - `domain`: the trace domain, which sets the weight of the transition selector.
    /// - `is_zk`: 1 when the PCS is zero-knowledge, 0 otherwise.
    ///
    /// # Panics
    ///
    /// Panics if `is_zk` is neither 0 nor 1.
    pub fn log_num_quotient_chunks(
        &self,
        domain: impl PolynomialSpace<Val = F>,
        is_zk: usize,
    ) -> usize {
        assert!(is_zk <= 1, "is_zk must be either 0 or 1");

        // A zero transition weight marks a two-adic domain.
        // There the degree is measured against the trace size, including periodic columns.
        let transition_degree = domain.transition_degree_multiple();
        let symbolic_degree = if transition_degree == 0 {
            self.max_constraint_degree(domain.size())
        } else {
            let degrees = self
                .constraints
                .max_constraint_degrees_with_transition(transition_degree);
            degrees.base.max(degrees.extension)
        };
        let degree = symbolic_degree.max(self.degree_hint.unwrap_or(0));

        // Circle STARKs, section 5.3: an odd constraint degree d gives a quotient
        // in the FFT subspace L^-_{(d-1)N}. For even d >= 4, rounding d-1 up to a
        // power of two provides strict room; d <= 2 uses Circle's doubled disjoint
        // domain. Thus the same chunk formula applies once d counts all selectors.
        //
        // On a two-adic domain the quotient has degree about (d - 1) N.
        // The floor of 2 keeps a quotient argument meaningful for smaller degrees.
        log2_ceil_usize((degree + is_zk).max(2) - 1)
    }

    /// Maximum constraint degree on a two-adic trace of the given height.
    ///
    /// The degree hint plays no part here.
    pub fn max_constraint_degree(&self, trace_len: usize) -> usize {
        self.constraints
            .max_constraint_degree(trace_len, &self.periodic_periods)
    }

    /// Check that every next-row column the constraints read is one the AIR declares.
    ///
    /// # Errors
    ///
    /// Returns the first undeclared read in constraint order.
    pub fn check_next_row_reads(&self) -> Result<(), UndeclaredNextRowRead> {
        self.constraints.check_next_row_reads(&self.next_row)
    }
}

/// Two-adic quotient sizing; use [`get_log_num_quotient_chunks_for_domain`] for other domains.
#[instrument(skip_all, level = "debug")]
pub fn get_log_num_quotient_chunks<F, A>(
    air: &A,
    layout: AirLayout,
    trace_len: usize,
    is_zk: usize,
) -> usize
where
    F: Field,
    A: Air<SymbolicAirBuilder<F>>,
{
    get_log_quotient_degree_extension(air, layout, trace_len, is_zk)
}

/// Two-adic quotient sizing, including extension constraints.
#[instrument(
    name = "infer log of base and extension constraint degree",
    skip_all,
    level = "debug"
)]
pub fn get_log_quotient_degree_extension<F, EF, A>(
    air: &A,
    layout: AirLayout,
    trace_len: usize,
    is_zk: usize,
) -> usize
where
    F: Field,
    EF: ExtensionField<F>,
    A: Air<SymbolicAirBuilder<F, EF>>,
{
    assert!(is_zk <= 1, "is_zk must be either 0 or 1");

    let degree = get_max_constraint_degree_extension::<F, EF, A>(air, layout, trace_len)
        .max(air.max_constraint_degree().unwrap_or(0));
    // We pad to at least degree 2, since a quotient argument doesn't make sense with smaller degrees.
    let constraint_degree = (degree + is_zk).max(2);

    // We bound the degree of the quotient polynomial by constraint_degree - 1,
    // then choose the number of quotient chunks as the smallest power of two
    // >= (constraint_degree - 1). This function returns log2(#chunks).
    log2_ceil_usize(constraint_degree - 1)
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use p3_air::symbolic::{AirLayout, SymbolicAirBuilder, SymbolicVariable};
    use p3_air::{AirBuilder, BaseAir, BaseEntry};
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

    fn air_layout(air: &impl BaseAir<BabyBear>, preprocessed_width: usize) -> AirLayout {
        AirLayout {
            preprocessed_width,
            main_width: air.width(),
            num_public_values: air.num_public_values(),
            ..Default::default()
        }
    }

    #[test]
    fn test_get_log_num_quotient_chunks_no_constraints() {
        let air = MockAir {
            constraints: vec![],
            width: 4,
        };
        let log_degree = get_log_num_quotient_chunks(&air, air_layout(&air, 3), 8, 0);
        assert_eq!(log_degree, 0);
    }

    #[test]
    fn test_get_log_num_quotient_chunks_single_constraint() {
        let air = MockAir {
            constraints: vec![SymbolicVariable::new(BaseEntry::Main { offset: 0 }, 0)],
            width: 4,
        };
        let log_degree = get_log_num_quotient_chunks(&air, air_layout(&air, 3), 8, 0);
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
        let log_degree = get_log_num_quotient_chunks(&air, air_layout(&air, 3), 8, 0);
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
        let log_chunks = get_log_num_quotient_chunks(&air, air_layout(&air, 0), 8, 0);
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
        let log_chunks = get_log_num_quotient_chunks(&air, air_layout(&air, 0), 8, 0);
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
        let with_hint = get_log_num_quotient_chunks(&air, air_layout(&air, 0), 8, 0);

        let air_no_hint = HintedMockAir {
            constraints: vec![SymbolicVariable::new(BaseEntry::Main { offset: 0 }, 0)],
            width: 4,
            degree_hint: None,
        };
        let without_hint =
            get_log_num_quotient_chunks(&air_no_hint, air_layout(&air_no_hint, 0), 8, 0);

        assert_eq!(with_hint, without_hint);
    }

    #[test]
    fn test_max_constraint_degree_hint_too_small_uses_symbolic_degree() {
        let air = HintedMockAir {
            constraints: vec![SymbolicVariable::new(BaseEntry::Main { offset: 0 }, 0)],
            width: 4,
            degree_hint: Some(0),
        };
        assert_eq!(
            get_log_num_quotient_chunks(&air, air_layout(&air, 0), 8, 0),
            0
        );
    }

    #[test]
    fn undersized_cubic_hint_preserves_quotient_chunks() {
        struct CubicAir;
        impl BaseAir<BabyBear> for CubicAir {
            fn width(&self) -> usize {
                1
            }
            fn max_constraint_degree(&self) -> Option<usize> {
                Some(1)
            }
        }
        impl Air<SymbolicAirBuilder<BabyBear>> for CubicAir {
            fn eval(&self, builder: &mut SymbolicAirBuilder<BabyBear>) {
                let x = SymbolicVariable::new(BaseEntry::Main { offset: 0 }, 0);
                builder.assert_zero(x * x * x);
            }
        }
        let layout = air_layout(&CubicAir, 0);
        for (is_zk, expected) in [(0, 1), (1, 2)] {
            assert_eq!(
                get_log_num_quotient_chunks(&CubicAir, layout, 8, is_zk),
                expected
            );
            assert_eq!(
                get_log_quotient_degree_extension::<BabyBear, BabyBear, _>(
                    &CubicAir, layout, 8, is_zk
                ),
                expected
            );
        }
    }
}
