//! The next-row columns an AIR declares, and the check that its constraints respect them.

use alloc::vec;
use alloc::vec::Vec;
use core::fmt;

use hashbrown::HashMap;
use p3_field::{ExtensionField, Field};
use thiserror::Error;

use super::fold::fold_symbolic_expression_cached;
use super::{
    BaseEntry, BaseLeaf, ExtLeaf, SymbolicConstraints, SymbolicExprVisitor, SymbolicExpression,
    SymbolicVariable, fold_symbolic_expressions,
};
use crate::BaseAir;

/// A trace whose next row a constraint can read.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum NextRowTrace {
    /// The main trace.
    Main,
    /// The preprocessed trace.
    Preprocessed,
}

impl fmt::Display for NextRowTrace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Main => "main",
            Self::Preprocessed => "preprocessed",
        })
    }
}

/// A constraint reads a next-row column that the AIR does not declare.
///
/// Provers and verifiers skip the next-row opening when the declared set is empty.
///
/// Such a read then sees a row of zeros instead of the trace.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Error)]
#[error(
    "a constraint reads column {column} of the next {trace} row, which the AIR does not declare"
)]
pub struct UndeclaredNextRowRead {
    /// The trace the read comes from.
    pub trace: NextRowTrace,
    /// The column index within that trace.
    pub column: usize,
}

/// The next-row columns an AIR declares, one membership flag per column.
///
/// A declared index past the trace width is dropped: no constraint can read it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NextRowColumns {
    /// Whether each main column is declared.
    main: Vec<bool>,
    /// Whether each preprocessed column is declared.
    preprocessed: Vec<bool>,
}

impl NextRowColumns {
    /// Read the declaration of an AIR.
    pub fn from_air<F>(air: &impl BaseAir<F>) -> Self {
        // One flag per column turns each membership test into an index lookup.
        let flags = |declared: Vec<usize>, width: usize| {
            let mut flags = vec![false; width];
            for column in declared {
                if let Some(flag) = flags.get_mut(column) {
                    *flag = true;
                }
            }
            flags
        };
        Self {
            main: flags(air.main_next_row_columns(), air.width()),
            preprocessed: flags(
                air.preprocessed_next_row_columns(),
                air.preprocessed_width(),
            ),
        }
    }

    /// Whether the AIR declares this next-row column.
    pub fn contains(&self, trace: NextRowTrace, column: usize) -> bool {
        let flags = match trace {
            NextRowTrace::Main => &self.main,
            NextRowTrace::Preprocessed => &self.preprocessed,
        };
        flags.get(column).copied().unwrap_or(false)
    }
}

impl<F: Field, EF: ExtensionField<F>> SymbolicConstraints<F, EF> {
    /// Check that every next-row column the constraints read is declared.
    ///
    /// Only reads that reach a constraint count.
    ///
    /// A next-row value the AIR loads but never constrains changes nothing.
    ///
    /// # Errors
    ///
    /// Returns the first undeclared read in constraint order.
    ///
    /// Base constraints come first, then extension constraints.
    pub fn check_next_row_reads(
        &self,
        declared: &NextRowColumns,
    ) -> Result<(), UndeclaredNextRowRead> {
        let mut reads = NextRowReads {
            declared,
            first: None,
        };

        // Visit every distinct node once.
        // Shared subtrees appear in many constraints, so the memoized fold keeps this linear.
        let base_roots: Vec<_> = self.base_constraints.iter().collect();
        fold_symbolic_expressions(&base_roots, &mut reads);

        // Extension constraints embed base subtrees as leaves.
        // One base cache serves all of them.
        let extension_roots: Vec<_> = self.extension_constraints.iter().collect();
        let mut ext_reads = ExtNextRowReads {
            inner: &mut reads,
            bases: HashMap::new(),
            base_stack: Vec::new(),
        };
        fold_symbolic_expressions(&extension_roots, &mut ext_reads);

        reads.first.map_or(Ok(()), Err)
    }
}

/// Records the first next-row read outside the declared columns.
struct NextRowReads<'d> {
    /// The declaration the reads are checked against.
    declared: &'d NextRowColumns,
    /// The first undeclared read seen so far.
    first: Option<UndeclaredNextRowRead>,
}

impl NextRowReads<'_> {
    fn record<F>(&mut self, variable: &SymbolicVariable<F>) {
        // Only offset-1 trace entries are next-row reads.
        let trace = match variable.entry {
            BaseEntry::Main { offset: 1 } => NextRowTrace::Main,
            BaseEntry::Preprocessed { offset: 1 } => NextRowTrace::Preprocessed,
            _ => return,
        };
        // Keep the earliest violation so the report does not depend on later constraints.
        if self.first.is_none() && !self.declared.contains(trace, variable.index) {
            self.first = Some(UndeclaredNextRowRead {
                trace,
                column: variable.index,
            });
        }
    }
}

impl<'a, F: Field> SymbolicExprVisitor<'a, BaseLeaf<F>> for NextRowReads<'_> {
    type Output = ();

    fn leaf(&mut self, leaf: &'a BaseLeaf<F>) {
        if let BaseLeaf::Variable(variable) = leaf {
            self.record(variable);
        }
    }

    fn add(&mut self, _: &(), _: &()) {}

    fn sub(&mut self, _: &(), _: &()) {}

    fn neg(&mut self, _: &()) {}

    fn mul(&mut self, _: &(), _: &()) {}
}

/// Walks extension constraints, descending into their embedded base subtrees.
struct ExtNextRowReads<'a, 'r, 'd, F> {
    /// The base-field recorder shared with the base constraints.
    inner: &'r mut NextRowReads<'d>,
    /// Base subtrees already visited, keyed by address.
    bases: HashMap<*const SymbolicExpression<F>, ()>,
    /// Traversal stack reused across base subtrees.
    base_stack: Vec<(&'a SymbolicExpression<F>, bool)>,
}

impl<'a, F: Field, EF: ExtensionField<F>> SymbolicExprVisitor<'a, ExtLeaf<F, EF>>
    for ExtNextRowReads<'a, '_, '_, F>
{
    type Output = ();

    fn leaf(&mut self, leaf: &'a ExtLeaf<F, EF>) {
        // Permutation columns are not covered by the declaration, so only base subtrees matter.
        if let ExtLeaf::Base(base) = leaf {
            fold_symbolic_expression_cached(
                base,
                &mut *self.inner,
                &mut self.bases,
                &mut self.base_stack,
            );
        }
    }

    fn add(&mut self, _: &(), _: &()) {}

    fn sub(&mut self, _: &(), _: &()) {}

    fn neg(&mut self, _: &()) {}

    fn mul(&mut self, _: &(), _: &()) {}
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;
    use p3_field::extension::BinomialExtensionField;

    use super::*;
    use crate::{
        AirBuilder, AirLayout, ExtensionBuilder, SymbolicAirBuilder, SymbolicConstraints,
        WindowAccess, capture_symbolic_constraints,
    };

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;

    /// Declarations only: the constraints under test are captured separately.
    struct Declared {
        main_next: Vec<usize>,
        preprocessed_next: Vec<usize>,
    }

    impl BaseAir<F> for Declared {
        fn width(&self) -> usize {
            3
        }

        fn preprocessed_width(&self) -> usize {
            2
        }

        fn main_next_row_columns(&self) -> Vec<usize> {
            self.main_next.clone()
        }

        fn preprocessed_next_row_columns(&self) -> Vec<usize> {
            self.preprocessed_next.clone()
        }
    }

    /// Capture constraints over three main and two preprocessed columns.
    fn capture(eval: impl FnOnce(&mut SymbolicAirBuilder<F, EF>)) -> SymbolicConstraints<F, EF> {
        let layout = AirLayout {
            main_width: 3,
            preprocessed_width: 2,
            ..Default::default()
        };
        capture_symbolic_constraints(layout, eval)
    }

    fn check(
        air: &Declared,
        constraints: &SymbolicConstraints<F, EF>,
    ) -> Result<(), UndeclaredNextRowRead> {
        constraints.check_next_row_reads(&NextRowColumns::from_air(air))
    }

    /// Constrain the next row of main column 2 against the current row.
    fn main_transition(builder: &mut SymbolicAirBuilder<F, EF>) {
        let main = builder.main();
        let local = main.current(2).unwrap();
        let next = main.next(2).unwrap();
        builder.when_transition().assert_eq(next, local);
    }

    #[test]
    fn declaration_membership() {
        // Fixture state: main declares {0, 2, 2, 7} over 3 columns, preprocessed declares {1}.
        //
        //     main:          0 yes   1 no    2 yes   (7 lies past the width)
        //     preprocessed:  0 no    1 yes
        let air = Declared {
            main_next: vec![0, 2, 2, 7],
            preprocessed_next: vec![1],
        };
        let declared = NextRowColumns::from_air(&air);

        // Invariant: membership follows the declaration, and indices past the width are absent.
        let main: Vec<bool> = (0..4)
            .map(|column| declared.contains(NextRowTrace::Main, column))
            .collect();
        assert_eq!(main, [true, false, true, false]);
        assert!(!declared.contains(NextRowTrace::Main, 7));
        let preprocessed: Vec<bool> = (0..3)
            .map(|column| declared.contains(NextRowTrace::Preprocessed, column))
            .collect();
        assert_eq!(preprocessed, [false, true, false]);
    }

    #[test]
    fn declared_reads_pass() {
        // Fixture state: the only next-row read is main column 2, and it is declared.
        let constraints = capture(main_transition);
        let air = Declared {
            main_next: vec![2],
            preprocessed_next: vec![],
        };

        // Invariant: a read inside the declared set is accepted.
        assert_eq!(check(&air, &constraints), Ok(()));
    }

    #[test]
    fn undeclared_main_read_is_reported() {
        let constraints = capture(main_transition);

        // Mutation: the declaration covers other columns, or none at all.
        //
        //     declared {}     read {2}  -> column 2 reported
        //     declared {0, 1} read {2}  -> column 2 reported
        for main_next in [vec![], vec![0, 1]] {
            let air = Declared {
                main_next,
                preprocessed_next: vec![],
            };
            assert_eq!(
                check(&air, &constraints),
                Err(UndeclaredNextRowRead {
                    trace: NextRowTrace::Main,
                    column: 2,
                })
            );
        }
    }

    #[test]
    fn undeclared_preprocessed_read_is_reported() {
        // Fixture state: main column 0 equals preprocessed column 1 of the next row.
        let constraints = capture(|builder| {
            let local = builder.main().current(0).unwrap();
            let next = builder.preprocessed().next(1).unwrap();
            builder.assert_eq(local, next);
        });

        // Mutation: every main column is declared, no preprocessed column is.
        let air = Declared {
            main_next: vec![0, 1, 2],
            preprocessed_next: vec![],
        };

        // Invariant: the report names the preprocessed trace, not the main one.
        assert_eq!(
            check(&air, &constraints),
            Err(UndeclaredNextRowRead {
                trace: NextRowTrace::Preprocessed,
                column: 1,
            })
        );
    }

    #[test]
    fn unconstrained_next_row_load_passes() {
        // Fixture state: the next row is loaded but never reaches a constraint.
        let constraints = capture(|builder| {
            let main = builder.main();
            let _unused = main.next(1).unwrap();
            builder.assert_bool(main.current(1).unwrap());
        });
        let air = Declared {
            main_next: vec![],
            preprocessed_next: vec![],
        };

        // Invariant: only constrained reads count.
        assert_eq!(check(&air, &constraints), Ok(()));
    }

    #[test]
    fn read_inside_an_extension_constraint_is_reported() {
        // Fixture state: the only next-row read sits in a base subtree of an extension constraint.
        //
        //     ext constraint:  EF::from(main[next][0]) * 3 = 0
        let constraints = capture(|builder| {
            let next = builder.main().next(0).unwrap();
            let lifted: <SymbolicAirBuilder<F, EF> as ExtensionBuilder>::ExprEF = next.into();
            builder.assert_zero_ext(lifted * EF::from_u8(3));
        });
        assert!(constraints.base_constraints().is_empty());
        let air = Declared {
            main_next: vec![],
            preprocessed_next: vec![],
        };

        // Invariant: extension constraints are walked down to their base leaves.
        assert_eq!(
            check(&air, &constraints),
            Err(UndeclaredNextRowRead {
                trace: NextRowTrace::Main,
                column: 0,
            })
        );
    }

    #[test]
    fn first_violation_in_constraint_order_is_reported() {
        // Fixture state: two undeclared reads in two constraints.
        //
        //     constraint 0 reads main[next][1]
        //     constraint 1 reads main[next][0]
        let constraints = capture(|builder| {
            let main = builder.main();
            builder.assert_zero(main.next(1).unwrap());
            builder.assert_zero(main.next(0).unwrap());
        });
        let air = Declared {
            main_next: vec![],
            preprocessed_next: vec![],
        };

        // Invariant: the earliest constraint decides the report.
        assert_eq!(
            check(&air, &constraints),
            Err(UndeclaredNextRowRead {
                trace: NextRowTrace::Main,
                column: 1,
            })
        );
    }
}
