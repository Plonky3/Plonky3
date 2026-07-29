//! Structural metadata an AIR exposes to the multilinear prover and verifier.

use alloc::vec::Vec;

use p3_air::{Air, AirLayout, SymbolicAirBuilder, get_all_symbolic_constraints};
use p3_field::{ExtensionField, Field};

/// Structural facts about an AIR needed before any proving begins.
///
/// These are derived once from the AIR definition.
/// They depend only on the constraint structure, not on the trace contents or the field of evaluation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConstraintMetadata {
    /// Number of base constraints the AIR asserts.
    ///
    /// Covers transition and boundary constraints.
    /// Excludes lookup and permutation arguments, matching the AIR's own constraint-count hint.
    pub num_constraints: usize,
    /// Upper bound on the largest total degree of any asserted constraint.
    ///
    /// Exact when derived by symbolic evaluation.
    /// May overestimate when the AIR supplies a hint, which need only be an upper bound.
    ///
    /// Degree follows the symbolic selector convention:
    /// - first-row and last-row selectors each count as degree one,
    /// - the transition selector counts as degree zero,
    /// - every trace column counts as degree one.
    pub max_constraint_degree: usize,
    /// Main-trace column indices whose next row is read by some constraint.
    ///
    /// These columns need a shifted-successor opening claim in addition to a point claim.
    pub next_row_main_columns: Vec<usize>,
    /// Preprocessed-trace column indices whose next row is read by some constraint.
    ///
    /// Empty when the AIR has no preprocessed trace.
    pub next_row_preprocessed_columns: Vec<usize>,
}

impl ConstraintMetadata {
    /// Derive the metadata from an AIR definition and its layout.
    ///
    /// # Arguments
    ///
    /// - `air`: the AIR whose constraints are inspected.
    /// - `layout`: column widths and public-value counts used to size the symbolic pass.
    ///
    /// # Performance
    ///
    /// Runs at most one symbolic evaluation of the AIR.
    /// When the AIR supplies both a constraint-count hint and a max-degree hint, no symbolic pass runs.
    pub fn from_air<F, EF, A>(air: &A, layout: AirLayout) -> Self
    where
        F: Field,
        EF: ExtensionField<F>,
        A: Air<SymbolicAirBuilder<F, EF>>,
    {
        // Shift structure is declared by the AIR directly; reading it needs no symbolic pass.
        let next_row_main_columns = air.main_next_row_columns();
        let next_row_preprocessed_columns = air.preprocessed_next_row_columns();

        // An AIR may hint its own constraint count and max degree to avoid the symbolic pass.
        let hinted_count = air.num_constraints();
        let hinted_degree = air.max_constraint_degree();

        let (num_constraints, max_constraint_degree) = match (hinted_count, hinted_degree) {
            // Both hints present: trust them and never evaluate the AIR.
            (Some(count), Some(degree)) => (count, degree),
            // At least one hint missing: evaluate symbolically once and fill in the gaps.
            _ => {
                // Collect every asserted constraint as a symbolic expression tree.
                //
                //     base[i] : i-th base-field constraint
                //     ext[j]  : j-th extension-field constraint (lookups / permutation args)
                let (base, ext) = get_all_symbolic_constraints::<F, EF, A>(air, layout);

                // Count covers base constraints only, matching the scope of the AIR's count hint.
                let count = hinted_count.unwrap_or(base.len());

                // Max degree is taken over both constraint families under the symbolic convention.
                let degree = hinted_degree.unwrap_or_else(|| {
                    // Largest degree among base-field constraints, or zero if there are none.
                    let base_degree = base.iter().map(|c| c.degree_multiple()).max().unwrap_or(0);
                    // Largest degree among extension-field constraints, or zero if there are none.
                    let ext_degree = ext.iter().map(|c| c.degree_multiple()).max().unwrap_or(0);
                    base_degree.max(ext_degree)
                });

                (count, degree)
            }
        };

        Self {
            num_constraints,
            max_constraint_degree,
            next_row_main_columns,
            next_row_preprocessed_columns,
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use p3_air::{Air, AirBuilder, AirLayout, BaseAir, ExtensionBuilder, WindowAccess};
    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;
    use p3_field::extension::BinomialExtensionField;

    use super::*;

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;

    /// Fibonacci AIR: two boundary constraints, two transition constraints, one final constraint.
    ///
    /// - first row: `local[0] == public[0]` and `local[1] == public[1]`
    /// - transition: `local[1] == next[0]` and `local[0] + local[1] == next[1]`
    /// - last row: `local[1] == public[2]`
    struct FibAir;

    impl<X> BaseAir<X> for FibAir {
        fn width(&self) -> usize {
            2
        }
        fn num_public_values(&self) -> usize {
            3
        }
    }

    impl<AB: AirBuilder> Air<AB> for FibAir {
        fn eval(&self, builder: &mut AB) {
            // Pull the two-row window and the public inputs into local bindings.
            let main = builder.main();
            let local = main.current_slice();
            let next = main.next_slice();
            let pis = builder.public_values();
            let (a, b, x) = (pis[0], pis[1], pis[2]);

            // Boundary at the first row pins the seed pair: degree-1 trace times a degree-1 selector.
            let mut first = builder.when_first_row();
            first.assert_eq(local[0], a);
            first.assert_eq(local[1], b);

            // Transition couples adjacent rows: degree-1 trace times a degree-0 selector.
            let mut trans = builder.when_transition();
            trans.assert_eq(local[1], next[0]);
            trans.assert_eq(local[0] + local[1], next[1]);

            // Boundary at the last row pins the claimed output.
            builder.when_last_row().assert_eq(local[1], x);
        }
    }

    /// Degree-3 AIR with one transition constraint that reads only the next row of column 0.
    ///
    /// Constraint: `local[0] * local[1] * local[2] == next[0]` on transition rows.
    struct ProductAir;

    impl<X> BaseAir<X> for ProductAir {
        fn width(&self) -> usize {
            3
        }
        fn main_next_row_columns(&self) -> Vec<usize> {
            // Only the next row of column 0 is read; columns 1 and 2 are current-row only.
            vec![0]
        }
    }

    impl<AB: AirBuilder> Air<AB> for ProductAir {
        fn eval(&self, builder: &mut AB) {
            // Bind the current row and the single shifted entry the constraint reads.
            let main = builder.main();
            let local = main.current_slice();
            let next0 = main.next_slice()[0];

            // Product of three trace columns has degree-3 in the trace variables.
            let product = local[0] * local[1] * local[2];
            builder.when_transition().assert_eq(product, next0);
        }
    }

    /// AIR with a preprocessed trace whose only next-row access is preprocessed column 1.
    struct PreprocessedAir;

    impl<X> BaseAir<X> for PreprocessedAir {
        fn width(&self) -> usize {
            1
        }
        fn preprocessed_width(&self) -> usize {
            2
        }
        fn preprocessed_next_row_columns(&self) -> Vec<usize> {
            // Only preprocessed column 1 has its next row read.
            vec![1]
        }
    }

    impl<AB: AirBuilder> Air<AB> for PreprocessedAir {
        fn eval(&self, builder: &mut AB) {
            // A single degree-1 constraint on the main column keeps the symbolic pass non-empty.
            let local = builder.main().current_slice()[0];
            builder.assert_zero(local);
        }
    }

    /// AIR that supplies both hints and panics if evaluated.
    ///
    /// Used to prove that the metadata path skips symbolic evaluation when both hints are present.
    struct HintedNoEvalAir;

    impl<X> BaseAir<X> for HintedNoEvalAir {
        fn width(&self) -> usize {
            3
        }
        fn num_constraints(&self) -> Option<usize> {
            Some(7)
        }
        fn max_constraint_degree(&self) -> Option<usize> {
            Some(4)
        }
    }

    impl<AB: AirBuilder> Air<AB> for HintedNoEvalAir {
        fn eval(&self, _builder: &mut AB) {
            // Reaching here means the metadata path ran a symbolic pass it should have skipped.
            panic!("eval must not run when both hints are present");
        }
    }

    /// AIR with one base constraint and one extension constraint.
    ///
    /// The extension constraint stands in for a lookup or permutation argument.
    struct MixedAir;

    impl<X> BaseAir<X> for MixedAir {
        fn width(&self) -> usize {
            1
        }
    }

    impl<AB: ExtensionBuilder> Air<AB> for MixedAir {
        fn eval(&self, builder: &mut AB) {
            // One base-field constraint on the single column.
            let local = builder.main().current_slice()[0];
            builder.assert_zero(local);
            // One extension-field constraint, counted separately from the base ones.
            builder.assert_zero_ext(AB::ExprEF::ZERO);
        }
    }

    #[test]
    fn metadata_matches_symbolic_for_fibonacci() {
        // Fixture state: width-2 AIR, 3 public values, 5 asserted constraints.
        //
        // Symbolic degrees per constraint:
        //
        //     first-row    : selector(1) + trace(1) = 2
        //     transition   : selector(0) + trace(1) = 1
        //     last-row     : selector(1) + trace(1) = 2
        //
        // Max over all five is 2.
        let layout = AirLayout::from_air::<F>(&FibAir);
        let meta = ConstraintMetadata::from_air::<F, EF, _>(&FibAir, layout);

        // All five constraints are counted.
        assert_eq!(meta.num_constraints, 5);
        // The boundary constraints dominate the symbolic degree at 2.
        assert_eq!(meta.max_constraint_degree, 2);
        // Default next-row access covers every main column.
        assert_eq!(meta.next_row_main_columns, vec![0, 1]);
        // No preprocessed trace means no preprocessed shift claims.
        assert!(meta.next_row_preprocessed_columns.is_empty());
    }

    #[test]
    fn metadata_reports_high_degree_and_selective_next_columns() {
        // Fixture state: one transition constraint that multiplies three trace columns.
        //
        //     local[0] * local[1] * local[2]  ->  trace degree 3
        //     transition selector             ->  symbolic degree 0
        //     constraint symbolic degree      =   3
        let layout = AirLayout::from_air::<F>(&ProductAir);
        let meta = ConstraintMetadata::from_air::<F, EF, _>(&ProductAir, layout);

        // Exactly one constraint is asserted.
        assert_eq!(meta.num_constraints, 1);
        // The triple product gives symbolic degree 3.
        assert_eq!(meta.max_constraint_degree, 3);
        // Only column 0 has its next row read.
        assert_eq!(meta.next_row_main_columns, vec![0]);
        // No preprocessed trace.
        assert!(meta.next_row_preprocessed_columns.is_empty());
    }

    #[test]
    fn metadata_reports_preprocessed_next_columns() {
        // Fixture state: width-1 main trace, width-2 preprocessed trace.
        //
        // Only preprocessed column 1 is declared as next-row accessed.
        let layout = AirLayout::from_air::<F>(&PreprocessedAir);
        let meta = ConstraintMetadata::from_air::<F, EF, _>(&PreprocessedAir, layout);

        // The single main constraint is counted.
        assert_eq!(meta.num_constraints, 1);
        // A bare column assertion is degree 1.
        assert_eq!(meta.max_constraint_degree, 1);
        // The main column reads its own next row by default.
        assert_eq!(meta.next_row_main_columns, vec![0]);
        // Only preprocessed column 1 needs a shift claim.
        assert_eq!(meta.next_row_preprocessed_columns, vec![1]);
    }

    #[test]
    fn metadata_uses_hints_without_evaluating() {
        // Fixture state: an AIR whose `eval` panics, but which hints both count and degree.
        //
        // Invariant: with both hints present the symbolic pass is skipped,
        // so `eval` is never called and the hinted values are returned verbatim.
        let layout = AirLayout::from_air::<F>(&HintedNoEvalAir);
        let meta = ConstraintMetadata::from_air::<F, EF, _>(&HintedNoEvalAir, layout);

        // Hinted constraint count is returned unchanged.
        assert_eq!(meta.num_constraints, 7);
        // Hinted max degree is returned unchanged.
        assert_eq!(meta.max_constraint_degree, 4);
        // Next-row access still defaults to every main column.
        assert_eq!(meta.next_row_main_columns, vec![0, 1, 2]);
    }

    #[test]
    fn metadata_counts_base_constraints_only() {
        // The AIR asserts one base constraint and one extension constraint.
        // The count must match the AIR hint's scope, which excludes lookup and permutation arguments.
        let layout = AirLayout::from_air::<F>(&MixedAir);
        let meta = ConstraintMetadata::from_air::<F, EF, _>(&MixedAir, layout);

        // The extension constraint is excluded from the count.
        assert_eq!(meta.num_constraints, 1);
        // Degree still spans both families: the base column is degree 1, the extension constant degree 0.
        assert_eq!(meta.max_constraint_degree, 1);
    }
}
