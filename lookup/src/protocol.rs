//! The lookup protocol trait.

use p3_air::{Air, PermutationAirBuilder};
use p3_field::Field;
use p3_matrix::dense::RowMajorMatrix;
use p3_uni_stark::{StarkGenericConfig, Val};

use crate::types::{Lookup, LookupError, LookupTerminal};

/// Lookup protocol over the single-terminal LogUp layout.
///
/// # Responsibilities
///
/// - Emit the per-row pinning constraint for each lookup's fraction column.
/// - Emit the accumulator constraints that bind the AIR's committed terminal.
/// - Generate the auxiliary permutation trace from the main trace.
/// - Verify the cross-AIR sum of committed terminals balances to zero.
///
/// # Trace layout
///
/// ```text
///     permutation column 0:      shared accumulator
///     permutation column i + 1:  fraction column for lookup i
/// ```
///
/// Every AIR declaring any lookup commits exactly one extension-field terminal.
pub trait LookupProtocol {
    /// Random challenges per lookup (2 for LogUp: `α`, `β`).
    fn num_challenges(&self) -> usize;

    /// Pin one lookup's fraction column to its per-row LogUp value.
    ///
    /// # Constraint
    ///
    /// On every row, the fraction column is forced to equal `V_i / U_i`:
    ///
    /// - `V_i` — numerator (signed sum of multiplicities times the cross-products).
    /// - `U_i` — denominator (product of all `(alpha - combined_tuple)` terms).
    fn eval_fraction<AB: PermutationAirBuilder>(&self, builder: &mut AB, lookup: &Lookup<AB::F>);

    /// Constrain the shared accumulator and bind the AIR's committed terminal.
    ///
    /// # Constraints
    ///
    /// - **First row** — the accumulator is anchored to zero.
    /// - **Transition** — each step adds the per-row sum of every fraction column.
    /// - **Last row** — accumulator plus the last row's fractions equals the committed terminal.
    fn eval_accumulator<AB: PermutationAirBuilder>(
        &self,
        builder: &mut AB,
        lookups: &[Lookup<AB::F>],
        terminal: AB::ExprEF,
    );

    /// Evaluate every lookup constraint for one AIR.
    fn eval_all<AB: PermutationAirBuilder>(&self, builder: &mut AB, lookups: &[Lookup<AB::F>]) {
        // No lookups means no permutation column and no committed terminal.
        if lookups.is_empty() {
            assert_eq!(
                0,
                builder.permutation_values().len(),
                "permutation values count mismatch"
            );
            return;
        }

        // Exactly one terminal per AIR with lookups.
        assert_eq!(
            1,
            builder.permutation_values().len(),
            "permutation values count mismatch"
        );
        let terminal = builder.permutation_values()[0].clone();

        // Pin each lookup's fraction column to its per-row rational value.
        for lookup in lookups {
            self.eval_fraction(builder, lookup);
        }

        // Pin the accumulator: anchor, transition, and terminal binding.
        self.eval_accumulator(builder, lookups, terminal.into());
    }

    /// Generate the permutation trace and the AIR's single terminal.
    ///
    /// # Returns
    ///
    /// - A trace matrix with the accumulator at column `0` and one fraction
    ///   column per declared lookup.
    /// - The AIR's terminal: `Some(_)` when any lookup is declared, `None` otherwise.
    fn generate_permutation<SC: StarkGenericConfig>(
        &self,
        main: &RowMajorMatrix<Val<SC>>,
        preprocessed: &Option<RowMajorMatrix<Val<SC>>>,
        public_values: &[Val<SC>],
        lookups: &[Lookup<Val<SC>>],
        challenges: &[SC::Challenge],
    ) -> (
        RowMajorMatrix<SC::Challenge>,
        Option<LookupTerminal<SC::Challenge>>,
    );

    /// Verify the cross-AIR sum of committed terminals is zero.
    ///
    /// - Present terminals contribute to the total.
    /// - Absent terminals (AIRs without lookups) contribute nothing.
    /// - A non-zero total signals an unbalanced lookup in the batch.
    fn verify_terminal_sum<EF: Field>(
        &self,
        terminals: &[Option<LookupTerminal<EF>>],
    ) -> Result<(), LookupError>;

    /// Polynomial degree of the highest-degree constraint emitted for one lookup.
    fn constraint_degree<F: Field>(&self, lookup: &Lookup<F>) -> usize;

    /// Evaluate AIR constraints followed by lookup constraints.
    fn eval_air_and_lookups<AB, A>(&self, air: &A, builder: &mut AB, lookups: &[Lookup<AB::F>])
    where
        AB: PermutationAirBuilder,
        A: Air<AB>,
    {
        air.eval(builder);
        if !lookups.is_empty() {
            self.eval_all(builder, lookups);
        } else {
            // No lookups declared: catch the inconsistent case where the builder
            // was nonetheless given permutation values to consume.
            assert_eq!(
                0,
                builder.permutation_values().len(),
                "permutation values count mismatch"
            );
        }
    }
}
