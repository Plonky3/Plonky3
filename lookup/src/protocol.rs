//! The lookup protocol trait.

use p3_air::{Air, PermutationAirBuilder};
use p3_field::Field;
use p3_matrix::dense::RowMajorMatrix;
use p3_uni_stark::{StarkGenericConfig, Val};

use crate::types::{Kind, Lookup, LookupData, LookupError};

/// A lookup protocol that can evaluate constraints, generate permutation
/// traces, and verify global sums.
///
/// Each lookup uses exactly one auxiliary column in the permutation trace,
/// matching the single [`Lookup::column`] field.
pub trait LookupProtocol {
    /// Random challenges per lookup (2 for LogUp: `α`, `β`).
    fn num_challenges(&self) -> usize;

    /// Evaluate a local (intra-AIR) lookup constraint.
    fn eval_local<AB: PermutationAirBuilder>(&self, builder: &mut AB, lookup: &Lookup<AB::F>);

    /// Evaluate a global (cross-AIR) lookup constraint.
    fn eval_global<AB: PermutationAirBuilder>(
        &self,
        builder: &mut AB,
        lookup: &Lookup<AB::F>,
        cumulative_sum: AB::ExprEF,
    );

    /// Evaluate all lookups, dispatching by [`Kind`].
    fn eval_all<AB: PermutationAirBuilder>(&self, builder: &mut AB, lookups: &[Lookup<AB::F>]) {
        let mut pv_idx = 0;
        for lookup in lookups {
            match &lookup.kind {
                Kind::Local => self.eval_local(builder, lookup),
                Kind::Global(_) => {
                    let expected = builder.permutation_values()[pv_idx].clone();
                    pv_idx += 1;
                    self.eval_global(builder, lookup, expected.into());
                }
            }
        }
        assert_eq!(pv_idx, builder.permutation_values().len());
    }

    /// Generate the permutation trace matrix.
    fn generate_permutation<SC: StarkGenericConfig>(
        &self,
        main: &RowMajorMatrix<Val<SC>>,
        preprocessed: &Option<RowMajorMatrix<Val<SC>>>,
        public_values: &[Val<SC>],
        lookups: &[Lookup<Val<SC>>],
        lookup_data: &mut [LookupData<SC::Challenge>],
        challenges: &[SC::Challenge],
    ) -> RowMajorMatrix<SC::Challenge>;

    /// Verify global cumulative sums balance to zero.
    fn verify_global_sum<EF: Field>(&self, cumulative_sums: &[EF]) -> Result<(), LookupError>;

    /// Polynomial degree of the transition constraint for a given lookup.
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
