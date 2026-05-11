//! Trait for recursive AIR constraint evaluation.

use hashbrown::HashMap;
use p3_air::symbolic::AirLayout;
use p3_air::{Air, SymbolicExpressionExt};
use p3_batch_stark::symbolic::{get_log_num_quotient_chunks, get_symbolic_constraints};
use p3_circuit::CircuitBuilder;
use p3_circuit::symbolic::{ColumnsTargets, SymbolicCompiler};
use p3_field::{Algebra, ExtensionField, Field};
use p3_lookup::lookup_traits::{Kind, Lookup, LookupData, LookupGadget};
use p3_uni_stark::{SymbolicAirBuilder, SymbolicExpression};

use crate::Target;
use crate::types::RecursiveLagrangeSelectors;

/// Structure holding lookup verification data:
///
/// - `contexts`: Slice of lookup contexts used in the AIR.
/// - `lookup_data`: Slice of lookup data for global lookups.
pub struct LookupMetadata<'a, F: Field> {
    pub contexts: &'a [Lookup<F>],
    pub lookup_data: &'a [LookupData<usize>],
}
/// Trait for evaluating AIR constraints within a recursive verification circuit.
///
/// This trait provides methods for computing constraint evaluations over circuit targets
/// rather than concrete field values.
pub trait RecursiveAir<F: Field, EF: ExtensionField<F>, LG: LookupGadget> {
    /// Returns the number of columns in the AIR's execution trace.
    ///
    /// This corresponds to the width of the trace matrix.
    fn width(&self) -> usize;

    /// Evaluate all AIR constraints and fold them into a single target.
    ///
    /// This method:
    /// 1. Retrieves all symbolic constraints from the AIR
    /// 2. Converts them to circuit targets
    /// 3. Folds them using powers of alpha: acc = acc * alpha + constraint
    ///
    /// # Parameters
    /// - `builder`: Circuit builder for creating operations
    /// - `sels`: Row selectors and vanishing inverse for constraint evaluation
    /// - `alpha`: Challenge used for folding constraints
    /// - `contexts`: Lookup contexts used in the AIR
    /// - `lookup_data`: Data for global lookups
    /// - `columns`: Trace columns (local, next) and public values
    /// - `lookup_gadget`: Gadget for handling lookups in the circuit
    ///
    /// # Returns
    /// A single target representing the folded constraint evaluation
    fn eval_folded_circuit(
        &self,
        builder: &mut CircuitBuilder<EF>,
        sels: &RecursiveLagrangeSelectors,
        alpha: &Target,
        lookup_metadata: &LookupMetadata<'_, F>,
        columns: ColumnsTargets<'_>,
        lookup_gadget: &LG,
    ) -> Target;

    /// Compute the log of the quotient polynomial degree.
    ///
    /// The quotient polynomial is formed by dividing the constraint polynomial
    /// by the vanishing polynomial. Its degree depends on:
    /// - The maximum constraint degree
    /// - Number of public values
    /// - Whether ZK randomization is used
    ///
    /// # Parameters
    /// - `num_public_values`: Number of public input values
    /// - `is_zk`: Whether ZK mode is enabled (0 or 1)
    ///
    /// # Returns
    /// Log₂ of the number of quotient chunks
    fn get_log_num_quotient_chunks(
        &self,
        preprocessed_width: usize,
        contexts: &[Lookup<F>],
        lookup_data: &[LookupData<usize>],
        is_zk: usize,
        lookup_gadget: &LG,
    ) -> usize;
}

impl<F: Field, EF: ExtensionField<F>, A, LG: LookupGadget> RecursiveAir<F, EF, LG> for A
where
    A: Air<SymbolicAirBuilder<F, EF>>,
    SymbolicExpressionExt<F, EF>: Algebra<SymbolicExpression<F>> + Algebra<EF>,
{
    fn width(&self) -> usize {
        Self::width(self)
    }

    fn eval_folded_circuit(
        &self,
        builder: &mut CircuitBuilder<EF>,
        sels: &RecursiveLagrangeSelectors,
        alpha: &Target,
        lookup_metadata: &LookupMetadata<'_, F>,
        columns: ColumnsTargets<'_>,
        lookup_gadget: &LG,
    ) -> Target {
        builder.push_scope("eval_folded_circuit");

        let LookupMetadata {
            contexts,
            lookup_data: _,
        } = lookup_metadata;

        let num_preprocessed = columns.local_prep_values.len();
        let num_permutation_values = contexts
            .iter()
            .filter(|c| matches!(&c.kind, Kind::Global(_)))
            .count();
        let layout = AirLayout {
            preprocessed_width: num_preprocessed,
            main_width: self.width(),
            num_public_values: self.num_public_values(),
            num_permutation_values,
            ..Default::default()
        };
        let (base_symbolic_constraints, extension_symbolic_constraints) =
            get_symbolic_constraints(self, layout, contexts, lookup_gadget);

        // Fold all constraints: result = c₀ + α·c₁ + α²·c₂ + ...
        //
        // Converting directly the tree SymbolicExpression<F> → SymbolicExpression<EF>
        // destroys Arc-based sub-expression sharing and causes exponential blowup.
        // Instead, we lift F → EF constants directly.
        //
        // Additionally, the cache is shared across all constraint calls to reuse circuit
        // operations for sub-expressions shared between different constraints.
        let compiler = SymbolicCompiler::new(sels.row_selectors, &columns);
        let mut acc = builder.define_const(EF::ZERO);
        let mut base_cache = HashMap::new();
        for s_c in &base_symbolic_constraints {
            let constraints = compiler.compile_base(s_c, builder, &mut base_cache);
            acc = builder.mul_add(acc, *alpha, constraints);
        }

        let mut ext_cache = HashMap::new();
        for s_c in &extension_symbolic_constraints {
            let constraints = compiler.compile_ext(s_c, builder, &mut base_cache, &mut ext_cache);
            acc = builder.mul_add(acc, *alpha, constraints);
        }

        builder.pop_scope();
        acc
    }

    fn get_log_num_quotient_chunks(
        &self,
        preprocessed_width: usize,
        contexts: &[Lookup<F>],
        _lookup_data: &[LookupData<usize>],
        is_zk: usize,
        lookup_gadget: &LG,
    ) -> usize
    where
        F: Field,
        EF: ExtensionField<F>,
        SymbolicExpressionExt<F, EF>: Algebra<SymbolicExpression<F>>,
        LG: LookupGadget,
    {
        let layout = AirLayout {
            preprocessed_width,
            main_width: self.width(),
            num_public_values: self.num_public_values(),
            ..Default::default()
        };
        get_log_num_quotient_chunks(self, layout, contexts, is_zk, lookup_gadget)
    }
}
