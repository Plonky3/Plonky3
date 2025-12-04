use alloc::vec::Vec;

use p3_field::{ExtensionField, Field};
use p3_lookup::lookup_traits::{AirLookupHandler, Lookup, LookupData, LookupGadget};
use p3_uni_stark::{SymbolicAirBuilder, SymbolicExpression};
use p3_util::log2_ceil_usize;
use tracing::instrument;

#[instrument(name = "infer log of constraint degree", skip_all)]
pub fn get_log_num_quotient_chunks<F, EF, A, LG>(
    air: &A,
    preprocessed_width: usize,
    num_public_values: usize,
    contexts: &[Lookup<F>],
    lookup_data: &[LookupData<EF>],
    is_zk: usize,
    lookup_gadget: &LG,
) -> usize
where
    F: Field,
    EF: ExtensionField<F>,
    A: AirLookupHandler<SymbolicAirBuilder<F, EF>>,
    SymbolicExpression<EF>: From<SymbolicExpression<F>>,
    LG: LookupGadget,
{
    assert!(is_zk <= 1, "is_zk must be either 0 or 1");
    // We pad to at least degree 2, since a quotient argument doesn't make sense with smaller degrees.
    let constraint_degree = (get_max_constraint_degree(
        air,
        preprocessed_width,
        num_public_values,
        contexts,
        lookup_data,
        lookup_gadget,
    ) + is_zk)
        .max(2);

    // The quotient's actual degree is approximately (max_constraint_degree - 1) n,
    // where subtracting 1 comes from division by the vanishing polynomial.
    // But we pad it to a power of two so that we can efficiently decompose the quotient.
    log2_ceil_usize(constraint_degree - 1)
}

#[instrument(name = "infer constraint degree", skip_all, level = "debug")]
pub fn get_max_constraint_degree<F, EF, A, LG>(
    air: &A,
    preprocessed_width: usize,
    num_public_values: usize,
    contexts: &[Lookup<F>],
    lookup_data: &[LookupData<EF>],
    lookup_gadget: &LG,
) -> usize
where
    F: Field,
    EF: ExtensionField<F>,
    A: AirLookupHandler<SymbolicAirBuilder<F, EF>>,
    SymbolicExpression<EF>: From<SymbolicExpression<F>>,
    LG: LookupGadget,
{
    let (base, extension) = get_symbolic_constraints(
        air,
        preprocessed_width,
        num_public_values,
        contexts,
        lookup_data,
        lookup_gadget,
    );
    let base_degree = base.iter().map(|c| c.degree_multiple()).max().unwrap_or(0);
    let extension_degree = extension
        .iter()
        .map(|c| c.degree_multiple())
        .max()
        .unwrap_or(0);
    base_degree.max(extension_degree)
}

#[instrument(name = "evaluate constraints symbolically", skip_all, level = "debug")]
pub fn get_symbolic_constraints<F, EF, A, LG>(
    air: &A,
    preprocessed_width: usize,
    num_public_values: usize,
    contexts: &[Lookup<F>],
    lookup_data: &[LookupData<EF>],
    lookup_gadget: &LG,
) -> (Vec<SymbolicExpression<F>>, Vec<SymbolicExpression<EF>>)
where
    F: Field,
    EF: ExtensionField<F>,
    A: AirLookupHandler<SymbolicAirBuilder<F, EF>>,
    SymbolicExpression<EF>: From<SymbolicExpression<F>>,
    LG: LookupGadget,
{
    let num_lookups = contexts.len();
    let num_aux_cols = num_lookups * lookup_gadget.num_aux_cols();
    let num_challenges = num_lookups * lookup_gadget.num_challenges();
    let mut builder = SymbolicAirBuilder::new(
        preprocessed_width,
        air.width(),
        num_public_values,
        num_aux_cols,
        num_challenges,
    );

    // Evaluate AIR and lookup constraints.
    <A as AirLookupHandler<_>>::eval(air, &mut builder, contexts, lookup_data, lookup_gadget);
    let base_constraints = builder.base_constraints();
    let extension_constraints = builder.extension_constraints();
    (base_constraints, extension_constraints)
}
