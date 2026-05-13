use alloc::vec::Vec;

use p3_air::Air;
use p3_air::symbolic::{AirLayout, ConstraintLayout, SymbolicExpression, SymbolicExpressionExt};
use p3_field::{Algebra, ExtensionField, Field};
use p3_lookup::{InteractionSymbolicBuilder, Kind, Lookup, LookupProtocol};
use p3_util::log2_ceil_usize;
use tracing::instrument;

#[instrument(
    name = "compute constraint layout with lookups",
    skip_all,
    level = "debug"
)]
pub fn get_constraint_layout<F, EF, A, LG>(
    air: &A,
    layout: AirLayout,
    contexts: &[Lookup<F>],
    lookup_gadget: &LG,
) -> ConstraintLayout
where
    F: Field,
    EF: ExtensionField<F>,
    A: Air<InteractionSymbolicBuilder<F, EF>>,
    SymbolicExpressionExt<F, EF>: Algebra<EF>,
    LG: LookupProtocol,
{
    let num_challenges = contexts.len() * lookup_gadget.num_challenges();
    let num_permutation_values = contexts
        .iter()
        .filter(|c| matches!(&c.kind, Kind::Global(_)))
        .count();
    let layout = AirLayout {
        permutation_width: contexts.len(),
        num_permutation_challenges: num_challenges,
        num_permutation_values,
        ..layout
    };
    let mut builder = InteractionSymbolicBuilder::new(layout);
    lookup_gadget.eval_air_and_lookups(air, &mut builder, contexts);
    builder.constraint_layout()
}

pub fn get_log_num_quotient_chunks<F, EF, A, LG>(
    air: &A,
    layout: AirLayout,
    contexts: &[Lookup<F>],
    is_zk: usize,
    lookup_gadget: &LG,
) -> usize
where
    F: Field,
    EF: ExtensionField<F>,
    A: Air<InteractionSymbolicBuilder<F, EF>>,
    SymbolicExpressionExt<F, EF>: Algebra<EF>,
    LG: LookupProtocol,
{
    assert!(is_zk <= 1, "is_zk must be either 0 or 1");

    if let Some(degree_hint) = air.max_constraint_degree() {
        let lookup_degree = contexts
            .iter()
            .map(|ctx| lookup_gadget.constraint_degree(ctx))
            .max()
            .unwrap_or(0);
        let max_degree = degree_hint.max(lookup_degree);
        let constraint_degree = (max_degree + is_zk).max(2);
        let result = log2_ceil_usize(constraint_degree - 1);

        debug_assert!(
            {
                let actual = get_max_constraint_degree(air, layout, contexts, lookup_gadget);
                max_degree >= actual
            },
            "max_constraint_degree() hint {} with lookup degree {} is too small; \
             symbolic evaluation found a larger degree",
            degree_hint,
            lookup_degree
        );

        return result;
    }

    // We pad to at least degree 2, since a quotient argument doesn't make sense with smaller degrees.
    let constraint_degree =
        (get_max_constraint_degree(air, layout, contexts, lookup_gadget) + is_zk).max(2);

    // The quotient's actual degree is approximately (max_constraint_degree - 1) n,
    // where subtracting 1 comes from division by the vanishing polynomial.
    // But we pad it to a power of two so that we can efficiently decompose the quotient.
    log2_ceil_usize(constraint_degree - 1)
}

#[instrument(name = "infer constraint degree", skip_all, level = "debug")]
pub fn get_max_constraint_degree<F, EF, A, LG>(
    air: &A,
    layout: AirLayout,
    contexts: &[Lookup<F>],
    lookup_gadget: &LG,
) -> usize
where
    F: Field,
    EF: ExtensionField<F>,
    A: Air<InteractionSymbolicBuilder<F, EF>>,
    SymbolicExpressionExt<F, EF>: Algebra<EF>,
    LG: LookupProtocol,
{
    let (base, extension) = get_symbolic_constraints(air, layout, contexts, lookup_gadget);
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
    layout: AirLayout,
    contexts: &[Lookup<F>],
    lookup_gadget: &LG,
) -> (
    Vec<SymbolicExpression<F>>,
    Vec<SymbolicExpressionExt<F, EF>>,
)
where
    F: Field,
    EF: ExtensionField<F>,
    A: Air<InteractionSymbolicBuilder<F, EF>>,
    SymbolicExpressionExt<F, EF>: Algebra<EF>,
    LG: LookupProtocol,
{
    let num_lookups = contexts.len();
    let num_challenges = num_lookups * lookup_gadget.num_challenges();
    let num_permutation_values = contexts
        .iter()
        .filter(|c| matches!(&c.kind, Kind::Global(_)))
        .count();
    let layout = AirLayout {
        permutation_width: num_lookups,
        num_permutation_challenges: num_challenges,
        num_permutation_values,
        ..layout
    };
    let mut builder = InteractionSymbolicBuilder::new(layout);

    // Evaluate AIR and lookup constraints.
    lookup_gadget.eval_air_and_lookups(air, &mut builder, contexts);
    let base_constraints = builder.base_constraints();
    let extension_constraints = builder.extension_constraints();
    (base_constraints, extension_constraints)
}
