use alloc::vec::Vec;

use p3_air::Air;
use p3_air::symbolic::{
    AirLayout, ConstraintLayout, SymbolicConstraints, SymbolicExpression, SymbolicExpressionExt,
};
use p3_commit::PolynomialSpace;
use p3_field::{Algebra, ExtensionField, Field};
use p3_lookup::{InteractionSymbolicBuilder, Lookup, LookupProtocol};
use p3_uni_stark::AirConstraints;
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
    layout.validate_against_air(air);
    // Permutation trace: accumulator column plus one fraction column per
    // lookup, or empty when the AIR declares no lookup.
    //
    // Each AIR with lookups commits exactly one terminal extension value.
    let (permutation_width, num_permutation_values) = if contexts.is_empty() {
        (0, 0)
    } else {
        (contexts.len() + 1, 1)
    };
    let num_challenges = contexts.len() * lookup_gadget.num_challenges();
    let layout = AirLayout {
        permutation_width,
        num_permutation_challenges: num_challenges,
        num_permutation_values,
        ..layout
    };
    let mut builder = InteractionSymbolicBuilder::new(layout);
    lookup_gadget.eval_air_and_lookups(air, &mut builder, contexts);
    builder.constraint_layout()
}

/// Quotient sizing using the domain's transition-selector degree.
///
/// Circle needs a full symbolic bound, including periodic columns and every
/// transition-selector factor. A domain-independent hint alone cannot supply it.
pub fn get_log_num_quotient_chunks_for_domain<F, EF, A, LG>(
    air: &A,
    layout: AirLayout,
    domain: impl PolynomialSpace<Val = F>,
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
    let constraints = evaluate_constraints(air, layout, contexts, lookup_gadget);
    AirConstraints::new(air, constraints).log_num_quotient_chunks(domain, is_zk)
}

/// Two-adic quotient sizing; use [`get_log_num_quotient_chunks_for_domain`] for other domains.
pub fn get_log_num_quotient_chunks<F, EF, A, LG>(
    air: &A,
    layout: AirLayout,
    trace_len: usize,
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

    let degree = get_max_constraint_degree(air, layout, trace_len, contexts, lookup_gadget)
        .max(air.max_constraint_degree().unwrap_or(0));
    // We pad to at least degree 2, since a quotient argument doesn't make sense with smaller degrees.
    let constraint_degree = (degree + is_zk).max(2);

    // The quotient's actual degree is approximately (max_constraint_degree - 1) n,
    // where subtracting 1 comes from division by the vanishing polynomial.
    // But we pad it to a power of two so that we can efficiently decompose the quotient.
    log2_ceil_usize(constraint_degree - 1)
}

#[instrument(name = "infer constraint degree", skip_all, level = "debug")]
pub fn get_max_constraint_degree<F, EF, A, LG>(
    air: &A,
    layout: AirLayout,
    trace_len: usize,
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
    let constraints = evaluate_constraints(air, layout, contexts, lookup_gadget);
    AirConstraints::new(air, constraints).max_constraint_degree(trace_len)
}

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
    let (base, extension, _) =
        evaluate_constraints(air, layout, contexts, lookup_gadget).into_parts();
    (base, extension)
}

/// Evaluate an AIR and its lookups symbolically, keeping both constraint streams.
///
/// # Panics
///
/// Panics if a layout width that the AIR itself determines disagrees with the AIR.
#[instrument(name = "evaluate constraints symbolically", skip_all, level = "debug")]
pub(crate) fn evaluate_constraints<F, EF, A, LG>(
    air: &A,
    layout: AirLayout,
    contexts: &[Lookup<F>],
    lookup_gadget: &LG,
) -> SymbolicConstraints<F, EF>
where
    F: Field,
    EF: ExtensionField<F>,
    A: Air<InteractionSymbolicBuilder<F, EF>>,
    SymbolicExpressionExt<F, EF>: Algebra<EF>,
    LG: LookupProtocol,
{
    layout.validate_against_air(air);
    let num_lookups = contexts.len();
    let num_challenges = num_lookups * lookup_gadget.num_challenges();
    // Permutation trace: accumulator column plus one fraction column per
    // lookup, or empty when the AIR declares no lookup.
    //
    // Each AIR with lookups commits exactly one terminal extension value.
    let (permutation_width, num_permutation_values) = if num_lookups == 0 {
        (0, 0)
    } else {
        (num_lookups + 1, 1)
    };
    let layout = AirLayout {
        permutation_width,
        num_permutation_challenges: num_challenges,
        num_permutation_values,
        ..layout
    };
    let mut builder = InteractionSymbolicBuilder::new(layout);

    // Evaluate AIR and lookup constraints.
    lookup_gadget.eval_air_and_lookups(air, &mut builder, contexts);
    builder.finish()
}

#[cfg(test)]
mod tests {
    use p3_air::{AirBuilder, BaseAir, WindowAccess};
    use p3_baby_bear::BabyBear;
    use p3_lookup::LogUpGadget;

    use super::*;

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
        impl Air<InteractionSymbolicBuilder<BabyBear, BabyBear>> for CubicAir {
            fn eval(&self, builder: &mut InteractionSymbolicBuilder<BabyBear, BabyBear>) {
                let x = builder.main().current_slice()[0];
                builder.assert_zero(x * x * x);
            }
        }
        let layout = AirLayout {
            main_width: 1,
            ..Default::default()
        };
        for (is_zk, expected) in [(0, 1), (1, 2)] {
            assert_eq!(
                get_log_num_quotient_chunks::<BabyBear, BabyBear, _, _>(
                    &CubicAir,
                    layout,
                    8,
                    &[],
                    is_zk,
                    &LogUpGadget::new()
                ),
                expected
            );
        }
    }
}
