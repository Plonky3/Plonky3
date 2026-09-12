//! Protocol-instance shape: data orthogonal to the runtime config.

use p3_air::Air;
use p3_air::symbolic::{
    AirLayout, SymbolicAirBuilder, get_all_symbolic_constraints,
    get_max_constraint_degree_extension,
};
use p3_commit::PolynomialSpace;
use p3_field::{ExtensionField, Field};

/// AIR-derived shape used in DEEP-ALI and composition-error bounds.
#[derive(Copy, Clone, Debug)]
pub struct StarkAirParams {
    pub num_constraints: usize,
    pub max_constraint_degree: usize,
    /// Exact number of committed quotient chunks, including ZK degree padding and doubling.
    pub num_quotient_chunks: usize,
    /// DEEP-ALI `max_combo`: maximum number of out-of-domain points
    /// referenced per column (typically 2 for `local`/`next`).
    pub max_combo: usize,
}

impl StarkAirParams {
    /// Derive `num_constraints` and `max_constraint_degree` by symbolically
    /// evaluating the AIR's constraints. The caller supplies `max_combo`
    /// (typically `2` for an AIR using `local`/`next` rotations, `1` if
    /// there is no transition constraint).
    /// `trace_domain` must be the prover's original trace domain (before ZK padding):
    /// its selector model and size determine the constraint degree. `is_zk` selects
    /// the prover's degree padding and quotient-chunk doubling.
    ///
    /// # `layout` must include every committed column width
    ///
    /// `layout` controls which column widths the symbolic builder allocates
    /// when evaluating constraints. A base-only layout (e.g.
    /// `AirLayout::from_air`, which fills only the `BaseAir` widths)
    /// leaves the permutation / lookup / preprocessed widths at `0`, so
    /// any constraints over those columns evaluate as identically zero and
    /// are dropped — resulting in an **overstated** security bound.
    ///
    /// For an AIR that uses lookups or other permutation arguments,
    /// construct the layout with the full set of widths (base + permutation
    /// + preprocessed). For pure base AIRs, `AirLayout::from_air` is safe.
    pub fn from_air<F, EF, A>(
        air: &A,
        layout: AirLayout,
        trace_domain: impl PolynomialSpace<Val = F>,
        max_combo: usize,
        is_zk: bool,
    ) -> Self
    where
        F: Field,
        EF: ExtensionField<F>,
        A: Air<SymbolicAirBuilder<F, EF>>,
    {
        let (base, ext) = get_all_symbolic_constraints::<F, EF, A>(air, layout);
        let num_constraints = base.len() + ext.len();
        let transition_degree = trace_domain.transition_degree_multiple();
        let max_constraint_degree = if transition_degree == 0 {
            // Two-adic periodic columns have trace-size-dependent degrees.
            get_max_constraint_degree_extension::<F, EF, A>(air, layout, trace_domain.size())
        } else {
            // Circle selectors and periodic columns occupy the full trace space.
            base.iter()
                .map(|c| c.degree_multiple_with_transition(transition_degree))
                .chain(
                    ext.iter()
                        .map(|c| c.degree_multiple_with_transition(transition_degree)),
                )
                .max()
                .unwrap_or(0)
        }
        .max(1);
        let committed_degree = air
            .max_constraint_degree()
            .unwrap_or(max_constraint_degree)
            .max(max_constraint_degree);
        let zk = usize::from(is_zk);
        let num_quotient_chunks = committed_degree
            .checked_add(zk)
            .and_then(|degree| (degree.max(2) - 1).checked_next_power_of_two())
            .and_then(|chunks| chunks.checked_mul(1 << zk))
            .unwrap_or(usize::MAX);
        Self {
            num_constraints,
            max_constraint_degree,
            num_quotient_chunks,
            max_combo,
        }
    }
}

/// Per-instance shape data not in the protocol params.
#[derive(Copy, Clone, Debug)]
pub struct InstanceShape {
    pub log_trace_length: usize,
    /// Bit-length of the field FRI/WHIR operates over (typically the
    /// extension field).
    pub modulus_bits: usize,
    /// Collision resistance of the commitment hash, in bits.
    pub collision_resistance: usize,
    /// Number of committed codewords random-linear-combined into a single
    /// low-degree-test instance (trace segments, quotient chunks, …). `1`
    /// means no batching. Read by the batched-openings proximity term in
    /// [`crate::stark::proven_security_report`].
    pub num_batched_functions: usize,
}
