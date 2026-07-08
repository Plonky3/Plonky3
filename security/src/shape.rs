//! Protocol-instance shape: data orthogonal to the runtime config.

use p3_air::Air;
use p3_air::symbolic::{AirLayout, SymbolicAirBuilder, get_all_symbolic_constraints};
use p3_field::{ExtensionField, Field};

/// AIR-derived shape used in DEEP-ALI and composition-error bounds.
#[derive(Copy, Clone, Debug)]
pub struct StarkAirParams {
    pub num_constraints: usize,
    pub max_constraint_degree: usize,
    /// DEEP-ALI `max_combo`: maximum number of out-of-domain points
    /// referenced per column (typically 2 for `local`/`next`).
    pub max_combo: usize,
}

impl StarkAirParams {
    /// Derive `num_constraints` and `max_constraint_degree` by symbolically
    /// evaluating the AIR's constraints. The caller supplies `max_combo`
    /// (typically `2` for an AIR using `local`/`next` rotations, `1` if
    /// there is no transition constraint).
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
    pub fn from_air<F, EF, A>(air: &A, layout: AirLayout, max_combo: usize) -> Self
    where
        F: Field,
        EF: ExtensionField<F>,
        A: Air<SymbolicAirBuilder<F, EF>>,
    {
        let (base, ext) = get_all_symbolic_constraints::<F, EF, A>(air, layout);
        let num_constraints = base.len() + ext.len();
        let max_constraint_degree = base
            .iter()
            .map(|c| c.degree_multiple())
            .chain(ext.iter().map(|c| c.degree_multiple()))
            .max()
            .unwrap_or(1)
            .max(1);
        Self {
            num_constraints,
            max_constraint_degree,
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
}
