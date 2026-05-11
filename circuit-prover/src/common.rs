use alloc::boxed::Box;
use alloc::vec::Vec;
use core::any::Any;

use hashbrown::HashMap;
use p3_circuit::ops::{NonPrimitivePreprocessedMap, NpoTypeId, PrimitiveOpType};
use p3_circuit::{Circuit, CircuitError};
use p3_field::{Algebra, ExtensionField, Field, PrimeCharacteristicRing, PrimeField64};
use p3_uni_stark::{StarkGenericConfig, SymbolicExpression, SymbolicExpressionExt, Val};
use p3_util::log2_ceil_usize;

use crate::air::{AluAir, ConstAir, PublicAir};
use crate::config::StarkField;
use crate::field_params::ExtractBinomialW;
use crate::{ConstraintProfile, DynamicAirEntry, TablePacking};

/// Plugin trait for NPO-owned preprocessing over generic circuits.
///
/// Each implementation can update `PreprocessedColumns` (ext_reads, multiplicities, etc.)
/// and return base-field non-primitive preprocessed rows for its own `NpoTypeId`s.
pub trait NpoPreprocessor<F>: Send + Sync
where
    F: StarkField + PrimeField64,
{
    /// Run plugin-owned preprocessing over a generic circuit.
    ///
    /// `circuit` and `preprocessed` are type-erased; implementations downcast to the
    /// `PreprocessedColumns<ExtF>` shapes they support and return an empty map otherwise.
    fn preprocess(
        &self,
        circuit: &dyn Any,
        preprocessed: &mut dyn Any,
    ) -> Result<NonPrimitivePreprocessedMap<F>, CircuitError>;
}

/// Builds (AIR, degree) from preprocessed base data for a given NPO op_type.
/// Used by `get_airs_and_degrees_with_prep` so that AIR construction is plugin-driven
/// without requiring generic methods on the preprocessor trait (object safety).
pub trait NpoAirBuilder<SC, const D: usize>: Send + Sync
where
    SC: StarkGenericConfig,
    SymbolicExpressionExt<Val<SC>, SC::Challenge>: Algebra<SymbolicExpression<Val<SC>>>,
{
    /// Number of operations packed into a single AIR row for this NPO.
    ///
    /// Must match the `lanes` value returned by the corresponding [`TableProver`] implementation.
    /// Defaults to 1.
    fn lanes(&self) -> usize {
        1
    }

    /// Attempt to build an AIR and compute its degree from committed preprocessed data.
    ///
    /// The `lanes` argument is `self.lanes()` forwarded by the framework.
    fn try_build(
        &self,
        op_type: &NpoTypeId,
        prep_base: &[Val<SC>],
        min_height: usize,
        lanes: usize,
        constraint_profile: ConstraintProfile,
    ) -> Option<(CircuitTableAir<SC, D>, usize)>;
}

/// Enum wrapper to allow heterogeneous table AIRs in a single batch STARK aggregation.
///
/// This enables different AIR types to be collected into a single vector for
/// batch STARK proving/verification while maintaining type safety.
pub enum CircuitTableAir<SC, const D: usize>
where
    SC: StarkGenericConfig,
    SymbolicExpressionExt<Val<SC>, SC::Challenge>: Algebra<SymbolicExpression<Val<SC>>>,
{
    Const(ConstAir<Val<SC>, D>),
    Public(PublicAir<Val<SC>, D>),
    /// Unified ALU table for all arithmetic operations
    Alu(AluAir<Val<SC>, D>),
    Dynamic(DynamicAirEntry<SC>),
}

impl<SC, const D: usize> Clone for CircuitTableAir<SC, D>
where
    SC: StarkGenericConfig,
    SymbolicExpressionExt<Val<SC>, SC::Challenge>: Algebra<SymbolicExpression<Val<SC>>>,
{
    fn clone(&self) -> Self {
        match self {
            Self::Const(air) => Self::Const(air.clone()),
            Self::Public(air) => Self::Public(air.clone()),
            Self::Alu(air) => Self::Alu(air.clone()),
            Self::Dynamic(air) => Self::Dynamic(air.clone()),
        }
    }
}

/// Type alias for a vector of circuit table AIRs paired with their respective degrees (log of their trace height).
type CircuitAirsWithDegrees<SC, const D: usize> = Vec<(CircuitTableAir<SC, D>, usize)>;

/// Output of [`get_airs_and_degrees_with_prep`]: AIRs with degrees, primitive columns, and non-primitive columns.
type PrepOutput<SC, const D: usize> = (
    CircuitAirsWithDegrees<SC, D>,
    Vec<Vec<Val<SC>>>,
    NonPrimitivePreprocessedMap<Val<SC>>,
);

pub fn get_airs_and_degrees_with_prep<
    SC: StarkGenericConfig + 'static + Send + Sync,
    ExtF: Field + ExtensionField<Val<SC>> + ExtractBinomialW<Val<SC>>,
    const D: usize,
>(
    circuit: &Circuit<ExtF>,
    packing: &TablePacking,
    non_primitive_preprocessors: &[Box<dyn NpoPreprocessor<Val<SC>>>],
    non_primitive_air_builders: &[Box<dyn NpoAirBuilder<SC, D>>],
    constraint_profile: ConstraintProfile,
) -> Result<PrepOutput<SC, D>, CircuitError>
where
    SymbolicExpressionExt<Val<SC>, SC::Challenge>: Algebra<SymbolicExpression<Val<SC>>>,
    Val<SC>: StarkField,
{
    let mut preprocessed = circuit.generate_preprocessed_columns::<D>()?;

    // Check if Public/Alu tables are empty and lanes > 1.
    // Using lanes > 1 with empty tables causes issues in recursive verification
    // due to a bug in how multi-lane padding interacts with lookup constraints.
    // We automatically reduce lanes to 1 in these cases with a warning.
    // IMPORTANT: This must be synchronized with prove_all_tables in batch_stark_prover.rs
    let public_idx = PrimitiveOpType::Public as usize;
    let alu_idx = PrimitiveOpType::Alu as usize;

    let public_rows = preprocessed.primitive[public_idx].len();
    let public_trace_only_dummy = public_rows <= 1;
    let effective_public_lanes = if public_trace_only_dummy && packing.public_lanes() > 1 {
        tracing::warn!(
            "Public table has <=1 row but public_lanes={} > 1. Reducing to public_lanes=1 to avoid \
             recursive verification issues. Consider using public_lanes=1 when few public inputs \
             are expected.",
            packing.public_lanes()
        );
        1
    } else {
        packing.public_lanes()
    };

    let alu_empty = preprocessed.primitive[alu_idx].is_empty();
    let effective_alu_lanes = if alu_empty && packing.alu_lanes() > 1 {
        tracing::warn!(
            "ALU table is empty but alu_lanes={} > 1. Reducing to alu_lanes=1 to avoid \
             recursive verification issues. Consider using alu_lanes=1 when no additions \
             are expected.",
            packing.alu_lanes()
        );
        1
    } else {
        packing.alu_lanes()
    };

    let w_binomial = ExtF::extract_w();

    // First, get base field elements for the preprocessed primitive values.
    let mut base_prep: Vec<Vec<Val<SC>>> = preprocessed
        .primitive
        .iter()
        .map(|vals| {
            vals.iter()
                .map(|v| v.as_base().ok_or(CircuitError::InvalidPreprocessedValues))
                .collect::<Result<Vec<_>, CircuitError>>()
        })
        .collect::<Result<Vec<_>, CircuitError>>()?;

    // Let plugins handle non-primitive preprocessing (ext_reads, multiplicities, etc.).
    let mut non_primitive_base: NonPrimitivePreprocessedMap<Val<SC>> = HashMap::new();
    let circuit_any: &dyn Any = circuit;
    let preprocessed_any: &mut dyn Any = &mut preprocessed;
    for plugin in non_primitive_preprocessors {
        let plugin_prep = plugin.preprocess(circuit_any, preprocessed_any)?;
        non_primitive_base.extend(plugin_prep);
    }

    // Get min_height from packing configuration and pass it to AIRs
    let min_height = packing.min_trace_height();

    // Helper to compute degree that respects min_height
    let compute_degree = |num_rows: usize| -> usize {
        let natural_height = num_rows.next_power_of_two();
        let min_rows = min_height.next_power_of_two();
        log2_ceil_usize(natural_height.max(min_rows))
    };

    let mut table_preps: Vec<(CircuitTableAir<SC, D>, usize)> =
        Vec::with_capacity(base_prep.len() + non_primitive_base.len());

    #[allow(clippy::needless_range_loop)]
    for idx in 0..base_prep.len() {
        let table = PrimitiveOpType::from(idx);
        match table {
            PrimitiveOpType::Alu => {
                // ALU preprocessed per op from circuit.rs: 12 values
                // [sel_add_vs_mul, sel_bool, sel_muladd, sel_horner, a_idx, b_idx, c_idx, out_idx,
                //  mult_a_eff, b_is_creator, mult_c_eff, out_is_creator]
                //
                // mult_a_eff / mult_c_eff: -1 (reader or later unconstrained), or +N (first
                // unconstrained creator). We convert to 12 values for AluAir (same order, mult_c_eff last).
                let lane_12 = 12_usize;
                let neg_one = <Val<SC>>::ZERO - <Val<SC>>::ONE;

                let mut chunks = base_prep[idx].chunks_exact(lane_12);
                let mut prep_13col: Vec<Val<SC>> = Vec::with_capacity(
                    chunks.len() * lane_12 + if alu_empty { 0 } else { lane_12 },
                );
                for chunk in &mut chunks {
                    let sel1 = chunk[0];
                    let sel2 = chunk[1];
                    let sel3 = chunk[2];
                    let sel4 = chunk[3];
                    let a_idx = chunk[4];
                    let b_idx = chunk[5];
                    let c_idx = chunk[6];
                    let out_idx = chunk[7];
                    let a_state = chunk[8].as_canonical_u64();
                    let b_is_creator = chunk[9].as_canonical_u64() != 0;
                    let c_state = chunk[10].as_canonical_u64();
                    let out_is_creator = chunk[11].as_canonical_u64() != 0;

                    // mult_a = -1 for all active rows; active = -mult_a = 1 always.
                    // Effective a-lookup mult = mult_a * a_reader_col (in get_alu_index_lookups).
                    // Effective c-lookup mult = mult_a * c_reader_col (in get_alu_index_lookups).
                    //
                    // a_state / c_state encoding:
                    //   0 → skip: col = 0, eff = 0
                    //   1 → reader: col = 1, eff = (-1)*1 = -1
                    //   2 → private creator: col = -(n_reads), eff = (-1)*(-(n_reads)) = +n_reads
                    let mult_a = neg_one;
                    let a_reader_col = match a_state {
                        0 => <Val<SC>>::ZERO,
                        1 => <Val<SC>>::ONE,
                        2 => {
                            let a_wid = a_idx.as_canonical_u64() as usize / D;
                            let n_reads = preprocessed.ext_reads.get(a_wid).copied().unwrap_or(0);
                            <Val<SC>>::ZERO - <Val<SC>>::from_u32(n_reads)
                        }
                        _ => <Val<SC>>::ZERO,
                    };
                    let c_reader_col = match c_state {
                        0 => <Val<SC>>::ZERO,
                        1 => <Val<SC>>::ONE,
                        2 => {
                            let c_wid = c_idx.as_canonical_u64() as usize / D;
                            let n_reads = preprocessed.ext_reads.get(c_wid).copied().unwrap_or(0);
                            <Val<SC>>::ZERO - <Val<SC>>::from_u32(n_reads)
                        }
                        _ => <Val<SC>>::ZERO,
                    };

                    // b: creator if b_is_creator, reader otherwise.
                    let mult_b = if b_is_creator {
                        let b_wid = b_idx.as_canonical_u64() as usize / D;
                        let n_reads = preprocessed.ext_reads.get(b_wid).copied().unwrap_or(0);
                        <Val<SC>>::from_u32(n_reads)
                    } else {
                        neg_one
                    };

                    // out: creator if out_is_creator, reader otherwise.
                    let mult_out = if out_is_creator {
                        let out_wid = out_idx.as_canonical_u64() as usize / D;
                        let n_reads = preprocessed.ext_reads.get(out_wid).copied().unwrap_or(0);
                        <Val<SC>>::from_u32(n_reads)
                    } else {
                        neg_one
                    };

                    prep_13col.extend([
                        mult_a,
                        sel1,
                        sel2,
                        sel3,
                        sel4,
                        a_idx,
                        b_idx,
                        c_idx,
                        out_idx,
                        mult_b,
                        mult_out,
                        a_reader_col,
                        c_reader_col,
                    ]);
                }
                debug_assert!(chunks.remainder().is_empty());

                // If ALU was empty, add a dummy row (all zeros = padding, no logup contribution).
                if alu_empty {
                    prep_13col.extend([<Val<SC>>::ZERO; 13]);
                }

                let num_ops = prep_13col.len() / 13;
                let horner_k = packing.horner_packed_steps();
                // Store the converted 13-col format before building the AIR.
                base_prep[idx] = prep_13col;
                let alu_air = if D == 1 {
                    AluAir::new_with_preprocessed(
                        num_ops,
                        effective_alu_lanes,
                        base_prep[idx].clone(),
                        horner_k,
                    )
                    .with_min_height(min_height)
                } else if D == 5 && ExtF::alu_is_quintic_trinomial() {
                    AluAir::new_quintic_trinomial_with_preprocessed(
                        num_ops,
                        effective_alu_lanes,
                        base_prep[idx].clone(),
                        horner_k,
                    )
                    .with_min_height(min_height)
                } else {
                    let w = w_binomial.expect(
                        "ALU preprocessed path needs binomial W when D>1 and the element field is \
                         not the quintic-trinomial ALU variant. Use D=1 for base-field circuits \
                         (ExtF = Val<SC>); for extension circuits use D = ExtF::DIMENSION and a \
                         binomial or supported quintic ExtF.",
                    );
                    AluAir::new_binomial_with_preprocessed(
                        num_ops,
                        effective_alu_lanes,
                        w,
                        base_prep[idx].clone(),
                        horner_k,
                    )
                    .with_min_height(min_height)
                };
                let num_entries = alu_air.scheduled_entry_count();
                let num_rows = num_entries.div_ceil(effective_alu_lanes);
                table_preps.push((CircuitTableAir::Alu(alu_air), compute_degree(num_rows)));
            }
            PrimitiveOpType::Public => {
                // Public preprocessed per op from circuit.rs: 1 value (D-scaled out_idx).
                // Convert to [ext_mult, out_idx] pairs using ext_reads.
                let mut prep_2col: Vec<Val<SC>> = Vec::with_capacity(base_prep[idx].len() * 2);
                for &out_idx in &base_prep[idx] {
                    let out_wid =
                        (<Val<SC> as PrimeField64>::as_canonical_u64(&out_idx) as usize) / D;
                    let n_reads = preprocessed.ext_reads.get(out_wid).copied().unwrap_or(0);
                    prep_2col.push(<Val<SC>>::from_u32(n_reads));
                    prep_2col.push(out_idx);
                }

                let num_ops = prep_2col.len() / 2;
                // Store the converted 2-col format before building the AIR.
                base_prep[idx] = prep_2col;
                let public_air = PublicAir::new_with_preprocessed(
                    num_ops,
                    effective_public_lanes,
                    base_prep[idx].clone(),
                )
                .with_min_height(min_height);
                let num_rows = num_ops.div_ceil(effective_public_lanes);
                table_preps.push((
                    CircuitTableAir::Public(public_air),
                    compute_degree(num_rows),
                ));
            }
            PrimitiveOpType::Const => {
                // Const preprocessed per op from circuit.rs: 1 value (D-scaled out_idx).
                // Convert to [ext_mult, out_idx] pairs using ext_reads.
                let mut prep_2col: Vec<Val<SC>> = Vec::with_capacity(base_prep[idx].len() * 2);
                for &out_idx in &base_prep[idx] {
                    let out_wid = out_idx.as_canonical_u64() as usize / D;
                    let n_reads = preprocessed.ext_reads.get(out_wid).copied().unwrap_or(0);
                    prep_2col.push(<Val<SC>>::from_u32(n_reads));
                    prep_2col.push(out_idx);
                }

                let height = prep_2col.len() / 2;
                // Store the converted 2-col format before building the AIR.
                base_prep[idx] = prep_2col;
                let const_air = ConstAir::new_with_preprocessed(height, base_prep[idx].clone())
                    .with_min_height(min_height);
                table_preps.push((CircuitTableAir::Const(const_air), compute_degree(height)));
            }
        }
    }

    // Iterate air builders first (fixed registration order) so that the
    // resulting AIR ordering matches the prover's non_primitive_provers order.
    for builder in non_primitive_air_builders {
        for (op_type, prep_base) in non_primitive_base.iter() {
            // TablePacking overrides the builder's own default lane count.
            let lanes = packing
                .npo_lanes(op_type)
                .unwrap_or_else(|| builder.lanes());
            if let Some((air, degree)) =
                builder.try_build(op_type, prep_base, min_height, lanes, constraint_profile)
            {
                table_preps.push((air, degree));
                break;
            }
        }
    }

    Ok((table_preps, base_prep, non_primitive_base))
}
