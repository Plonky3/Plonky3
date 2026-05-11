//! Recompose table prover: builds `RecomposeAir` instances for the batch STARK prover.

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;

use hashbrown::HashMap;
use p3_baby_bear::BabyBear;
use p3_batch_stark::{StarkGenericConfig, Val};
use p3_circuit::ops::recompose::RecomposeTrace;
use p3_circuit::ops::{NonPrimitivePreprocessedMap, NpoTypeId};
use p3_circuit::tables::Traces;
use p3_circuit::{CircuitError, PreprocessedColumns};
use p3_field::extension::{BinomialExtensionField, QuinticTrinomialExtensionField};
use p3_field::{Algebra, ExtensionField, Field, PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::Goldilocks;
use p3_koala_bear::KoalaBear;
use p3_uni_stark::{SymbolicExpression, SymbolicExpressionExt};
use p3_util::log2_ceil_usize;

use super::dynamic_air::{
    BatchAir, BatchTableInstance, DynamicAirEntry, TableProver, transmute_traces,
};
use super::{NonPrimitiveTableEntry, TablePacking};
use crate::air::RecomposeAir;
use crate::common::{CircuitTableAir, NpoAirBuilder, NpoPreprocessor};
use crate::config::StarkField;
use crate::{ConstraintProfile, impl_table_prover_batch_instances_from_base};

impl<SC, const D: usize> BatchAir<SC> for RecomposeAir<Val<SC>, D>
where
    SC: StarkGenericConfig + Send + Sync,
    Val<SC>: StarkField,
    SymbolicExpressionExt<Val<SC>, SC::Challenge>:
        Algebra<SymbolicExpression<Val<SC>>> + Algebra<SC::Challenge>,
{
}

/// Table prover for the recompose (BF→EF packing) NPO.
///
/// `lanes` controls how many operations are packed into a single AIR row.
/// Increasing this value reduces the trace height proportionally, at the cost of
/// a wider trace. Must be kept in sync with the corresponding [`RecomposeAirBuilder`].
pub struct RecomposeProver<const D: usize> {
    lanes: usize,
    /// When true, extra WitnessChecks receives are registered so a D=1 Poseidon2 inside a D>1
    /// circuit can read BF coefficients (per-coefficient receives per lane).
    coeff_lookups: bool,
}

impl<const D: usize> RecomposeProver<D> {
    /// Create a prover that packs `lanes` recompose operations per row.
    pub fn new(lanes: usize, coeff_lookups: bool) -> Self {
        Self {
            lanes: lanes.max(1),
            coeff_lookups,
        }
    }

    fn batch_instance_base<SC>(
        &self,
        _config: &SC,
        packing: &TablePacking,
        traces: &Traces<Val<SC>>,
    ) -> Option<BatchTableInstance<SC>>
    where
        SC: StarkGenericConfig + 'static + Send + Sync,
        Val<SC>: StarkField,
        SymbolicExpressionExt<Val<SC>, SC::Challenge>:
            Algebra<SymbolicExpression<Val<SC>>> + Algebra<SC::Challenge>,
    {
        let op_type = if self.coeff_lookups {
            NpoTypeId::recompose_with_coeff_lookups()
        } else {
            NpoTypeId::recompose()
        };
        let trace = traces.non_primitive_traces.get(&op_type)?;
        if trace.rows() == 0 {
            return None;
        }

        let t = trace.as_any().downcast_ref::<RecomposeTrace<Val<SC>>>()?;

        let num_ops = t.total_rows();
        // Prefer the per-op override from TablePacking; fall back to the prover's own default.
        let lanes = packing
            .npo_lanes(&op_type)
            .or_else(|| {
                if self.coeff_lookups {
                    packing.npo_lanes(&NpoTypeId::recompose())
                } else {
                    None
                }
            })
            .unwrap_or(self.lanes);
        let min_height = packing.min_trace_height();

        let coeff_lookups = self.coeff_lookups;
        let prep_lane_width =
            RecomposeAir::<Val<SC>, D>::preprocessed_lane_width_for(coeff_lookups);
        let mut preprocessed = Val::<SC>::zero_vec(num_ops * prep_lane_width);
        for (i, row) in t.operations.iter().enumerate() {
            let base = i * prep_lane_width;
            preprocessed[base] = row.output_wid.base_field_index::<Val<SC>, D>();
            if coeff_lookups {
                for (j, &coeff_wid) in row.input_wids.iter().enumerate().take(D) {
                    preprocessed[base + 2 + j * 2] = coeff_wid.base_field_index::<Val<SC>, D>();
                }
            }
        }

        let air = RecomposeAir::<Val<SC>, D>::new_with_preprocessed(
            lanes,
            preprocessed,
            min_height,
            coeff_lookups,
        );
        let matrix = RecomposeAir::<Val<SC>, D>::trace_to_matrix(&t.operations, lanes);

        Some(BatchTableInstance {
            op_type,
            air: DynamicAirEntry::new(Box::new(air)),
            trace: matrix,
            public_values: Vec::new(),
            rows: num_ops,
            lanes,
        })
    }
}

impl<SC, const D: usize> TableProver<SC> for RecomposeProver<D>
where
    SC: StarkGenericConfig + 'static + Send + Sync,
    Val<SC>: StarkField,
    SymbolicExpressionExt<Val<SC>, SC::Challenge>:
        Algebra<SymbolicExpression<Val<SC>>> + Algebra<SC::Challenge>,
{
    fn op_type(&self) -> NpoTypeId {
        if self.coeff_lookups {
            NpoTypeId::recompose_with_coeff_lookups()
        } else {
            NpoTypeId::recompose()
        }
    }

    fn lanes(&self) -> usize {
        self.lanes
    }

    impl_table_prover_batch_instances_from_base!(batch_instance_base);

    fn batch_air_from_table_entry(
        &self,
        _config: &SC,
        _degree: usize,
        _circuit_extension_degree: u32,
        table_entry: &NonPrimitiveTableEntry<SC>,
    ) -> Result<DynamicAirEntry<SC>, String> {
        let air = RecomposeAir::<Val<SC>, D>::new_with_preprocessed(
            table_entry.lanes,
            Vec::new(),
            1,
            self.coeff_lookups,
        );
        Ok(DynamicAirEntry::new(Box::new(air)))
    }

    fn air_with_committed_preprocessed(
        &self,
        committed_prep: Vec<Val<SC>>,
        min_height: usize,
        lanes: usize,
        _circuit_extension_degree: u32,
    ) -> Option<DynamicAirEntry<SC>> {
        let air = RecomposeAir::<Val<SC>, D>::new_with_preprocessed(
            lanes,
            committed_prep,
            min_height,
            self.coeff_lookups,
        );
        Some(DynamicAirEntry::new(Box::new(air)))
    }
}

// ============================================================================
// Preprocessor
// ============================================================================

/// NpoPreprocessor for the recompose table(s).
///
/// Converts EF preprocessed data to BF and sets `out_mult` from `ext_reads`.
/// When `split_coeff_tables` is true, emits separate base rows for `recompose` and `recompose/coeff`.
#[derive(Default, Clone)]
pub struct RecomposePreprocessor {
    pub split_coeff_tables: bool,
}

impl RecomposePreprocessor {
    pub const fn new(split_coeff_tables: bool) -> Self {
        Self { split_coeff_tables }
    }
}

impl NpoPreprocessor<KoalaBear> for RecomposePreprocessor {
    fn preprocess(
        &self,
        _circuit: &dyn core::any::Any,
        preprocessed: &mut dyn core::any::Any,
    ) -> Result<NonPrimitivePreprocessedMap<KoalaBear>, CircuitError> {
        type F = KoalaBear;
        let split = self.split_coeff_tables;
        if let Some(prep) =
            preprocessed.downcast_mut::<PreprocessedColumns<BinomialExtensionField<F, 4>, 4>>()
        {
            return recompose_preprocess_impl::<F, _, 4>(prep, split);
        }
        if let Some(prep) =
            preprocessed.downcast_mut::<PreprocessedColumns<QuinticTrinomialExtensionField<F>, 5>>()
        {
            return recompose_preprocess_impl::<F, _, 5>(prep, split);
        }
        if let Some(prep) = preprocessed.downcast_mut::<PreprocessedColumns<F, 1>>() {
            return recompose_preprocess_impl::<F, _, 1>(prep, split);
        }
        Ok(HashMap::new())
    }
}

impl NpoPreprocessor<BabyBear> for RecomposePreprocessor {
    fn preprocess(
        &self,
        _circuit: &dyn core::any::Any,
        preprocessed: &mut dyn core::any::Any,
    ) -> Result<NonPrimitivePreprocessedMap<BabyBear>, CircuitError> {
        type F = BabyBear;
        let split = self.split_coeff_tables;
        if let Some(prep) =
            preprocessed.downcast_mut::<PreprocessedColumns<BinomialExtensionField<F, 4>, 4>>()
        {
            return recompose_preprocess_impl::<F, _, 4>(prep, split);
        }
        if let Some(prep) = preprocessed.downcast_mut::<PreprocessedColumns<F, 1>>() {
            return recompose_preprocess_impl::<F, _, 1>(prep, split);
        }
        Ok(HashMap::new())
    }
}

impl NpoPreprocessor<Goldilocks> for RecomposePreprocessor {
    fn preprocess(
        &self,
        _circuit: &dyn core::any::Any,
        preprocessed: &mut dyn core::any::Any,
    ) -> Result<NonPrimitivePreprocessedMap<Goldilocks>, CircuitError> {
        type F = Goldilocks;
        let split = self.split_coeff_tables;
        if let Some(prep) =
            preprocessed.downcast_mut::<PreprocessedColumns<BinomialExtensionField<F, 2>, 2>>()
        {
            return recompose_preprocess_impl::<F, _, 2>(prep, split);
        }
        if let Some(prep) = preprocessed.downcast_mut::<PreprocessedColumns<F, 1>>() {
            return recompose_preprocess_impl::<F, _, 1>(prep, split);
        }
        Ok(HashMap::new())
    }
}

fn recompose_preprocess_impl<F, EF, const D: usize>(
    prep: &PreprocessedColumns<EF, D>,
    split_coeff_tables: bool,
) -> Result<NonPrimitivePreprocessedMap<F>, CircuitError>
where
    F: StarkField + PrimeField64,
    EF: Field + ExtensionField<F> + 'static,
{
    let mut result = HashMap::new();
    result.extend(recompose_preprocess_for_op::<F, EF, D>(
        prep,
        &NpoTypeId::recompose(),
        false,
    )?);
    if split_coeff_tables {
        result.extend(recompose_preprocess_for_op::<F, EF, D>(
            prep,
            &NpoTypeId::recompose_with_coeff_lookups(),
            true,
        )?);
    }
    Ok(result)
}

/// Extract preprocessed rows for one recompose `NpoTypeId` and set output / coeff multiplicities.
fn recompose_preprocess_for_op<F, EF, const D: usize>(
    prep: &PreprocessedColumns<EF, D>,
    op_type: &NpoTypeId,
    coeff_lookups: bool,
) -> Result<NonPrimitivePreprocessedMap<F>, CircuitError>
where
    F: StarkField + PrimeField64,
    EF: Field + ExtensionField<F> + 'static,
{
    let ef_data = match prep.non_primitive.get(op_type) {
        Some(d) if !d.is_empty() => d,
        _ => return Ok(HashMap::new()),
    };

    let prep_width = if coeff_lookups { 2 + 2 * D } else { 2 };

    let mut prep_base: Vec<F> = ef_data
        .iter()
        .map(|v| v.as_base().ok_or(CircuitError::InvalidPreprocessedValues))
        .collect::<Result<Vec<_>, CircuitError>>()?;

    if !prep_base.len().is_multiple_of(prep_width) {
        return Err(CircuitError::InvalidPreprocessedValues);
    }

    let neg_one = F::ZERO - F::ONE;
    let num_rows = prep_base.len() / prep_width;

    for row_idx in 0..num_rows {
        let row_start = row_idx * prep_width;

        let output_idx_val = prep_base[row_start];
        let out_wid = F::as_canonical_u64(&output_idx_val) as usize / D;

        let is_dup = prep
            .dup_npo_outputs
            .get(op_type)
            .and_then(|d| d.get(out_wid).copied())
            .unwrap_or(false);

        if is_dup {
            prep_base[row_start + 1] = neg_one;
        } else {
            let n_reads = prep.ext_reads.get(out_wid).copied().unwrap_or(0);
            prep_base[row_start + 1] = F::from_u32(n_reads);
        }

        if coeff_lookups {
            for i in 0..D {
                let coeff_idx_val = prep_base[row_start + 2 + i * 2];
                let coeff_wid = F::as_canonical_u64(&coeff_idx_val) as usize / D;
                let n_coeff_reads = if prep.hint_output_wids.contains(&(coeff_wid as u32)) {
                    prep.ext_reads.get(coeff_wid).copied().unwrap_or(0)
                } else {
                    0
                };
                prep_base[row_start + 2 + i * 2 + 1] = F::from_u32(n_coeff_reads);
            }
        }
    }

    let mut result = HashMap::new();
    result.insert(op_type.clone(), prep_base);
    Ok(result)
}

// ============================================================================
// AIR Builder
// ============================================================================

/// NpoAirBuilder for the recompose table.
///
/// `lanes` must match the value used in the paired [`RecomposeProver`].
#[derive(Clone)]
pub struct RecomposeAirBuilder<const D: usize> {
    lanes: usize,
    coeff_lookups: bool,
}

impl<const D: usize> RecomposeAirBuilder<D> {
    /// Create a builder that expects `lanes` operations packed per AIR row.
    pub fn new(lanes: usize, coeff_lookups: bool) -> Self {
        Self {
            lanes: lanes.max(1),
            coeff_lookups,
        }
    }
}

impl<SC, const D: usize> NpoAirBuilder<SC, D> for RecomposeAirBuilder<D>
where
    SC: StarkGenericConfig + 'static + Send + Sync,
    Val<SC>: StarkField,
    SymbolicExpressionExt<Val<SC>, SC::Challenge>:
        Algebra<SymbolicExpression<Val<SC>>> + Algebra<SC::Challenge>,
{
    fn lanes(&self) -> usize {
        self.lanes
    }

    fn try_build(
        &self,
        op_type: &NpoTypeId,
        prep_base: &[Val<SC>],
        min_height: usize,
        lanes: usize,
        _constraint_profile: ConstraintProfile,
    ) -> Option<(CircuitTableAir<SC, D>, usize)> {
        let matches = if !self.coeff_lookups {
            op_type.as_str() == "recompose"
        } else {
            op_type.as_str() == "recompose/coeff"
        };
        if !matches {
            return None;
        }

        let prep_lane_width =
            RecomposeAir::<Val<SC>, D>::preprocessed_lane_width_for(self.coeff_lookups);
        let num_ops = prep_base.len() / prep_lane_width;
        let num_rows = num_ops.div_ceil(lanes).max(1);

        let air = RecomposeAir::<Val<SC>, D>::new_with_preprocessed(
            lanes,
            prep_base.to_vec(),
            min_height,
            self.coeff_lookups,
        );

        let padded_rows = num_rows
            .next_power_of_two()
            .max(min_height.next_power_of_two());
        let degree = log2_ceil_usize(padded_rows);

        Some((
            CircuitTableAir::Dynamic(DynamicAirEntry::new(Box::new(air))),
            degree,
        ))
    }
}
