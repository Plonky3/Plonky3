//! Batch STARK prover and verifier that unifies all circuit tables
//! into a single batched STARK proof using `p3-batch-stark`.

use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;
use alloc::{format, vec};

#[cfg(debug_assertions)]
use p3_air::DebugConstraintBuilder;
use p3_air::{Air, BaseAir};
use p3_batch_stark::common::{GlobalPreprocessed, PreprocessedInstanceMeta};
use p3_batch_stark::{BatchProof, CommonData, ProverData, StarkGenericConfig, StarkInstance, Val};
use p3_circuit::ops::{NonPrimitivePreprocessedMap, NpoTypeId, Poseidon2Config, PrimitiveOpType};
use p3_circuit::tables::Traces;
use p3_commit::Pcs;
use p3_field::extension::{BinomialExtensionField, BinomiallyExtendable};
use p3_field::{Algebra, BasedVectorSpace, Field, PrimeCharacteristicRing, PrimeField};
use p3_lookup::LookupAir;
use p3_lookup::folder::{ProverConstraintFolderWithLookups, VerifierConstraintFolderWithLookups};
use p3_lookup::lookup_traits::Lookup;
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_uni_stark::{SymbolicAirBuilder, SymbolicExpression, SymbolicExpressionExt};
use p3_util::log2_strict_usize;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::instrument;

use crate::air::{AluAir, ConstAir, PublicAir};
use crate::batch_stark_prover::dynamic_air::transmute_traces;
use crate::batch_stark_prover::packing::{AirTableShape, TraceTablesLayout};
use crate::common::{CircuitTableAir, NpoAirBuilder, NpoPreprocessor};
use crate::config::StarkField;
use crate::constraint_profile::ConstraintProfile;
use crate::field_params::ExtractBinomialW;

mod dynamic_air;
mod packing;
mod poseidon2;
mod recompose;

pub use dynamic_air::{
    BatchAir, BatchTableInstance, CloneableBatchAir, DynamicAirEntry, TableProver,
};
pub use packing::TablePacking;
pub use poseidon2::{
    Poseidon2AirBuilder, Poseidon2AirWrapperInner, Poseidon2Preprocessor, Poseidon2Prover,
    Poseidon2ProverD2, poseidon2_preprocessor, poseidon2_verifier_air_from_config,
};
pub use recompose::{RecomposeAirBuilder, RecomposePreprocessor, RecomposeProver};

/// Prime modulus of the BabyBear field (`2^31 - 2^27 + 1`).
pub const BABY_BEAR_MODULUS: u64 = 0x7800_0001;
/// Prime modulus of the KoalaBear field (`2^31 - 2^24 + 1`).
pub const KOALA_BEAR_MODULUS: u64 = 0x7f00_0001;

/// Opaque variant tag for a non-primitive AIR in a batch proof.
///
/// Each [`NonPrimitiveTableEntry`] has one tag. The **meaning** of the tag is
/// defined by that entry's `op_type`: the corresponding [`TableProver`] interprets
/// it when building the AIR in [`TableProver::batch_air_from_table_entry`].
#[derive(Clone, Copy, Default, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum AirVariant {
    /// Baseline AIR for this op type (default behaviour).
    #[default]
    Baseline = 0,
    /// Recursion-optimized variant.
    Optimized = 1,
}

/// Metadata describing a non-primitive table inside a batch proof.
///
/// Every non-primitive dynamic plugin produces exactly one `NonPrimitiveTableEntry`
/// per batch instance. The entry is stored inside a `BatchStarkProof` and later provided
/// back to the plugin during verification through
/// [`TableProver::batch_air_from_table_entry`].
const fn default_npo_lanes() -> usize {
    1
}

#[derive(Serialize, Deserialize)]
#[serde(bound = "")]
pub struct NonPrimitiveTableEntry<SC>
where
    SC: StarkGenericConfig,
{
    /// Operation type (it should match `TableProver::op_type`).
    pub op_type: NpoTypeId,
    /// Number of logical operations (before lane packing) produced for this table.
    pub rows: usize,
    /// Number of operations packed per AIR row (lane count). Defaults to 1.
    #[serde(default = "default_npo_lanes")]
    pub lanes: usize,
    /// Public values exposed by this table (if any).
    pub public_values: Vec<Val<SC>>,
    /// AIR variant used for this non-primitive table.
    #[serde(default)]
    pub air_variant: AirVariant,
}

/// Combined data for circuit proving, including STARK prover data and preprocessed columns.
///
/// This struct bundles the upstream [`ProverData`] with circuit-specific preprocessed data,
/// providing a cleaner API for `prove_all_tables`.
///
/// Preprocessed columns are stored as flat base-field vectors rather than a
/// [`PreprocessedColumns<F, D>`](p3_circuit::PreprocessedColumns) because `D` is only
/// determined at proving time (via `EF::DIMENSION`) while this struct is constructed
/// and stored beforehand. The `ext_reads` and `dup_npo_outputs` fields from
/// `PreprocessedColumns` are fully consumed during AIR construction in
/// [`get_airs_and_degrees_with_prep`](crate::common::get_airs_and_degrees_with_prep)
/// and are not needed here.
pub struct CircuitProverData<SC: StarkGenericConfig> {
    /// STARK prover data from p3_batch_stark.
    pub prover_data: ProverData<SC>,
    /// Preprocessed columns for primitive operations (Const, Public, ALU).
    pub primitive_columns: Vec<Vec<Val<SC>>>,
    /// Preprocessed columns for non-primitive operations.
    pub non_primitive_columns: NonPrimitivePreprocessedMap<Val<SC>>,
}

impl<SC: StarkGenericConfig> CircuitProverData<SC> {
    /// Create new circuit prover data from components.
    pub const fn new(
        prover_data: ProverData<SC>,
        primitive_columns: Vec<Vec<Val<SC>>>,
        non_primitive_columns: NonPrimitivePreprocessedMap<Val<SC>>,
    ) -> Self {
        Self {
            prover_data,
            primitive_columns,
            non_primitive_columns,
        }
    }

    /// Get a reference to the common data.
    pub const fn common_data(&self) -> &CommonData<SC> {
        &self.prover_data.common
    }
}

/// Convenience macro for deriving all degree-specific helpers from a single base
/// implementation.
///
/// Plugins usually implement a single `batch_instance_base` method that operates on
/// base-field traces. This macro reuses that method to provide the `batch_instance_d*`
/// variants by casting higher-degree traces back to the base field.
///
/// Users can invoke it inside their `TableProver` impl:
///
/// ```ignore
/// impl<SC> TableProver<SC> for MyPlugin {
///     fn op_type(&self) -> NpoTypeId {
///         NpoTypeId::Poseidon2Perm(Poseidon2Config::BabyBearD4Width16)
///     }
///
///     impl_table_prover_batch_instances_from_base!(batch_instance_base);
///
///     fn batch_air_from_table_entry(
///         &self,
///         config: &SC,
///         degree: usize,
///         circuit_extension_degree: u32,
///         table_entry: &NonPrimitiveTableEntry<SC>,
///     ) -> Result<DynamicAirEntry<SC>, String> {
///         Ok(DynamicAirEntry::new(Box::new(MyPluginAir::<Val<SC>>::new(config))))
///     }
/// }
/// ```
#[macro_export]
macro_rules! impl_table_prover_batch_instances_from_base {
    ($base:ident) => {
        fn batch_instance_d1(
            &self,
            config: &SC,
            packing: &TablePacking,
            traces: &p3_circuit::tables::Traces<p3_batch_stark::Val<SC>>,
        ) -> Option<BatchTableInstance<SC>> {
            self.$base::<SC>(config, packing, traces)
        }

        fn batch_instance_d2(
            &self,
            config: &SC,
            packing: &TablePacking,
            traces: &p3_circuit::tables::Traces<
                p3_field::extension::BinomialExtensionField<p3_batch_stark::Val<SC>, 2>,
            >,
        ) -> Option<BatchTableInstance<SC>> {
            let t: &p3_circuit::tables::Traces<p3_batch_stark::Val<SC>> =
                unsafe { transmute_traces(traces) };
            self.$base::<SC>(config, packing, t)
        }

        fn batch_instance_d4(
            &self,
            config: &SC,
            packing: &TablePacking,
            traces: &p3_circuit::tables::Traces<
                p3_field::extension::BinomialExtensionField<p3_batch_stark::Val<SC>, 4>,
            >,
        ) -> Option<BatchTableInstance<SC>> {
            let t: &p3_circuit::tables::Traces<p3_batch_stark::Val<SC>> =
                unsafe { transmute_traces(traces) };
            self.$base::<SC>(config, packing, t)
        }

        fn batch_instance_d6(
            &self,
            config: &SC,
            packing: &TablePacking,
            traces: &p3_circuit::tables::Traces<
                p3_field::extension::BinomialExtensionField<p3_batch_stark::Val<SC>, 6>,
            >,
        ) -> Option<BatchTableInstance<SC>> {
            let t: &p3_circuit::tables::Traces<p3_batch_stark::Val<SC>> =
                unsafe { transmute_traces(traces) };
            self.$base::<SC>(config, packing, t)
        }

        fn batch_instance_d8(
            &self,
            config: &SC,
            packing: &TablePacking,
            traces: &p3_circuit::tables::Traces<
                p3_field::extension::BinomialExtensionField<p3_batch_stark::Val<SC>, 8>,
            >,
        ) -> Option<BatchTableInstance<SC>> {
            let t: &p3_circuit::tables::Traces<p3_batch_stark::Val<SC>> =
                unsafe { transmute_traces(traces) };
            self.$base::<SC>(config, packing, t)
        }

        fn batch_instance_d5(
            &self,
            config: &SC,
            packing: &TablePacking,
            traces: &p3_circuit::tables::Traces<
                p3_field::extension::QuinticTrinomialExtensionField<p3_batch_stark::Val<SC>>,
            >,
        ) -> Option<BatchTableInstance<SC>> {
            let t: &p3_circuit::tables::Traces<p3_batch_stark::Val<SC>> =
                unsafe { transmute_traces(traces) };
            self.$base::<SC>(config, packing, t)
        }
    };
}

/// Type alias for the primitive operation table selector.
///
/// Used as an index into [`RowCounts`] and related per-table arrays.
pub type PrimitiveTable = PrimitiveOpType;

/// Number of primitive circuit tables included in the unified batch STARK proof.
pub const NUM_PRIMITIVE_TABLES: usize = PrimitiveTable::Alu as usize + 1;

/// Row counts wrapper with type-safe indexing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RowCounts([usize; NUM_PRIMITIVE_TABLES]);

impl RowCounts {
    /// Creates a new RowCounts with the given row counts for each table.
    pub const fn new(rows: [usize; NUM_PRIMITIVE_TABLES]) -> Self {
        // Validate that all row counts are non-zero
        let mut i = 0;
        while i < rows.len() {
            assert!(rows[i] > 0);
            i += 1;
        }
        Self(rows)
    }
}

impl core::ops::Index<PrimitiveTable> for RowCounts {
    type Output = usize;
    fn index(&self, table: PrimitiveTable) -> &Self::Output {
        &self.0[table as usize]
    }
}

/// Serializable mirror of [`PreprocessedInstanceMeta`].
///
/// Defined locally because the upstream type does not derive `Serialize`/`Deserialize`.
#[derive(Serialize, Deserialize)]
struct SerializedPreprocessedInstanceMeta {
    matrix_index: usize,
    width: usize,
    degree_bits: usize,
}

/// Serializable projection of [`CommonData::preprocessed`] used to bind the proof
/// to its prover-side common data across (de)serialization.
///
/// `lookups` are intentionally omitted: the verifier always rebuilds them from the
/// AIRs reconstructed from proof metadata, so they are not part of the binding.
#[derive(Serialize, Deserialize)]
#[serde(bound = "")]
struct SerializedStarkCommon<SC: StarkGenericConfig> {
    commitment: <SC::Pcs as Pcs<SC::Challenge, SC::Challenger>>::Commitment,
    instances: Vec<Option<SerializedPreprocessedInstanceMeta>>,
    matrix_to_instance: Vec<usize>,
}

impl<SC: StarkGenericConfig> SerializedStarkCommon<SC> {
    fn from_common(common: &CommonData<SC>) -> Option<Self> {
        common.preprocessed.as_ref().map(|gp| Self {
            commitment: gp.commitment.clone(),
            instances: gp
                .instances
                .iter()
                .map(|opt| {
                    opt.as_ref().map(|m| SerializedPreprocessedInstanceMeta {
                        matrix_index: m.matrix_index,
                        width: m.width,
                        degree_bits: m.degree_bits,
                    })
                })
                .collect(),
            matrix_to_instance: gp.matrix_to_instance.clone(),
        })
    }

    fn into_common(self) -> CommonData<SC> {
        CommonData::new(
            Some(GlobalPreprocessed {
                commitment: self.commitment,
                instances: self
                    .instances
                    .into_iter()
                    .map(|opt| {
                        opt.map(|m| PreprocessedInstanceMeta {
                            matrix_index: m.matrix_index,
                            width: m.width,
                            degree_bits: m.degree_bits,
                        })
                    })
                    .collect(),
                matrix_to_instance: self.matrix_to_instance,
            }),
            Vec::new(),
        )
    }
}

/// Clone a [`CommonData`] without requiring [`Clone`] on the upstream
/// [`GlobalPreprocessed`] / [`PreprocessedInstanceMeta`] types.
fn clone_common_data<SC: StarkGenericConfig>(common: &CommonData<SC>) -> CommonData<SC> {
    CommonData::new(
        common.preprocessed.as_ref().map(|gp| GlobalPreprocessed {
            commitment: gp.commitment.clone(),
            instances: gp
                .instances
                .iter()
                .map(|opt| {
                    opt.as_ref().map(|m| PreprocessedInstanceMeta {
                        matrix_index: m.matrix_index,
                        width: m.width,
                        degree_bits: m.degree_bits,
                    })
                })
                .collect(),
            matrix_to_instance: gp.matrix_to_instance.clone(),
        }),
        common.lookups.clone(),
    )
}

/// Custom (de)serialization for [`BatchStarkProof::stark_common`]. Persists only the
/// preprocessed binding (commitment + per-instance metadata): the part the verifier
/// needs to bind the proof to the [`CommonData`] it was generated against. `lookups`
/// are intentionally not serialized because the verifier always rebuilds them from
/// the AIRs reconstructed from proof metadata.
mod serde_stark_common {
    use alloc::vec::Vec;

    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    use super::{CommonData, SerializedStarkCommon, StarkGenericConfig};

    pub(super) fn serialize<S, SC>(value: &CommonData<SC>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
        SC: StarkGenericConfig,
    {
        SerializedStarkCommon::from_common(value).serialize(serializer)
    }

    pub(super) fn deserialize<'de, D, SC>(deserializer: D) -> Result<CommonData<SC>, D::Error>
    where
        D: Deserializer<'de>,
        SC: StarkGenericConfig,
    {
        let parsed: Option<SerializedStarkCommon<SC>> = Option::deserialize(deserializer)?;
        Ok(parsed
            .map(SerializedStarkCommon::into_common)
            .unwrap_or_else(|| CommonData::new(None, Vec::new())))
    }
}

/// Proof bundle and metadata for the unified batch STARK proof across all circuit tables.
#[derive(Serialize, Deserialize)]
#[serde(bound = "")]
pub struct BatchStarkProof<SC>
where
    SC: StarkGenericConfig,
{
    /// The core cryptographic proof generated by `p3-batch-stark`.
    pub proof: BatchProof<SC>,
    /// Packing configuration used for the Witness, Public, and unified ALU tables.
    pub table_packing: TablePacking,
    /// The number of rows in each of the circuit tables.
    pub rows: RowCounts,
    /// Variant used for the primitive ALU table.
    pub alu_variant: AirVariant,
    /// The degree of the field extension (`D`) used for the proof.
    pub ext_degree: usize,
    /// The binomial coefficient `W` for extension field multiplication, if `ext_degree > 1`.
    pub w_binomial: Option<Val<SC>>,
    /// When `true` with `ext_degree == 5`, the ALU uses quintic trinomial reduction (`X^5+X^2-1`).
    #[serde(default)]
    pub alu_quintic_trinomial: bool,
    /// Manifest describing batched non-primitive tables defined at runtime.
    pub non_primitives: Vec<NonPrimitiveTableEntry<SC>>,
    /// Common data derived from the final table AIRs after trace construction.
    #[serde(with = "serde_stark_common")]
    pub stark_common: CommonData<SC>,
}

impl<SC> core::fmt::Debug for BatchStarkProof<SC>
where
    SC: StarkGenericConfig,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let stark_common_summary = self.stark_common.preprocessed.as_ref().map(|gp| {
            (
                gp.instances.len(),
                gp.matrix_to_instance.len(),
                self.stark_common.lookups.len(),
            )
        });
        f.debug_struct("BatchStarkProof")
            .field("table_packing", &self.table_packing)
            .field("rows", &self.rows)
            .field("ext_degree", &self.ext_degree)
            .field("w_binomial", &self.w_binomial)
            .field("alu_quintic_trinomial", &self.alu_quintic_trinomial)
            .field(
                "stark_common(instances, matrices, lookups)",
                &stark_common_summary,
            )
            .finish()
    }
}

/// Produces a single batch STARK proof covering all circuit tables.
pub struct BatchStarkProver<SC>
where
    SC: StarkGenericConfig + 'static,
{
    config: SC,
    table_packing: TablePacking,
    /// Variant used for the primitive ALU AIR.
    alu_variant: AirVariant,
    /// Registered dynamic non-primitive table provers.
    non_primitive_provers: Vec<Box<dyn TableProver<SC>>>,
    /// When true, run the lookup debugger before proving to report imbalanced multisets.
    debug_lookups: bool,
}

/// Errors for the batch STARK table prover.
#[derive(Debug, Error)]
pub enum BatchStarkProverError {
    /// The extension field degree is not one of the supported values (1, 2, 4, 6, 8).
    #[error("unsupported extension degree: {0} (supported: 1,2,4,5,6,8)")]
    UnsupportedDegree(usize),

    /// An extension field with degree > 1 was requested but the binomial parameter `W` was not provided.
    #[error("missing binomial parameter W for extension-field multiplication")]
    MissingWForExtension,

    /// The batch STARK verifier rejected the proof.
    #[error("verification failed: {0}")]
    Verify(String),

    /// A non-primitive table entry references an op type for which no [`TableProver`] was registered.
    #[error("missing table prover for non-primitive op `{0:?}`")]
    MissingTableProver(NpoTypeId),
}

impl<SC, const D: usize> BaseAir<Val<SC>> for CircuitTableAir<SC, D>
where
    SC: StarkGenericConfig,
    SymbolicExpressionExt<Val<SC>, SC::Challenge>: Algebra<SymbolicExpression<Val<SC>>>,
{
    fn width(&self) -> usize {
        match self {
            Self::Const(a) => a.width(),
            Self::Public(a) => a.width(),
            Self::Alu(a) => a.width(),
            Self::Dynamic(a) => <dyn CloneableBatchAir<SC> as BaseAir<Val<SC>>>::width(a.air()),
        }
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<Val<SC>>> {
        match self {
            Self::Const(a) => a.preprocessed_trace(),
            Self::Public(a) => a.preprocessed_trace(),
            Self::Alu(a) => a.preprocessed_trace(),
            Self::Dynamic(a) => {
                <dyn CloneableBatchAir<SC> as BaseAir<Val<SC>>>::preprocessed_trace(a.air())
            }
        }
    }
}

macro_rules! impl_circuit_table_air_for_builder {
    ($builder_ty:ty) => {
        fn eval(&self, builder: &mut $builder_ty) {
            match self {
                Self::Const(a) => Air::<$builder_ty>::eval(a, builder),
                Self::Public(a) => Air::<$builder_ty>::eval(a, builder),
                Self::Alu(a) => Air::<$builder_ty>::eval(a, builder),
                Self::Dynamic(a) => Air::<$builder_ty>::eval(a, builder),
            }
        }
    };
}

impl<SC, const D: usize> Air<SymbolicAirBuilder<Val<SC>, SC::Challenge>> for CircuitTableAir<SC, D>
where
    SC: StarkGenericConfig,
    Val<SC>: PrimeField,
    SymbolicExpressionExt<Val<SC>, SC::Challenge>:
        Algebra<SymbolicExpression<Val<SC>>> + Algebra<SC::Challenge>,
{
    impl_circuit_table_air_for_builder!(SymbolicAirBuilder<Val<SC>, SC::Challenge>);
}

#[cfg(debug_assertions)]
impl<'a, SC, const D: usize> Air<DebugConstraintBuilder<'a, Val<SC>, SC::Challenge>>
    for CircuitTableAir<SC, D>
where
    SC: StarkGenericConfig,
    Val<SC>: PrimeField,
    SymbolicExpressionExt<Val<SC>, SC::Challenge>: Algebra<SymbolicExpression<Val<SC>>>,
{
    impl_circuit_table_air_for_builder!(DebugConstraintBuilder<'a, Val<SC>, SC::Challenge>);
}

impl<'a, SC, const D: usize> Air<ProverConstraintFolderWithLookups<'a, SC>>
    for CircuitTableAir<SC, D>
where
    SC: StarkGenericConfig,
    Val<SC>: PrimeField,
    SymbolicExpressionExt<Val<SC>, SC::Challenge>: Algebra<SymbolicExpression<Val<SC>>>,
{
    impl_circuit_table_air_for_builder!(ProverConstraintFolderWithLookups<'a, SC>);
}

impl<'a, SC, const D: usize> Air<VerifierConstraintFolderWithLookups<'a, SC>>
    for CircuitTableAir<SC, D>
where
    SC: StarkGenericConfig,
    Val<SC>: PrimeField,
    SymbolicExpressionExt<Val<SC>, SC::Challenge>: Algebra<SymbolicExpression<Val<SC>>>,
{
    impl_circuit_table_air_for_builder!(VerifierConstraintFolderWithLookups<'a, SC>);
}

impl<SC, const D: usize> LookupAir<Val<SC>> for CircuitTableAir<SC, D>
where
    SC: StarkGenericConfig,
    Val<SC>: PrimeField,
    SymbolicExpressionExt<Val<SC>, SC::Challenge>:
        Algebra<SymbolicExpression<Val<SC>>> + Algebra<SC::Challenge>,
{
    fn add_lookup_columns(&mut self) -> Vec<usize> {
        match self {
            Self::Const(a) => ConstAir::<Val<SC>, D>::add_lookup_columns(a),
            Self::Public(a) => PublicAir::<Val<SC>, D>::add_lookup_columns(a),
            Self::Alu(a) => AluAir::<Val<SC>, D>::add_lookup_columns(a),
            Self::Dynamic(a) => DynamicAirEntry::<SC>::add_lookup_columns(a),
        }
    }

    fn get_lookups(&mut self) -> Vec<Lookup<Val<SC>>> {
        match self {
            Self::Const(a) => ConstAir::<Val<SC>, D>::get_lookups(a),
            Self::Public(a) => PublicAir::<Val<SC>, D>::get_lookups(a),
            Self::Alu(a) => AluAir::<Val<SC>, D>::get_lookups(a),
            Self::Dynamic(a) => DynamicAirEntry::<SC>::get_lookups(a),
        }
    }
}

/// Const-generic dispatch for [`BatchStarkProver::register_poseidon2_table`]: only the chosen
/// extension degree's `BinomiallyExtendable` bound is required on `Val<SC>`.
#[doc(hidden)]
pub trait RegisterPoseidon2ForExt<const D: usize, SC>
where
    SC: StarkGenericConfig + 'static,
{
    fn register_poseidon2(prover: &mut BatchStarkProver<SC>, config: Poseidon2Config);
}

impl<SC> RegisterPoseidon2ForExt<2, SC> for ()
where
    SC: StarkGenericConfig + 'static + Send + Sync,
    Val<SC>: StarkField + BinomiallyExtendable<2>,
    SymbolicExpressionExt<Val<SC>, SC::Challenge>:
        Algebra<SymbolicExpression<Val<SC>>> + Algebra<SC::Challenge>,
{
    fn register_poseidon2(prover: &mut BatchStarkProver<SC>, config: Poseidon2Config) {
        prover.register_table_prover(Box::new(Poseidon2ProverD2::new(
            config,
            ConstraintProfile::Standard,
        )));
    }
}

impl<SC> RegisterPoseidon2ForExt<4, SC> for ()
where
    SC: StarkGenericConfig + 'static + Send + Sync,
    Val<SC>: StarkField + BinomiallyExtendable<4>,
    SymbolicExpressionExt<Val<SC>, SC::Challenge>:
        Algebra<SymbolicExpression<Val<SC>>> + Algebra<SC::Challenge>,
{
    fn register_poseidon2(prover: &mut BatchStarkProver<SC>, config: Poseidon2Config) {
        prover.register_table_prover(Box::new(Poseidon2Prover::new(
            config,
            ConstraintProfile::Standard,
        )));
    }
}

impl<SC> RegisterPoseidon2ForExt<5, SC> for ()
where
    SC: StarkGenericConfig + 'static + Send + Sync,
    Val<SC>: StarkField + BinomiallyExtendable<4>,
    SymbolicExpressionExt<Val<SC>, SC::Challenge>:
        Algebra<SymbolicExpression<Val<SC>>> + Algebra<SC::Challenge>,
{
    fn register_poseidon2(prover: &mut BatchStarkProver<SC>, config: Poseidon2Config) {
        prover.register_table_prover(Box::new(Poseidon2Prover::new(
            config,
            ConstraintProfile::Standard,
        )));
    }
}

impl<SC> BatchStarkProver<SC>
where
    SC: StarkGenericConfig + 'static,
    Val<SC>: StarkField,
    SymbolicExpressionExt<Val<SC>, SC::Challenge>:
        Algebra<SymbolicExpression<Val<SC>>> + Algebra<SC::Challenge>,
{
    /// Create a new prover with the given STARK config and default table packing.
    pub fn new(config: SC) -> Self {
        Self {
            config,
            table_packing: TablePacking::default(),
            alu_variant: AirVariant::Optimized,
            non_primitive_provers: Vec::new(),
            debug_lookups: false,
        }
    }

    /// Override the default [`TablePacking`] configuration (builder-style).
    #[must_use]
    pub fn with_table_packing(mut self, table_packing: TablePacking) -> Self {
        self.table_packing = table_packing;
        self
    }

    /// Enable the lookup debugger. When set, `prove_all_tables` will run
    /// `check_lookups` on the constructed traces before generating the proof,
    /// panicking with a detailed message on any multiset imbalance.
    #[must_use]
    pub const fn with_debug_lookups(mut self) -> Self {
        self.debug_lookups = true;
        self
    }

    /// Register a dynamic non-primitive table prover.
    pub fn register_table_prover(&mut self, prover: Box<dyn TableProver<SC>>) {
        self.non_primitive_provers.push(prover);
    }

    /// Builder-style registration for a dynamic non-primitive table prover.
    #[must_use]
    pub fn with_table_prover(mut self, prover: Box<dyn TableProver<SC>>) -> Self {
        self.register_table_prover(prover);
        self
    }

    /// Register the non-primitive Poseidon2 table prover for extension degree `D` (`2` or `4`).
    pub fn register_poseidon2_table<const D: usize>(&mut self, config: Poseidon2Config)
    where
        SC: Send + Sync,
        (): RegisterPoseidon2ForExt<D, SC>,
    {
        <() as RegisterPoseidon2ForExt<D, SC>>::register_poseidon2(self, config);
    }

    /// Register the recompose (BF→EF packing) table prover(s) for extension degree `D`.
    ///
    /// Set `split_coeff_tables` to `true` when the Poseidon2 permutation degree can differ
    /// from the circuit extension degree `D` (e.g. D=1 Poseidon2 in a D=5 circuit). That
    /// registers both the standard `recompose` table and `recompose/coeff` (per-coefficient
    /// WitnessChecks receives only where the circuit uses them).
    pub fn register_recompose_table<const D: usize>(&mut self, split_coeff_tables: bool)
    where
        SC: Send + Sync,
    {
        for prover in recompose_table_provers::<SC, D>(1, split_coeff_tables) {
            self.register_table_prover(prover);
        }
    }

    /// Builder-style registration for the recompose table prover.
    #[must_use]
    pub fn with_recompose_table<const D: usize>(mut self, split_coeff_tables: bool) -> Self
    where
        SC: Send + Sync,
    {
        self.register_recompose_table::<D>(split_coeff_tables);
        self
    }

    /// Return the current [`TablePacking`] configuration.
    #[inline]
    pub const fn table_packing(&self) -> &TablePacking {
        &self.table_packing
    }

    /// Select which ALU AIR variant to use for primitive tables.
    #[must_use]
    pub const fn with_alu_variant(mut self, variant: AirVariant) -> Self {
        self.alu_variant = variant;
        self
    }

    /// Generate a unified batch STARK proof for all circuit tables.
    #[instrument(skip_all)]
    pub fn prove_all_tables<EF>(
        &self,
        traces: &Traces<EF>,
        circuit_prover_data: &CircuitProverData<SC>,
    ) -> Result<BatchStarkProof<SC>, BatchStarkProverError>
    where
        EF: Field + BasedVectorSpace<Val<SC>> + ExtractBinomialW<Val<SC>>,
        SymbolicExpressionExt<Val<SC>, SC::Challenge>: Algebra<SymbolicExpression<Val<SC>>>,
    {
        let w_opt = EF::extract_w();
        match EF::DIMENSION {
            1 => self.prove::<EF, 1>(traces, None, circuit_prover_data),
            2 => self.prove::<EF, 2>(traces, w_opt, circuit_prover_data),
            4 => self.prove::<EF, 4>(traces, w_opt, circuit_prover_data),
            5 => self.prove::<EF, 5>(traces, w_opt, circuit_prover_data),
            6 => self.prove::<EF, 6>(traces, w_opt, circuit_prover_data),
            8 => self.prove::<EF, 8>(traces, w_opt, circuit_prover_data),
            d => Err(BatchStarkProverError::UnsupportedDegree(d)),
        }
    }

    /// Verify the unified batch STARK proof against all tables.
    pub fn verify_all_tables(
        &self,
        proof: &BatchStarkProof<SC>,
    ) -> Result<(), BatchStarkProverError> {
        let common = &proof.stark_common;
        match proof.ext_degree {
            1 => self.verify::<1>(proof, None, common),
            2 => self.verify::<2>(proof, proof.w_binomial, common),
            4 => self.verify::<4>(proof, proof.w_binomial, common),
            5 => self.verify::<5>(proof, proof.w_binomial, common),
            6 => self.verify::<6>(proof, proof.w_binomial, common),
            8 => self.verify::<8>(proof, proof.w_binomial, common),
            d => Err(BatchStarkProverError::UnsupportedDegree(d)),
        }
    }

    /// Generate a batch STARK proof for a specific extension field degree.
    ///
    /// This is the core proving logic that handles all circuit tables for a given
    /// extension field dimension. It constructs AIRs, converts traces to matrices,
    /// and generates the unified proof.
    fn prove<EF, const D: usize>(
        &self,
        traces: &Traces<EF>,
        w_binomial: Option<Val<SC>>,
        circuit_prover_data: &CircuitProverData<SC>,
    ) -> Result<BatchStarkProof<SC>, BatchStarkProverError>
    where
        EF: Field + BasedVectorSpace<Val<SC>> + ExtractBinomialW<Val<SC>>,
    {
        let primitive = &circuit_prover_data.primitive_columns;
        let non_primitive = &circuit_prover_data.non_primitive_columns;
        let prover_data = &circuit_prover_data.prover_data;

        // One lookup per NpoTypeId instead of repeated `op_type()` (clones inner id string).
        let prover_index_by_type: BTreeMap<NpoTypeId, usize> = self
            .non_primitive_provers
            .iter()
            .enumerate()
            .map(|(i, p)| (p.op_type(), i))
            .collect();

        // Build matrices and AIRs per table.
        let packing = &self.table_packing;
        let min_height = packing.min_trace_height();

        // Check if Alu table has only dummy operations (trace length <= 1).
        // The table implementation adds a dummy row when empty, so we check for <= 1.
        // Using lanes > 1 with only dummy operations causes issues in recursive verification
        // due to a bug in how multi-lane padding interacts with lookup constraints.
        // We automatically reduce lanes to 1 in these cases with a warning.
        let alu_trace_only_dummy = traces.alu_trace.op_kind.len() <= 1;

        let alu_lanes = if alu_trace_only_dummy && packing.alu_lanes() > 1 {
            tracing::warn!(
                "ALu table has only dummy operations but alu_lanes={} > 1. Reducing to \
                 alu_lanes=1 to avoid recursive verification issues. Consider using \
                 alu_lanes=1 when no additions are expected.",
                packing.alu_lanes()
            );
            1
        } else {
            packing.alu_lanes()
        };

        // Const — preprocessed is already in [ext_mult, index] 2-col format.
        let const_rows = traces.const_trace.values.len();
        let const_prep = primitive[PrimitiveOpType::Const as usize].clone();
        let const_air = ConstAir::<Val<SC>, D>::new_with_preprocessed(const_rows, const_prep)
            .with_min_height(min_height);
        let const_matrix: RowMajorMatrix<Val<SC>> =
            ConstAir::<Val<SC>, D>::trace_to_matrix(&traces.const_trace, min_height);

        // Public — reduce lanes to 1 if the table has only dummy operations.
        let public_trace_only_dummy = traces.public_trace.values.len() <= 1;
        let public_lanes = if public_trace_only_dummy && packing.public_lanes() > 1 {
            tracing::warn!(
                "Public table has only dummy operations but public_lanes={} > 1. Reducing to \
                 public_lanes=1 to avoid recursive verification issues. Consider using \
                 public_lanes=1 when few public inputs are expected.",
                packing.public_lanes()
            );
            1
        } else {
            packing.public_lanes()
        };

        // Preprocessed is already in [ext_mult, index] 2-col format.
        let public_rows = traces.public_trace.values.len();
        let public_prep = primitive[PrimitiveOpType::Public as usize].clone();
        let public_air =
            PublicAir::<Val<SC>, D>::new_with_preprocessed(public_rows, public_lanes, public_prep)
                .with_min_height(min_height);
        let public_matrix: RowMajorMatrix<Val<SC>> = PublicAir::<Val<SC>, D>::trace_to_matrix(
            &traces.public_trace,
            public_lanes,
            min_height,
        );

        // ALU — preprocessed is already in 10-col format (with multiplicities) from
        // get_airs_and_degrees_with_prep. When the trace is empty, a dummy row is included.
        let alu_rows = traces.alu_trace.values.len();
        let alu_prep = primitive[PrimitiveOpType::Alu as usize].clone();
        let alu_num_ops = alu_prep.len() / AluAir::<Val<SC>, D>::preprocessed_lane_width();
        let horner_k = packing.horner_packed_steps();
        let alu_quintic = D == 5 && EF::alu_is_quintic_trinomial();
        let alu_air: AluAir<Val<SC>, D> = if D == 1 {
            AluAir::<Val<SC>, D>::new_with_preprocessed(alu_num_ops, alu_lanes, alu_prep, horner_k)
                .with_min_height(min_height)
        } else if alu_quintic {
            AluAir::<Val<SC>, D>::new_quintic_trinomial_with_preprocessed(
                alu_num_ops,
                alu_lanes,
                alu_prep,
                horner_k,
            )
            .with_min_height(min_height)
        } else {
            let w = w_binomial.ok_or(BatchStarkProverError::MissingWForExtension)?;
            AluAir::<Val<SC>, D>::new_binomial_with_preprocessed(
                alu_num_ops,
                alu_lanes,
                w,
                alu_prep,
                horner_k,
            )
            .with_min_height(min_height)
        };
        let alu_matrix: RowMajorMatrix<Val<SC>> =
            alu_air.trace_to_matrix(&traces.alu_trace, min_height);
        let alu_scheduled_entries = alu_air.scheduled_entry_count();

        // We first handle all non-primitive tables dynamically, which will then be batched alongside primitive ones.
        // Each trace must have a corresponding registered prover for it to be provable.
        for (op_type, trace) in &traces.non_primitive_traces {
            if trace.rows() == 0 {
                continue;
            }
            if !prover_index_by_type.contains_key(op_type) {
                return Err(BatchStarkProverError::MissingTableProver(op_type.clone()));
            }
        }

        let mut dynamic_instances: Vec<BatchTableInstance<SC>> =
            Vec::with_capacity(self.non_primitive_provers.len());
        if D == 1 {
            let t: &Traces<Val<SC>> = unsafe { transmute_traces(traces) };
            for p in &self.non_primitive_provers {
                if let Some(instance) = p.batch_instance_d1(&self.config, packing, t) {
                    dynamic_instances.push(instance);
                }
            }
        } else if D == 2 {
            type EF2<F> = BinomialExtensionField<F, 2>;
            let t: &Traces<EF2<Val<SC>>> = unsafe { transmute_traces(traces) };
            for p in &self.non_primitive_provers {
                if let Some(instance) = p.batch_instance_d2(&self.config, packing, t) {
                    dynamic_instances.push(instance);
                }
            }
        } else if D == 4 {
            type EF4<F> = BinomialExtensionField<F, 4>;
            let t: &Traces<EF4<Val<SC>>> = unsafe { transmute_traces(traces) };
            for p in &self.non_primitive_provers {
                if let Some(instance) = p.batch_instance_d4(&self.config, packing, t) {
                    dynamic_instances.push(instance);
                }
            }
        } else if D == 6 {
            type EF6<F> = BinomialExtensionField<F, 6>;
            let t: &Traces<EF6<Val<SC>>> = unsafe { transmute_traces(traces) };
            for p in &self.non_primitive_provers {
                if let Some(instance) = p.batch_instance_d6(&self.config, packing, t) {
                    dynamic_instances.push(instance);
                }
            }
        } else if D == 8 {
            type EF8<F> = BinomialExtensionField<F, 8>;
            let t: &Traces<EF8<Val<SC>>> = unsafe { transmute_traces(traces) };
            for p in &self.non_primitive_provers {
                if let Some(instance) = p.batch_instance_d8(&self.config, packing, t) {
                    dynamic_instances.push(instance);
                }
            }
        } else if D == 5 {
            type EF5<F> = p3_field::extension::QuinticTrinomialExtensionField<F>;
            let t: &Traces<EF5<Val<SC>>> = unsafe { transmute_traces(traces) };
            for p in &self.non_primitive_provers {
                if let Some(instance) = p.batch_instance_d5(&self.config, packing, t) {
                    dynamic_instances.push(instance);
                }
            }
        }

        // The `batch_instance_dN` methods regenerate Poseidon2 preprocessed data from
        // runtime ops using `extract_preprocessed_from_operations`.
        //
        // Hence, we override here with the committed preprocessed data so the debug
        // lookup check is consistent with the committed preprocessed trace.
        for instance in &mut dynamic_instances {
            if let Some(committed_prep) = non_primitive.get(&instance.op_type)
                && let Some(&pi) = prover_index_by_type.get(&instance.op_type)
            {
                let p = &self.non_primitive_provers[pi];
                if let Some(new_air) = p.air_with_committed_preprocessed(
                    committed_prep.clone(),
                    min_height,
                    instance.lanes,
                    D as u32,
                ) {
                    instance.air = new_air;
                }
            }
        }

        TraceTablesLayout {
            const_: AirTableShape {
                main_cols: BaseAir::width(&const_air),
                prep_cols: ConstAir::<Val<SC>, D>::preprocessed_width(),
                rows: const_rows,
                lanes: 1,
            },
            public: AirTableShape {
                main_cols: BaseAir::width(&public_air),
                prep_cols: public_air.preprocessed_width(),
                rows: public_rows.div_ceil(public_lanes),
                lanes: public_lanes,
            },
            alu: AirTableShape {
                main_cols: BaseAir::width(&alu_air),
                prep_cols: alu_air.preprocessed_width(),
                rows: alu_scheduled_entries.div_ceil(alu_lanes),
                lanes: alu_lanes,
            },
            non_primitives: dynamic_instances
                .iter()
                .map(|inst| {
                    let prep_cols = BaseAir::preprocessed_trace(&inst.air)
                        .map(|m| m.width())
                        .unwrap_or(0);
                    let rows = traces
                        .non_primitive_traces
                        .get(&inst.op_type)
                        .map(|t| t.rows())
                        .unwrap_or(inst.rows);
                    (
                        inst.op_type.clone(),
                        AirTableShape {
                            main_cols: inst.trace.width(),
                            prep_cols,
                            rows: rows / inst.lanes,
                            lanes: inst.lanes,
                        },
                    )
                })
                .collect(),
        }
        .log();

        // Wrap AIRs in enum for heterogeneous batching and build instances in fixed order.
        let mut air_storage: Vec<CircuitTableAir<SC, D>> =
            Vec::with_capacity(NUM_PRIMITIVE_TABLES + dynamic_instances.len());
        let mut trace_storage: Vec<RowMajorMatrix<Val<SC>>> =
            Vec::with_capacity(NUM_PRIMITIVE_TABLES + dynamic_instances.len());
        let mut public_storage: Vec<Vec<Val<SC>>> =
            Vec::with_capacity(NUM_PRIMITIVE_TABLES + dynamic_instances.len());
        let mut non_primitive_meta: Vec<(NpoTypeId, usize, usize, AirVariant)> =
            Vec::with_capacity(dynamic_instances.len());

        // Pad all trace matrices to at least min_height (for FRI compatibility)
        air_storage.push(CircuitTableAir::Const(const_air));
        trace_storage.push(const_matrix);
        public_storage.push(Vec::new());

        air_storage.push(CircuitTableAir::Public(public_air));
        trace_storage.push(public_matrix);
        public_storage.push(Vec::new());

        air_storage.push(CircuitTableAir::Alu(alu_air));
        trace_storage.push(alu_matrix);
        public_storage.push(Vec::new());

        for instance in dynamic_instances {
            let BatchTableInstance {
                op_type,
                air,
                mut trace,
                public_values,
                lanes,
                rows,
            } = instance;
            air_storage.push(CircuitTableAir::Dynamic(air));
            trace.pad_to_min_power_of_two_height(min_height, Val::<SC>::ZERO);
            trace_storage.push(trace);
            public_storage.push(public_values);
            non_primitive_meta.push((op_type, rows, lanes, AirVariant::Baseline));
        }

        // Use the pre-computed ProverData when the AIR structure is unchanged (common case).
        // Recompute only when lane reduction altered the lookup layout, since the number of
        // lookups per table depends on lane count.
        let lanes_reduced = (alu_trace_only_dummy && packing.alu_lanes() > 1)
            || (public_trace_only_dummy && packing.public_lanes() > 1);
        let recomputed_data: Option<ProverData<SC>> = if lanes_reduced {
            let trace_ext_degree_bits: Vec<usize> = trace_storage
                .iter()
                .map(|m| log2_strict_usize(m.height()) + self.config.is_zk())
                .collect();
            Some(ProverData::from_airs_and_degrees(
                &self.config,
                &mut air_storage,
                &trace_ext_degree_bits,
            ))
        } else {
            None
        };
        let effective_prover_data = recomputed_data.as_ref().unwrap_or(prover_data);

        let proof = {
            let trace_refs: Vec<&RowMajorMatrix<Val<SC>>> = trace_storage.iter().collect();
            let instances: Vec<StarkInstance<'_, SC, CircuitTableAir<SC, D>>> =
                StarkInstance::new_multiple(
                    &air_storage,
                    &trace_refs,
                    &public_storage,
                    &effective_prover_data.common,
                );

            if self.debug_lookups {
                use p3_lookup::debug_util::{LookupDebugInstance, check_lookups};

                let mut preprocessed_traces: Vec<Option<RowMajorMatrix<Val<SC>>>> = instances
                    .iter()
                    .map(|inst| inst.air.preprocessed_trace())
                    .collect();

                for (j, (op_type, _, lanes, _)) in non_primitive_meta.iter().enumerate() {
                    if let Some(committed_prep) = non_primitive.get(op_type) {
                        let prover = self
                            .non_primitive_provers
                            .iter()
                            .find(|p| TableProver::op_type(p.as_ref()) == *op_type);
                        if let Some(prover) = prover
                            && let Some(air) = prover.air_with_committed_preprocessed(
                                committed_prep.clone(),
                                min_height,
                                *lanes,
                                D as u32,
                            )
                            && let Some(trace) = air.preprocessed_trace()
                        {
                            preprocessed_traces[NUM_PRIMITIVE_TABLES + j] = Some(trace);
                        }
                    }
                }

                let debug_instances: Vec<LookupDebugInstance<'_, Val<SC>>> = instances
                    .iter()
                    .zip(preprocessed_traces.iter())
                    .map(|(inst, prep)| LookupDebugInstance {
                        main_trace: inst.trace,
                        preprocessed_trace: prep,
                        public_values: &inst.public_values,
                        lookups: &inst.lookups,
                        permutation_challenges: &[],
                    })
                    .collect();
                check_lookups(&debug_instances);
            }

            p3_batch_stark::prove_batch(&self.config, &instances, effective_prover_data)
        };

        let dynamic_public_values = public_storage.drain(NUM_PRIMITIVE_TABLES..);
        let non_primitives: Vec<NonPrimitiveTableEntry<SC>> = non_primitive_meta
            .into_iter()
            .zip(dynamic_public_values)
            .map(
                |((op_type, rows, lanes, air_variant), public_values)| NonPrimitiveTableEntry {
                    op_type,
                    rows,
                    lanes,
                    public_values,
                    air_variant,
                },
            )
            .collect();

        // Ensure all primitive table row counts are at least 1
        // RowCounts::new requires non-zero counts, so pad zeros to 1
        let const_rows_padded = const_rows.max(1);
        let public_rows_padded = public_rows.max(1);
        let alu_rows_padded = alu_rows.max(1);

        // Store the effective packing (reduced lanes if applicable) so the verifier matches
        // proving. Clone full config so `horner_packed_steps`, NPO lane overrides, etc. are preserved.
        let effective_packing = self
            .table_packing
            .clone()
            .with_public_alu_lanes(public_lanes, alu_lanes);

        // Populate `stark_common` so the proof is self-binding to the preprocessed metadata.
        let stark_common = recomputed_data
            .map(|pd| pd.common)
            .unwrap_or_else(|| clone_common_data(&prover_data.common));

        Ok(BatchStarkProof {
            proof,
            table_packing: effective_packing,
            rows: RowCounts::new([const_rows_padded, public_rows_padded, alu_rows_padded]),
            alu_variant: self.alu_variant,
            ext_degree: D,
            w_binomial: if D > 1 { w_binomial } else { None },
            alu_quintic_trinomial: alu_quintic,
            non_primitives,
            stark_common,
        })
    }

    /// Verify a batch STARK proof for a specific extension field degree.
    ///
    /// This reconstructs the AIRs from the proof metadata and verifies the proof
    /// against all circuit tables. The AIRs are reconstructed using the same
    /// configuration that was used during proof generation.
    fn verify<const D: usize>(
        &self,
        proof: &BatchStarkProof<SC>,
        w_binomial: Option<Val<SC>>,
        common: &CommonData<SC>,
    ) -> Result<(), BatchStarkProverError> {
        let prover_index_by_type: BTreeMap<NpoTypeId, usize> = self
            .non_primitive_provers
            .iter()
            .enumerate()
            .map(|(i, p)| (p.op_type(), i))
            .collect();

        // Rebuild AIRs in the same order as prove.
        let packing = &proof.table_packing;
        let public_lanes = packing.public_lanes();
        let alu_lanes = packing.alu_lanes();
        let min_height = packing.min_trace_height();

        let const_air = CircuitTableAir::Const(
            ConstAir::<Val<SC>, D>::new(proof.rows[PrimitiveTable::Const])
                .with_min_height(min_height),
        );
        let public_air = CircuitTableAir::Public(
            PublicAir::<Val<SC>, D>::new(proof.rows[PrimitiveTable::Public], public_lanes)
                .with_min_height(min_height),
        );
        let horner_k = packing.horner_packed_steps();
        let alu_air: CircuitTableAir<SC, D> = if D == 1 {
            CircuitTableAir::Alu(
                AluAir::<Val<SC>, D>::new(proof.rows[PrimitiveTable::Alu], alu_lanes)
                    .with_horner_pack_k(horner_k)
                    .with_min_height(min_height),
            )
        } else if D == 5 && proof.alu_quintic_trinomial {
            CircuitTableAir::Alu(
                AluAir::<Val<SC>, D>::new_quintic_trinomial(
                    proof.rows[PrimitiveTable::Alu],
                    alu_lanes,
                )
                .with_horner_pack_k(horner_k)
                .with_min_height(min_height),
            )
        } else {
            let w = w_binomial.ok_or(BatchStarkProverError::MissingWForExtension)?;
            CircuitTableAir::Alu(
                AluAir::<Val<SC>, D>::new_binomial(proof.rows[PrimitiveTable::Alu], alu_lanes, w)
                    .with_horner_pack_k(horner_k)
                    .with_min_height(min_height),
            )
        };
        let mut airs = vec![const_air, public_air, alu_air];
        let mut pvs: Vec<Vec<Val<SC>>> =
            Vec::with_capacity(NUM_PRIMITIVE_TABLES + proof.non_primitives.len());
        pvs.resize_with(NUM_PRIMITIVE_TABLES, Vec::new);

        for entry in &proof.non_primitives {
            let pi = *prover_index_by_type.get(&entry.op_type).ok_or_else(|| {
                BatchStarkProverError::Verify(format!(
                    "unknown non-primitive op: {:?}",
                    entry.op_type
                ))
            })?;
            let plugin = &self.non_primitive_provers[pi];
            let air = plugin
                .batch_air_from_table_entry(&self.config, D, proof.ext_degree as u32, entry)
                .map_err(BatchStarkProverError::Verify)?;
            airs.push(CircuitTableAir::Dynamic(air));
            pvs.push(entry.public_values.clone());
        }

        // Derive lookups from the rebuilt AIRs so the layout always reflects the effective
        // lane counts stored in `proof.table_packing`. The serialized `stark_common` only
        // carries the preprocessed binding, not the lookup contexts.
        let lookups: Vec<Vec<Lookup<Val<SC>>>> = airs.iter_mut().map(|a| a.get_lookups()).collect();
        let effective_common = CommonData::new(
            common.preprocessed.as_ref().map(|g| GlobalPreprocessed {
                commitment: g.commitment.clone(),
                instances: g.instances.clone(),
                matrix_to_instance: g.matrix_to_instance.clone(),
            }),
            lookups,
        );

        p3_batch_stark::verify_batch(&self.config, &airs, &proof.proof, &pvs, &effective_common)
            .map_err(|e| BatchStarkProverError::Verify(format!("{e:?}")))
    }
}

/// Poseidon2 AIR builders for the given extension degree `D` (typically `2` or `4`).
pub fn poseidon2_air_builders<SC, const D: usize>() -> Vec<Box<dyn NpoAirBuilder<SC, D>>>
where
    SC: StarkGenericConfig + 'static + Send + Sync,
    Val<SC>: BinomiallyExtendable<D> + StarkField,
    SymbolicExpressionExt<Val<SC>, SC::Challenge>:
        Algebra<SymbolicExpression<Val<SC>>> + Algebra<SC::Challenge>,
    Poseidon2AirBuilder<D>: NpoAirBuilder<SC, D>,
{
    vec![Box::new(Poseidon2AirBuilder)]
}

/// Create Poseidon2 table provers for D=4 (e.g. BabyBear, KoalaBear).
pub fn poseidon2_table_provers_d4<SC>(config: Poseidon2Config) -> Vec<Box<dyn TableProver<SC>>>
where
    SC: StarkGenericConfig + 'static + Send + Sync,
    Val<SC>: BinomiallyExtendable<4> + StarkField,
    SymbolicExpressionExt<Val<SC>, SC::Challenge>:
        Algebra<SymbolicExpression<Val<SC>>> + Algebra<SC::Challenge>,
{
    vec![Box::new(Poseidon2Prover::new(
        config,
        ConstraintProfile::Standard,
    ))]
}

/// Create Poseidon2 table provers for `D = 5` circuit traces (e.g. Koala quintic with base-first Poseidon).
pub fn poseidon2_table_provers_d5<SC>(config: Poseidon2Config) -> Vec<Box<dyn TableProver<SC>>>
where
    SC: StarkGenericConfig + 'static + Send + Sync,
    Val<SC>: StarkField + BinomiallyExtendable<4>,
    SymbolicExpressionExt<Val<SC>, SC::Challenge>:
        Algebra<SymbolicExpression<Val<SC>>> + Algebra<SC::Challenge>,
{
    vec![Box::new(Poseidon2Prover::new(
        config,
        ConstraintProfile::Standard,
    ))]
}

/// Poseidon2 AIR builders for D=2 (e.g. Goldilocks).
pub fn poseidon2_air_builders_d2<SC>() -> Vec<Box<dyn NpoAirBuilder<SC, 2>>>
where
    SC: StarkGenericConfig + 'static + Send + Sync,
    Val<SC>: BinomiallyExtendable<2> + StarkField,
    SymbolicExpressionExt<Val<SC>, SC::Challenge>:
        Algebra<SymbolicExpression<Val<SC>>> + Algebra<SC::Challenge>,
{
    vec![Box::new(Poseidon2AirBuilder::<2>)]
}

/// Poseidon2 AIR builders for D=4 (e.g. BabyBear, KoalaBear).
pub fn poseidon2_air_builders_d4<SC>() -> Vec<Box<dyn NpoAirBuilder<SC, 4>>>
where
    SC: StarkGenericConfig + 'static + Send + Sync,
    Val<SC>: BinomiallyExtendable<4> + StarkField,
    SymbolicExpressionExt<Val<SC>, SC::Challenge>:
        Algebra<SymbolicExpression<Val<SC>>> + Algebra<SC::Challenge>,
{
    vec![Box::new(Poseidon2AirBuilder::<4>)]
}

/// Poseidon2 AIR builders for `D = 5` circuit traces (e.g. KoalaBear quintic).
pub fn poseidon2_air_builders_d5<SC>() -> Vec<Box<dyn NpoAirBuilder<SC, 5>>>
where
    SC: StarkGenericConfig + 'static + Send + Sync,
    Val<SC>: StarkField,
    SymbolicExpressionExt<Val<SC>, SC::Challenge>:
        Algebra<SymbolicExpression<Val<SC>>> + Algebra<SC::Challenge>,
{
    vec![Box::new(Poseidon2AirBuilder::<5>)]
}

/// Returns a type-erased Recompose preprocessor.
///
/// When `split_coeff_tables` is true, preprocesses both `recompose` and `recompose/coeff` rows.
pub fn recompose_preprocessor<F>(split_coeff_tables: bool) -> Box<dyn NpoPreprocessor<F>>
where
    F: StarkField + PrimeField,
    RecomposePreprocessor: NpoPreprocessor<F>,
{
    Box::new(RecomposePreprocessor::new(split_coeff_tables))
}

/// Recompose table provers for a given extension field degree.
///
/// When `split_coeff_tables` is true, returns both the standard table and the `recompose/coeff`
/// variant.
pub fn recompose_table_provers<SC, const D: usize>(
    lanes: usize,
    split_coeff_tables: bool,
) -> Vec<Box<dyn TableProver<SC>>>
where
    SC: StarkGenericConfig + 'static + Send + Sync,
    Val<SC>: StarkField,
    SymbolicExpressionExt<Val<SC>, SC::Challenge>:
        Algebra<SymbolicExpression<Val<SC>>> + Algebra<SC::Challenge>,
{
    if split_coeff_tables {
        vec![
            Box::new(RecomposeProver::<D>::new(lanes, false)),
            Box::new(RecomposeProver::<D>::new(lanes, true)),
        ]
    } else {
        vec![Box::new(RecomposeProver::<D>::new(lanes, false))]
    }
}

/// Recompose AIR builders for a given extension field degree.
///
/// `split_coeff_tables` must match the value used in the paired [`recompose_table_provers`].
pub fn recompose_air_builders<SC, const D: usize>(
    lanes: usize,
    split_coeff_tables: bool,
) -> Vec<Box<dyn NpoAirBuilder<SC, D>>>
where
    SC: StarkGenericConfig + 'static + Send + Sync,
    Val<SC>: StarkField,
    SymbolicExpressionExt<Val<SC>, SC::Challenge>:
        Algebra<SymbolicExpression<Val<SC>>> + Algebra<SC::Challenge>,
{
    if split_coeff_tables {
        vec![
            Box::new(RecomposeAirBuilder::<D>::new(lanes, false)),
            Box::new(RecomposeAirBuilder::<D>::new(lanes, true)),
        ]
    } else {
        vec![Box::new(RecomposeAirBuilder::<D>::new(lanes, false))]
    }
}

#[cfg(test)]
mod tests;
