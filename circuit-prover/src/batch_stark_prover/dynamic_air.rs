use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;

#[cfg(debug_assertions)]
use p3_air::DebugConstraintBuilder;
use p3_air::{Air, AirBuilder, BaseAir};
use p3_batch_stark::{StarkGenericConfig, Val};
use p3_circuit::ops::NpoTypeId;
use p3_circuit::tables::Traces;
use p3_field::extension::{BinomialExtensionField, QuinticTrinomialExtensionField};
use p3_field::{Algebra, PrimeField};
use p3_lookup::LookupAir;
use p3_lookup::folder::{ProverConstraintFolderWithLookups, VerifierConstraintFolderWithLookups};
use p3_lookup::lookup_traits::Lookup;
use p3_matrix::dense::RowMajorMatrix;
use p3_uni_stark::{SymbolicAirBuilder, SymbolicExpression, SymbolicExpressionExt};

use super::TablePacking;

/// Type-erased AIR implementation for dynamically registered non-primitive tables.
///
/// This allows the batch prover to mix primitive AIRs with plugin AIRs in a single heterogeneous
/// batch.
/// Internally,`DynamicAirEntry` wraps the boxed plugin AIR and exposes a shared accessor
/// so that both prover and verifier can operate without knowing the concrete underlying type.
pub struct DynamicAirEntry<SC>
where
    SC: StarkGenericConfig,
{
    air: Box<dyn CloneableBatchAir<SC>>,
}

impl<SC> DynamicAirEntry<SC>
where
    SC: StarkGenericConfig,
{
    /// Wrap a boxed [`CloneableBatchAir`] into a `DynamicAirEntry`.
    pub fn new(inner: Box<dyn CloneableBatchAir<SC>>) -> Self {
        Self { air: inner }
    }

    /// Return a shared reference to the inner AIR.
    pub fn air(&self) -> &dyn CloneableBatchAir<SC> {
        &*self.air
    }

    /// Return a mutable reference to the inner AIR.
    pub fn air_mut(&mut self) -> &mut dyn CloneableBatchAir<SC> {
        &mut *self.air
    }
}

impl<SC> Clone for DynamicAirEntry<SC>
where
    SC: StarkGenericConfig,
    SymbolicExpressionExt<Val<SC>, SC::Challenge>: Algebra<SymbolicExpression<Val<SC>>>,
{
    fn clone(&self) -> Self {
        Self {
            air: self.air.clone_box(),
        }
    }
}

impl<SC> BaseAir<Val<SC>> for DynamicAirEntry<SC>
where
    SC: StarkGenericConfig,
    SymbolicExpressionExt<Val<SC>, SC::Challenge>: Algebra<SymbolicExpression<Val<SC>>>,
{
    fn width(&self) -> usize {
        <dyn CloneableBatchAir<SC> as BaseAir<Val<SC>>>::width(self.air())
    }

    fn num_public_values(&self) -> usize {
        <dyn CloneableBatchAir<SC> as BaseAir<Val<SC>>>::num_public_values(self.air())
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<Val<SC>>> {
        <dyn CloneableBatchAir<SC> as BaseAir<Val<SC>>>::preprocessed_trace(self.air())
    }
}

macro_rules! impl_air_for_dynamic_entry {
    (
        $(#[$cfg:meta])?
        $lt:lifetime,
        $builder:ty,
        $eval_method:ident,
    ) => {
        $(#[$cfg])?
        impl<$lt, SC> Air<$builder> for DynamicAirEntry<SC>
        where
            SC: StarkGenericConfig,
            Val<SC>: PrimeField,
            SymbolicExpressionExt<Val<SC>, SC::Challenge>: Algebra<SymbolicExpression<Val<SC>>>,
        {
            fn eval(&self, builder: &mut $builder) {
                self.air().$eval_method(builder);
            }
        }
    };

    (
        $(#[$cfg:meta])?
        $builder:ty,
        $eval_method:ident,
        $add_lookup_method:ident,
        $get_lookup_method:ident
    ) => {
        $(#[$cfg])?
        impl<SC> Air<$builder> for DynamicAirEntry<SC>
        where
            SC: StarkGenericConfig,
            Val<SC>: PrimeField,
            SymbolicExpressionExt<Val<SC>, SC::Challenge>: Algebra<SymbolicExpression<Val<SC>>> + Algebra<SC::Challenge>,
        {
            fn eval(&self, builder: &mut $builder) {
                self.air().$eval_method(builder);
            }
        }

        impl<SC> LookupAir<Val<SC>> for DynamicAirEntry<SC>
        where
            SC: StarkGenericConfig,
            Val<SC>: PrimeField,
            SymbolicExpressionExt<Val<SC>, SC::Challenge>: Algebra<SymbolicExpression<Val<SC>>> + Algebra<SC::Challenge>,
        {
            fn add_lookup_columns(&mut self) -> Vec<usize> {
                self.air_mut().$add_lookup_method()
            }

            fn get_lookups(
                &mut self,
            ) -> Vec<Lookup<<$builder as AirBuilder>::F>> {
                self.air_mut().$get_lookup_method()
            }
        }
    };
}

impl_air_for_dynamic_entry!(
    SymbolicAirBuilder<Val<SC>, SC::Challenge>,
    eval_symbolic,
    add_lookup_columns_symbolic,
    get_lookups_symbolic
);

#[cfg(debug_assertions)]
impl_air_for_dynamic_entry!(
    'a,
    DebugConstraintBuilder<'a, Val<SC>, SC::Challenge>,
    eval_debug,
);

impl_air_for_dynamic_entry!(
    'a,
    ProverConstraintFolderWithLookups<'a, SC>,
    eval_prover,
);

impl_air_for_dynamic_entry!(
    'a,
    VerifierConstraintFolderWithLookups<'a, SC>,
    eval_verifier,
);

/// Simple super trait of [`Air`] describing the behaviour of a non-primitive
/// dynamically dispatched AIR used in batched proofs.
#[cfg(debug_assertions)]
pub trait BatchAir<SC>:
    BaseAir<Val<SC>>
    + Air<SymbolicAirBuilder<Val<SC>, SC::Challenge>>
    + for<'a> Air<DebugConstraintBuilder<'a, Val<SC>, SC::Challenge>>
    + for<'a> Air<ProverConstraintFolderWithLookups<'a, SC>>
    + for<'a> Air<VerifierConstraintFolderWithLookups<'a, SC>>
    + Send
    + Sync
where
    SC: StarkGenericConfig,
    SymbolicExpressionExt<Val<SC>, SC::Challenge>: Algebra<SymbolicExpression<Val<SC>>>,
{
}

/// Simple super trait of [`Air`] describing the behaviour of a non-primitive
/// dynamically dispatched AIR used in batched proofs.
#[cfg(not(debug_assertions))]
pub trait BatchAir<SC>:
    BaseAir<Val<SC>>
    + Air<SymbolicAirBuilder<Val<SC>, SC::Challenge>>
    + for<'a> Air<ProverConstraintFolderWithLookups<'a, SC>>
    + for<'a> Air<VerifierConstraintFolderWithLookups<'a, SC>>
    + Send
    + Sync
where
    SC: StarkGenericConfig,
    SymbolicExpressionExt<Val<SC>, SC::Challenge>: Algebra<SymbolicExpression<Val<SC>>>,
{
}

macro_rules! impl_cloneable_batch_air_forwarding {
    (
        $(#[$cfg:meta])?
        $lt:lifetime,
        $builder:ty,
        $eval_method:ident,
        $add_lookup_method:ident,
        $get_lookup_method:ident
    ) => {
        $(#[$cfg])?
        fn $eval_method<$lt>(&self, builder: &mut $builder) {
            <T as Air<$builder>>::eval(self, builder);
        }

        $(#[$cfg])?
        fn $add_lookup_method<$lt>(&mut self) -> Vec<usize> {
            LookupAir::add_lookup_columns(self)
        }

        $(#[$cfg])?
        fn $get_lookup_method<$lt>(&mut self) -> Vec<Lookup<Val<SC>>> {
            LookupAir::get_lookups(self)
        }
    };
    (
        $(#[$cfg:meta])?
        $builder:ty,
        $eval_method:ident,
        $add_lookup_method:ident,
        $get_lookup_method:ident
    ) => {
        $(#[$cfg])?
        fn $eval_method(&self, builder: &mut $builder) {
            <T as Air<$builder>>::eval(self, builder);
        }

        $(#[$cfg])?
        fn $add_lookup_method(&mut self) -> Vec<usize> {
            LookupAir::add_lookup_columns(self)
        }

        $(#[$cfg])?
        fn $get_lookup_method(&mut self) -> Vec<Lookup<Val<SC>>> {
            LookupAir::get_lookups(self)
        }
    };
}

/// Object-safe extension of [`BatchAir`] that adds cloning support.
///
/// This trait is automatically implemented for any `T: BatchAir<SC> + Clone + 'static`.
/// It is the concrete trait object type stored inside [`DynamicAirEntry`].
pub trait CloneableBatchAir<SC>: BaseAir<Val<SC>> + Send + Sync
where
    SC: StarkGenericConfig,
    SymbolicExpressionExt<Val<SC>, SC::Challenge>: Algebra<SymbolicExpression<Val<SC>>>,
{
    fn clone_box(&self) -> Box<dyn CloneableBatchAir<SC>>;

    #[cfg(debug_assertions)]
    fn eval_debug<'a>(&self, builder: &mut DebugConstraintBuilder<'a, Val<SC>, SC::Challenge>);
    fn eval_symbolic(&self, builder: &mut SymbolicAirBuilder<Val<SC>, SC::Challenge>);
    fn eval_prover<'a>(&self, builder: &mut ProverConstraintFolderWithLookups<'a, SC>);
    fn eval_verifier<'a>(&self, builder: &mut VerifierConstraintFolderWithLookups<'a, SC>);

    #[cfg(debug_assertions)]
    fn add_lookup_columns_debug(&mut self) -> Vec<usize>;
    fn add_lookup_columns_symbolic(&mut self) -> Vec<usize>;
    fn add_lookup_columns_prover(&mut self) -> Vec<usize>;
    fn add_lookup_columns_verifier(&mut self) -> Vec<usize>;

    #[cfg(debug_assertions)]
    fn get_lookups_debug(&mut self) -> Vec<Lookup<Val<SC>>>;
    fn get_lookups_symbolic(&mut self) -> Vec<Lookup<Val<SC>>>;
    fn get_lookups_prover(&mut self) -> Vec<Lookup<Val<SC>>>;
    fn get_lookups_verifier(&mut self) -> Vec<Lookup<Val<SC>>>;
}

impl<SC, T> CloneableBatchAir<SC> for T
where
    SC: StarkGenericConfig,
    T: BatchAir<SC> + LookupAir<Val<SC>> + Clone + 'static,
    SymbolicExpressionExt<Val<SC>, SC::Challenge>:
        Algebra<SymbolicExpression<Val<SC>>> + Algebra<SC::Challenge>,
{
    fn clone_box(&self) -> Box<dyn CloneableBatchAir<SC>> {
        Box::new(self.clone())
    }

    impl_cloneable_batch_air_forwarding!(
        SymbolicAirBuilder<Val<SC>, SC::Challenge>,
        eval_symbolic,
        add_lookup_columns_symbolic,
        get_lookups_symbolic
    );

    #[cfg(debug_assertions)]
    impl_cloneable_batch_air_forwarding!(
        'a,
        DebugConstraintBuilder<'a, Val<SC>, SC::Challenge>,
        eval_debug,
        add_lookup_columns_debug,
        get_lookups_debug
    );

    impl_cloneable_batch_air_forwarding!(
        'a,
        ProverConstraintFolderWithLookups<'a, SC>,
        eval_prover,
        add_lookup_columns_prover,
        get_lookups_prover
    );

    impl_cloneable_batch_air_forwarding!(
        'a,
        VerifierConstraintFolderWithLookups<'a, SC>,
        eval_verifier,
        add_lookup_columns_verifier,
        get_lookups_verifier
    );
}

/// Data needed to insert a dynamic table instance into the batched prover.
///
/// A `BatchTableInstance` bundles everything the batch prover needs from a
/// non-primitive table plugin: the AIR, its populated trace matrix, any
/// public values it exposes, and the number of rows it produces.
pub struct BatchTableInstance<SC>
where
    SC: StarkGenericConfig,
{
    /// Operation type (it should match `TableProver::op_type`).
    pub op_type: NpoTypeId,
    /// The AIR implementation for this table.
    pub air: DynamicAirEntry<SC>,
    /// The populated trace matrix for this table.
    pub trace: RowMajorMatrix<Val<SC>>,
    /// Public values exposed by this table.
    pub public_values: Vec<Val<SC>>,
    /// Number of logical operations (before lane packing) produced for this table.
    pub rows: usize,
    /// Number of operations packed per AIR row (lane count).
    pub lanes: usize,
}

#[inline(always)]
/// # Safety
///
/// Caller must ensure that both `Traces<FromEF>` and `Traces<ToEF>` share an
/// identical in-memory representation.
pub(crate) unsafe fn transmute_traces<FromEF, ToEF>(t: &Traces<FromEF>) -> &Traces<ToEF> {
    debug_assert_eq!(
        core::mem::size_of::<Traces<FromEF>>(),
        core::mem::size_of::<Traces<ToEF>>()
    );
    debug_assert_eq!(
        core::mem::align_of::<Traces<FromEF>>(),
        core::mem::align_of::<Traces<ToEF>>()
    );

    unsafe { &*(t as *const _ as *const Traces<ToEF>) }
}

/// Trait implemented by all non-primitive table plugins used by the batch prover.
///
/// Implementors would typically delegate to an existing AIR type, define a base case
/// for base-field traces, and then use the [`impl_table_prover_batch_instances_from_base!`]
/// macro to generate the degree-specific implementations.
pub trait TableProver<SC>: Send + Sync
where
    SC: StarkGenericConfig + 'static,
{
    /// Operation type for this prover.
    fn op_type(&self) -> NpoTypeId;

    /// Number of operations packed into a single AIR row.
    ///
    /// Defaults to 1. Override to pack multiple operations per row, reducing the trace
    /// height by a factor of `lanes`. The AIR and its lookups must be designed accordingly.
    fn lanes(&self) -> usize {
        1
    }

    /// Produce a batched table instance for base-field traces.
    fn batch_instance_d1(
        &self,
        config: &SC,
        packing: &TablePacking,
        traces: &Traces<Val<SC>>,
    ) -> Option<BatchTableInstance<SC>>;

    /// Produce a batched table instance for degree-2 extension traces.
    fn batch_instance_d2(
        &self,
        config: &SC,
        packing: &TablePacking,
        traces: &Traces<BinomialExtensionField<Val<SC>, 2>>,
    ) -> Option<BatchTableInstance<SC>>;

    /// Produce a batched table instance for degree-4 extension traces.
    fn batch_instance_d4(
        &self,
        config: &SC,
        packing: &TablePacking,
        traces: &Traces<BinomialExtensionField<Val<SC>, 4>>,
    ) -> Option<BatchTableInstance<SC>>;

    /// Produce a batched table instance for degree-6 extension traces.
    fn batch_instance_d6(
        &self,
        config: &SC,
        packing: &TablePacking,
        traces: &Traces<BinomialExtensionField<Val<SC>, 6>>,
    ) -> Option<BatchTableInstance<SC>>;

    /// Produce a batched table instance for degree-8 extension traces.
    fn batch_instance_d8(
        &self,
        config: &SC,
        packing: &TablePacking,
        traces: &Traces<BinomialExtensionField<Val<SC>, 8>>,
    ) -> Option<BatchTableInstance<SC>>;

    /// Produce a batched table instance for degree-5 quintic trinomial extension traces.
    fn batch_instance_d5(
        &self,
        _config: &SC,
        _packing: &TablePacking,
        _traces: &Traces<QuinticTrinomialExtensionField<Val<SC>>>,
    ) -> Option<BatchTableInstance<SC>> {
        None
    }

    /// Rebuild the AIR for verification from the recorded non-primitive table entry.
    ///
    /// `circuit_extension_degree` is the circuit's extension-field dimension (e.g. 4 or 5), used
    /// by D1 Poseidon2 wrappers to select the witness-bus width (Bus1 vs Bus5).
    fn batch_air_from_table_entry(
        &self,
        config: &SC,
        degree: usize,
        circuit_extension_degree: u32,
        table_entry: &super::NonPrimitiveTableEntry<SC>,
    ) -> Result<DynamicAirEntry<SC>, String>;

    /// Build an AIR entry using committed preprocessed data.
    ///
    /// `lanes` is the lane count that was used when building the table instance
    /// (may differ from the prover's own default when overridden via `TablePacking`).
    ///
    /// Returns `None` if not supported by this table prover.
    /// This is used to override the preprocessed data regenerated from runtime ops
    /// with the data that was committed during `get_airs_and_degrees_with_prep`.
    fn air_with_committed_preprocessed(
        &self,
        committed_prep: Vec<Val<SC>>,
        min_height: usize,
        lanes: usize,
        circuit_extension_degree: u32,
    ) -> Option<DynamicAirEntry<SC>> {
        let _ = (committed_prep, min_height, lanes, circuit_extension_degree);
        None
    }
}
