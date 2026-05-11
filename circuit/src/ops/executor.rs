use alloc::boxed::Box;
use alloc::vec::Vec;
use core::any::Any;
use core::fmt::Debug;

use p3_field::Field;

use super::context::ExecutionContext;
use super::npo::NpoTypeId;
use crate::CircuitError;
use crate::types::WitnessId;

/// Object-safe trait exposing the subset of [`PreprocessedColumns`](crate::PreprocessedColumns)
/// methods that [`NonPrimitiveExecutor::preprocess`] implementations need.
///
/// This allows `PreprocessedColumns<F, const D: usize>` to be passed through `dyn` dispatch
/// without requiring executors to know `D` at compile time.
pub trait PreprocessedWriter<F: Field> {
    /// Returns the D-scaled base-field index for a given witness ID as a field element.
    ///
    /// `WitnessId(n)` maps to base-field index `n * D`.
    fn witness_index_as_field(&self, wid: WitnessId) -> F;

    /// Increments the ext-field read count for each of the given witness indices.
    fn increment_ext_reads(&mut self, wids: &[WitnessId]);

    /// Extends the preprocessed data of `op_type`'s non-primitive operation
    /// with `wids`'s witness indices (D-scaled). Does NOT increment ext-field read counts.
    ///
    /// Use this for non-primitive OUTPUTS: the table creates these witnesses on the
    /// `WitnessChecks` bus, so they are not readers. The `out_ctl` multiplicity is
    /// set separately by `get_airs_and_degrees_with_prep` based on `ext_reads`.
    fn register_non_primitive_output_index(&mut self, op_type: &NpoTypeId, wids: &[WitnessId]);

    /// Extends the preprocessed data of `op_type`'s non-primitive operation
    /// with `wids`'s witness indices (D-scaled), and increments their ext-field read counts.
    ///
    /// Use this for non-primitive inputs that the table reads from the `WitnessChecks` bus.
    fn register_non_primitive_witness_reads(
        &mut self,
        op_type: &NpoTypeId,
        wids: &[WitnessId],
    ) -> Result<(), CircuitError>;

    /// Extends the preprocessed data of `op_type`'s non-primitive operation with `values`.
    /// Does not update read counts.
    fn register_non_primitive_preprocessed_no_read(&mut self, op_type: &NpoTypeId, values: &[F]);

    /// Witness-table extension packing width `D` for this circuit (matches [`PreprocessedColumns<F,
    /// D>`](crate::PreprocessedColumns)).
    ///
    /// Used by Poseidon2 D=1 preprocessing to align compact preprocessed rows with the chosen
    /// witness-bus Poseidon2 AIR (Bus1 / Bus5).
    fn witness_extension_degree_slots(&self) -> usize {
        1
    }
}

/// Trait for operation-specific execution state.
///
/// Each non-primitive operation type can define its own state struct that persists
/// across invocations during circuit execution. This enables features like:
/// - Permutation chaining (storing previous output for next input)
/// - Recording execution data for canonical trace generation
///
/// Automatically implemented for any type that is `Any + Send + Sync + Debug`.
pub trait OpExecutionState: Any + Send + Sync + Debug {}

impl<T: Any + Send + Sync + Debug> OpExecutionState for T {}

impl dyn OpExecutionState {
    /// Downcast to a concrete type by reference.
    pub fn downcast_ref<T: 'static>(&self) -> Option<&T> {
        let any: &dyn Any = self;
        any.downcast_ref()
    }

    /// Downcast to a concrete type by mutable reference.
    pub fn downcast_mut<T: 'static>(&mut self) -> Option<&mut T> {
        let any: &mut dyn Any = self;
        any.downcast_mut()
    }
}

/// Trait for executable non-primitive operations
///
/// This trait enables dynamic dispatch and allows each operation to control
/// its own execution logic with full access to the execution context.
pub trait NonPrimitiveExecutor<F: Field>: Debug {
    /// Execute the operation with full context access
    ///
    /// # Arguments
    /// * `inputs` - Input witness indices
    /// * `outputs` - Output witness indices
    /// * `ctx` - Execution context with access to witness table, private data, and configs
    fn execute(
        &self,
        inputs: &[Vec<WitnessId>],
        outputs: &[Vec<WitnessId>],
        ctx: &mut ExecutionContext<'_, F>,
    ) -> Result<(), CircuitError>;

    /// Get operation type identifier (for config lookup, error reporting)
    fn op_type(&self) -> &NpoTypeId;

    /// Allow downcasting to concrete executor types
    fn as_any(&self) -> &dyn Any;

    /// Update the preprocessed values related to this operation. This consists of:
    /// - the preprocessed values for the associated table
    /// - the multiplicity for the `Witness` table.
    ///
    /// Uses the [`PreprocessedWriter`] API to ensure witness multiplicities are updated
    /// consistently when reading from the witness table. Duplicate-output detection
    /// (which outputs were already defined by an earlier op) is handled generically
    /// by `generate_preprocessed_columns` after this method returns.
    fn preprocess(
        &self,
        _inputs: &[Vec<WitnessId>],
        _outputs: &[Vec<WitnessId>],
        _preprocessed: &mut dyn PreprocessedWriter<F>,
    ) -> Result<(), CircuitError> {
        Ok(())
    }

    /// How many leading output groups are exposed as creators on the witness-checks bus.
    /// `generate_preprocessed_columns` only performs duplicate-output detection on the
    /// first `n` groups returned here.
    ///
    /// Override when some outputs are private (e.g. capacity elements in a permutation).
    /// Default: all groups.
    fn num_exposed_outputs(&self) -> Option<usize> {
        None
    }

    /// Clone as trait object
    fn boxed(&self) -> Box<dyn NonPrimitiveExecutor<F>>;
}

impl<F: Field> Clone for Box<dyn NonPrimitiveExecutor<F>> {
    fn clone(&self) -> Self {
        self.boxed()
    }
}

/// Trait for executable hint operations.
///
/// Hints are non-deterministic witness assignments that do not have associated AIR tables
/// or traces. They operate directly on the witness array.
pub trait HintExecutor<F: Field>: Debug {
    /// Execute the hint.
    ///
    /// - `inputs`: Witness IDs to read from
    /// - `outputs`: Witness IDs to write to
    /// - `witness`: Mutable reference to the witness table
    fn execute(
        &self,
        inputs: &[WitnessId],
        outputs: &[WitnessId],
        witness: &mut [Option<F>],
    ) -> Result<(), CircuitError>;

    /// Clone as trait object.
    fn boxed(&self) -> Box<dyn HintExecutor<F>>;
}

impl<F: Field> Clone for Box<dyn HintExecutor<F>> {
    fn clone(&self) -> Self {
        self.boxed()
    }
}
