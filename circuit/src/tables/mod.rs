//! Execution trace tables for zkVM circuit operations.

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;
use core::any::Any;
use core::fmt;

use hashbrown::HashMap;

use crate::CircuitError;
use crate::ops::{NpoTypeId, OpStateMap};
use crate::types::WitnessId;

mod alu;
mod constant;
mod public;
mod runner;
mod witness;

pub use alu::AluTrace;
pub use constant::{ConstTrace, ConstTraceBuilder};
pub use public::{PublicTrace, PublicTraceBuilder};
pub use runner::CircuitRunner;
pub use witness::WitnessTrace;

/// Trait implemented by all non-primitive operation traces.
pub trait NonPrimitiveTrace<F>: Send + Sync {
    /// Operation type for this non-primitive trace.
    fn op_type(&self) -> NpoTypeId;
    /// Number of rows produced by this trace.
    fn rows(&self) -> usize;
    /// Type-erased access for downcasting.
    fn as_any(&self) -> &dyn Any;
    /// Clone the trace into a boxed trait object.
    fn boxed_clone(&self) -> Box<dyn NonPrimitiveTrace<F>>;
}

/// Function pointer for constructing a non-primitive trace from runner state.
///
/// The trace generator receives operation execution state (recorded row data, chaining state, etc.).
pub type TraceGeneratorFn<F> =
    fn(op_states: &OpStateMap) -> Result<Option<Box<dyn NonPrimitiveTrace<F>>>, CircuitError>;

/// Execution traces for all tables.
///
/// This structure holds the complete execution trace of a circuit,
/// containing all the data needed to generate proofs.
pub struct Traces<F> {
    /// Central witness table (bus) storing all intermediate values.
    pub witness_trace: WitnessTrace<F>,
    /// Constant table for compile-time known values.
    pub const_trace: ConstTrace<F>,
    /// Public input table for externally provided values.
    pub public_trace: PublicTrace<F>,
    /// Unified ALU operation table (Add, Mul, BoolCheck, MulAdd).
    pub alu_trace: AluTrace<F>,
    /// Dynamically registered non-primitive traces indexed by operation type.
    pub non_primitive_traces: HashMap<NpoTypeId, Box<dyn NonPrimitiveTrace<F>>>,
    /// Tag to witness index mapping for probing values by name.
    pub tag_to_witness: HashMap<String, WitnessId>,
}

impl<F> Traces<F> {
    /// Fetch a non-primitive trace by identifier and downcast to a concrete type.
    pub fn non_primitive_trace<T>(&self, op_type: &NpoTypeId) -> Option<&T>
    where
        T: NonPrimitiveTrace<F> + 'static,
    {
        self.non_primitive_traces
            .get(op_type)
            .and_then(|trace| trace.as_any().downcast_ref::<T>())
    }

    /// Probes the value of a tagged wire.
    ///
    /// Returns `None` if the tag was not registered during circuit construction.
    ///
    /// # Example
    /// ```ignore
    /// let value = traces.probe("my-tag").expect("tag should exist");
    /// ```
    pub fn probe(&self, tag: &str) -> Option<&F> {
        let witness_id = self.tag_to_witness.get(tag)?;
        self.witness_trace.get_value(*witness_id)
    }
}

impl<F: PartialEq> PartialEq for Traces<F> {
    fn eq(&self, other: &Self) -> bool {
        self.witness_trace == other.witness_trace
            && self.const_trace == other.const_trace
            && self.public_trace == other.public_trace
            && self.alu_trace == other.alu_trace
            && self.tag_to_witness == other.tag_to_witness
            && self.non_primitive_traces.len() == other.non_primitive_traces.len()
            && self
                .non_primitive_traces
                .keys()
                .all(|k| other.non_primitive_traces.contains_key(k))
    }
}

impl<F: Clone> Clone for Traces<F> {
    fn clone(&self) -> Self {
        Self {
            witness_trace: self.witness_trace.clone(),
            const_trace: self.const_trace.clone(),
            public_trace: self.public_trace.clone(),
            alu_trace: self.alu_trace.clone(),
            non_primitive_traces: self
                .non_primitive_traces
                .iter()
                .map(|(op_type, trace)| (op_type.clone(), trace.boxed_clone()))
                .collect(),
            tag_to_witness: self.tag_to_witness.clone(),
        }
    }
}

impl<F> fmt::Debug for Traces<F>
where
    WitnessTrace<F>: fmt::Debug,
    ConstTrace<F>: fmt::Debug,
    PublicTrace<F>: fmt::Debug,
    AluTrace<F>: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let extra_summary: Vec<_> = self
            .non_primitive_traces
            .iter()
            .map(|(op_type, trace)| (op_type.clone(), trace.rows()))
            .collect();
        f.debug_struct("Traces")
            .field("witness_trace", &self.witness_trace)
            .field("const_trace", &self.const_trace)
            .field("public_trace", &self.public_trace)
            .field("alu_trace", &self.alu_trace)
            .field("non_primitive_traces", &extra_summary)
            .finish()
    }
}
