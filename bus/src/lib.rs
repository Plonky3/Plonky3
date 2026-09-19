#![doc = include_str!("../README.md")]
#![no_std]

extern crate alloc;

mod argument;
mod builder;
mod debug;
mod evaluation;
mod leaf;
mod memory;
mod multilinear;
mod plan;
mod product;
mod security;

pub use argument::{BusArgumentError, BusChallenges, BusProof, BusReductionOutput};
pub use builder::{
    BusActivation, BusInteractionBuilder, BusInteractionRecorder, BusSymbolicBuilder, RecordToken,
    SymbolicBusInteraction,
};
pub use debug::{
    BusDebugError, BusDebugInstance, BusDebugLimits, BusDebugLocation, BusDebugOccurrence,
    BusDebugReport, BusImbalance, BusUnmatched,
};
pub use evaluation::{BusEvaluation, BusEvaluationError};
pub use leaf::{BusDirection, BusLeafDeclaration, BusLeafError, BusLeaves, BusSelector};
pub use memory::{
    MemoryFinalCounts, MemoryReadAddresses, MemoryReadCounts, MemoryReadValues, MemoryTableValues,
    ReadOnlyMemoryBus, ReadOnlyMemoryChallenges, ReadOnlyMemoryClaims, ReadOnlyMemoryColumns,
    ReadOnlyMemoryError, ReadOnlyMemoryInteractionBuilder, ReadOnlyMemoryPlan,
};
pub use plan::{
    BusBlock, BusBlockOwner, BusDomain, BusExpressionLocation, BusPlan, BusPlanError, BusPlanInput,
    BusSecurityGeometry, BusTerminalShare, BusTupleSlot, UnsupportedBusAccess,
};
pub use product::{
    ProductGkrError, ProductGkrLayerProof, ProductGkrOutput, ProductGkrProof, ProductGkrRootShape,
    ProductGkrShape, ProductGkrShapeError,
};
