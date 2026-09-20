#![doc = include_str!("../README.md")]
#![no_std]

extern crate alloc;

mod builder;
mod debug;
mod leaf;
mod memory;
mod plan;
mod product;
mod security;
mod transcript;

pub use builder::{
    BusActivation, BusInteractionBuilder, BusInteractionRecorder, BusSymbolicBuilder, RecordToken,
    SymbolicBusInteraction,
};
pub use debug::{
    BusDebugError, BusDebugInstance, BusDebugLimits, BusDebugLocation, BusDebugOccurrence,
    BusDebugReport, BusImbalance, BusUnmatched,
};
pub use leaf::{BusDirection, BusLeafDeclaration, BusLeafError, BusLeaves, BusSelector};
pub use memory::{
    MemoryFinalCounts, MemoryReadAddresses, MemoryReadCounts, MemoryReadValues, MemoryTableValues,
    ReadOnlyMemoryClaims, ReadOnlyMemoryColumns, ReadOnlyMemoryError,
    ReadOnlyMemoryInteractionBuilder, ReadOnlyMemoryLeaves, ReadOnlyMemoryPlan,
};
pub use plan::{
    BusBlock, BusBlockOwner, BusDomain, BusExpressionLocation, BusPlan, BusPlanError, BusPlanInput,
    BusSecurityGeometry, BusTerminalShare, BusTupleSlot, UnsupportedBusAccess,
};
pub use product::{
    ProductGkrError, ProductGkrLayerProof, ProductGkrOutput, ProductGkrProof, ProductGkrRootShape,
    ProductGkrShape, ProductGkrShapeError,
};
