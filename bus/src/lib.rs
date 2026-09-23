#![doc = include_str!("../README.md")]
#![no_std]

extern crate alloc;

#[cfg(test)]
extern crate std;

mod argument;
mod builder;
#[cfg(feature = "diagnostics")]
mod debug;
mod direction;
mod evaluation;
mod memory;
mod name;
mod plan;
mod product;
mod ram;
mod security;

pub use argument::{BusArgumentError, BusChallenges, BusProof, BusReductionOutput};
pub use builder::{
    BusActivation, BusBoundary, BusInteractionBuilder, BusInteractionRecorder, BusSymbolicBuilder,
    RecordToken, SymbolicBusInteraction,
};
#[cfg(feature = "diagnostics")]
pub use debug::{
    BusDebugError, BusDebugInstance, BusDebugLimits, BusDebugLocation, BusDebugOccurrence,
    BusDebugReport, BusImbalance, BusUnmatched,
};
pub use direction::BusDirection;
pub use evaluation::{BusEvaluation, BusEvaluationError, BusFactorPlan};
pub use memory::{
    MemoryFinalCounts, MemoryReadAddresses, MemoryReadCounts, MemoryReadValues, MemoryTableValues,
    ReadOnlyMemoryBus, ReadOnlyMemoryChallenges, ReadOnlyMemoryClaims, ReadOnlyMemoryColumns,
    ReadOnlyMemoryError, ReadOnlyMemoryInteractionBuilder, ReadOnlyMemoryPlan,
};
pub use name::{BusName, BusNameError};
pub use plan::{
    BusBlock, BusBlockOwner, BusDomain, BusExpressionLocation, BusPlan, BusPlanError, BusPlanInput,
    BusSecurityGeometry, BusTerminalShare, BusTupleSlot, UnsupportedBusAccess,
};
pub use product::{
    ProductGkrError, ProductGkrLayerProof, ProductGkrOutput, ProductGkrProof, ProductGkrRootShape,
    ProductGkrShape, ProductGkrShapeError,
};
pub use ram::{
    MAX_RAM_BIT_WIDTH, MIN_RAM_ACCESS_COUNT, RamAccess, RamAir, RamBoundary, RamError, RamLayout,
    RamStatement, RamTrace,
};
