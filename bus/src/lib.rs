#![doc = include_str!("../README.md")]
#![no_std]

extern crate alloc;

mod builder;
mod leaf;
mod plan;
mod product;
mod security;
mod transcript;

pub use builder::{
    BusActivation, BusInteractionBuilder, BusInteractionRecorder, BusSymbolicBuilder, RecordToken,
    SymbolicBusInteraction,
};
pub use leaf::{BusDirection, BusLeafDeclaration, BusLeafError, BusLeaves, BusSelector};
pub use plan::{
    BusBlock, BusBlockOwner, BusDomain, BusExpressionLocation, BusPlan, BusPlanError, BusPlanInput,
    BusSecurityGeometry, BusTerminalShare, BusTupleSlot, UnsupportedBusAccess,
};
pub use product::{
    ProductGkrError, ProductGkrLayerProof, ProductGkrOutput, ProductGkrProof, ProductGkrRootShape,
    ProductGkrShape, ProductGkrShapeError,
};
