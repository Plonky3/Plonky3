#![doc = include_str!("../README.md")]
#![no_std]

extern crate alloc;

mod builder;
mod leaf;
mod product;
mod transcript;

pub use builder::{
    BusActivation, BusInteractionBuilder, BusInteractionRecorder, BusSymbolicBuilder, RecordToken,
    SymbolicBusInteraction,
};
pub use leaf::{BusDirection, BusLeafDeclaration, BusLeafError, BusLeaves, BusSelector};
pub use product::{
    ProductGkrError, ProductGkrLayerProof, ProductGkrOutput, ProductGkrProof, ProductGkrRootShape,
    ProductGkrShape, ProductGkrShapeError,
};

#[cfg(test)]
mod tests;
