#![doc = include_str!("../README.md")]
#![no_std]

extern crate alloc;

mod leaf;
mod product;
mod transcript;

pub use leaf::{BusDirection, BusLeafDeclaration, BusLeafError, BusLeaves, BusSelector};
pub use product::{
    ProductGkrError, ProductGkrLayerProof, ProductGkrOutput, ProductGkrProof, ProductGkrRootShape,
    ProductGkrShape, ProductGkrShapeError,
};

#[cfg(test)]
mod tests;
