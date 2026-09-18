#![doc = include_str!("../README.md")]
#![no_std]

extern crate alloc;

mod leaf;
mod product;
mod transcript;

pub use leaf::{
    BusDirection, BusLeafDeclaration, BusLeafError, BusLeaves, BusSelector, materialize_bus_leaves,
};
pub use product::{
    ProductGkrError, ProductGkrLayerProof, ProductGkrOutput, ProductGkrProof, ProductGkrRootShape,
    ProductGkrShape, ProductGkrShapeError, prove_product_gkr, verify_product_gkr,
};

#[cfg(test)]
mod tests;
