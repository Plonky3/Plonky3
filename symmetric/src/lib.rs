//! A framework for symmetric cryptography primitives.

#![no_std]

extern crate alloc;

mod compression;
mod hasher;
mod permutation;
mod serializing_hasher;
mod sponge;

pub use compression::*;
pub use hasher::*;
pub use permutation::*;
pub use serializing_hasher::*;
pub use sponge::*;
