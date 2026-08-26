#![doc = include_str!("../README.md")]
#![no_std]

extern crate alloc;

mod challenger;
mod clmul;
mod extension;
mod gf2;
mod tower;

pub use challenger::BinaryChallenger;
pub use gf2::Gf2;
pub use tower::{
    BinaryField2, BinaryField4, BinaryField8, BinaryField16, BinaryField32, BinaryField64,
    BinaryField128,
};
