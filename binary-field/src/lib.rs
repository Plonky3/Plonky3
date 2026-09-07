#![doc = include_str!("../README.md")]
#![no_std]

extern crate alloc;

mod cantor;
mod challenger;
mod clmul;
mod extension;
mod gf2;
mod ghash;
mod packed;
pub mod poly_basis;
mod tables;
mod tower;

pub use challenger::BinaryChallenger;
pub use gf2::Gf2;
pub use ghash::Ghash128;
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "vpclmulqdq",
    any(target_feature = "avx2", target_feature = "avx512f")
))]
pub use packed::*;
pub use tower::{
    BinaryField2, BinaryField4, BinaryField8, BinaryField16, BinaryField32, BinaryField64,
    BinaryField128, TowerLevel,
};
