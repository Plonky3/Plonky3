#![doc = include_str!("../README.md")]
#![no_std]

extern crate alloc;

mod aes;
mod bitslice;
mod cantor;
mod challenger;
mod clmul;
mod extension;
mod gf2;
mod ghash;
mod linear;
mod packed;
mod poly192;
mod poly64;
pub mod poly_basis;
mod poly_slice;
mod tables;
mod tower;
mod transcript;

pub use aes::{ByteMatrix, LinearizedPoly8b, PackedRijndael8b, Rijndael8b};
pub use bitslice::{
    Divisible, M128, M256, M512, PackedGf2, PackedGf2x8, PackedGf2x16, PackedGf2x32, PackedGf2x64,
    PackedGf2x128, PackedGf2x256, PackedGf2x512, Underlier, Word,
};
pub use challenger::BinaryChallenger;
pub use gf2::Gf2;
pub use ghash::Ghash128;
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "vpclmulqdq",
    any(target_feature = "avx2", target_feature = "avx512f")
))]
pub use packed::*;
pub use poly64::Poly64;
pub use poly192::Poly192;
pub use tower::{
    BinaryField2, BinaryField4, BinaryField8, BinaryField16, BinaryField32, BinaryField64,
    BinaryField128, TowerLevel,
};
