#![doc = include_str!("../README.md")]
#![no_std]

extern crate alloc;

// The byte-map kernels and the register they run on.
//
// A build without the instruction never reaches them, so it compiles neither.
//
// Tests compile both on every target, against the scalar stand-in in the test module.
#[cfg(any(
    test,
    all(
        target_arch = "x86_64",
        target_feature = "gfni",
        target_feature = "avx512f",
        target_feature = "avx512bw"
    )
))]
mod affine;
mod butterfly;
mod domain;
mod encoder;
#[cfg(any(
    test,
    all(
        target_arch = "x86_64",
        target_feature = "gfni",
        target_feature = "avx512f",
        target_feature = "avx512bw"
    )
))]
mod lanes;
mod lch;
mod naive;
mod poly;
mod subfield;
#[cfg(test)]
pub(crate) mod test_util;
mod tower;
mod traits;

pub use butterfly::ButterflyField;
pub use domain::*;
pub use encoder::*;
pub use lch::*;
pub use naive::*;
pub use poly::*;
pub use subfield::*;
pub use traits::*;
