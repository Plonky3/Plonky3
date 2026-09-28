#![doc = include_str!("../README.md")]
#![no_std]

extern crate alloc;

// The byte-map kernels and the register they run on.
//
// A build without the instruction never reaches them, so it compiles none of them.
//
// Tests compile them on every target, against the scalar stand-in in the test module.
//
// Whatever needs an instruction the host may not have is gated again inside its own module.
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
mod interleaved;
#[cfg(any(
    test,
    all(
        target_arch = "x86_64",
        target_feature = "gfni",
        target_feature = "avx512f",
        target_feature = "avx512bw"
    ),
    all(
        target_arch = "aarch64",
        target_feature = "neon",
        target_endian = "little"
    )
))]
mod lanes;
mod lch;
mod naive;
#[cfg(any(
    test,
    all(
        target_arch = "aarch64",
        target_feature = "neon",
        target_endian = "little"
    )
))]
mod neon;
mod poly;
mod staging;
mod subfield;
#[cfg(test)]
pub(crate) mod test_util;
mod tower;
mod traits;

pub use butterfly::ButterflyField;
pub use domain::*;
pub use encoder::*;
pub use interleaved::*;
pub use lch::*;
pub use naive::*;
pub use poly::*;
pub use subfield::*;
pub use traits::*;
