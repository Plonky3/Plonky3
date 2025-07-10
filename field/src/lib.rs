//! A framework for finite fields.

#![no_std]
#![cfg_attr(
    all(
        feature = "nightly-features",
        target_arch = "x86_64",
        target_feature = "avx512f"
    ),
    feature(stdarch_x86_avx512)
)]

extern crate alloc;

mod array;
mod batch_inverse;
pub mod coset;
pub mod exponentiation;
pub mod extension;
mod field;
mod helpers;
pub mod integers;
mod interleaves;
pub mod op_assign_macros;
mod packed;

pub use array::*;
pub use batch_inverse::*;
pub use field::*;
pub use helpers::*;
pub use interleaves::*;
pub use packed::*;
