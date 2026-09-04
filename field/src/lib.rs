#![doc = include_str!("../README.md")]
#![no_std]

extern crate alloc;

mod array;
mod batch_inverse;
pub mod coset;
mod dup;
pub mod exponentiation;
pub mod extension;
mod field;
mod helpers;
pub mod integers;
pub mod op_assign_macros;
mod packed;
mod sqrt;
mod vectorized;

pub use array::*;
pub use batch_inverse::*;
pub use dup::Dup;
pub use field::*;
pub use helpers::*;
pub use packed::*;
// Re-exported so the macros this crate exports can name the token-pasting helper through
// `$crate`, which resolves at the definition site.
// Without it every crate expanding one of those macros would need its own dependency on it.
#[doc(hidden)]
pub use paste;
pub use sqrt::{tonelli_shanks, tonelli_shanks_two_adic};
pub use vectorized::*;
