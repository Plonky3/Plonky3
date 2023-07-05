//! A PCS using degree 2 tensor codes, based on BCG20 <https://eprint.iacr.org/2020/1426>.

#![no_std]

extern crate alloc;

mod reshape;
mod tensor_pcs;
mod wrapped_matrix;

pub use tensor_pcs::*;
