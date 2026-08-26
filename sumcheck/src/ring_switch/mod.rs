//! Ring switching (IACR eprint 2024/504, Construction 3.1): a reduction from an evaluation
//! claim about a small-field multilinear to a claim about its packed extension-field
//! multilinear.

pub mod equality;
pub mod pack;
pub mod tensor;
pub mod weights;

pub use pack::{compute_s_hat, pack, packed_vars};

#[cfg(test)]
pub(crate) mod test_util;
