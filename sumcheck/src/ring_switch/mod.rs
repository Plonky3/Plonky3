//! Ring switching (IACR eprint 2024/504, Construction 3.1): a reduction from an evaluation
//! claim about a small-field multilinear to a claim about its packed extension-field
//! multilinear.

pub mod tensor;

#[cfg(test)]
pub(crate) mod test_util;
