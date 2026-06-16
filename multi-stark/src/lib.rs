//! Multilinear SuperSpartan-flavored STARK prover for AIRs.
//!
//! # References
//!
//! - Setty, Thaler, Wahby. Customizable Constraint Systems for succinct arguments. <https://eprint.iacr.org/2023/552.pdf>
//! - Borgeaud, W. AIR-specific optimizations on top of SuperSpartan. <https://solvable.group/posts/super-air/>

#![no_std]

#[cfg(test)]
extern crate alloc;

pub mod folder;
pub mod selectors;
