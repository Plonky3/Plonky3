//! Multilinear SuperSpartan-flavored STARK prover for AIRs.
//!
//! # References
//!
//! - Setty, Thaler, Wahby. Customizable Constraint Systems for succinct arguments. <https://eprint.iacr.org/2023/552.pdf>
//! - Borgeaud, W. AIR-specific optimizations on top of SuperSpartan. <https://solvable.group/posts/super-air/>

#![no_std]

extern crate alloc;

pub mod commit;
pub mod config;
pub mod folder;
pub mod metadata;
pub mod opening;
pub mod rounds;
pub mod selectors;
pub mod zerocheck;
