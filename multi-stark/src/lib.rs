//! Multilinear SuperSpartan-flavored STARK prover for AIRs.
//!
//! # Overview
//!
//! - Proves satisfiability of any `Air` from `p3-air`.
//! - Uses a multilinear polynomial IOP based on SuperSpartan.
//! - Adopts the practical refinements from wborgeaud's "AIR as MCCCS" blog.
//!
//! # Protocol shape
//!
//! - One multilinear per trace column committed via a multilinear PCS.
//! - One outer sumcheck binds an `eq`-weighted constraint polynomial.
//! - One shift-reduction sumcheck collapses next-row claims to column claims.
//! - PCS openings finish the proof at the resulting challenge points.
//!
//! # References
//!
//! - Setty, Thaler, Wahby. Customizable Constraint Systems for succinct arguments. <https://eprint.iacr.org/2023/552.pdf>
//! - Borgeaud, W. AIR-specific optimizations on top of SuperSpartan. <https://solvable.group/posts/super-air/>

#![no_std]

extern crate alloc;

pub mod selectors;
