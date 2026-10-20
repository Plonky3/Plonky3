//! An AIR for the Rescue permutation.
//!
//! One permutation per row. Each Rescue round is
//! `x -> x^ALPHA`, MDS, constants, `x -> x^(1/ALPHA)`, MDS, constants,
//! matching [`p3_rescue::Rescue`].
//!
//! The inverse S-box is what makes Rescue cheap to constrain: rather than
//! expressing `x^(1/ALPHA)` as a polynomial, the trace commits to its output
//! `y` and the AIR asserts `y^ALPHA = x`. Both halves of a round therefore
//! cost one degree-`ALPHA` constraint per state element.

#![no_std]

extern crate alloc;

mod air;
mod columns;
mod generation;

pub use air::*;
pub use columns::*;
pub use generation::*;
