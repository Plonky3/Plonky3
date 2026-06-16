//! Symbolic claim tracking for the HVZK-WHIR pipeline.
//!
//! The committed-sumcheck relation (Definition 5.8 of eprint 2026/391):
//!
//! ```text
//!     <f, W> + sum_i <xi_i, u_i> = target
//!
//!     <f, W>        ->  linear claim on the current source message
//!     <xi_i, u_i>   ->  linear claim on carried mask oracle i
//! ```
//!
//! - The source covector `W` shrinks with every fold.
//!   The verifier tracks it as a list of symbolic terms.
//!   It materializes only at the base case, on a small message.
//! - Mask covectors `u_i` are dense from birth.
//!   Mask messages have size `O~(lambda)`, so density is free.

mod masks;
mod source;

pub use masks::MaskClaims;
pub use source::SourceClaim;
