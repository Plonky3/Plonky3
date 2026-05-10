//! Fiat-Shamir domain separators for WARP's WHIR-facing layers.
//!
//! These constants are WARP integration tags, not constants from the WHIR
//! paper or the original `p3-whir` implementation. They separate protocol
//! roles that reuse the same WHIR PCS and commitment types:
//!
//! - the final accumulator opening `f_hat(alpha) = mu`,
//! - the final Boolean PESAT opening.
//!
//! The numeric values are ASCII mnemonics packed into `u64`s. They only need
//! to be stable, distinct, and absorbed before role-specific challenges are
//! sampled.

/// Domain separator for final accumulator openings, `WARP_OPEN`.
pub(super) const WHIR_WARP_OPENING: u64 = 0x5741_5250_4f50_454e;

/// Domain separator for final Boolean PESAT openings, `WARP_PESA`.
pub(super) const WHIR_WARP_PESAT: u64 = 0x5741_5250_5045_5341;
