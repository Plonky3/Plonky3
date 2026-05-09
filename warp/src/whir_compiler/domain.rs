//! Fiat-Shamir domain separators for the native WARP-to-WHIR root compiler.
//!
//! These are WARP integration tags, not constants from the WHIR paper. The
//! root compiler absorbs them before using a WHIR PCS object in a WARP-specific
//! role. This prevents replaying a valid WHIR proof/commitment transcript from
//! another part of the protocol where the same commitment type appears with a
//! different meaning.

/// Domain separator for base-message root-oracle commitments, `WARP_RBAS`.
pub(super) const ROOT_WHIR_BASE_ORACLE: u64 = 0x5741_5250_5242_4153;
