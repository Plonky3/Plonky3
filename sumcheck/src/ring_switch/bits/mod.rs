//! Ring switching at a bit alphabet.
//!
//! The reduction beside this one is generic over a base field and an extension of it.
//!
//! It cannot be instantiated at `F_2`.
//!
//! The basis accessor it rests on hands out a borrowed slice.
//!
//! 128 one-byte coefficients cannot be borrowed out of a 16-byte element.
//!
//! # What changes at a bit alphabet
//!
//! The construction is the same one, but two things about it stop being incidental.
//!
//! A coefficient is one bit, so every product by a coefficient is a conditional add:
//!
//! ```text
//!     general alphabet   ->  d multiplications per point
//!     bit alphabet       ->  the set bits, added
//! ```
//!
//! And the tensor element is a `d x d` bit matrix, which is one `EF` element per row:
//!
//! ```text
//!     one byte per coefficient   ->  16 KB at d = 128
//!     one bit per coefficient    ->   2 KB
//! ```
//!
//! That matters because the tensor element crosses the wire.
//!
//! So the element is held by rows, and the two readings of the matrix are a transpose apart.
//!
//! # The pieces
//!
//! ```text
//!     Coefficients    the F_2-coordinates of one element
//!     BitPacking      a bit witness read as the multilinear a commitment holds
//!     BitTensor       an element of EF (x) EF, as a bit matrix
//!     BitRingSwitch   one reduction, and the five values it produces
//! ```
//!
//! The witness being bit-valued is what the reduction's soundness rests on.
//!
//! The packing carries that as a type rather than as a convention.
//!
//! The two points every value depends on live on the reduction.
//!
//! So the equality tables are built once, and the widths they agree on are checked once.
//!
//! # The basis
//!
//! A tower level may hold its elements in any `F_2`-basis of the field.
//!
//! This module fixes the one its byte representation already defines.
//!
//! Coefficient `j` is bit `j` of the little-endian byte string, so `beta_0 = 1`.
//!
//! Nothing here depends on which basis that is, only on both sides using the same one.
//!
//! The payoff is that packing becomes a reinterpretation rather than a computation.

pub mod basis;
pub mod packing;
pub mod reduction;
pub mod tensor;

pub use basis::Coefficients;
pub use packing::{BitPacking, BitPackingError};
pub use reduction::{BitRingSwitch, BitRingSwitchError};
pub use tensor::{BitTensor, MalformedBitTensor};
