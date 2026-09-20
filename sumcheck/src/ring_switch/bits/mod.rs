//! Ring switching at a bit alphabet.
//!
//! The reduction beside this one is generic over a field and an extension.
//! It cannot be instantiated at `F_2`, because its basis accessor borrows.
//! A borrowed slice cannot hold 128 coefficients of one bit each.
//!
//! # What changes at a bit alphabet
//!
//! Two things about the same construction stop being incidental.
//!
//! A coefficient is one bit, so a product by one is a conditional add:
//!
//! ```text
//!     general alphabet   ->  d multiplications per point
//!     bit alphabet       ->  the set bits, added
//! ```
//!
//! And the tensor element is a `d x d` bit matrix, one `EF` element per row:
//!
//! ```text
//!     one byte per coefficient   ->  16 KB at d = 128
//!     one bit per coefficient    ->   2 KB
//! ```
//!
//! That matters because the element crosses the wire, so it is held by rows.
//! The two readings of the matrix are then a transpose apart.
//!
//! # The pieces
//!
//! ```text
//!     Coefficients    the F_2-coordinates of one element
//!     BitPacking      a bit witness read as the multilinear a commitment holds
//!     BitTensor       an element of EF (x) EF, as a bit matrix
//!     BitRingSwitch   one reduction, before the batching draw
//!     BitRingSwitchBatch  the same reduction, after it
//!     SuccessorTensors    the two elements a successor view adds past one element's rows
//! ```
//!
//! The reduction owns the two sides that run it, as methods over its own transcript.
//! The module beside this one exposes free functions because it has no such type.
//!
//! Booleanity comes from the packing itself, not from the wrapper.
//! At a byte-aligned level a packed multilinear unpacks to one bit witness.
//!
//! The reduction is split where the protocol splits.
//! The batching challenge is drawn after the tensor element is bound.
//! A type taking it up front would invite the unsound order.
//!
//! # The basis
//!
//! A tower level may hold its elements in any `F_2`-basis.
//! This module fixes the one its byte representation already defines.
//! Coefficient `j` is bit `j` of the little-endian bytes, so `beta_0 = 1`.
//!
//! Nothing depends on which basis that is, only on both sides agreeing.
//! The payoff is that packing becomes a reinterpretation, not a computation.

pub mod basis;
mod equality;
pub mod packing;
pub mod reduction;
pub mod tensor;
pub mod transcript;

pub use basis::Coefficients;
pub use packing::{BitPacking, BitPackingError, BitPackingView};
pub use reduction::{
    BitRingSwitch, BitRingSwitchBatch, BitRingSwitchError, BitRingSwitchProof,
    BitRingSwitchProofError, SuccessorTensors,
};
pub use tensor::{BitTensor, MalformedBitTensor};
pub use transcript::{
    BitRingSwitchProverTranscript, BitRingSwitchShape, BitRingSwitchVerifierTranscript,
    TranscriptWidth,
};
