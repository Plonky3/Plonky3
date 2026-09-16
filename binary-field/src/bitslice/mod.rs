//! Bit-sliced `GF(2)`: one field element per bit of a machine word.
//!
//! # The two layers
//!
//! ```text
//!     underlier   a block of bits, 8 to 512 wide, with no field structure
//!     packing     that block read as one GF(2) element per bit
//! ```
//!
//! The underlier is the join point: the packing is generic over it, so a width is added by
//! naming a block rather than by writing another kernel.
//!
//! # Why bit-slicing
//!
//! A `GF(2)` element carries one bit of information and is held in a whole byte.
//!
//! Packing `W` independent elements into `W` bits recovers that factor of eight.
//!
//! It also turns the field operations into single instructions on whole registers:
//!
//! ```text
//!     a + b   ->  a XOR b
//!     a - b   ->  a XOR b        (characteristic 2, so -x = x)
//!     a * b   ->  a AND b
//!     a^2     ->  a              (squaring fixes every element of the prime field)
//! ```
//!
//! # The lane order, fixed once
//!
//! Lane `i` is bit `i mod B` of word `i / B`, where `B` is the width of the backing word,
//! counting bits from the least significant one.
//!
//! ```text
//!     lane:    0     1     2     3    ...
//!     word 0:  bit0  bit1  bit2  bit3 ...
//! ```
//!
//! Read as bytes, that puts lane `i` at bit `i mod 8` of byte `i / 8`, lowest bit first.
//!
//! This is the convention a bit witness already uses elsewhere in the proving stack, so such
//! a witness can be reinterpreted as a bit slice without moving a single bit.
//!
//! Every accessor, constructor, iterator, interleave and transpose here respects that map.
//!
//! # Why these types are not the scalar field's packing
//!
//! The packed-value contract in the field crate requires a packing to be castable to and from
//! an array of `WIDTH` scalars.
//!
//! That is what lets it hand out a scalar slice and reinterpret a scalar buffer as a buffer of
//! packings.
//!
//! A bit-sliced packing of `W` lanes occupies `W / 8` bytes while `W` scalars occupy `W`:
//!
//! ```text
//!     [Gf2; 64]        64 bytes, one element per byte
//!     PackedGf2x64      8 bytes, one element per bit
//! ```
//!
//! Reinterpreting the second as the first would read eight times past the end of the buffer,
//! so no bit-sliced type can honour that contract.
//!
//! The scalar field therefore keeps its trivial width-one packing, and these types expose the
//! same algebra through their own constructors and accessors instead.
//!
//! They implement the prime-characteristic ring trait and the algebra over `GF(2)`, so generic
//! ring code runs on them unchanged.
//!
//! # Reading a witness at another width
//!
//! A wide block is laid out as narrower blocks side by side, so a run of wide packings and a
//! run of narrow ones over the same bytes hold the same lanes in the same order:
//!
//! ```text
//!     one PackedGf2x512   ==   four PackedGf2x128   ==   sixty-four PackedGf2x8
//! ```
//!
//! Narrowing is therefore a borrow, not a conversion, and a committed bit witness can be read
//! at whatever width a consumer wants without copying it.

mod packing;
mod underlier;

pub use packing::{
    PackedGf2, PackedGf2x8, PackedGf2x16, PackedGf2x32, PackedGf2x64, PackedGf2x128, PackedGf2x256,
    PackedGf2x512,
};
pub use underlier::{Divisible, M128, M256, M512, Underlier, Word};
