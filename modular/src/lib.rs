//! Rings `Z/M` with a modulus chosen at run time: Montgomery prime fields, `Z/2^k`, and compile-time prime fields behind one interface.
//!
//! Also primality tests (trial division, Miller-Rabin, strong Lucas, BPSW) and uniform prime sampling from a byte stream.

#![no_std]

extern crate alloc;
#[cfg(test)]
extern crate std;

mod error;
mod mont;
mod power_of_two;
pub mod prime;
mod ring;
mod static_field;
mod uint;

#[cfg(test)]
mod tests;

pub use error::{ModulusError, PrimeError};
pub use mont::{CanonicalResidue, MontField, MontModulus, Residue};
pub use power_of_two::PowerOfTwo;
pub use prime::{PrimePolicy, PrimeSample, sample_prime};
pub use ring::{ModField, ModRing, RingId, RingKind};
pub use static_field::{StaticBigPrime, StaticPrime};
pub use uint::Uint;
