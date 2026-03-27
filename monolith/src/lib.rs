//! The Monolith permutation family for ZK-friendly hashing.
//!
//! Monolith is a cryptographic permutation designed for efficient
//! evaluation both natively and inside arithmetic circuits. All
//! algebraic constraints are degree 2, making it faster to prove
//! in STARKs than Poseidon2 (degree 5-7) or Rescue (degree 5+).
//!
//! The paper defines two instantiations over different primes:
//! - **Monolith-31**: Mersenne31 (p = 2^31 - 1), state width 16 or 24
//! - **Monolith-64**: Goldilocks (p = 2^64 - 2^32 + 1), state width 8 or 12
//!
//! Reference: <https://eprint.iacr.org/2023/1025.pdf>

#![no_std]

pub mod bars;
pub mod mds;
mod monolith;
mod util;

pub use bars::MonolithBars;
pub use bars::goldilocks::MonolithBarsGoldilocks;
pub use bars::mersenne31::MonolithBarsM31;
pub use mds::goldilocks::MonolithMdsMatrixGoldilocks;
pub use mds::mersenne31::MonolithMdsMatrixMersenne31;
pub use monolith::{Monolith, MonolithGoldilocks8, MonolithGoldilocks16, MonolithMersenne31};
