//! XHash hash function family.
//!
//! XHash interleaves Rescue-Prime Optimized rounds with an extension-field
//! S-box round per
//! [eprint 2024/1635](https://eprint.iacr.org/2024/1635) (XHash-M31)
//! and the Rescue-Prime eXtension construction of
//! [eprint 2023/1045](https://eprint.iacr.org/2023/1045) (Goldilocks).
//!
//! Each round consists of three sub-rounds:
//!   - **F** (forward): `MDS → +ARK → x^α`
//!   - **B** (backward): `MDS → +ARK → x^(1/α)`
//!   - **E** (extension): `+ARK → x^β` where the state is interpreted as
//!     blocks of an extension-field element and the S-box is applied in that
//!     extension. No MDS precedes the extension step.
//!
//! After all `num_rounds` rounds, a concluding linear layer (`MDS → +ARK`)
//! is applied.

#![no_std]

extern crate alloc;

mod baby_bear;
mod goldilocks;
mod koala_bear;
mod mersenne_31;
mod util;
mod xhash;

pub use baby_bear::*;
pub use goldilocks::*;
pub use koala_bear::*;
pub use mersenne_31::*;
