//! Zero-knowledge codes (Section 3.2.1 of eprint 2026/391).

#![no_std]

extern crate alloc;

pub mod encoding;
pub mod reed_solomon;

pub use encoding::*;
pub use reed_solomon::*;
