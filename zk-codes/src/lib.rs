#![no_std]

extern crate alloc;

pub mod encoding;
pub mod reed_solomon;

pub use encoding::*;
pub use reed_solomon::*;
