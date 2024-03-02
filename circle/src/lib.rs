// #![no_std]
#![cfg_attr(not(test), no_std)]

extern crate alloc;

mod cfft;
mod domain;
mod pcs;
mod twiddles;
mod util;

pub use cfft::*;
pub use pcs::*;

#[cfg(test)]
mod tests {
    use super::*;
}
