//! A framework for symmetric cryptography primitives.

#![no_std]
#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

extern crate alloc;

pub mod compression;
pub mod hasher;
pub mod permutation;
pub mod sponge;
