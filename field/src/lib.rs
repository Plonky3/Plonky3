//! A framework for finite fields.

#![no_std]
#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

extern crate alloc;

pub mod field;
pub mod matrix;
pub mod mersenne31;
pub mod packed;
pub mod trivial_extension;
