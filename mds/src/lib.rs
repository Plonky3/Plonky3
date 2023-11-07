//! A library dealing with maximum distance separable transformations.

#![no_std]

extern crate alloc;

use p3_symmetric::Permutation;

pub mod babybear;
mod butterflies;
mod types;
pub mod coset_mds;
pub mod goldilocks;
pub mod integrated_coset_mds;
pub mod karatsuba_convolution;
pub mod mersenne31;
pub mod util;

pub trait MdsPermutation<T: Clone, const WIDTH: usize>: Permutation<[T; WIDTH]> {}
