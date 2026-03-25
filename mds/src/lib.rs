//! Maximum distance separable (MDS) matrix multiplication.
//!
//! MDS matrices provide optimal diffusion for algebraic hash functions.
//! Any k x k submatrix is invertible, so the associated linear code
//! achieves the Singleton bound.
//!
//! Three strategies are provided:
//! - Reed-Solomon coset evaluation via Bowers butterfly networks.
//! - An optimized variant that integrates coset shifts into twiddle factors.
//! - Karatsuba-style convolution for circulant matrices.

#![no_std]

extern crate alloc;

use p3_symmetric::Permutation;

mod butterflies;
pub mod coset_mds;
pub mod integrated_coset_mds;
pub mod karatsuba_convolution;
pub mod util;

/// Marker trait for permutations that are MDS.
///
/// An MDS permutation guarantees that any square submatrix of the
/// underlying linear map is invertible.
pub trait MdsPermutation<T: Clone, const WIDTH: usize>: Permutation<[T; WIDTH]> {}
