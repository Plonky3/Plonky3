//! Goldilocks field implementation using Montgomery arithmetic with extension field support.
//!
//! This crate provides a Montgomery form implementation of the Goldilocks prime field,
//! with optional AVX2/AVX512 vectorization support for improved performance.
//!
//! ## SIMD Support
//!
//! When compiled with SIMD support, this crate provides vectorized operations:
//! - `PackedGoldilocksMontyAVX2`: processes 4 field elements simultaneously (AVX2)  
//! - `PackedGoldilocksMontyAVX512`: processes 8 field elements simultaneously (AVX512)
//!
//! ### Building with SIMD
//!
//! To enable AVX2 optimizations:
//! ```bash
//! RUSTFLAGS="-C target-feature=+avx2" cargo build --release
//! ```
//!
//! To enable AVX512 optimizations:
//! ```bash
//! RUSTFLAGS="-C target-feature=+avx512f" cargo build --release
//! ```
//!
//! ### Benchmarking
//!
//! To run benchmarks comparing scalar vs vectorized performance:
//! ```bash
//! ./bench_avx2.sh  # For AVX2
//! ```
//!
//! Or manually:
//! ```bash
//! RUSTFLAGS="-C target-feature=+avx2" cargo bench --bench bench_field
//! RUSTFLAGS="-C target-feature=+avx512f" cargo bench --bench bench_field  # For AVX512
//! ```

#![no_std]

extern crate alloc;

mod extension;
mod goldilocks;
mod mds;
mod poseidon2;

#[cfg(test)]
mod tests;

pub use goldilocks::*;
pub use mds::*;
pub use poseidon2::*;

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    not(target_feature = "avx512f")
))]
mod x86_64_avx2;

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    not(target_feature = "avx512f")
))]
pub use x86_64_avx2::*;

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
mod x86_64_avx512;

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
pub use x86_64_avx512::*;
