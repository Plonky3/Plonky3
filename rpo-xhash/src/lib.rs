//! RPO and xHash hash permutations.
//!
//! Uses Plonky3's native MDS matrices (`MdsMatrixBabyBear`, `MdsMatrixKoalaBear`)
//! at width 24 for BabyBear and KoalaBear. M31 width-24 has two choices:
//! the paper's truncated circulant (`RpoCirMds24`, see `rpo::m31`) and a
//! Karatsuba-convolution variant using BabyBear's MDS column lifted to
//! Mersenne31 (`Mds24M31BBCol`, see `mds_m31_bb`). Goldilocks width-12 uses
//! a frequency-domain MDS (`MdsBase12`, see `mds_goldilocks`) matching the
//! row used by miden-crypto's `Rpo256` / `Rpx256`.
//!
//! On aarch64, the base-field S-box and round-constant additions run via
//! `<F as Field>::Packing` (4 elements per NEON vector for BB/KB/M31,
//! 2 elements for Goldilocks).

#![no_std]
// Tight numerical loops with explicit indexing read better than iterator
// chains for these S-box / chain kernels.
#![allow(clippy::needless_range_loop)]
// Mul/add chains in addition chains; `x = x * y` is more readable than `x *= y`
// when the multiply is one of many composing into an expression.
#![allow(clippy::assign_op_pattern)]
// We're intentionally non-const for hash factory APIs (they take RNGs).
#![allow(clippy::missing_const_for_fn)]

extern crate alloc;

pub mod ext_arith;
pub mod fft_mds;
pub mod mds_goldilocks;
pub mod mds_m31_bb;
pub mod pow_map;
pub mod reduce;
pub mod rpo;
pub mod xhash;
