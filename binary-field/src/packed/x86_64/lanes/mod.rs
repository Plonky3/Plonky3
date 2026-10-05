//! The wide registers, as the backends of both shared algebras.
//!
//! ```text
//!     GF(2^128)    the widest register:   256 bits, or 512 with avx512f
//!     GF(2^64)     always 256 bits:       4 elements, 12 coordinates of GF(2^192)
//! ```
//!
//! The prover holds one packed value per trace column in scratch buffers.
//!
//! At 512 bits a `GF(2^192)` value is 192 bytes, and a wide trace's scratch leaves the L2 cache.
//!
//! Measured on Zen 5 (Ryzen 9 9950X3D), Keccak-f proof at 2^16 rows, 32 threads:
//!
//! ```text
//!     one element per value         0.602 s
//!     8 lanes, 512-bit registers    0.633 s      twice the products per instruction
//!     4 lanes, 256-bit registers    0.570 s
//! ```
//!
//! So the `GF(2^64)` packings stay at 256 bits and take the ternary logic op where it exists.

mod avx2;
#[cfg(target_feature = "avx512f")]
mod avx512;
// The compile-time plan for moving between three-quadword elements and coordinate registers.
mod triples;

#[cfg(not(target_feature = "avx512f"))]
pub(crate) use avx2::*;
#[cfg(target_feature = "avx512f")]
pub(crate) use avx512::*;

use crate::clmul::wide::{Lanes64, TOP_NIBBLE_FOLD};
use crate::packed::split::Lanes;

/// Swaps the two quadwords of every lane, so `x ^ swap(x)` holds `x_lo ^ x_hi` in both halves.
const SWAP_QUADWORDS: i32 = 0x4e;

/// The truth table of `a ^ b ^ c` for a ternary logic instruction.
#[cfg(target_feature = "avx512vl")]
const XOR3: i32 = 0x96;

/// The 256-bit register the `GF(2^64)` packings use, on every build.
pub(crate) mod gf64 {
    use core::array;

    pub(crate) use super::avx2::{Reg, interleave_64};
    use super::avx2::{WIDTH, deinterleave_3, interleave_3, load, store};

    /// 64-bit elements per register.
    pub(crate) const WIDTH_64: usize = 2 * WIDTH;

    /// Three registers of consecutive three-quadword elements, as three coordinate registers.
    ///
    /// # Safety
    ///
    /// The address must be readable for three registers of quadwords.
    ///
    /// No alignment is required.
    #[inline(always)]
    pub(crate) unsafe fn gather_3(from: *const u64) -> [Reg; 3] {
        // SAFETY: the readability of the three registers is the caller's obligation.
        let rows = unsafe { array::from_fn(|r| load(from.add(r * WIDTH_64).cast())) };

        // Six blends and three permutes gather each coordinate into its own register.
        deinterleave_3(rows)
    }

    /// Three coordinate registers, written back as consecutive three-quadword elements.
    ///
    /// # Safety
    ///
    /// The address must be writable for three registers of quadwords.
    ///
    /// No alignment is required.
    #[inline(always)]
    pub(crate) unsafe fn scatter_3(to: *mut u64, coordinates: [Reg; 3]) {
        // The same instructions as the gather, in the opposite order.
        for (r, row) in interleave_3(coordinates).into_iter().enumerate() {
            // SAFETY: the writability of the three registers is the caller's obligation.
            unsafe { store(to.add(r * WIDTH_64).cast(), row) };
        }
    }
}

// The register is the production backend of the shared split-multiplier algebra.
//
// Every method is one intrinsic.
//
// So the generic code monomorphizes to the instructions a hand-written kernel would emit.
impl Lanes for Reg {
    #[inline(always)]
    fn zero() -> Self {
        zero()
    }

    #[inline(always)]
    fn broadcast(value: u128) -> Self {
        broadcast(value)
    }

    #[inline(always)]
    fn xor(self, other: Self) -> Self {
        xor(self, other)
    }

    #[inline(always)]
    fn unpack_low_64(self, other: Self) -> Self {
        unpack_low_64(self, other)
    }

    #[inline(always)]
    fn clmul<const IMM: i32>(self, other: Self) -> Self {
        clmul::<IMM>(self, other)
    }
}

// The 256-bit register as the backend of the `GF(2^64)` algebra.
impl Lanes64 for avx2::Reg {
    #[inline(always)]
    fn zero() -> Self {
        avx2::zero()
    }

    #[inline(always)]
    fn xor(self, other: Self) -> Self {
        avx2::xor(self, other)
    }

    #[inline(always)]
    fn xor3(self, b: Self, c: Self) -> Self {
        avx2::xor3(self, b, c)
    }

    #[inline(always)]
    fn clmul<const IMM: i32>(self, other: Self) -> Self {
        avx2::clmul::<IMM>(self, other)
    }

    #[inline(always)]
    fn unpack_low(self, other: Self) -> Self {
        avx2::unpack_low_64(self, other)
    }

    #[inline(always)]
    fn unpack_high(self, other: Self) -> Self {
        avx2::unpack_high_64(self, other)
    }

    #[inline(always)]
    fn shl<const N: i32>(self) -> Self {
        avx2::shl_64::<N>(self)
    }

    #[inline(always)]
    fn shr<const N: i32>(self) -> Self {
        avx2::shr_64::<N>(self)
    }

    // Two instructions where the shift form takes nine.
    //
    // Measured on Zen 5 (Ryzen 9 9950X3D), 512-bit registers: 0.173 ns per product against 0.182.
    #[inline(always)]
    fn fold_top_nibble(self) -> Self {
        // The nibble lands in the low byte of each quadword, and every other byte is zero.
        //
        // Entry zero of the table is zero, so those bytes look up nothing.
        let table = avx2::broadcast(u128::from_le_bytes(TOP_NIBBLE_FOLD));
        avx2::shuffle_bytes(table, avx2::shr_64::<60>(self))
    }
}
