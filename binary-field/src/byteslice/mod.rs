//! Byte-sliced GF(2)-linear maps from bytes into `Poly192`.
//!
//! A map from bytes to the challenge field is one `8 x 8` bit matrix per output byte.
//!
//! A sliced block holds 64 values, one register per byte of the values.
//!
//! The kernels are generic over the register, so a scalar model runs them on every target.

mod gfni;
mod lanes;
#[cfg(any(test, feature = "test-utils"))]
mod model;

use p3_field::PrimeCharacteristicRing;

pub use self::gfni::{
    IDENTITY, OUT_BYTES, Stage, plane_matrices, store_f192, transpose_rows, weight_matrices,
};
pub use self::lanes::SliceRegister;
#[cfg(any(test, feature = "test-utils"))]
pub use self::model::Model;
use crate::{Poly192, transpose_8x8};

/// The register the sliced kernels run on, where the build has one.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512bw",
    target_feature = "avx512vbmi",
    target_feature = "gfni",
    target_feature = "vpclmulqdq"
))]
pub type Native = core::arch::x86_64::__m512i;

/// Values in a sliced block.
pub const BLOCK: usize = 64;

/// The 256 subset sums of eight weights: entry `v` sums the weights of the set bits of `v`.
pub fn subset_sums(w: &[Poly192; 8]) -> [Poly192; 256] {
    let mut sums = [Poly192::ZERO; 256];
    for (i, &e) in w.iter().enumerate() {
        // The entries with bit `i` set are those below `2^i`, plus its weight.
        let (lo, hi) = sums.split_at_mut(1 << i);
        for (h, &l) in hi.iter_mut().zip(lo.iter()) {
            *h = l + e;
        }
    }
    sums
}

/// Transpose eight rows of eight bytes: bit `k` of output byte `s` is bit `s` of row `k`.
#[inline]
pub fn transpose_bits(rows: &[u8; 64]) -> [u8; 64] {
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512bw",
        target_feature = "avx512vbmi",
        target_feature = "gfni",
        target_feature = "vpclmulqdq"
    ))]
    {
        let mut out = [0u8; 64];
        // SAFETY: both arrays are 64 bytes.
        unsafe {
            Native::store(
                out.as_mut_ptr(),
                gfni::transpose_rows(Native::load(rows.as_ptr())),
            );
        }
        out
    }
    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512bw",
        target_feature = "avx512vbmi",
        target_feature = "gfni",
        target_feature = "vpclmulqdq"
    )))]
    transpose_bits_portable(rows)
}

/// The transpose one `8 x 8` bit block at a time: the reference.
#[cfg_attr(
    all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512bw",
        target_feature = "avx512vbmi",
        target_feature = "gfni",
        target_feature = "vpclmulqdq"
    ),
    allow(dead_code)
)]
pub(crate) fn transpose_bits_portable(rows: &[u8; 64]) -> [u8; 64] {
    let mut out = [0u8; 64];
    for (j, chunk) in out.as_chunks_mut::<8>().0.iter_mut().enumerate() {
        // Byte `j` of every row, row `k` in byte `k` of a word.
        let column = u64::from_le_bytes(core::array::from_fn(|k| rows[8 * k + j]));
        *chunk = transpose_8x8(column).to_le_bytes();
    }
    out
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(32))]

        #[test]
        fn the_transpose_moves_every_bit(input in prop::array::uniform32(any::<u8>()), more in prop::array::uniform32(any::<u8>())) {
            let mut rows = [0u8; 64];
            rows[..32].copy_from_slice(&input);
            rows[32..].copy_from_slice(&more);
            let out = transpose_bits(&rows);
            prop_assert_eq!(out, transpose_bits_portable(&rows));
            for k in 0..8 {
                for s in 0..64 {
                    prop_assert_eq!(out[s] >> k & 1, rows[8 * k + s / 8] >> (s % 8) & 1);
                }
            }
        }

        #[test]
        fn subset_sums_add_the_set_bits(seed: u64) {
            let mut rng = SmallRng::seed_from_u64(seed);
            let w: [Poly192; 8] = core::array::from_fn(|_| rng.random());
            let sums = subset_sums(&w);
            for v in [0usize, 1, 0x80, 0x5A, 0xFF] {
                let want: Poly192 = (0..8).filter(|s| v >> s & 1 == 1).map(|s| w[s]).sum();
                prop_assert_eq!(sums[v], want);
            }
        }
    }
}
