//! Weighted sums of packed bit rows: `fold(row) = sum of w_s over the set bits s`.
//!
//! The map is GF(2)-linear in the row, so it splits into one 8-bit piece per byte.
//!
//! - Portable: a 256-entry subset-sum table per byte.
//! - Byte-sliced: one `8 x 8` bit matrix per input and output byte, applied to 64 rows at once.

#[cfg(any(
    test,
    all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512bw",
        target_feature = "avx512vbmi",
        target_feature = "gfni",
        target_feature = "vpclmulqdq"
    )
))]
mod gfni;
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
mod portable;

use alloc::vec::Vec;

use p3_binary_field::Poly192;
use p3_binary_field::byteslice::BLOCK;
use p3_multilinear_util::point::Point;

/// The fold each target runs.
mod selected {
    /// The byte-sliced fold.
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512bw",
        target_feature = "avx512vbmi",
        target_feature = "gfni",
        target_feature = "vpclmulqdq"
    ))]
    pub(super) type Kernel = super::gfni::Fold<p3_binary_field::byteslice::Native>;

    /// The subset-sum tables.
    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512bw",
        target_feature = "avx512vbmi",
        target_feature = "gfni",
        target_feature = "vpclmulqdq"
    )))]
    pub(super) type Kernel = super::portable::Tables;
}

/// The weights of every bit of a row, prepared for folding.
#[derive(Clone, Debug)]
pub(crate) struct BitFold {
    /// Bytes per row.
    n_chunks: usize,
    /// The target's kernel data.
    kernel: selected::Kernel,
}

impl BitFold {
    /// Prepare `weights`, one per bit of a row of 8 to 128 bytes.
    ///
    /// # Panics
    ///
    /// - A row is not 8, 16, 32, 64 or 128 bytes.
    pub(crate) fn new(weights: &[Poly192]) -> Self {
        let n_chunks = weights.len() / 8;
        assert!(
            weights.len() == 8 * n_chunks
                && n_chunks.is_power_of_two()
                && (8..=128).contains(&n_chunks),
            "a row is 8 to 128 bytes, a power of two"
        );
        Self {
            n_chunks,
            kernel: selected::Kernel::new(weights),
        }
    }

    /// The fold at multilinear level `t`: row `u` of a position weighted by `eq(rho, u) L_s(z)`.
    pub(crate) fn at_level(lagrange: &[Poly192], rho: &[Poly192]) -> Self {
        let weights: Vec<Poly192> = Point::new(rho.to_vec())
            .equality_weights_lsb()
            .iter()
            .flat_map(|&e| lagrange.iter().map(move |&l| e * l))
            .collect();
        Self::new(&weights)
    }

    /// Bytes per row.
    pub(crate) const fn n_chunks(&self) -> usize {
        self.n_chunks
    }

    /// Fold up to 64 consecutive rows into `out[..rows.len()]`.
    #[inline]
    pub(crate) fn fold_block<const CHUNKS: usize>(
        &self,
        rows: &[[u8; CHUNKS]],
        out: &mut [Poly192; BLOCK],
    ) {
        debug_assert_eq!(CHUNKS, self.n_chunks);
        assert!(rows.len() <= BLOCK);
        self.kernel.fold_block(rows, out);
    }

    /// Fold 64 rows into coefficient planes, grouped by quad.
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512bw",
        target_feature = "avx512vbmi",
        target_feature = "gfni",
        target_feature = "vpclmulqdq"
    ))]
    #[inline]
    pub(crate) fn fold_quads<const CHUNKS: usize>(
        &self,
        rows: &[[u8; CHUNKS]; BLOCK],
    ) -> [[p3_binary_field::byteslice::Native; 8]; 3] {
        self.kernel.fold_quads(rows)
    }
}

#[cfg(test)]
mod tests {
    use p3_field::PrimeCharacteristicRing;
    use proptest::prelude::*;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::*;

    /// The fold by its definition.
    fn reference<const CHUNKS: usize>(weights: &[Poly192], row: &[u8; CHUNKS]) -> Poly192 {
        weights
            .iter()
            .enumerate()
            .filter(|(bit, _)| row[bit / 8] >> (bit % 8) & 1 == 1)
            .map(|(_, &w)| w)
            .sum()
    }

    /// Full blocks then a short one, against the definition and the portable tables.
    fn check<const CHUNKS: usize>(seed: u64) {
        let mut rng = SmallRng::seed_from_u64(seed);
        let weights: Vec<Poly192> = (0..8 * CHUNKS).map(|_| rng.random()).collect();
        let fold = BitFold::new(&weights);
        let tables = portable::Tables::new(&weights);
        for len in [BLOCK, 5] {
            let rows: Vec<[u8; CHUNKS]> = (0..len)
                .map(|_| core::array::from_fn(|_| rng.random()))
                .collect();
            let mut out = [Poly192::ZERO; BLOCK];
            fold.fold_block(&rows, &mut out);
            for (p, row) in rows.iter().enumerate() {
                assert_eq!(out[p], reference(&weights, row), "CHUNKS={CHUNKS}, row {p}");
                assert_eq!(tables.fold_row(row), out[p], "CHUNKS={CHUNKS}, row {p}");
            }
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(8))]

        #[test]
        fn the_dispatched_fold_is_the_definition(seed: u64) {
            check::<8>(seed);
            check::<16>(seed);
            check::<32>(seed);
            check::<64>(seed);
            check::<128>(seed);
        }
    }
}
