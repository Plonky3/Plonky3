//! The fold by one 256-entry subset-sum table per byte of a row: the reference.

use alloc::vec::Vec;

use p3_binary_field::Poly192;
use p3_field::PrimeCharacteristicRing;

use crate::byteslice::{BLOCK, subset_sums};

/// One 256-entry subset-sum table per byte of a row.
#[derive(Clone, Debug)]
pub(crate) struct Tables {
    /// Entry `[j][v]` sums the weights of the set bits of `v` at byte `j`.
    tables: Vec<[Poly192; 256]>,
}

impl Tables {
    /// The tables of one weight per bit.
    pub(crate) fn new(weights: &[Poly192]) -> Self {
        Self {
            tables: weights.as_chunks::<8>().0.iter().map(subset_sums).collect(),
        }
    }

    /// The fold of one row.
    #[inline(always)]
    pub(crate) fn fold_row<const CHUNKS: usize>(&self, row: &[u8; CHUNKS]) -> Poly192 {
        row.iter()
            .zip(&self.tables)
            .fold(Poly192::ZERO, |acc, (&v, sums)| acc + sums[usize::from(v)])
    }

    /// Fold up to 64 rows.
    #[inline]
    pub(crate) fn fold_block<const CHUNKS: usize>(
        &self,
        rows: &[[u8; CHUNKS]],
        out: &mut [Poly192; BLOCK],
    ) {
        for (o, row) in out.iter_mut().zip(rows) {
            *o = self.fold_row(row);
        }
    }
}
