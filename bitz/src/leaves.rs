//! The grand-product leaves of Phase 3: bit `(c, b)` of claim `t` is `y^t_b` when set and one otherwise (Construction 4.4, eq. 39).

use alloc::borrow::Cow;
use alloc::vec::Vec;

use p3_field::PrimeCharacteristicRing;
use p3_gkr::{ProductLeaves, RadixFourBatch};
use p3_maybe_rayon::prelude::*;
use tracing::instrument;

use crate::params::BitShape;
use crate::{Challenge, Val};

/// Leaves of every claim's trees, laid out `(t, c, b)` with `b` least significant, materialized densely.
pub(crate) struct BitLeaves<'a> {
    /// The committed words of the shape's bits.
    pub(crate) words: &'a [u64],
    /// The shape.
    pub(crate) shape: BitShape,
    /// `y^t = g^(gamma^t)`, one row image per row of each claim.
    pub(crate) images: &'a [Vec<Challenge>],
}

impl ProductLeaves<Challenge> for BitLeaves<'_> {
    type Layer<'a>
        = RadixFourBatch<'a, Val, Challenge, 4>
    where
        Self: 'a;

    fn own_layers(&self) -> usize {
        0
    }

    #[instrument(skip_all, name = "bitz leaves")]
    fn materialize(&self) -> Cow<'_, [Challenge]> {
        let rows = self.shape.rows();
        let column_words = self.shape.column_words();
        let mut leaves = Challenge::zero_vec(self.images.len() * self.words.len() * 64);
        leaves
            .par_chunks_exact_mut(rows)
            .enumerate()
            .for_each_min_task_bytes(rows * size_of::<Challenge>(), |(tree, out)| {
                let images = &self.images[tree >> self.shape.log_columns()];
                let column = tree & (self.shape.columns() - 1);
                let words = &self.words[column * column_words..][..column_words];
                for ((out, images), &word) in out
                    .as_chunks_mut::<64>()
                    .0
                    .iter_mut()
                    .zip(images.as_chunks::<64>().0)
                    .zip(words)
                {
                    for (s, (leaf, &image)) in out.iter_mut().zip(images).enumerate() {
                        *leaf = if word >> s & 1 == 1 {
                            image
                        } else {
                            Challenge::ONE
                        };
                    }
                }
            });
        Cow::Owned(leaves)
    }

    fn layer(&self, _: usize, _: &[Challenge]) -> Self::Layer<'_> {
        unreachable!("dense bit leaves prove no layer of their own")
    }
}
