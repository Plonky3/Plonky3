//! The fold byte-sliced: one register is 64 positions of a stripe, one matrix per stripe and output byte.
//!
//! A stripe of eight instances is read one cache line per instance, eight groups at a time.

use alloc::vec;
use alloc::vec::Vec;

use p3_field::PrimeCharacteristicRing;
use p3_maybe_rayon::prelude::*;

use super::Stripes;
use crate::Challenge;
use crate::byteslice::{OUT_BYTES, SliceRegister, store_f192, transpose_rows, weight_matrices};

/// Stripes one sweep over the accumulators covers, when the batch has enough of them.
pub(crate) const WIDE_TILE: usize = 32;

/// Stripes per sweep in a smaller batch.
pub(crate) const NARROW_TILE: usize = 8;

/// Line blocks the loads run ahead of the sweep.
const PREFETCH: usize = 4;

/// Whether the kernel takes this batch: lines inside instances, whole tiles.
pub(crate) const fn fits(stripes: &Stripes<'_>, tile: usize) -> bool {
    stripes.words >= 8 && stripes.len().is_multiple_of(tile)
}

/// The fold, a tile of stripes per step, each worker summing into its own accumulators.
///
/// Group `g`, register `o` of an accumulator is byte `o` of positions `64g ..`.
///
/// # Panics
///
/// - The batch does not fit the kernel.
pub(crate) fn partial_fold<R: SliceRegister + Send, const TILE: usize>(
    stripes: Stripes<'_>,
    groups: usize,
    eq_outer: &[Challenge],
) -> Vec<Challenge> {
    assert!(fits(&stripes, TILE), "whole lines and whole tiles");
    let blocks = groups.div_ceil(8);
    let tiles = stripes.len() / TILE;
    // One run of tiles per worker: each extra accumulator costs a zeroing and a merge of every group.
    let acc = (0..tiles)
        .into_par_iter()
        .with_min_len(tiles.div_ceil(current_num_threads()))
        .par_fold_reduce(
            || vec![[R::zero(); OUT_BYTES]; 8 * blocks],
            |mut acc, tile| {
                fold_tile::<R, TILE>(
                    stripes,
                    &eq_outer[8 * TILE * tile..][..8 * TILE],
                    tile,
                    &mut acc,
                );
                acc
            },
            |mut x, y| {
                for (x, y) in x.iter_mut().flatten().zip(y.iter().flatten()) {
                    *x = x.xor(*y);
                }
                x
            },
        );
    // Groups past the useful ones fold zero padding, so only the useful ones are stored.
    let mut out = vec![Challenge::ZERO; stripes.positions()];
    for (acc, out) in acc.iter().take(groups).zip(out.as_chunks_mut::<64>().0) {
        store_f192(acc, out);
    }
    out
}

/// Add one tile of stripes, eight eq weights per stripe in `eq`, a block of eight groups at a time.
#[inline(always)]
fn fold_tile<R: SliceRegister, const TILE: usize>(
    stripes: Stripes<'_>,
    eq: &[Challenge],
    tile: usize,
    acc: &mut [[R; OUT_BYTES]],
) {
    let (weights, _) = eq.as_chunks::<8>();
    let matrices: [[u64; OUT_BYTES]; TILE] =
        core::array::from_fn(|t| weight_matrices::<R>(&weights[t]));
    let words = stripes.words;
    // Stripe t's eight bit-sliced group registers.
    let mut buf = [[R::zero(); 8]; TILE];
    for (block, acc) in acc.as_chunks_mut::<8>().0.iter_mut().enumerate() {
        // Phase 1: each stripe's eight lines, transposed to one register per group, then bit-sliced.
        for (t, slot) in buf.iter_mut().enumerate() {
            let first = 8 * (TILE * tile + t) * words + 8 * block;
            // SAFETY: instance r's line is eight words inside instance r.
            let lines: [R; 8] = core::array::from_fn(|r| unsafe {
                R::load(stripes.z.as_ptr().add(first + r * words).cast())
            });
            if 8 * (block + PREFETCH) < words {
                for r in 0..8 {
                    R::prefetch(
                        stripes
                            .z
                            .as_ptr()
                            .wrapping_add(first + r * words + 8 * PREFETCH)
                            .cast(),
                    );
                }
            }
            *slot = R::transpose_qwords(lines).map(transpose_rows);
        }
        // Phase 2: each group's 24 sums stay in registers across the tile, two stripes per three-way XOR.
        for (g, acc) in acc.iter_mut().enumerate() {
            let mut r = *acc;
            for (x, m) in buf
                .as_chunks::<2>()
                .0
                .iter()
                .zip(matrices.as_chunks::<2>().0)
            {
                for (o, r) in r.iter_mut().enumerate() {
                    let g0 = x[0][g].affine(R::splat(m[0][o]));
                    let g1 = x[1][g].affine(R::splat(m[1][o]));
                    *r = r.xor3(g0, g1);
                }
            }
            *acc = r;
        }
    }
}
