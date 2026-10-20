//! The fold by one 256-entry sum table per stripe: the reference.

use alloc::vec;
use alloc::vec::Vec;

use p3_field::PrimeCharacteristicRing;
use p3_maybe_rayon::prelude::*;

use super::Stripes;
use crate::Challenge;
use crate::byteslice::subset_sums;

/// The fold, enough stripes per task to amortize each stripe's table.
pub(crate) fn fold_tables(
    stripes: Stripes<'_>,
    groups: usize,
    eq_outer: &[Challenge],
) -> Vec<Challenge> {
    let k = stripes.positions();
    let per_task = (stripes.len() / 256).max(1);
    let tasks = stripes.len().div_ceil(per_task);
    let (weights, _) = eq_outer.as_chunks::<8>();
    (0..tasks).into_par_iter().par_fold_reduce(
        || vec![Challenge::ZERO; k],
        |mut acc, task| {
            let first = task * per_task;
            for (s, w) in weights
                .iter()
                .enumerate()
                .take(stripes.len())
                .skip(first)
                .take(per_task)
            {
                let table = subset_sums(w);
                for (g, acc) in acc
                    .as_chunks_mut::<64>()
                    .0
                    .iter_mut()
                    .take(groups)
                    .enumerate()
                {
                    for (acc, byte) in acc.iter_mut().zip(stripes.bytes(s, g)) {
                        *acc += table[usize::from(byte)];
                    }
                }
            }
            acc
        },
        |mut x, y| {
            x.iter_mut().zip(&y).for_each(|(x, &y)| *x += y);
            x
        },
    )
}
