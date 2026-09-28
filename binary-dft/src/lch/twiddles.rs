//! The twiddle state of one transform, walked block by block instead of tabulated.

use alloc::vec::Vec;

use p3_binary_field::TowerLevel;

use crate::domain::{domain_point, domain_point_steps};

/// Stage bases and the increments between consecutive block twiddles.
///
/// The twiddle of stage `j`, block `b`, over the coset `shift + S_l` is
///
/// ```text
///     t(j, b) = W_j(shift) + domain_point(2 b)
/// ```
///
/// The second term is linear in `b`, so consecutive blocks differ by a table entry.
pub(super) struct Twiddles<F> {
    /// The twiddle of the first block of each stage, `W_j(shift)`.
    bases: Vec<F>,
    /// The step from block `b - 1` to block `b`, indexed by the trailing zeros of `b`.
    steps: Vec<F>,
}

impl<F: TowerLevel> Twiddles<F> {
    /// The twiddle state of a transform of `2^log_n` rows over the given coset.
    pub(super) fn new(log_n: usize, shift: F) -> Self {
        // W_0 is the identity and W_{j+1} = W_j^2 + W_j, so the bases form a chain of squarings.
        let mut base = shift;
        let bases = (0..log_n)
            .map(|_| {
                let current = base;
                base = base.square() + base;
                current
            })
            .collect();

        // A step depends only on the trailing zeros of the block index, never on the stage.
        //
        // So one table serves every stage, and a height of one needs no table.
        let steps = domain_point_steps::<F>(log_n.saturating_sub(1));

        Self { bases, steps }
    }

    /// The twiddle of one block, computed from its index.
    #[inline]
    pub(super) fn at(&self, stage: usize, block: usize) -> F {
        self.bases[stage] + domain_point::<F>(block << 1)
    }

    /// What to add to the twiddle of block `index - 1` to reach that of block `index`.
    #[inline]
    pub(super) fn step(&self, index: usize) -> F {
        self.steps[index.trailing_zeros() as usize]
    }
}
