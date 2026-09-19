//! Later zerocheck rounds computed in a field isomorphic to the challenge field.
//!
//! Two representations of one field can multiply at very different costs:
//!
//! ```text
//!     GF(2^128), tower basis      : three changes of basis around one carryless multiply
//!     GF(2^128), polynomial basis : one carryless multiply
//! ```
//!
//! A stage bound into `R` keeps every value its row loop reads in `R`.
//! The claims, the zerocheck point, and the interpolators stay in the challenge field.
//! Each round's per-node sums cross back into it once.
//!
//! The conversions are field isomorphisms, so every round polynomial is the challenge field's.

use alloc::vec::Vec;

use p3_air::{Air, BaseAir};
use p3_field::{Algebra, ExtensionField, Field, HasSubfield};
use p3_maybe_rayon::prelude::*;
use p3_multilinear_util::poly::Poly;

use super::subfield::PARALLEL_FOLD_CELLS;
use super::{ExtColumns, InteractionCoupling, RoundStateBase, RoundStateExt};
use crate::folder::{InteractionMultilinearFolder, MultilinearFolder};
use crate::selectors::BoundaryEvals;

/// Entries of a subfield fold table, one per value of a byte.
const TABLE_ENTRIES: usize = 1 << u8::BITS;

/// The table entry of a trace-field element: the first byte of its serialization.
#[inline]
fn table_index<F: Field>(value: F) -> usize {
    value.into_bytes().into_iter().next().map_or(0, usize::from)
}

/// The fold of every cell pair of a stage that fits the subfield `S`, tabulated in `R`.
///
/// Every cell and every difference of two cells lies in `S`, so two lookups replace the product:
///
/// ```text
///     R::from(lo + r * (hi - lo)) = low[index(lo)] + scaled[index(hi - lo)]
///
///     low[index(s)]    = R::from(s)
///     scaled[index(s)] = R::from(r * s)          for every s in S
/// ```
///
/// A table exists only when the index tells every element of `S` apart.
struct SubfieldFoldTables<R> {
    /// Each element of `S`, carried into `R`.
    low: [R; TABLE_ENTRIES],
    /// Each element of `S` scaled by the challenge, carried into `R`.
    scaled: [R; TABLE_ENTRIES],
}

impl<R: Field> SubfieldFoldTables<R> {
    /// Tabulate the fold at `r` over every element of `S`.
    ///
    /// The elements are zero and the powers of the generator of `S`.
    ///
    /// # Returns
    ///
    /// `None` when two elements of `S` share an index, or when those elements are not all of `S`.
    fn new<S, F, EF>(r: EF) -> Option<Self>
    where
        S: Field,
        F: HasSubfield<S>,
        EF: ExtensionField<F>,
        R: From<EF>,
    {
        let mut tables = Self {
            low: [R::ZERO; TABLE_ENTRIES],
            scaled: [R::ZERO; TABLE_ENTRIES],
        };
        let mut filled = [false; TABLE_ENTRIES];
        let mut insert = |element: S| {
            let cell = F::from(element);
            let index = table_index(cell);
            if core::mem::replace(&mut filled[index], true) {
                return false;
            }
            let value = EF::from(cell);
            tables.low[index] = R::from(value);
            tables.scaled[index] = R::from(r * value);
            true
        };

        // A repeated index ends the walk, so it takes at most one step per entry.
        let mut count = 1_u64;
        if !insert(S::ZERO) {
            return None;
        }
        let mut power = S::ONE;
        loop {
            if !insert(power) {
                return None;
            }
            count += 1;
            power *= S::GENERATOR;
            if power == S::ONE {
                break;
            }
        }
        (S::order().to_u64_digits() == [count]).then_some(tables)
    }
}

impl<'air, 'data, A, F, EF> RoundStateBase<'air, 'data, A, F, EF>
where
    F: Field,
    EF: ExtensionField<F>,
    A: BaseAir<F>,
{
    /// Bind the first variable at `r`, folding every column straight into `R`.
    ///
    /// Each pair of cells crosses into `R` and folds there:
    ///
    /// ```text
    ///     R::from(lo) + R::from(r) * (R::from(hi) - R::from(lo))
    /// ```
    ///
    /// The repeat-last tails fold in the challenge field and cross afterwards.
    /// The alpha powers, the lookup coefficients, and the selector prefix cross once here.
    #[tracing::instrument(skip_all, level = "debug")]
    pub(crate) fn fold_into<R>(self, r: EF) -> RoundStateExt<'air, 'data, A, F, EF, R>
    where
        R: Field + From<EF>,
    {
        let challenge = R::from(r);
        self.fold_pairs_into(r, |lo, hi| {
            let lo = R::from(EF::from(lo));
            lo + challenge * (R::from(EF::from(hi)) - lo)
        })
    }

    /// Bind the first variable at `r` into `R` without a product per cell.
    ///
    /// Every pair of cells folds through two table lookups, see [`SubfieldFoldTables`].
    /// When the tables cannot be built, the pairs fold as in [`Self::fold_into`].
    ///
    /// # Precondition
    ///
    /// The first round found this stage to fit `S`, as [`Self::fits_subfield`] reports.
    /// Every cell then lies in `S`, so every lookup lands on a filled entry.
    #[tracing::instrument(skip_all, level = "debug")]
    pub(crate) fn fold_subfield_into<S, R>(self, r: EF) -> RoundStateExt<'air, 'data, A, F, EF, R>
    where
        S: Field,
        F: HasSubfield<S>,
        R: Field + From<EF>,
    {
        debug_assert!(self.fits_subfield());
        let Some(tables) = SubfieldFoldTables::<R>::new::<S, F, EF>(r) else {
            tracing::debug!("the subfield's elements share a fold table index");
            return self.fold_into(r);
        };
        self.fold_pairs_into(r, |lo, hi| {
            tables.low[table_index(lo)] + tables.scaled[table_index(hi - lo)]
        })
    }

    /// Bind the first variable at `r`, folding each pair of cells into `R` with `fold_pair`.
    fn fold_pairs_into<R, U>(
        mut self,
        r: EF,
        fold_pair: U,
    ) -> RoundStateExt<'air, 'data, A, F, EF, R>
    where
        R: Field + From<EF>,
        U: Fn(F, F) -> R + Sync,
    {
        let next_tail = self.fold_claims_and_tails(r);

        // Columns already fold in parallel, so a short column folds on its own thread.
        let columns = self.fold_each_column(|column| {
            let (lo, hi) = column.split_at(column.len() / 2);
            let fold_cells = |(&lo, &hi): (&F, &F)| fold_pair(lo, hi);
            Poly::new(if column.len() < PARALLEL_FOLD_CELLS {
                lo.iter().zip(hi).map(fold_cells).collect()
            } else {
                lo.par_iter().zip(hi).map(fold_cells).collect()
            })
        });

        let lift = |values: &[EF]| {
            values
                .iter()
                .map(|&value| R::from(value))
                .collect::<Vec<_>>()
        };
        RoundStateExt {
            public_values: self.public_values,
            alpha: R::from(self.alpha),
            alpha_powers: self
                .alpha_powers
                .iter()
                .map(|powers| lift(powers))
                .collect(),
            betas: self.betas,
            constraint_groups: self.constraint_groups,
            interaction_groups: self.interaction_groups,
            slots: self.slots,
            tau: self.tau,
            round: 1,
            columns: ExtColumns::Scalar(columns),
            next_tail: lift(&next_tail),
            coupling: InteractionCoupling {
                links: self
                    .coupling
                    .links
                    .iter()
                    .map(|link| link.map(R::from))
                    .collect(),
                theta_beta_powers: lift(&self.coupling.theta_beta_powers),
            },
            lookup_scale: self.eta,
            boundary: BoundaryEvals::new(R::from(EF::ONE - r), R::from(r), R::from(EF::ONE - r)),
        }
    }
}

impl<'air, 'data, A, F, EF, R> RoundStateExt<'air, 'data, A, F, EF, R>
where
    F: Field,
    EF: ExtensionField<F> + From<R>,
    R: Field + From<EF>,
{
    /// Evaluate this round's polynomial with every row value in `R`.
    ///
    /// The eq weights cross into `R` here, once per round, split across threads past the length
    /// at which a column fold would split.
    #[tracing::instrument(skip_all, level = "debug")]
    pub(crate) fn round_poly_repr(&mut self, eq_suffix: &Poly<EF>) -> Vec<EF>
    where
        R: Algebra<F>,
        A: for<'b> Air<MultilinearFolder<'b, F, R, R>>
            + for<'b> Air<InteractionMultilinearFolder<'b, F, R, R>>,
    {
        let weights = eq_suffix.as_slice();
        let lift = |&weight: &EF| R::from(weight);
        let eq_suffix = Poly::new(if weights.len() < PARALLEL_FOLD_CELLS {
            weights.iter().map(lift).collect()
        } else {
            weights.par_iter().map(lift).collect()
        });
        self.round_poly_unpacked(&eq_suffix)
    }

    /// Bind the next variable at `r`, folding the scalar columns in `R`.
    ///
    /// # Panics
    ///
    /// Panics if the columns are packed, which a stage folded into `R` never is.
    #[tracing::instrument(skip_all, level = "debug")]
    pub(crate) fn fold_repr(&mut self, r: EF) {
        self.fold_claims_and_tails(r);

        let r = R::from(r);
        match &mut self.columns {
            ExtColumns::Scalar(cols) => cols
                .par_iter_mut()
                .for_each(|col| col.fix_prefix_var_mut(r)),
            ExtColumns::Packed(_) | ExtColumns::Sliced(_) => {
                unreachable!("a stage folded into R keeps scalar columns")
            }
        }

        self.boundary.apply(r);
        self.round += 1;
    }
}

#[cfg(test)]
mod tests;
