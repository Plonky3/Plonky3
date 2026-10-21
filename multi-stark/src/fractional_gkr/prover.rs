use alloc::vec;
use alloc::vec::Vec;
use core::mem::MaybeUninit;
use core::slice::ChunksMut;

use p3_challenger::FieldChallenger;
use p3_challenger::fs::TranscriptField;
use p3_field::{
    Algebra, ExtensionField, Field, PackedFieldExtension, PackedValue, PrimeCharacteristicRing,
};
#[cfg(test)]
use p3_field::{batch_multiplicative_inverse, dot_product};
use p3_maybe_rayon::prelude::*;
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::{Poly, PolyMaybePacked};
use p3_sumcheck::generic_degree::RoundPolyInterpolator;
use p3_util::log2_strict_usize;

use super::transcript::{FractionGkrProverTranscript, FractionGkrShape};
use super::{
    Fraction, FractionGkrLayerProof, FractionGkrOutput, FractionGkrProof, LeafFraction,
    LeafNumerator, MIXED_LEAF_STORAGE, SplitFraction,
};

enum SplitFractionMaybePacked<F: Field, EF: ExtensionField<F>> {
    Packed(SplitFraction<Poly<EF::ExtensionPacking>>),
    Scalar(SplitFraction<Poly<EF>>),
}

fn reduce_fraction<N, A>(n0: &[N], n1: &[N], d0: &[A], d1: &[A]) -> SplitFraction<Poly<A>>
where
    N: PrimeCharacteristicRing + Copy + Send + Sync,
    A: Algebra<N> + Copy + Send + Sync,
{
    debug_assert_eq!(n0.len(), n1.len());
    debug_assert_eq!(n0.len(), d0.len());
    debug_assert_eq!(n0.len(), d1.len());
    let (n00, n01) = n0.split_at(n0.len() / 2);
    let (n10, n11) = n1.split_at(n1.len() / 2);
    let (d00, d01) = d0.split_at(d0.len() / 2);
    let (d10, d11) = d1.split_at(d1.len() / 2);

    let reduce = |n0: &[N], n1: &[N], d0: &[A], d1: &[A]| {
        n0.par_iter()
            .zip(n1.par_iter())
            .zip(d0.par_iter())
            .zip(d1.par_iter())
            .map(|(((&n0, &n1), &d0), &d1)| {
                (A::mixed_dot_product::<2>(&[d1, d0], &[n0, n1]), d0 * d1)
            })
            .unzip::<_, _, Vec<_>, Vec<_>>()
    };

    let (n0, d0) = reduce(n00, n10, d00, d10);
    let (n1, d1) = reduce(n01, n11, d01, d11);
    SplitFraction {
        n0: Poly::new(n0),
        d0: Poly::new(d0),
        n1: Poly::new(n1),
        d1: Poly::new(d1),
    }
}

/// The one or two tree levels above four child tables.
///
/// Two levels come out of one pass whenever the level after the next still has a row.
fn reduce_levels<N, A>(n0: &[N], n1: &[N], d0: &[A], d1: &[A]) -> Vec<SplitFraction<Poly<A>>>
where
    N: PrimeCharacteristicRing + Copy + Send + Sync,
    A: Algebra<N> + Copy + Send + Sync,
{
    let tables = Tables { n0, n1, d0, d1 };
    if tables.len() >= 4 {
        let (next, after) = tables.reduce_two();
        vec![next.into_polys(), after.into_polys()]
    } else {
        vec![reduce_fraction(n0, n1, d0, d1)]
    }
}

impl<'a, F: Field, EF: ExtensionField<F>> LeafFraction<'a, F, EF> {
    /// Fold adjacent fraction pairs into the first extension-field layer.
    ///
    /// The four arms are the two numerator storages crossed with the two denominator storages.
    ///
    /// Every arm lands in the extension field, so no layer below this one branches again.
    fn reduce(&self) -> Vec<SplitFractionMaybePacked<F, EF>> {
        let mut levels = match (&self.n, self.d) {
            (LeafNumerator::Base(numer), PolyMaybePacked::Packed(denom)) => {
                // Base-field numerators are stored one scalar per evaluation.
                // Packing them here lines them up with the packed denominator.
                let numer = F::Packing::pack_slice(numer.as_slice());
                let (n0, n1) = numer.split_at(numer.len() / 2);
                let (d0, d1) = denom.as_slice().split_at(denom.num_evals() / 2);
                reduce_levels(n0, n1, d0, d1)
                    .into_iter()
                    .map(SplitFractionMaybePacked::Packed)
                    .collect::<Vec<_>>()
            }
            (LeafNumerator::Base(numer), PolyMaybePacked::Scalar(denom)) => {
                let (n0, n1) = numer.as_slice().split_at(numer.num_evals() / 2);
                let (d0, d1) = denom.as_slice().split_at(denom.num_evals() / 2);
                reduce_levels(n0, n1, d0, d1)
                    .into_iter()
                    .map(SplitFractionMaybePacked::Scalar)
                    .collect()
            }
            (
                LeafNumerator::Ext(PolyMaybePacked::Packed(numer)),
                PolyMaybePacked::Packed(denom),
            ) => {
                // Both halves already agree on storage, so the kernel runs with one type.
                let (n0, n1) = numer.as_slice().split_at(numer.num_evals() / 2);
                let (d0, d1) = denom.as_slice().split_at(denom.num_evals() / 2);
                reduce_levels(n0, n1, d0, d1)
                    .into_iter()
                    .map(SplitFractionMaybePacked::Packed)
                    .collect::<Vec<_>>()
            }
            (
                LeafNumerator::Ext(PolyMaybePacked::Scalar(numer)),
                PolyMaybePacked::Scalar(denom),
            ) => {
                let (n0, n1) = numer.as_slice().split_at(numer.num_evals() / 2);
                let (d0, d1) = denom.as_slice().split_at(denom.num_evals() / 2);
                reduce_levels(n0, n1, d0, d1)
                    .into_iter()
                    .map(SplitFractionMaybePacked::Scalar)
                    .collect()
            }
            _ => unreachable!("{MIXED_LEAF_STORAGE}"),
        };
        for level in &mut levels {
            level.transition();
        }
        levels
    }
}

/// Bytes of child values one task of a pass reads.
///
/// A fused fold writes a quarter of them, which stays in cache while its sums are taken.
const TASK_BYTES: usize = 1 << 16;

/// Blocks one task of a pass covers, for child values of type `A`.
///
/// A block is four quarters of four tables.
const fn task_blocks<A>() -> usize {
    let blocks = TASK_BYTES / (16 * size_of::<A>());
    if blocks == 0 { 1 } else { blocks }
}

/// The next task's window of a table cut into task-sized chunks.
fn next_window<'a, T>(chunks: &mut ChunksMut<'a, T>) -> &'a mut [T] {
    chunks.next().expect("every table has a window per task")
}

/// The polynomial of a pair of rounds, summed at the points both messages are read from.
///
/// The pair binds the two top variables, `Y1` and then `Y2`:
///
/// ```text
///     P(Y1, Y2) = sum_j eq(j) * gate(n0, n1, d0, d1 at (Y1, Y2, j))
/// ```
///
/// Each table is bilinear in `(Y1, Y2)`, so `P` is quadratic in each.
///
/// It is held at `{0, 1, inf}^2`, indexed `[Y2][Y1]`, where `inf` is a leading coefficient.
///
/// A line's value at `inf` is a difference, so no point costs a product by an interpolation node.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Grid<A>([[A; 3]; 3]);

impl<A: PrimeCharacteristicRing + Copy> Grid<A> {
    const fn zero() -> Self {
        Self([[A::ZERO; 3]; 3])
    }

    fn merge(mut self, other: Self) -> Self {
        for (lhs, rhs) in self
            .0
            .as_flattened_mut()
            .iter_mut()
            .zip(other.0.as_flattened())
        {
            *lhs += *rhs;
        }
        self
    }

    /// Adds one block's gate, weighted by `eq`.
    ///
    /// Each table's corners are indexed `2 * Y1 + Y2`, its quarters in memory order.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn add_block<N>(&mut self, lambda: A, eq: A, n0: [N; 4], n1: [N; 4], d0: [A; 4], d1: [A; 4])
    where
        N: PrimeCharacteristicRing + Copy,
        A: Algebra<N>,
    {
        // The weight and the batching term go into two factors, once per corner:
        //
        //     eq (d1 n0 + d0 n1 + lambda d0 d1) = d1 (eq n0 + lambda eq d0) + (eq d0) n1
        //
        // Both new factors are bilinear too, so their corners are all it takes.
        let d0: [A; 4] = d0.map(|d0| eq * d0);
        let m: [A; 4] = core::array::from_fn(|q| eq * n0[q] + lambda * d0[q]);
        let (n1, d0, d1, m) = (spread(n1), spread(d0), spread(d1), spread(m));

        // The numerator stays in its own type at every point.
        //
        // The point (1, 1) is skipped: the round's claim fixes it, once per pass.
        for (y2, sums) in self.0.iter_mut().enumerate() {
            for (y1, sum) in sums.iter_mut().enumerate() {
                if (y1, y2) != (1, 1) {
                    *sum += d1[y2][y1] * m[y2][y1] + d0[y2][y1] * n1[y2][y1];
                }
            }
        }
    }
}

impl<EF: Field> Grid<EF> {
    /// Fills in the point `(1, 1)`, the one value the claim of the first round determines.
    ///
    /// With `qA(Y1) = (1 - cB) P(Y1, 0) + cB P(Y1, 1)` the first round's polynomial:
    ///
    /// ```text
    ///     claim   = (1 - cA) qA(0) + cA qA(1)
    ///     qA(1)   = (1 - cB) P(1, 0) + cB P(1, 1)
    /// ```
    ///
    /// Both coordinates are inverted by their own rounds anyway, so neither may be zero.
    fn complete(mut self, claim: EF, [first, second]: [EF; 2]) -> Self {
        let [low, high, _] = &mut self.0;
        let at_zero = (EF::ONE - second) * low[0] + second * high[0];
        let at_one = (claim - (EF::ONE - first) * at_zero) * first.inverse();
        high[1] = (at_one - (EF::ONE - second) * low[1]) * second.inverse();
        self
    }

    /// The first round's message, `Y2` summed against its equality weight.
    ///
    /// Returns its values at zero and at the second interpolation node.
    fn first_round(&self, coordinate: EF) -> [EF; 2] {
        let [low, high, _] = self.0;
        let node = EF::interpolation_node(2);
        let weigh =
            |at_low: EF, at_high: EF| (EF::ONE - coordinate) * at_low + coordinate * at_high;
        [
            weigh(low[0], high[0]),
            weigh(quadratic_at(low, node), quadratic_at(high, node)),
        ]
    }

    /// The second round's message, once `Y1` is bound to `challenge`.
    ///
    /// Returns its values at zero and at the second interpolation node.
    fn second_round(&self, challenge: EF) -> [EF; 2] {
        let bound = self.0.map(|row| quadratic_at(row, challenge));
        [bound[0], quadratic_at(bound, EF::interpolation_node(2))]
    }
}

/// A bilinear table's values at `{0, 1, inf}^2`, indexed `[Y2][Y1]`, from its corners.
///
/// The corners are indexed `2 * Y1 + Y2`.
#[inline(always)]
fn spread<T: PrimeCharacteristicRing + Copy>([c00, c01, c10, c11]: [T; 4]) -> [[T; 3]; 3] {
    let low = [c00, c10, c10 - c00];
    let high = [c01, c11, c11 - c01];
    [low, high, core::array::from_fn(|y1| high[y1] - low[y1])]
}

/// The quadratic with values `a` at zero, `b` at one and leading coefficient `c`, at `x`.
#[inline]
fn quadratic_at<EF: Field>([a, b, c]: [EF; 3], x: EF) -> EF {
    a + (b - a - c) * x + c * x.square()
}

/// The four corners of block `j`, one per quarter of a table.
#[inline(always)]
const fn corners<T: Copy>(table: &[T], quarter: usize, j: usize) -> [T; 4] {
    [
        table[j],
        table[quarter + j],
        table[2 * quarter + j],
        table[3 * quarter + j],
    ]
}

/// Binds a block's two variables, `Y1` to `low` and then `Y2` to `high`.
#[inline(always)]
fn fold_corners<N, A>([c00, c01, c10, c11]: [N; 4], [low, high]: [A; 2]) -> A
where
    N: PrimeCharacteristicRing + Copy,
    A: Algebra<N> + Copy,
{
    let lower = low * (c10 - c00) + c00;
    let upper = low * (c11 - c01) + c01;
    lower + high * (upper - lower)
}

/// One layer's four child tables, borrowed.
///
/// Row `j` of a table is its child at the remaining variables, the top one most significant.
#[derive(Clone, Copy)]
struct Tables<'a, N, A> {
    n0: &'a [N],
    n1: &'a [N],
    d0: &'a [A],
    d1: &'a [A],
}

impl<'a, N, A> Tables<'a, N, A>
where
    N: PrimeCharacteristicRing + Copy + Send + Sync,
    A: Algebra<N> + Copy + Send + Sync,
{
    const fn len(&self) -> usize {
        self.d0.len()
    }

    /// Block `j`'s corners in every table.
    #[inline(always)]
    const fn block(&self, quarter: usize, j: usize) -> ([N; 4], [N; 4], [A; 4], [A; 4]) {
        (
            corners(self.n0, quarter, j),
            corners(self.n1, quarter, j),
            corners(self.d0, quarter, j),
            corners(self.d1, quarter, j),
        )
    }

    /// Sums the next pair of rounds' polynomial, `eq` weighing each block.
    fn grid(&self, eq: &[A], lambda: A) -> Grid<A> {
        let quarter = eq.len();
        debug_assert_eq!(self.len(), 4 * quarter);
        let blocks = task_blocks::<A>();
        eq.par_chunks(blocks)
            .enumerate()
            .map_collect_min_task_bytes(4 * TASK_BYTES, |(task, eq)| {
                let mut grid = Grid::zero();
                for (j, &eq) in (task * blocks..).zip(eq) {
                    let (n0, n1, d0, d1) = self.block(quarter, j);
                    grid.add_block(lambda, eq, n0, n1, d0, d1);
                }
                grid
            })
            .into_iter()
            .fold(Grid::zero(), Grid::merge)
    }

    /// Binds the two top variables to `challenges` and writes the folded tables to `out`.
    ///
    /// Given the next pair's weights, it also sums that pair's polynomial from the values it writes.
    ///
    /// Those values are still in registers, so the next pair reads no memory.
    fn fold_two(
        &self,
        challenges: [A; 2],
        out: &mut SplitFraction<Vec<A>>,
        next: Option<(&[A], A)>,
    ) -> Grid<A> {
        let len = self.len() / 4;
        let mut slots = out.spare_rows(len);

        let grid = if len < 4 {
            // Too short for quarters: one or two rows, folded in place of a task.
            debug_assert!(next.is_none());
            for j in 0..len {
                let (n0, n1, d0, d1) = self.block(len, j);
                let values = [
                    fold_corners(n0, challenges),
                    fold_corners(n1, challenges),
                    fold_corners(d0, challenges),
                    fold_corners(d1, challenges),
                ];
                for (slot, value) in slots.iter_mut().zip(values) {
                    slot[j].write(value);
                }
            }
            Grid::zero()
        } else {
            // Output row `q * next_quarter + j` is quarter `q` of the next pair's block `j`.
            //
            // A task folds every quarter of a run of those blocks, so it writes 16 windows.
            let next_quarter = len / 4;
            let blocks = task_blocks::<A>();
            let mut windows = slots.each_mut().map(|table| {
                let mut quarters = table.chunks_exact_mut(next_quarter);
                core::array::from_fn::<_, 4, _>(|_| {
                    quarters
                        .next()
                        .expect("a table has four quarters")
                        .chunks_mut(blocks)
                })
            });
            let tasks = (0..next_quarter.div_ceil(blocks))
                .map(|_| {
                    windows
                        .each_mut()
                        .map(|quarters| quarters.each_mut().map(next_window))
                })
                .collect::<Vec<_>>();

            tasks
                .into_par_iter()
                .enumerate()
                .map_collect_min_task_bytes(5 * TASK_BYTES, |(task, mut windows)| {
                    let start = task * blocks;
                    let rows = windows[0][0].len();

                    // One quarter at a time, so a pass reads 16 source streams rather than 64.
                    for q in 0..4 {
                        let [n0s, n1s, d0s, d1s] = windows.each_mut().map(|table| &mut *table[q]);
                        let slots = n0s.iter_mut().zip(n1s).zip(d0s).zip(d1s);
                        for (row, (((n0s, n1s), d0s), d1s)) in slots.enumerate() {
                            let (n0, n1, d0, d1) = self.block(len, q * next_quarter + start + row);
                            n0s.write(fold_corners(n0, challenges));
                            n1s.write(fold_corners(n1, challenges));
                            d0s.write(fold_corners(d0, challenges));
                            d1s.write(fold_corners(d1, challenges));
                        }
                    }

                    // The rows just written are still in cache, so the next sums read no memory.
                    let mut grid = Grid::zero();
                    if let Some((eq, lambda)) = next {
                        // SAFETY: the loop above wrote every slot of every window.
                        let windows = windows.each_ref().map(|table| {
                            table.each_ref().map(|window| unsafe {
                                core::slice::from_raw_parts(window.as_ptr().cast::<A>(), rows)
                            })
                        });
                        for (row, &eq) in eq[start..start + rows].iter().enumerate() {
                            let [n0, n1, d0, d1] =
                                windows.map(|table| table.map(|window| window[row]));
                            grid.add_block::<A>(lambda, eq, n0, n1, d0, d1);
                        }
                    }
                    grid
                })
                .into_iter()
                .fold(Grid::zero(), Grid::merge)
        };

        // SAFETY: every row below `len` was written exactly once above.
        unsafe { out.set_rows(len) };
        grid
    }

    /// Adds each pair of fractions, two tree levels in one pass.
    ///
    /// Row `i` of the next level is the sum of the fractions at row `i` of both halves:
    ///
    /// ```text
    ///     n0 / d0 + n1 / d1 = (d1 n0 + d0 n1) / (d0 d1)
    /// ```
    ///
    /// The level after takes those sums in pairs again while they are still in registers.
    fn reduce_two(&self) -> (SplitFraction<Vec<A>>, SplitFraction<Vec<A>>) {
        let quarter = self.len() / 4;
        debug_assert!(quarter >= 1);
        let mut next = SplitFraction::with_capacity(2 * quarter);
        let mut after = SplitFraction::with_capacity(quarter);

        // Row `i` of the next level and its partner `i + 2 quarter` sum into row `i` of the one after.
        //
        // That row is in the first half of the level after when `i < quarter`, else in the second.
        let blocks = task_blocks::<A>();
        let [n0, n1, d0, d1] = next
            .spare_rows(2 * quarter)
            .map(|table| table.split_at_mut(quarter));
        let [after_n0, after_n1, after_d0, after_d1] = after.spare_rows(quarter);
        let halves = [
            (0, [n0.0, n1.0, d0.0, d1.0], [after_n0, after_d0]),
            (quarter, [n0.1, n1.1, d0.1, d1.1], [after_n1, after_d1]),
        ];
        let tasks = halves
            .into_iter()
            .flat_map(|(offset, next, after)| {
                let mut next = next.map(|table| table.chunks_mut(blocks));
                let mut after = after.map(|table| table.chunks_mut(blocks));
                (0..quarter.div_ceil(blocks)).map(move |task| {
                    (
                        offset + task * blocks,
                        next.each_mut().map(next_window),
                        after.each_mut().map(next_window),
                    )
                })
            })
            .collect::<Vec<_>>();

        tasks.into_par_iter().for_each_min_task_bytes(
            4 * TASK_BYTES,
            |(start, [n0s, n1s, d0s, d1s], [after_n, after_d])| {
                let rows = n0s.len();
                for row in 0..rows {
                    let i = start + row;
                    let sum = |j: usize| {
                        let (n0, n1, d0, d1) = (self.n0[j], self.n1[j], self.d0[j], self.d1[j]);
                        (A::mixed_dot_product::<2>(&[d1, d0], &[n0, n1]), d0 * d1)
                    };
                    let ((n_low, d_low), (n_high, d_high)) = (sum(i), sum(i + 2 * quarter));
                    n0s[row].write(n_low);
                    d0s[row].write(d_low);
                    n1s[row].write(n_high);
                    d1s[row].write(d_high);
                    after_n[row].write(A::dot_product::<2>(&[d_high, d_low], &[n_low, n_high]));
                    after_d[row].write(d_low * d_high);
                }
            },
        );

        // SAFETY: the tasks' windows cover every row of both levels, each written once.
        unsafe {
            next.set_rows(2 * quarter);
            after.set_rows(quarter);
        }
        (next, after)
    }
}

impl<A> SplitFraction<Vec<A>> {
    /// An empty set of tables, the spare a first fold allocates into.
    const fn empty() -> Self {
        Self {
            n0: Vec::new(),
            d0: Vec::new(),
            n1: Vec::new(),
            d1: Vec::new(),
        }
    }

    /// Empty tables with room for `len` rows each.
    fn with_capacity(len: usize) -> Self {
        Self {
            n0: Vec::with_capacity(len),
            d0: Vec::with_capacity(len),
            n1: Vec::with_capacity(len),
            d1: Vec::with_capacity(len),
        }
    }

    /// The first `len` unwritten rows of every table, cleared first.
    fn spare_rows(&mut self, len: usize) -> [&mut [MaybeUninit<A>]; 4] {
        [&mut self.n0, &mut self.n1, &mut self.d0, &mut self.d1].map(|table| {
            table.clear();
            table.reserve(len);
            &mut table.spare_capacity_mut()[..len]
        })
    }

    /// Marks the first `len` rows of every table written.
    ///
    /// # Safety
    ///
    /// Every table must have its first `len` rows initialized.
    unsafe fn set_rows(&mut self, len: usize) {
        for table in [&mut self.n0, &mut self.n1, &mut self.d0, &mut self.d1] {
            // SAFETY: the caller wrote the first `len` rows.
            unsafe { table.set_len(len) };
        }
    }

    fn into_polys(self) -> SplitFraction<Poly<A>> {
        SplitFraction {
            n0: Poly::new(self.n0),
            d0: Poly::new(self.d0),
            n1: Poly::new(self.n1),
            d1: Poly::new(self.d1),
        }
    }

    fn view(&self) -> Tables<'_, A, A> {
        Tables {
            n0: &self.n0,
            n1: &self.n1,
            d0: &self.d0,
            d1: &self.d1,
        }
    }
}

impl<A> SplitFraction<Poly<A>> {
    fn view(&self) -> Tables<'_, A, A> {
        Tables {
            n0: self.n0.as_slice(),
            n1: self.n1.as_slice(),
            d0: self.d0.as_slice(),
            d1: self.d1.as_slice(),
        }
    }
}

/// The sum of a packed value's lanes.
fn sum_lanes<F: Field, EF: ExtensionField<F>>(value: EF::ExtensionPacking) -> EF {
    EF::ExtensionPacking::to_ext_iter([value]).sum()
}

/// One layer's sumcheck, from its claim down to the two children's claims.
struct LayerSumcheck<'a, 't, C, F: TranscriptField, EF> {
    transcript: &'a mut FractionGkrProverTranscript<'t, C, F, EF>,
    /// The parent's point, one coordinate per round.
    point: &'a [EF],
    /// The challenge batching the layer's numerator and denominator claims.
    lambda: EF,
    /// The claimed sum of the next round's equality-stripped polynomial.
    claim: EF,
    /// The equality factor of the bound variables.
    eq_prefix: EF,
    interpolator: &'a RoundPolyInterpolator<EF>,
    round_polys: Vec<[EF; 3]>,
    challenges: Vec<EF>,
}

impl<'a, 't, C, F, EF> LayerSumcheck<'a, 't, C, F, EF>
where
    F: TranscriptField,
    EF: ExtensionField<F>,
    C: FieldChallenger<F>,
{
    fn new(
        transcript: &'a mut FractionGkrProverTranscript<'t, C, F, EF>,
        point: &'a Point<EF>,
        lambda: EF,
        claim: EF,
        interpolator: &'a RoundPolyInterpolator<EF>,
    ) -> Self {
        Self {
            transcript,
            point: point.as_slice(),
            lambda,
            claim,
            eq_prefix: EF::ONE,
            interpolator,
            round_polys: Vec::with_capacity(point.num_variables()),
            challenges: Vec::with_capacity(point.num_variables()),
        }
    }

    /// The coordinates of the rounds after the next `skip`.
    fn coordinates_after(&self, skip: usize) -> &'a [EF] {
        &self.point[self.challenges.len() + skip..]
    }

    /// Sends one round from its equality-stripped values at zero and the second node.
    fn round(&mut self, evals: [EF; 2]) -> EF {
        let coordinate = self.point[self.challenges.len()];
        let (round_poly, q_sum) = restore_equality_factor(
            evals,
            self.claim,
            coordinate,
            self.eq_prefix,
            self.interpolator,
        );
        let challenge = self.transcript.round(&round_poly);
        self.round_polys.push(round_poly);
        self.challenges.push(challenge);
        self.claim = self.interpolator.eval(&evals, q_sum, challenge);
        self.eq_prefix *= (EF::ONE - coordinate) + challenge * (coordinate.double() - EF::ONE);
        challenge
    }

    /// Sends a pair of rounds from their polynomial.
    fn two_rounds(&mut self, grid: Grid<EF>) -> [EF; 2] {
        let k = self.challenges.len();
        let grid = grid.complete(self.claim, [self.point[k], self.point[k + 1]]);
        let low = self.round(grid.first_round(self.point[k + 1]));
        let high = self.round(grid.second_round(low));
        [low, high]
    }

    /// Runs every pair of rounds while a table has at least four rows.
    ///
    /// Returns the tables left, of one or two rows.
    fn prove_pairs<N, A>(
        &mut self,
        source: Tables<'_, N, A>,
        eq: impl Fn(&[EF]) -> Poly<A>,
        lanes: impl Fn(A) -> EF,
    ) -> SplitFraction<Vec<A>>
    where
        N: PrimeCharacteristicRing + Copy + Send + Sync,
        A: Algebra<N> + From<EF> + Copy + Send + Sync,
    {
        debug_assert!(source.len() >= 4);
        let lambda = A::from(self.lambda);
        // The weights of the pair after the next, when its tables will have four rows.
        let next_eq = |this: &Self, len: usize| (len >= 16).then(|| eq(this.coordinates_after(2)));

        let grid = source.grid(eq(self.coordinates_after(2)).as_slice(), lambda);
        let mut challenges = self.two_rounds(Grid(grid.0.map(|row| row.map(&lanes))));

        // The first fold leaves the source in place; later ones ping-pong two buffers.
        let mut tables = SplitFraction::empty();
        let mut spare = SplitFraction::empty();
        let weights = next_eq(self, source.len());
        let mut grid = source.fold_two(
            challenges.map(A::from),
            &mut tables,
            weights.as_ref().map(|eq| (eq.as_slice(), lambda)),
        );
        while tables.d0.len() >= 4 {
            challenges = self.two_rounds(Grid(grid.0.map(|row| row.map(&lanes))));
            let weights = next_eq(self, tables.d0.len());
            grid = tables.view().fold_two(
                challenges.map(A::from),
                &mut spare,
                weights.as_ref().map(|eq| (eq.as_slice(), lambda)),
            );
            core::mem::swap(&mut tables, &mut spare);
        }
        tables
    }

    /// Proves a layer stored packed, `W` rows per element.
    ///
    /// The pairs run packed while a table has four elements, then the scalar tail takes over.
    fn prove_packed<N>(&mut self, source: Tables<'_, N, EF::ExtensionPacking>) -> SplitFraction<EF>
    where
        N: PrimeCharacteristicRing + Copy + Send + Sync,
        EF::ExtensionPacking: Algebra<N>,
    {
        let eq = |point: &[EF]| Poly::new_packed_from_point(point, EF::ONE);
        let tables = self.prove_pairs(source, eq, sum_lanes::<F, EF>);
        let unpack =
            |table: Vec<EF::ExtensionPacking>| Poly::new(table).unpack::<F, EF>().into_evals();
        let tables = SplitFraction {
            n0: unpack(tables.n0),
            d0: unpack(tables.d0),
            n1: unpack(tables.n1),
            d1: unpack(tables.d1),
        };
        self.prove_scalar(tables.view())
    }

    /// Proves a layer stored one row per element.
    fn prove_scalar<N>(&mut self, source: Tables<'_, N, EF>) -> SplitFraction<EF>
    where
        N: PrimeCharacteristicRing + Copy + Send + Sync,
        EF: Algebra<N>,
    {
        let tables = if source.len() >= 4 {
            let eq = |point: &[EF]| Poly::new_from_point(point, EF::ONE);
            self.prove_pairs(source, eq, |value| value)
        } else {
            // At most two rows, lifted for the last round.
            let lift = |table: &[N]| table.iter().map(|&value| EF::from(value)).collect();
            SplitFraction {
                n0: lift(source.n0),
                d0: source.d0.to_vec(),
                n1: lift(source.n1),
                d1: source.d1.to_vec(),
            }
        };

        let row = |i: usize| SplitFraction {
            n0: tables.n0[i],
            d0: tables.d0[i],
            n1: tables.n1[i],
            d1: tables.d1[i],
        };
        if tables.d0.len() == 1 {
            return row(0);
        }

        // One variable left: each table is a line through its two rows.
        let (lo, hi) = (row(0), row(1));
        let at = |x: EF| SplitFraction {
            n0: lo.n0 + x * (hi.n0 - lo.n0),
            d0: lo.d0 + x * (hi.d0 - lo.d0),
            n1: lo.n1 + x * (hi.n1 - lo.n1),
            d1: lo.d1 + x * (hi.d1 - lo.d1),
        };
        let node = EF::interpolation_node(2);
        let challenge = self.round([lo.gate(self.lambda), at(node).gate(self.lambda)]);
        at(challenge)
    }

    /// Proves a layer of the reduced tree.
    fn prove_reduced(&mut self, fraction: &SplitFractionMaybePacked<F, EF>) -> SplitFraction<EF> {
        match fraction {
            SplitFractionMaybePacked::Packed(fraction) if fraction.n0.num_evals() >= 4 => {
                self.prove_packed(fraction.view())
            }
            SplitFractionMaybePacked::Packed(fraction) => {
                let fraction = SplitFraction {
                    n0: fraction.n0.unpack::<F, EF>(),
                    d0: fraction.d0.unpack::<F, EF>(),
                    n1: fraction.n1.unpack::<F, EF>(),
                    d1: fraction.d1.unpack::<F, EF>(),
                };
                self.prove_scalar(fraction.view())
            }
            SplitFractionMaybePacked::Scalar(fraction) => self.prove_scalar(fraction.view()),
        }
    }

    /// Proves the leaf layer, reading the caller's tables in place.
    fn prove_leaves(&mut self, fraction: &LeafFraction<'_, F, EF>) -> SplitFraction<EF> {
        const fn halves<T>(table: &[T]) -> (&[T], &[T]) {
            table.split_at(table.len() / 2)
        }
        match (&fraction.n, fraction.d) {
            (LeafNumerator::Base(numer), PolyMaybePacked::Packed(denom)) => {
                // Base-field numerators are stored one scalar per row.
                // Packing them lines them up with the packed denominator.
                let (n0, n1) = halves(F::Packing::pack_slice(numer.as_slice()));
                let (d0, d1) = halves(denom.as_slice());
                self.prove_packed(Tables { n0, n1, d0, d1 })
            }
            (LeafNumerator::Base(numer), PolyMaybePacked::Scalar(denom)) => {
                let (n0, n1) = halves(numer.as_slice());
                let (d0, d1) = halves(denom.as_slice());
                self.prove_scalar(Tables { n0, n1, d0, d1 })
            }
            (
                LeafNumerator::Ext(PolyMaybePacked::Packed(numer)),
                PolyMaybePacked::Packed(denom),
            ) => {
                let (n0, n1) = halves(numer.as_slice());
                let (d0, d1) = halves(denom.as_slice());
                self.prove_packed(Tables { n0, n1, d0, d1 })
            }
            (
                LeafNumerator::Ext(PolyMaybePacked::Scalar(numer)),
                PolyMaybePacked::Scalar(denom),
            ) => {
                let (n0, n1) = halves(numer.as_slice());
                let (d0, d1) = halves(denom.as_slice());
                self.prove_scalar(Tables { n0, n1, d0, d1 })
            }
            _ => unreachable!("{MIXED_LEAF_STORAGE}"),
        }
    }

    /// Hands back the layer's round messages and challenges, once it reached `claims`.
    fn finish(self, claims: &SplitFraction<EF>) -> (Vec<[EF; 3]>, Vec<EF>) {
        debug_assert_eq!(self.challenges.len(), self.point.len());
        debug_assert_eq!(self.claim, claims.gate(self.lambda));
        (self.round_polys, self.challenges)
    }
}

impl<A> SplitFraction<Poly<A>>
where
    A: PrimeCharacteristicRing + Copy + Send + Sync,
{
    fn num_variables(&self) -> usize {
        self.n0.num_variables() + 1
    }

    fn reduce(&self) -> Vec<Self> {
        reduce_levels(
            self.n0.as_slice(),
            self.n1.as_slice(),
            self.d0.as_slice(),
            self.d1.as_slice(),
        )
    }

    fn into_constants(self) -> SplitFraction<A> {
        SplitFraction {
            n0: self
                .n0
                .as_constant()
                .expect("root numerator must be constant"),
            d0: self
                .d0
                .as_constant()
                .expect("root denominator must be constant"),
            n1: self
                .n1
                .as_constant()
                .expect("root numerator must be constant"),
            d1: self
                .d1
                .as_constant()
                .expect("root denominator must be constant"),
        }
    }
}

#[cfg(test)]
impl<A: Field> SplitFraction<Poly<A>> {
    pub(super) fn sum(&self) -> A {
        let mut denom = Vec::with_capacity(self.d0.num_evals() + self.d1.num_evals());
        denom.extend_from_slice(self.d0.as_slice());
        denom.extend_from_slice(self.d1.as_slice());
        let denom_inv = batch_multiplicative_inverse(&denom);

        dot_product(
            denom_inv.into_iter(),
            self.n0.as_slice().iter().chain(self.n1.as_slice()).copied(),
        )
    }

    pub(super) fn eval(&self, point: &Point<A>) -> (A, A) {
        assert_eq!(point.num_variables(), self.num_variables());
        let branch = point[0];
        let suffix = point.get_subpoint_over_range(1..);
        let n0 = self.n0.eval_ext::<A>(&suffix);
        let n1 = self.n1.eval_ext::<A>(&suffix);
        let d0 = self.d0.eval_ext::<A>(&suffix);
        let d1 = self.d1.eval_ext::<A>(&suffix);

        (n0 + branch * (n1 - n0), d0 + branch * (d1 - d0))
    }
}

impl<F: Field, EF: ExtensionField<F>> SplitFractionMaybePacked<F, EF> {
    fn num_variables(&self) -> usize {
        match self {
            Self::Packed(fraction) => {
                fraction.num_variables() + log2_strict_usize(F::Packing::WIDTH)
            }
            Self::Scalar(fraction) => fraction.num_variables(),
        }
    }

    /// The one or two tree levels above this one.
    fn reduce(&self) -> Vec<Self> {
        let mut levels = match self {
            Self::Packed(fraction) => fraction.reduce().into_iter().map(Self::Packed).collect(),
            Self::Scalar(fraction) => fraction
                .reduce()
                .into_iter()
                .map(Self::Scalar)
                .collect::<Vec<_>>(),
        };
        for level in &mut levels {
            level.transition();
        }
        levels
    }

    /// Leave packed mode after its final useful SIMD reduction.
    fn transition(&mut self) {
        match self {
            Self::Packed(fraction) if fraction.n0.num_evals() == 1 => {
                *self = Self::Scalar(SplitFraction {
                    n0: fraction.n0.unpack::<F, EF>(),
                    d0: fraction.d0.unpack::<F, EF>(),
                    n1: fraction.n1.unpack::<F, EF>(),
                    d1: fraction.d1.unpack::<F, EF>(),
                });
            }
            _ => {}
        }
    }

    fn unpack(self) -> SplitFraction<Poly<EF>> {
        match self {
            Self::Packed(fraction) => SplitFraction {
                n0: fraction.n0.unpack::<F, EF>(),
                n1: fraction.n1.unpack::<F, EF>(),
                d0: fraction.d0.unpack::<F, EF>(),
                d1: fraction.d1.unpack::<F, EF>(),
            },
            Self::Scalar(fraction) => fraction,
        }
    }
}

/// Restore the current equality factor to the internally quadratic round polynomial.
///
/// The returned message contains the cubic evaluations expected by the existing verifier.
/// `q_sum` is the normalized quadratic sum `q(0) + q(1)` used only by the prover.
///
/// # Interpolation domain
///
/// The message carries the cubic at the four nodes the verifier interpolates over.
///
/// Those nodes are the field's own enumeration of the first four values, not the integers.
///
/// The two coincide in odd characteristic, extensions included.
///
/// Over a binary tower they do not.
///
/// There the integer three collapses onto the second node, and two onto the first.
fn restore_equality_factor<EF: Field>(
    quadratic_evals: [EF; 2],
    normalized_sum: EF,
    coordinate: EF,
    equality_prefix: EF,
    interpolator: &RoundPolyInterpolator<EF>,
) -> ([EF; 3], EF) {
    let q1 = (normalized_sum - (EF::ONE - coordinate) * quadratic_evals[0]) * coordinate.inverse();
    let q_sum = quadratic_evals[0] + q1;
    let q3 = interpolator.eval(&quadratic_evals, q_sum, EF::interpolation_node(3));

    // The equality factor in this variable is affine, so two values fix it everywhere:
    //
    //     eq(0)         = 1 - coordinate
    //     eq(1) - eq(0) = 2 * coordinate - 1
    //
    // Each node scales the step.
    //
    // Stepping would reach the next node only in odd characteristic.
    let equality_0 = EF::ONE - coordinate;
    let equality_step = coordinate.double() - EF::ONE;
    let equality_2 = equality_0 + EF::interpolation_node(2) * equality_step;
    let equality_3 = equality_0 + EF::interpolation_node(3) * equality_step;
    let round_poly = [
        equality_prefix * equality_0 * quadratic_evals[0],
        equality_prefix * equality_2 * quadratic_evals[1],
        equality_prefix * equality_3 * q3,
    ];
    (round_poly, q_sum)
}

/// Proves that a table of fractions sums to zero via a binary-tree reduction.
///
/// The run is driven through this protocol's own transcript, seeded from the variable count.
/// It returns the numerator and denominator openings of the input tables at the
/// resulting transcript-derived point.
///
/// The challenger is borrowed for the run and handed back where the reduction stops.
///
/// # Panics
///
/// Panics if the numerator and denominator have different variable counts, if
/// they have no variables, or if the fully reduced root denominator is zero.
///
/// Panics if an extension-field numerator and its denominator disagree on storage.
pub fn prove_fractional_gkr<F, EF, Challenger>(
    fraction: LeafFraction<'_, F, EF>,
    challenger: &mut Challenger,
) -> (FractionGkrProof<EF>, FractionGkrOutput<EF>)
where
    F: TranscriptField,
    EF: ExtensionField<F>,
    Challenger: FieldChallenger<F>,
{
    let num_variables = fraction.n.num_variables();
    assert_eq!(num_variables, fraction.d.num_variables());

    // The round kernel reads the two halves side by side.
    //
    // An extension-field numerator therefore has to be stored the way its denominator is.
    //
    // Nothing downstream can recover from a mixed pair.
    //
    // Where it would surface depends on the size, so it is rejected here instead.
    if let LeafNumerator::Ext(numer) = fraction.n {
        assert_eq!(
            matches!(numer, PolyMaybePacked::Packed(_)),
            matches!(fraction.d, PolyMaybePacked::Packed(_)),
            "{MIXED_LEAF_STORAGE}"
        );
    }
    assert!(
        num_variables >= 1,
        "fraction GKR requires at least one input variable"
    );

    // With one input variable, the original two leaves are already the root's children. The
    // root layer is therefore also the input layer: it has no sumcheck rounds and directly opens
    // the input at the sampled branch coordinate.
    if num_variables == 1 {
        let denom = fraction.d.clone().unpack();
        let [n0, n1] = fraction.n.leaf_pair();
        let root_claims = SplitFraction {
            n0,
            d0: denom.as_slice()[0],
            n1,
            d1: denom.as_slice()[1],
        };
        let root_denominator = root_claims.d0 * root_claims.d1;
        assert_eq!(
            root_claims.d1 * root_claims.n0 + root_claims.d0 * root_claims.n1,
            EF::ZERO,
            "fraction GKR input fractions must sum to zero"
        );
        assert_ne!(
            root_denominator,
            EF::ZERO,
            "fraction GKR root denominator must be nonzero"
        );

        // One driver covers the whole run, which here is the single root layer.
        let mut transcript = FractionGkrProverTranscript::<Challenger, F, EF>::new(
            challenger,
            FractionGkrShape { num_variables },
            root_denominator,
        );

        let lambda = transcript.begin_layer();
        debug_assert_eq!(lambda * root_denominator, root_claims.gate(lambda));
        let branch = transcript.end_layer(&root_claims);
        transcript.finish();

        let numerator = root_claims.n0 + branch * (root_claims.n1 - root_claims.n0);
        let denominator = root_claims.d0 + branch * (root_claims.d1 - root_claims.d0);

        return (
            FractionGkrProof {
                root_denominator,
                layers: vec![FractionGkrLayerProof {
                    round_polys: Vec::new(),
                    claims: root_claims,
                }],
            },
            FractionGkrOutput {
                point: Point::new(vec![branch]),
                numerator,
                denominator,
            },
        );
    }

    // The packed input path needs three backing variables: two for its initial
    // branch splits and one for dropping the first equality-suffix variable
    // before transitioning to scalar storage. With fewer, those logical
    // variables live entirely inside SIMD lanes, where the backing-table folds
    // cannot address them. At this tiny boundary, unpack once and reuse the
    // scalar prover so its proof and transcript stay identical to scalar storage.
    let scalar_denominator = match fraction.d {
        PolyMaybePacked::Packed(denom) if denom.num_variables() < 3 => {
            Some(PolyMaybePacked::Scalar(denom.unpack::<F, EF>()))
        }
        _ => None,
    };
    // A packed extension-field numerator follows its denominator out of packed storage.
    // The round kernel reads the two side by side, so they cannot part ways here.
    let scalar_numerator = match (&scalar_denominator, &fraction.n) {
        (Some(_), LeafNumerator::Ext(PolyMaybePacked::Packed(numer))) => {
            Some(PolyMaybePacked::Scalar(numer.unpack::<F, EF>()))
        }
        _ => None,
    };
    let fraction = Fraction {
        n: scalar_numerator
            .as_ref()
            .map_or(fraction.n, LeafNumerator::Ext),
        d: scalar_denominator.as_ref().unwrap_or(fraction.d),
    };

    // Build every reduced tree layer while retaining the original input for the
    // final leaf layer, the only one whose numerator may sit in the base field.
    let mut fractions = fraction.reduce();
    while fractions.last().unwrap().num_variables() > 1 {
        let levels = fractions.last().unwrap().reduce();
        fractions.extend(levels);
    }

    let root_claims = fractions.pop().unwrap().unpack().into_constants();
    let root_denominator = root_claims.d0 * root_claims.d1;
    assert_eq!(
        root_claims.d1 * root_claims.n0 + root_claims.d0 * root_claims.n1,
        EF::ZERO,
        "fraction GKR input fractions must sum to zero"
    );
    assert_ne!(
        root_denominator,
        EF::ZERO,
        "fraction GKR root denominator must be nonzero"
    );

    // One driver covers every layer of the run, from the root down to the input tables.
    let mut transcript = FractionGkrProverTranscript::<Challenger, F, EF>::new(
        challenger,
        FractionGkrShape { num_variables },
        root_denominator,
    );
    let interpolator = RoundPolyInterpolator::new(2);
    let mut layer_proofs = Vec::with_capacity(num_variables);

    // Prove the final reduction from the two one-variable children to the scalar root. This
    // layer has no sumcheck variables, but its random linear combination binds the root
    // denominator to the zero-numerator statement.
    let lambda = transcript.begin_layer();
    debug_assert_eq!(lambda * root_denominator, root_claims.gate(lambda));
    let branch = transcript.end_layer(&root_claims);
    let mut numerator = root_claims.n0 + branch * (root_claims.n1 - root_claims.n0);
    let mut denominator = root_claims.d0 + branch * (root_claims.d1 - root_claims.d0);
    let mut point = Point::new(vec![branch]);
    layer_proofs.push(FractionGkrLayerProof {
        round_polys: Vec::new(),
        claims: root_claims,
    });

    // Reduced extension-field layers are consumed from the top of the tree down.
    for fraction in fractions.into_iter().rev() {
        let lambda = transcript.begin_layer();
        let mut layer = LayerSumcheck::new(
            &mut transcript,
            &point,
            lambda,
            numerator + lambda * denominator,
            &interpolator,
        );
        let claims = layer.prove_reduced(&fraction);
        let (round_polys, mut round_point) = layer.finish(&claims);
        drop(fraction);

        let branch = transcript.end_layer(&claims);
        numerator = claims.n0 + branch * (claims.n1 - claims.n0);
        denominator = claims.d0 + branch * (claims.d1 - claims.d0);
        round_point.insert(0, branch);
        point = Point::new(round_point);
        layer_proofs.push(FractionGkrLayerProof {
            round_polys,
            claims,
        });
    }

    // The leaf tables are the only layer whose numerator may sit in the base field.
    // Its first fold promotes the numerators.
    let lambda = transcript.begin_layer();
    let mut layer = LayerSumcheck::new(
        &mut transcript,
        &point,
        lambda,
        numerator + lambda * denominator,
        &interpolator,
    );
    let claims = layer.prove_leaves(&fraction);
    let (round_polys, mut round_point) = layer.finish(&claims);
    let branch = transcript.end_layer(&claims);
    transcript.finish();

    let numerator = claims.n0 + branch * (claims.n1 - claims.n0);
    let denominator = claims.d0 + branch * (claims.d1 - claims.d0);
    round_point.insert(0, branch);
    let point = Point::new(round_point);
    layer_proofs.push(FractionGkrLayerProof {
        round_polys,
        claims,
    });

    (
        FractionGkrProof {
            root_denominator,
            layers: layer_proofs,
        },
        FractionGkrOutput {
            point,
            numerator,
            denominator,
        },
    )
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use p3_baby_bear::BabyBear;
    use p3_binary_field::{Poly64, Poly192};
    use p3_field::extension::BinomialExtensionField;
    use p3_field::{
        Algebra, ExtensionField, Field, PackedFieldExtension, PackedValue, PrimeCharacteristicRing,
    };
    use rand::SeedableRng;
    use rand::distr::{Distribution, StandardUniform};
    use rand::rngs::SmallRng;

    use super::{Grid, SplitFraction, Tables, reduce_fraction, sum_lanes};

    type Bb4 = BinomialExtensionField<BabyBear, 4>;

    fn values<T>(rng: &mut SmallRng, len: usize) -> Vec<T>
    where
        StandardUniform: Distribution<T>,
    {
        (0..len).map(|_| StandardUniform.sample(rng)).collect()
    }

    /// Random tables of `4 * quarter` rows, with numerators of type `N`.
    fn tables<N, A>(rng: &mut SmallRng, quarter: usize) -> (Vec<N>, Vec<N>, Vec<A>, Vec<A>)
    where
        StandardUniform: Distribution<N> + Distribution<A>,
    {
        let len = 4 * quarter;
        (
            values(rng, len),
            values(rng, len),
            values(rng, len),
            values(rng, len),
        )
    }

    /// The layer's gate at `(y1, y2)` on block `j`, each table's line evaluated directly.
    fn gate_at<N, A>(t: &Tables<'_, N, A>, lambda: A, j: usize, y1: A, y2: A) -> A
    where
        N: PrimeCharacteristicRing + Copy + Send + Sync,
        A: Algebra<N> + Copy + Send + Sync,
    {
        let quarter = t.len() / 4;
        let at = |table: [A; 4]| {
            let lower = table[0] + y1 * (table[2] - table[0]);
            let upper = table[1] + y1 * (table[3] - table[1]);
            lower + y2 * (upper - lower)
        };
        let (n0, n1, d0, d1) = t.block(quarter, j);
        let (n0, n1) = (at(n0.map(A::from)), at(n1.map(A::from)));
        let (d0, d1) = (at(d0), at(d1));
        d1 * n0 + d0 * n1 + lambda * d0 * d1
    }

    /// The pair's claim and both its messages, from the gate summed point by point.
    fn expected_messages<N, EF>(
        t: &Tables<'_, N, EF>,
        eq: &[EF],
        lambda: EF,
        [first_coordinate, coordinate, challenge]: [EF; 3],
    ) -> (EF, [[EF; 2]; 2])
    where
        N: PrimeCharacteristicRing + Copy + Send + Sync,
        EF: Field + Algebra<N>,
    {
        let node = EF::interpolation_node(2);
        let sum = |y1: EF, y2: EF| {
            (0..eq.len())
                .map(|j| eq[j] * gate_at(t, lambda, j, y1, y2))
                .sum::<EF>()
        };
        // The first round sums `Y2` against its equality weight.
        let first =
            |y1: EF| (EF::ONE - coordinate) * sum(y1, EF::ZERO) + coordinate * sum(y1, EF::ONE);
        let claim =
            (EF::ONE - first_coordinate) * first(EF::ZERO) + first_coordinate * first(EF::ONE);
        (
            claim,
            [
                [first(EF::ZERO), first(node)],
                [sum(challenge, EF::ZERO), sum(challenge, node)],
            ],
        )
    }

    fn check_messages<N, EF>(seed: u64)
    where
        N: PrimeCharacteristicRing + Copy + Send + Sync,
        EF: Field + Algebra<N>,
        StandardUniform: Distribution<N> + Distribution<EF>,
    {
        let mut rng = SmallRng::seed_from_u64(seed);
        // One block, a few blocks, and a quarter split into several tasks.
        for quarter in [1, 2, 3, 17, 700] {
            let (n0, n1, d0, d1) = tables::<N, EF>(&mut rng, quarter);
            let t = Tables {
                n0: &n0,
                n1: &n1,
                d0: &d0,
                d1: &d1,
            };
            let eq: Vec<EF> = values(&mut rng, quarter);
            let [lambda, first_coordinate, coordinate, challenge] =
                [(); 4].map(|()| StandardUniform.sample(&mut rng));

            // The grid leaves out the point the claim determines.
            let (claim, expected) =
                expected_messages(&t, &eq, lambda, [first_coordinate, coordinate, challenge]);
            let grid = t
                .grid(&eq, lambda)
                .complete(claim, [first_coordinate, coordinate]);
            assert_eq!(
                grid.first_round(coordinate),
                expected[0],
                "quarter {quarter}"
            );
            assert_eq!(
                grid.second_round(challenge),
                expected[1],
                "quarter {quarter}"
            );
        }
    }

    #[test]
    fn a_pair_of_rounds_matches_the_gate_summed_point_by_point() {
        // The odd-characteristic arm, its numerators in the base field and in the extension.
        check_messages::<BabyBear, Bb4>(1);
        check_messages::<Bb4, Bb4>(2);

        // A binary tower names its second node by bit pattern, not as one plus one.
        check_messages::<Poly64, Poly192>(3);
        check_messages::<Poly192, Poly192>(4);
    }

    #[test]
    fn a_packed_grid_sums_to_the_scalar_grid() {
        // Packing puts the low variables in lanes, so the quarters stay the top ones.
        type Packed = <Bb4 as ExtensionField<BabyBear>>::ExtensionPacking;
        type PackedBase = <BabyBear as Field>::Packing;
        let mut rng = SmallRng::seed_from_u64(5);
        for quarter in [1, 5, 300] {
            let (n0, n1, d0, d1) = tables::<PackedBase, Packed>(&mut rng, quarter);
            let eq: Vec<Packed> = values(&mut rng, quarter);
            let lambda: Bb4 = StandardUniform.sample(&mut rng);
            let packed = Tables {
                n0: &n0,
                n1: &n1,
                d0: &d0,
                d1: &d1,
            }
            .grid(&eq, lambda.into());

            let unpack_base = |table: &[PackedBase]| {
                table
                    .iter()
                    .flat_map(|p| p.as_slice().iter().copied())
                    .collect::<Vec<_>>()
            };
            let unpack = |table: &[Packed]| {
                table
                    .iter()
                    .flat_map(|p| (0..PackedBase::WIDTH).map(move |lane| p.extract(lane)))
                    .collect::<Vec<Bb4>>()
            };
            let (n0, n1, d0, d1, eq) = (
                unpack_base(&n0),
                unpack_base(&n1),
                unpack(&d0),
                unpack(&d1),
                unpack(&eq),
            );
            let scalar = Tables {
                n0: &n0,
                n1: &n1,
                d0: &d0,
                d1: &d1,
            }
            .grid(&eq, lambda);

            assert_eq!(
                Grid(packed.0.map(|row| row.map(sum_lanes::<BabyBear, Bb4>))),
                scalar
            );
        }
    }

    fn check_fold<N, EF>(seed: u64)
    where
        N: PrimeCharacteristicRing + Copy + Send + Sync,
        EF: Field + Algebra<N>,
        StandardUniform: Distribution<N> + Distribution<EF>,
    {
        let mut rng = SmallRng::seed_from_u64(seed);
        // Outputs of one and two rows, a single next block, and several tasks.
        for quarter in [1, 2, 4, 64, 2048] {
            let (n0, n1, d0, d1) = tables::<N, EF>(&mut rng, quarter);
            let t = Tables {
                n0: &n0,
                n1: &n1,
                d0: &d0,
                d1: &d1,
            };
            let [low, high, lambda] = [(); 3].map(|()| StandardUniform.sample(&mut rng));

            // Fold the top variable, then the next, one half at a time.
            let fold = |table: Vec<EF>, r: EF| {
                let (lo, hi) = table.split_at(table.len() / 2);
                lo.iter()
                    .zip(hi)
                    .map(|(&lo, &hi)| lo + r * (hi - lo))
                    .collect::<Vec<_>>()
            };
            let twice = |table: &[N]| {
                fold(
                    fold(table.iter().map(|&v| EF::from(v)).collect(), low),
                    high,
                )
            };
            let expected = SplitFraction {
                n0: twice(&n0),
                d0: fold(fold(d0.clone(), low), high),
                n1: twice(&n1),
                d1: fold(fold(d1.clone(), low), high),
            };

            // The next pair's weights exist once the output has four rows.
            let eq: Vec<EF> = values(&mut rng, quarter / 4);
            let next = (quarter >= 4).then_some((eq.as_slice(), lambda));
            let mut out = SplitFraction::empty();
            let grid = t.fold_two([low, high], &mut out, next);

            assert_eq!(out, expected, "quarter {quarter}");
            if quarter >= 4 {
                assert_eq!(grid, expected.view().grid(&eq, lambda), "quarter {quarter}");
            }
        }
    }

    #[test]
    fn two_tree_levels_in_one_pass_match_two_passes() {
        let mut rng = SmallRng::seed_from_u64(9);
        // The smallest pair of levels, and enough rows for several tasks.
        for quarter in [1, 2, 4096] {
            let (n0, n1, d0, d1) = tables::<BabyBear, Bb4>(&mut rng, quarter);
            let (next, after) = Tables {
                n0: &n0,
                n1: &n1,
                d0: &d0,
                d1: &d1,
            }
            .reduce_two();

            // The reference adds one level per pass.
            let once = reduce_fraction(&n0, &n1, &d0, &d1);
            let view = once.view();
            let twice = reduce_fraction(view.n0, view.n1, view.d0, view.d1);
            assert_eq!(next.into_polys(), once, "quarter {quarter}");
            assert_eq!(after.into_polys(), twice, "quarter {quarter}");
        }
    }

    #[test]
    fn a_fused_fold_binds_two_variables_and_sums_the_next_pair() {
        check_fold::<BabyBear, Bb4>(6);
        check_fold::<Bb4, Bb4>(7);
        check_fold::<Poly64, Poly192>(8);
    }
}
