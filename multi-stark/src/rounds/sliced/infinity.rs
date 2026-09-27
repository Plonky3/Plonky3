//! The four-variable tensor of a bit-valued stage on the nodes `0`, `1`, and infinity.
//!
//! A polynomial of degree at most two in one variable is pinned down by its values at `0` and
//! `1` and by its leading coefficient, its value "at infinity":
//!
//! ```text
//!     q(x) = q(0) (1 - x) + q(1) x + q(inf) (x^2 - x)
//! ```
//!
//! Fix every tensor coordinate but `v`. Each AIR input, a column or a boundary selector, is the
//! multilinear extension of its rows, so affine in `v`, and a constraint is `P = Q + L + K` of
//! total degree at most two in its inputs, with `B` the bilinear form of `Q`:
//!
//! ```text
//!     input(v)    = lo + v (hi - lo)
//!     P(input(v)) = P(lo) + v (B(lo, hi - lo) + L(hi - lo)) + v^2 Q(hi - lo)
//! ```
//!
//! Its value at infinity is `Q(hi - lo)`, where `hi - lo` is again multilinear in the other
//! coordinates. In a second coordinate `u`, `Q(a + u b) = Q(a) + u B(a, b) + u^2 Q(b)` has value
//! `Q(b)` at infinity, and at `0` or `1` it is `Q` of the half `u` selects. The maps of different
//! coordinates act on different variables and commute with the eq-weighted sum over the rows, so
//! every cell with a coordinate at infinity holds `Q` at the inputs each coordinate picks from
//! its corners:
//!
//! ```text
//!     0: lo        1: hi        inf: hi - lo = lo + hi
//! ```
//!
//! A cell with every coordinate at `0` or `1` is a row of the trace and holds `P` itself.
//!
//! The boundary selectors' rows are bits like any bit column's, and a public boundary pin is the
//! quadratic `selector * (column - public)` with its public value a constant. So over bit-valued
//! corners every picked input is a bit and, while every constant the AIRs use is a bit, `P` and
//! `Q` are polynomials over `GF(2)` that [`SlicedQuadraticFolder`] evaluates sixty-four lanes at a
//! time. Any other constant poisons the pass, and a corner outside `GF(2)` discards it.

use alloc::vec;
use alloc::vec::Vec;

use p3_air::Air;
use p3_field::Field;
use p3_maybe_rayon::prelude::*;
use p3_multilinear_util::poly::Poly;

use super::{
    LANE_VARIABLES, PlaneWords, Planes, PrefixFold, SlicedTensor, SlicedTrace, TensorNodes,
};
use crate::rounds::{AirSlot, rows_per_task};
use crate::selectors::BoundaryEvals;
use crate::sliced::{BitLaneSums, SLICED_CELLS, SlicedBit, SlicedQuadraticFolder};

/// Variables the tensor spans: three prefix variables, then the active variable `t`.
const DEPTH: usize = 4;

/// Nodes each tensor coordinate takes, `0`, `1`, and infinity, in tensor-index order.
const NODES: usize = 3;

/// Prefixes of the three variables before `t`, one per assignment of nodes.
const PREFIXES: usize = NODES.pow(DEPTH as u32 - 1);

/// What every task of the tensor pass shares.
struct InfinityTensor<'a, 'air, A, F, R> {
    /// The stage's planes.
    trace: &'a SlicedTrace<'a>,
    /// The stage's AIRs and their column spans.
    slots: &'a [AirSlot<'air, A>],
    /// Public inputs of each AIR.
    public_values: &'a [&'a [F]],
    /// Descending alpha powers of each AIR, in the accumulation field.
    alpha_powers: &'a [Vec<R>],
    /// The corners of every prefix, in prefix-index order.
    prefixes: Vec<Vec<usize>>,
    /// Words each corner block spans.
    words: usize,
    /// The lane weights: the eq factor of the variables inside a word.
    lanes: BitLaneSums<R>,
    /// The eq factor of the word variables, one per word of a corner block.
    word_weights: Vec<R>,
}

/// Per-worker sums of the tensor pass.
struct InfinityScratch<F, R> {
    /// `sums[air][prefix * 3 + node]`: eq-weighted, alpha-batched constraint sums.
    sums: Vec<Vec<R>>,
    /// Column inputs at `t = 0`, `t = 1`, and infinity, one word each.
    inputs: Vec<SlicedBit<F>>,
    /// The next-row inputs, which no AIR on the tensor declares.
    zeros: Vec<SlicedBit<F>>,
    /// Whether any evaluation was poisoned.
    poisoned: bool,
}

impl<F, R: Field> InfinityScratch<F, R> {
    fn new(airs: usize, width: usize) -> Self {
        Self {
            sums: vec![R::zero_vec(PREFIXES * NODES); airs],
            inputs: vec![SlicedBit::default(); width],
            zeros: vec![SlicedBit::default(); width],
            poisoned: false,
        }
    }

    /// Add another worker's sums into this one, poison included.
    fn merge(mut self, other: Self) -> Self {
        for (lhs, rhs) in self.sums.iter_mut().zip(other.sums) {
            R::add_slices(lhs, &rhs);
        }
        self.poisoned |= other.poisoned;
        self
    }
}

impl<A, F, R> InfinityTensor<'_, '_, A, F, R>
where
    F: Field,
    R: Field,
    A: for<'b> Air<SlicedQuadraticFolder<'b, F, R>>,
{
    /// Add one word of residual rows at one prefix to the scratch sums.
    fn accumulate(&self, scratch: &mut InfinityScratch<F, R>, word: usize, prefix: usize) {
        match self.prefixes[prefix].len() {
            2 => self.accumulate_corners::<2>(scratch, word, prefix),
            4 => self.accumulate_corners::<4>(scratch, word, prefix),
            8 => self.accumulate_corners::<8>(scratch, word, prefix),
            16 => self.accumulate_corners::<16>(scratch, word, prefix),
            corners => unreachable!("a tensor prefix reads 2 to 16 corners, not {corners}"),
        }
    }

    /// [`Self::accumulate`] at a prefix that reads `CORNERS` corners.
    fn accumulate_corners<const CORNERS: usize>(
        &self,
        scratch: &mut InfinityScratch<F, R>,
        word: usize,
        prefix: usize,
    ) {
        let corners = &self.prefixes[prefix];
        let width = self.trace.width;
        let starts: [usize; CORNERS] =
            core::array::from_fn(|corner| (corners[corner] * self.words + word) * width);
        let inputs = &mut scratch.inputs[..width];
        match &self.trace.cells {
            Planes::Low(words) => fold_corner_rows(&**words, starts, inputs),
            Planes::Pairs(pairs) => fold_corner_rows(&**pairs, starts, inputs),
        }

        let mut boundary = [[0; 3]; 2];
        for pair in corners.as_chunks::<2>().0 {
            for (selectors, &corner) in boundary.iter_mut().zip(pair) {
                let planes = self.trace.boundary[corner * self.words + word];
                for (selector, plane) in selectors.iter_mut().zip(planes) {
                    *selector ^= plane;
                }
            }
        }
        self.evaluate_nodes(scratch, boundary, self.word_weights[word], prefix);
    }

    /// Evaluate one word at `t = 0`, `t = 1`, and infinity, in one walk of each AIR.
    ///
    /// A prefix with no variable at infinity leaves `t = 0` and `t = 1` on the rows, where each
    /// constraint contributes its whole value. Every other cell takes the quadratic part, which
    /// an AIR below degree two does not have.
    ///
    /// Never inlined: the AIR evaluation needs a large stack frame, which inside the parallel
    /// fold would be reserved again at every level of Rayon's recursive split.
    #[inline(never)]
    fn evaluate_nodes(
        &self,
        scratch: &mut InfinityScratch<F, R>,
        [at_zero, at_one]: [[u64; 3]; 2],
        weight: R,
        prefix: usize,
    ) {
        let rows = self.prefixes[prefix].len() == 2;
        let [first, last, transition] = core::array::from_fn(|selector| {
            SlicedBit::new(node_words(at_zero[selector], at_one[selector]))
        });
        let boundary = BoundaryEvals::new(first, last, transition);
        let mut whole = [false; SLICED_CELLS];
        whole[..2].fill(rows);
        for slot in self.slots {
            if slot.constraint_degree == 0 || (!rows && slot.constraint_degree < 2) {
                continue;
            }
            let main = slot.main_offset..slot.main_offset + slot.main_width;
            let preprocessed =
                slot.preprocessed_offset..slot.preprocessed_offset + slot.preprocessed_width;
            let periodic = slot.periodic_offset..slot.periodic_offset + slot.periodic_width;
            let evaluation = SlicedQuadraticFolder::new(
                &scratch.inputs[main.clone()],
                &scratch.zeros[main],
                boundary,
                self.public_values[slot.stage_index],
                &self.alpha_powers[slot.stage_index],
                &self.lanes,
                whole,
            )
            .with_preprocessed(
                &scratch.inputs[preprocessed.clone()],
                &scratch.zeros[preprocessed],
            )
            .with_periodic(&scratch.inputs[periodic])
            .eval_air(slot.air);
            scratch.poisoned |= evaluation.poisoned;
            let sums = &mut scratch.sums[slot.stage_index][prefix * NODES..][..NODES];
            for (sum, &value) in sums.iter_mut().zip(&evaluation.value) {
                *sum += weight * value;
            }
        }
    }
}

/// Fold the corner rows starting at `starts`, `t = 0` and `t = 1` side by side, into every
/// column's words at `t = 0`, `t = 1`, and infinity.
///
/// Only the low planes are read: the pass runs on a stage whose high planes are clear.
fn fold_corner_rows<F, P: PlaneWords + ?Sized, const CORNERS: usize>(
    planes: &P,
    starts: [usize; CORNERS],
    inputs: &mut [SlicedBit<F>],
) {
    for (column, input) in inputs.iter_mut().enumerate() {
        let (mut at_zero, mut at_one) = (0, 0);
        for pair in starts.as_chunks::<2>().0 {
            at_zero ^= planes.planes(pair[0] + column)[0];
            at_one ^= planes.planes(pair[1] + column)[0];
        }
        *input = SlicedBit::new(node_words(at_zero, at_one));
    }
}

/// An input's words at `t = 0`, `t = 1`, and infinity, from its values at `t = 0` and `t = 1`.
///
/// The word past the three nodes is clear, so every evaluation there vanishes.
#[inline]
const fn node_words(at_zero: u64, at_one: u64) -> [u64; SLICED_CELLS] {
    const {
        assert!(
            SLICED_CELLS == NODES + 1,
            "one evaluation per node and one spare"
        );
    };
    [at_zero, at_one, at_zero ^ at_one, 0]
}

/// Evaluate all 81 tensor entries of a bit-valued stage on the nodes `0`, `1`, and infinity.
///
/// # Returns
///
/// `None` when a slot's degree exceeds two, a cell lies outside `GF(2)`, or a value outside
/// `GF(2)` poisoned an evaluation.
#[tracing::instrument(skip_all, level = "debug", fields(round = 3))]
pub(super) fn sliced_tensor_infinity<A, F, EF, R>(
    trace: &SlicedTrace<'_>,
    slots: &[AirSlot<'_, A>],
    public_values: &[&[F]],
    alpha_powers: &[Vec<R>],
    tau: &[EF],
) -> Option<SlicedTensor<EF>>
where
    F: Field,
    EF: Field + From<R>,
    R: Field + From<EF>,
    A: for<'b> Air<SlicedQuadraticFolder<'b, F, R>>,
{
    let num_vars = trace.num_vars;
    debug_assert!(trace.rounds == DEPTH - 1 && tau.len() == num_vars);
    // The tensor holds three nodes of each variable, which pin at most a quadratic, and the
    // folder drops every part of degree three and up.
    if slots.iter().any(|slot| slot.constraint_degree > 2) {
        tracing::debug!("an AIR above degree two keeps the tensor off the infinity nodes");
        return None;
    }
    if !trace.cells.high_planes_clear() {
        tracing::debug!("a cell outside GF(2) keeps the tensor off the infinity nodes");
        return None;
    }

    // Row variables split three ways: the tensor, the words of a corner block, and the lanes.
    let lane_weights = Poly::new_from_point(&tau[num_vars - LANE_VARIABLES..], EF::ONE);
    let word_weights = Poly::new_from_point(&tau[DEPTH..num_vars - LANE_VARIABLES], EF::ONE);
    let lift = |values: &[EF]| {
        values
            .iter()
            .map(|&value| R::from(value))
            .collect::<Vec<_>>()
    };
    let word_weights = lift(word_weights.as_slice());

    // The prefixes in index order, the last variable varying fastest. A variable at `0` or `1`
    // selects the low or the high half of the corners, and a variable at infinity reads both, as
    // `PrefixFold` does for any node outside `0` and `1`.
    let node_coordinates = [(false, false), (true, false), (false, true)];
    let prefixes = (0..PREFIXES)
        .map(|index| {
            let mut prefix = [(false, false); DEPTH - 1];
            let mut remaining = index;
            for node in prefix.iter_mut().rev() {
                *node = node_coordinates[remaining % NODES];
                remaining /= NODES;
            }
            PrefixFold::new(&prefix).corners
        })
        .collect();
    let context = InfinityTensor {
        trace,
        slots,
        public_values,
        alpha_powers,
        prefixes,
        words: word_weights.len(),
        lanes: BitLaneSums::new(&lift(lane_weights.as_slice())),
        word_weights,
    };

    // Every evaluation narrows the same AIR constants and public values, and the prefix of
    // nodes zero evaluates every AIR of nonzero degree. Its first task alone therefore meets any
    // value outside GF(2) before the parallel pass starts.
    let mut first = InfinityScratch::new(slots.len(), trace.width);
    context.accumulate(&mut first, 0, 0);
    let poisoned = |scratch: &InfinityScratch<F, R>| {
        if scratch.poisoned {
            tracing::debug!("an AIR constant outside GF(2) reached the infinity tensor");
        }
        scratch.poisoned
    };
    if poisoned(&first) {
        return None;
    }
    let tasks = context.words * PREFIXES;
    let scratch = (1..tasks)
        .into_par_iter()
        .with_min_len(rows_per_task(tasks))
        .par_fold_reduce(
            || InfinityScratch::new(slots.len(), trace.width),
            |mut scratch, task| {
                context.accumulate(&mut scratch, task / PREFIXES, task % PREFIXES);
                scratch
            },
            InfinityScratch::merge,
        )
        .merge(first);
    if poisoned(&scratch) {
        return None;
    }
    Some(SlicedTensor {
        values: scratch
            .sums
            .into_iter()
            .map(|sums| sums.into_iter().map(EF::from).collect())
            .collect(),
        depth: DEPTH,
        nodes: TensorNodes::Infinity,
    })
}
