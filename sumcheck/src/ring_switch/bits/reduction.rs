//! One ring-switching reduction at a bit alphabet, end to end.
//!
//! The five values the reduction is built from, the messages it sends, and the two sides that run it.
//!
//! Both sides are methods, because here the reduction has a type to hang them on.
//! The sibling module's are free functions because there the reduction has none.

use alloc::vec::Vec;
use core::borrow::Borrow;

use p3_binary_field::TowerLevel;
use p3_challenger::fs::TranscriptField;
use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::{Field, PackedValue};
use p3_maybe_rayon::prelude::*;
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;
use serde::{Deserialize, Serialize};

use super::basis::{Coefficients, CoordinateSums};
use super::equality::{FactoredEquality, scaled_sums_into};
use super::packing::BitPacking;
use super::products::LeftFactors;
use super::tensor::BitTensor;
use super::transcript::{
    BitRingSwitchProverTranscript, BitRingSwitchShape, BitRingSwitchVerifierTranscript,
    TranscriptWidth,
};
use crate::data::SumcheckData;
use crate::error::SumcheckError;
use crate::strategy::{Basis, FromTable, IntoTranscriptField, ReprSumcheckProver, VariableOrder};
use crate::transcript::{ProverTranscript, SumcheckShape};

/// Log of the hypercube points one task accumulates before its partial combines.
///
/// Large enough that the `d`-element accumulator and the block's own scaling are amortised,
/// and small enough both to keep every core fed at the heights this reduction runs at and to
/// keep the equality factor a block reads within cache.
#[cfg(not(test))]
const LOG_CHUNK: usize = PRODUCTION_LOG_CHUNK;

/// Small enough under test that a case above one variable splits in chunks.
///
/// The partial-combination fold is then on the path the tests take.
#[cfg(test)]
const LOG_CHUNK: usize = 1;

/// The value `LOG_CHUNK` takes outside test builds.
const PRODUCTION_LOG_CHUNK: usize = 14;

/// The hypercube points one task accumulates before its partial combines.
const CHUNK: usize = 1 << LOG_CHUNK;

/// Initial production compact depth. Benchmark controls can force another supported depth.
const COMPACT_ROUNDS: usize = 2;
/// The generic compact driver is intentionally bounded while its ranked shapes are measured.
const COMPACT_MAX_ROUNDS: usize = 6;
/// Keep at least this many equality blocks per bank in the production path.
const COMPACT_MIN_BLOCK_BITS: usize = 6;
/// The materializer keeps a fixed total EF scratch budget across all compact banks.
#[cfg(not(test))]
const COMPACT_TOTAL_SCRATCH: usize = 1 << 14;

/// Small enough under test that the materializer splits a tail in several chunks.
///
/// The bank offsets of every chunk past the first are then on the path the tests take.
#[cfg(test)]
const COMPACT_TOTAL_SCRATCH: usize = 1 << 4;

/// The fewest free variables an unforced compact run of `requested_k` rounds accepts.
///
/// Each of its `2^requested_k` banks then spans at least `2^COMPACT_MIN_BLOCK_BITS` equality
/// chunks of `2^log_chunk` points.
const fn compact_size_floor(requested_k: usize, log_chunk: usize) -> usize {
    requested_k + log_chunk + COMPACT_MIN_BLOCK_BITS
}

/// Bind prefix variables in one task-owned buffer without invoking Rayon recursively.
fn bind_scratch_prefix<R: Field>(values: &mut Vec<R>, challenges: &[R]) {
    for &challenge in challenges {
        let half = values.len() / 2;
        assert!(half > 0, "a compact prefix must leave a variable to bind");
        let (low, high) = values.split_at_mut(half);
        let (low_packed, low_tail) = R::Packing::pack_slice_with_suffix_mut(low);
        let (high_packed, high_tail) = R::Packing::pack_slice_with_suffix(high);
        let challenge_packed = R::Packing::broadcast(challenge);
        for (low, &high) in low_packed.iter_mut().zip(high_packed) {
            let old = *low;
            *low = old + (high - old) * challenge_packed;
        }
        for (low, &high) in low_tail.iter_mut().zip(high_tail) {
            *low += (high - *low) * challenge;
        }
        values.truncate(half);
    }
}

/// Materialize the witness after binding the compact head, using one fixed total scratch budget.
fn materialize_compact_packing<EF, R, S>(
    packing: &BitPacking<EF, S>,
    offset: usize,
    n: usize,
    compact_rounds: usize,
    challenges: &[EF],
) -> Poly<R>
where
    EF: TowerLevel + Send + Sync,
    R: IntoTranscriptField<EF> + Sync,
    S: Borrow<[EF]>,
{
    let _span = tracing::debug_span!("compact_packing").entered();
    let banks = 1usize << compact_rounds;
    let tail_evals = 1usize << (n - compact_rounds);
    let output_chunk = (COMPACT_TOTAL_SCRATCH >> compact_rounds).max(1);
    let prefix_challenges = challenges.iter().copied().map(R::from).collect::<Vec<_>>();
    let source = packing.poly().as_slice();
    let mut output = R::zero_vec(tail_evals);
    let output_chunk_len = output_chunk;
    output
        .par_chunks_mut(output_chunk)
        .enumerate()
        .for_each(|(chunk_index, output_chunk)| {
            let start = chunk_index * output_chunk_len;
            let mut scratch = Vec::with_capacity(banks * output_chunk.len());
            for bank in 0..banks {
                let source_start = offset + bank * tail_evals + start;
                scratch.extend_from_slice(&source[source_start..source_start + output_chunk.len()]);
            }
            let mut converted = R::from_table(scratch);
            bind_scratch_prefix(&mut converted, &prefix_challenges);
            output_chunk.copy_from_slice(&converted[..output_chunk.len()]);
        });
    Poly::new(output)
}

/// One ring-switching reduction at a bit alphabet, before the batching draw.
///
/// # Overview
///
/// The reduction turns a claim about a bit witness into one about its packing:
///
/// ```text
///     t(r) = s        a claim about the bits
///       ->  one degree-two sumcheck of at most l' rounds
///     t'(r') = s'     a claim about the packing, which a commitment answers
/// ```
///
/// # The order the protocol fixes
///
/// The batching challenge is drawn after the tensor element is bound.
/// So it cannot be a constructor argument.
///
/// ```text
///     new(r)                              tensor, incoming_claim
///     with_successor(r, a)                also successor_tensors, successor_claim
///     bind                                r, then every element, then draw r'' [and alpha]
///     batch(r'')                          initial_sum, closing_weight, weights
///     batch_with_successor(r'', alpha)    the same, when successor elements are sent
/// ```
///
/// Drawing `r''` first is unsound.
///
/// A bit matrix solving two `F_2`-linear systems moves the claim, sum held.
/// The rounds and the closing check then accept a true surviving claim.
///
/// # What the split settles, and what it leaves
///
/// It settles that `tensor` never needs the challenge.
/// Nothing is pushed into drawing one early just to obtain an element.
///
/// It does not settle the order.
///
/// The second stage borrows this one alone, so a caller can reach it first.
/// The forgery goes through with the element formed last.
///
/// Only a transcript binding `r` and the element before `r''` fixes that.
/// The two sides below own that transcript, so the ordering test is theirs.
///
/// # What it costs a verifier
///
/// Nothing witness-sized.
/// This stage holds the kept coordinates and one `d`-entry table.
///
/// Only `tensor`, `successor_tensors` and `weights` touch the `2^l'` equality table.
/// All three are prover-side, so each builds it rather than every caller paying.
///
/// # Why a bit alphabet is different
///
/// A coordinate is one bit, so a product by a coordinate is a conditional add.
///
/// Both the tensor accumulation and the weight multilinear are subset sums.
/// A general alphabet needs `d` multiplications per hypercube point instead.
///
/// # The successor view
///
/// The same reduction can also answer the witness one row on, the last row repeating.
/// The rows are the trailing `a` coordinates of the point; the selector leads it.
///
/// The successor weight splits at the packing boundary into three kept-side weights.
/// With `x(w)` the kept row bits of `w`:
///
/// ```text
///     eq      eq(r_high, w)                        the +1 settles inside the element
///     carry   eq(r_high, w - 1)  where x(w) != 0   the +1 ripples out of the element
///     last    eq(r_high, w)      where x(w) = max  the last row repeats
/// ```
///
/// The first is the tensor above, so the view adds two elements and no sumcheck.
/// Rows that fit in one element keep no row bits, and the view is a column reading alone.
///
/// ```text
///     s_next = sum_{v >= 1} eq_low[v - 1] * col_v(tensor)
///            + eq_low[d - 1] * ( col_0(carry) + col_{d-1}(last) )
/// ```
#[derive(Clone, Debug)]
pub struct BitRingSwitch<EF> {
    /// The evaluation point of the claim, over every variable of the witness.
    ///
    /// Held whole because the transcript binds it whole.
    point: Point<EF>,
    /// The equality table of the coordinates one packed element absorbs.
    eq_low: Poly<EF>,
    /// The trailing coordinates the successor view steps within, when there is one.
    successor: Option<usize>,
}

impl<EF: TowerLevel> BitRingSwitch<EF> {
    /// Number of coordinates one packed element absorbs.
    pub const ABSORBED: usize = Coefficients::<EF>::LOG_DIMENSION;

    /// Set up the reduction of a claim at one point.
    ///
    /// The batching challenge is not an argument.
    /// It belongs after the element is bound, which the second stage is for.
    ///
    /// # Arguments
    ///
    /// The evaluation point of the claim, over every variable of the witness.
    ///
    /// # Errors
    ///
    /// Returns an error when the point is narrower than one element absorbs.
    pub fn new(r: &Point<EF>) -> Result<Self, BitRingSwitchError> {
        if r.num_variables() < Self::ABSORBED {
            return Err(BitRingSwitchError::PointTooNarrow {
                needed: Self::ABSORBED,
                actual: r.num_variables(),
            });
        }

        // The packing keeps the leading coordinates and absorbs the rest.
        let (_, low) = r.split_at(r.num_variables() - Self::ABSORBED);

        Ok(Self {
            eq_low: Poly::new_from_point(low.as_slice(), EF::ONE),
            point: r.clone(),
            successor: None,
        })
    }

    /// Set up the reduction of a claim at one point, also answering its successor view.
    ///
    /// The successor view reads the witness one row on, the last row repeating.
    /// It steps within the trailing `row_variables` coordinates of the point.
    /// The coordinates before them select a column and never move.
    ///
    /// # Errors
    ///
    /// - The row count exceeds the coordinates the point names.
    /// - The point is narrower than one element absorbs.
    pub fn with_successor(r: &Point<EF>, row_variables: usize) -> Result<Self, BitRingSwitchError> {
        if row_variables > r.num_variables() {
            return Err(BitRingSwitchError::RowVariables {
                row_variables,
                num_variables: r.num_variables(),
            });
        }
        Ok(Self {
            successor: Some(row_variables),
            ..Self::new(r)?
        })
    }

    /// Variables the packed polynomial this reduction runs against must have.
    ///
    /// This is the maximum number of sumcheck rounds the reduction takes.
    /// Each leading Boolean selector coordinate selects a slot and removes one round.
    pub const fn num_variables(&self) -> usize {
        self.point.num_variables() - Self::ABSORBED
    }

    /// The coordinates of the evaluation point the packing keeps.
    ///
    /// They lead the point, so this is a borrow rather than a split.
    fn high(&self) -> &[EF] {
        &self.point.as_slice()[..self.num_variables()]
    }

    /// Row coordinates of the successor view the packing keeps, when there are any.
    ///
    /// Rows that fit in one element keep none, and the view is a column reading.
    fn kept_row_variables(&self) -> Option<usize> {
        self.successor
            .filter(|&rows| rows > Self::ABSORBED)
            .map(|rows| rows - Self::ABSORBED)
    }

    /// Whether the reduction sends the carry and last elements beside the tensor.
    ///
    /// True exactly for a successor view whose rows outrun one element.
    #[must_use]
    pub fn sends_successor_tensors(&self) -> bool {
        self.kept_row_variables().is_some()
    }

    /// The transcript description this reduction plays.
    fn shape(&self) -> BitRingSwitchShape {
        let num_variables = self.point.num_variables();
        match self.successor {
            Some(rows) if self.sends_successor_tensors() => {
                BitRingSwitchShape::with_successor_rows(num_variables, rows)
            }
            _ => BitRingSwitchShape::new(num_variables),
        }
    }

    /// Move on to the stage the batching challenge opens.
    ///
    /// # Errors
    ///
    /// - The reduction sends successor elements, which [`Self::batch_with_successor`] batches.
    /// - The challenge does not name the absorbed coordinates.
    pub fn batch<'a>(
        &'a self,
        r_batch: &Point<EF>,
    ) -> Result<BitRingSwitchBatch<'a, EF>, BitRingSwitchError> {
        if self.sends_successor_tensors() {
            return Err(BitRingSwitchError::SuccessorBatching { expected: true });
        }
        self.batch_stage(r_batch, None)
    }

    /// Move on to the stage the batching challenges open, successor elements included.
    ///
    /// `alpha` weighs the three elements' batched rows into one sum:
    ///
    /// ```text
    ///     rows(tensor) + alpha * rows(carry) + alpha^2 * rows(last)
    /// ```
    ///
    /// # Errors
    ///
    /// - The reduction sends no successor elements, which [`Self::batch`] batches.
    /// - The challenge does not name the absorbed coordinates.
    pub fn batch_with_successor<'a>(
        &'a self,
        r_batch: &Point<EF>,
        alpha: EF,
    ) -> Result<BitRingSwitchBatch<'a, EF>, BitRingSwitchError> {
        if !self.sends_successor_tensors() {
            return Err(BitRingSwitchError::SuccessorBatching { expected: false });
        }
        self.batch_stage(r_batch, Some(alpha))
    }

    /// The second stage over whatever the transcript drew, `alpha` present or not.
    fn batch_drawn(
        &self,
        r_batch: &Point<EF>,
        alpha: Option<EF>,
    ) -> Result<BitRingSwitchBatch<'_, EF>, BitRingSwitchError> {
        alpha.map_or_else(
            || self.batch(r_batch),
            |alpha| self.batch_with_successor(r_batch, alpha),
        )
    }

    /// The second stage over either kind of reduction, once its kind is checked.
    fn batch_stage(
        &self,
        r_batch: &Point<EF>,
        alpha: Option<EF>,
    ) -> Result<BitRingSwitchBatch<'_, EF>, BitRingSwitchError> {
        if r_batch.num_variables() != Self::ABSORBED {
            return Err(BitRingSwitchError::BatchWidthMismatch {
                expected: Self::ABSORBED,
                actual: r_batch.num_variables(),
            });
        }
        Ok(BitRingSwitchBatch {
            reduction: self,
            eq_batch: Poly::new_from_point(r_batch.as_slice(), EF::ONE),
            alpha,
        })
    }

    /// How many leading kept coordinates a Boolean prefix may fix.
    ///
    /// A successor view caps it at the selector.
    /// The equality factor of a Boolean row coordinate is an indicator too.
    /// Yet the carry reads the element one before, which that indicator would cut away.
    fn prefix_limit(&self) -> usize {
        let limit = self.num_variables();
        self.successor
            .map_or(limit, |rows| limit.min(self.point.num_variables() - rows))
    }

    /// The leading Boolean coordinates and the address they spell.
    ///
    /// Only the first [`Self::prefix_limit`] coordinates are scanned.
    fn fixed_prefix(&self) -> (usize, usize) {
        // Read the address from most significant bit to least significant bit.
        let mut prefix = 0;
        let mut address = 0usize;
        for &coordinate in &self.high()[..self.prefix_limit()] {
            // Zero selects the lower half of the remaining evaluation table.
            if coordinate == EF::ZERO {
                address <<= 1;
            // One selects its upper half.
            } else if coordinate == EF::ONE {
                address = (address << 1) | 1;
            // A field challenge needs an ordinary sumcheck round.
            } else {
                break;
            }
            // Count only coordinates whose values select a half exactly.
            prefix += 1;
        }
        // The pair identifies the selected slot without allocating its equality table.
        (prefix, address)
    }

    /// Where the equality table over the kept coordinates is nonzero, and its values there.
    ///
    /// # Algorithm
    ///
    /// The equality table factors over the variables.
    /// A coordinate that is already zero or one turns its factor into an indicator.
    ///
    /// ```text
    ///     high = (b_0 .. b_{p-1}, z_p .. z_{l'-1})   every b in {0, 1}
    ///     eq(high, w) = 0  unless the top p bits of w spell b
    /// ```
    ///
    /// The table is therefore supported on one run of `2^(l' - p)` consecutive elements,
    /// and on that run it is the equality table of the remaining coordinates alone.
    ///
    /// A claim about one column of a stacked trace arrives exactly this way, its slot
    /// address leading the row point.
    ///
    /// A point drawn from a transcript has no Boolean coordinate, so the run is the whole
    /// hypercube and this is the dense table.
    ///
    /// # Returns
    ///
    /// - The number of leading coordinates fixed to bits.
    /// - The first element of the supported run.
    /// - The equality table over that run, in the factored form its sweeps read.
    fn support(&self) -> (usize, usize, FactoredEquality<EF>) {
        // The equality table uses the first coordinate as the most significant index bit.
        // A Boolean prefix therefore selects one contiguous run.
        let (prefix, address) = self.fixed_prefix();

        // One element of the run per assignment of the coordinates that are left.
        let table = FactoredEquality::new(&self.high()[prefix..], LOG_CHUNK);
        (prefix, address * table.num_evals(), table)
    }

    /// The committed polynomial restricted to the Boolean prefix of the claim, held in `R`.
    ///
    /// A leading bit coordinate selects one half without a sumcheck round.
    /// Repeating that selection leaves exactly the slot the claim addresses.
    ///
    /// The slot is copied out of the packing, and the crossing into `R` runs over that copy.
    fn restricted_packing<R: FromTable<EF>, S: Borrow<[EF]>>(
        &self,
        packing: &BitPacking<EF, S>,
    ) -> Poly<R> {
        // The support identifies the same contiguous slot in both equality and witness order.
        let (prefix, address) = self.fixed_prefix();
        let len = 1usize << (self.num_variables() - prefix);
        let offset = address * len;

        // Only the selected slot feeds the sumcheck.
        // This avoids cloning and folding unrelated columns of a stacked trace.
        let values = &packing.poly().as_slice()[offset..offset + len];
        // The copy below is the only pass over the buffer, for a level handing back a zeroed
        // allocation rather than writing one element at a time. A level taking the trait's
        // default fills the slot serially first, at the width of the slot.
        let mut slot = EF::zero_vec(len);
        slot.par_chunks_mut(CHUNK)
            .zip(values.par_chunks(CHUNK))
            .for_each(|(slot, values)| slot.copy_from_slice(values));
        Poly::new(R::from_table(slot))
    }

    /// Restore the Boolean slot address in front of a point inside that slot.
    fn restore_prefix(&self, point: &Point<EF>) -> Point<EF> {
        // The prefix is public and fixed by the incoming evaluation point.
        let (prefix, _) = self.fixed_prefix();
        let mut coordinates = self.high()[..prefix].to_vec();

        // The random coordinates name the selected slot's remaining variables.
        coordinates.extend_from_slice(point.as_slice());
        Point::new(coordinates)
    }

    /// `sum_w eq(r_high, w) ⊗ t'(w)`, the element the prover sends.
    ///
    /// # Overview
    ///
    /// Read by columns it gives the bit planes at the kept coordinates.
    /// That is the reading which tests the incoming claim.
    ///
    /// Read by rows it gives the sumcheck its starting sum.
    ///
    /// The two readings are of the same coefficients.
    /// That is what makes a dishonest element catchable at all.
    ///
    /// # Performance
    ///
    /// Accumulated in place, one partial element per task.
    /// Forming each term separately would allocate `d` elements per point.
    ///
    /// The equality table is built here rather than held.
    /// That is strictly below the accumulation it feeds.
    ///
    /// It also keeps a verifier from paying for a table it never reads.
    ///
    /// Only the run the table is nonzero on is swept.
    /// A claim about one column of a stacked trace reaches `2^(l' - p)` of its `2^l'` elements.
    ///
    /// # Errors
    ///
    /// Returns an error unless the packing has the reduction's variables.
    pub fn tensor<S: Borrow<[EF]>>(
        &self,
        packing: &BitPacking<EF, S>,
    ) -> Result<BitTensor<EF>, BitRingSwitchError>
    where
        EF: Send + Sync,
    {
        self.check_width(packing.num_variables())?;
        let (_, offset, equality) = self.support();
        Ok(Self::tensor_over(packing, offset, &equality))
    }

    /// The element sent, accumulated against the equality table of the supported run.
    ///
    /// Elements outside the run weigh zero, so leaving them out changes no sum.
    ///
    /// # Algorithm
    ///
    /// The equality weight of a block is its own weight times the block's outer weight, and
    /// the first tensor leg carries the weight:
    ///
    /// ```text
    ///     sum_j (outer_i * inner_j) ⊗ t'(w)  =  (outer_i ⊗ 1) * sum_j inner_j ⊗ t'(w)
    /// ```
    ///
    /// So a block accumulates against the inner weights alone, and the outer weight is one
    /// scaling of the partial element rather than one multiplication per point.
    fn tensor_over<S: Borrow<[EF]>>(
        packing: &BitPacking<EF, S>,
        offset: usize,
        equality: &FactoredEquality<EF>,
    ) -> BitTensor<EF>
    where
        EF: Send + Sync,
    {
        let values = &packing.poly().as_slice()[offset..offset + equality.num_evals()];
        // Every block reads the same inner weights, so their layout is prepared once.
        let inner = LeftFactors::new(equality.inner());

        values
            .par_chunks(equality.block_len())
            .zip(equality.outer().par_iter())
            .par_fold_reduce(
                // The scratch is what a block accumulates into, so only a fold arm holds one.
                || (BitTensor::zero(), None),
                |(mut total, mut scratch), (values, &weight)| {
                    total.add_scaled_columns(&inner.sum(values, &mut scratch), weight);
                    (total, scratch)
                },
                |(mut total, scratch), (partial, _)| {
                    // Addition is associative, so regrouping cannot change it.
                    total += partial;
                    (total, scratch)
                },
            )
            .0
    }

    /// What the claim being reduced must equal, given the element sent.
    ///
    /// # Algorithm
    ///
    /// Column `v` is the witness's bit plane `v` read at the kept coordinates.
    /// The absorbed coordinates weigh those planes back together:
    ///
    /// ```text
    ///     t(r) = sum_v eq(r_low, v) * column v
    /// ```
    ///
    /// This is the reduction's only use of the absorbed coordinates.
    #[must_use]
    pub fn incoming_claim(&self, tensor: &BitTensor<EF>) -> EF {
        tensor
            .columns()
            .iter()
            .zip(self.eq_low.as_slice())
            .map(|(&column, &weight)| column * weight)
            .sum()
    }

    /// The carry and last elements the successor view adds, when its rows outrun one element.
    ///
    /// # Algorithm
    ///
    /// Over the run the equality table `E` is supported on, with `x(w)` the kept row bits:
    ///
    /// ```text
    ///     carry  =  sum_{x(w) != 0}   E[w - 1] ⊗ t'(w)    the element before, same column
    ///     last   =  sum_{x(w) = max}  E[w]     ⊗ t'(w)    the last element of each column
    /// ```
    ///
    /// The prefix never fixes a row coordinate, so `w - 1` stays inside the run.
    ///
    /// # Returns
    ///
    /// `None` unless the reduction sends successor elements.
    ///
    /// # Errors
    ///
    /// Returns an error unless the packing has the reduction's variables.
    pub fn successor_tensors<S: Borrow<[EF]>>(
        &self,
        packing: &BitPacking<EF, S>,
    ) -> Result<Option<SuccessorTensors<EF>>, BitRingSwitchError>
    where
        EF: Send + Sync,
    {
        self.check_width(packing.num_variables())?;
        if self.kept_row_variables().is_none() {
            return Ok(None);
        }
        let (_, offset, equality) = self.support();
        Ok(self.successor_tensors_over(packing, offset, &equality))
    }

    /// The successor elements, accumulated against the equality table of the supported run.
    ///
    /// `None` unless the reduction sends successor elements.
    fn successor_tensors_over<S: Borrow<[EF]>>(
        &self,
        packing: &BitPacking<EF, S>,
        offset: usize,
        equality: &FactoredEquality<EF>,
    ) -> Option<SuccessorTensors<EF>>
    where
        EF: Send + Sync,
    {
        let kept = self.kept_row_variables()?;

        // Elements outside the run weigh zero in all three elements.
        let block = equality.block_len();
        let values = &packing.poly().as_slice()[offset..offset + equality.num_evals()];
        // The run holds whole columns, so the kept row bits are the low bits of `w`.
        let column = 1usize << kept;
        let max = column - 1;
        let inner = equality.inner();

        // Past a block's first element, the element before lies in the block, and in the same
        // column unless the element starts one. Blocks and columns are both powers of two, so
        // that turns on the offset in the block alone: every block reads the same inner
        // weights shifted by one, which are therefore prepared once.
        let shifted = (0..block)
            .map(|j| if j & max == 0 { EF::ZERO } else { inner[j - 1] })
            .collect::<Vec<_>>();
        let shifted = LeftFactors::new(&shifted);

        let (carry, last, _) = values.par_chunks(block).enumerate().par_fold_reduce(
            // The scratch is what a block accumulates into, so only a fold arm holds one.
            || (BitTensor::zero(), BitTensor::zero(), None),
            |(mut carry, mut last, mut scratch), (index, values)| {
                let weight = equality.outer()[index];
                let start = (index * block) & max;
                // The +1 ripples out of the element before, inside the same column.
                // Before a block's first element, that closes the block before, under its weight.
                if start != 0 {
                    carry.add_exterior_product(equality.before_block(index), values[0]);
                }
                carry.add_scaled_columns(&shifted.sum(values, &mut scratch), weight);
                // The last element of a column reads itself again.
                // One element in a column qualifies, so its weight is formed directly.
                for j in (max - start..values.len()).step_by(column) {
                    last.add_exterior_product(weight * inner[j], values[j]);
                }
                (carry, last, scratch)
            },
            |(mut carry, mut last, scratch), (other_carry, other_last, _)| {
                // Addition is associative, so regrouping cannot change it.
                carry += other_carry;
                last += other_last;
                (carry, last, scratch)
            },
        );
        Some(SuccessorTensors { carry, last })
    }

    /// The weights the successor claim puts on the tensor's columns.
    ///
    /// # Algorithm
    ///
    /// With kept row coordinates, a column is read one bit on, and none reads bit 0:
    ///
    /// ```text
    ///     n[v] = eq_low[v - 1]   for v >= 1,   n[0] = 0
    /// ```
    ///
    /// The +1 out of bit `d - 1` goes to the carry element instead.
    ///
    /// With none, the `2^a` rows sit inside the element, `x = v mod 2^a`:
    ///
    /// ```text
    ///     n[v] = (x != 0 ? eq_low[v - 1] : 0) + (x = 2^a - 1 ? eq_low[v] : 0)
    /// ```
    fn successor_low_weights(&self, row_variables: usize) -> Vec<EF> {
        let eq_low = self.eq_low.as_slice();
        if self.sends_successor_tensors() {
            return (0..eq_low.len())
                .map(|v| if v == 0 { EF::ZERO } else { eq_low[v - 1] })
                .collect();
        }

        let last = (1usize << row_variables) - 1;
        (0..eq_low.len())
            .map(|v| {
                let row = v & last;
                let settled = if row == 0 { EF::ZERO } else { eq_low[v - 1] };
                let repeated = if row == last { eq_low[v] } else { EF::ZERO };
                settled + repeated
            })
            .collect()
    }

    /// What the successor claim must equal, given the elements sent.
    ///
    /// # Algorithm
    ///
    /// ```text
    ///     s_next = sum_v n[v] * col_v(tensor)
    ///            + eq_low[d - 1] * ( col_0(carry) + col_{d-1}(last) )   with successor elements
    /// ```
    ///
    /// `eq_low[d - 1]` is the product of the absorbed coordinates, the weight of the last bit.
    ///
    /// # Errors
    ///
    /// Returns an error when the reduction was set up without a successor view.
    ///
    /// # Panics
    ///
    /// Panics unless the successor elements are supplied exactly when the reduction sends them.
    pub fn successor_claim(
        &self,
        tensor: &BitTensor<EF>,
        successor: Option<&SuccessorTensors<EF>>,
    ) -> Result<EF, BitRingSwitchError> {
        let rows = self.successor.ok_or(BitRingSwitchError::NoSuccessorView)?;
        assert_eq!(
            successor.is_some(),
            self.sends_successor_tensors(),
            "the successor elements are supplied exactly when the reduction sends them"
        );

        let settled: EF = tensor
            .columns()
            .iter()
            .zip(self.successor_low_weights(rows))
            .map(|(&column, weight)| column * weight)
            .sum();
        Ok(successor.map_or(settled, |elements| {
            let d = BitTensor::<EF>::DIMENSION;
            let ripple = elements.carry.column(0) + elements.last.column(d - 1);
            settled + self.eq_low.as_slice()[d - 1] * ripple
        }))
    }

    /// Check that something names the variables this reduction runs over.
    const fn check_width(&self, actual: usize) -> Result<(), BitRingSwitchError> {
        if actual == self.num_variables() {
            Ok(())
        } else {
            Err(BitRingSwitchError::WidthMismatch {
                expected: self.num_variables(),
                actual,
            })
        }
    }

    /// The equality element at the kept coordinates and one other point.
    ///
    /// # Algorithm
    ///
    /// The equality polynomial factors over the variables:
    ///
    /// ```text
    ///     eq(X, Y) = prod_i ( X_i*Y_i + (1 - X_i)(1 - Y_i) )
    /// ```
    ///
    /// Each factor lands on a different tensor leg.
    /// From the identity, every variable adds an agree and a disagree term:
    ///
    /// ```text
    ///     agree     scale a copy's columns by a, its rows by b
    ///     disagree  scale the element's columns by 1 - a, its rows by 1 - b
    /// ```
    ///
    /// The two scalings commute, so either order gives the term.
    /// The alternative sums the element over the hypercube, exponentially.
    fn equality_element(&self, r_prime: &Point<EF>) -> BitTensor<EF> {
        // Boolean leading coordinates select a committed slot directly.
        // The equality element therefore spans only the coordinates left inside that slot.
        let (prefix, _) = self.fixed_prefix();
        let mut element = BitTensor::one();
        for (&a, &b) in self.high()[prefix..].iter().zip(r_prime.as_slice()) {
            let mut agree = element.clone();
            agree.scale_columns(a);
            agree.scale_rows(b);

            element.scale_columns(EF::ONE - a);
            element.scale_rows(EF::ONE - b);
            element += agree;
        }
        element
    }

    /// The carry and last weights lifted into the algebra at one other point.
    ///
    /// # Algorithm
    ///
    /// Split the coordinates past the prefix, and the point, into selector and kept rows:
    ///
    /// ```text
    ///     high[p..] = (selector, rho_h)       r' = (r'_sel, r'_rows)       |rho_h| = kept
    /// ```
    ///
    /// Both weights factor into a selector part, an equality factor per coordinate,
    /// and a row part:
    ///
    /// ```text
    ///     carry  =  successor_element(rho_h, r'_rows)   * prod_i (1 + sel_i ⊗ 1 + 1 ⊗ r'_sel,i)
    ///     last   =  prod(rho_h) ⊗ prod(r'_rows)         * prod_i (1 + sel_i ⊗ 1 + 1 ⊗ r'_sel,i)
    /// ```
    ///
    /// # Panics
    ///
    /// Panics unless the reduction sends successor elements.
    fn successor_elements(&self, r_prime: &Point<EF>) -> (BitTensor<EF>, BitTensor<EF>) {
        let kept = self
            .kept_row_variables()
            .expect("only a reduction sending successor elements closes on them");
        let (prefix, _) = self.fixed_prefix();
        let rest = &self.high()[prefix..];
        let (selector, rho) = rest.split_at(rest.len() - kept);
        let (r_selector, r_rows) = r_prime.as_slice().split_at(r_prime.num_variables() - kept);

        let mut carry = BitTensor::successor_element(rho, r_rows);
        let mut last = BitTensor::exterior_product(
            rho.iter().copied().product(),
            r_rows.iter().copied().product(),
        );
        for (&a, &b) in selector.iter().zip(r_selector) {
            carry.mul_equality_factor(a, b);
            last.mul_equality_factor(a, b);
        }
        (carry, last)
    }

    /// Common compact-depth guards, evaluated before bank slices or shifts are formed.
    fn compact_depth_is_eligible(&self, n: usize, requested_k: usize, force: bool) -> bool {
        if requested_k == 0 || requested_k > COMPACT_MAX_ROUNDS || n < requested_k {
            return false;
        }
        if let Some(kept_rows) = self.kept_row_variables()
            && (n < kept_rows || requested_k > n - kept_rows)
        {
            return false;
        }
        // At the production depth this floor is 2 + 14 + 6 = 22 free variables, but 2 + 1 + 6 = 9
        // under test, where `LOG_CHUNK` is 1, so no unit test runs the compact head at the shape
        // or the chunk size a production build first accepts.
        force || n >= compact_size_floor(requested_k, LOG_CHUNK)
    }
}

/// The same reduction, once the batching challenge has been drawn.
///
/// Holding one of these is not evidence that the element was bound first.
/// Enforcing that is the transcript's job, as the stage above records.
#[derive(Clone, Debug)]
pub struct BitRingSwitchBatch<'a, EF> {
    /// The stage the evaluation point alone fixes.
    reduction: &'a BitRingSwitch<EF>,
    /// The equality table of the batching challenge.
    eq_batch: Poly<EF>,
    /// The challenge weighing the carry and last elements, when the reduction sends them.
    alpha: Option<EF>,
}

impl<EF: TowerLevel> BitRingSwitchBatch<'_, EF> {
    /// The weight multilinear the sumcheck runs against the packing.
    ///
    /// # Algorithm
    ///
    /// The equality table decomposes over `F_2`.
    /// The challenge weighs those coordinates:
    ///
    /// ```text
    ///     A(w) = sum_u eq(u, r_batch) * coordinate u of eq(r_high, w)
    /// ```
    ///
    /// The coordinates are bits, so each entry is a subset sum.
    ///
    /// With successor elements the carry and last weights are the same table, moved:
    ///
    /// ```text
    ///     weights(w) = A(w)
    ///                + alpha   * (x(w) != 0  ? A(w - 1) : 0)
    ///                + alpha^2 * (x(w) = max ? A(w)     : 0)
    /// ```
    ///
    /// # Performance
    ///
    /// Prover-side, like the element passes, the other places the equality table is built.
    ///
    /// The Boolean prefix selects one slot before the sumcheck starts.
    /// The table therefore contains that slot alone, with no zero runs around it.
    pub fn weights(&self) -> Poly<EF>
    where
        EF: Send + Sync,
    {
        let (_, _, equality) = self.reduction.support();
        self.weights_over(&equality)
    }

    /// The weight multilinear, read off the equality table of the supported run, held in `R`.
    ///
    /// Every entry is a subset sum of the batching table, so carrying that table into `R`
    /// first puts the whole multilinear there for the cost of `d` conversions.
    ///
    /// The entries are sums and products of those images, so `R` must carry the arithmetic
    /// of `EF` and not merely its elements.
    fn weights_over<R>(&self, equality: &FactoredEquality<EF>) -> Poly<R>
    where
        EF: Send + Sync,
        R: IntoTranscriptField<EF> + Sync,
    {
        let mut table = Poly::zero(equality.num_variables());
        let batching: Vec<R> = self
            .eq_batch
            .as_slice()
            .iter()
            .copied()
            .map(R::from)
            .collect();
        let sums = CoordinateSums::<EF, R>::new(&batching);

        // Every entry belongs to the selected slot.
        // No zero run for another slot is allocated or folded.
        //
        // A block's entries are its own weights under one outer weight, and the coordinate
        // sums absorb that scale once per block instead of once per entry.
        //
        // The block is that amortisation unit and the split follows it, so a run holding
        // fewer blocks than there are threads sweeps on fewer tasks. Splitting below the
        // block does not recover them: a task spanning less than a block has to scale the
        // sums for itself, and one scaling costs more than the entries such a task writes.
        table
            .as_mut_slice()
            .par_chunks_mut(equality.block_len())
            .zip(equality.outer().par_iter())
            .for_each_init(
                // Scratch of the right shape, overwritten before each block reads it.
                || sums.clone(),
                |scaled, (block, &weight)| {
                    scaled_sums_into(scaled, &sums, weight);
                    scaled.apply_into(equality.inner(), block);
                },
            );

        let Some(alpha) = self.alpha else {
            return table;
        };
        let kept = self
            .reduction
            .kept_row_variables()
            .expect("a batch carries alpha only for a reduction sending successor elements");
        let column = 1usize << kept;
        let max = column - 1;
        let alpha = R::from(alpha);
        let alpha_squared = alpha.square();

        // The shift reads the entry before, which a column taken from its last row down still
        // finds unmoved. So the combination lands in the table it reads, in chunks of whole
        // columns.
        //
        // A column fits inside the run: `prefix_limit` lets no Boolean prefix fix a row
        // coordinate, so the table keeps at least the kept rows. Every chunk below is
        // therefore exactly `stride` long and splits into whole columns.
        let stride = equality.block_len().max(column).min(table.num_evals());
        debug_assert!(column.is_power_of_two() && stride.is_multiple_of(column));
        debug_assert!(table.num_evals().is_multiple_of(stride));
        table
            .as_mut_slice()
            .par_chunks_mut(stride)
            .for_each(|part| {
                for entries in part.chunks_mut(column) {
                    for row in (0..column).rev() {
                        let settled = entries[row];
                        let mut value = settled;
                        if row != 0 {
                            value += alpha * entries[row - 1];
                        }
                        if row == max {
                            value += alpha_squared * settled;
                        }
                        entries[row] = value;
                    }
                }
            });
        table
    }

    /// Build the tail weights from the compact prover's settled coordinate images.
    fn weights_from_generators<R>(
        &self,
        equality: &FactoredEquality<EF>,
        generators: &[R],
    ) -> Poly<R>
    where
        EF: Send + Sync,
        R: IntoTranscriptField<EF> + Sync,
    {
        let sums = CoordinateSums::<EF, R>::new(generators);
        let mut table = Poly::zero(equality.num_variables());
        table
            .as_mut_slice()
            .par_chunks_mut(equality.block_len())
            .zip(equality.outer().par_iter())
            .for_each_init(
                || sums.clone(),
                |scaled, (block, &weight)| {
                    scaled_sums_into(scaled, &sums, weight);
                    scaled.apply_into(equality.inner(), block);
                },
            );

        let Some(alpha) = self.alpha else {
            return table;
        };
        let kept = self
            .reduction
            .kept_row_variables()
            .expect("a compact alpha batch has successor rows");
        let column = 1usize << kept;
        let alpha = R::from(alpha);
        let alpha_squared = alpha.square();
        let max = column - 1;
        // A column fits inside the tail: the kept-row clause of `compact_depth_is_eligible` keeps
        // the compact head off the kept row coordinates, so the tail keeps at least the kept rows.
        // Every chunk below is therefore exactly `stride` long and splits into whole columns.
        let stride = equality.block_len().max(column).min(table.num_evals());
        debug_assert!(
            column.is_power_of_two() && stride.is_multiple_of(column),
            "the compact tail is shorter than one column; see compact_depth_is_eligible"
        );
        debug_assert!(
            table.num_evals().is_multiple_of(stride),
            "the compact tail does not split into whole strides; see compact_depth_is_eligible"
        );
        table
            .as_mut_slice()
            .par_chunks_mut(stride)
            .for_each(|part| {
                for entries in part.chunks_mut(column) {
                    for row in (0..column).rev() {
                        let settled = entries[row];
                        let mut value = settled;
                        if row != 0 {
                            value += alpha * entries[row - 1];
                        }
                        if row == max {
                            value += alpha_squared * settled;
                        }
                        entries[row] = value;
                    }
                }
            });
        table
    }

    /// The sum the reduction's sumcheck starts from.
    ///
    /// Derived from the elements' rows rather than taken from the prover.
    /// That ties the sumcheck to the coefficients the claims were checked on.
    ///
    /// ```text
    ///     rows(tensor) [+ alpha * rows(carry) + alpha^2 * rows(last)]
    /// ```
    ///
    /// # Panics
    ///
    /// Panics unless the successor elements are supplied exactly when the batch carries `alpha`.
    #[must_use]
    pub fn initial_sum(
        &self,
        tensor: &BitTensor<EF>,
        successor: Option<&SuccessorTensors<EF>>,
    ) -> EF {
        let sum = self.batch_rows(tensor);
        match (self.alpha, successor) {
            (None, None) => sum,
            (Some(alpha), Some(elements)) => {
                sum + self.successor_rows(alpha, &elements.carry, &elements.last)
            }
            _ => panic!("the successor elements are supplied exactly when the batch carries alpha"),
        }
    }

    /// The weight the surviving claim is scaled by, where the rounds ended.
    ///
    /// # Overview
    ///
    /// This is the weight multilinear read at that point.
    /// Taken through the equality element, not by a pass over the weights.
    ///
    /// # Errors
    ///
    /// Returns an error unless the point names the reduction's variables.
    pub fn closing_weight(&self, r_prime: &Point<EF>) -> Result<EF, BitRingSwitchError> {
        // The rounds run only over coordinates not fixed by the Boolean slot address.
        let (prefix, _) = self.reduction.fixed_prefix();
        self.reduction
            .check_width(prefix + r_prime.num_variables())?;
        let equality = self.batch_rows(&self.reduction.equality_element(r_prime));
        Ok(self.alpha.map_or(equality, |alpha| {
            let (carry, last) = self.reduction.successor_elements(r_prime);
            equality + self.successor_rows(alpha, &carry, &last)
        }))
    }

    /// The rows of an element, batched against the batching challenge.
    ///
    /// Both ends of the sumcheck are this operation on a different element.
    fn batch_rows(&self, tensor: &BitTensor<EF>) -> EF {
        tensor
            .rows()
            .iter()
            .zip(self.eq_batch.as_slice())
            .map(|(&row, &weight)| row * weight)
            .sum()
    }

    /// `alpha * rows(carry) + alpha^2 * rows(last)`, the successor share of either end.
    fn successor_rows(&self, alpha: EF, carry: &BitTensor<EF>, last: &BitTensor<EF>) -> EF {
        alpha * (self.batch_rows(carry) + alpha * self.batch_rows(last))
    }
}

/// Reasons a bit-alphabet reduction cannot be set up or run.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum BitRingSwitchError {
    /// The evaluation point does not cover the coordinates one element absorbs.
    #[error("the evaluation point names {actual} variables, at least {needed} are needed")]
    PointTooNarrow {
        /// Coordinates one packed element absorbs.
        needed: usize,
        /// Coordinates the point names.
        actual: usize,
    },
    /// The batching challenge does not name the absorbed coordinates exactly.
    ///
    /// A shorter one would zip-truncate, dropping identity rows.
    /// Both sides move the same way, so no later check would catch it.
    #[error("the batching challenge names {actual} variables, expected {expected}")]
    BatchWidthMismatch {
        /// Coordinates the reduction absorbs.
        expected: usize,
        /// Coordinates the challenge names.
        actual: usize,
    },
    /// A polynomial or point does not name the variables the reduction uses.
    #[error("the reduction runs over {expected} variables, got {actual}")]
    WidthMismatch {
        /// Variables the reduction runs over.
        expected: usize,
        /// Variables the argument names.
        actual: usize,
    },
    /// The successor view steps within more coordinates than the point names.
    #[error("the successor view names {row_variables} row variables, the point {num_variables}")]
    RowVariables {
        /// Row coordinates the successor view was set up with.
        row_variables: usize,
        /// Coordinates the evaluation point names.
        num_variables: usize,
    },
    /// A batching stage for the other kind of reduction was asked for.
    #[error(
        "the batching stage does not fit the reduction, successor challenge expected: {expected}"
    )]
    SuccessorBatching {
        /// Whether the reduction sends successor elements, so its stage takes `alpha`.
        expected: bool,
    },
    /// A successor claim was asked of a reduction set up without a successor view.
    #[error("the reduction was set up without a successor view")]
    NoSuccessorView,
}

/// The two elements a successor view adds when its rows outrun one packed element.
///
/// With `E` the equality table over the kept coordinates and `x(w)` the kept row bits:
///
/// ```text
///     carry  =  sum_{x(w) != 0}   E[w - 1] ⊗ t'(w)
///     last   =  sum_{x(w) = max}  E[w]     ⊗ t'(w)
/// ```
///
/// The claim reads one column of each, the rounds read all their rows.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(serialize = "EF: TowerLevel", deserialize = "EF: TowerLevel"))]
pub struct SuccessorTensors<EF> {
    /// The +1 rippling out of an element into the next one of its column.
    pub carry: BitTensor<EF>,
    /// The last element of each column, whose last row repeats.
    pub last: BitTensor<EF>,
}

/// The messages one bit-alphabet reduction puts on the wire.
///
/// The element travels by rows, one bit per matrix entry:
///
/// ```text
///     by rows          d elements
///     byte per entry   d^2 elements
/// ```
///
/// A successor view whose rows outrun one element sends two more, the same way.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "EF: TowerLevel", deserialize = "EF: TowerLevel"))]
pub struct BitRingSwitchProof<EF> {
    /// The tensor element both checks read, by rows and by columns.
    pub tensor: BitTensor<EF>,
    /// The carry and last elements, present exactly when the setup sends them.
    ///
    /// The verifier derives whether they belong from its own setup, never from this field.
    pub successor: Option<SuccessorTensors<EF>>,
    /// The batched degree-two rounds left after any Boolean slot prefix is fixed.
    pub sumcheck: SumcheckData<EF, EF>,
    /// The value of the surviving claim.
    pub final_eval: EF,
}

/// Why a bit-alphabet reduction was rejected.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum BitRingSwitchProofError {
    /// A list crossing the wire is not the width the description fixes.
    #[error(transparent)]
    Width(#[from] TranscriptWidth),

    /// The reduction could not be run over the point it was set up at.
    #[error(transparent)]
    Reduction(#[from] BitRingSwitchError),

    /// The claimed evaluation is not the column reading of the element.
    #[error("the claimed evaluation is not the column reading of the tensor element")]
    ClaimMismatch,

    /// The claimed successor evaluation is not the column reading of the elements.
    #[error("the claimed successor evaluation is not the column reading of the elements")]
    SuccessorClaimMismatch,

    /// Neither a current nor a successor evaluation was supplied, so nothing is checked.
    #[error("no claimed evaluation was supplied")]
    NoReading,

    /// A successor evaluation was supplied to a reduction without a successor view.
    #[error("a successor evaluation was supplied to a reduction without a successor view")]
    NoSuccessorView,

    /// A round of the batched sumcheck failed.
    #[error(transparent)]
    Sumcheck(#[from] SumcheckError),

    /// The surviving claim does not close the sumcheck against the closing weight.
    #[error("the surviving claim does not close the sumcheck")]
    FinalCheck,

    /// The sumcheck carries grinding witnesses this reduction never searches for.
    #[error("the sumcheck carries {actual} proof-of-work witnesses, expected none")]
    NonEmptyPowWitnesses {
        /// Witnesses the proof supplied.
        actual: usize,
    },
}

/// The two sides of the reduction, each over the transcript the other replays.
///
/// # A reduction, not a filter
///
/// A false input claim is not rejected outright.
/// It survives as a false surviving claim, except with the probability below.
///
/// Two cases the rounds cannot catch are left to whatever discharges the claim:
///
/// - a closing weight of zero, which constrains the surviving value not at all
/// - a tampered element, with the rest of the proof adapted to the sum it implies
///
/// # Booleanity is free
///
/// The packing is a bijection between bit strings and elements of the level.
/// Every bit pattern is an element, and every element is some bit pattern.
///
/// A commitment to a packed multilinear is therefore a commitment to a bit witness.
/// No range check, no auxiliary constraint, and nothing here to verify.
///
/// # Soundness
///
/// The reduction's own error is `(d_log + K - 1 + 2 l') / |EF|` (eprint 2024/504, Theorem 3.5).
/// Here `l'` is the number of rounds that actually run after a Boolean prefix is fixed.
/// Charging the full packed arity is a conservative upper bound.
///
/// `K` is the number of elements sent: three with successor elements, one otherwise.
///
/// The three sources are:
///
/// - `d_log / |EF|` from the batching point that collapses each element's row claims.
/// - `(K - 1) / |EF|` from `alpha`, which collapses the `K` batched row claims into one.
/// - `2 l' / |EF|` for the rounds of degree-two sumcheck.
///
/// A wrong element leaves a nonzero batched row claim.
/// It is multilinear in `r''` and of degree `K - 1` in `alpha`.
///
/// That bound needs `r''` and `alpha` drawn after every element is bound.
/// The transcript fixes that order.
/// It also needs `K` taken from the public setup, never read from the proof.
///
/// Every term is per-attempt, because the description holds no grinding step.
/// A protocol needing a total bound supplies the grinding outside this run.
impl<EF: TranscriptField + TowerLevel> BitRingSwitch<EF> {
    /// Reduce the claims this reduction was set up over to one about the packing.
    ///
    /// # The field the rounds run in
    ///
    /// Every round multiplies a packing entry by a weight, and both tables are witness-sized.
    /// `R` is the representation those two tables and the rounds over them are held in.
    ///
    /// ```text
    ///     once per reduction   the restricted packing, the batching table, alpha,
    ///                          the initial claim, and the surviving claim on its way back
    ///     once per round       the challenge, and the round message on its way back
    ///     never                the tensor elements, the equality table, the point
    /// ```
    ///
    /// The weight multilinear is accumulated in `R` from the batching table, so it is
    /// built there rather than carried across.
    ///
    /// The map is a field isomorphism, so each round message is the one the challenge
    /// field's own tables measure and the transcript is unchanged.
    ///
    /// # Returns
    ///
    /// The proof, the point the rounds ended at, and the surviving claim's value.
    /// Any Boolean slot prefix is restored in front of the random coordinates.
    ///
    /// # Panics
    ///
    /// Panics unless the packing has the variables the evaluation point leaves.
    pub fn prove<R, Challenger, S>(
        &self,
        packing: &BitPacking<EF, S>,
        challenger: &mut Challenger,
    ) -> (BitRingSwitchProof<EF>, Point<EF>, EF)
    where
        EF: Send + Sync,
        R: IntoTranscriptField<EF>,
        Challenger: FieldChallenger<EF> + GrindingChallenger<Witness = EF>,
        S: Borrow<[EF]>,
    {
        self.prove_with_compact_depth::<R, Challenger, S>(
            packing,
            challenger,
            COMPACT_ROUNDS,
            false,
        )
    }

    /// Run the compact driver at a requested depth. The `force` flag is test/benchmark-only and
    /// bypasses the conservative production size gate while retaining algebraic eligibility.
    pub(crate) fn prove_with_compact_depth<R, Challenger, S>(
        &self,
        packing: &BitPacking<EF, S>,
        challenger: &mut Challenger,
        requested_k: usize,
        force: bool,
    ) -> (BitRingSwitchProof<EF>, Point<EF>, EF)
    where
        EF: Send + Sync,
        R: IntoTranscriptField<EF>,
        Challenger: FieldChallenger<EF> + GrindingChallenger<Witness = EF>,
        S: Borrow<[EF]>,
    {
        let max_rounds = self.num_variables();
        assert_eq!(
            packing.num_variables(),
            max_rounds,
            "the packing must have the {max_rounds} variables the evaluation point leaves"
        );

        // Derive the selected slot before constructing any equality table. Compact runs only
        // need the suffix factor; the dense fallback constructs the full supported-run factor.
        let (prefix, address) = self.fixed_prefix();
        let n = max_rounds - prefix;
        let compact = self.compact_depth_is_eligible(n, requested_k, force);
        let offset = address * (1usize << n);
        let equality = (!compact).then(|| FactoredEquality::new(&self.high()[prefix..], LOG_CHUNK));
        let compact_equality =
            compact.then(|| FactoredEquality::new(&self.high()[prefix + requested_k..], LOG_CHUNK));
        let (tensor, successor, bank_tensors, bank_successors) = if compact {
            let _span = tracing::debug_span!("compact_stats", depth = requested_k).entered();
            let banks = 1usize << requested_k;
            let head = &self.high()[prefix..prefix + requested_k];
            let tail_equality = compact_equality.as_ref().expect("compact equality exists");
            let head_equality = Poly::new_from_point(head, EF::ONE);
            let bank_tensors = (0..banks)
                .map(|bank| {
                    Self::tensor_over(
                        packing,
                        offset + bank * tail_equality.num_evals(),
                        tail_equality,
                    )
                })
                .collect::<Vec<_>>();
            let bank_successors = self.kept_row_variables().map(|_| {
                (0..banks)
                    .map(|bank| {
                        self.successor_tensors_over(
                            packing,
                            offset + bank * tail_equality.num_evals(),
                            tail_equality,
                        )
                        .expect("eligible successor compact bank has successor tensors")
                    })
                    .collect::<Vec<_>>()
            });
            let mut tensor = BitTensor::zero();
            for (&weight, bank) in head_equality.as_slice().iter().zip(&bank_tensors) {
                tensor.add_scaled_columns(bank, weight);
            }
            let successor = bank_successors.as_ref().map(|banks| {
                let mut carry = BitTensor::zero();
                let mut last = BitTensor::zero();
                for (&weight, bank) in head_equality.as_slice().iter().zip(banks) {
                    carry.add_scaled_columns(&bank.carry, weight);
                    last.add_scaled_columns(&bank.last, weight);
                }
                SuccessorTensors { carry, last }
            });
            (tensor, successor, Some(bank_tensors), bank_successors)
        } else {
            (
                Self::tensor_over(
                    packing,
                    offset,
                    equality.as_ref().expect("dense equality exists"),
                ),
                self.successor_tensors_over(
                    packing,
                    offset,
                    equality.as_ref().expect("dense equality exists"),
                ),
                None,
                None,
            )
        };

        // The elements are functions of the kept coordinates alone.
        // They are therefore ready before the transcript needs them.
        let mut transcript =
            BitRingSwitchProverTranscript::<Challenger, EF>::new(challenger, self.shape());
        let (r_batch, alpha) = transcript.statement(
            &self.point,
            tensor.rows(),
            successor
                .as_ref()
                .map(|elements| (elements.carry.rows(), elements.last.rows())),
        );

        // The batching challenges arrived after the elements, which is the order soundness needs.
        let batch = self
            .batch_drawn(&r_batch, alpha)
            .expect("the transcript draws what the reduction's kind batches with");
        let mut sumcheck = SumcheckData::default();

        // A Boolean prefix is a public slot address. The compact branch keeps the first
        // requested coordinates in banked tensors and emits them through one n-round transcript.
        let (r_prime, final_eval) = if compact {
            let factors_span =
                tracing::debug_span!("compact_factors", depth = requested_k).entered();
            let bank_tensors = bank_tensors.expect("compact banks are present");
            let bank_successors = bank_successors.as_ref();
            let banks = 1usize << requested_k;
            let head = &self.high()[prefix..prefix + requested_k];
            let tail_equality = compact_equality.as_ref().expect("compact equality exists");
            let head_equality = Poly::new_from_point(head, EF::ONE);
            let mut batching = Vec::with_capacity(Coefficients::<EF>::DIMENSION);
            batching.extend(batch.eq_batch.as_slice().iter().copied().map(R::from));
            let coordinate_sums = CoordinateSums::<EF, R>::new(&batching);
            let mut factors = Vec::with_capacity(banks * BitTensor::<EF>::DIMENSION);
            let alpha = batch.alpha.map(R::from);
            let alpha_squared = alpha.map(|value| value.square());
            for bank in 0..banks {
                let successor = bank_successors.map(|all| &all[bank]);
                for coordinate in 0..BitTensor::<EF>::DIMENSION {
                    let mut value = R::from(bank_tensors[bank].rows()[coordinate]);
                    if let (Some(elements), Some(alpha), Some(alpha_squared)) =
                        (successor, alpha, alpha_squared)
                    {
                        value += alpha * R::from(elements.carry.rows()[coordinate])
                            + alpha_squared * R::from(elements.last.rows()[coordinate]);
                    }
                    factors.push(value);
                }
            }
            let mut generators = Vec::with_capacity(banks * BitTensor::<EF>::DIMENSION);
            for &weight in head_equality.as_slice() {
                for coordinate in 0..BitTensor::<EF>::DIMENSION {
                    let mut basis = Coefficients::<EF>::zero();
                    basis.set(coordinate);
                    generators.push(coordinate_sums.sum(weight * basis.element()));
                }
            }
            drop(bank_tensors);
            drop(factors_span);
            transcript.batched_sumcheck(|challenger| {
                let mut rounds_transcript = ProverTranscript::<Challenger, EF, EF>::new(
                    challenger,
                    SumcheckShape::new(n, 0, Basis::Evaluation),
                );
                let head_span = tracing::debug_span!("compact_head", depth = requested_k).entered();
                let mut compact_prover = ReprSumcheckProver::<EF, EF, R>::from_repr_tables(
                    VariableOrder::Prefix,
                    Poly::new(factors),
                    Poly::new(generators),
                    batch.initial_sum(&tensor, successor.as_ref()),
                );
                let mut challenges = Vec::with_capacity(n);
                for _ in 0..requested_k {
                    challenges.push(compact_prover.round(&mut sumcheck, &mut rounds_transcript));
                }
                let handoff_sum = compact_prover.claimed_sum();
                let generators = compact_prover.weights().into_evals();
                drop(compact_prover);
                drop(head_span);

                let bound_packing = materialize_compact_packing::<EF, R, _>(
                    packing,
                    offset,
                    n,
                    requested_k,
                    &challenges,
                );
                let bound_weights = {
                    let _span =
                        tracing::debug_span!("compact_weights", depth = requested_k).entered();
                    batch.weights_from_generators::<R>(tail_equality, &generators)
                };
                let _span = tracing::debug_span!("compact_tail", depth = requested_k).entered();
                let mut tail_prover = ReprSumcheckProver::<EF, EF, R>::from_repr_tables(
                    VariableOrder::Prefix,
                    bound_packing,
                    bound_weights,
                    handoff_sum,
                );
                for _ in requested_k..n {
                    challenges.push(tail_prover.round(&mut sumcheck, &mut rounds_transcript));
                }
                let final_eval = tail_prover.evals().as_slice()[0].into_transcript();
                rounds_transcript.finish();
                (Point::new(challenges), final_eval)
            })
        } else {
            let restricted = self.restricted_packing::<R, _>(packing);
            let rounds = restricted.num_variables();
            let weights =
                batch.weights_over::<R>(equality.as_ref().expect("dense equality exists"));
            let mut prover = ReprSumcheckProver::<EF, EF, R>::from_repr_tables(
                VariableOrder::Prefix,
                restricted,
                weights,
                batch.initial_sum(&tensor, successor.as_ref()),
            );
            let r_prime = transcript.batched_sumcheck(|challenger| {
                prover.compute_sumcheck_polynomials(&mut sumcheck, challenger, rounds, 0)
            });
            let final_eval = prover.evals().as_slice()[0].into_transcript();
            (r_prime, final_eval)
        };

        // After the last round the evaluation side has folded to the packing at that point.
        // No second pass over the packing is needed to find it.
        transcript.surviving_claim(final_eval);
        transcript.finish();

        (
            BitRingSwitchProof {
                tensor,
                successor,
                sumcheck,
                final_eval,
            },
            self.restore_prefix(&r_prime),
            final_eval,
        )
    }

    /// Replay the reduction against the current claim and return the claim it leaves behind.
    ///
    /// The same as [`Self::verify_readings`] with the current reading alone.
    ///
    /// # Errors
    ///
    /// As [`Self::verify_readings`].
    pub fn verify<Challenger>(
        &self,
        proof: &BitRingSwitchProof<EF>,
        claimed_sum: EF,
        challenger: &mut Challenger,
    ) -> Result<(Point<EF>, EF), BitRingSwitchProofError>
    where
        Challenger: FieldChallenger<EF> + GrindingChallenger<Witness = EF>,
    {
        self.verify_readings(proof, Some(claimed_sum), None, challenger)
    }

    /// Replay the reduction and return the claim it leaves behind, as a point and a value.
    ///
    /// Discharging that pair against a commitment to the packing is the caller's business.
    ///
    /// # Arguments
    ///
    /// - The proof.
    /// - The claimed evaluation of the witness at the point, when it is to be checked.
    /// - The claimed evaluation of its successor view there, when it is to be checked.
    /// - The challenger, advanced only once every structural check passes.
    ///
    /// # Errors
    ///
    /// Before the challenger is touched:
    ///
    /// - A malformed element.
    /// - Successor elements whose presence disagrees with the setup.
    /// - A non-empty grinding witness list, since this reduction never grinds.
    /// - No reading supplied, or a successor reading the setup does not answer.
    ///
    /// After it:
    ///
    /// - A claimed evaluation disagreeing with the elements' columns.
    /// - A failed sumcheck round, or a final claim that does not close it.
    pub fn verify_readings<Challenger>(
        &self,
        proof: &BitRingSwitchProof<EF>,
        current: Option<EF>,
        next: Option<EF>,
        challenger: &mut Challenger,
    ) -> Result<(Point<EF>, EF), BitRingSwitchProofError>
    where
        Challenger: FieldChallenger<EF> + GrindingChallenger<Witness = EF>,
    {
        // Every structural rejection runs before the challenger is touched.
        // A malformed proof therefore never leaves a half-advanced transcript.
        self.check_structure(proof, current, next)?;

        // The verifier derives the slot restriction from the public incoming point.
        // Proof data cannot choose how many rounds are expected.
        let (prefix, _) = self.fixed_prefix();
        let rounds = self.num_variables() - prefix;
        let mut transcript =
            BitRingSwitchVerifierTranscript::<Challenger, EF>::new(challenger, self.shape());
        let (r_batch, alpha) = transcript.statement(
            &self.point,
            proof.tensor.rows(),
            proof
                .successor
                .as_ref()
                .map(|elements| (elements.carry.rows(), elements.last.rows())),
        )?;

        if let Err(error) = self.check_readings(proof, current, next) {
            transcript.abort();
            return Err(error);
        }

        let batch = self
            .batch_drawn(&r_batch, alpha)
            .expect("the transcript draws what the reduction's kind batches with");

        // The initial sum is derived from the elements' rows, never taken from the prover.
        // That is what makes a dishonest element catchable at all.
        let mut sum = batch.initial_sum(&proof.tensor, proof.successor.as_ref());
        let replay = transcript.batched_sumcheck(|challenger| {
            proof
                .sumcheck
                .verify_rounds(challenger, &mut sum, rounds, 0, Basis::Evaluation)
        });
        let r_prime = match replay {
            Ok(point) => point,
            Err(error) => {
                transcript.abort();
                return Err(error.into());
            }
        };

        transcript.surviving_claim(proof.final_eval);
        transcript.finish();

        // The rounds close on the weight multilinear at their end point, times the value.
        // The weight comes through the lifted elements rather than another pass.
        let closing = batch.closing_weight(&r_prime)?;
        if sum != closing * proof.final_eval {
            return Err(BitRingSwitchProofError::FinalCheck);
        }

        Ok((self.restore_prefix(&r_prime), proof.final_eval))
    }

    /// Every rejection that needs neither the transcript nor a claim check.
    fn check_structure(
        &self,
        proof: &BitRingSwitchProof<EF>,
        current: Option<EF>,
        next: Option<EF>,
    ) -> Result<(), BitRingSwitchProofError> {
        // Every element is a square bit matrix, whichever of them the proof carries.
        let successor = proof
            .successor
            .iter()
            .flat_map(|elements| [&elements.carry, &elements.last]);
        if let Some(malformed) = core::iter::once(&proof.tensor)
            .chain(successor)
            .find(|element| !element.is_well_formed())
        {
            return Err(TranscriptWidth::TensorRows {
                expected: BitTensor::<EF>::DIMENSION,
                actual: malformed.rows().len(),
            }
            .into());
        }
        // The element count is the setup's, never the proof's.
        let expected = self.sends_successor_tensors();
        if proof.successor.is_some() != expected {
            return Err(
                TranscriptWidth::successor_elements(expected, proof.successor.is_some()).into(),
            );
        }
        if !proof.sumcheck.pow_witnesses.is_empty() {
            return Err(BitRingSwitchProofError::NonEmptyPowWitnesses {
                actual: proof.sumcheck.pow_witnesses.len(),
            });
        }
        if current.is_none() && next.is_none() {
            return Err(BitRingSwitchProofError::NoReading);
        }
        if next.is_some() && self.successor.is_none() {
            return Err(BitRingSwitchProofError::NoSuccessorView);
        }
        Ok(())
    }

    /// Check each supplied reading against the elements' columns, current before next.
    fn check_readings(
        &self,
        proof: &BitRingSwitchProof<EF>,
        current: Option<EF>,
        next: Option<EF>,
    ) -> Result<(), BitRingSwitchProofError> {
        // The columns are the witness's bit planes at the kept coordinates.
        // The absorbed coordinates weigh them back together, their only use here.
        if current.is_some_and(|claim| self.incoming_claim(&proof.tensor) != claim) {
            return Err(BitRingSwitchProofError::ClaimMismatch);
        }
        // The successor reads the same planes one bit on, and the successor elements' edges.
        if let Some(claim) = next
            && self.successor_claim(&proof.tensor, proof.successor.as_ref())? != claim
        {
            return Err(BitRingSwitchProofError::SuccessorClaimMismatch);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use p3_binary_field::{BinaryChallenger, BinaryField16, BinaryField128, Ghash128};
    use p3_challenger::{CanSample, HashChallenger};
    use p3_field::{Field, PrimeCharacteristicRing};
    use p3_keccak::Keccak256Hash;
    use proptest::prelude::*;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::*;

    type EF = BinaryField16;
    type Chal = BinaryChallenger<EF, HashChallenger<u8, Keccak256Hash, 32>>;
    type WideChal = BinaryChallenger<BinaryField128, HashChallenger<u8, Keccak256Hash, 32>>;

    /// A fresh sponge, so the two sides start from the same state.
    fn challenger() -> Chal {
        Chal::from_hasher(Vec::new(), Keccak256Hash)
    }

    /// A fresh wide-field sponge for representation differential checks.
    fn wide_challenger() -> WideChal {
        WideChal::from_hasher(Vec::new(), Keccak256Hash)
    }

    /// A random bit witness of the given byte length.
    fn bits(seed: u64, bytes: usize) -> Vec<u8> {
        let mut rng = SmallRng::seed_from_u64(seed);
        (0..bytes).map(|_| rng.random::<u8>()).collect()
    }

    /// One reduction over a witness of the given byte length.
    ///
    /// The batching challenge comes back beside it rather than folded in.
    /// A test then reaches the second stage the way the protocol does.
    fn fixture(
        seed: u64,
        bytes: usize,
    ) -> (BitRingSwitch<EF>, BitPacking<EF>, Point<EF>, Point<EF>) {
        let packing = BitPacking::<EF>::new(&bits(seed, bytes)).unwrap();
        let mut rng = SmallRng::seed_from_u64(seed ^ 0xFFFF);

        // The whole point, of which the packing keeps the leading coordinates.
        let r = Point::<EF>::rand(
            &mut rng,
            packing.num_variables() + BitRingSwitch::<EF>::ABSORBED,
        );
        let r_batch = Point::<EF>::rand(&mut rng, BitRingSwitch::<EF>::ABSORBED);
        let reduction = BitRingSwitch::new(&r).unwrap();

        (reduction, packing, r, r_batch)
    }

    fn assert_successor_proofs_match_dense(
        reduction: &BitRingSwitch<EF>,
        packing: &BitPacking<EF>,
        requested_k: usize,
    ) {
        let tensor = reduction.tensor(packing).unwrap();
        let successor = reduction.successor_tensors(packing).unwrap();
        let current = reduction.incoming_claim(&tensor);
        let next = reduction
            .successor_claim(&tensor, successor.as_ref())
            .unwrap();
        let mut dense_challenger = challenger();
        let dense =
            reduction.prove_with_compact_depth::<EF, _, _>(packing, &mut dense_challenger, 0, true);
        let dense_next: EF = dense_challenger.sample();
        let mut requested_challenger = challenger();
        let requested = reduction.prove_with_compact_depth::<EF, _, _>(
            packing,
            &mut requested_challenger,
            requested_k,
            true,
        );
        let requested_next: EF = requested_challenger.sample();
        assert_eq!(requested.0.tensor, dense.0.tensor);
        assert_eq!(requested.0.successor, dense.0.successor);
        assert_eq!(
            requested.0.sumcheck.polynomial_evaluations,
            dense.0.sumcheck.polynomial_evaluations
        );
        assert_eq!(
            requested.0.sumcheck.pow_witnesses,
            dense.0.sumcheck.pow_witnesses
        );
        assert_eq!(requested.0.final_eval, dense.0.final_eval);
        assert_eq!(requested.1, dense.1);
        assert_eq!(requested.2, dense.2);
        assert_eq!(requested_next, dense_next);
        assert!(
            reduction
                .verify_readings(&requested.0, Some(current), Some(next), &mut challenger())
                .is_ok()
        );
    }

    /// The witness as a multilinear over every variable, one element per bit.
    fn embedded(witness: &[u8]) -> Poly<EF> {
        embedded_over(witness)
    }

    /// The same embedding over any level the bits are read into.
    fn embedded_over<K: Field>(witness: &[u8]) -> Poly<K> {
        Poly::new(
            (0..witness.len() * 8)
                .map(|cell| {
                    if (witness[cell / 8] >> (cell % 8)) & 1 == 1 {
                        K::ONE
                    } else {
                        K::ZERO
                    }
                })
                .collect::<Vec<_>>(),
        )
    }

    #[test]
    fn compact_prefix_fold_matches_the_existing_prefix_kernel() {
        let mut rng = SmallRng::seed_from_u64(0xC0A7);
        for (variables, rounds) in [(0, 0), (1, 1), (4, 2), (7, 3), (8, 4)] {
            let mut rng_challenges = SmallRng::seed_from_u64(0xF01D + variables as u64);
            let challenges = if rounds >= 2 {
                vec![BinaryField128::ZERO, BinaryField128::ONE]
                    .into_iter()
                    .chain((2..rounds).map(|_| rng_challenges.random::<BinaryField128>()))
                    .collect::<Vec<_>>()
            } else {
                (0..rounds)
                    .map(|_| rng_challenges.random::<BinaryField128>())
                    .collect::<Vec<_>>()
            };
            let mut values: Vec<BinaryField128> = (0..(1usize << variables))
                .map(|_| rng.random::<BinaryField128>())
                .collect();
            let mut reference = Poly::new(values.clone());
            for &challenge in &challenges {
                reference.fix_prefix_var_mut(challenge);
            }
            bind_scratch_prefix(&mut values, &challenges);
            assert_eq!(values, reference.into_evals());
        }
    }

    #[test]
    fn compact_composed_weight_map_matches_dense_at_zero_and_one_heads() {
        let packing = BitPacking::<EF>::new(&bits(0xC01F, 16)).unwrap();
        let mut rng = SmallRng::seed_from_u64(0xD01F);
        let reduction = BitRingSwitch::new(&Point::<EF>::rand(
            &mut rng,
            packing.num_variables() + BitRingSwitch::<EF>::ABSORBED,
        ))
        .unwrap();
        let r_batch = Point::<EF>::rand(&mut rng, BitRingSwitch::<EF>::ABSORBED);
        let batch = reduction.batch(&r_batch).unwrap();
        let dense_equality = FactoredEquality::new(reduction.high(), LOG_CHUNK);
        let dense_weights = batch.weights_over::<EF>(&dense_equality);
        let head = &reduction.high()[..2];
        let head_equality = Poly::new_from_point(head, EF::ONE);
        let tail_equality = FactoredEquality::new(&reduction.high()[2..], LOG_CHUNK);
        let batching = batch.eq_batch.as_slice().to_vec();
        let coordinate_sums = CoordinateSums::<EF, EF>::new(&batching);

        for challenges in [
            [EF::ZERO, EF::ZERO],
            [EF::ZERO, EF::ONE],
            [EF::ONE, EF::ZERO],
            [EF::ONE, EF::ONE],
        ] {
            let lambda = Poly::new_from_point(&challenges, EF::ONE);
            let mut generators = Vec::with_capacity(BitTensor::<EF>::DIMENSION);
            for coordinate in 0..BitTensor::<EF>::DIMENSION {
                let mut basis = Coefficients::<EF>::zero();
                basis.set(coordinate);
                let mut generator = EF::ZERO;
                for bank in 0..4 {
                    generator += lambda.as_slice()[bank]
                        * coordinate_sums.sum(head_equality.as_slice()[bank] * basis.element());
                }
                generators.push(generator);
            }
            let compact_weights = batch.weights_from_generators(&tail_equality, &generators);
            let mut folded_dense = dense_weights.clone();
            for &challenge in &challenges {
                folded_dense.fix_prefix_var_mut(challenge);
            }
            assert_eq!(compact_weights, folded_dense);
        }
    }

    #[test]
    fn forced_compact_depths_match_the_dense_transcript() {
        for k in [1, 2, 4, 6] {
            for remaining_n in [k, k + 1] {
                let bytes = 2usize << remaining_n;
                let packing =
                    BitPacking::<EF>::new(&bits(0xC011 + k as u64 + remaining_n as u64, bytes))
                        .unwrap();
                let mut rng = SmallRng::seed_from_u64(0xD00D + k as u64 + remaining_n as u64);
                let mut point = Point::<EF>::rand(
                    &mut rng,
                    packing.num_variables() + BitRingSwitch::<EF>::ABSORBED,
                )
                .as_slice()
                .to_vec();
                point[0] = loop {
                    let value = rng.random::<EF>();
                    if value != EF::ZERO && value != EF::ONE {
                        break value;
                    }
                };
                let reduction = BitRingSwitch::new(&Point::new(point)).unwrap();
                assert_eq!(packing.num_variables(), remaining_n);
                assert_eq!(reduction.fixed_prefix().0, 0);
                assert!(reduction.compact_depth_is_eligible(remaining_n, k, true));
                let current = reduction.incoming_claim(&reduction.tensor(&packing).unwrap());
                let mut dense_challenger = challenger();
                let (dense, dense_point, dense_value) = reduction
                    .prove_with_compact_depth::<EF, _, _>(&packing, &mut dense_challenger, 0, true);
                let dense_next: EF = dense_challenger.sample();
                let mut compact_challenger = challenger();
                let (compact, compact_point, compact_value) = reduction
                    .prove_with_compact_depth::<EF, _, _>(
                        &packing,
                        &mut compact_challenger,
                        k,
                        true,
                    );
                let compact_next: EF = compact_challenger.sample();

                assert_eq!(compact.tensor, dense.tensor, "k={k}, n={remaining_n}");
                assert_eq!(compact.successor, dense.successor, "k={k}, n={remaining_n}");
                assert_eq!(
                    compact.sumcheck.polynomial_evaluations, dense.sumcheck.polynomial_evaluations,
                    "k={k}, n={remaining_n}"
                );
                assert_eq!(
                    compact.sumcheck.pow_witnesses, dense.sumcheck.pow_witnesses,
                    "k={k}, n={remaining_n}"
                );
                assert_eq!(compact_point, dense_point, "k={k}, n={remaining_n}");
                assert_eq!(compact_value, dense_value, "k={k}, n={remaining_n}");
                assert_eq!(
                    compact.final_eval, dense.final_eval,
                    "k={k}, n={remaining_n}"
                );
                assert_eq!(compact_next, dense_next, "k={k}, n={remaining_n}");
                assert!(
                    reduction
                        .verify(&compact, current, &mut challenger())
                        .is_ok()
                );
            }
        }

        // The common guard must fall back before attempting a two-bank slice when only one
        // free coordinate remains.
        let packing = BitPacking::<EF>::new(&bits(0xC012, 4)).unwrap();
        let mut rng = SmallRng::seed_from_u64(0xD012);
        let reduction = BitRingSwitch::new(&Point::<EF>::rand(
            &mut rng,
            packing.num_variables() + BitRingSwitch::<EF>::ABSORBED,
        ))
        .unwrap();
        let mut dense_challenger = challenger();
        let dense = reduction.prove_with_compact_depth::<EF, _, _>(
            &packing,
            &mut dense_challenger,
            0,
            true,
        );
        let mut compact_challenger = challenger();
        let compact = reduction.prove_with_compact_depth::<EF, _, _>(
            &packing,
            &mut compact_challenger,
            2,
            true,
        );
        assert_eq!(
            dense.0.sumcheck.polynomial_evaluations,
            compact.0.sumcheck.polynomial_evaluations
        );
        assert_eq!(dense.1, compact.1);
        assert_eq!(dense.2, compact.2);
    }

    #[test]
    fn compact_prefix_uses_a_nonzero_selected_slot() {
        let packing = BitPacking::<EF>::new(&bits(0xC016, 64)).unwrap();
        let mut rng = SmallRng::seed_from_u64(0xD016);
        let mut coordinates = Point::<EF>::rand(
            &mut rng,
            packing.num_variables() + BitRingSwitch::<EF>::ABSORBED,
        )
        .as_slice()
        .to_vec();
        coordinates[0] = EF::ONE;
        let reduction = BitRingSwitch::new(&Point::new(coordinates)).unwrap();
        let current = reduction.incoming_claim(&reduction.tensor(&packing).unwrap());
        let mut dense_challenger = challenger();
        let dense = reduction.prove_with_compact_depth::<EF, _, _>(
            &packing,
            &mut dense_challenger,
            0,
            true,
        );
        let dense_next: EF = dense_challenger.sample();
        let mut compact_challenger = challenger();
        let compact = reduction.prove_with_compact_depth::<EF, _, _>(
            &packing,
            &mut compact_challenger,
            2,
            true,
        );
        let compact_next: EF = compact_challenger.sample();
        assert_eq!(
            reduction.fixed_prefix().1,
            1,
            "the test must select the upper slot"
        );
        assert_eq!(compact.0.tensor, dense.0.tensor);
        assert_eq!(compact.0.successor, dense.0.successor);
        assert_eq!(
            compact.0.sumcheck.polynomial_evaluations,
            dense.0.sumcheck.polynomial_evaluations
        );
        assert_eq!(
            compact.0.sumcheck.pow_witnesses,
            dense.0.sumcheck.pow_witnesses
        );
        assert_eq!(compact.0.final_eval, dense.0.final_eval);
        assert_eq!(compact.1, dense.1);
        assert_eq!(compact.2, dense.2);
        assert_eq!(compact_next, dense_next);
        assert!(
            reduction
                .verify(&compact.0, current, &mut challenger())
                .is_ok()
        );
    }

    #[test]
    fn successor_compact_depth_refuses_fixed_selector_prefixes() {
        let n = 4;
        let kept_rows = 2;
        let row_variables = BitRingSwitch::<EF>::ABSORBED + kept_rows;
        let total = n + BitRingSwitch::<EF>::ABSORBED;
        let mut rng = SmallRng::seed_from_u64(0xD017);
        let mut non_boolean = || loop {
            let value = rng.random::<EF>();
            if value != EF::ZERO && value != EF::ONE {
                break value;
            }
        };

        // One fixed selector leaves only one free selector, so requested k2 is unsafe.
        let mut partial = vec![EF::ZERO; total];
        partial[0] = EF::ONE;
        partial[1] = non_boolean();
        partial[2..n].fill_with(&mut non_boolean);
        let partial_reduction =
            BitRingSwitch::with_successor(&Point::new(partial), row_variables).unwrap();
        let partial_prefix = partial_reduction.fixed_prefix().0;
        assert_eq!(partial_prefix, 1);
        assert!(!partial_reduction.compact_depth_is_eligible(n - partial_prefix, 2, true,));
        let fallback_packing = BitPacking::<EF>::new(&bits(0xD01B, 32)).unwrap();
        assert_successor_proofs_match_dense(&partial_reduction, &fallback_packing, 2);

        // Fixing every selector leaves only the kept rows, and therefore no compact head.
        let mut all = vec![EF::ZERO; total];
        all[..n - kept_rows].fill(EF::ONE);
        all[n - kept_rows..n].fill_with(&mut non_boolean);
        let all_reduction = BitRingSwitch::with_successor(&Point::new(all), row_variables).unwrap();
        let all_prefix = all_reduction.fixed_prefix().0;
        assert_eq!(all_prefix, n - kept_rows);
        assert!(!all_reduction.compact_depth_is_eligible(n - all_prefix, 1, true));
        assert_successor_proofs_match_dense(&all_reduction, &fallback_packing, 2);

        // A successor contained inside one packed element still takes the plain path and must
        // refuse a requested head wider than the one remaining variable.
        let inside_packing = BitPacking::<EF>::new(&bits(0xC018, 4)).unwrap();
        let inside_point = Point::<EF>::rand(
            &mut rng,
            inside_packing.num_variables() + BitRingSwitch::<EF>::ABSORBED,
        );
        let inside =
            BitRingSwitch::with_successor(&inside_point, BitRingSwitch::<EF>::ABSORBED - 1)
                .unwrap();
        assert!(!inside.compact_depth_is_eligible(inside.num_variables(), 2, true));
        assert_successor_proofs_match_dense(&inside, &inside_packing, 2);

        // A kept-row width above the test equality block still has complete columns in each
        // bank and must take the compact successor path.
        let wide_packing = BitPacking::<EF>::new(&bits(0xC019, 32)).unwrap();
        let mut wide_point = Point::<EF>::rand(
            &mut rng,
            wide_packing.num_variables() + BitRingSwitch::<EF>::ABSORBED,
        )
        .as_slice()
        .to_vec();
        wide_point[0] = loop {
            let value = rng.random::<EF>();
            if value != EF::ZERO && value != EF::ONE {
                break value;
            }
        };
        let wide = BitRingSwitch::with_successor(
            &Point::new(wide_point),
            BitRingSwitch::<EF>::ABSORBED + 2,
        )
        .unwrap();
        assert_eq!(wide.fixed_prefix().0, 0);
        assert!(wide.compact_depth_is_eligible(wide.num_variables(), 2, true));
        let wide_tensor = wide.tensor(&wide_packing).unwrap();
        let wide_successor = wide.successor_tensors(&wide_packing).unwrap();
        let wide_current = wide.incoming_claim(&wide_tensor);
        let wide_next = wide
            .successor_claim(&wide_tensor, wide_successor.as_ref())
            .unwrap();
        let mut wide_dense_challenger = challenger();
        let wide_dense = wide.prove_with_compact_depth::<EF, _, _>(
            &wide_packing,
            &mut wide_dense_challenger,
            0,
            true,
        );
        let mut wide_compact_challenger = challenger();
        let wide_compact = wide.prove_with_compact_depth::<EF, _, _>(
            &wide_packing,
            &mut wide_compact_challenger,
            2,
            true,
        );
        assert_eq!(wide_compact.0.successor, wide_dense.0.successor);
        assert_eq!(
            wide_compact.0.sumcheck.polynomial_evaluations,
            wide_dense.0.sumcheck.polynomial_evaluations
        );
        assert!(
            wide.verify_readings(
                &wide_compact.0,
                Some(wide_current),
                Some(wide_next),
                &mut challenger(),
            )
            .is_ok()
        );
    }

    #[test]
    fn compact_plain_path_handles_zero_one_and_sparse_packing() {
        let mut sparse = vec![0u8; 16];
        sparse[0] = 1;
        sparse[9] = 0x80;
        for (seed, witness) in [
            (0xD018, vec![0u8; 16]),
            (0xD019, vec![0xFFu8; 16]),
            (0xD01A, sparse),
        ] {
            let packing = BitPacking::<EF>::new(&witness).unwrap();
            let mut rng = SmallRng::seed_from_u64(seed);
            let reduction = BitRingSwitch::new(&Point::<EF>::rand(
                &mut rng,
                packing.num_variables() + BitRingSwitch::<EF>::ABSORBED,
            ))
            .unwrap();
            let mut dense_challenger = challenger();
            let dense = reduction.prove_with_compact_depth::<EF, _, _>(
                &packing,
                &mut dense_challenger,
                0,
                true,
            );
            let mut compact_challenger = challenger();
            let compact = reduction.prove_with_compact_depth::<EF, _, _>(
                &packing,
                &mut compact_challenger,
                2,
                true,
            );
            assert_eq!(compact.0.tensor, dense.0.tensor);
            assert_eq!(
                compact.0.sumcheck.polynomial_evaluations,
                dense.0.sumcheck.polynomial_evaluations
            );
            assert_eq!(
                compact.0.sumcheck.pow_witnesses,
                dense.0.sumcheck.pow_witnesses
            );
            assert_eq!(compact.1, dense.1);
            assert_eq!(compact.2, dense.2);
        }
    }

    #[test]
    fn compact_successor_and_ghash_paths_match_dense() {
        let packing = BitPacking::<EF>::new(&bits(0xC013, 16)).unwrap();
        let mut rng = SmallRng::seed_from_u64(0xD013);
        let reduction = BitRingSwitch::with_successor(
            &Point::<EF>::rand(
                &mut rng,
                packing.num_variables() + BitRingSwitch::<EF>::ABSORBED,
            ),
            BitRingSwitch::<EF>::ABSORBED + 1,
        )
        .unwrap();
        let tensor = reduction.tensor(&packing).unwrap();
        let successor = reduction.successor_tensors(&packing).unwrap();
        let current = reduction.incoming_claim(&tensor);
        let next = reduction
            .successor_claim(&tensor, successor.as_ref())
            .unwrap();
        let mut dense_challenger = challenger();
        let (dense, dense_point, dense_value) = reduction.prove_with_compact_depth::<EF, _, _>(
            &packing,
            &mut dense_challenger,
            0,
            true,
        );
        let mut compact_challenger = challenger();
        let (compact, compact_point, compact_value) = reduction
            .prove_with_compact_depth::<EF, _, _>(&packing, &mut compact_challenger, 2, true);
        assert_eq!(compact.successor, dense.successor);
        assert_eq!(
            compact.sumcheck.polynomial_evaluations,
            dense.sumcheck.polynomial_evaluations
        );
        assert_eq!(compact_point, dense_point);
        assert_eq!(compact_value, dense_value);
        assert!(
            reduction
                .verify_readings(&compact, Some(current), Some(next), &mut challenger())
                .is_ok()
        );

        for (bytes, requested_k, has_tail) in [(64, 2, false), (64, 1, true), (128, 2, true)] {
            let wide_packing =
                BitPacking::<BinaryField128>::new(&bits(0xC014 + bytes as u64, bytes)).unwrap();
            let mut wide_rng = SmallRng::seed_from_u64(0xD014 + bytes as u64);
            let mut wide_point = Point::<BinaryField128>::rand(
                &mut wide_rng,
                wide_packing.num_variables() + BitRingSwitch::<BinaryField128>::ABSORBED,
            )
            .as_slice()
            .to_vec();
            wide_point[0] = loop {
                let value = wide_rng.random::<BinaryField128>();
                if value != BinaryField128::ZERO && value != BinaryField128::ONE {
                    break value;
                }
            };
            let wide_reduction = BitRingSwitch::new(&Point::new(wide_point)).unwrap();
            assert_eq!(wide_reduction.fixed_prefix().0, 0);
            assert!(wide_packing.num_variables() >= requested_k);
            assert_eq!(wide_packing.num_variables() > requested_k, has_tail);
            assert!(wide_reduction.compact_depth_is_eligible(
                wide_packing.num_variables(),
                requested_k,
                true,
            ));
            let mut wide_dense_challenger = wide_challenger();
            let (wide_dense, wide_dense_point, wide_dense_value) = wide_reduction
                .prove_with_compact_depth::<BinaryField128, _, _>(
                    &wide_packing,
                    &mut wide_dense_challenger,
                    0,
                    true,
                );
            let mut wide_compact_challenger = wide_challenger();
            let (wide_compact, wide_compact_point, wide_compact_value) = wide_reduction
                .prove_with_compact_depth::<Ghash128, _, _>(
                    &wide_packing,
                    &mut wide_compact_challenger,
                    requested_k,
                    true,
                );
            let wide_dense_next: BinaryField128 = wide_dense_challenger.sample();
            let wide_compact_next: BinaryField128 = wide_compact_challenger.sample();
            assert_eq!(wide_compact.tensor, wide_dense.tensor);
            assert_eq!(wide_compact.successor, wide_dense.successor);
            assert_eq!(
                wide_compact.sumcheck.polynomial_evaluations,
                wide_dense.sumcheck.polynomial_evaluations
            );
            assert_eq!(
                wide_compact.sumcheck.pow_witnesses,
                wide_dense.sumcheck.pow_witnesses
            );
            assert_eq!(wide_compact_point, wide_dense_point);
            assert_eq!(wide_compact_value, wide_dense_value);
            assert_eq!(wide_compact.final_eval, wide_dense.final_eval);
            assert_eq!(wide_compact_next, wide_dense_next);
            let wide_current = wide_reduction.incoming_claim(&wide_compact.tensor);
            assert!(
                wide_reduction
                    .verify(&wide_compact, wide_current, &mut wide_challenger())
                    .is_ok()
            );
        }
    }

    #[test]
    fn production_gate_uses_compact_two_rounds_at_the_minimum_shape() {
        let packing = BitPacking::<EF>::new(&bits(0xC015, 1024)).unwrap();
        let mut rng = SmallRng::seed_from_u64(0xD015);
        let mut point = Point::<EF>::rand(
            &mut rng,
            packing.num_variables() + BitRingSwitch::<EF>::ABSORBED,
        )
        .as_slice()
        .to_vec();
        point[0] = loop {
            let value = rng.random::<EF>();
            if value != EF::ZERO && value != EF::ONE {
                break value;
            }
        };
        let reduction = BitRingSwitch::new(&Point::new(point)).unwrap();
        // A production build first accepts 22 free variables, and a test build 9.
        assert_eq!(compact_size_floor(COMPACT_ROUNDS, PRODUCTION_LOG_CHUNK), 22);
        assert_eq!(compact_size_floor(COMPACT_ROUNDS, LOG_CHUNK), 9);
        assert_eq!(reduction.num_variables(), 9);
        assert!(reduction.compact_depth_is_eligible(
            reduction.num_variables(),
            COMPACT_ROUNDS,
            false,
        ));
        assert!(!reduction.compact_depth_is_eligible(
            reduction.num_variables() - 1,
            COMPACT_ROUNDS,
            false,
        ));
        let mut dense_challenger = challenger();
        let dense = reduction.prove_with_compact_depth::<EF, _, _>(
            &packing,
            &mut dense_challenger,
            0,
            true,
        );
        let mut production_challenger = challenger();
        let production = reduction.prove::<EF, _, _>(&packing, &mut production_challenger);
        assert_eq!(
            production.0.sumcheck.polynomial_evaluations,
            dense.0.sumcheck.polynomial_evaluations
        );
        assert_eq!(production.1, dense.1);
        assert_eq!(production.2, dense.2);
    }

    #[test]
    fn the_reduction_runs_over_the_variables_the_packing_keeps() {
        // Fixture state: 128 cells is 7 variables, of which 16 bits absorb 4.
        let (reduction, packing, _, _) = fixture(0x5AE, 16);

        assert_eq!(BitRingSwitch::<EF>::ABSORBED, 4);
        assert_eq!(reduction.num_variables(), 3);
        assert_eq!(packing.num_variables(), 3);
    }

    /// The equality table over the kept coordinates, dense over the whole hypercube.
    ///
    /// The production path computes the same table without the run of zeros a Boolean
    /// coordinate puts in it, so this is what every case below checks it against.
    fn dense_eq_high(reduction: &BitRingSwitch<EF>) -> Poly<EF> {
        Poly::new_from_point(
            &reduction.point.as_slice()[..reduction.num_variables()],
            EF::ONE,
        )
    }

    #[test]
    fn the_incoming_claim_is_the_witness_at_the_point() {
        // Invariant: what the reduction demands of the claim is the witness's.
        //
        //     sum_v eq(r_low, v) * column v  ==  t(r)
        //
        // That identity is what the reduction's first check rests on.
        let witness = bits(0xC1A1, 16);
        let packing = BitPacking::<EF>::new(&witness).unwrap();
        let mut rng = SmallRng::seed_from_u64(0xF001);
        let r = Point::<EF>::rand(&mut rng, 7);
        let reduction = BitRingSwitch::new(&r).unwrap();

        let tensor = reduction.tensor(&packing).unwrap();

        assert_eq!(
            reduction.incoming_claim(&tensor),
            embedded(&witness).eval_base(&r)
        );
    }

    #[test]
    fn the_columns_are_the_bit_planes_at_the_kept_coordinates() {
        // Invariant: column `v` is cells `d*w + v` read at `r_high`.
        // Checked column by column, so a failure localises.
        let (reduction, packing, _, _) = fixture(0xB1A, 16);
        let eq = dense_eq_high(&reduction);

        for (v, &column) in reduction
            .tensor(&packing)
            .unwrap()
            .columns()
            .iter()
            .enumerate()
        {
            let expected: EF = (0..packing.len())
                .filter(|&w| packing.coefficients(w).get(v))
                .map(|w| eq.as_slice()[w])
                .sum();
            assert_eq!(column, expected, "column {v}");
        }
    }

    #[test]
    fn each_row_is_its_own_weighted_sum() {
        // Invariant: row `u` is the packing weighted by coordinate `u` of eq.
        //
        //     row u  =  sum_w A_{w,u} * t'(w)
        let (reduction, packing, _, _) = fixture(0x0A5, 16);
        let eq = dense_eq_high(&reduction);

        for (u, &row) in reduction
            .tensor(&packing)
            .unwrap()
            .rows()
            .iter()
            .enumerate()
        {
            let expected: EF = (0..packing.len())
                .filter(|&w| Coefficients::of(eq.as_slice()[w]).get(u))
                .map(|w| packing.poly().as_slice()[w])
                .sum();
            assert_eq!(row, expected, "row {u}");
        }
    }

    #[test]
    fn a_weight_is_the_subset_sum_the_coordinates_select() {
        // Invariant: the weight at `w` adds the table where bits are set.
        let (reduction, _, _, r_batch) = fixture(0x5E7, 16);
        let batch = reduction.batch(&r_batch).unwrap();
        let weights = batch.weights();

        // Built from the challenge here, not read off the stage under test.
        let eq_batch = Poly::<EF>::new_from_point(r_batch.as_slice(), EF::ONE);

        for (w, &value) in dense_eq_high(&reduction).as_slice().iter().enumerate() {
            let expected: EF = Coefficients::of(value)
                .iter()
                .zip(eq_batch.as_slice())
                .filter(|&(bit, _)| bit)
                .map(|(_, &weight)| weight)
                .sum();
            assert_eq!(weights.as_slice()[w], expected, "point {w}");
        }
        assert_eq!(weights.num_variables(), reduction.num_variables());
    }

    #[test]
    fn the_initial_sum_is_the_sum_the_sumcheck_proves() {
        // Invariant: the weighted sum over the hypercube is the batched rows.
        //
        //     sum_w A(w) * t'(w)  ==  sum_u eq(u, r_batch) * row u
        //
        // Both sides are derived, never sent, which a dishonest element hits.
        let (reduction, packing, _, r_batch) = fixture(0xBA7, 32);
        let batch = reduction.batch(&r_batch).unwrap();
        let tensor = reduction.tensor(&packing).unwrap();

        let dot: EF = batch
            .weights()
            .as_slice()
            .iter()
            .zip(packing.poly().as_slice())
            .map(|(&weight, &value)| weight * value)
            .sum();

        assert_eq!(dot, batch.initial_sum(&tensor, None));
    }

    #[test]
    fn the_batched_rows_weigh_by_the_challenge_itself() {
        // Invariant: row `u` carries the equality weight at point `u`.
        //
        //     initial_sum(s_hat) == sum_u eq(hypercube(u, k), r'') * row u
        //
        // Every other batch-stage test reads the table this stage holds.
        // A stage ignoring the challenge would pass all of them.
        //
        // That is the forgery with a challenge every prover knows.
        //
        // So the reference here is built from the challenge itself.
        let (reduction, packing, _, r_batch) = fixture(0xC4A, 16);
        let batch = reduction.batch(&r_batch).unwrap();
        let tensor = reduction.tensor(&packing).unwrap();
        let absorbed = BitRingSwitch::<EF>::ABSORBED;

        let expected: EF = tensor
            .rows()
            .iter()
            .enumerate()
            .map(|(u, &row)| {
                let point = Point::<EF>::hypercube(u, absorbed);
                row * Point::eval_eq(point.as_slice(), r_batch.as_slice())
            })
            .sum();

        assert_eq!(batch.initial_sum(&tensor, None), expected);
    }

    #[test]
    fn the_closing_weight_is_the_weight_multilinear_at_the_point() {
        // Invariant: the element closes the sumcheck with no weights pass.
        //
        //     closing_weight(r')  ==  A(r')
        let (reduction, _, _, r_batch) = fixture(0xA7, 16);
        let batch = reduction.batch(&r_batch).unwrap();
        let r_prime = Point::<EF>::rand(&mut SmallRng::seed_from_u64(0xA8), 3);

        assert_eq!(
            batch.closing_weight(&r_prime).unwrap(),
            batch.weights().eval_base(&r_prime)
        );
    }

    #[test]
    fn the_equality_element_matches_its_hypercube_definition() {
        // The recurrence is linear in the variables.
        // The definition below is exponential.
        //
        //     e = sum_w eq(r_high, w) ⊗ eq(w, r')
        let mut rng = SmallRng::seed_from_u64(0xE9);
        for num_variables in 0..5 {
            let r = Point::<EF>::rand(&mut rng, num_variables + BitRingSwitch::<EF>::ABSORBED);
            let reduction = BitRingSwitch::new(&r).unwrap();
            let r_prime = Point::<EF>::rand(&mut rng, num_variables);

            let eq_prime = Poly::<EF>::new_from_point(r_prime.as_slice(), EF::ONE);
            let mut expected = BitTensor::zero();
            let eq_high = dense_eq_high(&reduction);
            for (&a, &b) in eq_high.as_slice().iter().zip(eq_prime.as_slice()) {
                expected.add_exterior_product(a, b);
            }

            assert_eq!(
                reduction.equality_element(&r_prime),
                expected,
                "{num_variables} variables"
            );
        }
    }

    #[test]
    fn a_point_narrower_than_one_element_is_refused() {
        // Fewer coordinates than one element absorbs leaves nothing to keep.
        let mut rng = SmallRng::seed_from_u64(0xBAD);
        let r = Point::<EF>::rand(&mut rng, 3);

        assert_eq!(
            BitRingSwitch::new(&r).unwrap_err(),
            BitRingSwitchError::PointTooNarrow {
                needed: 4,
                actual: 3
            }
        );
    }

    #[test]
    fn a_batching_challenge_of_the_wrong_width_is_refused() {
        // A short challenge would zip-truncate, dropping identity rows.
        // Both sides move the same way, so no later check would catch it.
        let mut rng = SmallRng::seed_from_u64(0xBAD2);
        let r = Point::<EF>::rand(&mut rng, 7);
        let reduction = BitRingSwitch::new(&r).unwrap();

        for width in [3usize, 5] {
            let r_batch = Point::<EF>::rand(&mut rng, width);
            assert_eq!(
                reduction.batch(&r_batch).unwrap_err(),
                BitRingSwitchError::BatchWidthMismatch {
                    expected: 4,
                    actual: width,
                }
            );
        }
    }

    #[test]
    fn a_packing_of_the_wrong_width_is_refused() {
        // A mismatched packing would zip-truncate the accumulation.
        let (reduction, _, _, r_batch) = fixture(0xBAD3, 16);
        let batch = reduction.batch(&r_batch).unwrap();
        let wider = BitPacking::<EF>::new(&bits(0xBAD4, 32)).unwrap();

        assert_eq!(
            reduction.tensor(&wider).unwrap_err(),
            BitRingSwitchError::WidthMismatch {
                expected: 3,
                actual: 4
            }
        );
        assert_eq!(
            batch
                .closing_weight(&Point::<EF>::new(alloc::vec![EF::ONE; 4]))
                .unwrap_err(),
            BitRingSwitchError::WidthMismatch {
                expected: 3,
                actual: 4
            }
        );
    }

    #[test]
    fn a_boolean_prefix_restricts_the_sumcheck_without_changing_the_claim() {
        // Invariant: skipping the zeros of the equality table changes no sum.
        //
        // A claim about one column of a stacked trace arrives with a Boolean slot address
        // leading its row point, and that prefix makes the table an indicator:
        //
        //     high = (1, 0, z)   ->  eq(high, w) = 0 unless w is 100 or 101
        //
        // Fixture state: 2^7 cells, 4 absorbed, so 3 kept variables and 8 elements.
        //
        //     prefix 0   ->  the run is all 8, which is the dense case
        //     prefix 1   ->  4 of 8
        //     prefix 2   ->  2 of 8
        //     prefix 3   ->  1 of 8, the whole table an indicator
        //
        // The dense reference forms every term, zeros included, and must agree with each.
        let witness = bits(0xB0015, 16);
        let packing = BitPacking::<EF>::new(&witness).unwrap();
        let mut rng = SmallRng::seed_from_u64(0xB0016);

        for prefix in 0..=3usize {
            for address in 0..1usize << prefix {
                // Leading coordinates spell the address, highest bit first.
                let mut coordinates: Vec<EF> = (0..prefix)
                    .map(|bit| {
                        if (address >> (prefix - 1 - bit)) & 1 == 1 {
                            EF::ONE
                        } else {
                            EF::ZERO
                        }
                    })
                    .collect();
                // The rest of the point uses non-Boolean challenges.
                // This keeps the intended prefix length exact in every fixture.
                coordinates.extend((prefix..7).map(|_| {
                    loop {
                        let coordinate = rng.random::<EF>();
                        if coordinate != EF::ZERO && coordinate != EF::ONE {
                            break coordinate;
                        }
                    }
                }));
                let r = Point::new(coordinates);

                let reduction = BitRingSwitch::new(&r).unwrap();
                let r_batch = Point::<EF>::rand(&mut rng, BitRingSwitch::<EF>::ABSORBED);
                let batch = reduction.batch(&r_batch).unwrap();

                // The element, against the term-by-term sum over the whole hypercube.
                let mut expected = BitTensor::zero();
                for (&weight, &value) in dense_eq_high(&reduction)
                    .as_slice()
                    .iter()
                    .zip(packing.poly().as_slice())
                {
                    expected.add_exterior_product(weight, value);
                }
                assert_eq!(
                    reduction.tensor(&packing).unwrap(),
                    expected,
                    "prefix {prefix} address {address}"
                );

                // The weight multilinear, against the same table read subset sum by subset sum.
                let eq_batch = Poly::<EF>::new_from_point(r_batch.as_slice(), EF::ONE);
                let weights = batch.weights();
                assert_eq!(weights.num_variables(), reduction.num_variables() - prefix);

                // The compact table is the selected run of the dense reference.
                let len = 1usize << (reduction.num_variables() - prefix);
                let offset = address * len;
                let dense = dense_eq_high(&reduction);
                for (w, &value) in dense.as_slice()[offset..offset + len].iter().enumerate() {
                    let want: EF = Coefficients::of(value)
                        .iter()
                        .zip(eq_batch.as_slice())
                        .filter(|&(bit, _)| bit)
                        .map(|(_, &weight)| weight)
                        .sum();
                    assert_eq!(weights.as_slice()[w], want, "prefix {prefix} point {w}");
                }

                // The full protocol runs only inside the selected slot.
                // It restores the address before handing the point to the commitment.
                let claim = embedded(&witness).eval_base(&r);
                let (proof, surviving_point, surviving_value) =
                    reduction.prove::<EF, _, _>(&packing, &mut challenger());
                assert_eq!(
                    proof.sumcheck.num_rounds(),
                    reduction.num_variables() - prefix
                );
                assert_eq!(
                    &surviving_point.as_slice()[..prefix],
                    &r.as_slice()[..prefix]
                );
                assert_eq!(surviving_value, packing.poly().eval_base(&surviving_point));

                // The replay derives the same restriction from the incoming point.
                let (verified_point, verified_value) =
                    reduction.verify(&proof, claim, &mut challenger()).unwrap();
                assert_eq!(verified_point, surviving_point);
                assert_eq!(verified_value, surviving_value);
            }
        }
    }

    proptest! {
        #[test]
        fn the_chunked_accumulation_matches_the_term_by_term_sum(seed: u64, log_n in 1usize..6) {
            // The parallel path accumulates in chunks, combining partials.
            // The reference forms every term as its own element and adds them.
            let (reduction, packing, _, _) = fixture(seed, (1 << log_n) * 2);

            let mut expected = BitTensor::zero();
            for (&weight, &value) in dense_eq_high(&reduction)
                .as_slice()
                .iter()
                .zip(packing.poly().as_slice())
            {
                expected += BitTensor::exterior_product(weight, value);
            }

            prop_assert_eq!(reduction.tensor(&packing).unwrap(), expected);
        }
    }

    #[test]
    fn the_reduction_round_trips_and_leaves_a_true_claim() {
        // Invariant: the surviving claim is the truth about the packed polynomial.
        //
        //     in    t(r) = s   at a random point over all 8 variables
        //     out   t'(r')     over the 4 the packing keeps, 4 being absorbed
        let witness = bits(0x81A5, 32);
        let packing = BitPacking::<EF>::new(&witness).unwrap();
        let r = Point::<EF>::rand(&mut SmallRng::seed_from_u64(0x81A6), 8);
        let reduction = BitRingSwitch::new(&r).unwrap();
        let claim = embedded(&witness).eval_base(&r);

        let mut prover_chal = challenger();
        let (proof, r_prime_p, s_prime_p) = reduction.prove::<EF, _, _>(&packing, &mut prover_chal);

        let mut verifier_chal = challenger();
        let (r_prime_v, s_prime_v) = reduction.verify(&proof, claim, &mut verifier_chal).unwrap();

        assert_eq!(r_prime_p, r_prime_v);
        assert_eq!(s_prime_p, s_prime_v);
        assert_eq!(s_prime_v, packing.poly().eval_base(&r_prime_v));
    }

    #[test]
    fn a_claim_the_element_does_not_support_is_rejected() {
        // The column reading is what ties the element to the incoming claim.
        let witness = bits(0x0AD, 32);
        let packing = BitPacking::<EF>::new(&witness).unwrap();
        let r = Point::<EF>::rand(&mut SmallRng::seed_from_u64(0x0AE), 8);
        let reduction = BitRingSwitch::new(&r).unwrap();
        let claim = embedded(&witness).eval_base(&r);

        let (proof, _, _) = reduction.prove::<EF, _, _>(&packing, &mut challenger());

        let err = reduction
            .verify(&proof, claim + EF::ONE, &mut challenger())
            .unwrap_err();
        assert_eq!(err, BitRingSwitchProofError::ClaimMismatch);
    }

    #[test]
    fn a_tampered_element_is_caught_by_whichever_reading_it_moves() {
        // Invariant: both readings are of the same coefficients.
        //
        //     rows    -> the sum the rounds start from
        //     columns -> the claim they are checked against
        //
        // A tamper has to survive the column reading before the rounds ever run.
        // The two cases below are therefore checked separately.
        let witness = bits(0x7A17, 32);
        let packing = BitPacking::<EF>::new(&witness).unwrap();
        let r = Point::<EF>::rand(&mut SmallRng::seed_from_u64(0x7A18), 8);
        let reduction = BitRingSwitch::new(&r).unwrap();
        let claim = embedded(&witness).eval_base(&r);
        let (honest, _, _) = reduction.prove::<EF, _, _>(&packing, &mut challenger());

        // Case 1: add one to a row.
        //
        // One is the first basis coordinate, so this flips entry (3, 0) of the matrix.
        // That moves column 0 as well, so the column reading moves with it.
        //
        // The run stops at the claim check without reaching a single round.
        let mut flipped = honest.clone();
        let mut rows = flipped.tensor.rows().to_vec();
        rows[3] += EF::ONE;
        flipped.tensor = BitTensor::try_from(rows).unwrap();

        assert_eq!(
            reduction
                .verify(&flipped, claim, &mut challenger())
                .unwrap_err(),
            BitRingSwitchProofError::ClaimMismatch,
        );

        // Case 2: a delta the column reading cannot see.
        //
        // The claim check weighs the columns by the absorbed coordinates:
        //
        //     claim   = sum_v column_v * eq_low[v]
        //     hidden  when  d_0 * eq_low[0] + d_1 * eq_low[1] = 0
        //
        // Over characteristic two that is d_1 = d_0 * eq_low[0] / eq_low[1].
        // Nothing constrains d_0, so any nonzero value does.
        //
        // It comes from a byte pattern, since an integer would reduce modulo two.
        let eq_low = reduction.eq_low.as_slice();
        assert_ne!(
            eq_low[1],
            EF::ZERO,
            "the construction divides by this entry"
        );

        let d_0 = EF::from_le_byte_iter([0x2C, 0x9B].into_iter());
        let d_1 = d_0 * eq_low[0] * eq_low[1].inverse();
        assert_ne!(d_0, EF::ZERO);

        // A row list read as columns is the transpose, which is its own inverse.
        let mut delta_columns = alloc::vec![EF::ZERO; BitTensor::<EF>::DIMENSION];
        delta_columns[0] = d_0;
        delta_columns[1] = d_1;
        let delta = BitTensor::try_from(delta_columns).unwrap().columns();

        let mut hidden = honest.clone();
        let rows = hidden
            .tensor
            .rows()
            .iter()
            .zip(&delta)
            .map(|(&row, &shift)| row + shift)
            .collect::<Vec<_>>();
        hidden.tensor = BitTensor::try_from(rows).unwrap();

        // The element genuinely changed, yet its column reading did not.
        assert_ne!(hidden.tensor, honest.tensor);
        assert_eq!(
            reduction.incoming_claim(&hidden.tensor),
            reduction.incoming_claim(&honest.tensor),
        );

        // The row reading is what the rounds start from, and that one moved.
        //
        // No round rejects on its own.
        //
        //     sent      one value of the round polynomial, and its leading coefficient
        //     derived   the other value, from the running claim, never checked
        //
        //     wrong starting sum  ->  survives every round
        //     closing check       ->  the first place the two sides disagree
        assert_eq!(
            reduction
                .verify(&hidden, claim, &mut challenger())
                .unwrap_err(),
            BitRingSwitchProofError::FinalCheck,
        );

        // The honest element passes the same path, so the rejection is the delta.
        assert!(reduction.verify(&honest, claim, &mut challenger()).is_ok());
    }

    #[test]
    fn a_stray_grinding_witness_is_refused_before_the_transcript() {
        // Invariant: this reduction never grinds, so a witness rides along unbound.
        //
        // The rejection is structural, so it may not advance the sponge.
        //
        // A short element cannot be built at all.
        // Every route into one checks the row count, deserialization included.
        //
        // The verifier's shape check is therefore defence in depth, not a reachable path.
        let witness = bits(0x5407, 32);
        let packing = BitPacking::<EF>::new(&witness).unwrap();
        let r = Point::<EF>::rand(&mut SmallRng::seed_from_u64(0x5408), 8);
        let reduction = BitRingSwitch::new(&r).unwrap();
        let claim = embedded(&witness).eval_base(&r);

        let (mut proof, _, _) = reduction.prove::<EF, _, _>(&packing, &mut challenger());
        proof.sumcheck.pow_witnesses.push(EF::ONE);

        assert_eq!(
            reduction
                .verify(&proof, claim, &mut challenger())
                .unwrap_err(),
            BitRingSwitchProofError::NonEmptyPowWitnesses { actual: 1 }
        );

        // A row count the level does not admit has no constructor.
        assert!(BitTensor::<EF>::try_from(alloc::vec![EF::ONE; 4]).is_err());
    }

    #[test]
    fn a_point_the_prover_did_not_run_over_is_rejected() {
        // Invariant: the two sides must run over the same point, not merely the same width.
        //
        // The other point keeps the absorbed coordinates and moves one kept coordinate.
        // The column reading is then unchanged, so the claim check passes and the rest must catch it.
        //
        // That is the discriminating case.
        // Moving an absorbed coordinate would stop at the claim check instead.
        let witness = bits(0x9107, 32);
        let packing = BitPacking::<EF>::new(&witness).unwrap();
        let r = Point::<EF>::rand(&mut SmallRng::seed_from_u64(0x9108), 8);
        let claim = embedded(&witness).eval_base(&r);
        let reduction = BitRingSwitch::new(&r).unwrap();
        let (proof, _, _) = reduction.prove::<EF, _, _>(&packing, &mut challenger());

        let mut moved = r.as_slice().to_vec();
        moved[0] += EF::ONE;
        let elsewhere = BitRingSwitch::new(&Point::new(moved)).unwrap();

        let err = elsewhere
            .verify(&proof, claim, &mut challenger())
            .unwrap_err();
        assert_ne!(err, BitRingSwitchProofError::ClaimMismatch, "{err:?}");
    }

    proptest! {
        // Each case runs a full reduction on both sides, so a few dozen keep the suite fast.
        #![proptest_config(ProptestConfig { cases: 24, ..ProptestConfig::default() })]

        /// Every witness length the level admits, at a random point each time.
        #[test]
        fn the_reduction_round_trips_over_random_inputs(
            log_bytes in 1usize..=6,
            witness_seed: u64,
            point_seed: u64,
        ) {
            let witness = bits(witness_seed, 1 << log_bytes);
            let packing = BitPacking::<EF>::new(&witness).unwrap();
            let variables = log_bytes + 3;
            let r = Point::<EF>::rand(&mut SmallRng::seed_from_u64(point_seed), variables);
            let reduction = BitRingSwitch::new(&r).unwrap();
            let claim = embedded(&witness).eval_base(&r);

            let (proof, _, s_prime_p) = reduction.prove::<EF, _, _>(&packing, &mut challenger());
            let (r_prime, s_prime) =
                reduction.verify(&proof, claim, &mut challenger()).unwrap();

            prop_assert_eq!(s_prime, s_prime_p);
            prop_assert_eq!(s_prime, packing.poly().eval_base(&r_prime));
        }

        /// Every witness length and every row count it admits, both readings each time.
        #[test]
        fn a_successor_view_round_trips_over_random_inputs(
            (log_bytes, row_variables) in (1usize..=6)
                .prop_flat_map(|log_bytes| (Just(log_bytes), 0..=log_bytes + 3)),
            witness_seed: u64,
            point_seed: u64,
        ) {
            let witness = bits(witness_seed, 1 << log_bytes);
            let packing = BitPacking::<EF>::new(&witness).unwrap();
            let variables = log_bytes + 3;
            let r = Point::<EF>::rand(&mut SmallRng::seed_from_u64(point_seed), variables);
            let reduction = BitRingSwitch::with_successor(&r, row_variables).unwrap();
            let current = embedded(&witness).eval_base(&r);
            let next = dense_successor_claim(&witness, &r, row_variables);

            let (proof, r_prime_p, s_prime_p) = reduction.prove::<EF, _, _>(&packing, &mut challenger());
            let (r_prime, s_prime) = reduction
                .verify_readings(&proof, Some(current), Some(next), &mut challenger())
                .unwrap();

            prop_assert_eq!(&r_prime, &r_prime_p);
            prop_assert_eq!(s_prime, s_prime_p);
            prop_assert_eq!(s_prime, packing.poly().eval_base(&r_prime));
        }
    }

    /// The witness one row further on, repeat-last, batched over the selector: the definition.
    ///
    ///     sum_{c, x} eq(selector, c) * t(c, min(z + 1, max)) * eq(rho, z)   summed over z
    ///
    /// The selector leads the point and the rows trail it, `row_variables` of them.
    fn dense_successor_claim(witness: &[u8], point: &Point<EF>, row_variables: usize) -> EF {
        let cells = embedded(witness);
        let (selector, rho) = point.split_at(point.num_variables() - row_variables);
        let eq_selector = Poly::<EF>::new_from_point(selector.as_slice(), EF::ONE);
        let eq_rho = Poly::<EF>::new_from_point(rho.as_slice(), EF::ONE);
        let rows = 1usize << row_variables;
        let mut claim = EF::ZERO;
        for (c, &gate) in eq_selector.as_slice().iter().enumerate() {
            for (z, &weight) in eq_rho.as_slice().iter().enumerate() {
                let x = (z + 1).min(rows - 1);
                claim += gate * weight * cells.as_slice()[c * rows + x];
            }
        }
        claim
    }

    /// A field challenge that is neither zero nor one.
    ///
    /// A Boolean coordinate would move the prefix a fixture means to fix.
    fn non_boolean(rng: &mut SmallRng) -> EF {
        loop {
            let coordinate = rng.random::<EF>();
            if coordinate != EF::ZERO && coordinate != EF::ONE {
                return coordinate;
            }
        }
    }

    /// How the selector coordinates, which lead the point, are chosen.
    #[derive(Clone, Copy, Debug)]
    enum Selector {
        /// Field challenges throughout, as a transcript draws them.
        Random,
        /// Bits throughout, spelling a slot address.
        Boolean,
        /// Two bits and a field challenge, repeating.
        Mixed,
    }

    impl Selector {
        /// Every kind, in the order the tests sweep them.
        const ALL: [Self; 3] = [Self::Random, Self::Boolean, Self::Mixed];
    }

    /// A point whose leading `num_variables - row_variables` coordinates are the selector.
    ///
    /// The row coordinates are always field challenges.
    fn successor_point(
        seed: u64,
        num_variables: usize,
        row_variables: usize,
        selector: Selector,
    ) -> Point<EF> {
        let mut rng = SmallRng::seed_from_u64(seed);
        let selector_variables = num_variables - row_variables;
        Point::new(
            (0..num_variables)
                .map(|i| {
                    let bit = if i % 2 == 0 { EF::ONE } else { EF::ZERO };
                    match selector {
                        _ if i >= selector_variables => non_boolean(&mut rng),
                        Selector::Random => non_boolean(&mut rng),
                        Selector::Boolean => bit,
                        Selector::Mixed if i % 3 == 2 => non_boolean(&mut rng),
                        Selector::Mixed => bit,
                    }
                })
                .collect(),
        )
    }

    /// Row counts covering every case the successor view splits into, over 9 variables.
    ///
    /// ```text
    ///     0, 2, 4   ->  the rows fit in one element, no successor element is sent
    ///     6         ->  a 3-coordinate selector and 2 kept row coordinates
    ///     9         ->  no selector, every kept coordinate a row coordinate
    /// ```
    const ROW_VARIABLES: [usize; 5] = [0, 2, 4, 6, 9];

    #[test]
    fn the_successor_claim_is_the_witness_one_row_on() {
        // Invariant: the column reading of the elements is the repeat-last successor claim.
        //
        //     sum_z eq(rho, z) * t(c, min(z + 1, max))   batched over the selector c
        //
        // Fixture state: 512 cells is 9 variables, 16 bits absorb 4, so 5 are kept.
        //
        // Every row count runs under a random, a Boolean and a mixed selector.
        // A Boolean selector moves the run the elements are accumulated over.
        let witness = bits(0x5CC1, 64);
        let packing = BitPacking::<EF>::new(&witness).unwrap();

        for selector in Selector::ALL {
            for (seed, &row_variables) in (0x5CC2..).zip(&ROW_VARIABLES) {
                let r = successor_point(seed, 9, row_variables, selector);
                let reduction = BitRingSwitch::with_successor(&r, row_variables).unwrap();
                let tensor = reduction.tensor(&packing).unwrap();
                let successor = reduction.successor_tensors(&packing).unwrap();

                assert_eq!(
                    successor.is_some(),
                    row_variables > BitRingSwitch::<EF>::ABSORBED,
                    "{selector:?} selector, {row_variables} row variables"
                );
                assert_eq!(
                    reduction
                        .successor_claim(&tensor, successor.as_ref())
                        .unwrap(),
                    dense_successor_claim(&witness, &r, row_variables),
                    "{selector:?} selector, {row_variables} row variables"
                );
            }
        }
    }

    #[test]
    fn the_successor_elements_are_their_term_by_term_sums() {
        // Invariant: the two elements are the terms the dense equality table names, added
        // one at a time. The production path adds a block against the inner weights alone
        // and scales the partial once, which is the same product regrouped.
        //
        //     carry  =  sum_{x(w) != 0}   E[w - 1] ⊗ t'(w)
        //     last   =  sum_{x(w) = max}  E[w]     ⊗ t'(w)
        let witness = bits(0x7E24, 64);
        let packing = BitPacking::<EF>::new(&witness).unwrap();

        for &row_variables in &ROW_VARIABLES {
            for selector in Selector::ALL {
                let r = successor_point(0x7E25, 9, row_variables, selector);
                let reduction = BitRingSwitch::with_successor(&r, row_variables).unwrap();
                let Some(kept) = reduction.kept_row_variables() else {
                    continue;
                };
                let table = dense_eq_high(&reduction);
                let max = (1usize << kept) - 1;

                let mut carry = BitTensor::zero();
                let mut last = BitTensor::zero();
                for (w, &value) in packing.poly().as_slice().iter().enumerate() {
                    let row = w & max;
                    if row != 0 {
                        carry += BitTensor::exterior_product(table.as_slice()[w - 1], value);
                    }
                    if row == max {
                        last += BitTensor::exterior_product(table.as_slice()[w], value);
                    }
                }

                let elements = reduction.successor_tensors(&packing).unwrap().unwrap();
                assert_eq!(elements.carry, carry, "{selector:?}, {row_variables} rows");
                assert_eq!(elements.last, last, "{selector:?}, {row_variables} rows");
            }
        }
    }

    #[test]
    fn the_successor_weights_are_the_dense_table_entry_by_entry() {
        // Invariant: the weight table is the one a pass over the dense equality table builds,
        // entry for entry, shift and all.
        //
        //     A(w)       = sum over the set coordinates of eq(r_high, w) of eq(r_batch, .)
        //     weights(w) = A(w) + alpha * A(w - 1) [x(w) != 0] + alpha^2 * A(w) [x(w) = max]
        //
        // The production path takes the coordinate sums under the block's own scale and
        // combines the shift in place. Both group the same factors, and the field's
        // multiplication is associative, so nothing may differ at all.
        let mut rng = SmallRng::seed_from_u64(0xDE96);
        let r = successor_point(0xDE97, 9, 6, Selector::Random);
        let reduction = BitRingSwitch::with_successor(&r, 6).unwrap();
        let r_batch = Point::<EF>::rand(&mut rng, BitRingSwitch::<EF>::ABSORBED);
        let alpha = non_boolean(&mut rng);

        // Built from the challenge here, not read off the stage under test.
        let eq_batch = Poly::<EF>::new_from_point(r_batch.as_slice(), EF::ONE);
        let base = dense_eq_high(&reduction)
            .as_slice()
            .iter()
            .map(|&value| {
                Coefficients::of(value)
                    .iter_set()
                    .map(|u| eq_batch.as_slice()[u])
                    .sum::<EF>()
            })
            .collect::<Vec<_>>();

        let max = (1usize << reduction.kept_row_variables().unwrap()) - 1;
        let expected = (0..base.len())
            .map(|w| {
                let row = w & max;
                let mut value = base[w];
                if row != 0 {
                    value += alpha * base[w - 1];
                }
                if row == max {
                    value += alpha.square() * base[w];
                }
                value
            })
            .collect::<Vec<_>>();

        let batch = reduction.batch_with_successor(&r_batch, alpha).unwrap();
        assert_eq!(batch.weights().as_slice(), expected.as_slice());
    }

    #[test]
    fn the_sweeps_read_the_same_table_at_every_block_size() {
        // Invariant: the block is an amortisation unit, not part of what a sweep computes.
        //
        //     table[i * block + j]  ==  outer[i] * inner[j]   for every split of the point
        //
        // The size the reduction ships is far above the one these tests run at, and it is
        // what decides how much of each sweep is boundary and how much is interior: the
        // carry's first entry, the weights' stride against a column, and the count of blocks
        // a fold combines. Sweeping the split covers all three shapes at one place.
        let mut rng = SmallRng::seed_from_u64(0xB10C5);
        let packing = BitPacking::<EF>::new(&bits(0xB10C6, 64)).unwrap();
        let r_batch = Point::<EF>::rand(&mut rng, BitRingSwitch::<EF>::ABSORBED);
        let alpha = non_boolean(&mut rng);

        for &row_variables in &ROW_VARIABLES {
            for selector in Selector::ALL {
                let r = successor_point(0xB10C7, 9, row_variables, selector);
                let reduction = BitRingSwitch::with_successor(&r, row_variables).unwrap();
                let (prefix, offset, _) = reduction.support();
                let run = &reduction.high()[prefix..];
                let batch = reduction
                    .batch_drawn(
                        &r_batch,
                        reduction.sends_successor_tensors().then_some(alpha),
                    )
                    .unwrap();

                // One entry per block leaves the outer factor the dense table itself.
                let dense = FactoredEquality::new(run, 0);
                let tensor = BitRingSwitch::tensor_over(&packing, offset, &dense);
                let successor = reduction.successor_tensors_over(&packing, offset, &dense);
                let weights = batch.weights_over::<EF>(&dense);

                for log_block in 0..=run.len() + 2 {
                    let equality = FactoredEquality::new(run, log_block);
                    let case =
                        alloc::format!("{selector:?}, {row_variables} rows, 2^{log_block} block");

                    assert_eq!(
                        BitRingSwitch::tensor_over(&packing, offset, &equality),
                        tensor,
                        "{case}"
                    );
                    assert_eq!(
                        reduction.successor_tensors_over(&packing, offset, &equality),
                        successor,
                        "{case}"
                    );
                    assert_eq!(
                        batch.weights_over::<EF>(&equality).as_slice(),
                        weights.as_slice(),
                        "{case}"
                    );
                }
            }
        }
    }

    #[test]
    fn the_wide_sweeps_read_the_same_elements_at_every_block_size() {
        // Invariant: at the level the bit-block kernel is shaped for, the split changes nothing.
        //
        // Blocks of eight elements and up run that kernel where the target has one. The carry
        // meets columns shorter than a block, as long as one, and longer.
        type Wide = BinaryField128;
        let mut rng = SmallRng::seed_from_u64(0x31DE);
        let packing = BitPacking::<Wide>::new(&bits(0x31DF, 1 << 13)).unwrap();
        let num_variables = packing.num_variables() + BitRingSwitch::<Wide>::ABSORBED;

        for row_variables in [8, 10, 12, num_variables] {
            let r = Point::<Wide>::rand(&mut rng, num_variables);
            let reduction = BitRingSwitch::with_successor(&r, row_variables).unwrap();
            let (prefix, offset, _) = reduction.support();
            let run = &reduction.high()[prefix..];

            // One entry per block sums every term through the buckets.
            let dense = FactoredEquality::new(run, 0);
            let tensor = BitRingSwitch::tensor_over(&packing, offset, &dense);
            let successor = reduction.successor_tensors_over(&packing, offset, &dense);

            for log_block in 0..=run.len() {
                let equality = FactoredEquality::new(run, log_block);
                let case = alloc::format!("{row_variables} rows, 2^{log_block} block");
                assert_eq!(
                    BitRingSwitch::tensor_over(&packing, offset, &equality),
                    tensor,
                    "{case}"
                );
                assert_eq!(
                    reduction.successor_tensors_over(&packing, offset, &equality),
                    successor,
                    "{case}"
                );
            }
        }
    }

    #[test]
    fn the_successor_weights_close_the_batched_sum() {
        // Invariant: the weights are what both ends of the sumcheck read.
        //
        //     sum_w weights(w) * t'(w)  ==  initial_sum          rows of the three elements
        //     weights(r')               ==  closing_weight(r')   through the lifted elements
        //
        // Fixture state: 9 variables, 6 of them rows, so 2 kept row coordinates.
        let witness = bits(0xC105, 64);
        let packing = BitPacking::<EF>::new(&witness).unwrap();
        let mut rng = SmallRng::seed_from_u64(0xC106);
        let r = successor_point(0xC107, 9, 6, Selector::Random);
        let reduction = BitRingSwitch::with_successor(&r, 6).unwrap();
        let tensor = reduction.tensor(&packing).unwrap();
        let successor = reduction.successor_tensors(&packing).unwrap().unwrap();

        let r_batch = Point::<EF>::rand(&mut rng, BitRingSwitch::<EF>::ABSORBED);
        let alpha = non_boolean(&mut rng);
        let batch = reduction.batch_with_successor(&r_batch, alpha).unwrap();
        let weights = batch.weights();

        let dot: EF = weights
            .as_slice()
            .iter()
            .zip(packing.poly().as_slice())
            .map(|(&weight, &value)| weight * value)
            .sum();
        assert_eq!(batch.initial_sum(&tensor, Some(&successor)), dot);

        let r_prime = Point::<EF>::rand(&mut rng, reduction.num_variables());
        assert_eq!(
            batch.closing_weight(&r_prime).unwrap(),
            weights.eval_base(&r_prime)
        );
    }

    #[test]
    fn a_successor_view_round_trips() {
        // Invariant: both readings verify, and the surviving claim is the truth about the packing.
        //
        //     in    t(r) = current,  t one row on at r = next
        //     out   t'(r') = e       one claim, whatever the row count
        let witness = bits(0x7219, 64);
        let packing = BitPacking::<EF>::new(&witness).unwrap();
        let cells = embedded(&witness);

        for selector in Selector::ALL {
            for (seed, &row_variables) in (0x721A..).zip(&ROW_VARIABLES) {
                let r = successor_point(seed, 9, row_variables, selector);
                let reduction = BitRingSwitch::with_successor(&r, row_variables).unwrap();
                let current = cells.eval_base(&r);
                let next = dense_successor_claim(&witness, &r, row_variables);

                let (proof, r_prime, value) =
                    reduction.prove::<EF, _, _>(&packing, &mut challenger());
                let verified = reduction
                    .verify_readings(&proof, Some(current), Some(next), &mut challenger())
                    .unwrap();

                let case = alloc::format!("{selector:?} selector, {row_variables} row variables");
                assert_eq!(verified, (r_prime.clone(), value), "{case}");
                assert_eq!(value, packing.poly().eval_base(&r_prime), "{case}");
            }
        }
    }

    #[test]
    fn a_successor_view_inside_one_element_plays_the_plain_transcript() {
        // Invariant: rows that fit in one element send nothing extra.
        //
        // The successor is then a column reading alone, which the verifier makes offline.
        // The prover's side of the run is the one a plain reduction at the point plays.
        let witness = bits(0x1D3, 64);
        let packing = BitPacking::<EF>::new(&witness).unwrap();

        for row_variables in 0..=BitRingSwitch::<EF>::ABSORBED {
            let r = successor_point(0x1D4, 9, row_variables, Selector::Random);
            let plain = BitRingSwitch::new(&r).unwrap();
            let successor = BitRingSwitch::with_successor(&r, row_variables).unwrap();
            assert!(!successor.sends_successor_tensors());

            let mut plain_challenger = challenger();
            let (plain_proof, plain_point, plain_value) =
                plain.prove::<EF, _, _>(&packing, &mut plain_challenger);
            let mut successor_challenger = challenger();
            let (successor_proof, successor_point, successor_value) =
                successor.prove::<EF, _, _>(&packing, &mut successor_challenger);

            assert_eq!(successor_proof.tensor, plain_proof.tensor);
            assert!(successor_proof.successor.is_none());
            assert_eq!(
                successor_point, plain_point,
                "{row_variables} row variables"
            );
            assert_eq!(
                successor_value, plain_value,
                "{row_variables} row variables"
            );
            assert_eq!(
                CanSample::<EF>::sample(&mut successor_challenger),
                CanSample::<EF>::sample(&mut plain_challenger),
                "{row_variables} row variables"
            );
        }
    }

    #[test]
    fn the_prefix_stops_before_the_row_coordinates() {
        // Invariant: the Boolean prefix fixes selector coordinates only.
        //
        // A row coordinate fixed to a bit is no slot address.
        // The carry reads the element before, which may sit outside a run it would select.
        //
        // Fixture state: 9 variables, 6 of them rows.
        //
        //     point     (1, 0, 1 | 0, z | z, z, z, z)
        //                selector  kept   absorbed
        //     prefix    3, though the first row coordinate is a bit as well
        //     rounds    5 - 3 = 2, where a plain reduction at the point runs 1
        let witness = bits(0x9EF1, 64);
        let packing = BitPacking::<EF>::new(&witness).unwrap();
        let mut rng = SmallRng::seed_from_u64(0x9EF2);
        let mut coordinates = alloc::vec![EF::ONE, EF::ZERO, EF::ONE, EF::ZERO];
        coordinates.extend((4..9).map(|_| non_boolean(&mut rng)));
        let r = Point::new(coordinates);

        let reduction = BitRingSwitch::with_successor(&r, 6).unwrap();
        let current = embedded(&witness).eval_base(&r);
        let next = dense_successor_claim(&witness, &r, 6);

        let tensor = reduction.tensor(&packing).unwrap();
        let successor = reduction.successor_tensors(&packing).unwrap();
        assert_eq!(
            reduction
                .successor_claim(&tensor, successor.as_ref())
                .unwrap(),
            next
        );

        let (proof, r_prime, value) = reduction.prove::<EF, _, _>(&packing, &mut challenger());
        assert_eq!(proof.sumcheck.num_rounds(), 2);
        assert_eq!(
            reduction
                .verify_readings(&proof, Some(current), Some(next), &mut challenger())
                .unwrap(),
            (r_prime, value)
        );

        // The plain reduction eats the Boolean row coordinate too, which the cap is for.
        let (plain, _, _) = BitRingSwitch::new(&r)
            .unwrap()
            .prove::<EF, _, _>(&packing, &mut challenger());
        assert_eq!(plain.sumcheck.num_rounds(), 1);
    }

    #[test]
    fn a_false_successor_claim_is_rejected() {
        // The column reading is what ties the elements to the successor claim.
        //
        // Checked with successor elements sent, and with the rows inside one element.
        let witness = bits(0xFA15, 64);
        let packing = BitPacking::<EF>::new(&witness).unwrap();

        for row_variables in [2, 6] {
            let r = successor_point(0xFA16, 9, row_variables, Selector::Random);
            let reduction = BitRingSwitch::with_successor(&r, row_variables).unwrap();
            let current = embedded(&witness).eval_base(&r);
            let next = dense_successor_claim(&witness, &r, row_variables);
            let (proof, _, _) = reduction.prove::<EF, _, _>(&packing, &mut challenger());

            assert_eq!(
                reduction
                    .verify_readings(
                        &proof,
                        Some(current),
                        Some(next + EF::ONE),
                        &mut challenger()
                    )
                    .unwrap_err(),
                BitRingSwitchProofError::SuccessorClaimMismatch,
                "{row_variables} row variables"
            );
        }
    }

    #[test]
    fn a_tampered_carry_element_is_caught_by_the_closing_check() {
        // Invariant: the successor elements enter the rows the sumcheck starts from.
        //
        // The claim reads one column of each:
        //
        //     carry  ->  column 0
        //     last   ->  column d - 1
        //
        // A tamper in any other column passes the claim check.
        // Only the closing check is then left to catch it.
        //
        // Mutation: entry (3, 5) of carry, then entry (5, 0) of last.
        let witness = bits(0x7A3, 64);
        let packing = BitPacking::<EF>::new(&witness).unwrap();
        let r = successor_point(0x7A4, 9, 6, Selector::Random);
        let reduction = BitRingSwitch::with_successor(&r, 6).unwrap();
        let current = embedded(&witness).eval_base(&r);
        let next = dense_successor_claim(&witness, &r, 6);
        let (honest, _, _) = reduction.prove::<EF, _, _>(&packing, &mut challenger());
        let elements = honest.successor.clone().unwrap();

        let flipped = |element: &BitTensor<EF>, row: usize, bit: EF| {
            let mut rows = element.rows().to_vec();
            rows[row] += bit;
            BitTensor::try_from(rows).unwrap()
        };
        let mut carry = honest.clone();
        carry.successor = Some(SuccessorTensors {
            carry: flipped(&elements.carry, 3, EF::from_repr(1 << 5)),
            last: elements.last.clone(),
        });
        let mut last = honest.clone();
        last.successor = Some(SuccessorTensors {
            carry: elements.carry.clone(),
            last: flipped(&elements.last, 5, EF::ONE),
        });

        for (element, tampered) in [("carry", carry), ("last", last)] {
            // The element genuinely changed, yet the successor claim did not.
            assert_ne!(tampered.successor, honest.successor, "{element}");
            assert_eq!(
                reduction
                    .successor_claim(&tampered.tensor, tampered.successor.as_ref())
                    .unwrap(),
                next,
                "{element}"
            );
            assert_eq!(
                reduction
                    .verify_readings(&tampered, Some(current), Some(next), &mut challenger())
                    .unwrap_err(),
                BitRingSwitchProofError::FinalCheck,
                "{element}"
            );
        }

        // The honest elements pass the same path, so the rejection is the tamper.
        assert!(
            reduction
                .verify_readings(&honest, Some(current), Some(next), &mut challenger())
                .is_ok()
        );
    }

    #[test]
    fn a_successor_element_count_disagreeing_with_the_setup_is_refused() {
        // Invariant: the element count comes from the public setup, never from the proof.
        //
        //     6 row variables  ->  two elements expected, a proof without them
        //     2 row variables  ->  none expected, a proof carrying two
        //
        // The rejection is structural, so it may not advance the sponge.
        let witness = bits(0xC0DE, 64);
        let packing = BitPacking::<EF>::new(&witness).unwrap();

        let r_sending = successor_point(0xC0DF, 9, 6, Selector::Random);
        let sending = BitRingSwitch::with_successor(&r_sending, 6).unwrap();
        let (mut missing, _, _) = sending.prove::<EF, _, _>(&packing, &mut challenger());
        let spare = missing.successor.take();

        let r_fitting = successor_point(0xC0E0, 9, 2, Selector::Random);
        let fitting = BitRingSwitch::with_successor(&r_fitting, 2).unwrap();
        let (mut spurious, _, _) = fitting.prove::<EF, _, _>(&packing, &mut challenger());
        spurious.successor = spare;

        for (reduction, proof, r, expected, actual) in [
            (&sending, &missing, &r_sending, 2, 0),
            (&fitting, &spurious, &r_fitting, 0, 2),
        ] {
            let current = embedded(&witness).eval_base(r);
            let mut used = challenger();
            let err = reduction
                .verify_readings(proof, Some(current), None, &mut used)
                .unwrap_err();

            assert_eq!(
                err,
                BitRingSwitchProofError::Width(TranscriptWidth::SuccessorElements {
                    expected,
                    actual
                })
            );
            assert_eq!(
                CanSample::<EF>::sample(&mut used),
                CanSample::<EF>::sample(&mut challenger()),
                "the rejection advanced the sponge"
            );
        }
    }

    #[test]
    fn a_reading_the_setup_does_not_answer_is_refused() {
        // A plain reduction has no successor view to check a next claim against.
        // A call supplying neither reading would check nothing at all.
        let (reduction, packing, _, _) = fixture(0x2EAD, 32);
        let (proof, _, _) = reduction.prove::<EF, _, _>(&packing, &mut challenger());

        assert_eq!(
            reduction
                .verify_readings(&proof, None, Some(EF::ONE), &mut challenger())
                .unwrap_err(),
            BitRingSwitchProofError::NoSuccessorView
        );
        assert_eq!(
            reduction
                .verify_readings(&proof, None, None, &mut challenger())
                .unwrap_err(),
            BitRingSwitchProofError::NoReading
        );
    }

    #[test]
    fn a_successor_setup_of_the_wrong_kind_is_refused() {
        // A row count wider than the point names no rows at all.
        // Each batching stage belongs to one kind of reduction.
        // A plain reduction has no successor claim to read.
        let mut rng = SmallRng::seed_from_u64(0x5E7C);
        let r = Point::<EF>::rand(&mut rng, 9);
        let r_batch = Point::<EF>::rand(&mut rng, BitRingSwitch::<EF>::ABSORBED);

        assert_eq!(
            BitRingSwitch::with_successor(&r, 10).unwrap_err(),
            BitRingSwitchError::RowVariables {
                row_variables: 10,
                num_variables: 9
            }
        );
        assert_eq!(
            BitRingSwitch::with_successor(&r, 6)
                .unwrap()
                .batch(&r_batch)
                .unwrap_err(),
            BitRingSwitchError::SuccessorBatching { expected: true }
        );
        assert_eq!(
            BitRingSwitch::with_successor(&r, 4)
                .unwrap()
                .batch_with_successor(&r_batch, EF::ONE)
                .unwrap_err(),
            BitRingSwitchError::SuccessorBatching { expected: false }
        );
        assert_eq!(
            BitRingSwitch::new(&r)
                .unwrap()
                .successor_claim(&BitTensor::zero(), None)
                .unwrap_err(),
            BitRingSwitchError::NoSuccessorView
        );
    }

    /// A 128-bit reduction, played once per representation its rounds can run in.
    ///
    /// The test level of this module is 16 bits wide and has no second representation,
    /// so the comparison widens to the level that does.
    ///
    /// # Returns
    ///
    /// The proof, the point the rounds ended at, the surviving claim, and the sponge left
    /// behind, once for the tower basis and once for the polynomial basis.
    #[allow(clippy::type_complexity)]
    fn both_representations(
        reduction: &BitRingSwitch<BinaryField128>,
        packing: &BitPacking<BinaryField128>,
    ) -> [(
        BitRingSwitchProof<BinaryField128>,
        Point<BinaryField128>,
        BinaryField128,
        WideChal,
    ); 2] {
        let mut tower_sponge = WideChal::from_hasher(Vec::new(), Keccak256Hash);
        let tower = reduction.prove::<BinaryField128, _, _>(packing, &mut tower_sponge);
        let mut poly_sponge = WideChal::from_hasher(Vec::new(), Keccak256Hash);
        let poly = reduction.prove::<Ghash128, _, _>(packing, &mut poly_sponge);

        [
            (tower.0, tower.1, tower.2, tower_sponge),
            (poly.0, poly.1, poly.2, poly_sponge),
        ]
    }

    #[test]
    fn a_borrowed_packing_plays_the_owned_packing_transcript() {
        // Invariant: the store the elements are read from reaches no wire.
        //
        // A commitment hands the reduction a view over the table it already holds,
        // and that is the only route production takes.
        let (reduction, packing, _, _) = fixture(0x5AE, 16);
        let borrowed = BitPacking::from_packed(packing.poly().as_view()).unwrap();
        assert_eq!(borrowed.num_variables(), packing.num_variables());

        let (owned_proof, owned_point, owned_value) =
            reduction.prove::<EF, _, _>(&packing, &mut challenger());
        let (borrowed_proof, borrowed_point, borrowed_value) =
            reduction.prove::<EF, _, _>(&borrowed, &mut challenger());

        assert_eq!(borrowed_proof.tensor, owned_proof.tensor);
        assert_eq!(borrowed_proof.successor, owned_proof.successor);
        assert_eq!(
            borrowed_proof.sumcheck.polynomial_evaluations,
            owned_proof.sumcheck.polynomial_evaluations
        );
        assert_eq!(borrowed_proof.final_eval, owned_proof.final_eval);
        assert_eq!(borrowed_point, owned_point);
        assert_eq!(borrowed_value, owned_value);
    }

    #[test]
    fn the_polynomial_basis_rounds_play_the_tower_basis_transcript() {
        // Invariant: the representation the rounds run in reaches no wire.
        //
        // The two maps are inverse field isomorphisms, so every round polynomial is the
        // one the tower-basis tables measure, message by message.
        let witness = bits(0x9F17, 512);
        let packing = BitPacking::<BinaryField128>::new(&witness).unwrap();
        let absorbed = BitRingSwitch::<BinaryField128>::ABSORBED;
        let mut rng = SmallRng::seed_from_u64(0x2C0D);
        let num_variables = packing.num_variables() + absorbed;

        // A random point, one whose Boolean prefix drops rounds, and one sending successor
        // elements, so every stage that crosses into the representation is on the path.
        let random = Point::<BinaryField128>::rand(&mut rng, num_variables);
        let mut addressed = random.as_slice().to_vec();
        addressed[0] = BinaryField128::ONE;
        addressed[1] = BinaryField128::ZERO;
        let addressed = Point::new(addressed);
        let reductions = [
            (BitRingSwitch::new(&random).unwrap(), &random),
            (BitRingSwitch::new(&addressed).unwrap(), &addressed),
            (
                BitRingSwitch::with_successor(&random, absorbed + 1).unwrap(),
                &random,
            ),
        ];
        let embedding = embedded_over::<BinaryField128>(&witness);

        for (case, (reduction, point)) in reductions.iter().enumerate() {
            let [
                (tower, tower_point, tower_value, mut tower_sponge),
                (poly, poly_point, poly_value, mut poly_sponge),
            ] = both_representations(reduction, &packing);

            // A Boolean prefix drops rounds, so the kept coordinates do not count them.
            assert!(poly.sumcheck.num_rounds() > 0, "case {case} runs no round");
            assert_eq!(poly.tensor, tower.tensor, "case {case}");
            assert_eq!(poly.successor, tower.successor, "case {case}");
            assert_eq!(
                poly.sumcheck.polynomial_evaluations, tower.sumcheck.polynomial_evaluations,
                "case {case}"
            );
            assert_eq!(
                poly.sumcheck.pow_witnesses, tower.sumcheck.pow_witnesses,
                "case {case}"
            );
            assert_eq!(poly.final_eval, tower.final_eval, "case {case}");
            assert_eq!(poly_point, tower_point, "case {case}");
            assert_eq!(poly_value, tower_value, "case {case}");
            assert_eq!(
                CanSample::<BinaryField128>::sample(&mut poly_sponge),
                CanSample::<BinaryField128>::sample(&mut tower_sponge),
                "case {case}"
            );

            // Agreeing legs would still agree if both were wrong in the same way.
            // The claim is read off the witness, so the replay has to close on it.
            let claim = embedding.eval_base(*point);
            let mut verifier_sponge = WideChal::from_hasher(Vec::new(), Keccak256Hash);
            assert!(
                reduction.verify(&poly, claim, &mut verifier_sponge).is_ok(),
                "case {case}"
            );
        }
    }
}
