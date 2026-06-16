//! The [`SvoPoint`] type: a challenge point split into an SVO portion and a
//! residual split-eq portion, with the opening and residual-weight methods that
//! drive the SVO sumcheck rounds.

use alloc::vec::Vec;

use itertools::Itertools;
use p3_field::{ExtensionField, Field, add_scaled_slice_in_place};
use p3_maybe_rayon::prelude::*;
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;
use p3_multilinear_util::split_eq::SplitEq;
use p3_util::log2_strict_usize;

use crate::layout::{EqPartials, EqSvoPartials, NextPartials, NextSvoPartials};
use crate::strategy::VariableOrder;

/// Output length at or above which residual weight accumulation runs in parallel.
///
/// # Why this value
///
/// - Below `4096` output entries the per-chunk work is too small to amortize thread spawning.
/// - At or above it the parallel sweep over split-variable chunks outpaces the serial sweep.
const PARALLEL_THRESHOLD: usize = 4096;

/// Challenge point split into an SVO prefix and a residual split-eq suffix.
///
/// The split direction depends on [`VariableOrder`]:
/// - `Prefix`: `z_svo` is the prefix `l0` variables and `z_split` represents the
///   remaining suffix.
/// - `Suffix`: `z_svo` is the suffix `l0` variables and `z_split` represents the
///   remaining prefix.
#[derive(Debug, Clone)]
pub struct SvoPoint<F: Field, EF: ExtensionField<F>> {
    /// The first `k_svo` coordinates of the original point, handled by the SVO
    /// accumulator rounds.
    pub(crate) z_svo: Point<EF>,
    /// A factored table for `eq(z_rest, ·)` on the remaining coordinates after
    /// removing `z_svo` from the original point.
    pub(crate) z_split: SplitEq<F, EF>,
    /// Variable processing order
    var_order: VariableOrder,
}

impl<F: Field, EF: ExtensionField<F>> SvoPoint<F, EF> {
    /// Splits a challenge point into the SVO portion and the residual split-eq
    /// portion according to `var_order`.
    ///
    /// `l0` is the number of variables handled by the SVO optimization.
    pub fn new_unpacked(l0: usize, point: &Point<EF>, var_order: VariableOrder) -> Self {
        let (svo, split) = Self::split_svo(l0, point, var_order);
        Self::from_parts(svo, SplitEq::new_unpacked(&split, EF::ONE), var_order)
    }

    /// Splits a challenge point into a prefix SVO portion and a residual suffix.
    ///
    /// `l0` is the number of variables handled by the SVO optimization.
    pub fn new_packed(l0: usize, point: &Point<EF>) -> Self {
        let (svo, split) = Self::split_svo(l0, point, VariableOrder::Prefix);
        Self::from_parts(
            svo,
            SplitEq::new_packed(&split, EF::ONE),
            VariableOrder::Prefix,
        )
    }

    /// Splits `point` into its SVO portion and the residual portion at depth `l0`.
    ///
    /// For `Prefix` the SVO portion is the leading `l0` coordinates.
    /// For `Suffix` it is the trailing `l0` coordinates.
    fn split_svo(l0: usize, point: &Point<EF>, var_order: VariableOrder) -> (Point<EF>, Point<EF>) {
        assert!(l0 <= point.num_variables());
        match var_order {
            VariableOrder::Prefix => point.split_at(l0),
            VariableOrder::Suffix => {
                let (split, svo) = point.split_at(point.num_variables() - l0);
                (svo, split)
            }
        }
    }

    /// Assembles an [`SvoPoint`] from its SVO portion, residual split-eq, and variable order.
    const fn from_parts(
        z_svo: Point<EF>,
        z_split: SplitEq<F, EF>,
        var_order: VariableOrder,
    ) -> Self {
        Self {
            z_svo,
            z_split,
            var_order,
        }
    }

    /// Accumulates this claim's residual equality table into a buffer.
    ///
    /// Once the SVO rounds have fixed the suffix variables to `rs`, the remaining
    /// weight vector is:
    /// ```text
    /// alpha  · eq(z_rest, x_rest) · eq(z_svo, rs)
    /// ```
    /// for every assignment `x_rest` to the non-SVO variables.
    ///
    /// This method computes the scalar factor `alpha · eq(z_svo, rs)` and then asks
    /// `split_eq` to materialize the residual `eq(z_rest, ·)` table into `out`.
    pub fn accumulate_into(&self, out: &mut [EF], rs: &Point<EF>, scale: EF) {
        self.z_split
            .accumulate_into(out, Some(self.accumulate_scale(rs, scale)));
    }

    /// Folds the SVO equality weight `eq(z_svo, rs)` into the batching coefficient `scale`.
    ///
    /// Returns `scale · eq(z_svo, rs)`, the scalar applied to the residual `eq(z_rest, ·)` table.
    fn accumulate_scale(&self, rs: &Point<EF>, scale: EF) -> EF {
        assert_eq!(rs.num_variables(), self.num_variables_svo());
        scale * Point::eval_eq(self.z_svo.as_slice(), rs.as_slice())
    }

    /// Accumulates this claim's residual equality table into a packed buffer.
    ///
    /// Once the SVO rounds have fixed the suffix variables to `rs`, the remaining
    /// weight vector is:
    /// ```text
    /// alpha  · eq(z_rest, x_rest) · eq(z_svo, rs)
    /// ```
    /// for every assignment `x_rest` to the non-SVO variables.
    ///
    /// This method computes the scalar factor `alpha · eq(z_svo, rs)` and then asks
    /// `split_eq` to materialize the residual `eq(z_rest, ·)` table into `out`.
    pub fn accumulate_into_packed(
        &self,
        out: &mut [EF::ExtensionPacking],
        rs: &Point<EF>,
        scale: EF,
    ) {
        assert!(matches!(self.var_order, VariableOrder::Prefix));
        self.z_split
            .accumulate_into_packed(out, Some(self.accumulate_scale(rs, scale)));
    }

    /// Evaluates `poly` at this point and returns all partial evaluations seen
    /// during the SVO rounds.
    ///
    /// The non-SVO prefix is compressed first using `SplitEq`. The result is a
    /// polynomial only in the SVO variables, which is then:
    /// - evaluated at `z_svo` to obtain the opening value
    /// - partially compressed after each SVO round to feed the accumulator path
    pub fn eval(&self, poly: &Poly<F>) -> (EF, EqSvoPartials<EF>) {
        assert_eq!(self.num_variables(), poly.num_variables());
        // Each per-round compression is wrapped as an equality payload as it is produced.
        let (compressed, partial_evals) = match self.var_order {
            VariableOrder::Prefix => {
                let compressed = self.z_split.compress_suffix(poly);
                let partial_evals = (1..=self.num_variables_svo())
                    .map(|i| {
                        let (_svo_active, svo_rest) = self.z_svo.split_at(i);
                        EqPartials::new(compressed.compress_suffix(&svo_rest, EF::ONE))
                    })
                    .collect::<Vec<_>>();
                (compressed, partial_evals)
            }
            VariableOrder::Suffix => {
                let compressed = self.z_split.compress_prefix(poly);
                let partial_evals = (1..=self.num_variables_svo())
                    .map(|i| {
                        let (svo_rest, _svo_active) =
                            self.z_svo.split_at(self.z_svo.num_variables() - i);
                        EqPartials::new(compressed.compress_prefix(&svo_rest, EF::ONE))
                    })
                    .collect::<Vec<_>>();
                (compressed, partial_evals)
            }
        };
        // Evaluate the fully compressed SVO-only polynomial to get the scalar opening value.
        let eval = compressed.eval_base(&self.z_svo);
        (eval, EqSvoPartials::new(partial_evals))
    }

    /// Returns the number of SVO variables (`l0`).
    ///
    /// This is the depth of the SVO optimization.
    /// These coordinates are processed via the accumulator-based Lagrange
    /// interpolation path rather than the standard fold-and-sum path.
    pub const fn num_variables_svo(&self) -> usize {
        self.z_svo.num_variables()
    }

    /// Returns the number of variables of the represented point.
    pub const fn num_variables(&self) -> usize {
        self.z_svo.num_variables() + self.z_split.num_variables()
    }

    /// Returns the SVO suffix of the represented point.
    pub const fn z_svo(&self) -> &Point<EF> {
        &self.z_svo
    }

    /// Returns the factored equality table for the non-SVO prefix of the represented point.
    pub const fn z_split(&self) -> &SplitEq<F, EF> {
        &self.z_split
    }

    /// Returns the original point represented by this struct.
    pub const fn var_order(&self) -> VariableOrder {
        self.var_order
    }

    /// Adds the residual suffix-layout successor weight over the split variables.
    ///
    /// # Overview
    ///
    /// Once the SVO variables are fixed to the round challenges, the leftover
    /// weight over the split variables is a sum of three terms:
    ///
    /// - a "done" scalar times the split equality weights,
    /// - a "carry" scalar times the split equality weights shifted up one row,
    /// - an "omega" scalar times the all-ones split boundary weight.
    ///
    /// The result is added into the output scaled by the batching coefficient,
    /// without ever materializing a dense successor table.
    ///
    /// # Arguments
    ///
    /// - `out`: the split-variable weight buffer to accumulate into.
    /// - `rs`: the SVO round challenges that fix the SVO variables.
    /// - `scale`: the batching coefficient applied to this opening.
    ///
    /// # Panics
    ///
    /// - If the point is not in suffix layout.
    /// - If the number of challenges does not match the number of SVO variables.
    /// - If the output length does not match the number of split variables.
    pub fn accumulate_next_suffix_into(&self, out: &mut [EF], rs: &Point<EF>, scale: EF) {
        // This closed form is only derived for the suffix SVO layout.
        assert!(
            matches!(self.var_order(), VariableOrder::Suffix),
            "next residual weights are implemented for suffix SVO only"
        );
        // One round challenge per SVO variable.
        assert_eq!(rs.num_variables(), self.num_variables_svo());
        // The output buffer spans exactly the split variables.
        assert_eq!(log2_strict_usize(out.len()), self.z_split.num_variables());

        // Closed-form successor decomposition of the SVO part at the round challenges.
        let (carry, done, omega) = Point::eval_next(self.z_svo.as_slice(), rs.as_slice());
        // Pre-multiply each state scalar by the batching coefficient.
        let done_scale = scale * done;
        let carry_scale = scale * carry;
        let omega_scale = scale * omega;

        // Reference path: build the same residual densely to cross-check the fast path.
        #[cfg(debug_assertions)]
        let expected = {
            let mut expected = out.to_vec();
            let eq = self.z_split.materialize();
            // Done term: split equality weights, aligned row by row.
            expected
                .iter_mut()
                .zip_eq(eq.iter())
                .for_each(|(out, &weight)| *out += done_scale * weight);
            // Carry term: split equality weights shifted up by one row.
            expected
                .iter_mut()
                .skip(1)
                .zip_eq(eq.as_slice()[..eq.num_evals() - 1].iter())
                .for_each(|(out, &weight)| *out += carry_scale * weight);
            // Omega term: only the final all-ones split row.
            *expected.last_mut().unwrap() += omega_scale * self.z_split.last_scalar();
            expected
        };

        // The split equality table is factored as an outer (eq0) over an inner (eq1) block.
        let eq1 = self.z_split.eq1();
        // Each outer entry owns one contiguous inner chunk of the output.
        let cs = eq1.scalar_chunk_size();
        // Weight of the all-ones inner row, used to stitch the carry across chunk boundaries.
        let eq1_last = eq1.last_scalar();
        // Serial path: cheaper than thread spawning for small outputs.
        if out.len() < PARALLEL_THRESHOLD {
            // Carry into the first row of a chunk comes from the last row of the previous chunk.
            let mut prev_last = EF::ZERO;
            out.chunks_mut(cs)
                .zip(self.z_split.eq0().iter())
                .for_each(|(chunk, &w0)| {
                    // Scale done and carry by the outer weight; pass the cross-chunk carry seed.
                    eq1.accumulate_next_chunk_into(
                        chunk,
                        done_scale * w0,
                        carry_scale * w0,
                        carry_scale * prev_last,
                    );
                    // Carry seed for the next chunk: this chunk's all-ones inner row.
                    prev_last = w0 * eq1_last;
                });
        } else {
            // Parallel path: each chunk recomputes its own cross-chunk carry from the outer table.
            let eq0 = self.z_split.eq0().as_slice();
            out.par_chunks_mut(cs)
                .enumerate()
                .zip(eq0.par_iter())
                .for_each(|((idx, chunk), &w0)| {
                    // Boundary carry depends on the previous outer entry, except for the first chunk.
                    let boundary = if idx > 0 {
                        carry_scale * eq0[idx - 1] * eq1_last
                    } else {
                        EF::ZERO
                    };
                    eq1.accumulate_next_chunk_into(
                        chunk,
                        done_scale * w0,
                        carry_scale * w0,
                        boundary,
                    );
                });
        }

        // Omega contributes only to the global all-ones split row.
        *out.last_mut().unwrap() += omega_scale * self.z_split.last_scalar();

        // Fast and dense paths must agree exactly.
        #[cfg(debug_assertions)]
        debug_assert!(out == expected.as_slice());
    }

    /// Adds the residual prefix-layout successor weight over the split variables.
    ///
    /// # Overview
    ///
    /// Once the prefix SVO variables are fixed to the round challenges, the
    /// leftover weight over the split variables is a sum of three terms:
    ///
    /// - a "done" scalar times the split equality weights shifted up one row,
    /// - a "carry" scalar landing only on the first (all-zeros) split row,
    /// - an "omega" scalar landing only on the last (all-ones) split row.
    ///
    /// The result is added into the output scaled by the batching coefficient.
    ///
    /// # Arguments
    ///
    /// - `out`: the split-variable weight buffer to accumulate into.
    /// - `rs`: the SVO round challenges that fix the SVO variables.
    /// - `scale`: the batching coefficient applied to this opening.
    ///
    /// # Panics
    ///
    /// - If the point is not in prefix layout.
    /// - If the number of challenges does not match the number of SVO variables.
    /// - If the output length does not match the number of split variables.
    pub fn accumulate_next_prefix_into(&self, out: &mut [EF], rs: &Point<EF>, scale: EF) {
        // This closed form is only derived for the prefix SVO layout.
        assert!(
            matches!(self.var_order(), VariableOrder::Prefix),
            "prefix next residual weights require prefix SVO"
        );
        // One round challenge per SVO variable.
        assert_eq!(rs.num_variables(), self.num_variables_svo());
        // The output buffer spans exactly the split variables.
        assert_eq!(log2_strict_usize(out.len()), self.z_split.num_variables());

        // Successor decomposition of the SVO part; prefix layout uses the done and omega states.
        let (_carry, done, omega) = Point::eval_next(self.z_svo.as_slice(), rs.as_slice());
        // Prefix layout also needs the plain equality weight of the SVO part.
        let eq = Point::eval_eq(self.z_svo.as_slice(), rs.as_slice());
        // Pre-multiply each state scalar by the batching coefficient.
        let done_scale = scale * eq;
        let carry_scale = scale * done;
        let omega_scale = scale * omega;

        // Reference path: build the same residual densely to cross-check the fast path.
        #[cfg(debug_assertions)]
        let expected = {
            let mut expected = out.to_vec();
            let eq = self.z_split.materialize();
            // Done term: split equality weights shifted up by one row.
            expected
                .iter_mut()
                .skip(1)
                .zip_eq(eq.as_slice()[..eq.num_evals() - 1].iter())
                .for_each(|(out, &weight)| *out += done_scale * weight);
            // Carry and omega land only on the two boundary rows.
            let boundary = self.z_split.last_scalar();
            *expected.first_mut().unwrap() += carry_scale * boundary;
            *expected.last_mut().unwrap() += omega_scale * boundary;
            expected
        };

        // The split equality table is factored as an outer (eq0) over an inner (eq1) block.
        let eq1 = self.z_split.eq1();
        // Each outer entry owns one contiguous inner chunk of the output.
        let cs = eq1.scalar_chunk_size();
        // Weight of the all-ones inner row, used to stitch the shift across chunk boundaries.
        let eq1_last = eq1.last_scalar();
        // Serial path: cheaper than thread spawning for small outputs.
        if out.len() < PARALLEL_THRESHOLD {
            // Shifted weight into the first row of a chunk comes from the last row of the previous chunk.
            let mut prev_last = EF::ZERO;
            out.chunks_mut(cs)
                .zip(self.z_split.eq0().iter())
                .for_each(|(chunk, &w0)| {
                    // No row-aligned done term here; only the shifted-up contribution and its seed.
                    eq1.accumulate_next_chunk_into(
                        chunk,
                        EF::ZERO,
                        done_scale * w0,
                        done_scale * prev_last,
                    );
                    // Seed for the next chunk: this chunk's all-ones inner row.
                    prev_last = w0 * eq1_last;
                });
        } else {
            // Parallel path: each chunk recomputes its own cross-chunk shift from the outer table.
            let eq0 = self.z_split.eq0().as_slice();
            out.par_chunks_mut(cs)
                .enumerate()
                .zip(eq0.par_iter())
                .for_each(|((idx, chunk), &w0)| {
                    // Boundary shift depends on the previous outer entry, except for the first chunk.
                    let boundary = if idx > 0 {
                        done_scale * eq0[idx - 1] * eq1_last
                    } else {
                        EF::ZERO
                    };
                    eq1.accumulate_next_chunk_into(chunk, EF::ZERO, done_scale * w0, boundary);
                });
        }

        // Carry and omega contribute only to the global first and last split rows.
        let boundary = self.z_split.last_scalar();
        *out.first_mut().unwrap() += carry_scale * boundary;
        *out.last_mut().unwrap() += omega_scale * boundary;

        // Fast and dense paths must agree exactly.
        #[cfg(debug_assertions)]
        debug_assert!(out == expected.as_slice());
    }

    /// Evaluates a suffix-layout repeat-last-successor opening and caches its SVO rounds.
    ///
    /// # Overview
    ///
    /// - The witness polynomial is compressed over the split prefix into three payloads over the SVO suffix.
    /// - The three payloads are weighted respectively by the equality, carry, and boundary split states.
    /// - The payloads give the scalar opening value and one cached per-round successor table.
    ///
    /// # Arguments
    ///
    /// - `poly`: the raw witness polynomial over all variables.
    /// - `d_eq`: an optional precomputed equality payload reused from an earlier opening.
    ///
    /// # Returns
    ///
    /// - The scalar opening value of the successor weight against the polynomial.
    /// - One cached successor table per SVO sumcheck round.
    ///
    /// # Panics
    ///
    /// - If the polynomial does not span all point variables.
    /// - If the point is not in suffix layout.
    /// - If a supplied equality payload does not span the SVO variables.
    pub fn eval_next_suffix(
        &self,
        poly: &Poly<F>,
        d_eq: Option<&Poly<EF>>,
    ) -> (EF, NextSvoPartials<EF>) {
        // The polynomial must cover both the split and SVO variables.
        assert_eq!(self.num_variables(), poly.num_variables());
        // This routine derives its compressions only for suffix layout.
        assert!(
            matches!(self.var_order(), VariableOrder::Suffix),
            "next openings are implemented for suffix SVO only"
        );

        // Compress the equality payload over the split prefix unless the caller supplied it.
        let d_eq_owned = d_eq.is_none().then(|| self.z_split.compress_prefix(poly));
        let d_eq = d_eq.unwrap_or_else(|| d_eq_owned.as_ref().unwrap());
        assert_eq!(d_eq.num_variables(), self.num_variables_svo());

        // A caller-supplied payload must equal the freshly computed one.
        #[cfg(debug_assertions)]
        if d_eq_owned.is_none() {
            debug_assert_eq!(*d_eq, self.z_split.compress_prefix(poly));
        }
        // Carry payload: compress the polynomial over the split prefix with a one-row shift.
        let d_t = self.z_split.compress_prefix_shifted(poly);

        // Number of Boolean rows over the SVO and split variables.
        let svo_rows = 1 << self.num_variables_svo();
        let split_rows = 1 << self.z_split.num_variables();
        // The boundary weight is the all-ones corner of the split equality table.
        let omega_scale = self.z_split.last_scalar();
        // Suffix layout stores the all-ones split block at the very end of the polynomial.
        let omega_start = (split_rows - 1) * svo_rows;
        // Boundary payload: that final split block scaled by the boundary weight.
        let d_omega = Poly::new(
            poly.as_slice()[omega_start..omega_start + svo_rows]
                .iter()
                .map(|&value| omega_scale * value)
                .collect(),
        );

        // Closed-form successor state tables of the SVO point over all rows, built once.
        let states = NextPartials::from_point(self.z_svo.as_slice());

        // Weight the three state tables against the payloads and cache one table per round.
        self.eval_next_with(
            [
                states.done().as_slice(),
                states.carry().as_slice(),
                states.omega().as_slice(),
            ],
            [d_eq, &d_t, &d_omega],
            |active_len| self.next_round_partials_suffix(d_eq, &d_t, &d_omega, active_len),
        )
    }

    /// Evaluates a prefix-layout repeat-last-successor opening and caches its SVO rounds.
    ///
    /// # Overview
    ///
    /// - The witness polynomial is compressed over the split suffix into three payloads over the SVO prefix.
    /// - The payloads are the shifted-done weight, the carry boundary, and the omega boundary.
    /// - The payloads give the scalar opening value and one cached per-round successor table.
    ///
    /// # Arguments
    ///
    /// - `poly`: the raw witness polynomial over all variables.
    ///
    /// # Returns
    ///
    /// - The scalar opening value of the successor weight against the polynomial.
    /// - One cached successor table per SVO sumcheck round.
    ///
    /// # Panics
    ///
    /// - If the polynomial does not span all point variables.
    /// - If the point is not in prefix layout.
    pub fn eval_next_prefix(&self, poly: &Poly<F>) -> (EF, NextSvoPartials<EF>) {
        // The polynomial must cover both the split and SVO variables.
        assert_eq!(self.num_variables(), poly.num_variables());
        // This routine derives its compressions only for prefix layout.
        assert!(
            matches!(self.var_order(), VariableOrder::Prefix),
            "prefix next openings require prefix SVO"
        );

        // Done payload: compress over the split suffix with a one-row shift.
        let d_done = self.z_split.compress_suffix_shifted(poly);

        // Number of Boolean rows over the SVO and split variables.
        let svo_rows = 1 << self.num_variables_svo();
        let split_rows = 1 << self.z_split.num_variables();
        // The boundary weight is the all-ones corner of the split equality table.
        let boundary = self.z_split.last_scalar();
        // Carry payload over the SVO prefix, one entry per SVO row.
        let mut d_carry = EF::zero_vec(svo_rows);
        // Omega (boundary) payload over the SVO prefix, one entry per SVO row.
        let mut d_omega = EF::zero_vec(svo_rows);

        // Prefix layout makes each SVO row a contiguous split chunk of the polynomial.
        d_carry
            .iter_mut()
            .zip_eq(d_omega.iter_mut())
            .zip_eq(poly.as_slice().chunks(split_rows))
            .for_each(|((d_carry, d_omega), chunk)| {
                // Carry enters at the all-zeros split corner (first chunk entry).
                *d_carry = boundary * chunk.first().copied().unwrap();
                // Omega exits at the all-ones split corner (last chunk entry).
                *d_omega = boundary * chunk.last().copied().unwrap();
            });

        // Wrap the boundary payloads as polynomials over the SVO prefix.
        let d_carry = Poly::new(d_carry);
        let d_omega = Poly::new(d_omega);

        // Closed-form successor state tables and plain equality weights, built once.
        let states = NextPartials::from_point(self.z_svo.as_slice());
        let eq = Poly::new_from_point(self.z_svo.as_slice(), EF::ONE);

        // Weight the three tables against the payloads and cache one table per round.
        self.eval_next_with(
            [
                eq.as_slice(),
                states.done().as_slice(),
                states.omega().as_slice(),
            ],
            [&d_done, &d_carry, &d_omega],
            |active_len| self.next_round_partials_prefix(&d_done, &d_carry, &d_omega, active_len),
        )
    }

    /// Assembles a repeat-last-successor opening value and its per-round successor tables.
    ///
    /// # Overview
    ///
    /// - Each SVO row pairs three precomputed successor-weight tables with the matching payloads.
    /// - The opening value sums those weighted contributions over every SVO row.
    /// - One successor table is cached per SVO sumcheck round through the round tail.
    ///
    /// # Arguments
    ///
    /// - `weights`: the three per-row weight tables, in the same order as the payloads.
    /// - `payloads`: the three per-row payloads, in the same order as the weights.
    /// - `round_tail`: maps the number of active SVO variables to the cached round table.
    ///
    /// # Returns
    ///
    /// - The scalar opening value of the successor weight against the payloads.
    /// - One cached successor table per SVO sumcheck round.
    fn eval_next_with(
        &self,
        weights: [&[EF]; 3],
        payloads: [&Poly<EF>; 3],
        mut round_tail: impl FnMut(usize) -> NextPartials<EF>,
    ) -> (EF, NextSvoPartials<EF>) {
        let [w0, w1, w2] = weights;
        let [p0, p1, p2] = payloads;
        // Number of Boolean rows over the SVO variables.
        let svo_rows = 1 << self.num_variables_svo();

        // Opening value: sum each weight table against its payload over every SVO row.
        let eval = (0..svo_rows)
            .map(|svo_idx| {
                w0[svo_idx] * p0.as_slice()[svo_idx]
                    + w1[svo_idx] * p1.as_slice()[svo_idx]
                    + w2[svo_idx] * p2.as_slice()[svo_idx]
            })
            .sum();

        // Cache one compressed successor table per SVO sumcheck round.
        let rounds = (1..=self.num_variables_svo())
            .map(&mut round_tail)
            .collect::<Vec<_>>();

        (eval, NextSvoPartials::new(rounds))
    }

    /// Compresses suffix-layout repeat-last-successor payloads to the active SVO variables.
    ///
    /// # Overview
    ///
    /// - The three input payloads are indexed by all SVO variables, with the active suffix as the low bits.
    /// - The already-folded SVO prefix is fixed at the corresponding `z_svo` coordinates.
    /// - The output is the three-table successor decomposition over only the active suffix variables.
    ///
    /// # Arguments
    ///
    /// - `d_eq`: payload weighted by the equality state of the successor map.
    /// - `d_t`: payload weighted by the carry-into-next state.
    /// - `d_omega`: payload weighted by the wrap-around boundary state.
    /// - `active_len`: the number of active suffix variables kept for this round.
    ///
    /// # Panics
    ///
    /// - If `active_len` is zero or exceeds the number of SVO variables.
    /// - If any payload does not span all SVO variables.
    fn next_round_partials_suffix(
        &self,
        d_eq: &Poly<EF>,
        d_t: &Poly<EF>,
        d_omega: &Poly<EF>,
        active_len: usize,
    ) -> NextPartials<EF> {
        // Total SVO variables carried by each payload.
        let svo_len = self.z_svo.num_variables();
        // A round always keeps at least one active variable, never more than all of them.
        assert!(active_len > 0);
        assert!(active_len <= svo_len);
        // Every payload must be indexed by all SVO variables before compression.
        assert_eq!(d_eq.num_variables(), svo_len);
        assert_eq!(d_t.num_variables(), svo_len);
        assert_eq!(d_omega.num_variables(), svo_len);

        // Number of already-folded prefix variables to fix.
        let rest_len = svo_len - active_len;
        // Number of Boolean rows over the active suffix variables.
        let active_rows = 1 << active_len;

        // No prefix to fold: the payloads already live over only the active variables.
        if rest_len == 0 {
            return NextPartials::new(d_eq.clone(), d_t.clone(), d_omega.clone());
        }

        // Suffix layout puts the folded prefix in the high bits of the index.
        let (p_rest, _p_active) = self.z_svo.split_at(rest_len);
        // Equality table over the prefix used to contract it away.
        let rest_eq = SplitEq::<EF, EF>::new_packed(&p_rest, EF::ONE);

        // Done state: contract the equality payload over the prefix at the matching active row.
        let done = rest_eq.compress_prefix(d_eq);
        // Carry state: contract the equality payload over the prefix shifted by one successor step.
        let mut carry = rest_eq.compress_prefix_shifted(d_eq);

        // The all-ones prefix corner carries the full product of prefix coordinates.
        let carry_scale = p_rest.iter().copied().product::<EF>();
        // Add the carry-into-next contribution from the first active block of the carry payload.
        // The shared kernel runs this fused multiply-add over SIMD lanes.
        add_scaled_slice_in_place(
            carry.as_mut_slice(),
            &d_t.as_slice()[..active_rows],
            carry_scale,
        );

        // Number of Boolean rows over the folded prefix variables.
        let rest_rows = 1 << rest_len;
        // The wrap-around boundary picks up the same all-ones prefix product.
        let omega_scale = carry_scale;
        // Omega lives only on the last prefix block, the all-ones prefix row.
        let omega_start = (rest_rows - 1) * active_rows;
        // Scale that final active block to obtain the boundary state over the active variables.
        let omega = Poly::new(
            d_omega.as_slice()[omega_start..omega_start + active_rows]
                .iter()
                .map(|&value| omega_scale * value)
                .collect(),
        );

        NextPartials::new(done, carry, omega)
    }

    /// Compresses prefix-layout repeat-last-successor payloads to the active SVO variables.
    ///
    /// # Overview
    ///
    /// - The three input payloads are indexed by all SVO variables, with the active prefix as the high bits.
    /// - The already-folded SVO suffix is fixed at the corresponding `z_svo` coordinates.
    /// - The output is the three-table successor decomposition over only the active prefix variables.
    ///
    /// # Arguments
    ///
    /// - `d_done`: payload weighted by the shifted equality state of the successor map.
    /// - `d_carry`: payload weighted by the carry-into-next state.
    /// - `d_omega`: payload weighted by the wrap-around boundary state.
    /// - `active_len`: the number of active prefix variables kept for this round.
    ///
    /// # Panics
    ///
    /// - If `active_len` is zero or exceeds the number of SVO variables.
    /// - If any payload does not span all SVO variables.
    fn next_round_partials_prefix(
        &self,
        d_done: &Poly<EF>,
        d_carry: &Poly<EF>,
        d_omega: &Poly<EF>,
        active_len: usize,
    ) -> NextPartials<EF> {
        // Total SVO variables carried by each payload.
        let svo_len = self.z_svo.num_variables();
        // A round always keeps at least one active variable, never more than all of them.
        assert!(active_len > 0);
        assert!(active_len <= svo_len);
        // Every payload must be indexed by all SVO variables before compression.
        assert_eq!(d_done.num_variables(), svo_len);
        assert_eq!(d_carry.num_variables(), svo_len);
        assert_eq!(d_omega.num_variables(), svo_len);

        // Number of already-folded suffix variables to fix.
        let rest_len = svo_len - active_len;

        // No suffix to fold: the payloads already live over only the active variables.
        if rest_len == 0 {
            return NextPartials::new(d_done.clone(), d_carry.clone(), d_omega.clone());
        }

        // Prefix layout puts the folded suffix in the low bits of the index.
        let (_p_active, p_rest) = self.z_svo.split_at(active_len);
        // Equality table over the suffix used to contract it away.
        let rest_eq = SplitEq::<EF, EF>::new_packed(&p_rest, EF::ONE);

        // Done state: contract the shifted-done payload over the suffix.
        let mut done = rest_eq.compress_suffix(d_done);
        // Carry crossing a row boundary lands one suffix step over, so contract it shifted.
        let carry_done = rest_eq.compress_suffix_shifted(d_carry);

        // Fold the boundary-crossing carry contribution into the done state.
        done.as_mut_slice()
            .iter_mut()
            .zip_eq(carry_done.as_slice().iter())
            .for_each(|(out, &carry_done)| *out += carry_done);

        // Number of Boolean rows over the active prefix variables.
        let active_rows = 1 << active_len;
        // Number of Boolean rows over the folded suffix variables.
        let rest_rows = 1 << rest_len;
        // The all-ones suffix corner weight, where carry and omega boundaries live.
        let boundary = rest_eq.last_scalar();
        // Carry state over the active prefix, one entry per active row.
        let mut carry = EF::zero_vec(active_rows);
        // Omega (wrap-around) state over the active prefix, one entry per active row.
        let mut omega = EF::zero_vec(active_rows);

        // Walk the active rows, each a contiguous suffix chunk of the payloads.
        carry
            .iter_mut()
            .zip_eq(omega.iter_mut())
            .zip_eq(d_carry.as_slice().chunks(rest_rows))
            .zip_eq(d_omega.as_slice().chunks(rest_rows))
            .for_each(|(((carry, omega), carry_chunk), omega_chunk)| {
                // Carry enters at the all-zeros suffix corner (first chunk entry).
                *carry = boundary * carry_chunk.first().copied().unwrap();
                // Omega exits at the all-ones suffix corner (last chunk entry).
                *omega = boundary * omega_chunk.last().copied().unwrap();
            });

        // Wrap the contracted active tables as polynomials.
        let carry = Poly::new(carry);
        let omega = Poly::new(omega);

        NextPartials::new(done, carry, omega)
    }
}

#[cfg(test)]
mod test {
    use alloc::vec::Vec;

    use p3_field::extension::BinomialExtensionField;
    use p3_field::{
        ExtensionField, Field, PackedFieldExtension, PackedValue, PrimeCharacteristicRing,
    };
    use p3_koala_bear::KoalaBear;
    use p3_multilinear_util::point::Point;
    use p3_multilinear_util::poly::Poly;
    use p3_util::log2_strict_usize;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::*;
    use crate::strategy::VariableOrder;
    use crate::svo::evals_01inf_grid_prefix;

    type F = KoalaBear;
    type EF = BinomialExtensionField<F, 4>;
    type PackedEF = <EF as ExtensionField<F>>::ExtensionPacking;

    #[test]
    fn test_svo_point_eval() {
        let assert_eval = |svo_point: &SvoPoint<F, EF>, poly: &Poly<F>, point: &Point<EF>| {
            let e0 = poly.eval_base(point);

            let (e1, partial_evals) = svo_point.eval(poly);
            assert_eq!(e0, e1);
            assert_eq!(partial_evals.rounds().len(), svo_point.num_variables_svo());

            match svo_point.var_order() {
                VariableOrder::Prefix => {
                    partial_evals
                        .rounds()
                        .iter()
                        .enumerate()
                        .for_each(|(i, pe0)| {
                            let (_point_lo, point_hi) = point.split_at(i + 1);
                            assert_eq!(pe0.poly(), &poly.compress_suffix(&point_hi, EF::ONE));
                            assert_eq!(
                                e0,
                                pe0.poly().eval_base(&svo_point.z_svo().split_at(i + 1).0)
                            );
                        });
                }
                VariableOrder::Suffix => {
                    partial_evals
                        .rounds()
                        .iter()
                        .enumerate()
                        .for_each(|(i, pe0)| {
                            let (point_lo, point_hi) =
                                point.split_at(point.num_variables() - i - 1);
                            assert_eq!(pe0.poly(), &poly.compress_prefix(&point_lo, EF::ONE));
                            assert_eq!(e0, pe0.poly().eval_base(&point_hi));
                        });
                }
            }
        };

        let k = 12;
        let mut rng = SmallRng::seed_from_u64(11);
        let poly = Poly::<F>::rand(&mut rng, k);
        let point = Point::<EF>::rand(&mut rng, k);

        for l0 in 0..=k {
            let unpacked_prefix =
                SvoPoint::<F, EF>::new_unpacked(l0, &point, VariableOrder::Prefix);
            assert_eval(&unpacked_prefix, &poly, &point);
        }

        for l0 in 0..=k {
            let unpacked_suffix =
                SvoPoint::<F, EF>::new_unpacked(l0, &point, VariableOrder::Suffix);
            assert_eval(&unpacked_suffix, &poly, &point);
        }

        for l0 in 0..=k {
            let packed_prefix = SvoPoint::<F, EF>::new_packed(l0, &point);
            assert_eval(&packed_prefix, &poly, &point);
        }
    }

    #[test]
    fn test_svo_point_accumulate() {
        let mut rng = SmallRng::seed_from_u64(0);

        let assert_accumulate_unpacked =
            |svo_point: &SvoPoint<F, EF>, point: &Point<EF>, scale: EF, rs: &Point<EF>| {
                let eq = Poly::new_from_point(point.as_slice(), EF::ONE);
                let expected = match svo_point.var_order() {
                    VariableOrder::Prefix => eq.compress_prefix(rs, scale),
                    VariableOrder::Suffix => eq.compress_suffix(rs, scale),
                };

                let mut out = Poly::<EF>::zero(expected.num_variables());
                svo_point.accumulate_into(out.as_mut_slice(), rs, scale);
                assert_eq!(out, expected);
            };

        let assert_accumulate_packed =
            |svo_point: &SvoPoint<F, EF>, point: &Point<EF>, scale: EF, rs: &Point<EF>| {
                let eq = Poly::new_from_point(point.as_slice(), EF::ONE);
                let expected = eq.compress_prefix(rs, scale);
                let k_pack = log2_strict_usize(<<F as Field>::Packing as PackedValue>::WIDTH);
                assert!(expected.num_variables() >= k_pack);

                let mut out = Poly::<PackedEF>::zero(expected.num_variables() - k_pack);
                svo_point.accumulate_into_packed(out.as_mut_slice(), rs, scale);
                let unpacked =
                    <PackedEF as PackedFieldExtension<F, EF>>::to_ext_iter(out.iter().copied())
                        .take(expected.num_evals())
                        .collect::<Vec<_>>();
                assert_eq!(unpacked, expected.as_slice());
            };

        let k = 12;
        let k_pack = log2_strict_usize(<<F as Field>::Packing as PackedValue>::WIDTH);
        let point = Point::<EF>::rand(&mut rng, k);
        let scale: EF = rng.random();

        for l0 in 0..=k {
            let unpacked_prefix =
                SvoPoint::<F, EF>::new_unpacked(l0, &point, VariableOrder::Prefix);
            assert_eq!(unpacked_prefix.var_order(), VariableOrder::Prefix);
            assert_eq!(unpacked_prefix.num_variables(), k);
            assert_eq!(unpacked_prefix.num_variables_svo(), l0);
            assert_accumulate_unpacked(&unpacked_prefix, &point, scale, &Point::rand(&mut rng, l0));
        }

        for l0 in 0..=k {
            let unpacked_suffix =
                SvoPoint::<F, EF>::new_unpacked(l0, &point, VariableOrder::Suffix);
            assert_eq!(unpacked_suffix.var_order(), VariableOrder::Suffix);
            assert_eq!(unpacked_suffix.num_variables(), k);
            assert_eq!(unpacked_suffix.num_variables_svo(), l0);
            assert_accumulate_unpacked(&unpacked_suffix, &point, scale, &Point::rand(&mut rng, l0));
        }

        for l0 in 0..=k {
            if k - l0 >= k_pack {
                let packed_prefix = SvoPoint::<F, EF>::new_packed(l0, &point);
                assert_eq!(packed_prefix.var_order(), VariableOrder::Prefix);
                assert_eq!(packed_prefix.num_variables(), k);
                assert_eq!(packed_prefix.num_variables_svo(), l0);
                assert_accumulate_packed(&packed_prefix, &point, scale, &Point::rand(&mut rng, l0));
            }
        }
    }

    #[test]
    fn test_svo_point_eval_next_suffix() {
        // Invariant: the suffix opening value matches the direct successor evaluation,
        // and every cached round table re-evaluates to that same value.
        let assert_eval = |svo_point: &SvoPoint<F, EF>, poly: &Poly<F>, point: &Point<EF>| {
            assert!(matches!(svo_point.var_order(), VariableOrder::Suffix));

            // Ground truth: the full successor evaluation over all variables.
            let expected = poly.eval_next_base(point);
            // Fast path: the SVO opening value plus per-round cached tables.
            let (actual, partials) = svo_point.eval_next_suffix(poly, None);
            assert_eq!(actual, expected);
            assert_eq!(partials.rounds().len(), svo_point.num_variables_svo());

            // Each cached round must independently reproduce the opening value.
            for (round_idx, round) in partials.rounds().iter().enumerate() {
                let active_len = round_idx + 1;
                // Suffix layout: the active variables are the trailing SVO coordinates.
                let (_svo_rest, svo_active) = svo_point
                    .z_svo()
                    .split_at(svo_point.num_variables_svo() - active_len);

                assert_eq!(round.done().num_variables(), active_len);
                assert_eq!(round.carry().num_variables(), active_len);
                assert_eq!(round.omega().num_variables(), active_len);

                // Re-evaluate the three-state successor weight against the cached tables.
                let round_eval = (0..1 << active_len)
                    .map(|row_idx| {
                        let row = Point::hypercube(row_idx, active_len);
                        let (carry, done, omega) =
                            Point::eval_next(svo_active.as_slice(), row.as_slice());
                        done * round.done().as_slice()[row_idx]
                            + carry * round.carry().as_slice()[row_idx]
                            + omega * round.omega().as_slice()[row_idx]
                    })
                    .sum::<EF>();

                assert_eq!(round_eval, expected);
            }
        };

        // Fixture state: 12-variable random witness and opening point.
        let k = 12;
        let mut rng = SmallRng::seed_from_u64(12);
        let poly = Poly::<F>::rand(&mut rng, k);
        let point = Point::<EF>::rand(&mut rng, k);

        // Sweep every SVO depth from no SVO variables up to all 12.
        for l0 in 0..=k {
            let svo_point = SvoPoint::<F, EF>::new_unpacked(l0, &point, VariableOrder::Suffix);
            assert_eval(&svo_point, &poly, &point);
        }
    }

    #[test]
    fn test_svo_point_eval_next_prefix() {
        // Invariant: the prefix opening value matches the direct successor evaluation,
        // and every cached round table re-evaluates to that same value.
        let assert_eval = |svo_point: &SvoPoint<F, EF>, poly: &Poly<F>, point: &Point<EF>| {
            assert!(matches!(svo_point.var_order(), VariableOrder::Prefix));

            // Ground truth: the full successor evaluation over all variables.
            let expected = poly.eval_next_base(point);
            // Fast path: the SVO opening value plus per-round cached tables.
            let (actual, partials) = svo_point.eval_next_prefix(poly);
            assert_eq!(actual, expected);
            assert_eq!(partials.rounds().len(), svo_point.num_variables_svo());

            // Each cached round must independently reproduce the opening value.
            for (round_idx, round) in partials.rounds().iter().enumerate() {
                let active_len = round_idx + 1;
                // Prefix layout: the active variables are the leading SVO coordinates.
                let (svo_active, _) = svo_point.z_svo().split_at(active_len);

                assert_eq!(round.done().num_variables(), active_len);
                assert_eq!(round.carry().num_variables(), active_len);
                assert_eq!(round.omega().num_variables(), active_len);

                // Prefix re-evaluation pairs the equality weight with the done payload, plus the two boundary states.
                let round_eval = (0..1 << active_len)
                    .map(|row_idx| {
                        let row = Point::hypercube(row_idx, active_len);
                        let (_carry, done, omega) =
                            Point::eval_next(svo_active.as_slice(), row.as_slice());
                        let eq = Point::eval_eq(svo_active.as_slice(), row.as_slice());
                        eq * round.done().as_slice()[row_idx]
                            + done * round.carry().as_slice()[row_idx]
                            + omega * round.omega().as_slice()[row_idx]
                    })
                    .sum::<EF>();

                assert_eq!(round_eval, expected);
            }
        };

        // Fixture state: 12-variable random witness and opening point.
        let k = 12;
        let mut rng = SmallRng::seed_from_u64(13);
        let poly = Poly::<F>::rand(&mut rng, k);
        let point = Point::<EF>::rand(&mut rng, k);

        // Sweep every SVO depth using the unpacked split-equality construction.
        for l0 in 0..=k {
            let svo_point = SvoPoint::<F, EF>::new_unpacked(l0, &point, VariableOrder::Prefix);
            assert_eval(&svo_point, &poly, &point);
        }

        // Repeat with the packed split-equality construction to cover both code paths.
        for l0 in 0..=k {
            let svo_point = SvoPoint::<F, EF>::new_packed(l0, &point);
            assert_eval(&svo_point, &poly, &point);
        }
    }

    #[test]
    fn test_svo_point_accumulate_next_prefix() {
        let mut rng = SmallRng::seed_from_u64(14);
        // Fixture state: 12-variable random opening point and a random batching coefficient.
        let k = 12;
        let point = Point::<EF>::rand(&mut rng, k);
        let scale: EF = rng.random();
        // Dense successor weight table over all variables, used as the reference.
        let next = Poly::new_next_from_point(point.as_slice());

        // Invariant: the residual accumulator equals the dense weight compressed at the round challenges.
        for l0 in 0..=k {
            let svo_point = SvoPoint::<F, EF>::new_packed(l0, &point);
            // Random SVO round challenges fixing the l0 SVO variables.
            let rs = Point::rand(&mut rng, l0);
            // Reference: compress the dense weight over the prefix challenges, scaled.
            let expected = next.compress_prefix(&rs, scale);

            // Fast path: accumulate the residual split weight in place.
            let mut out = Poly::<EF>::zero(expected.num_variables());
            svo_point.accumulate_next_prefix_into(out.as_mut_slice(), &rs, scale);
            assert_eq!(out, expected);
        }
    }

    #[test]
    fn test_svo_point_accumulate_next_suffix() {
        let mut rng = SmallRng::seed_from_u64(14);
        // Fixture state: 12-variable random opening point and a random batching coefficient.
        let k = 12;
        let point = Point::<EF>::rand(&mut rng, k);
        let scale: EF = rng.random();
        // Dense successor weight table over all variables, used as the reference.
        let next = Poly::new_next_from_point(point.as_slice());

        // Invariant: the residual accumulator equals the dense weight compressed at the round challenges.
        for l0 in 0..=k {
            // Suffix layout: the l0 SVO variables are the trailing point coordinates.
            let svo_point = SvoPoint::<F, EF>::new_unpacked(l0, &point, VariableOrder::Suffix);
            // Random SVO round challenges fixing the l0 SVO variables.
            let rs = Point::rand(&mut rng, l0);
            // Reference: compress the dense weight over the suffix challenges, scaled.
            let expected = next.compress_suffix(&rs, scale);

            // Fast path: accumulate the residual split weight in place.
            let mut out = Poly::<EF>::zero(expected.num_variables());
            svo_point.accumulate_next_suffix_into(out.as_mut_slice(), &rs, scale);
            assert_eq!(out, expected);
        }
    }

    // Brute-force reference: compress the polynomial over the split prefix into the three SVO payloads.
    fn split_compressions_for_next(
        poly: &[EF],
        point: &[EF],
        split_len: usize,
        svo_len: usize,
    ) -> [Vec<EF>; 3] {
        assert_eq!(point.len(), split_len + svo_len);
        assert_eq!(poly.len(), 1 << point.len());

        // Suffix layout: the split prefix occupies the leading point coordinates.
        let (p_split, _p_svo) = point.split_at(split_len);
        let split_rows = 1 << split_len;
        let svo_rows = 1 << svo_len;

        // Per split-row weights: equality, carry-into-next (done), and boundary (omega).
        let mut eq_split = EF::zero_vec(split_rows);
        let mut t_split = EF::zero_vec(split_rows);
        let mut omega_split = EF::zero_vec(split_rows);

        // Fill the split weights from the closed form at each split row.
        for split_idx in 0..split_rows {
            let row = Point::hypercube(split_idx, split_len);
            let (_carry, done, omega) = Point::eval_next(p_split, row.as_slice());
            eq_split[split_idx] = Point::eval_eq(p_split, row.as_slice());
            t_split[split_idx] = done;
            omega_split[split_idx] = omega;
        }

        // Resulting SVO payloads, one entry per SVO row.
        let mut d_eq = EF::zero_vec(svo_rows);
        let mut d_t = EF::zero_vec(svo_rows);
        let mut d_omega = EF::zero_vec(svo_rows);

        // Contract the split dimension: the polynomial index splits as (split_idx << svo_len) | svo_idx.
        for split_idx in 0..split_rows {
            let base = split_idx << svo_len;
            for svo_idx in 0..svo_rows {
                let value = poly[base | svo_idx];
                // Weight each polynomial value by the matching split state and sum into the SVO payload.
                d_eq[svo_idx] += value * eq_split[split_idx];
                d_t[svo_idx] += value * t_split[split_idx];
                d_omega[svo_idx] += value * omega_split[split_idx];
            }
        }

        [d_eq, d_t, d_omega]
    }

    // Brute-force reference: sum the full ternary-grid accumulator over every fixed rest assignment.
    fn dense_next_accumulator_round(
        d_eq: &[EF],
        d_t: &[EF],
        d_omega: &[EF],
        p_svo: &[EF],
        active_len: usize,
    ) -> [Vec<EF>; 2] {
        let svo_len = p_svo.len();
        assert!(active_len > 0);
        assert!(active_len <= svo_len);
        // The rest variables are the already-folded SVO prefix summed over below.
        let rest_len = svo_len - active_len;
        let rest_rows = 1 << rest_len;
        // Full ternary grid over the active variables: 3^active_len entries.
        let grid_len = 3usize.pow(active_len as u32);
        let mut grid = EF::zero_vec(grid_len);

        // Accumulate one grid contribution per Boolean assignment of the rest variables.
        for rest_idx in 0..rest_rows {
            let rest_row = Point::hypercube(rest_idx, rest_len);
            let active_rows = 1 << active_len;
            // This rest assignment owns a contiguous active block of the SVO payloads.
            let row_start = rest_idx << active_len;
            let row_range = row_start..row_start + active_rows;

            // Expand each payload's active block to the ternary grid.
            let d_eq_grid = evals_01inf_grid_prefix(&d_eq[row_range.clone()]);
            let d_t_grid = evals_01inf_grid_prefix(&d_t[row_range.clone()]);
            let d_omega_grid = evals_01inf_grid_prefix(&d_omega[row_range]);

            // Successor states of the full SVO point with the rest prefix fixed.
            let mut carry = EF::zero_vec(active_rows);
            let mut done = EF::zero_vec(active_rows);
            let mut omega = EF::zero_vec(active_rows);
            for active_idx in 0..active_rows {
                let active_row = Point::hypercube(active_idx, active_len);
                // Concatenate the fixed rest prefix with this active row to form a full SVO row.
                let mut row = Vec::with_capacity(svo_len);
                row.extend_from_slice(rest_row.as_slice());
                row.extend_from_slice(active_row.as_slice());
                let (c, d, o) = Point::eval_next(p_svo, &row);
                carry[active_idx] = c;
                done[active_idx] = d;
                omega[active_idx] = o;
            }

            // Expand the state tables to the same ternary grid.
            let carry_grid = evals_01inf_grid_prefix(&carry);
            let done_grid = evals_01inf_grid_prefix(&done);
            let omega_grid = evals_01inf_grid_prefix(&omega);

            // Add the three state-times-data products pointwise across the grid.
            for idx in 0..grid_len {
                grid[idx] += done_grid[idx] * d_eq_grid[idx]
                    + carry_grid[idx] * d_t_grid[idx]
                    + omega_grid[idx] * d_omega_grid[idx];
            }
        }

        // Keep only the 0 third and the inf third, matching the production accumulator pair.
        let stride = 3usize.pow((active_len - 1) as u32);
        [grid[..stride].to_vec(), grid[2 * stride..].to_vec()]
    }

    #[test]
    fn test_next_svo_accumulators_match_dense_ternary_reference() {
        let mut rng = SmallRng::seed_from_u64(4);

        // Invariant: the fast compress-then-accumulate path equals the brute-force ternary grid.
        // Fixture state: split lengths 0..=4, SVO lengths 1..=5, 20 random instances each.
        for split_len in 0..=4 {
            for svo_len in 1..=5 {
                let total_len = split_len + svo_len;

                for _ in 0..20 {
                    // Random witness polynomial over all variables.
                    let poly = (0..1 << total_len)
                        .map(|_| rng.random::<EF>())
                        .collect::<Vec<_>>();
                    // Random opening point over all variables.
                    let point = (0..total_len)
                        .map(|_| rng.random::<EF>())
                        .collect::<Vec<_>>();
                    // Suffix layout: SVO variables are the trailing point coordinates.
                    let (_p_split, p_svo) = point.split_at(split_len);
                    let p_svo = Point::new(p_svo.to_vec());
                    // An SVO point whose SVO suffix equals `p_svo` (folding = svo_len).
                    let svo_point = SvoPoint::<F, EF>::new_unpacked(
                        svo_len,
                        &Point::new(point.clone()),
                        VariableOrder::Suffix,
                    );
                    // Reference compression of the polynomial over the split prefix.
                    let [d_eq, d_t, d_omega] =
                        split_compressions_for_next(&poly, &point, split_len, svo_len);

                    // Compare both paths for every SVO sumcheck round.
                    for active_len in 1..=svo_len {
                        // Fast path: compress payloads to the active variables for this round.
                        let round = svo_point.next_round_partials_suffix(
                            &Poly::new(d_eq.clone()),
                            &Poly::new(d_t.clone()),
                            &Poly::new(d_omega.clone()),
                            active_len,
                        );
                        // Active suffix coordinates kept for this round.
                        let (_svo_rest, svo_active) = p_svo.split_at(svo_len - active_len);
                        let stride = 3usize.pow((active_len - 1) as u32);
                        // Production accumulators at 0 and inf.
                        let mut production0 = EF::zero_vec(stride);
                        let mut production_inf = EF::zero_vec(stride);
                        round.accumulate_suffix(
                            svo_active.as_slice(),
                            &mut production0,
                            &mut production_inf,
                        );
                        let production = [production0, production_inf];
                        // Slow path: full ternary grid summed over the rest variables.
                        let dense = dense_next_accumulator_round(
                            &d_eq,
                            &d_t,
                            &d_omega,
                            p_svo.as_slice(),
                            active_len,
                        );

                        assert_eq!(
                            production, dense,
                            "split_len={split_len}, svo_len={svo_len}, active_len={active_len}"
                        );
                    }
                }
            }
        }
    }
}
