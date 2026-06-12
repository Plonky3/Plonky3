use alloc::vec::Vec;

use itertools::Itertools;
use p3_field::{ExtensionField, Field};
use p3_maybe_rayon::prelude::*;
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;
use p3_multilinear_util::split_eq::SplitEq;
use p3_util::log2_strict_usize;

use crate::layout::{NextPartials, NextSvoPartials};
use crate::strategy::VariableOrder;
use crate::svo::{SvoPoint, calculate_product_accumulator, evals_01inf_grid_prefix};

const PARALLEL_THRESHOLD: usize = 4096;

/// Compresses suffix-layout Next payloads down to the active SVO variables.
///
/// `d_eq`, `d_t`, and `d_omega` are the split-compressed payloads produced by
/// `eval_next_suffix`, still indexed by all SVO variables. This fixes the
/// already-folded SVO prefix (`rest`) at `p_svo` and leaves a `NextPartials`
/// table over the `active_len` suffix variables for one sumcheck round.
fn next_round_partials_suffix<F: Field>(
    d_eq: &Poly<F>,
    d_t: &Poly<F>,
    d_omega: &Poly<F>,
    p_svo: &Point<F>,
    active_len: usize,
) -> NextPartials<F> {
    let svo_len = p_svo.num_variables();
    assert!(active_len > 0);
    assert!(active_len <= svo_len);
    assert_eq!(d_eq.num_variables(), svo_len);
    assert_eq!(d_t.num_variables(), svo_len);
    assert_eq!(d_omega.num_variables(), svo_len);

    let rest_len = svo_len - active_len;
    let active_rows = 1 << active_len;

    if rest_len == 0 {
        return NextPartials::new(d_eq.clone(), d_t.clone(), d_omega.clone());
    }

    let (p_rest, _p_active) = p_svo.split_at(rest_len);
    let rest_eq = SplitEq::<F, F>::new_packed(&p_rest, F::ONE);

    let done = rest_eq.compress_prefix(d_eq);
    let mut carry = rest_eq.compress_prefix_shifted(d_eq);

    let carry_scale = p_rest.iter().copied().product::<F>();
    carry
        .as_mut_slice()
        .iter_mut()
        .zip_eq(d_t.as_slice()[..active_rows].iter())
        .for_each(|(out, &value)| *out += carry_scale * value);

    let rest_rows = 1 << rest_len;
    let omega_scale = carry_scale;
    let omega_start = (rest_rows - 1) * active_rows;
    let omega = Poly::new(
        d_omega.as_slice()[omega_start..omega_start + active_rows]
            .iter()
            .map(|&value| omega_scale * value)
            .collect(),
    );

    NextPartials::new(done, carry, omega)
}

/// Compresses prefix-layout Next payloads down to the active SVO variables.
///
/// `d_done`, `d_carry`, and `d_omega` are indexed by all SVO variables after
/// the split side has already been compressed. This fixes the already-folded
/// SVO suffix (`rest`) at `p_svo` and returns the three active-variable
/// payloads needed by the current sumcheck round.
fn next_round_partials_prefix<F: Field>(
    d_done: &Poly<F>,
    d_carry: &Poly<F>,
    d_omega: &Poly<F>,
    p_svo: &Point<F>,
    active_len: usize,
) -> NextPartials<F> {
    let svo_len = p_svo.num_variables();
    assert!(active_len > 0);
    assert!(active_len <= svo_len);
    assert_eq!(d_done.num_variables(), svo_len);
    assert_eq!(d_carry.num_variables(), svo_len);
    assert_eq!(d_omega.num_variables(), svo_len);

    let rest_len = svo_len - active_len;

    if rest_len == 0 {
        return NextPartials::new(d_done.clone(), d_carry.clone(), d_omega.clone());
    }

    let (_p_active, p_rest) = p_svo.split_at(active_len);
    let rest_eq = SplitEq::<F, F>::new_packed(&p_rest, F::ONE);

    let mut done = rest_eq.compress_suffix(d_done);
    let carry_done = rest_eq.compress_suffix_shifted(d_carry);

    done.as_mut_slice()
        .iter_mut()
        .zip_eq(carry_done.as_slice().iter())
        .for_each(|(out, &carry_done)| *out += carry_done);

    let active_rows = 1 << active_len;
    let rest_rows = 1 << rest_len;
    let boundary = rest_eq.last_scalar();
    let mut carry = F::zero_vec(active_rows);
    let mut omega = F::zero_vec(active_rows);

    carry
        .iter_mut()
        .zip_eq(omega.iter_mut())
        .zip_eq(d_carry.as_slice().chunks(rest_rows))
        .zip_eq(d_omega.as_slice().chunks(rest_rows))
        .for_each(|(((carry, omega), carry_chunk), omega_chunk)| {
            *carry = boundary * carry_chunk.first().copied().unwrap();
            *omega = boundary * omega_chunk.last().copied().unwrap();
        });

    let carry = Poly::new(carry);
    let omega = Poly::new(omega);

    NextPartials::new(done, carry, omega)
}

/// Materializes the repeat-last Next carry-state tables for a point.
///
/// The returned `done`, `carry`, and `omega` tables are indexed by Boolean rows
/// of the same length as `point_suffix`. Their dot product with matching data
/// payloads evaluates the repeat-last Next decomposition without iterating the
/// closed form row by row.
fn next_state_evals<F: Field>(point_suffix: &[F]) -> NextPartials<F> {
    let num_variables = point_suffix.len();
    let num_rows = 1 << num_variables;

    let boundary = point_suffix.iter().copied().product::<F>();
    let mut carry = F::zero_vec(num_rows);
    let mut omega = F::zero_vec(num_rows);
    carry[0] = boundary;
    omega[num_rows - 1] = boundary;

    let eq = Poly::new_from_point(point_suffix, F::ONE);
    let mut done = F::zero_vec(num_rows);
    if num_rows > 1 {
        done[1..].copy_from_slice(&eq.as_slice()[..num_rows - 1]);
    }

    NextPartials::new(Poly::new(done), Poly::new(carry), Poly::new(omega))
}

impl<F: Field> NextPartials<F> {
    /// Adds suffix-layout SVO accumulator contributions for one round.
    ///
    /// `self` holds the active-variable data payloads for this round. This
    /// expands both the Next state tables at `p_active` and the payloads to the
    /// `{0, 1, inf}` grid, then accumulates the round evaluations at `0` and
    /// `inf` into `acc0` and `acc_inf`.
    pub(crate) fn accumulate_suffix(&self, p_active: &[F], acc0: &mut [F], acc_inf: &mut [F]) {
        let active_len = p_active.len();
        assert!(active_len > 0);
        assert_eq!(self.done().num_variables(), active_len);
        assert_eq!(self.carry().num_variables(), active_len);
        assert_eq!(self.omega().num_variables(), active_len);

        let stride = 3usize.pow((active_len - 1) as u32);
        assert_eq!(acc0.len(), stride);
        assert_eq!(acc_inf.len(), stride);

        // TODO: carry and omega polys are sparse.
        let active = next_state_evals(p_active);

        let carry_grid = evals_01inf_grid_prefix(active.carry().as_slice());
        let done_grid = evals_01inf_grid_prefix(active.done().as_slice());
        let omega_grid = evals_01inf_grid_prefix(active.omega().as_slice());
        let done_data_grid = evals_01inf_grid_prefix(self.done().as_slice());
        let carry_data_grid = evals_01inf_grid_prefix(self.carry().as_slice());
        let omega_data_grid = evals_01inf_grid_prefix(self.omega().as_slice());

        acc0.iter_mut()
            .zip(
                done_grid[..stride]
                    .iter()
                    .zip(done_data_grid[..stride].iter()),
            )
            .zip(
                carry_grid[..stride]
                    .iter()
                    .zip(carry_data_grid[..stride].iter()),
            )
            .zip(
                omega_grid[..stride]
                    .iter()
                    .zip(omega_data_grid[..stride].iter()),
            )
            .for_each(
                |(((out, (&done, &done_data)), (&carry, &carry_data)), (&omega, &omega_data))| {
                    *out += done * done_data + carry * carry_data + omega * omega_data;
                },
            );

        acc_inf
            .iter_mut()
            .zip(
                done_grid[2 * stride..]
                    .iter()
                    .zip(done_data_grid[2 * stride..].iter()),
            )
            .zip(
                carry_grid[2 * stride..]
                    .iter()
                    .zip(carry_data_grid[2 * stride..].iter()),
            )
            .zip(
                omega_grid[2 * stride..]
                    .iter()
                    .zip(omega_data_grid[2 * stride..].iter()),
            )
            .for_each(
                |(((out, (&done, &done_data)), (&carry, &carry_data)), (&omega, &omega_data))| {
                    *out += done * done_data + carry * carry_data + omega * omega_data;
                },
            );
    }

    /// Adds prefix-layout SVO accumulator contributions for one round.
    ///
    /// `self` holds the active-variable data payloads for this round. Prefix
    /// layout separates the active state as `Eq`, `done`, and `omega` products,
    /// so each term can use the usual product accumulator and add its `0` and
    /// `inf` contributions to the running round accumulators.
    pub(crate) fn accumulate_prefix(&self, p_active: &[F], acc0: &mut [F], acc_inf: &mut [F]) {
        let active_len = p_active.len();
        assert!(active_len > 0);
        assert_eq!(self.done().num_variables(), active_len);
        assert_eq!(self.carry().num_variables(), active_len);
        assert_eq!(self.omega().num_variables(), active_len);

        let stride = 3usize.pow((active_len - 1) as u32);
        assert_eq!(acc0.len(), stride);
        assert_eq!(acc_inf.len(), stride);

        let active = next_state_evals(p_active);
        let eq_active = Poly::new_from_point(p_active, F::ONE);

        let terms = [
            calculate_product_accumulator(active_len, eq_active.as_slice(), self.done().as_slice()),
            calculate_product_accumulator(
                active_len,
                active.done().as_slice(),
                self.carry().as_slice(),
            ),
            calculate_product_accumulator(
                active_len,
                active.omega().as_slice(),
                self.omega().as_slice(),
            ),
        ];

        for [term0, term_inf] in terms {
            acc0.iter_mut()
                .zip(term0.iter())
                .for_each(|(out, &value)| *out += value);
            acc_inf
                .iter_mut()
                .zip(term_inf.iter())
                .for_each(|(out, &value)| *out += value);
        }
    }
}

impl<F: Field, EF: ExtensionField<F>> SvoPoint<F, EF> {
    /// Accumulates residual suffix-layout Next weights over the split variables.
    ///
    /// After the SVO variables are fixed to `rs`, the remaining weight over the
    /// split variables is:
    /// `done_svo * Eq(split) + carry_svo * T(split) + omega_svo * Omega(split)`.
    /// This adds that residual table into `out`, scaled by `scale`, without
    /// materializing any dense Next table.
    pub fn accumulate_next_suffix_into(&self, out: &mut [EF], rs: &Point<EF>, scale: EF) {
        assert!(
            matches!(self.var_order(), VariableOrder::Suffix),
            "next residual weights are implemented for suffix SVO only"
        );
        assert_eq!(rs.num_variables(), self.num_variables_svo());
        assert_eq!(log2_strict_usize(out.len()), self.z_split.num_variables());

        let (carry, done, omega) = Point::eval_next(self.z_svo.as_slice(), rs.as_slice());
        let done_scale = scale * done;
        let carry_scale = scale * carry;
        let omega_scale = scale * omega;

        #[cfg(debug_assertions)]
        let expected = {
            let mut expected = out.to_vec();
            let eq = self.z_split.materialize();
            expected
                .iter_mut()
                .zip_eq(eq.iter())
                .for_each(|(out, &weight)| *out += done_scale * weight);
            expected
                .iter_mut()
                .skip(1)
                .zip_eq(eq.as_slice()[..eq.num_evals() - 1].iter())
                .for_each(|(out, &weight)| *out += carry_scale * weight);
            *expected.last_mut().unwrap() += omega_scale * self.z_split.last_scalar();
            expected
        };

        let eq1 = self.z_split.eq1();
        let cs = eq1.scalar_chunk_size();
        let eq1_last = eq1.last_scalar();
        if out.len() < PARALLEL_THRESHOLD {
            let mut prev_last = EF::ZERO;
            out.chunks_mut(cs)
                .zip(self.z_split.eq0().iter())
                .for_each(|(chunk, &w0)| {
                    eq1.accumulate_next_chunk_into(
                        chunk,
                        done_scale * w0,
                        carry_scale * w0,
                        carry_scale * prev_last,
                    );
                    prev_last = w0 * eq1_last;
                });
        } else {
            let eq0 = self.z_split.eq0().as_slice();
            out.par_chunks_mut(cs)
                .enumerate()
                .zip(eq0.par_iter())
                .for_each(|((idx, chunk), &w0)| {
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

        *out.last_mut().unwrap() += omega_scale * self.z_split.last_scalar();

        #[cfg(debug_assertions)]
        debug_assert!(out == expected.as_slice());
    }

    /// Accumulates residual prefix-layout Next weights over the split variables.
    ///
    /// After the prefix SVO variables are fixed to `rs`, the remaining suffix
    /// split weight is:
    /// `eq_svo * T(split) + done_svo * Carry(split) + omega_svo * Omega(split)`.
    /// This adds that residual table into `out`, scaled by `scale`, using the
    /// shifted-eq helper plus the two repeat-last boundary rows.
    pub fn accumulate_next_prefix_into(&self, out: &mut [EF], rs: &Point<EF>, scale: EF) {
        assert!(
            matches!(self.var_order(), VariableOrder::Prefix),
            "prefix next residual weights require prefix SVO"
        );
        assert_eq!(rs.num_variables(), self.num_variables_svo());
        assert_eq!(log2_strict_usize(out.len()), self.z_split.num_variables());

        let (_carry, done, omega) = Point::eval_next(self.z_svo.as_slice(), rs.as_slice());
        let eq = Point::eval_eq(self.z_svo.as_slice(), rs.as_slice());
        let done_scale = scale * eq;
        let carry_scale = scale * done;
        let omega_scale = scale * omega;

        #[cfg(debug_assertions)]
        let expected = {
            let mut expected = out.to_vec();
            let eq = self.z_split.materialize();
            expected
                .iter_mut()
                .skip(1)
                .zip_eq(eq.as_slice()[..eq.num_evals() - 1].iter())
                .for_each(|(out, &weight)| *out += done_scale * weight);
            let boundary = self.z_split.last_scalar();
            *expected.first_mut().unwrap() += carry_scale * boundary;
            *expected.last_mut().unwrap() += omega_scale * boundary;
            expected
        };

        let eq1 = self.z_split.eq1();
        let cs = eq1.scalar_chunk_size();
        let eq1_last = eq1.last_scalar();
        if out.len() < PARALLEL_THRESHOLD {
            let mut prev_last = EF::ZERO;
            out.chunks_mut(cs)
                .zip(self.z_split.eq0().iter())
                .for_each(|(chunk, &w0)| {
                    eq1.accumulate_next_chunk_into(
                        chunk,
                        EF::ZERO,
                        done_scale * w0,
                        done_scale * prev_last,
                    );
                    prev_last = w0 * eq1_last;
                });
        } else {
            let eq0 = self.z_split.eq0().as_slice();
            out.par_chunks_mut(cs)
                .enumerate()
                .zip(eq0.par_iter())
                .for_each(|((idx, chunk), &w0)| {
                    let boundary = if idx > 0 {
                        done_scale * eq0[idx - 1] * eq1_last
                    } else {
                        EF::ZERO
                    };
                    eq1.accumulate_next_chunk_into(chunk, EF::ZERO, done_scale * w0, boundary);
                });
        }

        let boundary = self.z_split.last_scalar();
        *out.first_mut().unwrap() += carry_scale * boundary;
        *out.last_mut().unwrap() += omega_scale * boundary;

        #[cfg(debug_assertions)]
        debug_assert!(out == expected.as_slice());
    }

    /// Evaluates a suffix-layout repeat-last Next opening and caches SVO rounds.
    ///
    /// The raw witness polynomial is first compressed over the split prefix into
    /// three payloads over the SVO suffix: `d_eq`, `d_t`, and `d_omega`. Those
    /// payloads give the scalar opening value and are further compressed into
    /// one `NextPartials` entry per SVO sumcheck round.
    pub fn eval_next_suffix(
        &self,
        poly: &Poly<F>,
        d_eq: Option<&Poly<EF>>,
    ) -> (EF, NextSvoPartials<EF>) {
        assert_eq!(self.num_variables(), poly.num_variables());
        assert!(
            matches!(self.var_order(), VariableOrder::Suffix),
            "next openings are implemented for suffix SVO only"
        );

        let d_eq_owned = d_eq.is_none().then(|| self.z_split.compress_prefix(poly));
        let d_eq = d_eq.unwrap_or_else(|| d_eq_owned.as_ref().unwrap());
        assert_eq!(d_eq.num_variables(), self.num_variables_svo());

        #[cfg(debug_assertions)]
        if d_eq_owned.is_none() {
            debug_assert_eq!(*d_eq, self.z_split.compress_prefix(poly));
        }
        let d_t = self.z_split.compress_prefix_shifted(poly);

        let svo_rows = 1 << self.num_variables_svo();
        let split_rows = 1 << self.z_split.num_variables();
        let omega_scale = self.z_split.last_scalar();
        let omega_start = (split_rows - 1) * svo_rows;
        let d_omega = Poly::new(
            poly.as_slice()[omega_start..omega_start + svo_rows]
                .iter()
                .map(|&value| omega_scale * value)
                .collect(),
        );

        let eval = (0..svo_rows)
            .map(|svo_idx| {
                let row = Point::hypercube(svo_idx, self.z_svo.num_variables());
                let (carry, done, omega) = Point::eval_next(self.z_svo.as_slice(), row.as_slice());
                done * d_eq.as_slice()[svo_idx]
                    + carry * d_t.as_slice()[svo_idx]
                    + omega * d_omega.as_slice()[svo_idx]
            })
            .sum();

        let rounds = (1..=self.num_variables_svo())
            .map(|active_len| {
                next_round_partials_suffix(d_eq, &d_t, &d_omega, &self.z_svo, active_len)
            })
            .collect::<Vec<_>>();

        (eval, NextSvoPartials::new(rounds))
    }

    /// Evaluates a prefix-layout repeat-last Next opening and caches SVO rounds.
    ///
    /// The raw witness polynomial is first compressed over the split suffix into
    /// three payloads over the SVO prefix: shifted-done, carry boundary, and
    /// omega boundary. Those payloads give the scalar opening value and are
    /// further compressed into one `NextPartials` entry per SVO sumcheck round.
    pub fn eval_next_prefix(&self, poly: &Poly<F>) -> (EF, NextSvoPartials<EF>) {
        assert_eq!(self.num_variables(), poly.num_variables());
        assert!(
            matches!(self.var_order(), VariableOrder::Prefix),
            "prefix next openings require prefix SVO"
        );

        let d_done = self.z_split.compress_suffix_shifted(poly);

        let svo_rows = 1 << self.num_variables_svo();
        let split_rows = 1 << self.z_split.num_variables();
        let boundary = self.z_split.last_scalar();
        let mut d_carry = EF::zero_vec(svo_rows);
        let mut d_omega = EF::zero_vec(svo_rows);

        d_carry
            .iter_mut()
            .zip_eq(d_omega.iter_mut())
            .zip_eq(poly.as_slice().chunks(split_rows))
            .for_each(|((d_carry, d_omega), chunk)| {
                *d_carry = boundary * chunk.first().copied().unwrap();
                *d_omega = boundary * chunk.last().copied().unwrap();
            });

        let d_carry = Poly::new(d_carry);
        let d_omega = Poly::new(d_omega);

        let eval = (0..svo_rows)
            .map(|svo_idx| {
                let row = Point::hypercube(svo_idx, self.z_svo.num_variables());
                let (_carry, done, omega) = Point::eval_next(self.z_svo.as_slice(), row.as_slice());
                let eq = Point::eval_eq(self.z_svo.as_slice(), row.as_slice());
                eq * d_done.as_slice()[svo_idx]
                    + done * d_carry.as_slice()[svo_idx]
                    + omega * d_omega.as_slice()[svo_idx]
            })
            .sum();

        let rounds = (1..=self.num_variables_svo())
            .map(|active_len| {
                next_round_partials_prefix(&d_done, &d_carry, &d_omega, &self.z_svo, active_len)
            })
            .collect::<Vec<_>>();

        (eval, NextSvoPartials::new(rounds))
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use p3_field::PrimeCharacteristicRing;
    use p3_field::extension::BinomialExtensionField;
    use p3_koala_bear::KoalaBear;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::*;

    type F = KoalaBear;
    type EF = BinomialExtensionField<F, 4>;

    fn next_state_evals_reference(point_suffix: &[F]) -> [Poly<F>; 3] {
        let num_variables = point_suffix.len();
        let num_rows = 1 << num_variables;
        let mut carry = F::zero_vec(num_rows);
        let mut done = F::zero_vec(num_rows);
        let mut omega = F::zero_vec(num_rows);

        for row_idx in 0..num_rows {
            let row = Point::hypercube(row_idx, num_variables);
            let (c, d, o) = Point::eval_next(point_suffix, row.as_slice());
            carry[row_idx] = c;
            done[row_idx] = d;
            omega[row_idx] = o;
        }

        [Poly::new(carry), Poly::new(done), Poly::new(omega)]
    }

    #[test]
    fn test_next_state_evals_matches_reference() {
        let mut rng = SmallRng::seed_from_u64(1);

        for num_variables in 0..=8 {
            for _ in 0..16 {
                let point = Point::<F>::rand(&mut rng, num_variables);
                let actual = next_state_evals(point.as_slice());
                let [carry, done, omega] = next_state_evals_reference(point.as_slice());

                assert_eq!(actual.carry(), &carry);
                assert_eq!(actual.done(), &done);
                assert_eq!(actual.omega(), &omega);
            }
        }
    }

    fn split_compressions_for_next(
        poly: &[EF],
        point: &[EF],
        split_len: usize,
        svo_len: usize,
    ) -> [Vec<EF>; 3] {
        assert_eq!(point.len(), split_len + svo_len);
        assert_eq!(poly.len(), 1 << point.len());

        let (p_split, _p_svo) = point.split_at(split_len);
        let split_rows = 1 << split_len;
        let svo_rows = 1 << svo_len;

        let mut eq_split = EF::zero_vec(split_rows);
        let mut t_split = EF::zero_vec(split_rows);
        let mut omega_split = EF::zero_vec(split_rows);

        for split_idx in 0..split_rows {
            let row = Point::hypercube(split_idx, split_len);
            let (_carry, done, omega) = Point::eval_next(p_split, row.as_slice());
            eq_split[split_idx] = Point::eval_eq(p_split, row.as_slice());
            t_split[split_idx] = done;
            omega_split[split_idx] = omega;
        }

        let mut d_eq = EF::zero_vec(svo_rows);
        let mut d_t = EF::zero_vec(svo_rows);
        let mut d_omega = EF::zero_vec(svo_rows);

        for split_idx in 0..split_rows {
            let base = split_idx << svo_len;
            for svo_idx in 0..svo_rows {
                let value = poly[base | svo_idx];
                d_eq[svo_idx] += value * eq_split[split_idx];
                d_t[svo_idx] += value * t_split[split_idx];
                d_omega[svo_idx] += value * omega_split[split_idx];
            }
        }

        [d_eq, d_t, d_omega]
    }

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
        let rest_len = svo_len - active_len;
        let rest_rows = 1 << rest_len;
        let grid_len = 3usize.pow(active_len as u32);
        let mut grid = EF::zero_vec(grid_len);

        for rest_idx in 0..rest_rows {
            let rest_row = Point::hypercube(rest_idx, rest_len);
            let active_rows = 1 << active_len;
            let row_start = rest_idx << active_len;
            let row_range = row_start..row_start + active_rows;

            let d_eq_grid = evals_01inf_grid_prefix(&d_eq[row_range.clone()]);
            let d_t_grid = evals_01inf_grid_prefix(&d_t[row_range.clone()]);
            let d_omega_grid = evals_01inf_grid_prefix(&d_omega[row_range]);

            let mut carry = EF::zero_vec(active_rows);
            let mut done = EF::zero_vec(active_rows);
            let mut omega = EF::zero_vec(active_rows);
            for active_idx in 0..active_rows {
                let active_row = Point::hypercube(active_idx, active_len);
                let mut row = Vec::with_capacity(svo_len);
                row.extend_from_slice(rest_row.as_slice());
                row.extend_from_slice(active_row.as_slice());
                let (c, d, o) = Point::eval_next(p_svo, &row);
                carry[active_idx] = c;
                done[active_idx] = d;
                omega[active_idx] = o;
            }

            let carry_grid = evals_01inf_grid_prefix(&carry);
            let done_grid = evals_01inf_grid_prefix(&done);
            let omega_grid = evals_01inf_grid_prefix(&omega);

            for idx in 0..grid_len {
                grid[idx] += done_grid[idx] * d_eq_grid[idx]
                    + carry_grid[idx] * d_t_grid[idx]
                    + omega_grid[idx] * d_omega_grid[idx];
            }
        }

        let stride = 3usize.pow((active_len - 1) as u32);
        [grid[..stride].to_vec(), grid[2 * stride..].to_vec()]
    }

    #[test]
    fn test_next_svo_accumulators_match_dense_ternary_reference() {
        let mut rng = SmallRng::seed_from_u64(4);

        for split_len in 0..=4 {
            for svo_len in 1..=5 {
                let total_len = split_len + svo_len;

                for _ in 0..20 {
                    let poly = (0..1 << total_len)
                        .map(|_| rng.random::<EF>())
                        .collect::<Vec<_>>();
                    let point = (0..total_len)
                        .map(|_| rng.random::<EF>())
                        .collect::<Vec<_>>();
                    let (_p_split, p_svo) = point.split_at(split_len);
                    let p_svo = Point::new(p_svo.to_vec());
                    let [d_eq, d_t, d_omega] =
                        split_compressions_for_next(&poly, &point, split_len, svo_len);

                    for active_len in 1..=svo_len {
                        let round = next_round_partials_suffix(
                            &Poly::new(d_eq.clone()),
                            &Poly::new(d_t.clone()),
                            &Poly::new(d_omega.clone()),
                            &p_svo,
                            active_len,
                        );
                        let (_svo_rest, svo_active) = p_svo.split_at(svo_len - active_len);
                        let stride = 3usize.pow((active_len - 1) as u32);
                        let mut production0 = EF::zero_vec(stride);
                        let mut production_inf = EF::zero_vec(stride);
                        round.accumulate_suffix(
                            svo_active.as_slice(),
                            &mut production0,
                            &mut production_inf,
                        );
                        let production = [production0, production_inf];
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

    #[test]
    fn test_svo_point_eval_next_suffix() {
        let assert_eval = |svo_point: &SvoPoint<F, EF>, poly: &Poly<F>, point: &Point<EF>| {
            assert!(matches!(svo_point.var_order(), VariableOrder::Suffix));

            let expected = poly.eval_next_base(point);
            let (actual, partials) = svo_point.eval_next_suffix(poly, None);
            assert_eq!(actual, expected);
            assert_eq!(partials.rounds().len(), svo_point.num_variables_svo());

            for (round_idx, round) in partials.rounds().iter().enumerate() {
                let active_len = round_idx + 1;
                let (_svo_rest, svo_active) = svo_point
                    .z_svo()
                    .split_at(svo_point.num_variables_svo() - active_len);

                assert_eq!(round.done().num_variables(), active_len);
                assert_eq!(round.carry().num_variables(), active_len);
                assert_eq!(round.omega().num_variables(), active_len);

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

        let k = 12;
        let mut rng = SmallRng::seed_from_u64(12);
        let poly = Poly::<F>::rand(&mut rng, k);
        let point = Point::<EF>::rand(&mut rng, k);

        for l0 in 0..=k {
            let svo_point = SvoPoint::<F, EF>::new_unpacked(l0, &point, VariableOrder::Suffix);
            assert_eval(&svo_point, &poly, &point);
        }
    }

    #[test]
    fn test_svo_point_eval_next_prefix() {
        let assert_eval = |svo_point: &SvoPoint<F, EF>, poly: &Poly<F>, point: &Point<EF>| {
            assert!(matches!(svo_point.var_order(), VariableOrder::Prefix));

            let expected = poly.eval_next_base(point);
            let (actual, partials) = svo_point.eval_next_prefix(poly);
            assert_eq!(actual, expected);
            assert_eq!(partials.rounds().len(), svo_point.num_variables_svo());

            for (round_idx, round) in partials.rounds().iter().enumerate() {
                let active_len = round_idx + 1;
                let (svo_active, _) = svo_point.z_svo().split_at(active_len);

                assert_eq!(round.done().num_variables(), active_len);
                assert_eq!(round.carry().num_variables(), active_len);
                assert_eq!(round.omega().num_variables(), active_len);

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

        let k = 12;
        let mut rng = SmallRng::seed_from_u64(13);
        let poly = Poly::<F>::rand(&mut rng, k);
        let point = Point::<EF>::rand(&mut rng, k);

        for l0 in 0..=k {
            let svo_point = SvoPoint::<F, EF>::new_unpacked(l0, &point, VariableOrder::Prefix);
            assert_eval(&svo_point, &poly, &point);
        }

        for l0 in 0..=k {
            let svo_point = SvoPoint::<F, EF>::new_packed(l0, &point);
            assert_eval(&svo_point, &poly, &point);
        }
    }

    #[test]
    fn test_svo_point_accumulate_next_prefix() {
        let mut rng = SmallRng::seed_from_u64(14);
        let k = 12;
        let point = Point::<EF>::rand(&mut rng, k);
        let scale: EF = rng.random();
        let next = Poly::new_next_from_point(point.as_slice());

        for l0 in 0..=k {
            let svo_point = SvoPoint::<F, EF>::new_packed(l0, &point);
            let rs = Point::rand(&mut rng, l0);
            let expected = next.compress_prefix(&rs, scale);

            let mut out = Poly::<EF>::zero(expected.num_variables());
            svo_point.accumulate_next_prefix_into(out.as_mut_slice(), &rs, scale);
            assert_eq!(out, expected);
        }
    }
}
