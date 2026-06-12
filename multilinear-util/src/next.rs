use itertools::Itertools;
use p3_field::{ExtensionField, Field, PackedFieldExtension, PackedValue, dot_product};
use p3_maybe_rayon::prelude::*;

use crate::point::Point;
use crate::poly::{PARALLEL_THRESHOLD, Poly};
use crate::split_eq::SplitEq;
use crate::split_eq::eq::EqMaybePacked;

impl<F: Field> Point<F> {
    /// Evaluates the repeat-last Next closed-form carry state.
    ///
    /// Coordinates are processed from last to first. Passing the full point and
    /// row gives the full repeat-last Next value as `done + omega`; passing only
    /// a low-bit suffix leaves `carry` alive for the unprocessed prefix.
    #[must_use]
    pub fn eval_next(point: &[F], row: &[F]) -> (F, F, F) {
        assert_eq!(point.len(), row.len());

        let mut carry = F::ONE;
        let mut done = F::ZERO;
        let mut omega = F::ONE;
        for (&point_bit, &row_bit) in point.iter().zip(row).rev() {
            let eq = row_bit.double() * point_bit - point_bit - row_bit + F::ONE;
            let prev_carry = carry;
            carry = prev_carry * point_bit * (F::ONE - row_bit);
            done = done * eq + prev_carry * (F::ONE - point_bit) * row_bit;
            omega *= point_bit * row_bit;
        }
        (carry, done, omega)
    }
}

impl<F: Field, EF: ExtensionField<F>> EqMaybePacked<F, EF> {
    /// Adds `eq_weight * eq1[i] + shifted_weight * eq1[i - 1]` into each row.
    ///
    /// `boundary` is the already-scaled shifted contribution for row 0, whose
    /// predecessor belongs to the previous outer chunk.
    #[doc(hidden)]
    pub fn accumulate_next_chunk_into(
        &self,
        out: &mut [EF],
        eq_weight: EF,
        shifted_weight: EF,
        boundary: EF,
    ) {
        match self {
            Self::Unpacked(eq1) => {
                let (first_out, rest_out) = out.split_first_mut().unwrap();
                let mut eq1 = eq1.iter().copied();
                let mut prev = eq1.next().unwrap();

                *first_out += eq_weight * prev + boundary;
                rest_out.iter_mut().zip_eq(eq1).for_each(|(out, w1)| {
                    *out += eq_weight * w1 + shifted_weight * prev;
                    prev = w1;
                });
            }
            Self::Packed(eq1) => {
                let (first_out, rest_out) = out.split_first_mut().unwrap();
                let mut eq1 = EF::ExtensionPacking::to_ext_iter(eq1.iter().copied());
                let mut prev = eq1.next().unwrap();

                *first_out += eq_weight * prev + boundary;
                rest_out.iter_mut().zip_eq(eq1).for_each(|(out, w1)| {
                    *out += eq_weight * w1 + shifted_weight * prev;
                    prev = w1;
                });
            }
        }
    }

    /// Weighted accumulation for prefix compression using shifted eq1 weights.
    ///
    /// The first row is intentionally skipped; callers handle it with the
    /// previous outer chunk's last equality weight.
    pub(super) fn compress_prefix_shifted_into(&self, out: &mut [EF], chunk: &[F], w0: EF) {
        let size_inner = out.len();
        match self {
            Self::Unpacked(eq1) => {
                chunk
                    .chunks(size_inner)
                    .skip(1)
                    .zip(eq1.iter())
                    .for_each(|(chunk, &w1)| {
                        let w = w0 * w1;
                        out.iter_mut()
                            .zip_eq(chunk.iter())
                            .for_each(|(acc, &f)| *acc += w * f);
                    });
            }
            Self::Packed(eq1) => {
                chunk[size_inner..]
                    .chunks(size_inner * F::Packing::WIDTH)
                    .zip(eq1.iter())
                    .for_each(|(chunk, &w1)| {
                        let w = w1 * w0;
                        chunk
                            .chunks(size_inner)
                            .zip(EF::ExtensionPacking::to_ext_iter([w]))
                            .for_each(|(chunk, w)| {
                                out.iter_mut()
                                    .zip_eq(chunk.iter())
                                    .for_each(|(acc, &f)| *acc += w * f);
                            });
                    });
            }
        }
    }

    /// Dots a base-field chunk with the beginning of this eq table.
    ///
    /// Used for shifted repeat-last Next weights, where callers pass
    /// `poly[1..]` chunks and the first eq weight corresponds to the previous
    /// row. The final chunk may be shorter than the packed eq table width.
    pub(super) fn dot_with_base_shifted(&self, chunk: &[F]) -> EF {
        debug_assert!(chunk.len() <= self.scalar_chunk_size());

        match self {
            Self::Unpacked(eq1) => {
                dot_product(eq1.iter().take(chunk.len()).copied(), chunk.iter().copied())
            }
            Self::Packed(eq1) => {
                let (packed, suffix) = F::Packing::pack_slice_with_suffix(chunk);
                let mut sum = EF::ZERO;
                if !packed.is_empty() {
                    let packed_sum = dot_product(eq1.iter().copied(), packed.iter().copied());
                    sum += EF::ExtensionPacking::to_ext_iter([packed_sum]).sum::<EF>();
                }

                if !suffix.is_empty() {
                    let w1 = eq1.as_slice()[packed.len()];
                    sum += EF::ExtensionPacking::to_ext_iter([w1])
                        .zip(suffix.iter())
                        .map(|(w1, &value)| w1 * value)
                        .sum::<EF>();
                }

                sum
            }
        }
    }

    /// Returns the final scalar entry of this eq table.
    ///
    /// For packed storage this extracts the last lane of the last packed value.
    /// Repeat-last Next uses this as the boundary weight for the repeated final
    /// row.
    #[doc(hidden)]
    pub fn last_scalar(&self) -> EF {
        match self {
            Self::Unpacked(eq1) => *eq1.as_slice().last().unwrap(),
            Self::Packed(eq1) => {
                EF::ExtensionPacking::to_ext_iter([*eq1.as_slice().last().unwrap()])
                    .last()
                    .unwrap()
            }
        }
    }
}

impl<F: Field, EF: ExtensionField<F>> SplitEq<F, EF> {
    /// Evaluates the repeat-last shifted view of a base-field polynomial.
    ///
    /// Computes:
    /// ```text
    /// sum_{x in {0,1}^k} eq(z, x) * poly(succ_repeat_last(x))
    /// ```
    ///
    /// This is the scalar value of a next-row opening, computed without
    /// materializing the shifted polynomial.
    ///
    /// # Panics
    ///
    /// Panics if the polynomial and eq table have different numbers of variables.
    pub fn eval_next_base(&self, poly: &Poly<F>) -> EF {
        assert_eq!(poly.num_variables(), self.num_variables());

        // A constant has only the repeated-last row, so shifting is a no-op.
        if let Some(constant) = poly.as_constant() {
            return constant.into();
        }

        if poly.num_variables() == 1 {
            return poly.as_slice()[1].into();
        }

        let evals = poly.as_slice();
        let last = *evals.last().unwrap();
        let cs = self.eq1.scalar_chunk_size();
        let mut sum = if poly.num_evals() < PARALLEL_THRESHOLD {
            self.eq0
                .iter()
                .zip_eq(evals[1..].chunks(cs))
                .map(|(&w0, chunk)| self.eq1.dot_with_base_shifted(chunk) * w0)
                .sum::<EF>()
        } else {
            self.eq0
                .0
                .par_iter()
                .zip_eq(evals[1..].par_chunks(cs))
                .map(|(&w0, chunk)| self.eq1.dot_with_base_shifted(chunk) * w0)
                .sum::<EF>()
        };

        sum += *self.eq0.as_slice().last().unwrap() * self.eq1.last_scalar() * last;
        sum
    }

    /// Fixes the prefix variables using the shifted eq table
    /// `T = [0, eq[0], ..., eq[last - 1]]`.
    ///
    /// Computes:
    /// ```text
    /// out(x_suffix) = sum_{y_prefix != 0}
    ///     eq(point, y_prefix - 1) * poly(y_prefix, x_suffix)
    /// ```
    ///
    /// This is the `T_split` compression used by repeat-last Next.
    pub fn compress_prefix_shifted(&self, poly: &Poly<F>) -> Poly<EF> {
        assert!(self.num_variables() <= poly.num_variables());

        let k_inner = poly.num_variables() - self.num_variables();
        let inner_size = 1 << k_inner;
        let size_outer = poly.num_evals() / self.eq0.num_evals();
        let mut out = Poly::<EF>::zero(k_inner);

        if self.num_variables() == 0 {
            return out;
        }

        if (1 << poly.num_variables()) < PARALLEL_THRESHOLD {
            let mut prev_last = EF::ZERO;
            let eq1_last = self.eq1.last_scalar();
            poly.as_slice()
                .chunks(size_outer)
                .zip_eq(self.eq0.iter())
                .for_each(|(chunk, &w0)| {
                    out.as_mut_slice()
                        .iter_mut()
                        .zip_eq(chunk[..inner_size].iter())
                        .for_each(|(out, &value)| *out += prev_last * value);
                    self.eq1
                        .compress_prefix_shifted_into(out.as_mut_slice(), chunk, w0);
                    prev_last = w0 * eq1_last;
                });
            out
        } else {
            let eq0 = self.eq0.as_slice();
            let eq1_last = self.eq1.last_scalar();
            poly.as_slice()
                .par_chunks(size_outer)
                .enumerate()
                .zip_eq(eq0.par_iter())
                .par_fold_reduce(
                    || Poly::<EF>::zero(k_inner),
                    |mut acc, ((idx, chunk), &w0)| {
                        if idx > 0 {
                            let boundary = eq0[idx - 1] * eq1_last;
                            acc.as_mut_slice()
                                .iter_mut()
                                .zip_eq(chunk[..inner_size].iter())
                                .for_each(|(out, &value)| *out += boundary * value);
                        }
                        self.eq1
                            .compress_prefix_shifted_into(acc.as_mut_slice(), chunk, w0);
                        acc
                    },
                    |mut acc, part| {
                        acc.as_mut_slice()
                            .iter_mut()
                            .zip_eq(part.iter())
                            .for_each(|(acc, &part)| *acc += part);
                        acc
                    },
                )
        }
    }

    /// Fixes the suffix variables using the shifted eq table
    /// `T = [0, eq[0], ..., eq[last - 1]]`.
    ///
    /// Computes:
    /// ```text
    /// out(x_prefix) = sum_{y_suffix != 0}
    ///     eq(point, y_suffix - 1) * poly(x_prefix, y_suffix)
    /// ```
    pub fn compress_suffix_shifted(&self, poly: &Poly<F>) -> Poly<EF> {
        assert!(self.num_variables() <= poly.num_variables());

        let suffix_rows = 1 << self.num_variables();
        let out_len = poly.num_evals() >> self.num_variables();
        let mut out = EF::zero_vec(out_len);
        let cs = self.eq1.scalar_chunk_size();

        if self.num_variables() == 0 {
            return Poly::new(out);
        }

        if poly.num_evals() < PARALLEL_THRESHOLD {
            out.iter_mut()
                .zip_eq(poly.as_slice().chunks(suffix_rows))
                .for_each(|(out, chunk)| {
                    *out = self
                        .eq0
                        .iter()
                        .zip_eq(chunk[1..].chunks(cs))
                        .map(|(&w0, chunk)| self.eq1.dot_with_base_shifted(chunk) * w0)
                        .sum();
                });
        } else {
            out.par_iter_mut()
                .zip_eq(poly.as_slice().par_chunks(suffix_rows))
                .for_each(|(out, chunk)| {
                    *out = self
                        .eq0
                        .iter()
                        .zip_eq(chunk[1..].chunks(cs))
                        .map(|(&w0, chunk)| self.eq1.dot_with_base_shifted(chunk) * w0)
                        .sum();
                });
        }

        Poly::new(out)
    }

    /// Returns the all-ones entry of the factored eq table.
    ///
    /// With unit scale this is `prod_i point_i`, which is the repeat-last
    /// `Omega` boundary weight used by the Next decomposition.
    pub fn last_scalar(&self) -> EF {
        *self.eq0.as_slice().last().unwrap() * self.eq1.last_scalar()
    }
}

impl<F: Field> Poly<F> {
    /// Evaluates the repeat-last shifted view of this polynomial at `point`.
    ///
    /// Computes
    /// ```text
    ///     sum_{x in {0,1}^n} eq(point, x) * self(succ_repeat_last(x)).
    /// ```
    #[must_use]
    #[inline]
    pub fn eval_next_base<EF: ExtensionField<F>>(&self, point: &Point<EF>) -> EF {
        SplitEq::new_packed(point, EF::ONE).eval_next_base(self)
    }

    /// Materializes the repeat-last Next weight table for `point`.
    ///
    /// If `eq = Eq(point, .)`, then:
    /// ```text
    /// Next(point, .) = [0, eq[0], eq[1], ..., eq[last - 1]] + Omega
    /// Omega[last] = eq[last]
    /// ```
    ///
    /// This is useful as a dense reference implementation. Hot paths should
    /// prefer shifted views or split/compressed forms instead of materializing.
    pub fn new_next_from_point(point: &[F]) -> Self {
        let mut res = Self::new_from_point(point, F::ONE).0;
        let n = res.len();

        let last = res[n - 1];
        res.copy_within(0..n - 1, 1);
        res[0] = F::ZERO;
        res[n - 1] += last;

        Self::new(res)
    }
}

#[cfg(test)]
mod test {
    use alloc::vec::Vec;

    use itertools::Itertools;
    use p3_baby_bear::BabyBear;
    use p3_field::extension::BinomialExtensionField;
    use p3_field::{ExtensionField, Field, PrimeCharacteristicRing, dot_product};
    use proptest::prelude::*;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use crate::point::Point;
    use crate::poly::Poly;
    use crate::split_eq::SplitEq;
    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;

    fn eval_reference<F: Field, EF: ExtensionField<F>>(evals: &[F], point: &[EF]) -> EF {
        let eq = Poly::new_from_point(point, EF::ONE);
        dot_product(eq.iter().copied(), evals.iter().copied())
    }

    fn eval_next_reference<F: Field, EF: ExtensionField<F>>(evals: &[F], point: &[EF]) -> EF {
        let mut shifted = evals.to_vec();
        let last = *shifted.last().unwrap();
        shifted.rotate_left(1);
        *shifted.last_mut().unwrap() = last;
        eval_reference(&shifted, point)
    }

    fn compress_prefix_shifted_poly_reference<F: Field, EF: ExtensionField<F>>(
        poly: &Poly<F>,
        point: &Point<EF>,
    ) -> Poly<EF> {
        assert!(point.num_variables() <= poly.num_variables());
        let inner_vars = poly.num_variables() - point.num_variables();
        let inner_rows = 1 << inner_vars;
        let split_rows = 1 << point.num_variables();
        let eq = Poly::new_from_point(point.as_slice(), EF::ONE);
        let mut out = Poly::<EF>::zero(inner_vars);

        for split_idx in 1..split_rows {
            let weight = eq.as_slice()[split_idx - 1];
            let chunk = &poly.as_slice()[split_idx * inner_rows..(split_idx + 1) * inner_rows];
            out.as_mut_slice()
                .iter_mut()
                .zip_eq(chunk.iter())
                .for_each(|(out, &value)| *out += weight * value);
        }

        out
    }

    fn compress_suffix_shifted_reference<F: Field, EF: ExtensionField<F>>(
        poly: &Poly<F>,
        point: &Point<EF>,
    ) -> Poly<EF> {
        assert!(point.num_variables() <= poly.num_variables());
        let suffix_vars = point.num_variables();
        let prefix_rows = poly.num_evals() >> suffix_vars;
        let suffix_rows = 1 << suffix_vars;
        let eq = Poly::new_from_point(point.as_slice(), EF::ONE);
        let mut out = Poly::<EF>::zero(poly.num_variables() - suffix_vars);

        for prefix_idx in 0..prefix_rows {
            for suffix_idx in 1..suffix_rows {
                let weight = eq.as_slice()[suffix_idx - 1];
                let value = poly.as_slice()[(prefix_idx << suffix_vars) | suffix_idx];
                out.as_mut_slice()[prefix_idx] += weight * value;
            }
        }

        out
    }

    fn accumulate_next_chunk_reference(
        eq: &[EF],
        out: &mut [EF],
        eq_weight: EF,
        shifted_weight: EF,
        boundary: EF,
    ) {
        assert_eq!(eq.len(), out.len());

        for idx in 0..out.len() {
            out[idx] += eq_weight * eq[idx]
                + if idx == 0 {
                    boundary
                } else {
                    shifted_weight * eq[idx - 1]
                };
        }
    }

    fn compress_prefix_shifted_reference(eq: &[EF], out: &mut [EF], chunk: &[F], w0: EF) {
        let size_inner = out.len();
        chunk
            .chunks(size_inner)
            .skip(1)
            .zip(eq.iter())
            .for_each(|(chunk, &w1)| {
                let w = w0 * w1;
                out.iter_mut()
                    .zip_eq(chunk.iter())
                    .for_each(|(out, &value)| *out += w * value);
            });
    }

    #[test]
    fn test_next_closed_forms() {
        let mut rng = SmallRng::seed_from_u64(0);
        for k in 0..14 {
            let p0 = Point::<F>::rand(&mut rng, k);
            let p1 = Point::<F>::rand(&mut rng, k);
            let (_carry, done, omega) = Point::eval_next(p1.as_slice(), p0.as_slice());
            let e0 = done + omega;
            let next = Poly::new_next_from_point(p1.as_slice());
            let eq = Poly::new_from_point(p0.as_slice(), F::ONE);
            let e1 = dot_product(eq.iter().copied(), next.iter().copied());
            assert_eq!(e0, e1);
            let e1 = next.eval_base(&p0);
            assert_eq!(e0, e1);
            let next = Poly::new_from_point(p0.as_slice(), F::ONE);
            let e1 = next.eval_next_base(&p1);
            assert_eq!(e0, e1);
        }
    }

    #[test]
    fn test_accumulate_next_chunk_into_matches_reference() {
        let mut rng = SmallRng::seed_from_u64(1);

        for k in 2..=10 {
            let point = Point::<EF>::rand(&mut rng, k);
            let split_at = point.num_variables() / 2;
            let (_prefix, suffix) = point.split_at(split_at);
            let eq = Poly::new_from_point(suffix.as_slice(), EF::ONE);

            let eq_weight = rng.random();
            let shifted_weight = rng.random();
            let boundary = rng.random();
            let initial = (0..eq.num_evals())
                .map(|_| rng.random())
                .collect::<Vec<_>>();

            let mut expected = initial.clone();
            accumulate_next_chunk_reference(
                eq.as_slice(),
                &mut expected,
                eq_weight,
                shifted_weight,
                boundary,
            );

            let mut unpacked = initial.clone();
            SplitEq::<F, EF>::new_unpacked(&point, EF::ONE)
                .eq1()
                .accumulate_next_chunk_into(
                    unpacked.as_mut_slice(),
                    eq_weight,
                    shifted_weight,
                    boundary,
                );
            assert_eq!(unpacked, expected);

            let mut packed = initial.clone();
            SplitEq::<F, EF>::new_packed(&point, EF::ONE)
                .eq1()
                .accumulate_next_chunk_into(
                    packed.as_mut_slice(),
                    eq_weight,
                    shifted_weight,
                    boundary,
                );
            assert_eq!(packed, expected);
        }
    }

    #[test]
    fn test_compress_prefix_shifted_into_matches_reference() {
        let mut rng = SmallRng::seed_from_u64(2);

        for k in 2..=10 {
            let point = Point::<EF>::rand(&mut rng, k);
            let split_at = point.num_variables() / 2;
            let (_prefix, suffix) = point.split_at(split_at);
            let eq = Poly::new_from_point(suffix.as_slice(), EF::ONE);
            let size_inner = 1 << split_at;
            let chunk = (0..(eq.num_evals() * size_inner))
                .map(|_| rng.random())
                .collect::<Vec<_>>();
            let w0 = rng.random();
            let initial = (0..size_inner).map(|_| rng.random()).collect::<Vec<_>>();

            let mut expected = initial.clone();
            compress_prefix_shifted_reference(eq.as_slice(), &mut expected, chunk.as_slice(), w0);

            let mut unpacked = initial.clone();
            SplitEq::<F, EF>::new_unpacked(&point, EF::ONE)
                .eq1()
                .compress_prefix_shifted_into(unpacked.as_mut_slice(), chunk.as_slice(), w0);
            assert_eq!(unpacked, expected);

            let mut packed = initial.clone();
            SplitEq::<F, EF>::new_packed(&point, EF::ONE)
                .eq1()
                .compress_prefix_shifted_into(packed.as_mut_slice(), chunk.as_slice(), w0);
            assert_eq!(packed, expected);
        }
    }

    proptest! {
        #[test]
        fn prop_eval_next_base_matches_shifted_poly_reference(k in 0usize..=14, seed in any::<u64>()) {
            let mut rng = SmallRng::seed_from_u64(seed);
            let poly = Poly::<F>::rand(&mut rng, k);
            let point = Point::<EF>::rand(&mut rng, k);
            let expected = eval_next_reference(poly.as_slice(), point.as_slice());

            prop_assert_eq!(
                expected,
                SplitEq::<F, EF>::new_unpacked(&point, EF::ONE).eval_next_base(&poly),
            );
            prop_assert_eq!(
                expected,
                SplitEq::<F, EF>::new_packed(&point, EF::ONE).eval_next_base(&poly),
            );
        }

        #[test]
        fn prop_compress_prefix_shifted_matches_reference(
            split_vars in 0usize..=8,
            inner_vars in 0usize..=6,
            seed in any::<u64>(),
        ) {
            let mut rng = SmallRng::seed_from_u64(seed);
            let total_vars = split_vars + inner_vars;
            let point = Point::<EF>::rand(&mut rng, split_vars);

            let base_poly = Poly::<F>::rand(&mut rng, total_vars);
            let expected = compress_prefix_shifted_poly_reference(&base_poly, &point);
            prop_assert_eq!(
                expected,
                SplitEq::<F, EF>::new_packed(&point, EF::ONE)
                    .compress_prefix_shifted(&base_poly),
            );

            let ext_poly = Poly::<EF>::rand(&mut rng, total_vars);
            let expected = compress_prefix_shifted_poly_reference(&ext_poly, &point);
            prop_assert_eq!(
                expected,
                SplitEq::<EF, EF>::new_packed(&point, EF::ONE)
                    .compress_prefix_shifted(&ext_poly),
            );
        }

        #[test]
        fn prop_compress_suffix_shifted_matches_reference(
            suffix_vars in 0usize..=8,
            prefix_vars in 0usize..=6,
            seed in any::<u64>(),
        ) {
            let mut rng = SmallRng::seed_from_u64(seed);
            let total_vars = suffix_vars + prefix_vars;
            let point = Point::<EF>::rand(&mut rng, suffix_vars);

            let base_poly = Poly::<F>::rand(&mut rng, total_vars);
            let expected = compress_suffix_shifted_reference(&base_poly, &point);
            prop_assert_eq!(
                expected,
                SplitEq::<F, EF>::new_packed(&point, EF::ONE)
                    .compress_suffix_shifted(&base_poly),
            );

            let ext_poly = Poly::<EF>::rand(&mut rng, total_vars);
            let expected = compress_suffix_shifted_reference(&ext_poly, &point);
            prop_assert_eq!(
                expected,
                SplitEq::<EF, EF>::new_packed(&point, EF::ONE)
                    .compress_suffix_shifted(&ext_poly),
            );
        }
    }
}
