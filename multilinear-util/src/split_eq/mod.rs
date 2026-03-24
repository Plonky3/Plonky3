pub(crate) mod eq;

use eq::EqMaybePacked;
use itertools::Itertools;
use p3_field::{ExtensionField, Field, PackedFieldExtension, PackedValue};
use p3_maybe_rayon::prelude::*;
use p3_util::log2_strict_usize;

use crate::evals::{PARALLEL_THRESHOLD, Poly};
use crate::multilinear::Point;

/// Factored eq polynomial table for `scale · eq(z, ·)`. Splits point `z` at the
/// midpoint into `(z_lo, z_hi)` and stores `scale · eq(z_lo, ·)` and `eq(z_hi, ·)`
/// separately, exploiting the identity `eq(z, x) = eq(z_lo, x_lo) · eq(z_hi, x_hi)`.
/// This avoids materializing the full `2^k` table, using `2 · 2^{k/2}` space instead.
#[derive(Debug, Clone)]
pub struct SplitEq<F: Field, EF: ExtensionField<F>> {
    /// Eq table for the low-half variables `z_lo`.
    pub(super) eq0: Poly<EF>,
    /// Eq table for the high-half variables `z_hi`.
    pub(super) eq1: EqMaybePacked<F, EF>,
}

impl<F: Field, EF: ExtensionField<F>> SplitEq<F, EF> {
    // --- Constructors ---

    /// Constructs a `SplitEq` with unpacked eq table for the low half.
    pub fn new_unpacked(point: &Point<EF>, scale: EF) -> Self {
        let (z0, z1) = point.split_at(point.num_vars() / 2);
        let eq0 = Poly::new_from_point(z0.as_slice(), scale);
        let eq1 = EqMaybePacked::new_unpacked(&z1);
        Self { eq0, eq1 }
    }

    /// Constructs a `SplitEq`, using packed field representation for the low half.
    /// Falls back to unpacked if the low half has fewer variables than the packing width.
    pub fn new_packed(point: &Point<EF>, scale: EF) -> Self {
        let (z0, z1) = point.split_at(point.num_vars() / 2);
        let eq0 = Poly::new_from_point(z0.as_slice(), scale);
        let eq1 = EqMaybePacked::new_packed(&z1);
        Self { eq0, eq1 }
    }

    /// Returns the number of variables.
    pub const fn num_vars(&self) -> usize {
        self.eq0.num_vars() + self.eq1.num_vars()
    }

    // --- Eval ---

    /// Evaluates a base-field polynomial against the split eq tables:
    /// ```text
    ///   Σ_{x ∈ {0,1}^k} eq(z, x) · poly(x)
    /// ```
    pub fn eval_base(&self, poly: &Poly<F>) -> EF {
        assert_eq!(poly.num_vars(), self.num_vars());

        if let Some(constant) = poly.as_constant() {
            return constant.into();
        }

        let cs = self.eq1.scalar_chunk_size();
        if (1 << self.num_vars()) < PARALLEL_THRESHOLD {
            poly.0
                .chunks(cs)
                .zip_eq(self.eq0.iter())
                .map(|(chunk, &w0)| self.eq1.dot_with_base(chunk) * w0)
                .sum::<EF>()
        } else {
            poly.0
                .par_chunks(cs)
                .zip_eq(self.eq0.0.par_iter())
                .map(|(chunk, &w0)| self.eq1.dot_with_base(chunk) * w0)
                .sum::<EF>()
        }
    }

    /// Evaluates an extension-field polynomial against the split eq tables:
    /// ```text
    ///   Σ_{x ∈ {0,1}^k} eq(z, x) · poly(x)
    /// ```
    pub fn eval_ext(&self, poly: &Poly<EF>) -> EF {
        assert_eq!(poly.num_vars(), self.num_vars());

        if let Some(constant) = poly.as_constant() {
            return constant;
        }

        let cs = self.eq1.scalar_chunk_size();
        if (1 << self.num_vars()) < PARALLEL_THRESHOLD {
            poly.0
                .chunks(cs)
                .zip_eq(self.eq0.iter())
                .map(|(chunk, &w0)| self.eq1.dot_with_ext(chunk) * w0)
                .sum::<EF>()
        } else {
            poly.0
                .par_chunks(cs)
                .zip_eq(self.eq0.0.par_iter())
                .map(|(chunk, &w0)| self.eq1.dot_with_ext(chunk) * w0)
                .sum::<EF>()
        }
    }

    /// Evaluates a packed extension-field polynomial against the split eq tables:
    /// ```text
    ///   Σ_{x ∈ {0,1}^k} eq(z, x) · poly(x)
    /// ```
    pub fn eval_packed(&self, poly: &Poly<EF::ExtensionPacking>) -> EF {
        assert_eq!(
            poly.num_vars() + log2_strict_usize(F::Packing::WIDTH),
            self.num_vars()
        );
        match &self.eq1 {
            EqMaybePacked::Packed(eq1) => {
                let cs = eq1.num_evals();
                if (1 << (self.num_vars() - log2_strict_usize(F::Packing::WIDTH)))
                    < PARALLEL_THRESHOLD
                {
                    poly.0
                        .chunks(cs)
                        .zip_eq(self.eq0.iter())
                        .map(|(chunk, &w0)| self.eq1.dot_with_ext_packed(chunk) * w0)
                        .sum::<EF>()
                } else {
                    poly.0
                        .par_chunks(cs)
                        .zip_eq(self.eq0.0.par_iter())
                        .map(|(chunk, &w0)| self.eq1.dot_with_ext_packed(chunk) * w0)
                        .sum::<EF>()
                }
            }
            EqMaybePacked::Unpacked(_) => self.eval_ext(&Poly::new(
                EF::ExtensionPacking::to_ext_iter(poly.iter().copied()).collect(),
            )),
        }
    }

    // --- Accumulate ---

    /// Accumulates the eq table into `out`, using the factored `eq0 · eq1`
    /// representation. When `scale` is `None`, uses `1`.
    /// ```text
    ///   out[x] += scale · eq(point, x)   for all x ∈ {0,1}^k
    /// ```
    pub fn accumulate_into(&self, out: &mut [EF], scale: Option<EF>) {
        assert_eq!(log2_strict_usize(out.len()), self.num_vars());
        let w_scale = scale.unwrap_or(EF::ONE);
        let cs = self.eq1.scalar_chunk_size();
        if (1 << self.num_vars()) < PARALLEL_THRESHOLD {
            out.chunks_mut(cs)
                .zip(self.eq0.iter())
                .for_each(|(chunk, &w0)| {
                    self.eq1.accumulate_scalar_into(chunk, w0 * w_scale);
                });
        } else {
            out.par_chunks_mut(cs)
                .zip(self.eq0.0.par_iter())
                .for_each(|(chunk, &w0)| {
                    self.eq1.accumulate_scalar_into(chunk, w0 * w_scale);
                });
        }
    }

    /// Like [`accumulate_into`](Self::accumulate_into), but writes into a packed
    /// extension-field buffer.
    pub fn accumulate_into_packed(&self, out: &mut [EF::ExtensionPacking], scale: Option<EF>) {
        assert_eq!(
            log2_strict_usize(F::Packing::WIDTH * out.len()),
            self.num_vars()
        );
        let w_scale = scale.unwrap_or(EF::ONE);
        let cs = self.eq1.scalar_chunk_size() / F::Packing::WIDTH;
        if (1 << self.num_vars()) < PARALLEL_THRESHOLD {
            out.chunks_mut(cs)
                .zip(self.eq0.iter())
                .for_each(|(chunk, &w0)| {
                    self.eq1.accumulate_packed_into(chunk, w0 * w_scale);
                });
        } else {
            out.par_chunks_mut(cs)
                .zip(self.eq0.0.par_iter())
                .for_each(|(chunk, &w0)| {
                    self.eq1.accumulate_packed_into(chunk, w0 * w_scale);
                });
        }
    }

    // --- Compress ---

    /// Fixes the low variables of a multilinear polynomial using the split eq
    /// tables, returning a reduced polynomial over the remaining high variables.
    ///
    /// Given `poly` with `n` variables and split eq with `m ≤ n` variables, computes:
    /// ```text
    ///   out(x_hi) = Σ_{y_lo ∈ {0,1}^m} eq(point, y_lo) · poly(y_lo, x_hi)
    /// ```
    pub fn compress_lo(&self, poly: &Poly<F>) -> Poly<EF> {
        assert!(self.num_vars() <= poly.num_vars());
        let k_inner = poly.num_vars() - self.num_vars();
        let size_outer = poly.num_evals() / self.eq0.num_evals();

        if (1 << poly.num_vars()) < PARALLEL_THRESHOLD {
            let mut out = Poly::<EF>::zero(k_inner);
            poly.0
                .chunks(size_outer)
                .zip_eq(self.eq0.iter())
                .for_each(|(chunk, &w0)| {
                    self.eq1.compress_lo_into(out.as_mut_slice(), chunk, w0);
                });
            out
        } else {
            poly.0
                .par_chunks(size_outer)
                .zip_eq(self.eq0.0.par_iter())
                .par_fold_reduce(
                    || Poly::<EF>::zero(k_inner),
                    |mut acc, (chunk, &w0)| {
                        self.eq1.compress_lo_into(acc.as_mut_slice(), chunk, w0);
                        acc
                    },
                    |mut acc, part| {
                        acc.0
                            .iter_mut()
                            .zip_eq(part.iter())
                            .for_each(|(acc, &part)| *acc += part);
                        acc
                    },
                )
        }
    }

    /// Like [`compress_lo`](Self::compress_lo), but returns the result in packed
    /// extension-field representation. Requires that `poly` has enough variables
    /// to fill at least one packed element after compression.
    ///
    /// ```text
    ///   out(x_hi) = Σ_{y_lo ∈ {0,1}^m} eq(point, y_lo) · poly(y_lo, x_hi)
    /// ```
    pub fn compress_lo_to_packed(&self, poly: &Poly<F>) -> Poly<EF::ExtensionPacking> {
        assert!(self.num_vars() <= poly.num_vars());
        assert!(poly.num_vars() >= (self.num_vars() + log2_strict_usize(F::Packing::WIDTH)));
        let k_pack = log2_strict_usize(F::Packing::WIDTH);
        let k_inner = poly.num_vars() - self.num_vars() - k_pack;
        let size_outer = poly.num_evals() / self.eq0.num_evals();

        if (1 << poly.num_vars()) < PARALLEL_THRESHOLD {
            let mut out = Poly::<EF::ExtensionPacking>::zero(k_inner);
            poly.0
                .chunks(size_outer)
                .zip_eq(self.eq0.iter())
                .for_each(|(chunk, &w0)| {
                    self.eq1
                        .compress_lo_to_packed_into(out.as_mut_slice(), chunk, w0);
                });
            out
        } else {
            poly.0
                .par_chunks(size_outer)
                .zip_eq(self.eq0.0.par_iter())
                .par_fold_reduce(
                    || Poly::zero(k_inner),
                    |mut acc, (chunk, &w0)| {
                        self.eq1
                            .compress_lo_to_packed_into(acc.as_mut_slice(), chunk, w0);
                        acc
                    },
                    |mut acc, part| {
                        acc.0
                            .iter_mut()
                            .zip_eq(part.iter())
                            .for_each(|(acc, &part)| *acc += part);
                        acc
                    },
                )
        }
    }

    /// Fixes the high variables of a multilinear polynomial using the split eq
    /// tables, returning a reduced polynomial over the remaining low variables.
    ///
    /// Given `poly` with `n` variables and split eq with `m ≤ n` variables, computes:
    /// ```text
    ///   out(x_lo) = Σ_{y_hi ∈ {0,1}^m} eq(point, y_hi) · poly(x_lo, y_hi)
    /// ```
    pub fn compress_hi(&self, poly: &Poly<F>) -> Poly<EF> {
        let mut out = Poly::zero(poly.num_vars() - self.num_vars());
        self.compress_hi_into(out.as_mut_slice(), poly);
        out
    }

    /// Like [`compress_hi`](Self::compress_hi), but writes into a pre-allocated buffer.
    pub fn compress_hi_into(&self, out: &mut [EF], poly: &Poly<F>) {
        assert!(self.num_vars() <= poly.num_vars());
        assert_eq!(out.len(), poly.num_evals() >> self.num_vars());

        if (1 << self.num_vars()) < PARALLEL_THRESHOLD {
            out.iter_mut()
                .zip_eq(poly.0.chunks(1 << self.num_vars()))
                .for_each(|(out, chunk)| {
                    *out = self.eq1.compress_hi_dot(chunk, &self.eq0);
                });
        } else {
            out.par_iter_mut()
                .zip(poly.0.par_chunks(1 << self.num_vars()))
                .for_each(|(out, chunk)| {
                    *out = self.eq1.compress_hi_dot(chunk, &self.eq0);
                });
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_baby_bear::BabyBear;
    use p3_field::extension::BinomialExtensionField;
    use p3_field::{ExtensionField, Field, PackedValue, PrimeCharacteristicRing, dot_product};
    use p3_util::log2_strict_usize;
    use proptest::prelude::*;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use crate::evals::Poly;
    use crate::multilinear::Point;
    use crate::split_eq::SplitEq;

    type F = BabyBear;
    type PackedF = <F as Field>::Packing;
    type EF = BinomialExtensionField<F, 4>;

    const K_PACK: usize = log2_strict_usize(PackedF::WIDTH);

    /// Naive multilinear evaluation: `sum_x eq(point, x) * evals[x]`.
    fn eval_reference<F: Field, EF: ExtensionField<F>>(evals: &[F], point: &[EF]) -> EF {
        let eq = Poly::new_from_point(point, EF::ONE);
        dot_product(eq.iter().copied(), evals.iter().copied())
    }

    /// Naive reference: materializes the full eq table and accumulates.
    fn accumulate_reference(out: &mut [EF], point: &Point<EF>, scale: EF) {
        let eq = Poly::new_from_point(point.as_slice(), scale);
        out.iter_mut().zip(eq.iter()).for_each(|(o, &e)| *o += e);
    }

    // -----------------------------------------------------------------------
    // Eval: roundtrip unpacked == packed == reference
    // -----------------------------------------------------------------------

    proptest! {
        #[test]
        fn prop_eval_base_matches_reference(k in 0usize..=14, seed in any::<u64>()) {
            let mut rng = SmallRng::seed_from_u64(seed);
            let poly = Poly::<F>::rand(&mut rng, k);
            let point = Point::<EF>::rand(&mut rng, k);
            let expected = eval_reference(poly.as_slice(), point.as_slice());

            prop_assert_eq!(
                expected,
                SplitEq::<F, EF>::new_unpacked(&point, EF::ONE).eval_base(&poly),
            );
            prop_assert_eq!(
                expected,
                SplitEq::<F, EF>::new_packed(&point, EF::ONE).eval_base(&poly),
            );
        }

        #[test]
        fn prop_eval_ext_matches_reference(k in 0usize..=14, seed in any::<u64>()) {
            let mut rng = SmallRng::seed_from_u64(seed);
            let poly = Poly::<EF>::rand(&mut rng, k);
            let point = Point::<EF>::rand(&mut rng, k);
            let expected = eval_reference(poly.as_slice(), point.as_slice());

            prop_assert_eq!(
                expected,
                SplitEq::<F, EF>::new_unpacked(&point, EF::ONE).eval_ext(&poly),
            );
            prop_assert_eq!(
                expected,
                SplitEq::<F, EF>::new_packed(&point, EF::ONE).eval_ext(&poly),
            );
        }

        #[test]
        fn prop_eval_packed_matches_reference(
            k in K_PACK..=14usize,
            seed in any::<u64>(),
        ) {
            let mut rng = SmallRng::seed_from_u64(seed);
            let poly = Poly::<EF>::rand(&mut rng, k);
            let point = Point::<EF>::rand(&mut rng, k);
            let expected = eval_reference(poly.as_slice(), point.as_slice());
            let packed = poly.pack::<F, EF>();

            prop_assert_eq!(
                expected,
                SplitEq::<F, EF>::new_packed(&point, EF::ONE).eval_packed(&packed),
            );
            prop_assert_eq!(
                expected,
                SplitEq::<F, EF>::new_unpacked(&point, EF::ONE).eval_packed(&packed),
            );
        }
    }

    // -----------------------------------------------------------------------
    // Accumulate: reference vs unpacked vs packed, scale in ctor vs argument
    // -----------------------------------------------------------------------

    proptest! {
        #[test]
        fn prop_accumulate_matches_reference(k in 0usize..=14, seed in any::<u64>()) {
            let mut rng = SmallRng::seed_from_u64(seed);
            let input = Poly::<EF>::rand(&mut rng, k);
            let point = Point::<EF>::rand(&mut rng, k);
            let alpha: EF = rng.random();

            // Reference
            let mut expected = input.clone();
            accumulate_reference(expected.as_mut_slice(), &point, alpha);

            // Scale as argument
            let mut out = input.clone();
            SplitEq::<F, EF>::new_unpacked(&point, EF::ONE)
                .accumulate_into(out.as_mut_slice(), Some(alpha));
            prop_assert_eq!(&expected, &out);

            // Scale in constructor
            let mut out = input.clone();
            SplitEq::<F, EF>::new_unpacked(&point, alpha)
                .accumulate_into(out.as_mut_slice(), None);
            prop_assert_eq!(&expected, &out);

            // Packed path, scale as argument
            let mut out = input.clone();
            SplitEq::<F, EF>::new_packed(&point, EF::ONE)
                .accumulate_into(out.as_mut_slice(), Some(alpha));
            prop_assert_eq!(&expected, &out);

            // Packed path, scale in constructor
            let mut out = input.clone();
            SplitEq::<F, EF>::new_packed(&point, alpha)
                .accumulate_into(out.as_mut_slice(), None);
            prop_assert_eq!(&expected, &out);
        }

        #[test]
        fn prop_accumulate_into_packed_matches_scalar(
            k in (K_PACK + 1)..=14usize,
            seed in any::<u64>(),
        ) {
            let mut rng = SmallRng::seed_from_u64(seed);
            let input = Poly::<EF>::rand(&mut rng, k);
            let point = Point::<EF>::rand(&mut rng, k);
            let alpha: EF = rng.random();

            // Scalar reference
            let mut expected = input.clone();
            accumulate_reference(expected.as_mut_slice(), &point, alpha);

            // Packed with scale as argument
            let mut out = input.clone().pack::<F, EF>();
            SplitEq::<F, EF>::new_packed(&point, EF::ONE)
                .accumulate_into_packed(out.as_mut_slice(), Some(alpha));
            prop_assert_eq!(&expected, &out.unpack::<F, EF>());

            // Packed with scale in constructor
            let mut out = input.pack::<F, EF>();
            SplitEq::<F, EF>::new_packed(&point, alpha)
                .accumulate_into_packed(out.as_mut_slice(), None);
            prop_assert_eq!(&expected, &out.unpack::<F, EF>());
        }
    }

    // -----------------------------------------------------------------------
    // Compress: roundtrip compress_lo → eval == direct eval
    // -----------------------------------------------------------------------

    proptest! {
        #[test]
        fn prop_compress_lo_roundtrip(
            k in 0usize..=14,
            point_k_ratio in 0.0f64..=1.0,
            seed in any::<u64>(),
        ) {
            let point_k = ((k as f64) * point_k_ratio) as usize;
            let mut rng = SmallRng::seed_from_u64(seed);
            let poly = Poly::<F>::rand(&mut rng, k);
            let point = Point::<EF>::rand(&mut rng, k);
            let expected = eval_reference(poly.as_slice(), point.as_slice());

            let (point_lo, point_hi) = point.split_at(point_k);
            let split_lo = SplitEq::<F, EF>::new_unpacked(&point_lo, EF::ONE);
            let split_hi = SplitEq::<F, EF>::new_unpacked(&point_hi, EF::ONE);

            let compressed = split_lo.compress_lo(&poly);
            prop_assert_eq!(expected, split_hi.eval_ext(&compressed));
        }

        #[test]
        fn prop_compress_packed_roundtrip(
            k in (2 * K_PACK)..=14usize,
            seed in any::<u64>(),
        ) {
            let mut rng = SmallRng::seed_from_u64(seed);
            let poly = Poly::<F>::rand(&mut rng, k);
            let point = Point::<EF>::rand(&mut rng, k);
            let expected = eval_reference(poly.as_slice(), point.as_slice());

            // Use midpoint split so both halves have enough vars for packing
            let point_k = k / 2;
            // Guard: both halves need enough vars
            prop_assume!(k >= point_k + K_PACK);

            let (point_lo, point_hi) = point.split_at(point_k);
            let split_lo = SplitEq::<F, EF>::new_packed(&point_lo, EF::ONE);
            let split_hi = SplitEq::<F, EF>::new_packed(&point_hi, EF::ONE);

            // compress_lo packed → eval
            let compressed = split_lo.compress_lo(&poly);
            prop_assert_eq!(expected, split_hi.eval_ext(&compressed));

            // compress_lo_to_packed → unpack → eval
            let compressed = split_lo.compress_lo_to_packed(&poly).unpack::<F, EF>();
            prop_assert_eq!(expected, split_hi.eval_ext(&compressed));

            // compress_hi packed → eval
            let compressed = split_hi.compress_hi(&poly);
            prop_assert_eq!(expected, split_lo.eval_ext(&compressed));
        }
    }

    // -----------------------------------------------------------------------
    // Edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn test_eval_constant_poly() {
        let mut rng = SmallRng::seed_from_u64(42);
        for k in 0..=4 {
            let point = Point::<EF>::rand(&mut rng, k);
            let poly = Poly::new(vec![F::TWO; 1 << k]);
            // eq(z,·) sums to 1 over the hypercube, so eval should be 2
            let result = SplitEq::<F, EF>::new_packed(&point, EF::ONE).eval_base(&poly);
            assert_eq!(result, EF::TWO);
        }
    }

    #[test]
    fn test_accumulate_zero_scale() {
        let mut rng = SmallRng::seed_from_u64(42);
        let k = 6;
        let input = Poly::<EF>::rand(&mut rng, k);
        let point = Point::<EF>::rand(&mut rng, k);

        let mut out = input.clone();
        SplitEq::<F, EF>::new_packed(&point, EF::ZERO).accumulate_into(out.as_mut_slice(), None);
        // Adding zero-scaled eq table should not change anything
        assert_eq!(input, out);
    }

    #[test]
    fn test_compress_lo_full_point_yields_scalar() {
        let mut rng = SmallRng::seed_from_u64(42);
        let k = 8;
        let poly = Poly::<F>::rand(&mut rng, k);
        let point = Point::<EF>::rand(&mut rng, k);
        let expected = eval_reference(poly.as_slice(), point.as_slice());

        // compress_lo with all vars → 0-var output (a scalar)
        let split = SplitEq::<F, EF>::new_unpacked(&point, EF::ONE);
        let compressed = split.compress_lo(&poly);
        assert_eq!(compressed.num_vars(), 0);
        assert_eq!(compressed.as_constant().unwrap(), expected);
    }

    #[test]
    fn test_compress_hi_no_vars_is_identity() {
        let mut rng = SmallRng::seed_from_u64(42);
        let k = 8;
        let poly = Poly::<F>::rand(&mut rng, k);
        let point = Point::<EF>::rand(&mut rng, 0);

        // compress_hi with 0-var eq → should return poly as-is (lifted to EF)
        let split = SplitEq::<F, EF>::new_unpacked(&point, EF::ONE);
        let compressed = split.compress_hi(&poly);
        assert_eq!(compressed.num_vars(), k);
        for (c, &p) in compressed.iter().zip(poly.iter()) {
            assert_eq!(*c, EF::from(p));
        }
    }
}
