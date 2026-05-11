//! Factored equality polynomial for space-efficient multilinear operations.
//!
//! # Mathematical Background
//!
//! The equality polynomial eq(z, x) for z in F^k evaluates to 1 when x = z
//! on the boolean hypercube and 0 elsewhere. Materializing the full table
//! requires 2^k space.
//!
//! This module splits z at the midpoint into (z_prefix, z_suffix) and exploits:
//!
//! ```text
//! eq(z, x) = eq(z_prefix, x_prefix) * eq(z_suffix, x_suffix)
//! ```
//!
//! By storing eq(z_prefix, .) and eq(z_suffix, .) separately, the total space
//! is 2 * 2^{k/2} instead of 2^k.
//!
//! The suffix-half table (eq1) may be stored in SIMD-packed form
//! for faster inner-loop throughput on supported architectures.

pub mod eq;
mod packed_kernel;

use eq::EqMaybePacked;
use itertools::Itertools;
use p3_field::{ExtensionField, Field, PackedFieldExtension, PackedValue};
use p3_maybe_rayon::prelude::*;
use p3_util::log2_strict_usize;

use crate::point::Point;
use crate::poly::{PARALLEL_THRESHOLD, Poly};

/// Factored eq polynomial table for scale * eq(z, .).
///
/// Splits the evaluation point z at the midpoint into (z_prefix, z_suffix):
/// - eq0 stores scale * eq(z_prefix, .) over {0,1}^{k/2}
/// - eq1 stores eq(z_suffix, .) over {0,1}^{k - k/2}, optionally SIMD-packed
///
/// # Memory Layout
///
/// ```text
/// Full table:    2^k extension-field elements
/// Factored:      2^{k/2} (eq0) + 2^{k - k/2} (eq1)
/// ```
///
/// For k = 20, this is 2 * 1024 instead of 1_048_576 elements.
#[derive(Debug, Clone)]
pub struct SplitEq<F: Field, EF: ExtensionField<F>> {
    /// Prefix-half eq table: scale * eq(z_prefix, .) with 2^{k/2} entries.
    pub(crate) eq0: Poly<EF>,
    /// Suffix-half eq table: eq(z_suffix, .), scalar or SIMD-packed.
    pub(crate) eq1: EqMaybePacked<F, EF>,
}

impl<F: Field, EF: ExtensionField<F>> SplitEq<F, EF> {
    /// Creates a factored eq table with a scalar (unpacked) suffix-half.
    ///
    /// The evaluation point is split at the midpoint.
    /// The scale factor is absorbed into the prefix-half table.
    pub fn new_unpacked(point: &Point<EF>, scale: EF) -> Self {
        // Split z into (z_prefix, z_suffix) at the midpoint.
        let (z0, z1) = point.split_at(point.num_variables() / 2);
        // Build eq0 = scale * eq(z_prefix, .) and eq1 = eq(z_suffix, .).
        let eq0 = Poly::new_from_point(z0.as_slice(), scale);
        let eq1 = EqMaybePacked::new_unpacked(&z1);
        Self { eq0, eq1 }
    }

    /// Creates a factored eq table, using SIMD packing for the suffix-half when possible.
    ///
    /// Falls back to scalar if the suffix-half has fewer variables than log_2(W).
    /// The scale factor is absorbed into the prefix-half table.
    pub fn new_packed(point: &Point<EF>, scale: EF) -> Self {
        // Split z into (z_prefix, z_suffix) at the midpoint.
        let (z0, z1) = point.split_at(point.num_variables() / 2);
        // Build eq0 with scale, and attempt SIMD packing for eq1.
        let eq0 = Poly::new_from_point(z0.as_slice(), scale);
        let eq1 = EqMaybePacked::new_packed(&z1);
        Self { eq0, eq1 }
    }

    /// Total number of variables: k = k_prefix + k_suffix.
    pub const fn num_variables(&self) -> usize {
        self.eq0.num_variables() + self.eq1.num_variables()
    }

    /// Returns the prefix-half eq table.
    pub const fn eq0(&self) -> &Poly<EF> {
        &self.eq0
    }

    /// Returns the suffix-half eq table.
    pub const fn eq1(&self) -> &EqMaybePacked<F, EF> {
        &self.eq1
    }

    /// Evaluates a base-field polynomial against the factored eq table.
    ///
    /// Computes:
    /// ```text
    /// sum_{x in {0,1}^k} eq(z, x) * poly(x)
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if the polynomial and eq table have different numbers of variables.
    pub fn eval_base(&self, poly: &Poly<F>) -> EF {
        assert_eq!(poly.num_variables(), self.num_variables());

        // Short-circuit: a constant polynomial evaluates to itself
        // since eq(z, .) sums to 1 over the hypercube.
        if let Some(constant) = poly.as_constant() {
            return constant.into();
        }

        // Number of scalar elements per eq1 block.
        let cs = self.eq1.scalar_chunk_size();
        if (1 << self.num_variables()) < PARALLEL_THRESHOLD {
            // Sequential: chunk poly by eq1 block size, pair with eq0 weights.
            // For each chunk, compute the inner dot product with eq1,
            // then multiply by the corresponding eq0 weight.
            poly.0
                .chunks(cs)
                .zip_eq(self.eq0.iter())
                .map(|(chunk, &w0)| self.eq1.dot_with_base(chunk) * w0)
                .sum::<EF>()
        } else {
            // Parallel: same logic with parallel iterators.
            poly.0
                .par_chunks(cs)
                .zip_eq(self.eq0.0.par_iter())
                .map(|(chunk, &w0)| self.eq1.dot_with_base(chunk) * w0)
                .sum::<EF>()
        }
    }

    /// Evaluates an extension-field polynomial against the factored eq table.
    ///
    /// Computes:
    /// ```text
    /// sum_{x in {0,1}^k} eq(z, x) * poly(x)
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if the polynomial and eq table have different numbers of variables.
    pub fn eval_ext(&self, poly: &Poly<EF>) -> EF {
        assert_eq!(poly.num_variables(), self.num_variables());

        // Short-circuit for constant polynomials.
        if let Some(constant) = poly.as_constant() {
            return constant;
        }

        let cs = self.eq1.scalar_chunk_size();
        if (1 << self.num_variables()) < PARALLEL_THRESHOLD {
            // Sequential outer loop: dot each chunk with eq1, weight by eq0.
            poly.0
                .chunks(cs)
                .zip_eq(self.eq0.iter())
                .map(|(chunk, &w0)| self.eq1.dot_with_ext(chunk) * w0)
                .sum::<EF>()
        } else {
            // Parallel path.
            poly.0
                .par_chunks(cs)
                .zip_eq(self.eq0.0.par_iter())
                .map(|(chunk, &w0)| self.eq1.dot_with_ext(chunk) * w0)
                .sum::<EF>()
        }
    }

    /// Evaluates a SIMD-packed extension-field polynomial against the factored eq table.
    ///
    /// Computes:
    /// ```text
    /// sum_{x in {0,1}^k} eq(z, x) * poly(x)
    /// ```
    ///
    /// The polynomial has k - log_2(W) packed entries representing k variables.
    ///
    /// # Panics
    ///
    /// Panics if the packed polynomial size is inconsistent with the eq table.
    pub fn eval_packed(&self, poly: &Poly<EF::ExtensionPacking>) -> EF {
        assert_eq!(
            poly.num_variables() + log2_strict_usize(F::Packing::WIDTH),
            self.num_variables()
        );
        match &self.eq1 {
            EqMaybePacked::Packed(eq1) => {
                // Both polynomial and eq1 are packed; use packed dot product kernel.
                let cs = eq1.num_evals();
                if (1 << (self.num_variables() - log2_strict_usize(F::Packing::WIDTH)))
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
            EqMaybePacked::Unpacked(_) => {
                // eq1 is scalar; unpack the polynomial and delegate to the scalar path.
                self.eval_ext(&Poly::new(
                    EF::ExtensionPacking::to_ext_iter(poly.iter().copied()).collect(),
                ))
            }
        }
    }

    /// Adds the factored eq table into a scalar output buffer.
    ///
    /// ```text
    /// out[x] += scale * eq(point, x)   for all x in {0,1}^k
    /// ```
    ///
    /// When scale is None, uses 1.
    ///
    /// # Panics
    ///
    /// Panics if the output buffer length is not 2^k.
    pub fn accumulate_into(&self, out: &mut [EF], scale: Option<EF>) {
        assert_eq!(log2_strict_usize(out.len()), self.num_variables());
        // Collapse optional scale into a single weight; identity if absent.
        let w_scale = scale.unwrap_or(EF::ONE);
        let cs = self.eq1.scalar_chunk_size();
        if (1 << self.num_variables()) < PARALLEL_THRESHOLD {
            // Sequential: for each eq0 entry, accumulate eq1 * (eq0_weight * scale).
            out.chunks_mut(cs)
                .zip(self.eq0.iter())
                .for_each(|(chunk, &w0)| {
                    self.eq1.accumulate_scalar_into(chunk, w0 * w_scale);
                });
        } else {
            // Parallel: same with parallel chunk iteration.
            out.par_chunks_mut(cs)
                .zip(self.eq0.0.par_iter())
                .for_each(|(chunk, &w0)| {
                    self.eq1.accumulate_scalar_into(chunk, w0 * w_scale);
                });
        }
    }

    /// Materializes the full eq table as a polynomial.
    pub fn materialize(&self) -> Poly<EF> {
        let mut out = Poly::zero(self.num_variables());
        self.accumulate_into(out.as_mut_slice(), None);
        out
    }

    /// Adds the factored eq table into a SIMD-packed output buffer.
    ///
    /// Same semantics as the scalar version, but the output is in packed form.
    ///
    /// # Panics
    ///
    /// Panics if the output buffer length * W != 2^k.
    pub fn accumulate_into_packed(&self, out: &mut [EF::ExtensionPacking], scale: Option<EF>) {
        assert_eq!(
            log2_strict_usize(F::Packing::WIDTH * out.len()),
            self.num_variables()
        );
        let w_scale = scale.unwrap_or(EF::ONE);

        // Apply naive method if number of variables is too small
        if log2_strict_usize(F::Packing::WIDTH) * 2 > self.num_variables() {
            out.iter_mut()
                .zip_eq(self.materialize().0.chunks(F::Packing::WIDTH))
                .for_each(|(out, chunk)| {
                    *out += EF::ExtensionPacking::from_ext_slice(chunk) * w_scale;
                });
        } else {
            // Chunk size in packed elements (scalar chunk size / W).
            let cs = self.eq1.scalar_chunk_size() / F::Packing::WIDTH;
            if (1 << self.num_variables()) < PARALLEL_THRESHOLD {
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
    }

    /// Fixes the prefix variables by summing against the factored eq table.
    ///
    /// Given a polynomial with n variables and this eq table with m <= n variables,
    /// computes:
    /// ```text
    /// out(x_suffix) = sum_{y_prefix in {0,1}^m} eq(point, y_prefix) * poly(y_prefix, x_suffix)
    /// ```
    ///
    /// The result has n - m variables.
    ///
    /// # Panics
    ///
    /// Panics if the eq table has more variables than the polynomial.
    pub fn compress_prefix(&self, poly: &Poly<F>) -> Poly<EF> {
        assert!(self.num_variables() <= poly.num_variables());
        // Number of remaining (output) variables.
        let k_inner = poly.num_variables() - self.num_variables();
        // Number of base-field elements per eq0 entry.
        let size_outer = poly.num_evals() / self.eq0.num_evals();

        if (1 << poly.num_variables()) < PARALLEL_THRESHOLD {
            // Sequential: accumulate each eq0 chunk into a shared output buffer.
            let mut out = Poly::<EF>::zero(k_inner);
            poly.0
                .chunks(size_outer)
                .zip_eq(self.eq0.iter())
                .for_each(|(chunk, &w0)| {
                    // Delegate inner-loop accumulation to the eq1 kernel.
                    self.eq1.compress_prefix_into(out.as_mut_slice(), chunk, w0);
                });
            out
        } else {
            // Parallel: each thread accumulates into a local buffer, then reduce.
            poly.0
                .par_chunks(size_outer)
                .zip_eq(self.eq0.0.par_iter())
                .par_fold_reduce(
                    || Poly::<EF>::zero(k_inner),
                    |mut acc, (chunk, &w0)| {
                        self.eq1.compress_prefix_into(acc.as_mut_slice(), chunk, w0);
                        acc
                    },
                    // Merge thread-local accumulators by element-wise addition.
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

    /// Fixes the prefix variables into a SIMD-packed output.
    ///
    /// Same operation as the scalar compression, but the result is in packed form.
    /// Requires n - m >= log_2(W) so at least one full packed element is produced.
    ///
    /// # Panics
    ///
    /// - Panics if the eq table has more variables than the polynomial.
    /// - Panics if the remaining variables are fewer than log_2(W).
    pub fn compress_prefix_to_packed(&self, poly: &Poly<F>) -> Poly<EF::ExtensionPacking> {
        assert!(self.num_variables() <= poly.num_variables());
        assert!(
            poly.num_variables() >= (self.num_variables() + log2_strict_usize(F::Packing::WIDTH))
        );
        let k_pack = log2_strict_usize(F::Packing::WIDTH);
        // Output has k_inner packed variables (each packed element holds W scalars).
        let k_inner = poly.num_variables() - self.num_variables() - k_pack;
        let size_outer = poly.num_evals() / self.eq0.num_evals();

        if (1 << poly.num_variables()) < PARALLEL_THRESHOLD {
            let mut out = Poly::<EF::ExtensionPacking>::zero(k_inner);
            poly.0
                .chunks(size_outer)
                .zip_eq(self.eq0.iter())
                .for_each(|(chunk, &w0)| {
                    self.eq1
                        .compress_prefix_to_packed_into(out.as_mut_slice(), chunk, w0);
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
                            .compress_prefix_to_packed_into(acc.as_mut_slice(), chunk, w0);
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

    /// Fixes the suffix variables by summing against the factored eq table.
    ///
    /// Given a polynomial with n variables and this eq table with m <= n variables,
    /// computes:
    /// ```text
    /// out(x_prefix) = sum_{y_suffix in {0,1}^m} eq(point, y_suffix) * poly(x_prefix, y_suffix)
    /// ```
    ///
    /// The result has n - m variables.
    ///
    /// # Panics
    ///
    /// Panics if the eq table has more variables than the polynomial.
    pub fn compress_suffix(&self, poly: &Poly<F>) -> Poly<EF> {
        // Allocate output and delegate to the in-place version.
        let mut out = Poly::zero(poly.num_variables() - self.num_variables());
        self.compress_suffix_into(out.as_mut_slice(), poly);
        out
    }

    /// Fixes the suffix variables into a pre-allocated buffer.
    ///
    /// # Panics
    ///
    /// - Panics if the eq table has more variables than the polynomial.
    /// - Panics if the output buffer length != 2^{n-m}.
    pub fn compress_suffix_into(&self, out: &mut [EF], poly: &Poly<F>) {
        assert!(self.num_variables() <= poly.num_variables());
        assert_eq!(out.len(), poly.num_evals() >> self.num_variables());

        if (1 << poly.num_variables()) < PARALLEL_THRESHOLD {
            // Sequential: each output element is a full dot product of one row
            // against the factored eq tables.
            out.iter_mut()
                .zip_eq(poly.0.chunks(1 << self.num_variables()))
                .for_each(|(out, chunk)| {
                    *out = self.eq1.compress_suffix_dot(chunk, &self.eq0);
                });
        } else {
            // Parallel: each output element is independent.
            out.par_iter_mut()
                .zip(poly.0.par_chunks(1 << self.num_variables()))
                .for_each(|(out, chunk)| {
                    *out = self.eq1.compress_suffix_dot(chunk, &self.eq0);
                });
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_baby_bear::BabyBear;
    use p3_field::extension::BinomialExtensionField;
    use p3_field::{PrimeCharacteristicRing, dot_product};
    use proptest::prelude::*;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::*;

    type F = BabyBear;
    type PackedF = <F as Field>::Packing;
    type EF = BinomialExtensionField<F, 4>;

    /// Minimum number of variables required for SIMD packing.
    const K_PACK: usize = log2_strict_usize(PackedF::WIDTH);

    /// Naive multilinear evaluation by materializing the full eq table.
    fn eval_reference<F: Field, EF: ExtensionField<F>>(evals: &[F], point: &[EF]) -> EF {
        let eq = Poly::new_from_point(point, EF::ONE);
        dot_product(eq.iter().copied(), evals.iter().copied())
    }

    /// Naive accumulation by materializing the full eq table.
    fn accumulate_reference(out: &mut [EF], point: &Point<EF>, scale: EF) {
        let eq = Poly::new_from_point(point.as_slice(), scale);
        out.iter_mut().zip(eq.iter()).for_each(|(o, &e)| *o += e);
    }

    // Eval: unpacked == packed == reference

    proptest! {
        #[test]
        fn prop_eval_base_matches_reference(k in 0usize..=14, seed in any::<u64>()) {
            let mut rng = SmallRng::seed_from_u64(seed);
            let poly = Poly::<F>::rand(&mut rng, k);
            let point = Point::<EF>::rand(&mut rng, k);
            // Compute the expected result via naive full-table evaluation.
            let expected = eval_reference(poly.as_slice(), point.as_slice());

            // Both scalar and packed construction must match the reference.
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
            // Convert to packed representation.
            let packed = poly.pack::<F, EF>();

            // Both packed and unpacked eq tables must produce the same result.
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

    // Accumulate: reference vs unpacked vs packed, scale in ctor vs argument

    proptest! {
        #[test]
        fn prop_accumulate_matches_reference(k in 0usize..=14, seed in any::<u64>()) {
            let mut rng = SmallRng::seed_from_u64(seed);
            let input = Poly::<EF>::rand(&mut rng, k);
            let point = Point::<EF>::rand(&mut rng, k);
            let alpha: EF = rng.random();

            // Reference: full eq table materialization.
            let mut expected = input.clone();
            accumulate_reference(expected.as_mut_slice(), &point, alpha);

            // Scale passed as argument, unpacked path.
            let mut out = input.clone();
            SplitEq::<F, EF>::new_unpacked(&point, EF::ONE)
                .accumulate_into(out.as_mut_slice(), Some(alpha));
            prop_assert_eq!(&expected, &out);

            // Scale baked into constructor, unpacked path.
            let mut out = input.clone();
            SplitEq::<F, EF>::new_unpacked(&point, alpha)
                .accumulate_into(out.as_mut_slice(), None);
            prop_assert_eq!(&expected, &out);

            // Scale as argument, packed path.
            let mut out = input.clone();
            SplitEq::<F, EF>::new_packed(&point, EF::ONE)
                .accumulate_into(out.as_mut_slice(), Some(alpha));
            prop_assert_eq!(&expected, &out);

            // Scale in constructor, packed path.
            let mut out = input;
            SplitEq::<F, EF>::new_packed(&point, alpha)
                .accumulate_into(out.as_mut_slice(), None);
            prop_assert_eq!(&expected, &out);
        }

        #[test]
        fn prop_accumulate_into_packed_matches_scalar(
            k in K_PACK..=14usize,
            seed in any::<u64>(),
        ) {
            let mut rng = SmallRng::seed_from_u64(seed);
            let input = Poly::<EF>::rand(&mut rng, k);
            let point = Point::<EF>::rand(&mut rng, k);
            let alpha: EF = rng.random();

            // Scalar reference via full eq table.
            let mut expected = input.clone();
            accumulate_reference(expected.as_mut_slice(), &point, alpha);

            // Scale as argument into packed output.
            let mut out = input.pack::<F, EF>();
            SplitEq::<F, EF>::new_packed(&point, EF::ONE)
                .accumulate_into_packed(out.as_mut_slice(), Some(alpha));
            prop_assert_eq!(&expected, &out.unpack::<F, EF>());

            // Scale in constructor into packed output.
            let mut out = input.pack::<F, EF>();
            SplitEq::<F, EF>::new_packed(&point, alpha)
                .accumulate_into_packed(out.as_mut_slice(), None);
            prop_assert_eq!(&expected, &out.unpack::<F, EF>());
        }
    }

    // Compress roundtrip: compress then eval == direct eval

    proptest! {
        #[test]
        fn prop_compress_prefix_roundtrip(
            k in 0usize..=14,
            point_k_ratio in 0.0f64..=1.0,
            seed in any::<u64>(),
        ) {
            // Split the point at a random position determined by the ratio.
            let point_k = ((k as f64) * point_k_ratio) as usize;
            let mut rng = SmallRng::seed_from_u64(seed);
            let poly = Poly::<F>::rand(&mut rng, k);
            let point = Point::<EF>::rand(&mut rng, k);
            // Ground truth: direct naive evaluation.
            let expected = eval_reference(poly.as_slice(), point.as_slice());

            // Split the point and create two factored tables.
            let (point_lo, point_hi) = point.split_at(point_k);
            let split_lo = SplitEq::<F, EF>::new_unpacked(&point_lo, EF::ONE);
            let split_hi = SplitEq::<F, EF>::new_unpacked(&point_hi, EF::ONE);

            // Compress prefix variables, then evaluate the remainder.
            let compressed = split_lo.compress_prefix(&poly);
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

            // Split at midpoint so both halves have enough vars for packing.
            let point_k = k / 2;
            prop_assume!(k >= point_k + K_PACK);

            let (point_lo, point_hi) = point.split_at(point_k);
            let split_lo = SplitEq::<F, EF>::new_packed(&point_lo, EF::ONE);
            let split_hi = SplitEq::<F, EF>::new_packed(&point_hi, EF::ONE);

            // Scalar compress then eval.
            let compressed = split_lo.compress_prefix(&poly);
            prop_assert_eq!(expected, split_hi.eval_ext(&compressed));

            // Packed compress, unpack, then eval.
            let compressed = split_lo.compress_prefix_to_packed(&poly).unpack::<F, EF>();
            prop_assert_eq!(expected, split_hi.eval_ext(&compressed));

            // Compress suffix variables, then evaluate in the other direction.
            let compressed = split_hi.compress_suffix(&poly);
            prop_assert_eq!(expected, split_lo.eval_ext(&compressed));
        }
    }

    // Edge cases

    #[test]
    fn test_eval_constant_poly() {
        let mut rng = SmallRng::seed_from_u64(42);
        for k in 0..=4 {
            let point = Point::<EF>::rand(&mut rng, k);
            // All evaluations are 2; eq(z,.) sums to 1, so the result should be 2.
            let poly = Poly::new(vec![F::TWO; 1 << k]);
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

        // Accumulating with zero scale should be a no-op.
        let mut out = input.clone();
        SplitEq::<F, EF>::new_packed(&point, EF::ZERO).accumulate_into(out.as_mut_slice(), None);
        assert_eq!(input, out);
    }

    #[test]
    fn test_compress_prefix_full_point_yields_scalar() {
        let mut rng = SmallRng::seed_from_u64(42);
        let k = 8;
        let poly = Poly::<F>::rand(&mut rng, k);
        let point = Point::<EF>::rand(&mut rng, k);
        let expected = eval_reference(poly.as_slice(), point.as_slice());

        // Compressing all variables should produce a 0-variable (scalar) result.
        let split = SplitEq::<F, EF>::new_unpacked(&point, EF::ONE);
        let compressed = split.compress_prefix(&poly);
        assert_eq!(compressed.num_variables(), 0);
        assert_eq!(compressed.as_constant().unwrap(), expected);
    }

    #[test]
    fn test_compress_suffix_no_vars_is_identity() {
        let mut rng = SmallRng::seed_from_u64(42);
        let k = 8;
        let poly = Poly::<F>::rand(&mut rng, k);
        // Zero-variable point: eq table is trivially 1.
        let point = Point::<EF>::rand(&mut rng, 0);

        // Compressing zero variables should return the polynomial lifted to the extension field.
        let split = SplitEq::<F, EF>::new_unpacked(&point, EF::ONE);
        let compressed = split.compress_suffix(&poly);
        assert_eq!(compressed.num_variables(), k);
        // Each element should equal its base-field counterpart promoted to the extension.
        for (c, &p) in compressed.iter().zip(poly.iter()) {
            assert_eq!(*c, EF::from(p));
        }
    }
}
