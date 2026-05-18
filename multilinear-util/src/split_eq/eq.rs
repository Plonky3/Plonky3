//! Equality polynomial table with optional SIMD packing.
//!
//! Stores evaluations of eq(z, .) over the boolean hypercube, either as:
//! - scalar extension-field elements or
//! - SIMD-packed extension-field elements.
//!
//! # Mathematical Background
//!
//! The equality polynomial for a point z in F^k is:
//!
//! ```text
//! eq(z, x) = prod_{i=0}^{k-1} (z_i * x_i + (1 - z_i) * (1 - x_i))
//! ```
//!
//! Evaluating eq(z, .) over all x in {0,1}^k produces a table of 2^k values.

use itertools::Itertools;
use p3_field::{ExtensionField, Field, PackedFieldExtension, PackedValue, dot_product};
use p3_util::log2_strict_usize;

use super::packed_kernel::{compress_hi_dot_packed, compress_prefix_to_packed_packed};
use crate::point::Point;
use crate::poly::Poly;

/// Equality polynomial table in either scalar or SIMD-packed form.
///
/// # Variants
///
/// - Scalar: stores 2^k extension-field elements directly.
/// - Packed: stores 2^{k - log_2(W)} packed elements, each holding W lanes.
///   Only used when k >= log_2(W), where W is the SIMD width.
///
/// All kernel methods dispatch internally on the variant, so callers do not need to match.
#[derive(Debug, Clone)]
pub enum EqMaybePacked<F: Field, EF: ExtensionField<F>> {
    /// Scalar extension-field evaluations, one per hypercube point.
    Unpacked(Poly<EF>),
    /// SIMD-packed evaluations, each element holds W consecutive lanes.
    Packed(Poly<<EF as ExtensionField<F>>::ExtensionPacking>),
}

impl<F: Field, EF: ExtensionField<F>> EqMaybePacked<F, EF> {
    /// Builds a scalar (unpacked) equality table from a multilinear point.
    ///
    /// Always produces the scalar variant regardless of the number of variables.
    pub(super) fn new_unpacked(point: &Point<EF>) -> Self {
        // Materialize eq(point, .) with unit scale into a scalar polynomial.
        Self::Unpacked(Poly::new_from_point(point.as_slice(), EF::ONE))
    }

    /// Builds a SIMD-packed equality table when the point has enough variables.
    ///
    /// Falls back to scalar if the number of variables is less than log_2(W),
    /// since there would be fewer evaluations than SIMD lanes.
    pub(super) fn new_packed(point: &Point<EF>) -> Self {
        // Check whether there are enough variables to fill at least one packed element.
        if point.num_variables() >= log2_strict_usize(F::Packing::WIDTH) {
            Self::Packed(Poly::new_packed_from_point(point.as_slice(), EF::ONE))
        } else {
            // Not enough evaluations to pack; fall back to scalar.
            Self::Unpacked(Poly::new_from_point(point.as_slice(), EF::ONE))
        }
    }

    /// Total number of boolean variables represented by this table.
    ///
    /// For packed tables, this accounts for the log_2(W) variables
    /// absorbed into each SIMD lane.
    pub const fn num_variables(&self) -> usize {
        match self {
            Self::Unpacked(poly) => poly.num_variables(),
            // Packed polynomial has k - log_2(W) stored entries,
            // but represents k total variables.
            Self::Packed(poly) => poly.num_variables() + log2_strict_usize(F::Packing::WIDTH),
        }
    }

    /// Number of scalar base-field elements per eq1 block.
    ///
    /// - For unpacked: 2^k scalar elements.
    /// - For packed: num_packed_entries * W scalar elements.
    ///
    /// This is used to determine chunk sizes when iterating over
    /// a polynomial paired with this eq table.
    pub const fn scalar_chunk_size(&self) -> usize {
        match self {
            Self::Unpacked(eq1) => eq1.num_evals(),
            Self::Packed(eq1) => eq1.num_evals() * F::Packing::WIDTH,
        }
    }

    /// Inner product of this eq table with a base-field slice.
    ///
    /// Computes sum_{i} eq1[i] * chunk[i].
    ///
    /// For packed tables, reinterprets the input slice as packed elements
    /// and reduces the SIMD lanes via horizontal sum at the end.
    pub(super) fn dot_with_base(&self, chunk: &[F]) -> EF {
        match self {
            // Scalar path: direct element-wise dot product.
            Self::Unpacked(eq1) => dot_product(eq1.iter().copied(), chunk.iter().copied()),
            Self::Packed(eq1) => {
                // Reinterpret the flat scalar slice as packed SIMD elements.
                // Compute packed dot product, then reduce lanes to a single scalar.
                let sum = dot_product(
                    eq1.iter().copied(),
                    F::Packing::pack_slice(chunk).iter().copied(),
                );
                // Horizontal reduction: sum the W lanes of each packed result element.
                EF::ExtensionPacking::to_ext_iter([sum]).sum()
            }
        }
    }

    /// Inner product of this eq table with an extension-field slice.
    ///
    /// Computes sum_{i} eq1[i] * chunk[i].
    ///
    /// For packed tables, groups the input into W-element sub-slices,
    /// converts each group to a packed element, then reduces at the end.
    pub(super) fn dot_with_ext(&self, chunk: &[EF]) -> EF {
        match self {
            // Scalar path: direct element-wise dot product.
            Self::Unpacked(eq1) => dot_product(eq1.iter().copied(), chunk.iter().copied()),
            Self::Packed(eq1) => {
                // Group W consecutive extension-field elements into packed form,
                // then dot product with the packed eq table.
                let sum: EF::ExtensionPacking = dot_product(
                    chunk
                        .chunks(F::Packing::WIDTH)
                        .map(EF::ExtensionPacking::from_ext_slice),
                    eq1.iter().copied(),
                );
                // Horizontal reduction across SIMD lanes.
                EF::ExtensionPacking::to_ext_iter([sum]).sum()
            }
        }
    }

    /// Inner product of this eq table with a pre-packed extension-field slice.
    ///
    /// Computes sum_{i} eq1[i] * chunk[i] where both sides are already packed.
    ///
    /// # Panics
    ///
    /// Panics (via unreachable) if the table is in unpacked form.
    pub(super) fn dot_with_ext_packed(&self, chunk: &[EF::ExtensionPacking]) -> EF {
        match self {
            Self::Packed(eq1) => {
                // Both sides are packed; direct dot product then reduce.
                let sum = dot_product(chunk.iter().copied(), eq1.iter().copied());
                EF::ExtensionPacking::to_ext_iter([sum]).sum()
            }
            Self::Unpacked(_) => unreachable!(),
        }
    }

    /// Adds weight * eq1[i] to each element of a scalar output buffer.
    ///
    /// ```text
    /// out[i] += weight * eq1[i]   for all i
    /// ```
    ///
    /// For packed tables, unpacks each SIMD element into W scalar lanes
    /// before accumulating into the output.
    pub(super) fn accumulate_scalar_into(&self, out: &mut [EF], weight: EF) {
        match self {
            Self::Unpacked(eq1) => {
                // Scalar path: direct weighted accumulation.
                out.iter_mut()
                    .zip(eq1.iter())
                    .for_each(|(out, &w1)| *out += weight * w1);
            }
            Self::Packed(eq1) => {
                // Unpack each SIMD element into W scalar lanes,
                // then accumulate weight * lane_value into the output.
                out.chunks_mut(F::Packing::WIDTH)
                    .zip(eq1.iter())
                    .for_each(|(out, &w1)| {
                        out.iter_mut()
                            .zip_eq(EF::ExtensionPacking::to_ext_iter([w1]))
                            .for_each(|(out, w1)| *out += w1 * weight);
                    });
            }
        }
    }

    /// Adds weight * eq1[i] to each element of a packed output buffer.
    ///
    /// ```text
    /// out[i] += weight * eq1[i]   for all i (packed elements)
    /// ```
    ///
    /// # Panics
    ///
    /// Panics (via unreachable) if the table is in unpacked form.
    pub fn accumulate_packed_into(&self, out: &mut [EF::ExtensionPacking], weight: EF) {
        match self {
            Self::Packed(eq1) => {
                // Both output and eq table are packed; direct accumulation.
                out.iter_mut()
                    .zip(eq1.iter())
                    .for_each(|(out, &w1)| *out += w1 * weight);
            }
            _ => unreachable!(),
        }
    }

    /// Weighted accumulation for prefix-variable compression.
    ///
    /// For each eq1 entry, accumulates w0 * eq1[i] * chunk_row[j] into out[j].
    ///
    /// The input chunk contains data for all eq1 entries interleaved
    /// with the inner (output) dimension. The output buffer has one entry
    /// per inner coordinate.
    ///
    /// # Arguments
    ///
    /// - out: accumulator buffer of size 2^{k_inner}
    /// - chunk: slice of base-field evaluations for one eq0 entry
    /// - w0: the eq0 weight to multiply by
    pub(super) fn compress_prefix_into(&self, out: &mut [EF], chunk: &[F], w0: EF) {
        let size_inner = out.len();
        match self {
            Self::Unpacked(eq1) => {
                // Iterate over eq1 entries; each owns size_inner base-field elements.
                chunk
                    .chunks(size_inner)
                    .zip_eq(eq1.iter())
                    .for_each(|(chunk, &w1)| {
                        // Combined weight for this eq0 * eq1 entry.
                        let w = w0 * w1;
                        // Accumulate weighted base-field values into the output.
                        out.iter_mut()
                            .zip_eq(chunk.iter())
                            .for_each(|(acc, &f)| *acc += w * f);
                    });
            }
            Self::Packed(eq1) => {
                // Each packed eq1 entry covers W scalar eq1 entries.
                // Outer chunk per packed entry: size_inner * W scalars.
                chunk
                    .chunks(size_inner * F::Packing::WIDTH)
                    .zip_eq(eq1.iter())
                    .for_each(|(chunk, &w1)| {
                        // Unpack the packed weight into W scalar lane weights.
                        // Inner chunk per lane: size_inner scalars.
                        chunk
                            .chunks(size_inner)
                            .zip_eq(EF::ExtensionPacking::to_ext_iter([w1 * w0]))
                            .for_each(|(chunk, w)| {
                                // Accumulate weighted base-field values for this lane.
                                out.iter_mut()
                                    .zip_eq(chunk.iter())
                                    .for_each(|(acc, &f)| *acc += w * f);
                            });
                    });
            }
        }
    }

    /// Weighted accumulation for prefix-variable compression into packed output.
    ///
    /// Same operation as the scalar compression kernel,
    /// but writes into a packed extension-field buffer.
    ///
    /// # Arguments
    ///
    /// - out: packed accumulator buffer of size 2^{k_inner} / W
    /// - chunk: slice of base-field evaluations for one eq0 entry
    /// - w0: the eq0 weight to multiply by
    pub(super) fn compress_prefix_to_packed_into(
        &self,
        out: &mut [EF::ExtensionPacking],
        chunk: &[F],
        w0: EF,
    ) {
        match self {
            Self::Unpacked(eq1) => {
                let packed_inner = out.len();
                // Pack the entire scalar chunk into SIMD elements.
                let chunk = F::Packing::pack_slice(chunk);
                // Iterate over eq1 entries; each owns packed_inner packed elements.
                chunk
                    .chunks(packed_inner)
                    .zip_eq(eq1.iter())
                    .for_each(|(chunk, &w1)| {
                        // Broadcast the combined weight into packed form.
                        let w = EF::ExtensionPacking::from(w0 * w1);
                        // Accumulate packed weighted values.
                        out.iter_mut()
                            .zip_eq(chunk.iter())
                            .for_each(|(acc, &f)| *acc += w * f);
                    });
            }
            // Packed path: delegate to the SIMD kernel.
            //     - basis split into D per-coefficient mixed dot products,
            //     - one Montgomery reduction per CHUNK multiplies,
            //     - tiled inner loop for ILP.
            Self::Packed(eq1) => {
                compress_prefix_to_packed_packed::<F, EF>(out, eq1.as_slice(), chunk, w0);
            }
        }
    }

    /// Inner product of one polynomial row with the eq table, weighted by eq0.
    ///
    /// Computes:
    /// ```text
    /// sum_{j} eq0[j] * (sum_{i} eq1[i] * chunk[j * |eq1| + i])
    /// ```
    ///
    /// This is the kernel for suffix-variable compression:
    /// each output element is a full dot product of one row against the split eq tables.
    ///
    /// # Arguments
    ///
    /// - chunk: base-field slice of size 2^{num_variables}, representing one output row
    /// - eq0: the prefix-half eq table weights
    pub(super) fn compress_suffix_dot(&self, chunk: &[F], eq0: &Poly<EF>) -> EF {
        match self {
            Self::Unpacked(eq1) => {
                // Group the chunk by eq1 size, pair with eq0 weights.
                // Inner dot product over eq1, then weight by eq0 and sum.
                chunk
                    .chunks(eq1.num_evals())
                    .zip_eq(eq0.iter())
                    .map(|(chunk, &w0)| {
                        dot_product::<EF, _, _>(eq1.iter().copied(), chunk.iter().copied()) * w0
                    })
                    .sum::<EF>()
            }
            // Packed path: delegate to the SIMD kernel.
            //     - basis split into four per-coefficient dot products,
            //     - one Montgomery reduction per four multiplies,
            //     - interleaved inner loop for ILP.
            Self::Packed(eq1) => {
                compress_hi_dot_packed::<F, EF>(eq1.as_slice(), chunk, eq0.as_slice())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;
    use p3_field::extension::BinomialExtensionField;
    use proptest::prelude::*;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::*;

    type F = BabyBear;
    type PackedF = <F as Field>::Packing;
    type EF = BinomialExtensionField<F, 4>;
    type EP = <EF as ExtensionField<F>>::ExtensionPacking;

    /// Minimum number of variables required for SIMD packing.
    const K_PACK: usize = log2_strict_usize(PackedF::WIDTH);

    /// Naive multilinear evaluation by materializing the full eq table.
    fn eval_reference<F: Field, EF: ExtensionField<F>>(evals: &[F], point: &[EF]) -> EF {
        let eq = Poly::new_from_point(point, EF::ONE);
        dot_product(eq.iter().copied(), evals.iter().copied())
    }

    // Construction

    #[test]
    fn test_new_packed_fallback_to_unpacked() {
        let mut rng = SmallRng::seed_from_u64(0);
        // For variable counts below the packing threshold, packed must fall back.
        #[allow(clippy::reversed_empty_ranges)]
        for k in 0..K_PACK {
            let point = Point::<EF>::rand(&mut rng, k);
            let eq = EqMaybePacked::<F, EF>::new_packed(&point);
            // Should be scalar despite requesting packed.
            assert!(matches!(eq, EqMaybePacked::Unpacked(_)));
            assert_eq!(eq.num_variables(), k);
        }
    }

    #[test]
    fn test_new_packed_uses_packed_when_enough_vars() {
        let mut rng = SmallRng::seed_from_u64(0);
        // At or above the threshold, packed construction should succeed.
        for k in K_PACK..=8 {
            let point = Point::<EF>::rand(&mut rng, k);
            let eq = EqMaybePacked::<F, EF>::new_packed(&point);
            // Should be in packed form.
            assert!(matches!(eq, EqMaybePacked::Packed(_)));
            assert_eq!(eq.num_variables(), k);
        }
    }

    #[test]
    fn test_new_unpacked_always_unpacked() {
        let mut rng = SmallRng::seed_from_u64(0);
        // Explicit unpacked construction never produces the packed variant.
        for k in 0..=8 {
            let point = Point::<EF>::rand(&mut rng, k);
            let eq = EqMaybePacked::<F, EF>::new_unpacked(&point);
            assert!(matches!(eq, EqMaybePacked::Unpacked(_)));
            assert_eq!(eq.num_variables(), k);
        }
    }

    // Chunk size consistency

    #[test]
    fn test_scalar_chunk_size() {
        let mut rng = SmallRng::seed_from_u64(0);
        // Both variants must report the same scalar chunk size for a given k.
        for k in 0..=8 {
            let point = Point::<EF>::rand(&mut rng, k);
            let unpacked = EqMaybePacked::<F, EF>::new_unpacked(&point);
            // Scalar chunk size is always 2^k regardless of packing.
            assert_eq!(unpacked.scalar_chunk_size(), 1 << k);

            let packed = EqMaybePacked::<F, EF>::new_packed(&point);
            assert_eq!(packed.scalar_chunk_size(), 1 << k);
        }
    }

    // Dot product with base-field input

    proptest! {
        #[test]
        fn prop_dot_with_base_packed_eq_unpacked(k in K_PACK..=10usize, seed in any::<u64>()) {
            let mut rng = SmallRng::seed_from_u64(seed);
            let point = Point::<EF>::rand(&mut rng, k);
            let chunk: Vec<F> = (0..1 << k).map(|_| rng.random()).collect();

            // Both representations must produce the same dot product.
            let unpacked = EqMaybePacked::<F, EF>::new_unpacked(&point);
            let packed = EqMaybePacked::<F, EF>::new_packed(&point);

            prop_assert_eq!(unpacked.dot_with_base(&chunk), packed.dot_with_base(&chunk));
        }

        #[test]
        fn prop_dot_with_base_matches_reference(k in 0usize..=10, seed in any::<u64>()) {
            let mut rng = SmallRng::seed_from_u64(seed);
            let point = Point::<EF>::rand(&mut rng, k);
            let chunk: Vec<F> = (0..1 << k).map(|_| rng.random()).collect();

            // Must equal the naive full-table evaluation.
            let expected = eval_reference(&chunk, point.as_slice());
            let unpacked = EqMaybePacked::<F, EF>::new_unpacked(&point);
            prop_assert_eq!(expected, unpacked.dot_with_base(&chunk));
        }
    }

    // Dot product with extension-field input

    proptest! {
        #[test]
        fn prop_dot_with_ext_packed_eq_unpacked(k in K_PACK..=10usize, seed in any::<u64>()) {
            let mut rng = SmallRng::seed_from_u64(seed);
            let point = Point::<EF>::rand(&mut rng, k);
            let chunk: Vec<EF> = (0..1 << k).map(|_| rng.random()).collect();

            let unpacked = EqMaybePacked::<F, EF>::new_unpacked(&point);
            let packed = EqMaybePacked::<F, EF>::new_packed(&point);

            prop_assert_eq!(unpacked.dot_with_ext(&chunk), packed.dot_with_ext(&chunk));
        }

        #[test]
        fn prop_dot_with_ext_matches_reference(k in 0usize..=10, seed in any::<u64>()) {
            let mut rng = SmallRng::seed_from_u64(seed);
            let point = Point::<EF>::rand(&mut rng, k);
            let chunk: Vec<EF> = (0..1 << k).map(|_| rng.random()).collect();

            let expected: EF = eval_reference(&chunk, point.as_slice());
            let unpacked = EqMaybePacked::<F, EF>::new_unpacked(&point);
            prop_assert_eq!(expected, unpacked.dot_with_ext(&chunk));
        }
    }

    // Dot product with pre-packed extension-field input

    proptest! {
        #[test]
        fn prop_dot_with_ext_packed_matches_scalar(k in K_PACK..=10usize, seed in any::<u64>()) {
            let mut rng = SmallRng::seed_from_u64(seed);
            let point = Point::<EF>::rand(&mut rng, k);
            let chunk_scalar: Vec<EF> = (0..1 << k).map(|_| rng.random()).collect();

            // Pack the scalar chunk into SIMD elements.
            let chunk_packed: Vec<_> = chunk_scalar
                .chunks(PackedF::WIDTH)
                .map(EP::from_ext_slice)
                .collect();

            // Packed dot product must match the scalar version.
            let packed = EqMaybePacked::<F, EF>::new_packed(&point);
            let unpacked = EqMaybePacked::<F, EF>::new_unpacked(&point);

            prop_assert_eq!(
                unpacked.dot_with_ext(&chunk_scalar),
                packed.dot_with_ext_packed(&chunk_packed),
            );
        }
    }

    // Scalar accumulation

    proptest! {
        #[test]
        fn prop_accumulate_scalar_packed_eq_unpacked(k in K_PACK..=10usize, seed in any::<u64>()) {
            let mut rng = SmallRng::seed_from_u64(seed);
            let point = Point::<EF>::rand(&mut rng, k);
            let weight: EF = rng.random();

            let unpacked = EqMaybePacked::<F, EF>::new_unpacked(&point);
            let packed = EqMaybePacked::<F, EF>::new_packed(&point);

            // Both variants must produce identical output buffers.
            let mut out_unpacked = EF::zero_vec(1 << k);
            let mut out_packed = EF::zero_vec(1 << k);

            unpacked.accumulate_scalar_into(&mut out_unpacked, weight);
            packed.accumulate_scalar_into(&mut out_packed, weight);

            prop_assert_eq!(out_unpacked, out_packed);
        }

        #[test]
        fn prop_accumulate_scalar_matches_naive(k in 0usize..=10, seed in any::<u64>()) {
            let mut rng = SmallRng::seed_from_u64(seed);
            let point = Point::<EF>::rand(&mut rng, k);
            let weight: EF = rng.random();

            // Naive: materialize the full weighted eq table.
            let eq = Poly::<EF>::new_from_point(point.as_slice(), weight);
            let expected: Vec<EF> = eq.iter().copied().collect();

            // Accumulation into a zero buffer should match.
            let unpacked = EqMaybePacked::<F, EF>::new_unpacked(&point);
            let mut out = EF::zero_vec(1 << k);
            unpacked.accumulate_scalar_into(&mut out, weight);

            prop_assert_eq!(expected, out);
        }
    }

    // Packed accumulation

    proptest! {
        #[test]
        fn prop_accumulate_packed_matches_scalar(k in K_PACK..=10usize, seed in any::<u64>()) {
            let mut rng = SmallRng::seed_from_u64(seed);
            let point = Point::<EF>::rand(&mut rng, k);
            let weight: EF = rng.random();

            let packed = EqMaybePacked::<F, EF>::new_packed(&point);

            // Scalar reference accumulation.
            let mut out_scalar = EF::zero_vec(1 << k);
            packed.accumulate_scalar_into(&mut out_scalar, weight);

            // Packed accumulation, then unpack for comparison.
            let mut out_packed = EP::zero_vec((1 << k) / PackedF::WIDTH);
            packed.accumulate_packed_into(&mut out_packed, weight);
            let out_unpacked: Vec<EF> =
                <EP as PackedFieldExtension<F, EF>>::to_ext_iter(out_packed.iter().copied())
                    .collect();

            prop_assert_eq!(out_scalar, out_unpacked);
        }
    }

    // Prefix-variable compression kernel

    proptest! {
        #[test]
        fn prop_compress_prefix_packed_eq_unpacked(
            eq_k in K_PACK..=8usize,
            inner_k in 1usize..=4,
            seed in any::<u64>(),
        ) {
            // Total variables = eq variables + inner (remaining) variables.
            let k = eq_k + inner_k;
            let mut rng = SmallRng::seed_from_u64(seed);
            let point = Point::<EF>::rand(&mut rng, eq_k);
            let chunk: Vec<F> = (0..1 << k).map(|_| rng.random()).collect();
            let w0: EF = rng.random();

            let unpacked = EqMaybePacked::<F, EF>::new_unpacked(&point);
            let packed = EqMaybePacked::<F, EF>::new_packed(&point);

            // Accumulate into separate buffers; both must match.
            let mut out_u = EF::zero_vec(1 << inner_k);
            let mut out_p = EF::zero_vec(1 << inner_k);

            unpacked.compress_prefix_into(&mut out_u, &chunk, w0);
            packed.compress_prefix_into(&mut out_p, &chunk, w0);

            prop_assert_eq!(out_u, out_p);
        }
    }

    // Packed prefix-variable compression kernel

    proptest! {
        #[test]
        fn prop_compress_prefix_to_packed_matches_scalar(
            eq_k in K_PACK..=8usize,
            inner_k in K_PACK..=4usize,
            seed in any::<u64>(),
        ) {
            let k = eq_k + inner_k;
            let mut rng = SmallRng::seed_from_u64(seed);
            let point = Point::<EF>::rand(&mut rng, eq_k);
            let chunk: Vec<F> = (0..1 << k).map(|_| rng.random()).collect();
            let w0: EF = rng.random();

            let eq = EqMaybePacked::<F, EF>::new_packed(&point);

            // Scalar reference compression.
            let mut out_scalar = EF::zero_vec(1 << inner_k);
            eq.compress_prefix_into(&mut out_scalar, &chunk, w0);

            // Packed compression, then unpack for comparison.
            let packed_inner = (1 << inner_k) / PackedF::WIDTH;
            let mut out_packed = EP::zero_vec(packed_inner);
            eq.compress_prefix_to_packed_into(&mut out_packed, &chunk, w0);
            let out_unpacked: Vec<EF> =
                <EP as PackedFieldExtension<F, EF>>::to_ext_iter(out_packed.iter().copied())
                    .collect();

            prop_assert_eq!(out_scalar, out_unpacked);
        }
    }

    // Suffix-variable compression kernel

    proptest! {
        #[test]
        fn prop_compress_suffix_dot_packed_eq_unpacked(
            eq_k in K_PACK..=8usize,
            seed in any::<u64>(),
        ) {
            let mut rng = SmallRng::seed_from_u64(seed);
            let point = Point::<EF>::rand(&mut rng, eq_k);

            // Split at midpoint, mirroring the factored structure.
            let (z0, z1) = point.split_at(eq_k / 2);
            let eq0 = Poly::<EF>::new_from_point(z0.as_slice(), EF::ONE);
            let chunk: Vec<F> = (0..1 << eq_k).map(|_| rng.random()).collect();

            // Both representations must produce the same weighted dot product.
            let unpacked = EqMaybePacked::<F, EF>::new_unpacked(&z1);
            let packed = EqMaybePacked::<F, EF>::new_packed(&z1);

            prop_assert_eq!(
                unpacked.compress_suffix_dot(&chunk, &eq0),
                packed.compress_suffix_dot(&chunk, &eq0),
            );
        }

        #[test]
        fn prop_compress_suffix_dot_matches_reference(
            eq_k in 0usize..=10,
            seed in any::<u64>(),
        ) {
            let mut rng = SmallRng::seed_from_u64(seed);
            let point = Point::<EF>::rand(&mut rng, eq_k);
            let chunk: Vec<F> = (0..1 << eq_k).map(|_| rng.random()).collect();

            // A full-row dot against the split eq must equal the naive evaluation.
            let expected = eval_reference(&chunk, point.as_slice());

            let (z0, z1) = point.split_at(eq_k / 2);
            let eq0 = Poly::<EF>::new_from_point(z0.as_slice(), EF::ONE);
            let unpacked = EqMaybePacked::<F, EF>::new_unpacked(&z1);

            prop_assert_eq!(expected, unpacked.compress_suffix_dot(&chunk, &eq0));
        }
    }

    // Edge cases

    #[test]
    fn test_zero_vars() {
        // A zero-variable eq table has exactly one entry (the empty product = 1).
        let point = Point::<EF>::rand(&mut SmallRng::seed_from_u64(0), 0);
        let eq = EqMaybePacked::<F, EF>::new_unpacked(&point);
        assert_eq!(eq.num_variables(), 0);
        assert_eq!(eq.scalar_chunk_size(), 1);
        // Dot with a single element should return that element times 1.
        assert_eq!(eq.dot_with_base(&[F::TWO]), EF::TWO);
        assert_eq!(eq.dot_with_ext(&[EF::TWO]), EF::TWO);
    }

    #[test]
    fn test_accumulate_zero_weight() {
        let mut rng = SmallRng::seed_from_u64(42);
        let point = Point::<EF>::rand(&mut rng, 4);
        let eq = EqMaybePacked::<F, EF>::new_unpacked(&point);
        // Initialize output to all ones.
        let mut out = vec![EF::ONE; 1 << 4];
        // Zero weight should not modify the buffer.
        eq.accumulate_scalar_into(&mut out, EF::ZERO);
        assert!(out.iter().all(|&v| v == EF::ONE));
    }
}
