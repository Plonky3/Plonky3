use itertools::Itertools;
use p3_field::{ExtensionField, Field, PackedFieldExtension, PackedValue, dot_product};
use p3_util::log2_strict_usize;

use crate::evals::Poly;

/// Eq polynomial table, either as scalar extension elements or packed representation.
#[derive(Debug, Clone)]
pub(crate) enum EqMaybePacked<F: Field, EF: ExtensionField<F>> {
    Unpacked(Poly<EF>),
    Packed(Poly<<EF as ExtensionField<F>>::ExtensionPacking>),
}

impl<F: Field, EF: ExtensionField<F>> EqMaybePacked<F, EF> {
    /// Constructs an unpacked eq table.
    pub(super) fn new_unpacked(point: &crate::multilinear::Point<EF>) -> Self {
        Self::Unpacked(Poly::new_from_point(point.as_slice(), EF::ONE))
    }

    /// Constructs a packed eq table when possible, otherwise falls back to unpacked.
    pub(super) fn new_packed(point: &crate::multilinear::Point<EF>) -> Self {
        if point.num_vars() >= log2_strict_usize(F::Packing::WIDTH) {
            Self::Packed(Poly::new_packed_from_point(point.as_slice(), EF::ONE))
        } else {
            Self::Unpacked(Poly::new_from_point(point.as_slice(), EF::ONE))
        }
    }

    /// Returns the number of variables.
    pub(super) const fn num_vars(&self) -> usize {
        match self {
            Self::Unpacked(poly) => poly.num_vars(),
            Self::Packed(poly) => poly.num_vars() + log2_strict_usize(F::Packing::WIDTH),
        }
    }

    /// Number of scalar F elements per eq1 block (accounts for packing width).
    pub(super) const fn scalar_chunk_size(&self) -> usize {
        match self {
            Self::Unpacked(eq1) => eq1.num_evals(),
            Self::Packed(eq1) => eq1.num_evals() * F::Packing::WIDTH,
        }
    }

    // --- dot-product kernels (for eval) ---

    /// Inner product with a base-field chunk: `sum_i eq1[i] * chunk[i]`.
    pub(super) fn dot_with_base(&self, chunk: &[F]) -> EF {
        match self {
            Self::Unpacked(eq1) => dot_product(eq1.iter().copied(), chunk.iter().copied()),
            Self::Packed(eq1) => {
                let sum = dot_product(
                    eq1.iter().copied(),
                    F::Packing::pack_slice(chunk).iter().copied(),
                );
                EF::ExtensionPacking::to_ext_iter([sum]).sum()
            }
        }
    }

    /// Inner product with an extension-field chunk: `sum_i eq1[i] * chunk[i]`.
    pub(super) fn dot_with_ext(&self, chunk: &[EF]) -> EF {
        match self {
            Self::Unpacked(eq1) => dot_product(eq1.iter().copied(), chunk.iter().copied()),
            Self::Packed(eq1) => {
                let sum: EF::ExtensionPacking = dot_product(
                    chunk
                        .chunks(F::Packing::WIDTH)
                        .map(EF::ExtensionPacking::from_ext_slice),
                    eq1.iter().copied(),
                );
                EF::ExtensionPacking::to_ext_iter([sum]).sum()
            }
        }
    }

    /// Inner product with a packed extension-field chunk: `sum_i eq1[i] * chunk[i]`.
    pub(super) fn dot_with_ext_packed(&self, chunk: &[EF::ExtensionPacking]) -> EF {
        match self {
            Self::Packed(eq1) => {
                let sum = dot_product(chunk.iter().copied(), eq1.iter().copied());
                EF::ExtensionPacking::to_ext_iter([sum]).sum()
            }
            Self::Unpacked(_) => unreachable!(),
        }
    }

    // --- accumulate kernels ---

    /// `out[i] += weight * eq1[i]`, handling packed unpacking internally.
    pub(super) fn accumulate_scalar_into(&self, out: &mut [EF], weight: EF) {
        match self {
            Self::Unpacked(eq1) => {
                out.iter_mut()
                    .zip(eq1.iter())
                    .for_each(|(out, &w1)| *out += weight * w1);
            }
            Self::Packed(eq1) => {
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

    /// Like [`accumulate_scalar_into`](Self::accumulate_scalar_into), but into packed output.
    pub(super) fn accumulate_packed_into(&self, out: &mut [EF::ExtensionPacking], weight: EF) {
        match self {
            Self::Packed(eq1) => {
                out.iter_mut()
                    .zip(eq1.iter())
                    .for_each(|(out, &w1)| *out += w1 * weight);
            }
            _ => unreachable!(),
        }
    }

    // --- compress kernels ---

    /// Weighted accumulation for compress_lo: for each eq1 entry, accumulate
    /// `w0 * eq1[i] * chunk_row[j]` into `out[j]`.
    pub(super) fn compress_lo_into(&self, out: &mut [EF], chunk: &[F], w0: EF) {
        let size_inner = out.len();
        match self {
            Self::Unpacked(eq1) => {
                chunk
                    .chunks(size_inner)
                    .zip_eq(eq1.iter())
                    .for_each(|(chunk, &w1)| {
                        let w = w0 * w1;
                        out.iter_mut()
                            .zip_eq(chunk.iter())
                            .for_each(|(acc, &f)| *acc += w * f);
                    });
            }
            Self::Packed(eq1) => {
                chunk
                    .chunks(size_inner * F::Packing::WIDTH)
                    .zip_eq(eq1.iter())
                    .for_each(|(chunk, &w1)| {
                        chunk
                            .chunks(size_inner)
                            .zip_eq(EF::ExtensionPacking::to_ext_iter([w1 * w0]))
                            .for_each(|(chunk, w)| {
                                out.iter_mut()
                                    .zip_eq(chunk.iter())
                                    .for_each(|(acc, &f)| *acc += w * f);
                            });
                    });
            }
        }
    }

    /// Like [`compress_lo_into`](Self::compress_lo_into), but into packed output.
    pub(super) fn compress_lo_to_packed_into(
        &self,
        out: &mut [EF::ExtensionPacking],
        chunk: &[F],
        w0: EF,
    ) {
        match self {
            Self::Unpacked(eq1) => {
                let packed_inner = out.len();
                let chunk = F::Packing::pack_slice(chunk);
                chunk
                    .chunks(packed_inner)
                    .zip_eq(eq1.iter())
                    .for_each(|(chunk, &w1)| {
                        let w = EF::ExtensionPacking::from(w0 * w1);
                        out.iter_mut()
                            .zip_eq(chunk.iter())
                            .for_each(|(acc, &f)| *acc += w * f);
                    });
            }
            Self::Packed(eq1) => {
                // size_inner is in scalar F terms (= out.len() * WIDTH)
                let scalar_inner = out.len() * F::Packing::WIDTH;
                chunk
                    .chunks(scalar_inner * F::Packing::WIDTH)
                    .zip_eq(eq1.iter())
                    .for_each(|(chunk, &w1)| {
                        chunk
                            .chunks(scalar_inner)
                            .zip_eq(EF::ExtensionPacking::to_ext_iter([w1 * w0]))
                            .for_each(|(chunk, w)| {
                                let w = EF::ExtensionPacking::from(w);
                                let chunk = F::Packing::pack_slice(chunk);
                                out.iter_mut()
                                    .zip_eq(chunk.iter())
                                    .for_each(|(acc, &f)| *acc += w * f);
                            });
                    });
            }
        }
    }

    /// Per-row inner product for compress_hi: computes the dot product of one
    /// row of `poly` with eq1, weighted by eq0.
    pub(super) fn compress_hi_dot(&self, chunk: &[F], eq0: &Poly<EF>) -> EF {
        match self {
            Self::Unpacked(eq1) => chunk
                .chunks(eq1.num_evals())
                .zip_eq(eq0.iter())
                .map(|(chunk, &w0)| {
                    dot_product::<EF, _, _>(eq1.iter().copied(), chunk.iter().copied()) * w0
                })
                .sum::<EF>(),
            Self::Packed(eq1) => {
                let chunk = F::Packing::pack_slice(chunk);
                let sum = chunk
                    .chunks(eq1.num_evals())
                    .zip_eq(eq0.iter())
                    .map(|(chunk, &w0)| {
                        dot_product::<EF::ExtensionPacking, _, _>(
                            eq1.iter().copied(),
                            chunk.iter().copied(),
                        ) * w0
                    })
                    .sum::<EF::ExtensionPacking>();
                EF::ExtensionPacking::to_ext_iter([sum]).sum()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use p3_baby_bear::BabyBear;
    use p3_field::extension::BinomialExtensionField;
    use p3_field::{
        ExtensionField, Field, PackedFieldExtension, PackedValue, PrimeCharacteristicRing,
        dot_product,
    };
    use p3_util::log2_strict_usize;
    use proptest::prelude::*;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::EqMaybePacked;
    use crate::evals::Poly;
    use crate::multilinear::Point;

    type F = BabyBear;
    type PackedF = <F as Field>::Packing;
    type EF = BinomialExtensionField<F, 4>;
    type EP = <EF as ExtensionField<F>>::ExtensionPacking;

    const K_PACK: usize = log2_strict_usize(PackedF::WIDTH);

    /// Naive multilinear evaluation: `sum_x eq(point, x) * evals[x]`.
    fn eval_reference<F: Field, EF: ExtensionField<F>>(evals: &[F], point: &[EF]) -> EF {
        let eq = Poly::new_from_point(point, EF::ONE);
        dot_product(eq.iter().copied(), evals.iter().copied())
    }

    // -----------------------------------------------------------------------
    // Construction: new_packed falls back to Unpacked for small num_vars
    // -----------------------------------------------------------------------

    #[test]
    fn test_new_packed_fallback_to_unpacked() {
        let mut rng = SmallRng::seed_from_u64(0);
        for k in 0..K_PACK {
            let point = Point::<EF>::rand(&mut rng, k);
            let eq = EqMaybePacked::<F, EF>::new_packed(&point);
            assert!(matches!(eq, EqMaybePacked::Unpacked(_)));
            assert_eq!(eq.num_vars(), k);
        }
    }

    #[test]
    fn test_new_packed_uses_packed_when_enough_vars() {
        let mut rng = SmallRng::seed_from_u64(0);
        for k in K_PACK..=8 {
            let point = Point::<EF>::rand(&mut rng, k);
            let eq = EqMaybePacked::<F, EF>::new_packed(&point);
            assert!(matches!(eq, EqMaybePacked::Packed(_)));
            assert_eq!(eq.num_vars(), k);
        }
    }

    #[test]
    fn test_new_unpacked_always_unpacked() {
        let mut rng = SmallRng::seed_from_u64(0);
        for k in 0..=8 {
            let point = Point::<EF>::rand(&mut rng, k);
            let eq = EqMaybePacked::<F, EF>::new_unpacked(&point);
            assert!(matches!(eq, EqMaybePacked::Unpacked(_)));
            assert_eq!(eq.num_vars(), k);
        }
    }

    // -----------------------------------------------------------------------
    // scalar_chunk_size consistency
    // -----------------------------------------------------------------------

    #[test]
    fn test_scalar_chunk_size() {
        let mut rng = SmallRng::seed_from_u64(0);
        for k in 0..=8 {
            let point = Point::<EF>::rand(&mut rng, k);
            let unpacked = EqMaybePacked::<F, EF>::new_unpacked(&point);
            assert_eq!(unpacked.scalar_chunk_size(), 1 << k);

            let packed = EqMaybePacked::<F, EF>::new_packed(&point);
            assert_eq!(packed.scalar_chunk_size(), 1 << k);
        }
    }

    // -----------------------------------------------------------------------
    // dot_with_base: packed == unpacked for all k
    // -----------------------------------------------------------------------

    proptest! {
        #[test]
        fn prop_dot_with_base_packed_eq_unpacked(k in K_PACK..=10usize, seed in any::<u64>()) {
            let mut rng = SmallRng::seed_from_u64(seed);
            let point = Point::<EF>::rand(&mut rng, k);
            let chunk: Vec<F> = (0..1 << k).map(|_| rng.random()).collect();

            let unpacked = EqMaybePacked::<F, EF>::new_unpacked(&point);
            let packed = EqMaybePacked::<F, EF>::new_packed(&point);

            prop_assert_eq!(unpacked.dot_with_base(&chunk), packed.dot_with_base(&chunk));
        }

        #[test]
        fn prop_dot_with_base_matches_reference(k in 0usize..=10, seed in any::<u64>()) {
            let mut rng = SmallRng::seed_from_u64(seed);
            let point = Point::<EF>::rand(&mut rng, k);
            let chunk: Vec<F> = (0..1 << k).map(|_| rng.random()).collect();

            let expected = eval_reference(&chunk, point.as_slice());
            let unpacked = EqMaybePacked::<F, EF>::new_unpacked(&point);
            prop_assert_eq!(expected, unpacked.dot_with_base(&chunk));
        }
    }

    // -----------------------------------------------------------------------
    // dot_with_ext: packed == unpacked for all k
    // -----------------------------------------------------------------------

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

    // -----------------------------------------------------------------------
    // dot_with_ext_packed: matches scalar dot_with_ext
    // -----------------------------------------------------------------------

    proptest! {
        #[test]
        fn prop_dot_with_ext_packed_matches_scalar(k in K_PACK..=10usize, seed in any::<u64>()) {
            let mut rng = SmallRng::seed_from_u64(seed);
            let point = Point::<EF>::rand(&mut rng, k);
            let chunk_scalar: Vec<EF> = (0..1 << k).map(|_| rng.random()).collect();

            // Pack the chunk
            let chunk_packed: Vec<_> = chunk_scalar
                .chunks(PackedF::WIDTH)
                .map(EP::from_ext_slice)
                .collect();

            let packed = EqMaybePacked::<F, EF>::new_packed(&point);
            let unpacked = EqMaybePacked::<F, EF>::new_unpacked(&point);

            prop_assert_eq!(
                unpacked.dot_with_ext(&chunk_scalar),
                packed.dot_with_ext_packed(&chunk_packed),
            );
        }
    }

    // -----------------------------------------------------------------------
    // accumulate_scalar_into: packed == unpacked
    // -----------------------------------------------------------------------

    proptest! {
        #[test]
        fn prop_accumulate_scalar_packed_eq_unpacked(k in K_PACK..=10usize, seed in any::<u64>()) {
            let mut rng = SmallRng::seed_from_u64(seed);
            let point = Point::<EF>::rand(&mut rng, k);
            let weight: EF = rng.random();

            let unpacked = EqMaybePacked::<F, EF>::new_unpacked(&point);
            let packed = EqMaybePacked::<F, EF>::new_packed(&point);

            let mut out_unpacked = vec![EF::ZERO; 1 << k];
            let mut out_packed = vec![EF::ZERO; 1 << k];

            unpacked.accumulate_scalar_into(&mut out_unpacked, weight);
            packed.accumulate_scalar_into(&mut out_packed, weight);

            prop_assert_eq!(out_unpacked, out_packed);
        }

        #[test]
        fn prop_accumulate_scalar_matches_naive(k in 0usize..=10, seed in any::<u64>()) {
            let mut rng = SmallRng::seed_from_u64(seed);
            let point = Point::<EF>::rand(&mut rng, k);
            let weight: EF = rng.random();

            // Naive: weight * eq(point, x) for each x
            let eq = Poly::<EF>::new_from_point(point.as_slice(), weight);
            let expected: Vec<EF> = eq.iter().copied().collect();

            let unpacked = EqMaybePacked::<F, EF>::new_unpacked(&point);
            let mut out = vec![EF::ZERO; 1 << k];
            unpacked.accumulate_scalar_into(&mut out, weight);

            prop_assert_eq!(expected, out);
        }
    }

    // -----------------------------------------------------------------------
    // accumulate_packed_into: matches scalar version
    // -----------------------------------------------------------------------

    proptest! {
        #[test]
        fn prop_accumulate_packed_matches_scalar(k in K_PACK..=10usize, seed in any::<u64>()) {
            let mut rng = SmallRng::seed_from_u64(seed);
            let point = Point::<EF>::rand(&mut rng, k);
            let weight: EF = rng.random();

            let packed = EqMaybePacked::<F, EF>::new_packed(&point);

            // Scalar reference
            let mut out_scalar = vec![EF::ZERO; 1 << k];
            packed.accumulate_scalar_into(&mut out_scalar, weight);

            // Packed
            let mut out_packed = vec![EP::ZERO; (1 << k) / PackedF::WIDTH];
            packed.accumulate_packed_into(&mut out_packed, weight);
            let out_unpacked: Vec<EF> =
                <EP as PackedFieldExtension<F, EF>>::to_ext_iter(out_packed.iter().copied())
                    .collect();

            prop_assert_eq!(out_scalar, out_unpacked);
        }
    }

    // -----------------------------------------------------------------------
    // compress_lo_into: packed == unpacked
    // -----------------------------------------------------------------------

    proptest! {
        #[test]
        fn prop_compress_lo_packed_eq_unpacked(
            eq_k in K_PACK..=8usize,
            inner_k in 1usize..=4,
            seed in any::<u64>(),
        ) {
            let k = eq_k + inner_k;
            let mut rng = SmallRng::seed_from_u64(seed);
            let point = Point::<EF>::rand(&mut rng, eq_k);
            let chunk: Vec<F> = (0..1 << k).map(|_| rng.random()).collect();
            let w0: EF = rng.random();

            let unpacked = EqMaybePacked::<F, EF>::new_unpacked(&point);
            let packed = EqMaybePacked::<F, EF>::new_packed(&point);

            let mut out_u = vec![EF::ZERO; 1 << inner_k];
            let mut out_p = vec![EF::ZERO; 1 << inner_k];

            unpacked.compress_lo_into(&mut out_u, &chunk, w0);
            packed.compress_lo_into(&mut out_p, &chunk, w0);

            prop_assert_eq!(out_u, out_p);
        }
    }

    // -----------------------------------------------------------------------
    // compress_lo_to_packed_into: matches scalar compress_lo_into
    // -----------------------------------------------------------------------

    proptest! {
        #[test]
        fn prop_compress_lo_to_packed_matches_scalar(
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

            // Scalar reference
            let mut out_scalar = vec![EF::ZERO; 1 << inner_k];
            eq.compress_lo_into(&mut out_scalar, &chunk, w0);

            // Packed
            let packed_inner = (1 << inner_k) / PackedF::WIDTH;
            let mut out_packed = vec![EP::ZERO; packed_inner];
            eq.compress_lo_to_packed_into(&mut out_packed, &chunk, w0);
            let out_unpacked: Vec<EF> =
                <EP as PackedFieldExtension<F, EF>>::to_ext_iter(out_packed.iter().copied())
                    .collect();

            prop_assert_eq!(out_scalar, out_unpacked);
        }
    }

    // -----------------------------------------------------------------------
    // compress_hi_dot: packed == unpacked
    // -----------------------------------------------------------------------

    proptest! {
        #[test]
        fn prop_compress_hi_dot_packed_eq_unpacked(
            eq_k in K_PACK..=8usize,
            seed in any::<u64>(),
        ) {
            let mut rng = SmallRng::seed_from_u64(seed);
            let point = Point::<EF>::rand(&mut rng, eq_k);

            // eq0 and eq1 split at midpoint, like SplitEq does
            let (z0, z1) = point.split_at(eq_k / 2);
            let eq0 = Poly::<EF>::new_from_point(z0.as_slice(), EF::ONE);
            let chunk: Vec<F> = (0..1 << eq_k).map(|_| rng.random()).collect();

            let unpacked = EqMaybePacked::<F, EF>::new_unpacked(&z1);
            let packed = EqMaybePacked::<F, EF>::new_packed(&z1);

            prop_assert_eq!(
                unpacked.compress_hi_dot(&chunk, &eq0),
                packed.compress_hi_dot(&chunk, &eq0),
            );
        }

        #[test]
        fn prop_compress_hi_dot_matches_reference(
            eq_k in 0usize..=10,
            seed in any::<u64>(),
        ) {
            let mut rng = SmallRng::seed_from_u64(seed);
            let point = Point::<EF>::rand(&mut rng, eq_k);
            let chunk: Vec<F> = (0..1 << eq_k).map(|_| rng.random()).collect();

            // compress_hi_dot on the full row should equal eval_reference
            let expected = eval_reference(&chunk, point.as_slice());

            let (z0, z1) = point.split_at(eq_k / 2);
            let eq0 = Poly::<EF>::new_from_point(z0.as_slice(), EF::ONE);
            let unpacked = EqMaybePacked::<F, EF>::new_unpacked(&z1);

            prop_assert_eq!(expected, unpacked.compress_hi_dot(&chunk, &eq0));
        }
    }

    // -----------------------------------------------------------------------
    // Edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn test_zero_vars() {
        let point = Point::<EF>::rand(&mut SmallRng::seed_from_u64(0), 0);
        let eq = EqMaybePacked::<F, EF>::new_unpacked(&point);
        assert_eq!(eq.num_vars(), 0);
        assert_eq!(eq.scalar_chunk_size(), 1);
        assert_eq!(eq.dot_with_base(&[F::TWO]), EF::TWO);
        assert_eq!(eq.dot_with_ext(&[EF::TWO]), EF::TWO);
    }

    #[test]
    fn test_accumulate_zero_weight() {
        let mut rng = SmallRng::seed_from_u64(42);
        let point = Point::<EF>::rand(&mut rng, 4);
        let eq = EqMaybePacked::<F, EF>::new_unpacked(&point);
        let mut out = vec![EF::ONE; 1 << 4];
        eq.accumulate_scalar_into(&mut out, EF::ZERO);
        // Adding zero weight should leave everything as ONE
        assert!(out.iter().all(|&v| v == EF::ONE));
    }
}
