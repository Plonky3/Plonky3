//! Delayed-reduction variant of `compress_hi_dot` for the Packed case.
//!
//! For each output row, decomposes the inner N-element `PackedEF × F::Packing`
//! dot product into `D` separate `F::Packing × F::Packing` dot products, one
//! per extension-field basis coefficient. Each per-coefficient dot product is
//! driven by `F::Packing::dot_product::<CHUNK>`, a compile-time-sized primitive
//! that accumulates u64 products via `dot_product_4` (VPMULUDQ + VPADDQ on x86,
//! analogous intrinsics on NEON) and applies a single Montgomery reduction per
//! 4-wide accumulation group.
//!
//! Relative to the eager `dot_product<PackedEF, _, _>` baseline this saves
//! roughly 4× Monty reductions across the inner loop, stacked with an
//! independent-accumulator ILP gain from the per-coefficient decomposition.

use alloc::vec;
use alloc::vec::Vec;

use p3_field::{
    BasedVectorSpace, ExtensionField, Field, PackedFieldExtension, PackedValue,
    PrimeCharacteristicRing,
};

use crate::poly::Poly;

/// Delayed-reduction `compress_hi_dot` for the Packed variant.
///
/// Works for any `eq1_packed` length. The inner loop uses a compile-time
/// `CHUNK` of 4 so that `F::Packing::dot_product::<CHUNK>` always resolves to
/// the hand-tuned `dot_product_4` primitive on each SIMD target. A short
/// scalar tail handles the `n mod CHUNK` residue.
///
/// Restricted to extension-field dimension 4; other dimensions fall back to
/// the eager packed dot-product path.
pub(super) fn compress_hi_dot_delayed_packed<F, EF>(
    eq1_packed: &[<EF as ExtensionField<F>>::ExtensionPacking],
    chunk: &[F],
    eq0: &Poly<EF>,
) -> EF
where
    F: Field,
    EF: ExtensionField<F>,
{
    let n = eq1_packed.len();
    if n == 0 {
        return EF::ZERO;
    }
    if <EF as BasedVectorSpace<F>>::DIMENSION != 4 {
        return fallback::<F, EF>(eq1_packed, chunk, eq0);
    }

    // `dot_product_4` is the delayed-reduction SIMD primitive on every supported
    // target. CHUNK = 4 hits it directly for all N >= 4; smaller residues fall
    // into the scalar tail. CHUNK could be raised to 8/16 to amortize outer-loop
    // overhead further, but 4 is the lowest-risk default that covers AVX-512,
    // AVX2, NEON, and scalar equivalently.
    const CHUNK: usize = 4;

    let chunk_packed = F::Packing::pack_slice(chunk);

    // Transpose `eq1_packed` into four per-coefficient slices sharing one
    // allocation: layout is `[c0 | c1 | c2 | c3]`, each sub-block of length n.
    // One heap round-trip per `compress_hi_dot` call, amortized across `eq0`
    // outer iterations. Separate per-coefficient slices give `dot_product_4`
    // contiguous loads.
    let mut transposed: Vec<F::Packing> = vec![F::Packing::default(); 4 * n];
    {
        let (c0, rest) = transposed.split_at_mut(n);
        let (c1, rest) = rest.split_at_mut(n);
        let (c2, c3) = rest.split_at_mut(n);
        for (i, item) in eq1_packed.iter().enumerate() {
            let coefs = item.as_basis_coefficients_slice();
            c0[i] = coefs[0];
            c1[i] = coefs[1];
            c2[i] = coefs[2];
            c3[i] = coefs[3];
        }
    }
    let (c0, rest) = transposed.split_at(n);
    let (c1, rest) = rest.split_at(n);
    let (c2, c3) = rest.split_at(n);
    let cs: [&[F::Packing]; 4] = [c0, c1, c2, c3];

    let mut acc = EF::ExtensionPacking::default();
    for (j, w0) in eq0.as_slice().iter().enumerate() {
        let piece = &chunk_packed[j * n..(j + 1) * n];

        let coef_results: [F::Packing; 4] = core::array::from_fn(|k| {
            let src = cs[k];
            let mut a = F::Packing::ZERO;
            let mut i = 0;
            while i + CHUNK <= n {
                let lhs: &[F::Packing; CHUNK] =
                    (&src[i..i + CHUNK]).try_into().unwrap();
                let rhs: &[F::Packing; CHUNK] =
                    (&piece[i..i + CHUNK]).try_into().unwrap();
                a += F::Packing::dot_product::<CHUNK>(lhs, rhs);
                i += CHUNK;
            }
            while i < n {
                a += src[i] * piece[i];
                i += 1;
            }
            a
        });

        let inner_j = EF::ExtensionPacking::from_basis_coefficients_fn(|k| coef_results[k]);
        acc += inner_j * *w0;
    }

    <EF::ExtensionPacking as PackedFieldExtension<F, EF>>::to_ext_iter([acc]).sum()
}

/// Eager-reduction fallback used for extension-field dimensions other than 4.
///
/// Kept as a correctness guardrail; unreachable on the BabyBear/KoalaBear ext4
/// configurations that drive the `compress_hi_dot` hot path.
#[inline]
fn fallback<F, EF>(
    eq1_packed: &[<EF as ExtensionField<F>>::ExtensionPacking],
    chunk: &[F],
    eq0: &Poly<EF>,
) -> EF
where
    F: Field,
    EF: ExtensionField<F>,
{
    use p3_field::dot_product;
    let chunk_packed = F::Packing::pack_slice(chunk);
    let inner_size = eq1_packed.len();

    let sum: EF::ExtensionPacking = (0..eq0.as_slice().len())
        .map(|j| {
            let chunk_piece = &chunk_packed[j * inner_size..(j + 1) * inner_size];
            let d: EF::ExtensionPacking = dot_product::<EF::ExtensionPacking, _, _>(
                eq1_packed.iter().copied(),
                chunk_piece.iter().copied(),
            );
            d * eq0.as_slice()[j]
        })
        .sum();
    <EF::ExtensionPacking as PackedFieldExtension<F, EF>>::to_ext_iter([sum]).sum()
}
