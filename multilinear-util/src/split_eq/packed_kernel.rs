//! SIMD kernel for the packed high-eq dot product.
//!
//! # Overview
//!
//! For
//!
//! - a packed extension-field row `eq1` of length `N`,
//! - a base-field row `chunk`,
//! - a scalar extension-field weight vector `eq0`,
//!
//! the kernel returns
//!
//! ```text
//!     sum_{j, i}  eq0[j] * eq1[i] * chunk[j * N + i].
//! ```
//!
//! # Algorithm
//!
//! Three optimisations stack on top of each other.
//!
//! ## Basis split
//!
//! A packed extension element of dimension `D` is a degree-`D`
//! polynomial over the base-field packing.
//!
//! A packed dot product therefore factors into:
//!
//! - `D` independent scalar-on-packed dot products,
//! - one per basis coefficient.
//!
//! ## Delayed Montgomery reduction
//!
//! Each per-coefficient dot product runs through a primitive that:
//!
//! - accumulates four `u64` products,
//! - then applies one Montgomery reduction.
//!
//! Net: one reduction per group of four multiplies, not one per multiply.
//!
//! ## Interleaved inner loop
//!
//! The four per-coefficient accumulators are independent.
//! Running them as four sequential passes can serialise their SIMD multiplies.
//!
//! Interleaving them in a single step:
//!
//! - exposes four independent chains to the out-of-order engine,
//! - loads each shared right-hand block once per step, not four times.
//!
//! # Memory Layout
//!
//! Input (array-of-structs):
//!
//! ```text
//!     eq1_packed:  [ (c0, c1, c2, c3)_0,
//!                    (c0, c1, c2, c3)_1,
//!                    ...                  ]
//! ```
//!
//! After transpose (structure-of-arrays, one buffer per coefficient):
//!
//! ```text
//!     c0 : [ c0_0, c0_1, ..., c0_{N-1} ]
//!     c1 : [ c1_0, c1_1, ..., c1_{N-1} ]
//!     c2 : [ c2_0, c2_1, ..., c2_{N-1} ]
//!     c3 : [ c3_0, c3_1, ..., c3_{N-1} ]
//! ```
//!
//! Stack-transpose properties:
//!
//! - Buffers are uninitialised storage — no zero-fill cost.
//! - Only the first `N` slots of each buffer are written.
//! - Above a fixed `N` limit the kernel skips the transpose and uses
//!   a simpler fallback, keeping stack usage bounded.

use core::mem::MaybeUninit;

use itertools::Itertools;
use p3_field::{
    BasedVectorSpace, ExtensionField, Field, PackedFieldExtension, PackedValue,
    PrimeCharacteristicRing, dot_product,
};

use crate::poly::Poly;

/// SIMD kernel for the packed high-eq dot product.
///
/// # Arguments
///
/// - `eq1_packed`: packed extension-field row of length `N`.
/// - `chunk`: base-field row of `|eq0|` groups of `N * W` elements,
///   where `W` is the base-field packing width.
/// - `eq0`: scalar extension-field weight vector; its length is the
///   number of rows to fold.
///
/// # Returns
///
/// ```text
///     sum_{j, i}  eq0[j] * eq1_packed[i] * chunk[j * N + i].
/// ```
///
/// # Algorithm
///
/// Two code paths, selected per call:
///
/// - `EF::DIMENSION ∈ {1, …, 8}` and `N <= 128`: basis-split plus
///   interleaved delayed-reduction dot products. Dispatched into a
///   const-generic inner function so each supported dimension gets its
///   own fully unrolled monomorphisation.
/// - Otherwise: one packed-extension dot product per row.
///
/// # Panics
///
/// - `chunk.len()` is not a multiple of the base-field packing width, or
/// - `chunk.len() < |eq0| * N * W`.
pub(super) fn compress_hi_dot_packed<F, EF>(
    eq1_packed: &[EF::ExtensionPacking],
    chunk: &[F],
    eq0: &Poly<EF>,
) -> EF
where
    F: Field,
    EF: ExtensionField<F>,
{
    // Number of packed slots on the left of each dot product.
    let n = eq1_packed.len();

    // View the base-field row as packed blocks for SIMD consumption.
    //
    // Requires `chunk.len() % W == 0`, which the caller guarantees.
    let chunk_packed = F::Packing::pack_slice(chunk);

    // Debug-only shape check:
    //     - `chunk` must carry one `n`-block row per entry of `eq0`,
    //     - i.e. `chunk_packed.len() == n * |eq0|` in packed units.
    //
    // A mismatch would panic later inside `chunks_exact` + `zip_eq`
    // with a generic message; this assertion names the invariant.
    debug_assert_eq!(
        chunk_packed.len(),
        n * eq0.as_slice().len(),
        "chunk must hold |eq0| * N * W base-field elements \
         (got {} packed, expected {} = N * |eq0|)",
        chunk_packed.len(),
        n * eq0.as_slice().len(),
    );

    // Fast-path guard: stack transpose needs bounded `N`.
    if n <= MAX_STACK_N {
        // Lift `EF::DIMENSION` (an associated const) into a true const
        // generic by an explicit match. Each arm gets a fully unrolled
        // monomorphisation of the inner kernel; non-matching arms are
        // dead at every concrete call site.
        match EF::DIMENSION {
            1 => return run_fast_path::<F, EF, 1>(eq1_packed, chunk_packed, eq0, n),
            2 => return run_fast_path::<F, EF, 2>(eq1_packed, chunk_packed, eq0, n),
            3 => return run_fast_path::<F, EF, 3>(eq1_packed, chunk_packed, eq0, n),
            4 => return run_fast_path::<F, EF, 4>(eq1_packed, chunk_packed, eq0, n),
            5 => return run_fast_path::<F, EF, 5>(eq1_packed, chunk_packed, eq0, n),
            6 => return run_fast_path::<F, EF, 6>(eq1_packed, chunk_packed, eq0, n),
            7 => return run_fast_path::<F, EF, 7>(eq1_packed, chunk_packed, eq0, n),
            8 => return run_fast_path::<F, EF, 8>(eq1_packed, chunk_packed, eq0, n),
            _ => {}
        }
    }

    // Fallback: one full packed-extension dot product per row.
    //     - Montgomery reduction per inner multiply,
    //     - used when the fast-path constraints do not hold.
    let sum: EF::ExtensionPacking = chunk_packed
        .chunks_exact(n)
        .zip_eq(eq0.as_slice())
        .map(|(piece, &w0)| {
            dot_product::<EF::ExtensionPacking, _, _>(
                eq1_packed.iter().copied(),
                piece.iter().copied(),
            ) * w0
        })
        .sum();

    // Lane-wise horizontal reduction to a scalar.
    EF::ExtensionPacking::to_ext_iter([sum]).sum()
}

/// Inner group size for the delayed-reduction primitive.
///
/// Fixed at 4 because `F::Packing::dot_product::<4>` maps to the
/// `dot_product_4` primitive on every SIMD target. `N mod 4` residues
/// fall to a scalar tail.
const CHUNK: usize = 4;

/// Upper bound on `N` for the stack-transpose fast path.
///
/// Stack footprint:
/// ```text
///     D * MAX_STACK_N * size_of::<Packing>()
///         = 32 KiB  on AVX-512 (D = 4, packing width 16)
///         =  8 KiB  on NEON    (D = 4, packing width  4)
/// ```
///
/// 128 comfortably covers the hottest workloads; larger `N` routes to
/// the fallback so the stack reservation stays bounded.
const MAX_STACK_N: usize = 128;

/// Const-generic inner kernel for the packed high-eq dot product.
///
/// Implements the basis-split / delayed-reduction / interleaved-loop
/// recipe parameterised by the extension dimension `D`. Each supported
/// `D` gets its own monomorphisation, so the per-coefficient loop is
/// fully unrolled and the per-coefficient accumulator array is
/// register-allocated rather than stack-spilled.
///
/// Caller invariant: `D == EF::DIMENSION` and `n <= MAX_STACK_N`.
#[inline]
fn run_fast_path<F, EF, const D: usize>(
    eq1_packed: &[EF::ExtensionPacking],
    chunk_packed: &[F::Packing],
    eq0: &Poly<EF>,
    n: usize,
) -> EF
where
    F: Field,
    EF: ExtensionField<F>,
{
    // Phase 1: transpose array-of-structs into structure-of-arrays.
    //
    //     input  : [ (c_0, ..., c_{D-1})_i ]
    //     output : per-coefficient buffers c_0[..], ..., c_{D-1}[..]
    //
    // `MaybeUninit` skips the zero-fill; only the first `n` slots of
    // each buffer are written by the loop below.
    let mut bufs: [[MaybeUninit<F::Packing>; MAX_STACK_N]; D] =
        [[const { MaybeUninit::uninit() }; MAX_STACK_N]; D];
    for (i, item) in eq1_packed.iter().enumerate() {
        // `coefs.len()` equals `EF::DIMENSION`, which the dispatcher
        // pinned to `D` at the call site.
        let coefs = item.as_basis_coefficients_slice();
        for k in 0..D {
            bufs[k][i].write(coefs[k]);
        }
    }

    // SAFETY:
    //
    // The loop above initialised exactly the first `n` slots of each
    // buffer via `MaybeUninit::write`. We expose only that prefix, and
    // the buffers are not aliased.
    let cs: [&[F::Packing]; D] = core::array::from_fn(|k| unsafe {
        core::slice::from_raw_parts(bufs[k].as_ptr().cast::<F::Packing>(), n)
    });

    // Phase 2: interleaved per-row dot product.
    //
    // One pass over `i` drives all `D` coefficient accumulators,
    // reusing a single load of each right-hand block per step.
    let mut acc = EF::ExtensionPacking::ZERO;
    for (j, w0) in eq0.as_slice().iter().enumerate() {
        // Right-hand side for row `j`: `n` packed base-field blocks.
        let piece = &chunk_packed[j * n..(j + 1) * n];

        // `D` independent accumulators, one per basis coefficient.
        // Small const-D array: LLVM promotes to scalar registers.
        let mut a = [F::Packing::ZERO; D];

        // Main loop: `D` `dot_product_4` calls per step.
        //     - `D` independent accumulator chains,
        //     - one Montgomery reduction per four multiplies.
        let mut i = 0;
        while i + CHUNK <= n {
            let rhs: &[F::Packing; CHUNK] = (&piece[i..i + CHUNK]).try_into().unwrap();
            for k in 0..D {
                let l: &[F::Packing; CHUNK] = (&cs[k][i..i + CHUNK]).try_into().unwrap();
                a[k] += F::Packing::dot_product::<CHUNK>(l, rhs);
            }
            i += CHUNK;
        }

        // Scalar tail: sweep up the `n mod CHUNK` trailing positions.
        while i < n {
            let p = piece[i];
            for k in 0..D {
                a[k] += cs[k][i] * p;
            }
            i += 1;
        }

        // - Reassemble into a packed extension element,
        // - Weight by the scalar from `eq0`,
        // - Fold into the running accumulator.
        acc += EF::ExtensionPacking::from_basis_coefficients_fn(|k| a[k]) * *w0;
    }

    // Phase 3: horizontal reduction.
    //
    // The accumulator carries `W` parallel partial sums, one per SIMD
    // lane. Flatten into a single scalar result.
    EF::ExtensionPacking::to_ext_iter([acc]).sum()
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use p3_baby_bear::BabyBear;
    use p3_field::extension::BinomialExtensionField;
    use proptest::prelude::*;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::*;

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;
    type PF = <F as Field>::Packing;
    type PEF = <EF as ExtensionField<F>>::ExtensionPacking;

    /// Naive triple-nested reference: no SIMD tricks, no basis split.
    fn reference(eq1_packed: &[PEF], chunk: &[F], eq0: &Poly<EF>) -> EF {
        // Repack the base-field row the same way the kernel does.
        let chunk_packed = PF::pack_slice(chunk);
        let n = eq1_packed.len();

        // Accumulate the triple sum over `(j, i)` in packed form.
        let mut total = PEF::ZERO;
        for (j, &w0) in eq0.as_slice().iter().enumerate() {
            let piece = &chunk_packed[j * n..(j + 1) * n];
            for (&eq1i, &pi) in eq1_packed.iter().zip(piece) {
                total += eq1i * pi * w0;
            }
        }

        // Horizontal-reduce to a scalar so the output type matches.
        <PEF as PackedFieldExtension<F, EF>>::to_ext_iter([total]).sum()
    }

    /// Deterministic random input generator for the kernel and reference.
    fn random_inputs(n: usize, m_vars: usize, seed: u64) -> (Vec<PEF>, Vec<F>, Poly<EF>) {
        // Number of base-field lanes per SIMD packing.
        let w = PF::WIDTH;
        // Length of `eq0`; the polynomial constructor requires a power of two.
        let m = 1usize << m_vars;
        let mut rng = SmallRng::seed_from_u64(seed);

        // One packed extension element per slot in `eq1`.
        let eq1 = (0..n).map(|_| rng.random()).collect();
        // `chunk` is `m` rows of `n * w` base-field elements.
        let chunk = (0..n * w * m).map(|_| rng.random()).collect();
        // `eq0` is `m` scalar extension-field weights.
        let eq0 = Poly::new((0..m).map(|_| rng.random()).collect());

        (eq1, chunk, eq0)
    }

    proptest! {
        #[test]
        fn matches_reference(
            n in 0usize..=40,
            m_vars in 0usize..=4,
            seed in any::<u64>(),
        ) {
            // Invariant: kernel output equals the naive sum at every shape.
            //
            // Fixture state:
            //     n       = 0..40   - straddles the fast-path range.
            //     m_vars  = 0..4    - from 1 row up to 16 rows.
            let (eq1, chunk, eq0) = random_inputs(n, m_vars, seed);

            // Kernel and naive reference on the same inputs.
            prop_assert_eq!(
                compress_hi_dot_packed::<F, EF>(&eq1, &chunk, &eq0),
                reference(&eq1, &chunk, &eq0),
            );
        }

        #[test]
        fn fast_path_matches_fallback_across_boundary(
            n in 126usize..=130,
            m_vars in 0usize..=3,
            seed in any::<u64>(),
        ) {
            // Invariant: the stack threshold is a performance switch,
            // not a semantic one. Outputs must agree either side.
            //
            // Fixture state:
            //     n = 126, 127, 128   -> fast path (stack transpose)
            //     n = 129, 130        -> fallback  (eager dot product)
            let (eq1, chunk, eq0) = random_inputs(n, m_vars, seed);

            // Both sides feed the same inputs to their respective path.
            prop_assert_eq!(
                compress_hi_dot_packed::<F, EF>(&eq1, &chunk, &eq0),
                reference(&eq1, &chunk, &eq0),
            );
        }
    }

    #[test]
    fn empty_eq1_is_zero() {
        // Invariant: empty left-hand row -> empty sum -> zero,
        // independent of the right-hand weights.
        //
        // Fixture state:
        //     eq1   = []
        //     chunk = []
        //     eq0   = [0, 0]
        let eq0 = Poly::<EF>::new([EF::ZERO, EF::ZERO].to_vec());

        // Expected: empty fold returns the additive identity.
        assert_eq!(compress_hi_dot_packed::<F, EF>(&[], &[], &eq0), EF::ZERO,);
    }

    #[test]
    fn scalar_tail_exercised() {
        // Invariant: the `n mod CHUNK != 0` tail agrees with the naive
        // reference at every residue class.
        //
        // Fixture state (residues picked below and near the threshold):
        //     n = 1, 5, 9, 17, 33, 125   -> residue 1 mod 4
        //     n = 2                      -> residue 2 mod 4
        //     n = 3, 7                   -> residue 3 mod 4
        for &n in &[1usize, 2, 3, 5, 7, 9, 17, 33, 125] {
            // Same seed per iteration for reproducibility; only `n` varies.
            let (eq1, chunk, eq0) = random_inputs(n, 2, 0xC0FFEE);

            // Kernel must match the naive reference at every residue.
            assert_eq!(
                compress_hi_dot_packed::<F, EF>(&eq1, &chunk, &eq0),
                reference(&eq1, &chunk, &eq0),
                "mismatch at n = {n}",
            );
        }
    }
}
