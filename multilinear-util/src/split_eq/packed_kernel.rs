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
    Algebra, BasedVectorSpace, ExtensionField, Field, PackedFieldExtension, PackedValue,
    PrimeCharacteristicRing, dot_product,
};

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
    eq0: &[EF],
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
        n * eq0.len(),
        "chunk must hold |eq0| * N * W base-field elements \
         (got {} packed, expected {} = N * |eq0|)",
        chunk_packed.len(),
        n * eq0.len(),
    );

    // Fast-path guard: stack transpose needs bounded `N`.
    if n <= MAX_STACK_N {
        // Lift `EF::DIMENSION` (associated const) into a true const generic:
        //
        //     - each arm pins `D` to a concrete number,
        //     - each arm becomes a fully unrolled monomorphisation,
        //     - non-matching arms are dead at every concrete call site.
        match EF::DIMENSION {
            1 => return basis_split_dot::<F, EF, 1>(eq1_packed, chunk_packed, eq0, n),
            2 => return basis_split_dot::<F, EF, 2>(eq1_packed, chunk_packed, eq0, n),
            3 => return basis_split_dot::<F, EF, 3>(eq1_packed, chunk_packed, eq0, n),
            4 => return basis_split_dot::<F, EF, 4>(eq1_packed, chunk_packed, eq0, n),
            5 => return basis_split_dot::<F, EF, 5>(eq1_packed, chunk_packed, eq0, n),
            6 => return basis_split_dot::<F, EF, 6>(eq1_packed, chunk_packed, eq0, n),
            7 => return basis_split_dot::<F, EF, 7>(eq1_packed, chunk_packed, eq0, n),
            8 => return basis_split_dot::<F, EF, 8>(eq1_packed, chunk_packed, eq0, n),
            _ => {}
        }
    }

    // Fallback: one full packed-extension dot product per row.
    //     - Montgomery reduction per inner multiply,
    //     - used when the fast-path constraints do not hold.
    let sum: EF::ExtensionPacking = chunk_packed
        .chunks_exact(n)
        .zip_eq(eq0)
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

/// Inner group size for the per-basis dot products.
///
/// # Two roles
///
/// - **Interleave grain** (field-agnostic): round-robin step across
///   the per-basis accumulators. Keeps the array in scalar registers
///   and amortises loop overhead.
/// - **Overflow ceiling** (Monty-31 only): largest delayed-reduction
///   group that fits in a `u64` accumulator:
///
/// ```text
///     element            < 2^31
///     product            < 2^62
///     4 products summed  < 2^64    (just fits)
/// ```
///
/// # Behaviour across fields
///
/// - **Monty-31**: hits a hand-tuned delayed-reduction primitive.
///   Source of the ~4x reduction-count win.
/// - **Other fields**: still correct. Basis-split and ILP wins apply;
///   the delayed-reduction win does not.
///
/// # TODO: field-tunable group size
///
/// - Lift this constant to a per-field associated value.
/// - Dispatch on it alongside the existing dimension match.
/// - Each field then hits its own hand-tuned specialisation.
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
/// 128 comfortably covers the hottest workloads;
/// Larger `N` routes to the fallback so the stack reservation stays bounded.
const MAX_STACK_N: usize = 128;

/// Const-generic inner kernel for the packed high-eq dot product.
///
/// # Caller invariants
///
/// - `D == EF::DIMENSION`,
/// - `n <= MAX_STACK_N`.
#[inline]
fn basis_split_dot<F, EF, const D: usize>(
    eq1_packed: &[EF::ExtensionPacking],
    chunk_packed: &[F::Packing],
    eq0: &[EF],
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
    // `MaybeUninit` skips the zero-fill;
    //
    // Only the first `n` slots of each buffer are written by the loop below.
    let mut bufs: [[MaybeUninit<F::Packing>; MAX_STACK_N]; D] =
        [[const { MaybeUninit::uninit() }; MAX_STACK_N]; D];
    for (i, item) in eq1_packed.iter().enumerate() {
        // `coefs.len()` equals `EF::DIMENSION`.
        //
        // The dispatcher pinned to `D` at the call site.
        let coefs = item.as_basis_coefficients_slice();
        for k in 0..D {
            bufs[k][i].write(coefs[k]);
        }
    }

    // SAFETY:
    //
    // The loop above initialised exactly the first `n` slots of each buffer.
    // We expose only that prefix, and the buffers are not aliased.
    let cs: [&[F::Packing]; D] = core::array::from_fn(|k| unsafe {
        core::slice::from_raw_parts(bufs[k].as_ptr().cast::<F::Packing>(), n)
    });

    // Phase 2: interleaved per-row dot product.
    //
    // One pass over `i` drives all `D` coefficient accumulators,
    // reusing a single load of each right-hand block per step.
    let mut acc = EF::ExtensionPacking::ZERO;
    for (j, w0) in eq0.iter().enumerate() {
        // Right-hand side for row `j`: `n` packed base-field blocks.
        let piece = &chunk_packed[j * n..(j + 1) * n];

        // Per-coefficient delayed-reduction dot product against `piece`.
        // Returns the D-coefficient F::Packing array.
        let a = packed_basis_dot::<F, D>(&cs, piece);

        // - Reassemble into a packed extension element,
        // - Weight by the scalar from `eq0`,
        // - Fold into the running accumulator.
        acc += EF::ExtensionPacking::from_basis_coefficients_fn(|k| a[k]) * *w0;
    }

    // Phase 3: horizontal reduction.
    //
    // The accumulator carries `W` parallel partial sums, one per SIMD lane.
    // Flatten into a single scalar result.
    EF::ExtensionPacking::to_ext_iter([acc]).sum()
}

/// Const-generic inner kernel for the prefix-fold packed-output path.
///
/// # Caller invariants
///
/// - `D == EF::DIMENSION`,
/// - `F::Packing::WIDTH % CHUNK == 0`,
/// - `eq1_packed.len() <= MAX_STACK_N` (bounds the pre-multiply stack buffer).
fn compress_prefix_to_packed_packed_kernel<F, EF, const D: usize>(
    out: &mut [EF::ExtensionPacking],
    eq1_packed: &[EF::ExtensionPacking],
    chunk: &[F],
    w0: EF,
) where
    F: Field,
    EF: ExtensionField<F>,
{
    let n_packed = eq1_packed.len();
    let packed_inner = out.len();
    let w = F::Packing::WIDTH;

    // CHUNK divides W: every batch of CHUNK lanes stays inside one w_idx.
    //
    // So the gathered scalar weights all come from the same packed entry.
    debug_assert!(
        w.is_multiple_of(CHUNK),
        "W = {w} must be a multiple of CHUNK = {CHUNK}"
    );
    debug_assert!(
        n_packed <= MAX_STACK_N,
        "n_packed = {n_packed} must fit in the stack-resident pre-multiply buffer (MAX_STACK_N = {MAX_STACK_N})"
    );
    let lane_batches = w / CHUNK;

    // View the base-field row as packed rows, indexed by (w_idx, lane).
    //
    //     chunk_packed[(w_idx * W + lane) * packed_inner + i_pkt]
    //         packs   chunk[(w_idx * W + lane) * packed_inner * W + i_pkt * W..]
    let chunk_packed = F::Packing::pack_slice(chunk);
    debug_assert_eq!(chunk_packed.len(), n_packed * w * packed_inner);

    // Phase 0: pre-multiply each eq1 slot by w0 once.
    //
    //     pw[w_idx] = eq1_packed[w_idx] * w0     (depends only on w_idx)
    //
    // - Without this pre-pass: recomputed once per tile per w_idx.
    // - With it: recomputed once per w_idx total.
    // - Buffer is stack-resident; only the first n_packed slots are written.
    let mut pw_buf = [const { MaybeUninit::uninit() }; MAX_STACK_N];
    for (slot, &eq1_i) in pw_buf.iter_mut().zip(eq1_packed.iter()) {
        slot.write(eq1_i * w0);
    }
    // SAFETY: The loop above wrote exactly:
    // - the first `n_packed` slots,
    // - we expose only that prefix,
    // - the buffer is not aliased,
    // - `MaybeUninit<T>` shares layout and alignment with `T`.
    let pw: &[EF::ExtensionPacking] = unsafe {
        core::slice::from_raw_parts(pw_buf.as_ptr().cast::<EF::ExtensionPacking>(), n_packed)
    };

    // Phase 1: outer tile over output positions.
    //
    // Each tile keeps `TILE * D` per-coefficient accumulators in registers
    // and only commits them to the output buffer at tile end.
    let mut tile_start = 0;
    while tile_start < packed_inner {
        let tile_len = TILE.min(packed_inner - tile_start);
        let mut acc = [[F::Packing::ZERO; D]; TILE];

        // Phase 2: walk every (w_idx, lane) pair contributing to the tile.
        for w_idx in 0..n_packed {
            // Pull the pre-multiplied weight from the buffer;
            //
            // No redundant packed-extension multiply per tile.
            let pw_coefs = pw[w_idx].as_basis_coefficients_slice();

            for batch in 0..lane_batches {
                let lane_start = batch * CHUNK;

                // Pre-extract the CHUNK scalar weights per coefficient
                // by reading the matching lanes of the packed coefficient.
                let scalars: [[F; CHUNK]; D] = core::array::from_fn(|k| {
                    let coef_lanes = pw_coefs[k].as_slice();
                    core::array::from_fn(|c| coef_lanes[lane_start + c])
                });

                // Per output position in the tile:
                //   - gather CHUNK packed values from chunk_packed,
                //   - run D delayed-reduction dot products over them.
                for (t, acc_t) in acc[..tile_len].iter_mut().enumerate() {
                    let i_pkt = tile_start + t;
                    let packed_vals: [F::Packing; CHUNK] = core::array::from_fn(|c| {
                        chunk_packed[(w_idx * w + lane_start + c) * packed_inner + i_pkt]
                    });
                    for (k, acc_tk) in acc_t.iter_mut().enumerate() {
                        *acc_tk +=
                            F::Packing::mixed_dot_product::<CHUNK>(&packed_vals, &scalars[k]);
                    }
                }
            }
        }

        // Phase 3: fold each per-output accumulator into the output buffer.
        for (t, acc_t) in acc[..tile_len].iter().enumerate() {
            out[tile_start + t] += EF::ExtensionPacking::from_basis_coefficients_fn(|k| acc_t[k]);
        }

        tile_start += TILE;
    }
}

/// Eager fallback for the prefix-fold packed-output path.
///
/// # Behaviour
///
/// - One Montgomery reduction per inner multiply.
/// - No basis split, no tiling.
///
/// # When this runs
///
/// - `EF::DIMENSION` outside `1..=8`, or
/// - `F::Packing::WIDTH % CHUNK != 0`.
fn compress_prefix_to_packed_packed_kernel_eager<F, EF>(
    out: &mut [EF::ExtensionPacking],
    eq1_packed: &[EF::ExtensionPacking],
    chunk: &[F],
    w0: EF,
) where
    F: Field,
    EF: ExtensionField<F>,
{
    // Empty output is a no-op:
    //
    // - With no output slots, the inner stride collapses to zero.
    // - Iterating the row in zero-sized chunks panics.
    // - Bail before that step.
    if out.is_empty() {
        return;
    }

    let scalar_inner = out.len() * F::Packing::WIDTH;
    chunk
        .chunks(scalar_inner * F::Packing::WIDTH)
        .zip_eq(eq1_packed.iter())
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

/// SIMD kernel for the prefix-fold accumulation into a packed output.
///
/// # Arguments
///
/// - `out`: packed accumulator buffer of length `packed_inner`.
/// - `eq1_packed`: packed extension-field row of length `n_packed`.
/// - `chunk`: base-field row of `n_packed * W` per-lane sub-rows of
///   `packed_inner * W` elements each, where `W` is the base-field
///   packing width.
/// - `w0`: scalar extension-field weight applied to every eq1 lane.
///
/// # Effect
///
/// ```text
///     out[i_pkt] += sum_{w_idx, lane}
///         (eq1_packed[w_idx] * w0).lane[lane]
///         * F::Packing(chunk[(w_idx * W + lane) * packed_inner * W + i_pkt * W..])
/// ```
///
/// # Algorithm
///
/// Two code paths, selected per call.
///
/// ## Fast path
///
/// Conditions:
///
/// - `EF::DIMENSION ∈ {1, …, 8}`,
/// - `W % CHUNK == 0`,
/// - `n_packed <= MAX_STACK_N`.
///
/// What it does:
///
/// - Basis split + delayed Montgomery reduction.
/// - Tiling over output positions.
/// - One const-generic monomorphisation per supported `D`.
///
/// ## Eager fallback
///
/// - Used when any fast-path condition fails.
/// - One full packed-extension multiply per inner step.
///
/// # Panics
///
/// - `chunk.len()` is not a multiple of the base-field packing width, or
/// - `chunk.len() < n_packed * W * packed_inner * W`.
pub(super) fn compress_prefix_to_packed_packed<F, EF>(
    out: &mut [EF::ExtensionPacking],
    eq1_packed: &[EF::ExtensionPacking],
    chunk: &[F],
    w0: EF,
) where
    F: Field,
    EF: ExtensionField<F>,
{
    if F::Packing::WIDTH.is_multiple_of(CHUNK) && eq1_packed.len() <= MAX_STACK_N {
        match EF::DIMENSION {
            1 => {
                compress_prefix_to_packed_packed_kernel::<F, EF, 1>(out, eq1_packed, chunk, w0);
                return;
            }
            2 => {
                compress_prefix_to_packed_packed_kernel::<F, EF, 2>(out, eq1_packed, chunk, w0);
                return;
            }
            3 => {
                compress_prefix_to_packed_packed_kernel::<F, EF, 3>(out, eq1_packed, chunk, w0);
                return;
            }
            4 => {
                compress_prefix_to_packed_packed_kernel::<F, EF, 4>(out, eq1_packed, chunk, w0);
                return;
            }
            5 => {
                compress_prefix_to_packed_packed_kernel::<F, EF, 5>(out, eq1_packed, chunk, w0);
                return;
            }
            6 => {
                compress_prefix_to_packed_packed_kernel::<F, EF, 6>(out, eq1_packed, chunk, w0);
                return;
            }
            7 => {
                compress_prefix_to_packed_packed_kernel::<F, EF, 7>(out, eq1_packed, chunk, w0);
                return;
            }
            8 => {
                compress_prefix_to_packed_packed_kernel::<F, EF, 8>(out, eq1_packed, chunk, w0);
                return;
            }
            _ => {}
        }
    }
    compress_prefix_to_packed_packed_kernel_eager::<F, EF>(out, eq1_packed, chunk, w0);
}

/// Tile size in output positions for the prefix-fold packed kernel.
///
/// # Why this value
///
/// - Holds `TILE * D` per-coefficient accumulators in scalar registers.
/// - Lets the inner loop become throughput-bound on the dot primitive.
/// - Keeps the touched output slice in L1 across one tile.
///
/// Stack footprint per call:
///
/// ```text
///     TILE * D * sizeof(F::Packing)
///         = 8 * 8 * 64 B = 4 KiB on AVX-512 (D = 8, packing width 16)
///         = 8 * 4 * 16 B = 512 B on NEON    (D = 4, packing width  4)
/// ```
const TILE: usize = 8;

/// Per-coefficient delayed-reduction dot product.
///
/// # Returns
///
/// A `D`-coefficient `F::Packing` array, where coefficient `k` holds
///
/// ```text
///     sum_i cs[k][i] * rhs[i].
/// ```
///
/// # Algorithm
///
/// - Main loop: `D` independent dot products on aligned blocks of
///   `CHUNK` items, one Montgomery reduction per block per coefficient.
/// - Scalar tail: sweeps the `n mod CHUNK` trailing positions.
///
/// # Caller invariants
///
/// - `cs[k].len() == rhs.len()` for every `k`.
#[inline]
fn packed_basis_dot<F, const D: usize>(
    cs: &[&[F::Packing]; D],
    rhs: &[F::Packing],
) -> [F::Packing; D]
where
    F: Field,
{
    let n = rhs.len();

    // `D` independent accumulators, one per basis coefficient.
    //
    // Small const-D array: LLVM promotes to scalar registers.
    let mut a = [F::Packing::ZERO; D];

    // Main loop: `D` `dot_product_<CHUNK>` calls per step.
    //     - `D` independent accumulator chains,
    //     - one Montgomery reduction per `CHUNK` multiplies,
    //     - one shared load of the `rhs` CHUNK across all `D`.
    let mut i = 0;
    while i + CHUNK <= n {
        let rhs_chunk: &[F::Packing; CHUNK] = (&rhs[i..i + CHUNK]).try_into().unwrap();
        for k in 0..D {
            let l: &[F::Packing; CHUNK] = (&cs[k][i..i + CHUNK]).try_into().unwrap();
            a[k] += F::Packing::dot_product::<CHUNK>(l, rhs_chunk);
        }
        i += CHUNK;
    }

    // Scalar tail: sweep up the `n mod CHUNK` trailing positions.
    while i < n {
        let p = rhs[i];
        for k in 0..D {
            a[k] += cs[k][i] * p;
        }
        i += 1;
    }

    a
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
    use crate::poly::Poly;

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;
    type PF = <F as Field>::Packing;
    type PackedEF = <EF as ExtensionField<F>>::ExtensionPacking;

    /// Naive triple-nested reference: no SIMD tricks, no basis split.
    fn reference(eq1_packed: &[PackedEF], chunk: &[F], eq0: &Poly<EF>) -> EF {
        // Repack the base-field row the same way the kernel does.
        let chunk_packed = PF::pack_slice(chunk);
        let n = eq1_packed.len();

        // Accumulate the triple sum over `(j, i)` in packed form.
        let mut total = PackedEF::ZERO;
        for (j, &w0) in eq0.as_slice().iter().enumerate() {
            let piece = &chunk_packed[j * n..(j + 1) * n];
            for (&eq1i, &pi) in eq1_packed.iter().zip(piece) {
                total += eq1i * pi * w0;
            }
        }

        // Horizontal-reduce to a scalar so the output type matches.
        <PackedEF as PackedFieldExtension<F, EF>>::to_ext_iter([total]).sum()
    }

    /// Deterministic random input generator for the kernel and reference.
    fn random_inputs(n: usize, m_vars: usize, seed: u64) -> (Vec<PackedEF>, Vec<F>, Poly<EF>) {
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
                compress_hi_dot_packed::<F, EF>(&eq1, &chunk, eq0.as_slice()),
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
                compress_hi_dot_packed::<F, EF>(&eq1, &chunk, eq0.as_slice()),
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
        let eq0 = [EF::ZERO, EF::ZERO];

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
                compress_hi_dot_packed::<F, EF>(&eq1, &chunk, eq0.as_slice()),
                reference(&eq1, &chunk, &eq0),
                "mismatch at n = {n}",
            );
        }
    }

    /// Builds random inputs for the prefix-fold packed-output kernel.
    ///
    /// Shape:
    ///
    /// - `n_packed`: packed eq1 entries.
    /// - `packed_inner`: packed output positions.
    /// - `chunk` length: `n_packed * W * packed_inner * W` scalars.
    fn random_prefix_inputs(
        n_packed: usize,
        packed_inner: usize,
        seed: u64,
    ) -> (Vec<PackedEF>, Vec<F>, EF) {
        // Number of base-field lanes per SIMD packing.
        let w = PF::WIDTH;
        // One RNG seeded per shape gives reproducible inputs.
        let mut rng = SmallRng::seed_from_u64(seed);

        // Packed eq1 row: one PackedEF per slot.
        let eq1: Vec<PackedEF> = (0..n_packed).map(|_| rng.random()).collect();
        // Chunk layout: `n_packed * W` per-lane sub-rows, each of length
        // `packed_inner * W` base-field elements.
        let chunk: Vec<F> = (0..n_packed * w * packed_inner * w)
            .map(|_| rng.random())
            .collect();
        // Outer scalar weight applied to every eq1 lane.
        let w0: EF = rng.random();

        (eq1, chunk, w0)
    }

    proptest! {
        #[test]
        fn prefix_to_packed_kernel_matches_eager(
            n_packed in 0usize..=8,
            inner_k in 0usize..=5,
            seed in any::<u64>(),
        ) {
            // Invariant: kernel output equals the eager fallback for
            // every shape in the supported range.
            //
            // Fixture state:
            //     n_packed   = 0..8 - covers the empty case + small N.
            //     inner_k    = 0..5 - 1 to 32 packed output positions,
            //                         covers multi-tile + the TILE = 8 boundary.
            let packed_inner = 1usize << inner_k;
            let (eq1, chunk, w0) = random_prefix_inputs(n_packed, packed_inner, seed);

            let mut out_kernel = PackedEF::zero_vec(packed_inner);
            let mut out_eager = PackedEF::zero_vec(packed_inner);

            compress_prefix_to_packed_packed::<F, EF>(&mut out_kernel, &eq1, &chunk, w0);
            compress_prefix_to_packed_packed_kernel_eager::<F, EF>(
                &mut out_eager,
                &eq1,
                &chunk,
                w0,
            );

            prop_assert_eq!(out_kernel, out_eager);
        }
    }

    #[test]
    fn prefix_to_packed_kernel_empty_out_does_not_panic() {
        // Invariant: an empty output slice is a no-op on both paths.
        //
        // Fixture state:
        //     n_packed     = 3        - eq1 sized as if the fast path would engage.
        //     packed_inner = 0        - empty output short-circuits both kernels.
        //     chunk        = []       - matches packed_inner = 0.
        let mut rng = SmallRng::seed_from_u64(0xCAFE);
        let mut out_fast: Vec<PackedEF> = Vec::new();
        let mut out_eager: Vec<PackedEF> = Vec::new();

        let eq1: Vec<PackedEF> = (0..3).map(|_| rng.random()).collect();
        let chunk: Vec<F> = Vec::new();
        let w0: EF = rng.random();

        // Both calls must return cleanly without writing or panicking.
        compress_prefix_to_packed_packed::<F, EF>(&mut out_fast, &eq1, &chunk, w0);
        compress_prefix_to_packed_packed_kernel_eager::<F, EF>(&mut out_eager, &eq1, &chunk, w0);

        // No slots → nothing was written.
        assert!(out_fast.is_empty());
        assert!(out_eager.is_empty());
    }

    #[test]
    fn prefix_to_packed_kernel_empty_eq1_is_noop() {
        // Invariant: empty eq1 -> empty sum -> out unchanged.
        //
        // Fixture state:
        //     n_packed     = 0
        //     packed_inner = 4
        //     out          = [random PackedEF; 4]
        let mut rng = SmallRng::seed_from_u64(0xFEED);
        let initial: Vec<PackedEF> = (0..4).map(|_| rng.random()).collect();
        let mut out = initial.clone();

        compress_prefix_to_packed_packed::<F, EF>(&mut out, &[], &[], rng.random());

        // Out should be byte-for-byte identical: nothing to add.
        for (a, b) in out.iter().zip(initial.iter()) {
            assert_eq!(a, b);
        }
    }

    #[test]
    fn prefix_to_packed_kernel_zero_w0_is_noop() {
        // Invariant: w0 = 0 -> every (eq1[i] * w0) = 0 -> out unchanged,
        // regardless of eq1 and chunk shape.
        //
        // Fixture state:
        //     n_packed     = 4
        //     packed_inner = 8         - exactly one TILE.
        //     w0           = EF::ZERO
        let (eq1, chunk, _) = random_prefix_inputs(4, 8, 0xBEEF);
        let mut rng = SmallRng::seed_from_u64(0xBEEF + 1);
        let initial: Vec<PackedEF> = (0..8).map(|_| rng.random()).collect();
        let mut out = initial.clone();

        compress_prefix_to_packed_packed::<F, EF>(&mut out, &eq1, &chunk, EF::ZERO);

        for (a, b) in out.iter().zip(initial.iter()) {
            assert_eq!(a, b);
        }
    }

    #[test]
    fn prefix_to_packed_kernel_partial_tile() {
        // Invariant: when packed_inner < TILE, the single-shorter-tile
        // path matches the eager fallback exactly.
        //
        // Fixture state:
        //     n_packed     = 3
        //     packed_inner = 5         - one tile of length 5 (< TILE = 8).
        let (eq1, chunk, w0) = random_prefix_inputs(3, 5, 0xC0DE);

        let mut out_kernel = PackedEF::zero_vec(5);
        let mut out_eager = PackedEF::zero_vec(5);

        compress_prefix_to_packed_packed::<F, EF>(&mut out_kernel, &eq1, &chunk, w0);
        compress_prefix_to_packed_packed_kernel_eager::<F, EF>(&mut out_eager, &eq1, &chunk, w0);

        assert_eq!(out_kernel, out_eager);
    }

    #[test]
    fn prefix_to_packed_kernel_multi_tile() {
        // Invariant: multiple tiles (packed_inner > TILE) reduce
        // identically to the eager fallback.
        //
        // Fixture state:
        //     n_packed     = 4
        //     packed_inner = 17        - two full tiles + one short tail.
        let (eq1, chunk, w0) = random_prefix_inputs(4, 17, 0xC0FFEE);

        let mut out_kernel = PackedEF::zero_vec(17);
        let mut out_eager = PackedEF::zero_vec(17);

        compress_prefix_to_packed_packed::<F, EF>(&mut out_kernel, &eq1, &chunk, w0);
        compress_prefix_to_packed_packed_kernel_eager::<F, EF>(&mut out_eager, &eq1, &chunk, w0);

        assert_eq!(out_kernel, out_eager);
    }

    #[test]
    fn prefix_to_packed_kernel_accumulates_into_out() {
        // Invariant: the kernel ADDS to `out`, it does not overwrite —
        // matching the eager fallback's `*acc += w * f` semantics.
        //
        // Fixture state:
        //     n_packed     = 2
        //     packed_inner = 8
        //     initial out  = nonzero random
        let (eq1, chunk, w0) = random_prefix_inputs(2, 8, 0xDEAD);
        let mut rng = SmallRng::seed_from_u64(0xDEAD + 1);
        let initial: Vec<PackedEF> = (0..8).map(|_| rng.random()).collect();

        let mut out_kernel = initial.clone();
        let mut out_eager = initial;

        compress_prefix_to_packed_packed::<F, EF>(&mut out_kernel, &eq1, &chunk, w0);
        compress_prefix_to_packed_packed_kernel_eager::<F, EF>(&mut out_eager, &eq1, &chunk, w0);

        assert_eq!(out_kernel, out_eager);
    }
}
