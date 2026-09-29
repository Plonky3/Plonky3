//! The passes of a polynomial-basis transform, over the matrix viewed as 128-bit words.

use alloc::vec::Vec;

use p3_binary_field::poly_basis::{LOW_STAGES, LowStageTwiddles};
use p3_binary_field::{BinaryField128, TowerLevel, poly_basis};
use p3_maybe_rayon::prelude::*;

use super::plan::{Plan, SHARED_CACHE_BYTES};
use crate::domain::domain_point;
use crate::lch::BUTTERFLY_GRAIN;
use crate::staging::{
    Dispatch, StagedRuns, Store, for_each_staged_tile, for_each_staged_tile_into_cosets_stored,
    prefault,
};

/// Stage bases and the steps between consecutive block twiddles, in polynomial coordinates.
pub(super) struct Twiddles {
    /// The Cantor basis vector `v_{j+1}`, for every stage `j`.
    basis: [u128; usize::BITS as usize],
    /// The step from block `b` to block `b + 1`, indexed by the trailing ones of `b`.
    pub(super) deltas: [u128; usize::BITS as usize],
    /// The twiddle of the first block of each stage, `W_j(shift)`.
    pub(super) shifts: [u128; usize::BITS as usize],
}

impl Twiddles {
    /// The twiddle state of a transform of `2^log_n` rows over the given coset.
    pub(super) fn new(log_n: usize, shift: BinaryField128) -> Self {
        let mut result = Self {
            basis: [0; usize::BITS as usize],
            deltas: [0; usize::BITS as usize],
            shifts: [0; usize::BITS as usize],
        };
        let mut delta = 0;
        let mut base = poly_basis::from_tower(shift);
        for j in 0..log_n {
            // Consecutive blocks differ by the sum of the basis vectors below the lowest clear bit.
            result.basis[j] = poly_basis::from_tower(BinaryField128::cantor_basis(j + 1));
            delta ^= result.basis[j];
            result.deltas[j] = delta;

            // W_{j+1} = W_j^2 + W_j, one squaring per stage.
            result.shifts[j] = base;
            base = poly_basis::square(base) ^ base;
        }
        result
    }

    /// The twiddle of one block, computed from its index.
    pub(super) const fn at(&self, stage: usize, mut block: usize) -> u128 {
        let mut t = self.shifts[stage];
        while block != 0 {
            t ^= self.basis[block.trailing_zeros() as usize];
            block &= block - 1;
        }
        t
    }

    /// The Cantor basis vectors a transform of `2^log_n` rows reads.
    pub(super) fn basis(&self, log_n: usize) -> &[u128] {
        &self.basis[..log_n]
    }
}

/// Whether `elements` of work repay handing it to the pool.
// The serial pool exposes a `const` thread count and the parallel one does not.
#[allow(clippy::missing_const_for_fn)]
fn use_parallel(elements: usize) -> bool {
    let threads = current_num_threads();
    threads > 1 && elements >= 2 * BUTTERFLY_GRAIN * threads
}

/// Run an operation over consecutive chunks, in parallel when `stages` passes over them repay it.
pub(super) fn for_chunks(
    values: &mut [u128],
    chunk_len: usize,
    stages: usize,
    operation: impl Fn((usize, &mut [u128])) + Send + Sync,
) {
    if values.len() > chunk_len && use_parallel(values.len().saturating_mul(stages.max(1))) {
        values
            .par_chunks_mut(chunk_len)
            .enumerate()
            .for_each(operation);
    } else {
        values.chunks_mut(chunk_len).enumerate().for_each(operation);
    }
}

/// Elements one task copies when a coset copy is split across workers.
const COSET_COPY_GRAIN: usize = 1 << 16;

/// Copy one coset's coefficients, split across workers when the copy is large.
///
/// A low rate leaves few cosets to spread, so a large copy is split on its own.
pub(super) fn copy_coset(dst: &mut [u128], src: &[u128]) {
    if dst.len() > COSET_COPY_GRAIN && use_parallel(dst.len()) {
        dst.par_chunks_mut(COSET_COPY_GRAIN)
            .zip(src.par_chunks(COSET_COPY_GRAIN))
            .for_each(|(d, s)| d.copy_from_slice(s));
    } else {
        dst.copy_from_slice(src);
    }
}

/// A change of basis applied to a whole run of elements.
///
/// The kernel behind it converts several elements at once, so a run must reach it unbroken.
pub(super) type Conversion = fn(&mut [u128]);

/// Tower-basis bit patterns to polynomial coordinates.
pub(super) const INTO_POLY: Conversion = poly_basis::from_tower_slice;

/// Polynomial coordinates to tower-basis bit patterns.
pub(super) const INTO_TOWER: Conversion = poly_basis::to_tower_slice;

/// Change the basis of a whole matrix, in a pass of its own.
pub(super) fn convert(values: &mut [u128], conversion: Conversion) {
    // A conversion does several dependent lookups per element, so it repays the pool sooner than a butterfly.
    //
    // A task is a whole grain, so every task still reaches the blocked kernel.
    if use_parallel(values.len().saturating_mul(4)) {
        values.par_chunks_mut(BUTTERFLY_GRAIN).for_each(conversion);
    } else {
        conversion(values);
    }
}

/// Where the two basis changes ride, instead of taking a pass over the matrix each.
///
/// A conversion is a per-element map, so it commutes with every butterfly ordering.
///
/// Applied where a pass already holds the element in cache, it costs the lookups alone.
#[derive(Copy, Clone, Debug)]
pub(super) struct Fold {
    /// Convert out of the tower basis where the transform first reads each element.
    pub(super) entry: bool,
    /// Convert back into the tower basis where the transform last writes each element.
    pub(super) exit: bool,
}

impl Fold {
    /// Tower-basis values in, tower-basis values out.
    pub(super) const BOTH: Self = Self {
        entry: true,
        exit: true,
    };
    /// Tower-basis values in, polynomial coordinates out.
    pub(super) const ENTRY: Self = Self {
        entry: true,
        exit: false,
    };
    /// Polynomial coordinates in, tower-basis values out.
    pub(super) const EXIT: Self = Self {
        entry: false,
        exit: true,
    };
}

/// One stage over the whole matrix, cut into tasks that each seed their own twiddle.
pub(super) fn stage(
    values: &mut [u128],
    half: usize,
    j: usize,
    twiddles: &Twiddles,
    inverse: bool,
) {
    if !use_parallel(values.len()) {
        local_stage(values, half, j, twiddles, inverse, 0);
        return;
    }
    let blocks_per_chunk = (BUTTERFLY_GRAIN / (half << 1)).max(1);
    values
        .par_chunks_mut((half << 1) * blocks_per_chunk)
        .enumerate()
        .for_each(|(chunk_index, chunk)| {
            let first = chunk_index * blocks_per_chunk;
            let mut t = twiddles.at(j, first);
            for (index, block) in chunk.chunks_mut(half << 1).enumerate() {
                let (lo, hi) = block.split_at_mut(half);
                let butterfly = |lo: &mut [u128], hi: &mut [u128]| {
                    if inverse {
                        poly_basis::butterfly_inverse(lo, hi, t);
                    } else {
                        poly_basis::butterfly_forward(lo, hi, t);
                    }
                };

                // Pairs are independent, so a block wider than the grain is split across workers.
                if half <= BUTTERFLY_GRAIN {
                    butterfly(lo, hi);
                } else {
                    lo.par_chunks_mut(BUTTERFLY_GRAIN)
                        .zip(hi.par_chunks_mut(BUTTERFLY_GRAIN))
                        .for_each(|(lo, hi)| butterfly(lo, hi));
                }

                // Step to the next block's twiddle.
                t ^= twiddles.deltas[(first + index).trailing_ones() as usize];
            }
        });
}

/// One stage over a run of blocks on one worker, starting at global block `first`.
fn local_stage(
    values: &mut [u128],
    half: usize,
    j: usize,
    twiddles: &Twiddles,
    inverse: bool,
    first: usize,
) {
    let mut t = twiddles.at(j, first);
    for (index, block) in values.chunks_mut(half << 1).enumerate() {
        let (lo, hi) = block.split_at_mut(half);
        if inverse {
            poly_basis::butterfly_inverse(lo, hi, t);
        } else {
            poly_basis::butterfly_forward(lo, hi, t);
        }

        // Step to the next block's twiddle.
        t ^= twiddles.deltas[(first + index).trailing_ones() as usize];
    }
}

/// Run the `depth` stages a tile of `2^depth` rows of `row` elements is closed under.
///
/// - Sub-layer `s` pairs rows `2^(depth-1-s)` apart, in `2^s` blocks of one twiddle each.
/// - Globally those are blocks `block * 2^s + g` of stage `top - 1 - s`.
/// - A tile row is one matrix row in a contiguous tile, and a run of them in a staging tile.
fn tile_stages(
    tile: &mut [u128],
    row: usize,
    depth: usize,
    top: usize,
    twiddles: &Twiddles,
    inverse: bool,
    block: usize,
) {
    for k in 0..depth {
        // Forward runs the widest sub-layer first, inverse the narrowest.
        let s = if inverse { depth - 1 - k } else { k };
        local_stage(
            tile,
            (1 << (depth - 1 - s)) * row,
            top - 1 - s,
            twiddles,
            inverse,
            block << s,
        );
    }
}

/// Run the bottom stages inside each contiguous tile before leaving it.
pub(super) fn local_stages(
    values: &mut [u128],
    plan: Plan,
    twiddles: &Twiddles,
    inverse: bool,
    fold: Fold,
) {
    let Plan {
        width,
        log_n,
        local,
        ..
    } = plan;
    let tile_len = (1 << local) * width;

    // A single column pairs fewer elements than a register holds in its lowest stages.
    //
    // A forward tile then runs those stages together, one register set per run of rows.
    let low = (!inverse && width == 1 && local >= LOW_STAGES)
        .then(|| LowStageTwiddles::new(&twiddles.shifts, twiddles.basis(log_n)));
    for_chunks(values, tile_len, local, |(index, tile)| {
        // The tile is each element's first read when it runs before every other pass.
        if fold.entry {
            INTO_POLY(tile);
        }
        if let Some(low) = &low {
            // The stages above the low ones read a run of adjacent rows as one row.
            let above = local - LOW_STAGES;
            tile_stages(tile, 1 << LOW_STAGES, above, local, twiddles, false, index);
            low.forward(tile, index << above);
        } else {
            tile_stages(tile, width, local, local, twiddles, inverse, index);
        }

        // And each element's last write when it runs after every other pass.
        if fold.exit {
            INTO_TOWER(tile);
        }
    });
}

/// Run stages `top - 1` down to `top - depth` through one staging tile per worker.
///
/// # Algorithm
///
/// A gather addresses runs of `L = 2^log_block` adjacent rows, which rereads the matrix with wider rows:
///
/// ```text
///     width * 2^log_n  =  (L * width) * 2^(log_n - log_block)
/// ```
///
/// - Stage `j >= log_block` of the matrix is stage `j - log_block` of the reread one.
/// - A run must fit the stride the walk takes, hence `log_block <= top - depth`.
///
/// With `S = 2^(top - log_block - depth)` and `offset < S`, one closed set of reread rows is
///
/// ```text
///     row(k) = block * 2^(top - log_block) + offset + k * S ,     k = 0 .. 2^depth
/// ```
///
/// - Stage `top - 1 - s` pairs those rows `2^(depth - 1 - s)` apart in `k`, so the set is closed.
/// - The `L` matrix rows of one run lie in the same block at every sub-layer, so they share a twiddle.
/// - Sub-layer `s` sees block `block * 2^s + (k >> (depth - s))`, what a contiguous tile would see.
pub(super) fn fused_stages(
    values: &mut [u128],
    plan: Plan,
    top: usize,
    depth: usize,
    twiddles: &Twiddles,
    inverse: bool,
    convert_basis: bool,
) {
    let (runs, run, dispatch) = staged_group(plan, top, depth, values.len());
    for_each_staged_tile(values, runs, dispatch, |tile, block| {
        // The gather is each element's first read in the first group of a forward transform.
        if convert_basis && !inverse {
            INTO_POLY(tile);
        }
        tile_stages(tile, run, depth, top, twiddles, inverse, block);

        // The scatter is each element's last write in the last group of an inverse transform.
        if convert_basis && inverse {
            INTO_TOWER(tile);
        }
    });
}

/// The runs a group of `depth` stages below `top` stages, their length, and how the pass spreads.
fn staged_group(
    plan: Plan,
    top: usize,
    depth: usize,
    elements: usize,
) -> (StagedRuns, usize, Dispatch) {
    // A group whose stride is shorter than the planned run shortens the run to match.
    let log_block = plan.log_block.min(top - depth);

    // One reread row, in elements.
    let run = plan.width << log_block;

    // One staging tile per worker, not per task, since a task is microseconds of work.
    let dispatch = if use_parallel(elements.saturating_mul(depth)) {
        Dispatch::Parallel { min_len: 1 }
    } else {
        Dispatch::Serial
    };

    // Consecutive staged runs are S = 2^(top - log_block - depth) reread rows apart.
    let runs = StagedRuns::new(run, top - log_block - depth, depth);
    (runs, run, dispatch)
}

/// Run the first staging group of every coset of a zero-padded message in one pass.
///
/// - The message, in the tower basis, is the separate source when one is given, and the leading coset otherwise.
/// - One gather of the message serves every coset, and changes its basis once.
/// - Each coset runs the group on its own copy of the tile and scatters it into its own rows.
///
/// No pass converts the message or copies it into the cosets beforehand.
pub(super) fn first_group_into_cosets(
    values: &mut [u128],
    source: Option<&[u128]>,
    message_len: usize,
    plan: Plan,
    depth: usize,
    twiddles: &[Twiddles],
) {
    let top = plan.log_n;
    let (runs, run, dispatch) = staged_group(plan, top, depth, values.len());
    // Stream 64-byte cache-line stores when the matrix exceeds shared cache: the first shared
    // pass reads every coset back from memory, so retaining this scatter in cache adds no reuse.
    let store = if cfg!(all(target_arch = "x86_64", target_feature = "avx512f"))
        && size_of_val(values) > SHARED_CACHE_BYTES
    {
        Store::Streamed
    } else {
        Store::Cached
    };
    for_each_staged_tile_into_cosets_stored(
        values,
        source,
        message_len,
        runs,
        dispatch,
        store,
        INTO_POLY,
        |tile, block, coset| tile_stages(tile, run, depth, top, &twiddles[coset], false, block),
    );
}

/// Encode a zero-padded message whose cosets all share a first staging group of `depth` stages.
///
/// - The message, in the tower basis, is the separate source when one is given, and the leading coset otherwise.
/// - Every coset comes back evaluated in the tower basis.
pub(super) fn padded_sharing_first_group(
    values: &mut [u128],
    source: Option<&[u128]>,
    plan: Plan,
    depth: usize,
    log_inv_rate: usize,
) {
    let log_message = plan.log_n;
    let len = values.len() >> log_inv_rate;

    // Coset c starts at domain point c * 2^log_message, which is its shift.
    let twiddles: Vec<Twiddles> = (0..1 << log_inv_rate)
        .map(|c| Twiddles::new(log_message, domain_point(c << log_message)))
        .collect();

    // The cosets the first group's scatter writes first hold zeros.
    //
    // A contiguous sweep faults their pages in beforehand.
    //
    // The leading coset is one of them unless it holds the message.
    let first_written = if source.is_some() { 0 } else { len };
    prefault(&mut values[first_written..], 0);
    first_group_into_cosets(values, source, len, plan, depth, &twiddles);

    // Each coset finishes below the shared group on its own rows.
    for_chunks(values, len, log_message, |(c, coset)| {
        forward_below(coset, plan, &twiddles[c], 1, Fold::EXIT);
    });
}

/// Forward transform over the coset `shift + S_log_n`, in place.
pub(super) fn forward(values: &mut [u128], plan: Plan, shift: BinaryField128, fold: Fold) {
    forward_below(values, plan, &Twiddles::new(plan.log_n, shift), 0, fold);
}

/// The forward passes below the first `done` staging groups, which have already run.
pub(super) fn forward_below(
    values: &mut [u128],
    plan: Plan,
    twiddles: &Twiddles,
    done: usize,
    fold: Fold,
) {
    let Plan {
        width,
        log_n,
        local,
        ..
    } = plan;

    // A group that has already run was every element's first read.
    debug_assert!(
        done == 0 || !fold.entry,
        "the entry conversion rides on the first group"
    );

    // Peel the staging groups from the top stage down.
    //
    // The first group's gather is every element's first read, so it carries the entry conversion.
    let mut entry = fold.entry;
    let mut top = log_n;
    for (group, take) in plan.group_sizes().enumerate() {
        if group >= done {
            fused_stages(values, plan, top, take, twiddles, false, entry);
            entry = false;
        }
        top -= take;
    }

    // A plain pass carries no per-element map, so a conversion still owed takes a pass of its own.
    if entry && top > local {
        convert(values, INTO_POLY);
        entry = false;
    }
    for j in (local..top).rev() {
        stage(values, (1 << j) * width, j, twiddles, false);
    }

    // The contiguous tiles finish the bottom stages, as every element's last write.
    local_stages(
        values,
        plan,
        twiddles,
        false,
        Fold {
            entry,
            exit: fold.exit,
        },
    );
}

/// Inverse transform over the coset `shift + S_log_n`, in place.
pub(super) fn inverse(values: &mut [u128], plan: Plan, shift: BinaryField128, fold: Fold) {
    let Plan {
        width,
        log_n,
        local,
        ..
    } = plan;
    let twiddles = Twiddles::new(log_n, shift);
    let leftover = plan.leftover();

    // The contiguous tiles run first, so they are every element's first read.
    //
    // They are its last write too when no stage runs above them.
    let tile_exit = fold.exit && local == log_n;
    let mut exit = fold.exit && !tile_exit;
    local_stages(
        values,
        plan,
        &twiddles,
        true,
        Fold {
            entry: fold.entry,
            exit: tile_exit,
        },
    );

    // The leftover plain passes run before the groups, so the top group can carry the exit conversion.
    for j in local..local + leftover {
        stage(values, (1 << j) * width, j, &twiddles, true);
    }

    // The groups are listed from the top down, so the inverse walks the list backwards.
    //
    // The list is no longer than the stage count.
    let mut sizes = [0; usize::BITS as usize];
    let mut count = 0;
    for take in plan.group_sizes() {
        sizes[count] = take;
        count += 1;
    }
    let mut base = local + leftover;
    for (index, take) in sizes[..count].iter().rev().enumerate() {
        base += take;
        let last = index + 1 == count;
        fused_stages(values, plan, base, *take, &twiddles, true, exit && last);
        exit &= !last;
    }

    // No group ran, so the exit conversion takes a pass of its own.
    if exit {
        convert(values, INTO_TOWER);
    }
}
