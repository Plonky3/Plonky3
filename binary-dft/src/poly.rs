//! The Lin–Chung–Han transform carried out in the polynomial basis of `GF(2^128)`.

use alloc::vec::Vec;

use p3_binary_field::{BinaryField128, TowerLevel, poly_basis};
use p3_field::PrimeCharacteristicRing;
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;
use p3_util::{log2_floor_usize, log2_strict_usize};

use crate::domain::domain_point;
use crate::lch::BUTTERFLY_GRAIN;
use crate::traits::AdditiveNtt;

/// [`LchNtt`](crate::LchNtt) over `BinaryField128`, with the data held in the polynomial basis throughout.
///
/// A tower-basis product converts both operands into the polynomial basis and the result back,
/// sixteen dependent table lookups apiece, which is most of what a butterfly costs. Converting
/// the whole matrix once on the way in and once on the way out pays `2` conversions per element
/// instead of `3ℓ/2`, and every twiddle multiply in between is a bare carryless multiply and a
/// reduction. Additions are `XOR` in both bases, so they are unaffected.
///
/// Without a carryless-multiply instruction that product is a bit-serial loop and slower than
/// the tower arithmetic it replaces, so on such a target the transform runs in the tower basis
/// instead, using typed subfield multiplication when the twiddle permits it.
/// The choice is a constant and only one arm survives compilation.
#[derive(Clone, Debug, Default)]
pub struct PolyBasisNtt {
    tower: crate::tower::TowerNtt,
}

/// Stage shifts and the XOR increments between consecutive block twiddles.
struct Twiddles {
    basis: [u128; usize::BITS as usize],
    deltas: [u128; usize::BITS as usize],
    shifts: [u128; usize::BITS as usize],
}

impl Twiddles {
    fn new(log_n: usize, shift: BinaryField128) -> Self {
        let mut result = Self {
            basis: [0; usize::BITS as usize],
            deltas: [0; usize::BITS as usize],
            shifts: [0; usize::BITS as usize],
        };
        let mut delta = 0;
        let mut base = poly_basis::from_tower(shift);
        for j in 0..log_n {
            result.basis[j] = poly_basis::from_tower(BinaryField128::cantor_basis(j + 1));
            delta ^= result.basis[j];
            result.deltas[j] = delta;
            result.shifts[j] = base;
            base = poly_basis::square(base) ^ base;
        }
        result
    }

    const fn at(&self, stage: usize, mut block: usize) -> u128 {
        let mut t = self.shifts[stage];
        while block != 0 {
            t ^= self.basis[block.trailing_zeros() as usize];
            block &= block - 1;
        }
        t
    }
}

/// Schedule enough butterfly-sized work per worker to amortize parallel dispatch.
// The serial dependency exposes a `const` thread-count query, while the parallel
// implementation cannot; keep this shared scheduler callable in both configurations.
#[allow(clippy::missing_const_for_fn)]
fn use_parallel(elements: usize) -> bool {
    let threads = p3_maybe_rayon::prelude::current_num_threads();
    threads > 1 && elements >= 2 * BUTTERFLY_GRAIN * threads
}

fn for_chunks(
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

/// The polynomial-basis coordinates of an element held in the tower basis.
#[inline]
fn into_poly(value: u128) -> u128 {
    poly_basis::from_tower(BinaryField128::from_repr(value))
}

/// The tower-basis bit pattern of an element held in polynomial coordinates.
#[inline]
fn into_tower(value: u128) -> u128 {
    poly_basis::to_tower(value).to_repr()
}

fn convert(values: &mut [u128], conversion: impl Fn(u128) -> u128 + Send + Sync) {
    // Basis conversion does several dependent lookups per element, more work than
    // a butterfly, so it amortizes dispatch at a smaller byte volume.
    if use_parallel(values.len().saturating_mul(4)) {
        values
            .par_iter_mut()
            .for_each(|value| *value = conversion(*value));
    } else {
        values
            .iter_mut()
            .for_each(|value| *value = conversion(*value));
    }
}

/// Apply a per-element map to a tile the schedule is already holding in cache.
fn convert_tile(tile: &mut [u128], conversion: impl Fn(u128) -> u128) {
    for value in tile {
        *value = conversion(*value);
    }
}

/// Where the two basis conversions ride, instead of taking a pass over the matrix each.
///
/// A conversion is a pure per-element map, so it commutes with every butterfly ordering.
///
/// Applied where the schedule already holds the element in cache it costs the lookups alone,
/// and no memory traffic at all.
#[derive(Copy, Clone, Debug)]
struct Fold {
    /// Map out of the tower basis where the transform first reads each element.
    entry: bool,
    /// Map back into the tower basis where the transform last writes each element.
    exit: bool,
}

impl Fold {
    /// Take tower-basis values and hand tower-basis values back.
    const BOTH: Self = Self {
        entry: true,
        exit: true,
    };
    /// Take tower-basis values and leave the result in polynomial coordinates.
    const ENTRY: Self = Self {
        entry: true,
        exit: false,
    };
    /// Take polynomial coordinates and hand tower-basis values back.
    const EXIT: Self = Self {
        entry: false,
        exit: true,
    };
}

/// A stage uses bounded chunks, each starting from its own independently indexed twiddle.
fn stage(values: &mut [u128], half: usize, j: usize, twiddles: &Twiddles, inverse: bool) {
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
                if half <= BUTTERFLY_GRAIN {
                    butterfly(lo, hi);
                } else {
                    lo.par_chunks_mut(BUTTERFLY_GRAIN)
                        .zip(hi.par_chunks_mut(BUTTERFLY_GRAIN))
                        .for_each(|(lo, hi)| butterfly(lo, hi));
                }
                t ^= twiddles.deltas[(first + index).trailing_ones() as usize];
            }
        });
}

// A conservative tile budget; the row count scales with the element size and matrix width.
const TILE_BYTES: usize = 32 * 1024;

/// Bytes one worker's staging tile may occupy.
///
/// The tile is streamed rather than randomly addressed, so it need not fit in L1.
/// What it must not do is spill out of the private cache level below the shared one.
///
/// A sweep of the fusion depth at width 16 puts the optimum at 64 KiB, which fuses 8 stages.
/// That is exactly the per-core private cache size of the machine the sweep ran on.
///
/// A 1 MiB tile fuses 12 stages and runs a third slower.
const STAGING_BYTES: usize = 64 * 1024;

/// The shape of one transform, and the two places its stage sequence is cut.
///
/// A stage pairs rows a power of two apart, so the whole matrix has to be traversed once per
/// stage unless a set of rows closed under several stages can be brought into cache.
///
/// There are two such sets, and one cut point each:
///
/// ```text
///     stages log_n-1 .. local+leftover   rows far apart, gathered into a staging tile
///     stages local+leftover-1 .. local   rows far apart, one plain full pass each
///     stages local-1 .. 0                rows already adjacent, so a contiguous tile holds them
/// ```
///
/// The middle band holds the stages a staging tile would not pay for.
///
/// It is one stage wide at most, unless the staging tile cannot hold two rows.
///
/// The count of contiguous-tile stages never exceeds the count of stages there are.
#[derive(Copy, Clone, Debug)]
struct Plan {
    /// Elements per row.
    width: usize,
    /// Base-two logarithm of the row count.
    log_n: usize,
    /// Bottom stages that run to completion inside one contiguous tile of rows.
    local: usize,
    /// Long-stride stages that one staging tile fuses into a single pass over the matrix.
    depth: usize,
}

impl Plan {
    /// The cut points a matrix of this shape gets, from the two cache budgets.
    fn new(width: usize, log_n: usize) -> Self {
        let element = core::mem::size_of::<u128>();
        let tile_rows = (TILE_BYTES / element / width).max(1);
        let staging_rows = (STAGING_BYTES / element / width).max(1);
        Self {
            width,
            log_n,
            local: log2_floor_usize(tile_rows).min(log_n),
            depth: log2_floor_usize(staging_rows),
        }
    }

    /// Stages above the contiguous tile that run as plain full passes.
    ///
    /// Fusing a single stage would move the same bytes the stage moves on its own, plus the
    /// copy in and out of the staging tile, so one stage is left over rather than fused.
    ///
    /// A staging tile too narrow to hold two rows leaves every stage over.
    const fn leftover(&self) -> usize {
        let above = self.log_n - self.local;
        if self.depth < 2 {
            above
        } else if above % self.depth == 1 {
            1
        } else {
            0
        }
    }
}

/// A tile stays on one worker across its adjacent stages.
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
        t ^= twiddles.deltas[(first + index).trailing_ones() as usize];
    }
}

/// Run the `depth` stages that a tile of `2^depth` consecutive rows holds.
///
/// The tile is a radix-2 network on its rows.
///
/// Sub-layer `s` pairs rows `2^(depth-1-s)` apart and splits the tile into `2^s` blocks of
/// `2^(depth-s)` rows, each block carrying one twiddle.
///
/// Globally those blocks are the blocks `block * 2^s + g`, `g = 0 .. 2^s`, of stage
/// `top - 1 - s`, so the twiddle walk starts at `block << s`.
fn tile_stages(
    tile: &mut [u128],
    width: usize,
    depth: usize,
    top: usize,
    twiddles: &Twiddles,
    inverse: bool,
    block: usize,
) {
    for k in 0..depth {
        // The forward direction runs the widest sub-layer first, the inverse the narrowest.
        let s = if inverse { depth - 1 - k } else { k };
        local_stage(
            tile,
            (1 << (depth - 1 - s)) * width,
            top - 1 - s,
            twiddles,
            inverse,
            block << s,
        );
    }
}

/// Complete the stages confined to one cache-sized set of rows before leaving it.
fn local_stages(values: &mut [u128], plan: Plan, twiddles: &Twiddles, inverse: bool, fold: Fold) {
    let Plan { width, local, .. } = plan;
    let tile_len = (1 << local) * width;
    for_chunks(values, tile_len, local, |(index, tile)| {
        // The tile is the first read of every element it holds when it runs before every
        // other stage, and the last write when it runs after them.
        if fold.entry {
            convert_tile(tile, into_poly);
        }
        tile_stages(tile, width, local, local, twiddles, inverse, index);
        if fold.exit {
            convert_tile(tile, into_tower);
        }
    });
}

/// A raw handle to the matrix, so tasks that own rows spaced apart can run side by side.
///
/// Slice splitters cut a slice into contiguous pieces only, and the rows one staging tile
/// gathers are a power of two apart, so the tasks share this handle and address their own
/// rows through it.
///
/// # Safety
/// The row sets two live tasks address must be disjoint, and the exclusive borrow the base
/// pointer comes from must outlive every task.
#[derive(Copy, Clone)]
struct Rows {
    /// First element of the matrix.
    base: *mut u128,
    /// Elements per row.
    width: usize,
    /// Rows in the matrix.
    count: usize,
}

// SAFETY: the handle is a pointer and two lengths, with no interior mutability and no `Drop`,
// so sending or sharing it moves no data.
//
// The only caller derives the row index of every task from a bijection onto the row range,
// which is what makes concurrent use race-free.
unsafe impl Send for Rows {}
// SAFETY: see the `Send` implementation.
unsafe impl Sync for Rows {}

impl Rows {
    /// Check that a walk of `rows` rows from `first` in steps of `stride` stays inside the
    /// matrix.
    ///
    /// The walk is increasing, so bounding its last row bounds all of them.
    ///
    /// This runs once per tile rather than once per row, which is why it is a hard check and
    /// not a debug one.
    ///
    /// A row index past the end would otherwise be a write past the end of the matrix.
    ///
    /// # Panics
    /// Panics if the last row of the walk is at or beyond the row count.
    fn check(&self, first: usize, stride: usize, rows: usize) {
        assert!(
            rows == 0 || first + (rows - 1) * stride < self.count,
            "staged row walk leaves the matrix"
        );
    }

    /// Copy the rows `first`, `first + stride`, ... into consecutive rows of the tile.
    ///
    /// The tile is emptied first and then grown one row at a time.
    ///
    /// So it holds no element the walk did not write.
    ///
    /// And a worker never has to zero a tile it is about to overwrite in full.
    ///
    /// A tile whose capacity already covers the walk grows without reallocating.
    ///
    /// # Safety
    /// No other live task may address any of the rows the walk names.
    unsafe fn gather(&self, first: usize, stride: usize, rows: usize, tile: &mut Vec<u128>) {
        self.check(first, stride, rows);
        tile.clear();
        for k in 0..rows {
            // SAFETY: the bound above puts every row of the walk inside the matrix, and the
            // exclusive borrow the base pointer came from outlives the task.
            //
            // No other live task addresses this row, so nothing can write it during the read.
            let row = unsafe {
                core::slice::from_raw_parts(
                    self.base.add((first + k * stride) * self.width),
                    self.width,
                )
            };
            tile.extend_from_slice(row);
        }
    }

    /// Write consecutive rows of the tile back over the rows they were gathered from.
    ///
    /// # Safety
    /// No other live task may address any of the rows the walk names.
    unsafe fn scatter(&self, first: usize, stride: usize, tile: &[u128]) {
        let rows = tile.chunks_exact(self.width);
        self.check(first, stride, rows.len());
        for (k, row) in rows.enumerate() {
            // SAFETY: as in the gather, with the direction of the copy reversed.
            unsafe {
                core::ptr::copy_nonoverlapping(
                    row.as_ptr(),
                    self.base.add((first + k * stride) * self.width),
                    self.width,
                );
            }
        }
    }
}

/// Run stages `top - 1` down to `top - depth` through one staging tile per worker.
///
/// Each of those stages pairs rows far apart, so on its own it reads and writes the whole
/// matrix.
///
/// The rows they touch split into small sets that are closed under all of them, so gathering
/// one such set into a contiguous tile runs the whole group inside the cache.
///
/// The matrix is then read once and written once per group instead of once per stage.
///
/// # Algorithm
///
/// With `S = 2^(top-depth)` and `offset < S`, one such set is the `2^depth` rows
///
/// ```text
///     row(k) = block * 2^top + offset + k * S ,     k = 0 .. 2^depth
/// ```
///
/// Stage `top-1-s` pairs rows `2^(top-1-s) = 2^(depth-1-s) * S` apart.
///
/// That is a distance of `2^(depth-1-s)` in `k`, which stays inside the set for every
/// `s < depth`, so the set is closed.
///
/// # Twiddles
///
/// A sub-layer needs the twiddle of the global block its pair lies in.
///
/// For sub-layer `s` that block index is
///
/// ```text
///     row(k) >> (top - s) = block * 2^s + (k >> (depth - s))
/// ```
///
/// Writing `k = q * 2^(depth-s) + r` gives `offset + r * S < 2^(top-s)`, so only
/// `q = k >> (depth-s)` survives the shift.
///
/// That is precisely the block index a contiguous run of `2^(depth-s)` staged rows carries.
///
/// So the tile runs as an ordinary radix-2 network whose twiddle walk starts at `block << s`.
fn fused_stages(
    values: &mut [u128],
    plan: Plan,
    top: usize,
    depth: usize,
    twiddles: &Twiddles,
    inverse: bool,
    convert_basis: bool,
) {
    let width = plan.width;
    let len = values.len();
    // Rows between two consecutive staged rows, and elements in one staging tile.
    let stride = 1 << (top - depth);
    let tile_len = width << depth;
    // One tile per `(block, offset)` pair, which is one tile per `2^depth` rows.
    let tiles = len / tile_len;
    debug_assert_eq!(stride << depth, 1 << top, "staged rows do not span a block");
    debug_assert_eq!(tiles * tile_len, len, "tiles do not partition the matrix");

    let rows = Rows {
        base: values.as_mut_ptr(),
        width,
        count: len / width,
    };
    let task = |tile: &mut Vec<u128>, index: usize| {
        // A tile index splits into the stage-`top` block it lies in and its offset inside
        // the stride, which together with `k` name a row:
        //
        //     index  = block * S + offset
        //     row(k) = block * 2^top + offset + k * S
        let block = index >> (top - depth);
        let first = (block << top) + (index & (stride - 1));
        // SAFETY: `index` runs over `0..tiles` and `k` over `0..2^depth`, so
        // `(block, offset, k) -> row(k)` is a mixed-radix decomposition of `0..2^log_n`.
        //
        // Every row is inside the matrix and belongs to exactly one tile index, hence to
        // exactly one task.
        //
        // The exclusive borrow of the matrix outlives the whole region.
        unsafe { rows.gather(first, stride, 1 << depth, tile) };
        // Invariant: every element of the tile comes from the walk the gather just ran, so
        // nothing below reads an element the gather did not write.
        debug_assert_eq!(tile.len(), tile_len, "the gather left the tile short");
        // The gather is the first read of every element when this is the first group of a
        // forward transform.
        if convert_basis && !inverse {
            convert_tile(tile, into_poly);
        }
        tile_stages(tile, width, depth, top, twiddles, inverse, block);
        // The scatter is the last write of every element when this is the last group of an
        // inverse transform.
        if convert_basis && inverse {
            convert_tile(tile, into_tower);
        }
        // SAFETY: the rows are the ones the gather read, so the argument above applies
        // unchanged.
        unsafe { rows.scatter(first, stride, tile) };
    };

    // One staging tile per worker, not per task: a task is a few tens of microseconds of
    // work and the tile is tens of kilobytes.
    //
    // The tile is capacity only, with no initialized elements: the gather grows it from
    // empty, so no worker spends bandwidth zeroing tens of kilobytes it is about to
    // overwrite in full.
    let new_tile = || Vec::with_capacity(tile_len);
    if use_parallel(len.saturating_mul(depth)) {
        (0..tiles).into_par_iter().for_each_init(new_tile, task);
    } else {
        let mut tile = new_tile();
        for index in 0..tiles {
            task(&mut tile, index);
        }
    }
}

/// Forward transform of polynomial-basis values in an existing allocation.
fn forward(values: &mut [u128], plan: Plan, shift: BinaryField128, fold: Fold) {
    let Plan {
        width,
        log_n,
        local,
        depth,
    } = plan;
    let twiddles = Twiddles::new(log_n, shift);
    let leftover = plan.leftover();

    // Peel fused groups from the top stage downwards, each replacing `take` full passes.
    //
    // The first group's gather is the first read of every element, so it carries the entry
    // conversion.
    let mut entry = fold.entry;
    let mut top = log_n;
    while top - local > leftover {
        let take = depth.min(top - local - leftover);
        fused_stages(values, plan, top, take, &twiddles, false, entry);
        entry = false;
        top -= take;
    }

    // A plain pass carries no per-element map, so a conversion still owed ahead of one takes
    // a pass of its own.
    if entry && top > local {
        convert(values, into_poly);
        entry = false;
    }
    for j in (local..top).rev() {
        stage(values, (1 << j) * width, j, &twiddles, false);
    }

    // The contiguous tile finishes the bottom stages, and is the last write of every
    // element.
    local_stages(
        values,
        plan,
        &twiddles,
        false,
        Fold {
            entry,
            exit: fold.exit,
        },
    );
}

/// Inverse transform with the data kept in the polynomial basis.
fn inverse(values: &mut [u128], plan: Plan, shift: BinaryField128, fold: Fold) {
    let Plan {
        width,
        log_n,
        local,
        depth,
    } = plan;
    let twiddles = Twiddles::new(log_n, shift);
    let leftover = plan.leftover();

    // The contiguous tile runs first, so it is the first read of every element, and the last
    // write too when no stage runs above it.
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

    // Stages the staging tile does not pay for go before the groups, so a group still
    // reaches the top stage and can carry the exit conversion.
    for j in local..local + leftover {
        stage(values, (1 << j) * width, j, &twiddles, true);
    }
    let mut base = local + leftover;
    while base < log_n {
        let take = depth.min(log_n - base);
        base += take;
        let last = base == log_n;
        fused_stages(values, plan, base, take, &twiddles, true, exit && last);
        exit &= !last;
    }

    // No group ran, so the exit conversion needs a pass of its own.
    if exit {
        convert(values, into_tower);
    }
}

impl AdditiveNtt<BinaryField128> for PolyBasisNtt {
    fn shifted_ntt_batch(
        &self,
        mut mat: RowMajorMatrix<BinaryField128>,
        shift: BinaryField128,
    ) -> RowMajorMatrix<BinaryField128> {
        if !poly_basis::HAS_HARDWARE_CLMUL {
            return self.tower.shifted_ntt_batch(mat, shift);
        }

        let width = mat.width();
        let log_n = log2_strict_usize(mat.height());
        // `BinaryField128` is `#[repr(transparent)]` over `u128`, so this reuses `mat.values`'s
        // allocation instead of allocating a second codeword buffer: `to_repr` is the identity
        // bit pattern, and the specialised `Vec` collect below reinterprets the buffer in place.
        let mut values: Vec<u128> = core::mem::take(&mut mat.values)
            .into_iter()
            .map(BinaryField128::to_repr)
            .collect();

        // Both conversions ride along with the transform's first and last touch of each
        // element, so neither costs a pass over the matrix.
        forward(&mut values, Plan::new(width, log_n), shift, Fold::BOTH);

        mat.values = values.into_iter().map(BinaryField128::from_repr).collect();
        mat
    }

    fn ntt_batch_padded(
        &self,
        mut mat: RowMajorMatrix<BinaryField128>,
        log_inv_rate: usize,
    ) -> RowMajorMatrix<BinaryField128> {
        let log_n = log2_strict_usize(mat.height());
        assert!(log_inv_rate <= log_n, "padding exceeds matrix height");
        if log_inv_rate == 0 || !poly_basis::HAS_HARDWARE_CLMUL {
            return self.ntt_batch(mat);
        }
        let width = mat.width;
        let log_message = log_n - log_inv_rate;
        let len = mat.values.len() >> log_inv_rate;
        let mut values: Vec<u128> = core::mem::take(&mut mat.values)
            .into_iter()
            .map(BinaryField128::to_repr)
            .collect();
        let plan = Plan::new(width, log_message);
        let (message, tail) = values.split_at_mut(len);
        // Every coset starts from a copy of the message, so one conversion of the message
        // alone serves all `2^log_inv_rate` of them.
        //
        // Converting each copy instead would repeat the sixteen dependent lookups per element
        // once per coset, for nothing.
        convert(message, into_poly);
        // Only the conversion back is left, and each coset's contiguous tile carries its own.
        if len >= 2 * BUTTERFLY_GRAIN * p3_maybe_rayon::prelude::current_num_threads() {
            // Keep large coefficient copies next to evaluation so the copied data
            // is still hot, including when only one worker is available.
            for_chunks(tail, len, log_message, |(c, chunk)| {
                chunk.copy_from_slice(message);
                forward(
                    chunk,
                    plan,
                    domain_point((c + 1) << log_message),
                    Fold::EXIT,
                );
            });
            forward(message, plan, BinaryField128::ZERO, Fold::EXIT);
        } else {
            // Small cosets can run together after all coefficient copies are made.
            for_chunks(tail, len, 1, |(_, chunk)| chunk.copy_from_slice(message));
            for_chunks(&mut values, len, log_message, |(c, chunk)| {
                forward(chunk, plan, domain_point(c << log_message), Fold::EXIT);
            });
        }
        mat.values = values.into_iter().map(BinaryField128::from_repr).collect();
        mat
    }

    fn shifted_lde_batch(
        &self,
        mut mat: RowMajorMatrix<BinaryField128>,
        added_bits: usize,
        shift: BinaryField128,
    ) -> RowMajorMatrix<BinaryField128> {
        if !poly_basis::HAS_HARDWARE_CLMUL {
            return self.tower.shifted_lde_batch(mat, added_bits, shift);
        }
        let log_n = log2_strict_usize(mat.height());
        if added_bits == 0 {
            return mat;
        }
        let width = mat.width;
        let len = mat.values.len();
        let padded_len = u32::try_from(added_bits)
            .ok()
            .and_then(|bits| len.checked_shl(bits))
            .filter(|&padded| padded >> added_bits == len)
            .expect("extended codeword length overflows usize");
        // Preserve the evaluation prefix in its final allocation, then reuse the
        // original allocation for coefficients instead of cloning it before a resize.
        let mut coeffs: Vec<u128> = core::mem::take(&mut mat.values)
            .into_iter()
            .map(BinaryField128::to_repr)
            .collect();
        let mut values = alloc::vec![0u128; padded_len];
        values[..len].copy_from_slice(&coeffs);
        // The coefficients stay in the polynomial basis: every coset transform starts from a
        // copy of them.
        let plan = Plan::new(width, log_n);
        inverse(&mut coeffs, plan, shift, Fold::ENTRY);

        // The input evaluations already are the first coset. Only new cosets need
        // evaluation and conversion back from the polynomial basis.
        for_chunks(&mut values[len..], len, log_n, |(c, chunk)| {
            chunk.copy_from_slice(&coeffs);
            forward(
                chunk,
                plan,
                shift + domain_point::<BinaryField128>((c + 1) << log_n),
                Fold::EXIT,
            );
        });
        mat.values = values.into_iter().map(BinaryField128::from_repr).collect();
        mat
    }

    fn shifted_intt_batch(
        &self,
        mut mat: RowMajorMatrix<BinaryField128>,
        shift: BinaryField128,
    ) -> RowMajorMatrix<BinaryField128> {
        if !poly_basis::HAS_HARDWARE_CLMUL {
            return self.tower.shifted_intt_batch(mat, shift);
        }

        let width = mat.width();
        let log_n = log2_strict_usize(mat.height());
        // See `shifted_ntt_batch`: reuses `mat.values`'s allocation instead of a second buffer.
        let mut values: Vec<u128> = core::mem::take(&mut mat.values)
            .into_iter()
            .map(BinaryField128::to_repr)
            .collect();

        // See the forward transform: neither conversion costs a pass of its own.
        inverse(&mut values, Plan::new(width, log_n), shift, Fold::BOTH);

        mat.values = values.into_iter().map(BinaryField128::from_repr).collect();
        mat
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use p3_binary_field::{BinaryField128, TowerLevel, poly_basis};
    use p3_matrix::dense::RowMajorMatrix;
    use proptest::prelude::*;

    use super::{Fold, Plan, PolyBasisNtt};
    use crate::lch::LchNtt;
    use crate::naive::NaiveAdditiveNtt;
    use crate::traits::AdditiveNtt;

    /// Cut points small enough to keep the test matrices tiny, one pair per branch of the
    /// schedule:
    ///
    /// ```text
    ///     (2, 3)  full groups, plus a one-stage leftover at local + depth + 1
    ///     (2, 0)  a staging tile too narrow for two rows, so no group runs at all
    ///     (0, 3)  no contiguous tile, so every stage is fused
    ///     (1, 1)  a depth of one, which never pays for a tile
    /// ```
    const CUTS: [(usize, usize); 4] = [(2, 3), (2, 0), (0, 3), (1, 1)];

    /// Cuts and heights whose staging groups number three or more, with the group depths
    /// each one produces:
    ///
    /// ```text
    ///     (0, 3, 8)   3 + 3 + 2       a short last group, no contiguous tile
    ///     (0, 3, 9)   3 + 3 + 3       three groups of full depth
    ///     (1, 2, 8)   2 + 2 + 2       three groups and a one-stage leftover
    ///     (0, 2, 8)   2 + 2 + 2 + 2   four groups of full depth
    /// ```
    ///
    /// The staging loop reseeds its twiddle walk and its conversion flag on every turn, so
    /// only a third turn shows that the reseeding is not accidentally right for two.
    const DEEP_CUTS: [(usize, usize, usize); 4] = [(0, 3, 8), (0, 3, 9), (1, 2, 8), (0, 2, 8)];

    /// Widths that cover a single element per row, an odd row, and rows of several elements.
    const WIDTHS: [usize; 4] = [1, 3, 16, 64];

    /// Neither basis conversion rides along, so only the stage schedule is under test.
    const NONE: Fold = Fold {
        entry: false,
        exit: false,
    };

    /// A shift with bits in both halves, so no twiddle is accidentally zero.
    fn test_shift() -> BinaryField128 {
        BinaryField128::from_repr((1 << 127) | 7919)
    }

    /// Values whose bits depend on the position, so a misapplied twiddle cannot cancel out.
    fn coefficients(log_n: usize, width: usize) -> Vec<u128> {
        (0..(width << log_n))
            .map(|i| {
                let low = (i as u64)
                    .wrapping_mul(0x9e37_79b9_7f4a_7c15)
                    .wrapping_add(0x5555_5555_5555_5555);
                let high = low.rotate_left(17).wrapping_mul(0xc2b2_ae3d_27d4_eb4f);
                (u128::from(high) << 64) | u128::from(low)
            })
            .collect()
    }

    /// One full pass per stage, straight off the twiddle accessor: the schedule every cut of
    /// the stage sequence has to reproduce bit for bit.
    fn per_stage_schedule(
        values: &mut [u128],
        width: usize,
        log_n: usize,
        shift: BinaryField128,
        inverse: bool,
    ) {
        let twiddles = super::Twiddles::new(log_n, shift);
        for k in 0..log_n {
            // The forward direction runs the widest stage first, the inverse the narrowest.
            let j = if inverse { k } else { log_n - 1 - k };
            // Stage `j` pairs rows `2^j` apart, so a block spans `2^(j+1)` rows and the
            // block index alone picks the twiddle.
            let half = (1 << j) * width;
            for (block, rows) in values.chunks_mut(half << 1).enumerate() {
                let t = twiddles.at(j, block);
                let (lo, hi) = rows.split_at_mut(half);
                if inverse {
                    poly_basis::butterfly_inverse(lo, hi, t);
                } else {
                    poly_basis::butterfly_forward(lo, hi, t);
                }
            }
        }
    }

    /// Run the scheduled transform of one direction in place.
    fn scheduled(values: &mut [u128], plan: Plan, inverse: bool, fold: Fold) {
        if inverse {
            super::inverse(values, plan, test_shift(), fold);
        } else {
            super::forward(values, plan, test_shift(), fold);
        }
    }

    /// Builds a matrix whose entries are distinct functions of the seed and the position.
    fn matrix(log_n: usize, width: usize, seed: u64) -> RowMajorMatrix<BinaryField128> {
        RowMajorMatrix::new(
            (0..(width << log_n))
                .map(|i| {
                    let bits = seed
                        .wrapping_mul(0x9e37_79b9_7f4a_7c15)
                        .wrapping_add(i as u64);
                    BinaryField128::from_le_byte_iter(bits.to_le_bytes().into_iter().cycle())
                })
                .collect(),
            width,
        )
    }

    /// One plan per branch of the cut, at every width.
    ///
    /// The synthetic heights are the branch boundaries of the cut:
    ///
    /// ```text
    ///     local            nothing above the tile, so the tile is the whole transform
    ///     local + 1        one stage above the tile, which is left unfused
    ///     local + depth    one full staging group
    ///     local + depth+1  a full group and a one-stage leftover
    ///     2*local + depth  several groups and several tiles
    /// ```
    ///
    /// The deep cuts then carry the heights that need three or more groups.
    fn cut_plans() -> Vec<Plan> {
        let mut plans = Vec::new();
        for (local, depth) in CUTS {
            for log_n in [
                local,
                local + 1,
                local + depth,
                local + depth + 1,
                2 * local + depth,
            ] {
                for width in WIDTHS {
                    plans.push(Plan {
                        width,
                        log_n,
                        local: local.min(log_n),
                        depth,
                    });
                }
            }
        }
        for (local, depth, log_n) in DEEP_CUTS {
            for width in WIDTHS {
                plans.push(Plan {
                    width,
                    log_n,
                    local,
                    depth,
                });
            }
        }
        plans
    }

    #[test]
    fn every_cut_of_the_stage_sequence_matches_the_per_stage_schedule() {
        // Invariant: cutting the stage sequence into staging groups and a contiguous tile is
        // a pure reordering of memory traffic.
        //
        // Every element must come out bit for bit what one full pass per stage produces, in
        // both directions.
        for plan in cut_plans() {
            let Plan { width, log_n, .. } = plan;
            for inverse in [false, true] {
                let mut expected = coefficients(log_n, width);
                let mut actual = expected.clone();
                per_stage_schedule(&mut expected, width, log_n, test_shift(), inverse);
                scheduled(&mut actual, plan, inverse, NONE);
                assert_eq!(actual, expected, "{plan:?} inverse={inverse}");
            }
        }

        // The production cut points, at heights that put the interesting branch on each width.
        //
        // Fixture state, from the two cache budgets:
        //
        //     width    1 @ 2^10   local 10, depth 12   the tile is the whole transform
        //     width    3 @ 2^10   local  9, depth 10   one stage above the tile, unfused
        //     width   16 @ 2^10   local  7, depth  8   one group of three stages
        //     width   64 @ 2^10   local  5, depth  6   one group of five stages
        //     width  512 @ 2^8    local  2, depth  3   two groups of three stages
        //     width 1024 @ 2^7    local  1, depth  2   three groups of two stages
        for (width, log_n) in [(1, 10), (3, 10), (16, 10), (64, 10), (512, 8), (1024, 7)] {
            let plan = Plan::new(width, log_n);
            for inverse in [false, true] {
                let mut expected = coefficients(log_n, width);
                let mut actual = expected.clone();
                per_stage_schedule(&mut expected, width, log_n, test_shift(), inverse);
                scheduled(&mut actual, plan, inverse, NONE);
                assert_eq!(actual, expected, "{plan:?} inverse={inverse}");
            }
        }
    }

    #[test]
    fn folded_conversions_match_standalone_conversion_passes() {
        // Invariant: a basis conversion carried by whichever phase first reads or last writes
        // an element is the same map as a standalone pass over the whole matrix before or
        // after the transform.
        //
        // The cut points decide which phase carries it — a staging group's gather, a
        // staging group's scatter, the contiguous tile, or a pass of its own — so the same
        // set of cuts as the schedule test runs here.
        //
        // A run of three or more groups is what shows that the entry conversion rides the
        // first group only and the exit conversion the last group only.
        for plan in cut_plans() {
            let Plan { width, log_n, .. } = plan;
            for inverse in [false, true] {
                let input = coefficients(log_n, width);

                // Conversion in, then the schedule.
                let mut expected = input.clone();
                super::convert(&mut expected, super::into_poly);
                scheduled(&mut expected, plan, inverse, NONE);
                let mut actual = input.clone();
                scheduled(&mut actual, plan, inverse, Fold::ENTRY);
                assert_eq!(actual, expected, "entry {plan:?} inverse={inverse}");

                // Conversion in, the schedule, conversion out.
                super::convert(&mut expected, super::into_tower);
                let mut actual = input.clone();
                scheduled(&mut actual, plan, inverse, Fold::BOTH);
                assert_eq!(actual, expected, "both {plan:?} inverse={inverse}");

                // The schedule, then conversion out.
                let mut expected = input.clone();
                scheduled(&mut expected, plan, inverse, NONE);
                super::convert(&mut expected, super::into_tower);
                let mut actual = input;
                scheduled(&mut actual, plan, inverse, Fold::EXIT);
                assert_eq!(actual, expected, "exit {plan:?} inverse={inverse}");
            }
        }
    }

    #[test]
    fn padded_transform_matches_naive_at_wide_widths() {
        use p3_field::PrimeCharacteristicRing;
        for width in [1, 4, 16, 64] {
            for added in [0, 1, 2, 3] {
                let mut mat = matrix(4, width, 13);
                mat.values
                    .resize(mat.values.len() << added, BinaryField128::ZERO);
                let expected = NaiveAdditiveNtt::default().ntt_batch(mat.clone());
                assert_eq!(
                    PolyBasisNtt::default().ntt_batch_padded(mat, added),
                    expected
                );
            }
        }
    }

    #[test]
    fn shifted_lde_matches_naive_at_wide_widths() {
        for width in [1, 4, 16, 64] {
            for added in [0, 1, 2, 3] {
                let mat = matrix(4, width, 17);
                let shift = BinaryField128::from_repr(1 << 127);
                let expected =
                    NaiveAdditiveNtt::default().shifted_lde_batch(mat.clone(), added, shift);
                let actual = PolyBasisNtt::default().shifted_lde_batch(mat.clone(), added, shift);
                assert_eq!(actual, expected);
                assert_eq!(&actual.values[..mat.values.len()], &mat.values);
            }
        }
    }

    #[test]
    #[should_panic = "extended codeword length overflows usize"]
    fn lde_rejects_length_overflow() {
        let _ = PolyBasisNtt::default().lde_batch(matrix(1, 1, 0), usize::BITS as usize - 1);
    }

    #[test]
    fn incremental_twiddles_match_independent_domain_points() {
        use p3_binary_field::poly_basis;

        use crate::domain::{domain_point, subspace_polynomial};
        let shift = BinaryField128::from_repr((1 << 127) | 123);
        let twiddles = super::Twiddles::new(usize::BITS as usize - 1, shift);
        for stage in [0, 1, 7, 15, 31]
            .into_iter()
            .filter(|&stage| stage < usize::BITS as usize - 1)
        {
            for start in [0, 1, 63, 127, (1usize << (usize::BITS - 3)) - 3] {
                let mut t = twiddles.at(stage, start);
                for block in start..start + 9 {
                    let expected = subspace_polynomial(stage, shift)
                        + domain_point::<BinaryField128>(block << 1);
                    assert_eq!(t, poly_basis::from_tower(expected));
                    t ^= twiddles.deltas[block.trailing_ones() as usize];
                }
            }
        }
    }

    #[test]
    fn transforms_cross_cache_boundaries_in_natural_order() {
        for width in [1, 3, 16, 64] {
            let mat = matrix(12, width, 29);
            let shift = BinaryField128::from_repr((1 << 127) | 7919);
            let expected = crate::LchNtt::default().shifted_ntt_batch(mat.clone(), shift);
            let actual = PolyBasisNtt::default().shifted_ntt_batch(mat.clone(), shift);
            assert_eq!(actual, expected);
            assert_eq!(
                PolyBasisNtt::default().shifted_intt_batch(actual, shift),
                mat
            );
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(32))]

        /// The polynomial-basis transform is the same map as the reference oracle.
        #[test]
        fn poly_basis_matches_naive(
            log_n in 0usize..=8,
            width in 1usize..=5,
            seed in any::<u64>(),
            shift in any::<u64>(),
        ) {
            let coeffs = matrix(log_n, width, seed);
            let shift = BinaryField128::from_le_byte_iter(
                shift.to_le_bytes().into_iter().cycle(),
            );

            let fast = PolyBasisNtt::default().shifted_ntt_batch(coeffs.clone(), shift);
            let slow = NaiveAdditiveNtt::<BinaryField128>::default()
                .shifted_ntt_batch(coeffs, shift);
            prop_assert_eq!(fast, slow);
        }

        #[test]
        fn poly_basis_round_trips(
            log_n in 0usize..=8,
            width in 1usize..=3,
            seed in any::<u64>(),
            shift in any::<u64>(),
        ) {
            let coeffs = matrix(log_n, width, seed);
            let shift = BinaryField128::from_le_byte_iter(
                shift.to_le_bytes().into_iter().cycle(),
            );
            let ntt = PolyBasisNtt::default();
            let evals = ntt.shifted_ntt_batch(coeffs.clone(), shift);
            prop_assert_eq!(ntt.shifted_intt_batch(evals, shift), coeffs);
        }

        /// `PolyBasisNtt`'s low-degree extension agrees with the oracle's, on a coset too, and
        /// the input rows reappear as the prefix: the correspondence Phase 3 folds along.
        #[test]
        fn poly_basis_lde_matches_naive(
            log_n in 0usize..=6,
            added in 0usize..=3,
            width in 1usize..=3,
            seed in any::<u64>(),
            shift in any::<u64>(),
        ) {
            let coeffs = matrix(log_n, width, seed);
            let shift = BinaryField128::from_le_byte_iter(
                shift.to_le_bytes().into_iter().cycle(),
            );

            let lde = PolyBasisNtt::default().shifted_lde_batch(coeffs.clone(), added, shift);
            let naive = NaiveAdditiveNtt::<BinaryField128>::default()
                .shifted_lde_batch(coeffs.clone(), added, shift);
            prop_assert_eq!(&lde, &naive);
            prop_assert_eq!(&lde.values[..coeffs.values.len()], &coeffs.values[..]);
        }
    }

    /// A height whose stages take more than one butterfly task, so a task seeds its twiddle at
    /// a block index of its own rather than at zero. The oracle tests all sit below that
    /// height, so `LchNtt` stands in for the oracle here, itself held to an independent
    /// twiddle walk at this same height by `lch_matches_a_twiddle_walk_across_several_tasks`.
    #[test]
    fn poly_basis_matches_the_tower_across_several_tasks() {
        const LOG_N: usize = 12;
        let poly = PolyBasisNtt::default();
        let tower = LchNtt::<BinaryField128>::default();
        for width in [1usize, 3] {
            for shift_bits in [0u64, 0x1234_5678_9abc_def0] {
                let coeffs = matrix(LOG_N, width, 5);
                let shift =
                    BinaryField128::from_le_byte_iter(shift_bits.to_le_bytes().into_iter().cycle());

                let evals = tower.shifted_ntt_batch(coeffs.clone(), shift);
                assert_eq!(
                    poly.shifted_ntt_batch(coeffs.clone(), shift),
                    evals,
                    "ntt width={width} shift={shift_bits:#x}"
                );
                assert_eq!(
                    poly.shifted_intt_batch(evals, shift),
                    coeffs,
                    "intt width={width} shift={shift_bits:#x}"
                );
            }
        }
    }

    /// A height that is not a power of two has no well-defined `ℓ`. `ℓ` exceeding the bit width
    /// of `BinaryField128` is covered generically at [`LchNtt`](crate::LchNtt), where the level
    /// is a type parameter and the panic is cheap to reach; at a fixed `BinaryField128` it is
    /// only reachable through a matrix of `2^129` rows, which is not a test worth writing.
    #[test]
    #[should_panic]
    fn shifted_ntt_batch_rejects_a_non_power_of_two_height() {
        let coeffs = RowMajorMatrix::new(matrix(0, 1, 0).values.repeat(3), 1);
        let _ = PolyBasisNtt::default().ntt_batch(coeffs);
    }
}
