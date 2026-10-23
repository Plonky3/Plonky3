//! Identity-padded storage and folding state used by the product prover.

use alloc::borrow::Cow;
use alloc::vec;
use alloc::vec::Vec;
use core::marker::PhantomData;
use core::mem::MaybeUninit;

use p3_field::{ExtensionField, Field, PackedFieldExtension, PackedValue, PrimeCharacteristicRing};
use p3_maybe_rayon::prelude::*;
use p3_multilinear_util::point::Point;

use super::ROUND_POLY_LEN;
use super::math::interpolate_pair;

/// An arbitrary table prefix whose omitted suffix is the multiplicative identity.
///
/// The leaves may stay borrowed from the caller, since the prover only reads them.
struct IdentityPrefix<'a, F: Clone> {
    /// Explicit values after removing trailing identities.
    values: Cow<'a, [F]>,
}

impl<'a, F: Field> IdentityPrefix<'a, F> {
    /// Creates a canonical explicit prefix from a complete or partial table.
    fn new(values: impl Into<Cow<'a, [F]>>) -> Self {
        // Trailing identities have the same semantics when left implicit.
        let mut prefix = Self {
            values: values.into(),
        };
        prefix.trim();
        prefix
    }

    /// Reads one logical value through implicit identity padding.
    #[inline]
    fn get(&self, index: usize) -> F {
        self.values.get(index).copied().unwrap_or(F::ONE)
    }

    /// Multiplies fixed-size groups into the next retained product layer.
    ///
    /// A partial final group is completed by implicit identity factors.
    fn reduce<Base: Field>(&self, arity: usize) -> IdentityPrefix<'static, F>
    where
        F: ExtensionField<Base>,
    {
        let values = &self.values;
        let lanes = Base::Packing::WIDTH;
        if arity != 4 || !(2..=MAX_LANES).contains(&lanes) {
            return IdentityPrefix::new(
                values
                    .par_chunks(arity)
                    .map_collect_min_task_bytes((arity + 1) * size_of::<F>(), |chunk| {
                        chunk.iter().copied().product()
                    }),
            );
        }

        // Rows of four multiply in tasks, each writing its own window of the next level.
        let rows = values.len().div_ceil(4);
        let mut next = Vec::with_capacity(rows);
        next.spare_capacity_mut()[..rows]
            .par_chunks_mut(TASK_BLOCKS * BLOCK)
            .zip(values.par_chunks(4 * TASK_BLOCKS * BLOCK))
            .for_each_min_task_bytes(5 * TASK_BLOCKS * BLOCK * size_of::<F>(), |(out, chunk)| {
                multiply_rows::<Base, F>(chunk, out);
            });
        // SAFETY: the tasks' windows cover the first `rows` slots, and each task wrote all of its own.
        unsafe { next.set_len(rows) };
        IdentityPrefix::new(next)
    }

    /// Restores the canonical shortest representation.
    fn trim(&mut self) {
        let ones = self
            .values
            .iter()
            .rev()
            .take_while(|&&value| value == F::ONE);
        let len = self.values.len() - ones.count();
        match &mut self.values {
            Cow::Borrowed(values) => *values = &values[..len],
            Cow::Owned(values) => values.truncate(len),
        }
    }
}

/// Product levels retained for one identity-padded input prefix.
pub(super) struct ProductLayers<'a, F: Clone> {
    /// Explicit prefixes indexed by their base-two depth above the leaves.
    layers: Vec<IdentityPrefix<'a, F>>,
}

impl<'a, F: Field> ProductLayers<'a, F> {
    /// Builds every product level needed by the radix-four descent, borrowing the leaves.
    pub(super) fn new<Base: Field>(leaves: impl Into<Cow<'a, [F]>>, log_height: usize) -> Self
    where
        F: ExtensionField<Base>,
    {
        Self::from_leaves::<Base>(IdentityPrefix::new(leaves), log_height)
    }

    /// Builds the retained levels while taking ownership of the leaf allocation.
    pub(super) fn new_owned<Base: Field>(leaves: Vec<F>, log_height: usize) -> Self
    where
        F: ExtensionField<Base>,
    {
        Self::from_leaves::<Base>(IdentityPrefix::new(leaves), log_height)
    }

    /// Builds every retained level above the given leaves.
    fn from_leaves<Base: Field>(leaves: IdentityPrefix<'a, F>, log_height: usize) -> Self
    where
        F: ExtensionField<Base>,
    {
        // Unvisited depths remain empty constant-one prefixes.
        let mut layers = (0..=log_height)
            .map(|_| IdentityPrefix::new(Vec::new()))
            .collect::<Vec<_>>();
        layers[0] = leaves;

        // Two multiplication levels are retained per radix-four layer.
        let mut depth = 0;
        while depth + 2 <= log_height {
            layers[depth + 2] = layers[depth].reduce::<Base>(4);
            depth += 2;
        }
        if depth < log_height {
            layers[log_height] = layers[depth].reduce::<Base>(2);
        }

        Self { layers }
    }

    /// Returns the fully reduced product root.
    #[inline]
    pub(super) fn root(&self) -> F {
        self.layers
            .last()
            .expect("every product tree retains its root layer")
            .get(0)
    }

    /// Returns the first `count` values of the root level, a batch of roots.
    pub(super) fn roots(&self, count: usize) -> Vec<F> {
        let top = self
            .layers
            .last()
            .expect("every product tree retains its root layer");
        (0..count).map(|index| top.get(index)).collect()
    }

    /// Reads the two children of a root-most binary layer.
    #[inline]
    pub(super) fn binary_children(&self, depth: usize) -> [F; 2] {
        [self.layers[depth].get(0), self.layers[depth].get(1)]
    }

    /// Copies the binary child claims and releases their consumed source level.
    pub(super) fn take_binary_children(&mut self, depth: usize) -> [F; 2] {
        let children = self.binary_children(depth);
        self.layers[depth].values = Cow::Owned(Vec::new());
        children
    }

    /// Takes a retained level, releasing it from the layers.
    fn take_layer(&mut self, depth: usize) -> IdentityPrefix<'a, F> {
        IdentityPrefix {
            values: core::mem::take(&mut self.layers[depth].values),
        }
    }
}

/// Values in one block: four rows of four children.
const BLOCK: usize = 16;

/// Values in one pair: two rows of four children.
const PAIR: usize = 8;

/// Most workers whose packed rows still run one round at a time.
const SINGLE_MAX_WORKERS: usize = 4;

/// Most lanes a packing may have for its rows to run one round at a time.
const SINGLE_MAX_LANES: usize = 8;

/// Values one chunk of a child level spans, a power of two.
///
/// A chunk is one task of a pass, and a fold keeps its live values at its start.
const CHUNK: usize = 1 << 12;

/// Fewest live values a chunk keeps in place.
///
/// Shorter chunks gather into one contiguous run, since a task that small no longer pays for its dispatch.
const MIN_LIVE: usize = 64;

/// Pairs whose products one dot product sums, so a field that defers reduction reduces once per run.
const DOT_PAIRS: usize = 8;

/// Blocks of four rows one task multiplies into the level above.
const TASK_BLOCKS: usize = 64;

/// Most lanes a packed extension may have for the packed kernels.
///
/// It sizes the stack buffer a packed group's results go through.
const MAX_LANES: usize = 16;

/// Most low variables of the split block weights.
///
/// Every task rereads the low table, so it stays small enough for the second-level cache.
const EQ_LOW_VARS: usize = 10;

/// One tree's child level, read in place as rows of four children.
///
/// Row `z` holds the children `values[4z..4z + 4]`.
///
/// The values sit in chunks of `stride`, each a task of a pass, and a fold keeps every chunk's live values at its start.
///
/// Rows past the explicit prefix are all ones.
struct RadixFourState<'a, F: Clone> {
    /// The chunked explicit values.
    values: Cow<'a, [F]>,
    /// Values between the starts of two chunks.
    stride: usize,
    /// Live values of every chunk but the last.
    live: usize,
    /// Live values in all.
    len: usize,
}

impl<'a, F: Field> RadixFourState<'a, F> {
    /// Wraps a retained level without moving its values.
    fn new(values: IdentityPrefix<'a, F>) -> Self {
        Self::contiguous(values.values)
    }

    /// Chunks contiguous values.
    fn contiguous(values: Cow<'a, [F]>) -> Self {
        Self {
            len: values.len(),
            values,
            stride: CHUNK,
            live: CHUNK,
        }
    }

    /// Chunks holding live values.
    const fn chunks(&self) -> usize {
        self.len.div_ceil(self.live)
    }

    /// Live values of chunk `chunk`.
    const fn live_of(&self, chunk: usize) -> usize {
        let start = chunk * self.live;
        if self.len - start < self.live {
            self.len - start
        } else {
            self.live
        }
    }

    /// Logical value `index`, one past the explicit prefix.
    #[inline]
    fn get(&self, index: usize) -> F {
        if index < self.len {
            self.values[index / self.live * self.stride + index % self.live]
        } else {
            F::ONE
        }
    }

    /// Bytes one task of a pass is charged.
    ///
    /// Each block costs dozens of products, so its bytes are weighted by a few products each.
    const fn task_bytes(&self) -> usize {
        4 * self.live * size_of::<F>()
    }

    /// Sums `sum` over the live values of every chunk, the chunk's first value given as its index.
    fn sum_chunks<S: Send>(&self, sum: impl Fn(&[F], usize) -> S + Sync) -> Vec<S> {
        let chunks = self.chunks();
        self.values
            .par_chunks(self.stride)
            .take(chunks)
            .enumerate()
            .map_collect_min_task_bytes(self.task_bytes(), |(chunk, values)| {
                sum(&values[..self.live_of(chunk)], chunk * self.live)
            })
    }

    /// Folds the live values of every chunk to `1 / ratio` of their count, keeping each chunk's output at its start.
    ///
    /// It folds in place where the values are owned and the chunks stay long enough, and into a fresh run otherwise.
    fn fold_chunks<C: ChunkFold<F>>(&mut self, fold: &C) -> Vec<C::Sums> {
        let chunks = self.chunks();
        let width = C::WIDTH;
        let live = self.live / (width / C::ROW);
        let output = |chunk: usize| C::ROW * self.live_of(chunk).div_ceil(width);
        let len = (0..chunks).map(output).sum::<usize>();
        let run = if chunks > 1 { output(0) } else { len.max(1) };
        let task_bytes = self.task_bytes();

        // A short last chunk can write up to a row past its input, which spare capacity holds.
        let end = chunks.saturating_sub(1) * self.stride + chunks.checked_sub(1).map_or(0, output);
        if live >= MIN_LIVE
            && let Cow::Owned(values) = &mut self.values
            && values.capacity() >= end
        {
            if values.len() < end {
                values.resize(end, F::ONE);
            }
            let lives = (self.live, self.len);
            let sums = values
                .par_chunks_mut(self.stride)
                .take(chunks)
                .enumerate()
                .map_collect_min_task_bytes(task_bytes, |(chunk, values)| {
                    let input = lives.0.min(lives.1 - chunk * lives.0);
                    fold.fold(&mut InPlace(values), input, chunk * live)
                });
            self.live = live;
            self.len = len;
            return sums;
        }

        // The outputs of the chunks but the last are equally long, so they abut in one run.
        let mut next = Vec::with_capacity(len + BLOCK);
        let source = &self.values;
        let (stride, lives) = (self.stride, (self.live, self.len));
        let sums = next.spare_capacity_mut()[..len]
            .par_chunks_mut(run)
            .zip(source.par_chunks(stride))
            .enumerate()
            .map_collect_min_task_bytes(task_bytes, |(chunk, (out, values))| {
                let input = lives.0.min(lives.1 - chunk * lives.0);
                fold.fold(&mut Split(values, out), input, chunk * live)
            });
        // SAFETY: the chunks of `next` cover its first `len` slots, and each fold wrote all of its own.
        unsafe { next.set_len(len) };
        *self = Self::new(IdentityPrefix::new(next));
        sums
    }

    /// Sums the polynomial of the next two rounds over every block.
    fn grid<Base: Field>(&self, weights: &BlockWeights<F>) -> GridSums<F>
    where
        F: ExtensionField<Base>,
    {
        self.sum_chunks(|values, first| {
            let mut sums = GridSums::new();
            sums.add_blocks::<Base>(values, first / BLOCK, weights);
            sums
        })
        .into_iter()
        .fold(GridSums::new(), GridSums::merge)
    }

    /// Binds the next two variables to `challenges`, lowest first.
    ///
    /// Given block weights, it also sums the next two rounds' polynomial from the rows it writes.
    fn fold_two<Base: Field>(
        &mut self,
        challenges: [F; 2],
        weights: Option<&BlockWeights<F>>,
    ) -> Option<GridSums<F>>
    where
        F: ExtensionField<Base>,
    {
        let partials = self.fold_chunks(&FoldTwo::<Base, F> {
            challenges,
            weights,
            base: PhantomData,
        });
        weights.map(|_| {
            partials
                .into_iter()
                .map(|sums| sums.expect("every task sums when weights are given"))
                .fold(GridSums::new(), GridSums::merge)
        })
    }

    /// Sums the polynomial of the next round over every pair of rows of `N` children.
    fn line<Base: Field, const N: usize>(
        &self,
        weights: &(impl PairWeights<F> + ?Sized),
    ) -> PairSums<F>
    where
        F: ExtensionField<Base>,
    {
        self.sum_chunks(|values, first| {
            let mut sums = PairSums::new();
            sums.add_pairs::<Base, N>(values, first / (2 * N), weights);
            sums
        })
        .into_iter()
        .fold(PairSums::new(), PairSums::merge)
    }

    /// Binds the next variable to `challenge`.
    ///
    /// Given pair weights, it also sums the next round's polynomial from the rows it writes.
    fn fold_one<Base: Field, const N: usize, W: PairWeights<F> + ?Sized>(
        &mut self,
        challenge: F,
        weights: Option<&W>,
    ) -> Option<PairSums<F>>
    where
        F: ExtensionField<Base>,
    {
        let partials = self.fold_chunks(&FoldOne::<Base, F, W, N> {
            challenge,
            weights,
            base: PhantomData,
        });
        weights.map(|_| {
            partials
                .into_iter()
                .map(|sums| sums.expect("every task sums when weights are given"))
                .fold(PairSums::new(), PairSums::merge)
        })
    }

    /// Binds the last variable of a layer with two rows left.
    fn fold_last(&mut self, challenge: F) {
        let children: [F; 4] = core::array::from_fn(|child| {
            interpolate_pair([self.get(child), self.get(4 + child)], challenge)
        });
        *self = Self::new(IdentityPrefix::new(children.to_vec()));
    }

    /// The four lines' product at every node, for a layer with two rows left.
    fn last_product(&self, nodes: &RoundNodes<F>) -> [F; ROUND_POLY_LEN] {
        let line = |child: usize| nodes.line(self.get(child), self.get(4 + child));
        let mut product = line(0);
        for child in 1..4 {
            for (product, value) in product.iter_mut().zip(line(child)) {
                *product *= value;
            }
        }
        product
    }

    /// Reads the `N` terminal child claims after every parent variable is bound.
    fn children<const N: usize>(&self) -> [F; N] {
        core::array::from_fn(|child| self.get(child))
    }
}

/// One fold of a chunk's live values, and the sums it takes of the rows it writes.
trait ChunkFold<F>: Sync {
    /// The sums of one chunk's output.
    type Sums: Send;

    /// Input values that fold into one row.
    const WIDTH: usize;

    /// Values in one row.
    const ROW: usize;

    /// Folds `input` values through `io`, its first output being value `first` of the folded level.
    fn fold(&self, io: &mut impl FoldIo<F>, input: usize, first: usize) -> Self::Sums;
}

/// Binds two variables, summing the next two rounds when given their block weights.
struct FoldTwo<'w, Base, F> {
    /// The two challenges, lowest first.
    challenges: [F; 2],
    /// The next pair of rounds' block weights.
    weights: Option<&'w BlockWeights<F>>,
    /// The field the extension packs over.
    base: PhantomData<fn() -> Base>,
}

impl<Base: Field, F: ExtensionField<Base>> ChunkFold<F> for FoldTwo<'_, Base, F> {
    type Sums = Option<GridSums<F>>;

    const WIDTH: usize = BLOCK;

    const ROW: usize = 4;

    #[inline]
    fn fold(&self, io: &mut impl FoldIo<F>, input: usize, first: usize) -> Self::Sums {
        let rows = fold_blocks::<Base, F>(io, input, self.challenges);

        // The rows just written are still in cache, so the next grid reads no memory.
        self.weights.map(|weights| {
            let mut sums = GridSums::new();
            sums.add_blocks::<Base>(io.output(rows), first / BLOCK, weights);
            sums
        })
    }
}

/// Binds one variable, summing the next round when given its pair weights.
struct FoldOne<'w, Base, F, W: ?Sized, const N: usize> {
    /// The challenge.
    challenge: F,
    /// The next round's pair weights.
    weights: Option<&'w W>,
    /// The field the extension packs over.
    base: PhantomData<fn() -> Base>,
}

impl<Base: Field, F: ExtensionField<Base>, W: PairWeights<F> + ?Sized, const N: usize> ChunkFold<F>
    for FoldOne<'_, Base, F, W, N>
{
    type Sums = Option<PairSums<F>>;

    const WIDTH: usize = 2 * N;

    const ROW: usize = N;

    #[inline]
    fn fold(&self, io: &mut impl FoldIo<F>, input: usize, first: usize) -> Self::Sums {
        let rows = fold_pairs::<Base, F, N>(io, input, self.challenge);

        // The rows just written are still in cache, so the next round reads no memory.
        self.weights.map(|weights| {
            let mut sums = PairSums::new();
            sums.add_pairs::<Base, N>(io.output(rows), first / (2 * N), weights);
            sums
        })
    }
}

/// The values a fold reads and the rows it writes, in one buffer or in two.
///
/// A fold reads its inputs in order and writes each output at or before inputs it has already read.
/// One buffer can therefore hold both.
trait FoldIo<F> {
    /// Input value `index`.
    fn input(&self, index: usize) -> F;

    /// Writes output value `index`.
    fn write(&mut self, index: usize, value: F);

    /// The first `len` output values, all written.
    fn output(&mut self, len: usize) -> &[F];
}

/// A fold into the start of its own input.
struct InPlace<'b, F>(&'b mut [F]);

impl<F: Copy> FoldIo<F> for InPlace<'_, F> {
    #[inline(always)]
    fn input(&self, index: usize) -> F {
        self.0[index]
    }

    #[inline(always)]
    fn write(&mut self, index: usize, value: F) {
        self.0[index] = value;
    }

    #[inline]
    fn output(&mut self, len: usize) -> &[F] {
        &self.0[..len]
    }
}

/// A fold from one buffer into another.
struct Split<'b, F>(&'b [F], &'b mut [MaybeUninit<F>]);

impl<F: Copy> FoldIo<F> for Split<'_, F> {
    #[inline(always)]
    fn input(&self, index: usize) -> F {
        self.0[index]
    }

    #[inline(always)]
    fn write(&mut self, index: usize, value: F) {
        self.1[index].write(value);
    }

    #[inline]
    fn output(&mut self, len: usize) -> &[F] {
        let written = &self.1[..len];
        // SAFETY: the fold wrote every slot of `written`, and `MaybeUninit<F>` has `F`'s layout.
        unsafe { core::slice::from_raw_parts(written.as_ptr().cast::<F>(), len) }
    }
}

/// Folds each block of the first `len` inputs into one output row, binding its two row bits.
///
/// A short last block is completed by implicit identities.
/// Returns the output values written.
#[inline]
fn fold_blocks<Base: Field, F: ExtensionField<Base>>(
    io: &mut impl FoldIo<F>,
    len: usize,
    [low, high]: [F; 2],
) -> usize {
    // Row `2 b_2 + b_1` of a block is its corner at `(Y_1, Y_2) = (b_1, b_2)`.
    //
    //     lower  = e_0 + r_0 (e_1 - e_0)
    //     upper  = e_2 + r_0 (e_3 - e_2)
    //     folded = lower + r_1 (upper - lower)
    let fold = |block: [F; BLOCK]| -> [F; 4] {
        core::array::from_fn(|child| {
            let corner = |index: usize| block[4 * index + child];
            let lower = interpolate_pair([corner(0), corner(1)], low);
            let upper = interpolate_pair([corner(2), corner(3)], low);
            interpolate_pair([lower, upper], high)
        })
    };
    let blocks = len / BLOCK;

    // Whole groups go through the packed extension, lane `4 b + c` holding child `c` of block `b`.
    // Those lanes are the group's output rows in order.
    let lanes = Base::Packing::WIDTH;
    let packed = if lanes % 4 == 0 && (4..=MAX_LANES).contains(&lanes) {
        let group_blocks = lanes / 4;
        let mut folded = [F::ZERO; MAX_LANES];
        for group in 0..blocks / group_blocks {
            let start = group * group_blocks * BLOCK;
            let corner = |index: usize| {
                F::ExtensionPacking::from_ext_fn(|lane| {
                    io.input(start + BLOCK * (lane / 4) + 4 * index + lane % 4)
                })
            };
            let (e0, e1, e2, e3) = (corner(0), corner(1), corner(2), corner(3));
            let lower = e0 + (e1 - e0) * low;
            let upper = e2 + (e3 - e2) * low;
            (lower + (upper - lower) * high).to_ext_slice(&mut folded[..lanes]);
            for (lane, &value) in folded[..lanes].iter().enumerate() {
                io.write(group * lanes + lane, value);
            }
        }
        blocks / group_blocks * group_blocks
    } else {
        0
    };

    // The blocks short of a group go one at a time, the last one completed by identities.
    for block in packed..len.div_ceil(BLOCK) {
        let start = block * BLOCK;
        let values = core::array::from_fn(|index| {
            if start + index < len {
                io.input(start + index)
            } else {
                F::ONE
            }
        });
        for (child, value) in fold(values).into_iter().enumerate() {
            io.write(4 * block + child, value);
        }
    }
    4 * len.div_ceil(BLOCK)
}

/// Folds each pair of rows of the first `len` inputs into one output row, binding its row bit.
///
/// A short last pair is completed by implicit identities.
/// Returns the output values written.
#[inline]
fn fold_pairs<Base: Field, F: ExtensionField<Base>, const N: usize>(
    io: &mut impl FoldIo<F>,
    len: usize,
    challenge: F,
) -> usize {
    let width = 2 * N;
    let pairs = len / width;

    // Whole groups go through the packed extension, lane `N p + c` holding child `c` of pair `p`.
    // Those lanes are the group's output rows in order.
    let lanes = Base::Packing::WIDTH;
    let packed = if lanes % N == 0 && (N..=MAX_LANES).contains(&lanes) {
        let group_pairs = lanes / N;
        let mut folded = [F::ZERO; MAX_LANES];
        for group in 0..pairs / group_pairs {
            let start = group * group_pairs * width;
            let row = |index: usize| {
                F::ExtensionPacking::from_ext_fn(|lane| {
                    io.input(start + width * (lane / N) + N * index + lane % N)
                })
            };
            let (lower, upper) = (row(0), row(1));
            (lower + (upper - lower) * challenge).to_ext_slice(&mut folded[..lanes]);
            for (lane, &value) in folded[..lanes].iter().enumerate() {
                io.write(group * lanes + lane, value);
            }
        }
        pairs / group_pairs * group_pairs
    } else {
        0
    };

    // The pairs short of a group go one at a time, the last one completed by identities.
    for pair in packed..len.div_ceil(width) {
        let start = pair * width;
        let value = |index: usize| {
            if start + index < len {
                io.input(start + index)
            } else {
                F::ONE
            }
        };
        let row: [F; N] = core::array::from_fn(|child| {
            interpolate_pair([value(child), value(N + child)], challenge)
        });
        for (child, value) in row.into_iter().enumerate() {
            io.write(N * pair + child, value);
        }
    }
    N * len.div_ceil(width)
}

/// Writes the product of each row of four of `source` into `out`.
///
/// Whole groups of rows go through the packed extension, row `l` of a group in lane `l`.
/// A short last row is completed by implicit identities.
#[inline]
fn multiply_rows<Base: Field, F: ExtensionField<Base>>(source: &[F], out: &mut [MaybeUninit<F>]) {
    let lanes = Base::Packing::WIDTH;
    let (rows, short) = source.as_chunks::<4>();
    let mut products = [F::ZERO; MAX_LANES];
    for (group, slots) in rows.chunks_exact(lanes).zip(out.chunks_exact_mut(lanes)) {
        let child = |c: usize| F::ExtensionPacking::from_ext_fn(|lane| group[lane][c]);
        ((child(0) * child(1)) * (child(2) * child(3))).to_ext_slice(&mut products[..lanes]);
        for (slot, &product) in slots.iter_mut().zip(&products) {
            slot.write(product);
        }
    }

    // The rows short of a group go one at a time.
    let packed = rows.len() / lanes * lanes;
    for (row, slot) in rows[packed..].iter().zip(&mut out[packed..]) {
        slot.write((row[0] * row[1]) * (row[2] * row[3]));
    }
    if !short.is_empty() {
        out[rows.len()].write(short.iter().copied().product());
    }
}

/// Completes a short block with implicit identities.
#[inline]
fn padded_block<F: Field>(short: &[F]) -> [F; BLOCK] {
    let mut block = [F::ONE; BLOCK];
    block[..short.len()].copy_from_slice(short);
    block
}

/// The equality weight of every pair of rows in one round.
trait PairWeights<F: Field>: Sync {
    /// The weight of pair `pair`.
    fn at(&self, pair: usize) -> F;

    /// The weights of the pairs from `start`, one per lane of the packed extension.
    fn packed_at<Base: Field>(&self, start: usize) -> F::ExtensionPacking
    where
        F: ExtensionField<Base>;
}

impl<F: Field> PairWeights<F> for BlockWeights<F> {
    #[inline]
    fn at(&self, pair: usize) -> F {
        Self::at(self, pair)
    }

    #[inline]
    fn packed_at<Base: Field>(&self, start: usize) -> F::ExtensionPacking
    where
        F: ExtensionField<Base>,
    {
        Self::packed_at::<Base>(self, start)
    }
}

/// A flat table of pair weights.
impl<F: Field> PairWeights<F> for [F] {
    #[inline]
    fn at(&self, pair: usize) -> F {
        self[pair]
    }

    #[inline]
    fn packed_at<Base: Field>(&self, start: usize) -> F::ExtensionPacking
    where
        F: ExtensionField<Base>,
    {
        F::ExtensionPacking::from_ext_slice(&self[start..start + Base::Packing::WIDTH])
    }
}

/// The equality weight of every block, factored into two short tables.
///
/// Block `q` splits into its low bits `l` and high bits `h`:
///
/// ```text
///     eq(a, q) = eq(a_low, l) * eq(a_high, h)
/// ```
struct BlockWeights<F> {
    /// Weights over the low block bits.
    low: Vec<F>,
    /// Weights over the high block bits.
    high: Vec<F>,
    /// Number of low block bits.
    low_vars: usize,
}

impl<F: Field> BlockWeights<F> {
    /// Factors `eq(point, .)`, its first coordinate binding the lowest block bit.
    fn new(point: &[F]) -> Self {
        let low_vars = point.len().min(EQ_LOW_VARS);
        Self {
            low: Point::new(&point[..low_vars]).equality_weights_lsb(),
            high: Point::new(&point[low_vars..]).equality_weights_lsb(),
            low_vars,
        }
    }

    /// The weight of block `block`.
    #[inline]
    fn at(&self, block: usize) -> F {
        self.low[block & (self.low.len() - 1)] * self.high[block >> self.low_vars]
    }

    /// The weights of the blocks from `start`, one per lane of the packed extension.
    #[inline]
    fn packed_at<Base: Field>(&self, start: usize) -> F::ExtensionPacking
    where
        F: ExtensionField<Base>,
    {
        let lanes = Base::Packing::WIDTH;
        if self.low.len().is_multiple_of(lanes) && start.is_multiple_of(lanes) {
            // The lanes share one high weight, since the low table is a whole number of groups.
            let low = start & (self.low.len() - 1);
            F::ExtensionPacking::from_ext_slice(&self.low[low..low + lanes])
                * self.high[start >> self.low_vars]
        } else {
            F::ExtensionPacking::from_ext_fn(|lane| self.at(start + lane))
        }
    }
}

/// Sums of the block products of two rounds, for one tree.
///
/// Each block's product is kept as its Karatsuba terms.
///
/// The terms combine into coefficients only once, after the sum.
#[derive(Clone, Copy)]
struct GridSums<F> {
    /// Term sums, indexed by the `Y_2` term then the `Y_1` term.
    terms: [[F; 6]; 6],
    /// Sum of the weights of every summed block.
    weight: F,
}

impl<F: Field> GridSums<F> {
    /// The empty sum.
    const fn new() -> Self {
        Self {
            terms: [[F::ZERO; 6]; 6],
            weight: F::ZERO,
        }
    }

    /// Adds two partial sums.
    fn merge(mut self, other: Self) -> Self {
        for (terms, other) in self.terms.iter_mut().zip(other.terms) {
            for (term, other) in terms.iter_mut().zip(other) {
                *term += other;
            }
        }
        self.weight += other.weight;
        self
    }

    /// Adds every block of `values`, the first being block `first`.
    ///
    /// Whole groups of blocks go through the packed extension, block `l` of a group in lane `l`.
    /// The blocks short of a group, the last one possibly short of rows, go one at a time.
    #[inline]
    fn add_blocks<Base: Field>(&mut self, values: &[F], first: usize, weights: &BlockWeights<F>)
    where
        F: ExtensionField<Base>,
    {
        let lanes = Base::Packing::WIDTH;
        let (blocks, _) = values.as_chunks::<BLOCK>();
        let packed = if (2..=MAX_LANES).contains(&lanes) {
            blocks.len() / lanes * lanes
        } else {
            0
        };
        if packed > 0 {
            self.add_packed::<Base>(&blocks[..packed], first, weights);
        }
        self.add_scalar(&values[packed * BLOCK..], first + packed, weights);
    }

    /// Adds whole groups of blocks, one block per lane of the packed extension.
    #[inline]
    fn add_packed<Base: Field>(
        &mut self,
        blocks: &[[F; BLOCK]],
        first: usize,
        weights: &BlockWeights<F>,
    ) where
        F: ExtensionField<Base>,
    {
        let lanes = Base::Packing::WIDTH;
        let mut terms = [[F::ExtensionPacking::ZERO; 6]; 6];
        let mut weight_sum = F::ExtensionPacking::ZERO;
        for (index, group) in blocks.chunks_exact(lanes).enumerate() {
            // Lane `l` carries block `first + lanes * index + l` and its weight.
            let weight = weights.packed_at::<Base>(first + index * lanes);
            weight_sum += weight;
            let corners: [F::ExtensionPacking; BLOCK] = core::array::from_fn(|corner| {
                F::ExtensionPacking::from_ext_fn(|lane| group[lane][corner])
            });

            // Each lane's term sums its own blocks.
            accumulate_block(&mut terms, &corners, weight);
        }

        // The lanes' sums add into the scalar sums once per call.
        self.weight += F::ExtensionPacking::to_ext_iter([weight_sum]).sum::<F>();
        for (sum, packed) in self
            .terms
            .as_flattened_mut()
            .iter_mut()
            .zip(terms.as_flattened())
        {
            *sum += F::ExtensionPacking::to_ext_iter([*packed]).sum::<F>();
        }
    }

    /// Adds every block of `values` one at a time, the first being block `first`.
    ///
    /// A short last block is completed by implicit identities.
    #[inline]
    fn add_scalar(&mut self, values: &[F], first: usize, weights: &BlockWeights<F>) {
        let (blocks, short) = values.as_chunks::<BLOCK>();
        for (index, block) in blocks.iter().enumerate() {
            self.add_block(block, weights.at(first + index));
        }
        if !short.is_empty() {
            self.add_block(&padded_block(short), weights.at(first + blocks.len()));
        }
    }

    /// Adds one block's product, scaled by its weight.
    #[inline]
    fn add_block(&mut self, block: &[F; BLOCK], weight: F) {
        accumulate_block(&mut self.terms, block, weight);
        self.weight += weight;
    }

    /// Combines the terms into the bivariate polynomial, adding the identity tail.
    ///
    /// Blocks never summed are all ones, so each contributes its weight to the constant.
    /// The weights of all blocks sum to one, so those blocks contribute one minus the summed weight.
    fn finish(self) -> [[F; 5]; 5] {
        let in_y1 = self.terms.map(karatsuba_three_combine);
        let mut grid = karatsuba_three_combine_with(in_y1, add_coefficients, sub_coefficients);
        grid[0][0] += F::ONE - self.weight;
        grid
    }
}

/// Sums of the pair products of one round, for one tree.
///
/// Each pair's product is kept as the six Karatsuba terms of its two quadratic factors.
///
/// The terms combine into coefficients only once, after the sum.
#[derive(Clone, Copy)]
struct PairSums<F> {
    /// Term sums, in the order of [`pair_factors`].
    terms: [F; 6],
    /// Sum of the weights of every summed pair.
    weight: F,
}

impl<F: Field> PairSums<F> {
    /// The empty sum.
    const fn new() -> Self {
        Self {
            terms: [F::ZERO; 6],
            weight: F::ZERO,
        }
    }

    /// Adds two partial sums.
    fn merge(mut self, other: Self) -> Self {
        for (term, other) in self.terms.iter_mut().zip(other.terms) {
            *term += other;
        }
        self.weight += other.weight;
        self
    }

    /// Adds every pair of `values`, the first being pair `first`.
    ///
    /// Whole groups of pairs go through the packed extension, pair `l` of a group in lane `l`.
    /// The pairs short of a group, the last one possibly short of rows, go one at a time.
    #[inline]
    fn add_pairs<Base: Field, const N: usize>(
        &mut self,
        values: &[F],
        first: usize,
        weights: &(impl PairWeights<F> + ?Sized),
    ) where
        F: ExtensionField<Base>,
    {
        let width = 2 * N;
        let lanes = Base::Packing::WIDTH;
        let pairs = values.len() / width;
        let packed = if (2..=MAX_LANES).contains(&lanes) {
            pairs / lanes * lanes
        } else {
            0
        };

        if packed > 0 {
            let mut terms = PairTerms::<F::ExtensionPacking>::new();
            for (index, group) in values[..packed * width]
                .chunks_exact(lanes * width)
                .enumerate()
            {
                // Lane `l` carries pair `first + lanes * index + l` and its weight.
                let weight = weights.packed_at::<Base>(first + index * lanes);
                let mut corners = [F::ExtensionPacking::ZERO; PAIR];
                for (corner, packed) in corners[..width].iter_mut().enumerate() {
                    *packed = F::ExtensionPacking::from_ext_fn(|lane| group[lane * width + corner]);
                }
                terms.add::<N>(&corners, weight);
            }

            self.add_lanes::<Base>(terms);
        }

        let mut terms = PairTerms::<F>::new();
        for (index, pair) in values[packed * width..]
            .chunks(width)
            .enumerate()
            .map(|(index, pair)| (packed + index, pair))
        {
            let mut padded = [F::ONE; PAIR];
            padded[..pair.len()].copy_from_slice(pair);
            terms.add::<N>(&padded, weights.at(first + index));
        }
        let (sums, weight) = terms.finish();
        self.weight += weight;
        for (sum, term) in self.terms.iter_mut().zip(sums) {
            *sum += term;
        }
    }

    /// Adds the sums of every lane of packed pair terms.
    #[inline]
    fn add_lanes<Base: Field>(&mut self, terms: PairTerms<F::ExtensionPacking>)
    where
        F: ExtensionField<Base>,
    {
        let (sums, weight) = terms.finish();
        self.weight += F::ExtensionPacking::to_ext_iter([weight]).sum::<F>();
        for (sum, packed) in self.terms.iter_mut().zip(sums) {
            *sum += F::ExtensionPacking::to_ext_iter([packed]).sum::<F>();
        }
    }

    /// Combines every tree's sums into the round's degree-four polynomial, adding the identity tails.
    ///
    /// Pairs never summed are all ones, so each contributes its weight to the constant.
    /// The weights of all pairs sum to one, so those pairs contribute one minus the summed weight.
    fn batch<const N: usize>(sums: Vec<Self>, batching: F) -> [F; 5] {
        let mut product = [F::ZERO; 5];
        for (sums, power) in sums.into_iter().zip(batching.powers()) {
            let mut coefficients = if N == 2 {
                // Two lines multiply into a quadratic, whose terms are `c_0`, `c_1` and `d_01`.
                let [c0, c1, d01, ..] = sums.terms;
                [c0, d01 - c0 - c1, c1, F::ZERO, F::ZERO]
            } else {
                karatsuba_three_combine(sums.terms)
            };
            coefficients[0] += F::ONE - sums.weight;
            for (total, coefficient) in product.iter_mut().zip(coefficients) {
                *total += power * coefficient;
            }
        }
        product
    }
}

/// The six term products of pairs, buffered so each term sums as one dot product.
struct PairTerms<R> {
    /// Sums of the runs already reduced.
    sums: [R; 6],
    /// Sum of the weights added.
    weight: R,
    /// Left factors of the buffered pairs, by term.
    left: [[R; DOT_PAIRS]; 6],
    /// Right factors of the buffered pairs, by term.
    right: [[R; DOT_PAIRS]; 6],
    /// Buffered pairs.
    len: usize,
}

impl<R: PrimeCharacteristicRing + Copy> PairTerms<R> {
    /// The empty sum.
    const fn new() -> Self {
        Self {
            sums: [R::ZERO; 6],
            weight: R::ZERO,
            left: [[R::ZERO; DOT_PAIRS]; 6],
            right: [[R::ZERO; DOT_PAIRS]; 6],
            len: 0,
        }
    }

    /// Buffers the factors of one pair of rows of `N` children, its weight carried by the first child.
    ///
    /// The pair's values are the first `2 N` of `pair`.
    #[inline(always)]
    fn add<const N: usize>(&mut self, pair: &[R; PAIR], weight: R) {
        self.weight += weight;
        let [left, right] = pair_factors::<R, N>(pair, weight);
        for term in 0..6 {
            self.left[term][self.len] = left[term];
            self.right[term][self.len] = right[term];
        }
        self.len += 1;
        if self.len == DOT_PAIRS {
            for term in 0..if N == 2 { 3 } else { 6 } {
                self.sums[term] += R::dot_product(&self.left[term], &self.right[term]);
            }
            self.len = 0;
        }
    }

    /// The term sums and the weight sum, the buffered pairs included.
    #[inline]
    fn finish(mut self) -> ([R; 6], R) {
        for term in 0..6 {
            for index in 0..self.len {
                self.sums[term] += self.left[term][index] * self.right[term][index];
            }
        }
        (self.sums, self.weight)
    }
}

/// The factors of the products whose sums give the coefficients of `w c_0 ... c_(N-1)` along a pair.
///
/// Child `c` is the line through rows zero and one.
/// Four lines multiply in two quadratics, whose product's Karatsuba terms pair up as `left[t] * right[t]`.
/// Two lines give the three Karatsuba terms of their own product.
#[inline(always)]
fn pair_factors<R: PrimeCharacteristicRing + Copy, const N: usize>(
    pair: &[R; PAIR],
    weight: R,
) -> [[R; 6]; 2] {
    let line = |child: usize| [pair[child], pair[N + child] - pair[child]];
    let [a, b] = line(0);
    if N == 2 {
        let [c, d] = line(1);
        let (a, b) = (a * weight, b * weight);
        return [
            [a, b, a + b, R::ZERO, R::ZERO, R::ZERO],
            [c, d, c + d, R::ZERO, R::ZERO, R::ZERO],
        ];
    }
    let left = line_product([a * weight, b * weight], line(1));
    let right = line_product(line(2), line(3));
    let spread = |q: [R; 3]| [q[0], q[1], q[2], q[0] + q[1], q[0] + q[2], q[1] + q[2]];
    [spread(left), spread(right)]
}

/// Adds one block's product terms into their sums, its weight carried by the first child.
///
/// Each term is multiplied as soon as its operands exist, so no operand waits in memory.
#[inline(always)]
fn accumulate_block<R: PrimeCharacteristicRing + Copy>(
    terms: &mut [[R; 6]; 6],
    block: &[R; BLOCK],
    weight: R,
) {
    // Child `c` as a bilinear polynomial, coefficients indexed `[Y_2 power][Y_1 power]`.
    //
    //     V_c = e_0 + Y_1 (e_1 - e_0) + Y_2 (e_2 - e_0) + Y_1 Y_2 (e_3 - e_2 - e_1 + e_0)
    let child = |c: usize| {
        let corner = |index: usize| block[4 * index + c];
        let y1 = corner(1) - corner(0);
        [
            [corner(0), y1],
            [corner(2) - corner(0), corner(3) - corner(2) - y1],
        ]
    };

    // The weight scales one factor, so the product carries it.
    let [[a, b], [c, d]] = child(0);
    let first = [[a * weight, b * weight], [c * weight, d * weight]];

    // Two pairs of bilinear factors make two biquadratics.
    let left = bilinear_product(first, child(1));
    let right = bilinear_product(child(2), child(3));

    // Their product's Karatsuba terms: six in `Y_2`, each six in `Y_1`.
    let outer = karatsuba_three_operands(left, right, add_coefficients);
    for (sums, (u, w)) in terms.iter_mut().zip(outer) {
        for (sum, (x, y)) in sums.iter_mut().zip(karatsuba_three_operands(u, w, R::add)) {
            *sum += x * y;
        }
    }
}

/// Adds two coefficient vectors.
#[inline(always)]
fn add_coefficients<R: PrimeCharacteristicRing + Copy, const N: usize>(
    a: [R; N],
    b: [R; N],
) -> [R; N] {
    core::array::from_fn(|index| a[index] + b[index])
}

/// Subtracts two coefficient vectors.
#[inline(always)]
fn sub_coefficients<R: PrimeCharacteristicRing + Copy, const N: usize>(
    a: [R; N],
    b: [R; N],
) -> [R; N] {
    core::array::from_fn(|index| a[index] - b[index])
}

/// The product of two lines, three products.
#[inline(always)]
fn line_product<R: PrimeCharacteristicRing + Copy>([a0, a1]: [R; 2], [b0, b1]: [R; 2]) -> [R; 3] {
    // Karatsuba: the middle coefficient is (a_0 + a_1)(b_0 + b_1) - a_0 b_0 - a_1 b_1.
    let low = a0 * b0;
    let high = a1 * b1;
    [low, (a0 + a1) * (b0 + b1) - low - high, high]
}

/// The product of two bilinear polynomials, nine products.
#[inline(always)]
fn bilinear_product<R: PrimeCharacteristicRing + Copy>(
    [a0, a1]: [[R; 2]; 2],
    [b0, b1]: [[R; 2]; 2],
) -> [[R; 3]; 3] {
    // Karatsuba in `Y_2`, each of its three products a product of lines in `Y_1`.
    let low = line_product(a0, b0);
    let high = line_product(a1, b1);
    let cross = line_product(add_coefficients(a0, a1), add_coefficients(b0, b1));
    [
        low,
        sub_coefficients(sub_coefficients(cross, low), high),
        high,
    ]
}

/// The six operand pairs of a three-coefficient Karatsuba product.
///
/// ```text
///     (a_0, b_0), (a_1, b_1), (a_2, b_2),
///     (a_0 + a_1, b_0 + b_1), (a_0 + a_2, b_0 + b_2), (a_1 + a_2, b_1 + b_2)
/// ```
#[inline(always)]
fn karatsuba_three_operands<T: Copy>(a: [T; 3], b: [T; 3], add: impl Fn(T, T) -> T) -> [(T, T); 6] {
    let sum = |i: usize, j: usize| (add(a[i], a[j]), add(b[i], b[j]));
    [
        (a[0], b[0]),
        (a[1], b[1]),
        (a[2], b[2]),
        sum(0, 1),
        sum(0, 2),
        sum(1, 2),
    ]
}

/// The five coefficients a three-coefficient Karatsuba product's terms combine into.
#[inline]
fn karatsuba_three_combine<F: Field>(terms: [F; 6]) -> [F; 5] {
    karatsuba_three_combine_with(terms, F::add, F::sub)
}

/// Combines the six terms `c_0, c_1, c_2, d_01, d_02, d_12` over any coefficient type.
///
/// ```text
///     p_0 = c_0
///     p_1 = d_01 - c_0 - c_1
///     p_2 = d_02 - c_0 - c_2 + c_1
///     p_3 = d_12 - c_1 - c_2
///     p_4 = c_2
/// ```
#[inline]
fn karatsuba_three_combine_with<T: Copy>(
    [c0, c1, c2, d01, d02, d12]: [T; 6],
    add: impl Fn(T, T) -> T,
    sub: impl Fn(T, T) -> T,
) -> [T; 5] {
    [
        c0,
        sub(sub(d01, c0), c1),
        add(sub(sub(d02, c0), c2), c1),
        sub(sub(d12, c1), c2),
        c2,
    ]
}

/// The bivariate polynomial `P(Y_1, Y_2) = sum_q eq(q) prod_c V_c(Y_1, Y_2)` of two rounds.
///
/// Coefficients are indexed `[Y_2 power][Y_1 power]`, and `Y_1` is bound first.
struct Grid<F>([[F; 5]; 5]);

impl<F: Field> Grid<F> {
    /// Combines every tree's sums with consecutive powers of the batching challenge.
    fn batch(sums: Vec<GridSums<F>>, batching: F) -> Self {
        let mut grid = [[F::ZERO; 5]; 5];
        for (sums, power) in sums.into_iter().zip(batching.powers()) {
            for (row, tree_row) in grid.iter_mut().zip(sums.finish()) {
                for (coefficient, tree_coefficient) in row.iter_mut().zip(tree_row) {
                    *coefficient += power * tree_coefficient;
                }
            }
        }
        Self(grid)
    }

    /// The first round's inner polynomial, its rows weighed by `eq(anchor, Y_2)`.
    ///
    /// ```text
    ///     [Y_1^k] = P_{0,k} + anchor * sum_{m >= 1} P_{m,k}
    /// ```
    fn first_round(&self, anchor: F) -> [F; 5] {
        let [low, upper @ ..] = &self.0;
        core::array::from_fn(|k| low[k] + anchor * upper.iter().map(|row| row[k]).sum::<F>())
    }

    /// The second round's inner polynomial, once `Y_1` is bound to `challenge`.
    ///
    /// ```text
    ///     [Y_2^m] = sum_k challenge^k P_{m,k}
    /// ```
    fn second_round(&self, challenge: F) -> [F; 5] {
        self.0.map(|row| horner(&row, challenge))
    }
}

/// Evaluates a polynomial given by its coefficients, constant first.
#[inline]
fn horner<F: Field>(coefficients: &[F], point: F) -> F {
    coefficients
        .iter()
        .rev()
        .fold(F::ZERO, |sum, &coefficient| sum * point + coefficient)
}

/// The equality factor `eq(point[..j], r[..j])` of the coordinates bound so far.
struct EqPrefix<F>(F);

impl<F: Field> EqPrefix<F> {
    /// The empty product.
    const fn new() -> Self {
        Self(F::ONE)
    }

    /// Sends one round's message and binds its coordinate.
    ///
    /// The message is the inner values times the line `eq(point[..j], r[..j]) eq(anchor, X)`.
    fn round(
        &mut self,
        nodes: &RoundNodes<F>,
        anchor: F,
        inner: [F; ROUND_POLY_LEN],
        send: &mut impl FnMut([F; ROUND_POLY_LEN]) -> F,
    ) -> F {
        let one = self.0 * anchor;
        let pair = [self.0 - one, one];
        let eq_values = nodes.line(pair[0], pair[1]);
        let challenge = send(core::array::from_fn(|index| {
            eq_values[index] * inner[index]
        }));
        self.0 = interpolate_pair(pair, challenge);
        challenge
    }
}

/// Whether a layer runs its rounds in fused pairs rather than one at a time.
///
/// - A fused pair makes one pass over the rows where two single rounds make two.
/// - It pays for that with about twice the products, and a single round sums its products as dot products.
/// - Rows that do not pack cost little more than their products, so they run one round at a time.
/// - A few workers leave memory bandwidth spare, so their packed rows run singly too.
/// - A wide packing gathers many base values a pass, which outweighs the products a single round saves.
// The serial thread count is a constant, the parallel one is not.
#[allow(clippy::missing_const_for_fn)]
fn fuses_rounds<Base: Field>() -> bool {
    let lanes = Base::Packing::WIDTH;
    lanes > 1 && (lanes > SINGLE_MAX_LANES || current_num_threads() > SINGLE_MAX_WORKERS)
}

/// Per-tree states reduced by one shared sumcheck over `N` children, packed over `B`.
///
/// A radix-four layer has four children; a binary layer whose parent point has coordinates has two.
pub struct RadixFourBatch<'a, B, F: Clone, const N: usize> {
    /// One folding state for each product tree.
    states: Vec<RadixFourState<'a, F>>,
    /// A bound challenge whose fold waits for the next round's pass, when rounds are played one by one.
    pending: Option<F>,
    /// The field the extension packs over.
    base: PhantomData<fn() -> B>,
}

impl<'a, B: Field, F: ExtensionField<B>, const N: usize> RadixFourBatch<'a, B, F, N> {
    /// One table's batch from its child level, `N` values per parent row, identity suffix optional.
    pub fn from_level(level: Vec<F>) -> Self {
        Self::from_states(vec![RadixFourState::new(IdentityPrefix::new(level))])
    }

    /// Wraps one state per tree.
    const fn from_states(states: Vec<RadixFourState<'a, F>>) -> Self {
        Self {
            states,
            pending: None,
            base: PhantomData,
        }
    }

    /// Creates the batched states from one retained level per tree, borrowing each.
    #[cfg(test)]
    pub(super) fn new(layers: &'a [ProductLayers<'_, F>], depth: usize) -> Self {
        let states = layers
            .iter()
            .map(|layers| RadixFourState::new(IdentityPrefix::new(&*layers.layers[depth].values)))
            .collect();
        Self::from_states(states)
    }

    /// Consumes one retained level per tree, reading it in place.
    pub(super) fn new_consuming(layers: &mut [ProductLayers<'a, F>], depth: usize) -> Self {
        let states = layers
            .iter_mut()
            .map(|layers| RadixFourState::new(layers.take_layer(depth)))
            .collect();
        Self::from_states(states)
    }

    /// Runs every sumcheck round of the layer.
    ///
    /// Round `j` binds the lowest remaining row bit.
    /// Its rows are weighed by `eq(point[j + 1..], .)`, and the line `eq(point[j], X)` scales its message.
    ///
    /// `send` takes each round's message and returns its challenge.
    pub(super) fn prove_rounds(
        &mut self,
        point: &[F],
        batching: F,
        send: impl FnMut([F; ROUND_POLY_LEN]) -> F,
    ) -> Vec<F> {
        if N == 4 && fuses_rounds::<B>() {
            self.prove_pairs(point, batching, send)
        } else {
            self.prove_singles(point, batching, send)
        }
    }

    /// Runs the rounds one at a time, one pass over the rows each.
    fn prove_singles(
        &mut self,
        point: &[F],
        batching: F,
        mut send: impl FnMut([F; ROUND_POLY_LEN]) -> F,
    ) -> Vec<F> {
        let rounds = point.len();
        let nodes = RoundNodes::<F>::new();
        let mut challenges = Vec::with_capacity(rounds);

        // Already bound coordinates contribute one scalar to the equality line.
        let mut eq = EqPrefix::new();

        let mut sums = (rounds >= 1).then(|| {
            let weights = BlockWeights::new(&point[1..]);
            self.states
                .iter()
                .map(|state| state.line::<B, N>(&weights))
                .collect::<Vec<_>>()
        });
        while let Some(trees) = sums {
            let j = challenges.len();
            let inner = nodes.evaluate(&PairSums::batch::<N>(trees, batching));
            let challenge = eq.round(&nodes, point[j], inner, &mut send);
            challenges.push(challenge);

            // The fold reads the rows once more and sums the next round on the way.
            let weights = (j + 2 <= rounds).then(|| BlockWeights::new(&point[j + 2..]));
            let next = self
                .states
                .iter_mut()
                .map(|state| state.fold_one::<B, N, _>(challenge, weights.as_ref()))
                .collect::<Vec<_>>();
            sums = next.into_iter().collect();
        }
        challenges
    }

    /// Runs the rounds in pairs, one pass over the rows per pair.
    ///
    /// An odd count leaves one round, over two rows.
    fn prove_pairs(
        &mut self,
        point: &[F],
        batching: F,
        mut send: impl FnMut([F; ROUND_POLY_LEN]) -> F,
    ) -> Vec<F> {
        let rounds = point.len();
        let nodes = RoundNodes::<F>::new();
        let mut challenges = Vec::with_capacity(rounds);

        // Already bound coordinates contribute one scalar to the equality line.
        let mut eq = EqPrefix::new();

        // Two rounds per pass: rounds `j` and `j + 1` read one polynomial of their two variables.
        let mut grids = (rounds >= 2).then(|| {
            let weights = BlockWeights::new(&point[2..]);
            self.states
                .iter()
                .map(|state| state.grid::<B>(&weights))
                .collect::<Vec<_>>()
        });
        while let Some(sums) = grids {
            let j = challenges.len();
            let grid = Grid::batch(sums, batching);
            let first = nodes.evaluate(&grid.first_round(point[j + 1]));
            let low = eq.round(&nodes, point[j], first, &mut send);
            let second = nodes.evaluate(&grid.second_round(low));
            let high = eq.round(&nodes, point[j + 1], second, &mut send);
            challenges.extend([low, high]);

            // The fold reads the rows once more and sums the next two rounds on the way.
            let weights = (j + 4 <= rounds).then(|| BlockWeights::new(&point[j + 4..]));
            let next = self
                .states
                .iter_mut()
                .map(|state| state.fold_two::<B>([low, high], weights.as_ref()))
                .collect::<Vec<_>>();
            grids = next.into_iter().collect();
        }

        // An odd count leaves one round, over two rows per tree.
        if challenges.len() < rounds {
            let mut product = [F::ZERO; ROUND_POLY_LEN];
            for (state, power) in self.states.iter().zip(batching.powers()) {
                for (sum, value) in product.iter_mut().zip(state.last_product(&nodes)) {
                    *sum += power * value;
                }
            }
            let challenge = eq.round(&nodes, point[rounds - 1], product, &mut send);
            for state in &mut self.states {
                state.fold_last(challenge);
            }
            challenges.push(challenge);
        }
        challenges
    }

    /// Collects the terminal child claims in tree order.
    pub(super) fn children(&self) -> Vec<[F; N]> {
        debug_assert!(self.pending.is_none());
        self.states
            .iter()
            .map(RadixFourState::children::<N>)
            .collect()
    }
}

/// The sumcheck of one layer with `N` children per tree, as the product prover plays it.
///
/// An implementation over a cheaper leaf representation must return what the dense batch returns, so the transcript is unchanged.
pub trait LayerRounds<F, const N: usize> {
    /// Whether `round` reads the flat equality table; a layer that tracks its parent point itself is handed an empty one.
    const READS_EQUALITY: bool = true;

    /// The round message from the normalized equality suffix, the bound coordinate's eq endpoints, the live parent rows and the tree-batching challenge.
    fn round(
        &mut self,
        equality: &[F],
        equality_pair: [F; 2],
        logical_len: usize,
        batching: F,
    ) -> [F; ROUND_POLY_LEN];

    /// Binds the lowest parent variable to `challenge`.
    fn fold(&mut self, challenge: F, logical_len: usize);

    /// The terminal child claims of every tree once all parent variables are bound.
    fn children(&self) -> Vec<[F; N]>;
}

/// The dense batch played one round at a time, each fold deferred into the next round's pass.
impl<B: Field, F: ExtensionField<B>, const N: usize> LayerRounds<F, N>
    for RadixFourBatch<'_, B, F, N>
{
    fn round(
        &mut self,
        equality: &[F],
        equality_pair: [F; 2],
        logical_len: usize,
        batching: F,
    ) -> [F; ROUND_POLY_LEN] {
        debug_assert!(logical_len >= 2);
        debug_assert_eq!(equality.len(), logical_len / 2);
        let pending = self.pending.take();
        let trees = self
            .states
            .iter_mut()
            .map(|state| match pending {
                Some(challenge) => state
                    .fold_one::<B, N, [F]>(challenge, Some(equality))
                    .expect("a fold sums when weights are given"),
                None => state.line::<B, N>(equality),
            })
            .collect();
        let nodes = RoundNodes::<F>::new();
        let inner = nodes.evaluate(&PairSums::batch::<N>(trees, batching));
        let eq_values = nodes.line(equality_pair[0], equality_pair[1]);
        core::array::from_fn(|index| eq_values[index] * inner[index])
    }

    fn fold(&mut self, challenge: F, logical_len: usize) {
        debug_assert!(self.pending.is_none());
        if logical_len > 2 {
            self.pending = Some(challenge);
            return;
        }
        // The last fold leaves the terminal claims, so it runs now.
        for state in &mut self.states {
            state.fold_one::<B, N, [F]>(challenge, None);
        }
    }

    fn children(&self) -> Vec<[F; N]> {
        Self::children(self)
    }
}

/// The transmitted round-message nodes, with each line evaluated by as few products as possible.
struct RoundNodes<F> {
    /// Node one is omitted because the running sum reconstructs it.
    nodes: [F; ROUND_POLY_LEN],
    /// Whether a node equals its predecessor plus one, so its value is one slope further.
    unit_steps: [bool; ROUND_POLY_LEN],
}

impl<F: Field> RoundNodes<F> {
    /// Enumerates the nodes and marks every unit step between neighbours.
    fn new() -> Self {
        let nodes = [0, 2, 3, 4, 5].map(F::interpolation_node);
        let unit_steps =
            core::array::from_fn(|index| index > 0 && nodes[index] == nodes[index - 1] + F::ONE);
        Self { nodes, unit_steps }
    }

    /// Evaluates a polynomial given by its coefficients at every node.
    #[inline]
    fn evaluate(&self, coefficients: &[F]) -> [F; ROUND_POLY_LEN] {
        self.nodes.map(|node| horner(coefficients, node))
    }

    /// Evaluates the line through `zero` and `one` at every node.
    #[inline]
    fn line(&self, zero: F, one: F) -> [F; ROUND_POLY_LEN] {
        let slope = one - zero;
        let mut values = [zero; ROUND_POLY_LEN];
        for index in 0..ROUND_POLY_LEN {
            values[index] = if self.unit_steps[index] {
                values[index - 1] + slope
            } else if self.nodes[index] == F::ZERO {
                zero
            } else {
                zero + self.nodes[index] * slope
            };
        }
        values
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use p3_baby_bear::BabyBear;
    use p3_binary_field::{BinaryField128, Ghash128, Poly64, Poly192};
    use p3_field::PrimeCharacteristicRing;
    use p3_field::extension::BinomialExtensionField;
    use p3_multilinear_util::point::Point;
    use rand::distr::{Distribution, StandardUniform};
    use rand::{RngExt, SeedableRng};
    use rand_xoshiro::Xoroshiro128Plus;

    use super::*;

    type BabyBearQuartic = BinomialExtensionField<BabyBear, 4>;

    /// Folds a fully materialized multilinear table at its lowest variable.
    fn fold_dense<F: Field>(values: &mut Vec<F>, challenge: F) {
        *values = values
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&pair| interpolate_pair(pair, challenge))
            .collect();
    }

    /// The capacity an owned level holds, zero for a borrowed one.
    fn reserved<F: Clone>(prefix: &IdentityPrefix<'_, F>) -> usize {
        match &prefix.values {
            Cow::Borrowed(_) => 0,
            Cow::Owned(values) => values.capacity(),
        }
    }

    /// Splits an interleaved level into four dense child tables of `logical_len` rows.
    fn dense_children<F: Field>(values: &[F], logical_len: usize) -> [Vec<F>; 4] {
        core::array::from_fn(|child| {
            (0..logical_len)
                .map(|row| values.get(4 * row + child).copied().unwrap_or(F::ONE))
                .collect()
        })
    }

    /// Interpolates every line separately at every node, as an obvious reference.
    fn per_node_round<F: Field>(
        trees: &[[Vec<F>; 4]],
        equality: &[F],
        batching: F,
    ) -> [F; ROUND_POLY_LEN] {
        let nodes = [0, 2, 3, 4, 5].map(F::interpolation_node);
        let mut evaluations = [F::ZERO; ROUND_POLY_LEN];
        for row in 0..equality.len() / 2 {
            for (node_index, node) in nodes.into_iter().enumerate() {
                let eq_value = interpolate_pair([equality[2 * row], equality[2 * row + 1]], node);
                let mut power = F::ONE;
                let mut batched_product = F::ZERO;
                for children in trees {
                    let product = children
                        .iter()
                        .map(|child| interpolate_pair([child[2 * row], child[2 * row + 1]], node))
                        .product::<F>();
                    batched_product += power * product;
                    power *= batching;
                }
                evaluations[node_index] += eq_value * batched_product;
            }
        }
        evaluations
    }

    /// Checks every node of random lines, after pinning which nodes take the slope step.
    fn check_lines<F: Field>(rng: &mut Xoroshiro128Plus, unit_steps: [bool; ROUND_POLY_LEN])
    where
        StandardUniform: Distribution<F>,
    {
        let nodes = RoundNodes::<F>::new();
        assert_eq!(nodes.unit_steps, unit_steps);
        for _ in 0..32 {
            let zero = rng.random::<F>();
            let one = rng.random::<F>();
            let expected = [0, 2, 3, 4, 5]
                .map(|index| interpolate_pair([zero, one], F::interpolation_node(index)));
            assert_eq!(nodes.line(zero, one), expected);
        }
    }

    /// Checks every round message of every radix-four layer against the per-node reference.
    fn check_rounds<Base: Field, F: ExtensionField<Base>>(
        rng: &mut Xoroshiro128Plus,
        log_height: usize,
        prefix_lens: &[usize],
    ) where
        StandardUniform: Distribution<F>,
    {
        let inputs = prefix_lens
            .iter()
            .map(|&len| (0..len).map(|_| rng.random::<F>()).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        check_input_rounds::<Base, F>(rng, log_height, &inputs);
    }

    fn check_input_rounds<Base: Field, F: ExtensionField<Base>>(
        rng: &mut Xoroshiro128Plus,
        log_height: usize,
        inputs: &[Vec<F>],
    ) where
        StandardUniform: Distribution<F>,
    {
        let layers = inputs
            .iter()
            .map(|input| ProductLayers::new::<Base>(input, log_height))
            .collect::<Vec<_>>();

        // Every retained even depth with at least one parent variable is a child level.
        for depth in (0..=log_height - 3).step_by(2) {
            let variables = log_height - depth - 2;
            for anchor_kind in 0..4 {
                let anchor = (0..variables)
                    .map(|index| match anchor_kind {
                        0 => F::ZERO,
                        1 => F::ONE,
                        2 => F::from_bool(index % 2 != 0),
                        _ => rng.random::<F>(),
                    })
                    .collect::<Vec<_>>();
                for (challenge_kind, pairs) in (0..3).flat_map(|kind| [(kind, false), (kind, true)])
                {
                    let random_batching = rng.random::<F>();
                    for batching in [F::ZERO, F::ONE, random_batching] {
                        // The reference keeps dense child tables and the full equality table.
                        let mut trees = layers
                            .iter()
                            .map(|layers| {
                                dense_children(&layers.layers[depth].values, 1 << variables)
                            })
                            .collect::<Vec<_>>();
                        let mut equality = Point::new(anchor.as_slice()).equality_weights_lsb();
                        let mut batch = RadixFourBatch::new(&layers, depth);
                        let mut round = 0;

                        // Each message must equal the reference before both bind the same challenge.
                        let send = |message| {
                            assert_eq!(
                                message,
                                per_node_round(&trees, &equality, batching),
                                "depth={depth} round={round} anchor={anchor_kind} challenge={challenge_kind} pairs={pairs}",
                            );
                            let challenge = match challenge_kind {
                                0 => F::ZERO,
                                1 => F::ONE,
                                _ => rng.random::<F>(),
                            };
                            for child in trees.iter_mut().flatten() {
                                fold_dense(child, challenge);
                            }
                            fold_dense(&mut equality, challenge);
                            round += 1;
                            challenge
                        };
                        let challenges = if pairs {
                            batch.prove_pairs(&anchor, batching, send)
                        } else {
                            batch.prove_singles(&anchor, batching, send)
                        };
                        assert_eq!(challenges.len(), variables);

                        // Once every variable is bound, each child table is its terminal claim.
                        let expected = trees
                            .iter()
                            .map(|children| children.each_ref().map(|child| child[0]))
                            .collect::<Vec<_>>();
                        assert_eq!(batch.children(), expected);
                    }
                }
            }
        }
    }

    #[test]
    fn slope_steps_match_interpolation_at_every_node() {
        // In characteristic two only nodes 3 and 5 are one past their predecessors.
        // In a prime field nodes 3, 4 and 5 all are, and node 2 never is in either.
        // Each field therefore checks both the slope step and the direct product.
        let mut rng = Xoroshiro128Plus::seed_from_u64(0x5107_0001);
        check_lines::<BinaryField128>(&mut rng, [false, false, true, false, true]);
        check_lines::<Ghash128>(&mut rng, [false, false, true, false, true]);
        check_lines::<BabyBearQuartic>(&mut rng, [false, false, true, true, true]);
    }

    fn check_levels<Base: Field, F: ExtensionField<Base>>(rng: &mut Xoroshiro128Plus)
    where
        StandardUniform: Distribution<F>,
    {
        // Lengths around a lane group, a task and a partial row of four.
        for len in [0, 1, 3, 4, 5, 63, 64, 65, 1023, 4093, 4096, 4097, 8192] {
            let leaves = (0..len).map(|_| rng.random::<F>()).collect::<Vec<_>>();
            let layers = ProductLayers::new::<Base>(&leaves, 13);

            // Every retained even level is the naive product of rows of four below it.
            let mut level = leaves.clone();
            for depth in (2..=12).step_by(2) {
                level = level
                    .chunks(4)
                    .map(|row| row.iter().copied().product())
                    .collect();
                let expected = IdentityPrefix::new(level.clone());
                assert_eq!(layers.layers[depth].values, expected.values, "len={len}");
            }
            assert_eq!(layers.root(), leaves.iter().copied().product(), "len={len}");
        }
    }

    #[test]
    fn product_levels_match_a_naive_tree() {
        let mut rng = Xoroshiro128Plus::seed_from_u64(0x5107_0009);
        check_levels::<Poly64, Poly192>(&mut rng);
        check_levels::<BabyBear, BabyBearQuartic>(&mut rng);
        check_levels::<BinaryField128, BinaryField128>(&mut rng);
    }

    #[test]
    fn borrowed_leaves_are_read_in_place_until_the_first_fold() {
        // The prover only reads the leaves, so a borrowed input is never copied.
        //
        // Fixture state: 2^6 leaves, two trailing ones the trim drops without copying.
        let mut leaves = vec![Poly192::interpolation_node(2); 64];
        leaves[62] = Poly192::ONE;
        leaves[63] = Poly192::ONE;
        let mut layers = ProductLayers::new::<Poly64>(&leaves, 6);
        let taken = layers.take_layer(0);
        assert!(
            matches!(taken.values, Cow::Borrowed(values) if values.as_ptr() == leaves.as_ptr())
        );
        assert_eq!(taken.values.len(), 62);

        // The first fold writes owned rows and leaves the borrowed source untouched.
        let mut state = RadixFourState::new(taken);
        let _ = state.fold_two::<Poly64>([Poly192::ZERO, Poly192::ONE], None);
        assert!(matches!(state.values, Cow::Owned(_)));
    }

    #[test]
    fn consuming_layers_preserves_children_and_releases_only_the_source() {
        let mut rng = Xoroshiro128Plus::seed_from_u64(0x5107_0007);
        for log_height in [2usize, 3, 6, 7] {
            let capacity = 1usize << log_height;
            for len in [0, 1, 3.min(capacity), capacity / 2, capacity] {
                let mut input = Vec::with_capacity(4 * capacity);
                input.extend((0..len).map(|index| match index % 5 {
                    0 => BinaryField128::ONE,
                    1 => BinaryField128::ZERO,
                    _ => rng.random::<BinaryField128>(),
                }));
                let mut layers = ProductLayers::new_owned::<BinaryField128>(input, log_height);
                let mut product_depth = log_height;
                while product_depth > 0 {
                    let arity: usize = if product_depth % 2 == 1 { 2 } else { 4 };
                    let depth = product_depth - arity.ilog2() as usize;
                    let before = layers
                        .layers
                        .iter()
                        .map(|layer| (layer.values.clone(), reserved(layer)))
                        .collect::<Vec<_>>();
                    if arity == 2 {
                        let expected = layers.binary_children(depth);
                        assert_eq!(layers.take_binary_children(depth), expected);
                    } else {
                        // The taken level is the retained allocation itself, not a copy.
                        let taken = layers.take_layer(depth);
                        assert_eq!(taken.values, before[depth].0);
                        assert_eq!(reserved(&taken), before[depth].1);
                    }
                    for (index, (values, capacity)) in before.iter().enumerate() {
                        if index == depth {
                            assert!(layers.layers[index].values.is_empty());
                            assert_eq!(reserved(&layers.layers[index]), 0);
                        } else {
                            assert_eq!(&layers.layers[index].values, values);
                            assert_eq!(reserved(&layers.layers[index]), *capacity);
                        }
                    }
                    product_depth = depth;
                }
            }
        }
    }

    #[test]
    fn round_messages_match_the_per_node_reference() {
        // Unequal, empty and non-multiple-of-four prefixes leave children of different lengths.
        // Several trees fix the batching power each tree must receive.
        let mut rng = Xoroshiro128Plus::seed_from_u64(0x5107_0002);
        let shapes: [&[usize]; 9] = [
            &[77],
            &[128, 0],
            &[0, 128],
            &[128, 32],
            &[32, 128],
            &[128, 120],
            &[120, 128],
            &[13, 128, 91],
            &[128, 16, 0],
        ];
        for prefix_lens in shapes {
            check_rounds::<BinaryField128, BinaryField128>(&mut rng, 7, prefix_lens);
            check_rounds::<Ghash128, Ghash128>(&mut rng, 7, prefix_lens);
            check_rounds::<BabyBear, BabyBearQuartic>(&mut rng, 7, prefix_lens);
        }
        for log_height in [3, 4, 6] {
            let capacity = 1usize << log_height;
            let prefix_lens = [capacity, 0, capacity - 1];
            check_rounds::<BinaryField128, BinaryField128>(&mut rng, log_height, &prefix_lens);
            check_rounds::<Ghash128, Ghash128>(&mut rng, log_height, &prefix_lens);
            check_rounds::<BabyBear, BabyBearQuartic>(&mut rng, log_height, &prefix_lens);
        }
    }

    #[test]
    fn rounds_spanning_several_tasks_match_the_per_node_reference() {
        // A task reads 1024 values, so leaf levels of 4096 values span four tasks.
        // Prefixes ending inside a task, a block and a row cut each pass at every granularity.
        let mut rng = Xoroshiro128Plus::seed_from_u64(0x5107_0008);
        let prefix_lens = [4096, 3001, 1037];
        check_rounds::<Poly64, Poly192>(&mut rng, 12, &prefix_lens);
        check_rounds::<BabyBear, BabyBearQuartic>(&mut rng, 13, &prefix_lens);
    }

    fn check_identity_tails<Base: Field, F: ExtensionField<Base>>(rng: &mut Xoroshiro128Plus)
    where
        StandardUniform: Distribution<F>,
    {
        // Put the longest prefix in each child slot, including in a later tree.
        // Odd lengths keep one partial pair; zero and full lengths cover both extremes.
        for child_len in [0, 1, 2, 3, 15, 16, 17, 31, 32] {
            for slot in 0..4 {
                for trees in 1..=3 {
                    let mut inputs = vec![vec![F::ONE; 128]; trees];
                    if child_len > 0 {
                        let last = 4 * (child_len - 1) + slot;
                        inputs[trees - 1][last] = F::interpolation_node(2);
                        if last > 4 {
                            // A zero product and interior identities must not move the tail cut.
                            inputs[trees - 1][slot] = F::ZERO;
                        }
                    }
                    check_input_rounds::<Base, F>(rng, 7, &inputs);
                }
            }
        }
    }

    #[test]
    fn identity_tails_match_the_full_domain_reference() {
        let mut rng = Xoroshiro128Plus::seed_from_u64(0x5107_0005);
        check_identity_tails::<BinaryField128, BinaryField128>(&mut rng);
        check_identity_tails::<Ghash128, Ghash128>(&mut rng);
        check_identity_tails::<BabyBear, BabyBearQuartic>(&mut rng);
    }
}
