//! Identity-padded storage and folding state used by the product prover.

use alloc::borrow::Cow;
use alloc::vec::Vec;
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
    fn reduce(&self, arity: usize) -> IdentityPrefix<'static, F> {
        // A partial final group is completed by implicit identity factors.
        let values = self
            .values
            .par_chunks(arity)
            .map_collect_min_task_bytes((arity + 1) * size_of::<F>(), |chunk| {
                chunk.iter().copied().product()
            });
        IdentityPrefix::new(values)
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
    pub(super) fn new(leaves: &'a [F], log_height: usize) -> Self {
        Self::from_leaves(IdentityPrefix::new(leaves), log_height)
    }

    /// Builds the retained levels while taking ownership of the leaf allocation.
    pub(super) fn new_owned(leaves: Vec<F>, log_height: usize) -> Self {
        Self::from_leaves(IdentityPrefix::new(leaves), log_height)
    }

    /// Builds every retained level above the given leaves.
    fn from_leaves(leaves: IdentityPrefix<'a, F>, log_height: usize) -> Self {
        // Unvisited depths remain empty constant-one prefixes.
        let mut layers = (0..=log_height)
            .map(|_| IdentityPrefix::new(Vec::new()))
            .collect::<Vec<_>>();
        layers[0] = leaves;

        // Two multiplication levels are retained per radix-four layer.
        let mut depth = 0;
        while depth + 2 <= log_height {
            layers[depth + 2] = layers[depth].reduce(4);
            depth += 2;
        }
        if depth < log_height {
            layers[log_height] = layers[depth].reduce(2);
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

/// Blocks one task of a pass reads.
///
/// A fused fold writes a quarter of them, which stays in the first-level cache for its sums.
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
/// Rows past the explicit prefix are all ones.
struct RadixFourState<'a, F: Clone> {
    /// Explicit child values after removing trailing identities.
    values: IdentityPrefix<'a, F>,
    /// The buffer a fold writes, then swaps with the values.
    spare: Vec<F>,
}

impl<'a, F: Field> RadixFourState<'a, F> {
    /// Wraps a retained level without moving its values.
    const fn new(values: IdentityPrefix<'a, F>) -> Self {
        Self {
            values,
            spare: Vec::new(),
        }
    }

    /// Bytes one task of a pass is charged.
    ///
    /// Each block costs dozens of products, so its bytes are weighted by a few products each.
    const fn task_bytes() -> usize {
        4 * TASK_BLOCKS * BLOCK * size_of::<F>()
    }

    /// Sums the polynomial of the next two rounds over every block.
    fn grid<Base: Field>(&self, weights: &BlockWeights<F>) -> GridSums<F>
    where
        F: ExtensionField<Base>,
    {
        let values = &self.values.values;
        values
            .par_chunks(TASK_BLOCKS * BLOCK)
            .enumerate()
            .map_collect_min_task_bytes(Self::task_bytes(), |(task, chunk)| {
                let mut sums = GridSums::new();
                sums.add_blocks::<Base>(chunk, task * TASK_BLOCKS, weights);
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
        let source = &self.values.values;

        // Every block, the last one possibly short, folds into one row.
        let len = 4 * source.len().div_ceil(BLOCK);
        self.spare.clear();
        self.spare.reserve(len);
        let destination = &mut self.spare.spare_capacity_mut()[..len];

        // Source and destination split into the same number of chunks, one per task.
        let partials = destination
            .par_chunks_mut(4 * TASK_BLOCKS)
            .zip(source.par_chunks(TASK_BLOCKS * BLOCK))
            .enumerate()
            .map_collect_min_task_bytes(Self::task_bytes(), |(task, (out, chunk))| {
                fold_blocks::<Base, F>(chunk, out, challenges);
                // SAFETY: the fold wrote every slot of `out`, and `MaybeUninit<F>` has `F`'s layout.
                let rows =
                    unsafe { core::slice::from_raw_parts(out.as_ptr().cast::<F>(), out.len()) };

                // The rows just written are still in cache, so the next grid reads no memory.
                weights.map(|weights| {
                    let mut sums = GridSums::new();
                    sums.add_blocks::<Base>(rows, task * TASK_BLOCKS / 4, weights);
                    sums
                })
            });

        // SAFETY: the tasks' chunks cover the first `len` slots, and each task wrote all of its own.
        unsafe { self.spare.set_len(len) };
        // The folded rows become the values, and an owned source becomes the next spare.
        let folded = Cow::Owned(core::mem::take(&mut self.spare));
        if let Cow::Owned(source) = core::mem::replace(&mut self.values.values, folded) {
            self.spare = source;
        }
        self.values.trim();

        weights.map(|_| {
            partials
                .into_iter()
                .map(|sums| sums.expect("every task sums when weights are given"))
                .fold(GridSums::new(), GridSums::merge)
        })
    }

    /// Binds the last variable of a layer with two rows left.
    fn fold_last(&mut self, challenge: F) {
        let children: [F; 4] = core::array::from_fn(|child| {
            interpolate_pair(
                [self.values.get(child), self.values.get(4 + child)],
                challenge,
            )
        });
        self.values = IdentityPrefix::new(children.to_vec());
    }

    /// The four lines' product at every node, for a layer with two rows left.
    fn last_product(&self, nodes: &RoundNodes<F>) -> [F; ROUND_POLY_LEN] {
        let line = |child: usize| nodes.line(self.values.get(child), self.values.get(4 + child));
        let mut product = line(0);
        for child in 1..4 {
            for (product, value) in product.iter_mut().zip(line(child)) {
                *product *= value;
            }
        }
        product
    }

    /// Reads the four terminal child claims after every parent variable is bound.
    fn children(&self) -> [F; 4] {
        core::array::from_fn(|child| self.values.get(child))
    }
}

/// Folds each block of `source` into one row of `out`, binding its two row bits.
///
/// A short last block is completed by implicit identities.
#[inline]
fn fold_blocks<Base: Field, F: ExtensionField<Base>>(
    source: &[F],
    out: &mut [MaybeUninit<F>],
    [low, high]: [F; 2],
) {
    // Row `2 b_2 + b_1` of a block is its corner at `(Y_1, Y_2) = (b_1, b_2)`.
    //
    //     lower  = e_0 + r_0 (e_1 - e_0)
    //     upper  = e_2 + r_0 (e_3 - e_2)
    //     folded = lower + r_1 (upper - lower)
    let fold = |block: &[F; BLOCK], row: &mut [MaybeUninit<F>]| {
        for (child, slot) in row.iter_mut().enumerate() {
            let corner = |index: usize| block[4 * index + child];
            let lower = interpolate_pair([corner(0), corner(1)], low);
            let upper = interpolate_pair([corner(2), corner(3)], low);
            slot.write(interpolate_pair([lower, upper], high));
        }
    };
    let (blocks, short) = source.as_chunks::<BLOCK>();

    // Whole groups go through the packed extension, lane `4 b + c` holding child `c` of block `b`.
    // Those lanes are the group's output rows in order.
    let lanes = Base::Packing::WIDTH;
    let packed = if lanes % 4 == 0 && (4..=MAX_LANES).contains(&lanes) {
        let group_blocks = lanes / 4;
        let mut folded = [F::ZERO; MAX_LANES];
        for (group, slots) in blocks
            .chunks_exact(group_blocks)
            .zip(out.chunks_exact_mut(lanes))
        {
            let corner = |index: usize| {
                F::ExtensionPacking::from_ext_fn(|lane| group[lane / 4][4 * index + lane % 4])
            };
            let (e0, e1, e2, e3) = (corner(0), corner(1), corner(2), corner(3));
            let lower = e0 + (e1 - e0) * low;
            let upper = e2 + (e3 - e2) * low;
            (lower + (upper - lower) * high).to_ext_slice(&mut folded[..lanes]);
            for (slot, &value) in slots.iter_mut().zip(&folded) {
                slot.write(value);
            }
        }
        blocks.len() / group_blocks * group_blocks
    } else {
        0
    };

    // The blocks short of a group go one at a time.
    let (rows, _) = out.as_chunks_mut::<4>();
    for (block, row) in blocks[packed..].iter().zip(&mut rows[packed..]) {
        fold(block, row);
    }
    if !short.is_empty() {
        fold(&padded_block(short), &mut rows[blocks.len()]);
    }
}

/// Completes a short block with implicit identities.
#[inline]
fn padded_block<F: Field>(short: &[F]) -> [F; BLOCK] {
    let mut block = [F::ONE; BLOCK];
    block[..short.len()].copy_from_slice(short);
    block
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

/// Per-tree states reduced by one shared radix-four sumcheck.
pub(super) struct RadixFourBatch<'a, F: Clone> {
    /// One folding state for each product tree.
    states: Vec<RadixFourState<'a, F>>,
}

impl<'a, F: Field> RadixFourBatch<'a, F> {
    /// Creates the batched states from one retained level per tree, borrowing each.
    #[cfg(test)]
    pub(super) fn new(layers: &'a [ProductLayers<'_, F>], depth: usize) -> Self {
        let states = layers
            .iter()
            .map(|layers| RadixFourState::new(IdentityPrefix::new(&*layers.layers[depth].values)))
            .collect();
        Self { states }
    }

    /// Consumes one retained level per tree, reading it in place.
    pub(super) fn new_consuming(layers: &mut [ProductLayers<'a, F>], depth: usize) -> Self {
        let states = layers
            .iter_mut()
            .map(|layers| RadixFourState::new(layers.take_layer(depth)))
            .collect();
        Self { states }
    }

    /// Runs every sumcheck round of the layer.
    ///
    /// Round `j` binds the lowest remaining row bit.
    /// Its rows are weighed by `eq(point[j + 1..], .)`, and the line `eq(point[j], X)` scales its message.
    ///
    /// Rounds run in pairs, one pass over the rows per pair.
    /// An odd count leaves one round, over two rows.
    ///
    /// `send` takes each round's message and returns its challenge.
    pub(super) fn prove_rounds<Base: Field>(
        &mut self,
        point: &[F],
        batching: F,
        mut send: impl FnMut([F; ROUND_POLY_LEN]) -> F,
    ) -> Vec<F>
    where
        F: ExtensionField<Base>,
    {
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
                .map(|state| state.grid::<Base>(&weights))
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
                .map(|state| state.fold_two::<Base>([low, high], weights.as_ref()))
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
    pub(super) fn children(&self) -> Vec<[F; 4]> {
        self.states.iter().map(RadixFourState::children).collect()
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
            .map(|input| ProductLayers::new(input, log_height))
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
                for challenge_kind in 0..3 {
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
                        let challenges = batch.prove_rounds::<Base>(&anchor, batching, |message| {
                            assert_eq!(
                                message,
                                per_node_round(&trees, &equality, batching),
                                "depth={depth} round={round} anchor={anchor_kind} challenge={challenge_kind}",
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
                        });
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

    #[test]
    fn borrowed_leaves_are_read_in_place_until_the_first_fold() {
        // The prover only reads the leaves, so a borrowed input is never copied.
        //
        // Fixture state: 2^6 leaves, two trailing ones the trim drops without copying.
        let mut leaves = vec![Poly192::interpolation_node(2); 64];
        leaves[62] = Poly192::ONE;
        leaves[63] = Poly192::ONE;
        let mut layers = ProductLayers::new(&leaves, 6);
        let taken = layers.take_layer(0);
        assert!(
            matches!(taken.values, Cow::Borrowed(values) if values.as_ptr() == leaves.as_ptr())
        );
        assert_eq!(taken.values.len(), 62);

        // The first fold writes owned rows and leaves the borrowed source untouched.
        let mut state = RadixFourState::new(taken);
        let _ = state.fold_two::<Poly64>([Poly192::ZERO, Poly192::ONE], None);
        assert!(matches!(state.values.values, Cow::Owned(_)));
        assert!(state.spare.is_empty());
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
                let mut layers = ProductLayers::new_owned(input, log_height);
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
