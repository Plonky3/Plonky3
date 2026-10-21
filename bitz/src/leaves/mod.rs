//! The grand-product leaves of Phase 3: bit `(c, b)` of claim `t` is `y^t_b` when set and one otherwise (Construction 4.4, eq. 39).
//!
//! A leaf takes one of `2^r1 + 1` values, so the bottom radix-four layers are proved off the committed words (Section 6.1).
//!
//! Their first rounds read the bits through buckets and per-row tables (Section 6.2), and patterns come straight from the words (Section 6.3).
//!
//! Only the level of word products and the folded children of a few late rounds are ever stored.

mod bucket;
mod tables;

#[cfg(test)]
mod tests;

use alloc::borrow::Cow;
use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_field::{ExtensionField, Field, PackedFieldExtension, PackedValue, PrimeCharacteristicRing};
use p3_gkr::{LayerRounds, ProductLeaves, ROUND_POLY_LEN, RadixFourBatch};
use p3_maybe_rayon::prelude::*;
use p3_multilinear_util::point::Point;
use p3_util::DisjointMutPtr;
use tracing::instrument;

use self::tables::{Bytes, Direct, FoldTables, Kind, Layout, Nodes, bits, deal};
use crate::params::BitShape;

/// Radix-four layers proved off the words: their children are at most the products of sixteen leaves.
const OWN_LAYERS: usize = 3;

/// A layer stores its folded children once they number at most `2^-HANDOFF` of the leaves.
///
/// A folded child then reads `2^HANDOFF` bits, and tables index patterns of at most 64.
const HANDOFF: usize = 6;

/// The packed extension of `EF` over `F`.
type Packed<F, EF> = <EF as ExtensionField<F>>::ExtensionPacking;

/// Words of one column a task reads per pass.
const TILE_WORDS: usize = 8;

/// Columns a table-driven task streams at least, so its tables pay for themselves.
const MIN_TASK_COLUMNS: usize = 256;

/// Leaves of every claim's trees, laid out `(t, c, b)` with `b` least significant, read off the committed words.
pub(crate) struct BitLeaves<'a, F, EF> {
    /// The committed words of the shape's bits.
    pub(crate) words: &'a [u64],
    /// The shape.
    pub(crate) shape: BitShape,
    /// `y^t = g^(gamma^t)`, one row image per row of each claim; trees past them are all ones.
    pub(crate) images: &'a [Vec<EF>],
    /// The base field the dense rounds pack over.
    pub(crate) base: PhantomData<F>,
}

impl<'a, F, EF: Field> BitLeaves<'a, F, EF> {
    /// Words of column `c` from word `w` on.
    #[inline]
    fn column(&self, c: usize, w: usize, len: usize) -> &[u64] {
        &self.words[c * self.shape.column_words() + w..][..len]
    }

    /// Words per tile, and tiles per column.
    const fn tiles(&self) -> (usize, usize) {
        let words = self.shape.column_words();
        let tile = if words < TILE_WORDS {
            words
        } else {
            TILE_WORDS
        };
        (tile, words / tile)
    }

    /// Column ranges splitting the columns just enough that every thread gets a few tasks.
    fn column_chunks(&self) -> Vec<core::ops::Range<usize>> {
        let columns = self.shape.columns();
        let tasks = self.images.len() * self.tiles().1;
        let chunks = (4 * current_num_threads())
            .div_ceil(tasks)
            .clamp(1, (columns / MIN_TASK_COLUMNS).max(1));
        let len = columns.div_ceil(chunks);
        (0..chunks)
            .map(|chunk| chunk * len..columns.min((chunk + 1) * len))
            .collect()
    }
}

impl<F: Field, EF: ExtensionField<F>> ProductLeaves<EF> for BitLeaves<'_, F, EF> {
    type Layer<'a>
        = BitLayer<'a, F, EF>
    where
        Self: 'a;

    fn own_layers(&self) -> usize {
        OWN_LAYERS
    }

    /// The product of each word's set images, laid out `(t, c, w)`.
    #[instrument(skip_all, name = "bitz leaves")]
    fn materialize(&self) -> Cow<'_, [EF]> {
        let (columns, column_words) = (self.shape.columns(), self.shape.column_words());
        let (tile_words, tiles) = self.tiles();
        let mut level = EF::zero_vec(self.images.len() * columns * column_words);
        let out = DisjointMutPtr::new(&mut level);
        let tasks = self.images.len() * tiles;
        (0..tasks).into_par_iter().for_each(|task| {
            let (t, tile) = (task / tiles, task % tiles);
            let first_word = tile * tile_words;
            // One product table per byte of the tile's rows.
            let mut tables = EF::zero_vec(tile_words * 8 * 256);
            for (byte, table) in tables.as_chunks_mut::<256>().0.iter_mut().enumerate() {
                table[0] = EF::ONE;
                for b in 0..8 {
                    let image = self.images[t][64 * first_word + 8 * byte + b];
                    for x in 0..1 << b {
                        table[x | 1 << b] = table[x] * image;
                    }
                }
            }
            // Lane `l` of a packed product belongs to column `c + l`.
            let lanes = F::Packing::WIDTH;
            for c in (0..columns).step_by(lanes) {
                let live = lanes.min(columns - c);
                for wi in 0..tile_words {
                    let word = |l: usize| {
                        self.words[(c + l.min(live - 1)) * column_words + first_word + wi]
                    };
                    let product: Packed<F, EF> = (0..8)
                        .map(|byte| {
                            Packed::<F, EF>::from_ext_fn(|l| {
                                tables[(8 * wi + byte) * 256
                                    + ((word(l) >> (8 * byte)) & 255) as usize]
                            })
                        })
                        .product();
                    for (l, value) in Packed::<F, EF>::to_ext_iter([product])
                        .take(live)
                        .enumerate()
                    {
                        // SAFETY: task `(t, tile)` alone writes words `first_word..+tile_words` of row `(t, c + l)`.
                        let row = unsafe {
                            out.slice_mut(
                                (t * columns + c + l) * column_words + first_word,
                                tile_words,
                            )
                        };
                        row[wi] = value;
                    }
                }
            }
        });
        Cow::Owned(level)
    }

    fn layer(&self, j: usize, point: &[EF]) -> Self::Layer<'_> {
        BitLayer::new(self, j, point)
    }
}

/// Bottom radix-four layer `j`, whose children are nodes of `4^j` leaves.
///
/// Its row rounds read the words; from round `k` with `k + 2j = HANDOFF` it folds its children out and plays the dense rounds.
pub(crate) struct BitLayer<'a, F: Field, EF: ExtensionField<F>> {
    /// The leaves.
    leaves: &'a BitLeaves<'a, F, EF>,
    /// The layer.
    j: usize,
    /// The parent point in binding order: row, column, then tree coordinates.
    point: Vec<EF>,
    /// Row challenges bound so far.
    bound: Vec<EF>,
    /// Rounds played so far.
    round: usize,
    /// The dense rounds once the children are stored.
    dense: Option<RadixFourBatch<'a, F, EF, 4>>,
}

impl<'a, F: Field, EF: ExtensionField<F>> BitLayer<'a, F, EF> {
    fn new(leaves: &'a BitLeaves<'a, F, EF>, j: usize, point: &[EF]) -> Self {
        let mut layer = Self {
            leaves,
            j,
            point: point.to_vec(),
            bound: Vec::new(),
            round: 0,
            dense: None,
        };
        layer.store_if_due();
        layer
    }

    /// Parent row variables, bound first.
    const fn row_rounds(&self) -> usize {
        self.leaves.shape.log_rows() - 2 * self.j - 2
    }

    /// The round from which the children are stored.
    fn handoff(&self) -> usize {
        HANDOFF.saturating_sub(2 * self.j).min(self.row_rounds())
    }

    /// Stores the children once the handoff round is reached.
    fn store_if_due(&mut self) {
        if self.dense.is_none() && self.round == self.handoff() {
            self.dense = Some(RadixFourBatch::from_level(self.folded_children()));
        }
    }

    /// `eq` over the row, column and tree coordinates after the current one.
    fn equalities(&self) -> [Vec<EF>; 3] {
        let (rows, log_columns) = (self.row_rounds(), self.leaves.shape.log_columns());
        [
            &self.point[self.round + 1..rows],
            &self.point[rows..rows + log_columns],
            &self.point[rows + log_columns..],
        ]
        .map(|coordinates| Point::new(coordinates).equality_weights_lsb())
    }

    /// The degree-four row sum of the current row round, in coefficients, before the current `eq` factor.
    #[instrument(skip_all, name = "bitz row round")]
    fn row_sum(&self) -> [EF; 5] {
        let [rows, columns, trees] = self.equalities();
        let sum = match (self.j, self.round) {
            (0, 0) => bucket::byte_round(self.leaves, &rows, &columns, &trees),
            (0, 1) => bucket::pair_round(self.leaves, &self.bound, &rows, &columns, &trees),
            _ => self.table_round(&rows, &columns, &trees),
        };
        // Trees past the claims are all ones, and every other `eq` sums to one over its cube.
        let mut sum = sum;
        sum[0] += trees[self.leaves.images.len()..]
            .iter()
            .copied()
            .sum::<EF>();
        sum
    }

    /// The row sum of a round whose endpoints are read from per-block tables.
    fn table_round(&self, rows: &[EF], columns: &[EF], trees: &[EF]) -> [EF; 5] {
        match FoldTables::<EF>::new(self.j, self.round, 0).kind() {
            Kind::Direct => self.table_round_as::<Direct, false>(rows, columns, trees),
            Kind::Bytes(2) => self.table_round_as::<Bytes<2>, false>(rows, columns, trees),
            Kind::Bytes(_) => self.table_round_as::<Bytes<4>, true>(rows, columns, trees),
            Kind::Nodes(1) => self.table_round_as::<Nodes<1>, false>(rows, columns, trees),
            Kind::Nodes(_) => self.table_round_as::<Nodes<2>, true>(rows, columns, trees),
        }
    }

    /// The row sum through tables of layout `L`, one task per tree, tile and column range; `WIDE` endpoints span two dealt words.
    ///
    /// A task deals its words once, then streams them block by block so each block's tables stay in cache.
    fn table_round_as<L: Layout, const WIDE: bool>(
        &self,
        rows: &[EF],
        columns: &[EF],
        trees: &[EF],
    ) -> [EF; 5] {
        let leaves = self.leaves;
        let (j, k) = (self.j, self.round);
        let width = 1 << (k + 2 * j);
        let mask = u64::MAX >> (64 - width);
        let (tile_words, tiles) = leaves.tiles();
        // Each block covers `2 width` lane bits of every child slot.
        let blocks = tile_words * 16 / (2 * width);
        let weights = Point::new(self.bound.as_slice()).equality_weights_lsb();
        let chunks = leaves.column_chunks();
        let tasks = leaves.images.len() * tiles * chunks.len();
        let lanes_width = F::Packing::WIDTH;
        let partials: Vec<[EF; 5]> = (0..tasks)
            .into_par_iter()
            .map(|task| {
                let (t, tile, chunk) = (
                    task / (tiles * chunks.len()),
                    task / chunks.len() % tiles,
                    task % chunks.len(),
                );
                let range = chunks[chunk].clone();
                let (dealt, column_weights) = deal_chunk::<F, EF>(leaves, j, tile, range, columns);
                let stride = column_weights.len() * lanes_width;
                let mut tables = FoldTables::new(j, k, 8);
                let mut sum = [EF::ZERO; 5];
                for b in 0..blocks {
                    let block = tile * blocks + b;
                    tables.clear();
                    for s in 0..4 {
                        for side in 0..2 {
                            tables.push(&leaves.images[t], &weights, s, (2 * block + side) << k);
                        }
                    }
                    let values = tables.values();
                    // A narrow pattern sits in one 16-bit lane of a dealt word, a 32-bit one in two consecutive words.
                    let at = |side: usize| 2 * width * b + side * width;
                    let words = |side: usize, next: usize| {
                        &dealt[(at(side) / 16 + next) * stride..][..stride]
                    };
                    let mut products = [Packed::<F, EF>::ZERO; 6];
                    for (g, &weight) in column_weights.iter().enumerate() {
                        let endpoint = |s: usize, side: usize| {
                            let lanes = &words(side, 0)[g * lanes_width..][..lanes_width];
                            let (base, shift) = (tables.base(2 * s + side), 16 * s + at(side) % 16);
                            if !WIDE {
                                L::eval_packed::<F, EF>(values, base, |l| {
                                    (lanes[l] >> shift) & mask
                                })
                            } else {
                                let high = &words(side, 1)[g * lanes_width..][..lanes_width];
                                L::eval_packed::<F, EF>(values, base, |l| {
                                    (lanes[l] >> shift) & 0xffff
                                        | ((high[l] >> shift) & 0xffff) << 16
                                })
                            }
                        };
                        let even = core::array::from_fn(|s| endpoint(s, 0));
                        let odd = core::array::from_fn(|s| endpoint(s, 1));
                        accumulate(&mut products, weight, even, odd);
                    }
                    let products = products.map(|p| Packed::<F, EF>::to_ext_iter([p]).sum::<EF>());
                    let weight = rows[block] * trees[t];
                    sum = add(sum, coefficients(products).map(|c| weight * c));
                }
                sum
            })
            .collect();
        partials.into_iter().fold([EF::ZERO; 5], add)
    }

    /// The children after the rounds bound so far, laid out `(t, c, r, s)`.
    #[instrument(skip_all, name = "bitz fold children")]
    fn folded_children(&self) -> Vec<EF> {
        match FoldTables::<EF>::new(self.j, self.round, 0).kind() {
            Kind::Direct => self.folded_children_as::<Direct>(),
            Kind::Bytes(2) => self.folded_children_as::<Bytes<2>>(),
            Kind::Bytes(4) => self.folded_children_as::<Bytes<4>>(),
            Kind::Bytes(_) => self.folded_children_as::<Bytes<8>>(),
            Kind::Nodes(1) => self.folded_children_as::<Nodes<1>>(),
            Kind::Nodes(2) => self.folded_children_as::<Nodes<2>>(),
            Kind::Nodes(_) => self.folded_children_as::<Nodes<4>>(),
        }
    }

    /// The children through tables of layout `L`.
    fn folded_children_as<L: Layout>(&self) -> Vec<EF> {
        let leaves = self.leaves;
        let (j, k) = (self.j, self.round);
        let width = 1 << (k + 2 * j);
        let columns = leaves.shape.columns();
        let rows = leaves.shape.rows() / (4 * width);
        let (tile_words, tiles) = leaves.tiles();
        let tile_rows = tile_words * 16 / width;
        let weights = Point::new(self.bound.as_slice()).equality_weights_lsb();
        let mut children = EF::zero_vec(4 * leaves.images.len() * columns * rows);
        let out = DisjointMutPtr::new(&mut children);
        let tasks = leaves.images.len() * tiles;
        (0..tasks).into_par_iter().for_each(|task| {
            let (t, tile) = (task / tiles, task % tiles);
            let mut tables = FoldTables::new(j, k, 4 * tile_rows);
            for r in tile * tile_rows..(tile + 1) * tile_rows {
                for s in 0..4 {
                    tables.push(&leaves.images[t], &weights, s, r << k);
                }
            }
            for c in 0..columns {
                let lanes = lanes(leaves.column(c, tile * tile_words, tile_words), j);
                // SAFETY: task `(t, tile)` alone writes rows `tile * tile_rows..+tile_rows` of `(t, c)`.
                let row = unsafe {
                    out.slice_mut(
                        4 * ((t * columns + c) * rows + tile * tile_rows),
                        4 * tile_rows,
                    )
                };
                for (r, entry) in row.as_chunks_mut::<4>().0.iter_mut().enumerate() {
                    for (s, value) in entry.iter_mut().enumerate() {
                        *value = L::eval(
                            tables.values(),
                            tables.base(4 * r + s),
                            bits(lanes[s], width * r, width),
                        );
                    }
                }
            }
        });
        children
    }
}

impl<F: Field, EF: ExtensionField<F>> LayerRounds<EF, 4> for BitLayer<'_, F, EF> {
    const READS_EQUALITY: bool = false;

    fn round(
        &mut self,
        _: &[EF],
        equality_pair: [EF; 2],
        logical_len: usize,
        batching: EF,
    ) -> [EF; ROUND_POLY_LEN] {
        if let Some(dense) = &mut self.dense {
            let equality = Point::new(&self.point[self.round + 1..]).equality_weights_lsb();
            return dense.round(&equality, equality_pair, logical_len, batching);
        }
        let sum = self.row_sum();
        [0, 2, 3, 4, 5].map(|node| {
            let node = EF::interpolation_node(node);
            let value = sum
                .iter()
                .rev()
                .fold(EF::ZERO, |value, &c| value * node + c);
            (equality_pair[0] + node * (equality_pair[1] - equality_pair[0])) * value
        })
    }

    fn fold(&mut self, challenge: EF, logical_len: usize) {
        match &mut self.dense {
            Some(dense) => dense.fold(challenge, logical_len),
            None => self.bound.push(challenge),
        }
        self.round += 1;
        self.store_if_due();
    }

    fn children(&self) -> Vec<[EF; 4]> {
        self.dense
            .as_ref()
            .expect("the children are stored by the last row round")
            .children()
    }
}

/// The dealt words of a tile over a column range, word-major and padded to whole lane groups, with the packed column weights.
fn deal_chunk<F: Field, EF: ExtensionField<F>>(
    leaves: &BitLeaves<'_, F, EF>,
    j: usize,
    tile: usize,
    range: core::ops::Range<usize>,
    columns: &[EF],
) -> (Vec<u64>, Vec<Packed<F, EF>>) {
    let lanes_width = F::Packing::WIDTH;
    let (tile_words, _) = leaves.tiles();
    let groups = range.len().div_ceil(lanes_width);
    let stride = groups * lanes_width;
    let mut dealt = alloc::vec![0u64; tile_words * stride];
    for (i, c) in range.clone().enumerate() {
        for (w, &word) in leaves
            .column(c, tile * tile_words, tile_words)
            .iter()
            .enumerate()
        {
            dealt[w * stride + i] = deal(word, j);
        }
    }
    // Lane `l` of group `g` is column `range.start + g * lanes + l`; padding weighs zero.
    let weights = (0..groups)
        .map(|g| {
            Packed::<F, EF>::from_ext_fn(|l| {
                let i = g * lanes_width + l;
                if i < range.len() {
                    columns[range.start + i]
                } else {
                    EF::ZERO
                }
            })
        })
        .collect();
    (dealt, weights)
}

/// Each child slot's bits of the given words, consecutive parents in order (Section 6.3).
#[inline]
fn lanes(words: &[u64], j: usize) -> [u128; 4] {
    let mut lanes = [0u128; 4];
    for (i, &word) in words.iter().enumerate() {
        let dealt = deal(word, j);
        for (s, lane) in lanes.iter_mut().enumerate() {
            *lane |= u128::from((dealt >> (16 * s)) as u16) << (16 * i);
        }
    }
    lanes
}

/// `(e0 + (o0 - e0) X) (e1 + (o1 - e1) X)` in coefficients, by three products.
#[inline]
fn quadratic<A: PrimeCharacteristicRing + Copy>(e0: A, o0: A, e1: A, o1: A) -> [A; 3] {
    let low = e0 * e1;
    let high = (o0 - e0) * (o1 - e1);
    [low, o0 * o1 - low - high, high]
}

/// Adds the six Karatsuba products of two quadratics to `products`.
#[inline]
fn karatsuba<A: PrimeCharacteristicRing + Copy>(products: &mut [A; 6], u: [A; 3], v: [A; 3]) {
    products[0] += u[0] * v[0];
    products[1] += u[1] * v[1];
    products[2] += u[2] * v[2];
    products[3] += (u[0] + u[1]) * (v[0] + v[1]);
    products[4] += (u[1] + u[2]) * (v[1] + v[2]);
    products[5] += (u[0] + u[2]) * (v[0] + v[2]);
}

/// Adds `weight prod_s (even_s + (odd_s - even_s) X)` to `products`, by fourteen products.
#[inline]
fn accumulate<A: PrimeCharacteristicRing + Copy>(
    products: &mut [A; 6],
    weight: A,
    even: [A; 4],
    odd: [A; 4],
) {
    let u = quadratic(weight * even[0], weight * odd[0], even[1], odd[1]);
    let v = quadratic(even[2], odd[2], even[3], odd[3]);
    karatsuba(products, u, v);
}

/// The quartic whose two quadratic halves gave the summed Karatsuba `products`.
#[inline]
fn coefficients<EF: Field>(p: [EF; 6]) -> [EF; 5] {
    [
        p[0],
        p[3] - p[0] - p[1],
        p[5] - p[0] - p[2] + p[1],
        p[4] - p[1] - p[2],
        p[2],
    ]
}

/// Coefficient-wise sum.
#[inline]
fn add<EF: Field>(mut a: [EF; 5], b: [EF; 5]) -> [EF; 5] {
    for (a, b) in a.iter_mut().zip(b) {
        *a += b;
    }
    a
}
