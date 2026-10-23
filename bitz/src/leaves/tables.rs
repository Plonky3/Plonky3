//! Per-row lookup tables of folded nodes, indexed by the bits under them (Section 6.1).

use alloc::vec::Vec;

use p3_field::{ExtensionField, Field, PackedFieldExtension, PrimeCharacteristicRing};

/// Gathers each child slot's bits of a word into its own 16-bit lane, for nodes of `4^j` leaves.
///
/// Row `64 w + 4^(j+1) q + 4^j s + i` lands at bit `16 s + 4^j q + i`, so lane `s` lists child `s` of consecutive parents.
#[inline]
pub(super) const fn deal(mut word: u64, j: usize) -> u64 {
    // Each delta swap exchanges one bit of the slot index with one bit of the parent index.
    const fn swap(word: u64, shift: u32, mask: u64) -> u64 {
        let t = (word ^ (word >> shift)) & mask;
        word ^ t ^ (t << shift)
    }
    match j {
        0 => {
            // Two perfect unshuffles move bit `4 q + s` to `16 s + q`.
            let mut round = 0;
            while round < 2 {
                word = swap(word, 1, 0x2222_2222_2222_2222);
                word = swap(word, 2, 0x0C0C_0C0C_0C0C_0C0C);
                word = swap(word, 4, 0x00F0_00F0_00F0_00F0);
                word = swap(word, 8, 0x0000_FF00_0000_FF00);
                word = swap(word, 16, 0x0000_0000_FFFF_0000);
                round += 1;
            }
            word
        }
        // A 4 x 4 transpose of nibbles moves nibble `4 q + s` to `4 s + q`.
        1 => swap(
            swap(word, 12, 0x0000_F0F0_0000_F0F0),
            24,
            0x0000_0000_FF00_FF00,
        ),
        _ => word,
    }
}

/// The bits of `width` consecutive positions starting at `offset` of a lane stream.
#[inline]
pub(super) const fn bits(stream: u128, offset: usize, width: usize) -> u64 {
    ((stream >> offset) as u64) & (u64::MAX >> (64 - width))
}

/// Tables of folded child values `sum_z eq(rho; z) node(p0 + z)` over `2^k` consecutive parents of one child slot.
///
/// A pattern lists the `4^j` leaf bits of each parent's node in parent order, as `deal` lays them out.
pub(super) struct FoldTables<EF> {
    /// Concatenated tables, `stride` entries per pushed fold.
    values: Vec<EF>,
    /// Leaves per node, as `log2`.
    log_node: usize,
    /// Bits of one pattern.
    width: usize,
    /// Entries per pushed fold.
    stride: usize,
}

impl<EF: Field> FoldTables<EF> {
    /// Tables for nodes of `4^j` leaves folded over `k` parent variables, with room for `capacity` folds.
    pub(super) fn new(j: usize, k: usize, capacity: usize) -> Self {
        let log_node = 2 * j;
        let width = (1 << k) << log_node;
        let stride = if log_node > 3 {
            // Two byte tables per node, multiplied together.
            (1 << k) * 512
        } else if width > 8 {
            (width / 8) * 256
        } else {
            1 << width
        };
        Self {
            values: Vec::with_capacity(capacity * stride),
            log_node,
            width,
            stride,
        }
    }

    /// Bits of one pattern.
    #[cfg(test)]
    pub(super) const fn width(&self) -> usize {
        self.width
    }

    /// Forgets every pushed fold, keeping the allocation.
    pub(super) fn clear(&mut self) {
        self.values.clear();
    }

    /// Appends the tables of child `s` over parents `p0..p0 + 2^k`, from the row images and `eq(rho; z)`.
    pub(super) fn push(&mut self, images: &[EF], weights: &[EF], s: usize, p0: usize) {
        let node = 1 << self.log_node;
        // Row of leaf `i` of child `s` of parent `p`.
        let row = |p: usize, i: usize| ((4 * p + s) << self.log_node) + i;
        if self.log_node > 3 {
            // One node is 16 bits: products over each byte, the weight in the low one.
            for (z, &weight) in weights.iter().enumerate() {
                for (half, start) in [(0, weight), (8, EF::ONE)] {
                    let at = self.values.len();
                    self.values.push(start);
                    for b in 0..8 {
                        let image = images[row(p0 + z, half + b)];
                        for x in 0..1 << b {
                            let value = self.values[at + x] * image;
                            self.values.push(value);
                        }
                    }
                }
            }
        } else if node == 1 {
            // Leaf nodes: each group of up to eight parents adds `eq(rho; z) (y - 1)` per set bit.
            for (g, group) in weights.chunks(8).enumerate() {
                let at = self.values.len();
                self.values.push(group.iter().copied().sum());
                for (b, &weight) in group.iter().enumerate() {
                    let step = weight * (images[row(p0 + 8 * g + b, 0)] - EF::ONE);
                    for x in 0..1 << b {
                        let value = self.values[at + x] + step;
                        self.values.push(value);
                    }
                }
            }
        } else {
            // Nibble nodes: a weighted product table per node, two nodes per byte table.
            let product = |z: usize| {
                let mut table = [EF::ZERO; 16];
                table[0] = weights[z];
                for b in 0..4 {
                    let image = images[row(p0 + z, b)];
                    for x in 0..1 << b {
                        table[x | 1 << b] = table[x] * image;
                    }
                }
                table
            };
            if weights.len() == 1 {
                self.values.extend(product(0));
            } else {
                for z in (0..weights.len()).step_by(2) {
                    let (low, high) = (product(z), product(z + 1));
                    self.values
                        .extend((0..256).map(|x| low[x & 15] + high[x >> 4]));
                }
            }
        }
    }

    /// How the pushed tables read a pattern.
    pub(super) const fn kind(&self) -> Kind {
        if self.log_node > 3 {
            Kind::Nodes(self.stride / 512)
        } else if self.width > 8 {
            Kind::Bytes(self.width / 8)
        } else {
            Kind::Direct
        }
    }

    /// The concatenated tables.
    pub(super) fn values(&self) -> &[EF] {
        &self.values
    }

    /// Where the tables of pushed fold `index` start.
    #[inline(always)]
    pub(super) const fn base(&self, index: usize) -> usize {
        index * self.stride
    }

    /// The folded value of pushed fold `index` at `pattern`.
    pub(super) fn eval(&self, index: usize, pattern: u64) -> EF {
        let base = self.base(index);
        match self.kind() {
            Kind::Direct => Direct::eval(&self.values, base, pattern),
            Kind::Bytes(2) => Bytes::<2>::eval(&self.values, base, pattern),
            Kind::Bytes(4) => Bytes::<4>::eval(&self.values, base, pattern),
            Kind::Bytes(_) => Bytes::<8>::eval(&self.values, base, pattern),
            Kind::Nodes(1) => Nodes::<1>::eval(&self.values, base, pattern),
            Kind::Nodes(2) => Nodes::<2>::eval(&self.values, base, pattern),
            Kind::Nodes(_) => Nodes::<4>::eval(&self.values, base, pattern),
        }
    }
}

/// How a set of tables reads a pattern, fixed per round so the hot lookups carry no branch.
pub(super) trait Layout {
    /// The value of the tables starting at `base` for `pattern`.
    fn eval<EF: Field>(values: &[EF], base: usize, pattern: u64) -> EF;

    /// The values at the patterns of every lane, packed.
    #[inline(always)]
    fn eval_packed<F: Field, EF: ExtensionField<F>>(
        values: &[EF],
        base: usize,
        pattern: impl Fn(usize) -> u64,
    ) -> EF::ExtensionPacking {
        EF::ExtensionPacking::from_ext_fn(|l| Self::eval(values, base, pattern(l)))
    }
}

/// The layout of a set of tables.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Kind {
    /// One table indexed by the whole pattern.
    Direct,
    /// Byte tables whose entries add, one per pattern byte.
    Bytes(usize),
    /// Nodes of two byte tables each, multiplied, then added.
    Nodes(usize),
}

/// One table indexed by the whole pattern.
pub(super) struct Direct;

/// `N` byte tables whose entries add.
pub(super) struct Bytes<const N: usize>;

/// `N` nodes of two byte tables each, whose entries multiply.
pub(super) struct Nodes<const N: usize>;

impl Layout for Direct {
    #[inline(always)]
    fn eval<EF: Field>(values: &[EF], base: usize, pattern: u64) -> EF {
        values[base + pattern as usize]
    }
}

impl<const N: usize> Layout for Bytes<N> {
    #[inline(always)]
    fn eval<EF: Field>(values: &[EF], base: usize, pattern: u64) -> EF {
        let mut value = values[base + (pattern & 255) as usize];
        for g in 1..N {
            value += values[base + 256 * g + ((pattern >> (8 * g)) & 255) as usize];
        }
        value
    }
}

impl<const N: usize> Layout for Nodes<N> {
    #[inline(always)]
    fn eval<EF: Field>(values: &[EF], base: usize, pattern: u64) -> EF {
        let mut value = EF::ZERO;
        for z in 0..N {
            let at = base + 512 * z;
            let bits = pattern >> (16 * z);
            value += values[at + (bits & 255) as usize]
                * values[at + 256 + ((bits >> 8) & 255) as usize];
        }
        value
    }

    /// Both byte tables gathered into lanes, so each node costs one packed product.
    #[inline(always)]
    fn eval_packed<F: Field, EF: ExtensionField<F>>(
        values: &[EF],
        base: usize,
        pattern: impl Fn(usize) -> u64,
    ) -> EF::ExtensionPacking {
        let mut value = EF::ExtensionPacking::ZERO;
        for z in 0..N {
            let at = base + 512 * z;
            let byte = |l: usize, half: usize| ((pattern(l) >> (16 * z + 8 * half)) & 255) as usize;
            let low = EF::ExtensionPacking::from_ext_fn(|l| values[at + byte(l, 0)]);
            let high = EF::ExtensionPacking::from_ext_fn(|l| values[at + 256 + byte(l, 1)]);
            value += low * high;
        }
        value
    }
}

#[cfg(test)]
mod tests {
    use p3_binary_field::Poly192;
    use proptest::prelude::*;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::*;

    /// Bit `index` of a word, one bit at a time.
    fn bit(word: u64, index: usize) -> u64 {
        word >> index & 1
    }

    proptest! {
        #[test]
        fn dealing_lists_each_child_slot_in_parent_order(word: u64, j in 0usize..3) {
            let dealt = deal(word, j);
            let node = 1 << (2 * j);
            for s in 0..4 {
                for q in 0..16 / node {
                    for i in 0..node {
                        let row = 4 * node * q + node * s + i;
                        prop_assert_eq!(bit(dealt, 16 * s + node * q + i), bit(word, row));
                    }
                }
            }
        }

        #[test]
        fn tables_fold_the_nodes_they_index(seed: u64, j in 0usize..3, k in 0usize..7, s in 0usize..4) {
            // Patterns reach at most 64 bits.
            prop_assume!(k + 2 * j <= 6);
            let mut rng = SmallRng::seed_from_u64(seed);
            let node = 1 << (2 * j);
            let parents = 1 << k;
            let images: Vec<Poly192> = (0..4 * node * (parents + 1)).map(|_| rng.random()).collect();
            let weights: Vec<Poly192> = (0..parents).map(|_| rng.random()).collect();
            let mut tables = FoldTables::new(j, k, 2);
            tables.push(&images, &weights, s, 0);
            tables.push(&images, &weights, s, 1);
            for _ in 0..8 {
                let pattern = rng.random::<u64>() & (u64::MAX >> (64 - tables.width()));
                for (index, p0) in [0, 1].into_iter().enumerate() {
                    // Each node is the product of its set leaves.
                    let expected: Poly192 = (0..parents)
                        .map(|z| {
                            let leaves: Poly192 = (0..node)
                                .filter(|&i| bit(pattern, z * node + i) == 1)
                                .map(|i| images[(4 * (p0 + z) + s) * node + i])
                                .product();
                            weights[z] * leaves
                        })
                        .sum();
                    prop_assert_eq!(tables.eval(index, pattern), expected);
                }
            }
            tables.clear();
            prop_assert!(tables.values.is_empty());
        }
    }
}
