use crate::compression::PseudoCompressionFunction;

/// A compression that runs `lower` below a fixed tree level and `upper` from that level up.
///
/// It builds mixed-hash Merkle trees.
/// The bulk of the tree, the levels next to the leaves, is hashed with a fast native function.
/// The levels near the root are hashed with an arithmetization-friendly one.
///
/// ```text
///                  root                    ┐
///               /        \                 │ upper: levels upper_from ..
///            u              u              ┘
///          /   \          /   \            ┐
///         l     l        l     l           │ lower: levels 0 .. upper_from
///        / \   / \      / \   / \          ┘
///       leaf digests (hashed by the tree's leaf hasher)
/// ```
///
/// The level at which the switch happens trades prover time against recursion cost.
/// A binary tree over `2^d` leaves does about `2^d - 2^(d - upper_from)` of its compressions with `lower`.
/// A Merkle path checks `upper_from` of those folds, plus one more for each matrix injected below the switch.
/// Raising `upper_from` by one roughly halves the number of `upper` nodes the prover builds.
/// It also adds one `lower` compression to every path a recursive verifier checks.
///
/// # Levels
///
/// Levels count compression steps from the leaves, as [`PseudoCompressionFunction::compress_at_level`] defines them.
/// Level `0` folds leaf digests into their parents.
/// In an `N`-ary tree one level is one `N`-to-1 step.
/// A binary bridge step, which the tree inserts to reach a matrix injected between two `N`-ary steps, is a level of its own.
/// The injection compression of a layer runs at the level of the step that built that layer.
///
/// Both inner functions see the absolute tree level.
/// A `MixedCompression` can therefore serve as `lower` or `upper` of another one, to give three or more tiers.
///
/// # Digests
///
/// Both functions take and return the same digest type, so the tree, its caps and its proofs stay uniform.
/// A byte hash does not natively share a digest type with a field hash.
/// [`SerializingCompression`](crate::SerializingCompression) runs a compression over 32-byte digests on field-element digests, which bridges the two families.
///
/// # Security
///
/// Collision resistance is meant in the hash-tree sense of [`PseudoCompressionFunction`]: the preimage of a non-leaf node must consist of compression outputs.
/// In that sense a mixed tree is as secure as the weaker of its two compressions and its leaf hasher.
/// The prover and the verifier apply the same function at each level.
/// Two different openings that authenticate against one root diverge at a first node, walking down from the root.
/// That node's output agrees while its children differ, and both children are compression outputs, so they form a tree-setting collision of the function used at that level.
/// At the switch level the children are outputs of `lower`, so the tree-setting analysis of `upper` must cover inputs produced by `lower`.
///
/// # Panics
///
/// [`compress`](PseudoCompressionFunction::compress) and [`compress_many`](PseudoCompressionFunction::compress_many) panic.
/// Without a level, a mixed compression does not know which function to run.
/// Silently picking one would build a tree that differs from the one the verifier checks.
#[derive(Clone, Debug)]
pub struct MixedCompression<Lower, Upper> {
    /// Compression used at the levels below `upper_from`.
    lower: Lower,

    /// Compression used at `upper_from` and every level above it.
    upper: Upper,

    /// The first level compressed by `upper`.
    upper_from: usize,
}

impl<Lower, Upper> MixedCompression<Lower, Upper> {
    /// Creates a compression that uses `lower` for levels `0 .. upper_from` and `upper` from `upper_from` on.
    ///
    /// `upper_from = 0` uses `upper` everywhere.
    /// A value at or above the tree's level count uses `lower` everywhere.
    pub const fn new(lower: Lower, upper: Upper, upper_from: usize) -> Self {
        Self {
            lower,
            upper,
            upper_from,
        }
    }

    /// The first level compressed by the upper function.
    pub const fn upper_from(&self) -> usize {
        self.upper_from
    }

    /// The compression used at the levels below [`Self::upper_from`].
    pub const fn lower(&self) -> &Lower {
        &self.lower
    }

    /// The compression used from [`Self::upper_from`] on.
    pub const fn upper(&self) -> &Upper {
        &self.upper
    }

    #[inline]
    const fn is_upper(&self, level: usize) -> bool {
        level >= self.upper_from
    }
}

impl<T, Lower, Upper, const N: usize> PseudoCompressionFunction<T, N>
    for MixedCompression<Lower, Upper>
where
    Lower: PseudoCompressionFunction<T, N>,
    Upper: PseudoCompressionFunction<T, N>,
{
    /// The wider of the two, so that a caller sizing its batches by it serves both functions.
    const LANES: usize = {
        let (lower, upper) = (Lower::LANES, Upper::LANES);
        if lower > upper { lower } else { upper }
    };

    fn compress(&self, _input: [T; N]) -> T {
        panic!("MixedCompression needs a tree level to choose its function; call compress_at_level")
    }

    fn compress_many(&self, _inputs: &[[T; N]], _out: &mut [T])
    where
        T: Clone,
    {
        panic!(
            "MixedCompression needs a tree level to choose its function; call compress_many_at_level"
        )
    }

    #[inline]
    fn compress_at_level(&self, input: [T; N], level: usize) -> T {
        if self.is_upper(level) {
            self.upper.compress_at_level(input, level)
        } else {
            self.lower.compress_at_level(input, level)
        }
    }

    #[inline]
    fn compress_many_at_level(&self, inputs: &[[T; N]], out: &mut [T], level: usize)
    where
        T: Clone,
    {
        if self.is_upper(level) {
            self.upper.compress_many_at_level(inputs, out, level);
        } else {
            self.lower.compress_many_at_level(inputs, out, level);
        }
    }

    #[inline]
    fn lanes_at_level(&self, level: usize) -> usize {
        if self.is_upper(level) {
            self.upper.lanes_at_level(level)
        } else {
            self.lower.lanes_at_level(level)
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use super::*;

    /// Tags its output so a test can tell which function ran, and at which level.
    #[derive(Clone, Debug)]
    struct Tag<const ID: u64, const LANES: usize>;

    impl<const ID: u64, const LANES: usize> PseudoCompressionFunction<[u64; 2], 2> for Tag<ID, LANES> {
        const LANES: usize = LANES;

        fn compress(&self, input: [[u64; 2]; 2]) -> [u64; 2] {
            [ID, input[0][1] + input[1][1]]
        }
    }

    type Two = MixedCompression<Tag<1, 4>, Tag<2, 1>>;

    #[test]
    fn levels_below_the_switch_use_lower_and_the_rest_upper() {
        let c = Two::new(Tag, Tag, 3);
        let input = [[0, 5], [0, 7]];
        for level in 0..3 {
            assert_eq!(c.compress_at_level(input, level), [1, 12], "level {level}");
            assert_eq!(c.lanes_at_level(level), 4);
        }
        for level in 3..8 {
            assert_eq!(c.compress_at_level(input, level), [2, 12], "level {level}");
            assert_eq!(c.lanes_at_level(level), 1);
        }
    }

    #[test]
    fn lanes_is_the_wider_of_the_two() {
        assert_eq!(<Two as PseudoCompressionFunction<[u64; 2], 2>>::LANES, 4);
        assert_eq!(
            <MixedCompression<Tag<1, 1>, Tag<2, 8>> as PseudoCompressionFunction<[u64; 2], 2>>::LANES,
            8
        );
    }

    #[test]
    fn switch_at_zero_is_upper_everywhere() {
        let c = Two::new(Tag, Tag, 0);
        assert_eq!(c.compress_at_level([[0, 1], [0, 2]], 0), [2, 3]);
    }

    #[test]
    fn batched_compression_follows_the_level() {
        let c = Two::new(Tag, Tag, 2);
        let inputs = vec![[[0, 1], [0, 2]], [[0, 3], [0, 4]]];
        let mut out = vec![[0; 2]; 2];

        c.compress_many_at_level(&inputs, &mut out, 1);
        assert_eq!(out, vec![[1, 3], [1, 7]]);

        c.compress_many_at_level(&inputs, &mut out, 2);
        assert_eq!(out, vec![[2, 3], [2, 7]]);
    }

    #[test]
    fn nested_mixes_give_three_tiers_on_absolute_levels() {
        let c = MixedCompression::new(
            MixedCompression::new(Tag::<1, 1>, Tag::<2, 1>, 2),
            Tag::<3, 1>,
            4,
        );
        let tiers: Vec<u64> = (0..6)
            .map(|level| c.compress_at_level([[0, 0], [0, 0]], level)[0])
            .collect();
        assert_eq!(tiers, vec![1, 1, 2, 2, 3, 3]);
    }

    #[test]
    #[should_panic(expected = "needs a tree level")]
    fn compress_without_a_level_panics() {
        let _ = Two::new(Tag, Tag, 1).compress([[0, 0], [0, 0]]);
    }

    #[test]
    #[should_panic(expected = "needs a tree level")]
    fn compress_many_without_a_level_panics() {
        let mut out = [[0; 2]];
        Two::new(Tag, Tag, 1).compress_many(&[[[0, 0], [0, 0]]], &mut out);
    }
}
