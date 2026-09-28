//! The T5 compression function of Dodis, Khovratovich, Mouha and Nandi.
//!
//! T5 compresses five digests with three calls to 2-to-1 compressions `h1`, `h2`, `h3`:
//!
//! ```text
//!     T5(m1, m2, m3, m4, m5) = h3(h1(m1, m2) + m5, h2(m3, m4) + m5) + m5
//! ```
//!
//! A plain two-level subtree spends the same three calls on only four digests.
//!
//! A Merkle tree built from T5 nodes therefore takes a quarter fewer compression calls.
//!
//! Its depth, counted in sequential calls, shrinks from `log2 t` to `2 log5 t`, about `0.86 log2 t`.
//!
//! # Security
//!
//! In the ideal-function model T5 keeps the birthday bound of its 2-to-1 compressions.
//!
//! - Collisions: advantage at most `(n^2 + 10) q^2 / 2^n` for `n`-bit digests and `q` queries (Theorem 1).
//! - Preimages: advantage at most `2 q^3 / 2^(2n) + O(n q / 2^n)` (Theorem 2).
//!
//! The three compressions must be independent functions.
//!
//! With `h1 = h2` or any other coincidence, a collision costs only `2^(n/4)` queries (Section 8.4).
//!
//! Instances of one permutation with independent round constants, or one hash under three keys, qualify.
//!
//! The final `+ m5` is load-bearing: without it the construction falls to a `2^(n/4)` attack too (Section 8.5).
//!
//! The paper works over bit strings with `+` as XOR.
//!
//! Its arguments use only the group law, so field digests with field addition get the same bounds.
//!
//! There `n` is `log2` of the digest space.
//!
//! # Openings
//!
//! A conservative opening of a child reveals its four siblings, like any 5-ary node.
//!
//! An aggressive opening reveals only three digests and recomputes two calls instead of three.
//!
//! See [`T5::open_aggressive`] for the price in security.
//!
//! # References
//!
//! - Dodis, Khovratovich, Mouha, Nandi. *T5: Hashing Five Inputs with Three Compression Calls*. [2021/373](https://eprint.iacr.org/2021/373)

use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_field::PrimeCharacteristicRing;

use crate::{CompressionFunction, PseudoCompressionFunction};

/// Number of children of a T5 node.
pub const T5_ARITY: usize = 5;

/// Number of digests an aggressive opening of one T5 child reveals.
pub const T5_AGGRESSIVE_OPENING_LEN: usize = 3;

/// Groups staged per batched call, bounding the scratch of [`T5::compress_many`].
///
/// It is a multiple of every lane count a batched compression reports.
const BATCH_CHUNK: usize = 256;

/// An abelian group law on digest words: the `+` that T5 masks its halves with.
///
/// T5 lifts the law word by word to whole digests.
///
/// Its security argument needs a group, so the law must be associative, commutative and invertible.
pub trait GroupLaw<W> {
    /// Combine two words.
    fn add(a: W, b: W) -> W;
}

/// Bitwise XOR, the law of the paper, for integer and byte digest words.
///
/// It extends lane by lane to arrays of words, the digest words of a lane-packed hasher.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Xor;

macro_rules! impl_xor {
    ($($word:ty),*) => {$(
        impl GroupLaw<$word> for Xor {
            #[inline(always)]
            fn add(a: $word, b: $word) -> $word {
                a ^ b
            }
        }
    )*};
}

impl_xor!(u8, u16, u32, u64, u128, usize);

impl<W: Copy, const LANES: usize> GroupLaw<[W; LANES]> for Xor
where
    Self: GroupLaw<W>,
{
    #[inline(always)]
    fn add(a: [W; LANES], b: [W; LANES]) -> [W; LANES] {
        core::array::from_fn(|lane| Self::add(a[lane], b[lane]))
    }
}

/// Field addition, for digests made of field elements or of packed field vectors.
///
/// Over a binary field it is XOR again.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FieldAdd;

impl<R: PrimeCharacteristicRing> GroupLaw<R> for FieldAdd {
    #[inline(always)]
    fn add(a: R, b: R) -> R {
        a + b
    }
}

/// The T5 5-to-1 compression, built from three independent 2-to-1 compressions.
///
/// - `C`: the type of the three inner compressions.
/// - `L`: the group law on digest words, [`Xor`] for bytes and integers, [`FieldAdd`] for fields.
///
/// A binary step of a tree built on T5 uses `h1` alone, through [`PseudoCompressionFunction::compress_prefix`].
#[derive(Clone, Debug)]
pub struct T5<C, L> {
    /// Compresses the first pair of children.
    h1: C,

    /// Compresses the second pair of children.
    h2: C,

    /// Compresses the two masked halves.
    h3: C,

    /// Binds the group law, which is a pure type-level choice.
    _law: PhantomData<L>,
}

impl<C, L> T5<C, L> {
    /// Build a T5 node from three compressions.
    ///
    /// They must be independent functions, for instance one permutation under three sets of round constants.
    ///
    /// Passing the same function three times voids every bound of the paper.
    pub const fn new(h1: C, h2: C, h3: C) -> Self {
        Self {
            h1,
            h2,
            h3,
            _law: PhantomData,
        }
    }
}

impl<C, L> T5<C, L> {
    /// Add a mask to every word of a digest.
    #[inline(always)]
    fn mask<W, const D: usize>(digest: [W; D], mask: [W; D]) -> [W; D]
    where
        W: Copy,
        L: GroupLaw<W>,
    {
        core::array::from_fn(|k| L::add(digest[k], mask[k]))
    }

    /// The aggressive opening of one child: three digests from which [`Self::bypass`] rebuilds the node.
    ///
    /// Positions `0..5` name the children `m1..m5`, and the opening holds, in order:
    ///
    /// - position 0: `m2`, `m5` and `d = h2(m3, m4) + m5`.
    /// - position 1: `m1`, `m5` and `d`.
    /// - position 2: `m4`, `m5` and `c = h1(m1, m2) + m5`.
    /// - position 3: `m3`, `m5` and `c`.
    /// - position 4: `m1`, `m2` and `d`.
    ///
    /// The masked half `c` or `d` is not a child, so it cannot be checked on its own.
    ///
    /// That is what costs security, per T5 node and therefore per tree (Theorem 3, Propositions 1 and 2):
    ///
    /// - Against a forged opening of an honestly built tree: `2^(n/3)` queries proven, `2^(n/2)` conjectured from 3-XOR.
    /// - Against two conflicting openings of an untrusted tree: `2^(n/4)` proven, `2^(n/3)` conjectured from 4-XOR.
    ///
    /// A commitment in a proof system is built by the prover, so the second line applies there.
    ///
    /// # Panics
    ///
    /// Panics if `position` is not below 5.
    pub fn open_aggressive<W, const D: usize>(
        &self,
        children: &[[W; D]; T5_ARITY],
        position: usize,
    ) -> [[W; D]; T5_AGGRESSIVE_OPENING_LEN]
    where
        W: Copy,
        C: PseudoCompressionFunction<[W; D], 2>,
        L: GroupLaw<W>,
    {
        let [m1, m2, m3, m4, m5] = *children;
        match position {
            0 => [m2, m5, Self::mask(self.h2.compress([m3, m4]), m5)],
            1 => [m1, m5, Self::mask(self.h2.compress([m3, m4]), m5)],
            2 => [m4, m5, Self::mask(self.h1.compress([m1, m2]), m5)],
            3 => [m3, m5, Self::mask(self.h1.compress([m1, m2]), m5)],
            4 => [m1, m2, Self::mask(self.h2.compress([m3, m4]), m5)],
            _ => panic!("position {position} is not a child of a T5 node"),
        }
    }

    /// Rebuild a node from one child and its aggressive opening, with two compression calls.
    ///
    /// It equals [`PseudoCompressionFunction::compress`] of the full node whenever the opening is honest.
    ///
    /// # Panics
    ///
    /// Panics if `position` is not below 5.
    pub fn bypass<W, const D: usize>(
        &self,
        child: [W; D],
        position: usize,
        opening: [[W; D]; T5_AGGRESSIVE_OPENING_LEN],
    ) -> [W; D]
    where
        W: Copy,
        C: PseudoCompressionFunction<[W; D], 2>,
        L: GroupLaw<W>,
    {
        let [x, y, z] = opening;

        // The recomputed half and the fifth child, for every position.
        //
        // - Positions 0 to 3 recompute one half from the child and its pair sibling.
        // - Position 4 is the fifth child itself and recomputes the first half from `m1, m2`.
        let (c, d, m5) = match position {
            0 => (Self::mask(self.h1.compress([child, x]), y), z, y),
            1 => (Self::mask(self.h1.compress([x, child]), y), z, y),
            2 => (z, Self::mask(self.h2.compress([child, x]), y), y),
            3 => (z, Self::mask(self.h2.compress([x, child]), y), y),
            4 => (Self::mask(self.h1.compress([x, y]), child), z, child),
            _ => panic!("position {position} is not a child of a T5 node"),
        };
        Self::mask(self.h3.compress([c, d]), m5)
    }
}

impl<W, C, L, const D: usize> PseudoCompressionFunction<[W; D], T5_ARITY> for T5<C, L>
where
    W: Copy + Default,
    C: PseudoCompressionFunction<[W; D], 2>,
    L: GroupLaw<W> + Clone,
{
    const LANES: usize = C::LANES;

    #[inline]
    fn compress(&self, input: [[W; D]; T5_ARITY]) -> [W; D] {
        let [m1, m2, m3, m4, m5] = input;

        // The two halves are independent, so they can overlap in the pipeline.
        let a = self.h1.compress([m1, m2]);
        let b = self.h2.compress([m3, m4]);
        let e = self.h3.compress([Self::mask(a, m5), Self::mask(b, m5)]);
        Self::mask(e, m5)
    }

    fn compress_many(&self, inputs: &[[[W; D]; T5_ARITY]], out: &mut [[W; D]]) {
        // A ragged batch is a caller bug, so it fails loudly rather than dropping groups.
        assert_eq!(
            inputs.len(),
            out.len(),
            "group count ({}) must equal the output count ({})",
            inputs.len(),
            out.len()
        );

        // A one-lane compression gains nothing from staging.
        if C::LANES == 1 {
            for (output, &group) in out.iter_mut().zip(inputs) {
                *output = self.compress(group);
            }
            return;
        }

        // Each chunk makes three batched passes, one per inner compression:
        //
        // - pass 1: `a = h1(m1, m2)` and `b = h2(m3, m4)` over the whole chunk.
        // - pass 2: `e = h3(a + m5, b + m5)` over the whole chunk.
        // - fold:  `out = e + m5`.
        let zero = [W::default(); D];
        let staged = BATCH_CHUNK.min(inputs.len());
        let mut pairs = Vec::with_capacity(staged);
        let mut halves = Vec::with_capacity(staged);
        let mut a = alloc::vec![zero; staged];
        let mut b = alloc::vec![zero; staged];

        for (groups, out) in inputs.chunks(BATCH_CHUNK).zip(out.chunks_mut(BATCH_CHUNK)) {
            let n = groups.len();

            pairs.clear();
            halves.clear();
            for &[m1, m2, m3, m4, _] in groups {
                pairs.push([m1, m2]);
                halves.push([m3, m4]);
            }
            self.h1.compress_many(&pairs, &mut a[..n]);
            self.h2.compress_many(&halves, &mut b[..n]);

            // Reuse the first staging buffer for the masked halves.
            pairs.clear();
            pairs.extend(
                groups
                    .iter()
                    .zip(&a)
                    .zip(&b)
                    .map(|((g, &left), &right)| [Self::mask(left, g[4]), Self::mask(right, g[4])]),
            );
            self.h3.compress_many(&pairs, out);

            for (digest, group) in out.iter_mut().zip(groups) {
                *digest = Self::mask(*digest, group[4]);
            }
        }
    }

    /// A group of at most two children is compressed by `h1` alone.
    ///
    /// A larger prefix is a padded T5 node.
    #[inline]
    fn compress_prefix(&self, inputs: [[W; D]; T5_ARITY], len: usize) -> [W; D] {
        if len <= 2 {
            self.h1.compress([inputs[0], inputs[1]])
        } else {
            self.compress(inputs)
        }
    }

    fn compress_prefix_many(&self, inputs: &[[[W; D]; T5_ARITY]], len: usize, out: &mut [[W; D]]) {
        if len > 2 {
            return self.compress_many(inputs, out);
        }

        assert_eq!(
            inputs.len(),
            out.len(),
            "group count ({}) must equal the output count ({})",
            inputs.len(),
            out.len()
        );
        let pairs: Vec<[[W; D]; 2]> = inputs.iter().map(|g| [g[0], g[1]]).collect();
        self.h1.compress_many(&pairs, out);
    }
}

/// T5 is collision resistant in general, not only in a tree, when its compressions are.
impl<W, C, L, const D: usize> CompressionFunction<[W; D], T5_ARITY> for T5<C, L>
where
    W: Copy + Default,
    C: CompressionFunction<[W; D], 2>,
    L: GroupLaw<W> + Clone,
{
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;
    use core::array;

    use p3_field::PrimeCharacteristicRing;
    use p3_koala_bear::KoalaBear;
    use proptest::prelude::*;

    use super::*;

    /// A keyed 2-to-1 mixing function, distinct per key, that reports several lanes.
    ///
    /// It is no hash, but it is injective enough that a wrong wiring changes the output.
    #[derive(Clone, Debug)]
    struct Mix {
        key: u64,
    }

    impl Mix {
        const fn new(key: u64) -> Self {
            Self { key }
        }
    }

    impl<const D: usize> PseudoCompressionFunction<[u64; D], 2> for Mix {
        const LANES: usize = 4;

        fn compress(&self, [x, y]: [[u64; D]; 2]) -> [u64; D] {
            array::from_fn(|k| {
                let h = x[k].wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ y[k].rotate_left(17);
                (h ^ self.key).wrapping_mul(0xff51_afd7_ed55_8ccd) ^ (k as u64)
            })
        }
    }

    impl PseudoCompressionFunction<[KoalaBear; 2], 2> for Mix {
        fn compress(&self, [x, y]: [[KoalaBear; 2]; 2]) -> [KoalaBear; 2] {
            let key = KoalaBear::from_u64(self.key);
            array::from_fn(|k| (x[k] + key).cube() + y[k] * x[k] + y[k].square())
        }
    }

    type Node = T5<Mix, Xor>;

    fn node() -> Node {
        T5::new(Mix::new(1), Mix::new(2), Mix::new(3))
    }

    /// The paper's definition, written out with no shared helper.
    fn reference(children: [[u64; 2]; 5]) -> [u64; 2] {
        let [m1, m2, m3, m4, m5] = children;
        let xor = |x: [u64; 2], y: [u64; 2]| [x[0] ^ y[0], x[1] ^ y[1]];
        let a = Mix::new(1).compress([m1, m2]);
        let b = Mix::new(2).compress([m3, m4]);
        let e = Mix::new(3).compress([xor(a, m5), xor(b, m5)]);
        xor(e, m5)
    }

    fn children() -> impl Strategy<Value = [[u64; 2]; 5]> {
        prop::array::uniform5(prop::array::uniform2(any::<u64>()))
    }

    proptest! {
        #[test]
        fn compress_matches_the_paper(children in children()) {
            prop_assert_eq!(node().compress(children), reference(children));
        }

        #[test]
        fn batched_compress_matches_one_at_a_time(
            groups in prop::collection::vec(children(), 0..600),
        ) {
            // Lengths up to 600 cross two chunk boundaries and leave a partial last chunk.
            let node = node();
            let mut batched = vec![[0u64; 2]; groups.len()];
            node.compress_many(&groups, &mut batched);

            let expected: Vec<_> = groups.iter().map(|&g| node.compress(g)).collect();
            prop_assert_eq!(batched, expected);
        }

        #[test]
        fn every_aggressive_opening_rebuilds_the_node(
            children in children(),
            position in 0usize..5,
        ) {
            let node = node();
            let opening = node.open_aggressive(&children, position);
            prop_assert_eq!(
                node.bypass(children[position], position, opening),
                node.compress(children)
            );
        }

        #[test]
        fn a_changed_child_breaks_its_aggressive_opening(
            children in children(),
            position in 0usize..5,
            flip in 1u64..,
        ) {
            // Any other value at the opened position must land on a different node.
            let node = node();
            let opening = node.open_aggressive(&children, position);
            let mut forged = children[position];
            forged[0] ^= flip;
            prop_assert_ne!(node.bypass(forged, position, opening), node.compress(children));
        }

        #[test]
        fn field_digests_mask_with_field_addition(
            words in prop::array::uniform10(any::<u32>()),
        ) {
            // Field digests take `+` as the group law, with the same wiring as XOR.
            let children: [[KoalaBear; 2]; 5] =
                array::from_fn(|i| array::from_fn(|k| KoalaBear::from_u32(words[2 * i + k])));
            let node = T5::<Mix, FieldAdd>::new(Mix::new(1), Mix::new(2), Mix::new(3));

            let [m1, m2, m3, m4, m5] = children;
            let add = |x: [KoalaBear; 2], y: [KoalaBear; 2]| [x[0] + y[0], x[1] + y[1]];
            let a = Mix::new(1).compress([m1, m2]);
            let b = Mix::new(2).compress([m3, m4]);
            let e = Mix::new(3).compress([add(a, m5), add(b, m5)]);
            prop_assert_eq!(node.compress(children), add(e, m5));
        }
    }

    #[test]
    fn a_short_prefix_is_compressed_by_the_first_function_alone() {
        // A binary step reads only the pair, whatever sits in the padding slots.
        let node = node();
        let pair = [[1, 2], [3, 4]];
        let padded = [pair[0], pair[1], [9, 9], [9, 9], [9, 9]];
        assert_eq!(node.compress_prefix(padded, 2), Mix::new(1).compress(pair));

        let mut out = [[0; 2]; 2];
        node.compress_prefix_many(&[padded, padded], 2, &mut out);
        assert_eq!(out, [Mix::new(1).compress(pair); 2]);
    }

    #[test]
    fn a_long_prefix_is_a_padded_node() {
        let node = node();
        let group = [[1, 2], [3, 4], [5, 6], [0, 0], [0, 0]];
        assert_eq!(node.compress_prefix(group, 3), node.compress(group));
    }

    #[test]
    fn the_final_mask_is_applied() {
        // Only `m5` differs, and it enters both halves and the output.
        //
        // An implementation missing the final mask would still differ, so check the exact value.
        let node = node();
        let base = [[1, 2], [3, 4], [5, 6], [7, 8], [0, 0]];
        let mut shifted = base;
        shifted[4] = [0xff, 0];
        assert_eq!(node.compress(shifted), reference(shifted));
        assert_ne!(node.compress(shifted), node.compress(base));
    }

    #[test]
    #[should_panic(expected = "is not a child of a T5 node")]
    fn opening_a_sixth_child_panics() {
        node().open_aggressive(&[[0u64; 2]; 5], 5);
    }

    #[test]
    #[should_panic(expected = "must equal the output count")]
    fn batched_compress_rejects_a_ragged_batch() {
        node().compress_many(&[[[0u64; 2]; 5]; 3], &mut [[0; 2]; 2]);
    }

    #[test]
    fn xor_masks_every_lane_of_a_packed_word() {
        assert_eq!(Xor::add([1u64, 2, 3], [3, 2, 1]), [2, 0, 2]);
    }
}
