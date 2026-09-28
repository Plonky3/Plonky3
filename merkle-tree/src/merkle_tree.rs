use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_field::PackedValue;
use p3_matrix::Matrix;
use p3_symmetric::{CryptographicHasher, Hash, MerkleCap, PseudoCompressionFunction};
use serde::{Deserialize, Serialize};
use tracing::instrument;

use crate::builder::TreeBuilder;

/// An N-ary Merkle tree whose leaves are vectors of matrix rows.
///
/// * `F` – scalar element type inside each matrix row.
/// * `W` – scalar element type of every digest word.
/// * `M` – matrix type. Must implement [`Matrix<F>`].
/// * `N` – arity of the compression function.
/// * `DIGEST_ELEMS` – number of `W` words in one digest.
///
/// The tree is **balanced only at the digest layer**.
/// Leaf matrices may have arbitrary heights, but every height must sit on the
/// `ceil(max_height / 2^k)` ladder anchored at the tallest matrix — the same
/// requirement `Mmcs::commit` enforces before building a tree.
///
/// Use [`Self::root`] to fetch the final digest once the tree is built.
///
/// This generally shouldn't be used directly. If you're using a Merkle tree as an MMCS,
/// see `MerkleTreeMmcs`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MerkleTree<F, W, M, const N: usize, const DIGEST_ELEMS: usize> {
    /// All leaf matrices in insertion order.
    ///
    /// Each matrix contributes rows to one or more digest layers, depending on its height.
    /// Specifically, only the tallest matrices are included in the first digest layer,
    /// while shorter matrices are injected into higher digest layers at positions determined
    /// by their padded heights.
    ///
    /// This vector is retained only for inspection or re-opening of the tree; it is not used
    /// after construction time.
    pub(crate) leaves: Vec<M>,

    /// All intermediate digest layers, index 0 being the first layer above
    /// the leaves and the last layer containing exactly one root digest.
    ///
    /// Every inner vector holds contiguous digests `[left₀, right₀, left₁,
    /// right₁, …]`; higher layers refer to these by index.
    ///
    /// Serialization requires that `[W; DIGEST_ELEMS]` implements `Serialize` and
    /// `Deserialize`. This is automatically satisfied when `W` is a fixed-size type.
    #[serde(
        bound(serialize = "[W; DIGEST_ELEMS]: Serialize"),
        bound(deserialize = "[W; DIGEST_ELEMS]: Deserialize<'de>")
    )]
    pub(crate) digest_layers: Vec<Vec<[W; DIGEST_ELEMS]>>,

    /// The compression arity used at each tree level (transition from
    /// `digest_layers[i]` to `digest_layers[i+1]`).
    ///
    /// Each entry is either `N` (full N-ary step) as specified by the `N`
    /// parameter associated to the compression function, or `2` (binary step)
    /// when a matrix injection falls between N-ary levels.
    pub(crate) arity_schedule: Vec<usize>,

    /// Zero-sized marker that binds the generic `F` but occupies no space.
    _phantom: PhantomData<F>,
}

impl<F: Clone + Send + Sync, W: Clone, M: Matrix<F>, const N: usize, const DIGEST_ELEMS: usize>
    MerkleTree<F, W, M, N, DIGEST_ELEMS>
{
    /// Build a tree from **one or more matrices**.
    ///
    /// * `h` – hashing function used on raw rows.
    /// * `c` – N-to-1 compression function used on digests.
    /// * `leaves` – matrices to commit to. Must be non-empty.
    ///
    /// Matrices do **not** need to have power-of-two heights. However, every height must sit
    /// on the `ceil(max_height / 2^k)` ladder anchored at the tallest matrix — i.e. at `k`
    /// halvings above the leaves, the only admissible height is `ceil(max_height / 2^k)`. This
    /// ensures proper balancing when folding digests layer-by-layer, and that every global leaf
    /// index maps to a row in every committed matrix.
    ///
    /// All matrices are hashed row-by-row with `h`. The resulting digests are
    /// then folded upwards with `c` until a single root remains.
    ///
    /// # Panics
    /// * If `leaves` is empty, or every leaf has height 0.
    /// * If the packing widths of `P` and `PW` differ.
    /// * If any leaf height is off the `ceil(max_height / 2^k)` ladder.
    #[instrument(name = "build merkle tree", level = "debug", skip_all,
                 fields(dimensions = alloc::format!("{:?}", leaves.iter().map(|l| l.dimensions()).collect::<Vec<_>>())))]
    pub fn new<P, PW, H, C>(h: &H, c: &C, leaves: Vec<M>) -> Self
    where
        P: PackedValue<Value = F>,
        PW: PackedValue<Value = W>,
        H: CryptographicHasher<F, [W; DIGEST_ELEMS]>
            + CryptographicHasher<P, [PW; DIGEST_ELEMS]>
            + Sync,
        C: PseudoCompressionFunction<[W; DIGEST_ELEMS], N>
            + PseudoCompressionFunction<[PW; DIGEST_ELEMS], N>
            + Sync,
    {
        assert!(!leaves.is_empty(), "No matrices given?");
        const {
            assert!(N >= 2, "Arity N must be at least 2");
            assert!(N.is_power_of_two(), "Arity N must be a power of two");
            assert!(P::WIDTH == PW::WIDTH, "Packing widths must match");
        }

        // Geometry gate: every height must sit on the `ceil(max_height / 2^k)`
        // ladder anchored at the tallest matrix, or no tree can be built from
        // them. `Mmcs::commit` enforces the same gate; this constructor is
        // public, so it must enforce it too rather than fail later with an
        // out-of-bounds panic deep inside layer construction.
        if let Err(err) =
            crate::mmcs::validate_commit_reachable_heights(leaves.iter().map(|l| l.height()))
        {
            panic!("{err}");
        }

        // The plan borrows the matrices, so it lives only until the layers are built.
        let (digest_layers, arity_schedule) = {
            let builder = TreeBuilder::<P, PW, H, C, M, N, DIGEST_ELEMS>::new(h, c, &leaves);
            (builder.build(), builder.arity_schedule())
        };

        Self {
            leaves,
            digest_layers,
            arity_schedule,
            _phantom: PhantomData,
        }
    }

    /// Return the root digest of the tree.
    #[must_use]
    pub fn root(&self) -> Hash<F, W, DIGEST_ELEMS>
    where
        W: Copy,
    {
        self.digest_layers.last().unwrap()[0].into()
    }

    /// Return the Merkle cap at the specified height from the root.
    ///
    /// A cap height of 0 returns just the root (1 element).
    /// A cap height of h returns `product(arity_schedule[layer_idx..])` elements,
    /// where each arity is either N or 2 depending on the tree layout.
    ///
    /// # Panics
    /// Panics if `cap_height` exceeds the tree depth.
    #[must_use]
    pub fn cap(&self, cap_height: usize) -> MerkleCap<F, [W; DIGEST_ELEMS]>
    where
        W: Clone,
    {
        let num_layers = self.digest_layers.len();
        assert!(
            cap_height < num_layers,
            "cap_height {} exceeds tree depth {}",
            cap_height,
            num_layers
        );

        let layer_idx = num_layers - 1 - cap_height;
        let layer = &self.digest_layers[layer_idx];

        let cap_len: usize = self.arity_schedule[layer_idx..].iter().product();
        let cap_len = cap_len.min(layer.len());

        MerkleCap::new(layer[..cap_len].to_vec())
    }

    #[must_use]
    pub const fn num_layers(&self) -> usize {
        self.digest_layers.len()
    }
}

/// Select the compression arity for the current layer.
///
/// Returns `N` for a full N-ary step, or `2` for a binary bridge step when a
/// matrix injection must happen before the next N-ary target level.
pub(crate) fn select_arity_step<const N: usize>(
    curr_height_padded: usize,
    leaf_height_npt: usize,
    remaining_heights_tallest_first: impl Iterator<Item = usize>,
) -> usize {
    if curr_height_padded < N {
        return 2;
    }

    let n_ary_target = (curr_height_padded / N).next_power_of_two();
    let has_intermediate = remaining_heights_tallest_first
        .filter(|height| height.next_power_of_two() != leaf_height_npt)
        .any(|height| height.next_power_of_two() > n_ary_target);

    if has_intermediate { 2 } else { N }
}

/// Compute the padded output length for a compression step.
///
/// The output layer must be large enough for the *next* compression step
/// to form complete groups. There are three cases:
///
/// - `raw_len <= 1`: this is the root, no padding needed.
/// - `raw_len >= n`: pad up to the next multiple of `n`.
/// - `1 < raw_len < n`: pad to exactly `n` so that the next step can do a
///   single full N-to-1 compression to produce the root. This is safe
///   because the extra slots are filled with the default digest — the same
///   value that `compress` would use as padding internally.
pub(crate) const fn padded_len(raw_len: usize, n: usize) -> usize {
    if raw_len <= 1 {
        raw_len
    } else if raw_len >= n {
        raw_len.div_ceil(n) * n
    } else {
        n
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
    use p3_field::Field;
    use p3_keccak::Keccak256Hash;
    use p3_matrix::dense::RowMajorMatrix;
    use p3_symmetric::{
        CompressionFunctionFromHasher, PaddingFreeSponge, PseudoCompressionFunction,
        SerializingHasher, TruncatedPermutation,
    };
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    use super::*;

    type F = BabyBear;
    type Perm = Poseidon2BabyBear<16>;
    type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
    type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;

    /// Tree shapes the batched and unbatched drivers must agree on, as matrix heights and width.
    ///
    /// Between them they cover every boundary the batching introduces:
    ///
    /// - A tree of a single row, which has no level above the leaves at all.
    /// - Odd heights, which leave a padding tail at the top of a level.
    /// - Node counts that leave a partial final group in a hash call.
    ///
    /// - Several matrices sharing the tallest height, so a leaf message spans two rows.
    /// - A ragged height ladder, which forces injection levels.
    ///
    /// - Levels wide enough to fan out across threads instead of staying serial.
    /// - Rows long enough to make the sponge absorb more than one block.
    ///
    /// - A row past the serializing hasher's 8 KiB group budget, which it hashes one lane
    ///   group at a time. 2100 `BabyBear` columns are 8400 bytes, past that budget.
    const SHAPES: &[(&[usize], usize)] = &[
        (&[1], 1),
        (&[3], 4),
        (&[13], 1),
        (&[8, 8, 4], 3),
        (&[17, 9, 5, 3], 2),
        (&[1100], 1),
        (&[2049, 1025, 513], 5),
        (&[64], 135),
        (&[3], 2100),
    ];

    /// A hasher and compressor pair that reports one lane, forcing the unbatched driver.
    ///
    /// Every digest it produces is the wrapped primitive's, so the two drivers are compared on
    /// the same hash function and any difference is the driver's alone.
    #[derive(Clone, Copy, Debug)]
    struct Unbatched<T>(T);

    impl<Item, Out, T> CryptographicHasher<Item, Out> for Unbatched<T>
    where
        Item: Clone,
        T: CryptographicHasher<Item, Out>,
    {
        const LANES: usize = 1;

        fn hash_iter<I>(&self, input: I) -> Out
        where
            I: IntoIterator<Item = Item>,
        {
            self.0.hash_iter(input)
        }

        fn hash_iter_slices<'a, I>(&self, input: I) -> Out
        where
            I: IntoIterator<Item = &'a [Item]>,
            Item: 'a,
        {
            self.0.hash_iter_slices(input)
        }
    }

    impl<T, Inner, const N: usize> PseudoCompressionFunction<T, N> for Unbatched<Inner>
    where
        Inner: PseudoCompressionFunction<T, N>,
    {
        const LANES: usize = 1;

        fn compress(&self, input: [T; N]) -> T {
            self.0.compress(input)
        }
    }

    /// A byte hasher reporting three lanes, with no batched implementation of its own.
    ///
    /// Three is deliberately neither a power of two nor a divisor of any test height, so every
    /// group the driver forms ends in a short remainder.
    ///
    /// Leaving the batched hash at its default also isolates the driver: any disagreement comes
    /// from how the driver assembles messages, not from a vectorized sponge.
    #[derive(Clone, Copy, Debug)]
    struct ThreeLaneMix;

    impl CryptographicHasher<u8, [u8; 32]> for ThreeLaneMix {
        const LANES: usize = 3;

        fn hash_iter<I>(&self, input: I) -> [u8; 32]
        where
            I: IntoIterator<Item = u8>,
        {
            // A four-word state absorbing one byte per step, mixed with an odd multiplier and a
            // rotation so that byte order and message length both change the result.
            let mut state = [0x243f_6a88_85a3_08d3u64; 4];
            let mut count = 0u64;
            for byte in input {
                let word = &mut state[(count % 4) as usize];
                *word = word.rotate_left(11).wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ u64::from(byte);
                count += 1;
            }

            // Fold the length in so a truncated message cannot collide with a longer one.
            state[0] ^= count.wrapping_mul(0xc2b2_ae3d_27d4_eb4f);

            let mut digest = [0u8; 32];
            for (word, slot) in state.iter().zip(digest.as_chunks_mut::<8>().0) {
                *slot = word.to_le_bytes();
            }
            digest
        }
    }

    /// Build the same tree with both drivers and compare every node of every layer.
    ///
    /// Comparing whole layers rather than just the root pins where a divergence begins, and a
    /// root match alone could hide two compensating errors at a lower level.
    fn assert_drivers_agree<H, C, const N: usize>(h: &H, c: &C, heights: &[usize], width: usize)
    where
        H: CryptographicHasher<F, [u8; 32]> + Sync,
        C: PseudoCompressionFunction<[u8; 32], N> + Sync,
    {
        // Fixture: one random matrix per requested height, all at the same width.
        let mut rng = SmallRng::seed_from_u64(heights[0] as u64 * 1_000_003 + width as u64);
        let leaves: Vec<RowMajorMatrix<F>> = heights
            .iter()
            .map(|&height| RowMajorMatrix::rand(&mut rng, height, width))
            .collect();

        let batched =
            MerkleTree::<F, u8, RowMajorMatrix<F>, N, 32>::new::<F, u8, H, C>(h, c, leaves.clone());

        let unbatched_h = Unbatched(h.clone());
        let unbatched_c = Unbatched(c.clone());
        let unbatched = MerkleTree::<F, u8, RowMajorMatrix<F>, N, 32>::new::<
            F,
            u8,
            Unbatched<H>,
            Unbatched<C>,
        >(&unbatched_h, &unbatched_c, leaves);

        assert_eq!(
            batched.arity_schedule, unbatched.arity_schedule,
            "arity schedule differs for heights {heights:?} width {width}"
        );
        assert_eq!(
            batched.digest_layers.len(),
            unbatched.digest_layers.len(),
            "layer count differs for heights {heights:?} width {width}"
        );
        for (level, (left, right)) in batched
            .digest_layers
            .iter()
            .zip(&unbatched.digest_layers)
            .enumerate()
        {
            assert_eq!(
                left, right,
                "layer {level} differs for heights {heights:?} width {width}"
            );
        }
        assert_eq!(batched.root(), unbatched.root());
    }

    #[test]
    fn keccak_batched_tree_matches_unbatched_binary() {
        // Binary arity is the configuration every byte-digest scheme in the workspace uses.
        let h = SerializingHasher::new(Keccak256Hash);
        let c = CompressionFunctionFromHasher::<_, 2, 32>::new(Keccak256Hash);

        for &(heights, width) in SHAPES {
            assert_drivers_agree::<_, _, 2>(&h, &c, heights, width);
        }
    }

    #[test]
    fn keccak_batched_tree_matches_unbatched_quaternary() {
        // At arity four a level takes either a full four-to-one step, which the batched arm
        // handles, or a binary bridge step before an injection, which falls back to the
        // unbatched arm.
        //
        // Both must land on the same digests.
        let h = SerializingHasher::new(Keccak256Hash);
        let c = CompressionFunctionFromHasher::<_, 4, 32>::new(Keccak256Hash);

        for &(heights, width) in SHAPES {
            assert_drivers_agree::<_, _, 4>(&h, &c, heights, width);
        }
    }

    #[test]
    fn three_lane_batched_tree_matches_unbatched() {
        // A lane count of three exercises the driver's group arithmetic away from the powers of
        // two the vectorized sponges use.
        let h = SerializingHasher::new(ThreeLaneMix);
        let c = CompressionFunctionFromHasher::<_, 2, 32>::new(ThreeLaneMix);

        for &(heights, width) in SHAPES {
            assert_drivers_agree::<_, _, 2>(&h, &c, heights, width);
        }
    }

    #[test]
    fn keccak_tree_is_deterministic_across_builds() {
        // Padding slots take part in the next level's compression, so an unwritten slot would
        // show up as a root that changes between two builds of the same input.
        let h = SerializingHasher::new(Keccak256Hash);
        let c = CompressionFunctionFromHasher::<_, 2, 32>::new(Keccak256Hash);

        let mut rng = SmallRng::seed_from_u64(7);
        // Height 13 pads the leaf layer to 14 and every level above it to an even count.
        let leaves = vec![RowMajorMatrix::<F>::rand(&mut rng, 13, 3)];

        let first = MerkleTree::<F, u8, RowMajorMatrix<F>, 2, 32>::new::<F, u8, _, _>(
            &h,
            &c,
            leaves.clone(),
        );
        let second =
            MerkleTree::<F, u8, RowMajorMatrix<F>, 2, 32>::new::<F, u8, _, _>(&h, &c, leaves);

        assert_eq!(first.digest_layers, second.digest_layers);
    }

    #[test]
    fn test_padded_len_n2() {
        assert_eq!(padded_len(0, 2), 0);
        assert_eq!(padded_len(1, 2), 1);
        assert_eq!(padded_len(2, 2), 2);
        assert_eq!(padded_len(3, 2), 4);
        assert_eq!(padded_len(4, 2), 4);
        assert_eq!(padded_len(5, 2), 6);
        assert_eq!(padded_len(7, 2), 8);
        assert_eq!(padded_len(8, 2), 8);
        assert_eq!(padded_len(9, 2), 10);
        assert_eq!(padded_len(15, 2), 16);
        assert_eq!(padded_len(16, 2), 16);
    }

    #[test]
    fn test_padded_len_n4() {
        assert_eq!(padded_len(0, 4), 0);
        assert_eq!(padded_len(1, 4), 1);
        // Below-arity case: pad to exactly N
        assert_eq!(padded_len(2, 4), 4);
        assert_eq!(padded_len(3, 4), 4);
        // At or above arity: pad to next multiple of N
        assert_eq!(padded_len(4, 4), 4);
        assert_eq!(padded_len(5, 4), 8);
        assert_eq!(padded_len(7, 4), 8);
        assert_eq!(padded_len(8, 4), 8);
        assert_eq!(padded_len(9, 4), 12);
    }

    #[test]
    fn test_padded_len_n8() {
        assert_eq!(padded_len(0, 8), 0);
        assert_eq!(padded_len(1, 8), 1);
        // Below-arity: all pad to exactly N=8
        assert_eq!(padded_len(2, 8), 8);
        assert_eq!(padded_len(3, 8), 8);
        assert_eq!(padded_len(5, 8), 8);
        assert_eq!(padded_len(7, 8), 8);
        // At or above arity: next multiple of 8
        assert_eq!(padded_len(8, 8), 8);
        assert_eq!(padded_len(9, 8), 16);
        assert_eq!(padded_len(15, 8), 16);
        assert_eq!(padded_len(16, 8), 16);
    }

    #[test]
    fn test_padded_len_always_admits_full_groups() {
        // For any N in {2, 4, 8} and any raw_len > 1,
        // padded_len must be >= N and divisible by N (so a full compression
        // group is always possible), OR padded_len == raw_len <= 1 (root).
        for n in [2, 4, 8] {
            for raw_len in 2..=128 {
                let pl = padded_len(raw_len, n);
                assert!(
                    pl >= n && pl.is_multiple_of(n),
                    "padded_len({raw_len}, {n}) = {pl} is not a valid multiple of {n}",
                );
            }
        }
    }

    #[test]
    #[should_panic(expected = "matrix height 4 incompatible with tallest height 6")]
    fn new_rejects_heights_off_ladder() {
        // `MerkleTree::new` is a public constructor that bypasses `Mmcs::commit`'s
        // geometry gate. Heights 6 and 4 pass the weaker "equal within the same
        // power-of-two bucket" rule (next_power_of_two(6) = 8, next_power_of_two(4) = 4
        // — different buckets), but 4 is off the ceil(6 / 2^k) ladder: at k = 1 the
        // only admissible height is ceil(6 / 2) = 3. Building a tree from these would
        // panic later, out of bounds, inside a rayon closure — the constructor must
        // reject it up front instead.
        let mut rng = SmallRng::seed_from_u64(0);
        let perm = Perm::new_from_rng_128(&mut rng);
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm);

        let mat6 = RowMajorMatrix::<F>::rand(&mut rng, 6, 1);
        let mat4 = RowMajorMatrix::<F>::rand(&mut rng, 4, 1);

        let _ = MerkleTree::new::<<F as Field>::Packing, <F as Field>::Packing, MyHash, MyCompress>(
            &hash,
            &compress,
            vec![mat6, mat4],
        );
    }

    #[test]
    #[should_panic]
    fn new_rejects_single_zero_height_matrix() {
        // A single height-0 matrix used to build `digest_layers == [[]]`,
        // and `root()` would panic later. The constructor must reject it directly.
        let mut rng = SmallRng::seed_from_u64(0);
        let perm = Perm::new_from_rng_128(&mut rng);
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm);

        let empty_mat = RowMajorMatrix::<F>::new(Vec::new(), 1);

        let _ = MerkleTree::new::<<F as Field>::Packing, <F as Field>::Packing, MyHash, MyCompress>(
            &hash,
            &compress,
            vec![empty_mat],
        );
    }
}
