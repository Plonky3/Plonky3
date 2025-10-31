use alloc::vec;
use alloc::vec::Vec;
use core::array;
use core::cmp::Reverse;
use core::marker::PhantomData;

use itertools::Itertools;
use p3_field::PackedValue;
use p3_matrix::Matrix;
use p3_maybe_rayon::prelude::*;
use p3_symmetric::{CryptographicHasher, Hash, PseudoCompressionFunction};
use serde::{Deserialize, Serialize};
use tracing::instrument;

/// A binary Merkle tree whose leaves are vectors of matrix rows.
///
/// * `F` – scalar element type inside each matrix row.
/// * `W` – scalar element type of every digest word.
/// * `M` – matrix type. Must implement [`Matrix<F>`].
/// * `DIGEST_ELEMS` – number of `W` words in one digest.
///
/// The tree is **balanced only at the digest layer**.
/// Every leaf matrix must have a power-of-two height, though matrices may have
/// different heights from each other.
///
/// Use [`root`] to fetch the final digest once the tree is built.
///
/// This generally shouldn't be used directly. If you're using a Merkle tree as an MMCS,
/// see `MerkleTreeMmcs`.
#[derive(Debug, Serialize, Deserialize)]
pub struct MerkleTree<F, W, M, const DIGEST_ELEMS: usize> {
    /// All leaf matrices in insertion order.
    ///
    /// Each matrix contributes rows to one or more digest layers, depending on its height.
    /// Specifically, the tallest matrices populate the first digest layer, while shorter
    /// matrices are injected into higher digest layers once the running height matches theirs.
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

    /// Zero-sized marker that binds the generic `F` but occupies no space.
    _phantom: PhantomData<F>,
}

impl<F: Clone + Send + Sync, W: Clone, M: Matrix<F>, const DIGEST_ELEMS: usize>
    MerkleTree<F, W, M, DIGEST_ELEMS>
{
    /// Build a tree from **one or more matrices**.
    ///
    /// * `h` – hashing function used on raw rows.
    /// * `c` – 2-to-1 compression function used on digests.
    /// * `leaves` – matrices to commit to. Must be non-empty.
    ///
    /// Every matrix must have a non-zero, power-of-two height. Matrices of different heights
    /// are injected at the layer whose height matches theirs.
    ///
    /// All matrices are hashed row-by-row with `h`. The resulting digests are
    /// then folded upwards with `c` until a single root remains.
    ///
    /// # Panics
    /// * If `leaves` is empty.
    /// * If the packing widths of `P` and `PW` differ.
    /// * If any leaf has a height that is not a non-zero power of two.
    #[instrument(name = "build merkle tree", level = "debug", skip_all,
                 fields(dimensions = alloc::format!("{:?}", leaves.iter().map(|l| l.dimensions()).collect::<Vec<_>>())))]
    pub fn new<P, PW, H, C>(h: &H, c: &C, leaves: Vec<M>) -> Self
    where
        P: PackedValue<Value = F>,
        PW: PackedValue<Value = W>,
        H: CryptographicHasher<F, [W; DIGEST_ELEMS]>
            + CryptographicHasher<P, [PW; DIGEST_ELEMS]>
            + Sync,
        C: PseudoCompressionFunction<[W; DIGEST_ELEMS], 2>
            + PseudoCompressionFunction<[PW; DIGEST_ELEMS], 2>
            + Sync,
    {
        assert!(!leaves.is_empty(), "No matrices given?");
        const {
            assert!(P::WIDTH == PW::WIDTH, "Packing widths must match");
        }
        assert!(
            leaves
                .iter()
                .all(|matrix| matrix.height().is_power_of_two() && matrix.height() > 0),
            "matrix heights must be non-zero powers of two"
        );

        let mut leaves_largest_first = leaves
            .iter()
            .sorted_by_key(|l| Reverse(l.height()))
            .peekable();

        debug_assert!(
            leaves_largest_first
                .clone()
                .map(|m| m.height())
                .tuple_windows()
                .all(|(curr, next)| curr >= next && curr % next == 0),
            "matrix heights should descend by factors of two"
        );

        let max_height = leaves_largest_first.peek().unwrap().height();
        let tallest_matrices = leaves_largest_first
            .peeking_take_while(|m| m.height() == max_height)
            .collect_vec();

        let mut digest_layers = vec![first_digest_layer::<P, _, _, _, DIGEST_ELEMS>(
            h,
            &tallest_matrices,
        )];
        loop {
            let prev_layer = digest_layers.last().unwrap().as_slice();
            if prev_layer.len() == 1 {
                break;
            }
            let next_layer_len = prev_layer.len() / 2;

            // The matrices that get injected at this layer.
            let matrices_to_inject = leaves_largest_first
                .peeking_take_while(|m| m.height() == next_layer_len)
                .collect_vec();

            let next_digests = compress_and_inject::<P, _, _, _, _, DIGEST_ELEMS>(
                prev_layer,
                &matrices_to_inject,
                h,
                c,
            );
            digest_layers.push(next_digests);
        }

        Self {
            leaves,
            digest_layers,
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
}

/// Hash every row of the tallest matrices and build the first digest layer.
///
/// This function is responsible for creating the first layer of Merkle digests,
/// starting from raw rows of the tallest matrices. Each row is hashed using the
/// provided cryptographic hasher `h`. The result is a vector of digests that serve
/// as the base (leaf-level) nodes for the rest of the Merkle tree.
///
/// # Details
/// - When every matrix height is a power of two, each digest layer contains exactly the
///   expected number of elements—no padding digests are introduced.
/// - Matrices are "vertically packed" to allow SIMD-friendly parallel hashing,
///   meaning rows can be processed in batches.
/// - If the total number of rows isn't a multiple of the SIMD packing width,
///   the final few rows are handled using a fallback scalar path.
///
/// # Arguments
/// - `h`: Reference to the cryptographic hasher.
/// - `tallest_matrices`: References to the tallest matrices (all must have same height).
///
/// # Returns
/// A vector of `[PW::Value; DIGEST_ELEMS]`, containing the digests of each row.
#[instrument(name = "first digest layer", level = "debug", skip_all)]
fn first_digest_layer<P, PW, H, M, const DIGEST_ELEMS: usize>(
    h: &H,
    tallest_matrices: &[&M],
) -> Vec<[PW::Value; DIGEST_ELEMS]>
where
    P: PackedValue,
    PW: PackedValue,
    H: CryptographicHasher<P::Value, [PW::Value; DIGEST_ELEMS]>
        + CryptographicHasher<P, [PW; DIGEST_ELEMS]>
        + Sync,
    M: Matrix<P::Value>,
{
    // The number of rows to pack and hash together in one SIMD batch.
    let width = PW::WIDTH;

    // Get the height of the tallest matrices (they are guaranteed to be equal).
    let max_height = tallest_matrices[0].height();

    debug_assert!(max_height.is_power_of_two());

    // Allocate the digest vector; every entry is overwritten in the loops below.
    let mut digests = vec![[PW::Value::default(); DIGEST_ELEMS]; max_height];

    // Parallel loop: process complete batches of `width` rows at a time.
    digests[0..max_height]
        .par_chunks_exact_mut(width)
        .enumerate()
        .for_each(|(i, digests_chunk)| {
            // Compute the starting row index for this chunk.
            let first_row = i * width;

            // Collect all vertically packed rows from each matrix at `first_row`.
            // These packed rows are then hashed together using `h`.
            let packed_digest: [PW; DIGEST_ELEMS] = h.hash_iter(
                tallest_matrices
                    .iter()
                    .flat_map(|m| m.vertically_packed_row(first_row)),
            );

            // Unpack the resulting packed digest into individual scalar digests.
            // Then, assign each to its slot in the current chunk.
            for (dst, src) in digests_chunk.iter_mut().zip(unpack_array(packed_digest)) {
                *dst = src;
            }
        });

    // Handle leftover rows that do not form a full SIMD batch (if any).
    #[allow(clippy::needless_range_loop)]
    for i in ((max_height / width) * width)..max_height {
        unsafe {
            // Safety: The loop guarantees i < max_height == matrix height.
            // Use `row_unchecked` to avoid bounds checks for performance.
            digests[i] = h.hash_iter(tallest_matrices.iter().flat_map(|m| m.row_unchecked(i)));
        }
    }

    // Return the final digest vector (now fully populated).
    digests
}

/// Fold one digest layer into the next and, when present, mix in rows
/// taken from smaller matrices whose height equals `prev_layer.len()/2`.
fn compress_and_inject<P, PW, H, C, M, const DIGEST_ELEMS: usize>(
    prev_layer: &[[PW::Value; DIGEST_ELEMS]],
    matrices_to_inject: &[&M],
    h: &H,
    c: &C,
) -> Vec<[PW::Value; DIGEST_ELEMS]>
where
    P: PackedValue,
    PW: PackedValue,
    H: CryptographicHasher<P::Value, [PW::Value; DIGEST_ELEMS]>
        + CryptographicHasher<P, [PW; DIGEST_ELEMS]>
        + Sync,
    C: PseudoCompressionFunction<[PW::Value; DIGEST_ELEMS], 2>
        + PseudoCompressionFunction<[PW; DIGEST_ELEMS], 2>
        + Sync,
    M: Matrix<P::Value>,
{
    if matrices_to_inject.is_empty() {
        return compress::<PW, _, DIGEST_ELEMS>(prev_layer, c);
    }

    debug_assert!(prev_layer.len().is_power_of_two());

    let width = PW::WIDTH;
    let next_len = prev_layer.len() / 2;
    debug_assert!(
        matrices_to_inject
            .iter()
            .all(|matrix| matrix.height() == next_len),
        "matrices injected at a layer must match that layer's length"
    );

    let mut next_digests = vec![[PW::Value::default(); DIGEST_ELEMS]; next_len];
    next_digests
        .par_chunks_exact_mut(width)
        .enumerate()
        .for_each(|(i, digests_chunk)| {
            let first_row = i * width;
            let left = array::from_fn(|j| PW::from_fn(|k| prev_layer[2 * (first_row + k)][j]));
            let right = array::from_fn(|j| PW::from_fn(|k| prev_layer[2 * (first_row + k) + 1][j]));
            let mut packed_digest = c.compress([left, right]);
            let tallest_digest = h.hash_iter(
                matrices_to_inject
                    .iter()
                    .flat_map(|m| m.vertically_packed_row(first_row)),
            );
            packed_digest = c.compress([packed_digest, tallest_digest]);
            for (dst, src) in digests_chunk.iter_mut().zip(unpack_array(packed_digest)) {
                *dst = src;
            }
        });

    // If our packing width did not divide next_len, fall back to single-threaded scalar code
    // for the last bit.
    for i in (next_len / width * width)..next_len {
        let left = prev_layer[2 * i];
        let right = prev_layer[2 * i + 1];
        let digest = c.compress([left, right]);
        let rows_digest = unsafe {
            // Safety: Clearly i < next_len = m.height().
            h.hash_iter(matrices_to_inject.iter().flat_map(|m| m.row_unchecked(i)))
        };
        next_digests[i] = c.compress([digest, rows_digest]);
    }

    next_digests
}

/// Pure compression step used when no extra rows are injected.
///
/// Takes pairs of digests from `prev_layer`, feeds them to `c`,
/// and writes the results in order.
fn compress<P, C, const DIGEST_ELEMS: usize>(
    prev_layer: &[[P::Value; DIGEST_ELEMS]],
    c: &C,
) -> Vec<[P::Value; DIGEST_ELEMS]>
where
    P: PackedValue,
    C: PseudoCompressionFunction<[P::Value; DIGEST_ELEMS], 2>
        + PseudoCompressionFunction<[P; DIGEST_ELEMS], 2>
        + Sync,
{
    debug_assert!(prev_layer.len().is_power_of_two());

    let width = P::WIDTH;
    let next_len = prev_layer.len() / 2;
    let mut next_digests = vec![[P::Value::default(); DIGEST_ELEMS]; next_len];

    next_digests[0..next_len]
        .par_chunks_exact_mut(width)
        .enumerate()
        .for_each(|(i, digests_chunk)| {
            let first_row = i * width;
            let left = array::from_fn(|j| P::from_fn(|k| prev_layer[2 * (first_row + k)][j]));
            let right = array::from_fn(|j| P::from_fn(|k| prev_layer[2 * (first_row + k) + 1][j]));
            let packed_digest = c.compress([left, right]);
            for (dst, src) in digests_chunk.iter_mut().zip(unpack_array(packed_digest)) {
                *dst = src;
            }
        });

    // If our packing width did not divide next_len, fall back to single-threaded scalar code
    // for the last bit.
    for i in (next_len / width * width)..next_len {
        let left = prev_layer[2 * i];
        let right = prev_layer[2 * i + 1];
        next_digests[i] = c.compress([left, right]);
    }

    // Everything has been initialized so we can safely cast.
    next_digests
}

/// Converts a packed array `[P; N]` into its underlying `P::WIDTH` scalar arrays.
///
/// Interprets `[P; N]` as the matrix `[[P::Value; P::WIDTH]; N]`, performs a transpose to
/// get `[[P::Value; N] P::WIDTH]` and returns these `P::Value` arrays as an iterator.
#[inline]
fn unpack_array<P: PackedValue, const N: usize>(
    packed_digest: [P; N],
) -> impl Iterator<Item = [P::Value; N]> {
    (0..P::WIDTH).map(move |j| packed_digest.map(|p| p.as_slice()[j]))
}

#[cfg(test)]
mod tests {
    use p3_symmetric::PseudoCompressionFunction;
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    use super::*;

    #[derive(Clone, Copy)]
    struct DummyCompressionFunction;

    impl PseudoCompressionFunction<[u8; 32], 2> for DummyCompressionFunction {
        fn compress(&self, input: [[u8; 32]; 2]) -> [u8; 32] {
            let mut output = [0u8; 32];
            for (i, o) in output.iter_mut().enumerate() {
                // Simple XOR-based compression
                *o = input[0][i] ^ input[1][i];
            }
            output
        }
    }

    #[test]
    fn test_compress_even_length() {
        let prev_layer = [[0x01; 32], [0x02; 32], [0x03; 32], [0x04; 32]];
        let compressor = DummyCompressionFunction;
        let expected = vec![
            [0x03; 32], // 0x01 ^ 0x02
            [0x07; 32], // 0x03 ^ 0x04
        ];
        let result = compress::<u8, DummyCompressionFunction, 32>(&prev_layer, &compressor);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_compress_random_values() {
        let mut rng = SmallRng::seed_from_u64(1);
        let prev_layer: Vec<[u8; 32]> = (0..8).map(|_| rng.random()).collect();
        let compressor = DummyCompressionFunction;
        let expected: Vec<[u8; 32]> = prev_layer
            .chunks_exact(2)
            .map(|pair| {
                let mut result = [0u8; 32];
                for (i, r) in result.iter_mut().enumerate() {
                    *r = pair[0][i] ^ pair[1][i];
                }
                result
            })
            .collect();
        let result = compress::<u8, DummyCompressionFunction, 32>(&prev_layer, &compressor);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_compress_root_case_single_pair() {
        // When `prev_layer.len() == 2` we are at the “root-formation” case and
        // the function must return exactly one digest.
        //
        // 0xAA ^ 0x55 = 0xFF
        let prev_layer = [[0xAA; 32], [0x55; 32]];
        let compressor = DummyCompressionFunction;
        let expected = vec![[0xFF; 32]];
        let result = compress::<u8, DummyCompressionFunction, 32>(&prev_layer, &compressor);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_unpack_array_basic() {
        // Validate that `unpack_array` emits WIDTH (= 4) scalar arrays in the
        // right order when the packed words are `[u8; 4]`.

        // Two packed “words”, each four lanes wide
        let packed: [[u8; 4]; 2] = [
            [0, 1, 2, 3], // first word
            [4, 5, 6, 7], // second word
        ];

        // After unpacking we expect four rows (the width),
        // each row picking lane *j* from every packed word.
        let rows: Vec<[u8; 2]> = unpack_array::<[u8; 4], 2>(packed).collect();

        assert_eq!(
            rows,
            vec![
                [0, 4], // lane-0 of both packed words
                [1, 5], // lane-1
                [2, 6], // lane-2
                [3, 7], // lane-3
            ]
        );
    }
}
