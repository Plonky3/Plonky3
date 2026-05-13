use alloc::vec;
use alloc::vec::Vec;
use core::array;
use core::cmp::Reverse;
use core::marker::PhantomData;

use itertools::Itertools;
use p3_field::PackedValue;
use p3_matrix::Matrix;
use p3_maybe_rayon::prelude::*;
use p3_symmetric::{CryptographicHasher, Hash, MerkleCap, PseudoCompressionFunction};
use serde::{Deserialize, Serialize};
use tracing::instrument;

/// An N-ary Merkle tree whose leaves are vectors of matrix rows.
///
/// * `F` – scalar element type inside each matrix row.
/// * `W` – scalar element type of every digest word.
/// * `M` – matrix type. Must implement [`Matrix<F>`].
/// * `N` – arity of the compression function.
/// * `DIGEST_ELEMS` – number of `W` words in one digest.
///
/// The tree is **balanced only at the digest layer**.
/// Leaf matrices may have arbitrary heights as long as any two heights
/// that round **up** to the same power-of-two are equal.
///
/// Use [`root`] to fetch the final digest once the tree is built.
///
/// This generally shouldn't be used directly. If you're using a Merkle tree as an MMCS,
/// see `MerkleTreeMmcs`.
#[derive(Debug, Serialize, Deserialize)]
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
    /// Matrices do **not** need to have power-of-two heights. However, any two matrices
    /// whose heights **round up** to the same power-of-two must have **equal actual height**.
    /// This ensures proper balancing when folding digests layer-by-layer.
    ///
    /// All matrices are hashed row-by-row with `h`. The resulting digests are
    /// then folded upwards with `c` until a single root remains.
    ///
    /// # Panics
    /// * If `leaves` is empty.
    /// * If the packing widths of `P` and `PW` differ.
    /// * If two leaf heights *round up* to the same power-of-two but are not
    ///   equal (violates balancing rule).
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

        let mut leaves_largest_first = leaves
            .iter()
            .sorted_by_key(|l| Reverse(l.height()))
            .peekable();

        // check height property
        assert!(
            leaves_largest_first
                .clone()
                .map(|m| m.height())
                .tuple_windows()
                .all(|(curr, next)| curr == next
                    || curr.next_power_of_two() != next.next_power_of_two()),
            "matrix heights that round up to the same power of two must be equal"
        );

        let max_height = leaves_largest_first.peek().unwrap().height();
        let leaf_height_npt = max_height.next_power_of_two();
        let tallest_matrices = leaves_largest_first
            .peeking_take_while(|m| m.height() == max_height)
            .collect_vec();

        let mut digest_layers = vec![first_digest_layer::<P, _, _, _, N, DIGEST_ELEMS>(
            h,
            &tallest_matrices,
        )];
        let mut arity_schedule = Vec::new();

        loop {
            let prev_layer = digest_layers.last().unwrap().as_slice();
            if prev_layer.len() <= 1 {
                break;
            }

            // Decide whether this level is a full N-ary step or a binary step.
            let step = select_arity_step::<N>(
                prev_layer.len(),
                leaf_height_npt,
                leaves_largest_first.clone().map(|m| m.height()),
            );

            let next_layer_len = (prev_layer.len() / step).next_power_of_two();

            // The matrices that get injected at this layer.
            let matrices_to_inject = leaves_largest_first
                .peeking_take_while(|m| m.height().next_power_of_two() == next_layer_len)
                .collect_vec();

            let next_digests = compress_and_inject::<P, _, _, _, _, N, DIGEST_ELEMS>(
                prev_layer,
                step,
                &matrices_to_inject,
                h,
                c,
            );
            arity_schedule.push(step);
            digest_layers.push(next_digests);
        }

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

/// Hash every row of the tallest matrices and build the first digest layer.
///
/// This function is responsible for creating the first layer of Merkle digests,
/// starting from raw rows of the tallest matrices. Each row is hashed using the
/// provided cryptographic hasher `h`. The result is a vector of digests that serve
/// as the base (leaf-level) nodes for the rest of the Merkle tree.
///
/// # Details
/// - We always return an *even number of digests* (except when height is 1), to
///   ensure even pairing at higher layers.
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
fn first_digest_layer<P, PW, H, M, const N: usize, const DIGEST_ELEMS: usize>(
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

    let max_height_padded = padded_len(max_height, N);

    // Prepare a default digest value to fill unused slots or padding.
    let default_digest = [PW::Value::default(); DIGEST_ELEMS];

    // Allocate the digest vector with padded size, initialized to default digest.
    let mut digests = vec![default_digest; max_height_padded];

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
            PW::unpack_into(&packed_digest, digests_chunk);
        });

    // Handle leftover rows that do not form a full SIMD batch (if any).
    // `digests` is padded to `max_height_padded`, so cap the slice at `max_height`
    // to leave the padding tail untouched.
    let leftover_start = (max_height / width) * width;
    for (offset, digest) in digests[leftover_start..max_height].iter_mut().enumerate() {
        let i = leftover_start + offset;
        unsafe {
            // Safety: i < max_height == matrix height.
            // Use `row_unchecked` to avoid bounds checks for performance.
            *digest = h.hash_iter(tallest_matrices.iter().flat_map(|m| m.row_unchecked(i)));
        }
    }

    // Return the final digest vector (now fully populated).
    digests
}

/// Fold one digest layer into the next and, when present, mix in rows
/// taken from smaller matrices.
///
/// `step` is the grouping size for this level (either `N` for a full N-ary
/// step or `2` for a binary step when a matrix injection falls between
/// N-ary layers). Groups of `step` children are taken from `prev_layer`,
/// padded to `N` inputs with the default digest, then compressed with the
/// N-to-1 compression function.
fn compress_and_inject<P, PW, H, C, M, const N: usize, const DIGEST_ELEMS: usize>(
    prev_layer: &[[PW::Value; DIGEST_ELEMS]],
    step: usize,
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
    C: PseudoCompressionFunction<[PW::Value; DIGEST_ELEMS], N>
        + PseudoCompressionFunction<[PW; DIGEST_ELEMS], N>
        + Sync,
    M: Matrix<P::Value>,
{
    if matrices_to_inject.is_empty() {
        return compress::<PW, _, N, DIGEST_ELEMS>(prev_layer, step, c);
    }

    let width = PW::WIDTH;
    let next_len = matrices_to_inject[0].height();
    let raw_next = prev_layer.len() / step;
    let next_len_padded = padded_len(raw_next, N);

    let default_digest = [PW::Value::default(); DIGEST_ELEMS];
    let mut next_digests = vec![default_digest; next_len_padded];

    next_digests[0..next_len]
        .par_chunks_exact_mut(width)
        .enumerate()
        .for_each(|(i, digests_chunk)| {
            let first_row = i * width;
            let default_packed: [PW; DIGEST_ELEMS] =
                array::from_fn(|_| PW::broadcast(PW::Value::default()));

            let children: [[PW; DIGEST_ELEMS]; N] = array::from_fn(|n| {
                if n < step {
                    PW::pack_columns_fn(|lane| prev_layer[step * (first_row + lane) + n])
                } else {
                    default_packed
                }
            });
            let mut packed_digest = c.compress(children);

            let tallest_digest = h.hash_iter(
                matrices_to_inject
                    .iter()
                    .flat_map(|m| m.vertically_packed_row(first_row)),
            );
            let inject_inputs: [[PW; DIGEST_ELEMS]; N] = array::from_fn(|n| {
                if n == 0 {
                    packed_digest
                } else if n == 1 {
                    tallest_digest
                } else {
                    default_packed
                }
            });
            packed_digest = c.compress(inject_inputs);
            PW::unpack_into(&packed_digest, digests_chunk);
        });

    for i in (next_len / width * width)..next_len {
        let children: [_; N] = array::from_fn(|n| {
            if n < step {
                prev_layer[step * i + n]
            } else {
                default_digest
            }
        });
        let digest = c.compress(children);
        let rows_digest =
            unsafe { h.hash_iter(matrices_to_inject.iter().flat_map(|m| m.row_unchecked(i))) };
        let inject_inputs: [_; N] = array::from_fn(|n| {
            if n == 0 {
                digest
            } else if n == 1 {
                rows_digest
            } else {
                default_digest
            }
        });
        next_digests[i] = c.compress(inject_inputs);
    }

    for i in next_len..raw_next {
        let children: [_; N] = array::from_fn(|n| {
            if n < step {
                prev_layer[step * i + n]
            } else {
                default_digest
            }
        });
        let digest = c.compress(children);
        let inject_inputs: [_; N] =
            array::from_fn(|n| if n == 0 { digest } else { default_digest });
        next_digests[i] = c.compress(inject_inputs);
    }

    next_digests
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

/// Pure compression step used when no extra rows are injected.
///
/// Takes groups of digests from `prev_layer`, feeds them to `c`,
/// and writes the results in order.
///
/// Groups `step` consecutive digests from `prev_layer`, pads each group
/// to `N` inputs with the default digest, then compresses N-to-1.
fn compress<P, C, const N: usize, const DIGEST_ELEMS: usize>(
    prev_layer: &[[P::Value; DIGEST_ELEMS]],
    step: usize,
    c: &C,
) -> Vec<[P::Value; DIGEST_ELEMS]>
where
    P: PackedValue,
    C: PseudoCompressionFunction<[P::Value; DIGEST_ELEMS], N>
        + PseudoCompressionFunction<[P; DIGEST_ELEMS], N>
        + Sync,
{
    let width = P::WIDTH;
    let next_len = prev_layer.len() / step;
    let next_len_padded = padded_len(next_len, N);

    let default_digest = [P::Value::default(); DIGEST_ELEMS];
    let mut next_digests = vec![default_digest; next_len_padded];

    next_digests[0..next_len]
        .par_chunks_exact_mut(width)
        .enumerate()
        .for_each(|(i, digests_chunk)| {
            let first_row = i * width;
            let default_packed: [P; DIGEST_ELEMS] =
                array::from_fn(|_| P::broadcast(P::Value::default()));
            let children: [[P; DIGEST_ELEMS]; N] = array::from_fn(|n| {
                if n < step {
                    P::pack_columns_fn(|lane| prev_layer[step * (first_row + lane) + n])
                } else {
                    default_packed
                }
            });
            let packed_digest = c.compress(children);
            P::unpack_into(&packed_digest, digests_chunk);
        });

    for i in (next_len / width * width)..next_len {
        let children: [_; N] = array::from_fn(|n| {
            if n < step {
                prev_layer[step * i + n]
            } else {
                default_digest
            }
        });
        next_digests[i] = c.compress(children);
    }

    next_digests
}

#[cfg(test)]
mod tests {
    use p3_symmetric::PseudoCompressionFunction;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

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
        let result = compress::<u8, DummyCompressionFunction, 2, 32>(&prev_layer, 2, &compressor);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_compress_odd_length() {
        let prev_layer = [[0x05; 32], [0x06; 32], [0x07; 32]];
        let compressor = DummyCompressionFunction;
        let expected = vec![
            [0x03; 32], // 0x05 ^ 0x06
        ];
        let result = compress::<u8, DummyCompressionFunction, 2, 32>(&prev_layer, 2, &compressor);
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
        let result = compress::<u8, DummyCompressionFunction, 2, 32>(&prev_layer, 2, &compressor);
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
        let result = compress::<u8, DummyCompressionFunction, 2, 32>(&prev_layer, 2, &compressor);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_compress_non_power_of_two_with_padding() {
        // The code intentionally pads to the next even length unless the output
        // would become the root.  With `len() == 6` the output length must be 4
        // (three real digests plus one zero digest).

        let prev_layer = [
            [0x01; 32], [0x02; 32], [0x03; 32], [0x04; 32], [0x05; 32], [0x06; 32],
        ];
        let compressor = DummyCompressionFunction;

        let mut expected = vec![
            [0x03; 32], // 01 ^ 02
            [0x07; 32], // 03 ^ 04
            [0x03; 32], // 05 ^ 06
        ];
        // extra padded digest filled with 0
        expected.push([0x00; 32]);

        let result = compress::<u8, DummyCompressionFunction, 2, 32>(&prev_layer, 2, &compressor);
        assert_eq!(result, expected);
        assert_eq!(result.len(), 4);
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
}
