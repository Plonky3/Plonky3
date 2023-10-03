use alloc::vec;
use alloc::vec::Vec;
use core::array;
use core::cmp::Reverse;
use core::marker::PhantomData;

use itertools::Itertools;
use p3_commit::{DirectMmcs, Mmcs};
use p3_field::{AbstractField, Field, PackedField};
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixView};
use p3_matrix::{Dimensions, Matrix, MatrixRowSlices, MatrixRows};
use p3_maybe_rayon::{IndexedParallelIterator, MaybeParChunksMut, ParallelIterator};
use p3_symmetric::compression::PseudoCompressionFunction;
use p3_symmetric::hasher::CryptographicHasher;
use p3_util::log2_ceil_usize;
use tracing::instrument;

/// A binary Merkle tree for field data. It has leaves of type `F` and digests of type
/// `[F; DIGEST_ELEMS]`.
///
/// This generally shouldn't be used directly. If you're using a Merkle tree as an MMCS,
/// see `FieldMerkleTreeMmcs`.
pub struct FieldMerkleTree<F: Field, const DIGEST_ELEMS: usize> {
    leaves: Vec<RowMajorMatrix<F>>,
    digest_layers: Vec<Vec<[F; DIGEST_ELEMS]>>,
}

impl<F: Field, const DIGEST_ELEMS: usize> FieldMerkleTree<F, DIGEST_ELEMS> {
    /// Matrix heights need not be powers of two. However, if the heights of two given matrices
    /// round up to the same power of two, they must be equal.
    #[instrument(name = "build merkle tree", level = "debug", skip_all,
                 fields(dimensions = alloc::format!("{:?}", leaves.iter().map(|l| l.dimensions()).collect::<Vec<_>>())))]
    pub fn new<P, H, C>(h: &H, c: &C, leaves: Vec<RowMajorMatrix<F>>) -> Self
    where
        P: PackedField<Scalar = F>,
        H: CryptographicHasher<F, [F; DIGEST_ELEMS]>,
        H: CryptographicHasher<P, [P; DIGEST_ELEMS]>,
        H: Sync,
        C: PseudoCompressionFunction<[F; DIGEST_ELEMS], 2>,
        C: PseudoCompressionFunction<[P; DIGEST_ELEMS], 2>,
        C: Sync,
    {
        assert!(!leaves.is_empty(), "No matrices given?");

        // check height property
        assert!(
            leaves
                .iter()
                .map(|m| m.height())
                .sorted()
                .tuple_windows()
                .all(|(curr, next)| curr == next
                    || curr.next_power_of_two() != next.next_power_of_two()),
            "matrix heights that round up to the same power of two must be equal"
        );

        let mut leaves_largest_first = leaves
            .iter()
            .sorted_by_key(|l| Reverse(l.height()))
            .peekable();

        let max_height = leaves_largest_first.peek().unwrap().height();
        let tallest_matrices = leaves_largest_first
            .peeking_take_while(|m| m.height() == max_height)
            .collect_vec();

        let mut digest_layers = vec![first_digest_layer::<P, H, DIGEST_ELEMS>(
            h,
            tallest_matrices,
        )];
        loop {
            let prev_layer = digest_layers.last().map(Vec::as_slice).unwrap_or_default();
            if prev_layer.len() == 1 {
                break;
            }
            let next_layer_len = prev_layer.len() / 2;

            // The matrices that get injected at this layer.
            let matrices_to_inject = leaves_largest_first
                .peeking_take_while(|m| m.height().next_power_of_two() == next_layer_len)
                .collect_vec();

            let next_digests =
                compress_and_inject::<P, H, C, DIGEST_ELEMS>(prev_layer, matrices_to_inject, h, c);
            digest_layers.push(next_digests);
        }

        Self {
            leaves,
            digest_layers,
        }
    }

    #[must_use]
    pub fn root(&self) -> [F; DIGEST_ELEMS] {
        self.digest_layers.last().unwrap()[0]
    }
}

fn first_digest_layer<P, H, const DIGEST_ELEMS: usize>(
    h: &H,
    tallest_matrices: Vec<&RowMajorMatrix<P::Scalar>>,
) -> Vec<[P::Scalar; DIGEST_ELEMS]>
where
    P: PackedField,
    H: CryptographicHasher<P::Scalar, [P::Scalar; DIGEST_ELEMS]>,
    H: CryptographicHasher<P, [P; DIGEST_ELEMS]>,
    H: Sync,
{
    let width = P::WIDTH;
    let max_height = tallest_matrices[0].height();
    let max_height_padded = max_height.next_power_of_two();

    let default_digest = [P::Scalar::ZERO; DIGEST_ELEMS];
    let mut digests = vec![default_digest; max_height_padded];

    digests[0..max_height]
        .par_chunks_exact_mut(width)
        .enumerate()
        .for_each(|(i, digests_chunk)| {
            let first_row = i * width;
            let packed_digest: [P; DIGEST_ELEMS] = h.hash_iter(
                tallest_matrices
                    .iter()
                    .flat_map(|m| m.packed_row(first_row)),
            );
            for (dst, src) in digests_chunk.iter_mut().zip(unpack_array(packed_digest)) {
                *dst = src;
            }
        });

    // If our packing width did not divide max_height, fall back to single-threaded scalar code
    // for the last bit.
    #[allow(clippy::needless_range_loop)]
    for i in (max_height / width * width)..max_height {
        digests[i] = h.hash_iter_slices(tallest_matrices.iter().map(|m| m.row_slice(i)));
    }

    digests
}

/// Compress `n` digests from the previous layer into `n/2` digests, while potentially mixing in
/// some leaf data, if there are input matrices with (padded) height `n/2`.
fn compress_and_inject<P, H, C, const DIGEST_ELEMS: usize>(
    prev_layer: &[[P::Scalar; DIGEST_ELEMS]],
    matrices_to_inject: Vec<&RowMajorMatrix<P::Scalar>>,
    h: &H,
    c: &C,
) -> Vec<[P::Scalar; DIGEST_ELEMS]>
where
    P: PackedField,
    H: CryptographicHasher<P::Scalar, [P::Scalar; DIGEST_ELEMS]>,
    H: CryptographicHasher<P, [P; DIGEST_ELEMS]>,
    H: Sync,
    C: PseudoCompressionFunction<[P::Scalar; DIGEST_ELEMS], 2>,
    C: PseudoCompressionFunction<[P; DIGEST_ELEMS], 2>,
    C: Sync,
{
    if matrices_to_inject.is_empty() {
        return compress::<P, C, DIGEST_ELEMS>(prev_layer, c);
    }

    let width = P::WIDTH;
    let next_len = matrices_to_inject[0].height();
    let next_len_padded = prev_layer.len() / 2;

    let default_digest = [P::Scalar::ZERO; DIGEST_ELEMS];
    let mut next_digests = vec![default_digest; next_len_padded];

    next_digests[0..next_len]
        .par_chunks_exact_mut(width)
        .enumerate()
        .for_each(|(i, digests_chunk)| {
            let first_row = i * width;
            let left = array::from_fn(|j| P::from_fn(|k| prev_layer[2 * (first_row + k)][j]));
            let right = array::from_fn(|j| P::from_fn(|k| prev_layer[2 * (first_row + k) + 1][j]));
            let mut packed_digest = c.compress([left, right]);
            let tallest_digest = h.hash_iter(
                matrices_to_inject
                    .iter()
                    .flat_map(|m| m.packed_row(first_row)),
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
        let rows_digest = h.hash_iter_slices(matrices_to_inject.iter().map(|m| m.row_slice(i)));
        next_digests[i] = c.compress([digest, rows_digest]);
    }

    // At this point, we've exceeded the height of the matrices to inject, so we continue the
    // process above except with default_digest in place of an input digest.
    for i in next_len..next_len_padded {
        let left = prev_layer[2 * i];
        let right = prev_layer[2 * i + 1];
        let digest = c.compress([left, right]);
        next_digests[i] = c.compress([digest, default_digest]);
    }

    next_digests
}

/// Compress `n` digests from the previous layer into `n/2` digests.
fn compress<P, C, const DIGEST_ELEMS: usize>(
    prev_layer: &[[P::Scalar; DIGEST_ELEMS]],
    c: &C,
) -> Vec<[P::Scalar; DIGEST_ELEMS]>
where
    P: PackedField,
    C: PseudoCompressionFunction<[P::Scalar; DIGEST_ELEMS], 2>,
    C: PseudoCompressionFunction<[P; DIGEST_ELEMS], 2>,
    C: Sync,
{
    debug_assert!(prev_layer.len().is_power_of_two());
    let width = P::WIDTH;
    let next_len = prev_layer.len() / 2;

    let default_digest = [P::Scalar::ZERO; DIGEST_ELEMS];
    let mut next_digests = vec![default_digest; next_len];

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
        let digest = c.compress([left, right]);
        next_digests[i] = digest;
    }

    next_digests
}

/// Converts a packed array `[P; N]` into its underlying `P::WIDTH` scalar arrays.
#[inline]
fn unpack_array<P: PackedField, const N: usize>(
    packed_digest: [P; N],
) -> impl Iterator<Item = [P::Scalar; N]> {
    (0..P::WIDTH).map(move |j| packed_digest.map(|p| p.as_slice()[j]))
}

/// A vector commitment scheme backed by a Merkle tree.
///
/// Generics:
/// - `L`: a leaf value
/// - `D`: a digest
/// - `H`: the leaf hasher
/// - `C`: the digest compression function
#[derive(Copy, Clone)]
pub struct FieldMerkleTreeMmcs<P, H, C, const DIGEST_ELEMS: usize> {
    hash: H,
    compress: C,
    _phantom_p: PhantomData<P>,
}

impl<P, H, C, const DIGEST_ELEMS: usize> FieldMerkleTreeMmcs<P, H, C, DIGEST_ELEMS> {
    pub fn new(hash: H, compress: C) -> Self {
        Self {
            hash,
            compress,
            _phantom_p: PhantomData,
        }
    }
}

impl<P, H, C, const DIGEST_ELEMS: usize> Mmcs<P::Scalar>
    for FieldMerkleTreeMmcs<P, H, C, DIGEST_ELEMS>
where
    P: PackedField,
    H: CryptographicHasher<P::Scalar, [P::Scalar; DIGEST_ELEMS]>,
    H: CryptographicHasher<P, [P; DIGEST_ELEMS]>,
    H: Sync,
    C: PseudoCompressionFunction<[P::Scalar; DIGEST_ELEMS], 2>,
    C: PseudoCompressionFunction<[P; DIGEST_ELEMS], 2>,
    C: Sync,
{
    type ProverData = FieldMerkleTree<P::Scalar, DIGEST_ELEMS>;
    type Commitment = [P::Scalar; DIGEST_ELEMS];
    type Proof = Vec<[P::Scalar; DIGEST_ELEMS]>;
    type Error = ();
    type Mat<'a> = RowMajorMatrixView<'a, P::Scalar> where H: 'a, C: 'a;

    fn open_batch(
        &self,
        index: usize,
        prover_data: &FieldMerkleTree<P::Scalar, DIGEST_ELEMS>,
    ) -> (Vec<Vec<P::Scalar>>, Vec<[P::Scalar; DIGEST_ELEMS]>) {
        let max_height = self.get_max_height(prover_data);
        let log_max_height = log2_ceil_usize(max_height);

        let openings = prover_data
            .leaves
            .iter()
            .map(|matrix| {
                let log2_height = log2_ceil_usize(matrix.height());
                let bits_reduced = log_max_height - log2_height;
                let reduced_index = index >> bits_reduced;
                matrix.row(reduced_index).collect()
            })
            .collect_vec();

        let proof = (0..log_max_height)
            .map(|i| prover_data.digest_layers[i][(index >> i) ^ 1])
            .collect();

        (openings, proof)
    }

    fn get_matrices<'a>(
        &'a self,
        prover_data: &'a Self::ProverData,
    ) -> Vec<RowMajorMatrixView<'a, P::Scalar>> {
        prover_data.leaves.iter().map(|mat| mat.as_view()).collect()
    }

    fn verify_batch(
        &self,
        commit: &[P::Scalar; DIGEST_ELEMS],
        dimensions: &[Dimensions],
        mut index: usize,
        opened_values: &[Vec<P::Scalar>],
        proof: &Vec<[P::Scalar; DIGEST_ELEMS]>,
    ) -> Result<(), Self::Error> {
        let mut heights_tallest_first = dimensions
            .iter()
            .enumerate()
            .sorted_by_key(|(_, dims)| Reverse(dims.height))
            .peekable();

        let mut curr_height_padded = heights_tallest_first
            .peek()
            .unwrap()
            .1
            .height
            .next_power_of_two();

        let mut root = self.hash.hash_iter_slices(
            heights_tallest_first
                .peeking_take_while(|(_, dims)| {
                    dims.height.next_power_of_two() == curr_height_padded
                })
                .map(|(i, _)| opened_values[i].as_slice()),
        );

        for &sibling in proof.iter() {
            let (left, right) = if index & 1 == 0 {
                (root, sibling)
            } else {
                (sibling, root)
            };

            root = self.compress.compress([left, right]);
            index >>= 1;
            curr_height_padded >>= 1;

            let next_height = heights_tallest_first
                .peek()
                .map(|(_, dims)| dims.height)
                .filter(|h| h.next_power_of_two() == curr_height_padded);
            if let Some(next_height) = next_height {
                let next_height_openings_digest = self.hash.hash_iter_slices(
                    heights_tallest_first
                        .peeking_take_while(|(_, dims)| dims.height == next_height)
                        .map(|(i, _)| opened_values[i].as_slice()),
                );

                root = self.compress.compress([root, next_height_openings_digest]);
            }
        }

        if root == *commit {
            Ok(())
        } else {
            Err(())
        }
    }
}

impl<P, H, C, const DIGEST_ELEMS: usize> DirectMmcs<P::Scalar>
    for FieldMerkleTreeMmcs<P, H, C, DIGEST_ELEMS>
where
    P: PackedField,
    H: CryptographicHasher<P::Scalar, [P::Scalar; DIGEST_ELEMS]>,
    H: CryptographicHasher<P, [P; DIGEST_ELEMS]>,
    H: Sync,
    C: PseudoCompressionFunction<[P::Scalar; DIGEST_ELEMS], 2>,
    C: PseudoCompressionFunction<[P; DIGEST_ELEMS], 2>,
    C: Sync,
{
    fn commit(
        &self,
        inputs: Vec<RowMajorMatrix<P::Scalar>>,
    ) -> (Self::Commitment, Self::ProverData) {
        let tree = FieldMerkleTree::new::<P, H, C>(&self.hash, &self.compress, inputs);
        let root = tree.root();
        (root, tree)
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use itertools::Itertools;
    use p3_baby_bear::BabyBear;
    use p3_commit::{DirectMmcs, Mmcs};
    use p3_field::{AbstractField, Field};
    use p3_matrix::dense::RowMajorMatrix;
    use p3_matrix::{Dimensions, Matrix};
    use p3_mds::coset_mds::CosetMds;
    use p3_poseidon2::{DiffusionMatrixBabybear, Poseidon2};
    use p3_symmetric::compression::{PseudoCompressionFunction, TruncatedPermutation};
    use p3_symmetric::hasher::CryptographicHasher;
    use p3_symmetric::sponge::PaddingFreeSponge;
    use rand::thread_rng;

    use crate::FieldMerkleTreeMmcs;

    type F = BabyBear;

    type MyMds = CosetMds<F, 16>;
    type Perm = Poseidon2<F, MyMds, DiffusionMatrixBabybear, 16, 5>;
    type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
    type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
    type MyMmcs = FieldMerkleTreeMmcs<<F as Field>::Packing, MyHash, MyCompress, 8>;

    #[test]
    fn commit_single_1x8() {
        let mds = MyMds::default();
        let perm = Perm::new_from_rng(8, 22, mds, DiffusionMatrixBabybear, &mut thread_rng());
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm);
        let mmcs = MyMmcs::new(hash.clone(), compress.clone());

        // v = [2, 1, 2, 2, 0, 0, 1, 0]
        let v = vec![
            F::TWO,
            F::ONE,
            F::TWO,
            F::TWO,
            F::ZERO,
            F::ZERO,
            F::ONE,
            F::ZERO,
        ];
        let (commit, _) = mmcs.commit_vec(v.clone());

        let expected_result = compress.compress([
            compress.compress([
                compress.compress([hash.hash_item(v[0]), hash.hash_item(v[1])]),
                compress.compress([hash.hash_item(v[2]), hash.hash_item(v[3])]),
            ]),
            compress.compress([
                compress.compress([hash.hash_item(v[4]), hash.hash_item(v[5])]),
                compress.compress([hash.hash_item(v[6]), hash.hash_item(v[7])]),
            ]),
        ]);
        assert_eq!(commit, expected_result);
    }

    #[test]
    fn commit_single_2x2() {
        let mds = MyMds::default();
        let perm = Perm::new_from_rng(8, 22, mds, DiffusionMatrixBabybear, &mut thread_rng());
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm);
        let mmcs = MyMmcs::new(hash.clone(), compress.clone());

        // mat = [
        //   0 1
        //   2 1
        // ]
        let mat = RowMajorMatrix::new(vec![F::ZERO, F::ONE, F::TWO, F::ONE], 2);

        let (commit, _) = mmcs.commit(vec![mat]);

        let expected_result = compress.compress([
            hash.hash_slice(&[F::ZERO, F::ONE]),
            hash.hash_slice(&[F::TWO, F::ONE]),
        ]);
        assert_eq!(commit, expected_result);
    }

    #[test]
    fn commit_single_2x3() {
        let mds = MyMds::default();
        let perm = Perm::new_from_rng(8, 22, mds, DiffusionMatrixBabybear, &mut thread_rng());
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm);
        let mmcs = MyMmcs::new(hash.clone(), compress.clone());
        let default_digest = [F::ZERO; 8];

        // mat = [
        //   0 1
        //   2 1
        //   2 2
        // ]
        let mat = RowMajorMatrix::new(vec![F::ZERO, F::ONE, F::TWO, F::ONE, F::TWO, F::TWO], 2);

        let (commit, _) = mmcs.commit(vec![mat]);

        let expected_result = compress.compress([
            compress.compress([
                hash.hash_slice(&[F::ZERO, F::ONE]),
                hash.hash_slice(&[F::TWO, F::ONE]),
            ]),
            compress.compress([hash.hash_slice(&[F::TWO, F::TWO]), default_digest]),
        ]);
        assert_eq!(commit, expected_result);
    }

    #[test]
    fn commit_mixed() {
        let mds = MyMds::default();
        let perm = Perm::new_from_rng(8, 22, mds, DiffusionMatrixBabybear, &mut thread_rng());
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm);
        let mmcs = MyMmcs::new(hash.clone(), compress.clone());
        let default_digest = [F::ZERO; 8];

        // mat_1 = [
        //   0 1
        //   2 1
        //   2 2
        // ]
        let mat_1 = RowMajorMatrix::new(vec![F::ZERO, F::ONE, F::TWO, F::ONE, F::TWO, F::TWO], 2);
        // mat_2 = [
        //   1 2 1
        //   0 2 2
        // ]
        let mat_2 = RowMajorMatrix::new(vec![F::ONE, F::TWO, F::ONE, F::ZERO, F::TWO, F::TWO], 3);

        let (commit, prover_data) = mmcs.commit(vec![mat_1, mat_2]);

        let mat_1_leaf_hashes = [
            hash.hash_slice(&[F::ZERO, F::ONE]),
            hash.hash_slice(&[F::TWO, F::ONE]),
            hash.hash_slice(&[F::TWO, F::TWO]),
        ];
        let mat_2_leaf_hashes = [
            hash.hash_slice(&[F::ONE, F::TWO, F::ONE]),
            hash.hash_slice(&[F::ZERO, F::TWO, F::TWO]),
        ];

        let expected_result = compress.compress([
            compress.compress([
                compress.compress([mat_1_leaf_hashes[0], mat_1_leaf_hashes[1]]),
                mat_2_leaf_hashes[0],
            ]),
            compress.compress([
                compress.compress([mat_1_leaf_hashes[2], default_digest]),
                mat_2_leaf_hashes[1],
            ]),
        ]);
        assert_eq!(commit, expected_result);

        let (opened_values, _proof) = mmcs.open_batch(2, &prover_data);
        assert_eq!(
            opened_values,
            vec![vec![F::TWO, F::TWO], vec![F::ZERO, F::TWO, F::TWO]]
        );
    }

    #[test]
    fn commit_either_order() {
        let mut rng = thread_rng();
        let mds = MyMds::default();
        let perm = Perm::new_from_rng(8, 22, mds, DiffusionMatrixBabybear, &mut rng);
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm);
        let mmcs = MyMmcs::new(hash, compress);

        let input_1 = RowMajorMatrix::<F>::rand(&mut rng, 5, 8);
        let input_2 = RowMajorMatrix::<F>::rand(&mut rng, 3, 16);

        let (commit_1_2, _) = mmcs.commit(vec![input_1.clone(), input_2.clone()]);
        let (commit_2_1, _) = mmcs.commit(vec![input_2, input_1]);
        assert_eq!(commit_1_2, commit_2_1);
    }

    #[test]
    #[should_panic]
    fn mismatched_heights() {
        let mut rng = thread_rng();
        let mds = MyMds::default();
        let perm = Perm::new_from_rng(8, 22, mds, DiffusionMatrixBabybear, &mut rng);
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm);
        let mmcs = MyMmcs::new(hash, compress);

        // attempt to commit to a mat with 8 rows and a mat with 7 rows. this should panic.
        let large_mat = RowMajorMatrix::new(
            [1, 2, 3, 4, 5, 6, 7, 8].map(F::from_canonical_u8).to_vec(),
            1,
        );
        let small_mat =
            RowMajorMatrix::new([1, 2, 3, 4, 5, 6, 7].map(F::from_canonical_u8).to_vec(), 1);
        let _ = mmcs.commit(vec![large_mat, small_mat]);
    }

    #[test]
    fn verify_tampered_proof_fails() {
        let mut rng = thread_rng();
        let mds = MyMds::default();
        let perm = Perm::new_from_rng(8, 22, mds, DiffusionMatrixBabybear, &mut rng);
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm);
        let mmcs = MyMmcs::new(hash, compress);

        // 4 8x1 matrixes, 4 8x2 matrixes
        let large_mats = (0..4).map(|_| RowMajorMatrix::<F>::rand(&mut thread_rng(), 8, 1));
        let large_mat_dims = (0..4).map(|_| Dimensions {
            height: 8,
            width: 1,
        });
        let small_mats = (0..4).map(|_| RowMajorMatrix::<F>::rand(&mut thread_rng(), 8, 2));
        let small_mat_dims = (0..4).map(|_| Dimensions {
            height: 8,
            width: 2,
        });

        let (commit, prover_data) = mmcs.commit(large_mats.chain(small_mats).collect_vec());

        // open the 3rd row of each matrix, mess with proof, and verify
        let (opened_values, mut proof) = mmcs.open_batch(3, &prover_data);
        proof[0][0] += F::ONE;
        mmcs.verify_batch(
            &commit,
            &large_mat_dims.chain(small_mat_dims).collect_vec(),
            3,
            &opened_values,
            &proof,
        )
        .expect_err("expected verification to fail");
    }

    #[test]
    fn size_gaps() {
        let mut rng = thread_rng();
        let mds = MyMds::default();
        let perm = Perm::new_from_rng(8, 22, mds, DiffusionMatrixBabybear, &mut rng);
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm);
        let mmcs = MyMmcs::new(hash, compress);

        // 4 mats with 1000 rows, 8 columns
        let large_mats = (0..4).map(|_| RowMajorMatrix::<F>::rand(&mut thread_rng(), 1000, 8));
        let large_mat_dims = (0..4).map(|_| Dimensions {
            height: 1000,
            width: 8,
        });

        // 5 mats with 70 rows, 8 columns
        let medium_mats = (0..5).map(|_| RowMajorMatrix::<F>::rand(&mut thread_rng(), 70, 8));
        let medium_mat_dims = (0..5).map(|_| Dimensions {
            height: 70,
            width: 8,
        });

        // 6 mats with 8 rows, 8 columns
        let small_mats = (0..6).map(|_| RowMajorMatrix::<F>::rand(&mut thread_rng(), 8, 8));
        let small_mat_dims = (0..6).map(|_| Dimensions {
            height: 8,
            width: 8,
        });

        let (commit, prover_data) = mmcs.commit(
            large_mats
                .chain(medium_mats)
                .chain(small_mats)
                .collect_vec(),
        );

        // open the 6th row of each matrix and verify
        let (opened_values, proof) = mmcs.open_batch(6, &prover_data);
        mmcs.verify_batch(
            &commit,
            &large_mat_dims
                .chain(medium_mat_dims)
                .chain(small_mat_dims)
                .collect_vec(),
            6,
            &opened_values,
            &proof,
        )
        .expect("expected verification to succeed");
    }

    #[test]
    fn different_widths() {
        let mut rng = thread_rng();
        let mds = MyMds::default();
        let perm = Perm::new_from_rng(8, 22, mds, DiffusionMatrixBabybear, &mut rng);
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm);
        let mmcs = MyMmcs::new(hash, compress);

        // 10 mats with 32 rows where the ith mat has i + 1 cols
        let mats = (0..10)
            .map(|i| RowMajorMatrix::<F>::rand(&mut thread_rng(), 32, i + 1))
            .collect_vec();
        let dims = mats.iter().map(|m| m.dimensions()).collect_vec();

        let (commit, prover_data) = mmcs.commit(mats);
        let (opened_values, proof) = mmcs.open_batch(17, &prover_data);
        mmcs.verify_batch(&commit, &dims, 17, &opened_values, &proof)
            .expect("expected verification to succeed");
    }
}
