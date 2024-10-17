use alloc::vec::Vec;
use core::cell::RefCell;
use core::cmp::Reverse;
use core::marker::PhantomData;

use itertools::Itertools;
use p3_commit::Mmcs;
use p3_field::PackedValue;
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::stack::HorizontalPair;
use p3_matrix::{Dimensions, Matrix};
use p3_symmetric::{CryptographicHasher, Hash, PseudoCompressionFunction};
use p3_util::log2_ceil_usize;
use rand::distributions::{Distribution, Standard};
use rand::Rng;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use crate::MerkleTreeError::{RootMismatch, WrongBatchSize, WrongHeight};
use crate::{MerkleTree, MerkleTreeError};

/// A vector commitment scheme backed by a `MerkleTree`.
///
/// This is similar to `MerkleTreeMmcs`, but each leaf is "salted" with random elements. This is
/// done to turn the Merkle tree into a hiding commitment. See e.g. Section 3 of
/// [Interactive Oracle Proofs](https://eprint.iacr.org/2016/116).
///
/// `SALT_ELEMS` should be set such that the product of `SALT_ELEMS` with the size of the value
/// (`P::Value`) is at least the target security parameter.
///
/// `R` should be an appropriately seeded cryptographically secure pseudorandom number generator
/// (CSPRNG). Something like `ThreadRng` may work, although it relies on the operating system to
/// provide sufficient entropy.
///
/// Generics:
/// - `P`: a leaf value
/// - `PW`: an element of a digest
/// - `H`: the leaf hasher
/// - `C`: the digest compression function
/// - `R`: a random number generator for blinding leaves
#[derive(Clone, Debug)]
pub struct MerkleTreeHidingMmcs<P, PW, H, C, R, const DIGEST_ELEMS: usize, const SALT_ELEMS: usize>
{
    hash: H,
    compress: C,
    rng: RefCell<R>,
    _phantom: PhantomData<(P, PW)>,
}

impl<P, PW, H, C, R, const DIGEST_ELEMS: usize, const SALT_ELEMS: usize>
    MerkleTreeHidingMmcs<P, PW, H, C, R, DIGEST_ELEMS, SALT_ELEMS>
{
    pub fn new(hash: H, compress: C, rng: R) -> Self {
        Self {
            hash,
            compress,
            rng: rng.into(),
            _phantom: PhantomData,
        }
    }
}

impl<P, PW, H, C, R, const DIGEST_ELEMS: usize, const SALT_ELEMS: usize> Mmcs<P::Value>
    for MerkleTreeHidingMmcs<P, PW, H, C, R, DIGEST_ELEMS, SALT_ELEMS>
where
    P: PackedValue,
    P::Value: Serialize + DeserializeOwned,
    PW: PackedValue,
    H: CryptographicHasher<P::Value, [PW::Value; DIGEST_ELEMS]>,
    H: CryptographicHasher<P, [PW; DIGEST_ELEMS]>,
    H: Sync,
    C: PseudoCompressionFunction<[PW::Value; DIGEST_ELEMS], 2>,
    C: PseudoCompressionFunction<[PW; DIGEST_ELEMS], 2>,
    C: Sync,
    R: Rng + Clone,
    PW::Value: Eq,
    [PW::Value; DIGEST_ELEMS]: Serialize + for<'de> Deserialize<'de>,
    Standard: Distribution<P::Value>,
{
    type ProverData<M> =
        MerkleTree<P::Value, PW::Value, HorizontalPair<M, RowMajorMatrix<P::Value>>, DIGEST_ELEMS>;
    type Commitment = Hash<P::Value, PW::Value, DIGEST_ELEMS>;
    /// The first item is salts; the second is the usual Merkle proof (sibling digests).
    type Proof = (Vec<Vec<P::Value>>, Vec<[PW::Value; DIGEST_ELEMS]>);
    type Error = MerkleTreeError;

    fn commit<M: Matrix<P::Value>>(
        &self,
        inputs: Vec<M>,
    ) -> (Self::Commitment, Self::ProverData<M>) {
        let salted_inputs = inputs
            .into_iter()
            .map(|mat| {
                let salts =
                    RowMajorMatrix::rand(&mut *self.rng.borrow_mut(), mat.height(), SALT_ELEMS);
                HorizontalPair::new(mat, salts)
            })
            .collect();
        let tree = MerkleTree::new::<P, PW, H, C>(&self.hash, &self.compress, salted_inputs);
        let root = tree.root();
        (root, tree)
    }

    fn open_batch<M: Matrix<P::Value>>(
        &self,
        index: usize,
        prover_data: &MerkleTree<
            P::Value,
            PW::Value,
            HorizontalPair<M, RowMajorMatrix<P::Value>>,
            DIGEST_ELEMS,
        >,
    ) -> (
        Vec<Vec<P::Value>>,
        (Vec<Vec<P::Value>>, Vec<[PW::Value; DIGEST_ELEMS]>),
    ) {
        let max_height = self.get_max_height(prover_data);
        let log_max_height = log2_ceil_usize(max_height);

        let (openings, salts): (Vec<_>, Vec<_>) = prover_data
            .leaves
            .iter()
            .map(|matrix| {
                let log2_height = log2_ceil_usize(matrix.height());
                let bits_reduced = log_max_height - log2_height;
                let reduced_index = index >> bits_reduced;
                let opening = matrix.first.row(reduced_index).collect();
                let salt = matrix.second.row(reduced_index).collect();
                (opening, salt)
            })
            .unzip();

        let siblings: Vec<_> = (0..log_max_height)
            .map(|i| prover_data.digest_layers[i][(index >> i) ^ 1])
            .collect();

        (openings, (salts, siblings))
    }

    fn get_matrices<'a, M: Matrix<P::Value>>(
        &self,
        prover_data: &'a Self::ProverData<M>,
    ) -> Vec<&'a M> {
        prover_data.leaves.iter().map(|mat| &mat.first).collect()
    }

    fn verify_batch(
        &self,
        commit: &Self::Commitment,
        dimensions: &[Dimensions],
        mut index: usize,
        opened_values: &[Vec<P::Value>],
        proof: &Self::Proof,
    ) -> Result<(), Self::Error> {
        let (salts, siblings) = proof;
        // Check that the openings have the correct shape.
        if dimensions.len() != opened_values.len() {
            return Err(WrongBatchSize);
        }

        // TODO: Disabled for now since TwoAdicFriPcs and CirclePcs currently pass 0 for width.
        // for (dims, opened_vals) in dimensions.iter().zip(opened_values) {
        //     if opened_vals.len() != dims.width {
        //         return Err(WrongWidth);
        //     }
        // }

        // TODO: Disabled for now, CirclePcs sometimes passes a height that's off by 1 bit.
        let max_height = dimensions.iter().map(|dim| dim.height).max().unwrap();
        let log_max_height = log2_ceil_usize(max_height);
        if siblings.len() != log_max_height {
            return Err(WrongHeight {
                max_height,
                num_siblings: siblings.len(),
            });
        }

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
                .flat_map(|(i, _)| [opened_values[i].as_slice(), salts[i].as_slice()]),
        );

        for &sibling in siblings.iter() {
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
                        .flat_map(|(i, _)| [opened_values[i].as_slice(), salts[i].as_slice()]),
                );

                root = self.compress.compress([root, next_height_openings_digest]);
            }
        }

        if commit == &root {
            Ok(())
        } else {
            Err(RootMismatch)
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use itertools::Itertools;
    use p3_baby_bear::{BabyBear, DiffusionMatrixBabyBear};
    use p3_commit::Mmcs;
    use p3_field::{AbstractField, Field};
    use p3_matrix::dense::RowMajorMatrix;
    use p3_matrix::Matrix;
    use p3_poseidon2::{Poseidon2, Poseidon2ExternalMatrixGeneral};
    use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
    use rand::prelude::*;

    use super::MerkleTreeHidingMmcs;
    use crate::MerkleTreeError;

    type F = BabyBear;
    const SALT_ELEMS: usize = 4;

    type Perm = Poseidon2<F, Poseidon2ExternalMatrixGeneral, DiffusionMatrixBabyBear, 16, 7>;
    type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
    type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
    type MyMmcs = MerkleTreeHidingMmcs<
        <F as Field>::Packing,
        <F as Field>::Packing,
        MyHash,
        MyCompress,
        ThreadRng,
        8,
        SALT_ELEMS,
    >;

    #[test]
    #[should_panic]
    fn mismatched_heights() {
        let mut rng = thread_rng();
        let perm = Perm::new_from_rng_128(
            Poseidon2ExternalMatrixGeneral,
            DiffusionMatrixBabyBear::default(),
            &mut rng,
        );
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm);
        let mmcs = MyMmcs::new(hash, compress, thread_rng());

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
    fn different_widths() -> Result<(), MerkleTreeError> {
        let mut rng = thread_rng();
        let perm = Perm::new_from_rng_128(
            Poseidon2ExternalMatrixGeneral,
            DiffusionMatrixBabyBear::default(),
            &mut rng,
        );
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm);
        let mmcs = MyMmcs::new(hash, compress, thread_rng());

        // 10 mats with 32 rows where the ith mat has i + 1 cols
        let mats = (0..10)
            .map(|i| RowMajorMatrix::<F>::rand(&mut thread_rng(), 32, i + 1))
            .collect_vec();
        let dims = mats.iter().map(|m| m.dimensions()).collect_vec();

        let (commit, prover_data) = mmcs.commit(mats);
        let (opened_values, proof) = mmcs.open_batch(17, &prover_data);
        mmcs.verify_batch(&commit, &dims, 17, &opened_values, &proof)
    }
}
