use alloc::vec::Vec;

use itertools::Itertools;
use p3_commit::{BatchOpening, BatchOpeningRef, Mmcs};
use p3_field::PackedValue;
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::stack::HorizontalPair;
use p3_matrix::{Dimensions, Matrix};
use p3_symmetric::{CryptographicHasher, PseudoCompressionFunction};
use p3_util::zip_eq::zip_eq;
use rand::Rng;
use rand::distr::{Distribution, StandardUniform};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use spin::Mutex;

use crate::mmcs::check_widths;
use crate::pruning::PrunedMerklePaths;
use crate::{MerkleCap, MerkleTree, MerkleTreeError, MerkleTreeMmcs};

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
#[derive(Debug)]
pub struct MerkleTreeHidingMmcs<
    P,
    PW,
    H,
    C,
    R,
    const N: usize,
    const DIGEST_ELEMS: usize,
    const SALT_ELEMS: usize,
> {
    inner: MerkleTreeMmcs<P, PW, H, C, N, DIGEST_ELEMS>,
    rng: Mutex<R>,
}

impl<P, PW, H, C, R, const N: usize, const DIGEST_ELEMS: usize, const SALT_ELEMS: usize>
    MerkleTreeHidingMmcs<P, PW, H, C, R, N, DIGEST_ELEMS, SALT_ELEMS>
{
    /// Create a new `MerkleTreeHidingMmcs` with the given hash and compression functions.
    ///
    /// # Arguments
    /// * `hash` - The hash function used to hash individual matrix rows (leaf level).
    /// * `compress` - The compression function used to hash internal tree nodes.
    /// * `cap_height` - The height of the Merkle cap. A cap_height of 0 uses only the root,
    ///   while a cap_height of h uses all hashes from the h-th level below the root (2^h
    ///   in the case of a binary tree)
    /// * `rng` - A random number generator for generating salts.
    pub const fn new(hash: H, compress: C, cap_height: usize, rng: R) -> Self {
        let inner = MerkleTreeMmcs::new(hash, compress, cap_height);
        Self {
            inner,
            rng: Mutex::new(rng),
        }
    }

    pub const fn cap_height(&self) -> usize {
        self.inner.cap_height()
    }
}

impl<P, PW, H, C, R, const N: usize, const DIGEST_ELEMS: usize, const SALT_ELEMS: usize> Clone
    for MerkleTreeHidingMmcs<P, PW, H, C, R, N, DIGEST_ELEMS, SALT_ELEMS>
where
    MerkleTreeMmcs<P, PW, H, C, N, DIGEST_ELEMS>: Clone,
    R: Clone,
{
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            rng: Mutex::new(self.rng.lock().clone()),
        }
    }
}

impl<P, PW, H, C, R, const N: usize, const DIGEST_ELEMS: usize, const SALT_ELEMS: usize>
    Mmcs<P::Value> for MerkleTreeHidingMmcs<P, PW, H, C, R, N, DIGEST_ELEMS, SALT_ELEMS>
where
    P: PackedValue,
    P::Value: Serialize + DeserializeOwned,
    PW: PackedValue,
    H: CryptographicHasher<P::Value, [PW::Value; DIGEST_ELEMS]>
        + CryptographicHasher<P, [PW; DIGEST_ELEMS]>
        + Sync,
    C: PseudoCompressionFunction<[PW::Value; DIGEST_ELEMS], N>
        + PseudoCompressionFunction<[PW; DIGEST_ELEMS], N>
        + Sync,
    R: Rng + Clone + Send,
    PW::Value: Eq + Clone,
    [PW::Value; DIGEST_ELEMS]: Serialize + for<'de> Deserialize<'de>,
    StandardUniform: Distribution<P::Value>,
{
    type ProverData<M> = MerkleTree<
        P::Value,
        PW::Value,
        HorizontalPair<M, RowMajorMatrix<P::Value>>,
        N,
        DIGEST_ELEMS,
    >;
    type Commitment = MerkleCap<P::Value, [PW::Value; DIGEST_ELEMS]>;
    /// The first item is salts; the second is the usual Merkle proof (sibling digests).
    type Proof = (Vec<Vec<P::Value>>, Vec<[PW::Value; DIGEST_ELEMS]>);
    /// The first item is the per-query salts (`salts[q][m]`).
    /// The second is the shared pruned multiproof.
    type MultiProof = (
        Vec<Vec<Vec<P::Value>>>,
        PrunedMerklePaths<PW::Value, DIGEST_ELEMS>,
    );
    type Error = MerkleTreeError;

    fn commit<M: Matrix<P::Value>>(
        &self,
        inputs: Vec<M>,
    ) -> (Self::Commitment, Self::ProverData<M>) {
        let mut rng = self.rng.lock();
        let salted_inputs = inputs
            .into_iter()
            .map(|mat| {
                let salts = RowMajorMatrix::rand(&mut *rng, mat.height(), SALT_ELEMS);
                HorizontalPair::new(mat, salts)
            })
            .collect();
        self.inner.commit(salted_inputs)
    }

    fn open_batch<M: Matrix<P::Value>>(
        &self,
        index: usize,
        prover_data: &Self::ProverData<M>,
    ) -> BatchOpening<P::Value, Self> {
        let (salted_openings, siblings) = self.inner.open_batch(index, prover_data).unpack();
        let (openings, salts) = salted_openings
            .into_iter()
            .map(|row| {
                let (a, b) = row.split_at(row.len() - SALT_ELEMS);
                (a.to_vec(), b.to_vec())
            })
            .unzip();
        BatchOpening::new(openings, (salts, siblings))
    }

    fn get_matrices<'a, M: Matrix<P::Value>>(
        &self,
        prover_data: &'a Self::ProverData<M>,
    ) -> Vec<&'a M> {
        prover_data.leaves.iter().map(|mat| &mat.left).collect()
    }

    fn verify_batch(
        &self,
        commit: &Self::Commitment,
        dimensions: &[Dimensions],
        index: usize,
        batch_opening: BatchOpeningRef<'_, P::Value, Self>,
    ) -> Result<(), Self::Error> {
        let (opened_values, (salts, siblings)) = batch_opening.unpack();

        // Pin each opened row to its matrix width before salting.
        // The inner tree only sees salted widths.
        // Without this, an over-long row could be masked by an under-long salt.
        check_widths(dimensions, opened_values)?;

        let opened_salted_values = zip_eq(opened_values, salts, MerkleTreeError::WrongBatchSize)?
            .map(|(opened, salt)| opened.iter().chain(salt.iter()).copied().collect_vec())
            .collect_vec();

        // The inner tree commits to rows widened by the salt columns,
        // so the widths must be widened the same way.
        let salted_dimensions = dimensions
            .iter()
            .map(|dims| Dimensions {
                width: dims.width + SALT_ELEMS,
                height: dims.height,
            })
            .collect_vec();

        self.inner.verify_batch(
            commit,
            &salted_dimensions,
            index,
            BatchOpeningRef::new(&opened_salted_values, siblings),
        )
    }

    fn open_multi_batch<M: Matrix<P::Value>>(
        &self,
        indices: &[usize],
        prover_data: &Self::ProverData<M>,
    ) -> (Vec<Vec<Vec<P::Value>>>, Self::MultiProof) {
        // The inner tree opens salted rows: each row is `real_row ++ salt`.
        let (salted_values, pruned) = self.inner.open_multi_batch(indices, prover_data);

        // Split every salted row back into its unsalted prefix and its salt suffix.
        // Salts move into the proof.
        // The unsalted rows are returned to the caller.
        let mut salts = Vec::with_capacity(salted_values.len());
        let opened_values = salted_values
            .into_iter()
            .map(|rows_at_index| {
                let (opened, salts_at_index): (Vec<_>, Vec<_>) = rows_at_index
                    .into_iter()
                    .map(|row| {
                        let (a, b) = row.split_at(row.len() - SALT_ELEMS);
                        (a.to_vec(), b.to_vec())
                    })
                    .unzip();
                salts.push(salts_at_index);
                opened
            })
            .collect();

        (opened_values, (salts, pruned))
    }

    fn verify_multi_batch<Row: AsRef<[P::Value]> + PartialEq>(
        &self,
        commit: &Self::Commitment,
        dimensions: &[Dimensions],
        indices: &[usize],
        opened_values: &[Vec<Row>],
        proof: &Self::MultiProof,
    ) -> Result<(), Self::Error> {
        let (salts, pruned) = proof;

        // Re-attach salts query by query, mirroring the single-opening path.
        let mut salted_values = Vec::with_capacity(opened_values.len());
        for (rows, salts_at_index) in zip_eq(opened_values, salts, MerkleTreeError::WrongBatchSize)?
        {
            // Pin each unsalted row to its matrix width before salting.
            // The inner tree only sees salted widths.
            // An over-long row could otherwise hide behind an under-long salt.
            check_widths(dimensions, rows)?;

            let salted_rows = zip_eq(rows, salts_at_index, MerkleTreeError::WrongBatchSize)?
                .map(|(opened, salt)| {
                    opened
                        .as_ref()
                        .iter()
                        .chain(salt.iter())
                        .copied()
                        .collect_vec()
                })
                .collect_vec();
            salted_values.push(salted_rows);
        }

        // The inner tree commits to salt-widened rows, so widen the dimensions to match.
        let salted_dimensions = dimensions
            .iter()
            .map(|dims| Dimensions {
                width: dims.width + SALT_ELEMS,
                height: dims.height,
            })
            .collect_vec();

        self.inner
            .verify_multi_batch(commit, &salted_dimensions, indices, &salted_values, pruned)
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use itertools::Itertools;
    use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
    use p3_commit::Mmcs;
    use p3_field::{Field, PrimeCharacteristicRing};
    use p3_matrix::Matrix;
    use p3_matrix::dense::RowMajorMatrix;
    use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
    use p3_util::assert_sync;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    use super::MerkleTreeHidingMmcs;
    use crate::MerkleTreeError;

    type F = BabyBear;
    const SALT_ELEMS: usize = 4;

    type Perm = Poseidon2BabyBear<16>;
    type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
    type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
    type MyMmcs = MerkleTreeHidingMmcs<
        <F as Field>::Packing,
        <F as Field>::Packing,
        MyHash,
        MyCompress,
        SmallRng,
        2,
        8,
        SALT_ELEMS,
    >;

    #[test]
    #[should_panic]
    fn mismatched_heights() {
        let mut rng = SmallRng::seed_from_u64(1);
        let perm = Perm::new_from_rng_128(&mut rng);
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm);
        let mmcs = MyMmcs::new(hash, compress, 0, rng);

        // attempt to commit to a mat with 8 rows and a mat with 7 rows. this should panic.
        let large_mat = RowMajorMatrix::new([1, 2, 3, 4, 5, 6, 7, 8].map(F::from_u8).to_vec(), 1);
        let small_mat = RowMajorMatrix::new([1, 2, 3, 4, 5, 6, 7].map(F::from_u8).to_vec(), 1);
        let _ = mmcs.commit(vec![large_mat, small_mat]);
    }

    #[test]
    fn different_widths() -> Result<(), MerkleTreeError> {
        let mut rng = SmallRng::seed_from_u64(1);
        // 10 mats with 32 rows where the ith mat has i + 1 cols
        let mats = (0..10)
            .map(|i| RowMajorMatrix::<F>::rand(&mut rng, 32, i + 1))
            .collect_vec();
        let perm = Perm::new_from_rng_128(&mut rng);
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm);
        let mmcs = MyMmcs::new(hash, compress, 0, rng);

        let dims = mats.iter().map(|m| m.dimensions()).collect_vec();

        let (commit, prover_data) = mmcs.commit(mats);
        let batch_proof = mmcs.open_batch(17, &prover_data);
        mmcs.verify_batch(&commit, &dims, 17, (&batch_proof).into())
    }

    #[test]
    fn verify_rejects_wrong_row_width() {
        let mut rng = SmallRng::seed_from_u64(2);
        let perm = Perm::new_from_rng_128(&mut rng);
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm);

        // Commit to one matrix of width 4 through the hiding wrapper.
        let mat = RowMajorMatrix::<F>::rand(&mut rng, 8, 4);
        let dims = vec![mat.dimensions()];
        let mmcs = MyMmcs::new(hash, compress, 0, rng);
        let (commit, prover_data) = mmcs.commit(vec![mat]);

        // Mutation: append one extra element to the opened row.
        //
        //     opened row:  [a, b, c, d, EXTRA]   (5 values, width is 4)
        let mut opening = mmcs.open_batch(3, &prover_data);
        opening.opened_values[0].push(F::ONE);

        // The width is checked on the unsalted row, before salt columns are appended.
        //
        //     expected: 4 (matrix width)
        //     got:      5 (opened row length)
        let err = mmcs
            .verify_batch(&commit, &dims, 3, (&opening).into())
            .expect_err("row longer than the matrix width must be rejected");
        assert!(matches!(
            err,
            MerkleTreeError::WrongWidth {
                matrix: 0,
                expected: 4,
                got: 5,
            }
        ));
    }

    #[test]
    fn hiding_mmcs_is_sync() {
        assert_sync::<MyMmcs>();
    }

    #[test]
    fn multi_opening_round_trip() {
        let mut rng = SmallRng::seed_from_u64(3);
        // Three matrices of equal height but widths 2, 3, 4.
        let mats = (0..3)
            .map(|i| RowMajorMatrix::<F>::rand(&mut rng, 16, i + 2))
            .collect_vec();
        let dims = mats.iter().map(|m| m.dimensions()).collect_vec();
        let perm = Perm::new_from_rng_128(&mut rng);
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm);
        let mmcs = MyMmcs::new(hash, compress, 0, rng);
        let (commit, prover_data) = mmcs.commit(mats);

        // Open four leaves at once.
        // Indices must be ascending and distinct.
        let indices = vec![1usize, 4, 9, 15];
        let (opened, proof) = mmcs.open_multi_batch(&indices, &prover_data);

        // One opened row set per query, one row per committed matrix.
        assert_eq!(opened.len(), indices.len());
        assert!(opened.iter().all(|rows| rows.len() == dims.len()));

        // The salt suffix is stripped: each returned row matches its matrix width.
        for rows in &opened {
            for (row, dim) in rows.iter().zip(&dims) {
                assert_eq!(row.len(), dim.width);
            }
        }

        // The single shared multiproof authenticates every opened row.
        mmcs.verify_multi_batch(&commit, &dims, &indices, &opened, &proof)
            .expect("round-trip multi-opening must verify");
    }

    #[test]
    fn verify_multi_rejects_wrong_row_width() {
        let mut rng = SmallRng::seed_from_u64(4);
        // Commit to one matrix of width 4 through the hiding wrapper.
        let mat = RowMajorMatrix::<F>::rand(&mut rng, 8, 4);
        let dims = vec![mat.dimensions()];
        let perm = Perm::new_from_rng_128(&mut rng);
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm);
        let mmcs = MyMmcs::new(hash, compress, 0, rng);
        let (commit, prover_data) = mmcs.commit(vec![mat]);

        let indices = vec![2usize, 5];
        let (mut opened, proof) = mmcs.open_multi_batch(&indices, &prover_data);

        // Mutation: append one extra element to the first query's only row.
        //
        //     query 0 row:  [a, b, c, d, EXTRA]   (5 values, width is 4)
        opened[0][0].push(F::ONE);

        // The unsalted width is checked before salt columns are re-appended.
        // An over-long row cannot hide behind an under-long salt.
        //
        //     expected: 4 (matrix width)
        //     got:      5 (opened row length)
        let err = mmcs
            .verify_multi_batch(&commit, &dims, &indices, &opened, &proof)
            .expect_err("row longer than the matrix width must be rejected");
        assert!(matches!(
            err,
            MerkleTreeError::WrongWidth {
                matrix: 0,
                expected: 4,
                got: 5,
            }
        ));
    }
}
