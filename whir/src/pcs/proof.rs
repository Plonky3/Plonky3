use alloc::vec;
use alloc::vec::Vec;

use p3_commit::Mmcs;
use p3_matrix::{Dimensions, Matrix};
use p3_multilinear_util::poly::Poly;
use p3_sumcheck::OpeningBatch;
pub use p3_sumcheck::SumcheckData;
use serde::{Deserialize, Serialize};

/// Complete WHIR proof.
#[derive(Serialize, Deserialize, Clone)]
#[serde(bound(
    serialize = "F: Serialize, EF: Serialize, MT::MultiProof: Serialize",
    deserialize = "F: Deserialize<'de>, EF: Deserialize<'de>, MT::MultiProof: Deserialize<'de>"
))]
pub struct WhirProof<F: Send + Sync + Clone, EF, MT: Mmcs<F>> {
    /// Initial OOD evaluations.
    pub initial_ood_answers: Vec<EF>,
    /// Initial sumcheck data.
    pub initial_sumcheck: SumcheckData<F, EF>,
    /// Per-round proofs.
    pub rounds: Vec<WhirRoundProof<F, EF, MT>>,
    /// Final polynomial coefficients (sent in the clear).
    pub final_poly: Option<Poly<EF>>,
    /// Final round PoW witness.
    pub final_pow_witness: F,
    /// Final round STIR query openings.
    pub final_openings: QueryOpenings<F, EF, MT::MultiProof>,
    /// Final sumcheck data (if `final_sumcheck_rounds > 0`).
    pub final_sumcheck: Option<SumcheckData<F, EF>>,
}

/// Public opening proof produced by the WHIR PCS adapter.
///
/// # Layout
///
/// Two pieces travel together so the verifier can replay the protocol from
/// a single proof object:
///
/// - The proximity transcript: sumcheck rounds, intermediate commitments,
///   STIR query openings, and the final polynomial sent in the clear.
/// - The public opening evaluations indexed by batch.
///   Each batch keeps the evaluations at the current opening points separate
///   from the evaluations at the repeat-last successor points.
///
/// # Ordering invariant
///
/// The batches appear in the same order as the public opening schedule, so
/// the verifier can walk both side-by-side without re-sorting. A length
/// mismatch on either axis causes the adapter to reject before any Merkle
/// or sumcheck check runs.
#[derive(Serialize, Deserialize, Clone)]
#[serde(bound(
    serialize = "F: Serialize, EF: Serialize, MT::Commitment: Serialize, MT::MultiProof: Serialize",
    deserialize = "F: Deserialize<'de>, EF: Deserialize<'de>, MT::Commitment: Deserialize<'de>, MT::MultiProof: Deserialize<'de>"
))]
pub struct PcsProof<F: Send + Sync + Clone, EF, MT: Mmcs<F>> {
    /// Proximity transcript: initial commitment, sumcheck rounds, per-round
    /// commitments, STIR query openings, and the final polynomial.
    pub whir: WhirProof<F, EF, MT>,
    /// Opening evaluations in schedule order.
    /// Each batch stores the current-point evaluations separately from the
    /// repeat-last successor-point evaluations.
    pub evals: Vec<OpeningBatch<EF>>,
}

/// Per-round proof data.
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(bound(
    serialize = "F: Serialize, EF: Serialize, MT::Commitment: Serialize, MT::MultiProof: Serialize",
    deserialize = "F: Deserialize<'de>, EF: Deserialize<'de>, MT::Commitment: Deserialize<'de>, MT::MultiProof: Deserialize<'de>"
))]
pub struct WhirRoundProof<F: Send + Sync + Clone, EF, MT: Mmcs<F>> {
    /// Round commitment (Merkle root).
    pub commitment: Option<MT::Commitment>,
    /// OOD evaluations for this round.
    pub ood_answers: Vec<EF>,
    /// PoW witness after commitment.
    pub pow_witness: F,
    /// STIR query openings against the previous round's commitment.
    pub openings: QueryOpenings<F, EF, MT::MultiProof>,
    /// Sumcheck data for this round.
    pub sumcheck: SumcheckData<F, EF>,
}

/// Rows opened at many queried positions, plus one proof shared across them.
///
/// One multiproof authenticates every row together,
/// so sibling digests shared between queries travel once.
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(bound(
    serialize = "T: Serialize, P: Serialize",
    deserialize = "T: Deserialize<'de>, P: Deserialize<'de>"
))]
pub struct SharedProofOpening<T, P> {
    /// `rows[q]` is the opened leaf row at the `q`-th queried position.
    pub rows: Vec<Vec<T>>,
    /// Compact multiproof authenticating every row at once.
    pub proof: P,
}

impl<T: Send + Sync + Clone, P> SharedProofOpening<T, P> {
    /// Opens `indices` on a commitment holding exactly one matrix.
    pub(crate) fn open<MT, M>(mmcs: &MT, indices: &[usize], prover_data: &MT::ProverData<M>) -> Self
    where
        MT: Mmcs<T, MultiProof = P>,
        M: Matrix<T>,
    {
        let (values, proof) = mmcs.open_multi_batch(indices, prover_data);
        let rows = values
            .into_iter()
            .map(|mut per_matrix| {
                // WHIR commits a single matrix per round, so each query opens one row.
                assert_eq!(
                    per_matrix.len(),
                    1,
                    "WHIR opens commitments holding exactly one matrix"
                );
                per_matrix.swap_remove(0)
            })
            .collect();
        Self { rows, proof }
    }

    /// Verifies the rows against `commit` at the verifier-derived `indices`.
    ///
    /// The multiproof binds each row to its index,
    /// so a row/index count mismatch or a substituted leaf is rejected.
    pub(crate) fn verify<MT>(
        &self,
        mmcs: &MT,
        commit: &MT::Commitment,
        dimensions: &[Dimensions],
        indices: &[usize],
    ) -> Result<(), MT::Error>
    where
        MT: Mmcs<T, MultiProof = P>,
        T: PartialEq,
    {
        // WHIR commits one matrix per round, so each query opens one row.
        //
        // - The multiproof wants a `[query][matrix]` row shape.
        // - Wrap each row in a one-element slice to add the matrix axis.
        // - The slice borrows the row, copying no field data.
        let opened_values: Vec<Vec<&[T]>> =
            self.rows.iter().map(|row| vec![row.as_slice()]).collect();
        mmcs.verify_multi_batch(commit, dimensions, indices, &opened_values, &self.proof)
    }
}

/// Field-tagged shared-proof opening for one queried oracle.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum QueryOpenings<F, EF, P> {
    /// Base-field rows (the initial commitment).
    Base(SharedProofOpening<F, P>),
    /// Extension-field rows (every folded round commitment).
    Extension(SharedProofOpening<EF, P>),
}

impl<F: Clone + Send + Sync, EF, MT: Mmcs<F>> WhirProof<F, EF, MT> {
    /// Retrieve the PoW witness at a given round index.
    pub(crate) fn get_pow_after_commitment(&self, round_index: usize) -> Option<F> {
        self.rounds
            .get(round_index)
            .map(|round| round.pow_witness.clone())
    }
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
    use p3_field::Field;
    use p3_matrix::dense::RowMajorMatrix;
    use p3_merkle_tree::{MerkleTreeError, MerkleTreeMmcs, PrunedProofError};
    use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::*;

    type F = BabyBear;
    type Perm = Poseidon2BabyBear<16>;
    type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
    type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
    type PackedF = <F as Field>::Packing;
    type MyMmcs = MerkleTreeMmcs<PackedF, PackedF, MyHash, MyCompress, 2, 8>;
    type MyCommitment = <MyMmcs as Mmcs<F>>::Commitment;
    type MyMultiProof = <MyMmcs as Mmcs<F>>::MultiProof;

    /// One committed matrix opened at a repeated, unsorted query list.
    ///
    /// Indices `3` and `61` each appear twice and the list is not sorted --
    /// exactly the shape independent-draw STIR sampling produces.
    struct RepeatedIndexFixture {
        mmcs: MyMmcs,
        commitment: MyCommitment,
        opening: SharedProofOpening<F, MyMultiProof>,
        dimensions: [Dimensions; 1],
        indices: [usize; 6],
    }

    impl RepeatedIndexFixture {
        fn new() -> Self {
            let mut rng = SmallRng::seed_from_u64(0);
            let perm = Perm::new_from_rng_128(&mut rng);
            let mmcs = MyMmcs::new(MyHash::new(perm.clone()), MyCompress::new(perm), 0);

            let height = 64usize;
            let width = 4usize;
            let matrix = RowMajorMatrix::new(
                (0..height * width).map(|_| rng.random::<F>()).collect(),
                width,
            );
            let (commitment, prover_data) = mmcs.commit_matrix(matrix);

            let indices = [3usize, 61, 3, 40, 61, 0];
            let opening = SharedProofOpening::open(&mmcs, &indices, &prover_data);

            Self {
                mmcs,
                commitment,
                opening,
                dimensions: [Dimensions { width, height }],
                indices,
            }
        }

        fn verify(&self) -> Result<(), MerkleTreeError> {
            self.opening.verify(
                &self.mmcs,
                &self.commitment,
                &self.dimensions,
                &self.indices,
            )
        }
    }

    #[test]
    fn shared_proof_opening_accepts_repeated_indices() {
        // Independent-draw STIR sampling (`get_challenge_stir_queries`, the
        // caller of this opening's `open`/`verify`) allows the same index
        // to appear more than once in one query list. The pruned-multiproof
        // MMCS format this opening delegates to must accept that, and must
        // return the same row content for every occurrence of a repeated
        // index.
        let fixture = RepeatedIndexFixture::new();

        // Every occurrence of a repeated index opens to the same row.
        assert_eq!(
            fixture.opening.rows[0], fixture.opening.rows[2],
            "both openings of index 3 must agree"
        );
        assert_eq!(
            fixture.opening.rows[1], fixture.opening.rows[4],
            "both openings of index 61 must agree"
        );

        fixture
            .verify()
            .expect("pruned-multiproof verification must accept repeated, unsorted indices");
    }

    #[test]
    fn shared_proof_opening_rejects_disagreeing_repeated_index() {
        // The soundness of allowing duplicates rests on *both* occurrences of
        // a repeated index staying bound to the same leaf. If one occurrence
        // were free, the prover could choose that opening's fold value at
        // will; the round's claim is a batched `sum_i gamma^i s_i`, so a
        // single free `s_i` makes the sumcheck target satisfiable for any
        // polynomial and the round's proximity check vacuous.
        //
        // Deduplication inside the pruned walk is what makes this a real
        // hazard: the two occupancies collapse to one leaf, so a disagreement
        // between them has to be an explicit rejection rather than a hash
        // mismatch. Overwrite the second occurrence of index 3 (slot 2) with
        // the row belonging to index 40 and check it is caught.
        let mut fixture = RepeatedIndexFixture::new();

        fixture.opening.rows[2] = fixture.opening.rows[3].clone();
        assert_ne!(
            fixture.opening.rows[0], fixture.opening.rows[2],
            "the two openings of index 3 must actually disagree for this test to bite"
        );

        // Unique leaves ascending are [0, 3, 40, 61], so index 3 is slot 1.
        match fixture
            .verify()
            .expect_err("disagreeing openings of a repeated index must be rejected")
        {
            MerkleTreeError::MalformedPrunedProof(
                PrunedProofError::InconsistentDuplicateOpenings { slot },
            ) => assert_eq!(slot, 1),
            other => panic!("expected an inconsistent-duplicate rejection, got {other:?}"),
        }
    }
}
