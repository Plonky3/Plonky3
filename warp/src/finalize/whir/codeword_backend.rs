//! WHIR backend for fresh base-field WARP input codewords.
//!
//! WARP VACC steps treat upstream segment outputs as externally committed RS
//! codewords. This backend provides the Plonky3-native WHIR commitment and
//! opening implementation for those fresh codewords. The commitment is sampled
//! before WARP derives its step challenges, and the same deferred WHIR prover
//! data is reused for every opening requested by that step.

use super::*;

/// WHIR-backed fresh codeword committed outside a WARP step.
///
/// This is the Plonky3-native analogue of an upstream segment commitment:
/// the codeword is committed once with WHIR, then WARP's VACC step opens it at
/// sampled shift indices via WHIR proofs.
pub struct WhirCommittedCodeword<F, EF, MT, Challenger, const DIGEST_ELEMS: usize>
where
    F: Field,
    EF: ExtensionField<F>,
    MT: p3_commit::Mmcs<F>,
{
    commitment: MT::Commitment,
    prover_data: WhirDeferredProverData<F, EF, MT, DIGEST_ELEMS>,
    challenger: Challenger,
    codeword: Vec<F>,
    witness: Vec<F>,
}

/// WHIR-backed opening backend for fresh base-field WARP codewords.
#[derive(Debug)]
pub struct WhirCodewordBackend<'a, F, EF, MT, Challenger, Dft, const DIGEST_ELEMS: usize>
where
    F: Field,
    EF: ExtensionField<F>,
    MT: p3_commit::Mmcs<F>,
{
    pcs: &'a WhirPcs<EF, F, MT, Challenger, Dft, DIGEST_ELEMS>,
    challenger_seed: Challenger,
    _ph: PhantomData<(F, EF)>,
}

impl<'a, F, EF, MT, Challenger, Dft, const DIGEST_ELEMS: usize>
    WhirCodewordBackend<'a, F, EF, MT, Challenger, Dft, DIGEST_ELEMS>
where
    F: Field,
    EF: ExtensionField<F>,
    MT: p3_commit::Mmcs<F>,
{
    /// Create a WHIR backend for fresh base-field codewords.
    pub const fn new(
        pcs: &'a WhirPcs<EF, F, MT, Challenger, Dft, DIGEST_ELEMS>,
        challenger_seed: Challenger,
    ) -> Self {
        Self {
            pcs,
            challenger_seed,
            _ph: PhantomData,
        }
    }
}

impl<'a, F, EF, MT, Challenger, Dft, const DIGEST_ELEMS: usize>
    WhirCodewordBackend<'a, F, EF, MT, Challenger, Dft, DIGEST_ELEMS>
where
    F: TwoAdicField + Ord,
    EF: ExtensionField<F> + TwoAdicField,
    MT: p3_commit::Mmcs<F>,
    MT::Commitment: Clone + PartialEq,
    Challenger:
        FieldChallenger<F> + GrindingChallenger<Witness = F> + CanObserve<MT::Commitment> + Clone,
    Dft: TwoAdicSubgroupDft<F>,
    WhirDeferredProverData<F, EF, MT, DIGEST_ELEMS>: Clone,
{
    fn codeword_challenger(&self) -> Challenger {
        let mut challenger = self.challenger_seed.clone();
        challenger.observe(F::from_u64(domain::WHIR_CODEWORD_BACKEND));
        challenger
    }

    /// Commit a fresh WARP input codeword with WHIR.
    pub fn commit_codeword(
        &self,
        codeword: Vec<F>,
        witness: Vec<F>,
    ) -> Result<WhirCommittedCodeword<F, EF, MT, Challenger, DIGEST_ELEMS>, &'static str> {
        if codeword.len() != 1 << self.pcs.num_vars() {
            return Err("fresh codeword length does not match WHIR variable count");
        }

        let mut challenger = self.codeword_challenger();
        let (commitment, prover_data) = self
            .pcs
            .commit_deferred(RowMajorMatrix::new(codeword.clone(), 1), &mut challenger);

        Ok(WhirCommittedCodeword {
            commitment,
            prover_data,
            challenger,
            codeword,
            witness,
        })
    }
}

impl<F, EF, MT, Challenger, const DIGEST_ELEMS: usize> ExternalCommittedCodeword<F>
    for WhirCommittedCodeword<F, EF, MT, Challenger, DIGEST_ELEMS>
where
    F: Field,
    EF: ExtensionField<F>,
    MT: p3_commit::Mmcs<F>,
    MT::Commitment: Clone + Serialize + serde::de::DeserializeOwned,
{
    type Commitment = MT::Commitment;

    fn commitment(&self) -> Self::Commitment {
        self.commitment.clone()
    }

    fn codeword(&self) -> &[F] {
        &self.codeword
    }

    fn witness(&self) -> &[F] {
        &self.witness
    }
}

impl<F, EF, MT, Challenger, const DIGEST_ELEMS: usize> ExternalCommitmentObserver<F, Challenger>
    for WhirCommittedCodeword<F, EF, MT, Challenger, DIGEST_ELEMS>
where
    F: Field,
    EF: ExtensionField<F>,
    MT: p3_commit::Mmcs<F>,
    MT::Commitment: Clone + Serialize + serde::de::DeserializeOwned,
    Challenger: FieldChallenger<F> + CanObserve<MT::Commitment>,
{
    fn observe_commitment(&self, challenger: &mut Challenger) {
        challenger.observe(self.commitment.clone());
    }
}

impl<'a, F, EF, MT, Challenger, Dft, const DIGEST_ELEMS: usize>
    ExternalCodewordOpeningProver<F, WhirCommittedCodeword<F, EF, MT, Challenger, DIGEST_ELEMS>>
    for WhirCodewordBackend<'a, F, EF, MT, Challenger, Dft, DIGEST_ELEMS>
where
    F: TwoAdicField + Ord,
    EF: ExtensionField<F> + TwoAdicField + Serialize + serde::de::DeserializeOwned,
    MT: p3_commit::Mmcs<F>,
    MT::Commitment: Clone + PartialEq + Serialize + serde::de::DeserializeOwned,
    WhirProof<F, EF, MT>: Clone + Serialize + serde::de::DeserializeOwned,
    WhirDeferredProverData<F, EF, MT, DIGEST_ELEMS>: Clone,
    Challenger:
        FieldChallenger<F> + GrindingChallenger<Witness = F> + CanObserve<MT::Commitment> + Clone,
    Dft: TwoAdicSubgroupDft<F>,
{
    type Proof = WhirProof<F, EF, MT>;
    type Error = p3_whir::pcs::verifier::errors::VerifierError;

    fn open(
        &self,
        committed: &WhirCommittedCodeword<F, EF, MT, Challenger, DIGEST_ELEMS>,
        index: usize,
    ) -> Result<(F, Self::Proof), Self::Error> {
        let num_vars = self.pcs.num_vars();
        let opening_points = [vec![Point::new(boolean_index_point(index, num_vars))]];
        let mut challenger = committed.challenger.clone();
        let (opened_values, proof) = self.pcs.open_deferred(
            committed.prover_data.clone(),
            &opening_points,
            &mut challenger,
        );
        let opened = opened_values
            .first()
            .and_then(|values| values.first())
            .copied()
            .expect("WHIR fresh opening returned no value");
        debug_assert_eq!(opened, EF::from(committed.codeword[index]));
        Ok((committed.codeword[index], proof))
    }
}

impl<'a, F, EF, MT, Challenger, Dft, const DIGEST_ELEMS: usize>
    ExternalCodewordBatchOpeningProver<
        F,
        WhirCommittedCodeword<F, EF, MT, Challenger, DIGEST_ELEMS>,
    > for WhirCodewordBackend<'a, F, EF, MT, Challenger, Dft, DIGEST_ELEMS>
where
    F: TwoAdicField + Ord,
    EF: ExtensionField<F> + TwoAdicField + Serialize + serde::de::DeserializeOwned,
    MT: p3_commit::Mmcs<F>,
    MT::Commitment: Clone + PartialEq + Serialize + serde::de::DeserializeOwned,
    WhirProof<F, EF, MT>: Clone + Serialize + serde::de::DeserializeOwned,
    WhirDeferredProverData<F, EF, MT, DIGEST_ELEMS>: Clone,
    Challenger:
        FieldChallenger<F> + GrindingChallenger<Witness = F> + CanObserve<MT::Commitment> + Clone,
    Dft: TwoAdicSubgroupDft<F>,
{
    type BatchProof = WhirProof<F, EF, MT>;

    fn open_batch(
        &self,
        committed: &WhirCommittedCodeword<F, EF, MT, Challenger, DIGEST_ELEMS>,
        indices: &[usize],
    ) -> Result<(Vec<F>, Self::BatchProof), Self::Error> {
        let num_vars = self.pcs.num_vars();
        let len = 1usize << num_vars;
        let points = indices
            .iter()
            .map(|&index| {
                assert!(index < len, "fresh WHIR opening index out of range");
                Point::new(boolean_index_point(index, num_vars))
            })
            .collect::<Vec<_>>();
        let opening_points = [points];
        let mut challenger = committed.challenger.clone();
        let (opened_values, proof) = self.pcs.open_deferred(
            committed.prover_data.clone(),
            &opening_points,
            &mut challenger,
        );
        let opened = opened_values
            .first()
            .expect("WHIR fresh opening returned no polynomial values");
        debug_assert_eq!(opened.len(), indices.len());
        let values = indices
            .iter()
            .zip(opened.iter())
            .map(|(&index, &opened)| {
                debug_assert_eq!(opened, EF::from(committed.codeword[index]));
                committed.codeword[index]
            })
            .collect();
        Ok((values, proof))
    }

    fn open_batch_owned(
        &self,
        committed: WhirCommittedCodeword<F, EF, MT, Challenger, DIGEST_ELEMS>,
        indices: &[usize],
    ) -> Result<(Vec<F>, Self::BatchProof), Self::Error> {
        let num_vars = self.pcs.num_vars();
        let len = 1usize << num_vars;
        let points = indices
            .iter()
            .map(|&index| {
                assert!(index < len, "fresh WHIR opening index out of range");
                Point::new(boolean_index_point(index, num_vars))
            })
            .collect::<Vec<_>>();
        let opening_points = [points];
        let mut challenger = committed.challenger;
        let (opened_values, proof) =
            self.pcs
                .open_deferred(committed.prover_data, &opening_points, &mut challenger);
        let opened = opened_values
            .first()
            .expect("WHIR fresh opening returned no polynomial values");
        debug_assert_eq!(opened.len(), indices.len());
        let values = indices
            .iter()
            .zip(opened.iter())
            .map(|(&index, &opened)| {
                debug_assert_eq!(opened, EF::from(committed.codeword[index]));
                committed.codeword[index]
            })
            .collect();
        Ok((values, proof))
    }
}

impl<'a, F, EF, MT, Challenger, Dft, const DIGEST_ELEMS: usize>
    ExternalCodewordOpeningVerifier<F, Challenger>
    for WhirCodewordBackend<'a, F, EF, MT, Challenger, Dft, DIGEST_ELEMS>
where
    F: TwoAdicField + Ord,
    EF: ExtensionField<F> + TwoAdicField,
    MT: p3_commit::Mmcs<F>,
    MT::Commitment: Clone + PartialEq + Serialize + serde::de::DeserializeOwned,
    WhirProof<F, EF, MT>: Clone + Serialize + serde::de::DeserializeOwned,
    WhirDeferredProverData<F, EF, MT, DIGEST_ELEMS>: Clone,
    Challenger:
        FieldChallenger<F> + GrindingChallenger<Witness = F> + CanObserve<MT::Commitment> + Clone,
    Dft: TwoAdicSubgroupDft<F>,
{
    type Commitment = MT::Commitment;
    type Proof = WhirProof<F, EF, MT>;
    type Error = p3_whir::pcs::verifier::errors::VerifierError;

    fn observe_commitment(&self, challenger: &mut Challenger, commitment: &Self::Commitment) {
        challenger.observe(commitment.clone());
    }

    fn verify_opening(
        &self,
        commitment: &Self::Commitment,
        log_codeword_len: usize,
        index: usize,
        value: F,
        proof: &Self::Proof,
    ) -> Result<(), Self::Error> {
        assert_eq!(
            log_codeword_len,
            self.pcs.num_vars(),
            "fresh codeword variable count mismatch"
        );
        let opening_claims = [vec![(
            Point::new(boolean_index_point(index, log_codeword_len)),
            EF::from(value),
        )]];
        let mut challenger = self.codeword_challenger();
        self.pcs
            .verify_deferred(commitment, &opening_claims, proof, &mut challenger)
    }
}

impl<'a, F, EF, MT, Challenger, Dft, const DIGEST_ELEMS: usize>
    ExternalCodewordBatchOpeningVerifier<F, Challenger>
    for WhirCodewordBackend<'a, F, EF, MT, Challenger, Dft, DIGEST_ELEMS>
where
    F: TwoAdicField + Ord,
    EF: ExtensionField<F> + TwoAdicField,
    MT: p3_commit::Mmcs<F>,
    MT::Commitment: Clone + PartialEq + Serialize + serde::de::DeserializeOwned,
    WhirProof<F, EF, MT>: Clone + Serialize + serde::de::DeserializeOwned,
    WhirDeferredProverData<F, EF, MT, DIGEST_ELEMS>: Clone,
    Challenger:
        FieldChallenger<F> + GrindingChallenger<Witness = F> + CanObserve<MT::Commitment> + Clone,
    Dft: TwoAdicSubgroupDft<F>,
{
    type BatchProof = WhirProof<F, EF, MT>;

    fn verify_batch_opening(
        &self,
        commitment: &Self::Commitment,
        log_codeword_len: usize,
        indices: &[usize],
        values: &[F],
        proof: &Self::BatchProof,
    ) -> Result<(), Self::Error> {
        assert_eq!(
            log_codeword_len,
            self.pcs.num_vars(),
            "fresh codeword variable count mismatch"
        );
        assert_eq!(indices.len(), values.len());
        let opening_claims = [indices
            .iter()
            .zip(values.iter())
            .map(|(&index, &value)| {
                (
                    Point::new(boolean_index_point(index, log_codeword_len)),
                    EF::from(value),
                )
            })
            .collect::<Vec<_>>()];
        let mut challenger = self.codeword_challenger();
        self.pcs
            .verify_deferred(commitment, &opening_claims, proof, &mut challenger)
    }
}
