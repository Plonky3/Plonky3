use super::*;
use p3_baby_bear::BabyBear;
use p3_challenger::{CanObserve, CanSample, CanSampleBits};
use p3_commit::{BatchOpening, BatchOpeningRef, Mmcs};
use p3_field::extension::BinomialExtensionField;
use p3_matrix::{Dimensions, Matrix};
use serde::{Deserialize, Serialize};

type F = BabyBear;
type EF = BinomialExtensionField<F, 4>;

#[derive(Clone, Copy, Debug)]
struct DummyChallenger;

impl CanObserve<F> for DummyChallenger {
    fn observe(&mut self, _value: F) {}
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct ToyCommitment(Vec<F>);

impl CanObserve<ToyCommitment> for DummyChallenger {
    fn observe(&mut self, _value: ToyCommitment) {}
}

#[derive(Clone, Debug)]
struct ToyMmcs;

impl Mmcs<F> for ToyMmcs {
    type ProverData<M> = Vec<M>;
    type Commitment = ToyCommitment;
    type Proof = ();
    type Error = RootIopError;

    fn commit<M: Matrix<F>>(&self, inputs: Vec<M>) -> (Self::Commitment, Self::ProverData<M>) {
        let mut digest = Vec::new();
        for input in &inputs {
            digest.push(F::from_usize(input.height()));
            digest.push(F::from_usize(input.width()));
            for r in 0..input.height() {
                for c in 0..input.width() {
                    digest.push(input.get(r, c).unwrap());
                }
            }
        }
        (ToyCommitment(digest), inputs)
    }

    fn open_batch<M: Matrix<F>>(
        &self,
        index: usize,
        prover_data: &Self::ProverData<M>,
    ) -> BatchOpening<F, Self> {
        let opened_values = prover_data
            .iter()
            .map(|matrix| {
                let row = index.min(matrix.height() - 1);
                (0..matrix.width())
                    .map(|col| matrix.get(row, col).unwrap())
                    .collect()
            })
            .collect();
        BatchOpening::new(opened_values, ())
    }

    fn get_matrices<'a, M: Matrix<F>>(&self, prover_data: &'a Self::ProverData<M>) -> Vec<&'a M> {
        prover_data.iter().collect()
    }

    fn verify_batch(
        &self,
        _commit: &Self::Commitment,
        _dimensions: &[Dimensions],
        _index: usize,
        _batch_opening: BatchOpeningRef<'_, F, Self>,
    ) -> Result<(), Self::Error> {
        Ok(())
    }
}

impl CanSample<F> for DummyChallenger {
    fn sample(&mut self) -> F {
        F::ZERO
    }
}

impl CanSampleBits<usize> for DummyChallenger {
    fn sample_bits(&mut self, _bits: usize) -> usize {
        0
    }
}

impl FieldChallenger<F> for DummyChallenger {}

#[test]
fn records_and_checks_base_and_extension_claims() {
    let prover = RootIopProver::<F, EF>::new(2);
    let fresh = prover
        .commit_fresh_codeword(
            alloc::vec![
                F::from_u64(3),
                F::from_u64(4),
                F::from_u64(5),
                F::from_u64(6)
            ],
            alloc::vec![F::from_u64(7), F::from_u64(8)],
        )
        .unwrap();
    let (_, fresh_proof) = <RootIopProver<F, EF> as ExternalCodewordBatchOpeningProver<
        F,
        RootIopCommittedCodeword<F>,
    >>::open_batch(&prover, &fresh, &[1, 3])
    .unwrap();
    assert_eq!(fresh_proof.claim_ids, alloc::vec![0, 1]);

    let (acc_commitment, acc_data) =
        <RootIopProver<F, EF> as AccumulatorCommitmentBackend<F, EF, DummyChallenger>>::commit(
            &prover,
            alloc::vec![
                EF::from_u64(11),
                EF::from_u64(12),
                EF::from_u64(13),
                EF::from_u64(14),
            ],
        )
        .unwrap();
    let (_, acc_proof) = <RootIopProver<F, EF> as AccumulatorBatchOpeningBackend<
        F,
        EF,
        DummyChallenger,
    >>::open_batch(&prover, &acc_data, &[0, 2])
    .unwrap();
    assert_eq!(acc_commitment.oracle_id, 1);
    assert_eq!(acc_proof.claim_ids, alloc::vec![2, 3]);

    let transcript = prover.transcript();
    transcript.verify_witnessed_claim_values().unwrap();
}

#[test]
fn verifier_expected_claims_match_prover_transcript() {
    let prover = RootIopProver::<F, EF>::new(2);
    let fresh = prover
        .commit_fresh_codeword(
            alloc::vec![
                F::from_u64(1),
                F::from_u64(2),
                F::from_u64(3),
                F::from_u64(4)
            ],
            alloc::vec![F::from_u64(1)],
        )
        .unwrap();
    let (values, proof) = <RootIopProver<F, EF> as ExternalCodewordBatchOpeningProver<
        F,
        RootIopCommittedCodeword<F>,
    >>::open_batch(&prover, &fresh, &[0, 2])
    .unwrap();

    let verifier = RootIopVerifier::<F, EF>::new(2);
    <RootIopVerifier<F, EF> as ExternalCodewordBatchOpeningVerifier<
            F,
            DummyChallenger,
        >>::verify_batch_opening(&verifier, &fresh.commitment(), 2, &[0, 2], &values, &proof)
        .unwrap();

    let transcript = prover.transcript();
    verifier.verify_against_transcript(&transcript).unwrap();
    transcript.verify_witnessed_claim_values().unwrap();

    let proof = WitnessRootIopProof::<F, EF>::prove(&transcript).unwrap();
    WitnessRootIopProof::<F, EF>::verify(&verifier.expected_claims(), &proof).unwrap();
}

#[test]
fn bound_recorder_binds_real_commitments_and_claim_ids() {
    let prover = RootIopBoundProver::<F, EF, ToyMmcs>::new(ToyMmcs, 2);
    let fresh = prover
        .commit_fresh_codeword(
            alloc::vec![
                F::from_u64(1),
                F::from_u64(2),
                F::from_u64(3),
                F::from_u64(4)
            ],
            alloc::vec![F::from_u64(1)],
        )
        .unwrap();
    let (fresh_values, fresh_proof) =
        <RootIopBoundProver<F, EF, ToyMmcs> as ExternalCodewordBatchOpeningProver<
            F,
            RootIopBoundCommittedCodeword<F, ToyCommitment>,
        >>::open_batch(&prover, &fresh, &[1, 3])
        .unwrap();

    let (acc_commitment, acc_data) =
        <RootIopBoundProver<F, EF, ToyMmcs> as AccumulatorCommitmentBackend<
            F,
            EF,
            DummyChallenger,
        >>::commit(
            &prover,
            alloc::vec![
                EF::from_u64(11),
                EF::from_u64(12),
                EF::from_u64(13),
                EF::from_u64(14),
            ],
        )
        .unwrap();
    let (acc_values, acc_proof) =
        <RootIopBoundProver<F, EF, ToyMmcs> as AccumulatorBatchOpeningBackend<
            F,
            EF,
            DummyChallenger,
        >>::open_batch(&prover, &acc_data, &[0, 2])
        .unwrap();

    let verifier = RootIopBoundVerifier::<F, EF, ToyCommitment>::new(2);
    let mut challenger = DummyChallenger;
    <RootIopBoundVerifier<F, EF, ToyCommitment> as ExternalCodewordOpeningVerifier<
        F,
        DummyChallenger,
    >>::observe_commitment(&verifier, &mut challenger, &fresh.commitment());
    <RootIopBoundVerifier<F, EF, ToyCommitment> as ExternalCodewordBatchOpeningVerifier<
        F,
        DummyChallenger,
    >>::verify_batch_opening(
        &verifier,
        &fresh.commitment(),
        2,
        &[1, 3],
        &fresh_values,
        &fresh_proof,
    )
    .unwrap();
    <RootIopBoundVerifier<F, EF, ToyCommitment> as AccumulatorCommitmentBackend<
        F,
        EF,
        DummyChallenger,
    >>::observe_commitment(&verifier, &mut challenger, &acc_commitment);
    <RootIopBoundVerifier<F, EF, ToyCommitment> as AccumulatorBatchOpeningBackend<
        F,
        EF,
        DummyChallenger,
    >>::verify_batch_opening(
        &verifier,
        &acc_commitment,
        2,
        &[0, 2],
        &acc_values,
        &acc_proof,
    )
    .unwrap();

    let transcript = prover.transcript();
    transcript.verify_witnessed_claim_values().unwrap();
    verifier.verify_against_transcript(&transcript).unwrap();
    let proof = WitnessRootIopBoundProof::<F, EF, ToyCommitment>::prove(&transcript).unwrap();
    WitnessRootIopBoundProof::<F, EF, ToyCommitment>::verify(
        &verifier.expected_commitments(),
        &verifier.expected_claims(),
        &proof,
    )
    .unwrap();
    assert_eq!(verifier.expected_commitments().len(), 2);
}

#[test]
fn witnessed_claim_check_rejects_tampered_value() {
    let prover = RootIopProver::<F, EF>::new(1);
    let fresh = prover
        .commit_fresh_codeword(
            alloc::vec![F::from_u64(1), F::from_u64(2)],
            alloc::vec![F::from_u64(1)],
        )
        .unwrap();
    let _ = <RootIopProver<F, EF> as ExternalCodewordOpeningProver<
        F,
        RootIopCommittedCodeword<F>,
    >>::open(&prover, &fresh, 1)
    .unwrap();
    let mut transcript = prover.transcript();
    transcript.claims[0].value = RootIopOpeningValue::Base(F::from_u64(9));
    assert_eq!(
        transcript.verify_witnessed_claim_values(),
        Err(RootIopError::ClaimValueMismatch(0))
    );
}
