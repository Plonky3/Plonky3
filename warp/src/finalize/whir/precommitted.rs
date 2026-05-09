use super::*;

pub trait AccumulatorPointOpeningBackend<F, EF, Challenger>:
    AccumulatorCommitmentBackend<F, EF, Challenger>
where
    F: Field,
    EF: ExtensionField<F>,
    Challenger: FieldChallenger<F>,
{
    /// Proof for arbitrary MLE point openings.
    type PointProof: Clone + Serialize + serde::de::DeserializeOwned;
    /// Verification/proving error from the opening backend.
    type PointError: core::fmt::Debug;

    /// Number of variables in the committed accumulator codeword MLE.
    fn num_vars(&self) -> usize;

    /// Prove openings against an existing accumulator commitment/prover data.
    fn prove_points(
        &self,
        prover_data: &Self::ProverData,
        opening_points: &[Vec<Point<EF>>],
    ) -> Result<(MultilinearOpenedValues<EF>, Self::PointProof), Self::PointError>;

    /// Verify openings against an existing accumulator commitment.
    fn verify_points(
        &self,
        commitment: &Self::Commitment,
        opening_claims: &[Vec<(Point<EF>, EF)>],
        proof: &Self::PointProof,
    ) -> Result<(), Self::PointError>;
}

#[derive(Clone, Debug)]
pub struct PrecommittedAccumulatorPcs<'a, F, EF, Backend, Challenger>
where
    F: Field,
    EF: ExtensionField<F>,
    Challenger: FieldChallenger<F>,
    Backend: AccumulatorPointOpeningBackend<F, EF, Challenger>,
{
    backend: &'a Backend,
    commitment: Option<&'a Backend::Commitment>,
    prover_data: Option<&'a Backend::ProverData>,
    _ph: PhantomData<(F, EF, Challenger)>,
}

impl<'a, F, EF, Backend, Challenger> PrecommittedAccumulatorPcs<'a, F, EF, Backend, Challenger>
where
    F: Field,
    EF: ExtensionField<F>,
    Challenger: FieldChallenger<F>,
    Backend: AccumulatorPointOpeningBackend<F, EF, Challenger>,
{
    pub const fn prover(
        backend: &'a Backend,
        commitment: &'a Backend::Commitment,
        prover_data: &'a Backend::ProverData,
    ) -> Self {
        Self {
            backend,
            commitment: Some(commitment),
            prover_data: Some(prover_data),
            _ph: PhantomData,
        }
    }

    pub const fn verifier(backend: &'a Backend) -> Self {
        Self {
            backend,
            commitment: None,
            prover_data: None,
            _ph: PhantomData,
        }
    }
}

#[derive(Clone, Debug)]
pub struct PrecommittedAccumulatorPcsProverData<EF, Proof> {
    opened_values: MultilinearOpenedValues<EF>,
    proof: Proof,
}

impl<'a, F, EF, Backend, Challenger> MultilinearPcs<EF, Challenger>
    for PrecommittedAccumulatorPcs<'a, F, EF, Backend, Challenger>
where
    F: Field,
    EF: ExtensionField<F>,
    Challenger: FieldChallenger<F>,
    Backend: AccumulatorPointOpeningBackend<F, EF, Challenger>,
{
    type Val = EF;
    type Commitment = Backend::Commitment;
    type ProverData = PrecommittedAccumulatorPcsProverData<EF, Backend::PointProof>;
    type Proof = Backend::PointProof;
    type Error = Backend::PointError;

    fn num_vars(&self) -> usize {
        self.backend.num_vars()
    }

    fn commit(
        &self,
        evaluations: RowMajorMatrix<Self::Val>,
        opening_points: &[Vec<Point<EF>>],
        _challenger: &mut Challenger,
    ) -> (Self::Commitment, Self::ProverData) {
        assert_eq!(
            evaluations.width(),
            1,
            "precommitted accumulator finalizer opens one accumulator polynomial"
        );
        assert_eq!(
            evaluations.height(),
            1 << self.backend.num_vars(),
            "precommitted accumulator length must match backend variable count"
        );
        let commitment = self
            .commitment
            .expect("precommitted accumulator PCS prover missing commitment")
            .clone();
        let prover_data = self
            .prover_data
            .expect("precommitted accumulator PCS prover missing prover data");
        let (opened_values, proof) = self
            .backend
            .prove_points(prover_data, opening_points)
            .unwrap_or_else(|err| panic!("precommitted accumulator opening failed: {err:?}"));
        (
            commitment,
            PrecommittedAccumulatorPcsProverData {
                opened_values,
                proof,
            },
        )
    }

    fn open(
        &self,
        prover_data: Self::ProverData,
        _challenger: &mut Challenger,
    ) -> (MultilinearOpenedValues<EF>, Self::Proof) {
        (prover_data.opened_values, prover_data.proof)
    }

    fn verify(
        &self,
        commitment: &Self::Commitment,
        opening_claims: &[Vec<(Point<EF>, EF)>],
        proof: &Self::Proof,
        _challenger: &mut Challenger,
    ) -> Result<(), Self::Error> {
        self.backend
            .verify_points(commitment, opening_claims, proof)
    }
}
