use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::{
    CanObserve, CanSample, CanSampleBits, DuplexChallenger, FieldChallenger, GrindingChallenger,
};
use p3_commit::{MultilinearOpenedValues, MultilinearPcs};
use p3_dft::Radix2DFTSmallBatch;
use p3_field::BasedVectorSpace;
use p3_field::extension::BinomialExtensionField;
use p3_field::{ExtensionField, Field, PrimeCharacteristicRing};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;
use p3_symmetric::{MerkleCap, PaddingFreeSponge, TruncatedPermutation};
use p3_whir::parameters::{
    FoldingFactor, ProtocolParameters, SecurityAssumption, SumcheckStrategy,
};
use p3_whir::pcs::WhirPcs;
use rand::SeedableRng;
use rand::rngs::SmallRng;

use super::*;

type F = BabyBear;
type EF = BinomialExtensionField<F, 4>;
type Dft = Radix2DFTSmallBatch<F>;
type Perm = Poseidon2BabyBear<16>;
type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
type MyChallenger = DuplexChallenger<F, Perm, 16, 8>;
type PackedF = <F as Field>::Packing;
type MyMmcs = MerkleTreeMmcs<PackedF, PackedF, MyHash, MyCompress, 2, 8>;
type MyCommitment = MerkleCap<F, [F; 8]>;
type TestWhirPcs = WhirPcs<EF, F, MyMmcs, MyChallenger, Dft, 8>;
type TestChallengerWhirPcs = WhirPcs<EF, F, MyMmcs, TestChallenger<F>, Dft, 8>;
type TestWhirProof = WhirProof<F, EF, MyMmcs>;
type TestWhirLimbProof = ExtensionLimbPcsProof<EF, TestWhirProof>;
type TestWhirFinalizerProof = WhirWarpFinalizerProof<EF, TestWhirLimbProof>;
type TestWhirRootProof = crate::WarpExternalRootProofBatched<
    F,
    EF,
    Vec<MyCommitment>,
    MyCommitment,
    TestWhirProof,
    TestWhirLimbProof,
    TestWhirFinalizerProof,
>;

#[derive(Clone, Debug)]
struct TestChallenger<F> {
    state: F,
}

impl<F: Field> TestChallenger<F> {
    const fn new(state: F) -> Self {
        Self { state }
    }
}

impl<F: Field> CanObserve<F> for TestChallenger<F> {
    fn observe(&mut self, value: F) {
        self.state += value;
    }
}

impl CanObserve<Vec<EF>> for TestChallenger<F> {
    fn observe(&mut self, values: Vec<EF>) {
        for value in values {
            for &coeff in <EF as BasedVectorSpace<F>>::as_basis_coefficients_slice(&value) {
                <Self as CanObserve<F>>::observe(self, coeff);
            }
        }
    }
}

impl<F: Field> CanObserve<Vec<Vec<F>>> for TestChallenger<F> {
    fn observe(&mut self, valuess: Vec<Vec<F>>) {
        for values in valuess {
            for value in values {
                <Self as CanObserve<F>>::observe(self, value);
            }
        }
    }
}

impl CanObserve<MyCommitment> for TestChallenger<F> {
    fn observe(&mut self, commitment: MyCommitment) {
        for root in commitment.roots() {
            for &value in root {
                <Self as CanObserve<F>>::observe(self, value);
            }
        }
    }
}

impl CanObserve<Vec<MyCommitment>> for TestChallenger<F> {
    fn observe(&mut self, commitments: Vec<MyCommitment>) {
        for commitment in commitments {
            <Self as CanObserve<MyCommitment>>::observe(self, commitment);
        }
    }
}

impl<F: Field> CanSample<F> for TestChallenger<F> {
    fn sample(&mut self) -> F {
        let out = self.state;
        self.state += F::ONE;
        out
    }
}

impl<F: Field> CanSampleBits<usize> for TestChallenger<F> {
    fn sample_bits(&mut self, _bits: usize) -> usize {
        0
    }
}

impl<F> FieldChallenger<F> for TestChallenger<F> where F: Field + Sync {}

impl<F> GrindingChallenger for TestChallenger<F>
where
    F: Field + Sync,
{
    type Witness = F;

    fn grind(&mut self, _bits: usize) -> Self::Witness {
        F::ZERO
    }
}

#[derive(Clone, Debug)]
struct RawEvalPcs<Val> {
    num_vars: usize,
    _ph: PhantomData<Val>,
}

impl<Val> RawEvalPcs<Val> {
    const fn new(num_vars: usize) -> Self {
        Self {
            num_vars,
            _ph: PhantomData,
        }
    }
}

struct RawEvalProverData<Val> {
    evaluations: Vec<Val>,
    opening_points: Vec<Vec<Point<Val>>>,
}

impl<Val, Challenger> MultilinearPcs<Val, Challenger> for RawEvalPcs<Val>
where
    Val: ExtensionField<Val> + Serialize + serde::de::DeserializeOwned,
{
    type Val = Val;
    type Commitment = Vec<Val>;
    type ProverData = RawEvalProverData<Val>;
    type Proof = ();
    type Error = ();

    fn num_vars(&self) -> usize {
        self.num_vars
    }

    fn commit(
        &self,
        evaluations: RowMajorMatrix<Self::Val>,
        opening_points: &[Vec<Point<Val>>],
        _challenger: &mut Challenger,
    ) -> (Self::Commitment, Self::ProverData) {
        assert_eq!(evaluations.width(), 1);
        assert_eq!(evaluations.height(), 1 << self.num_vars);
        (
            evaluations.values.clone(),
            RawEvalProverData {
                evaluations: evaluations.values,
                opening_points: opening_points.to_vec(),
            },
        )
    }

    fn open(
        &self,
        prover_data: Self::ProverData,
        _challenger: &mut Challenger,
    ) -> (MultilinearOpenedValues<Val>, Self::Proof) {
        let poly = Poly::<Val>::new(prover_data.evaluations);
        let values = prover_data
            .opening_points
            .iter()
            .map(|points| {
                points
                    .iter()
                    .map(|point| poly.eval_ext::<Val>(point))
                    .collect()
            })
            .collect();
        (values, ())
    }

    fn verify(
        &self,
        commitment: &Self::Commitment,
        opening_claims: &[Vec<(Point<Val>, Val)>],
        _proof: &Self::Proof,
        _challenger: &mut Challenger,
    ) -> Result<(), Self::Error> {
        let poly = Poly::<Val>::new(commitment.clone());
        for claims_for_poly in opening_claims {
            for (point, value) in claims_for_poly {
                if poly.eval_ext::<Val>(point) != *value {
                    return Err(());
                }
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
struct BaseEvalPcs<Base, Challenge> {
    num_vars: usize,
    _ph: PhantomData<(Base, Challenge)>,
}

impl<Base, Challenge> BaseEvalPcs<Base, Challenge> {
    const fn new(num_vars: usize) -> Self {
        Self {
            num_vars,
            _ph: PhantomData,
        }
    }
}

struct BaseEvalProverData<Base, Challenge> {
    evaluations: Vec<Base>,
    opening_points: Vec<Vec<Point<Challenge>>>,
}

impl<Base, Challenge, Challenger> MultilinearPcs<Challenge, Challenger>
    for BaseEvalPcs<Base, Challenge>
where
    Base: Field + Serialize + serde::de::DeserializeOwned,
    Challenge: ExtensionField<Base> + Serialize + serde::de::DeserializeOwned,
{
    type Val = Base;
    type Commitment = Vec<Base>;
    type ProverData = BaseEvalProverData<Base, Challenge>;
    type Proof = ();
    type Error = ();

    fn num_vars(&self) -> usize {
        self.num_vars
    }

    fn commit(
        &self,
        evaluations: RowMajorMatrix<Self::Val>,
        opening_points: &[Vec<Point<Challenge>>],
        _challenger: &mut Challenger,
    ) -> (Self::Commitment, Self::ProverData) {
        assert_eq!(evaluations.width(), 1);
        assert_eq!(evaluations.height(), 1 << self.num_vars);
        (
            evaluations.values.clone(),
            BaseEvalProverData {
                evaluations: evaluations.values,
                opening_points: opening_points.to_vec(),
            },
        )
    }

    fn open(
        &self,
        prover_data: Self::ProverData,
        _challenger: &mut Challenger,
    ) -> (MultilinearOpenedValues<Challenge>, Self::Proof) {
        let poly = Poly::<Base>::new(prover_data.evaluations);
        let values = prover_data
            .opening_points
            .iter()
            .map(|points| {
                points
                    .iter()
                    .map(|point| poly.eval_base::<Challenge>(point))
                    .collect()
            })
            .collect();
        (values, ())
    }

    fn verify(
        &self,
        commitment: &Self::Commitment,
        opening_claims: &[Vec<(Point<Challenge>, Challenge)>],
        _proof: &Self::Proof,
        _challenger: &mut Challenger,
    ) -> Result<(), Self::Error> {
        let poly = Poly::<Base>::new(commitment.clone());
        for claims_for_poly in opening_claims {
            for (point, value) in claims_for_poly {
                if poly.eval_base::<Challenge>(point) != *value {
                    return Err(());
                }
            }
        }
        Ok(())
    }
}

fn systematic_code() -> ReedSolomonCode<F, Dft> {
    ReedSolomonCode::new_systematic(4, 1, Dft::default())
}

fn whir_pcs(num_vars: usize) -> TestWhirPcs {
    let mut rng = SmallRng::seed_from_u64(0x5748_4952);
    let perm = Perm::new_from_rng_128(&mut rng);
    let merkle_hash = MyHash::new(perm.clone());
    let merkle_compress = MyCompress::new(perm);
    let mmcs = MyMmcs::new(merkle_hash, merkle_compress, 0);
    let whir_params = ProtocolParameters {
        security_level: 32,
        pow_bits: 0,
        rs_domain_initial_reduction_factor: 1,
        folding_factor: FoldingFactor::Constant(2),
        mmcs,
        soundness_type: SecurityAssumption::CapacityBound,
        starting_log_inv_rate: 1,
    };
    TestWhirPcs::new(
        num_vars,
        whir_params,
        Dft::default(),
        SumcheckStrategy::default(),
    )
}

fn whir_pcs_test_challenger(num_vars: usize) -> TestChallengerWhirPcs {
    let mut rng = SmallRng::seed_from_u64(0x5748_4952);
    let perm = Perm::new_from_rng_128(&mut rng);
    let merkle_hash = MyHash::new(perm.clone());
    let merkle_compress = MyCompress::new(perm);
    let mmcs = MyMmcs::new(merkle_hash, merkle_compress, 0);
    let whir_params = ProtocolParameters {
        security_level: 32,
        pow_bits: 0,
        rs_domain_initial_reduction_factor: 1,
        folding_factor: FoldingFactor::Constant(2),
        mmcs,
        soundness_type: SecurityAssumption::CapacityBound,
        starting_log_inv_rate: 1,
    };
    TestChallengerWhirPcs::new(
        num_vars,
        whir_params,
        Dft::default(),
        SumcheckStrategy::default(),
    )
}

fn whir_challenger(seed: u64) -> MyChallenger {
    let mut rng = SmallRng::seed_from_u64(seed);
    MyChallenger::new(Perm::new_from_rng_128(&mut rng))
}

fn accumulator_fixture() -> (
    ReedSolomonCode<F, Dft>,
    AccumulatorInstance<EF, Vec<EF>>,
    AccumulatorWitness<EF, ()>,
) {
    use rand::RngExt;

    let code = systematic_code();
    let mut rng = SmallRng::seed_from_u64(0x57484952);
    let w: Vec<EF> = (0..code.msg_len()).map(|_| rng.random()).collect();
    let f = code.encode_algebra::<EF>(&w);
    let alpha = Point::<EF>::rand(&mut rng, code.log_codeword_len());
    let mu = Poly::<EF>::new(f.clone()).eval_ext::<F>(&alpha);
    let instance = AccumulatorInstance {
        rt: f.clone(),
        alpha: alpha.as_slice().to_vec(),
        mu,
        beta: Vec::new(),
        eta: EF::ZERO,
    };
    let witness = AccumulatorWitness { td: (), f, w };
    (code, instance, witness)
}

fn boolean_pesat_fixture() -> (
    ReedSolomonCode<F, Dft>,
    BooleanPesat<F, EF>,
    AccumulatorInstance<EF, Vec<EF>>,
    AccumulatorWitness<EF, ()>,
) {
    let log_witness = 4;
    let pesat = BooleanPesat::<F, EF>::new(log_witness, b"BooleanPesat/whir".to_vec());
    let code = ReedSolomonCode::new_systematic(log_witness, 1, Dft::default());
    let w = (0..pesat.shape().witness_len())
        .map(|i| EF::from_bool(i % 3 == 0))
        .collect::<Vec<_>>();
    let f = code.encode_algebra::<EF>(&w);
    let beta = (0..pesat.shape().log_constraints)
        .map(|i| EF::from_u64((i as u64) + 2))
        .collect::<Vec<_>>();
    let beta_eq = Poly::<EF>::new_from_point(&beta, EF::ONE);
    let eta = pesat.evaluate_bundled(beta_eq.as_slice(), &w);
    let instance = AccumulatorInstance {
        rt: f.clone(),
        alpha: vec![EF::ZERO; code.log_codeword_len()],
        mu: Poly::<EF>::new(f.clone())
            .eval_ext::<F>(&Point::new(vec![EF::ZERO; code.log_codeword_len()])),
        beta,
        eta,
    };
    let witness = AccumulatorWitness { td: (), f, w };
    (code, pesat, instance, witness)
}

fn whir_native_root_fixture() -> (
    ReedSolomonCode<F, Dft>,
    BooleanPesat<F, EF>,
    MyMmcs,
    crate::WarpParams,
    TestChallenger<F>,
    TestChallengerWhirPcs,
    AccumulatorInstance<EF, Vec<MyCommitment>>,
    TestWhirRootProof,
) {
    use crate::{WarpParams, WarpRootProver};

    let code = systematic_code();
    let pesat = BooleanPesat::<F, EF>::new(code.log_msg_len(), b"BooleanPesat/whir-root".to_vec());
    let mut rng = SmallRng::seed_from_u64(0x5254);
    let perm = Perm::new_from_rng_128(&mut rng);
    let mmcs = MyMmcs::new(MyHash::new(perm.clone()), MyCompress::new(perm), 0);
    let params = WarpParams::new(1, 2);
    let base_challenger = TestChallenger::new(F::from_u64(17));
    let whir_pcs = whir_pcs_test_challenger(code.log_codeword_len());
    let fresh_backend = WhirCodewordBackend::<F, EF, _, TestChallenger<F>, Dft, 8>::new(
        &whir_pcs,
        TestChallenger::new(F::from_u64(23)),
    );
    let acc_backend = WhirLimbAccumulatorBackend::<F, EF, _, TestChallenger<F>, Dft, 8>::new(
        &whir_pcs,
        TestChallenger::new(F::from_u64(29)),
    );
    let finalizer =
        WhirPrecommittedBooleanWarpFinalizerProtocol::<F, EF, _, TestChallenger<F>, Dft>::new(
            &acc_backend,
            &code,
            &pesat,
            TestChallenger::new(F::from_u64(31)),
        );

    let make_witness = |seed: u64| -> Vec<F> {
        (0..code.msg_len())
            .map(|i| F::from_bool(((seed + i as u64) & 1) == 1))
            .collect()
    };
    let step_fresh_committed = vec![
        (0..4)
            .map(|i| {
                let witness = make_witness(100 + i as u64);
                fresh_backend
                    .commit_codeword(code.encode(&witness), witness)
                    .expect("WHIR fresh commit")
            })
            .collect::<Vec<_>>(),
        (0..3)
            .map(|i| {
                let witness = make_witness(200 + i as u64);
                fresh_backend
                    .commit_codeword(code.encode(&witness), witness)
                    .expect("WHIR fresh commit")
            })
            .collect::<Vec<_>>(),
    ];

    let root_prover = WarpRootProver::new(&mmcs, &code, &pesat, params);
    let (claimed_final, root_proof) = root_prover
        .prove_external_linear_chain_accumulator_batched(
            &base_challenger,
            &fresh_backend,
            &acc_backend,
            step_fresh_committed,
            &finalizer,
        )
        .expect("WHIR-native root prove");

    (
        code,
        pesat,
        mmcs,
        params,
        base_challenger,
        whir_pcs,
        claimed_final,
        root_proof,
    )
}

fn verify_whir_native_root(
    code: &ReedSolomonCode<F, Dft>,
    pesat: &BooleanPesat<F, EF>,
    mmcs: &MyMmcs,
    params: crate::WarpParams,
    base_challenger: &TestChallenger<F>,
    whir_pcs: &TestChallengerWhirPcs,
    root_proof: &TestWhirRootProof,
) -> Result<AccumulatorInstance<EF, Vec<MyCommitment>>, crate::WarpError> {
    use crate::WarpRootVerifier;

    let fresh_backend = WhirCodewordBackend::<F, EF, _, TestChallenger<F>, Dft, 8>::new(
        whir_pcs,
        TestChallenger::new(F::from_u64(23)),
    );
    let acc_backend = WhirLimbAccumulatorBackend::<F, EF, _, TestChallenger<F>, Dft, 8>::new(
        whir_pcs,
        TestChallenger::new(F::from_u64(29)),
    );
    let finalizer =
        WhirPrecommittedBooleanWarpFinalizerProtocol::<F, EF, _, TestChallenger<F>, Dft>::new(
            &acc_backend,
            code,
            pesat,
            TestChallenger::new(F::from_u64(31)),
        );
    let root_verifier = WarpRootVerifier::new(mmcs, code, pesat, params);
    root_verifier.verify_external_linear_chain_accumulator_batched(
        base_challenger,
        &fresh_backend,
        &acc_backend,
        root_proof,
        &finalizer,
    )
}

#[test]
fn opening_protocol_proves_and_verifies_accumulator_mle_claim() {
    let (code, instance, witness) = accumulator_fixture();
    let pcs = RawEvalPcs::<EF>::new(code.log_codeword_len());
    let protocol = WhirAccumulatorOpeningProtocol::<F, EF, _, (), _>::new(&pcs, &code, ());

    let proof = protocol.prove(&instance, &witness).unwrap();
    protocol.verify(&instance, &proof).unwrap();
}

#[test]
fn opening_protocol_rejects_unrelated_commitment() {
    let (code, mut instance, witness) = accumulator_fixture();
    let pcs = RawEvalPcs::<EF>::new(code.log_codeword_len());
    let protocol = WhirAccumulatorOpeningProtocol::<F, EF, _, (), _>::new(&pcs, &code, ());
    instance.rt[0] += EF::ONE;

    let err = protocol.prove(&instance, &witness).unwrap_err();
    assert!(matches!(
        err,
        FinalizerError::Decider(DeciderError::MerkleRoot)
    ));
}

#[test]
fn opening_protocol_rejects_bad_claimed_mu() {
    let (code, mut instance, witness) = accumulator_fixture();
    let pcs = RawEvalPcs::<EF>::new(code.log_codeword_len());
    let protocol = WhirAccumulatorOpeningProtocol::<F, EF, _, (), _>::new(&pcs, &code, ());
    instance.mu += EF::ONE;

    let err = protocol.prove(&instance, &witness).unwrap_err();
    assert!(matches!(err, FinalizerError::Decider(DeciderError::MlEval)));
}

#[test]
fn extension_limb_pcs_recomposes_base_field_openings() {
    let (code, instance, witness) = accumulator_fixture();
    let inner = BaseEvalPcs::<F, EF>::new(code.log_codeword_len());
    let pcs = ExtensionLimbPcs::<F, EF, _>::new(&inner);
    let opening_points = [vec![Point::new(instance.alpha.clone())]];

    let (commitment, prover_data) = pcs.commit(
        RowMajorMatrix::new(witness.f.clone(), 1),
        &opening_points,
        &mut TestChallenger::new(F::ZERO),
    );
    let (opened_values, proof) = pcs.open(prover_data, &mut TestChallenger::new(F::ZERO));

    assert_eq!(commitment.len(), <EF as BasedVectorSpace<F>>::DIMENSION);
    assert_eq!(opened_values[0][0], instance.mu);
    pcs.verify(
        &commitment,
        &[vec![(Point::new(instance.alpha), instance.mu)]],
        &proof,
        &mut TestChallenger::new(F::ZERO),
    )
    .unwrap();
}

#[test]
fn extension_limb_pcs_rejects_tampered_limb_opening() {
    let (code, instance, witness) = accumulator_fixture();
    let inner = BaseEvalPcs::<F, EF>::new(code.log_codeword_len());
    let pcs = ExtensionLimbPcs::<F, EF, _>::new(&inner);
    let opening_points = [vec![Point::new(instance.alpha.clone())]];

    let (commitment, prover_data) = pcs.commit(
        RowMajorMatrix::new(witness.f.clone(), 1),
        &opening_points,
        &mut TestChallenger::new(F::ZERO),
    );
    let (_, mut proof) = pcs.open(prover_data, &mut TestChallenger::new(F::ZERO));
    proof.limb_opened_values[0][0][0] += EF::ONE;

    let err = pcs
        .verify(
            &commitment,
            &[vec![(Point::new(instance.alpha), instance.mu)]],
            &proof,
            &mut TestChallenger::new(F::ZERO),
        )
        .unwrap_err();
    assert!(matches!(err, ExtensionLimbPcsError::RecompositionMismatch));
}

#[test]
fn opening_protocol_accepts_extension_limb_pcs_commitment() {
    let (code, raw_instance, witness) = accumulator_fixture();
    let inner = BaseEvalPcs::<F, EF>::new(code.log_codeword_len());
    let pcs = ExtensionLimbPcs::<F, EF, _>::new(&inner);
    let opening_points = [vec![Point::new(raw_instance.alpha.clone())]];
    let (rt, _) = pcs.commit(
        RowMajorMatrix::new(witness.f.clone(), 1),
        &opening_points,
        &mut TestChallenger::new(F::ZERO),
    );
    let instance = AccumulatorInstance {
        rt,
        alpha: raw_instance.alpha,
        mu: raw_instance.mu,
        beta: raw_instance.beta,
        eta: raw_instance.eta,
    };
    let protocol = WhirAccumulatorOpeningProtocol::<F, EF, _, TestChallenger<F>, _>::new(
        &pcs,
        &code,
        TestChallenger::new(F::ONE),
    );

    let proof = protocol.prove(&instance, &witness).unwrap();
    protocol.verify(&instance, &proof).unwrap();
}

#[test]
fn whir_limb_accumulator_backend_opens_committed_accumulator_index() {
    let (code, _raw_instance, witness) = accumulator_fixture();
    let pcs = whir_pcs(code.log_codeword_len());
    let backend = WhirLimbAccumulatorBackend::<F, EF, _, MyChallenger, Dft, 8>::new(
        &pcs,
        whir_challenger(0xacc),
    );

    let (rt, td) = backend.commit(witness.f.clone()).unwrap();
    let index = 7;
    let (value, proof) = backend.open(&td, index).unwrap();
    assert_eq!(value, witness.f[index]);
    backend
        .verify_opening(&rt, code.log_codeword_len(), index, value, &proof)
        .unwrap();

    let err = backend
        .verify_opening(&rt, code.log_codeword_len(), index, value + EF::ONE, &proof)
        .unwrap_err();
    assert!(matches!(err, ExtensionLimbPcsError::RecompositionMismatch));
}

#[test]
fn whir_codeword_backend_opens_fresh_codeword_index() {
    let code = systematic_code();
    let witness = (0..code.msg_len())
        .map(|i| F::from_u64(i as u64 + 3))
        .collect::<Vec<_>>();
    let codeword = code.encode(&witness);
    let pcs = whir_pcs(code.log_codeword_len());
    let backend =
        WhirCodewordBackend::<F, EF, _, MyChallenger, Dft, 8>::new(&pcs, whir_challenger(0x51));
    let committed = backend
        .commit_codeword(codeword.clone(), witness)
        .expect("WHIR fresh commit");

    let index = 9;
    let (value, proof) = backend.open(&committed, index).unwrap();
    assert_eq!(value, codeword[index]);
    backend
        .verify_opening(
            &committed.commitment(),
            code.log_codeword_len(),
            index,
            value,
            &proof,
        )
        .unwrap();

    let err = backend
        .verify_opening(
            &committed.commitment(),
            code.log_codeword_len(),
            index,
            value + F::ONE,
            &proof,
        )
        .unwrap_err();
    assert!(matches!(
        err,
        p3_whir::pcs::verifier::errors::VerifierError::SumcheckFailed { .. }
            | p3_whir::pcs::verifier::errors::VerifierError::StirChallengeFailed { .. }
            | p3_whir::pcs::verifier::errors::VerifierError::MerkleProofInvalid { .. }
    ));
}

#[test]
fn whir_native_root_proof_verifies_steps_and_whir_finalizer() {
    let (code, pesat, mmcs, params, base_challenger, whir_pcs, claimed_final, root_proof) =
        whir_native_root_fixture();
    let verified_final = verify_whir_native_root(
        &code,
        &pesat,
        &mmcs,
        params,
        &base_challenger,
        &whir_pcs,
        &root_proof,
    )
    .expect("WHIR-native root verify");

    assert_eq!(verified_final.mu, claimed_final.mu);
    assert_eq!(verified_final.eta, claimed_final.eta);
    assert_eq!(root_proof.steps.len(), 2);
}

#[test]
fn whir_native_root_rejects_tampered_fresh_commitment() {
    let (code, pesat, mmcs, params, base_challenger, whir_pcs, _, mut root_proof) =
        whir_native_root_fixture();
    let mut roots = root_proof.steps[0].fresh_commitments[0].roots().to_vec();
    roots[0][0] += F::ONE;
    root_proof.steps[0].fresh_commitments[0] = MerkleCap::new(roots);

    verify_whir_native_root(
        &code,
        &pesat,
        &mmcs,
        params,
        &base_challenger,
        &whir_pcs,
        &root_proof,
    )
    .expect_err("tampered fresh WHIR commitment must be rejected");
}

#[test]
fn whir_native_root_rejects_valid_fresh_commitment_substitution() {
    let (code, pesat, mmcs, params, base_challenger, whir_pcs, _, mut root_proof) =
        whir_native_root_fixture();
    let fresh_backend = WhirCodewordBackend::<F, EF, _, TestChallenger<F>, Dft, 8>::new(
        &whir_pcs,
        TestChallenger::new(F::from_u64(23)),
    );
    let witness = (0..code.msg_len())
        .map(|i| F::from_bool(i % 2 == 0))
        .collect::<Vec<_>>();
    let alternate = fresh_backend
        .commit_codeword(code.encode(&witness), witness)
        .expect("alternate WHIR fresh commit");

    root_proof.steps[0].fresh_commitments[0] = alternate.commitment();

    verify_whir_native_root(
        &code,
        &pesat,
        &mmcs,
        params,
        &base_challenger,
        &whir_pcs,
        &root_proof,
    )
    .expect_err("substituting a different valid fresh WHIR commitment must be rejected");
}

#[test]
fn whir_native_root_rejects_dropped_or_reordered_steps() {
    let (code, pesat, mmcs, params, base_challenger, whir_pcs, _, root_proof) =
        whir_native_root_fixture();

    let mut dropped = root_proof.clone();
    dropped.steps.remove(0);
    verify_whir_native_root(
        &code,
        &pesat,
        &mmcs,
        params,
        &base_challenger,
        &whir_pcs,
        &dropped,
    )
    .expect_err("dropping a WHIR-native WARP root step must be rejected");

    let mut reordered = root_proof;
    reordered.steps.swap(0, 1);
    verify_whir_native_root(
        &code,
        &pesat,
        &mmcs,
        params,
        &base_challenger,
        &whir_pcs,
        &reordered,
    )
    .expect_err("reordering WHIR-native WARP root steps must be rejected");
}

#[test]
fn whir_native_root_rejects_tampered_step_instance() {
    let (code, pesat, mmcs, params, base_challenger, whir_pcs, _, mut root_proof) =
        whir_native_root_fixture();
    root_proof.steps[0].instance.mu += EF::ONE;

    verify_whir_native_root(
        &code,
        &pesat,
        &mmcs,
        params,
        &base_challenger,
        &whir_pcs,
        &root_proof,
    )
    .expect_err("tampered WARP step accumulator instance must be rejected");
}

#[test]
fn whir_native_root_rejects_tampered_finalizer_terminal() {
    let (code, pesat, mmcs, params, base_challenger, whir_pcs, _, mut root_proof) =
        whir_native_root_fixture();
    root_proof.final_proof.pesat.terminal_values[0] += EF::ONE;

    verify_whir_native_root(
        &code,
        &pesat,
        &mmcs,
        params,
        &base_challenger,
        &whir_pcs,
        &root_proof,
    )
    .expect_err("tampered WHIR finalizer terminal value must be rejected");
}

#[test]
fn whir_boolean_pesat_protocol_proves_and_verifies_decider_claim() {
    let (code, pesat, instance, witness) = boolean_pesat_fixture();
    let pcs = RawEvalPcs::<EF>::new(code.log_codeword_len());
    let protocol = WhirBooleanPesatProtocol::<F, EF, _, TestChallenger<F>, Dft>::new(
        &pcs,
        &code,
        &pesat,
        TestChallenger::new(F::ONE),
    );

    let proof = protocol.prove(&instance, &witness).unwrap();
    protocol.verify(&instance, &proof).unwrap();
}

#[test]
fn whir_boolean_pesat_protocol_rejects_bad_eta() {
    let (code, pesat, mut instance, witness) = boolean_pesat_fixture();
    let pcs = RawEvalPcs::<EF>::new(code.log_codeword_len());
    let protocol = WhirBooleanPesatProtocol::<F, EF, _, TestChallenger<F>, Dft>::new(
        &pcs,
        &code,
        &pesat,
        TestChallenger::new(F::ONE),
    );

    let proof = protocol.prove(&instance, &witness).unwrap();
    instance.eta += EF::ONE;
    assert!(protocol.verify(&instance, &proof).is_err());
}

#[test]
fn whir_boolean_warp_finalizer_proves_and_verifies_both_decider_claims() {
    let (code, pesat, instance, witness) = boolean_pesat_fixture();
    let pcs = RawEvalPcs::<EF>::new(code.log_codeword_len());
    let protocol = WhirBooleanWarpFinalizerProtocol::<F, EF, _, TestChallenger<F>, Dft>::new(
        &pcs,
        &code,
        &pesat,
        TestChallenger::new(F::ONE),
    );

    let proof = protocol.prove(&instance, &witness).unwrap();
    protocol.verify(&instance, &proof).unwrap();
}

#[test]
fn whir_boolean_warp_finalizer_accepts_extension_limb_pcs_commitment() {
    let (code, pesat, raw_instance, witness) = boolean_pesat_fixture();
    let inner = BaseEvalPcs::<F, EF>::new(code.log_codeword_len());
    let pcs = ExtensionLimbPcs::<F, EF, _>::new(&inner);
    let opening_points = [vec![Point::new(raw_instance.alpha.clone())]];
    let (rt, _) = pcs.commit(
        RowMajorMatrix::new(witness.f.clone(), 1),
        &opening_points,
        &mut TestChallenger::new(F::ZERO),
    );
    let instance = AccumulatorInstance {
        rt,
        alpha: raw_instance.alpha,
        mu: raw_instance.mu,
        beta: raw_instance.beta,
        eta: raw_instance.eta,
    };
    let protocol = WhirBooleanWarpFinalizerProtocol::<F, EF, _, TestChallenger<F>, Dft>::new(
        &pcs,
        &code,
        &pesat,
        TestChallenger::new(F::ONE),
    );

    let proof = protocol.prove(&instance, &witness).unwrap();
    protocol.verify(&instance, &proof).unwrap();
}

#[test]
fn whir_boolean_warp_finalizer_rejects_bad_accumulator_opening_claim() {
    let (code, pesat, mut instance, witness) = boolean_pesat_fixture();
    let pcs = RawEvalPcs::<EF>::new(code.log_codeword_len());
    let protocol = WhirBooleanWarpFinalizerProtocol::<F, EF, _, TestChallenger<F>, Dft>::new(
        &pcs,
        &code,
        &pesat,
        TestChallenger::new(F::ONE),
    );
    let proof = protocol.prove(&instance, &witness).unwrap();
    instance.mu += EF::ONE;

    let err = protocol.verify(&instance, &proof).unwrap_err();
    assert!(matches!(err, FinalizerError::OpeningProof(_)));
}
