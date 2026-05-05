//! Kernel-style WARP benchmark mirroring `whir/benches/sumcheck.rs`.
//!
//! This is intentionally a Plonky3-native benchmark. It answers a smaller question:
//!
//! ```text
//! How does native WARP accumulation/finalization scale against N independent
//! WHIR sumcheck kernels on synthetic multilinear data?
//! ```
//!
//! The `n_whir_*` lanes measure only the WHIR sumcheck kernel. The
//! `n_whir_commit_*` lanes add WHIR's own PCS commitment phase first, including
//! transpose/pad, DFT-based Reed-Solomon encoding, and Merkle commitment.
//!
//! This file exists so we can inspect the algebraic kernels without pulling in
//! a full zkVM stack.

use std::cell::RefCell;
use std::env;
use std::hint::black_box;
use std::io::{self, Write};
use std::time::{Duration, Instant};
use std::vec::Vec;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use p3_baby_bear::{BabyBear, Poseidon2BabyBear, default_babybear_poseidon2_16};
use p3_challenger::{
    CanObserve, CanSample, CanSampleBits, DuplexChallenger, FieldChallenger, GrindingChallenger,
};
use p3_commit::MultilinearPcs;
use p3_dft::Radix2DFTSmallBatch;
use p3_field::extension::BinomialExtensionField;
use p3_field::{Field, PrimeCharacteristicRing};
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;
use p3_symmetric::{MerkleCap, PaddingFreeSponge, TruncatedPermutation};
use p3_warp::{
    Accumulator, AccumulatorBatchOpeningBackend, AccumulatorCommitmentBackend,
    AccumulatorFinalizer, AccumulatorInstance, AccumulatorPointOpeningBackend, AccumulatorWitness,
    BooleanPesat, ExtensionLimbPcs, ExtensionLimbPcsProof, ExternalCodewordBatchOpeningProver,
    ExternalCodewordBatchOpeningVerifier, ExternalCodewordOpeningProver,
    ExternalCodewordOpeningVerifier, ExternalCommittedCodeword, NativeWarpWhirRootCommitment,
    NativeWarpWhirRootOracleOpeningProof, NativeWarpWhirRootOracleProverData,
    NativeWarpWhirRootProof, NativeWarpWhirRootProofSystem, ReedSolomonCode,
    RootIopBoundCommitment, RootIopBoundTranscript, RootIopError, RootIopOpeningClaim,
    RootIopOpeningPoint, RootIopOpeningProof, RootIopOpeningValue, RootIopOracleField,
    RootIopOracleValues, WarpExternalRootProofBatched, WarpExternalRootStepBatched, WarpParams,
    WarpProver, WarpRootProver, WarpRootVerifier, WhirBooleanWarpFinalizerProtocol,
    WhirPrecommittedBooleanWarpFinalizerProtocol, WhirWarpFinalizerProof, WitnessFinalizer,
};
use p3_whir::constraints::statement::initial::InitialStatement;
use p3_whir::fiat_shamir::domain_separator::DomainSeparator;
use p3_whir::parameters::{
    FoldingFactor, ProtocolParameters, SecurityAssumption, SumcheckStrategy, WhirConfig,
};
use p3_whir::pcs::WhirPcs;
use p3_whir::pcs::committer::writer::CommitmentWriter;
use p3_whir::pcs::proof::WhirProof;
use p3_whir::sumcheck::SumcheckData;
use p3_whir::sumcheck::single::SingleSumcheck;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use serde::{Deserialize, Serialize};

type F = BabyBear;
type EF = BinomialExtensionField<F, 4>;
type Perm = Poseidon2BabyBear<16>;
type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
type PackedF = <F as Field>::Packing;
type MyMmcs = MerkleTreeMmcs<PackedF, PackedF, MyHash, MyCompress, 2, 8>;
type MyChallenger = DuplexChallenger<F, Perm, 16, 8>;
type MyDft = Radix2DFTSmallBatch<F>;
type MyWhirConfig = WhirConfig<EF, F, MyMmcs, MyChallenger>;
type MyWhirProof = WhirProof<F, EF, MyMmcs>;
type MyCommitment = MerkleCap<F, [F; WHIR_DIGEST_ELEMS]>;
type MyWhirPcs = WhirPcs<EF, F, MyMmcs, BenchChallenger, MyDft, WHIR_DIGEST_ELEMS>;
type MyWhirLimbProof = ExtensionLimbPcsProof<EF, MyWhirProof>;
type MyWarpWhirFinalizerProof = WhirWarpFinalizerProof<EF, MyWhirLimbProof>;
type MyRootWhirCommitment = NativeWarpWhirRootCommitment<MyCommitment>;
type MyRootWhirProof = NativeWarpWhirRootProof<F, EF, MyMmcs>;
type MyRootWhirOracleProverData =
    NativeWarpWhirRootOracleProverData<F, EF, MyMmcs, BenchChallenger, WHIR_DIGEST_ELEMS>;
type MyRootWhirFinalizerProof = WhirWarpFinalizerProof<EF, RootIopOpeningProof>;
type MyWarpWhirRootProof = WarpExternalRootProofBatched<
    F,
    EF,
    BenchRootCommitment,
    BenchRootCommitment,
    RootIopOpeningProof,
    RootIopOpeningProof,
    MyRootWhirFinalizerProof,
>;
type NativeAccumulator =
    Accumulator<EF, MyCommitment, p3_warp::protocol::prover::ExtProverData<F, EF, MyMmcs>>;

const LOG_INV_RATE: usize = 1;
const DEFAULT_WARP_FRESH_PER_STEP: usize = 4;
const WHIR_FOLDING_FACTOR: usize = 4;
const WHIR_CONSTRAINTS: usize = 4;
const WHIR_DIGEST_ELEMS: usize = 8;
const DEFAULT_NUM_VARIABLES: &[usize] = &[14, 16, 18];
const DEFAULT_N_VALUES: &[usize] = &[4, 7, 13];

type NativeBenchPesat = BooleanPesat<F, EF>;

struct WarpKernelFixture<P = NativeBenchPesat> {
    mmcs: MyMmcs,
    base_challenger: MyChallenger,
    code: ReedSolomonCode<F, MyDft>,
    pesat: P,
    params: WarpParams,
}

fn make_permutation() -> Perm {
    default_babybear_poseidon2_16()
}

fn make_challenger() -> MyChallenger {
    MyChallenger::new(make_permutation())
}

#[derive(Clone)]
struct BenchChallenger(MyChallenger);

impl BenchChallenger {
    fn new() -> Self {
        Self(make_challenger())
    }
}

impl CanObserve<F> for BenchChallenger {
    fn observe(&mut self, value: F) {
        self.0.observe(value);
    }
}

impl CanObserve<MyCommitment> for BenchChallenger {
    fn observe(&mut self, value: MyCommitment) {
        self.0.observe(value);
    }
}

impl CanObserve<Vec<MyCommitment>> for BenchChallenger {
    fn observe(&mut self, value: Vec<MyCommitment>) {
        for commitment in value {
            self.0.observe(commitment);
        }
    }
}

impl CanSample<F> for BenchChallenger {
    fn sample(&mut self) -> F {
        self.0.sample()
    }
}

impl CanSampleBits<usize> for BenchChallenger {
    fn sample_bits(&mut self, bits: usize) -> usize {
        self.0.sample_bits(bits)
    }
}

impl FieldChallenger<F> for BenchChallenger {}

impl GrindingChallenger for BenchChallenger {
    type Witness = F;

    fn grind(&mut self, bits: usize) -> Self::Witness {
        self.0.grind(bits)
    }
}

fn make_warp_fixture(num_variables: usize) -> WarpKernelFixture {
    let perm = make_permutation();
    let mmcs = MyMmcs::new(MyHash::new(perm.clone()), MyCompress::new(perm.clone()), 0);
    let base_challenger = MyChallenger::new(perm);
    let dft = MyDft::default();
    let pesat = BooleanPesat::new(num_variables, b"BooleanPesat/warp-sumcheck-bench".to_vec());
    let code = ReedSolomonCode::<F, MyDft>::new_systematic(num_variables, LOG_INV_RATE, dft);
    let params = WarpParams::new(1, 2);
    WarpKernelFixture {
        mmcs,
        base_challenger,
        code,
        pesat,
        params,
    }
}

fn warp_fresh_per_step() -> usize {
    let arity = parse_usize_env("P3_WARP_ARITY", DEFAULT_WARP_FRESH_PER_STEP);
    assert!(
        arity >= 2 && arity.is_power_of_two(),
        "P3_WARP_ARITY must be a power of two >= 2"
    );
    arity
}

fn make_whir_protocol_params(mmcs: &MyMmcs) -> ProtocolParameters<MyMmcs> {
    ProtocolParameters {
        security_level: 32,
        pow_bits: 0,
        rs_domain_initial_reduction_factor: 1,
        folding_factor: FoldingFactor::Constant(WHIR_FOLDING_FACTOR),
        mmcs: mmcs.clone(),
        soundness_type: SecurityAssumption::JohnsonBound,
        starting_log_inv_rate: LOG_INV_RATE,
    }
}

fn make_whir_config(mmcs: &MyMmcs, num_variables: usize) -> MyWhirConfig {
    WhirConfig::new(num_variables, make_whir_protocol_params(mmcs))
}

fn make_whir_domain_separator(config: &MyWhirConfig) -> DomainSeparator<EF, F> {
    let mut ds = DomainSeparator::new(Vec::new());
    ds.commit_statement::<MyMmcs, MyChallenger, WHIR_DIGEST_ELEMS>(config);
    ds.add_whir_proof::<MyMmcs, MyChallenger, WHIR_DIGEST_ELEMS>(config);
    ds
}

fn make_whir_statement(
    num_variables: usize,
    folding_factor: usize,
    num_constraints: usize,
    mode: SumcheckStrategy,
    seed: u64,
) -> InitialStatement<F, EF> {
    let mut rng = SmallRng::seed_from_u64(seed);
    let poly = Poly::new((0..1 << num_variables).map(|_| rng.random()).collect());
    let mut stmt = InitialStatement::<F, EF>::new(poly, folding_factor, mode);
    for _ in 0..num_constraints {
        let point = Point::<EF>::rand(&mut rng, num_variables);
        let _ = stmt.evaluate(&point);
    }
    stmt
}

fn make_whir_statements(
    num_variables: usize,
    n: usize,
    mode: SumcheckStrategy,
) -> Vec<InitialStatement<F, EF>> {
    (0..n)
        .map(|i| {
            make_whir_statement(
                num_variables,
                WHIR_FOLDING_FACTOR,
                WHIR_CONSTRAINTS,
                mode,
                0x5750_0000 ^ ((num_variables as u64) << 16) ^ i as u64,
            )
        })
        .collect()
}

fn prove_n_whir_sumchecks(statements: &[InitialStatement<F, EF>]) {
    for statement in statements {
        let mut data = SumcheckData::default();
        let mut challenger = make_challenger();
        black_box(SingleSumcheck::new(
            &mut data,
            &mut challenger,
            WHIR_FOLDING_FACTOR,
            0,
            statement,
        ));
    }
}

fn prove_n_whir_commit_sumchecks(
    config: &MyWhirConfig,
    protocol_params: &ProtocolParameters<MyMmcs>,
    statements: &[InitialStatement<F, EF>],
) {
    let dft = MyDft::default();
    for statement_template in statements {
        let mut statement = statement_template.clone();
        let mut proof =
            MyWhirProof::from_protocol_parameters(protocol_params, config.num_variables);
        let mut challenger = make_challenger();
        make_whir_domain_separator(config).observe_domain_separator(&mut challenger);

        let merkle_data = CommitmentWriter::new(config)
            .commit(&dft, &mut proof, &mut challenger, &mut statement)
            .expect("WHIR commitment phase");
        black_box(&proof.initial_commitment);
        black_box(&proof.initial_ood_answers);
        black_box(&merkle_data);

        let mut data = SumcheckData::default();
        black_box(SingleSumcheck::new(
            &mut data,
            &mut challenger,
            WHIR_FOLDING_FACTOR,
            0,
            &statement,
        ));
    }
}

fn make_boolean_witness(seed: u64, num_variables: usize) -> Vec<F> {
    let mut rng = SmallRng::seed_from_u64(seed);
    (0..1 << num_variables)
        .map(|_| F::from_u64(rng.random::<u64>() & 1))
        .collect()
}

fn make_boolean_witnesses(num_variables: usize, n: usize) -> Vec<Vec<F>> {
    (0..n)
        .map(|i| {
            make_boolean_witness(
                0xAACC_0000 ^ ((num_variables as u64) << 16) ^ i as u64,
                num_variables,
            )
        })
        .collect()
}

fn step_plan(n: usize, fresh_per_step: usize) -> Vec<usize> {
    assert!(n >= fresh_per_step, "need at least one WARP step");
    assert!(
        (n - fresh_per_step) % (fresh_per_step - 1) == 0,
        "n must be ell + k * (ell - 1)"
    );
    let mut plan = vec![fresh_per_step];
    let mut remaining = n - fresh_per_step;
    while remaining > 0 {
        plan.push(fresh_per_step - 1);
        remaining -= fresh_per_step - 1;
    }
    plan
}

fn step_witness_groups(witnesses: &[Vec<F>], fresh_per_step: usize) -> Vec<Vec<Vec<F>>> {
    let mut groups = Vec::new();
    let mut start = 0;
    for fresh_count in step_plan(witnesses.len(), fresh_per_step) {
        let end = start + fresh_count;
        groups.push(witnesses[start..end].to_vec());
        start = end;
    }
    groups
}

fn prove_warp_native(fixture: &WarpKernelFixture, witnesses: &[Vec<F>]) {
    let root_prover =
        WarpRootProver::new(&fixture.mmcs, &fixture.code, &fixture.pesat, fixture.params);
    let finalizer = WitnessFinalizer::new(&fixture.mmcs, &fixture.code, &fixture.pesat);
    let step_witnesses = step_witness_groups(witnesses, warp_fresh_per_step());
    black_box(
        root_prover
            .prove_linear_chain(&fixture.base_challenger, &step_witnesses, &finalizer)
            .expect("native WARP prove"),
    );
}

fn make_whir_pcs(fixture: &WarpKernelFixture) -> MyWhirPcs {
    make_whir_pcs_for_num_vars(&fixture.mmcs, fixture.code.log_codeword_len())
}

fn make_whir_pcs_for_num_vars(mmcs: &MyMmcs, num_variables: usize) -> MyWhirPcs {
    WhirPcs::new(
        num_variables,
        make_whir_protocol_params(mmcs),
        MyDft::default(),
        SumcheckStrategy::Svo,
    )
}

#[allow(dead_code)]
struct WhirFullBundle {
    commitments: Vec<MyCommitment>,
    claims: Vec<Vec<(Point<EF>, EF)>>,
    proofs: Vec<MyWhirProof>,
}

#[allow(dead_code)]
fn make_whir_opening_points(num_variables: usize, proof_index: usize) -> Vec<Point<EF>> {
    let mut rng =
        SmallRng::seed_from_u64(0x5750_4655 ^ ((num_variables as u64) << 16) ^ proof_index as u64);
    (0..WHIR_CONSTRAINTS)
        .map(|_| Point::expand_from_univariate(rng.random(), num_variables))
        .collect()
}

#[allow(dead_code)]
fn build_n_whir_full_pcs(fixture: &WarpKernelFixture, witnesses: &[Vec<F>]) -> WhirFullBundle {
    let num_variables = fixture.code.log_msg_len();
    let pcs = make_whir_pcs_for_num_vars(&fixture.mmcs, num_variables);
    let mut commitments = Vec::with_capacity(witnesses.len());
    let mut claims = Vec::with_capacity(witnesses.len());
    let mut proofs = Vec::with_capacity(witnesses.len());

    for (i, witness) in witnesses.iter().enumerate() {
        let opening_points = make_whir_opening_points(num_variables, i);
        let mut challenger = BenchChallenger::new();
        let (commitment, prover_data) = pcs.commit(
            RowMajorMatrix::new(witness.clone(), 1),
            core::slice::from_ref(&opening_points),
            &mut challenger,
        );
        let (opened_values, proof) = pcs.open(prover_data, &mut challenger);
        let proof_claims = opening_points
            .into_iter()
            .zip(opened_values[0].iter().copied())
            .collect::<Vec<_>>();
        commitments.push(commitment);
        claims.push(proof_claims);
        proofs.push(proof);
    }

    WhirFullBundle {
        commitments,
        claims,
        proofs,
    }
}

#[allow(dead_code)]
fn prove_n_whir_full_pcs(fixture: &WarpKernelFixture, witnesses: &[Vec<F>]) {
    black_box(build_n_whir_full_pcs(fixture, witnesses));
}

#[allow(dead_code)]
fn verify_n_whir_full_pcs_bundle(fixture: &WarpKernelFixture, bundle: &WhirFullBundle) {
    let pcs = make_whir_pcs_for_num_vars(&fixture.mmcs, fixture.code.log_msg_len());
    for ((commitment, claims), proof) in bundle
        .commitments
        .iter()
        .zip(bundle.claims.iter())
        .zip(bundle.proofs.iter())
    {
        pcs.verify(
            commitment,
            core::slice::from_ref(claims),
            proof,
            &mut BenchChallenger::new(),
        )
        .expect("N WHIR full PCS verify");
    }
}

fn build_warp_native_accumulator(
    fixture: &WarpKernelFixture,
    witnesses: &[Vec<F>],
) -> NativeAccumulator {
    let step_prover = WarpProver::new(&fixture.mmcs, &fixture.code, &fixture.pesat, fixture.params);
    let step_witnesses = step_witness_groups(witnesses, warp_fresh_per_step());
    let mut current: Option<NativeAccumulator> = None;

    for fresh in step_witnesses {
        let priors = match current.take() {
            Some(accumulator) => vec![accumulator],
            None => Vec::new(),
        };
        let mut challenger = fixture.base_challenger.clone();
        let (next, proof) = step_prover.prove(&mut challenger, &fresh, &priors);
        black_box(proof);
        current = Some(next);
    }

    current.expect("valid WARP benchmark plan has at least one step")
}

struct WarpWhirFinalizerBundle {
    instance: AccumulatorInstance<EF, Vec<MyCommitment>>,
    proof: MyWarpWhirFinalizerProof,
}

fn whir_limb_commitment_for_finalizer(
    fixture: &WarpKernelFixture,
    pcs: &ExtensionLimbPcs<'_, F, EF, MyWhirPcs>,
    final_acc: &NativeAccumulator,
) -> Vec<MyCommitment> {
    let opening_points = [vec![Point::new(final_acc.instance.alpha.clone())]];
    let (commitment, _) = pcs.commit(
        RowMajorMatrix::new(final_acc.witness.f.clone(), 1),
        &opening_points,
        &mut BenchChallenger::new(),
    );
    assert_eq!(
        pcs.num_vars(),
        fixture.code.log_codeword_len(),
        "WHIR PCS and WARP code must use the same MLE dimension",
    );
    commitment
}

fn build_warp_whir_finalizer_bundle(
    fixture: &WarpKernelFixture,
    witnesses: &[Vec<F>],
) -> WarpWhirFinalizerBundle {
    let final_acc = build_warp_native_accumulator(fixture, witnesses);
    let whir_pcs = make_whir_pcs(fixture);
    let limb_pcs = ExtensionLimbPcs::<F, EF, _>::new(&whir_pcs);
    let rt = whir_limb_commitment_for_finalizer(fixture, &limb_pcs, &final_acc);
    let instance = AccumulatorInstance {
        rt,
        alpha: final_acc.instance.alpha.clone(),
        mu: final_acc.instance.mu,
        beta: final_acc.instance.beta.clone(),
        eta: final_acc.instance.eta,
    };
    let witness = AccumulatorWitness {
        td: (),
        f: final_acc.witness.f.clone(),
        w: final_acc.witness.w.clone(),
    };
    let finalizer = WhirBooleanWarpFinalizerProtocol::<F, EF, _, BenchChallenger, MyDft>::new(
        &limb_pcs,
        &fixture.code,
        &fixture.pesat,
        BenchChallenger::new(),
    );
    let proof = finalizer
        .prove(&instance, &witness)
        .expect("WARP finalizer over WHIR limb PCS");

    WarpWhirFinalizerBundle { instance, proof }
}

fn prove_warp_whir_finalizer(fixture: &WarpKernelFixture, witnesses: &[Vec<F>]) {
    black_box(build_warp_whir_finalizer_bundle(fixture, witnesses));
}

fn verify_warp_whir_finalizer_bundle(
    fixture: &WarpKernelFixture,
    bundle: &WarpWhirFinalizerBundle,
) {
    let whir_pcs = make_whir_pcs(fixture);
    let limb_pcs = ExtensionLimbPcs::<F, EF, _>::new(&whir_pcs);
    let finalizer = WhirBooleanWarpFinalizerProtocol::<F, EF, _, BenchChallenger, MyDft>::new(
        &limb_pcs,
        &fixture.code,
        &fixture.pesat,
        BenchChallenger::new(),
    );
    finalizer
        .verify(&bundle.instance, &bundle.proof)
        .expect("WARP finalizer over WHIR limb PCS verification");
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct BenchRootCommitment(RootIopBoundCommitment<MyRootWhirCommitment>);

impl CanObserve<BenchRootCommitment> for BenchChallenger {
    fn observe(&mut self, value: BenchRootCommitment) {
        observe_bench_root_commitment(self, &value);
    }
}

fn observe_bench_root_commitment(
    challenger: &mut BenchChallenger,
    commitment: &BenchRootCommitment,
) {
    let inner = &commitment.0;
    challenger.observe(F::from_usize(inner.oracle_id));
    challenger.observe(F::from_usize(inner.log_len));
    challenger.observe(F::from_usize(match inner.field {
        RootIopOracleField::Base => 0,
        RootIopOracleField::Extension => 1,
    }));
    match &inner.commitment {
        NativeWarpWhirRootCommitment::Base(root) => {
            challenger.observe(F::ZERO);
            challenger.observe(root.clone());
        }
        NativeWarpWhirRootCommitment::BaseMessage(root) => {
            challenger.observe(F::from_u8(2));
            challenger.observe(root.clone());
        }
        NativeWarpWhirRootCommitment::BaseMessageShared {
            root,
            column,
            width,
        } => {
            challenger.observe(F::from_u8(5));
            challenger.observe(root.clone());
            challenger.observe(F::from_usize(*column));
            challenger.observe(F::from_usize(*width));
        }
        NativeWarpWhirRootCommitment::Extension(roots) => {
            challenger.observe(F::ONE);
            challenger.observe(F::from_usize(roots.len()));
            for root in roots {
                challenger.observe(root.clone());
            }
        }
        NativeWarpWhirRootCommitment::ExtensionNative(root) => {
            challenger.observe(F::from_u8(3));
            challenger.observe(root.clone());
        }
        NativeWarpWhirRootCommitment::ExtensionMessage(root) => {
            challenger.observe(F::from_u8(4));
            challenger.observe(root.clone());
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct BenchRootFreshCodeword {
    commitment: BenchRootCommitment,
    codeword: Vec<F>,
    witness: Vec<F>,
}

impl p3_warp::ExternalCommittedCodeword<F> for BenchRootFreshCodeword {
    type Commitment = BenchRootCommitment;

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

impl p3_warp::ExternalCommitmentObserver<F, BenchChallenger> for BenchRootFreshCodeword {
    fn observe_commitment(&self, challenger: &mut BenchChallenger) {
        observe_bench_root_commitment(challenger, &self.commitment);
    }
}

struct WarpWhirRootBundle {
    instance: AccumulatorInstance<EF, BenchRootCommitment>,
    proof: MyWarpWhirRootProof,
    root_iop_proof: MyRootWhirProof,
    oracle_count: usize,
    claim_count: usize,
    whir_opening_count: usize,
}

#[derive(Clone, Debug, Default)]
struct WarpWhirRootPhaseDurations {
    setup: Duration,
    encode: Duration,
    shared_commit: Duration,
    vacc_total: Duration,
    dacc: Duration,
    root_whir: Duration,
    total: Duration,
    step_times: Vec<Duration>,
}

#[derive(Clone, Debug, Default)]
struct WarpWhirRootVerifyPhaseDurations {
    setup: Duration,
    chain: Duration,
    root_system_setup: Duration,
    root_whir: Duration,
    total: Duration,
    claim_shape: RootClaimShape,
}

#[derive(Clone, Copy, Debug, Default)]
struct RootClaimShape {
    index: usize,
    systematic_index: usize,
    nonsystematic_index: usize,
    mle: usize,
    message_subspace_mle: usize,
    codeword_mle: usize,
}

struct BenchRootIopWhirAccumulatorData {
    commitment: BenchRootCommitment,
    codeword: Vec<EF>,
}

struct BenchRootIopWhirProver<'a> {
    root_system:
        NativeWarpWhirRootProofSystem<'a, F, EF, MyMmcs, BenchChallenger, MyDft, WHIR_DIGEST_ELEMS>,
    log_codeword_len: usize,
    transcript: RefCell<RootIopBoundTranscript<F, EF, MyRootWhirCommitment>>,
    prover_data: RefCell<Vec<MyRootWhirOracleProverData>>,
}

impl<'a> BenchRootIopWhirProver<'a> {
    fn new(
        whir_pcs: &'a MyWhirPcs,
        base_message_pcs: &'a MyWhirPcs,
        code: &'a ReedSolomonCode<F, MyDft>,
    ) -> Self {
        let log_codeword_len = code.log_codeword_len();
        Self {
            root_system: NativeWarpWhirRootProofSystem::new_with_base_message_pcs(
                whir_pcs,
                base_message_pcs,
                code,
                BenchChallenger::new(),
            ),
            log_codeword_len,
            transcript: RefCell::new(RootIopBoundTranscript::default()),
            prover_data: RefCell::new(Vec::new()),
        }
    }

    fn next_oracle_id(&self) -> usize {
        self.transcript.borrow().oracles.len()
    }

    fn push_claim(
        &self,
        oracle_id: usize,
        point: RootIopOpeningPoint<EF>,
        value: RootIopOpeningValue<F, EF>,
    ) -> usize {
        let mut transcript = self.transcript.borrow_mut();
        let claim_id = transcript.claims.len();
        transcript.claims.push(p3_warp::RootIopOpeningClaim {
            claim_id,
            oracle_id,
            point,
            value,
        });
        claim_id
    }

    fn commit_fresh_codewords_shared(
        &self,
        inputs: Vec<(Vec<F>, Vec<F>)>,
    ) -> Result<Vec<BenchRootFreshCodeword>, RootIopError> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }
        for (codeword, witness) in &inputs {
            if codeword.len() != (1 << self.log_codeword_len)
                || witness.len() != (1 << (self.log_codeword_len - LOG_INV_RATE))
            {
                return Err(RootIopError::ShapeMismatch);
            }
        }

        let start_oracle_id = self.next_oracle_id();
        let root_inputs = inputs
            .iter()
            .enumerate()
            .map(|(i, (codeword, witness))| {
                (start_oracle_id + i, codeword.clone(), witness.clone())
            })
            .collect::<Vec<_>>();
        let committed = self
            .root_system
            .commit_shared_base_message_oracles(root_inputs)
            .map_err(|_| RootIopError::ShapeMismatch)?;

        let mut out = Vec::with_capacity(inputs.len());
        for ((codeword, witness), (commitment, prover_data)) in
            inputs.into_iter().zip(committed.into_iter())
        {
            self.transcript.borrow_mut().oracles.push((
                commitment.clone(),
                RootIopOracleValues::Base(codeword.clone()),
            ));
            self.prover_data.borrow_mut().push(prover_data);
            out.push(BenchRootFreshCodeword {
                commitment: BenchRootCommitment(commitment),
                codeword,
                witness,
            });
        }

        Ok(out)
    }

    fn root_proof(&self) -> MyRootWhirProof {
        self.root_system
            .prove(
                &self.transcript.borrow(),
                &self.prover_data.borrow(),
                &mut BenchChallenger::new(),
                0,
            )
            .expect("root IOP WHIR proof")
    }

    fn transcript_shape(&self) -> (usize, usize) {
        let transcript = self.transcript.borrow();
        (transcript.oracles.len(), transcript.claims.len())
    }
}

impl<'a> ExternalCodewordOpeningProver<F, BenchRootFreshCodeword> for BenchRootIopWhirProver<'a> {
    type Proof = RootIopOpeningProof;
    type Error = RootIopError;

    fn open(
        &self,
        committed: &BenchRootFreshCodeword,
        index: usize,
    ) -> Result<(F, Self::Proof), Self::Error> {
        let commitment = committed.commitment();
        let value = *committed
            .codeword()
            .get(index)
            .ok_or(RootIopError::IndexOutOfBounds {
                oracle_id: commitment.0.oracle_id,
                index,
            })?;
        let claim_id = self.push_claim(
            commitment.0.oracle_id,
            RootIopOpeningPoint::Index(index),
            RootIopOpeningValue::Base(value),
        );
        Ok((
            value,
            RootIopOpeningProof {
                claim_ids: vec![claim_id],
            },
        ))
    }
}

impl<'a> ExternalCodewordBatchOpeningProver<F, BenchRootFreshCodeword>
    for BenchRootIopWhirProver<'a>
{
    type BatchProof = RootIopOpeningProof;

    fn open_batch(
        &self,
        committed: &BenchRootFreshCodeword,
        indices: &[usize],
    ) -> Result<(Vec<F>, Self::BatchProof), Self::Error> {
        let commitment = committed.commitment();
        let mut values = Vec::with_capacity(indices.len());
        let mut claim_ids = Vec::with_capacity(indices.len());
        for &index in indices {
            let value = *committed
                .codeword()
                .get(index)
                .ok_or(RootIopError::IndexOutOfBounds {
                    oracle_id: commitment.0.oracle_id,
                    index,
                })?;
            let claim_id = self.push_claim(
                commitment.0.oracle_id,
                RootIopOpeningPoint::Index(index),
                RootIopOpeningValue::Base(value),
            );
            values.push(value);
            claim_ids.push(claim_id);
        }
        Ok((values, RootIopOpeningProof { claim_ids }))
    }
}

impl<'a> AccumulatorCommitmentBackend<F, EF, BenchChallenger> for BenchRootIopWhirProver<'a> {
    type Commitment = BenchRootCommitment;
    type ProverData = BenchRootIopWhirAccumulatorData;
    type Proof = RootIopOpeningProof;
    type Error = RootIopError;

    fn commit(
        &self,
        codeword: Vec<EF>,
    ) -> Result<(Self::Commitment, Self::ProverData), Self::Error> {
        if codeword.len() != (1 << self.log_codeword_len) {
            return Err(RootIopError::ShapeMismatch);
        }
        let oracle_id = self.next_oracle_id();
        let (commitment, prover_data) = self
            .root_system
            .commit_extension_oracle(oracle_id, codeword.clone())
            .map_err(|_| RootIopError::ShapeMismatch)?;
        self.transcript.borrow_mut().oracles.push((
            commitment.clone(),
            RootIopOracleValues::Extension(codeword.clone()),
        ));
        self.prover_data.borrow_mut().push(prover_data);
        let commitment = BenchRootCommitment(commitment);
        Ok((
            commitment.clone(),
            BenchRootIopWhirAccumulatorData {
                commitment,
                codeword,
            },
        ))
    }

    fn open(
        &self,
        prover_data: &Self::ProverData,
        index: usize,
    ) -> Result<(EF, Self::Proof), Self::Error> {
        let value = *prover_data
            .codeword
            .get(index)
            .ok_or(RootIopError::IndexOutOfBounds {
                oracle_id: prover_data.commitment.0.oracle_id,
                index,
            })?;
        let claim_id = self.push_claim(
            prover_data.commitment.0.oracle_id,
            RootIopOpeningPoint::Index(index),
            RootIopOpeningValue::Extension(value),
        );
        Ok((
            value,
            RootIopOpeningProof {
                claim_ids: vec![claim_id],
            },
        ))
    }

    fn observe_commitment(&self, challenger: &mut BenchChallenger, commitment: &Self::Commitment) {
        observe_bench_root_commitment(challenger, commitment);
    }

    fn verify_opening(
        &self,
        _commitment: &Self::Commitment,
        _log_codeword_len: usize,
        _index: usize,
        _value: EF,
        _proof: &Self::Proof,
    ) -> Result<(), Self::Error> {
        Err(RootIopError::ProverUsedAsVerifier)
    }
}

impl<'a> AccumulatorBatchOpeningBackend<F, EF, BenchChallenger> for BenchRootIopWhirProver<'a> {
    type BatchProof = RootIopOpeningProof;

    fn open_batch(
        &self,
        prover_data: &Self::ProverData,
        indices: &[usize],
    ) -> Result<(Vec<EF>, Self::BatchProof), Self::Error> {
        let mut values = Vec::with_capacity(indices.len());
        let mut claim_ids = Vec::with_capacity(indices.len());
        for &index in indices {
            let value = *prover_data
                .codeword
                .get(index)
                .ok_or(RootIopError::IndexOutOfBounds {
                    oracle_id: prover_data.commitment.0.oracle_id,
                    index,
                })?;
            let claim_id = self.push_claim(
                prover_data.commitment.0.oracle_id,
                RootIopOpeningPoint::Index(index),
                RootIopOpeningValue::Extension(value),
            );
            values.push(value);
            claim_ids.push(claim_id);
        }
        Ok((values, RootIopOpeningProof { claim_ids }))
    }

    fn verify_batch_opening(
        &self,
        _commitment: &Self::Commitment,
        _log_codeword_len: usize,
        _indices: &[usize],
        _values: &[EF],
        _proof: &Self::BatchProof,
    ) -> Result<(), Self::Error> {
        Err(RootIopError::ProverUsedAsVerifier)
    }
}

impl<'a> AccumulatorPointOpeningBackend<F, EF, BenchChallenger> for BenchRootIopWhirProver<'a> {
    type PointProof = RootIopOpeningProof;
    type PointError = RootIopError;

    fn num_vars(&self) -> usize {
        self.log_codeword_len
    }

    fn prove_points(
        &self,
        prover_data: &Self::ProverData,
        opening_points: &[Vec<Point<EF>>],
    ) -> Result<(Vec<Vec<EF>>, Self::PointProof), Self::PointError> {
        if opening_points.len() != 1 {
            return Err(RootIopError::OpeningArityMismatch);
        }
        let poly = Poly::<EF>::new(prover_data.codeword.clone());
        let mut values = Vec::with_capacity(opening_points[0].len());
        let mut claim_ids = Vec::with_capacity(opening_points[0].len());
        for point in &opening_points[0] {
            let value = poly.eval_ext::<F>(point);
            let claim_id = self.push_claim(
                prover_data.commitment.0.oracle_id,
                RootIopOpeningPoint::Mle(point.as_slice().to_vec()),
                RootIopOpeningValue::Extension(value),
            );
            values.push(value);
            claim_ids.push(claim_id);
        }
        Ok((vec![values], RootIopOpeningProof { claim_ids }))
    }

    fn verify_points(
        &self,
        _commitment: &Self::Commitment,
        _opening_claims: &[Vec<(Point<EF>, EF)>],
        _proof: &Self::PointProof,
    ) -> Result<(), Self::PointError> {
        Err(RootIopError::ProverUsedAsVerifier)
    }
}

struct BenchRootIopWhirVerifier {
    log_codeword_len: usize,
    expected_commitments: RefCell<Vec<BenchRootCommitment>>,
    expected_claims: RefCell<Vec<RootIopOpeningClaim<F, EF>>>,
}

impl BenchRootIopWhirVerifier {
    fn new(log_codeword_len: usize) -> Self {
        Self {
            log_codeword_len,
            expected_commitments: RefCell::new(Vec::new()),
            expected_claims: RefCell::new(Vec::new()),
        }
    }

    fn record_commitment(&self, commitment: &BenchRootCommitment) {
        let mut expected = self.expected_commitments.borrow_mut();
        if !expected
            .iter()
            .any(|known| known.0.oracle_id == commitment.0.oracle_id)
        {
            expected.push(commitment.clone());
        }
    }

    fn record_expected_claim(
        &self,
        proof_claim_id: usize,
        oracle_id: usize,
        point: RootIopOpeningPoint<EF>,
        value: RootIopOpeningValue<F, EF>,
    ) {
        self.expected_claims.borrow_mut().push(RootIopOpeningClaim {
            claim_id: proof_claim_id,
            oracle_id,
            point,
            value,
        });
    }

    fn expected_claims(&self) -> Vec<RootIopOpeningClaim<F, EF>> {
        let mut claims = self.expected_claims.borrow().clone();
        claims.sort_by_key(|claim| claim.claim_id);
        claims
    }

    fn expected_inner_commitments(&self) -> Vec<RootIopBoundCommitment<MyRootWhirCommitment>> {
        let mut commitments = self
            .expected_commitments
            .borrow()
            .iter()
            .map(|commitment| commitment.0.clone())
            .collect::<Vec<_>>();
        commitments.sort_by_key(|commitment| commitment.oracle_id);
        commitments
    }
}

impl ExternalCodewordOpeningVerifier<F, BenchChallenger> for BenchRootIopWhirVerifier {
    type Commitment = BenchRootCommitment;
    type Proof = RootIopOpeningProof;
    type Error = RootIopError;

    fn observe_commitment(&self, challenger: &mut BenchChallenger, commitment: &Self::Commitment) {
        self.record_commitment(commitment);
        observe_bench_root_commitment(challenger, commitment);
    }

    fn verify_opening(
        &self,
        commitment: &Self::Commitment,
        log_codeword_len: usize,
        index: usize,
        value: F,
        proof: &Self::Proof,
    ) -> Result<(), Self::Error> {
        if commitment.0.field != RootIopOracleField::Base
            || commitment.0.log_len != log_codeword_len
        {
            return Err(RootIopError::ShapeMismatch);
        }
        if proof.claim_ids.len() != 1 {
            return Err(RootIopError::OpeningArityMismatch);
        }
        self.record_expected_claim(
            proof.claim_ids[0],
            commitment.0.oracle_id,
            RootIopOpeningPoint::Index(index),
            RootIopOpeningValue::Base(value),
        );
        Ok(())
    }
}

impl ExternalCodewordBatchOpeningVerifier<F, BenchChallenger> for BenchRootIopWhirVerifier {
    type BatchProof = RootIopOpeningProof;

    fn verify_batch_opening(
        &self,
        commitment: &Self::Commitment,
        log_codeword_len: usize,
        indices: &[usize],
        values: &[F],
        proof: &Self::BatchProof,
    ) -> Result<(), Self::Error> {
        if commitment.0.field != RootIopOracleField::Base
            || commitment.0.log_len != log_codeword_len
            || indices.len() != values.len()
            || indices.len() != proof.claim_ids.len()
        {
            return Err(RootIopError::ShapeMismatch);
        }
        for ((&index, &value), &claim_id) in indices
            .iter()
            .zip(values.iter())
            .zip(proof.claim_ids.iter())
        {
            self.record_expected_claim(
                claim_id,
                commitment.0.oracle_id,
                RootIopOpeningPoint::Index(index),
                RootIopOpeningValue::Base(value),
            );
        }
        Ok(())
    }
}

impl AccumulatorCommitmentBackend<F, EF, BenchChallenger> for BenchRootIopWhirVerifier {
    type Commitment = BenchRootCommitment;
    type ProverData = BenchRootIopWhirAccumulatorData;
    type Proof = RootIopOpeningProof;
    type Error = RootIopError;

    fn commit(
        &self,
        _codeword: Vec<EF>,
    ) -> Result<(Self::Commitment, Self::ProverData), Self::Error> {
        Err(RootIopError::VerifierUsedAsProver)
    }

    fn open(
        &self,
        _prover_data: &Self::ProverData,
        _index: usize,
    ) -> Result<(EF, Self::Proof), Self::Error> {
        Err(RootIopError::VerifierUsedAsProver)
    }

    fn observe_commitment(&self, challenger: &mut BenchChallenger, commitment: &Self::Commitment) {
        self.record_commitment(commitment);
        observe_bench_root_commitment(challenger, commitment);
    }

    fn verify_opening(
        &self,
        commitment: &Self::Commitment,
        log_codeword_len: usize,
        index: usize,
        value: EF,
        proof: &Self::Proof,
    ) -> Result<(), Self::Error> {
        if commitment.0.field != RootIopOracleField::Extension
            || commitment.0.log_len != log_codeword_len
            || proof.claim_ids.len() != 1
        {
            return Err(RootIopError::ShapeMismatch);
        }
        self.record_expected_claim(
            proof.claim_ids[0],
            commitment.0.oracle_id,
            RootIopOpeningPoint::Index(index),
            RootIopOpeningValue::Extension(value),
        );
        Ok(())
    }
}

impl AccumulatorBatchOpeningBackend<F, EF, BenchChallenger> for BenchRootIopWhirVerifier {
    type BatchProof = RootIopOpeningProof;

    fn open_batch(
        &self,
        _prover_data: &Self::ProverData,
        _indices: &[usize],
    ) -> Result<(Vec<EF>, Self::BatchProof), Self::Error> {
        Err(RootIopError::VerifierUsedAsProver)
    }

    fn verify_batch_opening(
        &self,
        commitment: &Self::Commitment,
        log_codeword_len: usize,
        indices: &[usize],
        values: &[EF],
        proof: &Self::BatchProof,
    ) -> Result<(), Self::Error> {
        if commitment.0.field != RootIopOracleField::Extension
            || commitment.0.log_len != log_codeword_len
            || indices.len() != values.len()
            || indices.len() != proof.claim_ids.len()
        {
            return Err(RootIopError::ShapeMismatch);
        }
        for ((&index, &value), &claim_id) in indices
            .iter()
            .zip(values.iter())
            .zip(proof.claim_ids.iter())
        {
            self.record_expected_claim(
                claim_id,
                commitment.0.oracle_id,
                RootIopOpeningPoint::Index(index),
                RootIopOpeningValue::Extension(value),
            );
        }
        Ok(())
    }
}

impl AccumulatorPointOpeningBackend<F, EF, BenchChallenger> for BenchRootIopWhirVerifier {
    type PointProof = RootIopOpeningProof;
    type PointError = RootIopError;

    fn num_vars(&self) -> usize {
        self.log_codeword_len
    }

    fn prove_points(
        &self,
        _prover_data: &Self::ProverData,
        _opening_points: &[Vec<Point<EF>>],
    ) -> Result<(Vec<Vec<EF>>, Self::PointProof), Self::PointError> {
        Err(RootIopError::VerifierUsedAsProver)
    }

    fn verify_points(
        &self,
        commitment: &Self::Commitment,
        opening_claims: &[Vec<(Point<EF>, EF)>],
        proof: &Self::PointProof,
    ) -> Result<(), Self::PointError> {
        if commitment.0.field != RootIopOracleField::Extension
            || opening_claims.len() != 1
            || opening_claims[0].len() != proof.claim_ids.len()
        {
            return Err(RootIopError::OpeningArityMismatch);
        }
        for ((point, value), &claim_id) in opening_claims[0].iter().zip(proof.claim_ids.iter()) {
            self.record_expected_claim(
                claim_id,
                commitment.0.oracle_id,
                RootIopOpeningPoint::Mle(point.as_slice().to_vec()),
                RootIopOpeningValue::Extension(*value),
            );
        }
        Ok(())
    }
}

fn build_warp_whir_root_bundle(
    fixture: &WarpKernelFixture,
    witnesses: &[Vec<F>],
) -> WarpWhirRootBundle {
    build_warp_whir_root_bundle_with_phases(fixture, witnesses).0
}

fn prove_warp_whir_root_chain_with_phase_times(
    fixture: &WarpKernelFixture,
    root_iop_backend: &BenchRootIopWhirProver<'_>,
    finalizer: &WhirPrecommittedBooleanWarpFinalizerProtocol<
        '_,
        F,
        EF,
        BenchRootIopWhirProver<'_>,
        BenchChallenger,
        MyDft,
    >,
    step_fresh_committed: Vec<Vec<BenchRootFreshCodeword>>,
) -> (
    AccumulatorInstance<EF, BenchRootCommitment>,
    MyWarpWhirRootProof,
    Vec<Duration>,
    Duration,
) {
    let step_prover = WarpProver::new(&fixture.mmcs, &fixture.code, &fixture.pesat, fixture.params);
    let mut current = None;
    let mut steps = Vec::with_capacity(step_fresh_committed.len());
    let mut step_times = Vec::with_capacity(step_fresh_committed.len());

    for fresh in step_fresh_committed {
        let step_start = Instant::now();
        let fresh_commitments = fresh.iter().map(|fresh| fresh.commitment()).collect();
        let priors = match current.take() {
            Some(acc) => vec![acc],
            None => Vec::new(),
        };
        let mut challenger = BenchChallenger::new();
        let (next, proof) = step_prover.prove_with_external_committed_accumulator_batched(
            &mut challenger,
            root_iop_backend,
            root_iop_backend,
            fresh,
            &priors,
        );
        step_times.push(step_start.elapsed());
        steps.push(WarpExternalRootStepBatched {
            fresh_commitments,
            instance: next.instance.clone(),
            proof,
        });
        current = Some(next);
    }

    let final_acc = current.expect("non-empty step_fresh_committed checked by caller");
    let dacc_start = Instant::now();
    let final_proof = finalizer
        .finalize(&final_acc.instance, &final_acc.witness)
        .expect("WHIR-backed WARP finalizer");
    let dacc = dacc_start.elapsed();
    (
        final_acc.instance,
        WarpExternalRootProofBatched { steps, final_proof },
        step_times,
        dacc,
    )
}

fn build_warp_whir_root_bundle_with_phases(
    fixture: &WarpKernelFixture,
    witnesses: &[Vec<F>],
) -> (WarpWhirRootBundle, WarpWhirRootPhaseDurations) {
    let total_start = Instant::now();
    let phase_start = Instant::now();
    let whir_pcs = make_whir_pcs(fixture);
    let base_message_pcs = make_whir_pcs_for_num_vars(&fixture.mmcs, fixture.code.log_msg_len());
    let root_iop_backend = BenchRootIopWhirProver::new(&whir_pcs, &base_message_pcs, &fixture.code);
    let finalizer =
        WhirPrecommittedBooleanWarpFinalizerProtocol::<F, EF, _, BenchChallenger, MyDft>::new(
            &root_iop_backend,
            &fixture.code,
            &fixture.pesat,
            BenchChallenger::new(),
        );
    let witness_groups = step_witness_groups(witnesses, warp_fresh_per_step());
    let setup = phase_start.elapsed();

    let phase_start = Instant::now();
    let flat_inputs = witness_groups
        .iter()
        .flat_map(|group| {
            group
                .iter()
                .map(|witness| (fixture.code.encode(witness), witness.clone()))
        })
        .collect::<Vec<_>>();
    let encode = phase_start.elapsed();

    let phase_start = Instant::now();
    let mut shared_fresh = root_iop_backend
        .commit_fresh_codewords_shared(flat_inputs)
        .expect("root IOP shared WHIR fresh commit")
        .into_iter();
    let step_fresh_committed = witness_groups
        .iter()
        .map(|group| {
            (0..group.len())
                .map(|_| shared_fresh.next().expect("shared fresh commitment exists"))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let shared_commit = phase_start.elapsed();

    let phase_start = Instant::now();
    let (instance, proof, step_times, dacc) = prove_warp_whir_root_chain_with_phase_times(
        fixture,
        &root_iop_backend,
        &finalizer,
        step_fresh_committed,
    );
    let vacc_total = phase_start.elapsed().saturating_sub(dacc);

    let phase_start = Instant::now();
    let root_iop_proof = root_iop_backend.root_proof();
    let root_whir = phase_start.elapsed();
    let (oracle_count, claim_count) = root_iop_backend.transcript_shape();
    let whir_opening_count = count_root_whir_openings(&root_iop_proof);
    (
        WarpWhirRootBundle {
            instance,
            proof,
            root_iop_proof,
            oracle_count,
            claim_count,
            whir_opening_count,
        },
        WarpWhirRootPhaseDurations {
            setup,
            encode,
            shared_commit,
            vacc_total,
            dacc,
            root_whir,
            total: total_start.elapsed(),
            step_times,
        },
    )
}

fn count_root_whir_openings(proof: &MyRootWhirProof) -> usize {
    let separate_openings = proof
        .openings
        .iter()
        .map(|opening| match opening {
            NativeWarpWhirRootOracleOpeningProof::Base(_) => 1,
            NativeWarpWhirRootOracleOpeningProof::Extension(proof) => proof.limb_proofs.len(),
            NativeWarpWhirRootOracleOpeningProof::ExtensionNative(_) => 1,
            NativeWarpWhirRootOracleOpeningProof::ExtensionMessage(_) => 1,
        })
        .sum::<usize>();
    separate_openings
        + usize::from(proof.batched_opening.is_some())
        + usize::from(proof.direct_batched_opening.is_some())
}

fn prove_warp_whir_root(fixture: &WarpKernelFixture, witnesses: &[Vec<F>]) {
    black_box(build_warp_whir_root_bundle(fixture, witnesses));
}

fn verify_warp_whir_root_bundle(fixture: &WarpKernelFixture, bundle: &WarpWhirRootBundle) {
    verify_warp_whir_root_bundle_with_phases(fixture, bundle);
}

fn verify_warp_whir_root_bundle_with_phases(
    fixture: &WarpKernelFixture,
    bundle: &WarpWhirRootBundle,
) -> WarpWhirRootVerifyPhaseDurations {
    let total_start = Instant::now();
    let phase_start = Instant::now();
    let whir_pcs = make_whir_pcs(fixture);
    let base_message_pcs = make_whir_pcs_for_num_vars(&fixture.mmcs, fixture.code.log_msg_len());
    let root_iop_verifier = BenchRootIopWhirVerifier::new(fixture.code.log_codeword_len());
    let finalizer =
        WhirPrecommittedBooleanWarpFinalizerProtocol::<F, EF, _, BenchChallenger, MyDft>::new(
            &root_iop_verifier,
            &fixture.code,
            &fixture.pesat,
            BenchChallenger::new(),
        );
    let root_verifier =
        WarpRootVerifier::new(&fixture.mmcs, &fixture.code, &fixture.pesat, fixture.params);
    let setup = phase_start.elapsed();

    let phase_start = Instant::now();
    let verified = root_verifier
        .verify_external_linear_chain_accumulator_batched(
            &BenchChallenger::new(),
            &root_iop_verifier,
            &root_iop_verifier,
            &bundle.proof,
            &finalizer,
        )
        .expect("WHIR-backed WARP root verify");
    assert_eq!(verified, bundle.instance);
    let expected_commitments = root_iop_verifier.expected_inner_commitments();
    let expected_claims = root_iop_verifier.expected_claims();
    let claim_shape = root_claim_shape(&fixture.code, &expected_claims);
    let chain = phase_start.elapsed();

    let phase_start = Instant::now();
    let root_system = NativeWarpWhirRootProofSystem::new_with_base_message_pcs(
        &whir_pcs,
        &base_message_pcs,
        &fixture.code,
        BenchChallenger::new(),
    );
    let root_system_setup = phase_start.elapsed();

    let phase_start = Instant::now();
    root_system
        .verify(
            &expected_commitments,
            &expected_claims,
            &bundle.root_iop_proof,
            &mut BenchChallenger::new(),
            0,
        )
        .expect("WARP root IOP WHIR proof verification");
    let root_whir = phase_start.elapsed();
    WarpWhirRootVerifyPhaseDurations {
        setup,
        chain,
        root_system_setup,
        root_whir,
        total: total_start.elapsed(),
        claim_shape,
    }
}

fn root_claim_shape(
    code: &ReedSolomonCode<F, MyDft>,
    claims: &[RootIopOpeningClaim<F, EF>],
) -> RootClaimShape {
    let stride = 1 << code.log_inv_rate();
    let mut shape = RootClaimShape::default();
    for claim in claims {
        match &claim.point {
            RootIopOpeningPoint::Index(index) => {
                shape.index += 1;
                if index.is_multiple_of(stride) {
                    shape.systematic_index += 1;
                } else {
                    shape.nonsystematic_index += 1;
                }
            }
            RootIopOpeningPoint::Mle(point) => {
                shape.mle += 1;
                if point.len() == code.log_codeword_len()
                    && point[code.log_msg_len()..]
                        .iter()
                        .all(|&coord| coord == EF::ZERO)
                {
                    shape.message_subspace_mle += 1;
                } else {
                    shape.codeword_mle += 1;
                }
            }
        }
    }
    shape
}

fn parse_usize_list_env(name: &str, default: &[usize]) -> Vec<usize> {
    match env::var(name) {
        Ok(raw) => {
            let values: Vec<_> = raw
                .split(',')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(|s| {
                    s.parse::<usize>()
                        .unwrap_or_else(|_| panic!("{name} must be a comma-separated usize list"))
                })
                .collect();
            if values.is_empty() {
                default.to_vec()
            } else {
                values
            }
        }
        Err(_) => default.to_vec(),
    }
}

fn parse_usize_env(name: &str, default: usize) -> usize {
    env::var(name)
        .ok()
        .map(|raw| {
            raw.parse::<usize>()
                .unwrap_or_else(|_| panic!("{name} must be a usize"))
        })
        .unwrap_or(default)
}

fn average_duration(total: Duration, iterations: usize) -> Duration {
    Duration::from_nanos((total.as_nanos() / iterations as u128) as u64)
}

#[derive(Clone, Copy, Debug)]
struct DurationStats {
    min: Duration,
    median: Duration,
    mean: Duration,
    max: Duration,
    stddev: Duration,
}

fn time_average(iterations: usize, mut f: impl FnMut()) -> Duration {
    let start = Instant::now();
    for _ in 0..iterations {
        f();
    }
    average_duration(start.elapsed(), iterations)
}

fn duration_from_nanos(nanos: u128) -> Duration {
    Duration::from_nanos(nanos.min(u64::MAX as u128) as u64)
}

fn duration_stats(samples: &mut [Duration]) -> DurationStats {
    assert!(
        !samples.is_empty(),
        "duration_stats needs at least one sample"
    );
    samples.sort_unstable();
    let len = samples.len();
    let median = if len % 2 == 0 {
        duration_from_nanos((samples[len / 2 - 1].as_nanos() + samples[len / 2].as_nanos()) / 2)
    } else {
        samples[len / 2]
    };
    let sum = samples.iter().map(Duration::as_nanos).sum::<u128>();
    let mean_nanos = sum / len as u128;
    let mean_f64 = mean_nanos as f64;
    let variance = samples
        .iter()
        .map(|sample| {
            let delta = sample.as_nanos() as f64 - mean_f64;
            delta * delta
        })
        .sum::<f64>()
        / len as f64;
    DurationStats {
        min: samples[0],
        median,
        mean: duration_from_nanos(mean_nanos),
        max: samples[len - 1],
        stddev: duration_from_nanos(variance.sqrt() as u128),
    }
}

fn time_paired_stats(
    iterations: usize,
    warmup: usize,
    mut left: impl FnMut(),
    mut right: impl FnMut(),
) -> (DurationStats, DurationStats) {
    for _ in 0..warmup {
        left();
        right();
    }

    let mut left_samples = Vec::with_capacity(iterations);
    let mut right_samples = Vec::with_capacity(iterations);
    for i in 0..iterations {
        if i % 2 == 0 {
            let start = Instant::now();
            left();
            left_samples.push(start.elapsed());
            let start = Instant::now();
            right();
            right_samples.push(start.elapsed());
        } else {
            let start = Instant::now();
            right();
            right_samples.push(start.elapsed());
            let start = Instant::now();
            left();
            left_samples.push(start.elapsed());
        }
    }

    (
        duration_stats(&mut left_samples),
        duration_stats(&mut right_samples),
    )
}

#[allow(dead_code)]
fn print_progress(message: impl core::fmt::Display) {
    eprintln!("{message}");
    let _ = io::stderr().flush();
}

fn format_duration(d: Duration) -> String {
    let secs = d.as_secs_f64();
    if secs >= 1.0 {
        format!("{secs:.3} s")
    } else if secs >= 1e-3 {
        format!("{:.2} ms", secs * 1e3)
    } else if secs >= 1e-6 {
        format!("{:.2} us", secs * 1e6)
    } else {
        format!("{:.0} ns", secs * 1e9)
    }
}

fn format_warp_over_whir(warp: Duration, whir: Duration) -> String {
    let ratio = warp.as_secs_f64() / whir.as_secs_f64();
    let pct = (ratio - 1.0) * 100.0;
    if ratio >= 1.0 {
        format!("{ratio:.1}x (+{pct:.0}%)")
    } else {
        format!("{ratio:.2}x ({pct:.0}%)")
    }
}

fn format_duration_stats(stats: DurationStats) -> String {
    format!(
        "min={} med={} mean={} max={} sd={}",
        format_duration(stats.min),
        format_duration(stats.median),
        format_duration(stats.mean),
        format_duration(stats.max),
        format_duration(stats.stddev),
    )
}

fn print_sumcheck_comparison(num_variable_cases: &[usize], n_values: &[usize]) {
    let iterations = parse_usize_env("P3_WARP_SUMCHECK_COMPARE_ITERS", 1).max(1);
    let arity = warp_fresh_per_step();
    eprintln!();
    eprintln!("=== WARP native vs N WHIR commit+SVO kernel comparison ===");
    eprintln!("    Comparison is warp_native / n_whir_commit_svo for the same k and N.");
    eprintln!(
        "    WARP arity: {arity} fresh inputs in the first step, then {} per chained step.",
        arity - 1
    );
    eprintln!("    Times are single-process wall-clock averages over {iterations} iteration(s).");
    eprintln!(
        "{:<6}{:<8}{:<8}{:<16}{:<16}{:<18}",
        "k", "N", "steps", "whir", "warp", "warp / whir"
    );

    for &num_variables in num_variable_cases {
        for &n in n_values {
            if n < arity || (n - arity) % (arity - 1) != 0 {
                eprintln!(
                    "{:<6}{:<8}{:<8}{:<16}{:<16}{:<18}",
                    num_variables, n, "-", "skip", "skip", "invalid WARP N"
                );
                continue;
            }

            let warp_fixture = make_warp_fixture(num_variables);
            let whir_config = make_whir_config(&warp_fixture.mmcs, num_variables);
            let whir_protocol_params = make_whir_protocol_params(&warp_fixture.mmcs);
            let svo_statements = make_whir_statements(num_variables, n, SumcheckStrategy::Svo);
            let warp_witnesses = make_boolean_witnesses(num_variables, n);
            let steps = step_plan(n, arity).len();

            let whir_time = time_average(iterations, || {
                prove_n_whir_commit_sumchecks(&whir_config, &whir_protocol_params, &svo_statements);
            });
            let warp_time = time_average(iterations, || {
                prove_warp_native(&warp_fixture, &warp_witnesses);
            });

            eprintln!(
                "{:<6}{:<8}{:<8}{:<16}{:<16}{:<18}",
                num_variables,
                n,
                steps,
                format_duration(whir_time),
                format_duration(warp_time),
                format_warp_over_whir(warp_time, whir_time)
            );
        }
    }
    eprintln!();
}

fn print_warp_whir_finalizer_comparison(num_variable_cases: &[usize], n_values: &[usize]) {
    let iterations = parse_usize_env("P3_WARP_WHIR_FINALIZER_COMPARE_ITERS", 1).max(1);
    let arity = warp_fresh_per_step();
    eprintln!();
    eprintln!("=== Native WARP steps + WHIR-limb finalizer comparison ===");
    eprintln!("    WHIR lane: N WHIR commit+SVO kernels from this benchmark.");
    eprintln!(
        "    WARP lane: native WARP accumulation steps, then direct Boolean WhirBooleanWarpFinalizerProtocol over ExtensionLimbPcs<WhirPcs>."
    );
    eprintln!(
        "    Note: this plugs real p3-whir into the finalizer. It is not yet a per-step WHIR accumulator backend because WhirPcs registers opening points at commit time."
    );
    eprintln!(
        "    WARP arity: {arity} fresh inputs in the first step, then {} per chained step.",
        arity - 1
    );
    eprintln!("    Times are single-process wall-clock averages over {iterations} iteration(s).");
    eprintln!(
        "{:<6}{:<8}{:<8}{:<16}{:<16}{:<24}{:<16}",
        "k", "N", "steps", "whir", "warp+whir", "warp / whir", "verify"
    );

    for &num_variables in num_variable_cases {
        for &n in n_values {
            if n < arity || (n - arity) % (arity - 1) != 0 {
                eprintln!(
                    "{:<6}{:<8}{:<8}{:<16}{:<16}{:<24}{:<16}",
                    num_variables, n, "-", "skip", "skip", "-", "invalid WARP N"
                );
                continue;
            }

            let fixture = make_warp_fixture(num_variables);
            let whir_config = make_whir_config(&fixture.mmcs, num_variables);
            let whir_protocol_params = make_whir_protocol_params(&fixture.mmcs);
            let svo_statements = make_whir_statements(num_variables, n, SumcheckStrategy::Svo);
            let warp_witnesses = make_boolean_witnesses(num_variables, n);
            let steps = step_plan(n, arity).len();

            let whir_time = time_average(iterations, || {
                prove_n_whir_commit_sumchecks(&whir_config, &whir_protocol_params, &svo_statements);
            });
            let warp_time = time_average(iterations, || {
                prove_warp_whir_finalizer(&fixture, &warp_witnesses);
            });
            let bundle = build_warp_whir_finalizer_bundle(&fixture, &warp_witnesses);
            let verify_time = time_average(iterations, || {
                verify_warp_whir_finalizer_bundle(&fixture, &bundle);
            });

            eprintln!(
                "{:<6}{:<8}{:<8}{:<16}{:<16}{:<24}{:<16}",
                num_variables,
                n,
                steps,
                format_duration(whir_time),
                format_duration(warp_time),
                format_warp_over_whir(warp_time, whir_time),
                format_duration(verify_time),
            );
        }
    }
    eprintln!();
}

fn print_warp_whir_root_comparison(num_variable_cases: &[usize], n_values: &[usize]) {
    let iterations = parse_usize_env("P3_WARP_WHIR_ROOT_COMPARE_ITERS", 1).max(1);
    let warmup = parse_usize_env("P3_WARP_WHIR_ROOT_COMPARE_WARMUP", 1);
    let print_phases = env::var("P3_WARP_WHIR_ROOT_PHASES").as_deref() == Ok("1");
    let print_stats = iterations > 1 || env::var("P3_WARP_WHIR_ROOT_STATS").as_deref() == Ok("1");
    let arity = warp_fresh_per_step();
    eprintln!();
    eprintln!("=== WHIR-backed WARP root vs N full WHIR PCS comparison ===");
    eprintln!("    WHIR lane: N full WhirPcs commit+open proofs and WhirPcs verifications.");
    eprintln!(
        "    WARP lane: WARP VACC/DACC root recorder + batched-eq section 8.2 reduction + one WHIR batched opening when possible."
    );
    eprintln!(
        "    WARP arity: {arity} fresh inputs in the first step, then {} per chained step.",
        arity - 1
    );
    eprintln!(
        "    Times are paired medians over {iterations} sample(s) after {warmup} warmup iteration(s)."
    );
    eprintln!(
        "    Each sample alternates WHIR/WARP order to reduce thermal, allocator, and scheduler bias."
    );
    if print_phases {
        eprintln!("    Extra WARP phase timings are single setup runs used for verifier input.");
    }
    eprintln!(
        "{:<6}{:<8}{:<8}{:<10}{:<10}{:<12}{:<16}{:<16}{:<24}{:<16}{:<16}{:<24}",
        "k",
        "N",
        "steps",
        "oracles",
        "claims",
        "WHIR opens",
        "whir prove",
        "warp prove",
        "prove Δ",
        "whir verify",
        "warp verify",
        "verify Δ"
    );

    for &num_variables in num_variable_cases {
        for &n in n_values {
            if n < arity || (n - arity) % (arity - 1) != 0 {
                eprintln!(
                    "{:<6}{:<8}{:<8}{:<10}{:<10}{:<12}{:<16}{:<16}{:<24}{:<16}{:<16}{:<24}",
                    num_variables,
                    n,
                    "-",
                    "-",
                    "-",
                    "-",
                    "skip",
                    "skip",
                    "-",
                    "skip",
                    "skip",
                    "invalid WARP N"
                );
                continue;
            }

            let fixture = make_warp_fixture(num_variables);
            let warp_witnesses = make_boolean_witnesses(num_variables, n);
            let steps = step_plan(n, arity).len();

            print_progress(format!(
                "    running k={num_variables}, N={n}: paired WHIR/WARP prover samples..."
            ));
            let (whir_prove_stats, warp_prove_stats) = time_paired_stats(
                iterations,
                warmup,
                || {
                    prove_n_whir_full_pcs(&fixture, &warp_witnesses);
                },
                || {
                    prove_warp_whir_root(&fixture, &warp_witnesses);
                },
            );
            print_progress(format!(
                "    running k={num_variables}, N={n}: verifier setup..."
            ));
            let whir_bundle = build_n_whir_full_pcs(&fixture, &warp_witnesses);
            let (bundle, warp_phases) =
                build_warp_whir_root_bundle_with_phases(&fixture, &warp_witnesses);
            let warp_verify_phases = if print_phases {
                Some(verify_warp_whir_root_bundle_with_phases(&fixture, &bundle))
            } else {
                None
            };
            print_progress(format!(
                "    running k={num_variables}, N={n}: paired WHIR/WARP verifier samples..."
            ));
            let (whir_verify_stats, warp_verify_stats) = time_paired_stats(
                iterations,
                warmup,
                || {
                    verify_n_whir_full_pcs_bundle(&fixture, &whir_bundle);
                },
                || {
                    verify_warp_whir_root_bundle(&fixture, &bundle);
                },
            );

            eprintln!(
                "{:<6}{:<8}{:<8}{:<10}{:<10}{:<12}{:<16}{:<16}{:<24}{:<16}{:<16}{:<24}",
                num_variables,
                n,
                steps,
                bundle.oracle_count,
                bundle.claim_count,
                bundle.whir_opening_count,
                format_duration(whir_prove_stats.median),
                format_duration(warp_prove_stats.median),
                format_warp_over_whir(warp_prove_stats.median, whir_prove_stats.median),
                format_duration(whir_verify_stats.median),
                format_duration(warp_verify_stats.median),
                format_warp_over_whir(warp_verify_stats.median, whir_verify_stats.median),
            );
            if print_stats {
                eprintln!(
                    "      Prove stats:  WHIR [{}] | WARP [{}]",
                    format_duration_stats(whir_prove_stats),
                    format_duration_stats(warp_prove_stats),
                );
                eprintln!(
                    "      Verify stats: WHIR [{}] | WARP [{}]",
                    format_duration_stats(whir_verify_stats),
                    format_duration_stats(warp_verify_stats),
                );
            }
            if print_phases {
                let steps = warp_phases
                    .step_times
                    .iter()
                    .enumerate()
                    .map(|(i, duration)| format!("s{i}={}", format_duration(*duration)))
                    .collect::<Vec<_>>()
                    .join(" ");
                eprintln!(
                    "      WARP phases: setup={} encode={} shared_commit={} vacc_total={} dacc={} root_whir={} total={}",
                    format_duration(warp_phases.setup),
                    format_duration(warp_phases.encode),
                    format_duration(warp_phases.shared_commit),
                    format_duration(warp_phases.vacc_total),
                    format_duration(warp_phases.dacc),
                    format_duration(warp_phases.root_whir),
                    format_duration(warp_phases.total),
                );
                eprintln!("      WARP step phases: {steps}");
                let verify_phases = warp_verify_phases.as_ref().expect("verify phases exist");
                eprintln!(
                    "      WARP verify phases: setup={} chain={} root_system_setup={} root_whir={} total={}",
                    format_duration(verify_phases.setup),
                    format_duration(verify_phases.chain),
                    format_duration(verify_phases.root_system_setup),
                    format_duration(verify_phases.root_whir),
                    format_duration(verify_phases.total),
                );
                eprintln!(
                    "      WARP verify claims: index={} systematic_index={} nonsystematic_index={} mle={} message_subspace_mle={} codeword_mle={}",
                    verify_phases.claim_shape.index,
                    verify_phases.claim_shape.systematic_index,
                    verify_phases.claim_shape.nonsystematic_index,
                    verify_phases.claim_shape.mle,
                    verify_phases.claim_shape.message_subspace_mle,
                    verify_phases.claim_shape.codeword_mle,
                );
            }
        }
    }
    eprintln!();
}

fn print_warp_whir_root_verify_profile(num_variable_cases: &[usize], n_values: &[usize]) {
    let iterations = parse_usize_env("P3_WARP_WHIR_ROOT_VERIFY_ITERS", 64).max(1);
    let warmup = parse_usize_env("P3_WARP_WHIR_ROOT_VERIFY_WARMUP", 2);
    let arity = warp_fresh_per_step();
    eprintln!();
    eprintln!("=== WARP root verifier-only profile harness ===");
    eprintln!(
        "    Builds one WARP root proof, then repeatedly verifies it to make verifier profiling visible."
    );
    eprintln!(
        "    Times are medians over {iterations} verifier sample(s) after {warmup} warmup iteration(s)."
    );
    eprintln!(
        "{:<6}{:<8}{:<8}{:<16}{:<16}{:<16}{:<16}{:<16}{:<16}{:<16}{:<16}",
        "k",
        "N",
        "steps",
        "verify",
        "setup",
        "chain",
        "root_whir",
        "nonsys_idx",
        "mle",
        "msg_mle",
        "cw_mle",
    );

    for &num_variables in num_variable_cases {
        for &n in n_values {
            if n < arity || (n - arity) % (arity - 1) != 0 {
                eprintln!(
                    "{:<6}{:<8}{:<8}{:<16}{:<16}{:<16}{:<16}{:<16}{:<16}{:<16}{:<16}",
                    num_variables, n, "-", "skip", "-", "-", "-", "-", "-", "-", "invalid N",
                );
                continue;
            }

            let fixture = make_warp_fixture(num_variables);
            let warp_witnesses = make_boolean_witnesses(num_variables, n);
            let steps = step_plan(n, arity).len();

            print_progress(format!(
                "    running k={num_variables}, N={n}: building one WARP root proof..."
            ));
            let (bundle, _warp_phases) =
                build_warp_whir_root_bundle_with_phases(&fixture, &warp_witnesses);

            for _ in 0..warmup {
                verify_warp_whir_root_bundle(&fixture, &bundle);
            }

            print_progress(format!(
                "    running k={num_variables}, N={n}: verifier-only samples..."
            ));
            let mut total_samples = Vec::with_capacity(iterations);
            let mut last_phases = WarpWhirRootVerifyPhaseDurations::default();
            for _ in 0..iterations {
                let phases = verify_warp_whir_root_bundle_with_phases(&fixture, &bundle);
                total_samples.push(phases.total);
                last_phases = phases;
            }
            let stats = duration_stats(&mut total_samples);

            eprintln!(
                "{:<6}{:<8}{:<8}{:<16}{:<16}{:<16}{:<16}{:<16}{:<16}{:<16}{:<16}",
                num_variables,
                n,
                steps,
                format_duration(stats.median),
                format_duration(last_phases.setup),
                format_duration(last_phases.chain),
                format_duration(last_phases.root_whir),
                last_phases.claim_shape.nonsystematic_index,
                last_phases.claim_shape.mle,
                last_phases.claim_shape.message_subspace_mle,
                last_phases.claim_shape.codeword_mle,
            );
            eprintln!(
                "      Verify stats: WARP [{}]",
                format_duration_stats(stats)
            );
            eprintln!(
                "      Claim shape: index={} systematic_index={} nonsystematic_index={} mle={} message_subspace_mle={} codeword_mle={}",
                last_phases.claim_shape.index,
                last_phases.claim_shape.systematic_index,
                last_phases.claim_shape.nonsystematic_index,
                last_phases.claim_shape.mle,
                last_phases.claim_shape.message_subspace_mle,
                last_phases.claim_shape.codeword_mle,
            );
        }
    }
    eprintln!();
}

fn bench_sumcheck_like_prover(c: &mut Criterion) {
    let num_variable_cases = parse_usize_list_env("P3_WARP_SUMCHECK_K", DEFAULT_NUM_VARIABLES);
    let n_values = parse_usize_list_env("P3_WARP_SUMCHECK_N", DEFAULT_N_VALUES);

    if env::var("P3_WARP_SUMCHECK_COMPARE").as_deref() == Ok("1") {
        print_sumcheck_comparison(&num_variable_cases, &n_values);
        return;
    }
    if env::var("P3_WARP_WHIR_FINALIZER_COMPARE").as_deref() == Ok("1") {
        print_warp_whir_finalizer_comparison(&num_variable_cases, &n_values);
        return;
    }
    if env::var("P3_WARP_WHIR_ROOT_COMPARE").as_deref() == Ok("1") {
        print_warp_whir_root_comparison(&num_variable_cases, &n_values);
        return;
    }
    if env::var("P3_WARP_WHIR_ROOT_VERIFY_PROFILE").as_deref() == Ok("1") {
        print_warp_whir_root_verify_profile(&num_variable_cases, &n_values);
        return;
    }

    let mut group = c.benchmark_group("warp/sumcheck_like_prover");
    group.sample_size(parse_usize_env("P3_WARP_SUMCHECK_SAMPLES", 10).max(10));
    group.measurement_time(Duration::from_secs(
        parse_usize_env("P3_WARP_SUMCHECK_SECONDS", 5) as u64,
    ));

    let arity = warp_fresh_per_step();
    for &num_variables in &num_variable_cases {
        let label = format!("k{num_variables}");
        let warp_label = format!("l{arity}_{label}");
        for &n in &n_values {
            let valid_warp_n = n >= arity && (n - arity) % (arity - 1) == 0;
            let classic_statements =
                make_whir_statements(num_variables, n, SumcheckStrategy::Classic);
            let svo_statements = make_whir_statements(num_variables, n, SumcheckStrategy::Svo);
            let warp_fixture = make_warp_fixture(num_variables);
            let whir_config = make_whir_config(&warp_fixture.mmcs, num_variables);
            let whir_protocol_params = make_whir_protocol_params(&warp_fixture.mmcs);
            let warp_witnesses = make_boolean_witnesses(num_variables, n);

            group.bench_with_input(
                BenchmarkId::new(format!("n_whir_classic_{label}"), n),
                &n,
                |b, _| b.iter(|| prove_n_whir_sumchecks(&classic_statements)),
            );
            group.bench_with_input(
                BenchmarkId::new(format!("n_whir_svo_{label}"), n),
                &n,
                |b, _| b.iter(|| prove_n_whir_sumchecks(&svo_statements)),
            );
            group.bench_with_input(
                BenchmarkId::new(format!("n_whir_commit_classic_{label}"), n),
                &n,
                |b, _| {
                    b.iter(|| {
                        prove_n_whir_commit_sumchecks(
                            &whir_config,
                            &whir_protocol_params,
                            &classic_statements,
                        )
                    })
                },
            );
            group.bench_with_input(
                BenchmarkId::new(format!("n_whir_commit_svo_{label}"), n),
                &n,
                |b, _| {
                    b.iter(|| {
                        prove_n_whir_commit_sumchecks(
                            &whir_config,
                            &whir_protocol_params,
                            &svo_statements,
                        )
                    })
                },
            );
            if valid_warp_n {
                group.bench_with_input(
                    BenchmarkId::new(format!("warp_native_{warp_label}"), n),
                    &n,
                    |b, _| b.iter(|| prove_warp_native(&warp_fixture, &warp_witnesses)),
                );
            }
            if valid_warp_n && env::var("P3_WARP_WHIR_FINALIZER_BENCH").as_deref() == Ok("1") {
                let warp_whir_bundle =
                    build_warp_whir_finalizer_bundle(&warp_fixture, &warp_witnesses);
                group.bench_with_input(
                    BenchmarkId::new(format!("warp_whir_finalizer_{warp_label}"), n),
                    &n,
                    |b, _| b.iter(|| prove_warp_whir_finalizer(&warp_fixture, &warp_witnesses)),
                );
                group.bench_with_input(
                    BenchmarkId::new(format!("warp_whir_finalizer_verify_{warp_label}"), n),
                    &n,
                    |b, _| {
                        b.iter(|| {
                            verify_warp_whir_finalizer_bundle(&warp_fixture, &warp_whir_bundle)
                        })
                    },
                );
            }
        }
    }

    group.finish();
}

criterion_group!(benches, bench_sumcheck_like_prover);
criterion_main!(benches);
