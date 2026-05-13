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
//! The recursive-WHIR comparison builds one circuit that verifies the native
//! `p3_whir::pcs::WhirProof` objects produced by the WHIR lane, then proves
//! that verifier trace with a WHIR-native table proof. WARP is not part of that
//! recursive circuit; it remains the separate accumulation pipeline being
//! compared against recursive aggregation.
//!
//! Minimal recursive WHIR outer-proof profiler recipe:
//!
//! ```bash
//! CARGO_PROFILE_BENCH_DEBUG=1 \
//! RUSTFLAGS="-C force-frame-pointers=yes" \
//! P3_WHIR_RECURSIVE_COMPARE=1 \
//! P3_WHIR_REQUIRE_FULL_SOUNDNESS=1 \
//! P3_WHIR_RECURSIVE_PHASES=1 \
//! P3_WHIR_NATIVE_PHASES=1 \
//! P3_WHIR_RECURSIVE_COMPARE_ITERS=1 \
//! P3_WHIR_RECURSIVE_COMPARE_WARMUP=0 \
//! P3_WARP_SUMCHECK_K=2 \
//! P3_WARP_SUMCHECK_N=4 \
//! P3_WARP_ARITY=2 \
//! samply record -- cargo bench -p p3-warp --features recursive-parallel --bench sumcheck
//! ```
//!
//! Linux `perf` fallback:
//!
//! ```bash
//! CARGO_PROFILE_BENCH_DEBUG=1 \
//! RUSTFLAGS="-C force-frame-pointers=yes" \
//! P3_WHIR_RECURSIVE_COMPARE=1 \
//! P3_WHIR_REQUIRE_FULL_SOUNDNESS=1 \
//! P3_WHIR_RECURSIVE_PHASES=1 \
//! P3_WHIR_NATIVE_PHASES=1 \
//! P3_WHIR_RECURSIVE_COMPARE_ITERS=1 \
//! P3_WHIR_RECURSIVE_COMPARE_WARMUP=0 \
//! P3_WARP_SUMCHECK_K=2 \
//! P3_WARP_SUMCHECK_N=4 \
//! P3_WARP_ARITY=2 \
//! perf record -g -- cargo bench -p p3-warp --features recursive-parallel --bench sumcheck
//! ```
//!
//! This file exists so we can inspect the algebraic kernels without pulling in
//! a full zkVM stack.

use std::cell::RefCell;
use std::env;
use std::fs::{File, OpenOptions};
use std::hint::black_box;
use std::io::{self, Write};
use std::time::{Duration, Instant};
use std::vec::Vec;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use p3_baby_bear::{BabyBear, Poseidon2BabyBear, default_babybear_poseidon2_16};
use p3_challenger::{
    CanObserve, CanSample, CanSampleBits, DuplexChallenger, FieldChallenger, GrindingChallenger,
};
use p3_circuit::CircuitBuilder;
use p3_circuit::ops::{Poseidon2Config, generate_poseidon2_trace, generate_recompose_trace};
use p3_circuit_prover::{
    WhirNativeCircuitError, WhirNativeCircuitOptions, WhirNativeCircuitProof,
    WhirNativeLocalConstraintKind, prove_whir_native_circuit, verify_whir_native_circuit_proof,
};
use p3_commit::MultilinearPcs;
use p3_dft::Radix2DFTSmallBatch;
use p3_field::extension::BinomialExtensionField;
use p3_field::{BasedVectorSpace, Field, PrimeCharacteristicRing};
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::current_num_threads;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;
use p3_poseidon2_circuit_air::BabyBearD4Width16;
use p3_recursion::CircuitChallenger;
use p3_recursion::pcs::fri::{MerkleCapTargets, RecValMmcs};
use p3_recursion::pcs::set_whir_mmcs_private_data;
use p3_recursion::pcs::whir::{
    WhirProofVerificationInput, WhirProofVerificationTargets, verify_native_whir_proof_circuit,
};
use p3_symmetric::{MerkleCap, PaddingFreeSponge, Permutation, TruncatedPermutation};
use p3_warp::{
    AccumulatorBatchOpeningBackend, AccumulatorCommitmentBackend, AccumulatorFinalizer,
    AccumulatorInstance, AccumulatorPointOpeningBackend, BooleanPesat,
    ExternalCodewordBatchOpeningProver, ExternalCodewordBatchOpeningVerifier,
    ExternalCodewordOpeningProver, ExternalCodewordOpeningVerifier, ExternalCommittedCodeword,
    NativeWarpWhirRootCommitment, NativeWarpWhirRootOracleProverData, NativeWarpWhirRootProof,
    NativeWarpWhirRootProofSystem, ReedSolomonCode, RootIopBoundCommitment, RootIopBoundTranscript,
    RootIopError, RootIopOpeningClaim, RootIopOpeningPoint, RootIopOpeningProof,
    RootIopOpeningValue, RootIopOracleField, RootIopOracleValues, WarpExternalRootProofBatched,
    WarpExternalRootStepBatched, WarpParams, WarpProver, WarpRootVerifier,
    WhirPrecommittedBooleanWarpFinalizerProtocol, WhirWarpFinalizerProof,
};
use p3_whir::constraints::statement::initial::InitialStatement;
use p3_whir::fiat_shamir::domain_separator::DomainSeparator;
use p3_whir::parameters::{
    FoldingFactor, ProtocolParameters, SecurityAssumption, SumcheckStrategy, WhirConfig,
};
use p3_whir::pcs::WhirPcs;
use p3_whir::pcs::committer::writer::CommitmentWriter;
use p3_whir::pcs::proof::{QueryOpening, WhirProof};
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
type MyWhirNativeCircuitProof = WhirNativeCircuitProof<F, EF, MyMmcs>;
type MyCommitment = MerkleCap<F, [F; WHIR_DIGEST_ELEMS]>;
type MyWhirPcs = WhirPcs<EF, F, MyMmcs, BenchChallenger, MyDft, WHIR_DIGEST_ELEMS>;
type RecursiveWhirMmcs = RecValMmcs<F, WHIR_DIGEST_ELEMS, MyHash, MyCompress>;
type RecursiveWhirCommitment = MerkleCapTargets<F, WHIR_DIGEST_ELEMS>;
type RecursiveWhirInput = WhirProofVerificationInput<F, EF, MyMmcs, MyCommitment>;
type RecursiveWhirTargets =
    WhirProofVerificationTargets<F, EF, RecursiveWhirCommitment, RecursiveWhirMmcs>;
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
const LOG_INV_RATE: usize = 1;
const DEFAULT_WARP_FRESH_PER_STEP: usize = 4;
const DEFAULT_WHIR_FOLDING_FACTOR: usize = 4;
const WHIR_CONSTRAINTS: usize = 4;
const WHIR_DIGEST_ELEMS: usize = 8;
const DEFAULT_NUM_VARIABLES: &[usize] = &[14, 16, 18];
const DEFAULT_N_VALUES: &[usize] = &[4, 7, 13];
const DEFAULT_WHIR_NATIVE_OUTER_OPENINGS: usize = 2;

type NativeBenchPesat = BooleanPesat<F, EF>;

fn malicious_test_mode() -> bool {
    env::var("P3_WHIR_RECURSIVE_MALICIOUS_TESTS").as_deref() == Ok("1")
}

struct WarpKernelFixture<P = NativeBenchPesat> {
    mmcs: MyMmcs,
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
    let dft = MyDft::default();
    let pesat = BooleanPesat::new(num_variables, b"BooleanPesat/warp-sumcheck-bench".to_vec());
    let code = ReedSolomonCode::<F, MyDft>::new_systematic(num_variables, LOG_INV_RATE, dft);
    let params = WarpParams::new(1, 2);
    WarpKernelFixture {
        mmcs,
        code,
        pesat,
        params,
    }
}

fn warp_fresh_per_step() -> usize {
    let default = if malicious_test_mode() {
        2
    } else {
        DEFAULT_WARP_FRESH_PER_STEP
    };
    let arity = parse_usize_env("P3_WARP_ARITY", default);
    assert!(
        arity >= 2 && arity.is_power_of_two(),
        "P3_WARP_ARITY must be a power of two >= 2"
    );
    arity
}

fn whir_folding_factor() -> usize {
    let factor = parse_usize_env("P3_WARP_WHIR_FOLDING_FACTOR", DEFAULT_WHIR_FOLDING_FACTOR);
    assert!(factor > 0, "P3_WARP_WHIR_FOLDING_FACTOR must be non-zero");
    factor
}

fn effective_whir_folding_factor(num_variables: usize) -> usize {
    whir_folding_factor().min(num_variables.max(1))
}

#[derive(Clone, Copy, Debug)]
struct WhirRoundShape {
    rounds: usize,
    final_sumcheck_rounds: usize,
    final_queries: usize,
}

fn whir_round_shape(mmcs: &MyMmcs, num_variables: usize) -> WhirRoundShape {
    let config = WhirConfig::<EF, F, MyMmcs, MyChallenger>::new(
        num_variables,
        make_whir_protocol_params(mmcs, num_variables),
    );
    WhirRoundShape {
        rounds: config.n_rounds(),
        final_sumcheck_rounds: config.final_sumcheck_rounds,
        final_queries: config.final_queries,
    }
}

fn make_whir_protocol_params(mmcs: &MyMmcs, num_variables: usize) -> ProtocolParameters<MyMmcs> {
    ProtocolParameters {
        security_level: 32,
        pow_bits: 0,
        rs_domain_initial_reduction_factor: 1,
        folding_factor: FoldingFactor::Constant(effective_whir_folding_factor(num_variables)),
        mmcs: mmcs.clone(),
        soundness_type: SecurityAssumption::JohnsonBound,
        starting_log_inv_rate: LOG_INV_RATE,
    }
}

fn make_whir_config(mmcs: &MyMmcs, num_variables: usize) -> MyWhirConfig {
    WhirConfig::new(
        num_variables,
        make_whir_protocol_params(mmcs, num_variables),
    )
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
                effective_whir_folding_factor(num_variables),
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
            effective_whir_folding_factor(statement.num_variables()),
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
            effective_whir_folding_factor(statement.num_variables()),
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

fn make_whir_pcs_for_num_vars(mmcs: &MyMmcs, num_variables: usize) -> MyWhirPcs {
    WhirPcs::new(
        num_variables,
        make_whir_protocol_params(mmcs, num_variables),
        MyDft::default(),
        SumcheckStrategy::Svo,
    )
}

fn whir_native_circuit_options() -> WhirNativeCircuitOptions {
    let default_openings = if malicious_test_mode() {
        1
    } else {
        DEFAULT_WHIR_NATIVE_OUTER_OPENINGS
    };
    WhirNativeCircuitOptions {
        openings_per_table: parse_usize_env("P3_WHIR_RECURSIVE_OUTER_OPENINGS", default_openings)
            .max(1),
        min_num_variables: whir_folding_factor().max(4),
    }
}

fn eval_bench_poseidon2(config: Poseidon2Config, input: &[EF]) -> Option<Vec<EF>> {
    if config != Poseidon2Config::BabyBearD4Width16 {
        return None;
    }
    if input.len() != config.width_ext() {
        return None;
    }

    let perm = make_permutation();
    let mut base_input = [F::ZERO; 16];
    for (i, ext_elem) in input.iter().enumerate() {
        let coeffs = ext_elem.as_basis_coefficients_slice();
        base_input[i * config.d()..(i + 1) * config.d()].copy_from_slice(coeffs);
    }
    let base_output = perm.permute(base_input);
    Some(
        (0..config.width_ext())
            .map(|i| {
                EF::from_basis_coefficients_slice(
                    &base_output[i * config.d()..(i + 1) * config.d()],
                )
                .expect("Poseidon2 output coefficients must reconstruct an extension element")
            })
            .collect(),
    )
}

#[allow(dead_code)]
struct WhirFullBundle {
    commitments: Vec<MyCommitment>,
    claims: Vec<Vec<(Point<EF>, EF)>>,
    proofs: Vec<MyWhirProof>,
}

fn whir_full_bundle_bytes(bundle: &WhirFullBundle) -> usize {
    postcard::to_stdvec(&(&bundle.commitments, &bundle.proofs))
        .expect("serialize WHIR full bundle")
        .len()
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

#[allow(dead_code)]
struct WhirRecursiveVerifierKey {
    outer_mmcs: MyMmcs,
    outer_options: WhirNativeCircuitOptions,
    verification_circuit: p3_circuit::Circuit<EF>,
}

#[allow(dead_code)]
struct WhirRecursiveProofBundle {
    outer_proof: MyWhirNativeCircuitProof,
    public_statement: Vec<EF>,
}

#[allow(dead_code)]
struct WhirRecursiveBundle {
    native: WhirFullBundle,
    verifier_key: WhirRecursiveVerifierKey,
    proof: WhirRecursiveProofBundle,
    recursive_public_inputs: usize,
    recursive_private_inputs: usize,
    recursive_mmcs_ops: usize,
}

#[derive(Clone, Copy, Debug, Default)]
struct WhirRecursivePhaseDurations {
    native_whir: Duration,
    circuit_build: Duration,
    trace_generation: Duration,
    outer_proof: Duration,
    total: Duration,
}

#[derive(Clone, Copy, Debug, Default)]
struct WhirRecursiveVerifyPhaseDurations {
    outer_verify: Duration,
    total: Duration,
}

#[allow(dead_code)]
fn query_index_bits(index: usize, bits: usize) -> Vec<EF> {
    (0..bits)
        .map(|bit| {
            if (index >> bit) & 1 == 1 {
                EF::ONE
            } else {
                EF::ZERO
            }
        })
        .collect()
}

#[allow(dead_code)]
fn make_recursive_whir_inputs(
    bundle: &WhirFullBundle,
    config: &MyWhirConfig,
) -> Vec<RecursiveWhirInput> {
    bundle
        .commitments
        .iter()
        .cloned()
        .zip(bundle.claims.iter().cloned())
        .zip(bundle.proofs.iter().cloned())
        .map(|((commitment, claims), proof)| {
            let round_query_index_bits = proof
                .rounds
                .iter()
                .zip(config.round_parameters.iter())
                .map(|(round, params)| {
                    let bits = (params.domain_size >> params.folding_factor).ilog2() as usize;
                    round
                        .query_indices
                        .iter()
                        .map(|&index| query_index_bits(index, bits))
                        .collect()
                })
                .collect();
            let final_round = config.final_round_config();
            let final_bits =
                (final_round.domain_size >> final_round.folding_factor).ilog2() as usize;
            let final_query_index_bits = proof
                .final_query_indices
                .iter()
                .map(|&index| query_index_bits(index, final_bits))
                .collect();
            WhirProofVerificationInput {
                commitment,
                opening_claims: vec![claims],
                proof,
                round_query_index_bits,
                final_query_index_bits,
            }
        })
        .collect()
}

#[allow(dead_code)]
fn whir_query_opening_public_labels(
    query: &p3_whir::pcs::proof::QueryOpening<F, EF, <MyMmcs as p3_commit::Mmcs<F>>::Proof>,
    prefix: &str,
) -> Vec<String> {
    match query {
        p3_whir::pcs::proof::QueryOpening::Base { values, proof: _ } => (0..values.len())
            .map(|i| format!("{prefix}.base_value[{i}]"))
            .chain((0..0).map(|i| format!("{prefix}.base_proof_public[{i}]")))
            .collect(),
        p3_whir::pcs::proof::QueryOpening::Extension { values, proof: _ } => (0..values.len())
            .map(|i| format!("{prefix}.extension_value[{i}]"))
            .chain((0..0).map(|i| format!("{prefix}.extension_proof_public[{i}]")))
            .collect(),
        p3_whir::pcs::proof::QueryOpening::SharedBase { values, proof: _ } => values
            .iter()
            .enumerate()
            .flat_map(|(row, values)| {
                (0..values.len()).map(move |i| format!("{prefix}.shared_row[{row}][{i}]"))
            })
            .chain((0..0).map(|i| format!("{prefix}.shared_proof_public[{i}]")))
            .collect(),
        p3_whir::pcs::proof::QueryOpening::Batched { openings } => openings
            .iter()
            .enumerate()
            .flat_map(|(i, opening)| {
                whir_query_opening_public_labels(opening, &format!("{prefix}.batched[{i}]"))
            })
            .collect(),
    }
}

#[allow(dead_code)]
fn recursive_whir_public_labels(input: &RecursiveWhirInput, proof_index: usize) -> Vec<String> {
    let prefix = format!("proof[{proof_index}]");
    let mut labels = Vec::new();
    labels.extend(
        (0..input.commitment.num_roots() * WHIR_DIGEST_ELEMS)
            .map(|i| format!("{prefix}.statement_commitment[{i}]")),
    );
    for (poly_index, claims) in input.opening_claims.iter().enumerate() {
        for (claim_index, (point, _)) in claims.iter().enumerate() {
            labels.extend(
                (0..point.num_variables())
                    .map(|i| format!("{prefix}.claim[{poly_index}][{claim_index}].point[{i}]")),
            );
            labels.push(format!("{prefix}.claim[{poly_index}][{claim_index}].value"));
        }
    }

    labels
}

#[allow(dead_code)]
fn try_build_n_whir_recursive_bundle(
    fixture: &WarpKernelFixture,
    witnesses: &[Vec<F>],
) -> Result<WhirRecursiveBundle, WhirNativeCircuitError> {
    try_build_n_whir_recursive_bundle_with_phases(fixture, witnesses).map(|(bundle, _)| bundle)
}

#[allow(dead_code)]
fn try_build_n_whir_recursive_bundle_with_phases(
    fixture: &WarpKernelFixture,
    witnesses: &[Vec<F>],
) -> Result<(WhirRecursiveBundle, WhirRecursivePhaseDurations), WhirNativeCircuitError> {
    let total_start = Instant::now();
    let phase_start = Instant::now();
    let native = build_n_whir_full_pcs(fixture, witnesses);
    let native_whir = phase_start.elapsed();
    try_build_n_whir_recursive_bundle_from_native_with_phases(
        fixture,
        native,
        native_whir,
        total_start,
    )
}

#[allow(dead_code)]
fn try_build_n_whir_recursive_bundle_from_native(
    fixture: &WarpKernelFixture,
    native: WhirFullBundle,
) -> Result<WhirRecursiveBundle, WhirNativeCircuitError> {
    try_build_n_whir_recursive_bundle_from_native_with_phases(
        fixture,
        native,
        Duration::ZERO,
        Instant::now(),
    )
    .map(|(bundle, _)| bundle)
}

fn try_build_n_whir_recursive_bundle_from_native_with_phases(
    fixture: &WarpKernelFixture,
    native: WhirFullBundle,
    native_whir: Duration,
    total_start: Instant,
) -> Result<(WhirRecursiveBundle, WhirRecursivePhaseDurations), WhirNativeCircuitError> {
    let phase_start = Instant::now();
    let whir_config = make_whir_config(&fixture.mmcs, fixture.code.log_msg_len());
    let recursive_inputs = make_recursive_whir_inputs(&native, &whir_config);
    let poseidon2_config = Poseidon2Config::BabyBearD4Width16;

    let mut circuit_builder = CircuitBuilder::new();
    circuit_builder.enable_poseidon2_perm::<BabyBearD4Width16, _>(
        generate_poseidon2_trace::<EF, BabyBearD4Width16>,
        make_permutation(),
    );
    circuit_builder.enable_recompose::<F>(generate_recompose_trace::<F, EF>);

    let mut per_proof_op_ids = Vec::with_capacity(recursive_inputs.len());
    for input in &recursive_inputs {
        let targets = RecursiveWhirTargets::new_private_proof(&mut circuit_builder, input);
        let mut challenger = CircuitChallenger::<16, 8, Poseidon2Config>::new_babybear();
        challenger.init::<F, EF>(&mut circuit_builder);
        let op_ids = verify_native_whir_proof_circuit::<
            F,
            EF,
            _,
            RecursiveWhirMmcs,
            MyChallenger,
            WHIR_DIGEST_ELEMS,
        >(
            &mut circuit_builder,
            &mut challenger,
            poseidon2_config,
            &whir_config,
            &targets,
        )
        .map_err(|err| {
            WhirNativeCircuitError::ConstraintViolation(format!(
                "build recursive WHIR verifier circuit failed: {err}"
            ))
        })?;
        per_proof_op_ids.push(op_ids);
    }

    let verification_circuit = circuit_builder.build().map_err(|err| {
        WhirNativeCircuitError::ConstraintViolation(format!(
            "build recursive WHIR verification circuit failed: {err}"
        ))
    })?;
    let public_inputs = recursive_inputs
        .iter()
        .flat_map(RecursiveWhirTargets::public_values_for_private_proof)
        .collect::<Vec<_>>();
    let private_inputs = recursive_inputs
        .iter()
        .flat_map(RecursiveWhirTargets::private_values_for_private_proof)
        .collect::<Vec<_>>();
    assert_eq!(
        public_inputs.len(),
        verification_circuit.public_flat_len,
        "recursive WHIR public input packing mismatch"
    );
    let public_labels = recursive_inputs
        .iter()
        .enumerate()
        .flat_map(|(proof_index, input)| recursive_whir_public_labels(input, proof_index))
        .collect::<Vec<_>>();
    assert_eq!(
        public_labels.len(),
        public_inputs.len(),
        "recursive WHIR public input label mismatch"
    );
    let mut seen_public_rows = std::collections::HashMap::new();
    for (pos, (&row, &value)) in verification_circuit
        .public_rows
        .iter()
        .zip(&public_inputs)
        .enumerate()
    {
        if let Some((prev_pos, prev_value)) = seen_public_rows.insert(row, (pos, value)) {
            assert_eq!(
                prev_value, value,
                "recursive WHIR public inputs {prev_pos} ({}) and {pos} ({}) alias witness {row:?} with different values",
                public_labels[prev_pos], public_labels[pos]
            );
        }
    }
    assert_eq!(
        private_inputs.len(),
        verification_circuit.private_flat_len,
        "recursive WHIR private input packing mismatch"
    );
    let circuit_build = phase_start.elapsed();

    let phase_start = Instant::now();
    let mut runner = verification_circuit.runner();
    runner.set_public_inputs(&public_inputs).map_err(|err| {
        WhirNativeCircuitError::ConstraintViolation(format!(
            "set recursive WHIR public inputs failed: {err}"
        ))
    })?;
    runner.set_private_inputs(&private_inputs).map_err(|err| {
        WhirNativeCircuitError::ConstraintViolation(format!(
            "set recursive WHIR private inputs failed: {err}"
        ))
    })?;
    for (input, op_ids) in recursive_inputs.iter().zip(&per_proof_op_ids) {
        set_whir_mmcs_private_data::<F, EF, MyMmcs, MyHash, MyCompress, WHIR_DIGEST_ELEMS>(
            &mut runner,
            op_ids,
            &input.proof,
        )
        .map_err(|err| {
            WhirNativeCircuitError::ConstraintViolation(format!(
                "set recursive WHIR MMCS private data failed: {err}"
            ))
        })?;
    }

    let verification_traces = runner.run().map_err(|err| {
        WhirNativeCircuitError::ConstraintViolation(format!(
            "run recursive WHIR verifier circuit failed: {err}"
        ))
    })?;
    let trace_generation = phase_start.elapsed();

    let phase_start = Instant::now();
    let outer_options = whir_native_circuit_options();
    let outer_proof = prove_whir_native_circuit(
        &verification_circuit,
        &public_inputs,
        &private_inputs,
        &[],
        &verification_traces,
        outer_options,
        |num_variables| make_whir_pcs_for_num_vars(&fixture.mmcs, num_variables),
        BenchChallenger::new,
        eval_bench_poseidon2,
    )?;
    let outer_proof_duration = phase_start.elapsed();

    let bundle = WhirRecursiveBundle {
        native,
        verifier_key: WhirRecursiveVerifierKey {
            outer_mmcs: fixture.mmcs.clone(),
            outer_options,
            verification_circuit,
        },
        proof: WhirRecursiveProofBundle {
            outer_proof,
            public_statement: public_inputs.clone(),
        },
        recursive_public_inputs: public_inputs.len(),
        recursive_private_inputs: private_inputs.len(),
        recursive_mmcs_ops: per_proof_op_ids.iter().map(Vec::len).sum(),
    };
    Ok((
        bundle,
        WhirRecursivePhaseDurations {
            native_whir,
            circuit_build,
            trace_generation,
            outer_proof: outer_proof_duration,
            total: total_start.elapsed(),
        },
    ))
}

#[allow(dead_code)]
fn build_n_whir_recursive_bundle(
    fixture: &WarpKernelFixture,
    witnesses: &[Vec<F>],
) -> WhirRecursiveBundle {
    try_build_n_whir_recursive_bundle(fixture, witnesses)
        .expect("build WHIR-native recursive circuit proof")
}

#[allow(dead_code)]
fn prove_n_whir_recursive(fixture: &WarpKernelFixture, witnesses: &[Vec<F>]) {
    black_box(build_n_whir_recursive_bundle(fixture, witnesses));
}

#[allow(dead_code)]
fn try_verify_n_whir_recursive_bundle(bundle: &WhirRecursiveBundle) -> Result<(), String> {
    try_verify_n_whir_recursive_proof(&bundle.verifier_key, &bundle.proof)
}

#[allow(dead_code)]
fn try_verify_n_whir_recursive_bundle_with_phases(
    bundle: &WhirRecursiveBundle,
) -> Result<WhirRecursiveVerifyPhaseDurations, String> {
    try_verify_n_whir_recursive_proof_with_phases(&bundle.verifier_key, &bundle.proof)
}

#[allow(dead_code)]
fn try_verify_n_whir_recursive_proof(
    verifier_key: &WhirRecursiveVerifierKey,
    proof: &WhirRecursiveProofBundle,
) -> Result<(), String> {
    try_verify_n_whir_recursive_proof_with_phases(verifier_key, proof).map(|_| ())
}

#[allow(dead_code)]
fn try_verify_n_whir_recursive_proof_with_phases(
    verifier_key: &WhirRecursiveVerifierKey,
    proof: &WhirRecursiveProofBundle,
) -> Result<WhirRecursiveVerifyPhaseDurations, String> {
    let total_start = Instant::now();
    let phase_start = Instant::now();
    verify_whir_native_circuit_proof(
        &verifier_key.verification_circuit,
        &proof.public_statement,
        verifier_key.outer_options,
        &proof.outer_proof,
        |num_variables| make_whir_pcs_for_num_vars(&verifier_key.outer_mmcs, num_variables),
        BenchChallenger::new,
        eval_bench_poseidon2,
    )
    .map_err(|err| err.to_string())?;
    let outer_verify = phase_start.elapsed();
    Ok(WhirRecursiveVerifyPhaseDurations {
        outer_verify,
        total: total_start.elapsed(),
    })
}

#[allow(dead_code)]
fn verify_n_whir_recursive_bundle(bundle: &WhirRecursiveBundle) {
    try_verify_n_whir_recursive_bundle(bundle).expect("verify WHIR-native recursive circuit proof");
}

#[derive(Clone, Copy, Debug, Default)]
struct LaneMetrics {
    proof_bytes: usize,
    verifier_payload_bytes: usize,
    total_artifact_bytes: usize,
    commitments: usize,
    opening_claims: usize,
    whir_queries: usize,
    sumcheck_rounds: usize,
    table_count: usize,
    max_table_height: usize,
    total_table_cells: usize,
}

fn serialized_len<T: Serialize>(value: &T, label: &str) -> usize {
    postcard::to_stdvec(value)
        .unwrap_or_else(|err| panic!("serialize {label}: {err}"))
        .len()
}

fn whir_proof_query_count<F, EF, MT>(proof: &WhirProof<F, EF, MT>) -> usize
where
    F: Send + Sync + Clone,
    MT: p3_commit::Mmcs<F>,
{
    proof
        .rounds
        .iter()
        .map(|round| {
            round
                .queries
                .iter()
                .map(query_opening_count::<F, EF, MT::Proof>)
                .sum::<usize>()
        })
        .sum::<usize>()
        + proof
            .final_queries
            .iter()
            .map(query_opening_count::<F, EF, MT::Proof>)
            .sum::<usize>()
}

fn query_opening_count<F, EF, Proof>(query: &QueryOpening<F, EF, Proof>) -> usize {
    match query {
        QueryOpening::Base { .. }
        | QueryOpening::Extension { .. }
        | QueryOpening::SharedBase { .. } => 1,
        QueryOpening::Batched { openings } => openings.iter().map(query_opening_count).sum(),
    }
}

fn whir_proof_sumcheck_rounds<F, EF, MT>(proof: &WhirProof<F, EF, MT>) -> usize
where
    F: Send + Sync + Clone,
    MT: p3_commit::Mmcs<F>,
{
    proof.initial_sumcheck.num_rounds()
        + proof
            .rounds
            .iter()
            .map(|round| round.sumcheck.num_rounds())
            .sum::<usize>()
        + proof
            .final_sumcheck
            .as_ref()
            .map_or(0, SumcheckData::num_rounds)
}

fn recursive_lane_metrics(bundle: &WhirRecursiveBundle) -> LaneMetrics {
    let outer_proof = &bundle.proof.outer_proof;
    let table_count = outer_proof.table_commitments.len();
    let max_table_height = bundle
        .proof
        .outer_proof
        .table_commitments
        .iter()
        .map(|table| table.metadata.padded_height)
        .max()
        .unwrap_or(0);
    let total_table_cells = bundle
        .proof
        .outer_proof
        .table_commitments
        .iter()
        .map(|table| table.metadata.padded_height * table.metadata.padded_width)
        .sum();
    LaneMetrics {
        proof_bytes: serialized_len(outer_proof, "recursive outer proof"),
        verifier_payload_bytes: serialized_len(
            &(outer_proof, &bundle.proof.public_statement),
            "recursive verifier payload",
        ),
        total_artifact_bytes: serialized_len(
            &(
                &bundle.native.commitments,
                &bundle.native.proofs,
                outer_proof,
            ),
            "recursive total artifact",
        ),
        commitments: bundle.native.commitments.len() + outer_proof.table_commitments.len(),
        opening_claims: bundle.native.claims.iter().map(Vec::len).sum::<usize>()
            + outer_proof
                .opening_proofs
                .iter()
                .map(|proof| proof.opening_claims.len())
                .sum::<usize>(),
        whir_queries: bundle
            .native
            .proofs
            .iter()
            .map(whir_proof_query_count)
            .sum::<usize>()
            + outer_proof
                .opening_proofs
                .iter()
                .map(|proof| whir_proof_query_count(&proof.proof))
                .sum::<usize>(),
        sumcheck_rounds: bundle
            .native
            .proofs
            .iter()
            .map(whir_proof_sumcheck_rounds)
            .sum::<usize>()
            + outer_proof
                .opening_proofs
                .iter()
                .map(|proof| whir_proof_sumcheck_rounds(&proof.proof))
                .sum::<usize>()
            + outer_proof
                .constraint_sumcheck_proofs
                .iter()
                .filter_map(|proof| proof.local_proof.as_ref())
                .map(|proof| proof.sumcheck.num_rounds())
                .sum::<usize>(),
        table_count,
        max_table_height,
        total_table_cells,
    }
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

fn warp_whir_root_bundle_bytes(bundle: &WarpWhirRootBundle) -> usize {
    postcard::to_stdvec(&(&bundle.proof, &bundle.root_iop_proof))
        .expect("serialize WARP root bundle")
        .len()
}

fn warp_lane_metrics(bundle: &WarpWhirRootBundle) -> LaneMetrics {
    LaneMetrics {
        proof_bytes: serialized_len(&(&bundle.proof, &bundle.root_iop_proof), "WARP proof bytes"),
        verifier_payload_bytes: serialized_len(
            &(&bundle.instance, &bundle.proof, &bundle.root_iop_proof),
            "WARP verifier payload",
        ),
        total_artifact_bytes: serialized_len(
            &(&bundle.instance, &bundle.proof, &bundle.root_iop_proof),
            "WARP total artifact",
        ),
        commitments: bundle.oracle_count,
        opening_claims: bundle.claim_count,
        whir_queries: whir_proof_query_count(&bundle.root_iop_proof.opening.opening),
        sumcheck_rounds: bundle
            .proof
            .steps
            .iter()
            .map(|step| {
                step.proof.twin_constraint_sumcheck.num_rounds()
                    + step.proof.batching_sumcheck.num_rounds()
            })
            .sum::<usize>()
            + bundle.proof.final_proof.pesat.decider_sumcheck.num_rounds()
            + whir_proof_sumcheck_rounds(&bundle.root_iop_proof.opening.opening),
        table_count: 0,
        max_table_height: 0,
        total_table_cells: 0,
    }
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
    fn new(message_pcs: &'a MyWhirPcs, code: &'a ReedSolomonCode<F, MyDft>) -> Self {
        let log_codeword_len = code.log_codeword_len();
        Self {
            root_system: NativeWarpWhirRootProofSystem::new(
                message_pcs,
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
            self.transcript
                .borrow_mut()
                .oracles
                .push((commitment.clone(), RootIopOracleValues::Base(Vec::new())));
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
            RootIopOpeningPoint::RsCodewordIndex(index),
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
                RootIopOpeningPoint::RsCodewordIndex(index),
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
            RootIopOracleValues::Extension(Vec::new()),
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

    fn commit_with_message(
        &self,
        codeword: Vec<EF>,
        message: &[EF],
    ) -> Result<(Self::Commitment, Self::ProverData), Self::Error> {
        if codeword.len() != (1 << self.log_codeword_len) {
            return Err(RootIopError::ShapeMismatch);
        }
        if message.len() != (1 << (self.log_codeword_len - LOG_INV_RATE)) {
            return Err(RootIopError::ShapeMismatch);
        }
        let oracle_id = self.next_oracle_id();
        let (commitment, prover_data) = self
            .root_system
            .commit_extension_message_oracle(oracle_id, message.to_vec())
            .map_err(|_| RootIopError::ShapeMismatch)?;
        self.transcript.borrow_mut().oracles.push((
            commitment.clone(),
            RootIopOracleValues::Extension(Vec::new()),
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
            RootIopOpeningPoint::RsCodewordIndex(index),
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
                RootIopOpeningPoint::RsCodewordIndex(index),
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
            RootIopOpeningPoint::RsCodewordIndex(index),
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
                RootIopOpeningPoint::RsCodewordIndex(index),
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
            RootIopOpeningPoint::RsCodewordIndex(index),
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
                RootIopOpeningPoint::RsCodewordIndex(index),
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
            priors,
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
    let base_message_pcs = make_whir_pcs_for_num_vars(&fixture.mmcs, fixture.code.log_msg_len());
    let root_iop_backend = BenchRootIopWhirProver::new(&base_message_pcs, &fixture.code);
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

fn count_root_whir_openings(_proof: &MyRootWhirProof) -> usize {
    1
}

fn prove_warp_whir_root(fixture: &WarpKernelFixture, witnesses: &[Vec<F>]) {
    black_box(build_warp_whir_root_bundle(fixture, witnesses));
}

fn try_verify_warp_whir_root_bundle(
    fixture: &WarpKernelFixture,
    bundle: &WarpWhirRootBundle,
) -> Result<(), String> {
    try_verify_warp_whir_root_bundle_with_phases(fixture, bundle).map(|_| ())
}

fn verify_warp_whir_root_bundle(fixture: &WarpKernelFixture, bundle: &WarpWhirRootBundle) {
    try_verify_warp_whir_root_bundle(fixture, bundle).expect("verify WHIR-backed WARP root bundle");
}

fn try_verify_warp_whir_root_bundle_with_phases(
    fixture: &WarpKernelFixture,
    bundle: &WarpWhirRootBundle,
) -> Result<WarpWhirRootVerifyPhaseDurations, String> {
    let total_start = Instant::now();
    let phase_start = Instant::now();
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
        .map_err(|err| format!("WHIR-backed WARP root chain verification failed: {err}"))?;
    if verified != bundle.instance {
        return Err("WHIR-backed WARP verified accumulator mismatch".to_string());
    }
    let expected_commitments = root_iop_verifier.expected_inner_commitments();
    let expected_claims = root_iop_verifier.expected_claims();
    let claim_shape = root_claim_shape(&fixture.code, &expected_claims);
    let chain = phase_start.elapsed();

    let phase_start = Instant::now();
    let root_system = NativeWarpWhirRootProofSystem::new(
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
        .map_err(|err| format!("WARP root IOP WHIR proof verification failed: {err}"))?;
    let root_whir = phase_start.elapsed();
    Ok(WarpWhirRootVerifyPhaseDurations {
        setup,
        chain,
        root_system_setup,
        root_whir,
        total: total_start.elapsed(),
        claim_shape,
    })
}

fn verify_warp_whir_root_bundle_with_phases(
    fixture: &WarpKernelFixture,
    bundle: &WarpWhirRootBundle,
) -> WarpWhirRootVerifyPhaseDurations {
    try_verify_warp_whir_root_bundle_with_phases(fixture, bundle)
        .expect("verify WHIR-backed WARP root bundle")
}

fn root_claim_shape(
    code: &ReedSolomonCode<F, MyDft>,
    claims: &[RootIopOpeningClaim<F, EF>],
) -> RootClaimShape {
    let stride = 1 << code.log_inv_rate();
    let mut shape = RootClaimShape::default();
    for claim in claims {
        match &claim.point {
            RootIopOpeningPoint::Index(index) | RootIopOpeningPoint::RsCodewordIndex(index) => {
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

fn parse_first_usize_list_env(name: &str, default: usize) -> usize {
    parse_usize_list_env(name, &[default])
        .into_iter()
        .next()
        .unwrap_or(default)
}

fn expect_malicious_reject(name: &str, field: &str, result: Result<(), String>) {
    match result {
        Ok(()) => panic!("{name} unexpectedly verified after mutating {field}"),
        Err(_) => eprintln!("ok reject {name}"),
    }
}

fn first_recursive_opening_claim_indices(
    proof: &MyWhirNativeCircuitProof,
) -> Option<(usize, usize)> {
    proof
        .opening_proofs
        .iter()
        .enumerate()
        .find_map(|(proof_index, opening_proof)| {
            (!opening_proof.opening_claims.is_empty()).then_some((proof_index, 0))
        })
}

fn first_poseidon2_terminal_opening_indices(
    proof: &MyWhirNativeCircuitProof,
) -> Option<(usize, usize)> {
    proof
        .constraint_sumcheck_proofs
        .iter()
        .enumerate()
        .find_map(|(proof_index, constraint_proof)| {
            let local_proof = constraint_proof.local_proof.as_ref()?;
            (local_proof.kind == WhirNativeLocalConstraintKind::Poseidon2Air
                && !local_proof.terminal_openings.is_empty())
            .then_some((proof_index, 0))
        })
}

fn first_whir_initial_sumcheck_eval_indices<MT>(
    proof: &WhirProof<F, EF, MT>,
) -> Option<(usize, usize)>
where
    MT: p3_commit::Mmcs<F>,
{
    proof
        .initial_sumcheck
        .polynomial_evaluations
        .iter()
        .enumerate()
        .find_map(|(round_index, evals)| (!evals.is_empty()).then_some((round_index, 0)))
}

fn tamper_first_query_opening_value(
    query: &mut QueryOpening<F, EF, <MyMmcs as p3_commit::Mmcs<F>>::Proof>,
) -> bool {
    match query {
        QueryOpening::Base { values, .. } => {
            if let Some(value) = values.first_mut() {
                *value += F::ONE;
                true
            } else {
                false
            }
        }
        QueryOpening::Extension { values, .. } => {
            if let Some(value) = values.first_mut() {
                *value += EF::ONE;
                true
            } else {
                false
            }
        }
        QueryOpening::SharedBase { values, .. } => {
            if let Some(value) = values.iter_mut().find_map(|row| row.first_mut()) {
                *value += F::ONE;
                true
            } else {
                false
            }
        }
        QueryOpening::Batched { openings } => {
            openings.iter_mut().any(tamper_first_query_opening_value)
        }
    }
}

fn tamper_first_whir_query_value(proof: &mut MyWhirProof) -> bool {
    proof
        .rounds
        .iter_mut()
        .flat_map(|round| round.queries.iter_mut())
        .chain(proof.final_queries.iter_mut())
        .any(tamper_first_query_opening_value)
}

fn tamper_first_whir_query_index(proof: &mut MyWhirProof) -> bool {
    if let Some(index) = proof
        .rounds
        .iter_mut()
        .find_map(|round| round.query_indices.first_mut())
    {
        *index += 1;
        true
    } else if let Some(index) = proof.final_query_indices.first_mut() {
        *index += 1;
        true
    } else {
        false
    }
}

fn tamper_first_merkle_cap_word(commitment: &mut MyCommitment) -> bool {
    let mut roots = commitment.roots().to_vec();
    if let Some(first_root) = roots.first_mut() {
        first_root[0] += F::ONE;
        *commitment = MerkleCap::new(roots);
        true
    } else {
        false
    }
}

fn tamper_first_whir_round_commitment(proof: &mut MyWhirProof) -> bool {
    proof
        .rounds
        .iter_mut()
        .find_map(|round| round.commitment.as_mut())
        .is_some_and(tamper_first_merkle_cap_word)
}

fn tamper_first_whir_final_poly_value(proof: &mut MyWhirProof) -> bool {
    if let Some(value) = proof
        .final_poly
        .as_mut()
        .and_then(|poly| poly.as_mut_slice().first_mut())
    {
        *value += EF::ONE;
        true
    } else {
        false
    }
}

fn run_malicious_soundness_tests() {
    let num_variables = parse_first_usize_list_env("P3_WARP_SUMCHECK_K", 2);
    let n = parse_first_usize_list_env("P3_WARP_SUMCHECK_N", 2);
    let arity = warp_fresh_per_step();
    assert!(
        n >= arity && (n - arity).is_multiple_of(arity - 1),
        "malicious tests need N >= arity and N = arity + k * (arity - 1); got N={n}, arity={arity}"
    );

    let fixture = make_warp_fixture(num_variables);
    let witnesses = make_boolean_witnesses(num_variables, n);

    let mut recursive_bundle = try_build_n_whir_recursive_bundle(&fixture, &witnesses)
        .expect("build honest WHIR-native recursive bundle");
    try_verify_n_whir_recursive_bundle(&recursive_bundle)
        .expect("honest WHIR-native recursive bundle must verify");

    let old_public_input = *recursive_bundle
        .proof
        .public_statement
        .first()
        .expect("recursive bundle has public inputs");
    recursive_bundle.proof.public_statement[0] += EF::ONE;
    expect_malicious_reject(
        "recursive_public_input_tamper",
        "recursive_public_statement[0]",
        try_verify_n_whir_recursive_bundle(&recursive_bundle),
    );
    recursive_bundle.proof.public_statement[0] = old_public_input;

    let old_public_digest = *recursive_bundle
        .proof
        .outer_proof
        .public_io_digest
        .first()
        .expect("recursive outer proof has public digest");
    recursive_bundle.proof.outer_proof.public_io_digest[0] += F::ONE;
    expect_malicious_reject(
        "recursive_public_digest_tamper",
        "outer_proof.public_io_digest[0]",
        try_verify_n_whir_recursive_bundle(&recursive_bundle),
    );
    recursive_bundle.proof.outer_proof.public_io_digest[0] = old_public_digest;

    let (opening_proof_index, opening_claim_index) =
        first_recursive_opening_claim_indices(&recursive_bundle.proof.outer_proof)
            .expect("recursive outer proof has WHIR opening claims");
    let old_opening_claim = recursive_bundle.proof.outer_proof.opening_proofs[opening_proof_index]
        .opening_claims[opening_claim_index]
        .1;
    recursive_bundle.proof.outer_proof.opening_proofs[opening_proof_index].opening_claims
        [opening_claim_index]
        .1 += EF::ONE;
    expect_malicious_reject(
        "recursive_outer_opening_claim_tamper",
        "outer_proof.opening_proofs[*].opening_claims[*].1",
        try_verify_n_whir_recursive_bundle(&recursive_bundle),
    );
    recursive_bundle.proof.outer_proof.opening_proofs[opening_proof_index].opening_claims
        [opening_claim_index]
        .1 = old_opening_claim;

    let (constraint_proof_index, terminal_opening_index) =
        first_poseidon2_terminal_opening_indices(&recursive_bundle.proof.outer_proof)
            .expect("recursive outer proof has a Poseidon2 terminal opening");
    let local_proof = recursive_bundle
        .proof
        .outer_proof
        .constraint_sumcheck_proofs[constraint_proof_index]
        .local_proof
        .as_mut()
        .expect("Poseidon2 local proof exists");
    let old_terminal_opening = local_proof.terminal_openings[terminal_opening_index].value;
    local_proof.terminal_openings[terminal_opening_index].value += EF::ONE;
    expect_malicious_reject(
        "recursive_local_terminal_opening_tamper",
        "outer_proof.constraint_sumcheck_proofs[*].local_proof.terminal_openings[*].value",
        try_verify_n_whir_recursive_bundle(&recursive_bundle),
    );
    recursive_bundle
        .proof
        .outer_proof
        .constraint_sumcheck_proofs[constraint_proof_index]
        .local_proof
        .as_mut()
        .expect("Poseidon2 local proof exists")
        .terminal_openings[terminal_opening_index]
        .value = old_terminal_opening;

    let mut native = build_n_whir_full_pcs(&fixture, &witnesses);
    let (sumcheck_round, sumcheck_eval) =
        first_whir_initial_sumcheck_eval_indices(&native.proofs[0])
            .expect("native WHIR proof has initial sumcheck evaluations");
    native.proofs[0].initial_sumcheck.polynomial_evaluations[sumcheck_round][sumcheck_eval] +=
        EF::ONE;
    expect_malicious_reject(
        "recursive_native_whir_sumcheck_tamper",
        "native.proofs[0].initial_sumcheck.polynomial_evaluations[*][*]",
        try_build_n_whir_recursive_bundle_from_native(&fixture, native)
            .map(|_| ())
            .map_err(|err| err.to_string()),
    );

    let mut native = build_n_whir_full_pcs(&fixture, &witnesses);
    assert!(
        tamper_first_whir_query_value(&mut native.proofs[0]),
        "native WHIR proof has query values"
    );
    expect_malicious_reject(
        "recursive_native_whir_query_value_tamper",
        "native.proofs[0].{rounds,final_queries}[*].queries[*].values[*]",
        try_build_n_whir_recursive_bundle_from_native(&fixture, native)
            .map(|_| ())
            .map_err(|err| err.to_string()),
    );

    let mut native = build_n_whir_full_pcs(&fixture, &witnesses);
    assert!(
        tamper_first_whir_query_index(&mut native.proofs[0]),
        "native WHIR proof has query indices"
    );
    expect_malicious_reject(
        "recursive_native_whir_query_index_tamper",
        "native.proofs[0].{round,final}_query_indices[*]",
        try_build_n_whir_recursive_bundle_from_native(&fixture, native)
            .map(|_| ())
            .map_err(|err| err.to_string()),
    );

    let mut native = build_n_whir_full_pcs(&fixture, &witnesses);
    if tamper_first_whir_round_commitment(&mut native.proofs[0]) {
        expect_malicious_reject(
            "recursive_native_whir_round_commitment_tamper",
            "native.proofs[0].rounds[*].commitment.roots()[0][0]",
            try_build_n_whir_recursive_bundle_from_native(&fixture, native)
                .map(|_| ())
                .map_err(|err| err.to_string()),
        );
    }

    let mut native = build_n_whir_full_pcs(&fixture, &witnesses);
    assert!(
        tamper_first_whir_final_poly_value(&mut native.proofs[0]),
        "native WHIR proof has final polynomial values"
    );
    expect_malicious_reject(
        "recursive_native_whir_final_poly_tamper",
        "native.proofs[0].final_poly[0]",
        try_build_n_whir_recursive_bundle_from_native(&fixture, native)
            .map(|_| ())
            .map_err(|err| err.to_string()),
    );

    let mut native = build_n_whir_full_pcs(&fixture, &witnesses);
    native.claims[0][0].1 += EF::ONE;
    expect_malicious_reject(
        "recursive_native_opening_claim_tamper",
        "native.claims[0][0].1",
        try_build_n_whir_recursive_bundle_from_native(&fixture, native)
            .map(|_| ())
            .map_err(|err| err.to_string()),
    );

    recursive_bundle
        .verifier_key
        .outer_options
        .min_num_variables += 1;
    expect_malicious_reject(
        "recursive_mismatched_verifier_key",
        "verifier_key.outer_options.min_num_variables",
        try_verify_n_whir_recursive_bundle(&recursive_bundle),
    );
    recursive_bundle
        .verifier_key
        .outer_options
        .min_num_variables -= 1;

    let mut warp_bundle = build_warp_whir_root_bundle(&fixture, &witnesses);
    try_verify_warp_whir_root_bundle(&fixture, &warp_bundle)
        .expect("honest WARP root bundle must verify");

    let old_instance_mu = warp_bundle.instance.mu;
    warp_bundle.instance.mu += EF::ONE;
    expect_malicious_reject(
        "warp_instance_mu_tamper",
        "bundle.instance.mu",
        try_verify_warp_whir_root_bundle(&fixture, &warp_bundle),
    );
    warp_bundle.instance.mu = old_instance_mu;

    let old_step_instance_mu = warp_bundle.proof.steps[0].instance.mu;
    warp_bundle.proof.steps[0].instance.mu += EF::ONE;
    expect_malicious_reject(
        "warp_step_instance_mu_tamper",
        "bundle.proof.steps[0].instance.mu",
        try_verify_warp_whir_root_bundle(&fixture, &warp_bundle),
    );
    warp_bundle.proof.steps[0].instance.mu = old_step_instance_mu;

    let old_step_mu_final = warp_bundle.proof.steps[0].proof.mu_final;
    warp_bundle.proof.steps[0].proof.mu_final += EF::ONE;
    expect_malicious_reject(
        "warp_step_mu_final_tamper",
        "bundle.proof.steps[0].proof.mu_final",
        try_verify_warp_whir_root_bundle(&fixture, &warp_bundle),
    );
    warp_bundle.proof.steps[0].proof.mu_final = old_step_mu_final;

    let old_fresh_log_len = warp_bundle.proof.steps[0].fresh_commitments[0].0.log_len;
    warp_bundle.proof.steps[0].fresh_commitments[0].0.log_len += 1;
    expect_malicious_reject(
        "warp_root_commitment_metadata_tamper",
        "bundle.proof.steps[0].fresh_commitments[0].0.log_len",
        try_verify_warp_whir_root_bundle(&fixture, &warp_bundle),
    );
    warp_bundle.proof.steps[0].fresh_commitments[0].0.log_len = old_fresh_log_len;

    let old_terminal_value = warp_bundle.proof.final_proof.pesat.terminal_values[0];
    warp_bundle.proof.final_proof.pesat.terminal_values[0] += EF::ONE;
    expect_malicious_reject(
        "warp_finalizer_terminal_value_tamper",
        "bundle.proof.final_proof.pesat.terminal_values[0]",
        try_verify_warp_whir_root_bundle(&fixture, &warp_bundle),
    );
    warp_bundle.proof.final_proof.pesat.terminal_values[0] = old_terminal_value;

    let dropped_step = warp_bundle.proof.steps.remove(0);
    expect_malicious_reject(
        "warp_root_step_drop",
        "bundle.proof.steps.remove(0)",
        try_verify_warp_whir_root_bundle(&fixture, &warp_bundle),
    );
    warp_bundle.proof.steps.insert(0, dropped_step);

    let reorder_n = arity + (arity - 1);
    let reorder_witnesses = make_boolean_witnesses(num_variables, reorder_n);
    let mut reorder_bundle = build_warp_whir_root_bundle(&fixture, &reorder_witnesses);
    assert!(
        reorder_bundle.proof.steps.len() >= 2,
        "reorder malicious case needs at least two WARP steps"
    );
    reorder_bundle.proof.steps.swap(0, 1);
    expect_malicious_reject(
        "warp_root_step_reorder",
        "bundle.proof.steps.swap(0, 1)",
        try_verify_warp_whir_root_bundle(&fixture, &reorder_bundle),
    );

    let (root_round, root_eval) =
        first_whir_initial_sumcheck_eval_indices(&warp_bundle.root_iop_proof.opening.opening)
            .expect("root WHIR proof has initial sumcheck evaluations");
    let old_root_eval = warp_bundle
        .root_iop_proof
        .opening
        .opening
        .initial_sumcheck
        .polynomial_evaluations[root_round][root_eval];
    warp_bundle
        .root_iop_proof
        .opening
        .opening
        .initial_sumcheck
        .polynomial_evaluations[root_round][root_eval] += EF::ONE;
    expect_malicious_reject(
        "warp_root_whir_tamper",
        "bundle.root_iop_proof.opening.opening.initial_sumcheck.polynomial_evaluations[*][*]",
        try_verify_warp_whir_root_bundle(&fixture, &warp_bundle),
    );
    warp_bundle
        .root_iop_proof
        .opening
        .opening
        .initial_sumcheck
        .polynomial_evaluations[root_round][root_eval] = old_root_eval;
}

#[derive(Clone, Copy, Debug)]
struct DurationStats {
    min: Duration,
    median: Duration,
    mean: Duration,
    max: Duration,
    stddev: Duration,
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

fn duration_nanos(duration: Duration) -> u128 {
    duration.as_nanos()
}

fn reset_jsonl_output(path: &str) {
    File::create(path).unwrap_or_else(|err| panic!("create JSONL output `{path}`: {err}"));
}

fn append_jsonl_output(path: &str, line: &str) {
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .unwrap_or_else(|err| panic!("open JSONL output `{path}`: {err}"));
    writeln!(file, "{line}").unwrap_or_else(|err| panic!("write JSONL output `{path}`: {err}"));
}

#[allow(clippy::too_many_arguments)]
fn write_recursive_compare_jsonl(
    path: &str,
    num_variables: usize,
    n: usize,
    steps: usize,
    arity: usize,
    folding_factor: usize,
    outer_openings: usize,
    iterations: usize,
    warmup: usize,
    recursive_prove_stats: DurationStats,
    warp_prove_stats: DurationStats,
    recursive_verify_stats: DurationStats,
    warp_verify_stats: DurationStats,
    recursive_metrics: LaneMetrics,
    warp_metrics: LaneMetrics,
    recursive_phases: WhirRecursivePhaseDurations,
    warp_phases: &WarpWhirRootPhaseDurations,
    recursive_bundle: &WhirRecursiveBundle,
    warp_bundle: &WarpWhirRootBundle,
) {
    append_jsonl_output(
        path,
        &format!(
            concat!(
                "{{",
                "\"benchmark\":\"recursive_whir_vs_warp_whir_root\",",
                "\"k\":{},",
                "\"n\":{},",
                "\"steps\":{},",
                "\"arity\":{},",
                "\"whir_folding_factor\":{},",
                "\"recursive_outer_openings\":{},",
                "\"iterations\":{},",
                "\"warmup\":{},",
                "\"parallel_feature\":{},",
                "\"rayon_threads\":{},",
                "\"cpu_available_parallelism\":{},",
                "\"recursive_public_inputs\":{},",
                "\"recursive_private_inputs\":{},",
                "\"recursive_native_whir_proofs\":{},",
                "\"recursive_mmcs_ops\":{},",
                "\"warp_oracles\":{},",
                "\"warp_claims\":{},",
                "\"timing_nanos\":{{",
                "\"recursive_prove_median\":{},",
                "\"warp_prove_median\":{},",
                "\"recursive_verify_median\":{},",
                "\"warp_verify_median\":{}",
                "}},",
                "\"recursive_phases_nanos\":{{",
                "\"native_whir\":{},",
                "\"circuit_build\":{},",
                "\"trace_generation\":{},",
                "\"outer_proof\":{},",
                "\"total\":{}",
                "}},",
                "\"warp_phases_nanos\":{{",
                "\"setup\":{},",
                "\"encode\":{},",
                "\"shared_commit\":{},",
                "\"vacc_total\":{},",
                "\"dacc\":{},",
                "\"root_whir\":{},",
                "\"total\":{}",
                "}},",
                "\"recursive_metrics\":{{",
                "\"proof_bytes\":{},",
                "\"verifier_payload_bytes\":{},",
                "\"total_artifact_bytes\":{},",
                "\"commitments\":{},",
                "\"opening_claims\":{},",
                "\"whir_queries\":{},",
                "\"sumcheck_rounds\":{},",
                "\"table_count\":{},",
                "\"max_table_height\":{},",
                "\"total_table_cells\":{}",
                "}},",
                "\"warp_metrics\":{{",
                "\"proof_bytes\":{},",
                "\"verifier_payload_bytes\":{},",
                "\"total_artifact_bytes\":{},",
                "\"commitments\":{},",
                "\"opening_claims\":{},",
                "\"whir_queries\":{},",
                "\"sumcheck_rounds\":{}",
                "}}",
                "}}"
            ),
            num_variables,
            n,
            steps,
            arity,
            folding_factor,
            outer_openings,
            iterations,
            warmup,
            cfg!(feature = "parallel"),
            current_num_threads(),
            std::thread::available_parallelism().map_or(1, usize::from),
            recursive_bundle.recursive_public_inputs,
            recursive_bundle.recursive_private_inputs,
            recursive_bundle.native.proofs.len(),
            recursive_bundle.recursive_mmcs_ops,
            warp_bundle.oracle_count,
            warp_bundle.claim_count,
            duration_nanos(recursive_prove_stats.median),
            duration_nanos(warp_prove_stats.median),
            duration_nanos(recursive_verify_stats.median),
            duration_nanos(warp_verify_stats.median),
            duration_nanos(recursive_phases.native_whir),
            duration_nanos(recursive_phases.circuit_build),
            duration_nanos(recursive_phases.trace_generation),
            duration_nanos(recursive_phases.outer_proof),
            duration_nanos(recursive_phases.total),
            duration_nanos(warp_phases.setup),
            duration_nanos(warp_phases.encode),
            duration_nanos(warp_phases.shared_commit),
            duration_nanos(warp_phases.vacc_total),
            duration_nanos(warp_phases.dacc),
            duration_nanos(warp_phases.root_whir),
            duration_nanos(warp_phases.total),
            recursive_metrics.proof_bytes,
            recursive_metrics.verifier_payload_bytes,
            recursive_metrics.total_artifact_bytes,
            recursive_metrics.commitments,
            recursive_metrics.opening_claims,
            recursive_metrics.whir_queries,
            recursive_metrics.sumcheck_rounds,
            recursive_metrics.table_count,
            recursive_metrics.max_table_height,
            recursive_metrics.total_table_cells,
            warp_metrics.proof_bytes,
            warp_metrics.verifier_payload_bytes,
            warp_metrics.total_artifact_bytes,
            warp_metrics.commitments,
            warp_metrics.opening_claims,
            warp_metrics.whir_queries,
            warp_metrics.sumcheck_rounds,
        ),
    );
}

fn print_warp_whir_root_comparison(num_variable_cases: &[usize], n_values: &[usize]) {
    let iterations = parse_usize_env("P3_WARP_WHIR_ROOT_COMPARE_ITERS", 1).max(1);
    let warmup = parse_usize_env("P3_WARP_WHIR_ROOT_COMPARE_WARMUP", 1);
    let print_phases = env::var("P3_WARP_WHIR_ROOT_PHASES").as_deref() == Ok("1");
    let print_stats = iterations > 1 || env::var("P3_WARP_WHIR_ROOT_STATS").as_deref() == Ok("1");
    let arity = warp_fresh_per_step();
    let folding_factor = whir_folding_factor();
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
    eprintln!("    WHIR folding factor: {folding_factor} variables per folding round.");
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
            let whir_shape = whir_round_shape(&fixture.mmcs, num_variables);
            let whir_bytes = whir_full_bundle_bytes(&whir_bundle);
            let warp_bytes = warp_whir_root_bundle_bytes(&bundle);
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
                eprintln!(
                    "      WHIR shape: rounds={} final_sumcheck_rounds={} final_queries={}; serialized bytes: WHIR_artifact={} WARP_proof={} ({})",
                    whir_shape.rounds,
                    whir_shape.final_sumcheck_rounds,
                    whir_shape.final_queries,
                    whir_bytes,
                    warp_bytes,
                    format_warp_over_whir(
                        Duration::from_nanos(warp_bytes as u64),
                        Duration::from_nanos(whir_bytes as u64),
                    ),
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

fn print_recursive_whir_vs_warp_comparison(num_variable_cases: &[usize], n_values: &[usize]) {
    let iterations = parse_usize_env("P3_WHIR_RECURSIVE_COMPARE_ITERS", 1).max(1);
    let warmup = parse_usize_env("P3_WHIR_RECURSIVE_COMPARE_WARMUP", 0);
    let arity = warp_fresh_per_step();
    let folding_factor = whir_folding_factor();
    let outer_openings = whir_native_circuit_options().openings_per_table;
    let cpu_parallelism = std::thread::available_parallelism().map_or(1, usize::from);
    let require_full_soundness = env::var("P3_WHIR_REQUIRE_FULL_SOUNDNESS").as_deref() == Ok("1");
    let print_phases = env::var("P3_WHIR_RECURSIVE_PHASES").as_deref() == Ok("1");
    let json_path = env::var("P3_WHIR_RECURSIVE_JSON")
        .ok()
        .filter(|path| !path.trim().is_empty());
    let native_json_path = env::var("P3_WHIR_NATIVE_JSON")
        .ok()
        .filter(|path| !path.trim().is_empty());
    if let Some(path) = &json_path {
        reset_jsonl_output(path);
    }
    if let Some(path) = &native_json_path {
        reset_jsonl_output(path);
    }
    eprintln!();
    eprintln!("=== Recursive N-WHIR aggregate proof vs WHIR-backed WARP root comparison ===");
    eprintln!(
        "    Recursive lane: build N native WHIR proofs, run one verifier circuit for those N proofs, then use the WHIR-native outer path for its trace."
    );
    eprintln!(
        "    Outer recursive proof: WHIR-native table proof with table commitments, shifted Poseidon2 next-row columns, direct witness-table openings, local sumchecks, WHIR openings, and public/shape binding."
    );
    eprintln!("    WARP lane: existing WARP VACC/DACC root proof, kept outside recursion.");
    eprintln!(
        "    Comparison contract: both lanes use the same Boolean witnesses, k/N grid, BabyBear D4 field, WHIR folding factor, MMCS, challenger, and Reed-Solomon code."
    );
    eprintln!(
        "    Soundness: full benchmark path; no count-only WitnessChecks summary is accepted{}.",
        if require_full_soundness {
            " (required by P3_WHIR_REQUIRE_FULL_SOUNDNESS=1)"
        } else {
            ""
        }
    );
    eprintln!(
        "    WHIR folding factor: configured {folding_factor}, clamped to k for tiny smoke cases; recursive outer openings per table: {outer_openings}."
    );
    eprintln!(
        "    parallel_feature={} rayon_threads={} cpu_available_parallelism={}",
        cfg!(feature = "parallel"),
        current_num_threads(),
        cpu_parallelism,
    );
    eprintln!(
        "    Recursive prove samples include native WHIR proofs, verifier-circuit witness generation, and the outer WHIR-native proof."
    );
    eprintln!(
        "    Times are medians over {iterations} sample(s) after {warmup} warmup iteration(s); verifier timings are paired."
    );
    if print_phases {
        eprintln!(
            "    Phase timings are single setup runs: recursive native-WHIR/circuit/trace/outer-proof and WARP setup/encode/commit/VACC/DACC/root-WHIR."
        );
        eprintln!(
            "    Inner outer-proof breakdown: set P3_WHIR_NATIVE_PHASES=1 for per-phase/table WHIR-native timings; set P3_WHIR_NATIVE_JSON=/path/file.jsonl for JSONL rows."
        );
    }
    if let Some(path) = &json_path {
        eprintln!("    JSONL rows: {path}");
    }
    if let Some(path) = &native_json_path {
        eprintln!("    WHIR-native JSONL rows: {path}");
    }
    eprintln!(
        "{:<6}{:<8}{:<8}{:<18}{:<18}{:<24}{:<18}{:<18}{:<24}{:<14}{:<14}{:<14}",
        "k",
        "N",
        "steps",
        "recursive prove",
        "warp prove",
        "prove Δ",
        "recursive verify",
        "warp verify",
        "verify Δ",
        "rec proof B",
        "warp proof B",
        "mmcs ops",
    );

    for &num_variables in num_variable_cases {
        for &n in n_values {
            if n < arity || (n - arity) % (arity - 1) != 0 {
                eprintln!(
                    "{:<6}{:<8}{:<8}{:<18}{:<18}{:<24}{:<18}{:<18}{:<24}{:<14}{:<14}{:<14}",
                    num_variables,
                    n,
                    "-",
                    "skip",
                    "skip",
                    "-",
                    "skip",
                    "skip",
                    "invalid WARP N",
                    "-",
                    "-",
                    "-",
                );
                continue;
            }

            let fixture = make_warp_fixture(num_variables);
            let warp_witnesses = make_boolean_witnesses(num_variables, n);
            let steps = step_plan(n, arity).len();

            print_progress(format!(
                "    running k={num_variables}, N={n}: recursive verifier setup..."
            ));
            let (recursive_bundle, recursive_phases) =
                match try_build_n_whir_recursive_bundle_with_phases(&fixture, &warp_witnesses) {
                    Ok(bundle) => bundle,
                    Err(WhirNativeCircuitError::UnsupportedSoundComponent(message)) => {
                        eprintln!(
                            "{:<6}{:<8}{:<8}{:<18}{:<18}{:<24}{:<18}{:<18}{:<24}{:<14}{:<14}{:<14}",
                            num_variables,
                            n,
                            steps,
                            "unsupported",
                            "not run",
                            "-",
                            "unsupported",
                            "not run",
                            "see below",
                            "-",
                            "-",
                            "-",
                        );
                        eprintln!("      Recursive lane unsupported: {message}");
                        continue;
                    }
                    Err(err) => panic!("build WHIR-native recursive circuit proof failed: {err}"),
                };

            print_progress(format!(
                "    running k={num_variables}, N={n}: paired recursive-WHIR/WARP prover samples..."
            ));
            let (recursive_prove_stats, warp_prove_stats) = time_paired_stats(
                iterations,
                warmup,
                || {
                    prove_n_whir_recursive(&fixture, &warp_witnesses);
                },
                || {
                    prove_warp_whir_root(&fixture, &warp_witnesses);
                },
            );

            print_progress(format!(
                "    running k={num_variables}, N={n}: WARP verifier setup..."
            ));
            let (warp_bundle, _warp_phases) =
                build_warp_whir_root_bundle_with_phases(&fixture, &warp_witnesses);
            let recursive_metrics = recursive_lane_metrics(&recursive_bundle);
            let warp_metrics = warp_lane_metrics(&warp_bundle);
            let recursive_verify_phases = if print_phases {
                Some(
                    try_verify_n_whir_recursive_bundle_with_phases(&recursive_bundle)
                        .expect("recursive verify phase timing"),
                )
            } else {
                None
            };
            let warp_verify_phases = if print_phases {
                Some(verify_warp_whir_root_bundle_with_phases(
                    &fixture,
                    &warp_bundle,
                ))
            } else {
                None
            };

            print_progress(format!(
                "    running k={num_variables}, N={n}: paired recursive-WHIR/WARP verifier samples..."
            ));
            let (recursive_verify_stats, warp_verify_stats) = time_paired_stats(
                iterations,
                warmup,
                || {
                    verify_n_whir_recursive_bundle(&recursive_bundle);
                },
                || {
                    verify_warp_whir_root_bundle(&fixture, &warp_bundle);
                },
            );

            eprintln!(
                "{:<6}{:<8}{:<8}{:<18}{:<18}{:<24}{:<18}{:<18}{:<24}{:<14}{:<14}{:<14}",
                num_variables,
                n,
                steps,
                format_duration(recursive_prove_stats.median),
                format_duration(warp_prove_stats.median),
                format_warp_over_whir(warp_prove_stats.median, recursive_prove_stats.median),
                format_duration(recursive_verify_stats.median),
                format_duration(warp_verify_stats.median),
                format_warp_over_whir(warp_verify_stats.median, recursive_verify_stats.median),
                recursive_metrics.proof_bytes,
                warp_metrics.proof_bytes,
                recursive_bundle.recursive_mmcs_ops,
            );
            eprintln!(
                "      Recursive stats: public_inputs={} private_inputs={} native_whir_proofs={}",
                recursive_bundle.recursive_public_inputs,
                recursive_bundle.recursive_private_inputs,
                recursive_bundle.native.proofs.len(),
            );
            eprintln!(
                "      Recursive metrics: proof_bytes={} verifier_payload_bytes={} total_artifact_bytes={} commitments={} opening_claims={} whir_queries={} sumcheck_rounds={} table_count={} max_table_height={} total_table_cells={}",
                recursive_metrics.proof_bytes,
                recursive_metrics.verifier_payload_bytes,
                recursive_metrics.total_artifact_bytes,
                recursive_metrics.commitments,
                recursive_metrics.opening_claims,
                recursive_metrics.whir_queries,
                recursive_metrics.sumcheck_rounds,
                recursive_metrics.table_count,
                recursive_metrics.max_table_height,
                recursive_metrics.total_table_cells,
            );
            eprintln!(
                "      WARP metrics: proof_bytes={} verifier_payload_bytes={} total_artifact_bytes={} commitments={} opening_claims={} whir_queries={} sumcheck_rounds={}",
                warp_metrics.proof_bytes,
                warp_metrics.verifier_payload_bytes,
                warp_metrics.total_artifact_bytes,
                warp_metrics.commitments,
                warp_metrics.opening_claims,
                warp_metrics.whir_queries,
                warp_metrics.sumcheck_rounds,
            );
            if print_phases {
                let recursive_verify_phases = recursive_verify_phases
                    .as_ref()
                    .expect("recursive verify phases exist");
                let warp_verify_phases = warp_verify_phases
                    .as_ref()
                    .expect("warp verify phases exist");
                eprintln!(
                    "      Recursive phases: native_whir={} circuit_build={} trace_generation={} outer_proof={} total={}",
                    format_duration(recursive_phases.native_whir),
                    format_duration(recursive_phases.circuit_build),
                    format_duration(recursive_phases.trace_generation),
                    format_duration(recursive_phases.outer_proof),
                    format_duration(recursive_phases.total),
                );
                eprintln!(
                    "      Recursive verify phases: outer_verify={} total={}",
                    format_duration(recursive_verify_phases.outer_verify),
                    format_duration(recursive_verify_phases.total),
                );
                eprintln!(
                    "      WARP phases: setup={} encode={} shared_commit={} vacc_total={} dacc={} root_whir={} total={}",
                    format_duration(_warp_phases.setup),
                    format_duration(_warp_phases.encode),
                    format_duration(_warp_phases.shared_commit),
                    format_duration(_warp_phases.vacc_total),
                    format_duration(_warp_phases.dacc),
                    format_duration(_warp_phases.root_whir),
                    format_duration(_warp_phases.total),
                );
                eprintln!(
                    "      WARP verify phases: setup={} chain={} root_system_setup={} root_whir={} total={}",
                    format_duration(warp_verify_phases.setup),
                    format_duration(warp_verify_phases.chain),
                    format_duration(warp_verify_phases.root_system_setup),
                    format_duration(warp_verify_phases.root_whir),
                    format_duration(warp_verify_phases.total),
                );
            }
            if let Some(path) = &json_path {
                write_recursive_compare_jsonl(
                    path,
                    num_variables,
                    n,
                    steps,
                    arity,
                    folding_factor,
                    outer_openings,
                    iterations,
                    warmup,
                    recursive_prove_stats,
                    warp_prove_stats,
                    recursive_verify_stats,
                    warp_verify_stats,
                    recursive_metrics,
                    warp_metrics,
                    recursive_phases,
                    &_warp_phases,
                    &recursive_bundle,
                    &warp_bundle,
                );
            }
        }
    }
    eprintln!();
}

fn bench_sumcheck_like_prover(c: &mut Criterion) {
    if malicious_test_mode() {
        run_malicious_soundness_tests();
        return;
    }

    let num_variable_cases = parse_usize_list_env("P3_WARP_SUMCHECK_K", DEFAULT_NUM_VARIABLES);
    let n_values = parse_usize_list_env("P3_WARP_SUMCHECK_N", DEFAULT_N_VALUES);

    if env::var("P3_WARP_WHIR_ROOT_COMPARE").as_deref() == Ok("1") {
        print_warp_whir_root_comparison(&num_variable_cases, &n_values);
        return;
    }
    if env::var("P3_WHIR_RECURSIVE_COMPARE").as_deref() == Ok("1") {
        print_recursive_whir_vs_warp_comparison(&num_variable_cases, &n_values);
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

    for &num_variables in &num_variable_cases {
        let label = format!("k{num_variables}");
        for &n in &n_values {
            let classic_statements =
                make_whir_statements(num_variables, n, SumcheckStrategy::Classic);
            let svo_statements = make_whir_statements(num_variables, n, SumcheckStrategy::Svo);
            let warp_fixture = make_warp_fixture(num_variables);
            let whir_config = make_whir_config(&warp_fixture.mmcs, num_variables);
            let whir_protocol_params = make_whir_protocol_params(&warp_fixture.mmcs, num_variables);

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
        }
    }

    group.finish();
}

criterion_group!(benches, bench_sumcheck_like_prover);
criterion_main!(benches);
