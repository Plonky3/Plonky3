//! Kernel-style WARP benchmark mirroring `whir/benches/sumcheck.rs`.
//!
//! This is intentionally **not** an OpenVM/stark-backend benchmark and not a
//! succinct root-proof benchmark. It answers a smaller question:
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
//! The end-to-end comparison against real zkVM AIRs should use OpenVM/
//! stark-backend traces. This file exists so we can also inspect the algebraic
//! kernels without pulling in a full zkVM stack.

use std::cell::RefCell;
use std::env;
use std::hint::black_box;
use std::io::{self, Write};
#[cfg(feature = "stark-backend")]
use std::sync::Arc;
use std::time::{Duration, Instant};
use std::vec::Vec;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
#[cfg(feature = "stark-backend")]
use openvm_stark_backend::keygen::types::MultiStarkVerifyingKey;
#[cfg(feature = "stark-backend")]
use openvm_stark_backend::proof::Proof as OpenVmProof;
use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_baby_bear::{BabyBear, Poseidon2BabyBear, default_babybear_poseidon2_16};
#[cfg(feature = "stark-backend")]
use p3_challenger::CanFinalizeDigest;
use p3_challenger::{
    CanObserve, CanSample, CanSampleBits, DuplexChallenger, FieldChallenger, GrindingChallenger,
};
use p3_commit::MultilinearPcs;
use p3_dft::Radix2DFTSmallBatch;
#[cfg(feature = "stark-backend")]
use p3_field::BasedVectorSpace;
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
    AirAsPesat, BooleanPesat, ExtensionLimbPcs, ExtensionLimbPcsProof,
    ExternalCodewordBatchOpeningProver,
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
#[cfg(feature = "stark-backend")]
use p3_warp::{
    BackendWitnessFinalizer, StarkBackendAccumulatorBackend, StarkBackendAccumulatorClaim,
    StarkBackendAccumulatorOpeningProof, StarkBackendOpeningProof, StarkBackendOpeningVerifier,
    StarkBackendSegmentClaim, WarpExternalRootProof, WarpExternalRootReceipt,
    WarpStepSumcheckAirWitness, WitnessProof,
};
#[cfg(feature = "stark-backend")]
use p3_warp::{StarkBackendOpeningBackend, StarkBackendSegment};
#[cfg(feature = "stark-backend")]
use p3_warp::{
    binomial_bool_pesat_air, binomial_bool_pesat_air_context, binomial_mle_eval_air,
    binomial_mle_eval_air_context, binomial_rs_encoding_air, binomial_rs_encoding_air_context,
    binomial_sumcheck_air, binomial_sumcheck_air_context, root_receipt_air,
    root_receipt_air_context, root_receipt_proving_context,
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

#[cfg(feature = "stark-backend")]
use openvm_stark_backend::{
    AirRef, PartitionedBaseAir, StarkEngine, StarkProtocolConfig, SystemParams, WhirParams,
    WhirProximityStrategy,
    interaction::LogUpSecurityParameters,
    prover::{
        AirProvingContext, ColMajorMatrix, CpuColMajorBackend, DeviceDataTransporter,
        DeviceMultiStarkProvingKey, ProvingContext,
    },
};
#[cfg(feature = "stark-backend")]
use openvm_stark_sdk::config::baby_bear_poseidon2::{
    BabyBearPoseidon2Config as OpenVmSC, BabyBearPoseidon2CpuEngine,
};

type F = BabyBear;
type EF = BinomialExtensionField<F, 4>;
type Perm = Poseidon2BabyBear<16>;
type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
type PackedF = <F as Field>::Packing;
type MyMmcs = MerkleTreeMmcs<PackedF, PackedF, MyHash, MyCompress, 2, 8>;
type MyChallenger = DuplexChallenger<F, Perm, 16, 8>;
#[cfg(feature = "stark-backend")]
type MyChallengerDigest = <MyChallenger as CanFinalizeDigest>::Digest;
type MyDft = Radix2DFTSmallBatch<F>;
type MyWhirConfig = WhirConfig<EF, F, MyMmcs, MyChallenger>;
type MyWhirProof = WhirProof<F, EF, MyMmcs>;
type MyCommitment = MerkleCap<F, [F; WHIR_DIGEST_ELEMS]>;
#[cfg(feature = "stark-backend")]
type MyProof = <MyMmcs as p3_commit::Mmcs<F>>::Proof;
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
#[cfg(feature = "stark-backend")]
type OpenVmDigest = <OpenVmSC as StarkProtocolConfig>::Digest;
#[cfg(feature = "stark-backend")]
type OpenVmWarpExternalRootProof = WarpExternalRootProof<
    F,
    EF,
    MyCommitment,
    StarkBackendSegmentClaim<F, OpenVmDigest>,
    StarkBackendOpeningProof<F, OpenVmDigest>,
    MyProof,
    WitnessProof<EF>,
>;
#[cfg(feature = "stark-backend")]
type OpenVmWarpExternalRootReceipt = WarpExternalRootReceipt<
    EF,
    MyCommitment,
    StarkBackendSegmentClaim<F, OpenVmDigest>,
    MyChallengerDigest,
>;
#[cfg(feature = "stark-backend")]
type OpenVmWarpSwirlAccRootProof = WarpExternalRootProof<
    F,
    EF,
    StarkBackendAccumulatorClaim<OpenVmDigest>,
    StarkBackendSegmentClaim<F, OpenVmDigest>,
    StarkBackendOpeningProof<F, OpenVmDigest>,
    StarkBackendAccumulatorOpeningProof<EF, OpenVmDigest>,
    WitnessProof<EF>,
>;
#[cfg(feature = "stark-backend")]
type OpenVmWarpSwirlAccRootReceipt = WarpExternalRootReceipt<
    EF,
    StarkBackendAccumulatorClaim<OpenVmDigest>,
    StarkBackendSegmentClaim<F, OpenVmDigest>,
    MyChallengerDigest,
>;

const LOG_INV_RATE: usize = 1;
const DEFAULT_WARP_FRESH_PER_STEP: usize = 4;
const WHIR_FOLDING_FACTOR: usize = 4;
const WHIR_CONSTRAINTS: usize = 4;
const WHIR_DIGEST_ELEMS: usize = 8;
const DEFAULT_NUM_VARIABLES: &[usize] = &[14, 16, 18];
const DEFAULT_N_VALUES: &[usize] = &[4, 7, 13];

#[derive(Clone, Debug)]
#[cfg_attr(not(feature = "stark-backend"), allow(dead_code))]
struct BoolAir;

impl<FF: Field> BaseAir<FF> for BoolAir {
    fn width(&self) -> usize {
        1
    }

    fn num_constraints(&self) -> Option<usize> {
        Some(1)
    }

    fn max_constraint_degree(&self) -> Option<usize> {
        Some(2)
    }

    fn main_next_row_columns(&self) -> Vec<usize> {
        Vec::new()
    }
}

impl<AB> Air<AB> for BoolAir
where
    AB: AirBuilder,
    AB::F: Field,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let cell: AB::Expr = main.current_slice()[0].into();
        builder.assert_zero(cell.clone() * (cell - AB::Expr::ONE));
    }
}

#[cfg(feature = "stark-backend")]
impl<FF: Field> PartitionedBaseAir<FF> for BoolAir {}

type NativeBenchPesat = BooleanPesat<F, EF>;
#[cfg(feature = "stark-backend")]
type OpenVmBenchPesat = AirAsPesat<BoolAir, F, EF>;

struct WarpKernelFixture<P = NativeBenchPesat> {
    mmcs: MyMmcs,
    base_challenger: MyChallenger,
    code: ReedSolomonCode<F, MyDft>,
    pesat: P,
    params: WarpParams,
}

#[cfg(feature = "stark-backend")]
type OpenVmEngine = BabyBearPoseidon2CpuEngine;
#[cfg(feature = "stark-backend")]
type OpenVmDevicePk = DeviceMultiStarkProvingKey<<OpenVmEngine as StarkEngine>::PB>;

#[cfg(feature = "stark-backend")]
struct OpenVmWarpFixture {
    engine: OpenVmEngine,
    device_pk: OpenVmDevicePk,
    vk: MultiStarkVerifyingKey<OpenVmSC>,
    warp: WarpKernelFixture<OpenVmBenchPesat>,
}

#[cfg(feature = "stark-backend")]
struct OpenVmReceiptFixture {
    engine: OpenVmEngine,
    device_pk: OpenVmDevicePk,
    vk: MultiStarkVerifyingKey<OpenVmSC>,
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

#[cfg_attr(not(feature = "stark-backend"), allow(dead_code))]
fn make_bool_air_pesat(num_variables: usize, description: &[u8]) -> AirAsPesat<BoolAir, F, EF> {
    AirAsPesat::new(BoolAir, num_variables, description.to_vec())
}

fn warp_fresh_per_step() -> usize {
    let arity = parse_usize_env("P3_WARP_ARITY", DEFAULT_WARP_FRESH_PER_STEP);
    assert!(
        arity >= 2 && arity.is_power_of_two(),
        "P3_WARP_ARITY must be a power of two >= 2"
    );
    arity
}

#[cfg(feature = "stark-backend")]
fn make_warp_coefficient_fixture(num_variables: usize) -> WarpKernelFixture<OpenVmBenchPesat> {
    let perm = make_permutation();
    let mmcs = MyMmcs::new(MyHash::new(perm.clone()), MyCompress::new(perm.clone()), 0);
    let base_challenger = MyChallenger::new(perm);
    let dft = MyDft::default();
    let pesat = make_bool_air_pesat(num_variables, b"BoolAir/openvm-warp-external-bench");
    let code = ReedSolomonCode::<F, MyDft>::new_coefficient(num_variables, LOG_INV_RATE, dft);
    let params = WarpParams::new(1, 2);
    WarpKernelFixture {
        mmcs,
        base_challenger,
        code,
        pesat,
        params,
    }
}

#[cfg(feature = "stark-backend")]
fn make_openvm_params(num_variables: usize) -> SystemParams {
    make_openvm_params_with_max_degree(num_variables, 2)
}

#[cfg(feature = "stark-backend")]
fn make_openvm_params_with_max_degree(
    num_variables: usize,
    max_constraint_degree: usize,
) -> SystemParams {
    let l_skip = 0;
    let n_stack = num_variables;
    let log_blowup = LOG_INV_RATE;
    let k_whir = 1;
    let log_final_poly_len = 0;
    let log_stacked_height = l_skip + n_stack;
    SystemParams {
        l_skip,
        n_stack,
        w_stack: 1 << 12,
        log_blowup,
        whir: openvm_stark_backend::WhirConfig::new(
            log_blowup,
            log_stacked_height,
            WhirParams {
                k: k_whir,
                log_final_poly_len,
                query_phase_pow_bits: 1,
                folding_pow_bits: 2,
                mu_pow_bits: 3,
                proximity: WhirProximityStrategy::SplitUniqueList {
                    m: 3,
                    list_start_round: 1,
                },
            },
            5,
        ),
        logup: LogUpSecurityParameters {
            max_interaction_count: 1 << 30,
            log_max_message_length: 7,
            pow_bits: 2,
        },
        max_constraint_degree,
    }
}

#[cfg(feature = "stark-backend")]
fn make_openvm_warp_fixture(num_variables: usize) -> OpenVmWarpFixture {
    let engine = OpenVmEngine::new(make_openvm_params(num_variables));
    let airs = vec![Arc::new(BoolAir) as AirRef<OpenVmSC>];
    let (pk, vk) = engine.keygen(&airs);
    let device_pk = engine.device().transport_pk_to_device(&pk);
    OpenVmWarpFixture {
        engine,
        device_pk,
        vk,
        warp: make_warp_coefficient_fixture(num_variables),
    }
}

#[cfg(feature = "stark-backend")]
fn make_openvm_receipt_fixture(digest_len: usize) -> OpenVmReceiptFixture {
    let engine = OpenVmEngine::new(make_openvm_params(4));
    let airs = vec![root_receipt_air::<OpenVmSC>(digest_len)];
    let (pk, vk) = engine.keygen(&airs);
    let device_pk = engine.device().transport_pk_to_device(&pk);
    OpenVmReceiptFixture {
        engine,
        device_pk,
        vk,
    }
}

#[cfg(feature = "stark-backend")]
fn make_openvm_sumcheck_receipt_fixture(
    digest_len: usize,
    sumchecks: &[WarpStepSumcheckAirWitness<EF>],
) -> OpenVmReceiptFixture {
    let max_degree = sumchecks
        .iter()
        .flat_map(|witness| [witness.twin_degree, witness.batching_degree])
        .map(|degree| degree + 2)
        .max()
        .unwrap_or(2);
    let engine = OpenVmEngine::new(make_openvm_params_with_max_degree(4, max_degree));
    let mut airs = Vec::with_capacity(1 + 2 * sumchecks.len());
    airs.push(root_receipt_air::<OpenVmSC>(digest_len));
    for witness in sumchecks {
        airs.push(binomial_sumcheck_air::<F, OpenVmSC, 4>(
            witness.twin_round_challenges.len(),
            witness.twin_degree,
        ));
        airs.push(binomial_sumcheck_air::<F, OpenVmSC, 4>(
            witness.batching_round_challenges.len(),
            witness.batching_degree,
        ));
    }
    let (pk, vk) = engine.keygen(&airs);
    let device_pk = engine.device().transport_pk_to_device(&pk);
    OpenVmReceiptFixture {
        engine,
        device_pk,
        vk,
    }
}

#[cfg(feature = "stark-backend")]
fn make_openvm_sumcheck_mle_receipt_fixture(
    digest_len: usize,
    sumchecks: &[WarpStepSumcheckAirWitness<EF>],
    log_values_len: usize,
) -> OpenVmReceiptFixture {
    let max_degree = sumchecks
        .iter()
        .flat_map(|witness| [witness.twin_degree, witness.batching_degree])
        .map(|degree| degree + 2)
        .max()
        .unwrap_or(2)
        .max(3);
    let engine = OpenVmEngine::new(make_openvm_params_with_max_degree(
        log_values_len.max(4),
        max_degree,
    ));
    let mut airs = Vec::with_capacity(2 + 2 * sumchecks.len());
    airs.push(root_receipt_air::<OpenVmSC>(digest_len));
    for witness in sumchecks {
        airs.push(binomial_sumcheck_air::<F, OpenVmSC, 4>(
            witness.twin_round_challenges.len(),
            witness.twin_degree,
        ));
        airs.push(binomial_sumcheck_air::<F, OpenVmSC, 4>(
            witness.batching_round_challenges.len(),
            witness.batching_degree,
        ));
    }
    airs.push(binomial_mle_eval_air::<F, OpenVmSC, 4>(log_values_len));
    let (pk, vk) = engine.keygen(&airs);
    let device_pk = engine.device().transport_pk_to_device(&pk);
    OpenVmReceiptFixture {
        engine,
        device_pk,
        vk,
    }
}

#[cfg(feature = "stark-backend")]
fn make_openvm_sumcheck_mle_bool_receipt_fixture(
    digest_len: usize,
    sumchecks: &[WarpStepSumcheckAirWitness<EF>],
    log_codeword_len: usize,
    log_witness_len: usize,
) -> OpenVmReceiptFixture {
    let max_degree = sumchecks
        .iter()
        .flat_map(|witness| [witness.twin_degree, witness.batching_degree])
        .map(|degree| degree + 2)
        .max()
        .unwrap_or(2)
        .max(3);
    let engine = OpenVmEngine::new(make_openvm_params_with_max_degree(
        log_codeword_len.max(log_witness_len).max(4),
        max_degree,
    ));
    let mut airs = Vec::with_capacity(3 + 2 * sumchecks.len());
    airs.push(root_receipt_air::<OpenVmSC>(digest_len));
    for witness in sumchecks {
        airs.push(binomial_sumcheck_air::<F, OpenVmSC, 4>(
            witness.twin_round_challenges.len(),
            witness.twin_degree,
        ));
        airs.push(binomial_sumcheck_air::<F, OpenVmSC, 4>(
            witness.batching_round_challenges.len(),
            witness.batching_degree,
        ));
    }
    airs.push(binomial_mle_eval_air::<F, OpenVmSC, 4>(log_codeword_len));
    airs.push(binomial_bool_pesat_air::<F, OpenVmSC, 4>(log_witness_len));
    let (pk, vk) = engine.keygen(&airs);
    let device_pk = engine.device().transport_pk_to_device(&pk);
    OpenVmReceiptFixture {
        engine,
        device_pk,
        vk,
    }
}

#[cfg(feature = "stark-backend")]
fn make_openvm_finalizer_receipt_fixture(
    digest_len: usize,
    sumchecks: &[WarpStepSumcheckAirWitness<EF>],
    log_codeword_len: usize,
    log_witness_len: usize,
) -> OpenVmReceiptFixture {
    let max_degree = sumchecks
        .iter()
        .flat_map(|witness| [witness.twin_degree, witness.batching_degree])
        .map(|degree| degree + 2)
        .max()
        .unwrap_or(2)
        .max(3);
    let engine = OpenVmEngine::new(make_openvm_params_with_max_degree(
        log_codeword_len.max(log_witness_len).max(4),
        max_degree,
    ));
    let mut airs = Vec::with_capacity(4 + 2 * sumchecks.len());
    airs.push(root_receipt_air::<OpenVmSC>(digest_len));
    for witness in sumchecks {
        airs.push(binomial_sumcheck_air::<F, OpenVmSC, 4>(
            witness.twin_round_challenges.len(),
            witness.twin_degree,
        ));
        airs.push(binomial_sumcheck_air::<F, OpenVmSC, 4>(
            witness.batching_round_challenges.len(),
            witness.batching_degree,
        ));
    }
    airs.push(binomial_mle_eval_air::<F, OpenVmSC, 4>(log_codeword_len));
    airs.push(binomial_bool_pesat_air::<F, OpenVmSC, 4>(log_witness_len));
    airs.push(binomial_rs_encoding_air::<F, OpenVmSC, 4>(
        log_witness_len,
        log_codeword_len,
    ));
    let (pk, vk) = engine.keygen(&airs);
    let device_pk = engine.device().transport_pk_to_device(&pk);
    OpenVmReceiptFixture {
        engine,
        device_pk,
        vk,
    }
}

#[cfg(feature = "stark-backend")]
fn make_openvm_bool_ctx<SC: StarkProtocolConfig>(
    witness: &[SC::F],
) -> ProvingContext<CpuColMajorBackend<SC>> {
    let trace = RowMajorMatrix::new_col(witness.to_vec());
    ProvingContext::new(vec![(
        0,
        AirProvingContext::simple_no_pis(ColMajorMatrix::from_row_major(&trace)),
    )])
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

    let root_system = NativeWarpWhirRootProofSystem::new_with_base_message_pcs(
        &whir_pcs,
        &base_message_pcs,
        &fixture.code,
        BenchChallenger::new(),
    );
    root_system
        .verify(
            &root_iop_verifier.expected_inner_commitments(),
            &root_iop_verifier.expected_claims(),
            &bundle.root_iop_proof,
            &mut BenchChallenger::new(),
            0,
        )
        .expect("WARP root IOP WHIR proof verification");
}

#[cfg(feature = "stark-backend")]
fn build_n_openvm_whir_full(
    fixture: &OpenVmWarpFixture,
    witnesses: &[Vec<F>],
) -> Vec<OpenVmProof<OpenVmSC>> {
    witnesses
        .iter()
        .map(|witness| {
            let host_ctx = make_openvm_bool_ctx::<OpenVmSC>(witness);
            let device_ctx = fixture
                .engine
                .device()
                .transport_proving_ctx_to_device(&host_ctx);
            fixture
                .engine
                .prove(&fixture.device_pk, device_ctx)
                .expect("OpenVM/SWIRL full proof")
        })
        .collect()
}

#[cfg(feature = "stark-backend")]
fn prove_n_openvm_whir_full(fixture: &OpenVmWarpFixture, witnesses: &[Vec<F>]) {
    black_box(build_n_openvm_whir_full(fixture, witnesses));
}

#[cfg(feature = "stark-backend")]
fn verify_n_openvm_whir_full(fixture: &OpenVmWarpFixture, proofs: &[OpenVmProof<OpenVmSC>]) {
    for proof in proofs {
        fixture
            .engine
            .verify(&fixture.vk, proof)
            .expect("OpenVM/SWIRL full proof verification");
    }
}

#[cfg(feature = "stark-backend")]
fn build_openvm_warp_external_bundle(
    fixture: &OpenVmWarpFixture,
    witnesses: &[Vec<F>],
) -> OpenVmWarpExternalRootProof {
    let committed = witnesses
        .iter()
        .map(|witness| {
            let host_ctx = make_openvm_bool_ctx::<OpenVmSC>(witness);
            let device_ctx = fixture
                .engine
                .device()
                .transport_proving_ctx_to_device(&host_ctx);
            fixture
                .engine
                .commit_proving_context(&fixture.device_pk, device_ctx)
                .expect("OpenVM/SWIRL committed context")
        })
        .collect::<Vec<_>>();
    let segments = committed
        .iter()
        .map(StarkBackendSegment::new)
        .collect::<Vec<_>>();
    let root_prover = WarpRootProver::new(
        &fixture.warp.mmcs,
        &fixture.warp.code,
        &fixture.warp.pesat,
        fixture.warp.params,
    );
    let finalizer =
        WitnessFinalizer::new(&fixture.warp.mmcs, &fixture.warp.code, &fixture.warp.pesat);
    let opening_backend = StarkBackendOpeningBackend;

    let mut start = 0;
    let step_fresh_committed = step_plan(witnesses.len(), warp_fresh_per_step())
        .into_iter()
        .map(|fresh_count| {
            let end = start + fresh_count;
            let fresh = segments[start..end].to_vec();
            start = end;
            fresh
        })
        .collect::<Vec<_>>();

    root_prover
        .prove_external_linear_chain(
            &fixture.warp.base_challenger,
            &opening_backend,
            step_fresh_committed,
            &finalizer,
        )
        .expect("OpenVM/SWIRL external WARP root proof")
        .1
}

#[cfg(feature = "stark-backend")]
fn prove_openvm_warp_external(fixture: &OpenVmWarpFixture, witnesses: &[Vec<F>]) {
    black_box(build_openvm_warp_external_bundle(fixture, witnesses));
}

#[cfg(feature = "stark-backend")]
fn verify_openvm_warp_external_bundle(
    fixture: &OpenVmWarpFixture,
    bundle: &OpenVmWarpExternalRootProof,
) -> OpenVmWarpExternalRootReceipt {
    let verifier = WarpRootVerifier::new(
        &fixture.warp.mmcs,
        &fixture.warp.code,
        &fixture.warp.pesat,
        fixture.warp.params,
    );
    let fresh_verifier = StarkBackendOpeningVerifier::new(fixture.engine.config());
    let finalizer =
        WitnessFinalizer::new(&fixture.warp.mmcs, &fixture.warp.code, &fixture.warp.pesat);

    let receipt = verifier
        .verify_external_linear_chain_with_receipt(
            &fixture.warp.base_challenger,
            &fresh_verifier,
            bundle,
            &finalizer,
        )
        .expect("OpenVM/SWIRL external WARP root proof verification");
    black_box(receipt.clone())
}

#[cfg(feature = "stark-backend")]
fn build_openvm_warp_swirl_acc_bundle(
    fixture: &OpenVmWarpFixture,
    witnesses: &[Vec<F>],
) -> OpenVmWarpSwirlAccRootProof {
    let committed = witnesses
        .iter()
        .map(|witness| {
            let host_ctx = make_openvm_bool_ctx::<OpenVmSC>(witness);
            let device_ctx = fixture
                .engine
                .device()
                .transport_proving_ctx_to_device(&host_ctx);
            fixture
                .engine
                .commit_proving_context(&fixture.device_pk, device_ctx)
                .expect("OpenVM/SWIRL committed context")
        })
        .collect::<Vec<_>>();
    let segments = committed
        .iter()
        .map(StarkBackendSegment::new)
        .collect::<Vec<_>>();
    let root_prover = WarpRootProver::new(
        &fixture.warp.mmcs,
        &fixture.warp.code,
        &fixture.warp.pesat,
        fixture.warp.params,
    );
    let opening_backend = StarkBackendOpeningBackend;
    let acc_backend = StarkBackendAccumulatorBackend::new(fixture.engine.config());
    let finalizer =
        BackendWitnessFinalizer::<F, EF, MyDft, AirAsPesat<BoolAir, F, EF>, _, MyChallenger>::new(
            &acc_backend,
            &fixture.warp.code,
            &fixture.warp.pesat,
        );

    let mut start = 0;
    let step_fresh_committed = step_plan(witnesses.len(), warp_fresh_per_step())
        .into_iter()
        .map(|fresh_count| {
            let end = start + fresh_count;
            let fresh = segments[start..end].to_vec();
            start = end;
            fresh
        })
        .collect::<Vec<_>>();

    root_prover
        .prove_external_linear_chain_accumulator(
            &fixture.warp.base_challenger,
            &opening_backend,
            &acc_backend,
            step_fresh_committed,
            &finalizer,
        )
        .expect("OpenVM/SWIRL accumulator WARP root proof")
        .1
}

#[cfg(feature = "stark-backend")]
fn prove_openvm_warp_swirl_acc(fixture: &OpenVmWarpFixture, witnesses: &[Vec<F>]) {
    black_box(build_openvm_warp_swirl_acc_bundle(fixture, witnesses));
}

#[cfg(feature = "stark-backend")]
fn verify_openvm_warp_swirl_acc_bundle(
    fixture: &OpenVmWarpFixture,
    bundle: &OpenVmWarpSwirlAccRootProof,
) -> OpenVmWarpSwirlAccRootReceipt {
    let verifier = WarpRootVerifier::new(
        &fixture.warp.mmcs,
        &fixture.warp.code,
        &fixture.warp.pesat,
        fixture.warp.params,
    );
    let fresh_verifier = StarkBackendOpeningVerifier::new(fixture.engine.config());
    let acc_backend = StarkBackendAccumulatorBackend::new(fixture.engine.config());
    let finalizer =
        BackendWitnessFinalizer::<F, EF, MyDft, AirAsPesat<BoolAir, F, EF>, _, MyChallenger>::new(
            &acc_backend,
            &fixture.warp.code,
            &fixture.warp.pesat,
        );

    let receipt = verifier
        .verify_external_linear_chain_accumulator_with_receipt(
            &fixture.warp.base_challenger,
            &fresh_verifier,
            &acc_backend,
            bundle,
            &finalizer,
        )
        .expect("OpenVM/SWIRL accumulator WARP root proof verification");
    black_box(receipt.clone())
}

#[cfg(feature = "stark-backend")]
fn verify_openvm_warp_swirl_acc_bundle_with_sumchecks(
    fixture: &OpenVmWarpFixture,
    bundle: &OpenVmWarpSwirlAccRootProof,
) -> (
    OpenVmWarpSwirlAccRootReceipt,
    Vec<WarpStepSumcheckAirWitness<EF>>,
) {
    let verifier = WarpRootVerifier::new(
        &fixture.warp.mmcs,
        &fixture.warp.code,
        &fixture.warp.pesat,
        fixture.warp.params,
    );
    let fresh_verifier = StarkBackendOpeningVerifier::new(fixture.engine.config());
    let acc_backend = StarkBackendAccumulatorBackend::new(fixture.engine.config());
    let finalizer =
        BackendWitnessFinalizer::<F, EF, MyDft, AirAsPesat<BoolAir, F, EF>, _, MyChallenger>::new(
            &acc_backend,
            &fixture.warp.code,
            &fixture.warp.pesat,
        );

    verifier
        .verify_external_linear_chain_accumulator_with_sumcheck_witnesses(
            &fixture.warp.base_challenger,
            &fresh_verifier,
            &acc_backend,
            bundle,
            &finalizer,
        )
        .expect("OpenVM/SWIRL accumulator WARP root proof verification with sumcheck witnesses")
}

#[cfg(feature = "stark-backend")]
fn verify_openvm_warp_external_bundle_with_sumchecks(
    fixture: &OpenVmWarpFixture,
    bundle: &OpenVmWarpExternalRootProof,
) -> (
    OpenVmWarpExternalRootReceipt,
    Vec<WarpStepSumcheckAirWitness<EF>>,
) {
    let verifier = WarpRootVerifier::new(
        &fixture.warp.mmcs,
        &fixture.warp.code,
        &fixture.warp.pesat,
        fixture.warp.params,
    );
    let fresh_verifier = StarkBackendOpeningVerifier::new(fixture.engine.config());
    let finalizer =
        WitnessFinalizer::new(&fixture.warp.mmcs, &fixture.warp.code, &fixture.warp.pesat);

    verifier
        .verify_external_linear_chain_with_sumcheck_witnesses(
            &fixture.warp.base_challenger,
            &fresh_verifier,
            bundle,
            &finalizer,
        )
        .expect("OpenVM/SWIRL external WARP root proof verification with sumcheck witnesses")
}

#[cfg(feature = "stark-backend")]
fn build_openvm_warp_receipt_outer_proof(
    fixture: &OpenVmReceiptFixture,
    receipt: &OpenVmWarpExternalRootReceipt,
) -> OpenVmProof<OpenVmSC> {
    let host_ctx = root_receipt_proving_context::<OpenVmSC>(0, &receipt.claim_digest);
    let device_ctx = fixture
        .engine
        .device()
        .transport_proving_ctx_to_device(&host_ctx);
    fixture
        .engine
        .prove(&fixture.device_pk, device_ctx)
        .expect("OpenVM/SWIRL WARP root receipt proof")
}

#[cfg(feature = "stark-backend")]
fn build_openvm_warp_sumcheck_outer_proof(
    fixture: &OpenVmReceiptFixture,
    receipt: &OpenVmWarpExternalRootReceipt,
    bundle: &OpenVmWarpExternalRootProof,
    sumchecks: &[WarpStepSumcheckAirWitness<EF>],
) -> OpenVmProof<OpenVmSC> {
    assert_eq!(
        bundle.steps.len(),
        sumchecks.len(),
        "one sumcheck witness pair per WARP step",
    );
    let mut per_air = Vec::with_capacity(1 + 2 * sumchecks.len());
    per_air.push((
        0,
        root_receipt_air_context::<OpenVmSC>(&receipt.claim_digest),
    ));
    let mut air_id = 1;
    for (step, witness) in bundle.steps.iter().zip(sumchecks.iter()) {
        let (twin_ctx, twin_final) = binomial_sumcheck_air_context::<OpenVmSC, 4>(
            witness.twin_degree,
            witness.twin_initial_claim,
            &step.proof.twin_constraint_sumcheck,
            &witness.twin_round_challenges,
        );
        assert_eq!(twin_final, witness.twin_final_claim);
        per_air.push((air_id, twin_ctx));
        air_id += 1;

        let (batching_ctx, batching_final) = binomial_sumcheck_air_context::<OpenVmSC, 4>(
            witness.batching_degree,
            witness.batching_initial_claim,
            &step.proof.batching_sumcheck,
            &witness.batching_round_challenges,
        );
        assert_eq!(batching_final, witness.batching_final_claim);
        per_air.push((air_id, batching_ctx));
        air_id += 1;
    }

    let host_ctx = ProvingContext::new(per_air);
    let device_ctx = fixture
        .engine
        .device()
        .transport_proving_ctx_to_device(&host_ctx);
    fixture
        .engine
        .prove(&fixture.device_pk, device_ctx)
        .expect("OpenVM/SWIRL WARP root receipt + sumcheck proof")
}

#[cfg(feature = "stark-backend")]
fn build_openvm_warp_swirl_acc_sumcheck_outer_proof(
    fixture: &OpenVmReceiptFixture,
    receipt: &OpenVmWarpSwirlAccRootReceipt,
    bundle: &OpenVmWarpSwirlAccRootProof,
    sumchecks: &[WarpStepSumcheckAirWitness<EF>],
) -> OpenVmProof<OpenVmSC> {
    assert_eq!(
        bundle.steps.len(),
        sumchecks.len(),
        "one sumcheck witness pair per WARP step",
    );
    let mut per_air = Vec::with_capacity(1 + 2 * sumchecks.len());
    per_air.push((
        0,
        root_receipt_air_context::<OpenVmSC>(&receipt.claim_digest),
    ));
    let mut air_id = 1;
    for (step, witness) in bundle.steps.iter().zip(sumchecks.iter()) {
        let (twin_ctx, twin_final) = binomial_sumcheck_air_context::<OpenVmSC, 4>(
            witness.twin_degree,
            witness.twin_initial_claim,
            &step.proof.twin_constraint_sumcheck,
            &witness.twin_round_challenges,
        );
        assert_eq!(twin_final, witness.twin_final_claim);
        per_air.push((air_id, twin_ctx));
        air_id += 1;

        let (batching_ctx, batching_final) = binomial_sumcheck_air_context::<OpenVmSC, 4>(
            witness.batching_degree,
            witness.batching_initial_claim,
            &step.proof.batching_sumcheck,
            &witness.batching_round_challenges,
        );
        assert_eq!(batching_final, witness.batching_final_claim);
        per_air.push((air_id, batching_ctx));
        air_id += 1;
    }

    let host_ctx = ProvingContext::new(per_air);
    let device_ctx = fixture
        .engine
        .device()
        .transport_proving_ctx_to_device(&host_ctx);
    fixture
        .engine
        .prove(&fixture.device_pk, device_ctx)
        .expect("OpenVM/SWIRL WARP SWIRL-accumulator root receipt + sumcheck proof")
}

#[cfg(feature = "stark-backend")]
fn build_openvm_warp_swirl_acc_sumcheck_mle_outer_proof(
    fixture: &OpenVmReceiptFixture,
    receipt: &OpenVmWarpSwirlAccRootReceipt,
    bundle: &OpenVmWarpSwirlAccRootProof,
    sumchecks: &[WarpStepSumcheckAirWitness<EF>],
) -> OpenVmProof<OpenVmSC> {
    assert_eq!(
        bundle.steps.len(),
        sumchecks.len(),
        "one sumcheck witness pair per WARP step",
    );
    assert_eq!(
        bundle.final_proof.f.len(),
        1usize << receipt.final_instance.alpha.len(),
        "final witness length must match the public alpha dimension",
    );

    let mut per_air = Vec::with_capacity(2 + 2 * sumchecks.len());
    per_air.push((
        0,
        root_receipt_air_context::<OpenVmSC>(&receipt.claim_digest),
    ));
    let mut air_id = 1;
    for (step, witness) in bundle.steps.iter().zip(sumchecks.iter()) {
        let (twin_ctx, twin_final) = binomial_sumcheck_air_context::<OpenVmSC, 4>(
            witness.twin_degree,
            witness.twin_initial_claim,
            &step.proof.twin_constraint_sumcheck,
            &witness.twin_round_challenges,
        );
        assert_eq!(twin_final, witness.twin_final_claim);
        per_air.push((air_id, twin_ctx));
        air_id += 1;

        let (batching_ctx, batching_final) = binomial_sumcheck_air_context::<OpenVmSC, 4>(
            witness.batching_degree,
            witness.batching_initial_claim,
            &step.proof.batching_sumcheck,
            &witness.batching_round_challenges,
        );
        assert_eq!(batching_final, witness.batching_final_claim);
        per_air.push((air_id, batching_ctx));
        air_id += 1;
    }

    let (mle_ctx, mle_claim) = binomial_mle_eval_air_context::<OpenVmSC, 4>(
        &bundle.final_proof.f,
        &receipt.final_instance.alpha,
    );
    assert_eq!(
        mle_claim, receipt.final_instance.mu,
        "final MLE AIR must prove f_hat(alpha) = mu",
    );
    per_air.push((air_id, mle_ctx));

    let host_ctx = ProvingContext::new(per_air);
    let device_ctx = fixture
        .engine
        .device()
        .transport_proving_ctx_to_device(&host_ctx);
    fixture
        .engine
        .prove(&fixture.device_pk, device_ctx)
        .expect("OpenVM/SWIRL WARP SWIRL-accumulator root receipt + sumcheck + MLE proof")
}

#[cfg(feature = "stark-backend")]
fn build_openvm_warp_swirl_acc_sumcheck_mle_bool_outer_proof(
    fixture: &OpenVmReceiptFixture,
    receipt: &OpenVmWarpSwirlAccRootReceipt,
    bundle: &OpenVmWarpSwirlAccRootProof,
    sumchecks: &[WarpStepSumcheckAirWitness<EF>],
) -> OpenVmProof<OpenVmSC> {
    assert_eq!(
        bundle.steps.len(),
        sumchecks.len(),
        "one sumcheck witness pair per WARP step",
    );
    assert_eq!(
        bundle.final_proof.f.len(),
        1usize << receipt.final_instance.alpha.len(),
        "final codeword length must match the public alpha dimension",
    );
    assert_eq!(
        bundle.final_proof.w.len(),
        1usize << receipt.final_instance.beta.len(),
        "BoolAir final witness length must match the public beta dimension",
    );

    let mut per_air = Vec::with_capacity(3 + 2 * sumchecks.len());
    per_air.push((
        0,
        root_receipt_air_context::<OpenVmSC>(&receipt.claim_digest),
    ));
    let mut air_id = 1;
    for (step, witness) in bundle.steps.iter().zip(sumchecks.iter()) {
        let (twin_ctx, twin_final) = binomial_sumcheck_air_context::<OpenVmSC, 4>(
            witness.twin_degree,
            witness.twin_initial_claim,
            &step.proof.twin_constraint_sumcheck,
            &witness.twin_round_challenges,
        );
        assert_eq!(twin_final, witness.twin_final_claim);
        per_air.push((air_id, twin_ctx));
        air_id += 1;

        let (batching_ctx, batching_final) = binomial_sumcheck_air_context::<OpenVmSC, 4>(
            witness.batching_degree,
            witness.batching_initial_claim,
            &step.proof.batching_sumcheck,
            &witness.batching_round_challenges,
        );
        assert_eq!(batching_final, witness.batching_final_claim);
        per_air.push((air_id, batching_ctx));
        air_id += 1;
    }

    let (mle_ctx, mle_claim) = binomial_mle_eval_air_context::<OpenVmSC, 4>(
        &bundle.final_proof.f,
        &receipt.final_instance.alpha,
    );
    assert_eq!(
        mle_claim, receipt.final_instance.mu,
        "final MLE AIR must prove f_hat(alpha) = mu",
    );
    per_air.push((air_id, mle_ctx));
    air_id += 1;

    let (bool_ctx, bool_eta) = binomial_bool_pesat_air_context::<OpenVmSC, 4>(
        &bundle.final_proof.w,
        &receipt.final_instance.beta,
    );
    assert_eq!(
        bool_eta, receipt.final_instance.eta,
        "final Boolean PESAT AIR must prove Pb(beta, w) = eta",
    );
    per_air.push((air_id, bool_ctx));

    let host_ctx = ProvingContext::new(per_air);
    let device_ctx = fixture
        .engine
        .device()
        .transport_proving_ctx_to_device(&host_ctx);
    fixture.engine.prove(&fixture.device_pk, device_ctx).expect(
        "OpenVM/SWIRL WARP SWIRL-accumulator root receipt + sumcheck + MLE + Bool PESAT proof",
    )
}

#[cfg(feature = "stark-backend")]
fn build_openvm_warp_swirl_acc_finalizer_outer_proof(
    fixture: &OpenVmReceiptFixture,
    receipt: &OpenVmWarpSwirlAccRootReceipt,
    bundle: &OpenVmWarpSwirlAccRootProof,
    sumchecks: &[WarpStepSumcheckAirWitness<EF>],
) -> OpenVmProof<OpenVmSC> {
    assert_eq!(
        bundle.steps.len(),
        sumchecks.len(),
        "one sumcheck witness pair per WARP step",
    );
    assert_eq!(
        bundle.final_proof.f.len(),
        1usize << receipt.final_instance.alpha.len(),
        "final codeword length must match the public alpha dimension",
    );
    assert_eq!(
        bundle.final_proof.w.len(),
        1usize << receipt.final_instance.beta.len(),
        "BoolAir final witness length must match the public beta dimension",
    );

    let mut per_air = Vec::with_capacity(4 + 2 * sumchecks.len());
    per_air.push((
        0,
        root_receipt_air_context::<OpenVmSC>(&receipt.claim_digest),
    ));
    let mut air_id = 1;
    for (step, witness) in bundle.steps.iter().zip(sumchecks.iter()) {
        let (twin_ctx, twin_final) = binomial_sumcheck_air_context::<OpenVmSC, 4>(
            witness.twin_degree,
            witness.twin_initial_claim,
            &step.proof.twin_constraint_sumcheck,
            &witness.twin_round_challenges,
        );
        assert_eq!(twin_final, witness.twin_final_claim);
        per_air.push((air_id, twin_ctx));
        air_id += 1;

        let (batching_ctx, batching_final) = binomial_sumcheck_air_context::<OpenVmSC, 4>(
            witness.batching_degree,
            witness.batching_initial_claim,
            &step.proof.batching_sumcheck,
            &witness.batching_round_challenges,
        );
        assert_eq!(batching_final, witness.batching_final_claim);
        per_air.push((air_id, batching_ctx));
        air_id += 1;
    }

    let (mle_ctx, mle_claim) = binomial_mle_eval_air_context::<OpenVmSC, 4>(
        &bundle.final_proof.f,
        &receipt.final_instance.alpha,
    );
    assert_eq!(
        mle_claim, receipt.final_instance.mu,
        "final MLE AIR must prove f_hat(alpha) = mu",
    );
    per_air.push((air_id, mle_ctx));
    air_id += 1;

    let (bool_ctx, bool_eta) = binomial_bool_pesat_air_context::<OpenVmSC, 4>(
        &bundle.final_proof.w,
        &receipt.final_instance.beta,
    );
    assert_eq!(
        bool_eta, receipt.final_instance.eta,
        "final Boolean PESAT AIR must prove Pb(beta, w) = eta",
    );
    per_air.push((air_id, bool_ctx));
    air_id += 1;

    let z = rs_encoding_check_point(receipt);
    let rs_ctx = binomial_rs_encoding_air_context::<OpenVmSC, 4>(
        &bundle.final_proof.w,
        &bundle.final_proof.f,
        z,
    );
    per_air.push((air_id, rs_ctx));

    let host_ctx = ProvingContext::new(per_air);
    let device_ctx = fixture
        .engine
        .device()
        .transport_proving_ctx_to_device(&host_ctx);
    fixture
        .engine
        .prove(&fixture.device_pk, device_ctx)
        .expect("OpenVM/SWIRL WARP finalizer arithmetic + RS encoding proof")
}

#[cfg(feature = "stark-backend")]
fn prove_openvm_warp_receipt_outer(
    fixture: &OpenVmReceiptFixture,
    receipt: &OpenVmWarpExternalRootReceipt,
) {
    black_box(build_openvm_warp_receipt_outer_proof(fixture, receipt));
}

#[cfg(feature = "stark-backend")]
fn prove_openvm_warp_sumcheck_outer(
    fixture: &OpenVmReceiptFixture,
    receipt: &OpenVmWarpExternalRootReceipt,
    bundle: &OpenVmWarpExternalRootProof,
    sumchecks: &[WarpStepSumcheckAirWitness<EF>],
) {
    black_box(build_openvm_warp_sumcheck_outer_proof(
        fixture, receipt, bundle, sumchecks,
    ));
}

#[cfg(feature = "stark-backend")]
fn prove_openvm_warp_swirl_acc_sumcheck_outer(
    fixture: &OpenVmReceiptFixture,
    receipt: &OpenVmWarpSwirlAccRootReceipt,
    bundle: &OpenVmWarpSwirlAccRootProof,
    sumchecks: &[WarpStepSumcheckAirWitness<EF>],
) {
    black_box(build_openvm_warp_swirl_acc_sumcheck_outer_proof(
        fixture, receipt, bundle, sumchecks,
    ));
}

#[cfg(feature = "stark-backend")]
fn prove_openvm_warp_swirl_acc_sumcheck_mle_outer(
    fixture: &OpenVmReceiptFixture,
    receipt: &OpenVmWarpSwirlAccRootReceipt,
    bundle: &OpenVmWarpSwirlAccRootProof,
    sumchecks: &[WarpStepSumcheckAirWitness<EF>],
) {
    black_box(build_openvm_warp_swirl_acc_sumcheck_mle_outer_proof(
        fixture, receipt, bundle, sumchecks,
    ));
}

#[cfg(feature = "stark-backend")]
fn prove_openvm_warp_swirl_acc_sumcheck_mle_bool_outer(
    fixture: &OpenVmReceiptFixture,
    receipt: &OpenVmWarpSwirlAccRootReceipt,
    bundle: &OpenVmWarpSwirlAccRootProof,
    sumchecks: &[WarpStepSumcheckAirWitness<EF>],
) {
    black_box(build_openvm_warp_swirl_acc_sumcheck_mle_bool_outer_proof(
        fixture, receipt, bundle, sumchecks,
    ));
}

#[cfg(feature = "stark-backend")]
fn prove_openvm_warp_swirl_acc_finalizer_outer(
    fixture: &OpenVmReceiptFixture,
    receipt: &OpenVmWarpSwirlAccRootReceipt,
    bundle: &OpenVmWarpSwirlAccRootProof,
    sumchecks: &[WarpStepSumcheckAirWitness<EF>],
) {
    black_box(build_openvm_warp_swirl_acc_finalizer_outer_proof(
        fixture, receipt, bundle, sumchecks,
    ));
}

#[cfg(feature = "stark-backend")]
fn prove_openvm_warp_swirl_acc_finalizer_pipeline(
    openvm_fixture: &OpenVmWarpFixture,
    finalizer_fixture: &OpenVmReceiptFixture,
    witnesses: &[Vec<F>],
) {
    let bundle = build_openvm_warp_swirl_acc_bundle(openvm_fixture, witnesses);
    let (receipt, sumchecks) =
        verify_openvm_warp_swirl_acc_bundle_with_sumchecks(openvm_fixture, &bundle);
    black_box(build_openvm_warp_swirl_acc_finalizer_outer_proof(
        finalizer_fixture,
        &receipt,
        &bundle,
        &sumchecks,
    ));
}

#[cfg(feature = "stark-backend")]
fn verify_openvm_warp_receipt_outer(
    fixture: &OpenVmReceiptFixture,
    receipt: &OpenVmWarpExternalRootReceipt,
    proof: &OpenVmProof<OpenVmSC>,
) {
    assert_eq!(
        proof.public_values.len(),
        1,
        "root receipt outer proof must expose one AIR public-value vector",
    );
    assert_eq!(
        proof.public_values[0], receipt.claim_digest,
        "root receipt outer proof public values must match the native WARP root claim digest",
    );
    fixture
        .engine
        .verify(&fixture.vk, proof)
        .expect("OpenVM/SWIRL WARP root receipt proof verification");
}

#[cfg(feature = "stark-backend")]
fn verify_openvm_warp_sumcheck_outer(
    fixture: &OpenVmReceiptFixture,
    receipt: &OpenVmWarpExternalRootReceipt,
    sumchecks: &[WarpStepSumcheckAirWitness<EF>],
    proof: &OpenVmProof<OpenVmSC>,
) {
    assert_eq!(
        proof.public_values.len(),
        1 + 2 * sumchecks.len(),
        "root sumcheck outer proof must expose one receipt AIR and two sumcheck AIRs per WARP step",
    );
    assert_eq!(
        proof.public_values[0], receipt.claim_digest,
        "root receipt public values must match the native WARP root claim digest",
    );
    let mut air_id = 1;
    for witness in sumchecks {
        assert_eq!(
            proof.public_values[air_id],
            sumcheck_public_values(
                witness.twin_initial_claim,
                witness.twin_final_claim,
                &witness.twin_round_challenges,
            ),
            "twin sumcheck public values must match transcript-derived challenges",
        );
        air_id += 1;
        assert_eq!(
            proof.public_values[air_id],
            sumcheck_public_values(
                witness.batching_initial_claim,
                witness.batching_final_claim,
                &witness.batching_round_challenges,
            ),
            "batching sumcheck public values must match transcript-derived challenges",
        );
        air_id += 1;
    }
    fixture
        .engine
        .verify(&fixture.vk, proof)
        .expect("OpenVM/SWIRL WARP root receipt + sumcheck proof verification");
}

#[cfg(feature = "stark-backend")]
fn verify_openvm_warp_swirl_acc_sumcheck_outer(
    fixture: &OpenVmReceiptFixture,
    receipt: &OpenVmWarpSwirlAccRootReceipt,
    sumchecks: &[WarpStepSumcheckAirWitness<EF>],
    proof: &OpenVmProof<OpenVmSC>,
) {
    assert_eq!(
        proof.public_values.len(),
        1 + 2 * sumchecks.len(),
        "root sumcheck outer proof must expose one receipt AIR and two sumcheck AIRs per WARP step",
    );
    assert_eq!(
        proof.public_values[0], receipt.claim_digest,
        "root receipt public values must match the native WARP root claim digest",
    );
    let mut air_id = 1;
    for witness in sumchecks {
        assert_eq!(
            proof.public_values[air_id],
            sumcheck_public_values(
                witness.twin_initial_claim,
                witness.twin_final_claim,
                &witness.twin_round_challenges,
            ),
            "twin sumcheck public values must match transcript-derived challenges",
        );
        air_id += 1;
        assert_eq!(
            proof.public_values[air_id],
            sumcheck_public_values(
                witness.batching_initial_claim,
                witness.batching_final_claim,
                &witness.batching_round_challenges,
            ),
            "batching sumcheck public values must match transcript-derived challenges",
        );
        air_id += 1;
    }
    fixture
        .engine
        .verify(&fixture.vk, proof)
        .expect("OpenVM/SWIRL WARP SWIRL-accumulator root receipt + sumcheck proof verification");
}

#[cfg(feature = "stark-backend")]
fn verify_openvm_warp_swirl_acc_sumcheck_mle_outer(
    fixture: &OpenVmReceiptFixture,
    receipt: &OpenVmWarpSwirlAccRootReceipt,
    sumchecks: &[WarpStepSumcheckAirWitness<EF>],
    proof: &OpenVmProof<OpenVmSC>,
) {
    assert_eq!(
        proof.public_values.len(),
        2 + 2 * sumchecks.len(),
        "root sumcheck+MLE outer proof must expose receipt, two sumchecks per step, and one final MLE AIR",
    );
    assert_eq!(
        proof.public_values[0], receipt.claim_digest,
        "root receipt public values must match the native WARP root claim digest",
    );
    let mut air_id = 1;
    for witness in sumchecks {
        assert_eq!(
            proof.public_values[air_id],
            sumcheck_public_values(
                witness.twin_initial_claim,
                witness.twin_final_claim,
                &witness.twin_round_challenges,
            ),
            "twin sumcheck public values must match transcript-derived challenges",
        );
        air_id += 1;
        assert_eq!(
            proof.public_values[air_id],
            sumcheck_public_values(
                witness.batching_initial_claim,
                witness.batching_final_claim,
                &witness.batching_round_challenges,
            ),
            "batching sumcheck public values must match transcript-derived challenges",
        );
        air_id += 1;
    }
    assert_eq!(
        proof.public_values[air_id],
        mle_eval_public_values(&receipt.final_instance.alpha, receipt.final_instance.mu),
        "final MLE public values must match the receipt's alpha and mu",
    );
    fixture.engine.verify(&fixture.vk, proof).expect(
        "OpenVM/SWIRL WARP SWIRL-accumulator root receipt + sumcheck + MLE proof verification",
    );
}

#[cfg(feature = "stark-backend")]
fn verify_openvm_warp_swirl_acc_sumcheck_mle_bool_outer(
    fixture: &OpenVmReceiptFixture,
    receipt: &OpenVmWarpSwirlAccRootReceipt,
    sumchecks: &[WarpStepSumcheckAirWitness<EF>],
    proof: &OpenVmProof<OpenVmSC>,
) {
    assert_eq!(
        proof.public_values.len(),
        3 + 2 * sumchecks.len(),
        "root finalizer outer proof must expose receipt, two sumchecks per step, final MLE, and final Bool PESAT",
    );
    assert_eq!(
        proof.public_values[0], receipt.claim_digest,
        "root receipt public values must match the native WARP root claim digest",
    );
    let mut air_id = 1;
    for witness in sumchecks {
        assert_eq!(
            proof.public_values[air_id],
            sumcheck_public_values(
                witness.twin_initial_claim,
                witness.twin_final_claim,
                &witness.twin_round_challenges,
            ),
            "twin sumcheck public values must match transcript-derived challenges",
        );
        air_id += 1;
        assert_eq!(
            proof.public_values[air_id],
            sumcheck_public_values(
                witness.batching_initial_claim,
                witness.batching_final_claim,
                &witness.batching_round_challenges,
            ),
            "batching sumcheck public values must match transcript-derived challenges",
        );
        air_id += 1;
    }
    assert_eq!(
        proof.public_values[air_id],
        mle_eval_public_values(&receipt.final_instance.alpha, receipt.final_instance.mu),
        "final MLE public values must match the receipt's alpha and mu",
    );
    air_id += 1;
    assert_eq!(
        proof.public_values[air_id],
        bool_pesat_public_values(&receipt.final_instance.beta, receipt.final_instance.eta),
        "final Bool PESAT public values must match the receipt's beta and eta",
    );
    fixture
        .engine
        .verify(&fixture.vk, proof)
        .expect("OpenVM/SWIRL WARP finalizer arithmetic proof verification");
}

#[cfg(feature = "stark-backend")]
fn verify_openvm_warp_swirl_acc_finalizer_outer(
    fixture: &OpenVmReceiptFixture,
    receipt: &OpenVmWarpSwirlAccRootReceipt,
    sumchecks: &[WarpStepSumcheckAirWitness<EF>],
    proof: &OpenVmProof<OpenVmSC>,
) {
    assert_eq!(
        proof.public_values.len(),
        4 + 2 * sumchecks.len(),
        "root finalizer outer proof must expose receipt, two sumchecks per step, final MLE, final Bool PESAT, and RS encoding check",
    );
    assert_eq!(
        proof.public_values[0], receipt.claim_digest,
        "root receipt public values must match the native WARP root claim digest",
    );
    let mut air_id = 1;
    for witness in sumchecks {
        assert_eq!(
            proof.public_values[air_id],
            sumcheck_public_values(
                witness.twin_initial_claim,
                witness.twin_final_claim,
                &witness.twin_round_challenges,
            ),
            "twin sumcheck public values must match transcript-derived challenges",
        );
        air_id += 1;
        assert_eq!(
            proof.public_values[air_id],
            sumcheck_public_values(
                witness.batching_initial_claim,
                witness.batching_final_claim,
                &witness.batching_round_challenges,
            ),
            "batching sumcheck public values must match transcript-derived challenges",
        );
        air_id += 1;
    }
    assert_eq!(
        proof.public_values[air_id],
        mle_eval_public_values(&receipt.final_instance.alpha, receipt.final_instance.mu),
        "final MLE public values must match the receipt's alpha and mu",
    );
    air_id += 1;
    assert_eq!(
        proof.public_values[air_id],
        bool_pesat_public_values(&receipt.final_instance.beta, receipt.final_instance.eta),
        "final Bool PESAT public values must match the receipt's beta and eta",
    );
    air_id += 1;
    assert_eq!(
        proof.public_values[air_id],
        rs_encoding_public_values(rs_encoding_check_point(receipt)),
        "RS encoding public value must match the transcript-derived point",
    );
    fixture
        .engine
        .verify(&fixture.vk, proof)
        .expect("OpenVM/SWIRL WARP finalizer arithmetic + RS encoding proof verification");
}

#[cfg(feature = "stark-backend")]
fn sumcheck_public_values(initial: EF, final_claim: EF, challenges: &[EF]) -> Vec<F> {
    let mut values = Vec::with_capacity(4 * (challenges.len() + 2));
    values.extend_from_slice(initial.as_basis_coefficients_slice());
    values.extend_from_slice(final_claim.as_basis_coefficients_slice());
    for challenge in challenges {
        values.extend_from_slice(challenge.as_basis_coefficients_slice());
    }
    values
}

#[cfg(feature = "stark-backend")]
fn mle_eval_public_values(point: &[EF], claim: EF) -> Vec<F> {
    let mut values = Vec::with_capacity(4 * (point.len() + 1));
    for challenge in point {
        values.extend_from_slice(challenge.as_basis_coefficients_slice());
    }
    values.extend_from_slice(claim.as_basis_coefficients_slice());
    values
}

#[cfg(feature = "stark-backend")]
fn bool_pesat_public_values(point: &[EF], eta: EF) -> Vec<F> {
    let mut values = Vec::with_capacity(4 * (point.len() + 1));
    for challenge in point {
        values.extend_from_slice(challenge.as_basis_coefficients_slice());
    }
    values.extend_from_slice(eta.as_basis_coefficients_slice());
    values
}

#[cfg(feature = "stark-backend")]
fn rs_encoding_public_values(z: EF) -> Vec<F> {
    z.as_basis_coefficients_slice().to_vec()
}

#[cfg(feature = "stark-backend")]
fn rs_encoding_check_point(receipt: &OpenVmWarpSwirlAccRootReceipt) -> EF {
    let mut challenger = make_challenger();
    challenger.observe_slice(&receipt.claim_digest);
    challenger.observe_algebra_element(receipt.final_instance.mu);
    challenger.observe_algebra_element(receipt.final_instance.eta);
    for &alpha in &receipt.final_instance.alpha {
        challenger.observe_algebra_element(alpha);
    }
    for &beta in &receipt.final_instance.beta {
        challenger.observe_algebra_element(beta);
    }
    let mut z: EF = challenger.sample_algebra_element();
    while is_in_base_rs_domain(z, receipt.final_instance.alpha.len()) {
        challenger.observe(F::ONE);
        z = challenger.sample_algebra_element();
    }
    z
}

#[cfg(feature = "stark-backend")]
fn is_in_base_rs_domain(z: EF, log_domain_len: usize) -> bool {
    let limbs: &[F] = z.as_basis_coefficients_slice();
    if limbs[1..].iter().any(|&limb| limb != F::ZERO) {
        return false;
    }
    limbs[0].exp_power_of_2(log_domain_len) == F::ONE
}

#[cfg(feature = "stark-backend")]
fn prove_and_verify_openvm_warp_external(fixture: &OpenVmWarpFixture, witnesses: &[Vec<F>]) {
    let bundle = build_openvm_warp_external_bundle(fixture, witnesses);
    let receipt = verify_openvm_warp_external_bundle(fixture, &bundle);
    black_box(receipt);
    black_box(bundle);
}

#[cfg(feature = "stark-backend")]
fn prove_and_verify_n_openvm_whir_full(fixture: &OpenVmWarpFixture, witnesses: &[Vec<F>]) {
    let proofs = build_n_openvm_whir_full(fixture, witnesses);
    verify_n_openvm_whir_full(fixture, &proofs);
    black_box(proofs);
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

fn time_average(iterations: usize, mut f: impl FnMut()) -> Duration {
    let start = Instant::now();
    for _ in 0..iterations {
        f();
    }
    average_duration(start.elapsed(), iterations)
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
        "    WARP lane: native WARP accumulation steps, then direct Boolean WhirWarpFinalizerProtocol over ExtensionLimbPcs<WhirPcs>."
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
    let print_phases = env::var("P3_WARP_WHIR_ROOT_PHASES").as_deref() == Ok("1");
    let arity = warp_fresh_per_step();
    eprintln!();
    eprintln!("=== WHIR-backed WARP root vs N full WHIR PCS comparison ===");
    eprintln!("    WHIR lane: N full WhirPcs commit+open proofs and WhirPcs verifications.");
    eprintln!(
        "    WARP lane: WARP VACC/DACC root recorder + direct batched linear-Sigma root reduction + one WHIR batched opening when possible."
    );
    eprintln!(
        "    WARP arity: {arity} fresh inputs in the first step, then {} per chained step.",
        arity - 1
    );
    eprintln!("    Times are single-process wall-clock averages over {iterations} iteration(s).");
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
                "    running k={num_variables}, N={n}: full WHIR prover..."
            ));
            let whir_prove_time = time_average(iterations, || {
                prove_n_whir_full_pcs(&fixture, &warp_witnesses);
            });
            print_progress(format!(
                "    running k={num_variables}, N={n}: full WHIR verifier setup..."
            ));
            let whir_bundle = build_n_whir_full_pcs(&fixture, &warp_witnesses);
            print_progress(format!(
                "    running k={num_variables}, N={n}: full WHIR verifier..."
            ));
            let whir_verify_time = time_average(iterations, || {
                verify_n_whir_full_pcs_bundle(&fixture, &whir_bundle);
            });
            print_progress(format!(
                "    running k={num_variables}, N={n}: WARP root prover..."
            ));
            let warp_prove_time = time_average(iterations, || {
                prove_warp_whir_root(&fixture, &warp_witnesses);
            });
            print_progress(format!(
                "    running k={num_variables}, N={n}: WARP root verifier setup..."
            ));
            let (bundle, warp_phases) =
                build_warp_whir_root_bundle_with_phases(&fixture, &warp_witnesses);
            print_progress(format!(
                "    running k={num_variables}, N={n}: WARP root verifier..."
            ));
            let verify_time = time_average(iterations, || {
                verify_warp_whir_root_bundle(&fixture, &bundle);
            });

            eprintln!(
                "{:<6}{:<8}{:<8}{:<10}{:<10}{:<12}{:<16}{:<16}{:<24}{:<16}{:<16}{:<24}",
                num_variables,
                n,
                steps,
                bundle.oracle_count,
                bundle.claim_count,
                bundle.whir_opening_count,
                format_duration(whir_prove_time),
                format_duration(warp_prove_time),
                format_warp_over_whir(warp_prove_time, whir_prove_time),
                format_duration(whir_verify_time),
                format_duration(verify_time),
                format_warp_over_whir(verify_time, whir_verify_time),
            );
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
            }
        }
    }
    eprintln!();
}

#[cfg(feature = "stark-backend")]
fn print_openvm_stark_comparison(num_variable_cases: &[usize], n_values: &[usize]) {
    let iterations = parse_usize_env("P3_WARP_STARK_COMPARE_ITERS", 1).max(1);
    let arity = warp_fresh_per_step();
    eprintln!();
    eprintln!("=== OpenVM/SWIRL full proofs vs SWIRL-accumulator WARP comparison ===");
    eprintln!("    WHIR lane: N full OpenVM/SWIRL proofs over the BoolAir trace.");
    eprintln!(
        "    WARP lane: N OpenVM/SWIRL commitments + WARP accumulation + outer SWIRL proof for step sumchecks, final MLE, Bool PESAT, and random RS consistency."
    );
    eprintln!(
        "    Verify lane: N SWIRL proof verification vs WARP outer SWIRL proof verification."
    );
    eprintln!(
        "    SystemParams use l_skip=0, width=1, log_blowup={LOG_INV_RATE}, so the SWIRL RS code matches WARP coefficient RS encoding."
    );
    eprintln!(
        "    WARP arity: {arity} fresh inputs in the first step, then {} per chained step.",
        arity - 1
    );
    eprintln!("    Times are single-process wall-clock averages over {iterations} iteration(s).");
    eprintln!(
        "{:<6}{:<8}{:<8}{:<16}{:<16}{:<18}{:<16}{:<16}{:<18}",
        "k",
        "N",
        "steps",
        "whir prove",
        "warp prove",
        "prove ratio",
        "whir verify",
        "warp verify",
        "verify ratio"
    );

    for &num_variables in num_variable_cases {
        for &n in n_values {
            if n < arity || (n - arity) % (arity - 1) != 0 {
                eprintln!(
                    "{:<6}{:<8}{:<8}{:<16}{:<16}{:<18}{:<16}{:<16}{:<18}",
                    num_variables, n, "-", "skip", "skip", "-", "skip", "skip", "invalid WARP N"
                );
                continue;
            }

            let fixture = make_openvm_warp_fixture(num_variables);
            let witnesses = make_boolean_witnesses(num_variables, n);
            let steps = step_plan(n, arity).len();
            let setup_bundle = build_openvm_warp_swirl_acc_bundle(&fixture, &witnesses);
            let (setup_receipt, setup_sumchecks) =
                verify_openvm_warp_swirl_acc_bundle_with_sumchecks(&fixture, &setup_bundle);
            let finalizer_fixture = make_openvm_finalizer_receipt_fixture(
                setup_receipt.claim_digest.len(),
                &setup_sumchecks,
                setup_receipt.final_instance.alpha.len(),
                setup_receipt.final_instance.beta.len(),
            );

            let whir_time = time_average(iterations, || {
                prove_n_openvm_whir_full(&fixture, &witnesses);
            });
            let warp_time = time_average(iterations, || {
                prove_openvm_warp_swirl_acc_finalizer_pipeline(
                    &fixture,
                    &finalizer_fixture,
                    &witnesses,
                );
            });
            let whir_proofs = build_n_openvm_whir_full(&fixture, &witnesses);
            let warp_outer_proof = build_openvm_warp_swirl_acc_finalizer_outer_proof(
                &finalizer_fixture,
                &setup_receipt,
                &setup_bundle,
                &setup_sumchecks,
            );
            let whir_verify_time = time_average(iterations, || {
                verify_n_openvm_whir_full(&fixture, &whir_proofs);
            });
            let warp_verify_time = time_average(iterations, || {
                verify_openvm_warp_swirl_acc_finalizer_outer(
                    &finalizer_fixture,
                    &setup_receipt,
                    &setup_sumchecks,
                    &warp_outer_proof,
                );
            });

            eprintln!(
                "{:<6}{:<8}{:<8}{:<16}{:<16}{:<18}{:<16}{:<16}{:<18}",
                num_variables,
                n,
                steps,
                format_duration(whir_time),
                format_duration(warp_time),
                format_warp_over_whir(warp_time, whir_time),
                format_duration(whir_verify_time),
                format_duration(warp_verify_time),
                format_warp_over_whir(warp_verify_time, whir_verify_time)
            );
        }
    }
    eprintln!();
}

#[cfg(not(feature = "stark-backend"))]
fn print_openvm_stark_comparison(_num_variable_cases: &[usize], _n_values: &[usize]) {
    eprintln!();
    eprintln!("OpenVM/SWIRL comparison requires `--features \"parallel stark-backend\"`.");
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
    if env::var("P3_WARP_STARK_COMPARE").as_deref() == Ok("1") {
        print_openvm_stark_comparison(&num_variable_cases, &n_values);
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
            #[cfg(feature = "stark-backend")]
            if valid_warp_n && env::var("P3_WARP_STARK_BENCH").as_deref() == Ok("1") {
                let openvm_fixture = make_openvm_warp_fixture(num_variables);
                group.bench_with_input(
                    BenchmarkId::new(format!("openvm_whir_full_{label}"), n),
                    &n,
                    |b, _| b.iter(|| prove_n_openvm_whir_full(&openvm_fixture, &warp_witnesses)),
                );
                group.bench_with_input(
                    BenchmarkId::new(format!("openvm_warp_external_{warp_label}"), n),
                    &n,
                    |b, _| b.iter(|| prove_openvm_warp_external(&openvm_fixture, &warp_witnesses)),
                );
                group.bench_with_input(
                    BenchmarkId::new(format!("openvm_warp_swirl_acc_{warp_label}"), n),
                    &n,
                    |b, _| b.iter(|| prove_openvm_warp_swirl_acc(&openvm_fixture, &warp_witnesses)),
                );
                let whir_proofs = build_n_openvm_whir_full(&openvm_fixture, &warp_witnesses);
                let warp_bundle =
                    build_openvm_warp_external_bundle(&openvm_fixture, &warp_witnesses);
                let warp_swirl_acc_bundle =
                    build_openvm_warp_swirl_acc_bundle(&openvm_fixture, &warp_witnesses);
                let (warp_receipt, warp_sumchecks) =
                    verify_openvm_warp_external_bundle_with_sumchecks(
                        &openvm_fixture,
                        &warp_bundle,
                    );
                let (warp_swirl_acc_receipt, warp_swirl_acc_sumchecks) =
                    verify_openvm_warp_swirl_acc_bundle_with_sumchecks(
                        &openvm_fixture,
                        &warp_swirl_acc_bundle,
                    );
                let receipt_fixture = make_openvm_receipt_fixture(warp_receipt.claim_digest.len());
                let receipt_proof =
                    build_openvm_warp_receipt_outer_proof(&receipt_fixture, &warp_receipt);
                let sumcheck_fixture = make_openvm_sumcheck_receipt_fixture(
                    warp_receipt.claim_digest.len(),
                    &warp_sumchecks,
                );
                let sumcheck_proof = build_openvm_warp_sumcheck_outer_proof(
                    &sumcheck_fixture,
                    &warp_receipt,
                    &warp_bundle,
                    &warp_sumchecks,
                );
                let swirl_acc_sumcheck_fixture = make_openvm_sumcheck_receipt_fixture(
                    warp_swirl_acc_receipt.claim_digest.len(),
                    &warp_swirl_acc_sumchecks,
                );
                let swirl_acc_sumcheck_proof = build_openvm_warp_swirl_acc_sumcheck_outer_proof(
                    &swirl_acc_sumcheck_fixture,
                    &warp_swirl_acc_receipt,
                    &warp_swirl_acc_bundle,
                    &warp_swirl_acc_sumchecks,
                );
                let swirl_acc_sumcheck_mle_fixture = make_openvm_sumcheck_mle_receipt_fixture(
                    warp_swirl_acc_receipt.claim_digest.len(),
                    &warp_swirl_acc_sumchecks,
                    warp_swirl_acc_receipt.final_instance.alpha.len(),
                );
                let swirl_acc_sumcheck_mle_proof =
                    build_openvm_warp_swirl_acc_sumcheck_mle_outer_proof(
                        &swirl_acc_sumcheck_mle_fixture,
                        &warp_swirl_acc_receipt,
                        &warp_swirl_acc_bundle,
                        &warp_swirl_acc_sumchecks,
                    );
                let swirl_acc_sumcheck_mle_bool_fixture =
                    make_openvm_sumcheck_mle_bool_receipt_fixture(
                        warp_swirl_acc_receipt.claim_digest.len(),
                        &warp_swirl_acc_sumchecks,
                        warp_swirl_acc_receipt.final_instance.alpha.len(),
                        warp_swirl_acc_receipt.final_instance.beta.len(),
                    );
                let swirl_acc_sumcheck_mle_bool_proof =
                    build_openvm_warp_swirl_acc_sumcheck_mle_bool_outer_proof(
                        &swirl_acc_sumcheck_mle_bool_fixture,
                        &warp_swirl_acc_receipt,
                        &warp_swirl_acc_bundle,
                        &warp_swirl_acc_sumchecks,
                    );
                let swirl_acc_finalizer_fixture = make_openvm_finalizer_receipt_fixture(
                    warp_swirl_acc_receipt.claim_digest.len(),
                    &warp_swirl_acc_sumchecks,
                    warp_swirl_acc_receipt.final_instance.alpha.len(),
                    warp_swirl_acc_receipt.final_instance.beta.len(),
                );
                let swirl_acc_finalizer_proof = build_openvm_warp_swirl_acc_finalizer_outer_proof(
                    &swirl_acc_finalizer_fixture,
                    &warp_swirl_acc_receipt,
                    &warp_swirl_acc_bundle,
                    &warp_swirl_acc_sumchecks,
                );
                group.bench_with_input(
                    BenchmarkId::new(format!("openvm_whir_full_verify_{label}"), n),
                    &n,
                    |b, _| b.iter(|| verify_n_openvm_whir_full(&openvm_fixture, &whir_proofs)),
                );
                group.bench_with_input(
                    BenchmarkId::new(format!("openvm_warp_external_verify_{warp_label}"), n),
                    &n,
                    |b, _| {
                        b.iter(|| verify_openvm_warp_external_bundle(&openvm_fixture, &warp_bundle))
                    },
                );
                group.bench_with_input(
                    BenchmarkId::new(format!("openvm_warp_swirl_acc_verify_{warp_label}"), n),
                    &n,
                    |b, _| {
                        b.iter(|| {
                            verify_openvm_warp_swirl_acc_bundle(
                                &openvm_fixture,
                                &warp_swirl_acc_bundle,
                            )
                        })
                    },
                );
                group.bench_with_input(
                    BenchmarkId::new(format!("openvm_warp_receipt_outer_{warp_label}"), n),
                    &n,
                    |b, _| {
                        b.iter(|| prove_openvm_warp_receipt_outer(&receipt_fixture, &warp_receipt))
                    },
                );
                group.bench_with_input(
                    BenchmarkId::new(format!("openvm_warp_receipt_outer_verify_{warp_label}"), n),
                    &n,
                    |b, _| {
                        b.iter(|| {
                            verify_openvm_warp_receipt_outer(
                                &receipt_fixture,
                                &warp_receipt,
                                &receipt_proof,
                            )
                        })
                    },
                );
                group.bench_with_input(
                    BenchmarkId::new(format!("openvm_warp_sumcheck_outer_{warp_label}"), n),
                    &n,
                    |b, _| {
                        b.iter(|| {
                            prove_openvm_warp_sumcheck_outer(
                                &sumcheck_fixture,
                                &warp_receipt,
                                &warp_bundle,
                                &warp_sumchecks,
                            )
                        })
                    },
                );
                group.bench_with_input(
                    BenchmarkId::new(format!("openvm_warp_sumcheck_outer_verify_{warp_label}"), n),
                    &n,
                    |b, _| {
                        b.iter(|| {
                            verify_openvm_warp_sumcheck_outer(
                                &sumcheck_fixture,
                                &warp_receipt,
                                &warp_sumchecks,
                                &sumcheck_proof,
                            )
                        })
                    },
                );
                group.bench_with_input(
                    BenchmarkId::new(
                        format!("openvm_warp_swirl_acc_sumcheck_outer_{warp_label}"),
                        n,
                    ),
                    &n,
                    |b, _| {
                        b.iter(|| {
                            prove_openvm_warp_swirl_acc_sumcheck_outer(
                                &swirl_acc_sumcheck_fixture,
                                &warp_swirl_acc_receipt,
                                &warp_swirl_acc_bundle,
                                &warp_swirl_acc_sumchecks,
                            )
                        })
                    },
                );
                group.bench_with_input(
                    BenchmarkId::new(
                        format!("openvm_warp_swirl_acc_sumcheck_outer_verify_{warp_label}"),
                        n,
                    ),
                    &n,
                    |b, _| {
                        b.iter(|| {
                            verify_openvm_warp_swirl_acc_sumcheck_outer(
                                &swirl_acc_sumcheck_fixture,
                                &warp_swirl_acc_receipt,
                                &warp_swirl_acc_sumchecks,
                                &swirl_acc_sumcheck_proof,
                            )
                        })
                    },
                );
                group.bench_with_input(
                    BenchmarkId::new(
                        format!("openvm_warp_swirl_acc_sumcheck_mle_outer_{warp_label}"),
                        n,
                    ),
                    &n,
                    |b, _| {
                        b.iter(|| {
                            prove_openvm_warp_swirl_acc_sumcheck_mle_outer(
                                &swirl_acc_sumcheck_mle_fixture,
                                &warp_swirl_acc_receipt,
                                &warp_swirl_acc_bundle,
                                &warp_swirl_acc_sumchecks,
                            )
                        })
                    },
                );
                group.bench_with_input(
                    BenchmarkId::new(
                        format!("openvm_warp_swirl_acc_sumcheck_mle_outer_verify_{warp_label}"),
                        n,
                    ),
                    &n,
                    |b, _| {
                        b.iter(|| {
                            verify_openvm_warp_swirl_acc_sumcheck_mle_outer(
                                &swirl_acc_sumcheck_mle_fixture,
                                &warp_swirl_acc_receipt,
                                &warp_swirl_acc_sumchecks,
                                &swirl_acc_sumcheck_mle_proof,
                            )
                        })
                    },
                );
                group.bench_with_input(
                    BenchmarkId::new(
                        format!("openvm_warp_swirl_acc_sumcheck_mle_bool_outer_{warp_label}"),
                        n,
                    ),
                    &n,
                    |b, _| {
                        b.iter(|| {
                            prove_openvm_warp_swirl_acc_sumcheck_mle_bool_outer(
                                &swirl_acc_sumcheck_mle_bool_fixture,
                                &warp_swirl_acc_receipt,
                                &warp_swirl_acc_bundle,
                                &warp_swirl_acc_sumchecks,
                            )
                        })
                    },
                );
                group.bench_with_input(
                    BenchmarkId::new(
                        format!(
                            "openvm_warp_swirl_acc_sumcheck_mle_bool_outer_verify_{warp_label}"
                        ),
                        n,
                    ),
                    &n,
                    |b, _| {
                        b.iter(|| {
                            verify_openvm_warp_swirl_acc_sumcheck_mle_bool_outer(
                                &swirl_acc_sumcheck_mle_bool_fixture,
                                &warp_swirl_acc_receipt,
                                &warp_swirl_acc_sumchecks,
                                &swirl_acc_sumcheck_mle_bool_proof,
                            )
                        })
                    },
                );
                group.bench_with_input(
                    BenchmarkId::new(
                        format!("openvm_warp_swirl_acc_finalizer_outer_{warp_label}"),
                        n,
                    ),
                    &n,
                    |b, _| {
                        b.iter(|| {
                            prove_openvm_warp_swirl_acc_finalizer_outer(
                                &swirl_acc_finalizer_fixture,
                                &warp_swirl_acc_receipt,
                                &warp_swirl_acc_bundle,
                                &warp_swirl_acc_sumchecks,
                            )
                        })
                    },
                );
                group.bench_with_input(
                    BenchmarkId::new(
                        format!("openvm_warp_swirl_acc_finalizer_outer_verify_{warp_label}"),
                        n,
                    ),
                    &n,
                    |b, _| {
                        b.iter(|| {
                            verify_openvm_warp_swirl_acc_finalizer_outer(
                                &swirl_acc_finalizer_fixture,
                                &warp_swirl_acc_receipt,
                                &warp_swirl_acc_sumchecks,
                                &swirl_acc_finalizer_proof,
                            )
                        })
                    },
                );
                group.bench_with_input(
                    BenchmarkId::new(format!("openvm_whir_full_e2e_{label}"), n),
                    &n,
                    |b, _| {
                        b.iter(|| {
                            prove_and_verify_n_openvm_whir_full(&openvm_fixture, &warp_witnesses)
                        })
                    },
                );
                group.bench_with_input(
                    BenchmarkId::new(format!("openvm_warp_external_e2e_{warp_label}"), n),
                    &n,
                    |b, _| {
                        b.iter(|| {
                            prove_and_verify_openvm_warp_external(&openvm_fixture, &warp_witnesses)
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
