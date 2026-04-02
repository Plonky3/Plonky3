//! Criterion benchmarks for the Monolith AIR: trace generation, proving, and verification.
//!
//! Covers **Monolith-31** (Mersenne31, WIDTH 16, circle STARKs) and **Monolith-64**
//! (Goldilocks, WIDTH 8, two-adic FRI + Keccak), mirroring the poseidon1-air bench layout.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use p3_challenger::{HashChallenger, SerializingChallenger32, SerializingChallenger64};
use p3_circle::CirclePcs;
use p3_commit::ExtensionMmcs;
use p3_dft::Radix2Bowers;
use p3_field::PrimeCharacteristicRing;
use p3_field::extension::BinomialExtensionField;
use p3_fri::{TwoAdicFriPcs, create_benchmark_fri_params};
use p3_goldilocks::Goldilocks;
use p3_keccak::{Keccak256Hash, KeccakF};
use p3_merkle_tree::MerkleTreeMmcs;
use p3_mersenne_31::Mersenne31;
use p3_monolith::bars::mersenne31::MonolithBarsM31;
use p3_monolith::mds::mersenne31::MonolithMdsMatrixMersenne31;
use p3_monolith::{
    MonolithBarsGoldilocks, MonolithGoldilocks8, MonolithMdsMatrixGoldilocks, MonolithMersenne31,
};
use p3_monolith_air::{
    GOLDILOCKS_8_LIMB_BITS, MERSENNE31_LIMB_BITS, MonolithAir, generate_trace_rows,
};
use p3_sha256::Sha256;
use p3_symmetric::{CompressionFunctionFromHasher, PaddingFreeSponge, SerializingHasher};
use p3_uni_stark::{StarkConfig, prove, verify};

// --- Monolith-31 (Mersenne31) ------------------------------------------------

type M31Val = Mersenne31;
type M31Challenge = BinomialExtensionField<M31Val, 3>;

const M31_WIDTH: usize = 16;
const M31_NUM_FULL_ROUNDS: usize = 5;
const M31_NUM_BARS: usize = 8;
const M31_FIELD_BITS: usize = 31;

fn build_monolith_air_m31() -> (
    MonolithAir<M31Val, M31_WIDTH, M31_NUM_FULL_ROUNDS, M31_NUM_BARS, M31_FIELD_BITS>,
    MonolithBarsM31,
) {
    let bars = MonolithBarsM31;
    let mds = MonolithMdsMatrixMersenne31::<6>;
    let mds_matrix = MonolithAir::<
        M31Val,
        M31_WIDTH,
        M31_NUM_FULL_ROUNDS,
        M31_NUM_BARS,
        M31_FIELD_BITS,
    >::extract_mds_matrix(&mds);
    let monolith = MonolithMersenne31::new(bars, mds);
    let air = MonolithAir::new(monolith.round_constants, mds_matrix, MERSENNE31_LIMB_BITS);
    (air, bars)
}

#[allow(clippy::type_complexity)]
fn make_stark_config_m31() -> (
    StarkConfig<
        CirclePcs<
            M31Val,
            MerkleTreeMmcs<
                M31Val,
                u8,
                SerializingHasher<Sha256>,
                CompressionFunctionFromHasher<Sha256, 2, 32>,
                2,
                32,
            >,
            ExtensionMmcs<
                M31Val,
                M31Challenge,
                MerkleTreeMmcs<
                    M31Val,
                    u8,
                    SerializingHasher<Sha256>,
                    CompressionFunctionFromHasher<Sha256, 2, 32>,
                    2,
                    32,
                >,
            >,
        >,
        M31Challenge,
        SerializingChallenger32<M31Val, HashChallenger<u8, Sha256, 32>>,
    >,
    usize,
) {
    let byte_hash = Sha256 {};
    let field_hash = SerializingHasher::new(Sha256);
    let compress = CompressionFunctionFromHasher::<_, 2, 32>::new(byte_hash);

    let val_mmcs = MerkleTreeMmcs::<M31Val, u8, _, _, 2, 32>::new(field_hash, compress, 3);
    let challenge_mmcs = ExtensionMmcs::<M31Val, M31Challenge, _>::new(val_mmcs.clone());

    let challenger = SerializingChallenger32::<M31Val, HashChallenger<u8, _, 32>>::from_hasher(
        vec![],
        byte_hash,
    );

    let mut fri_params = create_benchmark_fri_params(challenge_mmcs);
    fri_params.log_blowup = 2;
    let log_blowup = fri_params.log_blowup;

    let pcs = CirclePcs::new(val_mmcs, fri_params);
    let config = StarkConfig::new(pcs, challenger);
    (config, log_blowup)
}

fn bench_trace_generation_m31(c: &mut Criterion) {
    let (air, bars) = build_monolith_air_m31();

    let mut group = c.benchmark_group("monolith_air_m31_trace");

    for log_num_hashes in [10, 14, 16] {
        let num_hashes = 1usize << log_num_hashes;

        group.bench_function(
            BenchmarkId::new("generate_trace", format!("2^{log_num_hashes}")),
            |b| {
                b.iter(|| {
                    let inputs: Vec<[M31Val; M31_WIDTH]> = (0..num_hashes)
                        .map(|i| core::array::from_fn(|j| M31Val::from_usize(i * M31_WIDTH + j)))
                        .collect();

                    generate_trace_rows::<
                        _,
                        _,
                        M31_WIDTH,
                        M31_NUM_FULL_ROUNDS,
                        M31_NUM_BARS,
                        M31_FIELD_BITS,
                    >(inputs, &air, &bars, 0)
                });
            },
        );
    }

    group.finish();
}

fn bench_prove_m31(c: &mut Criterion) {
    let (air, bars) = build_monolith_air_m31();
    let (config, log_blowup) = make_stark_config_m31();

    let mut group = c.benchmark_group("monolith_air_m31_prove");
    group.sample_size(10);

    for log_num_hashes in [10, 14] {
        let num_hashes = 1usize << log_num_hashes;

        group.bench_function(
            BenchmarkId::new("prove", format!("2^{log_num_hashes}")),
            |b| {
                b.iter(|| {
                    let trace = air.generate_trace_rows(num_hashes, &bars, log_blowup);
                    prove(&config, &air, trace, &[])
                });
            },
        );
    }

    group.finish();
}

fn bench_prove_verify_m31(c: &mut Criterion) {
    let (air, bars) = build_monolith_air_m31();
    let (config, log_blowup) = make_stark_config_m31();

    let mut group = c.benchmark_group("monolith_air_m31_prove_verify");
    group.sample_size(10);

    let log_num_hashes = 14;
    let num_hashes = 1usize << log_num_hashes;

    group.bench_function(
        BenchmarkId::new("prove_verify", format!("2^{log_num_hashes}")),
        |b| {
            b.iter(|| {
                let trace = air.generate_trace_rows(num_hashes, &bars, log_blowup);
                let proof = prove(&config, &air, trace, &[]);
                verify(&config, &air, &proof, &[]).expect("verification failed");
            });
        },
    );

    group.finish();
}

// --- Monolith-64 (Goldilocks, WIDTH 8) ---------------------------------------

type GlVal = Goldilocks;
type GlChallenge = BinomialExtensionField<GlVal, 2>;

const GL_WIDTH: usize = 8;
const GL_NUM_FULL_ROUNDS: usize = 5;
const GL_NUM_BARS: usize = 4;
const GL_FIELD_BITS: usize = 64;

fn build_monolith_air_goldilocks() -> (
    MonolithAir<GlVal, GL_WIDTH, GL_NUM_FULL_ROUNDS, GL_NUM_BARS, GL_FIELD_BITS>,
    MonolithBarsGoldilocks<8>,
) {
    let bars = MonolithBarsGoldilocks::<8>;
    let mds = MonolithMdsMatrixGoldilocks;
    let mds_matrix = MonolithAir::<
        GlVal,
        GL_WIDTH,
        GL_NUM_FULL_ROUNDS,
        GL_NUM_BARS,
        GL_FIELD_BITS,
    >::extract_mds_matrix(&mds);
    let monolith: MonolithGoldilocks8<_, GL_WIDTH, 5> = MonolithGoldilocks8::new(bars, mds);
    let air = MonolithAir::new(monolith.round_constants, mds_matrix, GOLDILOCKS_8_LIMB_BITS);
    (air, bars)
}

#[allow(clippy::type_complexity)]
fn make_stark_config_goldilocks() -> (
    StarkConfig<
        TwoAdicFriPcs<
            GlVal,
            Radix2Bowers,
            MerkleTreeMmcs<
                [GlVal; p3_keccak::VECTOR_LEN],
                [u64; p3_keccak::VECTOR_LEN],
                SerializingHasher<PaddingFreeSponge<KeccakF, 25, 17, 4>>,
                CompressionFunctionFromHasher<PaddingFreeSponge<KeccakF, 25, 17, 4>, 2, 4>,
                2,
                4,
            >,
            ExtensionMmcs<
                GlVal,
                GlChallenge,
                MerkleTreeMmcs<
                    [GlVal; p3_keccak::VECTOR_LEN],
                    [u64; p3_keccak::VECTOR_LEN],
                    SerializingHasher<PaddingFreeSponge<KeccakF, 25, 17, 4>>,
                    CompressionFunctionFromHasher<PaddingFreeSponge<KeccakF, 25, 17, 4>, 2, 4>,
                    2,
                    4,
                >,
            >,
        >,
        GlChallenge,
        SerializingChallenger64<GlVal, HashChallenger<u8, Keccak256Hash, 32>>,
    >,
    usize,
) {
    let byte_hash = Keccak256Hash {};
    let u64_hash = PaddingFreeSponge::<KeccakF, 25, 17, 4>::new(KeccakF {});
    let field_hash = SerializingHasher::new(u64_hash);
    let compress = CompressionFunctionFromHasher::<_, 2, 4>::new(u64_hash);

    let val_mmcs = MerkleTreeMmcs::<
        [GlVal; p3_keccak::VECTOR_LEN],
        [u64; p3_keccak::VECTOR_LEN],
        _,
        _,
        2,
        4,
    >::new(field_hash, compress, 3);
    let challenge_mmcs = ExtensionMmcs::<GlVal, GlChallenge, _>::new(val_mmcs.clone());

    let challenger =
        SerializingChallenger64::<GlVal, HashChallenger<u8, _, 32>>::from_hasher(vec![], byte_hash);

    let mut fri_params = create_benchmark_fri_params(challenge_mmcs);
    fri_params.log_blowup = 2;
    let log_blowup = fri_params.log_blowup;

    let pcs = TwoAdicFriPcs::new(Radix2Bowers, val_mmcs, fri_params);
    let config = StarkConfig::new(pcs, challenger);
    (config, log_blowup)
}

fn bench_trace_generation_goldilocks(c: &mut Criterion) {
    let (air, bars) = build_monolith_air_goldilocks();

    let mut group = c.benchmark_group("monolith_air_goldilocks_trace");

    for log_num_hashes in [10, 14, 16] {
        let num_hashes = 1usize << log_num_hashes;

        group.bench_function(
            BenchmarkId::new("generate_trace", format!("2^{log_num_hashes}")),
            |b| {
                b.iter(|| {
                    let inputs: Vec<[GlVal; GL_WIDTH]> = (0..num_hashes)
                        .map(|i| core::array::from_fn(|j| GlVal::from_usize(i * GL_WIDTH + j)))
                        .collect();

                    generate_trace_rows::<
                        _,
                        _,
                        GL_WIDTH,
                        GL_NUM_FULL_ROUNDS,
                        GL_NUM_BARS,
                        GL_FIELD_BITS,
                    >(inputs, &air, &bars, 0)
                });
            },
        );
    }

    group.finish();
}

fn bench_prove_goldilocks(c: &mut Criterion) {
    let (air, bars) = build_monolith_air_goldilocks();
    let (config, log_blowup) = make_stark_config_goldilocks();

    let mut group = c.benchmark_group("monolith_air_goldilocks_prove");
    group.sample_size(10);

    for log_num_hashes in [10, 14] {
        let num_hashes = 1usize << log_num_hashes;

        group.bench_function(
            BenchmarkId::new("prove", format!("2^{log_num_hashes}")),
            |b| {
                b.iter(|| {
                    let trace = air.generate_trace_rows(num_hashes, &bars, log_blowup);
                    prove(&config, &air, trace, &[])
                });
            },
        );
    }

    group.finish();
}

fn bench_prove_verify_goldilocks(c: &mut Criterion) {
    let (air, bars) = build_monolith_air_goldilocks();
    let (config, log_blowup) = make_stark_config_goldilocks();

    let mut group = c.benchmark_group("monolith_air_goldilocks_prove_verify");
    group.sample_size(10);

    let log_num_hashes = 14;
    let num_hashes = 1usize << log_num_hashes;

    group.bench_function(
        BenchmarkId::new("prove_verify", format!("2^{log_num_hashes}")),
        |b| {
            b.iter(|| {
                let trace = air.generate_trace_rows(num_hashes, &bars, log_blowup);
                let proof = prove(&config, &air, trace, &[]);
                verify(&config, &air, &proof, &[]).expect("verification failed");
            });
        },
    );

    group.finish();
}

criterion_group!(
    benches,
    bench_trace_generation_m31,
    bench_prove_m31,
    bench_prove_verify_m31,
    bench_trace_generation_goldilocks,
    bench_prove_goldilocks,
    bench_prove_verify_goldilocks,
);
criterion_main!(benches);
