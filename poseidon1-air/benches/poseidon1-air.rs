//! Criterion benchmarks for the Poseidon1 AIR: trace generation, proving, and verification.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use p3_baby_bear::{
    BABYBEAR_POSEIDON1_HALF_FULL_ROUNDS, BABYBEAR_POSEIDON1_PARTIAL_ROUNDS_16,
    BABYBEAR_POSEIDON1_RC_16, BABYBEAR_S_BOX_DEGREE, BabyBear, MDSBabyBearData,
};
use p3_challenger::{HashChallenger, SerializingChallenger32};
use p3_commit::ExtensionMmcs;
use p3_field::PrimeCharacteristicRing;
use p3_field::extension::BinomialExtensionField;
use p3_fri::{FriParameters, TwoAdicFriPcs};
use p3_keccak::{Keccak256Hash, KeccakF};
use p3_merkle_tree::MerkleTreeMmcs;
use p3_monty_31::MDSUtils;
use p3_poseidon1::Poseidon1Constants;
use p3_poseidon1_air::{
    FullRoundConstants, PartialRoundConstants, Poseidon1Air, VectorizedPoseidon1Air,
    generate_trace_rows,
};
use p3_symmetric::{CompressionFunctionFromHasher, PaddingFreeSponge, SerializingHasher};
use p3_uni_stark::{StarkConfig, prove, verify};

/// Base field for all benchmarks.
type Val = BabyBear;

/// Degree-4 extension used as the challenge field in FRI.
type Challenge = BinomialExtensionField<Val, 4>;

/// Permutation state width (t = 16 in the Poseidon1 paper).
const WIDTH: usize = 16;

/// S-box exponent alpha. For this field, gcd(alpha, p - 1) = 1 is satisfied by 7.
const SBOX_DEGREE: u64 = BABYBEAR_S_BOX_DEGREE;

/// One intermediate column per S-box evaluation (commits x^3 to keep constraint degree at 3).
const SBOX_REGISTERS: usize = 1;

/// Number of full rounds in each half of the permutation (RF / 2).
const HALF_FULL_ROUNDS: usize = BABYBEAR_POSEIDON1_HALF_FULL_ROUNDS;

/// Number of partial rounds (RP) for width 16.
const PARTIAL_ROUNDS: usize = BABYBEAR_POSEIDON1_PARTIAL_ROUNDS_16;

/// Build AIR-optimized round constants from the canonical width-16 parameters.
///
/// Performs circulant-to-dense MDS expansion and forward constant substitution.
fn babybear_air_constants() -> (
    FullRoundConstants<Val, WIDTH>,
    PartialRoundConstants<Val, WIDTH>,
) {
    // Assemble the raw Poseidon1 parameter set.
    let raw = Poseidon1Constants {
        rounds_f: 2 * HALF_FULL_ROUNDS,
        rounds_p: PARTIAL_ROUNDS,
        mds_circ_col: MDSBabyBearData::MATRIX_CIRC_MDS_16_COL,
        round_constants: BABYBEAR_POSEIDON1_RC_16.to_vec(),
    };

    // Convert to the AIR representation (applies forward constant substitution).
    raw.to_optimized()
}

/// Build a complete STARK configuration backed by Keccak hashing and FRI.
///
/// # Returns
///
/// - The STARK configuration (PCS + challenger).
/// - The log_2 blowup factor needed when allocating trace memory.
#[allow(clippy::type_complexity)]
fn make_stark_config() -> (
    StarkConfig<
        TwoAdicFriPcs<
            Val,
            p3_dft::Radix2Bowers,
            MerkleTreeMmcs<
                [Val; p3_keccak::VECTOR_LEN],
                [u64; p3_keccak::VECTOR_LEN],
                SerializingHasher<PaddingFreeSponge<KeccakF, 25, 17, 4>>,
                CompressionFunctionFromHasher<PaddingFreeSponge<KeccakF, 25, 17, 4>, 2, 4>,
                2,
                4,
            >,
            ExtensionMmcs<
                Val,
                Challenge,
                MerkleTreeMmcs<
                    [Val; p3_keccak::VECTOR_LEN],
                    [u64; p3_keccak::VECTOR_LEN],
                    SerializingHasher<PaddingFreeSponge<KeccakF, 25, 17, 4>>,
                    CompressionFunctionFromHasher<PaddingFreeSponge<KeccakF, 25, 17, 4>, 2, 4>,
                    2,
                    4,
                >,
            >,
        >,
        Challenge,
        SerializingChallenger32<Val, HashChallenger<u8, Keccak256Hash, 32>>,
    >,
    usize,
) {
    // Leaf hash: Keccak-256 operating on raw bytes.
    let byte_hash = Keccak256Hash {};

    // Sponge hash: absorb 17 u64 lanes from the Keccak-f permutation, squeeze 4.
    let u64_hash = PaddingFreeSponge::<KeccakF, 25, 17, 4>::new(KeccakF {});

    // Serialize field elements to bytes before hashing.
    let field_hash = SerializingHasher::new(u64_hash);

    // Merkle tree node compression: hash two 4-u64 digests into one.
    let compress = CompressionFunctionFromHasher::<_, 2, 4>::new(u64_hash);

    // Vector commitment scheme over the base field.
    let val_mmcs = MerkleTreeMmcs::<
        [Val; p3_keccak::VECTOR_LEN],
        [u64; p3_keccak::VECTOR_LEN],
        _,
        _,
        2,
        4,
    >::new(field_hash, compress, 3);

    // Extension field commitment for FRI queries.
    let challenge_mmcs = ExtensionMmcs::<Val, Challenge, _>::new(val_mmcs.clone());

    // Fiat-Shamir challenger seeded with an empty transcript.
    let challenger =
        SerializingChallenger32::<Val, HashChallenger<u8, _, 32>>::from_hasher(vec![], byte_hash);

    // Derive FRI parameters (blowup, queries, etc.) from the commitment scheme.
    let fri_params = FriParameters::new_benchmark(challenge_mmcs);

    // Save the blowup factor for trace allocation.
    let log_blowup = fri_params.log_blowup;

    // Sequential DFT backend (no parallelism).
    let dft = p3_dft::Radix2Bowers;

    // Polynomial commitment scheme: FRI over a two-adic evaluation domain.
    let pcs = TwoAdicFriPcs::new(dft, val_mmcs, fri_params);

    // Bundle PCS + challenger into the final STARK config.
    let config = StarkConfig::new(pcs, challenger);
    (config, log_blowup)
}

/// Benchmark trace generation at various sizes.
///
/// Measures the time to build the execution trace (input generation + all
/// Poseidon1 round computations) without proving or committing.
fn bench_trace_generation(c: &mut Criterion) {
    // Pre-compute constants once, shared across all sizes.
    let (full_constants, partial_constants) = babybear_air_constants();

    let mut group = c.benchmark_group("poseidon1_air_trace");

    // Sweep over trace sizes: 2^10, 2^14, 2^16 permutations.
    for log_num_hashes in [10, 14, 16] {
        let num_hashes = 1usize << log_num_hashes;

        group.bench_function(
            BenchmarkId::new("generate_trace", format!("2^{log_num_hashes}")),
            |b| {
                b.iter(|| {
                    // Build deterministic inputs: each permutation gets a unique
                    // sequential state [i*W, i*W+1, ..., i*W+W-1].
                    let inputs: Vec<[Val; WIDTH]> = (0..num_hashes)
                        .map(|i| core::array::from_fn(|j| Val::from_u32((i * WIDTH + j) as u32)))
                        .collect();

                    // Generate the full execution trace (one row per permutation).
                    generate_trace_rows::<
                        _,
                        WIDTH,
                        SBOX_DEGREE,
                        SBOX_REGISTERS,
                        HALF_FULL_ROUNDS,
                        PARTIAL_ROUNDS,
                    >(inputs, &full_constants, &partial_constants, 0)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark STARK proving (trace generation + commitment + FRI).
///
/// Uses a reduced sample size (10) because each iteration includes a full
/// proof generation cycle.
fn bench_prove(c: &mut Criterion) {
    // Build the AIR with canonical constants.
    let (full_constants, partial_constants) = babybear_air_constants();
    let air: Poseidon1Air<
        Val,
        WIDTH,
        SBOX_DEGREE,
        SBOX_REGISTERS,
        HALF_FULL_ROUNDS,
        PARTIAL_ROUNDS,
    > = Poseidon1Air::new(full_constants, partial_constants);

    // Set up the proving infrastructure.
    let (config, log_blowup) = make_stark_config();

    let mut group = c.benchmark_group("poseidon1_air_prove");
    // Full prove is expensive; keep the sample count low.
    group.sample_size(10);

    // Sweep over two trace sizes: 2^10 and 2^14 permutations.
    for log_num_hashes in [10, 14] {
        let num_hashes = 1usize << log_num_hashes;

        group.bench_function(
            BenchmarkId::new("prove", format!("2^{log_num_hashes}")),
            |b| {
                b.iter(|| {
                    // Generate trace with extra capacity for the LDE blowup.
                    let trace = air.generate_trace_rows(num_hashes, log_blowup);

                    // Run the full STARK prover.
                    prove(&config, &air, trace, &[])
                });
            },
        );
    }

    group.finish();
}

/// Benchmark the full prove-then-verify cycle.
///
/// Includes both proof generation and verification to measure end-to-end latency.
fn bench_prove_verify(c: &mut Criterion) {
    // Build the AIR.
    let (full_constants, partial_constants) = babybear_air_constants();
    let air: Poseidon1Air<
        Val,
        WIDTH,
        SBOX_DEGREE,
        SBOX_REGISTERS,
        HALF_FULL_ROUNDS,
        PARTIAL_ROUNDS,
    > = Poseidon1Air::new(full_constants, partial_constants);

    let (config, log_blowup) = make_stark_config();

    let mut group = c.benchmark_group("poseidon1_air_prove_verify");
    group.sample_size(10);

    // Fixed at 2^14 permutations — large enough for meaningful timing.
    let log_num_hashes = 14;
    let num_hashes = 1usize << log_num_hashes;

    group.bench_function(
        BenchmarkId::new("prove_verify", format!("2^{log_num_hashes}")),
        |b| {
            b.iter(|| {
                // Generate trace.
                let trace = air.generate_trace_rows(num_hashes, log_blowup);

                // Prove.
                let proof = prove(&config, &air, trace, &[]);

                // Verify — panics on failure to catch regressions.
                verify(&config, &air, &proof, &[]).expect("verification failed");
            });
        },
    );

    group.finish();
}

/// Benchmark proving with the vectorized AIR layout.
///
/// Packs 8 independent permutations per trace row, increasing utilization of each NTT column.
fn bench_vectorized_prove(c: &mut Criterion) {
    let (full_constants, partial_constants) = babybear_air_constants();

    /// Number of permutations packed into each trace row.
    const VECTOR_LEN: usize = 8;

    // Build the vectorized AIR (wraps a standard AIR, applies it VECTOR_LEN times per row).
    let air: VectorizedPoseidon1Air<
        Val,
        WIDTH,
        SBOX_DEGREE,
        SBOX_REGISTERS,
        HALF_FULL_ROUNDS,
        PARTIAL_ROUNDS,
        VECTOR_LEN,
    > = VectorizedPoseidon1Air::new(full_constants, partial_constants);

    let (config, log_blowup) = make_stark_config();

    let mut group = c.benchmark_group("poseidon1_air_vectorized_prove");
    group.sample_size(10);

    // Sweep over row counts; total permutations = 2^log_num_rows * VECTOR_LEN.
    for log_num_rows in [10, 14] {
        let num_perms = (1usize << log_num_rows) * VECTOR_LEN;

        group.bench_function(
            BenchmarkId::new("vectorized_prove", format!("2^{log_num_rows}_rows")),
            |b| {
                b.iter(|| {
                    // Generate vectorized trace (VECTOR_LEN permutations per row).
                    let trace = air.generate_vectorized_trace_rows(num_perms, log_blowup);

                    // Run the full STARK prover.
                    prove(&config, &air, trace, &[])
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_trace_generation,
    bench_prove,
    bench_prove_verify,
    bench_vectorized_prove,
);
criterion_main!(benches);
