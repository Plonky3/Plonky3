//! Criterion benchmarks for the SHA-256 AIR.
//!
//! Three benchmark groups are measured, each sweeping over several trace
//! sizes:
//!
//! - `sha256_air_trace`: trace generation only (no proving).
//! - `sha256_air_prove`: trace generation plus STARK proving.
//! - `sha256_air_prove_verify`: full prove + verify round trip.
//!
//! Use this suite to compare the cost of the AIR before and after an
//! optimization. Run with:
//!
//! ```text
//! cargo bench -p p3-sha256-air -- --save-baseline <name>
//! cargo bench -p p3-sha256-air -- --baseline <name>
//! ```

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use p3_baby_bear::BabyBear;
use p3_challenger::{HashChallenger, SerializingChallenger32};
use p3_commit::ExtensionMmcs;
use p3_field::extension::BinomialExtensionField;
use p3_fri::{FriParameters, TwoAdicFriPcs};
use p3_keccak::{Keccak256Hash, KeccakF};
use p3_merkle_tree::MerkleTreeMmcs;
use p3_sha256_air::Sha256Air;
use p3_symmetric::{CompressionFunctionFromHasher, PaddingFreeSponge, SerializingHasher};
use p3_uni_stark::{StarkConfig, prove, verify};

/// Base field for all benchmarks.
type Val = BabyBear;

/// Degree-4 extension used as the challenge field in FRI.
type Challenge = BinomialExtensionField<Val, 4>;

/// Build a STARK configuration plus its LDE blowup factor.
///
/// # Returns
///
/// A tuple of the config ready to pass to `prove` / `verify`, and the log_2
/// blowup factor that must be reserved when allocating the trace.
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
    // Leaf hash: Keccak-256 over raw bytes.
    let byte_hash = Keccak256Hash {};
    // Sponge hash: absorb 17 u64 lanes from KeccakF, squeeze 4.
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

    // Extension field commitment used during FRI queries.
    let challenge_mmcs = ExtensionMmcs::<Val, Challenge, _>::new(val_mmcs.clone());

    // Fiat-Shamir challenger seeded with an empty transcript.
    let challenger =
        SerializingChallenger32::<Val, HashChallenger<u8, _, 32>>::from_hasher(vec![], byte_hash);

    // FRI parameters (blowup, query count, etc).
    let fri_params = FriParameters::new_benchmark(challenge_mmcs);
    let log_blowup = fri_params.log_blowup;

    // Sequential DFT backend; parallel versions exist but we keep runs
    // deterministic for comparison.
    let dft = p3_dft::Radix2Bowers;

    // Bundle everything into the final config.
    let pcs = TwoAdicFriPcs::new(dft, val_mmcs, fri_params);
    let config = StarkConfig::new(pcs, challenger);
    (config, log_blowup)
}

/// Trace-generation-only benchmarks.
///
/// Exercises the hot path that fills each compression row. No commitments,
/// no FRI, no Fiat-Shamir.
fn bench_trace_generation(c: &mut Criterion) {
    let air = Sha256Air;

    let mut group = c.benchmark_group("sha256_air_trace");

    // Sweep three trace sizes. 2^10 is a quick signal; 2^14 / 2^16 expose
    // cache and parallelism effects.
    for log_num_hashes in [10, 12, 14] {
        let num_hashes = 1usize << log_num_hashes;

        group.bench_function(
            BenchmarkId::new("generate_trace", format!("2^{log_num_hashes}")),
            |b| {
                b.iter(|| {
                    // Zero LDE blowup - we're timing row population, not PCS setup.
                    air.generate_trace_rows::<Val>(num_hashes, 0)
                });
            },
        );
    }

    group.finish();
}

/// Proving benchmarks (trace generation + full STARK prover).
///
/// Use a small sample count so the total runtime is manageable.
fn bench_prove(c: &mut Criterion) {
    let air = Sha256Air;
    let (config, log_blowup) = make_stark_config();

    let mut group = c.benchmark_group("sha256_air_prove");
    // Each sample runs a full prover; keep the count small so wall time
    // stays bounded.
    group.sample_size(10);

    for log_num_hashes in [10, 12] {
        let num_hashes = 1usize << log_num_hashes;

        group.bench_function(
            BenchmarkId::new("prove", format!("2^{log_num_hashes}")),
            |b| {
                b.iter(|| {
                    // Trace is regenerated per sample - fair comparison because
                    // the optimized version will reuse the same code path.
                    let trace = air.generate_trace_rows::<Val>(num_hashes, log_blowup);
                    prove(&config, &air, trace, &[])
                });
            },
        );
    }

    group.finish();
}

/// Full prove + verify round trip at a single trace size.
///
/// Catches any asymmetry between prover and verifier that would be invisible
/// when benchmarking them in isolation.
fn bench_prove_verify(c: &mut Criterion) {
    let air = Sha256Air;
    let (config, log_blowup) = make_stark_config();

    let mut group = c.benchmark_group("sha256_air_prove_verify");
    group.sample_size(10);

    // Fixed size large enough to be meaningful but small enough to run fast.
    let log_num_hashes = 10;
    let num_hashes = 1usize << log_num_hashes;

    group.bench_function(
        BenchmarkId::new("prove_verify", format!("2^{log_num_hashes}")),
        |b| {
            b.iter(|| {
                let trace = air.generate_trace_rows::<Val>(num_hashes, log_blowup);
                let proof = prove(&config, &air, trace, &[]);
                verify(&config, &air, &proof, &[]).expect("verification failed");
            });
        },
    );

    group.finish();
}

criterion_group!(
    benches,
    bench_trace_generation,
    bench_prove,
    bench_prove_verify,
);
criterion_main!(benches);
