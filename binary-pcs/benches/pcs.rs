//! `commit`, `open`, `verify` and `fold_codeword` across `num_variables` in `{16, 18, 20}`, all
//! through `SuffixProver` at `folding = 0`, the only configuration this crate's commit phase
//! accepts.
//!
//! Every group uses `BatchSize::PerIteration`: under `BatchSize::LargeInput` the batch size
//! Criterion allocates scales with the iteration count, so two arms measured at different
//! iteration counts end up under different memory pressure and the comparison can invert.
//! `bench_verify` also prints each proof's `postcard` size to stderr, since under unique
//! decoding the query count — not the polynomial arity — dominates proof size and is worth
//! stating plainly.
//!
//! Each phase runs at two committed alphabets, over the same number of committed cells:
//!
//! ```text
//!     gf128   columns and base codeword at the challenge field's own width
//!     gf64    columns and base codeword half as wide, challenges unchanged
//! ```

use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use p3_binary_field::{BinaryChallenger, BinaryField64, BinaryField128};
use p3_binary_pcs::{BinaryPcs, BinaryPcsConfig, BinaryPcsParams, fold_codeword, fold_pair};
use p3_challenger::HashChallenger;
use p3_commit::MultilinearPcs;
use p3_keccak::Keccak256Hash;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_sumcheck::layout::{Layout, SuffixProver, Table, Witness};
use p3_sumcheck::{OpeningBatch, OpeningProtocol, TableShape, TableSpec};
use p3_symmetric::{CompressionFunctionFromHasher, SerializingHasher};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

type F = BinaryField128;
type MyHash = SerializingHasher<Keccak256Hash>;
type MyCompress = CompressionFunctionFromHasher<Keccak256Hash, 2, 32>;
type MyMmcs = MerkleTreeMmcs<F, u8, MyHash, MyCompress, 2, 32>;
type MyChallenger = BinaryChallenger<F, HashChallenger<u8, Keccak256Hash, 32>>;
type MyPcs = BinaryPcs<F, F, MyMmcs, MyMmcs>;

/// The same scheme with columns and base codeword over a 64-bit alphabet.
type Narrow = BinaryField64;
type NarrowMmcs = MerkleTreeMmcs<Narrow, u8, MyHash, MyCompress, 2, 32>;
type NarrowChallenger = BinaryChallenger<Narrow, HashChallenger<u8, Keccak256Hash, 32>>;
type NarrowPcs = BinaryPcs<Narrow, F, NarrowMmcs, MyMmcs>;

/// The polynomial arities under test.
const NUMS_VARIABLES: [usize; 3] = [16, 18, 20];

const LOG_INV_RATE: usize = 2;
const POW_BITS: usize = 8;
const SECURITY_LEVEL: usize = 100;

const fn mmcs() -> MyMmcs {
    MyMmcs::new(
        MyHash::new(Keccak256Hash),
        MyCompress::new(Keccak256Hash),
        0,
    )
}

const fn challenger() -> MyChallenger {
    MyChallenger::from_hasher(Vec::new(), Keccak256Hash)
}

const fn narrow_mmcs() -> NarrowMmcs {
    NarrowMmcs::new(
        MyHash::new(Keccak256Hash),
        MyCompress::new(Keccak256Hash),
        0,
    )
}

const fn narrow_challenger() -> NarrowChallenger {
    NarrowChallenger::from_hasher(Vec::new(), Keccak256Hash)
}

const fn params() -> BinaryPcsParams {
    BinaryPcsParams {
        log_inv_rate: LOG_INV_RATE,
        pow_bits: POW_BITS,
        security_level: SECURITY_LEVEL,
    }
}

fn make_pcs(num_variables: usize) -> MyPcs {
    let config = BinaryPcsConfig::try_new::<F, F>(num_variables, params()).unwrap();
    BinaryPcs::new(config, mmcs(), mmcs())
}

fn make_narrow_pcs(num_variables: usize) -> NarrowPcs {
    let config = BinaryPcsConfig::try_new::<Narrow, F>(num_variables, params()).unwrap();
    BinaryPcs::new(config, narrow_mmcs(), mmcs())
}

fn make_narrow_witness(num_variables: usize, seed: u64) -> Witness<Narrow> {
    let mut rng = SmallRng::seed_from_u64(seed);
    let table = Table::rand(&mut rng, 1, num_variables);
    SuffixProver::<Narrow, F>::new_witness(vec![table], 0)
}

fn make_witness(num_variables: usize, seed: u64) -> Witness<F> {
    let mut rng = SmallRng::seed_from_u64(seed);
    let table = Table::rand(&mut rng, 1, num_variables);
    SuffixProver::<F, F>::new_witness(vec![table], 0)
}

fn make_protocol(num_variables: usize) -> OpeningProtocol {
    OpeningProtocol::new(vec![TableSpec::new(
        TableShape::new(num_variables, 1),
        vec![OpeningBatch::new(vec![0], Vec::new())],
    )])
}

fn bench_commit(c: &mut Criterion) {
    let mut group = c.benchmark_group("commit");
    group.sample_size(10);
    for &num_variables in &NUMS_VARIABLES {
        let pcs = make_pcs(num_variables);
        group.bench_with_input(
            BenchmarkId::new("gf128", num_variables),
            &num_variables,
            |b, &num_variables| {
                b.iter_batched(
                    || (make_witness(num_variables, 0), challenger()),
                    |(witness, mut ch)| pcs.commit(witness, &mut ch).unwrap(),
                    BatchSize::PerIteration,
                );
            },
        );

        // The same cell count, half the bytes per leaf and half the encoded width.
        let narrow = make_narrow_pcs(num_variables);
        group.bench_with_input(
            BenchmarkId::new("gf64", num_variables),
            &num_variables,
            |b, &num_variables| {
                b.iter_batched(
                    || (make_narrow_witness(num_variables, 0), narrow_challenger()),
                    |(witness, mut ch)| narrow.commit(witness, &mut ch).unwrap(),
                    BatchSize::PerIteration,
                );
            },
        );
    }
    group.finish();
}

/// One whole opening at each alphabet, plus the proof size each one produces.
fn bench_open_narrow(c: &mut Criterion) {
    let mut group = c.benchmark_group("open_narrow");
    group.sample_size(10);
    for &num_variables in &NUMS_VARIABLES {
        let pcs = make_narrow_pcs(num_variables);
        let protocol = make_protocol(num_variables);

        // The proof's size at this alphabet, for the record alongside the timing.
        let mut ch = narrow_challenger();
        let (_commitment, data) = pcs
            .commit(make_narrow_witness(num_variables, 2), &mut ch)
            .unwrap();
        let proof = pcs.open(data, protocol.clone(), &mut ch).unwrap();
        let proof_bytes = postcard::to_allocvec(&proof).unwrap().len();
        eprintln!("pcs/proof_size_gf64/{num_variables}: {proof_bytes} bytes");

        group.bench_with_input(
            BenchmarkId::from_parameter(num_variables),
            &num_variables,
            |b, &num_variables| {
                b.iter_batched(
                    || {
                        let witness = make_narrow_witness(num_variables, 1);
                        let mut ch = narrow_challenger();
                        let (_commitment, prover_data) = pcs.commit(witness, &mut ch).unwrap();
                        (prover_data, protocol.clone(), ch)
                    },
                    |(prover_data, protocol, mut ch)| {
                        pcs.open(prover_data, protocol, &mut ch).unwrap()
                    },
                    BatchSize::PerIteration,
                );
            },
        );
    }
    group.finish();
}

fn bench_open(c: &mut Criterion) {
    let mut group = c.benchmark_group("open");
    group.sample_size(10);
    for &num_variables in &NUMS_VARIABLES {
        let pcs = make_pcs(num_variables);
        let protocol = make_protocol(num_variables);
        group.bench_with_input(
            BenchmarkId::from_parameter(num_variables),
            &num_variables,
            |b, &num_variables| {
                b.iter_batched(
                    || {
                        let witness = make_witness(num_variables, 1);
                        let mut ch = challenger();
                        let (_commitment, prover_data) = pcs.commit(witness, &mut ch).unwrap();
                        (prover_data, protocol.clone(), ch)
                    },
                    |(prover_data, protocol, mut ch)| {
                        pcs.open(prover_data, protocol, &mut ch).unwrap()
                    },
                    BatchSize::PerIteration,
                );
            },
        );
    }
    group.finish();
}

fn bench_verify(c: &mut Criterion) {
    let mut group = c.benchmark_group("verify");
    group.sample_size(10);
    for &num_variables in &NUMS_VARIABLES {
        let pcs = make_pcs(num_variables);
        let protocol = make_protocol(num_variables);
        let witness = make_witness(num_variables, 2);

        let mut prover_challenger = challenger();
        let (commitment, prover_data) = pcs.commit(witness, &mut prover_challenger).unwrap();
        let proof = pcs
            .open(prover_data, protocol.clone(), &mut prover_challenger)
            .unwrap();

        let proof_bytes = postcard::to_allocvec(&proof).unwrap().len();
        eprintln!("pcs/proof_size/{num_variables}: {proof_bytes} bytes");

        group.bench_with_input(
            BenchmarkId::from_parameter(num_variables),
            &num_variables,
            |b, _| {
                b.iter_batched(
                    challenger,
                    |mut ch| {
                        pcs.verify(&commitment, &proof, &mut ch, protocol.clone())
                            .expect("verification failed in benchmark");
                    },
                    BatchSize::PerIteration,
                );
            },
        );
    }
    group.finish();
}

/// The per-pair form, evaluating one domain point per output symbol.
///
/// This is the shape a single query's verifier-side check uses.
///
/// It runs alongside the whole-codeword form so the two are compared under one measurement
/// rather than one being inferred from the other.
fn fold_codeword_per_pair(codeword: &[F], beta: F) -> Vec<F> {
    codeword
        .chunks(2)
        .enumerate()
        .map(|(index, pair)| fold_pair(index, beta, pair[0], pair[1]))
        .collect()
}

/// Folds a full base-round codeword once, in both the whole-codeword and the per-pair form.
///
/// This is the round every real proof spends the most time in.
///
/// Folding only borrows its input, so neither arm needs per-call setup.
///
/// A fold halves the symbol count, so the ratios below are stated per output symbol rather than
/// per input element.
fn bench_fold_codeword(c: &mut Criterion) {
    let mut group = c.benchmark_group("fold_codeword");
    group.sample_size(10);
    let mut rng = SmallRng::seed_from_u64(3);
    for &num_variables in &NUMS_VARIABLES {
        let len = 1usize << (num_variables + LOG_INV_RATE);
        let codeword: Vec<F> = (0..len).map(|_| rng.random()).collect();
        let beta: F = rng.random();

        group.bench_with_input(
            BenchmarkId::new("codeword", num_variables),
            &codeword,
            |b, codeword| b.iter(|| fold_codeword(codeword, beta)),
        );
        group.bench_with_input(
            BenchmarkId::new("per_pair", num_variables),
            &codeword,
            |b, codeword| b.iter(|| fold_codeword_per_pair(codeword, beta)),
        );

        // The same symbol count over the narrower alphabet, on the two routes it reaches.
        //
        //     lift_gf64    narrow symbols, wide challenge, so the output widens
        //     tower_gf64   narrow symbols and a narrow challenge, no widening at all
        let narrow: Vec<Narrow> = (0..len).map(|_| rng.random()).collect();
        let narrow_beta: Narrow = rng.random();
        group.bench_with_input(
            BenchmarkId::new("lift_gf64", num_variables),
            &narrow,
            |b, narrow| b.iter(|| fold_codeword::<Narrow, F>(narrow, beta)),
        );
        group.bench_with_input(
            BenchmarkId::new("tower_gf64", num_variables),
            &narrow,
            |b, narrow| b.iter(|| fold_codeword::<Narrow, Narrow>(narrow, narrow_beta)),
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_commit,
    bench_open,
    bench_open_narrow,
    bench_verify,
    bench_fold_codeword
);
criterion_main!(benches);
