//! Zero-knowledge overhead benchmarks: HVZK-WHIR against plain WHIR.
//!
//! - Measures prover time, verifier time, and serialized proof size.
//! - Both pipelines run the same polynomial size and round structure.
//! - Eprint 2026/391 predicts a `1 + o(1)` overhead.

use std::time::Duration;

use criterion::measurement::WallTime;
use criterion::{
    BatchSize, BenchmarkGroup, BenchmarkId, Criterion, criterion_group, criterion_main,
};
use p3_challenger::DuplexChallenger;
use p3_commit::MultilinearPcs;
use p3_dft::Radix2DFTSmallBatch;
use p3_field::Field;
use p3_field::extension::BinomialExtensionField;
use p3_koala_bear::{KoalaBear, Poseidon2KoalaBear};
use p3_merkle_tree::MerkleTreeMmcs;
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;
use p3_sumcheck::layout::{Layout, PrefixProver, Table};
use p3_sumcheck::{OpeningBatch, OpeningProtocol, TableShape, TableSpec};
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use p3_whir::fiat_shamir::domain_separator::DomainSeparator;
use p3_whir::parameters::{FoldingFactor, ProtocolParameters, SecurityAssumption, WhirConfig};
use p3_whir::pcs::prover::WhirProver;
use p3_whir::pcs::zk::{HidingWhirPcs, ZkParameters, ZkWhirConfig};
use rand::SeedableRng;
use rand::rngs::SmallRng;

type F = KoalaBear;
type EF = BinomialExtensionField<F, 4>;

type Poseidon16 = Poseidon2KoalaBear<16>;
type Poseidon24 = Poseidon2KoalaBear<24>;
type MerkleHash = PaddingFreeSponge<Poseidon24, 24, 16, 8>;
type MerkleCompress = TruncatedPermutation<Poseidon16, 2, 8, 16>;
type Challenger = DuplexChallenger<F, Poseidon16, 16, 8>;
type PackedF = <F as Field>::Packing;
type Mmcs = MerkleTreeMmcs<PackedF, PackedF, MerkleHash, MerkleCompress, 2, 8>;
type Dft = Radix2DFTSmallBatch<F>;

type PlainPcs = WhirProver<EF, F, Dft, Mmcs, Challenger, PrefixProver<F, EF>>;
type ZkPcs = HidingWhirPcs<EF, F, Dft, Mmcs, Challenger, SmallRng>;

// Polynomial sizes (log_2 of coefficient count) and shared knobs.
const SIZES: [usize; 2] = [16, 18];
const FOLDING: usize = 4;
const LOG_INV_RATE: usize = 2;
const SECURITY_LEVEL: usize = 100;

const fn protocol_params() -> ProtocolParameters {
    ProtocolParameters {
        security_level: SECURITY_LEVEL,
        pow_bits: 16,
        round_log_inv_rates: Vec::new(),
        folding_factor: FoldingFactor::Constant(FOLDING),
        soundness_type: SecurityAssumption::CapacityBound,
        starting_log_inv_rate: LOG_INV_RATE,
    }
}

fn mmcs() -> Mmcs {
    let mut rng = SmallRng::seed_from_u64(1);
    let p16 = Poseidon16::new_from_rng_128(&mut rng);
    let p24 = Poseidon24::new_from_rng_128(&mut rng);
    Mmcs::new(MerkleHash::new(p24), MerkleCompress::new(p16), 0)
}

fn challenger() -> Challenger {
    let mut rng = SmallRng::seed_from_u64(2);
    Challenger::new(Poseidon16::new_from_rng_128(&mut rng))
}

/// Benchmarks the plain (non-hiding) prover for one size.
fn bench_plain(group: &mut BenchmarkGroup<'_, WallTime>, num_variables: usize) {
    let config = WhirConfig::new(num_variables, protocol_params()).unwrap();
    let pcs = PlainPcs::new(config, Dft::default(), mmcs());

    let mut rng = SmallRng::seed_from_u64(3);
    let table = Table::new(vec![Poly::<F>::rand(&mut rng, num_variables)]);
    let protocol = OpeningProtocol::new(vec![TableSpec::new(
        TableShape::new(num_variables, 1),
        vec![OpeningBatch::new(vec![0], Vec::new())],
    )]);

    group.bench_function(BenchmarkId::new("plain", num_variables), |b| {
        b.iter_batched(
            || {
                let mut ch = challenger();
                let mut ds = DomainSeparator::new(vec![]);
                pcs.add_domain_separator::<8>(&mut ds);
                ds.observe_domain_separator(&mut ch);
                let witness = PrefixProver::<F, EF>::new_witness(vec![table.clone()], FOLDING);
                (ch, witness)
            },
            |(mut ch, witness)| {
                let (_, data) = pcs.commit(witness, &mut ch);
                pcs.open(data, protocol.clone(), &mut ch)
            },
            BatchSize::SmallInput,
        );
    });
}

/// Benchmarks the hiding prover for one size.
fn bench_zk(group: &mut BenchmarkGroup<'_, WallTime>, num_variables: usize) {
    let config = ZkWhirConfig::new(
        num_variables,
        protocol_params(),
        // Section 2.7 of eprint 2026/391: O~(lambda)-sized masks at low
        // rate, so few spot checks reach the security level.
        ZkParameters {
            ell_zk: 16,
            mask_log_inv_rate: 5,
        },
    )
    .unwrap();
    let pcs = ZkPcs::new(config, Dft::default(), mmcs(), SmallRng::seed_from_u64(4));

    let mut rng = SmallRng::seed_from_u64(3);
    let witness = Poly::<F>::rand(&mut rng, num_variables);
    let points = vec![Point::<EF>::rand(&mut rng, num_variables)];

    group.bench_function(BenchmarkId::new("zk", num_variables), |b| {
        b.iter_batched(
            || {
                let mut ch = challenger();
                let mut ds = DomainSeparator::new(vec![]);
                pcs.add_domain_separator::<8>(&mut ds);
                ds.observe_domain_separator(&mut ch);
                (ch, witness.clone())
            },
            |(mut ch, witness)| {
                let (_, data) = pcs.commit(witness, &mut ch);
                pcs.open(data, points.clone(), &mut ch)
            },
            BatchSize::SmallInput,
        );
    });
}

/// Reports serialized proof sizes side by side (printed once, not measured).
fn report_proof_sizes(num_variables: usize) {
    // Plain proof size.
    let config = WhirConfig::new(num_variables, protocol_params()).unwrap();
    let pcs = PlainPcs::new(config, Dft::default(), mmcs());
    let mut rng = SmallRng::seed_from_u64(3);
    let table = Table::new(vec![Poly::<F>::rand(&mut rng, num_variables)]);
    let protocol = OpeningProtocol::new(vec![TableSpec::new(
        TableShape::new(num_variables, 1),
        vec![OpeningBatch::new(vec![0], Vec::new())],
    )]);
    let mut ch = challenger();
    let mut ds = DomainSeparator::new(vec![]);
    pcs.add_domain_separator::<8>(&mut ds);
    ds.observe_domain_separator(&mut ch);
    let witness = PrefixProver::<F, EF>::new_witness(vec![table], FOLDING);
    let (_, data) = pcs.commit(witness, &mut ch);
    let plain_proof = pcs.open(data, protocol, &mut ch);
    let plain_size = postcard::to_allocvec(&plain_proof).unwrap().len();

    // Hiding proof size.
    let config = ZkWhirConfig::new(
        num_variables,
        protocol_params(),
        // Section 2.7 of eprint 2026/391: O~(lambda)-sized masks at low
        // rate, so few spot checks reach the security level.
        ZkParameters {
            ell_zk: 16,
            mask_log_inv_rate: 5,
        },
    )
    .unwrap();
    let pcs = ZkPcs::new(config, Dft::default(), mmcs(), SmallRng::seed_from_u64(4));
    let mut rng = SmallRng::seed_from_u64(3);
    let witness = Poly::<F>::rand(&mut rng, num_variables);
    let points = vec![Point::<EF>::rand(&mut rng, num_variables)];
    let mut ch = challenger();
    let mut ds = DomainSeparator::new(vec![]);
    pcs.add_domain_separator::<8>(&mut ds);
    ds.observe_domain_separator(&mut ch);
    let (_, data) = pcs.commit(witness, &mut ch);
    let zk_proof = pcs.open(data, points, &mut ch);
    let zk_size = postcard::to_allocvec(&zk_proof).unwrap().len();

    println!(
        "proof size @ 2^{num_variables}: plain = {plain_size} B, zk = {zk_size} B, overhead = {:.2}x",
        zk_size as f64 / plain_size as f64,
    );
}

fn bench_zk_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("whir_zk_overhead");
    group
        .sample_size(10)
        .measurement_time(Duration::from_secs(15));

    for num_variables in SIZES {
        report_proof_sizes(num_variables);
        bench_plain(&mut group, num_variables);
        bench_zk(&mut group, num_variables);
    }
    group.finish();
}

criterion_group!(benches, bench_zk_overhead);
criterion_main!(benches);
