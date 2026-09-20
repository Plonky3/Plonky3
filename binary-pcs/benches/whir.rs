//! Commit, prove, and verify throughput for WHIR over the additive binary domain.

use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use p3_binary_field::{BinaryChallenger, Poly64, Poly192};
use p3_binary_pcs::whir::{BinaryWhirDomain, recommended_cap_height};
use p3_challenger::HashChallenger;
use p3_commit::MultilinearPcs;
use p3_keccak::Keccak256Hash;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_sumcheck::layout::{Layout, PrefixProver, Table, Witness};
use p3_sumcheck::{OpeningBatch, OpeningProtocol, TableShape, TableSpec};
use p3_symmetric::{CompressionFunctionFromHasher, SerializingHasher};
use p3_whir::{FoldingFactor, ProtocolParameters, SecurityAssumption, WhirConfig, WhirProver};
use rand::SeedableRng;
use rand::rngs::SmallRng;

type Hash = SerializingHasher<Keccak256Hash>;
type Compress = CompressionFunctionFromHasher<Keccak256Hash, 2, 32>;
type Mmcs = MerkleTreeMmcs<Poly64, u8, Hash, Compress, 2, 32>;
type Challenger = BinaryChallenger<Poly64, HashChallenger<u8, Keccak256Hash, 32>>;
type Domain = BinaryWhirDomain;
type Pcs = WhirProver<Poly192, Poly64, Domain, Mmcs, Challenger, PrefixProver<Poly64, Poly192>>;

const FOLDING: usize = 4;

const fn challenger() -> Challenger {
    Challenger::from_hasher(Vec::new(), Keccak256Hash)
}

fn instance(num_variables: usize) -> (Pcs, Witness<Poly64>, OpeningProtocol) {
    let mut rng = SmallRng::seed_from_u64(0x00B1_A47E);
    let witness = PrefixProver::<Poly64, Poly192>::new_witness(
        vec![Table::rand(&mut rng, 1, num_variables)],
        FOLDING,
    );
    let protocol = OpeningProtocol::new(vec![TableSpec::new(
        TableShape::new(num_variables, 1),
        vec![OpeningBatch::new(vec![0], vec![])],
    )]);
    let domain = Domain::default();
    let config = WhirConfig::new_with_domain(
        num_variables,
        ProtocolParameters {
            security_level: 100,
            pow_bits: 20,
            round_log_inv_rates: vec![],
            folding_factor: FoldingFactor::Constant(FOLDING),
            soundness_type: SecurityAssumption::JohnsonBound,
            starting_log_inv_rate: 1,
        },
        &domain,
    )
    .unwrap();
    let mmcs = Mmcs::new(
        Hash::new(Keccak256Hash),
        Compress::new(Keccak256Hash),
        recommended_cap_height(&config),
    );
    (Pcs::new(config, domain, mmcs), witness, protocol)
}

fn bench_whir(c: &mut Criterion) {
    for num_variables in [12, 16] {
        let (pcs, witness, protocol) = instance(num_variables);
        let mut group = c.benchmark_group("binary_whir");

        group.bench_with_input(
            BenchmarkId::new("commit", num_variables),
            &num_variables,
            |b, _| {
                b.iter_batched(
                    || (witness.clone(), challenger()),
                    |(witness, mut challenger)| pcs.commit(witness, &mut challenger).unwrap(),
                    BatchSize::PerIteration,
                );
            },
        );

        group.bench_with_input(
            BenchmarkId::new("prove", num_variables),
            &num_variables,
            |b, _| {
                b.iter_batched(
                    || {
                        let mut challenger = challenger();
                        let (_, data) = pcs.commit(witness.clone(), &mut challenger).unwrap();
                        (data, challenger)
                    },
                    |(data, mut challenger)| {
                        pcs.open(data, protocol.clone(), &mut challenger).unwrap()
                    },
                    BatchSize::PerIteration,
                );
            },
        );

        let mut prover_challenger = challenger();
        let (commitment, data) = pcs.commit(witness.clone(), &mut prover_challenger).unwrap();
        let proof = pcs
            .open(data, protocol.clone(), &mut prover_challenger)
            .unwrap();
        group.bench_with_input(
            BenchmarkId::new("verify", num_variables),
            &num_variables,
            |b, _| {
                b.iter(|| {
                    pcs.verify(&commitment, &proof, &mut challenger(), protocol.clone())
                        .unwrap();
                });
            },
        );
        group.finish();
    }
}

fn report_proof_size(_c: &mut Criterion) {
    eprintln!("\nbinary_whir proof sizes (100-bit target)");
    for num_variables in [12, 16] {
        let (pcs, witness, protocol) = instance(num_variables);
        let mut challenger = challenger();
        let (_, data) = pcs.commit(witness, &mut challenger).unwrap();
        let proof = pcs.open(data, protocol, &mut challenger).unwrap();
        let bytes = postcard::to_allocvec(&proof).unwrap().len();
        eprintln!("2^{num_variables}: {bytes} bytes");
    }
    eprintln!();
}

criterion_group!(benches, bench_whir);
criterion_group!(reports, report_proof_size);
criterion_main!(benches, reports);
