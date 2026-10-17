//! Mixed trace commitment and Boolean trace opening phase benchmarks.
use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use p3_binary_field::{BinaryChallenger, BinaryField128};
use p3_binary_pcs::{BinaryPcsConfig, BinaryPcsParams, BooleanTracePcs, MixedTraceCommitment};
use p3_challenger::HashChallenger;
use p3_commit::MultilinearPcs;
use p3_keccak::Keccak256Hash;
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_sumcheck::TableShape;
use p3_sumcheck::layout::{Table, plan_stacked_layout};
use p3_symmetric::{CompressionFunctionFromHasher, SerializingHasher};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

type F = BinaryField128;
type M = MerkleTreeMmcs<
    F,
    u8,
    SerializingHasher<Keccak256Hash>,
    CompressionFunctionFromHasher<Keccak256Hash, 2, 32>,
    2,
    32,
>;
type Challenger = BinaryChallenger<F, HashChallenger<u8, Keccak256Hash, 32>>;
const fn challenger() -> Challenger {
    Challenger::from_hasher(Vec::new(), Keccak256Hash)
}
const fn mmcs() -> M {
    M::new(
        SerializingHasher::new(Keccak256Hash),
        CompressionFunctionFromHasher::new(Keccak256Hash),
        0,
    )
}
fn scheme(shapes: &[TableShape]) -> BooleanTracePcs<F, M, M> {
    let arity = plan_stacked_layout(shapes).0;
    let params = BinaryPcsParams {
        log_inv_rate: 2,
        pow_bits: 0,
        security_level: 40,
    };
    let config = BinaryPcsConfig::try_new::<F, F>(arity - 7, params).unwrap();
    BooleanTracePcs::new(config, mmcs(), mmcs(), arity).unwrap()
}
fn bench_mixed(c: &mut Criterion) {
    let mut group = c.benchmark_group("mixed_trace_commit");
    group.sample_size(10);
    for log_height in [6, 10, 14] {
        for density in [0u32, 1, 4, 8, 32, 64, 128] {
            let mut rng = SmallRng::seed_from_u64(0xD35E);
            let cells = (0..1usize << log_height)
                .map(|_| {
                    let mut bytes = [0u8; 16];
                    for _ in 0..density {
                        let bit = rng.random_range(0..128);
                        bytes[bit / 8] |= 1 << (bit % 8);
                    }
                    F::from_le_bytes(bytes)
                })
                .collect();
            let table = Table::new(RowMajorMatrix::new(cells, 1usize << log_height));
            let pcs =
                MixedTraceCommitment::new(scheme(&[TableShape::new(log_height, 128)]), vec![0]);
            group.bench_with_input(
                BenchmarkId::new(log_height.to_string(), density),
                &table,
                |b, table| {
                    b.iter_batched(
                        || (vec![table.clone()], challenger()),
                        |(tables, mut chal)| pcs.commit(tables, &mut chal).unwrap(),
                        BatchSize::PerIteration,
                    );
                },
            );
        }
    }
    group.finish();
}
criterion_group!(benches, bench_mixed);
criterion_main!(benches);
