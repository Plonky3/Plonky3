//! Mixed trace commitment and Boolean trace opening phase benchmarks.
use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use p3_binary_field::{BinaryChallenger, BinaryField128, PackedGf2x64};
use p3_binary_pcs::{
    BinaryPcsConfig, BinaryPcsParams, BitOpening, BitReadings, BooleanMultilinearPcs, BooleanPcs,
    BooleanTraceCommitmentProof, BooleanTracePcs, MixedTraceCommitment,
};
use p3_challenger::HashChallenger;
use p3_commit::MultilinearPcs;
use p3_field::PrimeCharacteristicRing;
use p3_keccak::Keccak256Hash;
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_multilinear_util::point::Point;
use p3_sumcheck::layout::{Table, plan_stacked_layout};
use p3_sumcheck::{OpeningBatch, OpeningProtocol, PrescribedPointPcs, TableShape, TableSpec};
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
fn backend(shapes: &[TableShape]) -> BooleanPcs<F, M, M> {
    let arity = plan_stacked_layout(shapes).0;
    let params = BinaryPcsParams {
        log_inv_rate: 2,
        pow_bits: 0,
        security_level: 40,
    };
    let config = BinaryPcsConfig::try_new::<F, F>(arity - 7, params).unwrap();
    BooleanPcs::new(config, mmcs(), mmcs(), arity).unwrap()
}
fn scheme(shapes: &[TableShape]) -> BooleanTracePcs<F, M, M> {
    BooleanTracePcs::from_commitment(backend(shapes))
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

fn legacy_values(
    readings: &[BitReadings<F>],
    width: usize,
    successors: usize,
    partial: bool,
) -> Vec<F> {
    let mut values: Vec<_> = readings[..width]
        .iter()
        .map(|reading| reading.current.unwrap())
        .collect();
    values.extend(
        readings[..successors]
            .iter()
            .map(|reading| reading.next.unwrap()),
    );
    if partial {
        values.extend(
            readings[width..]
                .iter()
                .map(|reading| reading.current.unwrap()),
        );
        values.push(readings[width + 1].next.unwrap());
    }
    values
}

fn bench_trace_openings(c: &mut Criterion) {
    let mut group = c.benchmark_group("boolean_trace_openings");
    group.sample_size(10);
    for (width, rows) in [(64, 3), (64, 10), (1024, 3)] {
        let shape = TableShape::new(rows, width);
        let pcs = scheme(&[shape]);
        let mut rng = SmallRng::seed_from_u64(0xBA7C);
        let cells = (0..width << rows)
            .map(|_| F::from_bool(rng.random()))
            .collect();
        let table = Table::new(RowMajorMatrix::new(cells, 1usize << rows));
        let legacy = backend(&[shape]);
        let (arity, placements) = plan_stacked_layout(&[shape]);
        let mut words = vec![0u64; 1usize << (arity - 6)];
        for column in 0..width {
            for row in 0..1usize << rows {
                if table.column(column).value(row) == F::ONE {
                    let bit = (column << rows) + row;
                    words[bit / 64] |= 1 << (bit % 64);
                }
            }
        }
        let bits: Vec<_> = words.into_iter().map(PackedGf2x64::new).collect();
        let mut legacy_after_commit = challenger();
        let (legacy_root, legacy_data) =
            legacy.commit_bits(&bits, &mut legacy_after_commit).unwrap();
        let point = Point::<F>::rand(&mut rng, rows);
        let mut after_commit = challenger();
        let (root, data) = pcs.commit(vec![table], &mut after_commit).unwrap();
        assert_eq!(
            postcard::to_allocvec(&root).unwrap(),
            postcard::to_allocvec(&legacy_root).unwrap()
        );
        for successors in [0, 1, 8, width] {
            for partial in [false, true] {
                let mut schedule = vec![OpeningBatch::new(
                    (0..width).collect(),
                    (0..successors).collect(),
                )];
                if partial {
                    schedule.push(OpeningBatch::new(vec![width - 1, 0], vec![0]));
                }
                let request = OpeningProtocol::new(vec![TableSpec::new(shape, schedule)]);
                let points = vec![point.clone(); request.num_openings()];
                let proof = pcs
                    .open_at(data.clone(), &request, &points, &mut after_commit.clone())
                    .unwrap();
                let id = format!("w{width}_h{rows}_s{successors}_indexed{partial}");
                // The original per-column route: each current column shares its one
                // reduction with its successor reading; indexed extras follow it.
                let mut legacy_openings: Vec<_> = (0..width)
                    .map(|column| BitOpening {
                        point: placements[0].selectors()[column].lift_prefix(&point),
                        row_variables: rows,
                        current: true,
                        next: column < successors,
                    })
                    .collect();
                if partial {
                    for column in [width - 1, 0] {
                        legacy_openings.push(BitOpening {
                            point: placements[0].selectors()[column].lift_prefix(&point),
                            row_variables: rows,
                            current: true,
                            next: column == 0,
                        });
                    }
                }
                let (legacy_readings, legacy_opening) = legacy
                    .open_readings(
                        legacy_data.clone(),
                        &legacy_openings,
                        &mut legacy_after_commit.clone(),
                    )
                    .unwrap();
                let legacy_proof = BooleanTraceCommitmentProof {
                    values: legacy_values(&legacy_readings, width, successors, partial),
                    opening: legacy_opening,
                };
                assert_eq!(proof.values, legacy_proof.values);
                let legacy_tensors: usize = legacy_proof
                    .opening
                    .reduction
                    .claims
                    .iter()
                    .map(|claim| 1 + 2 * usize::from(claim.successor.is_some()))
                    .sum();
                eprintln!(
                    "boolean_trace_legacy/{id}: {} proof bytes, {} claims, {legacy_tensors} tensors",
                    postcard::to_allocvec(&legacy_proof).unwrap().len(),
                    legacy_proof.opening.reduction.claims.len()
                );
                let mut legacy_verifier = challenger();
                legacy.observe_commitment(&legacy_root, &mut legacy_verifier);
                legacy
                    .verify_readings(
                        &legacy_root,
                        &legacy_openings,
                        &legacy_readings,
                        &legacy_proof.opening,
                        &mut legacy_verifier,
                    )
                    .unwrap();
                group.bench_function(BenchmarkId::new("legacy_open", &id), |b| {
                    b.iter_batched(
                        || (legacy_data.clone(), legacy_after_commit.clone()),
                        |(data, mut chal)| {
                            let (readings, opening) = legacy
                                .open_readings(data, &legacy_openings, &mut chal)
                                .unwrap();
                            BooleanTraceCommitmentProof {
                                values: legacy_values(&readings, width, successors, partial),
                                opening,
                            }
                        },
                        BatchSize::PerIteration,
                    );
                });
                group.bench_function(BenchmarkId::new("legacy_verify", &id), |b| {
                    b.iter_batched(
                        || {
                            let mut chal = challenger();
                            legacy.observe_commitment(&legacy_root, &mut chal);
                            chal
                        },
                        |mut chal| {
                            legacy
                                .verify_readings(
                                    &legacy_root,
                                    &legacy_openings,
                                    &legacy_readings,
                                    &legacy_proof.opening,
                                    &mut chal,
                                )
                                .unwrap();
                        },
                        BatchSize::PerIteration,
                    );
                });

                let tensors: usize = proof
                    .opening
                    .reduction
                    .claims
                    .iter()
                    .map(|claim| 1 + 2 * usize::from(claim.successor.is_some()))
                    .sum();
                eprintln!(
                    "boolean_trace/{id}: {} proof bytes, {} claims, {tensors} tensors",
                    postcard::to_allocvec(&proof).unwrap().len(),
                    proof.opening.reduction.claims.len()
                );
                group.bench_function(BenchmarkId::new("open", &id), |b| {
                    b.iter_batched(
                        || (data.clone(), after_commit.clone()),
                        |(data, mut chal)| pcs.open_at(data, &request, &points, &mut chal).unwrap(),
                        BatchSize::PerIteration,
                    );
                });
                group.bench_function(BenchmarkId::new("verify", &id), |b| {
                    b.iter_batched(
                        || {
                            let mut chal = challenger();
                            pcs.observe_commitment(&root, &mut chal);
                            chal
                        },
                        |mut chal| {
                            pcs.verify_at(&root, &proof, &request, &points, &mut chal)
                                .unwrap();
                        },
                        BatchSize::PerIteration,
                    );
                });
            }
        }
    }
    group.finish();
}

criterion_group!(benches, bench_mixed, bench_trace_openings);
criterion_main!(benches);
