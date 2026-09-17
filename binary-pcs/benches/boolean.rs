//! A Boolean commitment against the same bits committed one element per bit.
//!
//! ```text
//!     packed     bits packed 128 to the element, then ring switched on opening
//!     embedded   one element per bit, opened directly
//! ```
//!
//! Both arms commit the same function and open it at the same point.
//! The times and the proof sizes therefore compare directly.
//!
//! The embedded arm is what a commitment with no packing costs.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use p3_binary_field::{BinaryChallenger, BinaryField128, Gf2, PackedGf2x64};
use p3_binary_pcs::{
    BinaryPcs, BinaryPcsConfig, BinaryPcsParams, BooleanMultilinearPcs, BooleanPcs,
};
use p3_challenger::HashChallenger;
use p3_commit::MultilinearPcs;
use p3_field::PrimeCharacteristicRing;
use p3_keccak::Keccak256Hash;
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_multilinear_util::point::Point;
use p3_sumcheck::layout::{Layout, SuffixProver, Table};
use p3_sumcheck::{OpeningBatch, OpeningProtocol, PrescribedPointPcs, TableShape, TableSpec};
use p3_symmetric::{CompressionFunctionFromHasher, SerializingHasher};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

type EF = BinaryField128;
type MyHash = SerializingHasher<Keccak256Hash>;
type MyCompress = CompressionFunctionFromHasher<Keccak256Hash, 2, 32>;
type MyMmcs = MerkleTreeMmcs<EF, u8, MyHash, MyCompress, 2, 32>;
type MyChallenger = BinaryChallenger<EF, HashChallenger<u8, Keccak256Hash, 32>>;

/// Bit counts under test, in log bits of one Boolean column.
const LOG_BITS: [usize; 3] = [16, 18, 20];

/// Coordinates one element of the widest level absorbs.
const ABSORBED: usize = 7;

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

const fn params() -> BinaryPcsParams {
    BinaryPcsParams {
        log_inv_rate: 2,
        pow_bits: 0,
        security_level: 100,
    }
}

/// A random bit-sliced witness of the given log bit count.
fn witness(log_bits: usize) -> Vec<PackedGf2x64> {
    let mut rng = SmallRng::seed_from_u64(0xB001);
    (0..1 << (log_bits - 6))
        .map(|_| PackedGf2x64::new(rng.random::<u64>()))
        .collect()
}

/// The same bits as one element per bit, which is what no packing commits.
fn embedded(bits: &[PackedGf2x64]) -> Vec<EF> {
    bits.iter()
        .flat_map(|block| {
            (0..PackedGf2x64::WIDTH).map(move |lane| {
                if block.get(lane) == Gf2::ONE {
                    EF::ONE
                } else {
                    EF::ZERO
                }
            })
        })
        .collect()
}

/// A Boolean commitment over a witness of `log_bits` variables.
fn boolean_pcs(log_bits: usize) -> BooleanPcs<EF, MyMmcs, MyMmcs> {
    let config = BinaryPcsConfig::try_new::<EF, EF>(log_bits - ABSORBED, params()).unwrap();
    BooleanPcs::new(config, mmcs(), mmcs(), log_bits).unwrap()
}

/// The same bits committed with no packing at all.
fn embedded_pcs(log_bits: usize) -> BinaryPcs<EF, EF, MyMmcs, MyMmcs> {
    let config = BinaryPcsConfig::try_new::<EF, EF>(log_bits, params()).unwrap();
    BinaryPcs::new(config, mmcs(), mmcs())
}

/// The opening schedule the embedded arm uses: one column, one point.
fn embedded_protocol(log_bits: usize) -> OpeningProtocol {
    OpeningProtocol::new(vec![TableSpec::new(
        TableShape::new(log_bits, 1),
        vec![OpeningBatch::new(vec![0], Vec::new())],
    )])
}

fn bench_commit(c: &mut Criterion) {
    let mut group = c.benchmark_group("boolean_commit");
    group.sample_size(10);
    for &log_bits in &LOG_BITS {
        let bits = witness(log_bits);
        let packed = boolean_pcs(log_bits);
        group.bench_with_input(BenchmarkId::new("packed", log_bits), &bits, |b, bits| {
            b.iter(|| packed.commit_bits(bits, &mut challenger()).unwrap());
        });

        let plain = embedded_pcs(log_bits);
        let cells = embedded(&bits);
        group.bench_with_input(
            BenchmarkId::new("embedded", log_bits),
            &cells,
            |b, cells| {
                b.iter(|| {
                    let table = Table::new(RowMajorMatrix::new(cells.clone(), cells.len()));
                    let witness = SuffixProver::<EF, EF>::new_witness(vec![table], 0);
                    plain.commit(witness, &mut challenger()).unwrap()
                });
            },
        );
    }
    group.finish();
}

fn bench_open(c: &mut Criterion) {
    let mut group = c.benchmark_group("boolean_open");
    group.sample_size(10);
    for &log_bits in &LOG_BITS {
        let bits = witness(log_bits);
        let point = Point::<EF>::rand(&mut SmallRng::seed_from_u64(0xB002), log_bits);

        // The packed arm: a ring switch, then one opening of the packed commitment.
        let packed = boolean_pcs(log_bits);
        let mut chal = challenger();
        let (_, data) = packed.commit_bits(&bits, &mut chal).unwrap();
        let points = vec![point.clone()];
        let (_, proof) = packed
            .open_at_points(data.clone(), &points, &mut chal.clone())
            .unwrap();
        eprintln!(
            "boolean/proof_size_packed/{log_bits}: {} bytes",
            postcard::to_allocvec(&proof).unwrap().len()
        );
        group.bench_with_input(BenchmarkId::new("packed", log_bits), &point, |b, point| {
            b.iter_batched(
                || (data.clone(), chal.clone()),
                |(data, mut chal)| {
                    packed
                        .open_at_points(data, core::slice::from_ref(point), &mut chal)
                        .unwrap()
                },
                criterion::BatchSize::PerIteration,
            );
        });

        // The embedded arm: one opening of a commitment `d` times longer.
        let plain = embedded_pcs(log_bits);
        let protocol = embedded_protocol(log_bits);
        let cells = embedded(&bits);
        let table = Table::new(RowMajorMatrix::new(cells.clone(), cells.len()));
        let mut plain_chal = challenger();
        let (_, plain_data) = plain
            .commit(
                SuffixProver::<EF, EF>::new_witness(vec![table], 0),
                &mut plain_chal,
            )
            .unwrap();
        let plain_proof = plain
            .open_at(
                plain_data.clone(),
                &protocol,
                core::slice::from_ref(&point),
                &mut plain_chal.clone(),
            )
            .unwrap();
        eprintln!(
            "boolean/proof_size_embedded/{log_bits}: {} bytes",
            postcard::to_allocvec(&plain_proof).unwrap().len()
        );
        group.bench_with_input(
            BenchmarkId::new("embedded", log_bits),
            &point,
            |b, point| {
                b.iter_batched(
                    || (plain_data.clone(), plain_chal.clone()),
                    |(data, mut chal)| {
                        plain
                            .open_at(data, &protocol, core::slice::from_ref(point), &mut chal)
                            .unwrap()
                    },
                    criterion::BatchSize::PerIteration,
                );
            },
        );
    }
    group.finish();
}

/// Several points through one claim pool, against one proof per point.
///
/// Both arms answer for the same four claims, so the times and the sizes compare directly.
fn bench_pooled(c: &mut Criterion) {
    const NUM_POINTS: usize = 4;

    let mut group = c.benchmark_group("boolean_pooled");
    group.sample_size(10);
    for &log_bits in &LOG_BITS {
        let bits = witness(log_bits);
        let pcs = boolean_pcs(log_bits);
        let mut rng = SmallRng::seed_from_u64(0xB003);
        let points: Vec<Point<EF>> = (0..NUM_POINTS)
            .map(|_| Point::<EF>::rand(&mut rng, log_bits))
            .collect();

        let mut chal = challenger();
        let (_, data) = pcs.commit_bits(&bits, &mut chal).unwrap();

        let (_, pooled) = pcs
            .open_at_points(data.clone(), &points, &mut chal.clone())
            .unwrap();
        let separate: usize = points
            .iter()
            .map(|point| {
                let (_, proof) = pcs
                    .open_at_points(
                        data.clone(),
                        core::slice::from_ref(point),
                        &mut chal.clone(),
                    )
                    .unwrap();
                postcard::to_allocvec(&proof).unwrap().len()
            })
            .sum();
        eprintln!(
            "boolean/pooled_size/{log_bits}: {} bytes vs {separate} separate",
            postcard::to_allocvec(&pooled).unwrap().len(),
        );

        group.bench_with_input(
            BenchmarkId::new("pooled", log_bits),
            &points,
            |b, points| {
                b.iter_batched(
                    || (data.clone(), chal.clone()),
                    |(data, mut chal)| pcs.open_at_points(data, points, &mut chal).unwrap(),
                    criterion::BatchSize::PerIteration,
                );
            },
        );
        group.bench_with_input(
            BenchmarkId::new("separate", log_bits),
            &points,
            |b, points| {
                b.iter_batched(
                    || (data.clone(), chal.clone()),
                    |(data, chal)| {
                        for point in points {
                            let _ = pcs
                                .open_at_points(
                                    data.clone(),
                                    core::slice::from_ref(point),
                                    &mut chal.clone(),
                                )
                                .unwrap();
                        }
                    },
                    criterion::BatchSize::PerIteration,
                );
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_commit, bench_open, bench_pooled);
criterion_main!(benches);
