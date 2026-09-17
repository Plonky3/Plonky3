//! Additive NTT and Reed–Solomon encoder benchmarks.

use std::hint::black_box;

use criterion::measurement::Measurement;
use criterion::{
    BatchSize, BenchmarkGroup, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main,
};
use p3_baby_bear::BabyBear;
use p3_binary_dft::{
    AdditiveNtt, AdditiveRsEncoder, ButterflyField, LchNtt, PolyBasisNtt, interleaved_encode_batch,
    subfield_ntt_batch,
};
use p3_binary_field::{
    BinaryField8, BinaryField16, BinaryField32, BinaryField64, BinaryField128, Ghash128, TowerLevel,
};
use p3_commit::Encoder;
use p3_dft::Radix2DFTSmallBatch;
use p3_field::PrimeCharacteristicRing;
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixView, RowMajorMatrixViewMut};
use rand::distr::{Distribution, StandardUniform};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

/// WHIR's folding-block width.
const WIDTH: usize = 16;

/// A single-column matrix, the narrowest shape the transform runs on.
///
/// Half of the stages are then far below one SIMD register.
/// Those stages measure what the kernel costs when it can pack nothing at all.
const NARROW_WIDTH: usize = 1;

/// The base-two logarithms of the transform heights the sweep covers.
const LOG_HEIGHTS: [usize; 4] = [14, 16, 18, 20];

/// One added bit of domain, so the codewords have rate `1/2`.
const LOG_INV_RATE: usize = 1;

/// A `BinaryField128` symbol is 16 bytes and a `BabyBear` symbol 4.
/// So equal byte volume is four times as many rows.
const BABY_BEAR_ROW_RATIO: usize = 4;

/// The transform over the subspace itself, and over a coset of it.
///
/// The sweep starts at the four-byte level, because `S_ℓ` needs `ℓ` Cantor basis vectors.
///
/// A two-byte level has only sixteen of them, short of the heights measured here.
fn bench_ntt(c: &mut Criterion) {
    ntt(c, "32", WIDTH, &LchNtt::<BinaryField32>::default(), None);
    ntt(c, "64", WIDTH, &LchNtt::<BinaryField64>::default(), None);

    // The three routes to a `GF(2^128)` transform.
    //
    // ```text
    //     128/tower   the data stays in the tower basis the field is defined in
    //     128/hybrid  the matrix changes basis once on the way in and once on the way out
    //     128/ghash   the data is already in the basis the carryless multiply wants
    // ```
    let tower = LchNtt::<BinaryField128>::default();
    ntt(c, "128/tower", WIDTH, &tower, None);
    ntt(c, "128/hybrid", WIDTH, &PolyBasisNtt::default(), None);
    ntt(c, "128/ghash", WIDTH, &LchNtt::<Ghash128>::default(), None);

    // A coset shift puts every twiddle outside the small subfields.
    //
    // So this arm measures the transform with no subfield structure left to exploit.
    let shift = BinaryField128::from_repr(0x5555_1234_9abc_def0_0f1e_2d3c_4b5a_6978);
    ntt(c, "128/tower/shifted", WIDTH, &tower, Some(shift));

    // The same levels on a single column.
    //
    // A stage pairing rows less than a register apart cannot pack anything.
    // So these arms are where work done before that is discovered shows up.
    ntt(
        c,
        "32/narrow",
        NARROW_WIDTH,
        &LchNtt::<BinaryField32>::default(),
        None,
    );
    ntt(
        c,
        "64/narrow",
        NARROW_WIDTH,
        &LchNtt::<BinaryField64>::default(),
        None,
    );
    ntt(c, "128/tower/narrow", NARROW_WIDTH, &tower, None);
}

/// The forward transform of one matrix, at each height of the sweep.
///
/// A shift selects a coset of `S_ℓ`.
///
/// Without one the transform runs over the subspace itself.
fn ntt<F: TowerLevel, N: AdditiveNtt<F>>(
    c: &mut Criterion,
    name: &str,
    width: usize,
    ntt: &N,
    shift: Option<F>,
) where
    StandardUniform: Distribution<F>,
{
    let mut group = c.benchmark_group(format!("ntt/{name}"));
    group.sample_size(10);

    let mut rng = SmallRng::seed_from_u64(1);
    for log_height in LOG_HEIGHTS {
        let coeffs = RowMajorMatrix::<F>::rand(&mut rng, 1 << log_height, width);

        // Throughput counts the matrix entries, so arms at different element sizes compare.
        group.throughput(Throughput::Elements((width << log_height) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(log_height), ntt, |b, ntt| {
            b.iter_batched(
                || coeffs.clone(),
                |m| match shift {
                    Some(shift) => ntt.shifted_ntt_batch(m, shift),
                    None => ntt.ntt_batch(m),
                },
                BatchSize::PerIteration,
            );
        });
    }
}

/// Elements on each side of one butterfly measurement, small enough to stay in cache.
const BUTTERFLY_RUN: usize = 1 << 10;

/// Elements on each side of a run too short for one SIMD register, at every level.
///
/// The widest level is sixteen bytes, so three of them still fall below a register.
///
/// A narrow matrix presents this shape at its lowest stages.
/// The kernel has to rule packing out before doing any work towards it.
const BUTTERFLY_SHORT: usize = 3;

/// One twiddle per subfield the butterfly splits on, labelled by its bit width.
const TWIDDLES: [(&str, u128); 4] = [
    ("t8", 0xa5),
    ("t16", 0xa5b3),
    ("t32", 0xa5b3_c7d1),
    ("t128", 0xa5b3_c7d1_e9f2_0b47_5c8e_1d39_6a24_f80b),
];

/// The butterfly kernel alone, on runs small enough to stay in the first-level cache.
///
/// A transform's twiddle at stage `j` and block `b` is `W_j(shift) + domain_point(2b)`.
///
/// Over the subspace itself the stages near the top land in a small tower subfield.
///
/// The ones near the bottom do not.
///
/// This group measures each of those regimes on its own, by the width of the twiddle.
fn bench_butterfly(c: &mut Criterion) {
    let mut group = c.benchmark_group("butterfly");

    // One level's four twiddle widths, at one run length.
    fn arm<F: ButterflyField, M: Measurement>(
        group: &mut BenchmarkGroup<'_, M>,
        name: &str,
        run: usize,
    ) where
        StandardUniform: Distribution<F>,
    {
        let mut rng = SmallRng::seed_from_u64(5);
        let mut lo: Vec<F> = (0..run).map(|_| rng.random()).collect();
        let mut hi: Vec<F> = (0..run).map(|_| rng.random()).collect();

        // Throughput counts both sides, since a butterfly writes each of them once.
        group.throughput(Throughput::Elements(2 * run as u64));

        for (label, bits) in TWIDDLES {
            // A twiddle wider than the level keeps only the coordinates the level has.
            let t = F::from_le_byte_iter(bits.to_le_bytes().into_iter());
            group.bench_function(BenchmarkId::new(name, label), |b| {
                b.iter(|| {
                    F::butterfly::<false>(black_box(&mut lo), black_box(&mut hi), black_box(t));
                });
            });
        }
    }

    arm::<BinaryField16, _>(&mut group, "16", BUTTERFLY_RUN);
    arm::<BinaryField32, _>(&mut group, "32", BUTTERFLY_RUN);
    arm::<BinaryField64, _>(&mut group, "64", BUTTERFLY_RUN);
    arm::<BinaryField128, _>(&mut group, "128/tower", BUTTERFLY_RUN);
    arm::<Ghash128, _>(&mut group, "128/ghash", BUTTERFLY_RUN);

    // The same levels on a run no register covers.
    // A kernel that prepares before it checks pays there for work it throws away.
    arm::<BinaryField32, _>(&mut group, "32/short", BUTTERFLY_SHORT);
    arm::<BinaryField64, _>(&mut group, "64/short", BUTTERFLY_SHORT);
    arm::<BinaryField128, _>(&mut group, "128/tower/short", BUTTERFLY_SHORT);
    group.finish();
}

/// Reed–Solomon encoding at equal byte volume, so both arms carry the same message bytes.
///
/// Each pair of entries is parameterised by the binary-field height.
/// The prime-field matrix is taller by the ratio of the two symbol sizes.
fn bench_encode(c: &mut Criterion) {
    let mut group = c.benchmark_group("encode");
    group.sample_size(10);

    let mut rng = SmallRng::seed_from_u64(1);
    let binary = AdditiveRsEncoder::<BinaryField128>::default();
    let baby_bear = Radix2DFTSmallBatch::<BabyBear>::default();

    for log_height in LOG_HEIGHTS {
        let message = RowMajorMatrix::<BinaryField128>::rand(&mut rng, 1 << log_height, WIDTH);
        group.bench_with_input(
            BenchmarkId::new("BinaryField128", log_height),
            &binary,
            |b, encoder| {
                b.iter_batched(
                    || message.clone(),
                    |m| encoder.encode_batch(m, LOG_INV_RATE),
                    BatchSize::PerIteration,
                );
            },
        );

        let message =
            RowMajorMatrix::<BabyBear>::rand(&mut rng, BABY_BEAR_ROW_RATIO << log_height, WIDTH);
        group.bench_with_input(
            BenchmarkId::new("BabyBear", log_height),
            &baby_bear,
            |b, encoder| {
                b.iter_batched(
                    || message.clone(),
                    |m| encoder.encode_batch(m, LOG_INV_RATE),
                    BatchSize::PerIteration,
                );
            },
        );
    }
}

/// The two routes from a byte-valued message to a wide codeword.
///
/// Either way the caller ends up with `BinaryField128` evaluations.
///
/// ```text
///     wide      widen every entry, then transform at the wide element size
///     subfield  transform the closed layers at the byte size, then widen
/// ```
///
/// The wide arm carries the widening pass too, since a caller pays it in both routes.
fn bench_subfield(c: &mut Criterion) {
    let mut group = c.benchmark_group("subfield");
    group.sample_size(10);

    let mut rng = SmallRng::seed_from_u64(3);
    let lch = LchNtt::<BinaryField128>::default();
    let poly = PolyBasisNtt::default();

    for log_height in LOG_HEIGHTS {
        for width in [1, WIDTH] {
            let message = RowMajorMatrix::<BinaryField8>::rand(&mut rng, 1 << log_height, width);
            let parameter = format!("h{log_height}/w{width}");

            // Throughput counts the matrix entries, so every arm compares directly.
            group.throughput(Throughput::Elements((width << log_height) as u64));

            let widen = |m: &RowMajorMatrix<BinaryField8>| {
                RowMajorMatrix::new(
                    m.values.iter().copied().map(BinaryField128::from).collect(),
                    width,
                )
            };

            group.bench_function(BenchmarkId::new("wide/lch", &parameter), |b| {
                b.iter(|| lch.ntt_batch(widen(&message)));
            });
            group.bench_function(BenchmarkId::new("wide/poly", &parameter), |b| {
                b.iter(|| poly.ntt_batch(widen(&message)));
            });
            group.bench_function(BenchmarkId::new("subfield", &parameter), |b| {
                b.iter_batched(
                    || message.clone(),
                    subfield_ntt_batch::<BinaryField8, BinaryField128>,
                    BatchSize::PerIteration,
                );
            });
        }
    }
    group.finish();
}

/// The two routes from a column-major message to an interleaved codeword.
///
/// ```text
///     two-pass  interleave in a pass of its own, then encode the padded matrix
///     fused     interleave into the first coset, then transform each coset
/// ```
fn bench_interleaved(c: &mut Criterion) {
    let mut group = c.benchmark_group("interleaved");
    group.sample_size(10);

    let mut rng = SmallRng::seed_from_u64(13);
    let encoder = AdditiveRsEncoder::<BinaryField128>::default();

    for log_message in LOG_HEIGHTS {
        for log_inv_rate in [1usize, 2, 3] {
            let columns = (0..WIDTH << log_message)
                .map(|_| rng.random::<BinaryField128>())
                .collect::<Vec<_>>();
            let parameter = format!("h{log_message}/r{log_inv_rate}");

            // Throughput counts the codeword entries, so the rates compare.
            group.throughput(Throughput::Elements(
                ((WIDTH << log_message) << log_inv_rate) as u64,
            ));

            group.bench_function(BenchmarkId::new("two_pass", &parameter), |b| {
                b.iter(|| {
                    // The layout `commit_base` builds before it hands the encoder a matrix.
                    let mut values = BinaryField128::zero_vec(columns.len() << log_inv_rate);
                    let source = RowMajorMatrixView::new(&columns, 1 << log_message);
                    let mut target =
                        RowMajorMatrixViewMut::new(&mut values[..columns.len()], WIDTH);
                    source.transpose_into(&mut target);
                    encoder.encode_batch_padded(RowMajorMatrix::new(values, WIDTH), log_inv_rate)
                });
            });

            // The same two passes over the transform the fused path itself runs on.
            // The pair therefore isolates the fusion from the choice of backend.
            let lch = LchNtt::<BinaryField128>::default();
            group.bench_function(BenchmarkId::new("two_pass/lch", &parameter), |b| {
                b.iter(|| {
                    let mut values = BinaryField128::zero_vec(columns.len() << log_inv_rate);
                    let source = RowMajorMatrixView::new(&columns, 1 << log_message);
                    let mut target =
                        RowMajorMatrixViewMut::new(&mut values[..columns.len()], WIDTH);
                    source.transpose_into(&mut target);
                    lch.ntt_batch(RowMajorMatrix::new(values, WIDTH))
                });
            });

            group.bench_function(BenchmarkId::new("fused", &parameter), |b| {
                b.iter(|| interleaved_encode_batch(&columns, log_message, log_inv_rate));
            });
        }
    }
    group.finish();
}

/// Direct polynomial-backend workloads, including small later-round domains.
fn bench_poly(c: &mut Criterion) {
    eprintln!(
        "binary-dft: parallel={}, threads={}, hardware_clmul={}",
        cfg!(feature = "parallel"),
        p3_maybe_rayon::prelude::current_num_threads(),
        p3_binary_field::poly_basis::HAS_HARDWARE_CLMUL
    );
    let mut group = c.benchmark_group("poly");
    group.sample_size(10);
    let mut rng = SmallRng::seed_from_u64(7);
    let ntt = PolyBasisNtt::default();
    let encoder = AdditiveRsEncoder::<BinaryField128>::default();
    let shift = BinaryField128::from_repr(1 << 127);
    for log_height in [4, 8, 12, 16] {
        for width in [1, 4, 16, 64] {
            let mat = RowMajorMatrix::<BinaryField128>::rand(&mut rng, 1 << log_height, width);
            let parameter = format!("h{log_height}/w{width}");
            for op in ["forward", "inverse", "shifted"] {
                group.bench_with_input(BenchmarkId::new(op, &parameter), &mat, |b, mat| {
                    b.iter_batched(
                        || mat.clone(),
                        |m| match op {
                            "forward" => ntt.ntt_batch(m),
                            "inverse" => ntt.intt_batch(m),
                            _ => ntt.shifted_ntt_batch(m, shift),
                        },
                        BatchSize::PerIteration,
                    );
                });
            }
            for added in [0, 1, 2, 3] {
                group.bench_with_input(
                    BenchmarkId::new(format!("lde/r{added}"), &parameter),
                    &mat,
                    |b, mat| {
                        b.iter_batched(
                            || mat.clone(),
                            |m| ntt.shifted_lde_batch(m, added, shift),
                            BatchSize::PerIteration,
                        );
                    },
                );
                group.bench_with_input(
                    BenchmarkId::new(format!("encode/r{added}"), &parameter),
                    &mat,
                    |b, mat| {
                        b.iter_batched(
                            || mat.clone(),
                            |m| encoder.encode_batch(m, added),
                            BatchSize::PerIteration,
                        );
                    },
                );
            }
        }
    }
    group.finish();
}

/// The production layout, encoder and Merkle commitment together.
fn bench_commit(c: &mut Criterion) {
    use p3_keccak::Keccak256Hash;
    use p3_merkle_tree::MerkleTreeMmcs;
    use p3_multilinear_util::poly::Poly;
    use p3_sumcheck::commit::commit_base;
    use p3_sumcheck::strategy::VariableOrder;
    use p3_symmetric::{CompressionFunctionFromHasher, SerializingHasher};

    type Hash = SerializingHasher<Keccak256Hash>;
    type Compress = CompressionFunctionFromHasher<Keccak256Hash, 2, 32>;
    type Mmcs = MerkleTreeMmcs<BinaryField128, u8, Hash, Compress, 2, 32>;
    let mmcs = Mmcs::new(Hash::new(Keccak256Hash), Compress::new(Keccak256Hash), 0);
    // Retain the previous padded full transform as an allocation-matched comparison.
    struct FullPaddedEncoder;
    impl Encoder<BinaryField128> for FullPaddedEncoder {
        fn encode_batch(
            &self,
            message: RowMajorMatrix<BinaryField128>,
            rate: usize,
        ) -> RowMajorMatrix<BinaryField128> {
            AdditiveRsEncoder::<BinaryField128>::default().encode_batch(message, rate)
        }
    }
    let encoder = AdditiveRsEncoder::<BinaryField128>::default();
    let mut rng = SmallRng::seed_from_u64(11);
    let mut group = c.benchmark_group("commit_base");
    group.sample_size(10);
    for log_height in [8, 12, 16] {
        for folding in [0, 2, 4, 6] {
            let matrix =
                RowMajorMatrix::<BinaryField128>::rand(&mut rng, 1 << log_height, 1 << folding);
            let poly = Poly::new(matrix.values);
            for added in [1, 2, 3] {
                for order in [VariableOrder::Prefix, VariableOrder::Suffix] {
                    let parameter = format!("{order:?}/h{log_height}/w{}/r{added}", 1 << folding);
                    group.bench_function(format!("full/{parameter}"), |b| {
                        b.iter(|| {
                            commit_base(order, &FullPaddedEncoder, &mmcs, &poly, folding, added)
                        });
                    });
                    group.bench_function(parameter, |b| {
                        b.iter(|| commit_base(order, &encoder, &mmcs, &poly, folding, added));
                    });
                }
            }
        }
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_butterfly,
    bench_ntt,
    bench_subfield,
    bench_interleaved,
    bench_encode,
    bench_poly,
    bench_commit
);
criterion_main!(benches);
