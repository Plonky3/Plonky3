//! Additive NTT, butterfly and Reed-Solomon encoder benchmarks.

use std::hint::black_box;

use criterion::measurement::Measurement;
use criterion::{
    BatchSize, BenchmarkGroup, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main,
};
use p3_baby_bear::BabyBear;
use p3_binary_dft::{AdditiveNtt, AdditiveRsEncoder, ButterflyField, LchNtt, PolyBasisNtt};
use p3_binary_field::{
    BinaryField16, BinaryField32, BinaryField64, BinaryField128, Ghash128, Poly64, TowerLevel,
};
use p3_commit::Encoder;
use p3_dft::Radix2DFTSmallBatch;
use p3_matrix::dense::RowMajorMatrix;
use rand::distr::{Distribution, StandardUniform};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

/// The folding-block width WHIR commits at.
const WIDTH: usize = 16;

/// A single column, the width the binary PCS commits at.
///
/// Its lowest stages pair rows closer than one SIMD register.
const NARROW_WIDTH: usize = 1;

/// Base-two logarithms of the transform heights measured.
const LOG_HEIGHTS: [usize; 4] = [14, 16, 18, 20];

/// One added dimension, so every codeword has rate 1/2.
const LOG_INV_RATE: usize = 1;

/// Rows of `BabyBear` per row of `BinaryField128` at equal byte volume.
const BABY_BEAR_ROW_RATIO: usize = 4;

/// Elements on each side of one butterfly measurement, small enough to stay in L1.
const BUTTERFLY_RUN: usize = 1 << 10;

/// Elements on each side of a run shorter than one SIMD register at every level.
const BUTTERFLY_SHORT: usize = 3;

/// One twiddle per subfield the butterfly splits on, labelled by its bit width.
const TWIDDLES: [(&str, u128); 4] = [
    ("t8", 0xa5),
    ("t16", 0xa5b3),
    ("t32", 0xa5b3_c7d1),
    ("t128", 0xa5b3_c7d1_e9f2_0b47_5c8e_1d39_6a24_f80b),
];

/// The butterfly kernel alone, one arm per level and twiddle width.
fn bench_butterfly(c: &mut Criterion) {
    let mut group = c.benchmark_group("butterfly");

    /// One level's twiddle widths, at one run length.
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

        // A butterfly writes both sides, so both count.
        group.throughput(Throughput::Elements(2 * run as u64));

        for (label, bits) in TWIDDLES {
            // A twiddle wider than the level keeps the coordinates the level has.
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
    arm::<Poly64, _>(&mut group, "64/poly", BUTTERFLY_RUN);
    arm::<BinaryField128, _>(&mut group, "128/tower", BUTTERFLY_RUN);
    arm::<Ghash128, _>(&mut group, "128/ghash", BUTTERFLY_RUN);

    // Runs no register covers, where a kernel that prepares before it checks pays for nothing.
    arm::<BinaryField32, _>(&mut group, "32/short", BUTTERFLY_SHORT);
    arm::<Poly64, _>(&mut group, "64/poly/short", BUTTERFLY_SHORT);
    arm::<BinaryField128, _>(&mut group, "128/tower/short", BUTTERFLY_SHORT);
    group.finish();
}

/// The forward transform of one matrix at each height, over the subspace or a coset.
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

        // Counting entries lets arms of different element sizes compare.
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

/// Every backend's forward transform, at the folding width and on a single column.
///
/// - `128/tower` is the level's own transform, which changes basis in a separate pass each way where the target multiplies carrylessly and enough stages hold a twiddle wider than its byte map covers.
/// - `128/poly` changes basis once on the way in and once on the way out.
/// - `128/ghash` holds data already in the basis the carryless multiply wants.
fn bench_ntt(c: &mut Criterion) {
    let tower = LchNtt::<BinaryField128>::default();
    for (suffix, width) in [("", WIDTH), ("/narrow", NARROW_WIDTH)] {
        ntt(
            c,
            &format!("32{suffix}"),
            width,
            &LchNtt::<BinaryField32>::default(),
            None,
        );
        ntt(
            c,
            &format!("64{suffix}"),
            width,
            &LchNtt::<BinaryField64>::default(),
            None,
        );
        ntt(
            c,
            &format!("64/poly{suffix}"),
            width,
            &LchNtt::<Poly64>::default(),
            None,
        );
        ntt(c, &format!("128/tower{suffix}"), width, &tower, None);
        ntt(c, &format!("128/poly{suffix}"), width, &PolyBasisNtt, None);
        ntt(
            c,
            &format!("128/ghash{suffix}"),
            width,
            &LchNtt::<Ghash128>::default(),
            None,
        );
    }

    // A coset shift puts every twiddle outside the small subfields.
    let shift = BinaryField128::from_repr(0x5555_1234_9abc_def0_0f1e_2d3c_4b5a_6978);
    ntt(c, "128/tower/shifted", WIDTH, &tower, Some(shift));
}

/// One padded encode, the call a commitment makes, at each height of the message.
fn encode_padded<F, E>(c: &mut Criterion, name: &str, width: usize, encoder: &E)
where
    F: TowerLevel,
    E: Encoder<F>,
    StandardUniform: Distribution<F>,
{
    let mut group = c.benchmark_group(format!("encode_padded/{name}"));
    group.sample_size(10);

    let mut rng = SmallRng::seed_from_u64(3);
    for log_height in LOG_HEIGHTS {
        // The message rows, then the zero tail the codeword grows into.
        let mut message = RowMajorMatrix::<F>::rand(&mut rng, 1 << log_height, width);
        message
            .values
            .resize(message.values.len() << LOG_INV_RATE, F::ZERO);

        group.throughput(Throughput::Elements(message.values.len() as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(log_height),
            encoder,
            |b, encoder| {
                b.iter_batched(
                    || message.clone(),
                    |m| encoder.encode_batch_padded(m, LOG_INV_RATE),
                    BatchSize::PerIteration,
                );
            },
        );
    }
}

/// Padded encoding for each alphabet a commitment uses.
fn bench_encode_padded(c: &mut Criterion) {
    let wide = AdditiveRsEncoder::<BinaryField128>::default();
    encode_padded(c, "128/narrow", NARROW_WIDTH, &wide);
    encode_padded(c, "128", WIDTH, &wide);

    let tower = AdditiveRsEncoder::new(LchNtt::<BinaryField128>::default());
    encode_padded(c, "128/tower", WIDTH, &tower);

    let poly64 = AdditiveRsEncoder::new(LchNtt::<Poly64>::default());
    encode_padded(c, "64/poly/narrow", NARROW_WIDTH, &poly64);
    encode_padded(c, "64/poly", WIDTH, &poly64);

    let narrow = AdditiveRsEncoder::new(LchNtt::<BinaryField32>::default());
    encode_padded(c, "32", WIDTH, &narrow);
}

/// Binary against prime-field encoding at equal byte volume.
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

        // A prime-field symbol is a quarter of the size, so the matrix is four times taller.
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

/// The production layout, encoder and Merkle commitment together.
fn bench_commit(c: &mut Criterion) {
    use p3_keccak::Keccak256Hash;
    use p3_merkle_tree::MerkleTreeMmcs;
    use p3_multilinear_util::poly::Poly;
    use p3_sumcheck::commit::{commit_base, write_stacked_message};
    use p3_sumcheck::strategy::VariableOrder;
    use p3_symmetric::{CompressionFunctionFromHasher, SerializingHasher};

    type Hash = SerializingHasher<Keccak256Hash>;
    type Compress = CompressionFunctionFromHasher<Keccak256Hash, 2, 32>;
    type Mmcs = MerkleTreeMmcs<BinaryField128, u8, Hash, Compress, 2, 32>;

    let mmcs = Mmcs::new(Hash::new(Keccak256Hash), Compress::new(Keccak256Hash), 0);
    let encoder = AdditiveRsEncoder::<BinaryField128>::default();
    let mut rng = SmallRng::seed_from_u64(11);
    let mut group = c.benchmark_group("commit_base");
    group.sample_size(10);

    for log_height in [12, 16] {
        for folding in [0, 4] {
            let matrix =
                RowMajorMatrix::<BinaryField128>::rand(&mut rng, 1 << log_height, 1 << folding);
            let poly = Poly::new(matrix.values);
            let num_variables = poly.num_variables();
            for order in [VariableOrder::Prefix, VariableOrder::Suffix] {
                let parameter = format!("{order:?}/h{log_height}/w{}", 1 << folding);
                group.bench_function(parameter, |b| {
                    b.iter(|| {
                        commit_base(&encoder, &mmcs, num_variables, folding, LOG_INV_RATE, |m| {
                            write_stacked_message(order, &poly, folding, m);
                        })
                    });
                });
            }
        }
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_butterfly,
    bench_ntt,
    bench_encode_padded,
    bench_encode,
    bench_commit
);
criterion_main!(benches);
