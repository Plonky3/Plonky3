//! Criterion benches for the univariate-skip low-degree extension.
//!
//! - One row on its own, which is the narrowest call there is.
//! - A block of rows, which is what a caller holding the whole witness makes.
//! - The streaming round message, which is what a prover runs.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use p3_binary_field::{BinaryField8, BinaryField16, BinaryField128, TowerLevel};
use p3_sumcheck::univariate_skip::{CompressedLde, Conjunction, SkipDomain, SkipRound};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

/// Rows one batched measurement extends, a whole column of a modest trace.
const BATCH_ROWS: usize = 1 << 14;

/// Rows one streamed round message covers, tall enough to leave every cache.
const STREAM_ROWS: usize = 1 << 16;

/// The domain shapes the sweep covers.
///
/// ```text
///     k = 6, d = 2    the product form a zerocheck of R1CS shape runs
///     k = 6, d = 4    two extra dimensions, so three times as many points per row
///     k = 7, d = 2    twice the row bytes and twice the points
/// ```
const SHAPES: [(usize, usize); 3] = [(6, 2), (6, 4), (7, 2)];

/// One level's extension, at every shape of the sweep.
fn arm<F: TowerLevel + Send + Sync>(c: &mut Criterion, name: &str) {
    let mut group = c.benchmark_group(format!("skip_lde/{name}"));
    let mut rng = SmallRng::seed_from_u64(9);

    for (log_size, degree) in SHAPES {
        let domain = SkipDomain::<F>::for_degree(log_size, degree).unwrap();
        let lde = CompressedLde::new(&domain).unwrap();
        let parameter = format!("k{log_size}/d{degree}");

        // One row, which is the call the streaming prover makes.
        let row = (0..lde.num_chunks())
            .map(|_| rng.random::<u8>())
            .collect::<Vec<_>>();
        let mut out = F::zero_vec(lde.num_transmitted());

        // Throughput counts the values produced, so shapes of different width compare.
        group.throughput(Throughput::Elements(lde.num_transmitted() as u64));
        group.bench_function(BenchmarkId::new("row", &parameter), |b| {
            b.iter(|| lde.extend(&row, &mut out));
        });

        // A block of rows, which is the call a caller holding the whole witness makes.
        let rows = (0..BATCH_ROWS * lde.num_chunks())
            .map(|_| rng.random::<u8>())
            .collect::<Vec<_>>();
        let mut out = F::zero_vec(BATCH_ROWS * lde.num_transmitted());

        group.throughput(Throughput::Elements(
            (BATCH_ROWS * lde.num_transmitted()) as u64,
        ));
        group.bench_function(BenchmarkId::new("batch", &parameter), |b| {
            b.iter(|| lde.extend_batch(&rows, &mut out));
        });
    }
    group.finish();
}

/// The two levels a skipped round runs over, whose tables differ by a factor of two.
fn bench_skip_lde(c: &mut Criterion) {
    arm::<BinaryField8>(c, "8");
    arm::<BinaryField16>(c, "16");
}

/// The whole round message, formed straight from packed witness rows.
///
/// This is the prover's own call.
/// It shows what the extension is worth against the weighted composition around it.
fn bench_stream_message(c: &mut Criterion) {
    let mut group = c.benchmark_group("skip_stream");
    group.sample_size(20);

    let mut rng = SmallRng::seed_from_u64(11);
    for (log_size, degree) in SHAPES {
        // A conjunction is the three-operand product form a bit-valued rank-one system takes.
        let round = SkipRound::<BinaryField8>::new(log_size, degree).unwrap();
        let row_bytes = round.row_bytes();

        // One packed witness per operand, plus one equality weight per row.
        let operands = (0..3)
            .map(|_| {
                (0..STREAM_ROWS * row_bytes)
                    .map(|_| rng.random::<u8>())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let borrowed = operands.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let eq = (0..STREAM_ROWS)
            .map(|_| rng.random::<BinaryField128>())
            .collect::<Vec<_>>();

        // Throughput counts the extension values the round reads.
        group.throughput(Throughput::Elements(
            (STREAM_ROWS * round.num_transmitted()) as u64,
        ));
        group.bench_function(
            BenchmarkId::from_parameter(format!("k{log_size}/d{degree}")),
            |b| {
                b.iter(|| round.stream_round_message(&borrowed, &eq, &Conjunction));
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_skip_lde, bench_stream_message);
criterion_main!(benches);
