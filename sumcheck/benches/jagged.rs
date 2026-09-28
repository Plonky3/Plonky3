//! Criterion benches for the jagged reduction and its trace ingestion.
//!
//! Every throughput below is the live cell count of the trace.
//!
//! No group charges the power-of-two envelope or the row bound, which is why a sparse commitment exists.

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::DuplexChallenger;
use p3_field::PrimeCharacteristicRing;
use p3_field::extension::BinomialExtensionField;
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;
use p3_sumcheck::jagged::{
    CellBudget, ColumnSource, JaggedLayout, JaggedPoint, JaggedWitness, TraceSource,
};
use p3_util::log2_ceil_usize;
use rand::SeedableRng;
use rand::rngs::SmallRng;

type F = BabyBear;
type EF = BinomialExtensionField<F, 4>;
type Perm = Poseidon2BabyBear<16>;
type Challenger = DuplexChallenger<F, Perm, 16, 8>;

/// Base-two logarithms of the live area benched.
///
/// The smallest is the capacity-free shape, where the live area is a millionth of the row bound.
const AREAS: [usize; 4] = [10, 16, 18, 20];

/// Columns every fixture spreads its live area over.
const COLUMNS: usize = 64;

/// Row bound the machine provisions, far above what any column reaches.
///
/// This is the shape a sparse commitment exists for, and nothing below may scale with it.
const ROW_VARIABLES: usize = 40;

fn challenger() -> Challenger {
    let mut rng = SmallRng::seed_from_u64(1);
    Challenger::new(Perm::new_from_rng_128(&mut rng))
}

/// Unequal column heights summing to exactly one power of two.
///
/// The tallest column is four times the shortest, and the envelope adds nothing to the live area.
fn heights(log_area: usize) -> Vec<usize> {
    let area = 1usize << log_area;
    let shares = (0..COLUMNS)
        .map(|column| 1 + column % 4)
        .collect::<Vec<_>>();
    let total = shares.iter().sum::<usize>();
    let mut heights = shares
        .iter()
        .map(|share| area * share / total)
        .collect::<Vec<_>>();

    // Hand the rounding remainder out one cell at a time so the sum is exact.
    let mut rest = area - heights.iter().sum::<usize>();
    for height in &mut heights {
        if rest == 0 {
            break;
        }
        *height += 1;
        rest -= 1;
    }
    heights
}

/// Equality weights of the first row indices, without materializing the row bound.
fn row_weights(point: &Point<EF>, length: usize) -> Vec<EF> {
    let expanded = point.num_variables().min(log2_ceil_usize(length.max(1)));
    let (leading, trailing) = point.as_slice().split_at(point.num_variables() - expanded);
    let scale = leading
        .iter()
        .fold(EF::ONE, |weight, &value| weight * (EF::ONE - value));
    let mut weights = Poly::new_from_point(trailing, scale).into_evals();
    weights.truncate(length);
    weights
}

/// The value the committed vector takes at the sparse point.
fn claimed_value(layout: &JaggedLayout, witness: &[F], point: &JaggedPoint<EF>) -> EF {
    let columns = Poly::new_from_point(point.column().as_slice(), EF::ONE);
    let tallest = (0..layout.num_columns())
        .map(|column| layout.column_height(column))
        .max()
        .unwrap_or(0);
    let rows = row_weights(point.row(), tallest);

    (0..layout.num_columns())
        .map(|column| {
            let start = layout.cumulative_heights()[column];
            let end = layout.cumulative_heights()[column + 1];
            let inner = witness[start..end]
                .iter()
                .zip(&rows)
                .map(|(&cell, &weight)| weight * cell)
                .sum::<EF>();
            inner * columns.as_slice()[column]
        })
        .sum()
}

/// The geometry, the committed vector and the sparse point of one fixture.
fn fixture(log_area: usize) -> (JaggedLayout, Vec<F>, JaggedPoint<EF>) {
    let mut rng = SmallRng::seed_from_u64(1);
    let layout = JaggedLayout::new(ROW_VARIABLES, &heights(log_area)).unwrap();
    let witness = (0..layout.dense_capacity())
        .map(|cell| F::from_u64(cell as u64))
        .collect();
    let point = JaggedPoint::new(
        Point::rand(&mut rng, ROW_VARIABLES),
        Point::rand(&mut rng, layout.column_variables()),
    );
    (layout, witness, point)
}

fn bench_ingest(c: &mut Criterion) {
    let mut group = c.benchmark_group("jagged/ingest");
    for log_area in AREAS {
        let (layout, witness, _) = fixture(log_area);
        group.throughput(Throughput::Elements(layout.area() as u64));

        // A producer that already owns the committed vector reads nothing at all.
        group.bench_function(BenchmarkId::new("committed", log_area), |b| {
            b.iter(|| {
                black_box(
                    JaggedWitness::read(&layout, TraceSource::Committed(black_box(&witness)))
                        .unwrap(),
                )
            });
        });

        // Column-major cells, which is one copy per live cell.
        let mut start = 0;
        let columns = (0..layout.num_columns())
            .map(|column| {
                let end = start + layout.column_height(column);
                let slice = &witness[start..end];
                start = end;
                ColumnSource::Dense(slice)
            })
            .collect::<Vec<_>>();
        group.bench_function(BenchmarkId::new("column_major", log_area), |b| {
            b.iter(|| {
                black_box(
                    JaggedWitness::read(&layout, TraceSource::Columns(black_box(&columns)))
                        .unwrap(),
                )
            });
        });

        // One row-major block, which is one strided gather per live cell.
        // The block is rectangular at the tallest column, which is what a row-major trace is.
        let tallest = (0..layout.num_columns())
            .map(|column| layout.column_height(column))
            .max()
            .unwrap();
        let block = F::zero_vec(COLUMNS * tallest);
        let interleaved = (0..layout.num_columns())
            .map(|column| ColumnSource::Interleaved {
                cells: &block,
                first: column,
                stride: COLUMNS,
                height: layout.column_height(column),
            })
            .collect::<Vec<_>>();
        group.bench_function(BenchmarkId::new("row_major", log_area), |b| {
            b.iter(|| {
                black_box(
                    JaggedWitness::read(&layout, TraceSource::Columns(black_box(&interleaved)))
                        .unwrap(),
                )
            });
        });

        // Bit-sliced cells, which is one widening per live cell.
        let words = vec![0x9E37_79B9_7F4A_7C15u64; layout.area().div_ceil(64)];
        let packed: Vec<ColumnSource<'_, F>> = (0..layout.num_columns())
            .map(|column| ColumnSource::Bits {
                words: &words,
                height: layout.column_height(column),
            })
            .collect::<Vec<_>>();
        group.bench_function(BenchmarkId::new("bit_sliced", log_area), |b| {
            b.iter(|| {
                black_box(
                    JaggedWitness::read(&layout, TraceSource::Columns(black_box(&packed))).unwrap(),
                )
            });
        });
    }
    group.finish();
}

fn bench_reduction(c: &mut Criterion) {
    let mut group = c.benchmark_group("jagged/prove");
    for log_area in AREAS {
        let (layout, witness, point) = fixture(log_area);
        let value = claimed_value(&layout, &witness, &point);

        group.throughput(Throughput::Elements(layout.area() as u64));
        group.bench_function(BenchmarkId::from_parameter(log_area), |b| {
            b.iter(|| {
                black_box(
                    layout
                        .prove(
                            black_box(&witness),
                            black_box(&point),
                            black_box(value),
                            &mut challenger(),
                        )
                        .unwrap(),
                )
            });
        });
    }
    group.finish();
}

fn bench_verify(c: &mut Criterion) {
    // Not a timing, but the cell accounting a sparse commitment is chosen for.
    for log_area in AREAS {
        let heights = heights(log_area);
        let jagged = CellBudget::of(&JaggedLayout::new(ROW_VARIABLES, &heights).unwrap());
        let stacked = CellBudget::stacked(&heights);
        println!(
            "jagged/cells log_area={log_area} live={} jagged={} stacked={}",
            jagged.live(),
            jagged.provisioned(),
            stacked.provisioned(),
        );
    }

    // The verifier's whole cost is the terminal selector, under a forty-variable row bound.
    // A regression here would mean the row bound had re-entered that cost.
    let mut group = c.benchmark_group("jagged/verify");
    for log_area in AREAS {
        let (layout, witness, point) = fixture(log_area);
        let value = claimed_value(&layout, &witness, &point);
        let proof = layout
            .prove(&witness, &point, value, &mut challenger())
            .unwrap()
            .proof()
            .clone();

        group.throughput(Throughput::Elements(1));
        group.bench_function(BenchmarkId::from_parameter(log_area), |b| {
            b.iter(|| {
                black_box(
                    layout
                        .verify(
                            black_box(&point),
                            black_box(value),
                            black_box(&proof),
                            &mut challenger(),
                        )
                        .unwrap(),
                )
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_ingest, bench_reduction, bench_verify);
criterion_main!(benches);
