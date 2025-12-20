//! Benchmarks for matrix packing operations.
//!
//! These benchmarks measure the performance of:
//! - `vertically_packed_row`: Pack WIDTH consecutive rows into packed field elements
//! - `vertically_packed_row_pair`: Pack two sets of WIDTH rows (current + next)
//! - `horizontally_packed_row`: Pack a single row into packed field elements
//! - `padded_horizontally_packed_row`: Same as above, with zero-padding for remainder
//!
//! ## Running benchmarks
//!
//! ```bash
//! RUSTFLAGS="-Ctarget-cpu=native" cargo bench --bench packing -p p3-matrix
//! ```

use std::hint::black_box;

use criterion::measurement::WallTime;
use criterion::{
    BenchmarkGroup, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main,
};
use p3_baby_bear::BabyBear;
use p3_field::{Field, PackedValue};
use p3_goldilocks::Goldilocks;
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use rand::SeedableRng;
use rand::distr::{Distribution, StandardUniform};
use rand::rngs::SmallRng;

/// Matrix height for all benchmarks (2^16 = 65536 rows)
const LOG_HEIGHT: usize = 16;
const HEIGHT: usize = 1 << LOG_HEIGHT;

/// Matrix widths to test:
/// - 128, 1024: Powers of 2, aligned to all packing widths
/// - 129, 1025: Off-by-one, tests padding/remainder handling
const WIDTHS: &[usize] = &[128, 129, 1024, 1025];

/// Step size for vertically_packed_row_pair (simulates constraint evaluation stride)
/// Using height/2 is realistic for "next row" access patterns
const STEP: usize = HEIGHT / 2;

// ============================================================================
// Generic Benchmark Implementations
// ============================================================================

fn bench_vertically_packed_row_impl<F>(c: &mut Criterion, field_name: &str)
where
    F: Field,
    StandardUniform: Distribution<F>,
{
    let mut group = c.benchmark_group(format!("vertically_packed_row/{field_name}"));
    group.sample_size(50);

    let packing_width = <F as Field>::Packing::WIDTH;

    for &width in WIDTHS {
        let mut rng = SmallRng::seed_from_u64(42);
        let matrix = RowMajorMatrix::<F>::rand(&mut rng, HEIGHT, width);

        let num_calls = HEIGHT / packing_width;
        let total_elements = num_calls * width;
        group.throughput(Throughput::Elements(total_elements as u64));

        group.bench_with_input(
            BenchmarkId::new(format!("w{packing_width}"), format!("{width}x{HEIGHT}")),
            &matrix,
            |b, matrix| {
                b.iter(|| {
                    for r in (0..HEIGHT).step_by(packing_width) {
                        for packed in matrix.vertically_packed_row::<<F as Field>::Packing>(r) {
                            black_box(packed);
                        }
                    }
                });
            },
        );
    }
    group.finish();
}

fn bench_vertically_packed_row_pair_impl<F>(c: &mut Criterion, field_name: &str)
where
    F: Field,
    StandardUniform: Distribution<F>,
{
    let mut group = c.benchmark_group(format!("vertically_packed_row_pair/{field_name}"));
    group.sample_size(50);

    let packing_width = <F as Field>::Packing::WIDTH;

    for &width in WIDTHS {
        let mut rng = SmallRng::seed_from_u64(42);
        let matrix = RowMajorMatrix::<F>::rand(&mut rng, HEIGHT, width);

        let num_calls = HEIGHT / packing_width;
        let total_elements = num_calls * width * 2;
        group.throughput(Throughput::Elements(total_elements as u64));

        group.bench_with_input(
            BenchmarkId::new(format!("w{packing_width}"), format!("{width}x{HEIGHT}")),
            &matrix,
            |b, matrix| {
                b.iter(|| {
                    for r in (0..HEIGHT).step_by(packing_width) {
                        black_box(
                            matrix.vertically_packed_row_pair::<<F as Field>::Packing>(r, STEP),
                        );
                    }
                });
            },
        );
    }
    group.finish();
}

fn bench_horizontally_packed_row_impl<F>(c: &mut Criterion, field_name: &str)
where
    F: Field,
    StandardUniform: Distribution<F>,
{
    let mut group = c.benchmark_group(format!("horizontally_packed_row/{field_name}"));
    group.sample_size(50);

    let packing_width = <F as Field>::Packing::WIDTH;

    for &width in WIDTHS {
        let mut rng = SmallRng::seed_from_u64(42);
        let matrix = RowMajorMatrix::<F>::rand(&mut rng, HEIGHT, width);

        let total_elements = HEIGHT * width;
        group.throughput(Throughput::Elements(total_elements as u64));

        group.bench_with_input(
            BenchmarkId::new(format!("w{packing_width}"), format!("{width}x{HEIGHT}")),
            &matrix,
            |b, matrix| {
                b.iter(|| {
                    for r in 0..HEIGHT {
                        let (packed, suffix) =
                            matrix.horizontally_packed_row::<<F as Field>::Packing>(r);
                        for p in packed {
                            black_box(p);
                        }
                        for s in suffix {
                            black_box(s);
                        }
                    }
                });
            },
        );
    }
    group.finish();
}

fn bench_padded_horizontally_packed_row_impl<F>(c: &mut Criterion, field_name: &str)
where
    F: Field,
    StandardUniform: Distribution<F>,
{
    let mut group = c.benchmark_group(format!("padded_horizontally_packed_row/{field_name}"));
    group.sample_size(50);

    let packing_width = <F as Field>::Packing::WIDTH;

    for &width in WIDTHS {
        let mut rng = SmallRng::seed_from_u64(42);
        let matrix = RowMajorMatrix::<F>::rand(&mut rng, HEIGHT, width);

        let packed_per_row = width.div_ceil(packing_width);
        let total_packed = HEIGHT * packed_per_row;
        group.throughput(Throughput::Elements(total_packed as u64));

        group.bench_with_input(
            BenchmarkId::new(format!("w{packing_width}"), format!("{width}x{HEIGHT}")),
            &matrix,
            |b, matrix| {
                b.iter(|| {
                    for r in 0..HEIGHT {
                        for packed in
                            matrix.padded_horizontally_packed_row::<<F as Field>::Packing>(r)
                        {
                            black_box(packed);
                        }
                    }
                });
            },
        );
    }
    group.finish();
}

fn bench_single_call_for_field<F>(group: &mut BenchmarkGroup<'_, WallTime>, field_name: &str)
where
    F: Field,
    StandardUniform: Distribution<F>,
{
    let width = 16;
    let height = 1024;
    let packing_width = <F as Field>::Packing::WIDTH;

    let mut rng = SmallRng::seed_from_u64(42);
    let matrix = RowMajorMatrix::<F>::rand(&mut rng, height, width);

    group.throughput(Throughput::Elements(width as u64));
    group.bench_function(
        BenchmarkId::new(
            "vertically_packed_row",
            format!("{field_name}_w{packing_width}"),
        ),
        |b| {
            let mut r = 0usize;
            b.iter(|| {
                for packed in matrix.vertically_packed_row::<<F as Field>::Packing>(r) {
                    black_box(packed);
                }
                r = (r + packing_width) % height;
            });
        },
    );

    group.throughput(Throughput::Elements((width * 2) as u64));
    group.bench_function(
        BenchmarkId::new(
            "vertically_packed_row_pair",
            format!("{field_name}_w{packing_width}"),
        ),
        |b| {
            let mut r = 0usize;
            b.iter(|| {
                black_box(
                    matrix.vertically_packed_row_pair::<<F as Field>::Packing>(r, height / 2),
                );
                r = (r + packing_width) % height;
            });
        },
    );
}

// ============================================================================
// Entry Points
// ============================================================================

fn bench_vertically_packed_row(c: &mut Criterion) {
    bench_vertically_packed_row_impl::<Goldilocks>(c, "goldilocks");
    bench_vertically_packed_row_impl::<BabyBear>(c, "babybear");
}

fn bench_vertically_packed_row_pair(c: &mut Criterion) {
    bench_vertically_packed_row_pair_impl::<Goldilocks>(c, "goldilocks");
    bench_vertically_packed_row_pair_impl::<BabyBear>(c, "babybear");
}

fn bench_horizontally_packed_row(c: &mut Criterion) {
    bench_horizontally_packed_row_impl::<Goldilocks>(c, "goldilocks");
    bench_horizontally_packed_row_impl::<BabyBear>(c, "babybear");
}

fn bench_padded_horizontally_packed_row(c: &mut Criterion) {
    bench_padded_horizontally_packed_row_impl::<Goldilocks>(c, "goldilocks");
    bench_padded_horizontally_packed_row_impl::<BabyBear>(c, "babybear");
}

fn bench_single_call_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_call_overhead");
    group.sample_size(100);

    bench_single_call_for_field::<Goldilocks>(&mut group, "goldilocks");
    bench_single_call_for_field::<BabyBear>(&mut group, "babybear");

    group.finish();
}

criterion_group!(
    benches,
    bench_vertically_packed_row,
    bench_vertically_packed_row_pair,
    bench_horizontally_packed_row,
    bench_padded_horizontally_packed_row,
    bench_single_call_overhead,
);

criterion_main!(benches);
