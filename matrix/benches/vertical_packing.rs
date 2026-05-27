use core::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use p3_baby_bear::BabyBear;
use p3_field::Field;
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use rand::SeedableRng;
use rand::rngs::SmallRng;

type F = BabyBear;
type Packed = <F as Field>::Packing;

const CONFIGS: &[(usize, usize)] = &[(10, 32), (14, 32), (18, 32), (14, 256)];

fn vertically_packed_row_scalar(c: &mut Criterion) {
    let mut group = c.benchmark_group("vertically_packed_row_scalar");
    group.sample_size(20);

    for &(log_rows, width) in CONFIGS {
        let rows = 1usize << log_rows;
        let mut rng = SmallRng::seed_from_u64(0);
        let matrix = RowMajorMatrix::<F>::rand_nonzero(&mut rng, rows, width);
        let start_row = rows / 2;
        let param = format!("2^{log_rows}x{width}");

        group.bench_with_input(BenchmarkId::new("row", &param), &(), |b, _| {
            b.iter(|| {
                black_box(
                    matrix
                        .vertically_packed_row::<F>(start_row)
                        .collect::<Vec<_>>(),
                )
            });
        });
    }
}

fn vertically_packed_row_pair_scalar(c: &mut Criterion) {
    let mut group = c.benchmark_group("vertically_packed_row_pair_scalar");
    group.sample_size(20);

    for &(log_rows, width) in CONFIGS {
        let rows = 1usize << log_rows;
        let mut rng = SmallRng::seed_from_u64(1);
        let matrix = RowMajorMatrix::<F>::rand_nonzero(&mut rng, rows, width);
        let start_row = rows / 2;
        let step = 8;
        let param = format!("2^{log_rows}x{width}");

        group.bench_with_input(BenchmarkId::new("row_pair", &param), &(), |b, _| {
            b.iter(|| black_box(matrix.vertically_packed_row_pair::<F>(start_row, step)));
        });
    }
}

fn vertically_packed_row_packed(c: &mut Criterion) {
    let mut group = c.benchmark_group("vertically_packed_row_packed");
    group.sample_size(20);

    for &(log_rows, width) in CONFIGS {
        let rows = 1usize << log_rows;
        let mut rng = SmallRng::seed_from_u64(2);
        let matrix = RowMajorMatrix::<F>::rand_nonzero(&mut rng, rows, width);
        let start_row = rows / 2;
        let param = format!("2^{log_rows}x{width}");

        group.bench_with_input(BenchmarkId::new("row", &param), &(), |b, _| {
            b.iter(|| {
                black_box(
                    matrix
                        .vertically_packed_row::<Packed>(start_row)
                        .collect::<Vec<_>>(),
                )
            });
        });
    }
}

fn vertically_packed_row_pair_packed(c: &mut Criterion) {
    let mut group = c.benchmark_group("vertically_packed_row_pair_packed");
    group.sample_size(20);

    for &(log_rows, width) in CONFIGS {
        let rows = 1usize << log_rows;
        let mut rng = SmallRng::seed_from_u64(3);
        let matrix = RowMajorMatrix::<F>::rand_nonzero(&mut rng, rows, width);
        let start_row = rows / 2;
        let step = 8;
        let param = format!("2^{log_rows}x{width}");

        group.bench_with_input(BenchmarkId::new("row_pair", &param), &(), |b, _| {
            b.iter(|| black_box(matrix.vertically_packed_row_pair::<Packed>(start_row, step)));
        });
    }
}

criterion_group!(
    benches,
    vertically_packed_row_scalar,
    vertically_packed_row_pair_scalar,
    vertically_packed_row_packed,
    vertically_packed_row_pair_packed,
);
criterion_main!(benches);
