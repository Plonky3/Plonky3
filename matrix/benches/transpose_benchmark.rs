use criterion::{criterion_group, criterion_main, Criterion};
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::MatrixTranspose;

fn transpose_benchmark(c: &mut Criterion) {
    const WIDTH_10: usize = 10;
    const HEIGHT_10: usize = 10;

    let matrix_10x10 = RowMajorMatrix::new(vec![0; WIDTH_10 * HEIGHT_10], WIDTH_10);
    c.bench_function("transpose 10x10", |b| {
        b.iter(|| matrix_10x10.clone().transpose())
    });

    const WIDTH_100: usize = 100;
    const HEIGHT_100: usize = 100;

    let matrix_100x100 = RowMajorMatrix::new(vec![0; WIDTH_100 * HEIGHT_100], WIDTH_100);
    c.bench_function("transpose 100x100", |b| {
        b.iter(|| matrix_100x100.clone().transpose())
    });

    const WIDTH_200: usize = 200;
    const HEIGHT_200: usize = 200;

    let matrix_200x200 = RowMajorMatrix::new(vec![0; WIDTH_200 * HEIGHT_200], WIDTH_200);
    c.bench_function("transpose 200x200", |b| {
        b.iter(|| matrix_200x200.clone().transpose())
    });

    const WIDTH_500: usize = 500;
    const HEIGHT_500: usize = 500;

    let matrix_500x500 = RowMajorMatrix::new(vec![0; WIDTH_500 * HEIGHT_500], WIDTH_500);
    c.bench_function("transpose 500x500", |b| {
        b.iter(|| matrix_500x500.clone().transpose())
    });

    const WIDTH_1024: usize = 1024;
    const HEIGHT_1024: usize = 1024;

    let matrix_1024x1024 = RowMajorMatrix::new(vec![0; WIDTH_1024 * HEIGHT_1024], WIDTH_1024);
    c.bench_function("transpose 1024x1024", |b| {
        b.iter(|| matrix_1024x1024.clone().transpose())
    });

    const WIDTH_10_000: usize = 10_000;
    const HEIGHT_10_000: usize = 10_000;

    let matrix_10_000x10_000 =
        RowMajorMatrix::new(vec![0; WIDTH_10_000 * HEIGHT_10_000], WIDTH_10_000);
    c.bench_function("transpose 10_000x10_000", |b| {
        b.iter(|| matrix_10_000x10_000.clone().transpose())
    });
}

criterion_group!(benches, transpose_benchmark);
criterion_main!(benches);
