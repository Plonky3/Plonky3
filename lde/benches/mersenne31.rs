use criterion::{black_box, criterion_group, criterion_main, Criterion};
use p3_matrix::dense::RowMajorMatrix;
use p3_mersenne_31::{Mersenne31, Mersenne31ComplexLDE};
use rand::distributions::Standard;
use rand::{thread_rng, Rng};

type Base = Mersenne31;

fn bench_lde(c: &mut Criterion) {
    for (n, m) in [(12, 1), (12, 2), (12, 4), (16, 4), (20, 4)] {
        let input = RowMajorMatrix::new_col(
            thread_rng()
                .sample_iter(Standard)
                .take(1 << n)
                .collect::<Vec<Base>>(),
        );
        let compressed = Mersenne31ComplexLDE::lde_batch_compress(input.clone(), m);
        c.bench_function(&format!("lde_compression_{n}_{m}"), |b| {
            b.iter(|| {
                black_box(Mersenne31ComplexLDE::lde_batch_compress(
                    black_box(input.clone()),
                    m,
                ))
            });
        });

        c.bench_function(&format!("lde_decompression_{n}_{m}"), |b| {
            b.iter(|| black_box(compressed.decompress()));
        });
    }
}

criterion_group!(mersenne31_lde, bench_lde);
criterion_main!(mersenne31_lde);
