use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use p3_baby_bear::BabyBear;
use p3_field::extension::BinomialExtensionField;
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use rand::SeedableRng;
use rand_chacha::ChaChaRng;

fn columnwise_dot_product(c: &mut Criterion) {
    let mut rng = ChaChaRng::seed_from_u64(0);

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;
    let log_rows = 16;

    c.benchmark_group("babybear")
        .sample_size(20)
        .bench_function("columnwise_dot_product", |b| {
            b.iter_batched(
                || {
                    (
                        RowMajorMatrix::<F>::rand_nonzero(&mut rng, 1 << log_rows, 1 << 10),
                        RowMajorMatrix::<EF>::rand_nonzero(&mut rng, 1 << log_rows, 1).values,
                    )
                },
                |(m, v)| m.columnwise_dot_product(&v),
                BatchSize::PerIteration,
            );
        });
}

criterion_group!(benches, columnwise_dot_product);
criterion_main!(benches);
