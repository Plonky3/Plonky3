use criterion::{criterion_group, criterion_main, Criterion};
use p3_field::extension::binomial_extension::BinomialExtensionField;
use p3_field_testing::bench_func::{benchmark_inv, benchmark_mul, benchmark_square};
use p3_goldilocks::Goldilocks;

type EF2 = BinomialExtensionField<Goldilocks, 2>;

fn bench_qudratic_extension(c: &mut Criterion) {
    let name = "BinomialExtensionField<Goldilocks, 2>";
    benchmark_square::<EF2>(c, name);
    benchmark_inv::<EF2>(c, name);
    benchmark_mul::<EF2>(c, name);
}

criterion_group!(bench_goldilocks_ef2, bench_qudratic_extension);
criterion_main!(bench_goldilocks_ef2);
