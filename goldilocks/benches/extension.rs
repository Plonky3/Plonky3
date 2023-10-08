use criterion::{criterion_group, criterion_main, Criterion};
use p3_field::extension::binomial_extension::BinomialExtensionField;
use p3_field::extension::quadratic::QuadraticBef;
use p3_field_testing::bench_func::{benchmark_inv, benchmark_mul, benchmark_square};
use p3_goldilocks::Goldilocks;

type EF2 = BinomialExtensionField<Goldilocks, 2>;
type SEF2 = QuadraticBef<Goldilocks>;

fn bench_qudratic_extension(c: &mut Criterion) {
    let name = "BinomialExtensionField<Goldilocks, 2>";
    benchmark_square::<EF2>(c, name);
    benchmark_inv::<EF2>(c, name);
    benchmark_mul::<EF2>(c, name);
}

fn bench_qudratic_extension_speicalized(c: &mut Criterion) {
    let name = "QuadraticBef<Goldilocks>";
    benchmark_square::<SEF2>(c, name);
    benchmark_inv::<SEF2>(c, name);
    benchmark_mul::<SEF2>(c, name);
}

criterion_group!(
    bench_goldilocks_ef2,
    bench_qudratic_extension,
    bench_qudratic_extension_speicalized
);
criterion_main!(bench_goldilocks_ef2);
