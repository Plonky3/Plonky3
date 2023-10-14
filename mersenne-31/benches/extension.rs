use criterion::{criterion_group, criterion_main, Criterion};
use p3_field::extension::binomial_extension::BinomialExtensionField;
use p3_field_testing::bench_func::{benchmark_inv, benchmark_mul, benchmark_square};
use p3_mersenne_31::{Mersenne31, Mersenne31Complex};

type EF2 = BinomialExtensionField<Mersenne31Complex<Mersenne31>, 2>;
type EF3 = BinomialExtensionField<Mersenne31Complex<Mersenne31>, 3>;

fn bench_qudratic_extension(c: &mut Criterion) {
    let name = "BinomialExtensionField<Mersenne31Complex<Mersenne31>, 2>";
    benchmark_square::<EF2>(c, name);
    benchmark_inv::<EF2>(c, name);
    benchmark_mul::<EF2>(c, name);
}

fn bench_cubic_extension(c: &mut Criterion) {
    let name = "BinomialExtensionField<Mersenne31Complex<Mersenne31>, 3>";
    benchmark_square::<EF3>(c, name);
    benchmark_inv::<EF3>(c, name);
    benchmark_mul::<EF3>(c, name);
}

criterion_group!(bench_mersennecomplex_ef2, bench_qudratic_extension);
criterion_group!(bench_mersennecomplex_ef3, bench_cubic_extension);

criterion_main!(bench_mersennecomplex_ef2, bench_mersennecomplex_ef3);
