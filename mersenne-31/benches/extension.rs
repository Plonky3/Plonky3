use criterion::{criterion_group, criterion_main, Criterion};
use p3_field::extension::binomial_extension::BinomialExtensionField;
use p3_field::extension::cubic::CubicBef;
use p3_field::extension::quadratic::QuadraticBef;
use p3_field_testing::bench_func::{benchmark_inv, benchmark_mul, benchmark_square};
use p3_mersenne_31::{Mersenne31, Mersenne31Complex};

type EF2 = BinomialExtensionField<Mersenne31Complex<Mersenne31>, 2>;
type EF3 = BinomialExtensionField<Mersenne31Complex<Mersenne31>, 3>;
type SEF2 = QuadraticBef<Mersenne31Complex<Mersenne31>>;
type SEF3 = CubicBef<Mersenne31Complex<Mersenne31>>;

fn bench_qudratic_extension(c: &mut Criterion) {
    let name = "BinomialExtensionField<Mersenne31Complex<Mersenne31>, 2>";
    benchmark_square::<EF2>(c, name);
    benchmark_inv::<EF2>(c, name);
    benchmark_mul::<EF2>(c, name);
}

fn bench_qudratic_extension_speicalized(c: &mut Criterion) {
    let name = "QuadraticBef<Mersenne31Complex<Mersenne31>>";
    benchmark_square::<SEF2>(c, name);
    benchmark_inv::<SEF2>(c, name);
    benchmark_mul::<SEF2>(c, name);
}
fn bench_cubic_extension(c: &mut Criterion) {
    let name = "BinomialExtensionField<Mersenne31Complex<Mersenne31>, 3>";
    benchmark_square::<EF3>(c, name);
    benchmark_inv::<EF3>(c, name);
    benchmark_mul::<EF3>(c, name);
}

fn bench_cubic_extension_speicalized(c: &mut Criterion) {
    let name = "CubicBef<Mersenne31Complex<Mersenne31>>";
    benchmark_square::<SEF3>(c, name);
    benchmark_inv::<SEF3>(c, name);
    benchmark_mul::<SEF3>(c, name);
}

criterion_group!(
    bench_goldilocks_ef2,
    bench_qudratic_extension,
    bench_qudratic_extension_speicalized
);

criterion_group!(
    bench_goldilocks_ef3,
    bench_cubic_extension,
    bench_cubic_extension_speicalized
);
criterion_main!(bench_goldilocks_ef2, bench_goldilocks_ef3);
