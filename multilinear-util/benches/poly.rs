//! Benchmarks for [`Poly`] operations.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use p3_baby_bear::BabyBear;
use p3_binary_field::Poly192;
use p3_field::Field;
use p3_field::extension::BinomialExtensionField;
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;
use rand::distr::{Distribution, StandardUniform};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

/// Base field type.
type F = BabyBear;

/// Extension field: a degree-4 binomial extension of the base field.
type EF4 = BinomialExtensionField<F, 4>;

/// Packed extension field type used by the packed eq builder.
type Packed = <EF4 as p3_field::ExtensionField<F>>::ExtensionPacking;

/// Benchmarks the packed single-point eq builder over the 2^18..2^24 range.
fn bench_new_packed_from_point(c: &mut Criterion) {
    let mut group = c.benchmark_group("new_packed_from_point");
    let mut rng = SmallRng::seed_from_u64(987654321);

    for num_vars in [18usize, 20, 22, 24] {
        let point: Vec<EF4> = (0..num_vars).map(|_| rng.random()).collect();
        let scale: EF4 = rng.random();

        group.bench_with_input(BenchmarkId::from_parameter(num_vars), &num_vars, |b, _| {
            b.iter(|| Poly::<Packed>::new_packed_from_point::<F, EF4>(&point, scale));
        });
    }

    group.finish();
}

/// Benchmarks the materialized equality tables of a point, in both bit orders.
fn bench_equality_weights_of<EF>(c: &mut Criterion, field: &str)
where
    EF: Field,
    StandardUniform: Distribution<EF>,
{
    let mut group = c.benchmark_group(format!("equality_weights/{field}"));
    let mut rng = SmallRng::seed_from_u64(123456789);

    for num_vars in [10usize, 18, 22] {
        let point = Point::<EF>::rand(&mut rng, num_vars);
        group.bench_with_input(BenchmarkId::new("msb", num_vars), &num_vars, |b, _| {
            b.iter(|| point.equality_weights_msb());
        });
        group.bench_with_input(BenchmarkId::new("lsb", num_vars), &num_vars, |b, _| {
            b.iter(|| point.equality_weights_lsb());
        });
    }

    group.finish();
}

fn bench_equality_weights(c: &mut Criterion) {
    bench_equality_weights_of::<Poly192>(c, "poly192");
    bench_equality_weights_of::<EF4>(c, "babybear4");
}

criterion_group!(benches, bench_new_packed_from_point, bench_equality_weights);
criterion_main!(benches);
