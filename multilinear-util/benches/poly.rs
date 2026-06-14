//! Benchmarks for [`Poly`] operations.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use p3_baby_bear::BabyBear;
use p3_field::extension::BinomialExtensionField;
use p3_multilinear_util::poly::Poly;
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

criterion_group!(benches, bench_new_packed_from_point);
criterion_main!(benches);
