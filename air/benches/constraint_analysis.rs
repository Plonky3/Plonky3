//! Compare capture and extraction, including destruction of every captured value.

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use p3_air::{
    Air, AirLayout, BaseAir, ConstraintCounts, ConstraintLayout, ConstraintLayoutBuilder,
    PermutationAirBuilder, SymbolicAirBuilder, SymbolicConstraints, WindowAccess,
    capture_constraint_layout,
};
use p3_baby_bear::BabyBear as F;
use p3_field::extension::BinomialExtensionField;
use p3_field::{BasedVectorSpace, PrimeCharacteristicRing};

type EF = BinomialExtensionField<F, 4>;

/// One evaluator and layout are shared by all benchmark paths.
struct Workload {
    constraints: usize,
    rounds: usize,
}

impl BaseAir<F> for Workload {
    fn width(&self) -> usize {
        16
    }
}

impl<AB: PermutationAirBuilder<F = F, EF = EF>> Air<AB> for Workload {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let permutation = builder.permutation();
        let coefficient = EF::from_basis_coefficients_fn(|i| F::from_usize(i + 1));
        for i in 0..self.constraints {
            let x: AB::Expr = main.current(i % self.width()).unwrap().into();
            let next: AB::Expr = main.next((i + 1) % self.width()).unwrap().into();
            let mut base = x.clone() + F::from_usize(i + 1);
            let mut extension: AB::ExprEF = permutation.current(i % 8).unwrap().into();
            extension += coefficient;
            for _ in 0..self.rounds {
                base = base.clone() * base + x.clone();
                extension = extension.clone() * extension + base.clone();
            }
            builder.assert_zero(base - next);
            builder.assert_zero_ext(extension);
        }
    }
}

fn layout() -> AirLayout {
    AirLayout {
        main_width: 16,
        permutation_width: 8,
        ..Default::default()
    }
}

fn builder(air: &Workload) -> SymbolicAirBuilder<F, EF> {
    let mut builder = SymbolicAirBuilder::new(layout());
    air.eval(&mut builder);
    builder
}

fn clone_extraction(builder: &SymbolicAirBuilder<F, EF>) -> SymbolicConstraints<F, EF> {
    SymbolicConstraints {
        base_constraints: builder.base_constraints(),
        extension_constraints: builder.extension_constraints(),
        constraint_layout: builder.constraint_layout(),
    }
}

fn assert_layout_eq(a: &ConstraintLayout, b: &ConstraintLayout) {
    assert_eq!(a.base_indices, b.base_indices);
    assert_eq!(a.ext_indices, b.ext_indices);
}

fn verify(air: &Workload) {
    let owned = builder(air).finish();
    let cloned = clone_extraction(&builder(air));
    let cheap = capture_constraint_layout::<F, EF>(layout(), |b| air.eval(b));
    assert_layout_eq(&owned.constraint_layout, &cheap);
    assert_layout_eq(&owned.constraint_layout, &cloned.constraint_layout);
    assert_eq!(
        owned.constraint_counts(),
        ConstraintCounts {
            base: air.constraints,
            extension: air.constraints,
        }
    );
    assert_eq!(owned.constraint_counts(), cloned.constraint_counts());
    let inputs = ConstraintLayoutBuilder::<F, EF>::new(layout());
    for (a, b) in owned.base_constraints.iter().zip(&cloned.base_constraints) {
        assert_eq!(a.resolve(&inputs), b.resolve(&inputs));
        assert_eq!(a.degree_multiple(), b.degree_multiple());
    }
    for (a, b) in owned
        .extension_constraints
        .iter()
        .zip(&cloned.extension_constraints)
    {
        assert_eq!(
            serde_json::to_value(a).unwrap(),
            serde_json::to_value(b).unwrap()
        );
        assert_eq!(a.degree_multiple(), b.degree_multiple());
    }
}

fn bench_constraint_analysis(c: &mut Criterion) {
    for (name, rounds) in [("shallow", 0), ("arithmetic", 6)] {
        for size in [16, 512] {
            let air = Workload {
                constraints: size,
                rounds,
            };
            verify(&air);
            let mut group = c.benchmark_group(format!("constraint_analysis/{name}"));
            group.bench_function(BenchmarkId::new("symbolic_layout", size), |b| {
                b.iter(|| {
                    let builder = builder(black_box(&air));
                    let result = builder.constraint_layout();
                    black_box(&result);
                    drop(result);
                    drop(builder);
                });
            });
            group.bench_function(BenchmarkId::new("lightweight_layout", size), |b| {
                b.iter(|| {
                    let result = capture_constraint_layout::<F, EF>(layout(), |builder| {
                        black_box(&air).eval(builder);
                    });
                    black_box(&result);
                    drop(result);
                });
            });
            group.bench_function(BenchmarkId::new("clone_extraction", size), |b| {
                b.iter(|| {
                    let builder = builder(black_box(&air));
                    let result = clone_extraction(&builder);
                    black_box(&result);
                    drop(result);
                    drop(builder);
                });
            });
            group.bench_function(BenchmarkId::new("owned_finish", size), |b| {
                b.iter(|| {
                    let result = builder(black_box(&air)).finish();
                    black_box(&result);
                    drop(result);
                });
            });
            group.finish();
        }
    }
}

criterion_group!(benches, bench_constraint_analysis);
criterion_main!(benches);
