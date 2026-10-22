//! Compare capture and extraction, including destruction of every captured value.

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use p3_air::{
    Air, AirBuilder, AirLayout, BaseAir, CapturedArithmeticConstraints, ConstraintCounts,
    ConstraintLayout, ConstraintLayoutBuilder, ExtensionBuilder, PermutationAirBuilder,
    SymbolicAirBuilder, SymbolicExpression, SymbolicExpressionExt, WindowAccess,
    capture_arithmetic, capture_constraint_layout,
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

type ConstraintParts = (
    Vec<SymbolicExpression<F>>,
    Vec<SymbolicExpressionExt<F, EF>>,
    ConstraintLayout,
);

fn clone_extraction(builder: &SymbolicAirBuilder<F, EF>) -> ConstraintParts {
    (
        builder.base_constraints(),
        builder.extension_constraints(),
        builder.constraint_layout(),
    )
}

fn assert_layout_eq(a: &ConstraintLayout, b: &ConstraintLayout) {
    assert_eq!(a.base_indices, b.base_indices);
    assert_eq!(a.ext_indices, b.ext_indices);
}

fn verify(air: &Workload) {
    let owned = builder(air).finish();
    let (cloned_base, cloned_extension, cloned_layout) = clone_extraction(&builder(air));
    let cheap = capture_constraint_layout::<F, EF>(layout(), |b| air.eval(b));
    assert_layout_eq(owned.constraint_layout(), &cheap);
    assert_layout_eq(owned.constraint_layout(), &cloned_layout);
    assert_eq!(
        owned.constraint_counts(),
        ConstraintCounts {
            base: air.constraints,
            extension: air.constraints,
        }
    );
    assert_eq!(
        owned.constraint_counts(),
        ConstraintCounts {
            base: cloned_base.len(),
            extension: cloned_extension.len(),
        }
    );
    let inputs = ConstraintLayoutBuilder::<F, EF>::new(layout());
    for (a, b) in owned.base_constraints().iter().zip(&cloned_base) {
        assert_eq!(a.resolve(&inputs), b.resolve(&inputs));
        assert_eq!(a.degree_multiple(), b.degree_multiple());
    }
    for (a, b) in owned.extension_constraints().iter().zip(&cloned_extension) {
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

// Capture once so these cases measure lowering and result destruction only.
fn arithmetic_capture(air: &Workload, highly_shared: bool) -> CapturedArithmeticConstraints<F, EF> {
    capture_arithmetic(layout(), |builder| {
        if !highly_shared {
            air.eval(builder);
            return;
        }
        let x = builder.main().values[0];
        let mut base = p3_air::arithmetic::ArithmeticExpression::from(x);
        let mut extension =
            p3_air::arithmetic::ArithmeticExpressionExt::from(builder.permutation().values[0]);
        for _ in 0..air.rounds {
            base = base.clone() * base + x;
            extension = extension.clone() * extension + base.clone();
        }
        for i in 0..air.constraints {
            let offset = F::from_usize(i + 1);
            builder.assert_zero(base.clone() + offset);
            builder.assert_zero_ext(extension.clone() + offset);
        }
    })
}

fn lower_per_root(captured: &CapturedArithmeticConstraints<F, EF>) -> ConstraintParts {
    (
        captured.base.iter().map(|root| root.lower()).collect(),
        captured.extension.iter().map(|root| root.lower()).collect(),
        ConstraintLayout {
            base_indices: captured.layout.base_indices.clone(),
            ext_indices: captured.layout.ext_indices.clone(),
        },
    )
}

fn bench_arithmetic_lowering(c: &mut Criterion) {
    for (name, highly_shared) in [("highly_shared", true), ("mostly_unshared", false)] {
        let mut group = c.benchmark_group(format!("arithmetic_lowering/{name}"));
        for size in [16, 512] {
            let captured = arithmetic_capture(
                &Workload {
                    constraints: size,
                    rounds: 6,
                },
                highly_shared,
            );
            let whole = captured.lower().unwrap();
            let (base, extension, layout) = lower_per_root(&captured);
            assert_layout_eq(whole.constraint_layout(), &layout);
            assert_eq!(whole.base_constraints().len(), base.len());
            assert_eq!(whole.extension_constraints().len(), extension.len());
            let inputs = ConstraintLayoutBuilder::<F, EF>::new(self::layout());
            for (a, b) in whole.base_constraints().iter().zip(&base) {
                assert_eq!(a.resolve(&inputs), b.resolve(&inputs));
                assert_eq!(a.degree_multiple(), b.degree_multiple());
            }
            for (a, b) in whole.extension_constraints().iter().zip(&extension) {
                assert_eq!(a.resolve(&inputs), b.resolve(&inputs));
                assert_eq!(a.degree_multiple(), b.degree_multiple());
            }
            drop((whole, base, extension, layout));
            group.bench_function(BenchmarkId::new("per_root", size), |b| {
                b.iter(|| black_box(lower_per_root(black_box(&captured))));
            });
            group.bench_function(BenchmarkId::new("whole_capture", size), |b| {
                b.iter(|| black_box(black_box(&captured).lower().unwrap()));
            });
        }
        group.finish();
    }
}

criterion_group!(
    benches,
    bench_constraint_analysis,
    bench_arithmetic_lowering
);
criterion_main!(benches);
