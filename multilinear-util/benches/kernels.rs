//! Micro benches for every hot multilinear kernel, over a prime and a binary field pair.
//!
//! Each kernel runs at the shapes the provers use:
//! - equality tables, scalar and packed,
//! - evaluation of base and extension polynomials,
//! - the sumcheck folds, prefix and suffix,
//! - the factored-eq contractions.

use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use p3_baby_bear::BabyBear;
use p3_binary_field::{Poly64, Poly192};
use p3_field::extension::BinomialExtensionField;
use p3_field::{ExtensionField, Field};
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;
use p3_multilinear_util::split_eq::SplitEq;
use rand::SeedableRng;
use rand::distr::{Distribution, StandardUniform};
use rand::rngs::SmallRng;

/// Degree-4 extension of the 31-bit prime field.
type Bb4 = BinomialExtensionField<BabyBear, 4>;

/// Variable counts for the table builders and evaluators.
const SIZES: [usize; 4] = [10, 14, 18, 22];

/// Variable counts for the folds, which run once per sumcheck round.
const FOLD_SIZES: [usize; 3] = [12, 16, 20];

/// Registers every kernel for one field pair under the given label.
fn bench_field_pair<F, EF>(c: &mut Criterion, label: &str)
where
    F: Field,
    EF: ExtensionField<F>,
    StandardUniform: Distribution<F> + Distribution<EF>,
{
    let mut rng = SmallRng::seed_from_u64(0x5eed);

    // Equality tables.
    {
        let mut group = c.benchmark_group(format!("{label}/eq_table"));
        for n in SIZES {
            let point = Point::<EF>::rand(&mut rng, n);
            group.bench_function(BenchmarkId::new("new_from_point", n), |b| {
                b.iter(|| Poly::<EF>::new_from_point(point.as_slice(), EF::ONE));
            });
            group.bench_function(BenchmarkId::new("new_from_point_over", n), |b| {
                b.iter(|| Poly::<EF>::new_from_point_over::<F>(point.as_slice(), EF::ONE));
            });
            group.bench_function(BenchmarkId::new("new_packed_from_point", n), |b| {
                b.iter(|| {
                    Poly::<EF::ExtensionPacking>::new_packed_from_point::<F, EF>(
                        point.as_slice(),
                        EF::ONE,
                    )
                });
            });
        }
        group.finish();
    }

    // Evaluation at a random extension point.
    {
        let mut group = c.benchmark_group(format!("{label}/eval"));
        for n in SIZES {
            let point = Point::<EF>::rand(&mut rng, n);
            let base = Poly::<F>::rand(&mut rng, n);
            let ext = Poly::<EF>::rand(&mut rng, n);
            group.bench_function(BenchmarkId::new("eval_base", n), |b| {
                b.iter(|| base.eval_base(&point));
            });
            group.bench_function(BenchmarkId::new("eval_ext", n), |b| {
                b.iter(|| ext.eval_ext::<F>(&point));
            });
        }
        group.finish();
    }

    // Sumcheck folds.
    {
        let mut group = c.benchmark_group(format!("{label}/fold"));
        for n in FOLD_SIZES {
            let r: EF = rand::RngExt::random(&mut rng);
            let base = Poly::<F>::rand(&mut rng, n);
            let ext = Poly::<EF>::rand(&mut rng, n);
            group.bench_function(BenchmarkId::new("fix_prefix_var_base", n), |b| {
                b.iter(|| base.fix_prefix_var(r));
            });
            group.bench_function(BenchmarkId::new("fix_prefix_var_to_packed", n), |b| {
                b.iter(|| base.fix_prefix_var_to_packed(r));
            });
            group.bench_function(BenchmarkId::new("fix_prefix_var_mut_ext", n), |b| {
                b.iter_batched(
                    || ext.clone(),
                    |mut p| p.fix_prefix_var_mut(r),
                    BatchSize::LargeInput,
                );
            });
            group.bench_function(BenchmarkId::new("fix_suffix_var_mut_ext", n), |b| {
                b.iter_batched(
                    || ext.clone(),
                    |mut p| p.fix_suffix_var_mut(r),
                    BatchSize::LargeInput,
                );
            });
        }
        group.finish();
    }

    // Factored-eq contractions, with the table built outside the loop.
    {
        let mut group = c.benchmark_group(format!("{label}/split_eq"));
        for n in [16usize, 20] {
            let point = Point::<EF>::rand(&mut rng, n);
            let half = Point::<EF>::rand(&mut rng, n / 2);
            let base = Poly::<F>::rand(&mut rng, n);
            let full = SplitEq::<F, EF>::new_packed(&point, EF::ONE);
            let half_eq = SplitEq::<F, EF>::new_packed(&half, EF::ONE);
            group.bench_function(BenchmarkId::new("new_packed", n), |b| {
                b.iter(|| SplitEq::<F, EF>::new_packed(&point, EF::ONE));
            });
            group.bench_function(BenchmarkId::new("materialize", n), |b| {
                b.iter(|| full.materialize());
            });
            group.bench_function(BenchmarkId::new("compress_prefix", n), |b| {
                b.iter(|| half_eq.compress_prefix(base.as_view()));
            });
            group.bench_function(BenchmarkId::new("compress_suffix", n), |b| {
                b.iter(|| half_eq.compress_suffix(base.as_view()));
            });
            group.bench_function(BenchmarkId::new("eval_next_base", n), |b| {
                b.iter(|| full.eval_next_base(base.as_view()));
            });
        }
        group.finish();
    }
}

fn bench_baby_bear(c: &mut Criterion) {
    bench_field_pair::<BabyBear, Bb4>(c, "bb4");
}

fn bench_binary(c: &mut Criterion) {
    bench_field_pair::<Poly64, Poly192>(c, "gf192");
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(20)
        .warm_up_time(core::time::Duration::from_millis(300))
        .measurement_time(core::time::Duration::from_millis(900));
    targets = bench_baby_bear, bench_binary
}
criterion_main!(benches);
