use core::marker::PhantomData;
use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use p3_baby_bear::BabyBear;
use p3_field::extension::{BinomialExtensionField, Complex};
use p3_field::{ExtensionField, TwoAdicField};
use p3_fri::{FriFoldingStrategy, TwoAdicFriFolding};
use p3_goldilocks::Goldilocks;
use p3_matrix::dense::RowMajorMatrix;
use p3_mersenne_31::Mersenne31;
use p3_util::pretty_name;
use rand::distr::{Distribution, StandardUniform};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

const LOG_ARITIES: [usize; 3] = [1, 2, 3];

fn bench<F: TwoAdicField, EF: ExtensionField<F>>(c: &mut Criterion, log_sizes: &[usize])
where
    StandardUniform: Distribution<EF>,
{
    let name = format!("fold_matrix::<{}>", pretty_name::<EF>(),);
    let mut group = c.benchmark_group(&name);
    group.sample_size(10);
    let folding = TwoAdicFriFolding::<(), ()>(PhantomData);

    for log_arity in LOG_ARITIES {
        let arity = 1 << log_arity;

        for log_size in log_sizes {
            let n = 1 << log_size;

            let mut rng = SmallRng::seed_from_u64(n as u64);
            let beta = rng.sample(StandardUniform);
            // The input always holds `2 * n` evaluations, laid out as `arity` per folded output.
            let mat = RowMajorMatrix::<EF>::rand(&mut rng, (2 * n) / arity, arity);

            group.bench_function(BenchmarkId::new(format!("log_arity_{log_arity}"), n), |b| {
                b.iter(|| black_box(folding.fold_matrix(beta, log_arity, mat.as_view())));
            });
        }
    }
}

fn bench_fold_even_odd(c: &mut Criterion) {
    let log_sizes = [12, 14, 16, 18, 20, 22];

    bench::<BabyBear, BabyBear>(c, &log_sizes);
    bench::<BabyBear, BinomialExtensionField<BabyBear, 5>>(c, &log_sizes);
    bench::<BinomialExtensionField<BabyBear, 5>, BinomialExtensionField<BabyBear, 5>>(
        c, &log_sizes,
    );
    bench::<Goldilocks, Goldilocks>(c, &log_sizes);
    bench::<Complex<Mersenne31>, Complex<Mersenne31>>(c, &log_sizes);
}

criterion_group!(benches, bench_fold_even_odd);
criterion_main!(benches);
