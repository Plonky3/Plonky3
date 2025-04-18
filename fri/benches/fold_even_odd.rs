use core::any::type_name;
use std::marker::PhantomData;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use p3_baby_bear::BabyBear;
use p3_field::TwoAdicField;
use p3_field::extension::{BinomialExtensionField, Complex};
use p3_fri::{FriGenericConfig, TwoAdicFriGenericConfig};
use p3_goldilocks::Goldilocks;
use p3_matrix::dense::RowMajorMatrix;
use p3_mersenne_31::Mersenne31;
use rand::distr::{Distribution, StandardUniform};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

fn bench<F: TwoAdicField>(c: &mut Criterion, log_sizes: &[usize])
where
    StandardUniform: Distribution<F>,
{
    let name = format!("fold_matrix::<{}>", type_name::<F>(),);
    let mut group = c.benchmark_group(&name);
    group.sample_size(10);
    // let simple_config = create_benchmark_fri_config(val_mmcs);
    let config = TwoAdicFriGenericConfig::<(), ()>(PhantomData);

    for log_size in log_sizes {
        let n = 1 << log_size;

        let mut rng = SmallRng::seed_from_u64(n as u64);
        let beta = rng.sample(StandardUniform);
        let mat = RowMajorMatrix::rand(&mut rng, n, 2);

        group.bench_function(BenchmarkId::from_parameter(n), |b| {
            b.iter(|| {
                config.fold_matrix(beta, mat.clone());
            })
        });
    }
}

fn bench_fold_even_odd(c: &mut Criterion) {
    let log_sizes = [12, 14, 16, 18, 20, 22];

    bench::<BabyBear>(c, &log_sizes);
    bench::<BinomialExtensionField<BabyBear, 5>>(c, &log_sizes);
    bench::<Goldilocks>(c, &log_sizes);
    bench::<Complex<Mersenne31>>(c, &log_sizes);
}

criterion_group!(benches, bench_fold_even_odd);
criterion_main!(benches);
