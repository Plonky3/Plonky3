use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use p3_poly::test_utils::rand_poly;
use p3_stir::test_utils::{
    test_bb_challenger, test_bb_stir_config, test_gl_challenger, test_gl_stir_config,
    BB_EXT_SEC_LEVEL, GL_EXT_SEC_LEVEL,
};
use p3_stir::{commit, prove, verify, SecurityAssumption};

const SAMPLES: usize = 10;
const LOG_INV_RATE: usize = 1;
const LOG_FOLDING_FACTOR: usize = 4;

macro_rules! impl_bench_field {
    ($func_name:ident, $field_id:expr, $sec_level:expr, $config_fn:ident, $challenger_fn:ident) => {
        fn $func_name(c: &mut Criterion, log_degree: usize) {
            let mut group = c.benchmark_group(concat!("STIR-", $field_id));
            group.sample_size(SAMPLES);

            let num_rounds = log_degree / LOG_FOLDING_FACTOR - 1;
            let config = $config_fn(
                $sec_level,
                SecurityAssumption::CapacityBound,
                log_degree,
                LOG_INV_RATE,
                LOG_FOLDING_FACTOR,
                num_rounds,
            );

            let degree = 1 << log_degree;
            let polynomial = rand_poly(degree - 1);

            group.bench_function(BenchmarkId::new("commit", log_degree), |b| {
                b.iter_batched(
                    || polynomial.clone(),
                    |poly| commit(&config, poly),
                    criterion::BatchSize::SmallInput,
                );
            });

            let challenger = $challenger_fn();

            group.bench_function(BenchmarkId::new("prove", log_degree), |b| {
                b.iter_batched(
                    || {
                        let (witness, commitment) = commit(&config, polynomial.clone());
                        (witness, commitment, challenger.clone())
                    },
                    |(witness, commitment, mut challenger)| {
                        prove(&config, witness, commitment, &mut challenger)
                    },
                    criterion::BatchSize::SmallInput,
                );
            });

            let challenger = $challenger_fn();
            let (witness, commitment) = commit(&config, polynomial.clone());
            let proof = prove(&config, witness, commitment, &mut challenger.clone());

            group.bench_function(BenchmarkId::new("verify", log_degree), |b| {
                b.iter_batched(
                    || (proof.clone(), challenger.clone()),
                    |(proof, mut challenger)| verify(&config, proof, &mut challenger),
                    criterion::BatchSize::SmallInput,
                );
            });
        }
    };
}

impl_bench_field!(
    bench_bb_with_log_degree,
    "BabyBear",
    BB_EXT_SEC_LEVEL,
    test_bb_stir_config,
    test_bb_challenger
);

impl_bench_field!(
    bench_gl_with_log_degree,
    "Goldilocks",
    GL_EXT_SEC_LEVEL,
    test_gl_stir_config,
    test_gl_challenger
);

fn bench(c: &mut Criterion) {
    for log_degree in (14..=22).step_by(2) {
        bench_bb_with_log_degree(c, log_degree);
        bench_gl_with_log_degree(c, log_degree);
    }
}

criterion_group!(benches, bench);
criterion_main!(benches);
