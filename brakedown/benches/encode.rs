use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use hyperbrakedown::BrakedownCode;
use hypercode::{IdentityCode, SystematicCode};
use hyperfield::field::Field;
use hyperfield::matrix::dense::DenseMatrix;
use hyperfield::matrix::sparse::CsrMatrix;
use hyperfield::mersenne31::Mersenne31;
use rand::distributions::{Distribution, Standard};
use rand::thread_rng;
use std::any::type_name;

const BATCH_SIZE: usize = 100;
const A_ROW_WEIGHT: usize = 10;
const B_ROW_WEIGHT: usize = 20;

fn criterion_benchmark(c: &mut Criterion) {
    encode::<Mersenne31, 20>(c);
}

fn encode<F: Field, const ROW_WEIGHT: usize>(c: &mut Criterion)
where
    Standard: Distribution<F>,
{
    let mut group = c.benchmark_group(&format!("encode::<{}>", type_name::<F>()));
    group.sample_size(10);

    let mut rng = thread_rng();
    for n_log in [14, 15, 16, 17] {
        let n = 1 << n_log;

        let a = CsrMatrix::<F>::rand_fixed_row_weight(&mut rng, n / 4, n, A_ROW_WEIGHT);
        let b = CsrMatrix::<F>::rand_fixed_row_weight(&mut rng, 3 * n / 4, n / 4, B_ROW_WEIGHT);
        let code = BrakedownCode {
            a,
            b,
            inner_code: Box::new(IdentityCode { len: n / 2 }),
        };

        let mut messages = DenseMatrix::rand(&mut rng, n, BATCH_SIZE);

        group.bench_with_input(BenchmarkId::from_parameter(n), &code, |b, code| {
            b.iter(|| {
                messages.values.truncate(n * BATCH_SIZE);
                code.append_parity(&mut messages)
            });
        });
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
