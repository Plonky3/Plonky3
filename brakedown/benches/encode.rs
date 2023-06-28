use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use p3_brakedown::{fast_height_14, BrakedownCode};
use p3_code::{CodeOrFamily, IdentityCode};
use p3_field::Field;
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::sparse::CsrMatrix;
use p3_mersenne_31::Mersenne31;
use rand::distributions::{Distribution, Standard};
use rand::thread_rng;
use std::any::type_name;

const BATCH_SIZE: usize = 1 << 12;
const A_ROW_WEIGHT: usize = 10;
const B_ROW_WEIGHT: usize = 20;

fn bench_encode(c: &mut Criterion) {
    encode::<Mersenne31, 20>(c);
}

fn encode<F: Field, const ROW_WEIGHT: usize>(c: &mut Criterion)
where
    Standard: Distribution<F>,
{
    let mut group = c.benchmark_group(&format!("encode::<{}>", type_name::<F>()));
    group.sample_size(10);

    let mut rng = thread_rng();
    for n_log in [14] {
        let n = 1 << n_log;

        let code = fast_height_14();

        let mut messages = RowMajorMatrix::rand(&mut rng, n, BATCH_SIZE);

        group.bench_with_input(BenchmarkId::from_parameter(n), &code, |b, code| {
            b.iter(|| {
                messages.values.truncate(n * BATCH_SIZE);
                code.encode_batch(messages.clone());
            });
        });
    }
}

criterion_group!(benches, bench_encode);
criterion_main!(benches);
