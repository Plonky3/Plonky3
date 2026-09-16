use std::hint::black_box;
use std::time::Duration;

use criterion::{Criterion, criterion_group, criterion_main};
use p3_circle::{CircleDomain, CircleEvaluations, Point};
use p3_field::PrimeCharacteristicRing;
use p3_matrix::dense::RowMajorMatrix;
use p3_mersenne_31::{Mersenne31, QM31};

type F = Mersenne31;
type EF = QM31;

fn bench_lagrange(c: &mut Criterion) {
    let log_n = 18;
    let domain = CircleDomain::<F>::standard(log_n);
    let evals = CircleEvaluations::from_natural_order(
        domain,
        RowMajorMatrix::new(
            (0..1 << log_n).map(|i| F::from_u64(i as u64 + 1)).collect(),
            1,
        ),
    );
    let zeta = Point::<EF>::from_projective_line(EF::from_u8(9));

    c.bench_function("circle/lagrange/evaluate_at_point/log_n=18", |b| {
        b.iter(|| black_box(evals.evaluate_at_point(black_box(zeta))));
    });
}

criterion_group! {
    name = lagrange;
    config = Criterion::default()
        .sample_size(10)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(2));
    targets = bench_lagrange
}
criterion_main!(lagrange);
