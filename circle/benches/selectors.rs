use std::hint::black_box;
use std::time::Duration;

use criterion::{Criterion, criterion_group, criterion_main};
use p3_circle::CircleDomain;
use p3_commit::PolynomialSpace;
use p3_mersenne_31::Mersenne31;

fn bench_selectors(c: &mut Criterion) {
    let domain = CircleDomain::<Mersenne31>::standard(18);
    let coset = domain.create_disjoint_domain(domain.size());

    c.bench_function("circle/selectors/log_n=18", |b| {
        b.iter(|| black_box(domain).selectors_on_coset(black_box(coset)));
    });
}

criterion_group! {
    name = selectors;
    config = Criterion::default()
        .sample_size(10)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(2));
    targets = bench_selectors
}
criterion_main!(selectors);
