use core::array;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks;
use p3_mersenne_31::Mersenne31;
use p3_rescue::{RpoGoldilocks, RpoMersenne31};
use p3_symmetric::Permutation;

fn bench_rpo_goldilocks_12(c: &mut Criterion) {
    let rpo = RpoGoldilocks::from_standard_constants(4, 128);
    let input: [Goldilocks; 12] = array::from_fn(|_| Goldilocks::ZERO);
    let id = BenchmarkId::new("rpo::<Goldilocks, alpha=7>", 12);
    c.bench_with_input(id, &input, |b, input| b.iter(|| rpo.permute(*input)));
}

fn bench_rpo_mersenne31(c: &mut Criterion) {
    let rpo = RpoMersenne31::from_standard_constants();
    let input: [Mersenne31; 24] = array::from_fn(|_| Mersenne31::ZERO);
    let id = BenchmarkId::new("rpo::<Mersenne31, alpha=5>", 24);
    c.bench_with_input(id, &input, |b, input| b.iter(|| rpo.permute(*input)));
}

criterion_group!(benches, bench_rpo_goldilocks_12, bench_rpo_mersenne31);
criterion_main!(benches);
