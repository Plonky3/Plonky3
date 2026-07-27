use core::array;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use p3_baby_bear::BabyBear;
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks;
use p3_koala_bear::KoalaBear;
use p3_mersenne_31::Mersenne31;
use p3_rescue::{RpoBabyBear, RpoGoldilocks, RpoKoalaBear, RpoMersenne31};
use p3_symmetric::Permutation;

fn bench_rpo_goldilocks_12(c: &mut Criterion) {
    let rpo = RpoGoldilocks::from_standard_constants();
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

fn bench_rpo_babybear(c: &mut Criterion) {
    let rpo = RpoBabyBear::from_standard_constants();
    let input: [BabyBear; 24] = array::from_fn(|_| BabyBear::ZERO);
    let id = BenchmarkId::new("rpo::<BabyBear, alpha=7>", 24);
    c.bench_with_input(id, &input, |b, input| b.iter(|| rpo.permute(*input)));
}

fn bench_rpo_koalabear(c: &mut Criterion) {
    let rpo = RpoKoalaBear::from_standard_constants();
    let input: [KoalaBear; 24] = array::from_fn(|_| KoalaBear::ZERO);
    let id = BenchmarkId::new("rpo::<KoalaBear, alpha=3>", 24);
    c.bench_with_input(id, &input, |b, input| b.iter(|| rpo.permute(*input)));
}

criterion_group!(
    benches,
    bench_rpo_goldilocks_12,
    bench_rpo_mersenne31,
    bench_rpo_babybear,
    bench_rpo_koalabear,
);
criterion_main!(benches);
