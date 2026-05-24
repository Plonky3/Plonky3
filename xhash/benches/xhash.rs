use core::array;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use p3_baby_bear::BabyBear;
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks;
use p3_koala_bear::KoalaBear;
use p3_mersenne_31::Mersenne31;
use p3_symmetric::Permutation;
use p3_xhash::{XHashBabyBear, XHashGoldilocks, XHashKoalaBear, XHashMersenne31};

fn bench_xhash_goldilocks(c: &mut Criterion) {
    let xhash = XHashGoldilocks::from_standard_constants(4, 128);
    let input: [Goldilocks; 12] = array::from_fn(|_| Goldilocks::ZERO);
    let id = BenchmarkId::new("xhash::<Goldilocks, alpha=7>", 12);
    c.bench_with_input(id, &input, |b, input| b.iter(|| xhash.permute(*input)));
}

fn bench_xhash_mersenne31(c: &mut Criterion) {
    let xhash = XHashMersenne31::from_standard_constants();
    let input: [Mersenne31; 24] = array::from_fn(|_| Mersenne31::ZERO);
    let id = BenchmarkId::new("xhash::<Mersenne31, alpha=5>", 24);
    c.bench_with_input(id, &input, |b, input| b.iter(|| xhash.permute(*input)));
}

fn bench_xhash_babybear(c: &mut Criterion) {
    let xhash = XHashBabyBear::from_standard_constants();
    let input: [BabyBear; 24] = array::from_fn(|_| BabyBear::ZERO);
    let id = BenchmarkId::new("xhash::<BabyBear, alpha=7>", 24);
    c.bench_with_input(id, &input, |b, input| b.iter(|| xhash.permute(*input)));
}

fn bench_xhash_koalabear(c: &mut Criterion) {
    let xhash = XHashKoalaBear::from_standard_constants();
    let input: [KoalaBear; 24] = array::from_fn(|_| KoalaBear::ZERO);
    let id = BenchmarkId::new("xhash::<KoalaBear, alpha=3>", 24);
    c.bench_with_input(id, &input, |b, input| b.iter(|| xhash.permute(*input)));
}

criterion_group!(
    benches,
    bench_xhash_goldilocks,
    bench_xhash_mersenne31,
    bench_xhash_babybear,
    bench_xhash_koalabear,
);
criterion_main!(benches);
