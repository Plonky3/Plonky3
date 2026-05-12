use core::array;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks;
use p3_rescue::RpxGoldilocks;
use p3_symmetric::Permutation;

fn bench_rpx_goldilocks_12(c: &mut Criterion) {
    let rpx = RpxGoldilocks::from_standard_constants(4, 128);
    let input: [Goldilocks; 12] = array::from_fn(|_| Goldilocks::ZERO);
    let id = BenchmarkId::new("rpx::<Goldilocks, alpha=7>", 12);
    c.bench_with_input(id, &input, |b, input| b.iter(|| rpx.permute(*input)));
}

criterion_group!(benches, bench_rpx_goldilocks_12);
criterion_main!(benches);
