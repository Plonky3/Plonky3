use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use p3_baby_bear::{BabyBear, default_babybear_poseidon1_16, default_babybear_poseidon1_24};
use p3_field::PrimeCharacteristicRing;
use p3_symmetric::Permutation;

fn bench_poseidon(c: &mut Criterion) {
    poseidon1_babybear_16(c);
    poseidon1_babybear_24(c);
}

fn poseidon1_babybear_16(c: &mut Criterion) {
    let poseidon = default_babybear_poseidon1_16();
    let input = [BabyBear::ZERO; 16];
    let id = BenchmarkId::new("poseidon1::<BabyBear, 7>", 16);
    c.bench_with_input(id, &input, |b, input| {
        b.iter(|| poseidon.permute(*input));
    });
}

fn poseidon1_babybear_24(c: &mut Criterion) {
    let poseidon = default_babybear_poseidon1_24();
    let input = [BabyBear::ZERO; 24];
    let id = BenchmarkId::new("poseidon1::<BabyBear, 7>", 24);
    c.bench_with_input(id, &input, |b, input| {
        b.iter(|| poseidon.permute(*input));
    });
}

criterion_group!(benches, bench_poseidon);
criterion_main!(benches);
