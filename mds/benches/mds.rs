use std::any::type_name;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use p3_baby_bear::{BabyBear, MdsMatrixBabyBear};
use p3_field::{AbstractField, Field};
use p3_goldilocks::{Goldilocks, MdsMatrixGoldilocks};
use p3_mds::coset_mds::CosetMds;
use p3_mds::integrated_coset_mds::IntegratedCosetMds;
use p3_mds::MdsPermutation;
use p3_mersenne_31::{MdsMatrixMersenne31, Mersenne31};
use rand::distributions::{Distribution, Standard};
use rand::{thread_rng, Rng};

fn bench_all_mds(c: &mut Criterion) {
    bench_mds::<BabyBear, IntegratedCosetMds<BabyBear, 16>, 16>(c);
    bench_mds::<<BabyBear as Field>::Packing, IntegratedCosetMds<BabyBear, 16>, 16>(c);
    bench_mds::<BabyBear, CosetMds<BabyBear, 16>, 16>(c);
    bench_mds::<<BabyBear as Field>::Packing, CosetMds<BabyBear, 16>, 16>(c);

    bench_mds::<BabyBear, MdsMatrixBabyBear, 8>(c);
    bench_mds::<BabyBear, MdsMatrixBabyBear, 12>(c);
    bench_mds::<BabyBear, MdsMatrixBabyBear, 16>(c);
    bench_mds::<BabyBear, MdsMatrixBabyBear, 24>(c);
    bench_mds::<BabyBear, MdsMatrixBabyBear, 32>(c);
    bench_mds::<BabyBear, MdsMatrixBabyBear, 64>(c);

    bench_mds::<Goldilocks, MdsMatrixGoldilocks, 8>(c);
    bench_mds::<Goldilocks, MdsMatrixGoldilocks, 12>(c);
    bench_mds::<Goldilocks, MdsMatrixGoldilocks, 16>(c);
    bench_mds::<Goldilocks, MdsMatrixGoldilocks, 32>(c);
    bench_mds::<Goldilocks, MdsMatrixGoldilocks, 64>(c);

    bench_mds::<Mersenne31, MdsMatrixMersenne31, 8>(c);
    bench_mds::<Mersenne31, MdsMatrixMersenne31, 12>(c);
    bench_mds::<Mersenne31, MdsMatrixMersenne31, 16>(c);
    bench_mds::<Mersenne31, MdsMatrixMersenne31, 32>(c);
    bench_mds::<Mersenne31, MdsMatrixMersenne31, 64>(c);
}

fn bench_mds<AF, Mds, const WIDTH: usize>(c: &mut Criterion)
where
    AF: AbstractField,
    Standard: Distribution<AF>,
    Mds: MdsPermutation<AF, WIDTH> + Default,
{
    let mds = Mds::default();

    let mut rng = thread_rng();
    let input = rng.gen::<[AF; WIDTH]>();
    let id = BenchmarkId::new(type_name::<Mds>(), WIDTH);
    c.bench_with_input(id, &input, |b, input| b.iter(|| mds.permute(input.clone())));
}

criterion_group!(benches, bench_all_mds);
criterion_main!(benches);
