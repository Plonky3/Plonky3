use core::array;

use criterion::{Criterion, criterion_group, criterion_main};
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::{Goldilocks, MdsMatrixGoldilocks};
use p3_mds::MdsPermutation;
use p3_mersenne_31::{MdsMatrixMersenne31, Mersenne31};
use p3_monolith::{
    MonolithBarsGoldilocks, MonolithBarsM31, MonolithGoldilocks8, MonolithMdsMatrixGoldilocks,
    MonolithMdsMatrixMersenne31, MonolithMersenne31,
};
use p3_symmetric::Permutation;

fn bench_monolith(c: &mut Criterion) {
    // Mersenne31 with WIDTH=16 (the standard Monolith-31 width).
    monolith_m31::<_, 16>(c, "MdsMatrixMersenne31", MdsMatrixMersenne31);
    monolith_m31::<_, 16>(c, "MonolithMds", MonolithMdsMatrixMersenne31::<6>);

    // Goldilocks with WIDTH=8 and WIDTH=12.
    monolith_gl8::<_, 8>(c, "MdsMatrixGoldilocks", MdsMatrixGoldilocks);
    monolith_gl8::<_, 8>(c, "MonolithMds", MonolithMdsMatrixGoldilocks);
    monolith_gl8::<_, 12>(c, "MdsMatrixGoldilocks", MdsMatrixGoldilocks);
    monolith_gl8::<_, 12>(c, "MonolithMds", MonolithMdsMatrixGoldilocks);
}

fn monolith_m31<Mds, const WIDTH: usize>(c: &mut Criterion, mds_name: &str, mds: Mds)
where
    Mds: MdsPermutation<Mersenne31, WIDTH>,
{
    let bars = MonolithBarsM31;
    let monolith: MonolithMersenne31<_, WIDTH, 5> = MonolithMersenne31::new(bars, mds);

    let mut input = array::from_fn(Mersenne31::from_usize);

    let name = format!("monolith::<Mersenne31, {WIDTH}> ({mds_name})");
    c.bench_function(name.as_str(), |b| {
        b.iter(|| monolith.permute_mut(&mut input));
    });
}

fn monolith_gl8<Mds, const WIDTH: usize>(c: &mut Criterion, mds_name: &str, mds: Mds)
where
    Mds: MdsPermutation<Goldilocks, WIDTH>,
{
    let bars = MonolithBarsGoldilocks::<8>;
    let monolith: MonolithGoldilocks8<_, WIDTH, 5> = MonolithGoldilocks8::new(bars, mds);

    let mut input = array::from_fn(|i| Goldilocks::new(i as u64));

    let name = format!("monolith::<Goldilocks8, {WIDTH}> ({mds_name})");
    c.bench_function(name.as_str(), |b| {
        b.iter(|| monolith.permute_mut(&mut input));
    });
}

criterion_group!(benches, bench_monolith);
criterion_main!(benches);
