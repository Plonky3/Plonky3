use core::array;

use criterion::{Criterion, criterion_group, criterion_main};
use p3_field::PrimeCharacteristicRing;
use p3_mds::MdsPermutation;
use p3_mersenne_31::{MdsMatrixMersenne31, Mersenne31};
use p3_monolith::{MonolithBarsM31, MonolithMersenne31};
use p3_symmetric::Permutation;

fn bench_monolith(c: &mut Criterion) {
    monolith::<_, 12>(c, MdsMatrixMersenne31);
    monolith::<_, 16>(c, MdsMatrixMersenne31);
}

fn monolith<Mds, const WIDTH: usize>(c: &mut Criterion, mds: Mds)
where
    Mds: MdsPermutation<Mersenne31, WIDTH>,
{
    let bars = MonolithBarsM31;
    let monolith: MonolithMersenne31<_, WIDTH, 5> = MonolithMersenne31::new(bars, mds);

    let mut input = array::from_fn(Mersenne31::from_usize);

    let name = format!("monolith::<Mersenne31, {WIDTH}>");
    c.bench_function(name.as_str(), |b| {
        b.iter(|| monolith.permute_mut(&mut input));
    });
}

criterion_group!(benches, bench_monolith);
criterion_main!(benches);
