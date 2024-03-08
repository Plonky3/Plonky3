use criterion::{criterion_group, criterion_main, Criterion};
use p3_field::AbstractField;
use p3_mds::MdsPermutation;
use p3_mersenne_31::{MdsMatrixMersenne31, Mersenne31};
use p3_monolith::MonolithMersenne31;

fn bench_monolith(c: &mut Criterion) {
    monolith::<_, 12>(c, MdsMatrixMersenne31);
    monolith::<_, 16>(c, MdsMatrixMersenne31);
}

fn monolith<Mds, const WIDTH: usize>(c: &mut Criterion, mds: Mds)
where
    Mds: MdsPermutation<Mersenne31, WIDTH>,
{
    let monolith: MonolithMersenne31<_, WIDTH, 5> = MonolithMersenne31::new(mds);

    let mut input: [Mersenne31; WIDTH] = [Mersenne31::zero(); WIDTH];
    for (i, inp) in input.iter_mut().enumerate() {
        *inp = Mersenne31::from_canonical_usize(i);
    }

    let name = format!("monolith::<Mersenne31, {}>", WIDTH);
    c.bench_function(name.as_str(), |b| {
        b.iter(|| monolith.permutation(&mut input))
    });
}

criterion_group!(benches, bench_monolith);
criterion_main!(benches);
