use criterion::{criterion_group, criterion_main, Criterion};
use p3_field::AbstractField;
use p3_mersenne_31::Mersenne31;
use p3_monolith::{MonolithMdsMatrixMersenne31Width16, MonolithMersenne31Width16};

fn permute_u64_benchmark(c: &mut Criterion) {
    let mds = MonolithMdsMatrixMersenne31Width16;
    let monolith: MonolithMersenne31Width16<5> = MonolithMersenne31Width16::new(mds);

    let mut input: [Mersenne31; 16] = [Mersenne31::zero(); 16];
    for (i, inp) in input.iter_mut().enumerate() {
        *inp = Mersenne31::from_canonical_usize(i);
    }

    c.bench_function("monolith permutation u64", |b| {
        b.iter(|| monolith.permutation(&mut input))
    });
}

criterion_group!(benches, permute_u64_benchmark);
criterion_main!(benches);
