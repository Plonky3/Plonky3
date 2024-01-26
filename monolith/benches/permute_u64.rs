use criterion::{criterion_group, criterion_main, Criterion};
use p3_monolith::{MonolithMdsMatrixMersenne31U64Width16, MonolithMersenne31U64Width16};

fn permute_u64_benchmark(c: &mut Criterion) {
    let mds = MonolithMdsMatrixMersenne31U64Width16;
    let monolith: MonolithMersenne31U64Width16<5> = MonolithMersenne31U64Width16::new(mds);

    let mut input: [u64; 16] = [0; 16];
    for (i, inp) in input.iter_mut().enumerate() {
        *inp = i as u64;
    }

    c.bench_function("monolith permutation u64", |b| {
        b.iter(|| monolith.permutation(&mut input))
    });
}

criterion_group!(benches, permute_u64_benchmark);
criterion_main!(benches);
