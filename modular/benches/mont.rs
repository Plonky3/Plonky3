//! Montgomery arithmetic at 2, 3 and 4 limbs, its slice kernels, and prime sampling.
//!
//! Each field is a sampled prime with the top bit spare: 127, 191 and 255 bits.

use std::hint::black_box;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use p3_modular::prime::is_bpsw;
use p3_modular::{ModField, ModRing, MontField, PrimePolicy, Residue, sample_prime};
use rand::rngs::SmallRng;
use rand::{Rng, RngExt, SeedableRng};

/// Products in one iteration of a scalar benchmark.
const REPS: usize = 1024;

/// Independent chains a throughput benchmark interleaves.
const LANES: usize = 8;

/// Slice length of the kernel benchmarks: four arrays stay in L1.
const LEN: usize = 1 << 12;

/// A byte stream from a seeded generator.
fn stream(seed: u64) -> impl FnMut(&mut [u8]) {
    let mut rng = SmallRng::seed_from_u64(seed);
    move |out: &mut [u8]| rng.fill_bytes(out)
}

fn field<const L: usize>() -> MontField<L> {
    let bits = 64 * L as u32 - 1;
    MontField::from_sample(sample_prime(stream(1), &PrimePolicy::new(bits)).expect("prime"))
}

fn random<const L: usize>(f: &MontField<L>, rng: &mut SmallRng, n: usize) -> Vec<Residue<L>> {
    (0..n)
        .map(|_| {
            let limbs: Vec<u64> = (0..L + 1).map(|_| rng.random()).collect();
            f.reduce_limbs(&limbs)
        })
        .collect()
}

fn bench_limbs<const L: usize>(c: &mut Criterion) {
    let f = field::<L>();
    let mut rng = SmallRng::seed_from_u64(2);
    let ops = random(&f, &mut rng, REPS);
    let a = random(&f, &mut rng, LEN);
    let b = random(&f, &mut rng, LEN);
    let point = random(&f, &mut rng, 12);
    let bits = 64 * L - 1;

    let mut g = c.benchmark_group(format!("mont/{bits}"));
    g.throughput(Throughput::Elements(REPS as u64));
    g.bench_function("mul/latency", |bn| {
        bn.iter(|| {
            black_box(&ops)
                .iter()
                .fold(f.one(), |acc, &y| f.mul(acc, y))
        });
    });
    g.bench_function("square/latency", |bn| {
        bn.iter(|| (0..REPS).fold(black_box(ops[0]), |acc, _| f.square(acc)));
    });
    g.bench_function("mul/throughput", |bn| {
        bn.iter(|| {
            let mut acc: [Residue<L>; LANES] = core::array::from_fn(|i| ops[i]);
            for chunk in black_box(&ops).as_chunks::<LANES>().0 {
                for (x, &y) in acc.iter_mut().zip(chunk) {
                    *x = f.mul(*x, y);
                }
            }
            acc
        });
    });
    g.throughput(Throughput::Elements(LEN as u64));
    g.bench_function("dot", |bn| bn.iter(|| f.dot(black_box(&a), black_box(&b))));
    // The kernels run in place on one buffer, so its values drift but stay field elements.
    let mut x = a.clone();
    g.bench_function("mul_slices", |bn| {
        bn.iter(|| f.mul_slices(black_box(&mut x), black_box(&b)));
    });
    g.bench_function("scale_add", |bn| {
        bn.iter(|| f.scale_add(black_box(&mut x), black_box(&b), ops[3]));
    });
    g.bench_function("eq_table", |bn| bn.iter(|| f.eq_table(black_box(&point))));
    g.bench_function("batch_inverse", |bn| {
        bn.iter(|| f.batch_inverse(black_box(&mut x)));
    });
    g.throughput(Throughput::Elements(1));
    g.bench_function("inverse", |bn| bn.iter(|| f.inverse(black_box(ops[5]))));
    g.finish();
}

fn bench_prime<const L: usize>(c: &mut Criterion) {
    let f = field::<L>();
    let bits = 64 * L - 1;
    let mut g = c.benchmark_group(format!("prime/{bits}"));
    g.bench_function("bpsw", |bn| bn.iter(|| is_bpsw(black_box(f.ring()))));
    let policy = PrimePolicy::new(bits as u32);
    let mut seed = 0;
    g.bench_function("sample", |bn| {
        bn.iter(|| {
            seed += 1;
            sample_prime::<L>(stream(seed), &policy)
        });
    });
    g.finish();
}

fn bench_mont(c: &mut Criterion) {
    bench_limbs::<2>(c);
    bench_limbs::<3>(c);
    bench_limbs::<4>(c);
    bench_prime::<2>(c);
    bench_prime::<3>(c);
    bench_prime::<4>(c);
}

criterion_group!(benches, bench_mont);
criterion_main!(benches);
