//! Fixed-base exponentiation in `GF(2^192)` from windowed tables, for 127-bit and 192-bit exponents.

use core::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use p3_binary_field::{Poly64, Poly192};
use p3_field::{Field, FixedBase, PrimeCharacteristicRing};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

/// Exponents per batched call.
const BATCH: usize = 1 << 14;

/// Window widths the sweep compares.
const WINDOWS: [u32; 5] = [4, 6, 8, 10, 12];

/// Random exponents of at most `bits` bits.
fn exponents(bits: u32, n: usize) -> Vec<[u64; 3]> {
    let mut rng = SmallRng::seed_from_u64(u64::from(bits));
    (0..n)
        .map(|_| {
            let mut e: [u64; 3] = rng.random();
            for (i, limb) in e.iter_mut().enumerate() {
                let keep = bits.saturating_sub(64 * i as u32).min(64);
                *limb = if keep == 64 {
                    *limb
                } else {
                    *limb & ((1 << keep) - 1)
                };
            }
            e
        })
        .collect()
}

fn bench_table(c: &mut Criterion) {
    let mut group = c.benchmark_group("fixed_base/table");
    for w in WINDOWS {
        group.bench_with_input(BenchmarkId::new("192", w), &w, |b, &w| {
            b.iter(|| FixedBase::with_window(black_box(Poly192::GENERATOR), 192, w));
        });
    }
}

fn bench_pow(c: &mut Criterion) {
    for bits in [127, 192] {
        let exps = exponents(bits, BATCH);
        let mut out = vec![Poly192::ZERO; BATCH];

        let mut group = c.benchmark_group(format!("fixed_base/{bits}"));
        group.throughput(Throughput::Elements(BATCH as u64));
        for w in WINDOWS {
            let table = FixedBase::with_window(Poly192::GENERATOR, 192, w);
            group.bench_with_input(BenchmarkId::new("pow", w), &w, |b, _| {
                b.iter(|| {
                    for (o, e) in out.iter_mut().zip(black_box(&exps)) {
                        *o = table.pow(e);
                    }
                });
            });
            group.bench_with_input(BenchmarkId::new("pow_into", w), &w, |b, _| {
                b.iter(|| table.pow_into::<Poly64, _>(black_box(&exps), &mut out));
            });
        }
    }
}

criterion_group!(benches, bench_table, bench_pow);
criterion_main!(benches);
