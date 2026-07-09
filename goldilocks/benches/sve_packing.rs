//! Throughput comparison of the SVE packed backend against scalar `Goldilocks`.
//!
//! Goldilocks is the interesting case: NEON has no 64×64→128 multiply and falls back to scalar asm,
//! whereas SVE's `UMULH` vectorises it. Requires an SVE host (Graviton3) and nightly:
//! ```text
//! RUSTFLAGS="-C target-cpu=neoverse-v1 -C target-feature=+sve" \
//!     cargo +nightly bench -p p3-goldilocks --bench sve_packing
//! ```

use criterion::{Criterion, criterion_group, criterion_main};

#[cfg(all(target_arch = "aarch64", target_feature = "sve"))]
fn bench_sve_packing(c: &mut Criterion) {
    use std::hint::black_box;

    use criterion::Throughput;
    use p3_field::{PackedValue, PrimeCharacteristicRing};
    use p3_goldilocks::{Goldilocks, PackedGoldilocksSve};
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    const N: usize = 1 << 10;
    let width = PackedGoldilocksSve::WIDTH;
    let elems = (N * width) as u64;

    let mut rng = SmallRng::seed_from_u64(1);
    let sa: Vec<Goldilocks> = (0..N * width).map(|_| rng.random()).collect();
    let sb: Vec<Goldilocks> = (0..N * width).map(|_| rng.random()).collect();
    let pa: Vec<PackedGoldilocksSve> = PackedGoldilocksSve::pack_slice(&sa).to_vec();
    let pb: Vec<PackedGoldilocksSve> = PackedGoldilocksSve::pack_slice(&sb).to_vec();

    // The Goldilocks headline: NEON falls back to scalar asm here; SVE `UMULH` vectorises it.
    let mut g = c.benchmark_group("goldilocks_mul");
    g.throughput(Throughput::Elements(elems));
    g.bench_function("sve_packed", |bn| {
        bn.iter(|| {
            let mut acc = PackedGoldilocksSve::ONE;
            for (&x, &y) in pa.iter().zip(&pb) {
                acc = black_box(acc + x * y);
            }
            black_box(acc)
        })
    });
    g.bench_function("scalar", |bn| {
        bn.iter(|| {
            let mut acc = Goldilocks::ONE;
            for (&x, &y) in sa.iter().zip(&sb) {
                acc = black_box(acc + x * y);
            }
            black_box(acc)
        })
    });
    g.finish();

    // Vectorised add (double-overflow-correct) vs scalar.
    let mut g = c.benchmark_group("goldilocks_add");
    g.throughput(Throughput::Elements(elems));
    g.bench_function("sve_packed", |bn| {
        bn.iter(|| {
            let mut acc = PackedGoldilocksSve::ZERO;
            for (&x, &y) in pa.iter().zip(&pb) {
                acc = black_box(acc + (x + y));
            }
            black_box(acc)
        })
    });
    g.bench_function("scalar", |bn| {
        bn.iter(|| {
            let mut acc = Goldilocks::ZERO;
            for (&x, &y) in sa.iter().zip(&sb) {
                acc = black_box(acc + (x + y));
            }
            black_box(acc)
        })
    });
    g.finish();
}

#[cfg(not(all(target_arch = "aarch64", target_feature = "sve")))]
fn bench_sve_packing(_c: &mut Criterion) {
    eprintln!(
        "SVE backend disabled. Rebuild on nightly with \
         RUSTFLAGS=\"-C target-cpu=neoverse-v1 -C target-feature=+sve\" on an SVE (Graviton3) host."
    );
}

criterion_group!(benches, bench_sve_packing);
criterion_main!(benches);
