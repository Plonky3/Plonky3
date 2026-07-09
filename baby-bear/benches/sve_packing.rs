//! Throughput comparison of the SVE packed backend against scalar `BabyBear`.
//!
//! Requires an SVE host (Graviton3 / Neoverse V1) and nightly:
//! ```text
//! RUSTFLAGS="-C target-cpu=neoverse-v1 -C target-feature=+sve" \
//!     cargo +nightly bench -p p3-baby-bear --bench sve_packing
//! ```

use criterion::{Criterion, criterion_group, criterion_main};

#[cfg(all(target_arch = "aarch64", target_feature = "sve"))]
fn bench_sve_packing(c: &mut Criterion) {
    use std::hint::black_box;

    use criterion::Throughput;
    use p3_baby_bear::{BabyBear, PackedBabyBearSve};
    use p3_field::{PackedFieldPow2, PackedValue, PrimeCharacteristicRing};
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    const N: usize = 1 << 10;
    let width = PackedBabyBearSve::WIDTH;
    let elems = (N * width) as u64;

    let mut rng = SmallRng::seed_from_u64(1);
    let sa: Vec<BabyBear> = (0..N * width).map(|_| rng.random()).collect();
    let sb: Vec<BabyBear> = (0..N * width).map(|_| rng.random()).collect();
    let pa: Vec<PackedBabyBearSve> = PackedBabyBearSve::pack_slice(&sa).to_vec();
    let pb: Vec<PackedBabyBearSve> = PackedBabyBearSve::pack_slice(&sb).to_vec();

    let mut g = c.benchmark_group("baby_bear_mul");
    g.throughput(Throughput::Elements(elems));
    g.bench_function("sve_packed", |bn| {
        bn.iter(|| {
            let mut acc = PackedBabyBearSve::ONE;
            for (&x, &y) in pa.iter().zip(&pb) {
                acc = black_box(acc + x * y);
            }
            black_box(acc)
        })
    });
    g.bench_function("scalar", |bn| {
        bn.iter(|| {
            let mut acc = BabyBear::ONE;
            for (&x, &y) in sa.iter().zip(&sb) {
                acc = black_box(acc + x * y);
            }
            black_box(acc)
        })
    });
    g.finish();

    let mut g = c.benchmark_group("baby_bear_sbox_exp7");
    g.throughput(Throughput::Elements(elems));
    g.bench_function("sve_packed", |bn| {
        bn.iter(|| {
            let mut acc = PackedBabyBearSve::ZERO;
            for &x in &pa {
                acc = black_box(acc + x.exp_const_u64::<7>());
            }
            black_box(acc)
        })
    });
    g.bench_function("scalar", |bn| {
        bn.iter(|| {
            let mut acc = BabyBear::ZERO;
            for &x in &sa {
                acc = black_box(acc + x.exp_const_u64::<7>());
            }
            black_box(acc)
        })
    });
    g.finish();

    let mut g = c.benchmark_group("baby_bear_interleave_block1");
    g.throughput(Throughput::Elements(elems));
    g.bench_function("sve_packed", |bn| {
        bn.iter(|| {
            let mut acc = PackedBabyBearSve::ZERO;
            for (&x, &y) in pa.iter().zip(&pb) {
                let (u, v) = x.interleave(y, 1);
                acc = black_box(acc + u + v);
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
