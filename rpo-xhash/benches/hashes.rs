//! Criterion benches for RPO and xHash permutations.
//!
//! Small-prime variants use width 24; Goldilocks uses width 12.

use core::array;

use criterion::{criterion_group, criterion_main, Criterion};
use p3_rpo_xhash::rpo::{
    babybear::rpo_babybear,
    goldilocks::rpo_goldilocks,
    koalabear::rpo_koalabear,
    m31::{rpo_m31_bb_mds, rpo_m31_cir},
};
use p3_rpo_xhash::xhash::{
    babybear::xhash_babybear,
    goldilocks::xhash_goldilocks,
    koalabear::xhash_koalabear,
    m31::{xhash_m31_bb_mds, xhash_m31_cir},
};
use p3_baby_bear::BabyBear;
use p3_goldilocks::Goldilocks;
use p3_koala_bear::KoalaBear;
use p3_mersenne_31::Mersenne31;
use p3_symmetric::Permutation;
use rand::rngs::SmallRng;
use rand::SeedableRng;

fn bench_rpo(c: &mut Criterion) {
    let mut g = c.benchmark_group("rpo/permute24");

    let mut rng = SmallRng::seed_from_u64(1);
    let hash = rpo_babybear(&mut rng);
    let input: [BabyBear; 24] = array::from_fn(|i| BabyBear::new((i as u32).wrapping_add(1)));
    g.bench_function("babybear", |b| b.iter(|| hash.permute(input)));

    let mut rng = SmallRng::seed_from_u64(1);
    let hash = rpo_koalabear(&mut rng);
    let input: [KoalaBear; 24] = array::from_fn(|i| KoalaBear::new((i as u32).wrapping_add(1)));
    g.bench_function("koalabear", |b| b.iter(|| hash.permute(input)));

    let mut rng = SmallRng::seed_from_u64(1);
    let hash = rpo_m31_cir(&mut rng);
    let input: [Mersenne31; 24] = array::from_fn(|i| Mersenne31::new((i as u32).wrapping_add(1)));
    g.bench_function("m31_cir", |b| b.iter(|| hash.permute(input)));

    let mut rng = SmallRng::seed_from_u64(1);
    let hash = rpo_m31_bb_mds(&mut rng);
    let input: [Mersenne31; 24] = array::from_fn(|i| Mersenne31::new((i as u32).wrapping_add(1)));
    g.bench_function("m31_bb_mds", |b| b.iter(|| hash.permute(input)));

    g.finish();
}

fn bench_rpo_goldilocks(c: &mut Criterion) {
    let mut g = c.benchmark_group("rpo/permute12");
    let input: [Goldilocks; 12] =
        array::from_fn(|i| Goldilocks::new((i as u64 + 1) * 1_000_000_007));

    let mut rng = SmallRng::seed_from_u64(3);
    let hash = rpo_goldilocks(&mut rng);
    g.bench_function("goldilocks", |b| b.iter(|| hash.permute(input)));

    g.finish();
}

fn bench_xhash(c: &mut Criterion) {
    let mut g = c.benchmark_group("xhash/permute24");

    let mut rng = SmallRng::seed_from_u64(2);
    let hash = xhash_babybear(&mut rng);
    let input: [BabyBear; 24] = array::from_fn(|i| BabyBear::new((i as u32).wrapping_add(1)));
    g.bench_function("babybear", |b| b.iter(|| hash.permute(input)));

    let mut rng = SmallRng::seed_from_u64(2);
    let hash = xhash_koalabear(&mut rng);
    let input: [KoalaBear; 24] = array::from_fn(|i| KoalaBear::new((i as u32).wrapping_add(1)));
    g.bench_function("koalabear", |b| b.iter(|| hash.permute(input)));

    let mut rng = SmallRng::seed_from_u64(2);
    let hash = xhash_m31_cir(&mut rng);
    let input: [Mersenne31; 24] = array::from_fn(|i| Mersenne31::new((i as u32).wrapping_add(1)));
    g.bench_function("m31_cir", |b| b.iter(|| hash.permute(input)));

    let mut rng = SmallRng::seed_from_u64(2);
    let hash = xhash_m31_bb_mds(&mut rng);
    let input: [Mersenne31; 24] = array::from_fn(|i| Mersenne31::new((i as u32).wrapping_add(1)));
    g.bench_function("m31_bb_mds", |b| b.iter(|| hash.permute(input)));

    g.finish();
}

fn bench_xhash_goldilocks(c: &mut Criterion) {
    let mut g = c.benchmark_group("xhash/permute12");
    let input: [Goldilocks; 12] =
        array::from_fn(|i| Goldilocks::new((i as u64 + 1) * 1_000_000_007));

    let mut rng = SmallRng::seed_from_u64(4);
    let hash = xhash_goldilocks(&mut rng);
    g.bench_function("goldilocks", |b| b.iter(|| hash.permute(input)));

    g.finish();
}

criterion_group!(
    benches,
    bench_rpo,
    bench_rpo_goldilocks,
    bench_xhash,
    bench_xhash_goldilocks
);
criterion_main!(benches);
