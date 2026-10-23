//! Commit, prove and verify one Goldilocks claim on `2^n` committed bits at the balanced split.

use core::hint::black_box;

use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use p3_bitz::{BitShape, BitzParams, BitzPcs, Challenger, ColumnWeights, RowWeights, TensorClaim};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::Goldilocks;
use p3_keccak::Keccak256Hash;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

/// Committed variables benchmarked.
const SIZES: [usize; 4] = [20, 24, 26, 28];

/// A fresh sponge.
const fn sponge() -> Challenger {
    Challenger::from_hasher(Vec::new(), Keccak256Hash)
}

/// Random words and one dense-row, equality-column claim on them.
fn instance(shape: BitShape) -> (Vec<u64>, TensorClaim<Goldilocks>) {
    let mut rng = SmallRng::seed_from_u64(shape.num_variables() as u64);
    let words: Vec<u64> = (0..shape.words()).map(|_| rng.next_u64()).collect();
    let u1: Vec<Goldilocks> = (0..shape.rows())
        .map(|_| Goldilocks::from_u64(rng.next_u64()))
        .collect();
    let point: Vec<Goldilocks> = (0..shape.log_columns())
        .map(|_| Goldilocks::from_u64(rng.next_u64()))
        .collect();
    let mut claim = TensorClaim {
        rows: RowWeights::Dense(u1.clone()),
        columns: ColumnWeights::Equality(point.clone()),
        value: Goldilocks::ZERO,
    };
    // The value: each column's integer fold, weighed by its equality weight.
    let mut eq = vec![Goldilocks::ONE];
    for &x in &point {
        eq = eq
            .iter()
            .flat_map(|&e| [e * (Goldilocks::ONE - x), e * x])
            .collect();
    }
    claim.value = words
        .chunks_exact(shape.column_words())
        .zip(&eq)
        .map(|(column, &e)| {
            let fold: u128 = (0..shape.rows())
                .filter(|b| column[b / 64] >> (b % 64) & 1 == 1)
                .map(|b| u128::from(u1[b].as_canonical_u64()))
                .sum();
            e * Goldilocks::from_u128(fold)
        })
        .sum();
    (words, claim)
}

fn bench_pcs(c: &mut Criterion) {
    let mut group = c.benchmark_group("bitz_pcs");
    group.sample_size(10);
    for n in SIZES {
        let shape = BitShape::balanced(n).unwrap();
        let pcs = BitzPcs::<Goldilocks>::new(shape, &BitzParams::default()).unwrap();
        let (words, claim) = instance(shape);
        let claims = [claim];

        group.bench_function(BenchmarkId::new("commit", n), |b| {
            b.iter_batched(
                || words.clone(),
                |words| black_box(pcs.commit(words, &mut sponge()).unwrap()),
                BatchSize::LargeInput,
            );
        });
        group.bench_function(BenchmarkId::new("prove", n), |b| {
            b.iter_batched(
                || {
                    let mut challenger = sponge();
                    let (_, data) = pcs.commit(words.clone(), &mut challenger).unwrap();
                    (data, challenger)
                },
                |(data, mut challenger)| {
                    black_box(pcs.prove(data, &claims, &mut challenger).unwrap())
                },
                BatchSize::LargeInput,
            );
        });
        let mut challenger = sponge();
        let (root, data) = pcs.commit(words.clone(), &mut challenger).unwrap();
        let proof = pcs.prove(data, &claims, &mut challenger).unwrap();
        group.bench_function(BenchmarkId::new("verify", n), |b| {
            b.iter(|| pcs.verify(&root, &claims, &proof, &mut sponge()).unwrap());
        });
    }
    group.finish();
}

criterion_group!(benches, bench_pcs);
criterion_main!(benches);
