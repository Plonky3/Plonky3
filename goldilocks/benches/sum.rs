use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use p3_field::AbstractField;
use p3_goldilocks::{sum_u128, Goldilocks};
use rand::Rng;

type F = Goldilocks;

fn bench_sum(c: &mut Criterion) {
    sum::<1>(c);
    sum_delayed::<1>(c);

    sum::<2>(c);
    sum_delayed::<2>(c);

    sum::<4>(c);
    sum_delayed::<4>(c);

    sum::<8>(c);
    sum_delayed::<8>(c);

    sum::<12>(c);
    sum_delayed::<12>(c);

    sum::<16>(c);
    sum_delayed::<16>(c);

    sum::<32>(c);
    sum_delayed::<32>(c);
}

fn sum<const N: usize>(c: &mut Criterion) {
    let mut rng = rand::thread_rng();
    let mut input = Vec::new();
    for _ in 0..1000 {
        let mut row = Vec::new();
        for _ in 0..N {
            row.push(rng.gen::<F>())
        }
        input.push(row)
    }

    let id = BenchmarkId::new("Goldilocks sum", N);
    c.bench_with_input(id, &input, |b, input| {
        b.iter(|| {
            black_box(input);
            let mut res = [F::zero(); 1000];
            for i in 0..1000 {
                res[i] = input[i].iter().fold(F::zero(), |x, y| x + *y);
            }
            res
        });
    });
}

fn sum_delayed<const N: usize>(c: &mut Criterion) {
    let mut rng = rand::thread_rng();
    let mut input = Vec::new();
    for _ in 0..1000 {
        let mut row = Vec::new();
        for _ in 0..N {
            row.push(rng.gen::<F>())
        }
        input.push(row)
    }

    let id = BenchmarkId::new("Goldilocks sum_delayed", N);
    c.bench_with_input(id, &input, |b, input| {
        b.iter(|| {
            black_box(input);
            let mut res = [F::zero(); 1000];
            for i in 0..1000 {
                res[i] = sum_u128(&input[i]);
            }
            res
        });
    });
}

criterion_group!(goldilocks_arithmetic, bench_sum);
criterion_main!(goldilocks_arithmetic);
