use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use p3_field::AbstractField;
use p3_goldilocks::{sum_u128, Goldilocks};
use rand::Rng;

type F = Goldilocks;

fn bench_sum(c: &mut Criterion) {
    const REPS: usize = 1000;
    const SIZES: [usize; 6] = [1, 2, 4, 8, 12, 16];

    SIZES.map(|size| {
        sum::<REPS>(c, size);
        sum_delayed::<REPS>(c, size);
    });
}

fn sum<const REPS: usize>(c: &mut Criterion, size: usize) {
    let mut rng = rand::thread_rng();
    let mut input = Vec::new();
    for _ in 0..REPS {
        let mut row = Vec::new();
        for _ in 0..size {
            row.push(rng.gen::<F>())
        }
        input.push(row)
    }

    let id = BenchmarkId::new("Goldilocks sum", size);
    c.bench_with_input(id, &input, |b, input| {
        b.iter(|| {
            black_box(input);
            let mut res = [F::zero(); REPS];
            for i in 0..REPS {
                res[i] = input[i].iter().fold(F::zero(), |x, y| x + *y);
            }
            res
        });
    });
}

fn sum_delayed<const REPS: usize>(c: &mut Criterion, size: usize) {
    let mut rng = rand::thread_rng();
    let mut input = Vec::new();
    for _ in 0..REPS {
        let mut row = Vec::new();
        for _ in 0..size {
            row.push(rng.gen::<F>())
        }
        input.push(row)
    }

    let id = BenchmarkId::new("Goldilocks sum_delayed", size);
    c.bench_with_input(id, &input, |b, input| {
        b.iter(|| {
            black_box(input);
            let mut res = [F::zero(); REPS];
            for i in 0..REPS {
                res[i] = sum_u128(&input[i]);
            }
            res
        });
    });
}

criterion_group!(goldilocks_arithmetic, bench_sum);
criterion_main!(goldilocks_arithmetic);
