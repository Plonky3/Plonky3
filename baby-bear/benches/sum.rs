use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use p3_baby_bear::{sum_u64, BabyBear};
use p3_field::AbstractField;
use rand::Rng;

type F = BabyBear;

fn bench_sum(c: &mut Criterion) {
    const REPS: usize = 1000;
    const SIZES: [usize; 6] = [1, 2, 4, 8, 16, 32];

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

    let id = BenchmarkId::new("BabyBear sum", size);
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

    let id = BenchmarkId::new("BabyBear sum_delayed", size);
    c.bench_with_input(id, &input, |b, input| {
        b.iter(|| {
            black_box(input);
            let mut res = [F::zero(); REPS];
            for i in 0..REPS {
                res[i] = sum_u64(&input[i]);
            }
            res
        });
    });
}

criterion_group!(baby_bear_arithmetic, bench_sum);
criterion_main!(baby_bear_arithmetic);
