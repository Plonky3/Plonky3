use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use p3_baby_bear::{babybear_quad_mul, babybear_triple_mul, fast_exp_7, BabyBear};
use p3_field::AbstractField;
use rand::Rng;

type F = BabyBear;

fn bench_mul(c: &mut Criterion) {
    exp_two(c);
    exp_three(c);
    mul_three(c);
    mul_three_delayed(c);
    mul_four(c);
    mul_four_delayed(c);
    exp_seven(c);
    exp_seven_delayed_1(c);
    exp_seven_delayed_2(c);
    exp_seven_delayed_3(c);
}

fn exp_two(c: &mut Criterion) {
    let mut rng = rand::thread_rng();
    let input = rng.gen::<F>();
    let id = BenchmarkId::new("BabyBear Exponential", 2);
    c.bench_with_input(id, &input, |b, input| {
        b.iter(|| {
            black_box(input);
            let mut res = *input;
            for _ in 0..1000 {
                res = res.square();
            }
            res
        });
    });
}

fn exp_three(c: &mut Criterion) {
    let mut rng = rand::thread_rng();
    let input = rng.gen::<F>();
    let id = BenchmarkId::new("BabyBear Exponential", 3);
    c.bench_with_input(id, &input, |b, input| {
        b.iter(|| {
            black_box(input);
            let mut res = *input;
            for _ in 0..1000 {
                res = babybear_triple_mul(res, res, res);
            }
            res
        });
    });
}

fn mul_three(c: &mut Criterion) {
    let mut rng = rand::thread_rng();
    let input = rng.gen::<[F; 2001]>();
    let id = BenchmarkId::new("BabyBear Three Multiplications", 3);
    c.bench_with_input(id, &input, |b, input| {
        b.iter(|| {
            black_box(input);
            let mut res = input[0];
            for i in 0..1000 {
                res = res * input[2*i + 1] * input[2*i + 2];
            }
            res
        });
    });
}

fn mul_three_delayed(c: &mut Criterion) {
    let mut rng = rand::thread_rng();
    let input = rng.gen::<[F; 2001]>();
    let id = BenchmarkId::new("BabyBear Three Multiplications Delayed", 3);
    c.bench_with_input(id, &input, |b, input| {
        b.iter(|| {
            black_box(input);
            let mut res = input[0];
            for i in 0..1000 {
                res = babybear_triple_mul(res, input[2*i + 1], input[2*i + 2]);
            }
            res
            
        });
    });
}

fn mul_four(c: &mut Criterion) {
    let mut rng = rand::thread_rng();
    let input = rng.gen::<[F; 3001]>();
    let id = BenchmarkId::new("BabyBear Four Multiplications", 4);
    c.bench_with_input(id, &input, |b, input| {
        b.iter(|| {
            black_box(input);
            let mut res = input[0];
            for i in 0..1000 {
                res = res * input[3*i + 1] * input[3*i + 2] * input[3*i + 3];
            }
            res
        });
    });
}

fn mul_four_delayed(c: &mut Criterion) {
    let mut rng = rand::thread_rng();
    let input = rng.gen::<[F; 3001]>();
    let id = BenchmarkId::new("BabyBear Four Multiplications Delayed", 4);
    c.bench_with_input(id, &input, |b, input| {
        b.iter(|| {
            black_box(input);
            let mut res = input[0];
            for i in 0..1000 {
                res = babybear_quad_mul(res, input[3*i + 1], input[3*i + 2], input[3*i + 3]);
            }
            res
        });
    });
}

fn exp_seven(c: &mut Criterion) {
    let mut rng = rand::thread_rng();
    let input = rng.gen::<F>();
    let id = BenchmarkId::new("BabyBear Exponential", 7);
    c.bench_with_input(id, &input, |b, input| {
        b.iter(|| {
            black_box(input);
            let mut res = *input;
            for _ in 0..1000 {
                let square = res.square();
                let triple = square * res;
                let quad = square.square();
                res = triple * quad
            }
            res
        });
    });
}

fn exp_seven_delayed_1(c: &mut Criterion) {
    let mut rng = rand::thread_rng();
    let input = rng.gen::<F>();
    let id = BenchmarkId::new("BabyBear Exponential Delayed 1", 7);
    c.bench_with_input(id, &input, |b, input| {
        b.iter(|| {
            black_box(input);
            let mut res = *input;
            for _ in 0..1000 {
                let square = res.square();
                res = babybear_quad_mul(square, square, square, res);
            }
            res
        });
    });
}

fn exp_seven_delayed_2(c: &mut Criterion) {
    let mut rng = rand::thread_rng();
    let input = rng.gen::<F>();
    let id = BenchmarkId::new("BabyBear Exponential Delayed 2", 7);
    c.bench_with_input(id, &input, |b, input| {
        b.iter(|| {
            black_box(input);
            let mut res = *input;
            for _ in 0..1000 {
                let cube = babybear_triple_mul(res, res, res);
                res = babybear_triple_mul(cube, cube, res);
            }
            res
        });
    });
}

fn exp_seven_delayed_3(c: &mut Criterion) {
    let mut rng = rand::thread_rng();
    let input = rng.gen::<F>();
    let id = BenchmarkId::new("BabyBear Exponential Delayed 3", 7);
    c.bench_with_input(id, &input, |b, input| {
        b.iter(|| {
            black_box(input);
            let mut res = *input;
            for _ in 0..1000 {
                res = fast_exp_7(res);
            }
            res
        });
    });
}

criterion_group!(baby_bear_arithmetic, bench_mul);
criterion_main!(baby_bear_arithmetic);
