use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use p3_baby_bear::{babybear_quad_mul, babybear_triple_mul, BabyBear};
use p3_field::AbstractField;
use rand::Rng;

type F = BabyBear;

fn bench_mul(c: &mut Criterion) {
    mul_three(c);
    mul_three_delayed(c);
    mul_four(c);
    mul_four_delayed(c);
    exp_seven(c);
    exp_seven_delayed_1(c);
    exp_seven_delayed_2(c);
}

fn mul_three(c: &mut Criterion) {
    let mut rng = rand::thread_rng();
    let input = rng.gen::<[F; 3]>();
    let id = BenchmarkId::new("BabyBear Three MultiMultiplications", 3);
    c.bench_with_input(id, &input, |b, input| {
        b.iter(|| {
            black_box(input);
            input[0] * input[1] * input[2]
        });
    });
}

fn mul_three_delayed(c: &mut Criterion) {
    let mut rng = rand::thread_rng();
    let input = rng.gen::<[F; 3]>();
    let id = BenchmarkId::new("BabyBear Three MultiMultiplications Delayed", 3);
    c.bench_with_input(id, &input, |b, input| {
        b.iter(|| {
            black_box(input);
            babybear_triple_mul(input[0], input[1], input[2])
        });
    });
}

fn mul_four(c: &mut Criterion) {
    let mut rng = rand::thread_rng();
    let input = rng.gen::<[F; 4]>();
    let id = BenchmarkId::new("BabyBear Four MultiMultiplications", 4);
    c.bench_with_input(id, &input, |b, input| {
        b.iter(|| {
            black_box(input);
            input[0] * input[1] * input[2] * input[3]
        });
    });
}

fn mul_four_delayed(c: &mut Criterion) {
    let mut rng = rand::thread_rng();
    let input = rng.gen::<[F; 4]>();
    let id = BenchmarkId::new("BabyBear Four MultiMultiplications Delayed", 4);
    c.bench_with_input(id, &input, |b, input| {
        b.iter(|| {
            black_box(input);
            babybear_quad_mul(input[0], input[1], input[2], input[3])
        });
    });
}

fn exp_seven(c: &mut Criterion) {
    let mut rng = rand::thread_rng();
    let input = rng.gen::<F>();
    let id = BenchmarkId::new("BabyBear MultiExponential", 7);
    c.bench_with_input(id, &input, |b, input| {
        b.iter(|| {
            black_box(input);
            let square = input.square();
            let triple = square * *input;
            let quad = square.square();
            triple * quad
        });
    });
}

fn exp_seven_delayed_1(c: &mut Criterion) {
    let mut rng = rand::thread_rng();
    let input = rng.gen::<F>();
    let id = BenchmarkId::new("BabyBear MultiExponential Delayed 1", 7);
    c.bench_with_input(id, &input, |b, input| {
        b.iter(|| {
            black_box(input);
            let square = input.square();
            babybear_quad_mul(square, square, square, *input)
        });
    });
}

fn exp_seven_delayed_2(c: &mut Criterion) {
    let mut rng = rand::thread_rng();
    let input = rng.gen::<F>();
    let id = BenchmarkId::new("BabyBear MultiExponential Delayed 2", 7);
    c.bench_with_input(id, &input, |b, input| {
        b.iter(|| {
            black_box(input);
            let cube = babybear_triple_mul(*input, *input, *input);
            babybear_triple_mul(cube, cube, *input)
        });
    });
}

criterion_group!(baby_bear_arithmetic, bench_mul);
criterion_main!(baby_bear_arithmetic);
