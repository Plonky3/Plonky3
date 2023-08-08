use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use p3_field::{AbstractField, Field, PrimeField32};
use p3_mersenne_31::Mersenne31;
use rand::Rng;
type F = Mersenne31;

pub fn euclidean_inverse(mut a: u32, mut b: u32) -> Option<u32> {
    if a == 0 {
        return None;
    }
    if a == 1 {
        return Some(1);
    }
    let (mut x, mut y) = (1_i32, 0_i32);
    while b != 0 {
        let (q, r) = (a / b, a % b);
        (a, b) = (b, r);
        (x, y) = (y, x - (q as i32) * y);
    }
    if a == 1 {
        if x < 0 {
            Some(b - (-x as u32))
        } else {
            Some(x as u32)
        }
    } else {
        None
    }
}
fn shift_inverse(x: F) -> Option<F> {
    // Uses algorithm 9.4.5 in Crandall and Pomerance book
    // "Prime Numbers: A Computational Perspective" to compute the inverse.
    if x.is_zero() {
        return None;
    }
    let mut a = F::ONE;
    let mut b = F::ZERO;
    let mut u = x.as_canonical_u32();
    let mut v = F::ORDER_U32;
    loop {
        // Shift off trailing zeros
        let e = u.trailing_zeros() as u64;
        u >>= e;
        // Circular shift
        a = a.mul_2exp_u64(31 - e);
        if u == 1 {
            return Some(a);
        }
        (a, b, u, v) = (a + b, a, u + v, u);
    }
}
fn fermat_inverse_basic(x: F) -> Option<F> {
    if x.is_zero() {
        return None;
    }
    Some(x.exp_u64(F::ORDER_U32 as u64 - 2))
}
fn fermat_inverse_2(x: F) -> Option<F> {
    if x.is_zero() {
        return None;
    }
    let p2 = x.square() * x;
    let p4 = p2.exp_power_of_2(2) * p2;
    let p8 = p4.exp_power_of_2(4) * p4;
    let p16 = p8.exp_power_of_2(8) * p8;
    let p24 = p16.exp_power_of_2(8) * p8;
    let p28 = p24.exp_power_of_2(4) * p4;
    let p29 = p28.exp_power_of_2(1) * x;
    let p29_1 = p29.exp_power_of_2(2) * x;
    Some(p29_1)
}
fn fermat_inverse_3(x: F) -> Option<F> {
    if x.is_zero() {
        return None;
    }
    let p2 = x.square() * x;
    let p3 = p2.square() * x;
    let p6 = p3.exp_power_of_2(3) * p3;
    let p12 = p6.exp_power_of_2(6) * p6;
    let p24 = p12.exp_power_of_2(12) * p12;
    let p27 = p24.exp_power_of_2(3) * p3;
    let p29 = p27.exp_power_of_2(2) * p2;
    let p29_1 = p29.exp_power_of_2(2) * x;
    Some(p29_1)
}

fn try_inverse(c: &mut Criterion) {
    c.bench_function("euclidean_inverse", |b| {
        b.iter_batched(
            || {
                let mut rng = rand::thread_rng();
                rng.gen::<F>().as_canonical_u32()
            },
            |x| euclidean_inverse(x, F::ORDER_U32),
            BatchSize::SmallInput,
        )
    });
    c.bench_function("shift_inverse", |b| {
        b.iter_batched(
            || {
                let mut rng = rand::thread_rng();
                rng.gen::<F>()
            },
            shift_inverse,
            BatchSize::SmallInput,
        )
    });
    c.bench_function("fermat_inverse_basic", |b| {
        b.iter_batched(
            || {
                let mut rng = rand::thread_rng();
                rng.gen::<F>()
            },
            fermat_inverse_basic,
            BatchSize::SmallInput,
        )
    });
    c.bench_function("fermat_inverse_2", |b| {
        b.iter_batched(
            || {
                let mut rng = rand::thread_rng();
                rng.gen::<F>()
            },
            fermat_inverse_2,
            BatchSize::SmallInput,
        )
    });
    c.bench_function("fermat_inverse_3", |b| {
        b.iter_batched(
            || {
                let mut rng = rand::thread_rng();
                rng.gen::<F>()
            },
            fermat_inverse_3,
            BatchSize::SmallInput,
        )
    });
}
criterion_group!(mersenne31_arithmetics, try_inverse);
criterion_main!(mersenne31_arithmetics);
