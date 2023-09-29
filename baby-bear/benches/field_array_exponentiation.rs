use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use p3_baby_bear::BabyBear;
use p3_field::{AbstractField, FieldArray};
// use p3_field::{AbstractField};

use rand::Rng;

type F = BabyBear;
// const ZERO: F = BabyBear{ value: 0 };

fn field_array_exp(c: &mut Criterion) {
    c.bench_function("field_array_exp", |b| {
        b.iter_batched(
            || {
                let mut rng = rand::thread_rng();
                // rng.gen::<F>()
                let mut vec = Vec::new();
                for _ in 0..8 {
                    vec.push(rng.gen::<F>())
                }
                FieldArray::<BabyBear, 8>(vec.try_into().unwrap())
                // vec
            },
            |mut x| {
                for _ in 0..100 {
                    x = x.exp_const_u64::<7>();
                }
                x
            },
            // |x| {
            //     let mut acc = ZERO;
            //     for y in x {acc += y.exp_const_u64::<7>();}
            //     if acc == ZERO {
            //         println!("{:?}", acc);
            //     }
            // },
            // |x| x.exp_const_u64::<7>(),
            BatchSize::SmallInput,
        )
    });
}

criterion_group!(baby_bear_arithmetic, field_array_exp);
criterion_main!(baby_bear_arithmetic);
