//! Utilities for testing field implementations.

#![no_std]

extern crate alloc;

pub mod bench_func;

pub use bench_func::*;
use p3_field::{
    cyclic_subgroup_coset_known_order, cyclic_subgroup_known_order, two_adic_coset_zerofier,
    two_adic_subgroup_zerofier, Field, TwoAdicField,
};
use rand::distributions::{Distribution, Standard};
use rand::Rng;

#[allow(clippy::eq_op)]
pub fn test_add_neg_sub_mul<F: Field>()
where
    Standard: Distribution<F>,
{
    let mut rng = rand::thread_rng();
    let x = rng.gen::<F>();
    let y = rng.gen::<F>();
    let z = rng.gen::<F>();
    assert_eq!(x + (-x), F::ZERO);
    assert_eq!(-x, F::ZERO - x);
    assert_eq!(x + x, x * F::TWO);
    assert_eq!(x * (-x), -x.square());
    assert_eq!(x + y, y + x);
    assert_eq!(x * y, y * x);
    assert_eq!(x * (y * z), (x * y) * z);
    assert_eq!(x - (y + z), (x - y) - z);
    assert_eq!((x + y) - z, x + (y - z));
    assert_eq!(x * (y + z), x * y + x * z);
}

pub fn test_inv_div<F: Field>()
where
    Standard: Distribution<F>,
{
    let mut rng = rand::thread_rng();
    let x = rng.gen::<F>();
    let y = rng.gen::<F>();
    let z = rng.gen::<F>();
    assert_eq!(x * x.inverse(), F::ONE);
    assert_eq!(x.inverse() * x, F::ONE);
    assert_eq!(x.square().inverse(), x.inverse().square());
    assert_eq!((x / y) * y, x);
    assert_eq!(x / (y * z), (x / y) / z);
    assert_eq!((x * y) / z, x * (y / z));
}

pub fn test_inverse<F: Field>()
where
    Standard: Distribution<F>,
{
    assert_eq!(None, F::ZERO.try_inverse());

    assert_eq!(Some(F::ONE), F::ONE.try_inverse());

    let mut rng = rand::thread_rng();
    for _ in 0..1000 {
        let x = rng.gen::<F>();
        if !x.is_zero() && !x.is_one() {
            let z = x.inverse();
            assert_ne!(x, z);
            assert_eq!(x * z, F::ONE);
        }
    }
}

pub fn test_two_adic_subgroup_zerofier<F: TwoAdicField>() {
    for log_n in 0..5 {
        let g = F::two_adic_generator(log_n);
        for x in cyclic_subgroup_known_order(g, 1 << log_n) {
            let zerofier_eval = two_adic_subgroup_zerofier(log_n, x);
            assert_eq!(zerofier_eval, F::ZERO);
        }
    }
}

pub fn test_two_adic_coset_zerofier<F: TwoAdicField>() {
    for log_n in 0..5 {
        let g = F::two_adic_generator(log_n);
        let shift = F::generator();
        for x in cyclic_subgroup_coset_known_order(g, shift, 1 << log_n) {
            let zerofier_eval = two_adic_coset_zerofier(log_n, shift, x);
            assert_eq!(zerofier_eval, F::ZERO);
        }
    }
}

#[macro_export]
macro_rules! test_field {
    ($field:ty) => {
        mod field_tests {
            #[test]
            fn test_add_neg_sub_mul() {
                $crate::test_add_neg_sub_mul::<$field>();
            }
            #[test]
            fn test_inv_div() {
                $crate::test_inv_div::<$field>();
            }
            #[test]
            fn test_inverse() {
                $crate::test_inverse::<$field>();
            }
        }
    };
}

#[macro_export]
macro_rules! test_two_adic_field {
    ($field:ty) => {
        mod two_adic_field_tests {
            #[test]
            fn test_two_adic_field_subgroup_zerofier() {
                $crate::test_two_adic_subgroup_zerofier::<$field>();
            }
            #[test]
            fn test_two_adic_coset_zerofier() {
                $crate::test_two_adic_coset_zerofier::<$field>();
            }
        }
    };
}
