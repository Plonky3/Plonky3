//! Utilities for testing field implementations.

#![no_std]

extern crate alloc;

pub mod bench_func;
pub mod dft_testing;
pub mod packedfield_testing;

pub use bench_func::*;
pub use dft_testing::*;
use num_bigint::BigUint;
use num_traits::identities::One;
use p3_field::{
    cyclic_subgroup_coset_known_order, cyclic_subgroup_known_order, two_adic_coset_zerofier,
    two_adic_subgroup_zerofier, ExtensionField, Field, TwoAdicField,
};
pub use packedfield_testing::*;
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
    assert_eq!(x + (-x), F::zero());
    assert_eq!(-x, F::zero() - x);
    assert_eq!(x + x, x * F::two());
    assert_eq!(x, x.halve() * F::two());
    assert_eq!(x * (-x), -x.square());
    assert_eq!(x + y, y + x);
    assert_eq!(x * y, y * x);
    assert_eq!(x * (y * z), (x * y) * z);
    assert_eq!(x - (y + z), (x - y) - z);
    assert_eq!((x + y) - z, x + (y - z));
    assert_eq!(x * (y + z), x * y + x * z);
    assert_eq!(
        x + y + z + x + y + z,
        [x, x, y, y, z, z].iter().cloned().sum()
    );
}

pub fn test_inv_div<F: Field>()
where
    Standard: Distribution<F>,
{
    let mut rng = rand::thread_rng();
    let x = rng.gen::<F>();
    let y = rng.gen::<F>();
    let z = rng.gen::<F>();
    assert_eq!(x * x.inverse(), F::one());
    assert_eq!(x.inverse() * x, F::one());
    assert_eq!(x.square().inverse(), x.inverse().square());
    assert_eq!((x / y) * y, x);
    assert_eq!(x / (y * z), (x / y) / z);
    assert_eq!((x * y) / z, x * (y / z));
}

pub fn test_inverse<F: Field>()
where
    Standard: Distribution<F>,
{
    assert_eq!(None, F::zero().try_inverse());

    assert_eq!(Some(F::one()), F::one().try_inverse());

    let mut rng = rand::thread_rng();
    for _ in 0..1000 {
        let x = rng.gen::<F>();
        if !x.is_zero() && !x.is_one() {
            let z = x.inverse();
            assert_ne!(x, z);
            assert_eq!(x * z, F::one());
        }
    }
}

pub fn test_multiplicative_group_factors<F: Field>() {
    let product: BigUint = F::multiplicative_group_factors()
        .into_iter()
        .map(|(factor, exponent)| factor.pow(exponent as u32))
        .product();
    assert_eq!(product + BigUint::one(), F::order());
}

pub fn test_two_adic_subgroup_zerofier<F: TwoAdicField>() {
    for log_n in 0..5 {
        let g = F::two_adic_generator(log_n);
        for x in cyclic_subgroup_known_order(g, 1 << log_n) {
            let zerofier_eval = two_adic_subgroup_zerofier(log_n, x);
            assert_eq!(zerofier_eval, F::zero());
        }
    }
}

pub fn test_two_adic_coset_zerofier<F: TwoAdicField>() {
    for log_n in 0..5 {
        let g = F::two_adic_generator(log_n);
        let shift = F::generator();
        for x in cyclic_subgroup_coset_known_order(g, shift, 1 << log_n) {
            let zerofier_eval = two_adic_coset_zerofier(log_n, shift, x);
            assert_eq!(zerofier_eval, F::zero());
        }
    }
}

pub fn test_two_adic_generator_consistency<F: TwoAdicField>() {
    let log_n = F::TWO_ADICITY;
    let g = F::two_adic_generator(log_n);
    for bits in 0..=log_n {
        assert_eq!(g.exp_power_of_2(bits), F::two_adic_generator(log_n - bits));
    }
}

pub fn test_ef_two_adic_generator_consistency<
    F: TwoAdicField,
    EF: TwoAdicField + ExtensionField<F>,
>() {
    assert_eq!(
        EF::from_base(F::two_adic_generator(F::TWO_ADICITY)),
        EF::two_adic_generator(F::TWO_ADICITY)
    );
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
            #[test]
            fn test_multiplicative_group_factors() {
                $crate::test_multiplicative_group_factors::<$field>();
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
            #[test]
            fn test_two_adic_consisitency() {
                $crate::test_two_adic_generator_consistency::<$field>();
            }
        }
    };
}

#[macro_export]
macro_rules! test_two_adic_extension_field {
    ($field:ty, $ef:ty) => {
        use $crate::test_two_adic_field;

        test_two_adic_field!($ef);

        mod two_adic_extension_field_tests {

            #[test]
            fn test_ef_two_adic_generator_consistency() {
                $crate::test_ef_two_adic_generator_consistency::<$field, $ef>();
            }
        }
    };
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use p3_baby_bear::BabyBear;
    use p3_field::extension::{BinomialExtensionField, HasFrobenius};
    use p3_field::{binomial_expand, eval_poly, AbstractExtensionField, AbstractField};
    use rand::random;

    use super::*;

    #[test]
    fn test_minimal_poly() {
        type F = BabyBear;
        type EF = BinomialExtensionField<F, 4>;
        for _ in 0..1024 {
            let x: EF = random();
            let m: Vec<EF> = x.minimal_poly().into_iter().map(EF::from_base).collect();
            assert!(eval_poly(&m, x).is_zero());
        }
    }

    #[test]
    fn test_binomial_expand() {
        type F = BabyBear;
        // (x - 1)(x - 2) = x^2 - 3x + 2
        assert_eq!(
            binomial_expand(&[F::one(), F::two()]),
            vec![F::two(), -F::from_canonical_usize(3), F::one()]
        );
    }
}
