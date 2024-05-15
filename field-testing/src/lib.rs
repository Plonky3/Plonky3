//! Utilities for testing field implementations.

#![no_std]

extern crate alloc;

pub mod bench_func;

pub use bench_func::*;
use num_bigint::BigUint;
use p3_field::{
    cyclic_subgroup_coset_known_order, cyclic_subgroup_known_order, two_adic_coset_zerofier,
    two_adic_subgroup_zerofier, AbstractField, Field, TwoAdicField,
};
use rand::distributions::{Distribution, Standard};
use rand::Rng;

#[allow(clippy::eq_op)]
pub fn test_add_neg_sub_mul<F: Field>()
where
    Standard: Distribution<F>,
{
    let mut rng = rand::thread_rng();
    for _ in 0..1000 {
        let [x, y, z] = rng.gen::<[F; 3]>();
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
}

pub fn test_inv_div<F: Field>()
where
    Standard: Distribution<F>,
{
    let mut rng = rand::thread_rng();
    for _ in 0..1000 {
        let [x, y, z] = rng.gen::<[F; 3]>();
        assert_eq!(x * x.inverse(), F::one());
        assert_eq!(x.inverse() * x, F::one());
        assert_eq!(x.square().inverse(), x.inverse().square());
        assert_eq!((x / y) * y, x);
        assert_eq!(x / (y * z), (x / y) / z);
        assert_eq!((x * y) / z, x * (y / z));
    }
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

pub fn test_multiplicative_group_order<F: Field>() {
    // Check that product of factors actually matches p - 1.
    let product: BigUint = F::multiplicative_group_factors()
        .into_iter()
        .map(|(factor, exponent)| factor.pow(exponent as u32))
        .product();
    assert_eq!(product + 1u32, F::order());

    // Check that g^(p-1) = 1.
    // (this is true for any element but we might as well check)
    assert_eq!(F::generator().exp_biguint(F::order() - 1u32), F::one());

    // Check that g does not actually generate some smaller subgroup.
    // Say we have p - 1 = a0^n0 * a1^n1 * ...
    // And g generates a subgroup of order a0^m0 * a1^m1 * ...
    // If all m_i = n_i, we are good
    // If g doesn't generate the full subgroup, it will have some m_i < n_i.
    // So for each i, we can test g^((p-1)/a_i), equivalent to setting m_i = n_i - 1,
    // and the result will be 1 iff m_i < n_i.
    for (factor, _) in F::multiplicative_group_factors() {
        assert_ne!(
            F::generator().exp_biguint((F::order() - 1u32) / factor.clone()),
            F::one(),
            "Generator does not generate the full multiplicative subgroup; missing a subgroup for {factor}",
        );
    }
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

/*
pub fn test_ef_two_adic_generator_consistency<
    F: TwoAdicField,
    A: A
    EF: TwoAdicField + ExtensionField<F>,
>() {
    assert_eq!(
        EF::from_base(F::two_adic_generator(F::TWO_ADICITY)),
        EF::two_adic_generator(F::TWO_ADICITY)
    );
}
*/

#[macro_export]
macro_rules! test_field {
    ($field:ty) => {
        mod field_tests {
            use super::*;
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
            fn test_multiplicative_group_order() {
                $crate::test_multiplicative_group_order::<$field>();
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
