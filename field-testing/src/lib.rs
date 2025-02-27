//! Utilities for testing field implementations.

#![no_std]

extern crate alloc;

pub mod bench_func;
pub mod dft_testing;
pub mod from_integer_tests;
pub mod packedfield_testing;

pub use bench_func::*;
pub use dft_testing::*;
use num_bigint::BigUint;
use num_traits::identities::One;
use p3_field::{
    ExtensionField, Field, PrimeCharacteristicRing, TwoAdicField,
    cyclic_subgroup_coset_known_order, cyclic_subgroup_known_order,
    two_adic_coset_vanishing_polynomial, two_adic_subgroup_vanishing_polynomial,
};
pub use packedfield_testing::*;
use rand::Rng;
use rand::distr::{Distribution, StandardUniform};

#[allow(clippy::eq_op)]
pub fn test_add_neg_sub_mul<F: Field>()
where
    StandardUniform: Distribution<F>,
{
    let mut rng = rand::rng();
    let x = rng.random::<F>();
    let y = rng.random::<F>();
    let z = rng.random::<F>();
    assert_eq!(F::ONE + F::NEG_ONE, F::ZERO);
    assert_eq!(x + (-x), F::ZERO);
    assert_eq!(F::ONE + F::ONE, F::TWO);
    assert_eq!(-x, F::ZERO - x);
    assert_eq!(x + x, x * F::TWO);
    assert_eq!(x * F::TWO, x.double());
    assert_eq!(x, x.halve() * F::TWO);
    assert_eq!(x * (-x), -x.square());
    assert_eq!(x + y, y + x);
    assert_eq!(x * F::ZERO, F::ZERO);
    assert_eq!(x * F::ONE, x);
    assert_eq!(x * y, y * x);
    assert_eq!(x * (y * z), (x * y) * z);
    assert_eq!(x - (y + z), (x - y) - z);
    assert_eq!((x + y) - z, x + (y - z));
    assert_eq!(x * (y + z), x * y + x * z);
    assert_eq!(x + y + z + x + y + z, [x, x, y, y, z, z].into_iter().sum());
    test_tree_sum(&[x, y, z, x, y, z, x, y, z, x, y, z, x, y, z, x]);
}

pub fn test_inv_div<F: Field>()
where
    StandardUniform: Distribution<F>,
{
    let mut rng = rand::rng();
    let x = rng.random::<F>();
    let y = rng.random::<F>();
    let z = rng.random::<F>();
    assert_eq!(x * x.inverse(), F::ONE);
    assert_eq!(x.inverse() * x, F::ONE);
    assert_eq!(x.square().inverse(), x.inverse().square());
    assert_eq!((x / y) * y, x);
    assert_eq!(x / (y * z), (x / y) / z);
    assert_eq!((x * y) / z, x * (y / z));
}

pub fn test_inverse<F: Field>()
where
    StandardUniform: Distribution<F>,
{
    assert_eq!(None, F::ZERO.try_inverse());

    assert_eq!(Some(F::ONE), F::ONE.try_inverse());

    let mut rng = rand::rng();
    for _ in 0..1000 {
        let x = rng.random::<F>();
        if !x.is_zero() && !x.is_one() {
            let z = x.inverse();
            assert_ne!(x, z);
            assert_eq!(x * z, F::ONE);
        }
    }
}

pub fn test_dot_product<const N: usize, R: PrimeCharacteristicRing + Eq + Copy>(
    u: &[R; N],
    v: &[R; N],
) {
    let output = R::dot_product(u, v);
    let expected = u.iter().zip(v).map(|(&x, &y)| x * y).sum();
    assert_eq!(output, expected)
}

pub fn test_tree_sum<R: PrimeCharacteristicRing + Eq + Copy>(u: &[R; 16]) {
    let mut sum = R::ZERO;
    assert_eq!(sum, R::tree_sum::<0>(u[..0].try_into().unwrap()));
    sum += u[0];
    assert_eq!(sum, R::tree_sum::<1>(u[..1].try_into().unwrap()));
    sum += u[1];
    assert_eq!(sum, R::tree_sum::<2>(u[..2].try_into().unwrap()));
    sum += u[2];
    assert_eq!(sum, R::tree_sum::<3>(u[..3].try_into().unwrap()));
    sum += u[3];
    assert_eq!(sum, R::tree_sum::<4>(u[..4].try_into().unwrap()));
    sum += u[4];
    assert_eq!(sum, R::tree_sum::<5>(u[..5].try_into().unwrap()));
    sum += u[5];
    assert_eq!(sum, R::tree_sum::<6>(u[..6].try_into().unwrap()));
    sum += u[6];
    assert_eq!(sum, R::tree_sum::<7>(u[..7].try_into().unwrap()));
    sum += u[7];
    assert_eq!(sum, R::tree_sum::<8>(u[..8].try_into().unwrap()));
    sum += u[8];
    assert_eq!(sum, R::tree_sum::<9>(u[..9].try_into().unwrap()));
    sum += u[9];
    assert_eq!(sum, R::tree_sum::<10>(u[..10].try_into().unwrap()));
    sum += u[10];
    assert_eq!(sum, R::tree_sum::<11>(u[..11].try_into().unwrap()));
    sum += u[11];
    assert_eq!(sum, R::tree_sum::<12>(u[..12].try_into().unwrap()));
    sum += u[12];
    assert_eq!(sum, R::tree_sum::<13>(u[..13].try_into().unwrap()));
    sum += u[13];
    assert_eq!(sum, R::tree_sum::<14>(u[..14].try_into().unwrap()));
    sum += u[14];
    assert_eq!(sum, R::tree_sum::<15>(u[..15].try_into().unwrap()));
    sum += u[15];
    assert_eq!(sum, R::tree_sum::<16>(u));
}

pub fn test_multiplicative_group_factors<F: Field>() {
    let product: BigUint = F::multiplicative_group_factors()
        .into_iter()
        .map(|(factor, exponent)| factor.pow(exponent as u32))
        .product();
    assert_eq!(product + BigUint::one(), F::order());
}

pub fn test_two_adic_subgroup_vanishing_polynomial<F: TwoAdicField>() {
    for log_n in 0..5 {
        let g = F::two_adic_generator(log_n);
        for x in cyclic_subgroup_known_order(g, 1 << log_n) {
            let vanishing_polynomial_eval = two_adic_subgroup_vanishing_polynomial(log_n, x);
            assert_eq!(vanishing_polynomial_eval, F::ZERO);
        }
    }
}

pub fn test_two_adic_coset_vanishing_polynomial<F: TwoAdicField>() {
    for log_n in 0..5 {
        let g = F::two_adic_generator(log_n);
        let shift = F::GENERATOR;
        for x in cyclic_subgroup_coset_known_order(g, shift, 1 << log_n) {
            let vanishing_polynomial_eval = two_adic_coset_vanishing_polynomial(log_n, shift, x);
            assert_eq!(vanishing_polynomial_eval, F::ZERO);
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
        Into::<EF>::into(F::two_adic_generator(F::TWO_ADICITY)),
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
macro_rules! test_prime_field {
    ($field:ty) => {
        mod from_integer_small_tests {
            use p3_field::integers::QuotientMap;
            use p3_field::{Field, PrimeCharacteristicRing};

            #[test]
            fn test_small_integer_conversions() {
                $crate::generate_from_small_int_tests!(
                    $field,
                    [
                        u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize
                    ]
                );
            }

            #[test]
            fn test_small_signed_integer_conversions() {
                $crate::generate_from_small_neg_int_tests!(
                    $field,
                    [i8, i16, i32, i64, i128, isize]
                );
            }
        }
    };
}

#[macro_export]
macro_rules! test_prime_field_64 {
    ($field:ty) => {
        mod from_integer_tests_prime_field_64 {
            use p3_field::integers::QuotientMap;
            use p3_field::{Field, PrimeCharacteristicRing, PrimeField64};
            use rand::Rng;

            #[test]
            fn test_as_canonical_u64() {
                let mut rng = rand::rng();
                let x: u64 = rng.random();
                let x_mod_order = x % <$field>::ORDER_U64;

                assert_eq!(<$field>::ZERO.as_canonical_u64(), 0);
                assert_eq!(<$field>::ONE.as_canonical_u64(), 1);
                assert_eq!(<$field>::TWO.as_canonical_u64(), 2 % <$field>::ORDER_U64);
                assert_eq!(
                    <$field>::NEG_ONE.as_canonical_u64(),
                    <$field>::ORDER_U64 - 1
                );

                assert_eq!(
                    <$field>::from_int(<$field>::ORDER_U64).as_canonical_u64(),
                    0
                );
                assert_eq!(<$field>::from_int(x).as_canonical_u64(), x_mod_order);
                assert_eq!(
                    unsafe { <$field>::from_canonical_unchecked(x_mod_order).as_canonical_u64() },
                    x_mod_order
                );
            }

            #[test]
            fn test_as_unique_u64() {
                assert_ne!(
                    <$field>::ZERO.to_unique_u64(),
                    <$field>::ONE.to_unique_u64()
                );
                assert_ne!(
                    <$field>::ZERO.to_unique_u64(),
                    <$field>::NEG_ONE.to_unique_u64()
                );
                assert_eq!(
                    <$field>::from_int(<$field>::ORDER_U64).to_unique_u64(),
                    <$field>::ZERO.to_unique_u64()
                );
            }

            #[test]
            fn test_large_unsigned_integer_conversions() {
                $crate::generate_from_large_u_int_tests!($field, <$field>::ORDER_U64, [u64, u128]);
            }

            #[test]
            fn test_large_signed_integer_conversions() {
                $crate::generate_from_large_i_int_tests!($field, <$field>::ORDER_U64, [i64, i128]);
            }
        }
    };
}

#[macro_export]
macro_rules! test_prime_field_32 {
    ($field:ty) => {
        mod from_integer_tests_prime_field_32 {
            use p3_field::integers::QuotientMap;
            use p3_field::{Field, PrimeCharacteristicRing, PrimeField32};
            use rand::Rng;

            #[test]
            fn test_as_canonical_u32() {
                let mut rng = rand::rng();
                let x: u32 = rng.random();
                let x_mod_order = x % <$field>::ORDER_U32;

                assert_eq!(<$field>::ZERO.as_canonical_u32(), 0);
                assert_eq!(<$field>::ONE.as_canonical_u32(), 1);
                assert_eq!(<$field>::TWO.as_canonical_u32(), 2 % <$field>::ORDER_U32);
                assert_eq!(
                    <$field>::NEG_ONE.as_canonical_u32(),
                    <$field>::ORDER_U32 - 1
                );
                assert_eq!(
                    <$field>::from_int(<$field>::ORDER_U32).as_canonical_u32(),
                    0
                );
                assert_eq!(<$field>::from_int(x).as_canonical_u32(), x_mod_order);
                assert_eq!(
                    unsafe { <$field>::from_canonical_unchecked(x_mod_order).as_canonical_u32() },
                    x_mod_order
                );
            }

            #[test]
            fn test_as_unique_u32() {
                assert_ne!(
                    <$field>::ZERO.to_unique_u32(),
                    <$field>::ONE.to_unique_u32()
                );
                assert_ne!(
                    <$field>::ZERO.to_unique_u32(),
                    <$field>::NEG_ONE.to_unique_u32()
                );
                assert_eq!(
                    <$field>::from_int(<$field>::ORDER_U32).to_unique_u32(),
                    <$field>::ZERO.to_unique_u32()
                );
            }

            #[test]
            fn test_large_unsigned_integer_conversions() {
                $crate::generate_from_large_u_int_tests!(
                    $field,
                    <$field>::ORDER_U32,
                    [u32, u64, u128]
                );
            }

            #[test]
            fn test_large_signed_integer_conversions() {
                $crate::generate_from_large_i_int_tests!(
                    $field,
                    <$field>::ORDER_U32,
                    [i32, i64, i128]
                );
            }
        }
    };
}

#[macro_export]
macro_rules! test_two_adic_field {
    ($field:ty) => {
        mod two_adic_field_tests {
            #[test]
            fn test_two_adic_field_subgroup_vanishing_polynomial() {
                $crate::test_two_adic_subgroup_vanishing_polynomial::<$field>();
            }
            #[test]
            fn test_two_adic_coset_vanishing_polynomial() {
                $crate::test_two_adic_coset_vanishing_polynomial::<$field>();
            }
            #[test]
            fn test_two_adic_consistency() {
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
    use p3_field::{PrimeCharacteristicRing, binomial_expand, eval_poly};
    use rand::random;

    use super::*;

    #[test]
    fn test_minimal_poly() {
        type F = BabyBear;
        type EF = BinomialExtensionField<F, 4>;
        for _ in 0..1024 {
            let x: EF = random();
            let m: Vec<EF> = x.minimal_poly().into_iter().map(Into::<EF>::into).collect();
            assert!(eval_poly(&m, x).is_zero());
        }
    }

    #[test]
    fn test_binomial_expand() {
        type F = BabyBear;
        // (x - 1)(x - 2) = x^2 - 3x + 2
        assert_eq!(
            binomial_expand(&[F::ONE, F::TWO]),
            vec![F::TWO, -F::from_u8(3), F::ONE]
        );
    }
}
