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
pub fn test_ring_with_eq<R: PrimeCharacteristicRing + Copy + Eq>(zeros: &[R], ones: &[R])
where
    StandardUniform: Distribution<R> + Distribution<[R; 16]>,
{
    // zeros should be a vector containing differenent representatives of `R::ZERO`.
    // ones should be a vector containing differenent representatives of `R::ONE`.
    let mut rng = rand::rng();
    let x = rng.random::<R>();
    let y = rng.random::<R>();
    let z = rng.random::<R>();
    assert_eq!(R::ONE + R::NEG_ONE, R::ZERO, "Error 1 + (-1) =/= 0");
    assert_eq!(R::NEG_ONE + R::TWO, R::ONE, "Error -1 + 2 =/= 1");
    assert_eq!(x + (-x), R::ZERO, "Error x + (-x) =/= 0");
    assert_eq!(R::ONE + R::ONE, R::TWO, "Error 1 + 1 =/= 2");
    assert_eq!(-(-x), x, "Error when testing double negation");
    assert_eq!(x + x, x * R::TWO, "Error when comparing x * 2 to x + x");
    assert_eq!(
        x * R::TWO,
        x.double(),
        "Error when comparing x.double() to x * 2"
    );

    // Check different representatives of Zero.
    for zero in zeros.iter().copied() {
        assert_eq!(zero, R::ZERO);
        assert_eq!(x + zero, x, "Error when testing additive identity right.");
        assert_eq!(zero + x, x, "Error when testing additive identity left.");
        assert_eq!(x - zero, x, "Error when testing subtracting zero.");
        assert_eq!(zero - x, -x, "Error when testing subtracting  from zero.");
        assert_eq!(
            x * zero,
            zero,
            "Error when testing right multiplication by 0."
        );
        assert_eq!(
            zero * x,
            zero,
            "Error when testing left multiplication by 0."
        );
    }

    // Check different representatives of One.
    for one in ones.iter().copied() {
        assert_eq!(one, R::ONE);
        assert_eq!(
            x * one,
            x,
            "Error when testing multaplicative identity right."
        );
        assert_eq!(
            one * x,
            x,
            "Error when testing multaplicative identity left."
        );
    }

    assert_eq!(
        x * R::NEG_ONE,
        -x,
        "Error when testing right multiplication by -1."
    );
    assert_eq!(
        R::NEG_ONE * x,
        -x,
        "Error when testing left multiplication by -1."
    );
    assert_eq!(x * x, x.square(), "Error when testing x * x = x.square()");
    assert_eq!(
        x * x * x,
        x.cube(),
        "Error when testing x * x * x = x.cube()"
    );
    assert_eq!(x + y, y + x, "Error when testing commutativity of addition");
    assert_eq!(
        (x - y),
        -(y - x),
        "Error when testing anticommutativity of sub."
    );
    assert_eq!(
        x * y,
        y * x,
        "Error when testing commutativity of multiplication."
    );
    assert_eq!(
        x + (y + z),
        (x + y) + z,
        "Error when testing associativity of addition"
    );
    assert_eq!(
        x * (y * z),
        (x * y) * z,
        "Error when testing associativity of multiplication."
    );
    assert_eq!(
        x - (y - z),
        (x - y) + z,
        "Error when testing subtraction and addition"
    );
    assert_eq!(
        x - (y + z),
        (x - y) - z,
        "Error when testing subtraction and addition"
    );
    assert_eq!(
        (x + y) - z,
        x + (y - z),
        "Error when testing subtraction and addition"
    );
    assert_eq!(
        x * (-y),
        -(x * y),
        "Error when testing distributivity of mul and right neg."
    );
    assert_eq!(
        (-x) * y,
        -(x * y),
        "Error when testing distributivity of mul and left neg."
    );

    assert_eq!(
        x * (y + z),
        x * y + x * z,
        "Error when testing distributivity of add and left mul."
    );
    assert_eq!(
        (x + y) * z,
        x * z + y * z,
        "Error when testing distributivity of add and right mul."
    );
    assert_eq!(
        x * (y - z),
        x * y - x * z,
        "Error when testing distributivity of sub and left mul."
    );
    assert_eq!(
        (x - y) * z,
        x * z - y * z,
        "Error when testing distributivity of sub and right mul."
    );

    let vec1: [R; 64] = rng.random();
    let vec2: [R; 64] = rng.random();
    test_sums(&vec1[..16].try_into().unwrap());
    test_dot_product(&vec1, &vec2);

    assert_eq!(
        x.exp_const_u64::<0>(),
        R::ONE,
        "Error when comparing x.exp_const_u64::<0> to R::ONE."
    );
    assert_eq!(
        x.exp_const_u64::<1>(),
        x,
        "Error when comparing x.exp_const_u64::<3> to x."
    );
    assert_eq!(
        x.exp_const_u64::<2>(),
        x * x,
        "Error when comparing x.exp_const_u64::<3> to x*x."
    );
    assert_eq!(
        x.exp_const_u64::<3>(),
        x * x * x,
        "Error when comparing x.exp_const_u64::<3> to x*x*x."
    );
    assert_eq!(
        x.exp_const_u64::<4>(),
        x * x * x * x,
        "Error when comparing x.exp_const_u64::<3> to x*x*x*x."
    );
    assert_eq!(
        x.exp_const_u64::<5>(),
        x * x * x * x * x,
        "Error when comparing x.exp_const_u64::<5> to x*x*x*x*x."
    );
    assert_eq!(
        x.exp_const_u64::<6>(),
        x * x * x * x * x * x,
        "Error when comparing x.exp_const_u64::<7> to x*x*x*x*x*x."
    );
    assert_eq!(
        x.exp_const_u64::<7>(),
        x * x * x * x * x * x * x,
        "Error when comparing x.exp_const_u64::<7> to x*x*x*x*x*x*x."
    );

    test_binary_ops(zeros, ones, x, y, z);
}

pub fn test_inv_div<F: Field>()
where
    StandardUniform: Distribution<F>,
{
    let mut rng = rand::rng();
    let x = rng.random::<F>();
    let y = rng.random::<F>();
    let z = rng.random::<F>();
    assert_eq!(x, x.halve() * F::TWO);
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

pub fn test_dot_product<R: PrimeCharacteristicRing + Eq + Copy>(u: &[R; 64], v: &[R; 64]) {
    let mut dot = R::ZERO;
    assert_eq!(
        dot,
        R::dot_product::<0>(u[..0].try_into().unwrap(), v[..0].try_into().unwrap())
    );
    dot += u[0] * v[0];
    assert_eq!(
        dot,
        R::dot_product::<1>(u[..1].try_into().unwrap(), v[..1].try_into().unwrap())
    );
    dot += u[1] * v[1];
    assert_eq!(
        dot,
        R::dot_product::<2>(u[..2].try_into().unwrap(), v[..2].try_into().unwrap())
    );
    dot += u[2] * v[2];
    assert_eq!(
        dot,
        R::dot_product::<3>(u[..3].try_into().unwrap(), v[..3].try_into().unwrap())
    );
    dot += u[3] * v[3];
    assert_eq!(
        dot,
        R::dot_product::<4>(u[..4].try_into().unwrap(), v[..4].try_into().unwrap())
    );
    dot += u[4] * v[4];
    assert_eq!(
        dot,
        R::dot_product::<5>(u[..5].try_into().unwrap(), v[..5].try_into().unwrap())
    );
    dot += u[5] * v[5];
    assert_eq!(
        dot,
        R::dot_product::<6>(u[..6].try_into().unwrap(), v[..6].try_into().unwrap())
    );
    dot += u[6] * v[6];
    assert_eq!(
        dot,
        R::dot_product::<7>(u[..7].try_into().unwrap(), v[..7].try_into().unwrap())
    );
    dot += u[7] * v[7];
    assert_eq!(
        dot,
        R::dot_product::<8>(u[..8].try_into().unwrap(), v[..8].try_into().unwrap())
    );
    dot += u[8] * v[8];
    assert_eq!(
        dot,
        R::dot_product::<9>(u[..9].try_into().unwrap(), v[..9].try_into().unwrap())
    );
    dot += u[9] * v[9];
    assert_eq!(
        dot,
        R::dot_product::<10>(u[..10].try_into().unwrap(), v[..10].try_into().unwrap())
    );
    dot += u[10] * v[10];
    assert_eq!(
        dot,
        R::dot_product::<11>(u[..11].try_into().unwrap(), v[..11].try_into().unwrap())
    );
    dot += u[11] * v[11];
    assert_eq!(
        dot,
        R::dot_product::<12>(u[..12].try_into().unwrap(), v[..12].try_into().unwrap())
    );
    dot += u[12] * v[12];
    assert_eq!(
        dot,
        R::dot_product::<13>(u[..13].try_into().unwrap(), v[..13].try_into().unwrap())
    );
    dot += u[13] * v[13];
    assert_eq!(
        dot,
        R::dot_product::<14>(u[..14].try_into().unwrap(), v[..14].try_into().unwrap())
    );
    dot += u[14] * v[14];
    assert_eq!(
        dot,
        R::dot_product::<15>(u[..15].try_into().unwrap(), v[..15].try_into().unwrap())
    );
    dot += u[15] * v[15];
    assert_eq!(
        dot,
        R::dot_product::<16>(u[..16].try_into().unwrap(), v[..16].try_into().unwrap())
    );

    let dot_64: R = u
        .iter()
        .zip(v.iter())
        .fold(R::ZERO, |acc, (&lhs, &rhs)| acc + (lhs * rhs));
    assert_eq!(dot_64, R::dot_product::<64>(u, v));
}

pub fn test_sums<R: PrimeCharacteristicRing + Eq + Copy>(u: &[R; 16]) {
    let mut sum = R::ZERO;
    assert_eq!(sum, R::tree_sum::<0>(u[..0].try_into().unwrap()));
    assert_eq!(sum, u[..0].iter().copied().sum());
    sum += u[0];
    assert_eq!(sum, R::tree_sum::<1>(u[..1].try_into().unwrap()));
    assert_eq!(sum, u[..1].iter().copied().sum());
    sum += u[1];
    assert_eq!(sum, R::tree_sum::<2>(u[..2].try_into().unwrap()));
    assert_eq!(sum, u[..2].iter().copied().sum());
    sum += u[2];
    assert_eq!(sum, R::tree_sum::<3>(u[..3].try_into().unwrap()));
    assert_eq!(sum, u[..3].iter().copied().sum());
    sum += u[3];
    assert_eq!(sum, R::tree_sum::<4>(u[..4].try_into().unwrap()));
    assert_eq!(sum, u[..4].iter().copied().sum());
    sum += u[4];
    assert_eq!(sum, R::tree_sum::<5>(u[..5].try_into().unwrap()));
    assert_eq!(sum, u[..5].iter().copied().sum());
    sum += u[5];
    assert_eq!(sum, R::tree_sum::<6>(u[..6].try_into().unwrap()));
    assert_eq!(sum, u[..6].iter().copied().sum());
    sum += u[6];
    assert_eq!(sum, R::tree_sum::<7>(u[..7].try_into().unwrap()));
    assert_eq!(sum, u[..7].iter().copied().sum());
    sum += u[7];
    assert_eq!(sum, R::tree_sum::<8>(u[..8].try_into().unwrap()));
    assert_eq!(sum, u[..8].iter().copied().sum());
    sum += u[8];
    assert_eq!(sum, R::tree_sum::<9>(u[..9].try_into().unwrap()));
    assert_eq!(sum, u[..9].iter().copied().sum());
    sum += u[9];
    assert_eq!(sum, R::tree_sum::<10>(u[..10].try_into().unwrap()));
    assert_eq!(sum, u[..10].iter().copied().sum());
    sum += u[10];
    assert_eq!(sum, R::tree_sum::<11>(u[..11].try_into().unwrap()));
    assert_eq!(sum, u[..11].iter().copied().sum());
    sum += u[11];
    assert_eq!(sum, R::tree_sum::<12>(u[..12].try_into().unwrap()));
    assert_eq!(sum, u[..12].iter().copied().sum());
    sum += u[12];
    assert_eq!(sum, R::tree_sum::<13>(u[..13].try_into().unwrap()));
    assert_eq!(sum, u[..13].iter().copied().sum());
    sum += u[13];
    assert_eq!(sum, R::tree_sum::<14>(u[..14].try_into().unwrap()));
    assert_eq!(sum, u[..14].iter().copied().sum());
    sum += u[14];
    assert_eq!(sum, R::tree_sum::<15>(u[..15].try_into().unwrap()));
    assert_eq!(sum, u[..15].iter().copied().sum());
    sum += u[15];
    assert_eq!(sum, R::tree_sum::<16>(u));
    assert_eq!(sum, u.iter().copied().sum());
}

pub fn test_binary_ops<R: PrimeCharacteristicRing + Eq + Copy>(
    zeros: &[R],
    ones: &[R],
    x: R,
    y: R,
    z: R,
) {
    for zero in zeros {
        for one in ones {
            assert_eq!(one.xor(one), R::ZERO, "Error when testing xor(1, 1) = 0.");
            assert_eq!(zero.xor(one), R::ONE, "Error when testing xor(0, 1) = 1.");
            assert_eq!(one.xor(zero), R::ONE, "Error when testing xor(1, 0) = 1.");
            assert_eq!(zero.xor(zero), R::ZERO, "Error when testing xor(0, 0) = 0.");
            assert_eq!(one.andn(one), R::ZERO, "Error when testing andn(1, 1) = 0.");
            assert_eq!(zero.andn(one), R::ONE, "Error when testing andn(0, 1) = 1.");
            assert_eq!(
                one.andn(zero),
                R::ZERO,
                "Error when testing andn(1, 0) = 0."
            );
            assert_eq!(
                zero.andn(zero),
                R::ZERO,
                "Error when testing andn(0, 0) = 0."
            );
            assert_eq!(
                zero.bool_check(),
                R::ZERO,
                "Error when testing bool_check(0) = 0."
            );
            assert_eq!(
                one.bool_check(),
                R::ZERO,
                "Error when testing bool_check(1) = 0."
            );
        }
    }

    assert_eq!(
        R::ONE.xor(&R::NEG_ONE),
        R::TWO,
        "Error when testing xor(1, -1) = 2."
    );
    assert_eq!(
        R::NEG_ONE.xor(&R::ONE),
        R::TWO,
        "Error when testing xor(-1, 1) = 2."
    );
    assert_eq!(
        R::NEG_ONE.xor(&R::NEG_ONE),
        R::from_i8(-4),
        "Error when testing xor(-1, -1) = -4."
    );
    assert_eq!(
        R::ONE.andn(&R::NEG_ONE),
        R::ZERO,
        "Error when testing andn(1, -1) = 0."
    );
    assert_eq!(
        R::NEG_ONE.andn(&R::ONE),
        R::TWO,
        "Error when testing andn(-1, 1) = 2."
    );
    assert_eq!(
        R::NEG_ONE.andn(&R::NEG_ONE),
        -R::TWO,
        "Error when testing andn(-1, -1) = -2."
    );

    assert_eq!(x.xor(&y), x + y - x * y.double(), "Error when testing xor.");

    assert_eq!(x.andn(&y), (R::ONE - x) * y, "Error when testing andn.");

    assert_eq!(
        x.xor3(&y, &z),
        x + y + z - (x * y + x * z + y * z).double() + x * y * z.double().double(),
        "Error when testing xor3."
    );
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
    ($field:ty, $zeros: expr, $ones: expr) => {
        mod field_tests {
            #[test]
            fn test_ring_with_eq() {
                $crate::test_ring_with_eq::<$field>($zeros, $ones);
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
