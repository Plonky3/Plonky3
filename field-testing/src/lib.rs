//! Utilities for testing field implementations.

#![no_std]

extern crate alloc;

pub mod bench_func;
pub mod dft_testing;
pub mod extension_testing;
pub mod from_integer_tests;
pub mod packedfield_testing;

use alloc::vec::Vec;
use core::array;
use core::iter::successors;

pub use bench_func::*;
pub use dft_testing::*;
pub use extension_testing::*;
use num_bigint::BigUint;
use p3_field::coset::TwoAdicMultiplicativeCoset;
use p3_field::{
    ExtensionField, Field, PackedValue, PrimeCharacteristicRing, PrimeField32, PrimeField64,
    TwoAdicField, batch_multiplicative_inverse,
};
use p3_util::iter_array_chunks_padded;
pub use packedfield_testing::*;
use proptest::prelude::*;
use rand::distr::{Distribution, StandardUniform};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use serde::Serialize;
use serde::de::DeserializeOwned;

/// Generate a random field element from a u64 seed, for use in proptest strategies.
fn arb_field<F>() -> impl Strategy<Value = F>
where
    F: core::fmt::Debug + 'static,
    StandardUniform: Distribution<F>,
{
    any::<u64>().prop_map(|seed| {
        let mut rng = SmallRng::seed_from_u64(seed);
        rng.random()
    })
}

#[allow(clippy::eq_op)]
pub fn test_ring_with_eq<R: PrimeCharacteristicRing + Copy + Eq>(zeros: &[R], ones: &[R])
where
    StandardUniform: Distribution<R> + Distribution<[R; 16]>,
{
    // zeros should be a vector containing different representatives of `R::ZERO`.
    // ones should be a vector containing different representatives of `R::ONE`.
    let mut rng = SmallRng::seed_from_u64(1);
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
    assert_eq!(x, x.halve() * R::TWO, "Error when testing halve.");

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
        assert_eq!(one * one, one);
        assert_eq!(
            x * one,
            x,
            "Error when testing multiplicative identity right."
        );
        assert_eq!(
            one * x,
            x,
            "Error when testing multiplicative identity left."
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

    // Edge case tests with special values
    for &a in &[R::ZERO, R::ONE, R::TWO, R::NEG_ONE] {
        for &b in &[R::ZERO, R::ONE, R::TWO, R::NEG_ONE] {
            assert_eq!(a + b, b + a, "commutativity with special values");
            assert_eq!(a * b, b * a, "commutativity with special values");
        }
        assert_eq!(a * a, a.square(), "square with special value");
        assert_eq!(a * a * a, a.cube(), "cube with special value");
        assert_eq!(a.halve().double(), a, "halve/double with special value");
    }

    // Test that Product of empty iterator returns ONE (the multiplicative identity)
    let empty: [R; 0] = [];
    let product_result: R = empty.into_iter().product();
    assert_eq!(
        product_result,
        R::ONE,
        "Product of empty iterator should return ONE, not ZERO"
    );
}

pub fn test_mul_2exp_u64<R: PrimeCharacteristicRing + Eq>()
where
    StandardUniform: Distribution<R>,
{
    let mut rng = SmallRng::seed_from_u64(1);
    let x = rng.random::<R>();
    assert_eq!(x.mul_2exp_u64(0), x);
    assert_eq!(x.mul_2exp_u64(1), x.double());
    for i in 0..128 {
        assert_eq!(
            x.clone().mul_2exp_u64(i),
            x.clone() * R::from_u128(1_u128 << i)
        );
    }
    // Goldilocks behaviour changes at 96, 192 so we want to test larger numbers than that.
    for i in 128..256 {
        assert_eq!(x.clone().mul_2exp_u64(i), x.clone() * R::TWO.exp_u64(i));
    }
}

pub fn test_div_2exp_u64<R: PrimeCharacteristicRing + Eq>()
where
    StandardUniform: Distribution<R>,
{
    let mut rng = SmallRng::seed_from_u64(1);
    let x = rng.random::<R>();
    assert_eq!(x.div_2exp_u64(0), x);
    assert_eq!(x.div_2exp_u64(1), x.halve());
    for i in 0..128 {
        assert_eq!(x.mul_2exp_u64(i).div_2exp_u64(i), x);
        assert_eq!(
            x.div_2exp_u64(i),
            // Best to invert in the prime subfield in case F is an extension field.
            x.clone() * R::from_prime_subfield(R::PrimeSubfield::from_u128(1_u128 << i).inverse())
        );
    }
    // Goldilocks behaviour changes at 96, 192 so we want to test larger numbers than that.
    for i in 128..256 {
        assert_eq!(x.mul_2exp_u64(i).div_2exp_u64(i), x);
        assert_eq!(
            x.div_2exp_u64(i),
            // Best to invert in the prime subfield in case F is an extension field.
            x.clone() * R::from_prime_subfield(R::PrimeSubfield::TWO.inverse().exp_u64(i))
        );
    }
}

pub fn test_add_slice<F: Field>()
where
    StandardUniform: Distribution<F>,
{
    let mut rng = SmallRng::seed_from_u64(1);
    let lengths = [
        F::Packing::WIDTH - 1,
        F::Packing::WIDTH,
        (F::Packing::WIDTH - 1) + (F::Packing::WIDTH << 10),
    ];
    for len in lengths {
        let mut slice_1: Vec<_> = (&mut rng).sample_iter(StandardUniform).take(len).collect();
        let slice_1_copy = slice_1.clone();
        let slice_2: Vec<_> = (&mut rng).sample_iter(StandardUniform).take(len).collect();

        F::add_slices(&mut slice_1, &slice_2);
        for i in 0..len {
            assert_eq!(slice_1[i], slice_1_copy[i] + slice_2[i]);
        }
    }
}

pub fn test_inverse<F: Field>()
where
    StandardUniform: Distribution<F>,
{
    assert_eq!(None, F::ZERO.try_inverse());
    assert_eq!(Some(F::ONE), F::ONE.try_inverse());
    assert_eq!(F::NEG_ONE.inverse(), F::NEG_ONE, "-1 is its own inverse");
    let two_inv = F::TWO
        .try_inverse()
        .expect("2 must be invertible in this field (test_inverse assumes characteristic != 2)");
    assert_eq!(two_inv, F::ONE.halve(), "inverse of 2 == halve(1)");
    let mut rng = SmallRng::seed_from_u64(1);
    for _ in 0..1000 {
        let x = rng.random::<F>();
        if !x.is_zero() && !x.is_one() {
            let z = x.inverse();
            assert_ne!(x, z);
            assert_eq!(x * z, F::ONE);
        }
    }
}

/// Verify [`batch_multiplicative_inverse`] against the naive per-element inverse
/// across a range of input lengths.
///
/// Sizes are chosen to exercise:
/// - the empty input,
/// - lengths below the packing width (tail-only path),
/// - lengths whose remainder mod the packing width is 1, 2, or 3
///   (mixed prefix-packed + tail-serial path),
/// - lengths that straddle the internal `par_chunks` boundary (1024)
///   so the trailing chunk receives a non-aligned tail.
pub fn test_batch_multiplicative_inverse<F: Field>()
where
    StandardUniform: Distribution<F>,
{
    let mut rng = SmallRng::seed_from_u64(0xBA7C);

    let lengths = [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 15, 16, 17, 63, 64, 65, 1023, 1024, 1025, 1027, 2049, 4099,
    ];

    for &n in &lengths {
        let xs: Vec<F> = (0..n)
            .map(|_| {
                // Reject zero so every input is invertible.
                let mut x = rng.random::<F>();
                while x.is_zero() {
                    x = rng.random::<F>();
                }
                x
            })
            .collect();

        let got = batch_multiplicative_inverse(&xs);
        assert_eq!(got.len(), n, "result length mismatch for n = {n}");

        for (i, (x, inv)) in xs.iter().zip(&got).enumerate() {
            assert_eq!(
                *x * *inv,
                F::ONE,
                "x[{i}] * inv[{i}] != 1 for input length {n}"
            );
        }
    }
}

/// Test JSON serialization and deserialization for a set of field values.
///
/// This function tests that:
/// 1. Each value can be serialized and deserialized correctly
/// 2. Double round-trip serialization is consistent
pub fn test_field_json_serialization<F>(values: &[F])
where
    F: PrimeCharacteristicRing + Serialize + DeserializeOwned + Eq,
{
    for value in values {
        // Single round-trip
        let serialized = serde_json::to_string(value).expect("Failed to serialize field element");
        let deserialized: F =
            serde_json::from_str(&serialized).expect("Failed to deserialize field element");
        assert_eq!(
            *value, deserialized,
            "Single round-trip serialization failed"
        );

        // Double round-trip to ensure consistency
        let serialized_again = serde_json::to_string(&deserialized)
            .expect("Failed to serialize field element (second time)");
        let deserialized_again: F = serde_json::from_str(&serialized_again)
            .expect("Failed to deserialize field element (second time)");
        assert_eq!(
            *value, deserialized_again,
            "Double round-trip serialization failed"
        );
        assert_eq!(
            deserialized, deserialized_again,
            "Deserialized values should be equal"
        );
    }
}

/// Test JSON deserialization boundary behavior for 32-bit prime fields.
///
/// Most fields only accept values in `[0, ORDER_U32)`, while some fields (e.g. Mersenne31)
/// have a redundant representation of zero and also accept `ORDER_U32`.
pub fn test_prime_field_32_json_deserialization_boundaries<F>(accepts_order_repr: bool)
where
    F: PrimeField32 + Serialize + DeserializeOwned + Eq,
{
    let zero: F = serde_json::from_str("0").expect("Failed to deserialize zero");
    assert_eq!(zero, F::ZERO, "Deserializing 0 should produce ZERO");

    let original: F = serde_json::from_str("42").expect("Failed to deserialize test value");
    let serialized = serde_json::to_string(&original).expect("Failed to serialize test value");
    let deserialized: F =
        serde_json::from_str(&serialized).expect("Failed to deserialize serialized test value");
    assert_eq!(
        deserialized, original,
        "Round-trip serialization should preserve the value"
    );

    let max_valid = if accepts_order_repr {
        F::ORDER_U32
    } else {
        F::ORDER_U32 - 1
    };
    let max_valid_json = serde_json::to_string(&max_valid).expect("Failed to encode max valid u32");
    let max_valid_result: Result<F, _> = serde_json::from_str(&max_valid_json);
    assert!(
        max_valid_result.is_ok(),
        "Expected max valid representation to deserialize successfully"
    );

    if let Some(first_invalid) = max_valid.checked_add(1) {
        let first_invalid_json =
            serde_json::to_string(&first_invalid).expect("Failed to encode first invalid value");
        let first_invalid_result: Result<F, _> = serde_json::from_str(&first_invalid_json);
        assert!(
            first_invalid_result.is_err(),
            "Expected first out-of-range representation to fail deserialization"
        );
    }

    if max_valid != u32::MAX {
        let max_u32_json = serde_json::to_string(&u32::MAX).expect("Failed to encode u32::MAX");
        let max_u32_result: Result<F, _> = serde_json::from_str(&max_u32_json);
        assert!(
            max_u32_result.is_err(),
            "Expected u32::MAX to fail deserialization"
        );
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
    assert_eq!(sum, R::sum_array::<0>(u[..0].try_into().unwrap()));
    assert_eq!(sum, u[..0].iter().copied().sum());
    sum += u[0];
    assert_eq!(sum, R::sum_array::<1>(u[..1].try_into().unwrap()));
    assert_eq!(sum, u[..1].iter().copied().sum());
    sum += u[1];
    assert_eq!(sum, R::sum_array::<2>(u[..2].try_into().unwrap()));
    assert_eq!(sum, u[..2].iter().copied().sum());
    sum += u[2];
    assert_eq!(sum, R::sum_array::<3>(u[..3].try_into().unwrap()));
    assert_eq!(sum, u[..3].iter().copied().sum());
    sum += u[3];
    assert_eq!(sum, R::sum_array::<4>(u[..4].try_into().unwrap()));
    assert_eq!(sum, u[..4].iter().copied().sum());
    sum += u[4];
    assert_eq!(sum, R::sum_array::<5>(u[..5].try_into().unwrap()));
    assert_eq!(sum, u[..5].iter().copied().sum());
    sum += u[5];
    assert_eq!(sum, R::sum_array::<6>(u[..6].try_into().unwrap()));
    assert_eq!(sum, u[..6].iter().copied().sum());
    sum += u[6];
    assert_eq!(sum, R::sum_array::<7>(u[..7].try_into().unwrap()));
    assert_eq!(sum, u[..7].iter().copied().sum());
    sum += u[7];
    assert_eq!(sum, R::sum_array::<8>(u[..8].try_into().unwrap()));
    assert_eq!(sum, u[..8].iter().copied().sum());
    sum += u[8];
    assert_eq!(sum, R::sum_array::<9>(u[..9].try_into().unwrap()));
    assert_eq!(sum, u[..9].iter().copied().sum());
    sum += u[9];
    assert_eq!(sum, R::sum_array::<10>(u[..10].try_into().unwrap()));
    assert_eq!(sum, u[..10].iter().copied().sum());
    sum += u[10];
    assert_eq!(sum, R::sum_array::<11>(u[..11].try_into().unwrap()));
    assert_eq!(sum, u[..11].iter().copied().sum());
    sum += u[11];
    assert_eq!(sum, R::sum_array::<12>(u[..12].try_into().unwrap()));
    assert_eq!(sum, u[..12].iter().copied().sum());
    sum += u[12];
    assert_eq!(sum, R::sum_array::<13>(u[..13].try_into().unwrap()));
    assert_eq!(sum, u[..13].iter().copied().sum());
    sum += u[13];
    assert_eq!(sum, R::sum_array::<14>(u[..14].try_into().unwrap()));
    assert_eq!(sum, u[..14].iter().copied().sum());
    sum += u[14];
    assert_eq!(sum, R::sum_array::<15>(u[..15].try_into().unwrap()));
    assert_eq!(sum, u[..15].iter().copied().sum());
    sum += u[15];
    assert_eq!(sum, R::sum_array::<16>(u));
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

/// Tests the optimized implementation of `powers.take(n).collect()`
pub fn test_powers_collect<F: Field>() {
    // Small using serial implementation
    let small_powers_serial = [0, 1, 2, 3, 4, 15];
    // Small using packed implementation
    let small_powers_packed = [16, 17];
    // Large powers of two
    let powers_of_two = [5, 6, 7, 8, 9, 10, 13];

    let num_powers_tests: Vec<usize> = small_powers_serial
        .into_iter()
        .chain(small_powers_packed)
        .chain(powers_of_two.iter().flat_map(|exp| {
            // Check boundaries at power of 2
            let n = 1 << exp;
            [n - 1, n, n + 1]
        }))
        .collect();

    let base = F::TWO;
    let shift = F::GENERATOR;

    // Manual implementation of `Powers`
    let expected_iter = successors(Some(shift), |prev| Some(*prev * base));

    for num_powers in num_powers_tests {
        let expected: Vec<_> = expected_iter.clone().take(num_powers).collect();
        let actual = base.shifted_powers(shift).collect_n(num_powers);
        assert_eq!(
            expected, actual,
            "Got different powers when taking {num_powers}"
        );
    }
}

/// A function which extends the `exp_u64` code to handle `BigUints`.
///
/// This solution is slow (particularly when dealing with extension fields
/// which should really be making use of the frobenius map) but should be
/// fast enough for testing purposes.
pub(crate) fn exp_biguint<F: Field>(x: F, exponent: &BigUint) -> F {
    let digits = exponent.to_u64_digits();
    let size = digits.len();

    let mut power = F::ONE;

    let bases = (0..size).map(|i| x.exp_power_of_2(64 * i));
    digits
        .iter()
        .zip(bases)
        .for_each(|(digit, base)| power *= base.exp_u64(*digit));
    power
}

/// Given a list of the factors of the multiplicative group of a field, check
/// that the defined generator is actually a generator of that group.
pub fn test_generator<F: Field>(multiplicative_group_factors: &[(BigUint, u32)]) {
    // First we check that the given factors multiply to the order of the
    // multiplicative group (|F| - 1). Ideally this would also check that
    // the given factors are prime but as factors can be large that check
    // can end up being quite expensive so ignore that for now. As the factors
    // are hardcoded and public, these prime checks can be easily done using
    // sage or wolfram alpha.
    let product: BigUint = multiplicative_group_factors
        .iter()
        .map(|(factor, exponent)| factor.pow(*exponent))
        .product();
    assert_eq!(product + BigUint::from(1u32), F::order());

    // Given a prime factorization r = p1^e1 * p2^e2 * ... * pk^ek, an element g has order
    // r if and only if g^r = 1 and g^(r/pi) != 1 for all pi in the prime factorization of r.
    let mut partial_products: Vec<F> = (0..=multiplicative_group_factors.len())
        .map(|i| {
            let mut generator_power = F::GENERATOR;
            multiplicative_group_factors
                .iter()
                .enumerate()
                .for_each(|(j, (factor, exponent))| {
                    let modified_exponent = if i == j { exponent - 1 } else { *exponent };
                    for _ in 0..modified_exponent {
                        generator_power = exp_biguint(generator_power, factor);
                    }
                });
            generator_power
        })
        .collect();

    assert_eq!(partial_products.pop().unwrap(), F::ONE);

    for elem in partial_products.into_iter() {
        assert_ne!(elem, F::ONE);
    }
}

pub fn test_two_adic_generator_consistency<F: TwoAdicField>() {
    let log_n = F::TWO_ADICITY;
    let g = F::two_adic_generator(log_n);
    for bits in 0..=log_n {
        assert_eq!(g.exp_power_of_2(bits), F::two_adic_generator(log_n - bits));
    }
}

pub fn test_two_adic_point_collection<F: TwoAdicField>() {
    let log_n = F::TWO_ADICITY.min(15);
    for bits in 0..=log_n {
        let group = TwoAdicMultiplicativeCoset::new(F::ONE, bits).unwrap();
        let points = group.iter().collect();
        // Add `map` to avoid calling `BoundedPowers::collect()`
        #[allow(clippy::map_identity)]
        let points_expected = group.iter().map(|x| x).collect::<Vec<_>>();
        assert_eq!(points, points_expected);
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

pub fn test_into_bytes_32<F: PrimeField32>(zeros: &[F], ones: &[F])
where
    StandardUniform: Distribution<F>,
{
    let mut rng = SmallRng::seed_from_u64(1);
    let x = rng.random::<F>();

    assert_eq!(
        x.into_bytes().into_iter().collect::<Vec<_>>(),
        x.to_unique_u32().to_le_bytes()
    );
    for one in ones {
        assert_eq!(
            one.into_bytes().into_iter().collect::<Vec<_>>(),
            F::ONE.to_unique_u32().to_le_bytes()
        );
    }
    for zero in zeros {
        assert_eq!(zero.into_bytes().into_iter().collect::<Vec<_>>(), [0; 4]);
    }
}

pub fn test_into_bytes_64<F: PrimeField64>(zeros: &[F], ones: &[F])
where
    StandardUniform: Distribution<F>,
{
    let mut rng = SmallRng::seed_from_u64(1);
    let x = rng.random::<F>();

    assert_eq!(
        x.into_bytes().into_iter().collect::<Vec<_>>(),
        x.to_unique_u64().to_le_bytes()
    );
    for one in ones {
        assert_eq!(
            one.into_bytes().into_iter().collect::<Vec<_>>(),
            F::ONE.to_unique_u64().to_le_bytes()
        );
    }
    for zero in zeros {
        assert_eq!(zero.into_bytes().into_iter().collect::<Vec<_>>(), [0; 8]);
    }
}

pub fn test_into_stream<F: Field>()
where
    StandardUniform: Distribution<[F; 16]>,
{
    let mut rng = SmallRng::seed_from_u64(1);
    let xs: [F; 16] = rng.random();

    let byte_vec = F::into_byte_stream(xs).into_iter().collect::<Vec<_>>();
    let u32_vec = F::into_u32_stream(xs).into_iter().collect::<Vec<_>>();
    let u64_vec = F::into_u64_stream(xs).into_iter().collect::<Vec<_>>();

    let expected_bytes = xs
        .into_iter()
        .flat_map(|x| x.into_bytes())
        .collect::<Vec<_>>();
    let expected_u32s = iter_array_chunks_padded(byte_vec.iter().copied(), 0)
        .map(u32::from_le_bytes)
        .collect::<Vec<_>>();
    let expected_u64s = iter_array_chunks_padded(byte_vec.iter().copied(), 0)
        .map(u64::from_le_bytes)
        .collect::<Vec<_>>();

    assert_eq!(byte_vec, expected_bytes);
    assert_eq!(u32_vec, expected_u32s);
    assert_eq!(u64_vec, expected_u64s);

    let ys: [F; 16] = rng.random();
    let zs: [F; 16] = rng.random();

    let combs: [[F; 3]; 16] = array::from_fn(|i| [xs[i], ys[i], zs[i]]);

    let byte_vec_ys = F::into_byte_stream(ys).into_iter().collect::<Vec<_>>();
    let byte_vec_zs = F::into_byte_stream(zs).into_iter().collect::<Vec<_>>();
    let u32_vec_ys = F::into_u32_stream(ys).into_iter().collect::<Vec<_>>();
    let u32_vec_zs = F::into_u32_stream(zs).into_iter().collect::<Vec<_>>();
    let u64_vec_ys = F::into_u64_stream(ys).into_iter().collect::<Vec<_>>();
    let u64_vec_zs = F::into_u64_stream(zs).into_iter().collect::<Vec<_>>();

    let combined_bytes = F::into_parallel_byte_streams(combs)
        .into_iter()
        .collect::<Vec<_>>();
    let combined_u32s = F::into_parallel_u32_streams(combs)
        .into_iter()
        .collect::<Vec<_>>();
    let combined_u64s = F::into_parallel_u64_streams(combs)
        .into_iter()
        .collect::<Vec<_>>();

    let expected_combined_bytes: Vec<[u8; 3]> = (0..byte_vec.len())
        .map(|i| [byte_vec[i], byte_vec_ys[i], byte_vec_zs[i]])
        .collect();
    let expected_combined_u32s: Vec<[u32; 3]> = (0..u32_vec.len())
        .map(|i| [u32_vec[i], u32_vec_ys[i], u32_vec_zs[i]])
        .collect();
    let expected_combined_u64s: Vec<[u64; 3]> = (0..u64_vec.len())
        .map(|i| [u64_vec[i], u64_vec_ys[i], u64_vec_zs[i]])
        .collect();

    assert_eq!(combined_bytes, expected_combined_bytes);
    assert_eq!(combined_u32s, expected_combined_u32s);
    assert_eq!(combined_u64s, expected_combined_u64s);
}

/// Test ring axioms with 256 random (x, y, z) triplets via proptest.
///
/// Tests commutativity, associativity, distributivity, negation,
/// subtraction identities, square/cube, double/halve, and
/// multiplication by zero and negative one.
pub fn test_ring_axioms_proptest<R>()
where
    R: PrimeCharacteristicRing + Copy + Eq + core::fmt::Debug + 'static,
    StandardUniform: Distribution<R>,
{
    let config = ProptestConfig::with_cases(256);
    proptest!(config, |(x in arb_field::<R>(), y in arb_field::<R>(), z in arb_field::<R>())| {
        // Commutativity
        prop_assert_eq!(x + y, y + x, "addition commutativity");
        prop_assert_eq!(x * y, y * x, "multiplication commutativity");

        // Associativity
        prop_assert_eq!(x + (y + z), (x + y) + z, "addition associativity");
        prop_assert_eq!(x * (y * z), (x * y) * z, "multiplication associativity");

        // Distributivity
        prop_assert_eq!(x * (y + z), x * y + x * z, "left distributivity");
        prop_assert_eq!((x + y) * z, x * z + y * z, "right distributivity");

        // Negation
        prop_assert_eq!(x + (-x), R::ZERO, "additive inverse");
        prop_assert_eq!(-(-x), x, "double negation");

        // Subtraction identities
        prop_assert_eq!(x - (y - z), (x - y) + z, "sub-sub identity");
        prop_assert_eq!(x - (y + z), (x - y) - z, "sub-add identity");

        // Square and cube
        prop_assert_eq!(x * x, x.square(), "square");
        prop_assert_eq!(x * x * x, x.cube(), "cube");

        // Double and halve
        prop_assert_eq!(x.double(), x + x, "double");
        prop_assert_eq!(x.halve().double(), x, "halve roundtrip");

        // Multiplication by zero and negative one
        prop_assert_eq!(x * R::ZERO, R::ZERO, "x * 0 == 0");
        prop_assert_eq!(R::NEG_ONE * x, -x, "-1 * x == -x");
    });
}

/// Test field axioms (inverse, division) with deterministic edge cases
/// and 256 random non-zero (x, y, z) triplets via proptest.
pub fn test_field_axioms_proptest<F>()
where
    F: Field + core::fmt::Debug + 'static,
    StandardUniform: Distribution<F>,
{
    // Deterministic edge cases
    assert_eq!(F::TWO.inverse(), F::ONE.halve());
    assert_eq!(F::NEG_ONE.inverse(), F::NEG_ONE, "-1 is its own inverse");
    assert_eq!(
        F::GENERATOR.inverse() * F::GENERATOR,
        F::ONE,
        "generator inverse roundtrip"
    );

    // Proptest: 256 random triplets, all non-zero
    let config = ProptestConfig::with_cases(256);
    proptest!(config, |(x in arb_field::<F>(), y in arb_field::<F>(), z in arb_field::<F>())| {
        // Skip if any element is zero
        if x.is_zero() || y.is_zero() || z.is_zero() {
            return Ok(());
        }

        // Inverse properties
        prop_assert_eq!(x * x.inverse(), F::ONE, "x * x^-1 == 1");
        prop_assert_eq!(x.inverse().inverse(), x, "double inverse");
        prop_assert_eq!(x.square().inverse(), x.inverse().square(), "square-inverse commutativity");

        // Division roundtrip
        prop_assert_eq!((x / y) * y, x, "division roundtrip");

        // Division associativity
        prop_assert_eq!(x / (y * z), (x / y) / z, "division-multiplication associativity");
        prop_assert_eq!((x * y) / z, x * (y / z), "multiplication-division associativity");
    });
}

#[macro_export]
macro_rules! test_ring_with_eq {
    ($ring:ty, $zeros: expr, $ones: expr) => {
        mod ring_tests {
            use p3_field::PrimeCharacteristicRing;

            #[test]
            fn test_ring_with_eq() {
                $crate::test_ring_with_eq::<$ring>($zeros, $ones);
            }
            #[test]
            fn test_mul_2exp_u64() {
                $crate::test_mul_2exp_u64::<$ring>();
            }
            #[test]
            fn test_div_2exp_u64() {
                $crate::test_div_2exp_u64::<$ring>();
            }
        }
    };
}

#[macro_export]
macro_rules! test_field {
    ($field:ty, $zeros: expr, $ones: expr, $factors: expr) => {
        $crate::test_ring_with_eq!($field, $zeros, $ones);

        mod field_tests {
            #[test]
            fn test_inverse() {
                $crate::test_inverse::<$field>();
            }
            #[test]
            fn test_batch_multiplicative_inverse() {
                $crate::test_batch_multiplicative_inverse::<$field>();
            }
            #[test]
            fn test_generator() {
                $crate::test_generator::<$field>($factors);
            }
            #[test]
            fn test_streaming() {
                $crate::test_into_stream::<$field>();
            }
            #[test]
            fn test_powers_collect() {
                $crate::test_powers_collect::<$field>();
            }
            #[test]
            fn test_ring_axioms_proptest() {
                $crate::test_ring_axioms_proptest::<$field>();
            }
            #[test]
            fn test_field_axioms_proptest() {
                $crate::test_field_axioms_proptest::<$field>();
            }
        }

        // Looks a little strange but we also check that everything works
        // when the field is considered as a trivial extension of itself.
        mod trivial_extension_tests {
            #[test]
            fn test_to_from_trivial_extension() {
                $crate::test_to_from_extension_field::<$field, $field>();
            }

            #[test]
            fn test_trivial_packed_extension() {
                $crate::test_packed_extension::<$field, $field>();
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
    ($field:ty, $zeros: expr, $ones: expr) => {
        mod from_integer_tests_prime_field_64 {
            use p3_field::integers::QuotientMap;
            use p3_field::{Field, PrimeCharacteristicRing, PrimeField64, RawDataSerializable};
            use rand::rngs::SmallRng;
            use rand::{RngExt, SeedableRng};

            #[test]
            fn test_as_canonical_u64() {
                let mut rng = SmallRng::seed_from_u64(1);
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

            #[test]
            fn test_raw_data_serializable() {
                // Only do the 64-bit test if the field is 64 bits.
                // This will error if tested on smaller fields.
                if <$field>::NUM_BYTES == 8 {
                    $crate::test_into_bytes_64::<$field>($zeros, $ones);
                }
            }
        }
    };
}

#[macro_export]
macro_rules! test_prime_field_32 {
    ($field:ty, $zeros: expr, $ones: expr) => {
        mod from_integer_tests_prime_field_32 {
            use p3_field::integers::QuotientMap;
            use p3_field::{Field, PrimeCharacteristicRing, PrimeField32, PrimeField64};
            use rand::rngs::SmallRng;
            use rand::{RngExt, SeedableRng};

            #[test]
            fn test_as_canonical_u32() {
                let mut rng = SmallRng::seed_from_u64(1);
                let x: u32 = rng.random();
                let x_mod_order = x % <$field>::ORDER_U32;

                for zero in $zeros {
                    assert_eq!(zero.as_canonical_u32(), 0);
                    assert_eq!(zero.to_unique_u32() as u64, zero.to_unique_u64());
                }
                for one in $ones {
                    assert_eq!(one.as_canonical_u32(), 1);
                    assert_eq!(one.to_unique_u32() as u64, one.to_unique_u64());
                }
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
                    <$field>::from_int(x).to_unique_u32() as u64,
                    <$field>::from_int(x).to_unique_u64()
                );
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

            #[test]
            fn test_raw_data_serializable() {
                $crate::test_into_bytes_32::<$field>($zeros, $ones);
            }

            #[test]
            fn test_json_deserialization_boundaries() {
                let accepts_order_repr = $zeros.len() > 1;
                $crate::test_prime_field_32_json_deserialization_boundaries::<$field>(
                    accepts_order_repr,
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
            fn test_two_adic_consistency() {
                $crate::test_two_adic_generator_consistency::<$field>();
                $crate::test_two_adic_point_collection::<$field>();
            }

            // Looks a little strange but we also check that everything works
            // when the field is considered as a trivial extension of itself.
            #[test]
            fn test_two_adic_generator_consistency_as_trivial_extension() {
                $crate::test_ef_two_adic_generator_consistency::<$field, $field>();
            }
        }
    };
}

#[macro_export]
macro_rules! test_extension_field {
    ($field:ty, $ef:ty) => {
        mod extension_field_tests {
            #[test]
            fn test_to_from_extension() {
                $crate::test_to_from_extension_field::<$field, $ef>();
            }

            #[test]
            fn test_galois_extension() {
                $crate::test_galois_extension::<$field, $ef>();
            }

            #[test]
            fn test_packed_extension() {
                $crate::test_packed_extension::<$field, $ef>();
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

#[macro_export]
macro_rules! test_frobenius {
    ($field:ty, $ef:ty) => {
        mod frobenius_tests {
            #[test]
            fn test_frobenius_fixes_base_field() {
                $crate::test_frobenius_fixes_base_field::<$field, $ef>();
            }

            #[test]
            fn test_frobenius_proptest() {
                $crate::test_frobenius_proptest::<$field, $ef>();
            }
        }
    };
}
