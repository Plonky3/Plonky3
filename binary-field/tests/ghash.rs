//! Independent vectors and boundary coverage for polynomial-basis arithmetic.

use core::array;

use p3_binary_field::{Gf2, Ghash128};
use p3_field::{Field, PackedField, PackedValue, PrimeCharacteristicRing};
use proptest::prelude::*;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

/// Read a NIST block written as a big-endian integer into polynomial coordinates.
const fn nist_block(block: u128) -> Ghash128 {
    // NIST assigns x^0 to the leftmost bit.
    // Polynomial coordinates assign it to the low bit.
    Ghash128::from_le_bytes(block.reverse_bits().to_le_bytes())
}

#[test]
fn nist_gcm_aes128_example_2() {
    // Source: NIST GCM examples, GCM-AES128 Example #2, pages 2–3.
    // https://csrc.nist.gov/CSRC/media/Projects/Cryptographic-Standards-and-Guidelines/documents/examples/AES_GCM.pdf
    let h = nist_block(0xb83b533708bf535d0aa6e52980d53b78);
    let blocks = [
        0x42831ec2217774244b7221b784d0d49c,
        0xe3aa212f2c02a4e035c17e2329aca12e,
        0x21d514b25466931c7d8f6a5aac84aa05,
        0x1ba30b396a0aac973d58e091473f5985,
        // No associated data; 512 ciphertext bits in the final length block.
        512,
    ];

    // GHASH absorbs each block by adding it and multiplying by the authentication key.
    let result = blocks
        .into_iter()
        .fold(Ghash128::ZERO, |acc, block| (acc + nist_block(block)) * h);
    assert_eq!(result, nist_block(0x7f1b32b81b820d02614f8895ac1d4eac));
}

/// Check every lane against independently reduced scalar products.
fn check_dot<P: PackedField<Scalar = Ghash128> + Eq, const N: usize>(seed: u64) {
    let mut rng = SmallRng::seed_from_u64(seed);
    // Different values in adjacent lanes expose accidental cross-lane carries.
    let a: [P; N] = array::from_fn(|_| P::from_fn(|_| rng.random()));
    let b: [P; N] = array::from_fn(|_| P::from_fn(|_| rng.random()));
    let actual = P::dot_product(&a, &b);
    for lane in 0..P::WIDTH {
        let expected: Ghash128 = a
            .iter()
            .zip(&b)
            .map(|(x, y)| x.as_slice()[lane] * y.as_slice()[lane])
            .sum();
        assert_eq!(actual.as_slice()[lane], expected);
    }
}

/// Check one packing over the accumulation lengths that exercise each code path.
///
/// The length is a const generic, so the cases are spelled out rather than looped over.
fn check_dot_lengths<P: PackedField<Scalar = Ghash128> + Eq>(seed: u64) {
    // Empty, singleton, odd, and two longer runs.
    check_dot::<P, 0>(seed);
    check_dot::<P, 1>(seed);
    check_dot::<P, 3>(seed);
    check_dot::<P, 16>(seed);
    check_dot::<P, 65>(seed);
}

proptest! {
    #[test]
    fn dot_products_match_scalar_reduction(seed: u64) {
        // A scalar is a width-one packing, so both go through the same checks.
        check_dot_lengths::<Ghash128>(seed);
        check_dot_lengths::<<Ghash128 as Field>::Packing>(seed);
    }
}

#[test]
fn dot_product_cancellation_and_extremes() {
    type P = <Ghash128 as Field>::Packing;
    // Shift the extreme values between lanes so every lane sees every boundary.
    let extremes = [0, 1, u128::MAX, 1 << 127, 1 << 64, u64::MAX as u128, 0x87];
    for offset in 0..extremes.len() {
        let a = P::from_fn(|i| {
            Ghash128::from_le_bytes(extremes[(offset + i) % extremes.len()].to_le_bytes())
        });
        let b = P::from_fn(|i| {
            Ghash128::from_le_bytes(extremes[(offset + i + 1) % extremes.len()].to_le_bytes())
        });
        // Two identical products cancel before reduction in characteristic two.
        assert_eq!(P::dot_product(&[a, a], &[b, b]), P::ZERO);
        assert_eq!(P::dot_product(&[a, a, a], &[b, b, b]), a * b);
    }
}

#[test]
fn packed_property_suite_supports_zero_in_small_fields() {
    // Half the random GF(2) denominators are zero; the suite must handle them explicitly.
    p3_field_testing::test_packed_vs_scalar_proptest::<Gf2>();
}
