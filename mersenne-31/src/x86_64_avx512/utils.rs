//! Shared AVX512 utilities for Poseidon hash functions on Mersenne31.

use core::arch::x86_64::{self, __m512i};

use p3_field::PrimeField32;

use crate::{Mersenne31, PackedMersenne31AVX512, exp5};

/// Convert a Mersenne31 element from standard form {0, ..., P} to negative form {-P, ..., 0},
/// broadcast to all 16 AVX512 lanes.
#[inline(always)]
pub(super) fn convert_to_vec_neg_form(input: i32) -> __m512i {
    let input_sub_p = input - (Mersenne31::ORDER_U32 as i32);
    unsafe {
        // Safety: If this code got compiled then AVX512-F intrinsics are available.
        x86_64::_mm512_set1_epi32(input_sub_p)
    }
}

/// Compute the map x -> (x + rc)^5 on Mersenne-31 field elements.
/// x must be represented as a value in {0..P}.
/// rc must be represented as a value in {-P, ..., 0}.
/// If the inputs do not conform to these representations, the result is undefined.
/// The output will be represented as a value in {0..P}.
#[inline(always)]
pub(super) fn add_rc_and_sbox(
    input: PackedMersenne31AVX512,
    rc: __m512i,
) -> PackedMersenne31AVX512 {
    unsafe {
        // Safety: If this code got compiled then AVX512-F intrinsics are available.
        let input_vec = input.to_vector();
        let input_plus_rc = x86_64::_mm512_add_epi32(input_vec, rc);

        // Due to the representations of input and rc, input_plus_rc is in {-P, ..., P}.
        // This is exactly the required bound to apply sbox.
        let input_post_sbox = exp5(input_plus_rc);
        PackedMersenne31AVX512::from_vector(input_post_sbox)
    }
}

#[cfg(test)]
mod tests {
    use p3_field::{InjectiveMonomial, PrimeCharacteristicRing};
    use proptest::prelude::*;

    use crate::{Mersenne31, PackedMersenne31AVX512};

    type F = Mersenne31;

    fn arb_f() -> impl Strategy<Value = F> {
        prop::num::u32::ANY.prop_map(F::from_u32)
    }

    proptest! {
        #[test]
        fn add_rc_and_sbox_matches_scalar(
            inputs in prop::array::uniform16(arb_f()),
            rc in arb_f(),
        ) {
            let expected: [F; 16] = inputs.map(|x| (x + rc).injective_exp_n());

            let packed = PackedMersenne31AVX512(inputs);
            let rc_neg = super::convert_to_vec_neg_form(rc.value as i32);
            let result = super::add_rc_and_sbox(packed, rc_neg);

            prop_assert_eq!(result.0, expected);
        }
    }
}
