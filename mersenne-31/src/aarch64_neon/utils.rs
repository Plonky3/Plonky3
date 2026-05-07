//! Shared NEON utilities for Poseidon hash functions on Mersenne31.

use core::arch::aarch64::{self, uint32x4_t};
use core::mem::transmute;

use super::packing::exp5;
use crate::PackedMersenne31Neon;

/// The prime P = 2^31 - 1 as a packed NEON vector.
const P: uint32x4_t = unsafe { transmute::<[u32; 4], _>([0x7fffffff; 4]) };

/// Compute the fused AddRoundConstant and S-Box operation: `x -> (x + rc)^5`.
///
/// # Optimization
/// 1. Adds `rc` (positive form). Result is in `[0, 2P]`.
/// 2. Performs "Min-Reduction": `min(sum, sum - P)`.
///    - If `sum < P`, `sum - P` wraps to a huge value, `min` selects `sum`.
///    - If `sum >= P`, `sum - P` is small, `min` selects `sum - P`.
///    - Cost: 2 instructions (`sub`, `min`).
/// 3. Calls `exp5`. Since input is now strictly `[0, P]`, `exp5` does not need
///    to handle signs or absolute values.
///
/// # Safety
/// - `input` must contain elements in canonical form `{0, ..., P}`.
/// - `rc` must contain round constants in positive form `{0, ..., P}`.
#[inline(always)]
pub(super) fn add_rc_and_sbox(input: &mut PackedMersenne31Neon, rc: uint32x4_t) {
    unsafe {
        // Safety: If this code got compiled then NEON intrinsics are available.
        let input_vec = input.to_vector();

        // 1. Add round constant. Result in [0, 2P].
        let sum = aarch64::vaddq_u32(input_vec, rc);

        // 2. Fast Reduction to [0, P].
        // If sum >= P, we want (sum - P).
        // If sum < P, (sum - P) underflows to > P.
        // Unsigned min selects the correct modular result.
        let diff = aarch64::vsubq_u32(sum, P);
        let reduced = aarch64::vminq_u32(sum, diff);

        // 3. Apply S-box (optimized for positive inputs).
        let output = exp5(reduced);

        *input = PackedMersenne31Neon::from_vector(output);
    }
}

#[cfg(test)]
mod tests {
    use core::arch::aarch64;

    use p3_field::{InjectiveMonomial, PrimeCharacteristicRing};
    use proptest::prelude::*;

    use crate::{Mersenne31, PackedMersenne31Neon};

    type F = Mersenne31;

    fn arb_f() -> impl Strategy<Value = F> {
        prop::num::u32::ANY.prop_map(F::from_u32)
    }

    proptest! {
        #[test]
        fn add_rc_and_sbox_matches_scalar(
            inputs in prop::array::uniform4(arb_f()),
            rcs in prop::array::uniform4(arb_f()),
        ) {
            // Scalar reference: (x + rc)^5 per lane.
            let expected: [F; 4] = core::array::from_fn(|i| {
                (inputs[i] + rcs[i]).injective_exp_n()
            });

            // NEON path.
            let mut packed = PackedMersenne31Neon(inputs);
            let rc_vec = unsafe { aarch64::vld1q_u32(rcs.map(|f| f.value).as_ptr()) };
            super::add_rc_and_sbox(&mut packed, rc_vec);

            prop_assert_eq!(packed.0, expected);
        }
    }
}
