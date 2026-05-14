//! BabyBear pow map: x^7 over F_p^2 at width 24 (12 pairs).
//!
//! Uses canonical (non-Montgomery) u32 arithmetic for the Fp2 S-box.
//! Although this requires Montgomery conversion at the boundaries
//! (as_canonical_u32 on read, BabyBear::new on write), the canonical
//! Barrett reduction inside the Fp2 arithmetic is cheaper than
//! Montgomery reduction per intermediate product.

use p3_baby_bear::BabyBear;
use p3_field::PrimeField32;
use p3_symmetric::Permutation;

use crate::ext_arith::fp2::babybear as fp2;

#[inline(always)]
fn apply_pow7_fp2_w24(state: &mut [BabyBear; 24]) {
    let z: [[u32; 2]; 12] = core::array::from_fn(|s| {
        [
            state[2 * s].as_canonical_u32(),
            state[2 * s + 1].as_canonical_u32(),
        ]
    });
    let z2: [[u32; 2]; 12] = core::array::from_fn(|s| fp2::sqr(z[s]));
    let z3: [[u32; 2]; 12] = core::array::from_fn(|s| fp2::mul(z2[s], z[s]));
    let z6: [[u32; 2]; 12] = core::array::from_fn(|s| fp2::sqr(z3[s]));
    let z7: [[u32; 2]; 12] = core::array::from_fn(|s| fp2::mul(z6[s], z[s]));

    for s in 0..12 {
        state[2 * s] = BabyBear::new(z7[s][0]);
        state[2 * s + 1] = BabyBear::new(z7[s][1]);
    }
}

/// BabyBear width 24: x^7 over F_p² (nonlinear layer only).
#[derive(Debug, Copy, Clone, Default, Eq, PartialEq)]
pub struct PowMap24;

impl Permutation<[BabyBear; 24]> for PowMap24 {
    #[inline(always)]
    fn permute_mut(&self, state: &mut [BabyBear; 24]) {
        apply_pow7_fp2_w24(state);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};

    #[test]
    fn w24_changes_state() {
        let mut rng = StdRng::seed_from_u64(444);
        let state: [BabyBear; 24] =
            core::array::from_fn(|_| BabyBear::new(rng.random::<u32>() % BabyBear::ORDER_U32));
        let mut result = state;
        PowMap24.permute_mut(&mut result);
        assert_ne!(state, result);
    }
}
