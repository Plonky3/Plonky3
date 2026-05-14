//! M31 pow map: x^5 over F_p^3 at width 24 (8 triples).

use p3_field::PrimeField32;
use p3_mersenne_31::Mersenne31;
use p3_symmetric::Permutation;

use crate::ext_arith::fp3::m31 as fp3;

#[inline(always)]
fn apply_pow5_fp3_w24(state: &mut [Mersenne31; 24]) {
    let z: [[u32; 3]; 8] = core::array::from_fn(|s| {
        let b = s * 3;
        [
            state[b].as_canonical_u32(),
            state[b + 1].as_canonical_u32(),
            state[b + 2].as_canonical_u32(),
        ]
    });
    let z2: [[u32; 3]; 8] = core::array::from_fn(|s| fp3::sqr(z[s]));
    let z4: [[u32; 3]; 8] = core::array::from_fn(|s| fp3::sqr(z2[s]));
    let z5: [[u32; 3]; 8] = core::array::from_fn(|s| fp3::mul(z4[s], z[s]));

    for s in 0..8 {
        let b = s * 3;
        state[b] = Mersenne31::new(z5[s][0]);
        state[b + 1] = Mersenne31::new(z5[s][1]);
        state[b + 2] = Mersenne31::new(z5[s][2]);
    }
}

/// M31 width 24: x^5 over F_p³ (nonlinear layer only). 8 S-boxes.
#[derive(Debug, Copy, Clone, Default, Eq, PartialEq)]
pub struct PowMap24;

impl Permutation<[Mersenne31; 24]> for PowMap24 {
    #[inline(always)]
    fn permute_mut(&self, state: &mut [Mersenne31; 24]) {
        apply_pow5_fp3_w24(state);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};

    #[test]
    fn w24_changes_state() {
        let mut rng = StdRng::seed_from_u64(222);
        let state: [Mersenne31; 24] =
            core::array::from_fn(|_| Mersenne31::new(rng.random::<u32>() % Mersenne31::ORDER_U32));
        let mut result = state;
        PowMap24.permute_mut(&mut result);
        assert_ne!(state, result);
    }
}
