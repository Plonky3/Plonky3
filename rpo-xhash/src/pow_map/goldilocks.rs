//! Goldilocks pow map: x^7 over F_p^3 at width 12 (4 triples).

use p3_goldilocks::Goldilocks;
use p3_symmetric::Permutation;

use crate::ext_arith::fp3::goldilocks as fp3;

#[inline(always)]
fn apply_pow7_fp3_w12(state: &mut [Goldilocks; 12]) {
    let z: [[Goldilocks; 3]; 4] = core::array::from_fn(|s| {
        let b = s * 3;
        [state[b], state[b + 1], state[b + 2]]
    });

    let z2: [[Goldilocks; 3]; 4] = core::array::from_fn(|s| fp3::sqr(z[s]));
    let z3: [[Goldilocks; 3]; 4] = core::array::from_fn(|s| fp3::mul(z2[s], z[s]));
    let z6: [[Goldilocks; 3]; 4] = core::array::from_fn(|s| fp3::sqr(z3[s]));
    let z7: [[Goldilocks; 3]; 4] = core::array::from_fn(|s| fp3::mul(z6[s], z[s]));

    for s in 0..4 {
        let b = s * 3;
        state[b] = z7[s][0];
        state[b + 1] = z7[s][1];
        state[b + 2] = z7[s][2];
    }
}

/// Goldilocks width 12: x^7 over F_p³.
#[derive(Debug, Copy, Clone, Default, Eq, PartialEq)]
pub struct PowMap12;

impl Permutation<[Goldilocks; 12]> for PowMap12 {
    #[inline(always)]
    fn permute_mut(&self, state: &mut [Goldilocks; 12]) {
        apply_pow7_fp3_w12(state);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};

    #[test]
    fn w12_changes_state() {
        let mut rng = StdRng::seed_from_u64(777);
        let state: [Goldilocks; 12] =
            core::array::from_fn(|_| Goldilocks::new(rng.random::<u64>()));
        let mut result = state;
        PowMap12.permute_mut(&mut result);
        assert_ne!(state, result);
    }
}
