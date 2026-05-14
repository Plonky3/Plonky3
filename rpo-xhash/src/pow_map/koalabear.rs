//! KoalaBear pow map: x^3 over F_p^3 at width 24 (8 triples).

use p3_koala_bear::KoalaBear;
use p3_symmetric::Permutation;

use crate::ext_arith::fp3::koalabear_field as fp3;

#[inline(always)]
fn apply_pow3_fp3_w24(state: &mut [KoalaBear; 24]) {
    let z: [[KoalaBear; 3]; 8] = core::array::from_fn(|s| {
        let b = s * 3;
        [state[b], state[b + 1], state[b + 2]]
    });
    let z2: [[KoalaBear; 3]; 8] = core::array::from_fn(|s| fp3::sqr(z[s]));
    let z3: [[KoalaBear; 3]; 8] = core::array::from_fn(|s| fp3::mul(z2[s], z[s]));

    for s in 0..8 {
        let b = s * 3;
        state[b] = z3[s][0];
        state[b + 1] = z3[s][1];
        state[b + 2] = z3[s][2];
    }
}

/// KoalaBear width 24: x^3 over F_p³ (trinomial α³+α+4=0).
#[derive(Debug, Copy, Clone, Default, Eq, PartialEq)]
pub struct PowMap24;

impl Permutation<[KoalaBear; 24]> for PowMap24 {
    #[inline(always)]
    fn permute_mut(&self, state: &mut [KoalaBear; 24]) {
        apply_pow3_fp3_w24(state);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use p3_field::PrimeField32;
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};

    #[test]
    fn w24_changes_state() {
        let mut rng = StdRng::seed_from_u64(666);
        let state: [KoalaBear; 24] =
            core::array::from_fn(|_| KoalaBear::new(rng.random::<u32>() % KoalaBear::ORDER_U32));
        let mut result = state;
        PowMap24.permute_mut(&mut result);
        assert_ne!(state, result);
    }
}
