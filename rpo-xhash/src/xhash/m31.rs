//! XHash-M31: 3 rounds, base d=5, extension X^5 over F_{p^3} (α³=5).
//!
//! Sage:
//! ```sage
//! p = 2^31 - 1; assert gcd(5, p^3-1) == 1
//! R.<x> = GF(p)[]; assert (x^3-5).is_irreducible()
//! ```

use p3_field::PrimeField32;
use p3_mds::MdsPermutation;
use p3_mersenne_31::Mersenne31;
use rand::RngExt;

use crate::mds_m31_bb::Mds24M31BBCol;
use crate::pow_map::m31::PowMap24;
use crate::rpo::m31::{RpoCirMds24, SboxM31};

use super::XHash;

/// 3 rounds for XHash-M31 (Section 3, paper specifies 3 rounds).
pub const XHASH_M31_ROUNDS: usize = 3;

/// XHash-M31 with the paper's truncated circulant MDS.
pub type XHashM31Cir = XHash<Mersenne31, SboxM31, PowMap24, RpoCirMds24, 24>;

/// XHash-M31 with BabyBear's MDS column (perf comparison; see `crate::mds_m31_bb`).
pub type XHashM31BBMds = XHash<Mersenne31, SboxM31, PowMap24, Mds24M31BBCol, 24>;

fn new_from_rng<Mds: MdsPermutation<Mersenne31, 24> + Default>(
    rng: &mut impl rand::Rng,
) -> XHash<Mersenne31, SboxM31, PowMap24, Mds, 24> {
    let num_constants = (3 * XHASH_M31_ROUNDS + 1) * 24;
    let round_constants = (0..num_constants)
        .map(|_| Mersenne31::new(rng.random::<u32>() % Mersenne31::ORDER_U32))
        .collect();
    XHash::new_from_constants(XHASH_M31_ROUNDS, round_constants)
}

pub fn xhash_m31_cir(rng: &mut impl rand::Rng) -> XHashM31Cir {
    new_from_rng(rng)
}

pub fn xhash_m31_bb_mds(rng: &mut impl rand::Rng) -> XHashM31BBMds {
    new_from_rng(rng)
}

#[cfg(test)]
mod tests {
    use super::*;
    use p3_symmetric::Permutation;
    use rand::rngs::SmallRng;
    use rand::SeedableRng;

    #[test]
    fn xhash_m31_deterministic() {
        let mut rng = SmallRng::seed_from_u64(1);
        let hash = xhash_m31_cir(&mut rng);
        let input: [Mersenne31; 24] =
            core::array::from_fn(|i| Mersenne31::new((i as u32 + 1) * 37));
        let out1 = hash.permute(input);
        let out2 = hash.permute(input);
        assert_eq!(out1, out2);
    }
}
