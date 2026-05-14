//! XHash-BabyBear: 3 rounds, base d=7, extension X^7 over F_{p^2} (α²=11).
//!
//! d=5 fails in the base (5 | p−1). d=7 is the smallest prime with gcd(7, p−1)=1,
//! and also satisfies gcd(7, p²−1)=1: p ≡ 2 (mod 7), so p+1 ≡ 3 (mod 7), and
//! prime factors of p−1 are {2, 3, 5}. So x^7 is a permutation over F_{p^2}.
//!
//! Using F_{p^2} with d=7 here (rather than F_{p^3} which would force d=11)
//! roughly halves the extension-layer cost.
//!
//! Sage:
//! ```sage
//! p = 2013265921
//! assert gcd(5, p-1) != 1     # d=5 fails in F_p
//! assert gcd(7, p-1) == 1     # d=7 OK in F_p
//! assert gcd(7, p^2-1) == 1   # d=7 OK in F_{p^2}
//! R.<x> = GF(p)[]; assert (x^2-11).is_irreducible()
//! ```

use p3_baby_bear::{BabyBear, MdsMatrixBabyBear};
use p3_field::PrimeField32;
use rand::RngExt;

use crate::pow_map::babybear::PowMap24;
use crate::rpo::babybear::SboxBB;

use super::XHash;

/// 3 rounds for XHash-BabyBear.
pub const XHASH_BB_ROUNDS: usize = 3;

/// XHash-BabyBear with x^7 over F_{p^2} extension S-box and Plonky3's native 24×24 MDS.
pub type XHashBabyBear = XHash<BabyBear, SboxBB, PowMap24, MdsMatrixBabyBear, 24>;

fn new_bb_from_rng<Mds: p3_mds::MdsPermutation<BabyBear, 24> + Default>(
    rng: &mut impl rand::Rng,
) -> XHash<BabyBear, SboxBB, PowMap24, Mds, 24> {
    let num_constants = (3 * XHASH_BB_ROUNDS + 1) * 24;
    let round_constants = (0..num_constants)
        .map(|_| BabyBear::new(rng.random::<u32>() % BabyBear::ORDER_U32))
        .collect();
    XHash::new_from_constants(XHASH_BB_ROUNDS, round_constants)
}

pub fn xhash_babybear(rng: &mut impl rand::Rng) -> XHashBabyBear {
    new_bb_from_rng(rng)
}

#[cfg(test)]
mod tests {
    use super::*;
    use p3_symmetric::Permutation;
    use rand::rngs::SmallRng;
    use rand::SeedableRng;

    #[test]
    fn xhash_babybear_deterministic() {
        let mut rng = SmallRng::seed_from_u64(1);
        let hash = xhash_babybear(&mut rng);
        let input: [BabyBear; 24] = core::array::from_fn(|i| BabyBear::new((i as u32 + 1) * 37));
        let out1 = hash.permute(input);
        let out2 = hash.permute(input);
        assert_eq!(out1, out2);
    }
}
