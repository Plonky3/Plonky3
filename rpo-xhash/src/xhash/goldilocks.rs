//! XHash-Goldilocks: 3 rounds, base d=7, extension X^7 over F_{p^3} (α³=2).
//!
//! Same `(FB)(E)(FB)(E)(FB)(E)(M)`-style construction as miden-crypto's `Rpx256`
//! permutation, but uses our binomial Fp³ (α³ = 2). For byte-exact compatibility
//! with `Rpx256` the trinomial irreducible `x³ - x - 1` would need to be wired
//! in — deferred.
//!
//! Sage:
//! ```sage
//! p = 2^64 - 2^32 + 1
//! assert gcd(7, p^3 - 1) == 1
//! R.<x> = GF(p)[]; assert (x^3 - 2).is_irreducible()
//! ```

use p3_goldilocks::Goldilocks;
use rand::RngExt;

use crate::mds_goldilocks::MdsBase12;
use crate::pow_map::goldilocks::PowMap12;
use crate::rpo::goldilocks::SboxGL;

use super::XHash;

/// 3 rounds for XHash-Goldilocks (matches miden-crypto's RPX256 round count).
pub const XHASH_GL_ROUNDS: usize = 3;

/// XHash-Goldilocks: 12-element state, x^7 / x^{1/7} base S-box, x^7 over Fp³
/// extension S-box, 12×12 FFT MDS matching miden-crypto's Rpx256 row.
pub type XHashGoldilocks = XHash<Goldilocks, SboxGL, PowMap12, MdsBase12, 12>;

pub fn xhash_goldilocks(rng: &mut impl rand::Rng) -> XHashGoldilocks {
    let num_constants = (3 * XHASH_GL_ROUNDS + 1) * 12;
    let round_constants = (0..num_constants)
        .map(|_| Goldilocks::new(rng.random::<u64>()))
        .collect();
    XHash::new_from_constants(XHASH_GL_ROUNDS, round_constants)
}

#[cfg(test)]
mod tests {
    use super::*;
    use p3_symmetric::Permutation;
    use rand::rngs::SmallRng;
    use rand::SeedableRng;

    #[test]
    fn xhash_goldilocks_deterministic() {
        let mut rng = SmallRng::seed_from_u64(1);
        let hash = xhash_goldilocks(&mut rng);
        let input: [Goldilocks; 12] =
            core::array::from_fn(|i| Goldilocks::new((i as u64 + 1) * 37));
        let out1 = hash.permute(input);
        let out2 = hash.permute(input);
        assert_eq!(out1, out2);
    }
}
