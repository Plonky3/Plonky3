//! XHash-KoalaBear: 3 rounds, base d=5, extension X^3 over F_{p^3} (α³+α+4=0).
//!
//! d=3 is the cheapest possible extension S-box (1 sqr + 1 mul per triplet).
//!
//! Sage:
//! ```sage
//! p = 2130706433
//! assert gcd(3, p^3-1) == 1
//! R.<x> = GF(p)[]; assert (x^3+x+4).is_irreducible()
//! ```

use p3_field::PrimeField32;
use p3_koala_bear::{KoalaBear, MdsMatrixKoalaBear};
use rand::RngExt;

use crate::pow_map::koalabear::PowMap24;
use crate::rpo::koalabear::SboxKB;

use super::XHash;

/// 3 rounds for XHash-KoalaBear.
pub const XHASH_KB_ROUNDS: usize = 3;

/// XHash-KoalaBear with Plonky3's native 24×24 Karatsuba MDS.
pub type XHashKoalaBear = XHash<KoalaBear, SboxKB, PowMap24, MdsMatrixKoalaBear, 24>;

fn new_kb_from_rng<Mds: p3_mds::MdsPermutation<KoalaBear, 24> + Default>(
    rng: &mut impl rand::Rng,
) -> XHash<KoalaBear, SboxKB, PowMap24, Mds, 24> {
    let num_constants = (3 * XHASH_KB_ROUNDS + 1) * 24;
    let round_constants = (0..num_constants)
        .map(|_| KoalaBear::new(rng.random::<u32>() % KoalaBear::ORDER_U32))
        .collect();
    XHash::new_from_constants(XHASH_KB_ROUNDS, round_constants)
}

pub fn xhash_koalabear(rng: &mut impl rand::Rng) -> XHashKoalaBear {
    new_kb_from_rng(rng)
}

#[cfg(test)]
mod tests {
    use super::*;
    use p3_symmetric::Permutation;
    use rand::rngs::SmallRng;
    use rand::SeedableRng;

    #[test]
    fn xhash_koalabear_deterministic() {
        let mut rng = SmallRng::seed_from_u64(1);
        let hash = xhash_koalabear(&mut rng);
        let input: [KoalaBear; 24] = core::array::from_fn(|i| KoalaBear::new((i as u32 + 1) * 37));
        let out1 = hash.permute(input);
        let out2 = hash.permute(input);
        assert_eq!(out1, out2);
    }
}
