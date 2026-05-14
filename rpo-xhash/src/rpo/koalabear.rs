//! RPO for KoalaBear: x^5 / x^{1/5} S-boxes over F_p.
//!
//! KoalaBear p = 2^31 - 2^24 + 1 = 2130706433, p-1 = 2^24 * 127.
//! d = 5 works because gcd(5, p-1) = 1  (127 mod 5 = 2).
//! Inverse exponent: 5^{-1} mod (p-1) = 852282573 = 0x32CCCCCD.
//!
//! Verified with Sage:
//! ```sage
//! p = 2^31 - 2^24 + 1          # 2130706433
//! assert p.is_prime()
//! assert gcd(5, p - 1) == 1
//! e = inverse_mod(5, p - 1)    # 852282573 = 0x32CCCCCD
//! assert (5 * e) % (p - 1) == 1
//! F = GF(p); x = F(123456789)
//! assert (x^5)^e == x          # roundtrip
//! # 2-bit pairs from MSB:
//! # bin(e) = '110010110011001100110011001101'
//! # pairs: [3, 0, 2, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 1]
//! ```

use p3_field::PrimeField32;
#[cfg(target_arch = "aarch64")]
use p3_field::{Field, PackedValue, PrimeCharacteristicRing};
use p3_koala_bear::{KoalaBear, MdsMatrixKoalaBear};
use rand::RngExt;

#[cfg(not(target_arch = "aarch64"))]
use crate::reduce::monty31::from_raw_monty_u32;
#[cfg(not(target_arch = "aarch64"))]
use crate::reduce::monty31::monty_reduce_kb;

use super::{RpoHash, RpoSbox};

/// 7 rounds (same as RPO-M31 for comparable security at similar field size).
pub const RPO_KB_ROUNDS: usize = 7;

/// RPO S-box for KoalaBear: x^5 forward, x^{1/5} backward.
#[derive(Debug, Copy, Clone, Default, Eq, PartialEq)]
pub struct SboxKB;

impl RpoSbox<KoalaBear, 24> for SboxKB {
    #[inline(always)]
    fn forward(&self, state: &mut [KoalaBear; 24]) {
        apply_pow5(state);
    }

    #[inline(always)]
    fn backward(&self, state: &mut [KoalaBear; 24]) {
        apply_pow_inv5(state);
    }
}

/// x^5 for 24 KoalaBear elements.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn apply_pow5(state: &mut [KoalaBear; 24]) {
    type P = <KoalaBear as Field>::Packing;
    let mut p: [P; 6] = core::array::from_fn(|i| P::from_fn(|j| state[4 * i + j]));
    // x^5 = (x²)²·x
    for s in p.iter_mut() {
        let x2 = s.square();
        let x4 = x2.square();
        *s = x4 * *s;
    }
    for i in 0..6 {
        state[4 * i..4 * (i + 1)].copy_from_slice(p[i].as_slice());
    }
}

#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
fn apply_pow5(state: &mut [KoalaBear; 24]) {
    let mut raw = [0u32; 24];
    for i in 0..24 {
        raw[i] = state[i].to_unique_u32();
    }
    pow5_scalar(&mut raw);
    for i in 0..24 {
        state[i] = from_raw_monty_u32(raw[i]);
    }
}

/// x^{1/5} = x^{852282573} for 24 KoalaBear elements (operates on raw Montgomery residues).
///
/// Exponent 0x32CCCCCD (30 bits). 2-bit pairs from MSB:
///   [3, 0, 2, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 1]
///
/// Six zero-pairs mean no multiply, and pairs 5-12 form a repeating
/// [3, 0] pattern (shift-2 + mul x^3 per pair).
///
/// Cost: 29 squarings + 9 multiplications = 38 ops per element.
/// On aarch64 NEON: processes 4 elements per instruction via PackedField.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn apply_pow_inv5(state: &mut [KoalaBear; 24]) {
    type P = <KoalaBear as Field>::Packing;
    let x: [P; 6] = core::array::from_fn(|i| P::from_fn(|j| state[4 * i + j]));
    let x2: [P; 6] = core::array::from_fn(|i| x[i].square());
    let x3: [P; 6] = core::array::from_fn(|i| x2[i] * x[i]);

    // acc = x³ (pair 0 = 3)
    let mut acc: [P; 6] = x3;

    // pair 1 = 0
    for _ in 0..2 {
        for i in 0..6 {
            acc[i] = acc[i].square();
        }
    }
    // pair 2 = 2
    for _ in 0..2 {
        for i in 0..6 {
            acc[i] = acc[i].square();
        }
    }
    for i in 0..6 {
        acc[i] = acc[i] * x2[i];
    }
    // pair 3 = 3
    for _ in 0..2 {
        for i in 0..6 {
            acc[i] = acc[i].square();
        }
    }
    for i in 0..6 {
        acc[i] = acc[i] * x3[i];
    }
    // pair 4 = 0
    for _ in 0..2 {
        for i in 0..6 {
            acc[i] = acc[i].square();
        }
    }
    // pairs 5-12: [3, 0] × 4
    for _ in 0..4 {
        for _ in 0..2 {
            for i in 0..6 {
                acc[i] = acc[i].square();
            }
        }
        for i in 0..6 {
            acc[i] = acc[i] * x3[i];
        }
        for _ in 0..2 {
            for i in 0..6 {
                acc[i] = acc[i].square();
            }
        }
    }
    // pair 13 = 3
    for _ in 0..2 {
        for i in 0..6 {
            acc[i] = acc[i].square();
        }
    }
    for i in 0..6 {
        acc[i] = acc[i] * x3[i];
    }
    // pair 14 = 1
    for _ in 0..2 {
        for i in 0..6 {
            acc[i] = acc[i].square();
        }
    }
    for i in 0..6 {
        acc[i] = acc[i] * x[i];
    }

    for i in 0..6 {
        state[4 * i..4 * (i + 1)].copy_from_slice(acc[i].as_slice());
    }
}

#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
fn apply_pow_inv5(state: &mut [KoalaBear; 24]) {
    let mut raw = [0u32; 24];
    for i in 0..24 {
        raw[i] = state[i].to_unique_u32();
    }
    pow_inv5_scalar(&mut raw);
    for i in 0..24 {
        state[i] = from_raw_monty_u32(raw[i]);
    }
}

// ============================================================
// Scalar fallback (non-aarch64)
// ============================================================

#[cfg(not(target_arch = "aarch64"))]
fn pow5_scalar(s: &mut [u32; 24]) {
    let reduce = monty_reduce_kb;
    let x = *s;
    let mut x2 = [0u32; 24];
    for i in 0..24 {
        x2[i] = reduce(x[i] as u64 * x[i] as u64);
    }
    let mut x4 = [0u32; 24];
    for i in 0..24 {
        x4[i] = reduce(x2[i] as u64 * x2[i] as u64);
    }
    for i in 0..24 {
        s[i] = reduce(x4[i] as u64 * x[i] as u64);
    }
}

#[cfg(not(target_arch = "aarch64"))]
fn pow_inv5_scalar(s: &mut [u32; 24]) {
    let reduce = monty_reduce_kb;
    let x = *s;
    let mut x2 = [0u32; 24];
    for i in 0..24 {
        x2[i] = reduce(x[i] as u64 * x[i] as u64);
    }
    let mut x3 = [0u32; 24];
    for i in 0..24 {
        x3[i] = reduce(x2[i] as u64 * x[i] as u64);
    }
    let mut acc = x3;
    // pair 1 = 0
    for _ in 0..2 {
        for i in 0..24 {
            acc[i] = reduce(acc[i] as u64 * acc[i] as u64);
        }
    }
    // pair 2 = 2: shift-2 + mul x^2
    for _ in 0..2 {
        for i in 0..24 {
            acc[i] = reduce(acc[i] as u64 * acc[i] as u64);
        }
    }
    for i in 0..24 {
        acc[i] = reduce(acc[i] as u64 * x2[i] as u64);
    }
    // pair 3 = 3
    for _ in 0..2 {
        for i in 0..24 {
            acc[i] = reduce(acc[i] as u64 * acc[i] as u64);
        }
    }
    for i in 0..24 {
        acc[i] = reduce(acc[i] as u64 * x3[i] as u64);
    }
    // pair 4 = 0
    for _ in 0..2 {
        for i in 0..24 {
            acc[i] = reduce(acc[i] as u64 * acc[i] as u64);
        }
    }
    // pairs 5-12: [3, 0] × 4
    for _ in 0..4 {
        for _ in 0..2 {
            for i in 0..24 {
                acc[i] = reduce(acc[i] as u64 * acc[i] as u64);
            }
        }
        for i in 0..24 {
            acc[i] = reduce(acc[i] as u64 * x3[i] as u64);
        }
        for _ in 0..2 {
            for i in 0..24 {
                acc[i] = reduce(acc[i] as u64 * acc[i] as u64);
            }
        }
    }
    // pair 13 = 3
    for _ in 0..2 {
        for i in 0..24 {
            acc[i] = reduce(acc[i] as u64 * acc[i] as u64);
        }
    }
    for i in 0..24 {
        acc[i] = reduce(acc[i] as u64 * x3[i] as u64);
    }
    // pair 14 = 1
    for _ in 0..2 {
        for i in 0..24 {
            acc[i] = reduce(acc[i] as u64 * acc[i] as u64);
        }
    }
    for i in 0..24 {
        s[i] = reduce(acc[i] as u64 * x[i] as u64);
    }
}

// ============================================================
// Hash type alias and factory
// ============================================================

/// RPO-KoalaBear with Plonky3's native 24×24 circulant MDS.
pub type RpoKoalaBear = RpoHash<KoalaBear, SboxKB, MdsMatrixKoalaBear, 24>;

fn new_kb_from_rng<Mds: p3_mds::MdsPermutation<KoalaBear, 24> + Default>(
    rng: &mut impl rand::Rng,
) -> RpoHash<KoalaBear, SboxKB, Mds, 24> {
    let num_constants = (2 * RPO_KB_ROUNDS + 1) * 24;
    let round_constants = (0..num_constants)
        .map(|_| KoalaBear::new(rng.random::<u32>() % KoalaBear::ORDER_U32))
        .collect();
    RpoHash::new_from_constants(RPO_KB_ROUNDS, round_constants)
}

/// RPO-KoalaBear with Plonky3's native 24×24 MDS, 7 rounds.
pub fn rpo_koalabear(rng: &mut impl rand::Rng) -> RpoKoalaBear {
    new_kb_from_rng(rng)
}

#[cfg(test)]
mod tests {
    use super::*;
    use p3_symmetric::Permutation;
    use rand::rngs::SmallRng;
    use rand::SeedableRng;

    #[test]
    fn pow5_roundtrip() {
        let mut rng = SmallRng::seed_from_u64(42);
        let state: [KoalaBear; 24] =
            core::array::from_fn(|_| KoalaBear::new(rng.random::<u32>() % KoalaBear::ORDER_U32));
        let mut s = state;
        apply_pow5(&mut s);
        assert_ne!(s, state, "pow5 should change state");
        apply_pow_inv5(&mut s);
        assert_eq!(s, state, "pow5 then pow_inv5 should roundtrip");
    }

    #[test]
    fn pow_inv5_roundtrip() {
        let mut rng = SmallRng::seed_from_u64(99);
        let state: [KoalaBear; 24] =
            core::array::from_fn(|_| KoalaBear::new(rng.random::<u32>() % KoalaBear::ORDER_U32));
        let mut s = state;
        apply_pow_inv5(&mut s);
        assert_ne!(s, state, "pow_inv5 should change state");
        apply_pow5(&mut s);
        assert_eq!(s, state, "pow_inv5 then pow5 should roundtrip");
    }

    /// Known-answer test: Sage: `GF(2130706433)(123456789)^inverse_mod(5, 2130706432)`
    #[test]
    fn pow_inv5_known_answer() {
        let mut state = [KoalaBear::new(0); 24];
        state[0] = KoalaBear::new(123456789);
        apply_pow_inv5(&mut state);
        assert_eq!(state[0], KoalaBear::new(864749829));
    }

    #[test]
    fn rpo_koalabear_deterministic() {
        let mut rng = SmallRng::seed_from_u64(1);
        let hash = rpo_koalabear(&mut rng);
        let input: [KoalaBear; 24] = core::array::from_fn(|i| KoalaBear::new((i as u32 + 1) * 37));
        let out1 = hash.permute(input);
        let out2 = hash.permute(input);
        assert_eq!(out1, out2);
    }
}
