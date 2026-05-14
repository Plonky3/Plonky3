//! RPO for BabyBear: x^7 / x^{1/7} S-boxes over F_p.
//!
//! BabyBear p = 15 * 2^27 + 1 = 2013265921, p-1 = 2^27 * 3 * 5.
//! d = 5 does NOT work: gcd(5, p-1) = 5.
//! d = 7 is the smallest prime with gcd(7, p-1) = 1.
//! Inverse exponent: 7^{-1} mod (p-1) = 1725656503 = 0x66DB6DB7.
//!
//! Verified with Sage:
//! ```sage
//! p = 15 * 2^27 + 1              # 2013265921
//! assert p.is_prime()
//! assert gcd(5, p - 1) == 5      # d=5 fails
//! assert gcd(7, p - 1) == 1      # d=7 works
//! e = inverse_mod(7, p - 1)      # 1725656503 = 0x66DB6DB7
//! assert (7 * e) % (p - 1) == 1
//! F = GF(p); x = F(123456789)
//! assert (x^7)^e == x            # roundtrip
//! ```

use p3_baby_bear::{BabyBear, MdsMatrixBabyBear};
use p3_field::PrimeField32;
#[cfg(target_arch = "aarch64")]
use p3_field::{Field, PackedValue, PrimeCharacteristicRing};
use rand::RngExt;

#[cfg(not(target_arch = "aarch64"))]
use crate::reduce::monty31::from_raw_monty_u32;
#[cfg(not(target_arch = "aarch64"))]
use crate::reduce::monty31::monty_reduce_bb;

use super::{RpoHash, RpoSbox};

/// 7 rounds (same as RPO-M31 for comparable security at similar field size).
pub const RPO_BB_ROUNDS: usize = 7;

/// RPO S-box for BabyBear: x^7 forward, x^{1/7} backward.
#[derive(Debug, Copy, Clone, Default, Eq, PartialEq)]
pub struct SboxBB;

impl RpoSbox<BabyBear, 24> for SboxBB {
    #[inline(always)]
    fn forward(&self, state: &mut [BabyBear; 24]) {
        apply_pow7(state);
    }

    #[inline(always)]
    fn backward(&self, state: &mut [BabyBear; 24]) {
        apply_pow_inv7(state);
    }
}

/// x^7 for 24 BabyBear elements.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn apply_pow7(state: &mut [BabyBear; 24]) {
    type P = <BabyBear as Field>::Packing;
    let mut p: [P; 6] = core::array::from_fn(|i| P::from_fn(|j| state[4 * i + j]));
    // x^7 = (x²·x)²·x
    for s in p.iter_mut() {
        let x2 = s.square();
        let x3 = x2 * *s;
        let x6 = x3.square();
        *s = x6 * *s;
    }
    for i in 0..6 {
        state[4 * i..4 * (i + 1)].copy_from_slice(p[i].as_slice());
    }
}

#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
fn apply_pow7(state: &mut [BabyBear; 24]) {
    let mut raw = [0u32; 24];
    for i in 0..24 {
        raw[i] = state[i].to_unique_u32();
    }
    pow7_scalar(&mut raw);
    for i in 0..24 {
        state[i] = from_raw_monty_u32(raw[i]);
    }
}

/// x^{1/7} = x^{1725656503} for 24 BabyBear elements (operates on raw Montgomery residues).
///
/// Exponent 0x66DB6DB7 = 0o14_66666666_7 (octal): digit '14' prefix,
/// eight repeating '6's, final '7'.
///
/// Addition chain (octal digit-at-a-time):
///   precomp: x^2, x^3, x^6, x^7, x^12
///   acc = x^12                    (= 0o14)
///   8 × (shift-3 + mul x^6)      (append eight octal 6's)
///   1 × (shift-3 + mul x^7)      (append final 7)
///
/// Sage: `assert int('14666666667', 8) == inverse_mod(7, 2^27*15)`
///
/// Cost: 30 squarings + 11 multiplications = 41 ops per element.
/// On aarch64 NEON: processes 4 elements per instruction via PackedField.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn apply_pow_inv7(state: &mut [BabyBear; 24]) {
    type P = <BabyBear as Field>::Packing;
    let mut x: [P; 6] = core::array::from_fn(|i| P::from_fn(|j| state[4 * i + j]));

    // Precompute x², x³, x⁶, x⁷, then acc = x¹².
    let x2: [P; 6] = core::array::from_fn(|i| x[i].square());
    let x3: [P; 6] = core::array::from_fn(|i| x2[i] * x[i]);
    let x6: [P; 6] = core::array::from_fn(|i| x3[i].square());
    let x7: [P; 6] = core::array::from_fn(|i| x6[i] * x[i]);
    let _ = &mut x; // silence "mut not needed"

    let mut acc: [P; 6] = core::array::from_fn(|i| x6[i].square()); // x^12 = 0o14

    // Append eight octal '6's: 8 × (shift-3 + mul x⁶)
    for _ in 0..8 {
        for _ in 0..3 {
            for i in 0..6 {
                acc[i] = acc[i].square();
            }
        }
        for i in 0..6 {
            acc[i] = acc[i] * x6[i];
        }
    }

    // Final octal '7': shift-3 + mul x⁷
    for _ in 0..3 {
        for i in 0..6 {
            acc[i] = acc[i].square();
        }
    }
    for i in 0..6 {
        acc[i] = acc[i] * x7[i];
    }

    for i in 0..6 {
        state[4 * i..4 * (i + 1)].copy_from_slice(acc[i].as_slice());
    }
}

#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
fn apply_pow_inv7(state: &mut [BabyBear; 24]) {
    let mut raw = [0u32; 24];
    for i in 0..24 {
        raw[i] = state[i].to_unique_u32();
    }
    pow_inv7_scalar(&mut raw);
    for i in 0..24 {
        state[i] = from_raw_monty_u32(raw[i]);
    }
}

// ============================================================
// Scalar fallback (non-aarch64)
// ============================================================

#[cfg(not(target_arch = "aarch64"))]
fn pow7_scalar(s: &mut [u32; 24]) {
    let reduce = monty_reduce_bb;
    let x = *s;

    let mut x2 = [0u32; 24];
    for i in 0..24 {
        x2[i] = reduce(x[i] as u64 * x[i] as u64);
    }
    let mut x3 = [0u32; 24];
    for i in 0..24 {
        x3[i] = reduce(x2[i] as u64 * x[i] as u64);
    }
    let mut x6 = [0u32; 24];
    for i in 0..24 {
        x6[i] = reduce(x3[i] as u64 * x3[i] as u64);
    }
    for i in 0..24 {
        s[i] = reduce(x6[i] as u64 * x[i] as u64);
    }
}

#[cfg(not(target_arch = "aarch64"))]
fn pow_inv7_scalar(s: &mut [u32; 24]) {
    let reduce = monty_reduce_bb;
    let x = *s;

    let mut x2 = [0u32; 24];
    for i in 0..24 {
        x2[i] = reduce(x[i] as u64 * x[i] as u64);
    }
    let mut x3 = [0u32; 24];
    for i in 0..24 {
        x3[i] = reduce(x2[i] as u64 * x[i] as u64);
    }
    let mut x6 = [0u32; 24];
    for i in 0..24 {
        x6[i] = reduce(x3[i] as u64 * x3[i] as u64);
    }
    let mut x7 = [0u32; 24];
    for i in 0..24 {
        x7[i] = reduce(x6[i] as u64 * x[i] as u64);
    }
    let mut acc = [0u32; 24];
    for i in 0..24 {
        acc[i] = reduce(x6[i] as u64 * x6[i] as u64);
    }
    // acc = x^12 = 0o14
    for _ in 0..8 {
        for _ in 0..3 {
            for i in 0..24 {
                acc[i] = reduce(acc[i] as u64 * acc[i] as u64);
            }
        }
        for i in 0..24 {
            acc[i] = reduce(acc[i] as u64 * x6[i] as u64);
        }
    }
    for _ in 0..3 {
        for i in 0..24 {
            acc[i] = reduce(acc[i] as u64 * acc[i] as u64);
        }
    }
    for i in 0..24 {
        s[i] = reduce(acc[i] as u64 * x7[i] as u64);
    }
}

// ============================================================
// Hash type alias and factory
// ============================================================

/// RPO-BabyBear with Plonky3's native 24×24 circulant MDS.
pub type RpoBabyBear = RpoHash<BabyBear, SboxBB, MdsMatrixBabyBear, 24>;

fn new_bb_from_rng<Mds: p3_mds::MdsPermutation<BabyBear, 24> + Default>(
    rng: &mut impl rand::Rng,
) -> RpoHash<BabyBear, SboxBB, Mds, 24> {
    let num_constants = (2 * RPO_BB_ROUNDS + 1) * 24;
    let round_constants = (0..num_constants)
        .map(|_| BabyBear::new(rng.random::<u32>() % BabyBear::ORDER_U32))
        .collect();
    RpoHash::new_from_constants(RPO_BB_ROUNDS, round_constants)
}

/// RPO-BabyBear with Plonky3's native 24×24 MDS, 7 rounds.
pub fn rpo_babybear(rng: &mut impl rand::Rng) -> RpoBabyBear {
    new_bb_from_rng(rng)
}

#[cfg(test)]
mod tests {
    use super::*;
    use p3_symmetric::Permutation;
    use rand::rngs::SmallRng;
    use rand::SeedableRng;

    #[test]
    fn pow7_roundtrip() {
        let mut rng = SmallRng::seed_from_u64(42);
        let state: [BabyBear; 24] =
            core::array::from_fn(|_| BabyBear::new(rng.random::<u32>() % BabyBear::ORDER_U32));
        let mut s = state;
        apply_pow7(&mut s);
        assert_ne!(s, state, "pow7 should change state");
        apply_pow_inv7(&mut s);
        assert_eq!(s, state, "pow7 then pow_inv7 should roundtrip");
    }

    #[test]
    fn pow_inv7_roundtrip() {
        let mut rng = SmallRng::seed_from_u64(99);
        let state: [BabyBear; 24] =
            core::array::from_fn(|_| BabyBear::new(rng.random::<u32>() % BabyBear::ORDER_U32));
        let mut s = state;
        apply_pow_inv7(&mut s);
        assert_ne!(s, state, "pow_inv7 should change state");
        apply_pow7(&mut s);
        assert_eq!(s, state, "pow_inv7 then pow7 should roundtrip");
    }

    /// Known-answer test: Sage: `GF(2013265921)(123456789)^inverse_mod(7, 2013265920)`
    #[test]
    fn pow_inv7_known_answer() {
        let mut state = [BabyBear::new(0); 24];
        state[0] = BabyBear::new(123456789);
        apply_pow_inv7(&mut state);
        assert_eq!(state[0], BabyBear::new(187825366));
    }

    #[test]
    fn rpo_babybear_deterministic() {
        let mut rng = SmallRng::seed_from_u64(1);
        let hash = rpo_babybear(&mut rng);
        let input: [BabyBear; 24] = core::array::from_fn(|i| BabyBear::new((i as u32 + 1) * 37));
        let out1 = hash.permute(input);
        let out2 = hash.permute(input);
        assert_eq!(out1, out2);
    }
}
