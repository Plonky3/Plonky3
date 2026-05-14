//! RPO-M31: S-boxes (x^5 and x^{1/5}) and MDS variants for Mersenne31.
//!
//! M31 p = 2^31 - 1 = 2147483647, p-1 = 2 * 3^2 * 7 * 11 * 31 * 151 * 331.
//! d = 5: gcd(5, p-1) = 1.
//! Inverse exponent: 5^{-1} mod (p-1) = 1717986917 = 0x66666665.
//!
//! Verified with Sage:
//! ```sage
//! p = 2^31 - 1                   # 2147483647
//! assert p.is_prime()
//! assert gcd(5, p - 1) == 1
//! e = inverse_mod(5, p - 1)      # 1717986917 = 0x66666665
//! assert (5 * e) % (p - 1) == 1
//! F = GF(p); x = F(123456789)
//! assert (x^5)^e == x            # roundtrip
//! ```
//!
//! On aarch64, the S-box uses NEON SIMD to process pairs of elements in
//! parallel via `vmull_u32` (2× u32→u64 widening multiply). The Mersenne
//! reduction (shift-and-mask) maps directly to NEON shift/and/add ops.
//!
//! Implements the RPO-M31 specification from "RPO-M31 and XHash-M31"
//! (Ashur & Tariq), with the paper's 24x24 truncated circulant MDS
//! (`RpoCirMds24`).

use p3_field::PrimeField32;
#[cfg(target_arch = "aarch64")]
use p3_field::{Field, PackedValue, PrimeCharacteristicRing};
use p3_mds::MdsPermutation;
use p3_mersenne_31::Mersenne31;
use p3_symmetric::Permutation;
use rand::RngExt;

use crate::reduce::m31::reduce;

use super::{RpoHash, RpoSbox};

/// Number of rounds for RPO-M31 (Section 3, 7 rounds = 14 steps + final).
pub const RPO_M31_ROUNDS: usize = 7;

// ============================================================
// S-box: x^5 (forward) and x^{1/5} (backward) over F_p
// ============================================================

/// RPO S-box for M31: x^5 forward, x^{1/5} backward.
#[derive(Debug, Copy, Clone, Default, Eq, PartialEq)]
pub struct SboxM31;

impl RpoSbox<Mersenne31, 24> for SboxM31 {
    #[inline(always)]
    fn forward(&self, state: &mut [Mersenne31; 24]) {
        apply_pow5(state);
    }

    #[inline(always)]
    fn backward(&self, state: &mut [Mersenne31; 24]) {
        apply_pow_inv5(state);
    }
}

/// x^5 for 24 M31 elements.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn apply_pow5(state: &mut [Mersenne31; 24]) {
    type P = <Mersenne31 as Field>::Packing;
    let mut p: [P; 6] = core::array::from_fn(|i| P::from_fn(|j| state[4 * i + j]));
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
fn apply_pow5(state: &mut [Mersenne31; 24]) {
    let mut raw = [0u32; 24];
    for i in 0..24 {
        raw[i] = state[i].as_canonical_u32();
    }
    pow5_scalar(&mut raw);
    for i in 0..24 {
        state[i] = Mersenne31::new(raw[i]);
    }
}

/// x^{1/5} = x^{1717986917} for 24 M31 elements.
///
/// Exponent 0x66666665. Addition chain exploiting the repeating hex `6`:
///   x -> x^2 -> x^3 -> x^5, x^6
///   then 6 × (shift-4, mul x^6)
///   final shift-4, mul x^5
///
/// Cost: 30 squarings + 9 multiplications per element.
/// On aarch64 NEON: processes 4 elements per instruction via PackedField.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn apply_pow_inv5(state: &mut [Mersenne31; 24]) {
    type P = <Mersenne31 as Field>::Packing;
    let x: [P; 6] = core::array::from_fn(|i| P::from_fn(|j| state[4 * i + j]));

    // Precompute x², x³, x⁵, x⁶.
    let x2: [P; 6] = core::array::from_fn(|i| x[i].square());
    let x3: [P; 6] = core::array::from_fn(|i| x2[i] * x[i]);
    let x5: [P; 6] = core::array::from_fn(|i| x3[i] * x2[i]);
    let x6: [P; 6] = core::array::from_fn(|i| x3[i].square());

    // acc = x^6
    let mut acc: [P; 6] = x6;

    // Build 0x66: shift-4 + mul x^6
    for _ in 0..4 {
        for i in 0..6 {
            acc[i] = acc[i].square();
        }
    }
    for i in 0..6 {
        acc[i] = acc[i] * x6[i];
    }

    // 0x666 through 0x6666666: 5 × (shift-4, mul x^6)
    for _ in 0..5 {
        for _ in 0..4 {
            for i in 0..6 {
                acc[i] = acc[i].square();
            }
        }
        for i in 0..6 {
            acc[i] = acc[i] * x6[i];
        }
    }

    // Final: shift-4 + mul x^5
    for _ in 0..4 {
        for i in 0..6 {
            acc[i] = acc[i].square();
        }
    }
    for i in 0..6 {
        acc[i] = acc[i] * x5[i];
    }

    for i in 0..6 {
        state[4 * i..4 * (i + 1)].copy_from_slice(acc[i].as_slice());
    }
}

#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
fn apply_pow_inv5(state: &mut [Mersenne31; 24]) {
    let mut raw = [0u32; 24];
    for i in 0..24 {
        raw[i] = state[i].as_canonical_u32();
    }
    pow_inv5_scalar(&mut raw);
    for i in 0..24 {
        state[i] = Mersenne31::new(raw[i]);
    }
}

// ============================================================
// Scalar fallback (non-aarch64)
// ============================================================

#[cfg(not(target_arch = "aarch64"))]
fn pow5_scalar(s: &mut [u32; 24]) {
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
    let x = *s;
    let mut x2 = [0u32; 24];
    for i in 0..24 {
        x2[i] = reduce(x[i] as u64 * x[i] as u64);
    }
    let mut x3 = [0u32; 24];
    for i in 0..24 {
        x3[i] = reduce(x2[i] as u64 * x[i] as u64);
    }
    let mut x5 = [0u32; 24];
    for i in 0..24 {
        x5[i] = reduce(x3[i] as u64 * x2[i] as u64);
    }
    let mut x6 = [0u32; 24];
    for i in 0..24 {
        x6[i] = reduce(x3[i] as u64 * x3[i] as u64);
    }
    let mut acc = x6;
    for _ in 0..4 {
        for i in 0..24 {
            acc[i] = reduce(acc[i] as u64 * acc[i] as u64);
        }
    }
    for i in 0..24 {
        acc[i] = reduce(acc[i] as u64 * x6[i] as u64);
    }
    for _ in 0..5 {
        for _ in 0..4 {
            for i in 0..24 {
                acc[i] = reduce(acc[i] as u64 * acc[i] as u64);
            }
        }
        for i in 0..24 {
            acc[i] = reduce(acc[i] as u64 * x6[i] as u64);
        }
    }
    for _ in 0..4 {
        for i in 0..24 {
            acc[i] = reduce(acc[i] as u64 * acc[i] as u64);
        }
    }
    for i in 0..24 {
        s[i] = reduce(acc[i] as u64 * x5[i] as u64);
    }
}

// ============================================================
// Paper's 24x24 MDS: truncated 32x32 circulant (Appendix A.3)
// ============================================================

/// First row of the 32x32 circulant MDS (Eq. 16 of the paper).
const CIRCULANT_ROW: [u32; 32] = [
    185870542, 2144994796, 1696461115, 215190769, 930115258, 766567118, 2003379079, 1770558586,
    1779722644, 434368282, 289154277, 1979813463, 1436360233, 1342944808, 63026005, 903393155,
    1512525948, 105409451, 1072974295, 979558870, 436105640, 2126764826, 1981550821, 636196459,
    645360517, 412540024, 1649351985, 1485803845, 53244687, 719457988, 270924307, 82564914,
];

/// Paper's 24x24 MDS for RPO-M31 (truncated 32x32 circulant).
///
/// M\[i\]\[j\] = CIRCULANT_ROW\[(j - i) mod 32\] for 0 <= i,j < 24.
#[derive(Debug, Copy, Clone, Default, Eq, PartialEq)]
pub struct RpoCirMds24;

impl Permutation<[Mersenne31; 24]> for RpoCirMds24 {
    #[inline(always)]
    fn permute_mut(&self, state: &mut [Mersenne31; 24]) {
        let input: [u32; 24] = core::array::from_fn(|i| state[i].as_canonical_u32());
        for i in 0..24 {
            let mut sum = 0u64;
            for j in 0..24 {
                let idx = ((j as isize - i as isize).rem_euclid(32)) as usize;
                let prod = input[j] as u64 * CIRCULANT_ROW[idx] as u64;
                sum += reduce(prod) as u64;
            }
            state[i] = Mersenne31::new(reduce(sum));
        }
    }
}

impl MdsPermutation<Mersenne31, 24> for RpoCirMds24 {}

// ============================================================
// Hash type aliases and factory functions
// ============================================================

/// RPO-M31 with the paper's circulant MDS.
pub type RpoM31Cir = RpoHash<Mersenne31, SboxM31, RpoCirMds24, 24>;

/// RPO-M31 with BabyBear's MDS column (for perf comparison; see `crate::mds_m31_bb`).
pub type RpoM31BBMds = RpoHash<Mersenne31, SboxM31, crate::mds_m31_bb::Mds24M31BBCol, 24>;

fn new_from_rng<Mds: MdsPermutation<Mersenne31, 24> + Default>(
    num_rounds: usize,
    rng: &mut impl rand::Rng,
) -> RpoHash<Mersenne31, SboxM31, Mds, 24> {
    let num_constants = (2 * num_rounds + 1) * 24;
    let round_constants = (0..num_constants)
        .map(|_| Mersenne31::new(rng.random::<u32>() % Mersenne31::ORDER_U32))
        .collect();
    RpoHash::new_from_constants(num_rounds, round_constants)
}

/// RPO-M31 with paper's circulant MDS, 7 rounds.
pub fn rpo_m31_cir(rng: &mut impl rand::Rng) -> RpoM31Cir {
    new_from_rng(RPO_M31_ROUNDS, rng)
}

/// RPO-M31 with BabyBear's MDS column, 7 rounds.
pub fn rpo_m31_bb_mds(rng: &mut impl rand::Rng) -> RpoM31BBMds {
    new_from_rng(RPO_M31_ROUNDS, rng)
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use p3_symmetric::Permutation;
    use rand::rngs::SmallRng;
    use rand::SeedableRng;

    #[test]
    fn pow5_roundtrip() {
        let mut rng = SmallRng::seed_from_u64(42);
        let state: [Mersenne31; 24] =
            core::array::from_fn(|_| Mersenne31::new(rng.random::<u32>() % Mersenne31::ORDER_U32));
        let mut s = state;
        apply_pow5(&mut s);
        assert_ne!(s, state, "pow5 should change state");
        apply_pow_inv5(&mut s);
        assert_eq!(s, state, "pow5 then pow_inv5 should roundtrip");
    }

    #[test]
    fn pow_inv5_roundtrip() {
        let mut rng = SmallRng::seed_from_u64(99);
        let state: [Mersenne31; 24] =
            core::array::from_fn(|_| Mersenne31::new(rng.random::<u32>() % Mersenne31::ORDER_U32));
        let mut s = state;
        apply_pow_inv5(&mut s);
        assert_ne!(s, state, "pow_inv5 should change state");
        apply_pow5(&mut s);
        assert_eq!(s, state, "pow_inv5 then pow5 should roundtrip");
    }

    /// Known-answer test: Sage: `GF(2^31-1)(123456789)^inverse_mod(5, 2^31-2)`
    #[test]
    fn pow_inv5_known_answer() {
        let mut state: [Mersenne31; 24] = core::array::from_fn(|i| {
            Mersenne31::new((i as u32 + 1) * 123456789 % Mersenne31::ORDER_U32)
        });
        let x0 = state[0]; // = 123456789
        apply_pow_inv5(&mut state);
        assert_eq!(x0, Mersenne31::new(123456789));
        assert_eq!(state[0], Mersenne31::new(1401007331));
    }

    #[test]
    fn rpo_m31_cir_deterministic() {
        let mut rng = SmallRng::seed_from_u64(1);
        let hash = rpo_m31_cir(&mut rng);
        let input: [Mersenne31; 24] =
            core::array::from_fn(|i| Mersenne31::new((i as u32 + 1) * 37));
        let out1 = hash.permute(input);
        let out2 = hash.permute(input);
        assert_eq!(out1, out2);
    }

    #[test]
    fn rpo_m31_bb_mds_deterministic() {
        let mut rng = SmallRng::seed_from_u64(1);
        let hash = rpo_m31_bb_mds(&mut rng);
        let input: [Mersenne31; 24] =
            core::array::from_fn(|i| Mersenne31::new((i as u32 + 1) * 37));
        let out1 = hash.permute(input);
        let out2 = hash.permute(input);
        assert_eq!(out1, out2);
    }

    #[test]
    fn cir_mds_changes_state() {
        let mut state: [Mersenne31; 24] =
            core::array::from_fn(|i| Mersenne31::new((i as u32 + 1) * 7));
        let orig = state;
        RpoCirMds24.permute_mut(&mut state);
        assert_ne!(state, orig);
    }
}
