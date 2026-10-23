//! The byte-sliced moves, written once over the register.
//!
//! Butterflies of two-register byte permutes move between row-major values and byte-sliced registers.
//!
//! One affine instruction applies an `8 x 8` bit matrix to 64 input bytes.

use super::BLOCK;
use super::lanes::SliceRegister;
use crate::Poly192;

/// Bytes of a challenge-field element.
pub const OUT_BYTES: usize = 24;

/// Every byte offset in place.
pub const IDENTITY: [u8; 64] = {
    let mut sigma = [0u8; 64];
    let mut f = 0;
    while f < 64 {
        sigma[f] = f as u8;
        f += 1;
    }
    sigma
};

/// Byte offset `(o, p bits 0..3)` to quadwords.
const TO_QWORDS: [u8; 64] = {
    let mut sigma = [0u8; 64];
    let mut f = 0;
    while f < 64 {
        sigma[f] = ((f >> 3) | ((f & 7) << 3)) as u8;
        f += 1;
    }
    sigma
};

/// One butterfly stage: registers `z` and `z | zbit` exchange bytes through two permutes.
#[derive(Clone, Copy, Debug)]
pub struct Stage {
    /// Byte sources of the low register of each pair; bit 6 picks the high one.
    lo: [u8; 64],
    /// Byte sources of the high register of each pair.
    hi: [u8; 64],
}

impl Stage {
    /// Swap byte-offset bit `fbit` with the stage's register bit, then permute each output by `sigma`.
    pub const fn swap(fbit: usize, sigma: &[u8; 64]) -> Self {
        let (mut lo, mut hi) = ([0u8; 64], [0u8; 64]);
        let mut f = 0;
        while f < 64 {
            let g = sigma[f] as usize;
            // A byte keeps its offset except bit `fbit`, which trades places with the register bit.
            lo[f] = (if g & fbit == 0 { g } else { 64 | (g ^ fbit) }) as u8;
            hi[f] = (if g & fbit == 0 { g | fbit } else { 64 | g }) as u8;
            f += 1;
        }
        Self { lo, hi }
    }

    /// Exchange bytes between registers `z` and `z | zbit`, for every `z` with that bit clear.
    #[inline(always)]
    pub fn apply<R: SliceRegister, const N: usize>(&self, zbit: usize, regs: &mut [R; N]) {
        // SAFETY: both index arrays are 64 bytes.
        let (lo, hi) = unsafe { (R::load(self.lo.as_ptr()), R::load(self.hi.as_ptr())) };
        for k in 0..N / 2 {
            // Pair `k` with a zero inserted at bit `zbit`.
            let z = (k & (zbit - 1)) | ((k & !(zbit - 1)) << 1);
            let (u, w) = (regs[z], regs[z | zbit]);
            regs[z] = u.permute2(lo, w);
            regs[z | zbit] = u.permute2(hi, w);
        }
    }
}

/// Stages taking eight registers of one coefficient, register = byte `o` and offset = value `p`, to quadwords.
///
/// Register `k`, quadword `l` then holds the coefficient of value `8k + l`.
const OUTPUT_STAGES: [Stage; 3] = [
    Stage::swap(8, &IDENTITY),
    Stage::swap(16, &IDENTITY),
    Stage::swap(32, &TO_QWORDS),
];

/// Quadword sources interleaving the first two coefficients of eight values; a zero is filled from the third.
const INTERLEAVE01: [[u64; 8]; 3] = [
    [0, 8, 0, 1, 9, 0, 2, 10],
    [0, 3, 11, 0, 4, 12, 0, 5],
    [13, 0, 6, 14, 0, 7, 15, 0],
];

/// Quadword sources of the third coefficient.
const INTERLEAVE2: [[u64; 8]; 3] = [
    [0, 0, 0, 0, 0, 1, 0, 0],
    [2, 0, 0, 3, 0, 0, 4, 0],
    [0, 5, 0, 0, 6, 0, 0, 7],
];

/// Quadwords of each output register the third coefficient fills.
const INTERLEAVE2_MASK: [u8; 3] = [0b0010_0100, 0b0100_1001, 0b1001_0010];

/// Store 24 registers of output bytes, register `o` holding byte `o` of 64 values, as those values.
#[inline(always)]
pub fn store_f192<R: SliceRegister>(acc: &[R; OUT_BYTES], out: &mut [Poly192; BLOCK]) {
    // Per coefficient, eight registers of output bytes become eight registers of quadwords.
    let mut c: [[R; 8]; 3] = core::array::from_fn(|i| core::array::from_fn(|o| acc[8 * i + o]));
    for (s, stage) in OUTPUT_STAGES.iter().enumerate() {
        for coefficient in &mut c {
            stage.apply(1 << s, coefficient);
        }
    }

    // Interleave the three coefficients of values 8k..8k+8 into 24 consecutive quadwords.
    // SAFETY: each index array is 64 bytes.
    let (idx01, idx2): ([R; 3], [R; 3]) = unsafe {
        (
            INTERLEAVE01.map(|i| R::load(i.as_ptr().cast())),
            INTERLEAVE2.map(|i| R::load(i.as_ptr().cast())),
        )
    };
    let dst = out.as_mut_ptr().cast::<u8>();
    let [c0, c1, c2] = c;
    for (k, ((c0, c1), c2)) in c0.into_iter().zip(c1).zip(c2).enumerate() {
        for i in 0..3 {
            let v = c0.permute2_qwords(idx01[i], c1).permute_merge_qwords(
                INTERLEAVE2_MASK[i],
                idx2[i],
                c2,
            );
            // SAFETY: an element is three contiguous quadwords, so the stores of group k cover out[8k..8k+8].
            unsafe { R::store(dst.add(192 * k + 64 * i), v) };
        }
    }
}

/// Byte sources gathering byte `o` of eight weights into quadword `o`, in reverse weight order.
///
/// Output register `g`, quadword `l`, byte `7 - s` takes byte `8g + l` of weight `s`.
const fn gather_index(g: usize) -> ([u8; 64], [u8; 64], u64) {
    let (mut lo, mut hi, mut mask) = ([0u8; 64], [0u8; 64], 0u64);
    let mut p = 0;
    while p < 64 {
        let (l, s) = (p / 8, 7 - p % 8);
        let src = 24 * s + 8 * g + l;
        if src < 128 {
            lo[p] = src as u8;
        } else {
            hi[p] = (src - 128) as u8;
            mask |= 1 << p;
        }
        p += 1;
    }
    (lo, hi, mask)
}

/// The gathers of the three output registers.
const GATHER: [([u8; 64], [u8; 64], u64); 3] = [gather_index(0), gather_index(1), gather_index(2)];

/// Byte `t` of each quadword is the unit vector `1 << (7 - t)`.
const REVERSED_IDENTITY: u64 = 0x0102_0408_1020_4080;

/// The matrices of eight weights: matrix `o` maps byte `x` to byte `o` of `sum_{s in x} w_s`.
#[inline(always)]
pub fn weight_matrices<R: SliceRegister>(w: &[Poly192; 8]) -> [u64; OUT_BYTES] {
    let src = w.as_ptr().cast::<u8>();
    // SAFETY: eight weights are 192 bytes, exactly three registers.
    let v: [R; 3] = core::array::from_fn(|i| unsafe { R::load(src.add(64 * i)) });
    let unit = R::splat(REVERSED_IDENTITY);
    let mut out = [0u64; OUT_BYTES];
    let dst = out.as_mut_ptr().cast::<u8>();
    for (g, (lo, hi, mask)) in GATHER.iter().enumerate() {
        // SAFETY: the index arrays are 64 bytes, and store g fills out[8g..8g+8].
        unsafe {
            let q = v[0].permute2(R::load(lo.as_ptr()), v[1]).permute_merge(
                *mask,
                R::load(hi.as_ptr()),
                v[2],
            );
            R::store(dst.add(64 * g), unit.affine(q));
        }
    }
    out
}

/// Byte sources of one coefficient plane's transpose: quadword `j`, byte `7 - s` takes byte `j` of quadword `s`.
const PLANE_GATHER: [u8; 64] = {
    let mut index = [0u8; 64];
    let mut p = 0;
    while p < 64 {
        let (j, s) = (p / 8, 7 - p % 8);
        index[p] = (8 * s + j) as u8;
        p += 1;
    }
    index
};

/// The matrices of eight weights given as coefficient planes, one register per plane.
///
/// Quadword `j` of register `k` maps an input byte to byte `8 k + j` of the weighted sum, as in `weight_matrices`.
#[inline(always)]
pub fn plane_matrices<R: SliceRegister>(planes: &[R; 3]) -> [R; 3] {
    // SAFETY: the index array is 64 bytes.
    let gather = unsafe { R::load(PLANE_GATHER.as_ptr()) };
    let unit = R::splat(REVERSED_IDENTITY);
    planes.map(|p| unit.affine(p.permute(gather)))
}

/// Byte sources of the bit transpose: quadword `j`, byte `t` takes row `7 - t`, column `j`.
const COLUMNS: [u8; 64] = {
    let mut index = [0u8; 64];
    let mut i = 0;
    while i < 64 {
        index[i] = ((7 - i % 8) * 8 + i / 8) as u8;
        i += 1;
    }
    index
};

/// The block that reverses the bits of a byte.
const BIT_REVERSE: u64 = 0x8040_2010_0804_0201;

/// Transpose eight rows of eight bytes: bit `k` of output byte `s` is bit `s` of row `k`.
///
/// Quadword `k` of the input is row `k`.
#[inline(always)]
pub fn transpose_rows<R: SliceRegister>(rows: R) -> R {
    // SAFETY: the index array is 64 bytes.
    let columns = rows.permute(unsafe { R::load(COLUMNS.as_ptr()) });
    R::splat(BIT_REVERSE).affine(columns)
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use p3_field::PrimeCharacteristicRing;
    use proptest::prelude::*;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::*;
    use crate::Poly64;
    use crate::byteslice::model::Model;

    /// The weighted sum of the set bits of a byte, the definition every matrix is checked against.
    fn weighted(w: &[Poly192; 8], x: u8) -> Poly192 {
        (0..8).filter(|s| x >> s & 1 == 1).map(|s| w[s]).sum()
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(16))]

        #[test]
        fn the_model_transposes_bits(rows in prop::array::uniform32(any::<u8>()), more in prop::array::uniform32(any::<u8>())) {
            let mut input = [0u8; 64];
            input[..32].copy_from_slice(&rows);
            input[32..].copy_from_slice(&more);
            let mut out = [0u8; 64];
            // SAFETY: both arrays are 64 bytes.
            unsafe { Model::store(out.as_mut_ptr(), transpose_rows(Model::load(input.as_ptr()))) };
            prop_assert_eq!(out, crate::byteslice::transpose_bits_portable(&input));
        }

        #[test]
        fn the_model_matrices_map_bytes_to_weighted_sums(seed: u64) {
            let mut rng = SmallRng::seed_from_u64(seed);
            let w: [Poly192; 8] = core::array::from_fn(|_| rng.random());
            let m = weight_matrices::<Model>(&w);
            for x in [0u8, 1, 0x80, 0x5A, 0xFF, rng.random()] {
                // Byte `o` of the weighted sum is matrix `o` applied to `x`.
                let want = weighted(&w, x);
                let bytes: Vec<u8> = (0..OUT_BYTES)
                    .map(|o| {
                        let mut out = [0u8; 64];
                        // SAFETY: the array is 64 bytes.
                        unsafe {
                            Model::store(out.as_mut_ptr(), Model::splat(u64::from(x)).affine(Model::splat(m[o])));
                        }
                        out[0]
                    })
                    .collect();
                let got = Poly192::new(core::array::from_fn(|k| {
                    Poly64::from_le_bytes(core::array::from_fn(|t| bytes[8 * k + t]))
                }));
                prop_assert_eq!(got, want);
            }
        }

        #[test]
        fn the_model_plane_matrices_are_the_weight_matrices(seed: u64) {
            // Invariant: eight weights as coefficient planes give the matrices of the weights themselves.
            let mut rng = SmallRng::seed_from_u64(seed);
            let w: [Poly192; 8] = core::array::from_fn(|_| rng.random());
            let planes: [Model; 3] = core::array::from_fn(|k| {
                let plane: [u64; 8] = core::array::from_fn(|l| w[l].coefficients()[k].to_bits());
                // SAFETY: the array is 64 bytes.
                unsafe { Model::load(plane.as_ptr().cast()) }
            });
            let mut got = [0u64; OUT_BYTES];
            for (k, reg) in plane_matrices(&planes).into_iter().enumerate() {
                // SAFETY: each store fills eight of the 24 matrices.
                unsafe { Model::store(got[8 * k..].as_mut_ptr().cast(), reg) };
            }
            prop_assert_eq!(got, weight_matrices::<Model>(&w));
        }

        #[test]
        fn the_model_stores_sliced_values(seed: u64) {
            // Invariant: register `o`, byte `p` is byte `o` of value `p`.
            let mut rng = SmallRng::seed_from_u64(seed);
            let values: [Poly192; BLOCK] = core::array::from_fn(|_| rng.random());
            let acc: [Model; OUT_BYTES] = core::array::from_fn(|o| {
                let column: [u8; 64] =
                    core::array::from_fn(|p| values[p].coefficients()[o / 8].to_bits().to_le_bytes()[o % 8]);
                // SAFETY: the array is 64 bytes.
                unsafe { Model::load(column.as_ptr()) }
            });
            let mut out = [Poly192::ZERO; BLOCK];
            store_f192(&acc, &mut out);
            prop_assert_eq!(out, values);
        }
    }
}
