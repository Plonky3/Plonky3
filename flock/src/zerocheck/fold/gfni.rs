//! The fold of 64 packed bit rows at a time, byte-sliced: one matrix per input and output byte.

use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_binary_field::Poly192;
use p3_util::log2_strict_usize;

use crate::byteslice::{
    BLOCK, IDENTITY, OUT_BYTES, SliceRegister, Stage, store_f192, weight_matrices,
};

/// Byte offset `(o0, o1, p2, p3, p4, o2)` to quad planes.
const TO_QUADS: [u8; 64] = {
    let mut sigma = [0u8; 64];
    let mut f = 0;
    while f < 64 {
        sigma[f] = ((f & 3) | ((f >> 3) << 2) | (((f >> 2) & 1) << 5)) as u8;
        f += 1;
    }
    sigma
};

/// Stages taking eight registers of one coefficient to quad planes.
///
/// Register `u + 2v + 4g`, quadword `l` then holds the coefficient of value `4 (8g + l) + u + 2v`.
const QUAD_STAGES: [Stage; 3] = [
    Stage::swap(1, &IDENTITY),
    Stage::swap(2, &IDENTITY),
    Stage::swap(32, &TO_QUADS),
];

/// The fold of packed bit rows by fixed weights, 64 rows at a time.
#[derive(Clone, Debug)]
pub(crate) struct Fold<R> {
    /// `matrices[24 r + o]` maps register `r` of the input transpose to output byte `o`.
    matrices: Vec<u64>,
    /// Stages taking row-major bytes to one register per input byte.
    input: Vec<Stage>,
    /// The register the fold runs on.
    _register: PhantomData<fn() -> R>,
}

impl<R: SliceRegister> Fold<R> {
    /// Prepare one weight per bit of a row.
    pub(crate) fn new(weights: &[Poly192]) -> Self {
        let n_chunks = weights.len() / 8;
        let c = log2_strict_usize(n_chunks);

        // Byte address bits 0..c are the byte in the row, bits c..c+6 the row.
        // After the stages, the offset is the row and the register is the byte, rotated when c > 6.
        let n_stages = c.min(6);
        let skew = c.saturating_sub(6);
        // Without a rotation, a short row leaves offset = (row high bits, then row low bits).
        let plain = |p: usize| {
            if c >= 6 {
                p
            } else {
                (p >> (6 - c)) | ((p & ((1 << (6 - c)) - 1)) << c)
            }
        };
        let input = (0..n_stages)
            .map(|s| {
                let last = s + 1 == n_stages;
                let sigma: [u8; 64] =
                    core::array::from_fn(|f| if last { plain(f) } else { f } as u8);
                Stage::swap(1 << s, &sigma)
            })
            .collect();

        // Register `r` then holds input byte `j(r)`.
        let byte_of_reg = |r: usize| (r >> skew) | ((r & ((1 << skew) - 1)) << (c - skew));
        let bytes: &[[Poly192; 8]] = weights.as_chunks().0;
        let matrices = (0..n_chunks)
            .flat_map(|r| weight_matrices::<R>(&bytes[byte_of_reg(r)]))
            .collect();
        Self {
            matrices,
            input,
            _register: PhantomData,
        }
    }

    /// Fold up to 64 rows into `out[..rows.len()]`.
    #[inline]
    pub(crate) fn fold_block<const CHUNKS: usize>(
        &self,
        rows: &[[u8; CHUNKS]],
        out: &mut [Poly192; BLOCK],
    ) {
        match <&[[u8; CHUNKS]; BLOCK]>::try_from(rows) {
            Ok(full) => store_f192(&self.fold_bytes(full), out),
            // A short block folds from a zero-padded copy.
            Err(_) => {
                let mut padded = [[0u8; CHUNKS]; BLOCK];
                padded[..rows.len()].copy_from_slice(rows);
                store_f192(&self.fold_bytes(&padded), out);
            }
        }
    }

    /// Fold a full block into quad planes: plane `k`, register `u + 2v + 4g`, quadword `l` is row `4 (8g + l) + u + 2v`.
    #[inline(always)]
    pub(crate) fn fold_quads<const CHUNKS: usize>(
        &self,
        rows: &[[u8; CHUNKS]; BLOCK],
    ) -> [[R; 8]; 3] {
        let acc = self.fold_bytes(rows);
        let mut planes: [[R; 8]; 3] =
            core::array::from_fn(|i| core::array::from_fn(|o| acc[8 * i + o]));
        for (s, stage) in QUAD_STAGES.iter().enumerate() {
            for plane in &mut planes {
                stage.apply(1 << s, plane);
            }
        }
        planes
    }

    /// Fold a full block into 24 registers, register `o` holding byte `o` of the 64 values.
    #[inline(always)]
    fn fold_bytes<const CHUNKS: usize>(&self, rows: &[[u8; CHUNKS]; BLOCK]) -> [R; OUT_BYTES] {
        // Phase 1: CHUNKS registers of 64 bytes, row-major, then one register per input byte.
        let base = rows.as_ptr().cast::<u8>();
        // SAFETY: the block is 64 * CHUNKS bytes, exactly CHUNKS registers.
        let mut regs: [R; CHUNKS] = core::array::from_fn(|i| unsafe { R::load(base.add(64 * i)) });
        // Up to three stages per pass, eight registers at a time, so a wide row stays in registers.
        let n_stages = log2_strict_usize(CHUNKS).min(6);
        for first in (0..n_stages).step_by(3) {
            match n_stages - first {
                1 => self.input_stages::<2, CHUNKS>(first, &mut regs),
                2 => self.input_stages::<4, CHUNKS>(first, &mut regs),
                _ => self.input_stages::<8, CHUNKS>(first, &mut regs),
            }
        }

        // Phase 2: every output byte accumulates one affine product per input register.
        let mut acc = [R::zero(); OUT_BYTES];
        let matrices: &[[u64; OUT_BYTES]] = self.matrices.as_chunks().0;
        for (pair, m) in regs
            .as_chunks::<2>()
            .0
            .iter()
            .zip(matrices.as_chunks::<2>().0)
        {
            for (o, acc) in acc.iter_mut().enumerate() {
                let g0 = pair[0].affine(R::splat(m[0][o]));
                let g1 = pair[1].affine(R::splat(m[1][o]));
                *acc = acc.xor3(g0, g1);
            }
        }
        acc
    }

    /// Input stages `first..first + log2(G)`, on each group of `G` registers they pair.
    #[inline(always)]
    fn input_stages<const G: usize, const CHUNKS: usize>(
        &self,
        first: usize,
        regs: &mut [R; CHUNKS],
    ) {
        // Stage `s` pairs on register bit `s + skew`.
        let low = first + log2_strict_usize(CHUNKS).saturating_sub(6);
        let group_bits = (G - 1) << low;
        for base in (0..CHUNKS).filter(|r| r & group_bits == 0) {
            let mut group: [R; G] = core::array::from_fn(|k| regs[base | (k << low)]);
            for j in 0..log2_strict_usize(G) {
                self.input[first + j].apply(1 << j, &mut group);
            }
            for (k, g) in group.iter().enumerate() {
                regs[base | (k << low)] = *g;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use p3_field::PrimeCharacteristicRing;
    use proptest::prelude::*;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::*;
    use crate::byteslice::Model;
    use crate::zerocheck::fold::portable::Tables;

    /// Full blocks then a short one, the model's fold against the portable tables.
    fn check<const CHUNKS: usize>(seed: u64) {
        let mut rng = SmallRng::seed_from_u64(seed);
        let weights: Vec<Poly192> = (0..8 * CHUNKS).map(|_| rng.random()).collect();
        let fold = Fold::<Model>::new(&weights);
        let tables = Tables::new(&weights);
        for len in [BLOCK, 5] {
            let rows: Vec<[u8; CHUNKS]> = (0..len)
                .map(|_| core::array::from_fn(|_| rng.random()))
                .collect();
            let mut out = [Poly192::ZERO; BLOCK];
            fold.fold_block(&rows, &mut out);
            for (p, row) in rows.iter().enumerate() {
                assert_eq!(out[p], tables.fold_row(row), "CHUNKS={CHUNKS}, row {p}");
            }
        }

        // The quad planes hold the same values, regrouped by quad.
        let rows: [[u8; CHUNKS]; BLOCK] =
            core::array::from_fn(|_| core::array::from_fn(|_| rng.random()));
        let planes = fold.fold_quads(&rows);
        for (k, plane) in planes.iter().enumerate() {
            for (reg, &register) in plane.iter().enumerate() {
                let mut bytes = [0u8; 64];
                // SAFETY: the array is 64 bytes.
                unsafe { Model::store(bytes.as_mut_ptr(), register) };
                let (uv, g) = (reg % 4, reg / 4);
                for (l, word) in bytes.as_chunks::<8>().0.iter().enumerate() {
                    let row = 4 * (8 * g + l) + uv;
                    let want = tables.fold_row(&rows[row]).coefficients()[k].to_bits();
                    assert_eq!(
                        u64::from_le_bytes(*word),
                        want,
                        "CHUNKS={CHUNKS}, row {row}"
                    );
                }
            }
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(4))]

        #[test]
        fn the_model_fold_is_the_table_fold(seed: u64) {
            check::<8>(seed);
            check::<16>(seed);
            check::<32>(seed);
            check::<64>(seed);
            check::<128>(seed);
        }
    }
}
