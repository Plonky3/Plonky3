//! The Lin–Chung–Han transform carried out in the polynomial basis of `GF(2^128)`.

use alloc::vec::Vec;

use p3_binary_field::{BinaryField128, TowerLevel, poly_basis};
use p3_field::PrimeCharacteristicRing;
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;
use p3_util::log2_strict_usize;

use crate::domain::domain_point;
use crate::lch::BUTTERFLY_GRAIN;
use crate::traits::AdditiveNtt;

/// [`LchNtt`](crate::LchNtt) over `BinaryField128`, with the data held in the polynomial basis throughout.
///
/// A tower-basis product converts both operands into the polynomial basis and the result back,
/// sixteen dependent table lookups apiece, which is most of what a butterfly costs. Converting
/// the whole matrix once on the way in and once on the way out pays `2` conversions per element
/// instead of `3ℓ/2`, and every twiddle multiply in between is a bare carryless multiply and a
/// reduction. Additions are `XOR` in both bases, so they are unaffected.
///
/// Without a carryless-multiply instruction that product is a bit-serial loop and slower than
/// the tower arithmetic it replaces, so on such a target the transform runs in the tower basis
/// instead, using typed subfield multiplication when the twiddle permits it.
/// The choice is a constant and only one arm survives compilation.
#[derive(Clone, Debug, Default)]
pub struct PolyBasisNtt {
    tower: crate::tower::TowerNtt,
}

/// Stage shifts and the XOR increments between consecutive block twiddles.
struct Twiddles {
    basis: [u128; usize::BITS as usize],
    deltas: [u128; usize::BITS as usize],
    shifts: [u128; usize::BITS as usize],
}

impl Twiddles {
    fn new(log_n: usize, shift: BinaryField128) -> Self {
        let mut result = Self {
            basis: [0; usize::BITS as usize],
            deltas: [0; usize::BITS as usize],
            shifts: [0; usize::BITS as usize],
        };
        let mut delta = 0;
        let mut base = poly_basis::from_tower(shift);
        for j in 0..log_n {
            result.basis[j] = poly_basis::from_tower(BinaryField128::cantor_basis(j + 1));
            delta ^= result.basis[j];
            result.deltas[j] = delta;
            result.shifts[j] = base;
            base = poly_basis::square(base) ^ base;
        }
        result
    }

    const fn at(&self, stage: usize, mut block: usize) -> u128 {
        let mut t = self.shifts[stage];
        while block != 0 {
            t ^= self.basis[block.trailing_zeros() as usize];
            block &= block - 1;
        }
        t
    }
}

/// A stage uses bounded chunks, each starting from its own independently indexed twiddle.
fn stage(values: &mut [u128], half: usize, j: usize, twiddles: &Twiddles, inverse: bool) {
    let blocks_per_chunk = (BUTTERFLY_GRAIN / (half << 1)).max(1);
    values
        .par_chunks_mut((half << 1) * blocks_per_chunk)
        .enumerate()
        .for_each(|(chunk_index, chunk)| {
            let first = chunk_index * blocks_per_chunk;
            let mut t = twiddles.at(j, first);
            for (index, block) in chunk.chunks_mut(half << 1).enumerate() {
                let (lo, hi) = block.split_at_mut(half);
                let butterfly = |lo: &mut [u128], hi: &mut [u128]| {
                    if inverse {
                        poly_basis::butterfly_inverse(lo, hi, t);
                    } else {
                        poly_basis::butterfly_forward(lo, hi, t);
                    }
                };
                if half <= BUTTERFLY_GRAIN {
                    butterfly(lo, hi);
                } else {
                    lo.par_chunks_mut(BUTTERFLY_GRAIN)
                        .zip(hi.par_chunks_mut(BUTTERFLY_GRAIN))
                        .for_each(|(lo, hi)| butterfly(lo, hi));
                }
                t ^= twiddles.deltas[(first + index).trailing_ones() as usize];
            }
        });
}

// A conservative tile budget; the row count scales with the element size and matrix width.
const TILE_BYTES: usize = 32 * 1024;

/// A tile stays on one worker across its adjacent stages.
fn local_stage(
    values: &mut [u128],
    half: usize,
    j: usize,
    twiddles: &Twiddles,
    inverse: bool,
    first: usize,
) {
    let mut t = twiddles.at(j, first);
    for (index, block) in values.chunks_mut(half << 1).enumerate() {
        let (lo, hi) = block.split_at_mut(half);
        if inverse {
            poly_basis::butterfly_inverse(lo, hi, t);
        } else {
            poly_basis::butterfly_forward(lo, hi, t);
        }
        t ^= twiddles.deltas[(first + index).trailing_ones() as usize];
    }
}

/// Complete the stages confined to one cache-sized set of rows before leaving it.
fn local_stages(
    values: &mut [u128],
    width: usize,
    log_n: usize,
    twiddles: &Twiddles,
    inverse: bool,
) -> usize {
    let rows = (TILE_BYTES / core::mem::size_of::<u128>() / width).max(1);
    let local = p3_util::log2_floor_usize(rows).min(log_n);
    let tile_len = (1 << local) * width;
    values
        .par_chunks_mut(tile_len)
        .enumerate()
        .for_each(|(tile, values)| {
            for k in 0..local {
                let j = if inverse { k } else { local - 1 - k };
                local_stage(
                    values,
                    (1 << j) * width,
                    j,
                    twiddles,
                    inverse,
                    tile << (local - j - 1),
                );
            }
        });
    local
}

/// Forward transform of polynomial-basis values in an existing allocation.
fn forward(values: &mut [u128], width: usize, log_n: usize, shift: BinaryField128) {
    let twiddles = Twiddles::new(log_n, shift);
    if core::mem::size_of_val(values) <= TILE_BYTES {
        for j in (0..log_n).rev() {
            stage(values, (1 << j) * width, j, &twiddles, false);
        }
        return;
    }
    let rows = (TILE_BYTES / core::mem::size_of::<u128>() / width).max(1);
    let local = p3_util::log2_floor_usize(rows).min(log_n);
    for j in (local..log_n).rev() {
        stage(values, (1 << j) * width, j, &twiddles, false);
    }
    local_stages(values, width, log_n, &twiddles, false);
}

/// Inverse transform with the data kept in the polynomial basis.
fn inverse(values: &mut [u128], width: usize, log_n: usize, shift: BinaryField128) {
    let twiddles = Twiddles::new(log_n, shift);
    if core::mem::size_of_val(values) <= TILE_BYTES {
        for j in 0..log_n {
            stage(values, (1 << j) * width, j, &twiddles, true);
        }
        return;
    }
    let local = local_stages(values, width, log_n, &twiddles, true);
    for j in local..log_n {
        stage(values, (1 << j) * width, j, &twiddles, true);
    }
}

impl AdditiveNtt<BinaryField128> for PolyBasisNtt {
    fn shifted_ntt_batch(
        &self,
        mut mat: RowMajorMatrix<BinaryField128>,
        shift: BinaryField128,
    ) -> RowMajorMatrix<BinaryField128> {
        if !poly_basis::HAS_HARDWARE_CLMUL {
            return self.tower.shifted_ntt_batch(mat, shift);
        }

        let width = mat.width();
        let log_n = log2_strict_usize(mat.height());
        // `BinaryField128` is `#[repr(transparent)]` over `u128`, so this reuses `mat.values`'s
        // allocation instead of allocating a second codeword buffer: `to_repr` is the identity
        // bit pattern, and the specialised `Vec` collect below reinterprets the buffer in place.
        let mut values: Vec<u128> = core::mem::take(&mut mat.values)
            .into_iter()
            .map(BinaryField128::to_repr)
            .collect();
        values
            .par_iter_mut()
            .for_each(|v| *v = poly_basis::from_tower(BinaryField128::from_repr(*v)));

        forward(&mut values, width, log_n, shift);

        values
            .par_iter_mut()
            .for_each(|v| *v = poly_basis::to_tower(*v).to_repr());
        mat.values = values.into_iter().map(BinaryField128::from_repr).collect();
        mat
    }

    fn ntt_batch_padded(
        &self,
        mut mat: RowMajorMatrix<BinaryField128>,
        log_inv_rate: usize,
    ) -> RowMajorMatrix<BinaryField128> {
        let log_n = log2_strict_usize(mat.height());
        assert!(log_inv_rate <= log_n, "padding exceeds matrix height");
        if log_inv_rate == 0 || !poly_basis::HAS_HARDWARE_CLMUL {
            return self.ntt_batch(mat);
        }
        let width = mat.width;
        let log_message = log_n - log_inv_rate;
        let len = mat.values.len() >> log_inv_rate;
        let mut values: Vec<u128> = core::mem::take(&mut mat.values)
            .into_iter()
            .map(BinaryField128::to_repr)
            .collect();
        let (message, tail) = values.split_at_mut(len);
        message.par_iter_mut().for_each(|v| {
            *v = poly_basis::from_tower(BinaryField128::from_repr(*v));
        });
        // Keep the coefficient prefix immutable until every other coset has copied it.
        // Each worker transforms its final destination, without a temporary coset matrix.
        tail.par_chunks_mut(len).enumerate().for_each(|(c, chunk)| {
            chunk.copy_from_slice(message);
            forward(
                chunk,
                width,
                log_message,
                domain_point((c + 1) << log_message),
            );
        });
        forward(message, width, log_message, BinaryField128::ZERO);
        values
            .par_iter_mut()
            .for_each(|v| *v = poly_basis::to_tower(*v).to_repr());
        mat.values = values.into_iter().map(BinaryField128::from_repr).collect();
        mat
    }

    fn shifted_lde_batch(
        &self,
        mut mat: RowMajorMatrix<BinaryField128>,
        added_bits: usize,
        shift: BinaryField128,
    ) -> RowMajorMatrix<BinaryField128> {
        if !poly_basis::HAS_HARDWARE_CLMUL {
            return self.tower.shifted_lde_batch(mat, added_bits, shift);
        }
        let log_n = log2_strict_usize(mat.height());
        if added_bits == 0 {
            return mat;
        }
        let width = mat.width;
        let len = mat.values.len();
        let padded_len = u32::try_from(added_bits)
            .ok()
            .and_then(|bits| len.checked_shl(bits))
            .filter(|&padded| padded >> added_bits == len)
            .expect("extended codeword length overflows usize");
        // Preserve the evaluation prefix in its final allocation, then reuse the
        // original allocation for coefficients instead of cloning it before a resize.
        let mut coeffs: Vec<u128> = core::mem::take(&mut mat.values)
            .into_iter()
            .map(BinaryField128::to_repr)
            .collect();
        let mut values = alloc::vec![0u128; padded_len];
        values[..len].copy_from_slice(&coeffs);
        coeffs
            .par_iter_mut()
            .for_each(|v| *v = poly_basis::from_tower(BinaryField128::from_repr(*v)));
        inverse(&mut coeffs, width, log_n, shift);

        // The input evaluations already are the first coset. Only new cosets need
        // evaluation and conversion back from the polynomial basis.
        values[len..]
            .par_chunks_mut(len)
            .enumerate()
            .for_each(|(c, chunk)| {
                chunk.copy_from_slice(&coeffs);
                forward(
                    chunk,
                    width,
                    log_n,
                    shift + domain_point::<BinaryField128>((c + 1) << log_n),
                );
                for v in chunk {
                    *v = poly_basis::to_tower(*v).to_repr();
                }
            });
        mat.values = values.into_iter().map(BinaryField128::from_repr).collect();
        mat
    }

    fn shifted_intt_batch(
        &self,
        mut mat: RowMajorMatrix<BinaryField128>,
        shift: BinaryField128,
    ) -> RowMajorMatrix<BinaryField128> {
        if !poly_basis::HAS_HARDWARE_CLMUL {
            return self.tower.shifted_intt_batch(mat, shift);
        }

        let width = mat.width();
        let log_n = log2_strict_usize(mat.height());
        // See `shifted_ntt_batch`: reuses `mat.values`'s allocation instead of a second buffer.
        let mut values: Vec<u128> = core::mem::take(&mut mat.values)
            .into_iter()
            .map(BinaryField128::to_repr)
            .collect();
        values
            .par_iter_mut()
            .for_each(|v| *v = poly_basis::from_tower(BinaryField128::from_repr(*v)));

        inverse(&mut values, width, log_n, shift);

        values
            .par_iter_mut()
            .for_each(|v| *v = poly_basis::to_tower(*v).to_repr());
        mat.values = values.into_iter().map(BinaryField128::from_repr).collect();
        mat
    }
}

#[cfg(test)]
mod tests {
    use p3_binary_field::{BinaryField128, TowerLevel};
    use p3_matrix::dense::RowMajorMatrix;
    use proptest::prelude::*;

    use super::PolyBasisNtt;
    use crate::naive::NaiveAdditiveNtt;
    use crate::traits::AdditiveNtt;

    /// Builds a matrix whose entries are distinct functions of the seed and the position.
    fn matrix(log_n: usize, width: usize, seed: u64) -> RowMajorMatrix<BinaryField128> {
        RowMajorMatrix::new(
            (0..(width << log_n))
                .map(|i| {
                    let bits = seed
                        .wrapping_mul(0x9e37_79b9_7f4a_7c15)
                        .wrapping_add(i as u64);
                    BinaryField128::from_le_byte_iter(bits.to_le_bytes().into_iter().cycle())
                })
                .collect(),
            width,
        )
    }

    #[test]
    fn padded_transform_matches_naive_at_wide_widths() {
        use p3_field::PrimeCharacteristicRing;
        for width in [1, 4, 16, 64] {
            for added in [0, 1, 2, 3] {
                let mut mat = matrix(4, width, 13);
                mat.values
                    .resize(mat.values.len() << added, BinaryField128::ZERO);
                let expected = NaiveAdditiveNtt::default().ntt_batch(mat.clone());
                assert_eq!(
                    PolyBasisNtt::default().ntt_batch_padded(mat, added),
                    expected
                );
            }
        }
    }

    #[test]
    fn shifted_lde_matches_naive_at_wide_widths() {
        for width in [1, 4, 16, 64] {
            for added in [0, 1, 2, 3] {
                let mat = matrix(4, width, 17);
                let shift = BinaryField128::from_repr(1 << 127);
                let expected =
                    NaiveAdditiveNtt::default().shifted_lde_batch(mat.clone(), added, shift);
                let actual = PolyBasisNtt::default().shifted_lde_batch(mat.clone(), added, shift);
                assert_eq!(actual, expected);
                assert_eq!(&actual.values[..mat.values.len()], &mat.values);
            }
        }
    }

    #[test]
    #[should_panic = "extended codeword length overflows usize"]
    fn lde_rejects_length_overflow() {
        let _ = PolyBasisNtt::default().lde_batch(matrix(1, 1, 0), usize::BITS as usize - 1);
    }

    #[test]
    fn incremental_twiddles_match_independent_domain_points() {
        use p3_binary_field::poly_basis;

        use crate::domain::{domain_point, subspace_polynomial};
        let shift = BinaryField128::from_repr((1 << 127) | 123);
        let twiddles = super::Twiddles::new(usize::BITS as usize - 1, shift);
        for stage in [0, 1, 7, 15, 31]
            .into_iter()
            .filter(|&stage| stage < usize::BITS as usize - 1)
        {
            for start in [0, 1, 63, 127, (1usize << (usize::BITS - 3)) - 3] {
                let mut t = twiddles.at(stage, start);
                for block in start..start + 9 {
                    let expected = subspace_polynomial(stage, shift)
                        + domain_point::<BinaryField128>(block << 1);
                    assert_eq!(t, poly_basis::from_tower(expected));
                    t ^= twiddles.deltas[block.trailing_ones() as usize];
                }
            }
        }
    }

    #[test]
    fn transforms_cross_cache_boundaries_in_natural_order() {
        for width in [1, 3, 16, 64] {
            let mat = matrix(12, width, 29);
            let shift = BinaryField128::from_repr((1 << 127) | 7919);
            let expected = crate::LchNtt::default().shifted_ntt_batch(mat.clone(), shift);
            let actual = PolyBasisNtt::default().shifted_ntt_batch(mat.clone(), shift);
            assert_eq!(actual, expected);
            assert_eq!(
                PolyBasisNtt::default().shifted_intt_batch(actual, shift),
                mat
            );
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(32))]

        /// The polynomial-basis transform is the same map as the reference oracle.
        #[test]
        fn poly_basis_matches_naive(
            log_n in 0usize..=8,
            width in 1usize..=5,
            seed in any::<u64>(),
            shift in any::<u64>(),
        ) {
            let coeffs = matrix(log_n, width, seed);
            let shift = BinaryField128::from_le_byte_iter(
                shift.to_le_bytes().into_iter().cycle(),
            );

            let fast = PolyBasisNtt::default().shifted_ntt_batch(coeffs.clone(), shift);
            let slow = NaiveAdditiveNtt::<BinaryField128>::default()
                .shifted_ntt_batch(coeffs, shift);
            prop_assert_eq!(fast, slow);
        }

        #[test]
        fn poly_basis_round_trips(
            log_n in 0usize..=8,
            width in 1usize..=3,
            seed in any::<u64>(),
            shift in any::<u64>(),
        ) {
            let coeffs = matrix(log_n, width, seed);
            let shift = BinaryField128::from_le_byte_iter(
                shift.to_le_bytes().into_iter().cycle(),
            );
            let ntt = PolyBasisNtt::default();
            let evals = ntt.shifted_ntt_batch(coeffs.clone(), shift);
            prop_assert_eq!(ntt.shifted_intt_batch(evals, shift), coeffs);
        }

        /// `PolyBasisNtt`'s low-degree extension agrees with the oracle's, on a coset too, and
        /// the input rows reappear as the prefix: the correspondence Phase 3 folds along.
        #[test]
        fn poly_basis_lde_matches_naive(
            log_n in 0usize..=6,
            added in 0usize..=3,
            width in 1usize..=3,
            seed in any::<u64>(),
            shift in any::<u64>(),
        ) {
            let coeffs = matrix(log_n, width, seed);
            let shift = BinaryField128::from_le_byte_iter(
                shift.to_le_bytes().into_iter().cycle(),
            );

            let lde = PolyBasisNtt::default().shifted_lde_batch(coeffs.clone(), added, shift);
            let naive = NaiveAdditiveNtt::<BinaryField128>::default()
                .shifted_lde_batch(coeffs.clone(), added, shift);
            prop_assert_eq!(&lde, &naive);
            prop_assert_eq!(&lde.values[..coeffs.values.len()], &coeffs.values[..]);
        }
    }

    /// A height that is not a power of two has no well-defined `ℓ`. `ℓ` exceeding the bit width
    /// of `BinaryField128` is covered generically at [`LchNtt`](crate::LchNtt), where the level
    /// is a type parameter and the panic is cheap to reach; at a fixed `BinaryField128` it is
    /// only reachable through a matrix of `2^129` rows, which is not a test worth writing.
    #[test]
    #[should_panic]
    fn shifted_ntt_batch_rejects_a_non_power_of_two_height() {
        let coeffs = RowMajorMatrix::new(matrix(0, 1, 0).values.repeat(3), 1);
        let _ = PolyBasisNtt::default().ntt_batch(coeffs);
    }
}
