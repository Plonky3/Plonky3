//! The portable `BinaryField128` transform with typed subfield twiddles.

use core::ops::Mul;

use p3_binary_field::{
    BinaryField8, BinaryField16, BinaryField32, BinaryField64, BinaryField128, TowerLevel,
};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;
use p3_util::log2_strict_usize;

use crate::domain::{domain_point, subspace_polynomial};
use crate::lch::BUTTERFLY_GRAIN;

/// The specialized portable implementation also supplies the trait's generic LDE path.
#[derive(Clone, Debug, Default)]
pub(crate) struct TowerNtt;

impl crate::AdditiveNtt<BinaryField128> for TowerNtt {
    fn shifted_ntt_batch(
        &self,
        mut mat: RowMajorMatrix<BinaryField128>,
        shift: BinaryField128,
    ) -> RowMajorMatrix<BinaryField128> {
        transform(&mut mat, shift, false);
        mat
    }

    fn shifted_intt_batch(
        &self,
        mut mat: RowMajorMatrix<BinaryField128>,
        shift: BinaryField128,
    ) -> RowMajorMatrix<BinaryField128> {
        transform(&mut mat, shift, true);
        mat
    }
}

fn butterfly<T: Copy>(lo: &mut [BinaryField128], hi: &mut [BinaryField128], t: T, inverse: bool)
where
    BinaryField128: Mul<T, Output = BinaryField128>,
{
    if inverse {
        for (u, v) in lo.iter_mut().zip(hi) {
            *v += *u;
            *u += *v * t;
        }
    } else {
        for (u, v) in lo.iter_mut().zip(hi) {
            *u += *v * t;
            *v += *u;
        }
    }
}

pub(crate) fn transform(
    mat: &mut RowMajorMatrix<BinaryField128>,
    shift: BinaryField128,
    inverse: bool,
) {
    let log_n = log2_strict_usize(mat.height());
    for k in 0..log_n {
        let j = if inverse { k } else { log_n - 1 - k };
        let half = (1 << j) * mat.width;
        let base = subspace_polynomial(j, shift);
        mat.values
            .par_chunks_mut(half << 1)
            .enumerate()
            .for_each(|(block, values)| {
                let t = base + domain_point::<BinaryField128>(block << 1);
                let bits = t.to_repr();
                let apply = |(lo, hi): (&mut [BinaryField128], &mut [BinaryField128])| {
                    // A shifted twiddle is not necessarily in the domain's small subfield.
                    // Check its actual coordinates before constructing a typed coefficient.
                    if bits == 0 {
                        for (u, v) in lo.iter().zip(hi) {
                            *v += *u;
                        }
                    } else if bits <= u8::MAX as u128 {
                        butterfly(lo, hi, BinaryField8::from_repr(bits as u8), inverse);
                    } else if bits <= u16::MAX as u128 {
                        butterfly(lo, hi, BinaryField16::from_repr(bits as u16), inverse);
                    } else if bits <= u32::MAX as u128 {
                        butterfly(lo, hi, BinaryField32::from_repr(bits as u32), inverse);
                    } else if bits <= u64::MAX as u128 {
                        butterfly(lo, hi, BinaryField64::from_repr(bits as u64), inverse);
                    } else {
                        butterfly(lo, hi, t, inverse);
                    }
                };
                let (lo, hi) = values.split_at_mut(half);
                if half <= BUTTERFLY_GRAIN {
                    apply((lo, hi));
                } else {
                    lo.par_chunks_mut(BUTTERFLY_GRAIN)
                        .zip(hi.par_chunks_mut(BUTTERFLY_GRAIN))
                        .for_each(apply);
                }
            });
    }
}

#[cfg(test)]
mod tests {
    use p3_binary_field::{BinaryField128, TowerLevel};
    use p3_field::PrimeCharacteristicRing;
    use p3_matrix::dense::RowMajorMatrix;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    use crate::{AdditiveNtt, LchNtt, NaiveAdditiveNtt};

    #[test]
    fn mixed_twiddles_preserve_full_and_shifted_transforms() {
        let mut rng = SmallRng::seed_from_u64(41);
        for width in [1, 4, 16, 64] {
            for shift in [
                BinaryField128::ZERO,
                BinaryField128::cantor_basis(15),
                BinaryField128::from_repr(1 << 127),
            ] {
                let mat = RowMajorMatrix::<BinaryField128>::rand(&mut rng, 16, width);
                let expected = NaiveAdditiveNtt::default().shifted_ntt_batch(mat.clone(), shift);
                let mut actual = mat.clone();
                super::transform(&mut actual, shift, false);
                assert_eq!(actual, expected);
                super::transform(&mut actual, shift, true);
                assert_eq!(actual, mat);
            }
        }
    }

    #[test]
    fn mixed_twiddles_agree_with_generic_tower_at_larger_sizes() {
        let mut rng = SmallRng::seed_from_u64(43);
        for shift in [BinaryField128::ZERO, BinaryField128::from_repr(1 << 127)] {
            let mat = RowMajorMatrix::<BinaryField128>::rand(&mut rng, 1024, 16);
            let expected = LchNtt::default().shifted_ntt_batch(mat.clone(), shift);
            let mut actual = mat.clone();
            super::transform(&mut actual, shift, false);
            assert_eq!(actual, expected);
            super::transform(&mut actual, shift, true);
            assert_eq!(actual, mat);
        }
    }
}
