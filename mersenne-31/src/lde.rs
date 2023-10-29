use alloc::vec::Vec;

use itertools::Itertools;
use p3_field::{AbstractField, Field, TwoAdicField};
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::{Matrix, MatrixRowSlices};
use p3_util::log2_strict_usize;

use crate::{Mersenne31, Mersenne31Complex, Mersenne31ComplexRadix2Dit, Mersenne31Dft};

type Base = Mersenne31;
type Ext = Mersenne31Complex<Base>;
type Dft = Mersenne31ComplexRadix2Dit;

pub struct Mersenne31ComplexLDE {
    c: Vec<Ext>,
    base_bits: usize,
    mat: RowMajorMatrix<Base>,
}

impl Mersenne31ComplexLDE {
    pub fn lde_batch_compress(
        points: RowMajorMatrix<Base>,
        added_bits: usize,
    ) -> Mersenne31ComplexLDE {
        let base_bits = log2_strict_usize(points.height());
        let bits = base_bits + added_bits;

        // this function only works when the evaluation domain is in the unit circle
        // x^{p+1} = 1
        assert!(bits <= 31);

        let mut polys = Mersenne31Dft::dft_batch::<Dft>(points);

        // store c0
        let c = polys
            .row_slice(0)
            .iter()
            .map(|x| (*x).div_2exp_u64(base_bits as u64))
            .collect_vec();

        // p(x)-c0
        polys.row_mut(0).into_iter().for_each(|x| *x = Ext::zero());

        // The coset is the chracterized by the quotient set
        // NB: current DFT is based on inverse group (1, g^-1,g^-2,...) so we need to inverse the generator
        let cosets = Ext::two_adic_generator(base_bits + added_bits)
            .inverse()
            .powers()
            .take(1 << added_bits)
            .map(|tau| {
                // tau^{-|H|/2} = (tau^{|H|/2})*
                let phi_inv = tau.exp_power_of_2(base_bits - 1).inverse();

                // { tau^k }
                let tks: Vec<Mersenne31Complex<Mersenne31>> =
                    tau.powers().take(polys.height()).collect_vec();

                let scaled = RowMajorMatrix::new(
                    polys
                        .rows()
                        .zip(&tks)
                        .flat_map(|(row, tk)| row.iter().map(|&x| *tk * x * phi_inv))
                        .collect_vec(),
                    polys.width(),
                );
                Mersenne31Dft::idft_batch::<Dft>(scaled)
            })
            .collect_vec();

        let mut value = Vec::with_capacity(polys.width() * (1 << bits));

        // reorganize the value to allign with the group structure of evaluation domain.
        for i in 0..(1 << base_bits) {
            value.extend(cosets.iter().map(|x| x.row_slice(i)).flatten());
        }

        Self {
            c,
            base_bits,
            mat: RowMajorMatrix::new(value, polys.width()),
        }
    }

    pub fn decompress(&self) -> RowMajorMatrix<Ext> {
        let bits = log2_strict_usize(self.mat.height());
        let added_bits = bits - self.base_bits;

        let mut output = Vec::with_capacity(self.mat.height() * self.mat.width());

        let phis = Ext::two_adic_generator(bits)
            .powers()
            .take(1 << added_bits)
            .map(|x| x.exp_power_of_2(self.base_bits - 1))
            .collect_vec();

        for chunk in self.mat.rows().chunks(1 << added_bits).into_iter() {
            output.extend(
                chunk.into_iter().zip(&phis).flat_map(|(row, phi)| {
                    row.iter().zip(&self.c).map(|(&x, &c0)| ((*phi * x) + c0))
                }),
            )
        }

        RowMajorMatrix::new(output, self.mat.width())
    }
}

#[cfg(test)]
mod tests {
    use p3_lde::{NaiveSubgroupLde, TwoAdicLde};
    use rand::distributions::{Distribution, Standard};
    use rand::{thread_rng, Rng};

    use super::*;
    use crate::Mersenne31;

    type Base = Mersenne31;

    #[test]
    fn consistency()
    where
        Standard: Distribution<Base>,
    {
        const BASE_BITS: usize = 4;
        const ADDED_BITS: usize = 1;
        const N: usize = 1 << BASE_BITS;

        let value = thread_rng()
            .sample_iter(Standard)
            .take(N)
            .collect::<Vec<Base>>();
        let input_real = RowMajorMatrix::new_col(value);

        let expected: RowMajorMatrix<Ext> =
            NaiveSubgroupLde.lde_batch(input_real.clone(), ADDED_BITS);

        let compressed = Mersenne31ComplexLDE::lde_batch_compress(input_real, ADDED_BITS);
        let output = compressed.decompress();

        assert_eq!(expected, output);
    }
}
