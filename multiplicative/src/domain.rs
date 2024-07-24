use itertools::Itertools;
use p3_commit::{LagrangeSelectors, PolynomialSpace};
use p3_field::{
    batch_multiplicative_inverse, cyclic_subgroup_coset_known_order, ExtensionField, TwoAdicField,
};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_util::{log2_ceil_usize, log2_strict_usize};

#[derive(Copy, Clone, Debug)]
pub struct MultiplicativeCoset<Val: TwoAdicField> {
    pub log_n: usize,
    pub shift: Val,
}

impl<Val: TwoAdicField> MultiplicativeCoset<Val> {
    fn gen(&self) -> Val {
        Val::two_adic_generator(self.log_n)
    }
}

impl<Val: TwoAdicField> PolynomialSpace for MultiplicativeCoset<Val> {
    type Val = Val;

    fn size(&self) -> usize {
        1 << self.log_n
    }

    fn first_point(&self) -> Val {
        self.shift
    }
    fn next_point<Ext: ExtensionField<Val>>(&self, x: Ext) -> Option<Ext> {
        Some(x * self.gen())
    }

    fn create_disjoint_domain(&self, min_size: usize) -> Self {
        Self {
            log_n: log2_ceil_usize(min_size),
            shift: self.shift * Val::generator(),
        }
    }
    fn zp_at_point<Ext: ExtensionField<Val>>(&self, point: Ext) -> Ext {
        (point * self.shift.inverse()).exp_power_of_2(self.log_n) - Ext::one()
    }

    fn split_domains(&self, num_chunks: usize) -> Vec<Self> {
        let log_chunks = log2_strict_usize(num_chunks);
        (0..num_chunks)
            .map(|i| Self {
                log_n: self.log_n - log_chunks,
                shift: self.shift * self.gen().exp_u64(i as u64),
            })
            .collect()
    }
    fn split_evals(
        &self,
        num_chunks: usize,
        evals: RowMajorMatrix<Val>,
    ) -> Vec<RowMajorMatrix<Val>> {
        // todo less copy
        (0..num_chunks)
            .map(|i| {
                evals
                    .as_view()
                    .vertically_strided(num_chunks, i)
                    .to_row_major_matrix()
            })
            .collect()
    }

    fn selectors_at_point<Ext: ExtensionField<Val>>(&self, point: Ext) -> LagrangeSelectors<Ext> {
        let unshifted_point = point * self.shift.inverse();
        let z_h = unshifted_point.exp_power_of_2(self.log_n) - Ext::one();
        LagrangeSelectors {
            is_first_row: z_h / (unshifted_point - Ext::one()),
            is_last_row: z_h / (unshifted_point - self.gen().inverse()),
            is_transition: unshifted_point - self.gen().inverse(),
            inv_zeroifier: z_h.inverse(),
        }
    }

    fn selectors_on_coset(&self, coset: Self) -> LagrangeSelectors<Vec<Val>> {
        assert_eq!(self.shift, Val::one());
        assert_ne!(coset.shift, Val::one());
        assert!(coset.log_n >= self.log_n);
        let rate_bits = coset.log_n - self.log_n;

        let s_pow_n = coset.shift.exp_power_of_2(self.log_n);
        // evals of Z_H(X) = X^n - 1
        let evals = Val::two_adic_generator(rate_bits)
            .powers()
            .take(1 << rate_bits)
            .map(|x| s_pow_n * x - Val::one())
            .collect_vec();

        let xs = cyclic_subgroup_coset_known_order(coset.gen(), coset.shift, 1 << coset.log_n)
            .collect_vec();

        let single_point_selector = |i: u64| {
            let coset_i = self.gen().exp_u64(i);
            let denoms = xs.iter().map(|&x| x - coset_i).collect_vec();
            let invs = batch_multiplicative_inverse(&denoms);
            evals
                .iter()
                .cycle()
                .zip(invs)
                .map(|(&z_h, inv)| z_h * inv)
                .collect_vec()
        };

        let subgroup_last = self.gen().inverse();

        LagrangeSelectors {
            is_first_row: single_point_selector(0),
            is_last_row: single_point_selector((1 << self.log_n) - 1),
            is_transition: xs.into_iter().map(|x| x - subgroup_last).collect(),
            inv_zeroifier: batch_multiplicative_inverse(&evals)
                .into_iter()
                .cycle()
                .take(1 << coset.log_n)
                .collect(),
        }
    }
}
