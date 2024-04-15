use alloc::vec::Vec;
use itertools::{izip, Itertools};
use p3_commit::PolynomialSpace;
use p3_field::{
    extension::{Complex, ComplexExtendable},
    ExtensionField,
};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_util::log2_strict_usize;
use tracing::instrument;

use crate::{domain::CircleDomain, util::v_n, Cfft};

fn reduce_matrix<F: ComplexExtendable, EF: ExtensionField<F>>(
    domain: CircleDomain<F>,
    p: RowMajorMatrix<F>,
    zeta: Complex<EF>,
    ps_at_zeta: &[EF],
    mu: EF,
) -> Vec<EF> {
    let mu_pow_width = mu.exp_u64(p.width() as u64);
    p.rows()
        .zip(domain.points())
        .map(|(row, x)| {
            let x_rotate_zeta: Complex<EF> = x.rotate(zeta.conjugate());
            let v_gamma_re: EF = EF::one() - x_rotate_zeta.real();
            let v_gamma_im: EF = x_rotate_zeta.imag();
            let lhs: EF = (v_gamma_re - mu_pow_width * v_gamma_im)
                / (v_gamma_re.square() + v_gamma_im.square());
            lhs * izip!(row, ps_at_zeta, mu.powers())
                .map(|(p_at_x, &p_at_zeta, mu_pow)| mu_pow * (-p_at_zeta + p_at_x))
                .sum::<EF>()
        })
        .collect()
}

#[instrument(skip_all, fields(bits = log2_strict_usize(lde.len())))]
pub fn extract_lambda<F: ComplexExtendable, EF: ExtensionField<F>>(
    orig_domain: CircleDomain<F>,
    lde_domain: CircleDomain<F>,
    lde: &mut [EF],
) -> EF {
    // TODO: precompute
    let v_d = lde_domain
        .points()
        .map(|x| v_n(x.real(), log2_strict_usize(orig_domain.size())))
        .collect_vec();

    let v_d_2: F = v_d.iter().map(|x| x.square()).sum();

    let lde_dot_v_d: EF = izip!(lde.iter(), &v_d).map(|(&a, &b)| a * b).sum();
    let lambda = lde_dot_v_d * v_d_2.inverse();

    for (y, v_x) in izip!(lde, v_d) {
        *y -= lambda * v_x;
    }

    lambda
}

pub fn is_low_degree<F: ComplexExtendable>(evals: &RowMajorMatrix<F>) -> bool {
    let cfft = Cfft::default();
    cfft.cfft_batch(evals.clone())
        .rows()
        .skip(1)
        .step_by(2)
        .all(|row| row.into_iter().all(|col| col.is_zero()))
}

#[cfg(test)]
mod tests {
    use p3_field::extension::{BinomialExtensionField, Complex};
    use p3_matrix::{dense::RowMajorMatrix, Matrix};
    use p3_mersenne_31::Mersenne31;
    use p3_util::log2_strict_usize;
    use rand::{thread_rng, Rng};

    use crate::{
        domain::CircleDomain,
        util::{univariate_to_point, v_n},
        Cfft,
    };

    use super::*;

    type F = Mersenne31;
    type EF = BinomialExtensionField<Mersenne31, 2>;
    // type EF = Complex<Complex<Mersenne31>>;

    fn open_mat_at_point(
        domain: CircleDomain<F>,
        p: RowMajorMatrix<F>,
        pt: Complex<EF>,
    ) -> Vec<EF> {
        let log_n = log2_strict_usize(p.height());
        let basis: Vec<_> = domain.lagrange_basis(pt);
        let v_n_at_zeta = v_n(pt.real(), log_n) - v_n(domain.shift.real(), log_n);
        p.columnwise_dot_product(&basis)
            .into_iter()
            .map(|x| x * v_n_at_zeta)
            .collect()
    }

    #[test]
    fn test_quotienting() {
        let mut rng = thread_rng();
        let log_n = 2;
        let cfft = Cfft::<F>::default();

        let trace_domain = CircleDomain::<F>::standard(log_n);
        let lde_domain = CircleDomain::<F>::standard(log_n + 1);

        let trace = RowMajorMatrix::<F>::rand(&mut rng, 1 << log_n, 1);
        let lde = cfft.lde(trace.clone(), trace_domain, lde_domain);
        let zeta: EF = rng.gen();
        let zeta_pt: Complex<EF> = univariate_to_point(zeta).unwrap();
        let trace_at_zeta = open_mat_at_point(trace_domain, trace, zeta_pt);
        assert_eq!(
            trace_at_zeta,
            open_mat_at_point(lde_domain, lde.clone(), zeta_pt)
        );

        assert!(is_low_degree(&lde));

        let mu: EF = rng.gen();
        let q = reduce_matrix(lde_domain, lde.clone(), zeta_pt, &trace_at_zeta, mu);

        // dbg!(cfft.cfft_batch(RowMajorMatrix::new_col(q.clone()).flatten_to_base()));

        let mut q_corr = q.clone();
        let lambda = extract_lambda(trace_domain, lde_domain, &mut q_corr);
        // dbg!(&lambda);

        // dbg!(cfft.cfft_batch(RowMajorMatrix::new_col(q_corr.clone()).flatten_to_base()));
    }
}
