use alloc::vec::Vec;

use itertools::{izip, Itertools};
use p3_commit::PolynomialSpace;
use p3_field::extension::{Complex, ComplexExtendable};
use p3_field::{batch_multiplicative_inverse, ExtensionField};
use p3_fri::PowersReducer;
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;
use p3_maybe_rayon::prelude::*;
use p3_util::log2_strict_usize;
use tracing::instrument;

use crate::domain::CircleDomain;
use crate::util::v_n;
use crate::Cfft;

/// Compute numerator and denominator of the left hand side of the DEEP quotient
/// Section 6, Remark 21 of Circle Starks (page 30 of first edition PDF)
/// Re(1/v_gamma) + alpha^L Im(1/v_gamma)
/// ("right hand side" is \bar g - \bar v_gamma)
pub(crate) fn deep_quotient_lhs<F: ComplexExtendable, EF: ExtensionField<F>>(
    x: Complex<F>,
    zeta: Complex<EF>,
    alpha_pow_width: EF,
) -> (EF, EF) {
    let x_rotate_zeta: Complex<EF> = x.rotate(zeta.conjugate());

    let v_gamma_re: EF = EF::one() - x_rotate_zeta.real();
    let v_gamma_im: EF = x_rotate_zeta.imag();

    (
        v_gamma_re - alpha_pow_width * v_gamma_im,
        v_gamma_re.square() + v_gamma_im.square(),
    )
}

pub(crate) fn deep_quotient_reduce_row<F: ComplexExtendable, EF: ExtensionField<F>>(
    alpha_reducer: &PowersReducer<F, EF>,
    lhs_num: EF,
    lhs_denom_inv: EF,
    ps_at_x: &[F],
    alpha_reduced_ps_at_zeta: EF,
) -> EF {
    lhs_num * lhs_denom_inv * (alpha_reducer.reduce_base(&ps_at_x) - alpha_reduced_ps_at_zeta)
}

#[instrument(skip_all, fields(log_n = domain.log_n))]
pub(crate) fn deep_quotient_reduce_matrix<F: ComplexExtendable, EF: ExtensionField<F>>(
    domain: &CircleDomain<F>,
    mat: &RowMajorMatrix<F>,
    zeta: Complex<EF>,
    ps_at_zeta: &[EF],
    alpha_reducer: &PowersReducer<F, EF>,
    alpha_pow_width: EF,
) -> Vec<EF> {
    let (lhs_nums, lhs_denoms): (Vec<_>, Vec<_>) = domain
        .points()
        .map(|x| deep_quotient_lhs(x, zeta, alpha_pow_width))
        .unzip();
    let lhs_denom_invs = batch_multiplicative_inverse(&lhs_denoms);
    let alpha_reduced_ps_at_zeta = alpha_reducer.reduce_ext(&ps_at_zeta);
    mat.par_row_slices()
        .zip(lhs_nums)
        .zip(lhs_denom_invs)
        .map(|((ps_at_x, lhs_num), lhs_denom_inv)| {
            deep_quotient_reduce_row(
                alpha_reducer,
                lhs_num,
                lhs_denom_inv,
                ps_at_x,
                alpha_reduced_ps_at_zeta,
            )
        })
        .collect()
}

/// Given evaluations over lde_domain, extract the multiple of the vanishing poly of orig_domain
/// |lde_domain| > |orig_domain|
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
