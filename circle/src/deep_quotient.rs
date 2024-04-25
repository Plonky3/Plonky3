use alloc::vec::Vec;

use itertools::{izip, Itertools};
use p3_commit::PolynomialSpace;
use p3_field::extension::{Complex, ComplexExtendable, ExtensionPowersReducer};
use p3_field::{batch_multiplicative_inverse, ExtensionField};
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;
use p3_maybe_rayon::prelude::*;
use p3_util::log2_strict_usize;
use tracing::instrument;

use crate::domain::CircleDomain;
use crate::util::v_n;

/// Compute numerator and denominator of the "vanishing part" of the DEEP quotient
/// Section 6, Remark 21 of Circle Starks (page 30 of first edition PDF)
/// Re(1/v_gamma) + alpha^L Im(1/v_gamma)
/// (Other "part" is \bar g - \bar v_gamma)
pub(crate) fn deep_quotient_vanishing_part<F: ComplexExtendable, EF: ExtensionField<F>>(
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
    alpha_reducer: &ExtensionPowersReducer<F, EF>,
    x: Complex<F>,
    zeta: Complex<EF>,
    ps_at_x: &[F],
    ps_at_zeta: &[EF],
) -> EF {
    let (vp_num, vp_denom) =
        deep_quotient_vanishing_part(x, zeta, alpha_reducer.factor.exp_u64(ps_at_x.len() as u64));
    vp_num
        * vp_denom.inverse()
        * (alpha_reducer.reduce_base(ps_at_x) - alpha_reducer.reduce_ext(ps_at_zeta))
}

/// Same as `deep_quotient_reduce_row`, but reduces a whole matrix into a column, taking advantage of batch inverses.
#[instrument(skip_all, fields(log_n = domain.log_n))]
pub(crate) fn deep_quotient_reduce_matrix<F: ComplexExtendable, EF: ExtensionField<F>>(
    alpha_reducer: &mut ExtensionPowersReducer<F, EF>,
    domain: &CircleDomain<F>,
    mat: &RowMajorMatrix<F>,
    zeta: Complex<EF>,
    ps_at_zeta: &[EF],
) -> Vec<EF> {
    // +1 so we can get alpha^width
    alpha_reducer.prepare_for_width(mat.width() + 1);
    let alpha_pow_width = alpha_reducer.powers[mat.width()];
    let (vp_nums, vp_denoms): (Vec<_>, Vec<_>) = domain
        .points()
        .map(|x| deep_quotient_vanishing_part(x, zeta, alpha_pow_width))
        .unzip();
    let vp_denom_invs = batch_multiplicative_inverse(&vp_denoms);
    let alpha_reduced_ps_at_zeta = alpha_reducer.reduce_ext(ps_at_zeta);
    mat.par_row_slices()
        .zip(vp_nums)
        .zip(vp_denom_invs)
        .map(|((ps_at_x, vp_num), vp_denom_inv)| {
            vp_num * vp_denom_inv * (alpha_reducer.reduce_base(ps_at_x) - alpha_reduced_ps_at_zeta)
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
