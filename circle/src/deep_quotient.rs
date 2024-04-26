use alloc::vec::Vec;

use itertools::{izip, Itertools};
use p3_field::extension::{Complex, ComplexExtendable};
use p3_field::{batch_multiplicative_inverse, dot_product, ExtensionField};
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;
use p3_maybe_rayon::prelude::*;
use p3_util::log2_strict_usize;
use tracing::instrument;

use crate::domain::CircleDomain;
use crate::util::{v_n, v_p};

/// Compute numerator and denominator of the "vanishing part" of the DEEP quotient
/// Section 6, Remark 21 of Circle Starks (page 30 of first edition PDF)
/// Re(1/v_gamma) + alpha^L Im(1/v_gamma)
/// (Other "part" is \bar g - \bar v_gamma)
pub(crate) fn deep_quotient_vanishing_part<F: ComplexExtendable, EF: ExtensionField<F>>(
    x: Complex<F>,
    zeta: Complex<EF>,
    alpha_pow_width: EF,
) -> (EF, EF) {
    let [re_v_zeta, im_v_zeta] = v_p(zeta, x).to_array();
    (
        re_v_zeta - alpha_pow_width * im_v_zeta,
        re_v_zeta.square() + im_v_zeta.square(),
    )
}

pub(crate) fn deep_quotient_reduce_row<F: ComplexExtendable, EF: ExtensionField<F>>(
    alpha: EF,
    x: Complex<F>,
    zeta: Complex<EF>,
    ps_at_x: &[F],
    ps_at_zeta: &[EF],
) -> EF {
    let (vp_num, vp_denom) =
        deep_quotient_vanishing_part(x, zeta, alpha.exp_u64(ps_at_x.len() as u64));
    (vp_num / vp_denom)
        * dot_product::<EF, _, _>(
            alpha.powers(),
            izip!(ps_at_x, ps_at_zeta).map(|(&p_at_x, &p_at_zeta)| -p_at_zeta + p_at_x),
        )
}

/// Same as `deep_quotient_reduce_row`, but reduces a whole matrix into a column, taking advantage of batch inverses.
#[instrument(skip_all, fields(log_n = domain.log_n))]
pub(crate) fn deep_quotient_reduce_matrix<F: ComplexExtendable, EF: ExtensionField<F>>(
    alpha: EF,
    domain: &CircleDomain<F>,
    mat: &RowMajorMatrix<F>,
    zeta: Complex<EF>,
    ps_at_zeta: &[EF],
) -> Vec<EF> {
    let alpha_pow_width = alpha.exp_u64(mat.width() as u64);
    let (vp_nums, vp_denoms): (Vec<_>, Vec<_>) = domain
        .points()
        .map(|x| deep_quotient_vanishing_part(x, zeta, alpha_pow_width))
        .unzip();
    let vp_denom_invs = batch_multiplicative_inverse(&vp_denoms);

    let alpha_reduced_ps_at_zeta: EF = dot_product(alpha.powers(), ps_at_zeta.iter().copied());

    mat.dot_ext_powers(alpha)
        .zip(vp_nums.into_par_iter())
        .zip(vp_denom_invs.into_par_iter())
        .map(|((reduced_ps_at_x, vp_num), vp_denom_inv)| {
            vp_num * vp_denom_inv * (reduced_ps_at_x - alpha_reduced_ps_at_zeta)
        })
        .collect()
}

/// Given evaluations over lde_domain, extract the multiple of the vanishing poly of orig_domain
/// See Section 4.3, Lemma 6: < v_n, f > = 0 for any f in FFT space
/// So, we find the "error" (a scalar multiple of v_n) and remove it
/// |lde_domain| > |orig_domain|
#[instrument(skip_all, fields(bits = log2_strict_usize(lde.len())))]
pub fn extract_lambda<F: ComplexExtendable, EF: ExtensionField<F>>(
    orig_domain: CircleDomain<F>,
    lde_domain: CircleDomain<F>,
    lde: &mut [EF],
) -> EF {
    let num_cosets = 1 << (lde_domain.log_n - orig_domain.log_n);

    // v_n is constant on cosets of the same size as orig_domain, so we only have
    // as many unique values as we have cosets.
    let v_d_init = lde_domain
        .points()
        .take(num_cosets)
        .map(|x| v_n(x.real(), orig_domain.log_n))
        .collect_vec();

    // The unique values are repeated over the rest of the domain like
    // 0 1 2 .. n-1 n n n-1 .. 1 0 0 1 ..
    let v_d = v_d_init
        .iter()
        .chain(v_d_init.iter().rev())
        .cycle()
        .copied();

    // < v_d, v_d >
    // This formula was determined experimentally...
    let v_d_2 = F::two().exp_u64(lde_domain.log_n as u64 - 1);

    let lambda = dot_product::<EF, _, _>(lde.iter().copied(), v_d.clone()) * v_d_2.inverse();

    for (y, v_x) in izip!(lde, v_d) {
        *y -= lambda * v_x;
    }

    lambda
}
