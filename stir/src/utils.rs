use alloc::vec;
use alloc::vec::Vec;
use core::iter;

use itertools::{iterate, izip, Itertools};
use p3_challenger::{CanObserve, FieldChallenger};
use p3_dft::{Radix2Dit, TwoAdicSubgroupDft};
use p3_field::coset::TwoAdicMultiplicativeCoset;
use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_poly::Polynomial;

// Syntactic sugar for the proof-of-work computation
#[inline]
pub(crate) fn compute_pow(security_level: usize, error: f64) -> f64 {
    0f64.max(security_level as f64 - error)
}

// Given a polynomial f and a folding coefficient c, this function computes the usual folding
// (same as in FRI) of the requested arity/folding factor:
//   folded(x) = f_0(x) + c * f_1(x) + ... + c^(arity - 1) * f_(arity - 1)(x)
// where f_i is the polynomial whose j_th coefficient is the (i + j * arity)-th
// coefficient of f.
pub(crate) fn fold_polynomial<F: TwoAdicField>(
    // The polynomial to fold
    polynomial: &Polynomial<F>,
    // The folding coefficient
    c: F,
    // The log2 of the folding factor
    log_folding_factor: usize,
) -> Polynomial<F> {
    let deg = if let Some(deg) = polynomial.degree() {
        deg
    } else {
        return Polynomial::zero();
    };

    let folding_factor = 1 << log_folding_factor;
    let fold_size = (deg + 1).div_ceil(folding_factor);

    // Powers of c
    let folding_powers = iter::successors(Some(F::ONE), |&x| Some(x * c))
        .take(folding_factor)
        .collect_vec();

    let mut folded_coeffs = vec![F::ZERO; fold_size];

    // Computing the folded coefficients as described above
    for (i, coeff) in polynomial.coeffs().iter().enumerate() {
        folded_coeffs[i / folding_factor] += *coeff * folding_powers[i % folding_factor];
    }

    Polynomial::from_coeffs(folded_coeffs)
}

// Multiply the given polynomial by the power polynomial
//   1 + c * x + c^2 * x^2 + ... + c^d * x^d
// using the more efficient method of multiplying by
// (c * x - 1)^(d + 1) and then dividing by (c * x - 1).
pub(crate) fn multiply_by_power_polynomial<F: Field>(
    // The polynomial to multiply
    polynomial: &Polynomial<F>,
    // The coefficient c defining the power polynomial
    c: F,
    // The degree d of the desired power polynomial
    degree: usize,
) -> Polynomial<F> {
    // The power polynomial of degree 0 is identically 1
    if degree == 0 {
        return polynomial.clone();
    }

    // Multiplication by ((c * x)^(d + 1) - 1) / (c * x - 1) can be done more efficiently
    // by a special-case multiplication followed by a special-case division.

    // We first compute polynomial * ((c * x)^(d + 1) - 1), i. e.:
    //   [0 ... 0] || [c^(degree + 1) * coeffs]
    // -  coeffs   || [0         ...         0]
    let c_pow_n_1 = c.exp_u64((degree + 1) as u64);
    let mut new_coeffs = vec![F::ZERO; degree + 1];
    new_coeffs.extend(polynomial.coeffs().iter().map(|&coeff| coeff * c_pow_n_1));
    for (c1, c2) in new_coeffs.iter_mut().zip(polynomial.coeffs().iter()) {
        *c1 -= *c2;
    }

    // Now we divide by c*x - 1 by dividing by x - (1/c) and multiplying by c afterwards
    let mut last = *new_coeffs.iter().last().unwrap();
    let c_inv = c.inverse();
    for new_c in new_coeffs.iter_mut().rev().skip(1) {
        *new_c += c_inv * last;
        last = *new_c;
    }

    assert!(new_coeffs.remove(0) == F::ZERO);

    new_coeffs.iter_mut().for_each(|c| *c *= c_inv);

    Polynomial::from_coeffs(new_coeffs)
}

// Compute the evaluation of a folded polynomial at a point given the
// evaluations of the original polynomial at the k-th roots of that point, where
// k is the folding factor
pub(crate) fn fold_evaluations<F: TwoAdicField>(
    // The evaluations of the original polynomial
    evals: Vec<F>,
    // point_root is a k-th root of the point where we evaluate the folded
    // polynomial. Its inverse can be supplied to make use of batch inversion
    // outside of this function.
    (point_root, point_root_inv_hint): (F, Option<F>),
    // The log2 of the folding factor
    log_folding_factor: usize,
    // omega is the generator of the coset of k-th roots of the point - in other
    // words, this is simply a primitive k-th root of unity. Its inverse can be
    // suplied for efficiency, since it will be used across many calls.
    (omega, omega_inv_hint): (F, Option<F>),
    // The folding coefficient
    c: F,
    // The inverse of 2 can be supplied to avoid recomputation in every call
    two_inv_hint: Option<F>,
) -> F {
    // Let fold_k(g, b) denote the k-ary folding of a polynomial g with
    // coefficient b. Then one has:
    //   fold_k(h, coeff) = fold_2(..., fold_2(h, c), c^2, c^4, ..., ),
    // where the ellipses denote the log2(k)-fold composition of fold_2. The
    // same applies to the evaluations, i. e. one can compute the k-ary folding
    // of evals through repeated binary foldings.

    let arity = 1 << log_folding_factor;
    assert!(evals.len() == arity);

    // We first construct the list `gammas` of values c / y_j, where y_j runs
    // over half the k-th roots of point_root^k. In particular, for each pair
    // of roots {y_j, -y_j} (note that k is even), we only store one of the
    // two.

    let inv_omega = omega_inv_hint.unwrap_or_else(|| omega.inverse());
    let inv_point_root = point_root_inv_hint.unwrap_or_else(|| point_root.inverse());

    let mut gammas = iterate(inv_point_root * c, |&x| x * inv_omega)
        .take(arity / 2)
        .collect_vec();

    let mut result = evals;

    // Repeatedly binary-fold until only one evaluation is left
    while result.len() > 1 {
        result = fold_evaluations_binary(result, &gammas, two_inv_hint);

        // The gammas for the next step are half the squares of the current
        // gammas
        gammas = gammas[..(gammas.len() / 2)]
            .iter()
            .map(|&gamma| gamma.square())
            .collect_vec();
    }

    result.pop().unwrap()
}

// Compute the binary folding of the given list of evaluations
fn fold_evaluations_binary<F: TwoAdicField>(
    // The evaluations to fold
    evals: Vec<F>,
    // The list of c / y_j with y_j as in fold_evaluations()
    gammas: &[F],
    // The inverse of 2 can be supplied to avoid recomputation in every call
    two_inv_hint: Option<F>,
) -> Vec<F> {
    let cutoff = evals.len() / 2;
    let low_evals = evals[..cutoff].iter();
    let high_evals = evals[cutoff..].iter();

    izip!(low_evals, high_evals, gammas.iter())
        .map(|(&eval_1, &eval_2, &gamma)| fold_evaluation_pair(eval_1, eval_2, gamma, two_inv_hint))
        .collect_vec()
}

// Compute the binary folding of the evaluations of a polynomial at a point and
// its additive inverse
#[inline]
fn fold_evaluation_pair<F: TwoAdicField>(
    // f(w), where w^2 is the point we are evaluating fold_2(f) at
    eval_1: F,
    // f(-w) with w as above
    eval_2: F,
    // c / w, where c is the folding coefficient
    gamma: F,
    // The inverse of 2 can be supplied to avoid recomputation in every call
    two_inv_hint: Option<F>,
) -> F {
    let two_inv = two_inv_hint.unwrap_or_else(|| F::TWO.inverse());
    // This expression uses two multiplications, as opposed to three in the more
    // symmetric 1/2 * ((1 + gamma) * eval_1 + (1 - gamma) * eval_2)
    two_inv * (eval_1 + eval_2 + gamma * (eval_1 - eval_2))
}

// Observe a list of extension field elements, preceded by its length for
// security
pub(crate) fn observe_ext_slice_with_size<F: Field, E: ExtensionField<F>, C: FieldChallenger<F>>(
    challenger: &mut C,
    values: &[E],
) {
    let size: u64 = values.len().try_into().expect("Slice too long to observe");
    challenger.observe_algebra_element(E::from_u64(size));
    values
        .iter()
        .for_each(|&v| challenger.observe_algebra_element(v));
}

// Observe a list of usize, preceded by its length for security.
pub(crate) fn observe_usize_slice<F: Field, C: CanObserve<F>>(
    challenger: &mut C,
    values: &[usize],
    absorb_size: bool,
) {
    if absorb_size {
        let size: u64 = values.len().try_into().expect("Slice too long to observe");
        challenger.observe(F::from_u64(size));
    }

    values.iter().for_each(|&v| {
        challenger.observe(F::from_u64(
            v.try_into().expect("Value too large to observe"),
        ))
    });
}

// Returns the list of evaluations of the polynomial with given coefficients
// over the coset, that is, `polynomial(shift * g^0), polynomial(shift *
// g^1), ..., polynomial(shift * g^(2^log_size - 1))`.
//
// This function panics if the degree of the polynomial is greater than or
// equal to the size of the coset. In this case, a larger domain should be
// used instead.
pub fn domain_dft<F: TwoAdicField>(
    domain: TwoAdicMultiplicativeCoset<F>,
    poly_coeffs: Vec<F>,
    dft: &Radix2Dit<F>,
) -> Vec<F> {
    let size = domain.size();
    assert!(
        poly_coeffs.len() <= size,
        "More coefficients were supplied than the size of the coset (note \
            that leading zeros are not removed inside this function). Consider \
            constructing a larger coset, evaluating therein and retaining the \
            appropriate evaluations (which will be interleaved with those in \
            the rest of the large domain)."
    );

    if poly_coeffs.is_empty() {
        return vec![F::ZERO; size];
    } else if poly_coeffs.len() == 1 {
        return vec![poly_coeffs[0]; size];
    }

    let mut coeffs = poly_coeffs;
    coeffs.resize(size, F::ZERO);

    if domain.shift() == domain.subgroup_generator() {
        // In this case it is more efficient to use a plain FFT without
        // shift, and then (cyclically) rotate the resulting evaluations by
        // one position. This case is particularly frequentNote that this case is not unusual and is, e. g.
        // used in this repository's implementation of STIR
        let mut evals = dft.dft(coeffs);
        evals.rotate_left(1);
        evals
    } else {
        dft.coset_dft(coeffs, domain.shift())
    }
}

#[cfg(test)]
mod tests {

    use core::assert_eq;

    use iter::Iterator;
    use p3_baby_bear::BabyBear;
    use p3_field::coset::TwoAdicMultiplicativeCoset;
    use p3_field::PrimeCharacteristicRing;
    use p3_poly::test_utils::rand_poly_rng;
    use p3_poly::Polynomial;
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    use super::*;

    type BB = BabyBear;

    #[test]
    // Checks that fold_polynomial returns the correct polynomial computed
    // manually. All polynomials are hard-coded, but the folding randomness is
    // not.
    fn test_fold_polynomial() {
        let polynomial = Polynomial::<BB>::from_coeffs(vec![BB::ONE; 16]);
        let folding_randomness = BB::from_u32(3);

        // folding_factor = 2
        assert_eq!(
            fold_polynomial(&polynomial, folding_randomness, 1).coeffs(),
            vec![4, 4, 4, 4, 4, 4, 4, 4]
                .into_iter()
                .map(BB::from_u32)
                .collect_vec()
        );

        // folding_factor = 4
        assert_eq!(
            fold_polynomial(&polynomial, folding_randomness, 2).coeffs(),
            vec![40, 40, 40, 40]
                .into_iter()
                .map(BB::from_u32)
                .collect_vec()
        );
    }

    #[test]
    // Checks that fold_polynomial returns the correct polynomial. The
    // polynomial to be folded is constructed by collating its
    // randomly-generated limbs, which are also used to compute the expected
    // folded polynomial.
    fn test_fold_backwards() {
        let fold_degree = 5;
        let log_folding_factor = 4;

        let folding_factor = 1 << log_folding_factor;

        let mut rng = SmallRng::seed_from_u64(914);

        let folding_randomness: BB = rng.random();

        // Generating the limbs
        let folds = (0..folding_factor)
            .map(|_| rand_poly_rng::<BB>(fold_degree - 1, &mut rng))
            .collect_vec();

        let powers_of_x = iter::successors(Some(Polynomial::one()), |p| Some(&Polynomial::x() * p))
            .take(folding_factor)
            .collect_vec();

        // Constructing the polynomial to be folded
        let polynomial = folds
            .iter()
            .map(|fold| fold.compose_with_exponent(folding_factor))
            .zip(powers_of_x.iter())
            .fold(Polynomial::zero(), |acc, (raised_fold, power_of_x)| {
                &acc + &(&raised_fold * power_of_x)
            });

        let powers_of_r = iter::successors(Some(BB::ONE), |&x| Some(x * folding_randomness))
            .take(folding_factor)
            .collect_vec();

        // Constructing the expected folded polynomial as the linear combination
        // of the limbs with coefficients equal to the list of powers of r
        let expected_folded_polynomial = folds
            .iter()
            .zip(powers_of_r.iter())
            .map(|(p, r)| p * r)
            .fold(Polynomial::zero(), |acc, p| &acc + &p);

        assert_eq!(
            fold_polynomial(&polynomial, folding_randomness, log_folding_factor),
            expected_folded_polynomial
        );
    }

    #[test]
    // Checks that multiply_by_power_polynomial returns the same as
    // multiplication by power_polynomial() (the latter being more transparent,
    // but less efficient)
    fn test_multiply_by_power_polynomial() {
        let degree_polynomial = 5;
        let degree_power_polynomial = 6;

        let mut rng = SmallRng::seed_from_u64(1923);
        let coeff: BB = rng.random();
        let polynomial = rand_poly_rng(degree_polynomial, &mut rng);

        let expected = &Polynomial::power_polynomial(coeff, degree_power_polynomial) * &polynomial;

        assert_eq!(
            multiply_by_power_polynomial(&polynomial, coeff, degree_power_polynomial),
            expected
        );
    }

    // Macro to test evaluation folding with various parameters
    macro_rules! test_fold_evals_with_log_arity {
        ($log_arity:expr, $polynomial:expr, $folding_randomness:expr) => {{
            let mut rng = SmallRng::seed_from_u64(87);
            let domain = TwoAdicMultiplicativeCoset::new(rng.random(), $log_arity).unwrap();

            // Evaluating the polynomial over the domain
            let evaluations = domain
                .iter()
                .map(|x| $polynomial.evaluate(&x))
                .collect_vec();

            // We first compute the folded evaluations using the method
            // fold_evaluations()
            let folded_evaluation = fold_evaluations(
                evaluations,
                (domain.shift(), None),
                $log_arity,
                (domain.subgroup_generator(), None),
                $folding_randomness,
                None,
            );

            // The above needs to coincide with the result of evaluating the
            // folded polynomial at shift^k
            let folded_polynomial = fold_polynomial(&$polynomial, $folding_randomness, $log_arity);
            let expected_folded_evaluation =
                folded_polynomial.evaluate(&domain.shift().exp_power_of_2($log_arity));

            assert_eq!(
                folded_evaluation, expected_folded_evaluation,
                "log_arity = {}",
                $log_arity
            );
        }};
    }

    #[test]
    // Checks that fold_evaluations() returns the expected results for arities
    // k = 2^1, ..., 2^9
    fn test_fold_evaluations() {
        let mut rng = SmallRng::seed_from_u64(43);
        let polynomial = rand_poly_rng((1 << 10) - 1, &mut rng);
        let folding_randomness: BB = rng.random();

        for log_arity in 1..10 {
            test_fold_evals_with_log_arity!(log_arity, polynomial, folding_randomness)
        }
    }

    #[test]
    // Checks that fold_evaluations_binary() returns the expected results
    fn test_fold_evaluations_binary() {
        let log_domain_size = 4;
        let poly_deg = 7;

        let mut rng = SmallRng::seed_from_u64(93);

        let polynomial = rand_poly_rng(poly_deg, &mut rng);

        // Folding coefficient
        let c = rng.random();

        // Points with the same 16-th power
        let point_root: BB = rng.random();
        let point = point_root.exp_power_of_2(log_domain_size);
        let omega = BB::two_adic_generator(log_domain_size);
        let roots = iterate(point_root, |&x| x * omega)
            .take(1 << log_domain_size)
            .collect_vec();

        assert!(roots
            .iter()
            .all(|&x| x.exp_power_of_2(log_domain_size) == point));

        // Computing folded evaluations using the method fold_evaluations_binary
        let gammas = roots[0..(1 << log_domain_size) / 2]
            .iter()
            .map(|&root| c * root.inverse())
            .collect_vec();
        let evals = roots
            .iter()
            .map(|&root| polynomial.evaluate(&root))
            .collect_vec();
        let folded_evals = fold_evaluations_binary(evals.clone(), &gammas, None);

        // Computing folded evaluations by evaluating the folded polynomial
        let folded_poly = fold_polynomial(&polynomial, c, 1);
        let roots_squared = roots[0..(1 << log_domain_size) / 2]
            .iter()
            .map(|&root| root.square())
            .collect_vec();
        let expected_folded_evals = roots_squared
            .iter()
            .map(|root| folded_poly.evaluate(root))
            .collect_vec();

        assert_eq!(folded_evals, expected_folded_evals);
    }
}
