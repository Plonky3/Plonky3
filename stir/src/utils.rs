use alloc::vec;
use alloc::vec::Vec;
use core::cmp::Ordering;
#[cfg(test)]
use core::iter;

use itertools::{iterate, izip, Itertools};
use p3_challenger::{CanObserve, FieldChallenger};
use p3_dft::{Radix2Dit, TwoAdicSubgroupDft};
use p3_field::coset::TwoAdicMultiplicativeCoset;
use p3_field::{batch_multiplicative_inverse, ExtensionField, Field, TwoAdicField};
#[cfg(test)]
use rand::{
    distr::{Distribution, StandardUniform},
    Rng,
};

// Syntactic sugar for the proof-of-work computation
#[inline]
pub(crate) fn compute_pow(security_level: usize, error: f64) -> f64 {
    0f64.max(security_level as f64 - error)
}

pub(crate) fn fold_evaluations_at_domain<F: TwoAdicField>(
    evals: Vec<F>,
    domain: TwoAdicMultiplicativeCoset<F>,
    log_folding_factor: usize,
    c: F,
) -> Vec<F> {
    // NP TODO better way to handle 1/2?
    let two_inv = F::TWO.inverse();
    let denominators: Vec<F> = domain.iter().take(domain.size() / 2).collect();
    let inv_denominators = batch_multiplicative_inverse(&denominators);

    fold_evaluations_at_domain_inner(
        evals,
        domain,
        log_folding_factor,
        c,
        two_inv,
        inv_denominators,
    )
}

fn fold_evaluations_at_domain_inner<F: TwoAdicField>(
    evals: Vec<F>,
    domain: TwoAdicMultiplicativeCoset<F>,
    log_folding_factor: usize,
    c: F,
    two_inv: F,
    half_domain_invs: Vec<F>,
) -> Vec<F> {
    if log_folding_factor == 0 {
        return evals;
    }

    // We modify the evaluations and domain inverses in place to avoid
    // allocating new memory
    let mut evals = evals;
    let mut half_domain_invs = half_domain_invs;

    let half_size = half_domain_invs.len();

    let (evals_plus, evals_minus) = evals.split_at_mut(half_size);

    evals_plus
        .iter_mut()
        .zip(evals_minus.iter())
        .zip(half_domain_invs.iter())
        .for_each(|((eval_p, eval_m), inv)| {
            *eval_p = two_inv * (*eval_p + *eval_m + c * *inv * (*eval_p - *eval_m));
        });

    evals.resize(half_size, F::ZERO);

    half_domain_invs.resize(half_size / 2, F::ZERO);

    half_domain_invs.iter_mut().for_each(|inv| {
        *inv = inv.square();
    });

    let c_squared = c.square();

    // NP TODO remove this and pass only the generator
    let domain_squared = domain.exp_power_of_2(1).unwrap();

    fold_evaluations_at_domain_inner(
        evals,
        domain_squared,
        log_folding_factor - 1,
        c_squared,
        two_inv,
        half_domain_invs,
    )
}

// NP TODO now this is largely duplicated in fold_evaluationsa_at_domain

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
pub(crate) fn domain_dft<F: TwoAdicField>(
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

// NP TODO explain that this is here simply for symmetry with domain_dft
#[inline]
pub(crate) fn domain_idft<F: TwoAdicField>(
    evals: Vec<F>,
    domain: TwoAdicMultiplicativeCoset<F>,
    dft: &Radix2Dit<F>,
) -> Vec<F> {
    dft.coset_idft(evals, domain.shift())
}

// Adds two polynomials given their coefficients
pub(crate) fn add_polys<F: Field>(coeffs_1: &[F], coeffs_2: &[F]) -> Vec<F> {
    let mut sum = coeffs_1
        .iter()
        .zip(coeffs_2.iter())
        .map(|(a, b)| *a + *b)
        .collect_vec();

    match coeffs_1.len().cmp(&coeffs_2.len()) {
        Ordering::Greater => sum.extend(coeffs_1[coeffs_2.len()..].iter().cloned()),
        Ordering::Less => sum.extend(coeffs_2[coeffs_1.len()..].iter().cloned()),
        Ordering::Equal => (),
    }

    sum
}

// Returns the polynomial 1, r*x, r^2*x^2, ..., r^degree*x^degree
pub(crate) fn power_polynomial<F: Field>(r: F, degree: usize) -> Vec<F> {
    if r == F::ZERO {
        vec![F::ONE]
    } else {
        iterate(F::ONE, |&prev| prev * r).take(degree + 1).collect()
    }
}

// Returns the vanishing polynomial at the given points, i. e. the product of (x
// - p) as p runs over `points`. If the list contains duplicates of the same
// point, only one is kept.
pub(crate) fn vanishing_polynomial<F: Field>(points: impl IntoIterator<Item = F>) -> Vec<F> {
    // Deduplicating the points
    let mut points = points.into_iter().unique().collect_vec();

    assert!(
        !points.is_empty(),
        "The vanishing polynomial of an empty set is undefined"
    );

    // We iteratively multiply the polynomial (x - points[0]) by each of the
    // vanishing polynomials (x - points[i]) for i > 0
    let mut coeffs = vec![-points.pop().unwrap(), F::ONE];

    while let Some(point) = points.pop() {
        // Basic idea: add shifted and scaled versions of the current polynomial
        // For instance, if f has coefficients
        //   [2, -3, 4, 1],
        // then (x - 5) * f has coefficients
        //   [0, 2, -3, 4, 1] + (-5) * [2, -3, 4, 1, 0]

        let mut prev_coeff = F::ZERO;

        for coeff in coeffs.iter_mut() {
            let current_coeff = *coeff;
            *coeff = prev_coeff - *coeff * point;
            prev_coeff = current_coeff;
        }

        coeffs.push(F::ONE);
    }

    coeffs
}

// Returns the unique polynomial of degree <= point_to_evals.len() which maps x
// to y as (x, y) runs over point_to_evals. Panics if point_to_evals contains
// the same coordinate x in two of its pairs which differ in the evaluation y.
pub(crate) fn lagrange_interpolation<F: Field>(point_to_evals: Vec<(F, F)>) -> Vec<F> {
    if point_to_evals.is_empty() {
        panic!("The Lagrange interpolation of an empty set is undefined");
    }

    // Testing for consistency and removing duplicate points
    let point_to_evals = point_to_evals.into_iter().unique().collect_vec();

    let points = point_to_evals
        .iter()
        .map(|(x, _)| *x)
        .unique()
        .collect_vec();

    assert_eq!(
        points.len(),
        point_to_evals.len(),
        "One point has two different requested evaluations"
    );

    let vanishing_poly = vanishing_polynomial(points);

    let mut result = vec![];

    for (point, eval) in point_to_evals.into_iter() {
        // We obtain the (non-normalised) vanishing polynomial at all points
        // other than point by removing the (x - point) factor from the full
        // vanishing polynomial
        let (polynomial, _) = divide_by_vanishing_linear_polynomial(&vanishing_poly, point);

        // We normalise it so that it takes the value `eval` at `point`
        let denominator = eval_poly(&polynomial, point);
        result = add_polys(&result, &scale_poly(&polynomial, eval / denominator));
    }

    result
}

#[cfg(test)]
// Test function which subtracts two polynomials given their coefficients
pub(crate) fn subtract_polys<F: Field>(coeffs_1: &[F], coeffs_2: &[F]) -> Vec<F> {
    let minus_coeffs_2 = coeffs_2.iter().map(|c| -*c).collect_vec();

    add_polys(coeffs_1, &minus_coeffs_2)
}

#[cfg(test)]
// Test function which multiplies two polynomials given their coefficients
pub(crate) fn mul_polys<F: Field>(coeffs_1: &[F], coeffs_2: &[F]) -> Vec<F> {
    let mut result = vec![F::ZERO; coeffs_1.len() + coeffs_2.len() - 1];

    for (i, coeff_1) in coeffs_1.iter().enumerate() {
        for (j, coeff_2) in coeffs_2.iter().enumerate() {
            result[i + j] += *coeff_1 * *coeff_2;
        }
    }
    result
}

#[inline]
// Scales a polynomial by a scalar given the former's coefficients
fn scale_poly<F: Field>(coeffs: &[F], scalar: F) -> Vec<F> {
    coeffs.iter().map(|c| *c * scalar).collect_vec()
}

// Divides the polynomial with the given coefficients by (x - point), returning
// the quotient and the remainder. The latter coincides with the polynomial
// evaluated at `point` and is returned as a field element.
pub(crate) fn divide_by_vanishing_linear_polynomial<F: Field>(
    coeffs: &[F],
    point: F,
) -> (Vec<F>, F) {
    if coeffs.is_empty() {
        return (vec![], F::ZERO);
    }

    let mut quotient_coeffs = coeffs.to_vec();

    // Special case: division by x - 0 = x
    if point == F::ZERO {
        let remainder = quotient_coeffs.remove(0);
        return (quotient_coeffs, remainder);
    }

    // General case: use Ruffini's rule
    let mut quotient_coeffs_iter = quotient_coeffs.iter_mut().rev();

    let mut last = *quotient_coeffs_iter.next().unwrap();

    for new_c in quotient_coeffs_iter {
        *new_c += point * last;
        last = *new_c;
    }

    let remainder = quotient_coeffs.remove(0);

    (quotient_coeffs, remainder)
}

// Evaluates a polynomial given its coefficients at a point
pub(crate) fn eval_poly<F: Field>(coeffs: &[F], point: F) -> F {
    coeffs
        .iter()
        .rev()
        .fold(F::ZERO, |acc, coeff| acc * point + *coeff)
}

#[cfg(test)]
// Test function which, given a polynomial f and a folding coefficient
// c, computes the usual folding of the requested arity/folding factor:
//   folded(x) = f_0(x) + c * f_1(x) + ... + c^(arity - 1) * f_(arity - 1)(x)
// where f_i is the polynomial whose j_th coefficient is the (i + j * arity)-th
// coefficient of f.
pub(crate) fn fold_polynomial<F: TwoAdicField>(
    // The polynomial to fold
    polynomial: &[F],
    // The folding coefficient
    c: F,
    // The log2 of the folding factor
    log_folding_factor: usize,
) -> Vec<F> {
    if polynomial.is_empty() {
        // Folding the zero polynomial
        return vec![F::ZERO];
    }

    let deg = polynomial.len() - 1;

    let folding_factor = 1 << log_folding_factor;
    let fold_size = (deg + 1).div_ceil(folding_factor);

    // Powers of c
    let folding_powers = iter::successors(Some(F::ONE), |&x| Some(x * c))
        .take(folding_factor)
        .collect_vec();

    let mut folded_coeffs = vec![F::ZERO; fold_size];

    // Computing the folded coefficients as described above
    for (i, coeff) in polynomial.iter().enumerate() {
        folded_coeffs[i / folding_factor] += *coeff * folding_powers[i % folding_factor];
    }

    folded_coeffs
}

#[cfg(test)]
// Test function which returns a random polynomial of the exact given degree
pub(crate) fn rand_poly_coeffs<F: Field>(degree: usize, rng: &mut impl Rng) -> Vec<F>
where
    StandardUniform: Distribution<F>,
{
    let mut coeffs: Vec<F> = (0..degree).map(|_| rng.random()).collect();

    let mut leading_coeff = F::ZERO;

    while leading_coeff == F::ZERO {
        leading_coeff = rng.random();
    }

    coeffs.push(leading_coeff);

    coeffs
}

#[cfg(test)]
// Test function which returns a random polynomial of the exact given degree
// generated using seeded ChaCha20Rng.
pub(crate) fn rand_poly_coeffs_seeded<F: Field>(degree: usize, seed: Option<u64>) -> Vec<F>
where
    StandardUniform: Distribution<F>,
{
    use rand::rngs::SmallRng;
    use rand::SeedableRng;

    let mut rng = SmallRng::seed_from_u64(seed.unwrap_or(42));

    rand_poly_coeffs(degree, &mut rng)
}

#[cfg(test)]
// Test function which, given the coefficients of the polynomial f(x), returns the coefficients of
// the polynomial f(x^exponent)
fn compose_poly_with_exponent<F: Field>(coeffs: &[F], exponent: usize) -> Vec<F> {
    if coeffs.is_empty() {
        return vec![F::ZERO];
    }

    let d = coeffs.len() - 1;

    // We "stretch out" the vector of coefficients by a factor of exponent
    // filling the gaps with zeros
    let mut new_coeffs = vec![F::ZERO; d * exponent + 1];

    for (i, coeff) in coeffs.iter().enumerate() {
        new_coeffs[i * exponent] = *coeff;
    }

    new_coeffs
}

#[cfg(test)]
// Test function which divides a polynomial by another polynomial given their
// coefficients, returning the quotient and remainder
pub(crate) fn divide_poly_with_remainder<F: Field>(
    dividend: Vec<F>,
    divisor: Vec<F>,
) -> (Vec<F>, Vec<F>) {
    assert!(!divisor.is_empty(), "Cannot divide by the zero polynomial");

    // Trivial division cases
    if dividend.is_empty() {
        return (vec![], vec![]);
    }

    let d_dividend = dividend.len() - 1;
    let d_divisor = divisor.len() - 1;

    if d_dividend < d_divisor {
        return (vec![], dividend);
    }

    let mut quotient_coeffs = vec![F::ZERO; d_dividend - d_divisor + 1];
    let mut remainder = dividend.clone();

    let divisor_leading_coeff_inv = divisor.last().unwrap().inverse();

    // Ieratively compute the coefficients of the quotient
    while !remainder.is_empty() && remainder.len() - 1 >= d_divisor {
        let cur_q_coeff = *remainder.last().unwrap() * divisor_leading_coeff_inv;
        let cur_q_degree = remainder.len() - 1 - d_divisor;
        quotient_coeffs[cur_q_degree] = cur_q_coeff;

        for (i, div_coeff) in divisor.iter().cloned().enumerate() {
            remainder[cur_q_degree + i] -= cur_q_coeff * div_coeff;
        }
        while let Some(true) = remainder.last().map(|c| c.is_zero()) {
            remainder.pop();
        }
    }

    truncate_leading_zeros(&mut remainder);

    (quotient_coeffs, remainder)
}

#[cfg(test)]
// Test function which removes leading zeros from a polynomial given its
// coefficients
pub(crate) fn truncate_leading_zeros<F: Field>(coeffs: &mut Vec<F>) {
    if coeffs.is_empty() {
        return;
    }

    let mut leading_index = coeffs.len() - 1;

    while coeffs[leading_index].is_zero() {
        if leading_index == 0 {
            coeffs.clear();
            return;
        }

        leading_index -= 1;
    }

    coeffs.truncate(leading_index + 1);
}

#[cfg(test)]
mod tests {

    use core::assert_eq;

    use iter::Iterator;
    use p3_baby_bear::BabyBear;
    use p3_field::coset::TwoAdicMultiplicativeCoset;
    use p3_field::PrimeCharacteristicRing;
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    use super::*;

    type BB = BabyBear;

    #[test]
    // Checks that fold_polynomial returns the correct polynomial computed
    // manually. All polynomials are hard-coded, but the folding randomness is
    // not.
    fn test_fold_polynomial() {
        let polynomial = vec![BB::ONE; 16];
        let folding_randomness = BB::from_u32(3);

        // folding_factor = 2
        assert_eq!(
            fold_polynomial(&polynomial, folding_randomness, 1),
            vec![4, 4, 4, 4, 4, 4, 4, 4]
                .into_iter()
                .map(BB::from_u32)
                .collect_vec()
        );

        // folding_factor = 4
        assert_eq!(
            fold_polynomial(&polynomial, folding_randomness, 2),
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
            .map(|_| rand_poly_coeffs::<BB>(fold_degree, &mut rng))
            .collect_vec();

        let powers_of_x = (0..folding_factor)
            .map(|i| {
                let mut coeffs = vec![BB::ZERO; i];
                coeffs.push(BB::ONE);
                coeffs
            })
            .collect_vec();

        // Constructing the polynomial to be folded
        let polynomial = folds
            .iter()
            .map(|fold| compose_poly_with_exponent(fold, folding_factor))
            .zip(powers_of_x.iter())
            .fold(vec![], |acc, (raised_fold, power_of_x)| {
                add_polys(&acc, &mul_polys(&raised_fold, power_of_x))
            });

        let powers_of_r = iter::successors(Some(BB::ONE), |&x| Some(x * folding_randomness))
            .take(folding_factor)
            .collect_vec();

        // Constructing the expected folded polynomial as the linear combination
        // of the limbs with coefficients equal to the list of powers of r
        let expected_folded_polynomial = folds
            .iter()
            .zip(powers_of_r.iter())
            .map(|(p, r)| scale_poly(p, *r))
            .fold(vec![], |acc, p| add_polys(&acc, &p));

        assert_eq!(
            fold_polynomial(&polynomial, folding_randomness, log_folding_factor),
            expected_folded_polynomial
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
                .map(|x| eval_poly(&$polynomial, x))
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
            let expected_folded_evaluation = eval_poly(
                &folded_polynomial,
                domain.shift().exp_power_of_2($log_arity),
            );

            assert_eq!(
                folded_evaluation, expected_folded_evaluation,
                "log_arity = {}",
                $log_arity
            );
        }};
    }

    // NP TODO remove duplication with test_fold_evaluations
    // NP TODO pass rng here and in the above macro

    macro_rules! test_fold_evals_at_domain_with_log_arity {
        ($log_arity:expr, $polynomial:expr, $folding_randomness:expr, $dft:expr) => {{
            let mut rng = SmallRng::seed_from_u64(87);
            let domain = TwoAdicMultiplicativeCoset::new(rng.random(), 10).unwrap();

            // Evaluating the polynomial over the domain
            let evaluations = domain_dft(domain, $polynomial.clone(), $dft);

            // We first compute the folded evaluations using the method
            // fold_evaluations_at_domain()
            let folded_evaluation =
                fold_evaluations_at_domain(evaluations, domain, $log_arity, $folding_randomness);

            // The above needs to coincide with the result of evaluating the
            // folded polynomial at the the k^th power of the domain
            let folded_polynomial = fold_polynomial(&$polynomial, $folding_randomness, $log_arity);

            let expected_folded_evaluations = domain_dft(
                domain.exp_power_of_2($log_arity).unwrap(),
                folded_polynomial.to_vec(),
                $dft,
            );

            assert_eq!(folded_evaluation, expected_folded_evaluations);
        }};
    }

    #[test]
    // Checks that fold_evaluations() returns the expected results for arities
    // k = 2^1, ..., 2^9
    fn test_fold_evaluations() {
        let mut rng = SmallRng::seed_from_u64(43);
        let polynomial = rand_poly_coeffs((1 << 10) - 1, &mut rng);
        let folding_randomness: BB = rng.random();

        for log_arity in 1..10 {
            test_fold_evals_with_log_arity!(log_arity, &polynomial, folding_randomness)
        }
    }

    #[test]
    // Checks that fold_evaluations_at_domain() returns the expected results for
    // arities k = 2^1, ..., 2^9
    fn test_fold_evaluations_at_domain() {
        let dft = Radix2Dit::default();

        let mut rng = SmallRng::seed_from_u64(999);
        let polynomial = rand_poly_coeffs((1 << 10) - 1, &mut rng);
        let folding_randomness: BB = rng.random();

        for log_arity in 1..10 {
            test_fold_evals_at_domain_with_log_arity!(
                log_arity,
                &polynomial,
                folding_randomness,
                &dft
            )
        }
    }

    #[test]
    // Checks that fold_evaluations_binary() returns the expected results
    fn test_fold_evaluations_binary() {
        let log_domain_size = 4;
        let poly_deg = 7;

        let mut rng = SmallRng::seed_from_u64(93);

        let polynomial = rand_poly_coeffs(poly_deg, &mut rng);

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
            .map(|&root| eval_poly(&polynomial, root))
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
            .map(|root| eval_poly(&folded_poly, *root))
            .collect_vec();

        assert_eq!(folded_evals, expected_folded_evals);
    }
}
