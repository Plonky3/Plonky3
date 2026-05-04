//! Unit tests for STIR polynomial arithmetic utilities.

use p3_baby_bear::BabyBear;
use p3_field::extension::BinomialExtensionField;
use p3_field::{Field, PrimeCharacteristicRing};
use p3_stir::utils::{
    add_polys, check_shake_consistency, compute_shake_polynomial, divide_by_linear, eval_poly,
    interpolate_poly, scale_poly,
};

type F = BabyBear;
type EF = BinomialExtensionField<F, 4>;

fn f(n: u64) -> F {
    F::from_u64(n)
}

fn ef(n: u64) -> EF {
    EF::from(f(n))
}

// ---------------------------------------------------------------------------
// eval_poly
// ---------------------------------------------------------------------------

#[test]
fn test_eval_poly_constant() {
    // p(x) = 7
    let poly = vec![f(7)];
    assert_eq!(eval_poly(&poly, f(3)), f(7));
    assert_eq!(eval_poly(&poly, f(0)), f(7));
}

#[test]
fn test_eval_poly_linear() {
    // p(x) = 2 + 3x  →  p(5) = 2 + 15 = 17
    let poly = vec![f(2), f(3)];
    assert_eq!(eval_poly(&poly, f(5)), f(17));
    assert_eq!(eval_poly(&poly, f(0)), f(2));
}

#[test]
fn test_eval_poly_quadratic() {
    // p(x) = 1 + 2x + 3x²  →  p(4) = 1 + 8 + 48 = 57
    let poly = vec![f(1), f(2), f(3)];
    assert_eq!(eval_poly(&poly, f(4)), f(57));
}

#[test]
fn test_eval_poly_empty() {
    let poly: Vec<F> = vec![];
    assert_eq!(eval_poly(&poly, f(5)), F::ZERO);
}

// ---------------------------------------------------------------------------
// divide_by_linear
// ---------------------------------------------------------------------------

#[test]
fn test_divide_by_linear_remainder_equals_eval() {
    // p(x) = 1 + 2x + 3x²
    let poly = vec![f(1), f(2), f(3)];
    let point = f(4);
    let (q, r) = divide_by_linear(&poly, point);
    // Remainder must equal p(point)
    assert_eq!(r, eval_poly(&poly, point));
    // Verify p(x) = (x - point) * q(x) + r at x = 10
    let x = f(10);
    let qx = eval_poly(&q, x);
    let px = eval_poly(&poly, x);
    let reconstructed = (x - point) * qx + r;
    assert_eq!(px, reconstructed);
}

#[test]
fn test_divide_by_linear_exact_divisor() {
    // p(x) = x² - 1 = (x-1)(x+1); dividing by (x - 1) should give remainder 0.
    // p(x) = -1 + 0*x + 1*x²  → root at x=1: p(1) = -1 + 0 + 1 = 0
    let neg_one = F::ZERO - F::ONE;
    let poly = vec![neg_one, F::ZERO, F::ONE];
    let (_q, r) = divide_by_linear(&poly, F::ONE);
    assert_eq!(r, F::ZERO);
}

#[test]
fn test_divide_by_linear_empty() {
    let poly: Vec<F> = vec![];
    let (q, r) = divide_by_linear(&poly, f(5));
    assert!(q.is_empty());
    assert_eq!(r, F::ZERO);
}

// ---------------------------------------------------------------------------
// add_polys / scale_poly
// ---------------------------------------------------------------------------

#[test]
fn test_add_polys_same_degree() {
    // (1 + 2x) + (3 + 4x) = 4 + 6x
    let a = vec![f(1), f(2)];
    let b = vec![f(3), f(4)];
    let c = add_polys(&a, &b);
    assert_eq!(c, vec![f(4), f(6)]);
}

#[test]
fn test_add_polys_different_degree() {
    // (1 + 2x + 3x²) + (5 + 6x) = 6 + 8x + 3x²
    let a = vec![f(1), f(2), f(3)];
    let b = vec![f(5), f(6)];
    let c = add_polys(&a, &b);
    assert_eq!(c, vec![f(6), f(8), f(3)]);
}

#[test]
fn test_scale_poly() {
    // 2 * (1 + 3x + 5x²) = 2 + 6x + 10x²
    let poly = vec![f(1), f(3), f(5)];
    let scaled = scale_poly(&poly, f(2));
    assert_eq!(scaled, vec![f(2), f(6), f(10)]);
}

// ---------------------------------------------------------------------------
// interpolate_poly
// ---------------------------------------------------------------------------

#[test]
fn test_interpolate_poly_degree_0() {
    // Single point: p(7) = 42  →  p(x) = 42
    let poly = interpolate_poly(&[f(7)], &[f(42)]);
    assert_eq!(eval_poly(&poly, f(7)), f(42));
    assert_eq!(eval_poly(&poly, f(0)), f(42));
}

#[test]
fn test_interpolate_poly_linear() {
    // Two points: p(0)=2, p(1)=5  →  p(x) = 2 + 3x
    let poly = interpolate_poly(&[f(0), f(1)], &[f(2), f(5)]);
    assert_eq!(eval_poly(&poly, f(0)), f(2));
    assert_eq!(eval_poly(&poly, f(1)), f(5));
    assert_eq!(eval_poly(&poly, f(2)), f(8));
}

#[test]
fn test_interpolate_poly_quadratic() {
    // Three points: p(0)=1, p(1)=3, p(2)=9  →  p(x) = 1 + x + x²
    let poly = interpolate_poly(&[f(0), f(1), f(2)], &[f(1), f(3), f(7)]);
    // p(x) = 1 + x + x²: p(0)=1, p(1)=3, p(2)=7
    assert_eq!(eval_poly(&poly, f(0)), f(1));
    assert_eq!(eval_poly(&poly, f(1)), f(3));
    assert_eq!(eval_poly(&poly, f(2)), f(7));
    // Degree-2 polynomial, check at a new point x=3: p(3)=1+3+9=13
    assert_eq!(eval_poly(&poly, f(3)), f(13));
}

#[test]
fn test_interpolate_poly_roundtrip() {
    // Choose a known polynomial p(x) = x³ + 2x + 1, sample 4 points, interpolate, verify.
    // p(x) = 1 + 2x + 0x² + x³
    let known_poly = vec![f(1), f(2), F::ZERO, F::ONE];
    let xs: Vec<F> = (0u64..4).map(f).collect();
    let ys: Vec<F> = xs.iter().map(|&x| eval_poly(&known_poly, x)).collect();
    let recovered = interpolate_poly(&xs, &ys);
    // Must agree at the interpolation points and a new one
    for (&x, &y) in xs.iter().zip(ys.iter()) {
        assert_eq!(eval_poly(&recovered, x), y);
    }
    assert_eq!(eval_poly(&recovered, f(5)), eval_poly(&known_poly, f(5)));
}

// ---------------------------------------------------------------------------
// compute_shake_polynomial + check_shake_consistency
// ---------------------------------------------------------------------------

#[test]
fn test_shake_polynomial_degree_bound() {
    // shake = sum_y (ans(X) - ans(y)) / (X - y); ans has degree d,
    // each term has degree d-1, sum has degree d-1.
    // Use extension field points for more variety.
    let ans = vec![ef(1), ef(2), ef(3), ef(4)]; // degree-3 polynomial
    let points = vec![ef(10), ef(20), ef(30)];
    let shake = compute_shake_polynomial(&ans, &points);
    // shake should be of degree <= 2 (ans degree - 1)
    assert!(shake.len() <= 3);
}

#[test]
fn test_shake_consistency_check() {
    // Build ans by interpolation, compute shake, and check at a random rho.
    let points = vec![ef(1), ef(2), ef(3)];
    let values = vec![ef(5), ef(11), ef(23)];
    let ans = interpolate_poly(&points, &values);
    let shake = compute_shake_polynomial(&ans, &points);

    // Any rho outside the points should pass.
    let rho = ef(100);
    assert!(check_shake_consistency(&ans, &shake, &points, &values, rho));
}

#[test]
fn test_shake_consistency_tampered_answer_fails() {
    let points = vec![ef(1), ef(2), ef(3)];
    let values = vec![ef(5), ef(11), ef(23)];
    let ans = interpolate_poly(&points, &values);
    let shake = compute_shake_polynomial(&ans, &points);

    // Tamper with the shake polynomial.
    let mut bad_shake = shake;
    bad_shake[0] += ef(1);
    let rho = ef(42);
    assert!(!check_shake_consistency(
        &ans, &bad_shake, &points, &values, rho
    ));
}

#[test]
fn test_shake_consistency_tampered_value_fails() {
    let points = vec![ef(1), ef(2), ef(3)];
    let values = vec![ef(5), ef(11), ef(23)];
    let ans = interpolate_poly(&points, &values);
    let shake = compute_shake_polynomial(&ans, &points);

    // Tamper with a claimed value (so ans no longer matches).
    let mut bad_values = values;
    bad_values[0] = ef(999);
    let rho = ef(42);
    // The ans was built from the original values, so the reconstructed ans from
    // bad_values will differ → consistency check must fail.
    let bad_ans = interpolate_poly(&points, &bad_values);
    assert!(!check_shake_consistency(
        &bad_ans,
        &shake,
        &points,
        &bad_values,
        rho
    ));
}

// ---------------------------------------------------------------------------
// fold_codeword correctness
// ---------------------------------------------------------------------------

#[test]
fn test_fold_codeword_arity2_constant_polynomial() {
    use p3_dft::Radix2DitParallel;
    use p3_stir::prover::codeword_from_coeffs;
    use p3_stir::utils::fold_codeword;

    // A constant polynomial p(x) = 5 should fold to p(x) = 5 regardless of gamma.
    let log_domain = 4;
    let domain_size = 1 << log_domain;
    let dft = Radix2DitParallel::<F>::default();
    let shift = F::GENERATOR;

    let coeffs: Vec<EF> = {
        let mut c = vec![EF::ZERO; domain_size];
        c[0] = ef(5);
        c
    };

    // DFT to get codeword.
    let codeword = codeword_from_coeffs(&dft, coeffs, shift, log_domain);

    let gamma = ef(7);
    let folded = fold_codeword::<F, EF>(&codeword, gamma, 1, log_domain);

    // Folded codeword should have all entries equal to 5 (constant polynomial).
    for &v in &folded {
        assert_eq!(v, ef(5), "folded constant polynomial must stay constant");
    }
}

#[test]
fn test_fold_codeword_agrees_with_fold_fiber() {
    use p3_dft::Radix2DitParallel;
    use p3_stir::utils::{fold_codeword, fold_fiber};

    let log_domain = 4;
    let log_arity = 1;
    let domain_size = 1 << log_domain;
    let dft = Radix2DitParallel::<F>::default();
    let shift = F::GENERATOR;

    // Random-ish coefficients using field arithmetic.
    let coeffs: Vec<EF> = (1..=domain_size).map(|i| ef(i as u64)).collect();

    use p3_stir::prover::codeword_from_coeffs;
    let codeword = codeword_from_coeffs(&dft, coeffs, shift, log_domain);

    let gamma = ef(42);
    let folded = fold_codeword::<F, EF>(&codeword, gamma, log_arity, log_domain);

    let new_height = codeword.len() >> log_arity;
    let log_new_height = log_domain - log_arity;

    // For each new-domain index j, fold_fiber should give the same result as folded[j].
    for j in 0..new_height {
        let fiber: Vec<EF> = (0..(1 << log_arity))
            .map(|k| codeword[j + k * new_height])
            .collect();
        let expected = fold_fiber::<F, EF>(&fiber, j, log_new_height, log_arity, gamma);
        assert_eq!(
            folded[j], expected,
            "fold_codeword and fold_fiber disagree at j={j}"
        );
    }
}
