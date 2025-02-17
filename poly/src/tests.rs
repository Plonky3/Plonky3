use alloc::vec;
use alloc::vec::Vec;

use itertools::Itertools;
use p3_baby_bear::BabyBear;
use p3_dft::{Radix2Dit, TwoAdicSubgroupDft};
use p3_field::{Field, PrimeCharacteristicRing, TwoAdicField};
use p3_goldilocks::Goldilocks;
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;
use rand::Rng;

use crate::test_utils::rand_poly;
use crate::Polynomial;

type BB = BabyBear;
type GL = Goldilocks;

const TEST_ITERATIONS: usize = 30;

fn field_element_from_i64<F: Field>(x: i64) -> F {
    let sign = if x >= 0 { F::ONE } else { -F::ONE };
    let value = F::from_u64(x.abs() as u64);
    sign * value
}

fn field_elements_from_i64<F: Field>(xs: Vec<i64>) -> Vec<F> {
    xs.into_iter().map(|x| field_element_from_i64(x)).collect()
}

#[test]
// Checks that leading zeros are trimmed appropriately
fn test_trimmed_coeffs() {
    let coeffs: Vec<GL> = field_elements_from_i64(vec![1, 9, -9, 2, 0, -3, 0, 0]);
    let poly = Polynomial::from_coeffs(coeffs);

    assert_eq!(
        poly.coeffs,
        field_elements_from_i64(vec![1, 9, -9, 2, 0, -3])
    );

    let coeffs = vec![GL::ZERO; 22];
    let poly = Polynomial::from_coeffs(coeffs);

    assert_eq!(poly.coeffs, vec![]);
}

#[test]
// Checks the evaluate method manually and using an FFT
fn test_evaluate() {
    // a(x) = 3 + 2x + 5x^2 + 6x^3 + 8x^4
    let mut a_coeffs = field_elements_from_i64(vec![3, 2, 5, 6, 8]);

    let a = Polynomial::from_coeffs(a_coeffs.clone());
    let x = GL::ONE;
    let y = a.evaluate(&x);
    assert_eq!(y, GL::from_u32(24));

    a_coeffs.resize(8, GL::ZERO);
    let dft = Radix2Dit::default();
    let a_evals = dft
        .dft_batch(RowMajorMatrix::new(a_coeffs, 1))
        .transpose()
        .first_row()
        .collect_vec();

    let gen = GL::two_adic_generator(3);

    for (index, expected) in a_evals.into_iter().enumerate() {
        let point = gen.exp_u64(index as u64);
        assert_eq!(a.evaluate(&point), expected);
    }
}

#[test]
// Checks addition, subtraction, multiplication, and division where both operands
// are polynomials. The expected result is computed manually.
fn test_ops_manual() {
    // a(x) = 3 + 2x + 5x^2 + 6x^3
    let a_coeffs: Vec<BB> = field_elements_from_i64(vec![3, 2, 5, 6]);

    // b(x) = 1 + 2x + 3x^2 + 4x^3 + 5x^4 + 7x^5 + 8x^6
    let b_coeffs: Vec<BB> = field_elements_from_i64(vec![1, 2, 3, 4, 5, 7, 8]);

    // a(x) + b(x) = 4 + 4x + 8x^2 + 10x^3 + 5x^4 + 7x^5 + 8x^6
    let expected_coeffs_add = field_elements_from_i64(vec![4, 4, 8, 10, 5, 7, 8]);

    // a(x) - b(x) = 2 + 0x + 2x^2 + 2x^3 - 5x^4 - 7x^5 - 8x^6
    let expected_coeffs_sub = field_elements_from_i64(vec![2, 0, 2, 2, -5, -7, -8]);

    // a(x) * b(x) = 3 + 8x + 18x^2 + 34x^3 + 50x^4 + 69x^5 + 87x^6 + 81x^7 + 82x^8 + 48x^9
    let expected_coeffs_mul = field_elements_from_i64(vec![3, 8, 18, 34, 50, 69, 87, 81, 82, 48]);

    // b(x) / a(x) =  (q(x), r(x))

    // q(x) = -197/648 + (37 x)/108 + x^2/18 + (4 x^3)/3
    let q_coeffs_noms: Vec<BB> = field_elements_from_i64(vec![-197, 37, 1, 4]);
    let q_coeffs_denoms: Vec<BB> = field_elements_from_i64(vec![648, 108, 18, 3]);
    let expected_coeffs_q = q_coeffs_noms
        .into_iter()
        .zip(q_coeffs_denoms)
        .map(|(n, d)| n * d.inverse())
        .collect_vec();

    // r(x) = 413/216 + (128 x)/81 + (2377 x^2)/648
    let r_coeffs_noms: Vec<BB> = field_elements_from_i64(vec![413, 128, 2377]);
    let r_coeffs_denoms: Vec<BB> = field_elements_from_i64(vec![216, 81, 648]);
    let expected_coeffs_r = r_coeffs_noms
        .into_iter()
        .zip(r_coeffs_denoms)
        .map(|(n, d)| n * d.inverse())
        .collect_vec();

    let a = Polynomial::from_coeffs(a_coeffs);
    let b = Polynomial::from_coeffs(b_coeffs);

    let add = &a + &b;
    assert_eq!(add.coeffs, expected_coeffs_add);

    let sub = &a - &b;
    assert_eq!(sub.coeffs, expected_coeffs_sub);

    let mul = &a.mul_naive(&b);
    assert_eq!(mul.coeffs, expected_coeffs_mul);

    let (q, r) = &b.divide_with_remainder(&a);
    assert_eq!(q.coeffs, expected_coeffs_q);
    assert_eq!(r.coeffs, expected_coeffs_r);
}

#[test]
// Checks addition, subtraction, multiplication, and division where both operands
// are polynomials. Correctness is checked by evaluating at ()(degree + 1) points.
fn test_ops_random() {
    let mut rng = rand::rng();

    for _ in 0..TEST_ITERATIONS {
        let deg_a: usize = rng.random_range(0..100);
        let a = rand_poly::<GL>(100);

        let deg_b: usize = rng.random_range(0..100);
        let b = rand_poly::<GL>(100);

        let add = &a + &b;
        let sub = &a - &b;
        let mul = &a * &b;
        let (q, r) = &a.divide_with_remainder(&b);

        // In the case of addition, subtraction and multiplication, this tests
        // at more points than necessary (which would be degree + 1), but these
        // checks are cheap
        for i in 0..(deg_a + deg_b + 1) {
            let i_f = GL::from_u32(i as u32);
            assert_eq!(add.evaluate(&i_f), a.evaluate(&i_f) + b.evaluate(&i_f));
            assert_eq!(sub.evaluate(&i_f), a.evaluate(&i_f) - b.evaluate(&i_f));
            assert_eq!(mul.evaluate(&i_f), a.evaluate(&i_f) * b.evaluate(&i_f));
            assert_eq!(
                a.evaluate(&i_f),
                q.evaluate(&i_f) * b.evaluate(&i_f) + r.evaluate(&i_f)
            );
        }
    }
}

#[test]
// Checks that addition and subtraction operations that produce polynomials with leading
// zeros are correctly trimmed.
fn test_ops_leading_zeros() {
    // a(x) = 3 + 2x + 5x^2 + 6x^3
    let a_coeffs: Vec<BB> = field_elements_from_i64(vec![3, 2, 5, 6, 5, 7, 8]);

    // b_add(x) = 1 + 2x + 3x^2 + 4x^3 - 5x^4 - 7x^5 - 8x^6
    let b_add_coeffs: Vec<BB> = field_elements_from_i64(vec![1, 2, 3, 4, -5, -7, -8]);

    // a(x) + b_add(x) = 4 + 4x + 8x^2 + 10x^3
    let expected_coeffs_add = field_elements_from_i64(vec![4, 4, 8, 10]);

    // b_sub(x) = 2 + 0x + 2x^2 + 2x^3 + 5x^4 + 7x^5 + 8x^6
    let b_sub_coeffs: Vec<BB> = field_elements_from_i64(vec![1, 2, 3, 4, 5, 7, 8]);

    // a(x) - b_sub(x) = 4 + 4x + 8x^2 + 10x^3
    let expected_coeffs_sub = field_elements_from_i64(vec![2, 0, 2, 2]);

    let a = Polynomial::from_coeffs(a_coeffs);
    let b_add = Polynomial::from_coeffs(b_add_coeffs);
    let b_sub = Polynomial::from_coeffs(b_sub_coeffs);

    let add = &a + &b_add;
    assert_eq!(add.coeffs, expected_coeffs_add);

    let sub = &a - &b_sub;
    assert_eq!(sub.coeffs, expected_coeffs_sub);
}

#[test]
// Checks addition, subtraction, multiplication, and division where the right operand is
// a random constant. Correctness is checked by manually computing the expected result.
fn test_ops_constants() {
    let mut rng = rand::rng();

    for _ in 0..TEST_ITERATIONS {
        let deg = rng.random_range(1..20);

        let polynomial: Polynomial<GL> = rand_poly(deg);
        let constant = GL::from_u32(rng.random_range(1..100));

        let expected_add = {
            let mut coeffs = polynomial.coeffs.clone();
            coeffs[0] += constant;
            Polynomial::from_coeffs(coeffs)
        };

        let expected_sub = {
            let mut coeffs = polynomial.coeffs.clone();
            coeffs[0] -= constant;
            Polynomial::from_coeffs(coeffs)
        };

        let expected_mul = {
            let mut coeffs = polynomial.coeffs.clone();
            for c in coeffs.iter_mut() {
                *c *= constant;
            }
            Polynomial::from_coeffs(coeffs)
        };

        let expected_div = {
            let mut coeffs = polynomial.coeffs.clone();
            for c in coeffs.iter_mut() {
                *c *= constant.inverse();
            }
            Polynomial::from_coeffs(coeffs)
        };

        assert_eq!(&polynomial + &constant, expected_add);
        assert_eq!(&polynomial - &constant, expected_sub);
        assert_eq!(&polynomial * &constant, expected_mul);
        assert_eq!(&polynomial / &constant, expected_div);
    }
}

#[test]
// Checks that operations where one of the polynomials is zero (in the case of
// division: always the dividend) yields the expected results
fn test_ops_zero() {
    let mut rng = rand::rng();

    let zero_poly: Polynomial<GL> = Polynomial::zero();

    for _ in 0..TEST_ITERATIONS {
        let deg = rng.random_range(0..20);

        // op can never be the zero polynomial (degree 0 corresponds to non-zero constants)
        let op = rand_poly(deg);

        assert_eq!(&op + &zero_poly, op);
        assert_eq!(&zero_poly + &op, op);

        assert_eq!(&op - &zero_poly, op);
        assert_eq!(&zero_poly - &op, -&op);

        assert_eq!(&op * &zero_poly, zero_poly);
        assert_eq!(&zero_poly * &op, zero_poly);

        let (q, r) = zero_poly.divide_with_remainder(&op);
        assert_eq!(q, zero_poly);
        assert_eq!(r, zero_poly);
    }
}

#[test]
#[should_panic(expected = "Cannot divide by the zero polynomial")]
// Checks that division by zero panics
fn test_div_by_zero() {
    let a = rand_poly::<BB>(100);
    a.divide_with_remainder(&Polynomial::zero());
}

#[test]
#[should_panic(expected = "Cannot divide by the zero polynomial")]
// Checks that dividing zero by zero also panics
fn test_div_zero_by_zero() {
    Polynomial::<GL>::zero().divide_with_remainder(&Polynomial::zero());
}

#[test]
// Checks that multiplication using the FFT and the naive algorithm return the
// same result
fn test_mul_fft_naive_consistency() {
    for _ in 0..TEST_ITERATIONS {
        let a = rand_poly::<GL>(63);
        let b = rand_poly::<GL>(64);

        // This uses the FFT algorithm due to the chosen degrees
        let prod = &a * &b;

        assert_eq!(prod, a.mul_naive(&b));
    }
}

#[test]
// Checks that Lagrange interpolation yields the expected polynomial
fn test_lagrange_interpolation() {
    let mut rng = rand::rng();

    for _ in 0..TEST_ITERATIONS {
        let deg = rng.random_range(15..20);

        let polynomial = rand_poly::<GL>(deg);

        let mut points = Vec::new();

        while points.len() < deg + 1 {
            let point = field_element_from_i64(rng.random::<i64>());
            if !points.contains(&point) {
                points.push(point);
            }
        }

        let evals = points.iter().map(|p| polynomial.evaluate(p)).collect_vec();

        let interpolator =
            Polynomial::lagrange_interpolation(points.into_iter().zip(evals).collect_vec());
        assert_eq!(interpolator, polynomial);
    }
}

#[test]
#[should_panic(expected = "One point has two different requested evaluations")]
// Checks that Lagrange interpolation panics when two different evaluations are requested of the
fn test_lagrange_interpolation_with_inconsistent_evals() {
    let point_to_evals: Vec<(GL, GL)> = vec![(3, 4), (2, 6), (8, 7), (9, 15), (3, 5)]
        .into_iter()
        .map(|(p, e)| (field_element_from_i64(p), field_element_from_i64(e)))
        .collect();
    Polynomial::<GL>::lagrange_interpolation(point_to_evals);
}

#[test]
#[should_panic(expected = "The Lagrange interpolation of an empty set is undefined")]
// Checks that Lagrange interpolation panics when the set of points is empty
fn test_lagrange_interpolation_empty() {
    Polynomial::<GL>::lagrange_interpolation(Vec::new());
}

#[test]
// Checks that Lagrange interpolation correctly handles duplicate (point,
// evaluation) pairs
fn test_lagrange_interpolation_duplicates() {
    let mut rng = rand::rng();

    for _ in 0..10 * TEST_ITERATIONS {
        let num_points = rng.random_range(15..20);
        let num_duplicates = rng.random_range(1..num_points);

        let deg = num_points - num_duplicates - 1;

        let polynomial = rand_poly::<GL>(deg);

        let mut points = Vec::new();

        // Creating a list of distinct points
        while points.len() < deg + 1 {
            let point = field_element_from_i64(rng.random::<i64>());
            if !points.contains(&point) {
                points.push(point);
            }
        }

        // Adding duplicate points
        for _ in 0..num_duplicates {
            let index = rng.random_range(0..points.len());
            points.push(points[index]);
        }

        let evals = points.iter().map(|p| polynomial.evaluate(p)).collect_vec();

        let interpolator =
            Polynomial::lagrange_interpolation(points.into_iter().zip(evals).collect_vec());
        assert_eq!(interpolator, polynomial);
    }
}

#[test]
// Checks the correctness of compose_with_exponent by manually calculating the
// expected result. Ensures the output matches the input when given a constant
// or zero polynomial as input.
fn test_compose_with_exponent() {
    let rng = &mut rand::rng();

    // p(x) = 3 + 2x + 6x^2 + 7x^3 + 9x^4
    let polynomial = Polynomial::<BB>::from_coeffs(field_elements_from_i64(vec![3, 2, 6, 7, 9]));

    let zero_poly = Polynomial::<BB>::zero();

    let constant_poly = Polynomial::<BB>::constant(BB::from_u32(rng.random()));

    let exponent = 3;

    // p(x^3) = 3 + 2x^3 + 6x^6 + 7x^9 + 9x^12
    assert_eq!(
        polynomial.compose_with_exponent(exponent).coeffs(),
        field_elements_from_i64(vec![3, 0, 0, 2, 0, 0, 6, 0, 0, 7, 0, 0, 9])
    );

    assert_eq!(zero_poly.compose_with_exponent(exponent), zero_poly);

    assert_eq!(constant_poly.compose_with_exponent(exponent), constant_poly);
}

#[test]
// Checks that vanishing_polynomial returns the expected polynomial computed
// manually
fn test_vanishing_manual() {
    let points = field_elements_from_i64(vec![3, -5, 7, 2]);

    // p(x) = (x - 3)(x + 5)(x - 7)(x - 2) = -210 + 163x - 19x^2 - 7x^3 + x^4
    let expected_poly =
        Polynomial::<BB>::from_coeffs(field_elements_from_i64(vec![-210, 163, -19, -7, 1]));

    assert_eq!(
        Polynomial::<BB>::vanishing_polynomial(points),
        expected_poly
    );
}

#[test]
// Checks that vanishing_polynomial returns the expected polynomial by asserting
// that the latter is zero at all given points and has the expected degree
fn test_vanishing_random() {
    let max_num_points = 100;

    let mut rng = rand::rng();
    let points = field_elements_from_i64(
        (0..max_num_points)
            .map(|_| rng.random::<i64>())
            .unique()
            .collect_vec(),
    );

    let num_points = points.len();

    let vanishing_poly = Polynomial::<BB>::vanishing_polynomial(points.clone());

    assert_eq!(vanishing_poly.degree().unwrap(), num_points);
    assert!(points
        .iter()
        .all(|p| vanishing_poly.evaluate(p) == BB::ZERO));
}

#[test]
#[should_panic(expected = "The vanishing polynomial of an empty set is undefined")]
// Checks that vanishing_polynomial panics when the set of points is empty
fn test_vanishing_empty() {
    Polynomial::<BB>::vanishing_polynomial(Vec::new());
}

#[test]
// Checks that vanishing_polynomial and lagrange_interpolation recover the same
// polynomial
fn test_vanishing_and_lagrange_interpolation() {
    let max_num_points = 100;

    let mut rng = rand::rng();
    let points = field_elements_from_i64(
        (0..max_num_points)
            .map(|_| rng.random::<i64>())
            .collect_vec(),
    );

    let vanishing_poly = Polynomial::<BB>::vanishing_polynomial(points.clone());

    assert_eq!(vanishing_poly.degree().unwrap(), points.len());

    // In order to recover the same polynomial through interpolation, we need
    // one more value
    let mut point_evals = points.iter().map(|p| (*p, BB::ZERO)).collect_vec();

    let mut new_point = BB::ZERO;
    while new_point.is_zero() {
        new_point = field_element_from_i64(rng.random::<i64>());
    }
    let new_eval = vanishing_poly.evaluate(&new_point);
    point_evals.push((new_point, new_eval));

    let new_vanishing_poly = Polynomial::lagrange_interpolation(point_evals);

    assert_eq!(new_vanishing_poly, vanishing_poly);
}

#[test]
// Checks that division by a linear polynomial yields the expected quotient and
// evaluation
fn test_division_by_linear_polynomial() {
    let mut rng = rand::rng();

    for _ in 0..TEST_ITERATIONS {
        let deg = rng.random_range(1..100);
        let polynomial: Polynomial<GL> = rand_poly(deg);
        let point = rng.random();

        let divisor = Polynomial::vanishing_linear_polynomial(point);
        let (expected_q, _) = polynomial.divide_with_remainder(&divisor);

        let (q, r) = polynomial.divide_by_vanishing_linear_polynomial(point);

        assert_eq!(q, expected_q);
        assert_eq!(r, polynomial.evaluate(&point));
    }
}

#[test]
// Checks that dividing the vanishing polynomial of a set of points by the linar
// vanishing polynomials at each point results in the constant polynomial 1.
fn test_division_by_linear_polynomial_vanishing() {
    let max_num_points = 20;

    let mut rng = rand::rng();

    let poly_one = Polynomial::<GL>::constant(GL::ONE);

    for _ in 0..TEST_ITERATIONS {
        let points = field_elements_from_i64(
            (0..max_num_points)
                .map(|_| rng.random::<i64>())
                .unique()
                .collect_vec(),
        );

        let mut vanishing_poly = Polynomial::<GL>::vanishing_polynomial(points.clone());

        for point in points {
            (vanishing_poly, _) = vanishing_poly.divide_by_vanishing_linear_polynomial(point);
        }

        assert_eq!(vanishing_poly, poly_one);
    }
}

#[test]
fn test_power_polynomial() {
    let mut rng = rand::rng();

    let degree = rng.random_range(50..100);

    let mut r = GL::ZERO;

    while r == GL::ZERO {
        r = rng.random();
    }

    // (1 + rx + ... + (rx)^n) * (rx - 1) = (rx)^(n + 1) - 1
    let power_polynomial = Polynomial::power_polynomial(r, degree);
    let rxn_1 = &(&Polynomial::x().compose_with_exponent(degree + 1)
        * &r.exp_u64((degree + 1) as u64))
        - &GL::ONE;
    let rx_1 = &(&Polynomial::x() * &r) - &GL::ONE;

    assert_eq!(&power_polynomial * &rx_1, rxn_1);
}
