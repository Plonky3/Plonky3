//! Small monomial polynomial routines over binary tower fields.

use alloc::vec;
use alloc::vec::Vec;

use p3_binary_field::TowerLevel;

fn trim<F: TowerLevel>(p: &mut Vec<F>) {
    while p.last() == Some(&F::ZERO) {
        p.pop();
    }
}

pub(super) fn evaluate<F: TowerLevel>(p: &[F], x: F) -> F {
    p.iter().rev().fold(F::ZERO, |acc, &c| acc * x + c)
}

pub(super) fn add<F: TowerLevel>(a: &[F], b: &[F]) -> Vec<F> {
    let mut result = vec![F::ZERO; a.len().max(b.len())];
    for (i, &c) in a.iter().enumerate() {
        result[i] += c;
    }
    for (i, &c) in b.iter().enumerate() {
        result[i] += c;
    }
    trim(&mut result);
    result
}

pub(super) fn scale<F: TowerLevel>(p: &[F], scalar: F) -> Vec<F> {
    let mut result = p.iter().map(|&c| c * scalar).collect();
    trim(&mut result);
    result
}

pub(super) fn multiply<F: TowerLevel>(a: &[F], b: &[F]) -> Vec<F> {
    if a.is_empty() || b.is_empty() {
        return vec![];
    }
    let mut result = vec![F::ZERO; a.len() + b.len() - 1];
    for (i, &x) in a.iter().enumerate() {
        for (j, &y) in b.iter().enumerate() {
            result[i + j] += x * y;
        }
    }
    trim(&mut result);
    result
}

pub(super) fn exact_divide<F: TowerLevel>(
    mut numerator: Vec<F>,
    denominator: &[F],
) -> Option<Vec<F>> {
    let mut denominator = denominator.to_vec();
    trim(&mut denominator);
    let inverse = denominator.last()?.inverse();
    trim(&mut numerator);
    let mut quotient = vec![F::ZERO; numerator.len().saturating_sub(denominator.len()) + 1];
    while numerator.len() >= denominator.len() {
        let offset = numerator.len() - denominator.len();
        let leading = *numerator.last()? * inverse;
        quotient[offset] = leading;
        for (i, &c) in denominator.iter().enumerate() {
            numerator[offset + i] += leading * c;
        }
        trim(&mut numerator);
    }
    if !numerator.is_empty() {
        return None;
    }
    trim(&mut quotient);
    Some(quotient)
}

pub(super) fn domain_point<F: TowerLevel>(index: usize, log_n: usize) -> F {
    (0..log_n)
        .filter(|&i| index >> i & 1 == 1)
        .map(F::cantor_basis)
        .sum()
}

pub(super) fn vanishing<F: TowerLevel>(log_n: usize) -> Vec<F> {
    let mut w = vec![F::ZERO, F::ONE];
    for _ in 0..log_n {
        let mut square = vec![F::ZERO; 2 * w.len() - 1];
        for (i, &c) in w.iter().enumerate() {
            square[2 * i] = c * c;
        }
        w = add(&square, &w);
    }
    w
}

pub(super) fn interpolate<F: TowerLevel>(values: &[F], log_n: usize) -> Vec<F> {
    let w = vanishing::<F>(log_n);
    let mut result = vec![];
    // W'=1 for the unnormalised Cantor subspace polynomial. Therefore the
    // Lagrange basis at x_i is exactly W(X)/(X+x_i).
    for (i, &y) in values.iter().enumerate() {
        let basis = exact_divide(w.clone(), &[domain_point::<F>(i, log_n), F::ONE])
            .expect("Cantor subspace point is a root of its vanishing polynomial");
        result = add(&result, &scale(&basis, y));
    }
    result
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use p3_binary_field::{BinaryField8, BinaryField16, TowerLevel};
    use p3_field::{Field, PrimeCharacteristicRing};

    use super::*;

    #[test]
    fn interpolate_and_divide_cantor_polynomials() {
        type F = BinaryField16;
        let w = vanishing::<F>(2);
        assert_eq!(w, vec![F::ZERO, F::ONE, F::ZERO, F::ZERO, F::ONE]);
        let coefficients = vec![F::from_repr(7), F::from_repr(3), F::from_repr(4)];
        let values: Vec<_> = (0..4)
            .map(|i| evaluate(&coefficients, domain_point::<F>(i, 2)))
            .collect();
        assert_eq!(interpolate(&values, 2), coefficients);
        assert_eq!(
            exact_divide(multiply(&w, &coefficients), &w),
            Some(coefficients)
        );
        assert!(exact_divide(vec![F::ONE], &w).is_none());
    }

    #[test]
    fn cantor_vanishing_polynomials_cover_every_supported_trace_height() {
        use crate::zk::Val;
        for log_n in 1..=8 {
            let w = vanishing::<Val>(log_n);
            for index in 0..1 << log_n {
                assert_eq!(evaluate(&w, domain_point::<Val>(index, log_n)), Val::ZERO);
            }
            assert_ne!(evaluate(&w, Val::cantor_basis(log_n)), Val::ZERO);
        }
        let w = vanishing::<BinaryField16>(3);
        let values: Vec<_> = (0..8).map(|i| BinaryField16::from_repr(i * 3)).collect();
        let polynomial = interpolate(&values, 3);
        for (index, &value) in values.iter().enumerate() {
            assert_eq!(
                evaluate(&polynomial, domain_point::<BinaryField16>(index, 3)),
                value
            );
        }
        assert!(exact_divide(w, &[]).is_none());
    }

    #[test]
    fn exhaustive_same_statement_terminal_distributions() {
        // Both witnesses satisfy x*y=1 on H={0,1}. Enumerate every independent
        // semantic mask in GF(256): the terminal (t_x,t_y,q) distribution must
        // coincide, including the quotient determined by t and the statement.
        type F = BinaryField8;
        let w = vanishing::<F>(1);
        let z = F::cantor_basis(1);
        let wz = evaluate(&w, z);
        let mut distributions = vec![];
        for a in [F::ONE, F::from_repr(7)] {
            let b = a.inverse();
            let mut counts = vec![0u8; 256 * 256];
            for r in 0..256u16 {
                for s in 0..256u16 {
                    let x = add(&[a], &scale(&w, F::from_repr(r as u8)));
                    let y = add(&[b], &scale(&w, F::from_repr(s as u8)));
                    let q = exact_divide(add(&multiply(&x, &y), &[F::ONE]), &w).unwrap();
                    let tx = evaluate(&x, z);
                    let ty = evaluate(&y, z);
                    assert_eq!(evaluate(&q, z), (tx * ty + F::ONE) / wz);
                    counts[tx.to_repr() as usize * 256 + ty.to_repr() as usize] += 1;
                }
            }
            assert!(counts.iter().all(|&c| c == 1));
            distributions.push(counts);
        }
        assert_eq!(distributions[0], distributions[1]);
    }
}
