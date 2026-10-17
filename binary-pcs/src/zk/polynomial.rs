//! Small monomial polynomial routines over binary tower fields.

use alloc::vec;
use alloc::vec::Vec;

use p3_binary_field::TowerLevel;

fn trim<F: TowerLevel>(p: &mut Vec<F>) {
    while p.last() == Some(&F::ZERO) {
        p.pop();
    }
}

#[cfg(test)]
pub(super) fn evaluate<F: TowerLevel>(p: &[F], x: F) -> F {
    p.iter().rev().fold(F::ZERO, |acc, &c| acc * x + c)
}

/// Change a power-of-two row-major monomial matrix to the Cantor novel basis.
pub(super) fn monomial_to_novel<F: TowerLevel>(values: &mut [F], width: usize) {
    let rows = values.len() / width;
    debug_assert!(rows.is_power_of_two() && values.len().is_multiple_of(width));
    for log_block in (1..=rows.ilog2() as usize).rev() {
        let half = 1 << (log_block - 1);
        let j = log_block - 1;
        // W_j = sum_{r subset-of j} X^(2^r). Its leading term is X^half;
        // all remaining coefficients are one, so monic division uses additions.
        let support: Vec<_> = (0..j).filter(|&r| r & j == r).map(|r| 1 << r).collect();
        for block in values.chunks_exact_mut(2 * half * width) {
            for row in (half..2 * half).rev() {
                for column in 0..width {
                    let coefficient = block[row * width + column];
                    for &degree in &support {
                        block[(row - half + degree) * width + column] += coefficient;
                    }
                }
            }
        }
        // Each block now stores remainder followed by quotient. Descending
        // division corrected the quotient in place; retain it for the next level.
    }
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
    fn monomial_conversion_preserves_the_product_defined_novel_polynomial() {
        use crate::zk::Val;

        // Catch reversed recursion, an uncorrected quotient, or basis scaling.
        for log_n in 0..=6 {
            let n = 1 << log_n;
            for degree in 0..n {
                let mut converted = vec![Val::ZERO; n];
                converted[degree] = Val::ONE;
                monomial_to_novel(&mut converted, 1);
                for x in [
                    Val::ZERO,
                    Val::ONE,
                    Val::from_repr(0x9876_5432_10fe_dcba_1234_5678_9abc_def0),
                ] {
                    let mut w = x;
                    let mut basis = vec![Val::ONE; n];
                    for bit in 0..log_n {
                        for (index, value) in basis.iter_mut().enumerate() {
                            if index >> bit & 1 != 0 {
                                *value *= w;
                            }
                        }
                        w = w * w + w;
                    }
                    let actual: Val = converted.iter().zip(basis).map(|(&c, b)| c * b).sum();
                    assert_eq!(
                        actual,
                        x.exp_u64(degree as u64),
                        "log_n={log_n} degree={degree}"
                    );
                }
            }
        }
    }

    #[test]
    fn monomial_conversion_reconstructs_dense_and_padded_columns() {
        use p3_binary_dft::{AdditiveNtt, LchNtt};
        use p3_matrix::dense::RowMajorMatrix;
        use rand::rngs::StdRng;
        use rand::{RngExt, SeedableRng};

        use crate::zk::Val;

        let mut rng = StdRng::seed_from_u64(0x4d4f_4e4f);
        for length in [1usize, 3, 4, 5, 7, 8, 9, 31, 32, 33, 127, 128, 129] {
            let n = length.next_power_of_two();
            let width = 3;
            let mut coefficients = vec![Val::ZERO; n * width];
            for coefficient in &mut coefficients[..length * width] {
                *coefficient = rng.random();
            }
            let mut converted = coefficients.clone();
            monomial_to_novel(&mut converted, width);
            for shift in [
                Val::ZERO,
                Val::ONE,
                Val::from_repr(0xfedc_ba98_7654_3210_0123_4567_89ab_cdef),
            ] {
                let encoded = LchNtt::default()
                    .shifted_ntt_batch(RowMajorMatrix::new(converted.clone(), width), shift);
                for (index, row) in encoded.values.chunks_exact(width).enumerate() {
                    let point = shift + domain_point::<Val>(index, n.ilog2() as usize);
                    for (column, &value) in row.iter().enumerate() {
                        let monomials: Vec<_> = coefficients
                            .chunks_exact(width)
                            .map(|row| row[column])
                            .collect();
                        assert_eq!(value, evaluate(&monomials, point));
                    }
                }
            }
            // Reconstruct independently by multiplying the defining W_j polynomials.
            let mut reconstructed = vec![Val::ZERO; n * width];
            for index in 0..n {
                let mut basis = vec![Val::ONE];
                for bit in 0..n.ilog2() as usize {
                    if index >> bit & 1 != 0 {
                        basis = multiply(&basis, &vanishing::<Val>(bit));
                    }
                }
                for (degree, coefficient) in basis.into_iter().enumerate() {
                    for column in 0..width {
                        reconstructed[degree * width + column] +=
                            coefficient * converted[index * width + column];
                    }
                }
            }
            assert_eq!(reconstructed, coefficients, "length={length}");
        }
    }

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
        use p3_binary_dft::subspace_polynomial;
        use rand::rngs::StdRng;
        use rand::{RngExt, SeedableRng};

        use crate::zk::Val;
        let mut rng = StdRng::seed_from_u64(0x5641_4e49_5348);
        for log_n in 1..=8 {
            let w = vanishing::<Val>(log_n);
            for index in 0..1 << log_n {
                assert_eq!(evaluate(&w, domain_point::<Val>(index, log_n)), Val::ZERO);
                assert_eq!(
                    subspace_polynomial(log_n, domain_point::<Val>(index, log_n)),
                    Val::ZERO
                );
            }
            assert_ne!(evaluate(&w, Val::cantor_basis(log_n)), Val::ZERO);
            for _ in 0..16 {
                let point: Val = rng.random();
                assert_eq!(subspace_polynomial(log_n, point), evaluate(&w, point));
            }
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
