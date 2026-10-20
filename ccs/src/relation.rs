//! Constraint system: matrices and the products of their outputs that must vanish.

use alloc::vec;
use alloc::vec::Vec;

use p3_challenger::fs::TranscriptField;
use p3_field::{Algebra, Field};
use p3_maybe_rayon::prelude::*;

use crate::error::{StructureError, WitnessError};
use crate::operator::{LinearOperator, Operator, axis_len, encode_usize};

/// Coefficient times an entrywise product of matrix outputs; factors may repeat.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Term<F> {
    /// Scalar multiplying the product.
    coefficient: F,
    /// Indices of the multiplied matrices.
    factors: Vec<usize>,
}

impl<F> Term<F> {
    /// Coefficient times the product of the listed matrix outputs.
    #[must_use]
    pub fn new(coefficient: F, factors: impl Into<Vec<usize>>) -> Self {
        Self {
            coefficient,
            factors: factors.into(),
        }
    }

    /// Scalar multiplying the product.
    #[must_use]
    pub const fn coefficient(&self) -> &F {
        &self.coefficient
    }

    /// Indices of the multiplied matrices.
    #[must_use]
    pub fn factors(&self) -> &[usize] {
        &self.factors
    }
}

/// Rows a constraint applies to, evaluated in closed form by the verifier.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RowSelector {
    /// Every row.
    All,
    /// Only row zero.
    First,
    /// Only the last row.
    Last,
    /// Every row except the last.
    Transition,
}

impl RowSelector {
    /// Whether the selector holds at a row.
    pub(crate) const fn holds(self, row: usize, rows: usize) -> bool {
        match self {
            Self::All => true,
            Self::First => row == 0,
            Self::Last => row + 1 == rows,
            Self::Transition => row + 1 != rows,
        }
    }

    /// Multilinear extension of the selector at a row point.
    pub(crate) fn evaluate<EF: Field>(self, row: &[EF]) -> EF {
        // Last row is the all-ones vertex.
        let last = || row.iter().copied().product::<EF>();
        match self {
            Self::All => EF::ONE,
            Self::First => row.iter().map(|&r| EF::ONE - r).product(),
            Self::Last => last(),
            Self::Transition => EF::ONE - last(),
        }
    }

    /// Statement encoding byte.
    const fn tag(self) -> u8 {
        match self {
            Self::All => 0,
            Self::First => 1,
            Self::Last => 2,
            Self::Transition => 3,
        }
    }
}

/// Sum of terms that must vanish on every selected row.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Constraint<F> {
    /// Selected rows.
    selector: RowSelector,
    /// Terms whose sum must vanish.
    terms: Vec<Term<F>>,
}

impl<F> Constraint<F> {
    /// A constraint that must hold on every row.
    #[must_use]
    pub const fn new(terms: Vec<Term<F>>) -> Self {
        Self {
            selector: RowSelector::All,
            terms,
        }
    }

    /// Restrict the constraint to the rows a selector picks.
    #[must_use]
    pub const fn on(mut self, selector: RowSelector) -> Self {
        self.selector = selector;
        self
    }

    /// Selected rows.
    #[must_use]
    pub const fn selector(&self) -> RowSelector {
        self.selector
    }

    /// Terms whose sum must vanish.
    #[must_use]
    pub fn terms(&self) -> &[Term<F>] {
        &self.terms
    }

    /// Per-row degree, counting a restricting selector as one factor.
    pub(crate) fn degree(&self) -> usize {
        let terms = self
            .terms
            .iter()
            .map(|t| t.factors.len())
            .max()
            .unwrap_or(0);
        terms + usize::from(self.selector != RowSelector::All)
    }
}

/// Customizable constraint system over structured matrices.
///
/// All matrices share one shape, and every constraint, term and factor is non-empty and in range.
///
/// Every term has a factor, so the relation is homogeneous and the zero witness satisfies it.
///
/// Constants such as the one of an R1CS are public cells, pinned through the witness commitment.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Ccs<F, O = Operator<F>> {
    /// Number of variables indexing a constraint row.
    log_rows: usize,
    /// Number of variables indexing a witness entry.
    log_cols: usize,
    /// Matrices M_0, ..., M_{t-1}.
    operators: Vec<O>,
    /// Constraints.
    constraints: Vec<Constraint<F>>,
    /// Largest per-row degree of one constraint.
    degree: usize,
}

impl<F: Field, O: LinearOperator<F>> Ccs<F, O> {
    /// Assemble a constraint system, rejecting the first broken invariant.
    pub fn new(operators: Vec<O>, constraints: Vec<Constraint<F>>) -> Result<Self, StructureError> {
        // The first matrix fixes the shape.
        let first = operators.first().ok_or(StructureError::NoOperators)?;
        let (log_rows, log_cols) = (first.log_rows(), first.log_cols());
        for (operator, op) in operators.iter().enumerate() {
            if (op.log_rows(), op.log_cols()) != (log_rows, log_cols) {
                return Err(StructureError::ShapeMismatch {
                    operator,
                    log_rows: op.log_rows(),
                    log_cols: op.log_cols(),
                    expected_log_rows: log_rows,
                    expected_log_cols: log_cols,
                });
            }
        }

        // Every constraint needs a term, every term a factor, every factor a matrix.
        if constraints.is_empty() {
            return Err(StructureError::NoConstraints);
        }
        for (constraint, k) in constraints.iter().enumerate() {
            if k.terms.is_empty() {
                return Err(StructureError::EmptyConstraint { constraint });
            }
            for (term, t) in k.terms.iter().enumerate() {
                if t.factors.is_empty() {
                    return Err(StructureError::EmptyTerm { constraint, term });
                }
                if let Some(&factor) = t.factors.iter().find(|&&j| j >= operators.len()) {
                    return Err(StructureError::FactorOutOfRange {
                        constraint,
                        term,
                        factor,
                        operators: operators.len(),
                    });
                }
            }
        }

        // Bounds every zerocheck round polynomial.
        let degree = constraints
            .iter()
            .map(Constraint::degree)
            .max()
            .unwrap_or(0);

        Ok(Self {
            log_rows,
            log_cols,
            operators,
            constraints,
            degree,
        })
    }

    /// Number of variables indexing a constraint row.
    #[must_use]
    pub const fn log_rows(&self) -> usize {
        self.log_rows
    }

    /// Number of variables indexing a witness entry.
    #[must_use]
    pub const fn log_cols(&self) -> usize {
        self.log_cols
    }

    /// Matrices, in term index order.
    #[must_use]
    pub fn operators(&self) -> &[O] {
        &self.operators
    }

    /// Constraints.
    #[must_use]
    pub fn constraints(&self) -> &[Constraint<F>] {
        &self.constraints
    }

    /// Largest per-row constraint degree, counting selectors.
    #[must_use]
    pub const fn degree(&self) -> usize {
        self.degree
    }

    /// Every matrix output for a witness.
    ///
    /// # Panics
    ///
    /// - The witness has the wrong length.
    #[must_use]
    pub fn operand_tables(&self, z: &[F]) -> Vec<Vec<F>> {
        // Matrices are independent: compute in parallel.
        let rows = 1 << self.log_rows;
        self.operators
            .par_iter()
            .map(|op| {
                let mut out = F::zero_vec(rows);
                op.apply(z, &mut out);
                out
            })
            .collect()
    }

    /// Check a witness directly against every constraint on every row, reporting the first failure.
    pub fn check(&self, z: &[F]) -> Result<(), WitnessError> {
        // One entry per column.
        let expected = axis_len(self.log_cols);
        if z.len() != expected {
            return Err(WitnessError::LengthMismatch {
                expected,
                actual: z.len(),
            });
        }

        // Evaluate every constraint row by row.
        let tables = self.operand_tables(z);
        let rows = axis_len(self.log_rows);
        let mut row_values = vec![F::ZERO; self.operators.len()];
        for row in 0..rows {
            for (value, table) in row_values.iter_mut().zip(&tables) {
                *value = table[row];
            }
            for (constraint, k) in self.constraints.iter().enumerate() {
                if k.selector.holds(row, rows) && !combine(&k.terms, &row_values).is_zero() {
                    return Err(WitnessError::Unsatisfied { row, constraint });
                }
            }
        }
        Ok(())
    }

    /// Append an injective encoding of the system, bound into the transcript.
    pub(crate) fn encode(&self, out: &mut Vec<u8>)
    where
        F: TranscriptField,
    {
        // Counted matrices, each length-prefixed so no encoding can run into the next.
        encode_usize(out, self.operators.len());
        let mut bytes = Vec::new();
        for op in &self.operators {
            bytes.clear();
            op.encode(&mut bytes);
            encode_usize(out, bytes.len());
            out.extend_from_slice(&bytes);
        }

        // Counted constraints: selector, then counted terms.
        encode_usize(out, self.constraints.len());
        for constraint in &self.constraints {
            out.push(constraint.selector.tag());
            encode_usize(out, constraint.terms.len());
            for term in &constraint.terms {
                F::encode(&term.coefficient, out);
                encode_usize(out, term.factors.len());
                for &factor in &term.factors {
                    encode_usize(out, factor);
                }
            }
        }
    }
}

/// Term sum of one constraint from one row of matrix outputs, in any algebra over the field.
pub(crate) fn combine<F: Field, A: Algebra<F> + Copy>(terms: &[Term<F>], values: &[A]) -> A {
    terms
        .iter()
        .map(|term| {
            let product = term.factors.iter().fold(A::ONE, |acc, &j| acc * values[j]);
            product * term.coefficient
        })
        .fold(A::ZERO, |acc, v| acc + v)
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;

    use super::*;
    use crate::operator::{SparseMatrix, TraceColumn};

    /// Base field.
    type F = BabyBear;

    /// R1CS of x * y = 12 and x + y = 7 over the witness (1, x, y, 0).
    fn product_and_sum() -> Ccs<F> {
        let a = SparseMatrix::new(1, 2, vec![(0, 1, F::ONE), (1, 1, F::ONE), (1, 2, F::ONE)]);
        let b = SparseMatrix::new(1, 2, vec![(0, 2, F::ONE), (1, 0, F::ONE)]);
        let c = SparseMatrix::new(1, 2, vec![(0, 0, F::from_u32(12)), (1, 0, F::from_u32(7))]);
        Ccs::new(
            vec![a.into(), b.into(), c.into()],
            vec![Constraint::new(vec![
                Term::new(F::ONE, [0, 1]),
                Term::new(F::NEG_ONE, [2]),
            ])],
        )
        .unwrap()
    }

    #[test]
    fn the_honest_witness_satisfies_and_a_wrong_one_fails_the_sum() {
        // Fixture state: (3, 4) satisfies both rows.
        //
        // Mutation: (2, 6) fails row 1, since 2 + 6 != 7.
        let ccs = product_and_sum();
        assert_eq!(ccs.degree(), 2);
        assert_eq!(ccs.check(&[1, 3, 4, 0].map(F::from_u32)), Ok(()));
        assert_eq!(
            ccs.check(&[1, 2, 6, 0].map(F::from_u32)),
            Err(WitnessError::Unsatisfied {
                row: 1,
                constraint: 0
            })
        );
    }

    #[test]
    fn a_witness_of_the_wrong_length_is_reported() {
        let ccs = product_and_sum();
        assert_eq!(
            ccs.check(&[F::ONE; 3]),
            Err(WitnessError::LengthMismatch {
                expected: 4,
                actual: 3
            })
        );
    }

    #[test]
    fn a_selector_switches_a_constraint_off_on_the_rows_it_skips() {
        // Fixture state: a = (1, 2, 4, 8) with a_next = 2 * a.
        //
        // Row 3 reads a zero next row, so only transitions hold.
        let this = Operator::from(TraceColumn::new(2, 0, 0, 0));
        let next = Operator::from(TraceColumn::new(2, 0, 0, 1));
        let rule = Constraint::new(vec![Term::new(F::ONE, [1]), Term::new(-F::TWO, [0])]);
        let z = [1, 2, 4, 8].map(F::from_u32);

        // On every row, the boundary row fails.
        let everywhere = Ccs::new(vec![this.clone(), next.clone()], vec![rule.clone()]).unwrap();
        assert_eq!(everywhere.degree(), 1);
        assert_eq!(
            everywhere.check(&z),
            Err(WitnessError::Unsatisfied {
                row: 3,
                constraint: 0
            })
        );

        // On transitions it holds, at one more degree.
        let transitions =
            Ccs::new(vec![this, next], vec![rule.on(RowSelector::Transition)]).unwrap();
        assert_eq!(transitions.degree(), 2);
        assert_eq!(transitions.check(&z), Ok(()));
    }

    #[test]
    fn every_structural_invariant_is_enforced() {
        // A 4-row column and a 2-row matrix differ in shape.
        let col = Operator::from(TraceColumn::new(2, 1, 0, 0));
        let small = Operator::from(SparseMatrix::<F>::new(1, 3, vec![]));
        let one = |factors: Vec<usize>| Constraint::new(vec![Term::new(F::ONE, factors)]);

        // No matrices.
        assert_eq!(
            Ccs::<F>::new(vec![], vec![one(vec![0])]),
            Err(StructureError::NoOperators)
        );

        // No constraints.
        assert_eq!(
            Ccs::<F>::new(vec![col.clone()], vec![]),
            Err(StructureError::NoConstraints)
        );

        // Two shapes.
        assert_eq!(
            Ccs::new(vec![col.clone(), small], vec![one(vec![0])]),
            Err(StructureError::ShapeMismatch {
                operator: 1,
                log_rows: 1,
                log_cols: 3,
                expected_log_rows: 2,
                expected_log_cols: 3,
            })
        );

        // A constraint with no term.
        assert_eq!(
            Ccs::new(vec![col.clone()], vec![Constraint::new(vec![])]),
            Err(StructureError::EmptyConstraint { constraint: 0 })
        );

        // A term with no factor.
        assert_eq!(
            Ccs::new(vec![col.clone()], vec![one(vec![])]),
            Err(StructureError::EmptyTerm {
                constraint: 0,
                term: 0
            })
        );

        // A factor past the last matrix.
        assert_eq!(
            Ccs::new(vec![col], vec![one(vec![1])]),
            Err(StructureError::FactorOutOfRange {
                constraint: 0,
                term: 0,
                factor: 1,
                operators: 1
            })
        );
    }

    #[test]
    fn selectors_extend_their_row_indicators() {
        use p3_field::extension::BinomialExtensionField;
        use p3_multilinear_util::point::Point;

        /// Challenge field.
        type EF = BinomialExtensionField<F, 4>;

        // Invariant: each closed form extends its 0/1 row table.
        //
        // Fixture state: 3 row variables, a non-Boolean point.
        let r: Vec<EF> = [3, 7, 11].map(EF::from_u32).to_vec();
        for selector in [
            RowSelector::All,
            RowSelector::First,
            RowSelector::Last,
            RowSelector::Transition,
        ] {
            let dense: EF = (0..8)
                .filter(|&row| selector.holds(row, 8))
                .map(|row| Point::new(r.as_slice()).equality_at_vertex(row).unwrap())
                .sum();
            assert_eq!(selector.evaluate(&r), dense, "{selector:?}");
        }
    }
}
