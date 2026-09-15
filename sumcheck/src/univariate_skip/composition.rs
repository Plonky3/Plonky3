//! The constraint a binary zerocheck proves vanishing.

use p3_field::Algebra;

/// A constraint read over any algebra containing the witness alphabet.
///
/// # Overview
///
/// A skip round evaluates its constraint twice over different carriers:
///
/// ```text
///     on the transmitted points  ->  subfield values, from extending bit rows
///     in the residual rounds     ->  large-field values, after the first challenge
/// ```
///
/// One implementation covers both: the arithmetic is the same, only the carrier changes.
///
/// A trait rather than a closure is what lets the round read it in whichever carrier it holds.
///
/// # Degree
///
/// The declared degree sizes the transmitted domain.
///
/// It therefore has to be an upper bound on the total degree in the operands.
///
/// Declaring it too low leaves the round polynomial unrecoverable from what is sent.
///
/// That makes the reduction unsound.
///
/// Declaring it too high only costs cosets.
pub trait Composition<F> {
    /// Number of operands the constraint reads.
    fn arity(&self) -> usize;

    /// Total degree of the constraint in its operands.
    fn degree(&self) -> usize;

    /// Read the constraint on one tuple of operand values.
    ///
    /// # Panics
    ///
    /// May panic if the tuple does not carry one value per operand.
    fn eval<A: Algebra<F> + Copy>(&self, values: &[A]) -> A;
}

/// The conjunction constraint of a bit-valued rank-one system.
///
/// # Overview
///
/// Three operands, asserting that the third is the bitwise conjunction of the first two:
///
/// ```text
///     a & b = c        on bits,  a * b = c
/// ```
///
/// Characteristic two makes subtraction addition, so the constraint is read as `a*b + c`.
///
/// This is the shape a batch of Boolean circuits reduces to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Conjunction;

impl<F> Composition<F> for Conjunction {
    fn arity(&self) -> usize {
        3
    }

    fn degree(&self) -> usize {
        2
    }

    fn eval<A: Algebra<F> + Copy>(&self, values: &[A]) -> A {
        assert_eq!(values.len(), 3, "the conjunction reads three operands");

        // In characteristic two the difference and the sum coincide.
        values[0] * values[1] + values[2]
    }
}

#[cfg(test)]
mod tests {
    use p3_binary_field::{BinaryField8, BinaryField128, TowerLevel};
    use p3_field::PrimeCharacteristicRing;

    use super::*;

    /// The witness alphabet every test reads the constraint over.
    type F = BinaryField8;

    /// Read the constraint in the alphabet itself, as the transmitted points are.
    fn in_alphabet(values: &[F]) -> F {
        Composition::<F>::eval(&Conjunction, values)
    }

    /// Read the constraint in the large field, as the residual rounds are.
    fn in_extension(values: &[BinaryField128]) -> BinaryField128 {
        Composition::<F>::eval(&Conjunction, values)
    }

    #[test]
    fn the_conjunction_vanishes_exactly_on_satisfying_bits() {
        // Fixture state: every assignment of three bits.
        //
        //     a b c  ->  a*b + c
        //     0 0 0  ->  0        satisfied
        //     1 1 1  ->  0        satisfied
        //     1 1 0  ->  1        broken
        //     0 1 1  ->  1        broken
        for a in [0u8, 1] {
            for b in [0u8, 1] {
                for c in [0u8, 1] {
                    let bit = |v: u8| if v == 1 { F::ONE } else { F::ZERO };
                    let value = in_alphabet(&[bit(a), bit(b), bit(c)]);
                    let satisfied = (a & b) == c;
                    assert_eq!(value == F::ZERO, satisfied, "{a}{b}{c}");
                }
            }
        }
    }

    #[test]
    fn one_constraint_reads_in_both_carriers() {
        // The round evaluates the same constraint over subfield and large-field values.
        //
        // Embedding then reading must agree with reading then embedding.
        //
        // Otherwise the message and the residual rounds prove different things.
        let small = [1u8, 1, 0].map(F::from_repr);
        let large = small.map(BinaryField128::from);

        assert_eq!(
            BinaryField128::from(in_alphabet(&small)),
            in_extension(&large)
        );
    }

    #[test]
    fn the_declared_shape_matches_what_the_constraint_reads() {
        // The arity sizes the operand tuple and the degree sizes the transmitted domain.
        //
        // Both are read by the round rather than inferred, so they are pinned here.
        assert_eq!(Composition::<F>::arity(&Conjunction), 3);
        assert_eq!(Composition::<F>::degree(&Conjunction), 2);
    }
}
