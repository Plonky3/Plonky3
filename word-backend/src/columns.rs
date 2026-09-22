//! Packed relation columns derived from a checked witness.

use alloc::vec::Vec;
use core::array;

use p3_word::{ConstraintSystem, Operand, ValueIndex};

use crate::statement::Statement;
use crate::{Packed, PackedWitness, PackedWord, WitnessError};

/// Packed rows used by the relation reductions.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OperationColumns<W: PackedWord> {
    /// The operand each linear relation requires to vanish.
    zero: Vec<Packed<W>>,
    /// The left input, right input, and output of each bitwise product.
    bitwise_and: [Vec<Packed<W>>; 3],
    /// The factors and result limbs of each unsigned product.
    integer_mul: [Vec<Packed<W>>; 4],
}

impl<W: PackedWord> OperationColumns<W> {
    /// Evaluates the operands needed by the nonlinear reductions.
    ///
    /// A composed statement walks each gadget once per instance.
    ///
    /// A relation's column position is therefore exactly its composed relation index.
    ///
    /// # Errors
    ///
    /// Returns an error when the witness does not have the statement's shape.
    pub fn new(statement: &Statement<W>, values: &PackedWitness<W>) -> Result<Self, WitnessError> {
        values.check_shape(statement)?;
        let counts = statement.relation_counts();
        let mut columns = Self {
            zero: Vec::with_capacity(counts[0]),
            bitwise_and: array::from_fn(|_| Vec::with_capacity(counts[1])),
            integer_mul: array::from_fn(|_| Vec::with_capacity(counts[2])),
        };

        match statement {
            // A flat statement addresses the composed segments directly.
            Statement::Flat(system) => columns.extend(system, values, |index| index),
            Statement::Composed(composition) => {
                for (call, index) in composition.calls().iter().zip(0_usize..) {
                    let component = call.component();
                    for instance in 0..call.instances() {
                        // The affine map is the same one the compiled layout reverses.
                        columns.extend(component.body(), values, |slot| {
                            composition
                                .resolve(index, instance, slot)
                                .expect("the layout bounded every call and instance")
                        });
                    }
                }
            }
        }
        Ok(columns)
    }

    /// Appends one relation block, readdressing every term as it is read.
    fn extend(
        &mut self,
        system: &ConstraintSystem<W>,
        values: &PackedWitness<W>,
        remap: impl Fn(ValueIndex) -> ValueIndex,
    ) {
        let evaluate = |operand: &Operand<W>| {
            let mut result = W::pack(W::ZERO);
            for term in operand.terms() {
                let word = values
                    .get(remap(term.index()))
                    .expect("the checked system only contains in-bounds indices");
                result += W::pack(term.apply(word));
            }
            result
        };

        // The linear family has one semantic operand per relation.
        self.zero.extend(
            system
                .zero_constraints()
                .iter()
                .map(|constraint| evaluate(constraint.value())),
        );

        // One constraint-major pass fills all three bitwise reduction columns.
        for constraint in system.and_constraints() {
            self.bitwise_and[0].push(evaluate(constraint.left()));
            self.bitwise_and[1].push(evaluate(constraint.right()));
            self.bitwise_and[2].push(evaluate(constraint.output()));
        }

        // One constraint-major pass fills all four integer-product columns.
        for constraint in system.integer_mul_constraints() {
            self.integer_mul[0].push(evaluate(constraint.left()));
            self.integer_mul[1].push(evaluate(constraint.right()));
            self.integer_mul[2].push(evaluate(constraint.low()));
            self.integer_mul[3].push(evaluate(constraint.high()));
        }
    }

    /// Returns the operand of each linear relation.
    #[inline]
    pub fn zero(&self) -> &[Packed<W>] {
        &self.zero
    }

    /// Returns the left, right, and output AND operands.
    #[inline]
    pub const fn bitwise_and(&self) -> &[Vec<Packed<W>>; 3] {
        &self.bitwise_and
    }

    /// Returns the factors followed by the low and high product limbs.
    #[inline]
    pub const fn integer_mul(&self) -> &[Vec<Packed<W>>; 4] {
        &self.integer_mul
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_word::{
        AndConstraint, ConstraintSystem, IntegerMulConstraint, Operand, Shift, ShiftKind,
        ShiftedValue, ValueIndex, Word64,
    };
    use proptest::prelude::*;

    use super::*;

    fn operand(terms: Vec<ShiftedValue<Word64>>) -> Operand<Word64> {
        Operand::new(terms)
    }

    #[test]
    fn packed_columns_match_scalar_operand_evaluation() {
        let public_index = ValueIndex::public(0).unwrap();
        let witness_index = ValueIndex::witness(0).unwrap();
        let shift = Shift::new(ShiftKind::Lane32RotateRight, 7).unwrap();
        let left = operand(vec![ShiftedValue::single(witness_index, shift)]);
        let right = operand(vec![ShiftedValue::plain(public_index)]);
        let output = operand(vec![]);
        let and = AndConstraint::new(left.clone(), right.clone(), output);
        let mul =
            IntegerMulConstraint::new(left.clone(), right.clone(), left.clone(), right.clone());
        let system = ConstraintSystem::new(1, 1, vec![], vec![and], vec![mul]).unwrap();
        let public = [Word64::new(0x0102_0304_0506_0708)];
        let witness = [Word64::new(0xfedc_ba98_7654_3210)];
        let packed = PackedWitness::new(&system, &public, &witness).unwrap();
        let columns = OperationColumns::new(&Statement::from(system), &packed).unwrap();

        assert_eq!(
            columns.bitwise_and()[0][0].to_bits(),
            left.evaluate(&public, &witness).unwrap().get()
        );
        assert_eq!(
            columns.bitwise_and()[1][0].to_bits(),
            right.evaluate(&public, &witness).unwrap().get()
        );
        for (column, expected) in columns
            .integer_mul()
            .iter()
            .zip([&left, &right, &left, &right])
        {
            assert_eq!(
                column[0].to_bits(),
                expected.evaluate(&public, &witness).unwrap().get()
            );
        }
    }

    #[test]
    fn empty_families_have_no_protocol_padding() {
        let system = ConstraintSystem::<Word64>::new(0, 0, vec![], vec![], vec![]).unwrap();
        let packed = PackedWitness::new(&system, &[], &[]).unwrap();
        let columns = OperationColumns::new(&Statement::from(system), &packed).unwrap();

        assert!(columns.bitwise_and().iter().all(Vec::is_empty));
        assert!(columns.integer_mul().iter().all(Vec::is_empty));
    }

    #[test]
    fn rejects_a_witness_packed_for_another_shape() {
        let source = ConstraintSystem::<Word64>::new(0, 0, vec![], vec![], vec![]).unwrap();
        let target = ConstraintSystem::<Word64>::new(0, 1, vec![], vec![], vec![]).unwrap();
        let packed = PackedWitness::new(&source, &[], &[]).unwrap();

        assert_eq!(
            OperationColumns::new(&Statement::from(target), &packed),
            Err(WitnessError {
                segment: p3_word::Segment::Witness,
                expected: 1,
                actual: 0,
            })
        );
    }

    proptest! {
        #[test]
        fn random_packed_columns_match_the_scalar_reference(
            public_value in any::<u64>(),
            witness_value in any::<u64>(),
            full_amount in 1usize..64,
            lane_amount in 1usize..32,
        ) {
            let public_index = ValueIndex::public(0).unwrap();
            let witness_index = ValueIndex::witness(0).unwrap();
            let full = Shift::new(ShiftKind::RotateRight, full_amount).unwrap();
            let lanes = Shift::new(ShiftKind::Lane32ArithmeticRight, lane_amount).unwrap();
            let left = operand(vec![
                ShiftedValue::single(witness_index, full),
                ShiftedValue::plain(public_index),
            ]);
            let right = operand(vec![ShiftedValue::single(public_index, lanes)]);
            let output = operand(vec![ShiftedValue::plain(witness_index)]);
            let and = AndConstraint::new(left.clone(), right.clone(), output.clone());
            let mul = IntegerMulConstraint::new(
                left.clone(),
                right.clone(),
                output.clone(),
                left.clone(),
            );
            let system = ConstraintSystem::new(1, 1, vec![], vec![and], vec![mul]).unwrap();
            let public = [Word64::new(public_value)];
            let witness = [Word64::new(witness_value)];
            let packed = PackedWitness::new(&system, &public, &witness).unwrap();
            let columns = OperationColumns::new(&Statement::from(system), &packed).unwrap();

            for (column, expected) in columns
                .bitwise_and()
                .iter()
                .zip([&left, &right])
            {
                prop_assert_eq!(
                    column[0].to_bits(),
                    expected.evaluate(&public, &witness).unwrap().get()
                );
            }
            for (column, expected) in columns
                .integer_mul()
                .iter()
                .zip([&left, &right, &output, &left])
            {
                prop_assert_eq!(
                    column[0].to_bits(),
                    expected.evaluate(&public, &witness).unwrap().get()
                );
            }
        }
    }
}
