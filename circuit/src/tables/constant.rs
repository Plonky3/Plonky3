use alloc::vec::Vec;

use p3_field::PrimeCharacteristicRing;

use crate::CircuitError;
use crate::ops::Op;
use crate::types::WitnessId;

/// Constant values table.
///
/// Stores all compile-time known constant values used in the circuit.
/// Each constant binds to a specific witness ID.
/// Both prover and verifier know these values in advance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConstTrace<F> {
    /// Witness IDs that each constant binds to.
    ///
    /// Maps each constant to its location in the witness table.
    pub index: Vec<WitnessId>,
    /// Constant field element values.
    pub values: Vec<F>,
}

/// Builder for generating constant traces.
pub struct ConstTraceBuilder<'a, F> {
    primitive_ops: &'a [Op<F>],
}

impl<'a, F: PrimeCharacteristicRing> ConstTraceBuilder<'a, F> {
    /// Creates a new constant trace builder.
    pub const fn new(primitive_ops: &'a [Op<F>]) -> Self {
        Self { primitive_ops }
    }

    /// Builds the constant trace from circuit operations.
    pub fn build(self) -> Result<ConstTrace<F>, CircuitError> {
        let mut index = Vec::with_capacity(1 << 9);
        let mut values = Vec::with_capacity(1 << 9);

        for prim in self.primitive_ops {
            if let Op::Const { out, val } = prim {
                index.push(*out);
                values.push(val.dup());
            }
        }

        Ok(ConstTrace { index, values })
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_test_utils::baby_bear_params::{BabyBear, PrimeCharacteristicRing};

    use super::*;

    type F = BabyBear;

    #[test]
    fn test_single_constant() {
        // Create a single constant operation that loads a value into witness
        let val = F::from_u64(42);
        let out = WitnessId(0);

        let ops = vec![Op::Const { out, val }];

        // Build the trace using the builder pattern
        let builder = ConstTraceBuilder::new(&ops);
        let trace = builder.build().expect("Failed to build trace");

        assert_eq!(
            trace,
            ConstTrace {
                index: vec![out],
                values: vec![val],
            }
        );
    }

    #[test]
    fn test_multiple_constants() {
        // Create multiple constant operations with different values
        let val1 = F::from_u64(10);
        let out1 = WitnessId(0);

        let val2 = F::from_u64(20);
        let out2 = WitnessId(1);

        let val3 = F::from_u64(30);
        let out3 = WitnessId(2);

        let ops = vec![
            Op::Const {
                out: out1,
                val: val1,
            },
            Op::Const {
                out: out2,
                val: val2,
            },
            Op::Const {
                out: out3,
                val: val3,
            },
        ];

        // Build the trace
        let builder = ConstTraceBuilder::new(&ops);
        let trace = builder.build().expect("Failed to build trace");

        assert_eq!(
            trace,
            ConstTrace {
                index: vec![out1, out2, out3],
                values: vec![val1, val2, val3],
            }
        );
    }

    #[test]
    fn test_empty_operations() {
        // Provide an empty operations list
        let ops: Vec<Op<F>> = vec![];

        // Build the trace
        let builder = ConstTraceBuilder::new(&ops);
        let trace = builder.build().expect("Failed to build trace");

        assert_eq!(
            trace,
            ConstTrace {
                index: vec![],
                values: vec![],
            }
        );
    }
}
