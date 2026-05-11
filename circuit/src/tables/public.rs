use alloc::vec::Vec;

use p3_field::{Dup, PrimeCharacteristicRing};

use crate::CircuitError;
use crate::ops::Op;
use crate::types::WitnessId;

/// Public input table.
///
/// Unlike compile-time `Const` values, these inputs are provided at runtime
/// and are known to both the prover and the verifier.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PublicTrace<F> {
    /// Witness IDs of each public input.
    ///
    /// Identifies which witness slots contain public values.
    pub index: Vec<WitnessId>,

    /// Public input field element values.
    ///
    /// Provided at the start of the execution.
    /// Serve as the starting point for computation.
    pub values: Vec<F>,
}

/// Builder for generating public input traces.
pub struct PublicTraceBuilder<'a, F> {
    primitive_ops: &'a [Op<F>],
    witness: &'a [Option<F>],
}

impl<'a, F: PrimeCharacteristicRing> PublicTraceBuilder<'a, F> {
    /// Creates a new public trace builder.
    pub const fn new(primitive_ops: &'a [Op<F>], witness: &'a [Option<F>]) -> Self {
        Self {
            primitive_ops,
            witness,
        }
    }

    /// Builds the public input trace from circuit operations.
    pub fn build(self) -> Result<PublicTrace<F>, CircuitError> {
        let mut index = Vec::with_capacity(1 << 15);
        let mut values = Vec::with_capacity(1 << 15);

        for prim in self.primitive_ops {
            if let Op::Public { out, public_pos: _ } = prim {
                index.push(*out);
                let value = self
                    .witness
                    .get(out.0 as usize)
                    .and_then(|opt| opt.as_ref().map(Dup::dup))
                    .ok_or(CircuitError::WitnessNotSet { witness_id: *out })?;
                values.push(value);
            }
        }

        Ok(PublicTrace { index, values })
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_test_utils::baby_bear_params::{BabyBear, PrimeCharacteristicRing};

    use super::*;

    type F = BabyBear;

    #[test]
    fn test_single_public_input() {
        // Create a single public input operation that reads from witness
        let out = WitnessId(0);
        let val = F::from_u64(42);

        let ops = vec![Op::Public { out, public_pos: 0 }];

        // Prepare the witness table with the public input value
        let witness = vec![Some(val)];

        // Build the trace using the builder pattern
        let builder = PublicTraceBuilder::new(&ops, &witness);
        let trace = builder.build().expect("Failed to build trace");

        assert_eq!(
            trace,
            PublicTrace {
                index: vec![out],
                values: vec![val],
            }
        );
    }

    #[test]
    fn test_multiple_public_inputs() {
        // Create multiple public input operations with non-contiguous witness indices
        let out1 = WitnessId(0);
        let out2 = WitnessId(2);

        let val1 = F::from_u64(10);
        let val2 = F::from_u64(30);

        let ops = vec![
            Op::Public {
                out: out1,
                public_pos: 0,
            },
            Op::Public {
                out: out2,
                public_pos: 1,
            },
        ];

        // Prepare witness table with gaps (index 1 is unused)
        let witness = vec![Some(val1), None, Some(val2)];

        // Build the trace
        let builder = PublicTraceBuilder::new(&ops, &witness);
        let trace = builder.build().expect("Failed to build trace");

        assert_eq!(
            trace,
            PublicTrace {
                index: vec![out1, out2],
                values: vec![val1, val2],
            }
        );
    }

    #[test]
    fn test_empty_operations() {
        // Provide an empty operations list
        let ops: Vec<Op<F>> = vec![];
        let witness: Vec<Option<F>> = vec![];

        // Build the trace
        let builder = PublicTraceBuilder::new(&ops, &witness);
        let trace = builder.build().expect("Failed to build trace");

        assert_eq!(
            trace,
            PublicTrace {
                index: vec![],
                values: vec![],
            }
        );
    }

    #[test]
    fn test_witness_not_set_error() {
        // Create a public input operation referencing an unset witness slot
        let out = WitnessId(0);
        let ops = vec![Op::Public { out, public_pos: 0 }];

        // Witness table has the slot but value is None (not yet set)
        let witness: Vec<Option<F>> = vec![None];

        // Attempt to build the trace
        let builder = PublicTraceBuilder::new(&ops, &witness);
        let result = builder.build();

        // Verify the build fails with the expected error
        assert!(result.is_err(), "Should fail when witness is not set");
        assert!(matches!(
            result,
            Err(CircuitError::WitnessNotSet { witness_id }) if witness_id == out
        ));
    }
}
