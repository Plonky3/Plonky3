use alloc::vec::Vec;

use p3_field::Field;

use crate::ops::AluOpKind;
use crate::types::WitnessId;

/// Record of an ALU operation captured during execution (avoids re-reading witness).
#[derive(Debug, Clone)]
pub struct AluOpRecord<F> {
    /// The kind of ALU operation (Add, Mul, BoolCheck, MulAdd, HornerAcc).
    pub kind: AluOpKind,
    /// Witness index of the first operand `a`.
    pub a_index: WitnessId,
    /// Witness index of the second operand `b`.
    pub b_index: WitnessId,
    /// Witness index of the third operand `c` (only meaningful for MulAdd).
    pub c_index: WitnessId,
    /// Witness index of the output value.
    pub out_index: WitnessId,
    /// Concrete value of `a` at execution time.
    pub a_val: F,
    /// Concrete value of `b` at execution time.
    pub b_val: F,
    /// Concrete value of `c` at execution time (zero for non-MulAdd ops).
    pub c_val: F,
    /// Concrete output value at execution time.
    pub out_val: F,
}

/// Unified ALU operation table.
///
/// Records all ALU operations (Add, Mul, BoolCheck, MulAdd, HornerAcc) in the circuit.
/// Each row represents one constraint based on the operation kind:
/// - Add: a + b = out
/// - Mul: a * b = out
/// - BoolCheck: a * (a - 1) = 0, out = a
/// - MulAdd: a * b + c = out
/// - HornerAcc: out = prev_row_out * b + c - a
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AluTrace<F> {
    /// Operation kind for each row
    pub op_kind: Vec<AluOpKind>,
    /// Operand values (a, b, c, out)
    pub values: Vec<[F; 4]>,
    /// Operand indices (a, b, c, out)
    pub indices: Vec<[WitnessId; 4]>,
}

impl<F> AluTrace<F> {
    /// Builds an ALU trace from execution records (no witness lookups).
    pub fn from_records(records: Vec<AluOpRecord<F>>) -> Self
    where
        F: Field,
    {
        let mut op_kind = Vec::with_capacity(records.len());
        let mut values = Vec::with_capacity(records.len());
        let mut indices = Vec::with_capacity(records.len());

        for r in records {
            op_kind.push(r.kind);
            values.push([r.a_val, r.b_val, r.c_val, r.out_val]);
            indices.push([r.a_index, r.b_index, r.c_index, r.out_index]);
        }

        if op_kind.is_empty() {
            op_kind.push(AluOpKind::Add);
            values.push([F::ZERO, F::ZERO, F::ZERO, F::ZERO]);
            indices.push([WitnessId(0), WitnessId(0), WitnessId(0), WitnessId(0)]);
        }

        Self {
            op_kind,
            values,
            indices,
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_test_utils::baby_bear_params::{BabyBear, PrimeCharacteristicRing};

    use super::*;
    use crate::{CircuitError, Op};

    /// Builder for generating ALU traces.
    pub struct AluTraceBuilder<'a, F> {
        primitive_ops: &'a [Op<F>],
        witness: &'a [Option<F>],
    }

    impl<'a, F: Field> AluTraceBuilder<'a, F> {
        /// Creates a new ALU trace builder.
        pub const fn new(primitive_ops: &'a [Op<F>], witness: &'a [Option<F>]) -> Self {
            Self {
                primitive_ops,
                witness,
            }
        }

        /// Builds the ALU trace from circuit operations.
        pub fn build(self) -> Result<AluTrace<F>, CircuitError> {
            let mut op_kind = Vec::with_capacity(1 << 15);
            let mut values = Vec::with_capacity(1 << 15);
            let mut indices = Vec::with_capacity(1 << 15);

            for prim in self.primitive_ops {
                if let Op::Alu {
                    kind, a, b, c, out, ..
                } = prim
                {
                    let a_val = self.resolve(a)?;
                    let b_val = self.resolve(b)?;
                    let c_val = if let Some(c_id) = c {
                        self.resolve(c_id)?
                    } else {
                        F::ZERO
                    };
                    let out_val = self.resolve(out)?;

                    op_kind.push(*kind);
                    values.push([a_val, b_val, c_val, out_val]);
                    indices.push([*a, *b, c.unwrap_or(WitnessId(0)), *out]);
                }
            }

            // If trace is empty, add a dummy row: 0 + 0 = 0
            if values.is_empty() {
                op_kind.push(AluOpKind::Add);
                values.push([F::ZERO, F::ZERO, F::ZERO, F::ZERO]);
                indices.push([WitnessId(0), WitnessId(0), WitnessId(0), WitnessId(0)]);
            }

            Ok(AluTrace {
                op_kind,
                values,
                indices,
            })
        }

        /// Resolves a single witness value safely.
        #[inline]
        fn resolve(&self, id: &WitnessId) -> Result<F, CircuitError> {
            #[cfg(debug_assertions)]
            {
                self.witness
                    .get(id.0 as usize)
                    .and_then(|opt| opt.as_ref())
                    .copied()
                    .ok_or(CircuitError::WitnessNotSet { witness_id: *id })
            }

            #[cfg(not(debug_assertions))]
            {
                unsafe {
                    Ok(*self
                        .witness
                        .get_unchecked(id.0 as usize)
                        .as_ref()
                        .expect("witness not set?"))
                }
            }
        }
    }

    type F = BabyBear;

    #[test]
    fn test_single_addition() {
        let a = F::from_u64(5);
        let b = F::from_u64(3);
        let out = F::from_u64(8);
        let witness = vec![Some(a), Some(b), Some(out)];

        let ops = vec![Op::add(WitnessId(0), WitnessId(1), WitnessId(2))];

        let builder = AluTraceBuilder::new(&ops, &witness);
        let trace = builder.build().expect("Failed to build trace");

        assert_eq!(
            trace,
            AluTrace {
                op_kind: vec![AluOpKind::Add],
                values: vec![[a, b, F::ZERO, out]],
                indices: vec![[WitnessId(0), WitnessId(1), WitnessId(0), WitnessId(2)]],
            }
        );
    }

    #[test]
    fn test_single_multiplication() {
        let a = F::from_u64(5);
        let b = F::from_u64(3);
        let out = F::from_u64(15);
        let witness = vec![Some(a), Some(b), Some(out)];

        let ops = vec![Op::mul(WitnessId(0), WitnessId(1), WitnessId(2))];

        let builder = AluTraceBuilder::new(&ops, &witness);
        let trace = builder.build().expect("Failed to build trace");

        assert_eq!(
            trace,
            AluTrace {
                op_kind: vec![AluOpKind::Mul],
                values: vec![[a, b, F::ZERO, out]],
                indices: vec![[WitnessId(0), WitnessId(1), WitnessId(0), WitnessId(2)]],
            }
        );
    }

    #[test]
    fn test_mul_add() {
        // a * b + c = out => 5 * 3 + 2 = 17
        let a = F::from_u64(5);
        let b = F::from_u64(3);
        let c = F::from_u64(2);
        let out = F::from_u64(17);
        let witness = vec![Some(a), Some(b), Some(c), Some(out)];

        let ops = vec![Op::mul_add(
            WitnessId(0),
            WitnessId(1),
            WitnessId(2),
            WitnessId(3),
        )];

        let builder = AluTraceBuilder::new(&ops, &witness);
        let trace = builder.build().expect("Failed to build trace");

        assert_eq!(
            trace,
            AluTrace {
                op_kind: vec![AluOpKind::MulAdd],
                values: vec![[a, b, c, out]],
                indices: vec![[WitnessId(0), WitnessId(1), WitnessId(2), WitnessId(3)]],
            }
        );
    }

    #[test]
    fn test_bool_check() {
        // BoolCheck: a * (a - 1) = 0, out = a
        // For a = 1: 1 * 0 = 0 ✓
        let a = F::ONE;
        let witness = vec![Some(a), Some(F::ZERO)]; // a and placeholder for b

        let ops = vec![Op::bool_check(WitnessId(0), WitnessId(1), WitnessId(0))];

        let builder = AluTraceBuilder::new(&ops, &witness);
        let trace = builder.build().expect("Failed to build trace");

        assert_eq!(
            trace,
            AluTrace {
                op_kind: vec![AluOpKind::BoolCheck],
                values: vec![[a, F::ZERO, F::ZERO, a]],
                indices: vec![[WitnessId(0), WitnessId(1), WitnessId(0), WitnessId(0)]],
            }
        );
    }

    #[test]
    fn test_mixed_operations() {
        let a1 = F::from_u64(10);
        let b1 = F::from_u64(20);
        let out1 = F::from_u64(30); // add

        let a2 = F::from_u64(7);
        let b2 = F::from_u64(3);
        let out2 = F::from_u64(21); // mul

        let witness = vec![
            Some(a1),
            Some(b1),
            Some(out1),
            Some(a2),
            Some(b2),
            Some(out2),
        ];

        let ops = vec![
            Op::add(WitnessId(0), WitnessId(1), WitnessId(2)),
            Op::mul(WitnessId(3), WitnessId(4), WitnessId(5)),
        ];

        let builder = AluTraceBuilder::new(&ops, &witness);
        let trace = builder.build().expect("Failed to build trace");

        assert_eq!(
            trace,
            AluTrace {
                op_kind: vec![AluOpKind::Add, AluOpKind::Mul],
                values: vec![[a1, b1, F::ZERO, out1], [a2, b2, F::ZERO, out2]],
                indices: vec![
                    [WitnessId(0), WitnessId(1), WitnessId(0), WitnessId(2)],
                    [WitnessId(3), WitnessId(4), WitnessId(0), WitnessId(5)],
                ],
            }
        );
    }

    #[test]
    fn test_empty_operations_creates_dummy_row() {
        let witness = vec![Some(F::ZERO)];
        let ops: Vec<Op<F>> = vec![];

        let builder = AluTraceBuilder::new(&ops, &witness);
        let trace = builder.build().expect("Failed to build trace");

        assert_eq!(
            trace,
            AluTrace {
                op_kind: vec![AluOpKind::Add],
                values: vec![[F::ZERO, F::ZERO, F::ZERO, F::ZERO]],
                indices: vec![[WitnessId(0), WitnessId(0), WitnessId(0), WitnessId(0)]],
            }
        );
    }

    #[cfg(debug_assertions)]
    #[test]
    fn test_missing_witness_returns_error() {
        let witness = vec![None, Some(F::from_u64(5)), Some(F::from_u64(5))];

        let ops = vec![Op::add(WitnessId(0), WitnessId(1), WitnessId(2))];

        let builder = AluTraceBuilder::new(&ops, &witness);
        let result = builder.build();

        assert!(result.is_err());
        match result {
            Err(CircuitError::WitnessNotSet { witness_id }) => {
                assert_eq!(witness_id, WitnessId(0));
            }
            _ => panic!("Expected WitnessNotSet error"),
        }
    }
}
