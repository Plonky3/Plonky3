use alloc::boxed::Box;
use alloc::vec::Vec;

use hashbrown::HashMap;
use p3_field::Field;
use strum_macros::EnumCount;

use super::executor::{HintExecutor, NonPrimitiveExecutor};
use crate::types::{NonPrimitiveOpId, WitnessId};

/// ALU operation kinds for the unified arithmetic table.
///
/// This enum defines the different arithmetic operations that can be performed
/// in a single ALU row, selected by preprocessed selectors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AluOpKind {
    /// Addition: out = a + b
    Add,
    /// Multiplication: out = a * b
    Mul,
    /// Boolean check: a * (a - 1) = 0, out = a
    BoolCheck,
    /// Fused multiply-add: out = a * b + c
    MulAdd,
    /// Row-chained Horner accumulator: out = prev_row_out * b + c - a
    ///
    /// Uses the `out` column of the previous ALU row (same lane) as an implicit
    /// accumulator input. Fuses multiply-add-subtract into a single row.
    /// HornerAcc ops within the same chain must be placed in consecutive ALU rows
    /// of the same lane.
    HornerAcc,
}

/// Circuit operations.
///
/// Operations are distinguised as primitive, non-primitive, and hints:
///
/// - Primitive ops (`Const`, `Public`, `Alu`) are the basic arithmetic building blocks
/// - Non-primitive ops (`NonPrimitiveOpWithExecutor`) are table-backed plugin operations
/// - Hint ops (`Hint`) are non-deterministic witness assignments that do NOT have tables,
///   AIR, or traces; they are purely a convenience for filling witnesses.
#[derive(Debug)]
pub enum Op<F> {
    /// Load a constant value into the witness table
    ///
    /// Sets `witness[out] = val`. Used for literal constants and
    /// supports constant pooling optimization where identical constants
    /// reuse the same witness slot.
    Const { out: WitnessId, val: F },

    /// Load a public input value into the witness table
    ///
    /// Sets `witness[out] = public_inputs[public_pos]`. Public inputs
    /// are values known to both prover and verifier, typically used
    /// for circuit inputs and expected outputs.
    Public { out: WitnessId, public_pos: usize },

    /// Unified ALU operation supporting multiple arithmetic operations.
    ///
    /// The `kind` field determines the operation:
    /// - `Add`: out = a + b
    /// - `Mul`: out = a * b
    /// - `BoolCheck`: a * (a - 1) = 0, out = a
    /// - `MulAdd`: out = a * b + c
    /// - `HornerAcc`: out = acc * b + c - a (acc from previous row's out, same lane)
    Alu {
        kind: AluOpKind,
        a: WitnessId,
        b: WitnessId,
        /// Third operand, used for MulAdd and HornerAcc
        c: Option<WitnessId>,
        out: WitnessId,
        /// Intermediate output for MulAdd: stores a * b when fused from separate mul + add.
        /// For HornerAcc: stores the accumulator WitnessId (previous Horner step's out).
        /// The runner sets this witness value so dependent operations still work.
        intermediate_out: Option<WitnessId>,
    },

    /// Hint operation: non-deterministically fills witness values via a user-provided closure.
    ///
    /// Hints are NOT table-backed:
    /// - they do not have an AIR
    /// - they do not participate in non-primitive traces
    /// - they do not have private data or configs
    ///
    /// They are used for things like bit decompositions and extension-field decompositions.
    Hint {
        /// Input witnesses read by the hint.
        inputs: Vec<WitnessId>,
        /// Output witnesses written by the hint.
        outputs: Vec<WitnessId>,
        /// User-provided executor that implements the hint logic.
        executor: Box<dyn HintExecutor<F>>,
    },

    /// Non-primitive operation with executor-based dispatch
    NonPrimitiveOpWithExecutor {
        inputs: Vec<Vec<WitnessId>>,
        outputs: Vec<Vec<WitnessId>>,
        executor: Box<dyn NonPrimitiveExecutor<F>>,
        /// For private data lookup and error reporting
        op_id: NonPrimitiveOpId,
    },
}

impl<F> Op<F> {
    /// Create an addition operation (convenience wrapper for Op::Alu with AluOpKind::Add).
    pub const fn add(a: WitnessId, b: WitnessId, out: WitnessId) -> Self {
        Self::Alu {
            kind: AluOpKind::Add,
            a,
            b,
            c: None,
            out,
            intermediate_out: None,
        }
    }

    /// Create a multiplication operation (convenience wrapper for Op::Alu with AluOpKind::Mul).
    pub const fn mul(a: WitnessId, b: WitnessId, out: WitnessId) -> Self {
        Self::Alu {
            kind: AluOpKind::Mul,
            a,
            b,
            c: None,
            out,
            intermediate_out: None,
        }
    }

    /// Create a fused multiply-add operation: out = a * b + c.
    pub const fn mul_add(a: WitnessId, b: WitnessId, c: WitnessId, out: WitnessId) -> Self {
        Self::Alu {
            kind: AluOpKind::MulAdd,
            a,
            b,
            c: Some(c),
            out,
            intermediate_out: None,
        }
    }

    /// Create a Horner accumulator: out = acc * b + c - a.
    ///
    /// `acc` is the previous Horner step's output (used by runner for computation).
    /// In the AIR, the accumulator comes implicitly from the previous row's `out` column.
    pub const fn horner_acc(
        a: WitnessId,
        b: WitnessId,
        c: WitnessId,
        out: WitnessId,
        acc: WitnessId,
    ) -> Self {
        Self::Alu {
            kind: AluOpKind::HornerAcc,
            a,
            b,
            c: Some(c),
            out,
            intermediate_out: Some(acc),
        }
    }

    /// Create a boolean check operation: a * (a - 1) = 0.
    pub const fn bool_check(a: WitnessId, b: WitnessId, out: WitnessId) -> Self {
        Self::Alu {
            kind: AluOpKind::BoolCheck,
            a,
            b,
            c: None,
            out,
            intermediate_out: None,
        }
    }

    /// Check if this is an ALU operation of the given kind.
    pub fn is_alu_kind(&self, kind: AluOpKind) -> bool {
        matches!(self, Self::Alu { kind: k, .. } if *k == kind)
    }

    /// Rewrite witness IDs in place using the given map (follows chains to canonical ID).
    /// Used by the optimizer to apply ALU dedup without re-boxing non-primitive executors.
    pub fn apply_witness_rewrite(&mut self, rewrite: &HashMap<WitnessId, WitnessId>) {
        if rewrite.is_empty() {
            return;
        }
        match self {
            Self::Const { out, .. } => *out = out.resolve(rewrite),
            Self::Public { out, .. } => *out = out.resolve(rewrite),
            Self::Alu {
                a,
                b,
                c,
                out,
                intermediate_out,
                ..
            } => {
                *a = a.resolve(rewrite);
                *b = b.resolve(rewrite);
                *c = c.map(|id| id.resolve(rewrite));
                *out = out.resolve(rewrite);
                *intermediate_out = intermediate_out.map(|id| id.resolve(rewrite));
            }
            Self::Hint {
                inputs, outputs, ..
            } => {
                for w in inputs.iter_mut() {
                    *w = w.resolve(rewrite);
                }
                for w in outputs.iter_mut() {
                    *w = w.resolve(rewrite);
                }
            }
            Self::NonPrimitiveOpWithExecutor {
                inputs, outputs, ..
            } => {
                for g in inputs.iter_mut() {
                    for w in g.iter_mut() {
                        *w = w.resolve(rewrite);
                    }
                }
                for g in outputs.iter_mut() {
                    for w in g.iter_mut() {
                        *w = w.resolve(rewrite);
                    }
                }
            }
        }
    }
}

#[derive(EnumCount, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PrimitiveOpType {
    Const = 0,
    Public = 1,
    /// Unified ALU table (combines Add, Mul, BoolCheck, MulAdd)
    Alu = 2,
}

#[allow(clippy::fallible_impl_from)]
impl From<usize> for PrimitiveOpType {
    fn from(value: usize) -> Self {
        match value {
            0 => Self::Const,
            1 => Self::Public,
            2 => Self::Alu,
            _ => panic!("Invalid PrimitiveOpType value: {}", value),
        }
    }
}

impl<F: Field + Clone> Clone for Op<F> {
    fn clone(&self) -> Self {
        match self {
            Self::Const { out, val } => Self::Const {
                out: *out,
                val: *val,
            },
            Self::Public { out, public_pos } => Self::Public {
                out: *out,
                public_pos: *public_pos,
            },
            Self::Alu {
                kind,
                a,
                b,
                c,
                out,
                intermediate_out,
            } => Self::Alu {
                kind: *kind,
                a: *a,
                b: *b,
                c: *c,
                out: *out,
                intermediate_out: *intermediate_out,
            },
            Self::Hint {
                inputs,
                outputs,
                executor,
            } => Self::Hint {
                inputs: inputs.clone(),
                outputs: outputs.clone(),
                executor: executor.boxed(),
            },
            Self::NonPrimitiveOpWithExecutor {
                inputs,
                outputs,
                executor,
                op_id,
            } => Self::NonPrimitiveOpWithExecutor {
                inputs: inputs.clone(),
                outputs: outputs.clone(),
                executor: executor.boxed(),
                op_id: *op_id,
            },
        }
    }
}

impl<F: Field + PartialEq> PartialEq for Op<F> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Const { out: o1, val: v1 }, Self::Const { out: o2, val: v2 }) => {
                o1 == o2 && v1 == v2
            }
            (
                Self::Public {
                    out: o1,
                    public_pos: p1,
                },
                Self::Public {
                    out: o2,
                    public_pos: p2,
                },
            ) => o1 == o2 && p1 == p2,
            (
                Self::Alu {
                    kind: k1,
                    a: a1,
                    b: b1,
                    c: c1,
                    out: o1,
                    intermediate_out: io1,
                },
                Self::Alu {
                    kind: k2,
                    a: a2,
                    b: b2,
                    c: c2,
                    out: o2,
                    intermediate_out: io2,
                },
            ) => k1 == k2 && a1 == a2 && b1 == b2 && c1 == c2 && o1 == o2 && io1 == io2,
            (
                Self::Hint {
                    inputs: i1,
                    outputs: o1,
                    executor: _,
                },
                Self::Hint {
                    inputs: i2,
                    outputs: o2,
                    executor: _,
                },
            ) => {
                // Compare by value layout only; executors are opaque closures.
                i1 == i2 && o1 == o2
            }
            (
                Self::NonPrimitiveOpWithExecutor {
                    inputs: i1,
                    outputs: o1,
                    executor: e1,
                    op_id: id1,
                },
                Self::NonPrimitiveOpWithExecutor {
                    inputs: i2,
                    outputs: o2,
                    executor: e2,
                    op_id: id2,
                },
            ) => i1 == i2 && o1 == o2 && e1.op_type() == e2.op_type() && id1 == id2,
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use p3_test_utils::baby_bear_params::{BabyBear, PrimeCharacteristicRing};

    use super::*;

    type F = BabyBear;

    #[test]
    fn test_op_partial_eq_different_variants() {
        let const_op = Op::Const {
            out: WitnessId(0),
            val: F::from_u64(5),
        };
        let alu_op = Op::add(WitnessId(0), WitnessId(1), WitnessId(2));
        assert_ne!(const_op, alu_op);
    }

    #[test]
    fn test_op_partial_eq_same_variant_different_values() {
        let alu_op1: Op<F> = Op::add(WitnessId(0), WitnessId(1), WitnessId(2));
        let alu_op2: Op<F> = Op::add(WitnessId(3), WitnessId(4), WitnessId(5));
        assert_ne!(alu_op1, alu_op2);
    }

    #[test]
    fn test_op_partial_eq_alu_add_same_values() {
        let a = WitnessId(0);
        let b = WitnessId(1);
        let out = WitnessId(2);
        let alu_op1: Op<F> = Op::add(a, b, out);
        let alu_op2: Op<F> = Op::add(a, b, out);
        assert_eq!(alu_op1, alu_op2);
    }

    #[test]
    fn test_op_partial_eq_const_different_values() {
        let out = WitnessId(0);
        let const_op1: Op<F> = Op::Const {
            out,
            val: F::from_u64(10),
        };
        let const_op2: Op<F> = Op::Const {
            out,
            val: F::from_u64(20),
        };
        assert_ne!(const_op1, const_op2);
    }

    #[test]
    fn test_op_partial_eq_const_different_outputs() {
        let val = F::from_u64(42);
        let const_op1: Op<F> = Op::Const {
            out: WitnessId(0),
            val,
        };
        let const_op2: Op<F> = Op::Const {
            out: WitnessId(1),
            val,
        };
        assert_ne!(const_op1, const_op2);
    }

    #[test]
    fn test_op_partial_eq_const_same_values() {
        let out = WitnessId(0);
        let val = F::from_u64(99);
        let const_op1: Op<F> = Op::Const { out, val };
        let const_op2: Op<F> = Op::Const { out, val };
        assert_eq!(const_op1, const_op2);
    }

    #[test]
    fn test_op_partial_eq_public_different_positions() {
        let out = WitnessId(0);
        let public_op1: Op<F> = Op::Public { out, public_pos: 0 };
        let public_op2: Op<F> = Op::Public { out, public_pos: 1 };
        assert_ne!(public_op1, public_op2);
    }

    #[test]
    fn test_op_partial_eq_public_same_values() {
        let out = WitnessId(5);
        let public_pos = 3;
        let public_op1: Op<F> = Op::Public { out, public_pos };
        let public_op2: Op<F> = Op::Public { out, public_pos };
        assert_eq!(public_op1, public_op2);
    }

    #[test]
    fn test_op_partial_eq_alu_mul_different_values() {
        let mul_op1: Op<F> = Op::mul(WitnessId(0), WitnessId(1), WitnessId(2));
        let mul_op2: Op<F> = Op::mul(WitnessId(10), WitnessId(11), WitnessId(12));
        assert_ne!(mul_op1, mul_op2);
    }

    #[test]
    fn test_op_partial_eq_alu_mul_same_values() {
        let a = WitnessId(7);
        let b = WitnessId(8);
        let out = WitnessId(9);
        let mul_op1: Op<F> = Op::mul(a, b, out);
        let mul_op2: Op<F> = Op::mul(a, b, out);
        assert_eq!(mul_op1, mul_op2);
    }

    #[test]
    fn test_op_partial_eq_alu_partial_match() {
        let alu_op1: Op<F> = Op::add(WitnessId(0), WitnessId(1), WitnessId(2));
        let alu_op2: Op<F> = Op::add(WitnessId(0), WitnessId(1), WitnessId(99));
        assert_ne!(alu_op1, alu_op2);
    }

    #[test]
    fn test_op_partial_eq_alu_different_kinds() {
        let add_op: Op<F> = Op::add(WitnessId(0), WitnessId(1), WitnessId(2));
        let mul_op: Op<F> = Op::mul(WitnessId(0), WitnessId(1), WitnessId(2));
        assert_ne!(add_op, mul_op);
    }

    #[test]
    fn test_op_partial_eq_alu_muladd() {
        let muladd_op1: Op<F> = Op::mul_add(WitnessId(0), WitnessId(1), WitnessId(2), WitnessId(3));
        let muladd_op2: Op<F> = Op::mul_add(WitnessId(0), WitnessId(1), WitnessId(2), WitnessId(3));
        assert_eq!(muladd_op1, muladd_op2);
    }

    #[test]
    #[should_panic(expected = "Invalid PrimitiveOpType value")]
    fn test_primitive_op_type_invalid_conversion() {
        let _ = PrimitiveOpType::from(999);
    }
}
