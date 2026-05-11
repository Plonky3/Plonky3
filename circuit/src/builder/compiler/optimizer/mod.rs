mod analysis;
mod dedup;
mod fuse_mul_add;

use alloc::vec::Vec;

use dedup::Deduplicator;
use fuse_mul_add::MulAddFusion;
use hashbrown::HashMap;
use p3_field::Field;

use crate::ops::Op;
use crate::types::WitnessId;

/// Runs optimization passes on primitive operations.
///
/// Takes ownership of the op list and produces an optimized version plus a rewrite map:
/// any witness ID that was removed as a duplicate points to its canonical witness.
/// The caller should apply this map to expr_to_widx, public_rows, and tag_to_witness
/// so no reference stays to removed IDs.
///
/// Currently implements:
/// - ALU deduplication: removes duplicate ALU ops (identical or same inputs via connect)
/// - MulAdd fusion: detects `a * b + c` patterns and fuses them into MulAdd ops
///
/// *Note*: CSE is implemented within the
/// [ExpressionBuilder](crate::builder::expression_builder::ExpressionBuilder) itself.
pub struct Optimizer<F>(core::marker::PhantomData<F>);

impl<F: Field> Optimizer<F> {
    pub fn optimize(ops: Vec<Op<F>>) -> (Vec<Op<F>>, HashMap<WitnessId, WitnessId>) {
        let (ops, rewrite) = Deduplicator::new().run(ops);
        let ops = MulAddFusion::new(&ops).run(ops);
        (ops, rewrite)
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use p3_test_utils::baby_bear_params::{BabyBear, PrimeCharacteristicRing};

    use super::*;
    use crate::CircuitBuilder;
    use crate::ops::AluOpKind;

    type F = BabyBear;

    #[test]
    fn test_passthrough() {
        let ops: Vec<Op<F>> = vec![
            Op::Const {
                out: WitnessId(0),
                val: F::ZERO,
            },
            Op::add(WitnessId(0), WitnessId(1), WitnessId(2)),
        ];

        let (optimized, _) = Optimizer::optimize(ops.clone());
        assert_eq!(optimized, ops);
    }

    #[test]
    fn test_bool_check_passthrough_and_muladd() {
        // BoolCheck ops now come directly from lowering (not fusion).
        // The optimizer should pass them through unchanged while still fusing MulAdd.
        let ops: Vec<Op<F>> = vec![
            Op::Const {
                out: WitnessId(0),
                val: F::ONE,
            },
            Op::bool_check(WitnessId(1), WitnessId(1), WitnessId(2)),
            Op::mul(WitnessId(3), WitnessId(4), WitnessId(5)), // a * c
            Op::add(WitnessId(5), WitnessId(0), WitnessId(6)), // a*c + 1 -> MulAdd
        ];

        let (optimized, _) = Optimizer::optimize(ops);

        let bool_checks = optimized
            .iter()
            .filter(|op| op.is_alu_kind(AluOpKind::BoolCheck))
            .count();
        let mul_adds = optimized
            .iter()
            .filter(|op| op.is_alu_kind(AluOpKind::MulAdd))
            .count();

        assert_eq!(bool_checks, 1, "Expected 1 BoolCheck passthrough");
        assert!(mul_adds >= 1, "Expected at least 1 MulAdd");
    }

    #[test]
    fn test_single_op_circuit() {
        let mut builder = CircuitBuilder::<F>::new();
        builder.define_const(F::from_u64(42));
        let traces = builder.build().unwrap().runner().run().unwrap();
        assert!(!traces.const_trace.values.is_empty());
    }

    #[test]
    fn test_public_input_only_circuit() {
        let mut builder = CircuitBuilder::<F>::new();
        builder.public_input();
        let circuit = builder.build().unwrap();
        let mut runner = circuit.runner();
        runner.set_public_inputs(&[F::from_u64(99)]).unwrap();
        assert!(!runner.run().unwrap().public_trace.values.is_empty());
    }

    #[test]
    fn test_large_circuit_correctness() {
        const N: usize = 5000;
        let mut builder = CircuitBuilder::<F>::new();
        let expected = builder.alloc_public_input("expected");
        let mut acc = builder.define_const(F::ZERO);
        for i in 1..=N {
            let ci = builder.define_const(F::from_u64(i as u64));
            acc = builder.add(acc, ci);
        }
        builder.connect(acc, expected);

        let circuit = builder.build().unwrap();
        let mut runner = circuit.runner();
        runner
            .set_public_inputs(&[F::from_u64((N * (N + 1) / 2) as u64)])
            .unwrap();
        assert!(runner.run().unwrap().witness_trace.num_rows() > N);
    }

    #[test]
    fn test_empty_ops() {
        let ops: Vec<Op<F>> = vec![];
        let (optimized, rewrite) = Optimizer::optimize(ops);
        assert!(optimized.is_empty());
        assert!(rewrite.is_empty());
    }

    #[test]
    fn test_optimizer_idempotent() {
        let ops: Vec<Op<F>> = vec![
            Op::Const {
                out: WitnessId(0),
                val: F::ONE,
            },
            Op::bool_check(WitnessId(1), WitnessId(1), WitnessId(2)),
            Op::mul(WitnessId(3), WitnessId(4), WitnessId(5)),
            Op::add(WitnessId(5), WitnessId(0), WitnessId(6)),
        ];

        let (first_pass, _) = Optimizer::optimize(ops);
        let (second_pass, rewrite2) = Optimizer::optimize(first_pass.clone());

        assert_eq!(first_pass, second_pass, "Optimizer should be idempotent");
        assert!(
            rewrite2.is_empty(),
            "Second pass should produce no rewrites"
        );
    }

    #[test]
    fn test_optimizer_never_increases_op_count() {
        let ops: Vec<Op<F>> = vec![
            Op::Const {
                out: WitnessId(0),
                val: F::ZERO,
            },
            Op::mul(WitnessId(1), WitnessId(2), WitnessId(3)),
            Op::add(WitnessId(3), WitnessId(0), WitnessId(4)),
            Op::mul(WitnessId(1), WitnessId(2), WitnessId(5)), // dup
            Op::add(WitnessId(5), WitnessId(0), WitnessId(6)), // dup
        ];

        let original_len = ops.len();
        let (optimized, _) = Optimizer::optimize(ops);
        assert!(optimized.len() <= original_len);
    }
}
