use alloc::vec::Vec;

use hashbrown::HashMap;
use p3_field::Field;

use super::analysis::AluKey;
use crate::ops::Op;
use crate::types::WitnessId;

/// Removes duplicate ALU operations by tracking a canonical output per `AluKey`.
///
/// When a duplicate is found its output witness is rewritten to the canonical one.
/// Later ops see the canonical ID through `apply_witness_rewrite`.
pub(super) struct Deduplicator {
    rewrite: HashMap<WitnessId, WitnessId>,
    seen: HashMap<AluKey, WitnessId>,
}

impl Deduplicator {
    pub(super) fn new() -> Self {
        Self {
            rewrite: HashMap::new(),
            seen: HashMap::new(),
        }
    }

    /// Consumes the op list and returns deduplicated ops + the rewrite map.
    pub(super) fn run<F: Field>(
        mut self,
        ops: Vec<Op<F>>,
    ) -> (Vec<Op<F>>, HashMap<WitnessId, WitnessId>) {
        let mut result = Vec::with_capacity(ops.len());

        for mut op in ops {
            op.apply_witness_rewrite(&self.rewrite);

            if let Some((dup_out, canonical)) = self.detect_duplicate(&op) {
                let root = canonical.resolve(&self.rewrite);
                if dup_out != root {
                    self.rewrite.insert(dup_out, root);
                }
                continue;
            }

            result.push(op);
        }

        (result, self.rewrite)
    }

    /// Returns `Some((duplicate_out, canonical_out))` when `op` duplicates an earlier ALU.
    fn detect_duplicate<F: Field>(&mut self, op: &Op<F>) -> Option<(WitnessId, WitnessId)> {
        let Op::Alu {
            kind, a, b, c, out, ..
        } = op
        else {
            return None;
        };

        let key = AluKey::new(
            *kind,
            a.resolve(&self.rewrite),
            b.resolve(&self.rewrite),
            c.map(|id| id.resolve(&self.rewrite)),
        );

        if let Some(&canonical) = self.seen.get(&key) {
            Some((*out, canonical))
        } else {
            self.seen.insert(key, *out);
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use p3_test_utils::baby_bear_params::{BabyBear, PrimeCharacteristicRing};

    use super::*;
    use crate::CircuitBuilder;

    type F = BabyBear;

    #[test]
    fn test_duplicated_op_fusion() {
        let a = WitnessId(0);
        let b = WitnessId(1);
        let c = WitnessId(2);
        let mul_out = WitnessId(3);
        let mul_out2 = WitnessId(5);
        let add_out = WitnessId(4);

        let ops: Vec<Op<F>> = vec![
            Op::mul(a, b, mul_out),
            Op::mul(a, b, mul_out2),
            Op::add(mul_out, c, add_out),
        ];

        let (deduped, rewrite) = Deduplicator::new().run(ops);

        assert_eq!(
            deduped,
            vec![Op::mul(a, b, mul_out), Op::add(mul_out, c, add_out)]
        );
        assert_eq!(rewrite.get(&mul_out2), Some(&mul_out));
    }

    #[test]
    fn test_duplicated_op_fusion_in_builder() {
        let mut builder = CircuitBuilder::<F>::new();
        let a = builder.define_const(F::TWO);
        let b = builder.public_input();
        let c = builder.public_input();

        builder.connect(b, c);
        builder.alloc_mul(a, b, "mul_result1");
        builder.alloc_mul(a, c, "mul_result2");

        let circuit = builder.build().unwrap();
        let mut runner = circuit.runner();
        runner
            .set_public_inputs(&[F::from_u32(42), F::from_u32(42)])
            .unwrap();

        assert_eq!(runner.run().unwrap().alu_trace.values.len(), 1);
    }

    #[test]
    fn test_all_deduplicated_circuit() {
        let mut builder = CircuitBuilder::<F>::new();
        let a = builder.define_const(F::TWO);
        let b = builder.public_input();
        let r1 = builder.mul(a, b);
        let r2 = builder.mul(a, b);
        builder.connect(r1, r2);

        let circuit = builder.build().unwrap();
        let mut runner = circuit.runner();
        runner.set_public_inputs(&[F::from_u64(5)]).unwrap();

        let traces = runner.run().unwrap();
        let mul_count = traces
            .alu_trace
            .values
            .iter()
            .filter(|row| row[3] == F::from_u64(10))
            .count();
        assert_eq!(mul_count, 1, "Duplicate mul should be deduped");
    }

    #[test]
    fn test_empty_input() {
        let ops: Vec<Op<F>> = vec![];
        let (deduped, rewrite) = Deduplicator::new().run(ops);
        assert!(deduped.is_empty());
        assert!(rewrite.is_empty());
    }

    #[test]
    fn test_non_alu_ops_pass_through() {
        let ops: Vec<Op<F>> = vec![
            Op::Const {
                out: WitnessId(0),
                val: F::ONE,
            },
            Op::Const {
                out: WitnessId(1),
                val: F::TWO,
            },
        ];

        let (deduped, rewrite) = Deduplicator::new().run(ops.clone());
        assert_eq!(deduped, ops);
        assert!(rewrite.is_empty());
    }

    #[test]
    fn test_commutative_dedup_add() {
        let (a, b) = (WitnessId(0), WitnessId(1));
        let ops: Vec<Op<F>> = vec![
            Op::add(a, b, WitnessId(2)),
            Op::add(b, a, WitnessId(3)), // same as above (commutative)
        ];

        let (deduped, rewrite) = Deduplicator::new().run(ops);
        assert_eq!(deduped, vec![Op::add(a, b, WitnessId(2))]);
        assert_eq!(rewrite.get(&WitnessId(3)), Some(&WitnessId(2)));
    }

    #[test]
    fn test_commutative_dedup_mul() {
        let (a, b) = (WitnessId(0), WitnessId(1));
        let ops: Vec<Op<F>> = vec![Op::mul(a, b, WitnessId(2)), Op::mul(b, a, WitnessId(3))];

        let (deduped, rewrite) = Deduplicator::new().run(ops);
        assert_eq!(deduped, vec![Op::mul(a, b, WitnessId(2))]);
        assert_eq!(rewrite.get(&WitnessId(3)), Some(&WitnessId(2)));
    }

    #[test]
    fn test_chained_rewrite() {
        // op0: a + b = c
        // op1: a + b = d  (dup of op0, d -> c)
        // op2: d + a = e  (d rewrites to c, so this is c + a = e)
        // op3: c + a = f  (dup of op2 after rewrite, f -> e)
        let (a, b) = (WitnessId(0), WitnessId(1));
        let ops: Vec<Op<F>> = vec![
            Op::add(a, b, WitnessId(2)),
            Op::add(a, b, WitnessId(3)),
            Op::add(WitnessId(3), a, WitnessId(4)),
            Op::add(WitnessId(2), a, WitnessId(5)),
        ];

        let (deduped, rewrite) = Deduplicator::new().run(ops);
        assert_eq!(
            deduped,
            vec![
                Op::add(a, b, WitnessId(2)),
                Op::add(WitnessId(2), a, WitnessId(4)),
            ]
        );
        assert_eq!(rewrite.get(&WitnessId(3)), Some(&WitnessId(2)));
        assert_eq!(rewrite.get(&WitnessId(5)), Some(&WitnessId(4)));
    }

    #[test]
    fn test_distinct_ops_not_deduped() {
        let ops: Vec<Op<F>> = vec![
            Op::add(WitnessId(0), WitnessId(1), WitnessId(2)),
            Op::mul(WitnessId(0), WitnessId(1), WitnessId(3)),
        ];

        let (deduped, rewrite) = Deduplicator::new().run(ops.clone());
        assert_eq!(deduped, ops);
        assert!(rewrite.is_empty());
    }
}
