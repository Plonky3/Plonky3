use alloc::vec::Vec;

use hashbrown::HashMap;
use p3_field::Field;

use super::analysis::{IndexedDef, OpDef};
use crate::ops::{AluOpKind, Op};
use crate::types::WitnessId;

/// Detects `a * b + c` patterns and rewrites them as fused `MulAdd` ops.
///
/// # Algorithm
///
/// **Phase 1** — scan every `Add` for a single-use `Mul` operand; record it as a candidate.
/// **Phase 2** — iteratively discard candidates whose addend won't be available at the
///              mul's position (cross-fusion ordering).
/// **Phase 3** — build the final op list, replacing muls with MulAdds and dropping
///              consumed adds.
pub(super) struct MulAddFusion<F> {
    use_counts: HashMap<WitnessId, usize>,
    defs: HashMap<WitnessId, IndexedDef<F>>,
    backwards_computed: HashMap<WitnessId, usize>,
}

impl<F: Field> MulAddFusion<F> {
    /// Scans `ops` to build use-counts, definitions, and backwards-op tracking.
    pub(super) fn new(ops: &[Op<F>]) -> Self {
        let mut fusion = Self {
            use_counts: HashMap::new(),
            defs: HashMap::with_capacity(ops.len()),
            backwards_computed: HashMap::new(),
        };
        fusion.scan_use_counts(ops);
        fusion.scan_defs(ops);
        fusion
    }

    /// Runs the three-phase fusion and returns the rewritten op list.
    pub(super) fn run(self, ops: Vec<Op<F>>) -> Vec<Op<F>> {
        let candidates = self.identify_candidates(&ops);
        let valid = self.filter_valid(&ops, &candidates);
        Self::apply(ops, candidates, &valid)
    }

    fn def_idx(&self, id: &WitnessId) -> Option<usize> {
        self.defs.get(id).map(|d| d.idx)
    }

    fn is_const(&self, id: &WitnessId) -> bool {
        self.defs.get(id).is_some_and(|d| d.def.is_const())
    }

    fn uses(&self, id: &WitnessId) -> usize {
        self.use_counts.get(id).copied().unwrap_or(0)
    }

    fn is_backwards(&self, idx: usize, out: &WitnessId) -> bool {
        self.def_idx(out).is_some_and(|i| i < idx)
    }

    /// Inserts a def unless the witness is already a Const (connect aliasing).
    fn insert_def(&mut self, id: WitnessId, idx: usize, def: OpDef<F>) {
        if !self.is_const(&id) {
            self.defs.insert(id, IndexedDef::new(idx, def));
        }
    }

    /// Where `witness` will be available, accounting for fusions that move outputs.
    fn effective_position(
        &self,
        witness: WitnessId,
        fused_positions: &HashMap<WitnessId, usize>,
    ) -> Option<usize> {
        fused_positions
            .get(&witness)
            .copied()
            .or_else(|| self.def_idx(&witness))
    }

    fn scan_use_counts(&mut self, ops: &[Op<F>]) {
        for op in ops {
            match op {
                Op::Alu { a, b, c, .. } => {
                    *self.use_counts.entry(*a).or_default() += 1;
                    *self.use_counts.entry(*b).or_default() += 1;
                    if let Some(c) = c {
                        *self.use_counts.entry(*c).or_default() += 1;
                    }
                }
                Op::NonPrimitiveOpWithExecutor { inputs, .. } => {
                    for &id in inputs.iter().flatten() {
                        *self.use_counts.entry(id).or_default() += 1;
                    }
                }
                _ => {}
            }
        }
    }

    fn scan_defs(&mut self, ops: &[Op<F>]) {
        for (idx, op) in ops.iter().enumerate() {
            match op {
                Op::Const { out, val } => {
                    // Always insert consts (they win over any prior def).
                    self.defs
                        .insert(*out, IndexedDef::new(idx, OpDef::Const(*val)));
                }
                Op::Alu {
                    kind: AluOpKind::Mul,
                    a,
                    b,
                    out,
                    c: None,
                    ..
                } => {
                    self.track_backwards_op(idx, *out, *b);
                    self.insert_def(*out, idx, OpDef::Mul { a: *a, b: *b });
                }
                Op::Alu {
                    kind: AluOpKind::Add,
                    b,
                    out,
                    c: None,
                    ..
                } => {
                    self.track_backwards_op(idx, *out, *b);
                    self.insert_def(*out, idx, OpDef::Other);
                }
                Op::Alu { out, .. } | Op::Public { out, .. } => {
                    self.insert_def(*out, idx, OpDef::Other);
                }
                Op::NonPrimitiveOpWithExecutor { outputs, .. } => {
                    for &id in outputs.iter().flatten() {
                        self.insert_def(id, idx, OpDef::Other);
                    }
                }
                Op::Hint { outputs, .. } => {
                    for &id in outputs {
                        self.insert_def(id, idx, OpDef::Other);
                    }
                }
            }
        }
    }

    /// If `out` is already defined before `idx`, this is a "backwards" op (e.g. sub
    /// encoded as add). Record that `computed` is produced at `idx`.
    fn track_backwards_op(&mut self, idx: usize, out: WitnessId, computed: WitnessId) {
        if !self.is_const(&out) && self.is_backwards(idx, &out) {
            self.backwards_computed.insert(computed, idx);
            self.insert_def(computed, idx, OpDef::Other);
        }
    }

    fn identify_candidates(&self, ops: &[Op<F>]) -> HashMap<usize, (usize, Op<F>, WitnessId)> {
        let mut candidates = HashMap::new();

        for (add_idx, op) in ops.iter().enumerate() {
            let Op::Alu {
                kind: AluOpKind::Add,
                a,
                b,
                c: None,
                out,
                ..
            } = op
            else {
                continue;
            };

            if self.is_const(out) || self.is_backwards(add_idx, out) {
                continue;
            }

            // Try both orientations: a as mul result, then b.
            let candidate = self
                .try_fuse(*a, *b, *out, add_idx)
                .or_else(|| self.try_fuse(*b, *a, *out, add_idx));

            if let Some(c) = candidate {
                candidates.insert(add_idx, c);
            }
        }

        candidates
    }

    /// Checks whether `mul_result` comes from a fusable Mul and `addend` is safe to use.
    /// Returns `(mul_idx, replacement_op, addend)`.
    fn try_fuse(
        &self,
        mul_result: WitnessId,
        addend: WitnessId,
        out: WitnessId,
        add_idx: usize,
    ) -> Option<(usize, Op<F>, WitnessId)> {
        let indexed = self.defs.get(&mul_result)?;
        let (mul_a, mul_b) = indexed.def.as_mul()?;
        let mul_idx = indexed.idx;

        // Single-use, non-const mul
        if self.uses(&mul_result) != 1 || self.is_const(&mul_result) {
            return None;
        }

        // Addend must already be computed
        if self.def_idx(&addend).is_some_and(|i| i >= add_idx) {
            return None;
        }

        // Addend must not come from a backwards op at or after mul_idx
        if self
            .backwards_computed
            .get(&addend)
            .is_some_and(|&i| i >= mul_idx)
        {
            return None;
        }

        // Mul's second operand must be available at mul's position
        if self.def_idx(&mul_b).is_some_and(|i| i >= mul_idx) {
            return None;
        }

        let muladd = Op::Alu {
            kind: AluOpKind::MulAdd,
            a: mul_a,
            b: mul_b,
            c: Some(addend),
            out,
            intermediate_out: Some(mul_result),
        };

        Some((mul_idx, muladd, addend))
    }

    /// Keeps only candidates whose addend is available at the mul's position,
    /// re-checking after each removal (because removing a fusion changes positions).
    fn filter_valid(
        &self,
        ops: &[Op<F>],
        candidates: &HashMap<usize, (usize, Op<F>, WitnessId)>,
    ) -> hashbrown::HashSet<usize> {
        let mut valid: hashbrown::HashSet<usize> = candidates.keys().copied().collect();

        loop {
            let fused_positions: HashMap<WitnessId, usize> = valid
                .iter()
                .filter_map(|&add_idx| {
                    let (mul_idx, _, _) = candidates.get(&add_idx)?;
                    let Op::Alu { out, .. } = ops.get(add_idx)? else {
                        return None;
                    };
                    Some((*out, *mul_idx))
                })
                .collect();

            let before = valid.len();
            valid.retain(|&add_idx| {
                let Some((mul_idx, _, addend)) = candidates.get(&add_idx) else {
                    return true;
                };
                self.effective_position(*addend, &fused_positions)
                    .is_none_or(|pos| pos < *mul_idx)
            });

            if valid.len() == before {
                break;
            }
        }

        valid
    }

    /// Builds the final op list from the validated fusion set.
    fn apply(
        ops: Vec<Op<F>>,
        mut candidates: HashMap<usize, (usize, Op<F>, WitnessId)>,
        valid: &hashbrown::HashSet<usize>,
    ) -> Vec<Op<F>> {
        let mut consumed_adds = hashbrown::HashSet::new();
        let mut mul_replacements: HashMap<usize, Op<F>> = HashMap::new();

        for &add_idx in valid {
            if let Some((mul_idx, muladd, _)) = candidates.remove(&add_idx)
                && !mul_replacements.contains_key(&mul_idx)
            {
                mul_replacements.insert(mul_idx, muladd);
                consumed_adds.insert(add_idx);
            }
        }

        ops.into_iter()
            .enumerate()
            .filter(|(idx, _)| !consumed_adds.contains(idx))
            .map(|(idx, op)| mul_replacements.remove(&idx).unwrap_or(op))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use p3_test_utils::baby_bear_params::{BabyBear, PrimeCharacteristicRing};

    use super::*;
    use crate::ops::Op;

    type F = BabyBear;

    #[test]
    fn test_basic_fusion() {
        let (a, b, c) = (WitnessId(0), WitnessId(1), WitnessId(2));
        let (mul_out, add_out) = (WitnessId(3), WitnessId(4));

        let ops: Vec<Op<F>> = vec![Op::mul(a, b, mul_out), Op::add(mul_out, c, add_out)];

        let fused = MulAddFusion::new(&ops).run(ops);

        let expected = vec![Op::Alu {
            kind: AluOpKind::MulAdd,
            a,
            b,
            c: Some(c),
            out: add_out,
            intermediate_out: Some(mul_out),
        }];
        assert_eq!(fused, expected);
    }

    #[test]
    fn test_symmetric_fusion() {
        let (a, b, c) = (WitnessId(0), WitnessId(1), WitnessId(2));
        let (mul_out, add_out) = (WitnessId(3), WitnessId(4));

        let ops: Vec<Op<F>> = vec![Op::mul(a, b, mul_out), Op::add(c, mul_out, add_out)];

        let fused = MulAddFusion::new(&ops).run(ops);

        let expected = vec![Op::Alu {
            kind: AluOpKind::MulAdd,
            a,
            b,
            c: Some(c),
            out: add_out,
            intermediate_out: Some(mul_out),
        }];
        assert_eq!(fused, expected);
    }

    #[test]
    fn test_no_fusion_when_mul_has_multiple_uses() {
        let (a, b, c) = (WitnessId(0), WitnessId(1), WitnessId(2));
        let mul_out = WitnessId(3);

        let ops: Vec<Op<F>> = vec![
            Op::mul(a, b, mul_out),
            Op::add(mul_out, c, WitnessId(4)),
            Op::add(mul_out, a, WitnessId(5)),
        ];

        let fused = MulAddFusion::new(&ops).run(ops.clone());
        assert_eq!(fused, ops);
    }

    #[test]
    fn test_chained_fusion() {
        let acc0 = WitnessId(0);
        let (bit0, pow0, term0, acc1) = (WitnessId(1), WitnessId(2), WitnessId(3), WitnessId(4));
        let (bit1, pow1, term1, acc2) = (WitnessId(5), WitnessId(6), WitnessId(7), WitnessId(8));

        let ops: Vec<Op<F>> = vec![
            Op::Const {
                out: acc0,
                val: F::ZERO,
            },
            Op::Const {
                out: pow0,
                val: F::ONE,
            },
            Op::Const {
                out: pow1,
                val: F::TWO,
            },
            Op::mul(bit0, pow0, term0),
            Op::add(acc0, term0, acc1),
            Op::mul(bit1, pow1, term1),
            Op::add(acc1, term1, acc2),
        ];

        let fused = MulAddFusion::new(&ops).run(ops);

        assert_eq!(
            fused,
            vec![
                Op::Const {
                    out: acc0,
                    val: F::ZERO,
                },
                Op::Const {
                    out: pow0,
                    val: F::ONE,
                },
                Op::Const {
                    out: pow1,
                    val: F::TWO,
                },
                Op::Alu {
                    kind: AluOpKind::MulAdd,
                    a: bit0,
                    b: pow0,
                    c: Some(acc0),
                    out: acc1,
                    intermediate_out: Some(term0),
                },
                Op::Alu {
                    kind: AluOpKind::MulAdd,
                    a: bit1,
                    b: pow1,
                    c: Some(acc1),
                    out: acc2,
                    intermediate_out: Some(term1),
                },
            ]
        );
    }

    #[test]
    fn test_backwards_add_not_fused() {
        let b = WitnessId(0);
        let one = WitnessId(1);
        let b_minus_one = WitnessId(2);

        let ops: Vec<Op<F>> = vec![
            Op::Const {
                out: one,
                val: F::ONE,
            },
            Op::Const {
                out: WitnessId(3),
                val: F::TWO,
            },
            Op::Const {
                out: WitnessId(5),
                val: F::ZERO,
            },
            Op::add(one, b_minus_one, b), // backwards
            Op::mul(b, WitnessId(3), WitnessId(4)),
            Op::add(WitnessId(5), WitnessId(4), WitnessId(6)),
        ];

        let fused = MulAddFusion::new(&ops).run(ops);
        assert_eq!(
            fused,
            vec![
                Op::Const {
                    out: one,
                    val: F::ONE,
                },
                Op::Const {
                    out: WitnessId(3),
                    val: F::TWO,
                },
                Op::Const {
                    out: WitnessId(5),
                    val: F::ZERO,
                },
                Op::add(one, b_minus_one, b),
                Op::Alu {
                    kind: AluOpKind::MulAdd,
                    a: b,
                    b: WitnessId(3),
                    c: Some(WitnessId(5)),
                    out: WitnessId(6),
                    intermediate_out: Some(WitnessId(4)),
                },
            ]
        );
    }

    #[test]
    fn test_empty_input() {
        let ops: Vec<Op<F>> = vec![];
        let fused = MulAddFusion::new(&ops).run(ops);
        assert!(fused.is_empty());
    }

    #[test]
    fn test_no_fusion_when_mul_output_is_const() {
        // If the mul output is a const alias, skip fusion
        let ops: Vec<Op<F>> = vec![
            Op::Const {
                out: WitnessId(3),
                val: F::TWO,
            },
            Op::mul(WitnessId(0), WitnessId(1), WitnessId(3)), // output is already a const
            Op::add(WitnessId(3), WitnessId(2), WitnessId(4)),
        ];

        let fused = MulAddFusion::new(&ops).run(ops.clone());
        assert_eq!(fused, ops);
    }

    #[test]
    fn test_no_fusion_when_addend_not_yet_computed() {
        // addend is defined after the add — should not fuse
        let ops: Vec<Op<F>> = vec![
            Op::mul(WitnessId(0), WitnessId(1), WitnessId(2)),
            Op::add(WitnessId(2), WitnessId(3), WitnessId(4)),
            Op::Const {
                out: WitnessId(3),
                val: F::ONE,
            },
        ];

        let fused = MulAddFusion::new(&ops).run(ops.clone());
        assert_eq!(fused, ops);
    }

    #[test]
    fn test_no_fusion_when_mul_b_not_available() {
        // mul's b operand is defined after the mul itself
        let ops: Vec<Op<F>> = vec![
            Op::mul(WitnessId(0), WitnessId(5), WitnessId(2)),
            Op::Const {
                out: WitnessId(5),
                val: F::TWO,
            },
            Op::Const {
                out: WitnessId(3),
                val: F::ONE,
            },
            Op::add(WitnessId(2), WitnessId(3), WitnessId(4)),
        ];

        let fused = MulAddFusion::new(&ops).run(ops.clone());
        assert_eq!(fused, ops);
    }

    #[test]
    fn test_only_first_add_fused_per_mul() {
        // Two adds consuming the same mul — mul has 2 uses → no fusion at all
        let ops: Vec<Op<F>> = vec![
            Op::mul(WitnessId(0), WitnessId(1), WitnessId(2)),
            Op::add(WitnessId(2), WitnessId(3), WitnessId(4)),
            Op::add(WitnessId(2), WitnessId(5), WitnessId(6)),
        ];

        let fused = MulAddFusion::new(&ops).run(ops.clone());
        assert_eq!(fused, ops);
    }

    #[test]
    fn test_const_only_passthrough() {
        let ops: Vec<Op<F>> = vec![
            Op::Const {
                out: WitnessId(0),
                val: F::ZERO,
            },
            Op::Const {
                out: WitnessId(1),
                val: F::ONE,
            },
        ];

        let fused = MulAddFusion::new(&ops).run(ops.clone());
        assert_eq!(fused, ops);
    }

    #[test]
    fn test_muladd_preserves_intermediate_out() {
        let (a, b, c) = (WitnessId(0), WitnessId(1), WitnessId(2));
        let (mul_out, add_out) = (WitnessId(3), WitnessId(4));

        let ops: Vec<Op<F>> = vec![Op::mul(a, b, mul_out), Op::add(mul_out, c, add_out)];

        let fused = MulAddFusion::new(&ops).run(ops);
        assert_eq!(
            fused,
            vec![Op::Alu {
                kind: AluOpKind::MulAdd,
                a,
                b,
                c: Some(c),
                out: add_out,
                intermediate_out: Some(mul_out),
            }]
        );
    }
}
