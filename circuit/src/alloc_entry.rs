//! Structured allocation logging for circuit expression graphs.

use alloc::collections::BTreeSet;
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;
use core::ops::Deref;

use crate::ExprId;
use crate::ops::NpoTypeId;

/// Classification of an expression allocation.
#[derive(Debug, Clone)]
pub enum AllocationType {
    Public,
    Const,
    Add,
    Sub,
    Mul,
    Div,
    HornerAcc,
    BoolCheck,
    MulAdd,
    NonPrimitiveOp(NpoTypeId),
    WitnessHint,
}

impl AllocationType {
    /// Section header for grouped display.
    const fn group_name(&self) -> &'static str {
        match self {
            Self::Public => "Public Inputs",
            Self::Const => "Constants",
            Self::Add => "Additions",
            Self::Sub => "Subtractions",
            Self::Mul => "Multiplications",
            Self::Div => "Divisions",
            Self::HornerAcc => "Horner Accumulator Steps",
            Self::BoolCheck => "Bool Checks",
            Self::MulAdd => "Fused Multiply-Adds",
            Self::NonPrimitiveOp(_) => "Non-Primitive Operations",
            Self::WitnessHint => "Witness Hints",
        }
    }

    /// Arithmetic operator for binary ops, `None` for everything else.
    const fn operator(&self) -> Option<char> {
        match self {
            Self::Add => Some('+'),
            Self::Sub => Some('-'),
            Self::Mul => Some('*'),
            Self::Div => Some('/'),
            _ => None,
        }
    }
}

impl fmt::Display for AllocationType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Public => f.write_str("Public"),
            Self::Const => f.write_str("Const"),
            Self::Add => f.write_str("Add"),
            Self::Sub => f.write_str("Sub"),
            Self::Mul => f.write_str("Mul"),
            Self::Div => f.write_str("Div"),
            Self::HornerAcc => f.write_str("HornerAcc"),
            Self::BoolCheck => f.write_str("BoolCheck"),
            Self::MulAdd => f.write_str("MulAdd"),
            Self::NonPrimitiveOp(id) => write!(f, "NonPrimitiveOp({id:?})"),
            Self::WitnessHint => f.write_str("WitnessHint"),
        }
    }
}

/// A single recorded allocation with its metadata.
#[derive(Debug, Clone)]
pub struct AllocationEntry {
    /// The expression ID allocated.
    pub expr_id: ExprId,
    /// Classification of the allocation.
    pub alloc_type: AllocationType,
    /// User-provided label (empty when not set).
    pub label: &'static str,
    /// Dependencies: each inner `Vec` is one operand group.
    pub dependencies: Vec<Vec<ExprId>>,
    /// Scope/sub-circuit this allocation belongs to.
    pub scope: Option<String>,
}

impl fmt::Display for AllocationEntry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(op) = self.alloc_type.operator()
            && self.dependencies.len() == 2
        {
            return write!(
                f,
                "expr_{} = expr_{} {op} expr_{}",
                self.expr_id.0, self.dependencies[0][0].0, self.dependencies[1][0].0,
            );
        }

        if let AllocationType::NonPrimitiveOp(_) = &self.alloc_type {
            if self.dependencies.is_empty() {
                return write!(f, "{}", self.alloc_type);
            }
            let deps: Vec<_> = self
                .dependencies
                .iter()
                .flatten()
                .map(|e| format!("expr_{}", e.0))
                .collect();
            return write!(f, "{} (inputs: [{}])", self.alloc_type, deps.join(", "));
        }

        write!(f, "expr_{} ({})", self.expr_id.0, self.alloc_type)
    }
}

/// Structured log of every expression allocation in a circuit.
#[derive(Debug, Clone, Default)]
pub struct AllocationLog(Vec<AllocationEntry>);

impl Deref for AllocationLog {
    type Target = [AllocationEntry];

    fn deref(&self) -> &[AllocationEntry] {
        &self.0
    }
}

impl AllocationLog {
    /// Appends an entry to the log.
    pub fn push(&mut self, entry: AllocationEntry) {
        self.0.push(entry);
    }

    /// Sorted, deduplicated list of scope names in the log.
    pub fn scopes(&self) -> Vec<String> {
        self.iter()
            .filter_map(|e| e.scope.clone())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect()
    }

    /// Emit structured trace events for specific [`ExprId`]s.
    ///
    /// Intended for debugging witness-conflict errors where two `ExprId`s
    /// have been merged to the same `WitnessId`.
    pub fn dump_expr_ids(&self, expr_ids: &[ExprId]) {
        let _span = tracing::debug_span!("expr_id_lookup", requested = ?expr_ids).entered();

        for expr_id in expr_ids {
            if let Some(e) = self.iter().find(|e| e.expr_id == *expr_id) {
                tracing::debug!(
                    expr_id = e.expr_id.0,
                    alloc_type = %e.alloc_type,
                    label = e.label,
                    scope = ?e.scope,
                    deps = ?e.dependencies,
                    expression = %e,
                    "found",
                );
            } else {
                tracing::warn!(expr_id = expr_id.0, "expr_id not found in allocation log");
            }
        }
    }

    /// Dump the entire log: summary, then each scope grouped by type.
    pub fn dump(&self) {
        let _span = tracing::debug_span!("circuit_allocation_log", total = self.len()).entered();

        self.log_summary();

        for scope in self.scopes() {
            self.dump_scope(Some(&scope));
        }
        self.dump_scope(None);
    }

    /// Dump allocations for a single scope, grouped by type.
    pub fn dump_scope(&self, scope: Option<&str>) {
        let scope_name = scope.unwrap_or("main");
        let filtered: Vec<_> = self
            .iter()
            .filter(|e| e.scope.as_deref() == scope)
            .collect();

        if filtered.is_empty() {
            tracing::debug!(scope = scope_name, "no allocations in scope");
            return;
        }

        let _span = tracing::debug_span!("scope", name = scope_name, allocations = filtered.len())
            .entered();

        Self::dump_grouped(&filtered);
    }

    /// Emit one summary event with per-type counts as structured fields.
    fn log_summary(&self) {
        let (mut pub_n, mut cst, mut add, mut sub) = (0u32, 0, 0, 0);
        let (mut mul, mut div, mut hor, mut bck, mut mad, mut npo, mut wit) =
            (0u32, 0, 0, 0, 0, 0, 0);

        for e in self.iter() {
            match e.alloc_type {
                AllocationType::Public => pub_n += 1,
                AllocationType::Const => cst += 1,
                AllocationType::Add => add += 1,
                AllocationType::Sub => sub += 1,
                AllocationType::Mul => mul += 1,
                AllocationType::Div => div += 1,
                AllocationType::HornerAcc => hor += 1,
                AllocationType::BoolCheck => bck += 1,
                AllocationType::MulAdd => mad += 1,
                AllocationType::NonPrimitiveOp(_) => npo += 1,
                AllocationType::WitnessHint => wit += 1,
            }
        }

        tracing::debug!(
            publics = pub_n,
            constants = cst,
            additions = add,
            subtractions = sub,
            multiplications = mul,
            divisions = div,
            horner_steps = hor,
            bool_checks = bck,
            mul_adds = mad,
            non_primitive_ops = npo,
            witness_hints = wit,
            "allocation summary",
        );
    }

    /// Log entries grouped by allocation type inside nested spans.
    fn dump_grouped(entries: &[&AllocationEntry]) {
        /// Filter predicates in display order — one per [`AllocationType`] variant.
        const GROUPS: &[fn(&AllocationType) -> bool] = &[
            |a| matches!(a, AllocationType::Public),
            |a| matches!(a, AllocationType::Const),
            |a| matches!(a, AllocationType::Add),
            |a| matches!(a, AllocationType::Sub),
            |a| matches!(a, AllocationType::Mul),
            |a| matches!(a, AllocationType::Div),
            |a| matches!(a, AllocationType::HornerAcc),
            |a| matches!(a, AllocationType::BoolCheck),
            |a| matches!(a, AllocationType::MulAdd),
            |a| matches!(a, AllocationType::NonPrimitiveOp(_)),
            |a| matches!(a, AllocationType::WitnessHint),
        ];

        for predicate in GROUPS {
            let group: Vec<_> = entries
                .iter()
                .filter(|e| predicate(&e.alloc_type))
                .collect();
            if group.is_empty() {
                continue;
            }

            let group_name = group[0].alloc_type.group_name();
            let _span =
                tracing::debug_span!("group", name = group_name, count = group.len()).entered();

            for entry in &group {
                tracing::debug!(
                    expr_id = entry.expr_id.0,
                    alloc_type = %entry.alloc_type,
                    label = entry.label,
                    expression = %entry,
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::string::ToString;
    use alloc::vec;
    use alloc::vec::Vec;

    use super::*;

    fn entry(
        id: u32,
        alloc_type: AllocationType,
        label: &'static str,
        deps: Vec<Vec<ExprId>>,
        scope: Option<&str>,
    ) -> AllocationEntry {
        AllocationEntry {
            expr_id: ExprId(id),
            alloc_type,
            label,
            dependencies: deps,
            scope: scope.map(|s| s.to_string()),
        }
    }

    #[test]
    fn allocation_type_display_all_variants() {
        assert_eq!(AllocationType::Public.to_string(), "Public");
        assert_eq!(AllocationType::Const.to_string(), "Const");
        assert_eq!(AllocationType::Add.to_string(), "Add");
        assert_eq!(AllocationType::Sub.to_string(), "Sub");
        assert_eq!(AllocationType::Mul.to_string(), "Mul");
        assert_eq!(AllocationType::Div.to_string(), "Div");
        assert_eq!(AllocationType::HornerAcc.to_string(), "HornerAcc");
        assert_eq!(AllocationType::BoolCheck.to_string(), "BoolCheck");
        assert_eq!(AllocationType::MulAdd.to_string(), "MulAdd");
        assert_eq!(AllocationType::WitnessHint.to_string(), "WitnessHint");
        assert_eq!(
            AllocationType::NonPrimitiveOp(NpoTypeId::new("poseidon2")).to_string(),
            "NonPrimitiveOp(NpoTypeId(poseidon2))"
        );
    }

    #[test]
    fn allocation_type_group_names() {
        assert_eq!(AllocationType::Public.group_name(), "Public Inputs");
        assert_eq!(AllocationType::Const.group_name(), "Constants");
        assert_eq!(AllocationType::Add.group_name(), "Additions");
        assert_eq!(AllocationType::Sub.group_name(), "Subtractions");
        assert_eq!(AllocationType::Mul.group_name(), "Multiplications");
        assert_eq!(AllocationType::Div.group_name(), "Divisions");
        assert_eq!(
            AllocationType::HornerAcc.group_name(),
            "Horner Accumulator Steps"
        );
        assert_eq!(AllocationType::BoolCheck.group_name(), "Bool Checks");
        assert_eq!(AllocationType::MulAdd.group_name(), "Fused Multiply-Adds");
        assert_eq!(
            AllocationType::NonPrimitiveOp(NpoTypeId::new("x")).group_name(),
            "Non-Primitive Operations"
        );
        assert_eq!(AllocationType::WitnessHint.group_name(), "Witness Hints");
    }

    #[test]
    fn allocation_type_operator_binary_ops() {
        assert_eq!(AllocationType::Add.operator(), Some('+'));
        assert_eq!(AllocationType::Sub.operator(), Some('-'));
        assert_eq!(AllocationType::Mul.operator(), Some('*'));
        assert_eq!(AllocationType::Div.operator(), Some('/'));
    }

    #[test]
    fn allocation_type_operator_non_binary() {
        assert_eq!(AllocationType::Public.operator(), None);
        assert_eq!(AllocationType::Const.operator(), None);
        assert_eq!(AllocationType::HornerAcc.operator(), None);
        assert_eq!(AllocationType::BoolCheck.operator(), None);
        assert_eq!(AllocationType::MulAdd.operator(), None);
        assert_eq!(AllocationType::WitnessHint.operator(), None);
        assert_eq!(
            AllocationType::NonPrimitiveOp(NpoTypeId::new("x")).operator(),
            None
        );
    }

    #[test]
    fn display_binary_op_with_two_deps() {
        let e = entry(
            5,
            AllocationType::Add,
            "",
            vec![vec![ExprId(1)], vec![ExprId(2)]],
            None,
        );
        assert_eq!(e.to_string(), "expr_5 = expr_1 + expr_2");
    }

    #[test]
    fn display_binary_op_without_two_deps_falls_back() {
        let e = entry(5, AllocationType::Mul, "", vec![vec![ExprId(1)]], None);
        assert_eq!(e.to_string(), "expr_5 (Mul)");
    }

    #[test]
    fn display_non_primitive_op_with_deps() {
        let e = entry(
            10,
            AllocationType::NonPrimitiveOp(NpoTypeId::new("hash")),
            "",
            vec![vec![ExprId(1), ExprId(2)], vec![ExprId(3)]],
            None,
        );
        assert_eq!(
            e.to_string(),
            "NonPrimitiveOp(NpoTypeId(hash)) (inputs: [expr_1, expr_2, expr_3])"
        );
    }

    #[test]
    fn display_non_primitive_op_without_deps() {
        let e = entry(
            10,
            AllocationType::NonPrimitiveOp(NpoTypeId::new("hint")),
            "",
            vec![],
            None,
        );
        assert_eq!(e.to_string(), "NonPrimitiveOp(NpoTypeId(hint))");
    }

    #[test]
    fn display_leaf_types() {
        assert_eq!(
            entry(0, AllocationType::Public, "", vec![], None).to_string(),
            "expr_0 (Public)"
        );
        assert_eq!(
            entry(3, AllocationType::Const, "", vec![], None).to_string(),
            "expr_3 (Const)"
        );
        assert_eq!(
            entry(7, AllocationType::HornerAcc, "", vec![], None).to_string(),
            "expr_7 (HornerAcc)"
        );
        assert_eq!(
            entry(8, AllocationType::BoolCheck, "", vec![], None).to_string(),
            "expr_8 (BoolCheck)"
        );
        assert_eq!(
            entry(9, AllocationType::WitnessHint, "", vec![], None).to_string(),
            "expr_9 (WitnessHint)"
        );
    }

    #[test]
    fn log_default_is_empty() {
        let log = AllocationLog::default();
        assert!(log.is_empty());
        assert_eq!(log.len(), 0);
    }

    #[test]
    fn log_push_and_deref() {
        let mut log = AllocationLog::default();
        log.push(entry(1, AllocationType::Public, "a", vec![], None));
        log.push(entry(2, AllocationType::Const, "", vec![], None));
        assert_eq!(log.len(), 2);
        assert_eq!(log[0].expr_id, ExprId(1));
        assert_eq!(log[1].expr_id, ExprId(2));
    }

    #[test]
    fn log_scopes_empty() {
        assert!(AllocationLog::default().scopes().is_empty());
    }

    #[test]
    fn log_scopes_sorted_and_deduplicated() {
        let mut log = AllocationLog::default();
        log.push(entry(1, AllocationType::Add, "", vec![], Some("beta")));
        log.push(entry(2, AllocationType::Add, "", vec![], Some("alpha")));
        log.push(entry(3, AllocationType::Add, "", vec![], Some("beta")));
        log.push(entry(4, AllocationType::Add, "", vec![], None));
        assert_eq!(log.scopes(), vec!["alpha", "beta"]);
    }

    #[test]
    fn log_dump_does_not_panic_on_empty() {
        AllocationLog::default().dump();
    }

    #[test]
    fn log_dump_scope_does_not_panic_on_empty_scope() {
        let mut log = AllocationLog::default();
        log.push(entry(1, AllocationType::Const, "", vec![], Some("other")));
        log.dump_scope(Some("nonexistent"));
    }

    #[test]
    fn log_dump_expr_ids_does_not_panic() {
        let mut log = AllocationLog::default();
        log.push(entry(1, AllocationType::Const, "c", vec![], None));
        log.dump_expr_ids(&[ExprId(1), ExprId(999)]);
    }

    #[test]
    fn realistic_circuit_log() {
        let mut log = AllocationLog::default();

        // Public inputs (scoped)
        let a = ExprId(1);
        let b = ExprId(2);
        let c = ExprId(3);
        log.push(entry(
            1,
            AllocationType::Public,
            "input_a",
            vec![],
            Some("inputs"),
        ));
        log.push(entry(
            2,
            AllocationType::Public,
            "input_b",
            vec![],
            Some("inputs"),
        ));
        log.push(entry(
            3,
            AllocationType::Public,
            "input_c",
            vec![],
            Some("inputs"),
        ));

        // Constant (unscoped)
        let two = ExprId(4);
        log.push(entry(4, AllocationType::Const, "2", vec![], None));

        // Arithmetic (scoped under "arithmetic")
        let bc = ExprId(5);
        log.push(entry(
            5,
            AllocationType::Mul,
            "b_times_c",
            vec![vec![b], vec![c]],
            Some("arithmetic"),
        ));
        let sum = ExprId(6);
        log.push(entry(
            6,
            AllocationType::Add,
            "a_plus_bc",
            vec![vec![a], vec![bc]],
            Some("arithmetic"),
        ));
        let diff = ExprId(7);
        log.push(entry(
            7,
            AllocationType::Sub,
            "a_minus_bc",
            vec![vec![a], vec![bc]],
            Some("arithmetic"),
        ));
        let product = ExprId(8);
        log.push(entry(
            8,
            AllocationType::Mul,
            "sum_times_diff",
            vec![vec![sum], vec![diff]],
            Some("arithmetic"),
        ));
        log.push(entry(
            9,
            AllocationType::Div,
            "final_result",
            vec![vec![product], vec![two]],
            Some("arithmetic"),
        ));

        // Structural queries
        assert_eq!(log.len(), 9);
        assert_eq!(log.scopes(), vec!["arithmetic", "inputs"]);

        // Display for each binary op variant
        assert_eq!(log[4].to_string(), "expr_5 = expr_2 * expr_3");
        assert_eq!(log[5].to_string(), "expr_6 = expr_1 + expr_5");
        assert_eq!(log[6].to_string(), "expr_7 = expr_1 - expr_5");
        assert_eq!(log[7].to_string(), "expr_8 = expr_6 * expr_7");
        assert_eq!(log[8].to_string(), "expr_9 = expr_8 / expr_4");

        // Display for leaf types
        assert_eq!(log[0].to_string(), "expr_1 (Public)");
        assert_eq!(log[3].to_string(), "expr_4 (Const)");

        // dump / dump_scope / dump_expr_ids must not panic
        log.dump();
        log.dump_scope(Some("inputs"));
        log.dump_scope(Some("arithmetic"));
        log.dump_scope(None);
        log.dump_expr_ids(&[ExprId(5), ExprId(9), ExprId(42)]);
    }
}
