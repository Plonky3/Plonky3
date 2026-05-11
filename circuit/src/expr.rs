use alloc::format;
use alloc::string::ToString;
use alloc::vec::Vec;

use hashbrown::HashMap;

use crate::builder::CircuitBuilderError;
use crate::types::{ExprId, NonPrimitiveOpId};

/// Expression DAG for field operations
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Expr<F> {
    /// Constant field element
    Const(F),
    /// Public input at declaration position
    Public(usize),
    /// Private input that must be constrained by downstream use.
    PrivateInput(usize),
    /// Addition of two expressions
    Add { lhs: ExprId, rhs: ExprId },
    /// Subtraction of two expressions
    Sub { lhs: ExprId, rhs: ExprId },
    /// Multiplication of two expressions
    Mul { lhs: ExprId, rhs: ExprId },
    /// Division of two expressions
    Div { lhs: ExprId, rhs: ExprId },
    /// Horner accumulator step: result = acc * alpha + p_at_z - p_at_x
    ///
    /// Emits a single HornerAcc ALU op with no intermediate witnesses.
    HornerAcc {
        acc: ExprId,
        alpha: ExprId,
        p_at_z: ExprId,
        p_at_x: ExprId,
    },
    /// Boolean check: asserts val ∈ {0, 1}.
    ///
    /// Emits a single BoolCheck ALU op with no intermediate witnesses.
    BoolCheck { val: ExprId },
    /// Fused multiply-add: result = a * b + c.
    ///
    /// Emits a single MulAdd ALU op with no intermediate witnesses.
    MulAdd { a: ExprId, b: ExprId, c: ExprId },
    /// Anchor node for a non-primitive operation in the expression DAG.
    ///
    /// This node has no witness value itself, but it fixes the relative execution order
    /// of non-primitive ops w.r.t. other expressions during lowering.
    ///
    /// The `inputs` field contains all input expressions (flattened from witness_exprs),
    /// making dependencies explicit in the DAG structure. This enables proper topological
    /// analysis and ensures the lowerer emits ops after their inputs are available.
    ///
    /// For stateful ops (e.g., Poseidon2 perm chaining with `in_ctl=false`), `inputs` may
    /// be empty since chained values flow internally and are not materialized in the
    /// witness table. Execution order for such ops is determined by their position in
    /// the ops list during lowering.
    NonPrimitiveCall {
        op_id: NonPrimitiveOpId,
        inputs: Vec<ExprId>,
    },
    /// Output of a non-primitive operation.
    ///
    /// This node represents a value produced by a non-primitive op. The `call` field
    /// points to the `NonPrimitiveCall` expression node, making the dependency explicit
    /// in the DAG structure. `output_idx` selects which output of that op this refers to.
    NonPrimitiveOutput { call: ExprId, output_idx: u32 },
}

/// Graph for storing expression DAG nodes
#[derive(Debug, Clone, Default)]
pub struct ExpressionGraph<F> {
    nodes: Vec<Expr<F>>,
}

impl<F> ExpressionGraph<F> {
    pub const fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    /// Add an expression to the graph, returning its ID
    pub fn add_expr(&mut self, expr: Expr<F>) -> ExprId {
        let id = ExprId(self.nodes.len() as u32);
        self.nodes.push(expr);
        id
    }

    /// Get an expression by ID
    pub fn get_expr(&self, id: ExprId) -> &Expr<F> {
        &self.nodes[id.0 as usize]
    }

    /// Get all nodes in the graph
    pub fn nodes(&self) -> &[Expr<F>] {
        &self.nodes
    }

    /// Build a map from `NonPrimitiveOpId` to its sorted, validated output expressions.
    pub fn build_npo_output_map(
        &self,
    ) -> Result<HashMap<NonPrimitiveOpId, Vec<(u32, ExprId)>>, CircuitBuilderError> {
        let mut map: HashMap<NonPrimitiveOpId, Vec<(u32, ExprId)>> = HashMap::new();

        // Scan every node, collecting output nodes grouped by their parent operation.
        for (expr_idx, expr) in self.nodes.iter().enumerate() {
            // Skip non-output nodes.
            if let Expr::NonPrimitiveOutput { call, output_idx } = expr {
                // Validate that the referenced call node is actually a call.
                if let Expr::NonPrimitiveCall { op_id, .. } = self.get_expr(*call) {
                    // Record this output under its parent operation.
                    map.entry(*op_id)
                        .or_default()
                        .push((*output_idx, ExprId(expr_idx as u32)));
                } else {
                    return Err(CircuitBuilderError::MissingExprMapping {
                        expr_id: *call,
                        context: "NonPrimitiveOutput.call must reference a NonPrimitiveCall"
                            .to_string(),
                    });
                }
            }
        }

        // Sort and validate in a single pass
        for (&op_id, outputs) in &mut map {
            // Sort each operation's outputs by index for deterministic ordering.
            outputs.sort_unstable_by_key(|&(idx, _)| idx);

            // After sorting, a contiguous 0..N range means output_idx[pos] == pos for all pos.
            //
            // This single check catches both gaps and duplicates.
            if let Some((pos, &(bad_idx, _))) = outputs
                .iter()
                .enumerate()
                .find(|&(pos, &(idx, _))| idx != pos as u32)
            {
                return Err(CircuitBuilderError::MalformedNonPrimitiveOutputs {
                    op_id,
                    details: format!("expected contiguous output_idx {pos}, got {bad_idx}"),
                });
            }
        }

        Ok(map)
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;

    // Mock extension field element for testing
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    struct MockExtField(u64);

    #[test]
    fn test_expression_graph() {
        let mut graph = ExpressionGraph::<MockExtField>::new();

        let const_expr = Expr::Const(MockExtField(42));
        let public_expr = Expr::Public(0);

        let const_id = graph.add_expr(const_expr.clone());
        let public_id = graph.add_expr(public_expr.clone());

        assert_eq!(const_id, ExprId::ZERO);
        assert_eq!(public_id, ExprId(1));

        assert_eq!(graph.get_expr(const_id), &const_expr);
        assert_eq!(graph.get_expr(public_id), &public_expr);

        let add_expr = Expr::Add {
            lhs: const_id,
            rhs: public_id,
        };
        let add_id = graph.add_expr(add_expr.clone());
        assert_eq!(add_id, ExprId(2));
        assert_eq!(graph.get_expr(add_id), &add_expr);
    }

    #[test]
    fn npo_output_map_empty_graph() {
        // An empty graph has no output nodes, so the map should be empty.
        let graph = ExpressionGraph::<MockExtField>::new();
        let map = graph.build_npo_output_map().unwrap();
        assert!(map.is_empty());
    }

    #[test]
    fn npo_output_map_no_npo_nodes() {
        // A graph with only constants and publics has no output nodes.
        let mut graph = ExpressionGraph::<MockExtField>::new();
        graph.add_expr(Expr::Const(MockExtField(1)));
        graph.add_expr(Expr::Public(0));
        let map = graph.build_npo_output_map().unwrap();
        assert!(map.is_empty());
    }

    #[test]
    fn npo_output_map_valid_contiguous() {
        let mut graph = ExpressionGraph::<MockExtField>::new();
        // Create a call node with no inputs.
        let call = graph.add_expr(Expr::NonPrimitiveCall {
            op_id: NonPrimitiveOpId(0),
            inputs: Vec::new(),
        });
        // Insert outputs out of order to exercise the sorting logic.
        let out1 = graph.add_expr(Expr::NonPrimitiveOutput {
            call,
            output_idx: 1,
        });
        let out0 = graph.add_expr(Expr::NonPrimitiveOutput {
            call,
            output_idx: 0,
        });
        let map = graph.build_npo_output_map().unwrap();
        let outputs = &map[&NonPrimitiveOpId(0)];
        // After sorting, index 0 should come first despite being added second.
        assert_eq!(outputs, &[(0, out0), (1, out1)]);
    }

    #[test]
    fn npo_output_map_gap_rejected() {
        use crate::builder::CircuitBuilderError;

        let mut graph = ExpressionGraph::<MockExtField>::new();
        let call = graph.add_expr(Expr::NonPrimitiveCall {
            op_id: NonPrimitiveOpId(0),
            inputs: Vec::new(),
        });
        // Create outputs at indices 0 and 2, leaving a gap at index 1.
        graph.add_expr(Expr::NonPrimitiveOutput {
            call,
            output_idx: 0,
        });
        graph.add_expr(Expr::NonPrimitiveOutput {
            call,
            output_idx: 2,
        });
        // The contiguity check should reject this with a descriptive error.
        match graph.build_npo_output_map() {
            Err(CircuitBuilderError::MalformedNonPrimitiveOutputs { op_id, details }) => {
                assert_eq!(op_id, NonPrimitiveOpId(0));
                assert!(details.contains("expected contiguous"));
            }
            other => panic!("expected MalformedNonPrimitiveOutputs, got {other:?}"),
        }
    }

    #[test]
    fn npo_output_map_duplicate_rejected() {
        use crate::builder::CircuitBuilderError;

        let mut graph = ExpressionGraph::<MockExtField>::new();
        let call = graph.add_expr(Expr::NonPrimitiveCall {
            op_id: NonPrimitiveOpId(0),
            inputs: Vec::new(),
        });
        // Two outputs both claiming index 0 — a duplicate.
        graph.add_expr(Expr::NonPrimitiveOutput {
            call,
            output_idx: 0,
        });
        graph.add_expr(Expr::NonPrimitiveOutput {
            call,
            output_idx: 0,
        });
        // Duplicates cause two entries at position 0, making position 1 mismatch.
        match graph.build_npo_output_map() {
            Err(CircuitBuilderError::MalformedNonPrimitiveOutputs { op_id, details }) => {
                assert_eq!(op_id, NonPrimitiveOpId(0));
                assert!(details.contains("expected contiguous"));
            }
            other => panic!("expected MalformedNonPrimitiveOutputs, got {other:?}"),
        }
    }

    #[test]
    fn npo_output_map_call_points_to_non_call() {
        use crate::builder::CircuitBuilderError;

        let mut graph = ExpressionGraph::<MockExtField>::new();
        // Create a constant node and an output that incorrectly references it as a call.
        let not_a_call = graph.add_expr(Expr::Const(MockExtField(42)));
        graph.add_expr(Expr::NonPrimitiveOutput {
            call: not_a_call,
            output_idx: 0,
        });
        // Should fail because the referenced node is not a call.
        match graph.build_npo_output_map() {
            Err(CircuitBuilderError::MissingExprMapping { expr_id, context }) => {
                assert_eq!(expr_id, not_a_call);
                assert!(context.contains("NonPrimitiveCall"));
            }
            other => panic!("expected MissingExprMapping, got {other:?}"),
        }
    }

    proptest! {
        #[test]
        fn expr_get_returns_added(vals in prop::collection::vec(any::<u64>().prop_map(MockExtField), 1..30)) {
            let mut graph = ExpressionGraph::<MockExtField>::new();
            let mut ids = Vec::new();

            for val in &vals {
                let expr = Expr::Const(*val);
                let id = graph.add_expr(expr.clone());
                ids.push(id);
            }

            for (id, val) in ids.iter().zip(vals.iter()) {
                let retrieved = graph.get_expr(*id);
                prop_assert_eq!(retrieved, &Expr::Const(*val), "get should return added expression");
            }
        }

        #[test]
        fn expr_primitive_ops(val1 in any::<u64>().prop_map(MockExtField), val2 in any::<u64>().prop_map(MockExtField)) {
            let mut graph = ExpressionGraph::<MockExtField>::new();

            let id1 = graph.add_expr(Expr::Const(val1));
            let id2 = graph.add_expr(Expr::Const(val2));

            let add_id = graph.add_expr(Expr::Add { lhs: id1, rhs: id2 });
            match graph.get_expr(add_id) {
                Expr::Add { lhs, rhs } => {
                    prop_assert_eq!(*lhs, id1);
                    prop_assert_eq!(*rhs, id2);
                }
                _ => prop_assert!(false, "expected Add expr"),
            }

            let sub_id = graph.add_expr(Expr::Sub { lhs: id1, rhs: id2 });
            match graph.get_expr(sub_id) {
                Expr::Sub { lhs, rhs } => {
                    prop_assert_eq!(*lhs, id1);
                    prop_assert_eq!(*rhs, id2);
                }
                _ => prop_assert!(false, "expected Sub expr"),
            }

            let mul_id = graph.add_expr(Expr::Mul { lhs: id1, rhs: id2 });
            match graph.get_expr(mul_id) {
                Expr::Mul { lhs, rhs } => {
                    prop_assert_eq!(*lhs, id1);
                    prop_assert_eq!(*rhs, id2);
                }
                _ => prop_assert!(false, "expected Mul expr"),
            }

            let div_id = graph.add_expr(Expr::Div { lhs: id1, rhs: id2 });
            match graph.get_expr(div_id) {
                Expr::Div { lhs, rhs } => {
                    prop_assert_eq!(*lhs, id1);
                    prop_assert_eq!(*rhs, id2);
                }
                _ => prop_assert!(false, "expected Div expr"),
            }
        }

        #[test]
        fn expr_public_positions(positions in prop::collection::vec(0usize..100, 0..20)) {
            let mut graph = ExpressionGraph::<MockExtField>::new();
            let mut ids = Vec::new();

            for &pos in &positions {
                let id = graph.add_expr(Expr::Public(pos));
                ids.push(id);
            }

            for (&id, &expected_pos) in ids.iter().zip(positions.iter()) {
                match graph.get_expr(id) {
                    Expr::Public(pos) => {
                        prop_assert_eq!(*pos, expected_pos, "public position should match");
                    }
                    _ => prop_assert!(false, "expected Public expr"),
                }
            }
        }

        #[test]
        fn expr_equality(val in any::<u64>().prop_map(MockExtField)) {
            let expr1 = Expr::Const(val);
            let expr2 = Expr::Const(val);
            let expr3 = Expr::Const(MockExtField(val.0 + 1));

            prop_assert_eq!(&expr1, &expr2, "same expressions should be equal");
            prop_assert_ne!(&expr1, &expr3, "different expressions should not be equal");
        }
    }
}
