//! Flattened DAG types for symbolic expression trees.

use alloc::sync::Arc;
use alloc::vec::Vec;

use super::SymbolicExpr;

/// A single node in a flattened expression DAG.
///
/// Arithmetic variants reference children by position index
/// rather than heap-allocated pointers.
///
/// # Invariant
///
/// All child indices must be strictly less than this node's
/// own position in the containing vector.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SymbolicExprNode<A> {
    /// An atomic value: variable reference, field constant, or selector flag.
    Leaf(A),

    /// Sum of two sub-expressions.
    Add {
        /// Index of the left operand in the DAG node vector.
        left: usize,
        /// Index of the right operand in the DAG node vector.
        right: usize,
        /// Cached constraint degree: max of the two operand degrees.
        degree_multiple: usize,
    },

    /// Difference of two sub-expressions.
    Sub {
        /// Index of the left operand in the DAG node vector.
        left: usize,
        /// Index of the right operand in the DAG node vector.
        right: usize,
        /// Cached constraint degree: max of the two operand degrees.
        degree_multiple: usize,
    },

    /// Additive inverse of a sub-expression.
    Neg {
        /// Index of the operand in the DAG node vector.
        idx: usize,
        /// Cached constraint degree: same as the operand's degree.
        degree_multiple: usize,
    },

    /// Product of two sub-expressions.
    Mul {
        /// Index of the left operand in the DAG node vector.
        left: usize,
        /// Index of the right operand in the DAG node vector.
        right: usize,
        /// Cached constraint degree: sum of the two operand degrees.
        degree_multiple: usize,
    },
}

/// A topologically-sorted DAG of symbolic constraint expressions.
///
/// Built by flattening one or more expression trees.
/// Deduplication is by pointer identity, not structural equality:
/// - Shared sub-trees are stored once.
/// - Independent but structurally identical sub-trees are kept separate.
///
/// # Invariant
///
/// Every node at position `i` only references positions `< i`.
/// A single forward pass is sufficient to evaluate or reconstruct the entire graph.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SymbolicExprDag<A> {
    /// Nodes in dependency order (children always precede parents).
    pub nodes: Vec<SymbolicExprNode<A>>,

    /// Position of each constraint's root node inside the node vector.
    ///
    /// Preserves the input ordering from the slice passed to the constructor.
    pub constraint_idx: Vec<usize>,
}

impl<A: Clone> SymbolicExprDag<A> {
    /// Rebuild expression trees from this DAG.
    ///
    /// Returns one tree per constraint root, in the same order as the original input slice.
    /// Shared nodes in the DAG produce shared sub-trees in the output.
    #[must_use]
    pub fn to_expressions(&self) -> Vec<SymbolicExpr<A>> {
        // Rebuild every DAG node into a reference-counted tree.
        let all = self.reconstruct_all();

        // Collect only the constraint root trees, unwrapping the outer pointer.
        self.constraint_idx
            .iter()
            .map(|&idx| all[idx].as_ref().clone())
            .collect()
    }

    /// Total number of unique nodes in the DAG.
    #[inline]
    #[must_use]
    pub const fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Rebuild every node into a shared tree node.
    ///
    /// Walking in topological order so that children are available before parents.
    fn reconstruct_all(&self) -> Vec<Arc<SymbolicExpr<A>>> {
        // Pre-allocate one slot per DAG node.
        let mut exprs: Vec<Arc<SymbolicExpr<A>>> = Vec::with_capacity(self.nodes.len());

        // Forward pass: topological order guarantees all children are built
        // before any parent that references them.
        for node in &self.nodes {
            let expr = match node {
                // Leaves translate directly with no child references.
                SymbolicExprNode::Leaf(a) => SymbolicExpr::Leaf(a.clone()),
                // Binary and unary nodes clone child pointers from the
                // already-built prefix of the vector.
                SymbolicExprNode::Add {
                    left,
                    right,
                    degree_multiple,
                } => SymbolicExpr::Add {
                    x: Arc::clone(&exprs[*left]),
                    y: Arc::clone(&exprs[*right]),
                    degree_multiple: *degree_multiple,
                },
                SymbolicExprNode::Sub {
                    left,
                    right,
                    degree_multiple,
                } => SymbolicExpr::Sub {
                    x: Arc::clone(&exprs[*left]),
                    y: Arc::clone(&exprs[*right]),
                    degree_multiple: *degree_multiple,
                },
                SymbolicExprNode::Neg {
                    idx,
                    degree_multiple,
                } => SymbolicExpr::Neg {
                    x: Arc::clone(&exprs[*idx]),
                    degree_multiple: *degree_multiple,
                },
                SymbolicExprNode::Mul {
                    left,
                    right,
                    degree_multiple,
                } => SymbolicExpr::Mul {
                    x: Arc::clone(&exprs[*left]),
                    y: Arc::clone(&exprs[*right]),
                    degree_multiple: *degree_multiple,
                },
            };
            // Wrap the tree node and append to maintain index correspondence.
            exprs.push(Arc::new(expr));
        }

        exprs
    }
}

impl<A: Clone> From<&[SymbolicExpr<A>]> for SymbolicExprDag<A> {
    fn from(constraints: &[SymbolicExpr<A>]) -> Self {
        SymbolicExpr::flatten_to_dag(constraints)
    }
}

#[cfg(test)]
mod tests {
    use alloc::sync::Arc;
    use alloc::vec;
    use alloc::vec::Vec;

    use p3_baby_bear::BabyBear;
    use p3_field::{PrimeCharacteristicRing, PrimeField32};
    use proptest::prelude::*;

    use super::*;
    use crate::symbolic::expression::BaseLeaf;
    use crate::symbolic::variable::{BaseEntry, SymbolicVariable};
    use crate::symbolic::{SymbolicExpr, SymbolicExpression};

    type F = BabyBear;

    /// Build a symbolic variable leaf referencing a main trace column.
    fn var(offset: usize, index: usize) -> SymbolicExpression<F> {
        SymbolicExpr::Leaf(BaseLeaf::Variable(SymbolicVariable::new(
            BaseEntry::Main { offset },
            index,
        )))
    }

    /// Build a symbolic constant leaf from a raw field value.
    fn constant(val: u32) -> SymbolicExpression<F> {
        SymbolicExpr::Leaf(BaseLeaf::Constant(F::new(val)))
    }

    /// Assert that every node at index `i` only references indices `< i`.
    fn assert_topological_order<A>(dag: &SymbolicExprDag<A>) {
        for (i, node) in dag.nodes.iter().enumerate() {
            // Check that all child indices precede this node's position.
            let valid = match node {
                SymbolicExprNode::Leaf(_) => true,
                SymbolicExprNode::Add { left, right, .. }
                | SymbolicExprNode::Sub { left, right, .. }
                | SymbolicExprNode::Mul { left, right, .. } => *left < i && *right < i,
                SymbolicExprNode::Neg { idx, .. } => *idx < i,
            };
            assert!(valid, "node {i} violates topological order");
        }
    }

    /// Evaluate a symbolic expression on concrete field values.
    ///
    /// # Arguments
    ///
    /// - `var_values`: concrete values for main trace variables at offset 0.
    /// - `is_first_row`, `is_last_row`, `is_transition`: selector values.
    fn eval_expr(
        expr: &SymbolicExpression<F>,
        var_values: &[F],
        is_first_row: F,
        is_last_row: F,
        is_transition: F,
    ) -> F {
        match expr {
            // Return the embedded field constant directly.
            SymbolicExpr::Leaf(BaseLeaf::Constant(c)) => *c,
            // Look up the variable value by column index.
            SymbolicExpr::Leaf(BaseLeaf::Variable(v)) => match v.entry {
                BaseEntry::Main { offset: 0 } => var_values[v.index],
                _ => F::ZERO,
            },
            // Return the corresponding selector value.
            SymbolicExpr::Leaf(BaseLeaf::IsFirstRow) => is_first_row,
            SymbolicExpr::Leaf(BaseLeaf::IsLastRow) => is_last_row,
            SymbolicExpr::Leaf(BaseLeaf::IsTransition) => is_transition,
            // Recursively evaluate arithmetic operations.
            SymbolicExpr::Add { x, y, .. } => {
                eval_expr(x, var_values, is_first_row, is_last_row, is_transition)
                    + eval_expr(y, var_values, is_first_row, is_last_row, is_transition)
            }
            SymbolicExpr::Sub { x, y, .. } => {
                eval_expr(x, var_values, is_first_row, is_last_row, is_transition)
                    - eval_expr(y, var_values, is_first_row, is_last_row, is_transition)
            }
            SymbolicExpr::Neg { x, .. } => {
                -eval_expr(x, var_values, is_first_row, is_last_row, is_transition)
            }
            SymbolicExpr::Mul { x, y, .. } => {
                eval_expr(x, var_values, is_first_row, is_last_row, is_transition)
                    * eval_expr(y, var_values, is_first_row, is_last_row, is_transition)
            }
        }
    }

    /// Count the total number of nodes in a tree, following shared pointers each time.
    fn tree_node_count<A>(expr: &SymbolicExpr<A>) -> usize {
        match expr {
            SymbolicExpr::Leaf(_) => 1,
            SymbolicExpr::Neg { x, .. } => 1 + tree_node_count(x),
            SymbolicExpr::Add { x, y, .. }
            | SymbolicExpr::Sub { x, y, .. }
            | SymbolicExpr::Mul { x, y, .. } => 1 + tree_node_count(x) + tree_node_count(y),
        }
    }

    /// Number of distinct main trace variables available to generated expressions.
    const NUM_VARS: usize = 4;

    /// Generate a random field element uniformly in `[0, P)`.
    fn arb_field_element() -> impl Strategy<Value = F> {
        (0u32..F::ORDER_U32).prop_map(F::new)
    }

    /// Generate a random symbolic expression tree of bounded depth.
    ///
    /// Leaves include main trace variables, random constants, and all
    /// three selector types. Inner nodes use the four arithmetic operations.
    fn arb_expr(max_depth: u32) -> impl Strategy<Value = SymbolicExpression<F>> {
        // Base case: choose uniformly among the five leaf kinds.
        let leaf = prop_oneof![
            (0..NUM_VARS).prop_map(|i| var(0, i)),
            arb_field_element().prop_map(|c| SymbolicExpr::Leaf(BaseLeaf::Constant(c))),
            Just(SymbolicExpr::Leaf(BaseLeaf::IsFirstRow)),
            Just(SymbolicExpr::Leaf(BaseLeaf::IsLastRow)),
            Just(SymbolicExpr::Leaf(BaseLeaf::IsTransition)),
        ];

        // Recursive case: combine two sub-expressions with an arithmetic op.
        leaf.prop_recursive(max_depth, 64, 2, |inner| {
            prop_oneof![
                (inner.clone(), inner.clone()).prop_map(|(a, b)| a + b),
                (inner.clone(), inner.clone()).prop_map(|(a, b)| a - b),
                inner.clone().prop_map(|a| -a),
                (inner.clone(), inner).prop_map(|(a, b)| a * b),
            ]
        })
    }

    /// Generate a random evaluation context: variable values and selector values.
    fn arb_eval_context() -> impl Strategy<Value = (Vec<F>, F, F, F)> {
        (
            proptest::collection::vec(arb_field_element(), NUM_VARS),
            arb_field_element(),
            arb_field_element(),
            arb_field_element(),
        )
    }

    #[test]
    fn empty_constraints() {
        // Flatten an empty slice.
        let dag = SymbolicExpr::<BaseLeaf<F>>::flatten_to_dag(&[]);

        // The DAG should contain no nodes and no constraint roots.
        assert_eq!(dag.node_count(), 0);
        assert!(dag.constraint_idx.is_empty());
        // Round-trip should also produce an empty list.
        assert!(dag.to_expressions().is_empty());
    }

    #[test]
    fn arc_sharing_produces_single_node() {
        // Build e = var(0,0) + const(3), then square it via pointer sharing.
        let e_arc = Arc::new(var(0, 0) + constant(3));
        let sq = SymbolicExpr::Mul {
            x: Arc::clone(&e_arc),
            y: Arc::clone(&e_arc),
            degree_multiple: 2,
        };

        // The shared sub-expression should appear exactly once in the DAG.
        // The product node references the same index for both operands.
        assert_eq!(
            SymbolicExpr::flatten_to_dag(&[sq]),
            SymbolicExprDag {
                nodes: vec![
                    SymbolicExprNode::Leaf(BaseLeaf::Variable(SymbolicVariable::new(
                        BaseEntry::Main { offset: 0 },
                        0,
                    ))),
                    SymbolicExprNode::Leaf(BaseLeaf::Constant(F::new(3))),
                    SymbolicExprNode::Add {
                        left: 0,
                        right: 1,
                        degree_multiple: 1,
                    },
                    // Both operands point to index 2 — deduplication in action.
                    SymbolicExprNode::Mul {
                        left: 2,
                        right: 2,
                        degree_multiple: 2,
                    },
                ],
                constraint_idx: vec![3],
            }
        );
    }

    #[test]
    fn cross_constraint_sharing() {
        // Two constraints that share a sub-expression via pointer sharing.
        let shared = Arc::new(var(0, 0) * var(0, 1));
        let c1 = SymbolicExpr::Add {
            x: Arc::clone(&shared),
            y: Arc::new(constant(1)),
            degree_multiple: 2,
        };
        let c2 = SymbolicExpr::Sub {
            x: Arc::clone(&shared),
            y: Arc::new(constant(2)),
            degree_multiple: 2,
        };

        let dag = SymbolicExpr::flatten_to_dag(&[c1, c2]);

        // The shared product node at index 2 is referenced by both constraint roots.
        assert_eq!(
            dag,
            SymbolicExprDag {
                nodes: vec![
                    SymbolicExprNode::Leaf(BaseLeaf::Variable(SymbolicVariable::new(
                        BaseEntry::Main { offset: 0 },
                        0,
                    ))),
                    SymbolicExprNode::Leaf(BaseLeaf::Variable(SymbolicVariable::new(
                        BaseEntry::Main { offset: 0 },
                        1,
                    ))),
                    // Shared product — appears once, referenced by both constraints.
                    SymbolicExprNode::Mul {
                        left: 0,
                        right: 1,
                        degree_multiple: 2,
                    },
                    SymbolicExprNode::Leaf(BaseLeaf::Constant(F::new(1))),
                    // First constraint root: shared + const(1).
                    SymbolicExprNode::Add {
                        left: 2,
                        right: 3,
                        degree_multiple: 2,
                    },
                    SymbolicExprNode::Leaf(BaseLeaf::Constant(F::new(2))),
                    // Second constraint root: shared - const(2).
                    SymbolicExprNode::Sub {
                        left: 2,
                        right: 5,
                        degree_multiple: 2,
                    },
                ],
                constraint_idx: vec![4, 6],
            }
        );
    }

    #[test]
    fn distinct_arcs_are_not_deduplicated() {
        // Build two structurally identical but independently allocated expressions.
        let e1 = var(0, 0) + constant(1);
        let e2 = var(0, 0) + constant(1);

        let dag = SymbolicExpr::flatten_to_dag(&[e1, e2]);

        // Different allocations must produce separate root nodes.
        assert_ne!(
            dag.constraint_idx[0], dag.constraint_idx[1],
            "independent allocations must produce separate nodes"
        );
    }

    #[test]
    fn from_slice_trait() {
        // Verify the convenience conversion trait produces the same result.
        let constraints = [var(0, 0) + var(0, 1)];
        let dag = SymbolicExprDag::from(constraints.as_slice());
        // Two leaves + one addition = 3 nodes.
        assert_eq!(dag.node_count(), 3);
    }

    proptest! {
        #[test]
        fn round_trip_preserves_evaluation(
            expr in arb_expr(4),
            (var_values, is_first, is_last, is_trans) in arb_eval_context(),
        ) {
            // Flatten the expression tree into a DAG and reconstruct it.
            let dag = SymbolicExpr::flatten_to_dag(&[expr.clone()]);
            let reconstructed = dag.to_expressions();

            // Evaluate both the original and reconstructed trees on the same inputs.
            let original_val = eval_expr(&expr, &var_values, is_first, is_last, is_trans);
            let reconstructed_val = eval_expr(&reconstructed[0], &var_values, is_first, is_last, is_trans);

            // The round trip must preserve the evaluation result.
            prop_assert_eq!(original_val, reconstructed_val);
        }

        #[test]
        fn topological_order_always_holds(expr in arb_expr(5)) {
            // Flatten and verify the structural invariant.
            let dag = SymbolicExpr::flatten_to_dag(&[expr]);
            assert_topological_order(&dag);
        }

        #[test]
        fn dag_never_larger_than_tree(expr in arb_expr(4)) {
            // Count nodes via tree traversal (follows sharing, so may double-count).
            let tree_count = tree_node_count(&expr);
            let dag = SymbolicExpr::flatten_to_dag(&[expr]);
            // The DAG deduplicates, so it can only be equal or smaller.
            prop_assert!(dag.node_count() <= tree_count);
        }

        #[test]
        fn degree_preserved_through_round_trip(expr in arb_expr(4)) {
            // Record the original constraint degree.
            let original_degree = expr.degree_multiple();
            let dag = SymbolicExpr::flatten_to_dag(&[expr]);
            let reconstructed = dag.to_expressions();
            // The reconstructed expression must report the same degree.
            prop_assert_eq!(reconstructed[0].degree_multiple(), original_degree);
        }

        #[test]
        fn multi_constraint_round_trip(
            e1 in arb_expr(3),
            e2 in arb_expr(3),
            e3 in arb_expr(3),
            (var_values, is_first, is_last, is_trans) in arb_eval_context(),
        ) {
            // Flatten three constraints together into a single DAG.
            let constraints = vec![e1.clone(), e2.clone(), e3.clone()];
            let dag = SymbolicExpr::flatten_to_dag(&constraints);
            let reconstructed = dag.to_expressions();

            // The output must have the same number of constraint roots.
            prop_assert_eq!(reconstructed.len(), 3);

            // Each constraint must independently preserve its evaluation.
            for (orig, recon) in constraints.iter().zip(reconstructed.iter()) {
                let orig_val = eval_expr(orig, &var_values, is_first, is_last, is_trans);
                let recon_val = eval_expr(recon, &var_values, is_first, is_last, is_trans);
                prop_assert_eq!(orig_val, recon_val);
            }
        }

        #[test]
        fn single_leaf_produces_one_node(val in 0u32..F::ORDER_U32) {
            // A single constant leaf should flatten to exactly one DAG node.
            let expr = SymbolicExpr::Leaf(BaseLeaf::Constant(F::new(val)));
            let dag = SymbolicExpr::flatten_to_dag(&[expr]);

            prop_assert_eq!(dag.node_count(), 1);
            prop_assert_eq!(dag.constraint_idx.len(), 1);
            // The sole constraint root must be node 0.
            prop_assert_eq!(dag.constraint_idx[0], 0);
        }

        #[test]
        fn constraint_count_matches_input(
            exprs in proptest::collection::vec(arb_expr(2), 0..8),
        ) {
            let dag = SymbolicExpr::flatten_to_dag(&exprs);
            // Number of root indices must equal the number of input expressions.
            prop_assert_eq!(dag.constraint_idx.len(), exprs.len());
            // Round-trip must also produce the same count.
            prop_assert_eq!(dag.to_expressions().len(), exprs.len());
        }

        #[test]
        fn constraint_indices_in_bounds(expr in arb_expr(4)) {
            let dag = SymbolicExpr::flatten_to_dag(&[expr]);
            // Every root index must point to a valid node.
            for &idx in &dag.constraint_idx {
                prop_assert!(idx < dag.node_count());
            }
        }
    }
}
