//! Flattened wire format for [`SymbolicExpr`].
//!
//! An expression tree is an acyclic DAG whose interior nodes share `Arc`
//! children. Serializing the `Arc` pointers directly would expand each shared
//! sub-tree once per parent, so instead the tree is lowered to a topologically
//! ordered `Vec` of [`FlatNode`]s in which children are referenced by index.
//! Shared sub-trees collapse to a single node, giving the compact "list of
//! operations" encoding used as the stable on-wire form.

use alloc::collections::BTreeMap;
use alloc::sync::Arc;
use alloc::vec::Vec;

use serde::de::Error as _;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::symbolic::SymbolicExpr;

/// A single node in the flattened expression arena.
///
/// Interior nodes refer to their operands by their position in the node list.
/// Every operand index is strictly smaller than the node's own index, so the
/// list is a valid topological (post-order) ordering with the root last.
#[derive(Debug, Serialize, Deserialize)]
enum FlatNode<A> {
    Leaf(A),
    Add {
        x: usize,
        y: usize,
        degree_multiple: usize,
    },
    Sub {
        x: usize,
        y: usize,
        degree_multiple: usize,
    },
    Neg {
        x: usize,
        degree_multiple: usize,
    },
    Mul {
        x: usize,
        y: usize,
        degree_multiple: usize,
    },
}

/// Lower `node` into `nodes` in post-order, returning its index.
///
/// `seen` maps the address of an already-emitted node to its index so shared
/// `Arc` children are emitted exactly once. The whole tree is borrowed for the
/// duration, so addresses stay live and unique while flattening.
fn flatten_into<'a, A>(
    node: &'a SymbolicExpr<A>,
    nodes: &mut Vec<FlatNode<&'a A>>,
    seen: &mut BTreeMap<*const SymbolicExpr<A>, usize>,
) -> usize {
    let key: *const SymbolicExpr<A> = node;
    if let Some(&idx) = seen.get(&key) {
        return idx;
    }
    let flat = match node {
        SymbolicExpr::Leaf(a) => FlatNode::Leaf(a),
        SymbolicExpr::Add {
            x,
            y,
            degree_multiple,
        } => {
            let x = flatten_into(x, nodes, seen);
            let y = flatten_into(y, nodes, seen);
            FlatNode::Add {
                x,
                y,
                degree_multiple: *degree_multiple,
            }
        }
        SymbolicExpr::Sub {
            x,
            y,
            degree_multiple,
        } => {
            let x = flatten_into(x, nodes, seen);
            let y = flatten_into(y, nodes, seen);
            FlatNode::Sub {
                x,
                y,
                degree_multiple: *degree_multiple,
            }
        }
        SymbolicExpr::Neg { x, degree_multiple } => {
            let x = flatten_into(x, nodes, seen);
            FlatNode::Neg {
                x,
                degree_multiple: *degree_multiple,
            }
        }
        SymbolicExpr::Mul {
            x,
            y,
            degree_multiple,
        } => {
            let x = flatten_into(x, nodes, seen);
            let y = flatten_into(y, nodes, seen);
            FlatNode::Mul {
                x,
                y,
                degree_multiple: *degree_multiple,
            }
        }
    };
    let idx = nodes.len();
    nodes.push(flat);
    seen.insert(key, idx);
    idx
}

/// Rebuild a tree from a flattened arena, sharing `Arc`s for repeated indices.
///
/// Returns `None` if the list is empty, an operand index is not strictly
/// smaller than its node's index (a forward or self reference), or the root is
/// somehow still shared after construction.
fn unflatten<A>(nodes: Vec<FlatNode<A>>) -> Option<SymbolicExpr<A>> {
    let mut built: Vec<Arc<SymbolicExpr<A>>> = Vec::with_capacity(nodes.len());
    for (i, flat) in nodes.into_iter().enumerate() {
        let expr = match flat {
            FlatNode::Leaf(a) => SymbolicExpr::Leaf(a),
            FlatNode::Add {
                x,
                y,
                degree_multiple,
            } => {
                if x >= i || y >= i {
                    return None;
                }
                SymbolicExpr::Add {
                    x: built[x].clone(),
                    y: built[y].clone(),
                    degree_multiple,
                }
            }
            FlatNode::Sub {
                x,
                y,
                degree_multiple,
            } => {
                if x >= i || y >= i {
                    return None;
                }
                SymbolicExpr::Sub {
                    x: built[x].clone(),
                    y: built[y].clone(),
                    degree_multiple,
                }
            }
            FlatNode::Neg { x, degree_multiple } => {
                if x >= i {
                    return None;
                }
                SymbolicExpr::Neg {
                    x: built[x].clone(),
                    degree_multiple,
                }
            }
            FlatNode::Mul {
                x,
                y,
                degree_multiple,
            } => {
                if x >= i || y >= i {
                    return None;
                }
                SymbolicExpr::Mul {
                    x: built[x].clone(),
                    y: built[y].clone(),
                    degree_multiple,
                }
            }
        };
        built.push(Arc::new(expr));
    }
    // The root is the last node emitted; no other node references it, so it is
    // uniquely owned and can be unwrapped without cloning.
    Arc::try_unwrap(built.pop()?).ok()
}

impl<A: Serialize> Serialize for SymbolicExpr<A> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut nodes: Vec<FlatNode<&A>> = Vec::new();
        let mut seen: BTreeMap<*const Self, usize> = BTreeMap::new();
        flatten_into(self, &mut nodes, &mut seen);
        nodes.serialize(serializer)
    }
}

impl<'de, A: Deserialize<'de>> Deserialize<'de> for SymbolicExpr<A> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let nodes = Vec::<FlatNode<A>>::deserialize(deserializer)?;
        unflatten(nodes).ok_or_else(|| D::Error::custom("invalid flattened symbolic expression"))
    }
}

#[cfg(test)]
mod tests {
    use alloc::format;
    use alloc::sync::Arc;

    use p3_baby_bear::BabyBear;

    use crate::symbolic::expression::BaseLeaf;
    use crate::symbolic::variable::BaseEntry;
    use crate::symbolic::{SymbolicExpr, SymbolicExpression, SymbolicVariable};

    type F = BabyBear;

    fn var(index: usize) -> SymbolicExpression<F> {
        SymbolicExpression::from(SymbolicVariable::new(BaseEntry::Main { offset: 0 }, index))
    }

    #[test]
    fn json_round_trip_preserves_structure() {
        // A non-trivial tree: (x0 * x1) - x0 + 7, exercising every node kind.
        let expr = var(0) * var(1) - var(0) + SymbolicExpression::from(F::new(7));

        let json = serde_json::to_string(&expr).unwrap();
        let decoded: SymbolicExpression<F> = serde_json::from_str(&json).unwrap();

        // Re-serializing the decoded tree must reproduce the original wire form.
        assert_eq!(serde_json::to_string(&decoded).unwrap(), json);
        assert_eq!(format!("{decoded:?}"), format!("{expr:?}"));
    }

    #[test]
    fn shared_subtree_is_emitted_once() {
        // Build a node that references the same Arc child twice.
        let shared = Arc::new(var(0));
        let expr = SymbolicExpr::Add {
            x: shared.clone(),
            y: shared,
            degree_multiple: 1,
        };

        // The flattened form is a JSON array; with deduplication the shared leaf
        // contributes a single node, so the arena holds exactly two nodes.
        let value: serde_json::Value = serde_json::to_value(&expr).unwrap();
        assert_eq!(value.as_array().unwrap().len(), 2);

        let decoded: SymbolicExpression<F> = serde_json::from_value(value).unwrap();
        assert_eq!(format!("{decoded:?}"), format!("{expr:?}"));
    }

    #[test]
    fn rejects_forward_reference() {
        // A single Add node whose operands point past itself is not a valid arena.
        let json = r#"[{"Add":{"x":1,"y":2,"degree_multiple":1}}]"#;
        let decoded: Result<SymbolicExpression<F>, _> = serde_json::from_str(json);
        assert!(decoded.is_err());
    }

    #[test]
    fn rejects_empty_arena() {
        let decoded: Result<SymbolicExpression<F>, _> = serde_json::from_str("[]");
        assert!(decoded.is_err());
    }

    #[test]
    fn leaf_round_trips() {
        let expr = SymbolicExpression::<F>::Leaf(BaseLeaf::Constant(F::new(42)));
        let json = serde_json::to_string(&expr).unwrap();
        let decoded: SymbolicExpression<F> = serde_json::from_str(&json).unwrap();
        assert_eq!(format!("{decoded:?}"), format!("{expr:?}"));
    }
}
