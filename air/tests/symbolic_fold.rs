use std::sync::Arc;

use p3_air::{
    AirLayout, BaseEntry, BaseLeaf, DebugConstraintBuilder, ExtEntry, ExtLeaf,
    PermutationAirBuilder, SymbolicAirBuilder, SymbolicExpr, SymbolicExprVisitor,
    SymbolicExpression, SymbolicExpressionExt, SymbolicVariable, SymbolicVariableExt, WindowAccess,
    fold_symbolic_expressions, resolve_symbolic_expressions, resolve_symbolic_expressions_ext,
};
use p3_baby_bear::BabyBear as F;
use p3_field::extension::BinomialExtensionField;
use p3_field::{BasedVectorSpace, PrimeCharacteristicRing};
use p3_matrix::dense::RowMajorMatrixView;
use p3_matrix::stack::ViewPair;

type EF = BinomialExtensionField<F, 4>;

#[derive(Default)]
struct Evaluate {
    leaves: usize,
    operations: usize,
    events: Vec<(&'static str, i64)>,
}

impl<'a> SymbolicExprVisitor<'a, i64> for Evaluate {
    type Output = i64;
    fn leaf(&mut self, value: &'a i64) -> i64 {
        self.leaves += 1;
        self.events.push(("leaf", *value));
        *value
    }
    fn add(&mut self, x: &i64, y: &i64) -> i64 {
        self.operations += 1;
        self.events.push(("add", x + y));
        x + y
    }
    fn sub(&mut self, x: &i64, y: &i64) -> i64 {
        self.operations += 1;
        self.events.push(("sub", x - y));
        x - y
    }
    fn mul(&mut self, x: &i64, y: &i64) -> i64 {
        self.operations += 1;
        self.events.push(("mul", x * y));
        x * y
    }
    fn neg(&mut self, x: &i64) -> i64 {
        self.operations += 1;
        self.events.push(("neg", -x));
        -x
    }
}

#[test]
fn fold_preserves_order_and_visits_shared_nodes_once_across_roots() {
    let a = Arc::new(SymbolicExpr::Leaf(3));
    let b = Arc::new(SymbolicExpr::Leaf(5));
    let sum = Arc::new(SymbolicExpr::Add {
        x: a.clone(),
        y: b.clone(),
        degree_multiple: 0,
    });
    let product = SymbolicExpr::Mul {
        x: sum.clone(),
        y: sum.clone(),
        degree_multiple: 0,
    };
    let difference = SymbolicExpr::Sub {
        x: sum,
        y: a,
        degree_multiple: 0,
    };
    let neg = SymbolicExpr::Neg {
        x: b,
        degree_multiple: 0,
    };
    let mut visitor = Evaluate::default();
    assert_eq!(
        fold_symbolic_expressions(&[&product, &difference, &product, &neg], &mut visitor),
        [64, 5, 64, -5]
    );
    assert_eq!((visitor.leaves, visitor.operations), (2, 4));
    assert_eq!(
        visitor.events,
        [
            ("leaf", 3),
            ("leaf", 5),
            ("add", 8),
            ("mul", 64),
            ("sub", 5),
            ("neg", -5),
        ]
    );
    assert!(fold_symbolic_expressions::<i64, _>(&[], &mut visitor).is_empty());
    assert_eq!((visitor.leaves, visitor.operations), (2, 4));
    let equal_leaves = [SymbolicExpr::Leaf(3), SymbolicExpr::Leaf(3)];
    assert_eq!(
        fold_symbolic_expressions(&[&equal_leaves[0], &equal_leaves[1]], &mut visitor),
        [3, 3]
    );
    assert_eq!(visitor.leaves, 4);
}

#[test]
fn fold_handles_deep_chains_without_using_the_call_stack() {
    let mut nodes = vec![Arc::new(SymbolicExpr::Leaf(1))];
    for _ in 0..50_000 {
        nodes.push(Arc::new(SymbolicExpr::Neg {
            x: nodes.last().unwrap().clone(),
            degree_multiple: 0,
        }));
    }
    let mut visitor = Evaluate::default();
    assert_eq!(
        fold_symbolic_expressions(&[nodes.last().unwrap()], &mut visitor),
        [1]
    );
    assert_eq!(visitor.operations, 50_000);
    // Keep children alive while dropping each parent, so teardown is iterative too.
    while nodes.pop().is_some() {}
}

#[derive(Clone, Debug, PartialEq)]
struct Analysis<'a> {
    degree: usize,
    dependencies: Vec<&'a str>,
    id: usize,
}

#[derive(Default)]
struct Compile {
    instructions: Vec<(usize, usize)>,
}

impl<'a> SymbolicExprVisitor<'a, &'static str> for Compile {
    type Output = Analysis<'a>;
    fn leaf(&mut self, value: &'a &'static str) -> Self::Output {
        let id = self.instructions.len();
        self.instructions.push((id, id));
        Analysis {
            degree: 1,
            dependencies: vec![value],
            id,
        }
    }
    fn add(&mut self, x: &Self::Output, y: &Self::Output) -> Self::Output {
        let id = self.instructions.len();
        self.instructions.push((x.id, y.id));
        let mut dependencies = x.dependencies.clone();
        for dependency in &y.dependencies {
            if !dependencies.contains(dependency) {
                dependencies.push(dependency);
            }
        }
        Analysis {
            degree: x.degree.max(y.degree),
            dependencies,
            id,
        }
    }
    fn sub(&mut self, x: &Self::Output, y: &Self::Output) -> Self::Output {
        self.add(x, y)
    }
    fn neg(&mut self, x: &Self::Output) -> Self::Output {
        self.add(x, x)
    }
    fn mul(&mut self, x: &Self::Output, y: &Self::Output) -> Self::Output {
        let mut result = self.add(x, y);
        result.degree = x.degree + y.degree;
        result
    }
}

#[test]
fn visitor_can_borrow_dependencies_compute_degrees_and_emit_ir() {
    let shared = Arc::new(SymbolicExpr::Leaf("main"));
    let product = SymbolicExpr::Mul {
        x: shared.clone(),
        y: shared.clone(),
        degree_multiple: 2,
    };
    let mut visitor = Compile::default();
    let result = fold_symbolic_expressions(&[&product, &shared], &mut visitor);
    assert_eq!(
        result[0],
        Analysis {
            degree: 2,
            dependencies: vec!["main"],
            id: 1
        }
    );
    assert_eq!(result[1].id, 0);
    assert_eq!(visitor.instructions, [(0, 0), (0, 0)]);
}

fn base(entry: BaseEntry) -> SymbolicExpression<F> {
    SymbolicVariable::new(entry, 0).into()
}

fn ext(entry: ExtEntry) -> SymbolicExpressionExt<F, EF> {
    SymbolicVariableExt::new(entry, 0).into()
}

fn pair(rows: &[F]) -> ViewPair<'_, F> {
    ViewPair::new(
        RowMajorMatrixView::new(&rows[..1], 1),
        RowMajorMatrixView::new(&rows[1..], 1),
    )
}

#[test]
fn batched_resolution_handles_all_leaf_categories_and_true_extension_constants() {
    let main = [F::new(2), F::new(3)];
    let prep = [F::new(5), F::new(7)];
    let public = [F::new(11)];
    let periodic = [F::new(13)];
    let ec = EF::from_basis_coefficients_slice(&[F::new(17), F::new(19), F::new(23), F::new(29)])
        .unwrap();
    let permutation = [ec, ec + F::ONE];
    let challenges = [ec + F::TWO];
    let values = [ec + F::new(3)];
    let builder = DebugConstraintBuilder::new_with_permutation(
        0,
        pair(&main),
        pair(&prep),
        &public,
        F::new(31),
        F::new(37),
        F::new(41),
        ViewPair::new(
            RowMajorMatrixView::new(&permutation[..1], 1),
            RowMajorMatrixView::new(&permutation[1..], 1),
        ),
        &challenges,
        &values,
        &periodic,
    );
    let shared =
        Arc::new(base(BaseEntry::Main { offset: 0 }) * base(BaseEntry::Preprocessed { offset: 1 }));
    let lifted = SymbolicExpr::Add {
        x: shared.clone(),
        y: shared,
        degree_multiple: 2,
    };
    let bases = [
        lifted,
        base(BaseEntry::Main { offset: 1 }),
        base(BaseEntry::Preprocessed { offset: 0 }),
        base(BaseEntry::Public),
        base(BaseEntry::Periodic),
        SymbolicExpr::Leaf(BaseLeaf::IsFirstRow),
        SymbolicExpr::Leaf(BaseLeaf::IsLastRow),
        SymbolicExpr::Leaf(BaseLeaf::IsTransition),
        SymbolicExpression::from(F::new(43)),
    ];
    let roots: Vec<_> = bases.iter().collect();
    assert_eq!(
        resolve_symbolic_expressions(&roots, &builder),
        [28, 3, 5, 11, 13, 31, 37, 41, 43].map(F::new)
    );
    for (expression, resolved) in roots
        .iter()
        .zip(resolve_symbolic_expressions(&roots, &builder))
    {
        assert_eq!(expression.resolve(&builder), resolved);
    }
    let arithmetic = -(bases[0].clone() - bases[3].clone()) + bases[4].clone();
    assert_eq!(
        resolve_symbolic_expressions(&[&arithmetic], &builder),
        [-F::new(4)]
    );
    assert_eq!(arithmetic.resolve(&builder), -F::new(4));
    let lifted: SymbolicExpressionExt<F, EF> =
        bases.into_iter().sum::<SymbolicExpression<F>>().into();
    let current = ext(ExtEntry::Permutation { offset: 0 });
    let next = ext(ExtEntry::Permutation { offset: 1 });
    let challenge = ext(ExtEntry::Challenge);
    let value = ext(ExtEntry::PermutationValue);
    let expression =
        -(current * next - challenge) + value * SymbolicExpressionExt::from(ec) + lifted.clone();
    let expected = -(ec * (ec + F::ONE) - (ec + F::TWO)) + (ec + F::new(3)) * ec + F::new(212);
    assert_eq!(
        resolve_symbolic_expressions_ext(&[&expression, &lifted, &expression], &builder),
        [expected, EF::from(F::new(212)), expected]
    );
    assert_eq!(expression.resolve(&builder), expected);
    assert!(resolve_symbolic_expressions_ext::<F, EF, _>(&[], &builder).is_empty());
}

#[test]
fn extension_resolution_retains_base_subtree_arithmetic() {
    let builder = SymbolicAirBuilder::<F, EF>::new(AirLayout {
        main_width: 1,
        ..AirLayout::default()
    });
    let base = base(BaseEntry::Main { offset: 0 });
    let square = base.clone() * base;
    let expression = SymbolicExpressionExt::from(square);
    let resolved = expression.resolve(&builder);
    assert!(matches!(
        resolved,
        SymbolicExpr::Leaf(ExtLeaf::Base(SymbolicExpr::Mul { .. }))
    ));
}

#[test]
fn extension_resolution_shares_base_work_across_distinct_embedded_roots() {
    let builder = SymbolicAirBuilder::<F, EF>::new(AirLayout {
        main_width: 1,
        ..AirLayout::default()
    });
    let base = base(BaseEntry::Main { offset: 0 });
    let square = Arc::new(base.clone() * base);
    let first = SymbolicExpressionExt::from(SymbolicExpr::Neg {
        x: square.clone(),
        degree_multiple: 2,
    });
    let second = SymbolicExpressionExt::from(SymbolicExpr::Neg {
        x: square,
        degree_multiple: 2,
    });
    let resolved = resolve_symbolic_expressions_ext(&[&first, &second], &builder);
    // A symbolic builder allocates children when evaluating multiplication.
    // Reusing their addresses proves the shared multiplication was evaluated
    // once even though it is embedded in two distinct base roots.
    fn square_child(expression: &SymbolicExpressionExt<F, EF>) -> &Arc<SymbolicExpression<F>> {
        let SymbolicExpr::Leaf(ExtLeaf::Base(SymbolicExpr::Neg { x, .. })) = expression else {
            panic!("expected lifted base negation");
        };
        let SymbolicExpr::Mul { x, .. } = x.as_ref() else {
            panic!("expected base square");
        };
        x
    }
    assert!(Arc::ptr_eq(
        square_child(&resolved[0]),
        square_child(&resolved[1])
    ));
}

#[test]
#[should_panic(expected = "expressions cannot span more than two rows")]
fn extension_resolution_rejects_unsupported_row_offsets() {
    let builder = SymbolicAirBuilder::<F, EF>::new(AirLayout {
        permutation_width: 1,
        ..AirLayout::default()
    });
    let _ = ext(ExtEntry::Permutation { offset: 2 }).resolve(&builder);
}

#[test]
#[should_panic(expected = "permutation column index out of bounds")]
fn extension_resolution_rejects_out_of_bounds_columns() {
    let builder = SymbolicAirBuilder::<F, EF>::new(AirLayout {
        permutation_width: 1,
        ..AirLayout::default()
    });
    let _ = SymbolicExpressionExt::from(SymbolicVariableExt::new(
        ExtEntry::Permutation { offset: 0 },
        1,
    ))
    .resolve(&builder);
}

// Keep the reference independent of the extension resolver and its visitor.
fn resolve_extension_reference(
    expression: &SymbolicExpressionExt<F, EF>,
    builder: &DebugConstraintBuilder<'_, F, EF>,
) -> EF {
    match expression {
        SymbolicExpr::Leaf(ExtLeaf::Base(base)) => base.resolve(builder).into(),
        SymbolicExpr::Leaf(ExtLeaf::ExtConstant(value)) => *value,
        SymbolicExpr::Leaf(ExtLeaf::ExtVariable(variable)) => match variable.entry {
            ExtEntry::Permutation { offset: 0 } => {
                builder.permutation().current(variable.index).unwrap()
            }
            ExtEntry::Permutation { offset: 1 } => {
                builder.permutation().next(variable.index).unwrap()
            }
            ExtEntry::Permutation { .. } => unreachable!(),
            ExtEntry::Challenge => builder.permutation_randomness()[variable.index],
            ExtEntry::PermutationValue => builder.permutation_values()[variable.index],
        },
        SymbolicExpr::Add { x, y, .. } => {
            resolve_extension_reference(x, builder) + resolve_extension_reference(y, builder)
        }
        SymbolicExpr::Sub { x, y, .. } => {
            resolve_extension_reference(x, builder) - resolve_extension_reference(y, builder)
        }
        SymbolicExpr::Neg { x, .. } => -resolve_extension_reference(x, builder),
        SymbolicExpr::Mul { x, y, .. } => {
            resolve_extension_reference(x, builder) * resolve_extension_reference(y, builder)
        }
    }
}

// A fixed generator keeps failures reproducible without another dependency.
const fn random_index(state: &mut u64, limit: usize) -> usize {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 32) as usize) % limit
}

fn grow_random_dag<A>(pool: &mut Vec<Arc<SymbolicExpr<A>>>, state: &mut u64) {
    for _ in 0..48 {
        let x = pool[random_index(state, pool.len())].clone();
        let y = pool[random_index(state, pool.len())].clone();
        let degree_multiple = 0;
        let expression = match random_index(state, 4) {
            0 => SymbolicExpr::Add {
                x,
                y,
                degree_multiple,
            },
            1 => SymbolicExpr::Sub {
                x,
                y,
                degree_multiple,
            },
            2 => SymbolicExpr::Neg { x, degree_multiple },
            _ => SymbolicExpr::Mul {
                x,
                y,
                degree_multiple,
            },
        };
        pool.push(Arc::new(expression));
    }
}

#[test]
fn seeded_random_dags_match_recursive_base_and_extension_resolution() {
    let main = [F::new(2), F::new(3)];
    let prep = [F::new(5), F::new(7)];
    let public = [F::new(11)];
    let periodic = [F::new(13)];
    let ec = EF::from_basis_coefficients_slice(&[F::new(17), F::new(19), F::new(23), F::new(29)])
        .unwrap();
    let permutation = [ec, ec + F::ONE];
    let challenges = [ec + F::TWO];
    let values = [ec + F::new(3)];
    let builder = DebugConstraintBuilder::new_with_permutation(
        0,
        pair(&main),
        pair(&prep),
        &public,
        F::new(31),
        F::new(37),
        F::new(41),
        ViewPair::new(
            RowMajorMatrixView::new(&permutation[..1], 1),
            RowMajorMatrixView::new(&permutation[1..], 1),
        ),
        &challenges,
        &values,
        &periodic,
    );
    for seed in 0..300 {
        let mut state = seed;
        let mut bases: Vec<_> = [
            base(BaseEntry::Main { offset: 0 }),
            base(BaseEntry::Main { offset: 1 }),
            base(BaseEntry::Preprocessed { offset: 0 }),
            base(BaseEntry::Preprocessed { offset: 1 }),
            base(BaseEntry::Public),
            base(BaseEntry::Periodic),
            SymbolicExpr::Leaf(BaseLeaf::IsFirstRow),
            SymbolicExpr::Leaf(BaseLeaf::IsLastRow),
            SymbolicExpr::Leaf(BaseLeaf::IsTransition),
            SymbolicExpression::from(F::new(43)),
        ]
        .into_iter()
        .map(Arc::new)
        .collect();
        grow_random_dag(&mut bases, &mut state);
        let mut base_roots: Vec<_> = (0..7)
            .map(|_| bases[random_index(&mut state, bases.len())].as_ref())
            .collect();
        base_roots.push(base_roots[0]);
        assert_eq!(
            resolve_symbolic_expressions(&base_roots, &builder),
            base_roots
                .iter()
                .map(|root| root.resolve(&builder))
                .collect::<Vec<_>>(),
            "base seed {seed}"
        );

        // Lift the entire pool: distinct embedded roots still share Arc children.
        let mut extensions: Vec<_> = bases
            .iter()
            .map(|base| Arc::new(SymbolicExpressionExt::from(base.as_ref().clone())))
            .collect();
        extensions.extend(
            [
                ext(ExtEntry::Permutation { offset: 0 }),
                ext(ExtEntry::Permutation { offset: 1 }),
                ext(ExtEntry::Challenge),
                ext(ExtEntry::PermutationValue),
                SymbolicExpr::Leaf(ExtLeaf::ExtConstant(ec)),
            ]
            .into_iter()
            .map(Arc::new),
        );
        grow_random_dag(&mut extensions, &mut state);
        let mut roots: Vec<_> = (0..7)
            .map(|_| extensions[random_index(&mut state, extensions.len())].as_ref())
            .collect();
        roots.push(roots[0]);
        let expected: Vec<_> = roots
            .iter()
            .map(|root| resolve_extension_reference(root, &builder))
            .collect();
        assert_eq!(
            resolve_symbolic_expressions_ext(&roots, &builder),
            expected,
            "extension seed {seed}"
        );
        assert_eq!(
            roots
                .iter()
                .map(|root| root.resolve(&builder))
                .collect::<Vec<_>>(),
            expected,
            "recursive extension seed {seed}"
        );
    }
}
