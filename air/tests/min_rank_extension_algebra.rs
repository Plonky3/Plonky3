use core::array;
use std::collections::BTreeSet;

use p3_air::arithmetic::{ArithmeticExpression, ArithmeticNode, ArithmeticVariable, BaseOperand};
use p3_air::{
    AirBuilder, AirLayout, BaseEntry, BaseLeaf, CapturedArithmeticConstraints, ConstraintLayout,
    SymbolicAirBuilder, SymbolicExprVisitor, SymbolicExpression, SymbolicVariable,
    capture_arithmetic, fold_symbolic_expressions, resolve_symbolic_expressions,
};
use p3_baby_bear::BabyBear;
use p3_field::extension::{
    BinomialExtensionField, BinomiallyExtendable, CubicTrinomialExtensionField,
    QuinticTrinomialExtensionField, binomial_square, cubic_square, qm31_mul_min_rank,
    qm31_square_min_rank, quintic_square, trinomial_cubic_mul, trinomial_quintic_mul,
};
use p3_field::{BasedVectorSpace, Field, PrimeCharacteristicRing};
use p3_goldilocks::Goldilocks;
use p3_koala_bear::KoalaBear;
use p3_matrix::dense::RowMajorMatrix;
use p3_mersenne_31::{Mersenne31, QM31};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

/// Counts operations across a multi-root [`SymbolicExpression`] DAG (visiting each
/// shared `Arc` node exactly once, matching [`fold_symbolic_expressions`] and wire flattening).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct SymDagStats {
    /// Non-constant multiplications (`deg(x) > 0 && deg(y) > 0`, i.e., quadratic `A * A` muls).
    quadratic_muls: usize,
    /// Constant-scaling multiplications (`expr * const`).
    scalar_muls: usize,
    /// Additions and subtractions.
    add_subs: usize,
    /// Negations.
    negs: usize,
}

struct SymStatsVisitor<F> {
    stats: SymDagStats,
    _marker: core::marker::PhantomData<F>,
}

impl<F> SymStatsVisitor<F> {
    const fn new() -> Self {
        Self {
            stats: SymDagStats {
                quadratic_muls: 0,
                scalar_muls: 0,
                add_subs: 0,
                negs: 0,
            },
            _marker: core::marker::PhantomData,
        }
    }
}

impl<'a, F: Field> SymbolicExprVisitor<'a, BaseLeaf<F>> for SymStatsVisitor<F> {
    // Track whether the subtree is a compile-time constant (`true`) or non-constant (`false`).
    type Output = bool;

    fn leaf(&mut self, leaf: &'a BaseLeaf<F>) -> bool {
        matches!(leaf, BaseLeaf::Constant(_))
    }

    fn add(&mut self, x: &bool, y: &bool) -> bool {
        self.stats.add_subs += 1;
        *x && *y
    }

    fn sub(&mut self, x: &bool, y: &bool) -> bool {
        self.stats.add_subs += 1;
        *x && *y
    }

    fn neg(&mut self, x: &bool) -> bool {
        self.stats.negs += 1;
        *x
    }

    fn mul(&mut self, x: &bool, y: &bool) -> bool {
        if !*x && !*y {
            self.stats.quadratic_muls += 1;
        } else {
            self.stats.scalar_muls += 1;
        }
        *x && *y
    }
}

fn count_symbolic_dag<F: Field, const D: usize>(roots: &[SymbolicExpression<F>; D]) -> SymDagStats {
    let refs: [&SymbolicExpression<F>; D] = array::from_fn(|i| &roots[i]);
    let mut visitor = SymStatsVisitor::<F>::new();
    let _ = fold_symbolic_expressions(&refs, &mut visitor);
    visitor.stats
}

/// Counts distinct nodes across a multi-root [`ArithmeticExpression`] DAG by pointer identity.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct ArithDagStats {
    muls: usize,
    squares: usize,
    dots: usize,
    mixed_dots: usize,
    add_subs: usize,
}

fn visit_arith_operand<F: Field>(
    op: &BaseOperand<F>,
    seen: &mut BTreeSet<usize>,
    stats: &mut ArithDagStats,
) {
    if let BaseOperand::Expr(arc) = op {
        let key = std::sync::Arc::as_ptr(arc) as usize;
        if seen.insert(key) {
            visit_arith_expr(arc, seen, stats);
        }
    }
}

fn visit_arith_expr<F: Field>(
    expr: &ArithmeticExpression<F>,
    seen: &mut BTreeSet<usize>,
    stats: &mut ArithDagStats,
) {
    match expr.node() {
        ArithmeticNode::Leaf(_) | ArithmeticNode::Neg(_) => {
            if let ArithmeticNode::Neg(a) = expr.node() {
                visit_arith_operand(a, seen, stats);
            }
        }
        ArithmeticNode::Add(a, b) | ArithmeticNode::Sub(a, b) => {
            stats.add_subs += 1;
            visit_arith_operand(a, seen, stats);
            visit_arith_operand(b, seen, stats);
        }
        ArithmeticNode::Mul(a, b) => {
            stats.muls += 1;
            visit_arith_operand(a, seen, stats);
            visit_arith_operand(b, seen, stats);
        }
        ArithmeticNode::Square(a) => {
            stats.squares += 1;
            visit_arith_operand(a, seen, stats);
        }
        ArithmeticNode::Sum(items) => {
            for item in items.iter() {
                visit_arith_operand(item, seen, stats);
            }
        }
        ArithmeticNode::Dot(pairs) => {
            stats.dots += 1;
            for (a, b) in pairs.iter() {
                visit_arith_operand(a, seen, stats);
                visit_arith_operand(b, seen, stats);
            }
        }
        ArithmeticNode::MixedDot(pairs) => {
            stats.mixed_dots += 1;
            for (a, b) in pairs.iter() {
                visit_arith_operand(a, seen, stats);
                visit_arith_operand(b, seen, stats);
            }
        }
        _ => unreachable!(),
    }
}

fn count_arithmetic_dag<F: Field, const D: usize>(
    roots: &[ArithmeticExpression<F>; D],
) -> ArithDagStats {
    let mut seen = BTreeSet::new();
    let mut stats = ArithDagStats::default();
    for root in roots {
        visit_arith_expr(root, &mut seen, &mut stats);
    }
    stats
}

/// Minimal concrete `AirBuilder` to evaluate `SymbolicExpression` and `ArithmeticExpression`
/// on arbitrary trace row values `[a_0..a_{D-1}, b_0..b_{D-1}]`.
struct RowEvalBuilder<F> {
    main: RowMajorMatrix<F>,
}

impl<F: Field> RowEvalBuilder<F> {
    fn new(row: Vec<F>) -> Self {
        let width = row.len();
        let mut values = row.clone();
        values.extend(row);
        Self {
            main: RowMajorMatrix::new(values, width),
        }
    }
}

impl<F: Field> AirBuilder for RowEvalBuilder<F> {
    type F = F;
    type Expr = F;
    type Var = F;
    type PreprocessedWindow = RowMajorMatrix<F>;
    type MainWindow = RowMajorMatrix<F>;
    type PublicVar = F;
    type PeriodicVar = F;

    fn main(&self) -> Self::MainWindow {
        self.main.clone()
    }

    fn preprocessed(&self) -> &Self::PreprocessedWindow {
        &self.main
    }

    fn is_first_row(&self) -> F {
        F::ONE
    }

    fn is_last_row(&self) -> F {
        F::ZERO
    }

    fn is_transition(&self) -> F {
        F::ONE
    }

    fn assert_zero<I: Into<Self::Expr>>(&mut self, _x: I) {}
}

/// Baseline schoolbook binomial multiplication over `SymbolicExpression<F>` (mirroring `quartic_mul`,
/// `quintic_mul`, `octic_mul` dot-product expansion before minimum-rank `ExtensionAlgebra`).
fn schoolbook_binomial_sym<F: Field, const D: usize>(
    a: &[SymbolicExpression<F>; D],
    b: &[SymbolicExpression<F>; D],
    w: F,
) -> [SymbolicExpression<F>; D] {
    let mut out: [SymbolicExpression<F>; D] = array::from_fn(|_| SymbolicExpression::ZERO);
    for i in 0..D {
        for j in 0..D {
            let prod = a[i].clone() * b[j].clone();
            if i + j >= D {
                out[i + j - D] += prod * w;
            } else {
                out[i + j] += prod;
            }
        }
    }
    out
}

fn verify_binomial_air<F, const D: usize>(
    expected_before_muls: usize,
    expected_after_muls: usize,
    expected_sq_muls: usize,
    expected_arith_sq: (usize, usize),
) where
    F: BinomiallyExtendable<D>,
    rand::distr::StandardUniform: rand::distr::Distribution<F>,
{
    let w = <F as BinomiallyExtendable<D>>::W;
    let sym_a: [SymbolicExpression<F>; D] =
        array::from_fn(|i| SymbolicVariable::new(BaseEntry::Main { offset: 0 }, i).into());
    let sym_b: [SymbolicExpression<F>; D] =
        array::from_fn(|i| SymbolicVariable::new(BaseEntry::Main { offset: 0 }, D + i).into());

    // Before: schoolbook / dot_product expansion emits D^2 quadratic muls.
    let before_mul = schoolbook_binomial_sym(&sym_a, &sym_b, w);
    assert_eq!(
        count_symbolic_dag(&before_mul).quadratic_muls,
        expected_before_muls
    );

    let mut before_sq: [SymbolicExpression<F>; D] = array::from_fn(|_| SymbolicExpression::ZERO);
    binomial_square(&sym_a, &mut before_sq, w);
    let before_sq_muls = count_symbolic_dag(&before_sq).quadratic_muls;
    assert!(before_sq_muls > expected_sq_muls);

    // After: SymbolicExpression<F> implements ExtensionAlgebra<F, D, Binomial<F>>!
    let ext_sym_a = BinomialExtensionField::<F, D, SymbolicExpression<F>>::new(sym_a);
    let ext_sym_b = BinomialExtensionField::<F, D, SymbolicExpression<F>>::new(sym_b);
    let ext_sym_prod = ext_sym_a.clone() * ext_sym_b;
    let ext_sym_sq = ext_sym_a.square();

    let prod_roots: [SymbolicExpression<F>; D] = ext_sym_prod
        .as_basis_coefficients_slice()
        .to_vec()
        .try_into()
        .unwrap();
    let sq_roots: [SymbolicExpression<F>; D] = ext_sym_sq
        .as_basis_coefficients_slice()
        .to_vec()
        .try_into()
        .unwrap();

    assert_eq!(
        count_symbolic_dag(&prod_roots).quadratic_muls,
        expected_after_muls
    );
    assert_eq!(
        count_symbolic_dag(&sq_roots).quadratic_muls,
        expected_sq_muls
    );
    for r in &prod_roots {
        assert_eq!(r.degree_multiple(), 2);
    }
    for r in &sq_roots {
        assert_eq!(r.degree_multiple(), 2);
    }

    // ArithmeticExpression<F> implements ExtensionAlgebra<F, D, Binomial<F>>!
    let arith_a: [ArithmeticExpression<F>; D] =
        array::from_fn(|i| ArithmeticVariable::new(BaseEntry::Main { offset: 0 }, i).into());
    let arith_b: [ArithmeticExpression<F>; D] =
        array::from_fn(|i| ArithmeticVariable::new(BaseEntry::Main { offset: 0 }, D + i).into());
    let ext_arith_a = BinomialExtensionField::<F, D, ArithmeticExpression<F>>::new(arith_a);
    let ext_arith_b = BinomialExtensionField::<F, D, ArithmeticExpression<F>>::new(arith_b);
    let ext_arith_prod = ext_arith_a.clone() * ext_arith_b;
    let ext_arith_sq = ext_arith_a.square();

    let arith_prod_roots: [ArithmeticExpression<F>; D] = ext_arith_prod
        .as_basis_coefficients_slice()
        .to_vec()
        .try_into()
        .unwrap();
    let arith_sq_roots: [ArithmeticExpression<F>; D] = ext_arith_sq
        .as_basis_coefficients_slice()
        .to_vec()
        .try_into()
        .unwrap();

    let arith_prod_stats = count_arithmetic_dag(&arith_prod_roots);
    assert_eq!(arith_prod_stats.muls, expected_after_muls);
    assert_eq!(arith_prod_stats.squares, 0);

    let arith_sq_stats = count_arithmetic_dag(&arith_sq_roots);
    assert_eq!(
        (arith_sq_stats.muls, arith_sq_stats.squares),
        expected_arith_sq
    );

    // Lowering CapturedArithmeticConstraints preserves shared Arcs -> exact 2D - 1 quadratic muls!
    let captured = CapturedArithmeticConstraints::<F, F> {
        base: arith_prod_roots.to_vec(),
        extension: vec![],
        layout: ConstraintLayout {
            base_indices: (0..D).collect(),
            ext_indices: vec![],
        },
    };
    let lowered = captured.lower().unwrap();
    let lowered_roots: [SymbolicExpression<F>; D] =
        lowered.base_constraints().to_vec().try_into().unwrap();
    assert_eq!(
        count_symbolic_dag(&lowered_roots).quadratic_muls,
        expected_after_muls
    );

    // Exhaustive basis pairs + random evaluation agreement against concrete BinomialExtensionField<F, D>.
    let mut test_vectors: Vec<([F; D], [F; D])> = Vec::new();
    for i in 0..D {
        for j in 0..D {
            let mut va = [F::ZERO; D];
            let mut vb = [F::ZERO; D];
            va[i] = F::ONE;
            vb[j] = F::ONE;
            test_vectors.push((va, vb));
        }
    }
    let mut rng = SmallRng::seed_from_u64(0xA1B2_C3D4 ^ (D as u64));
    for _ in 0..64 {
        test_vectors.push((
            array::from_fn(|_| rng.random()),
            array::from_fn(|_| rng.random()),
        ));
    }

    let prod_refs: [&SymbolicExpression<F>; D] = array::from_fn(|i| &prod_roots[i]);
    let sq_refs: [&SymbolicExpression<F>; D] = array::from_fn(|i| &sq_roots[i]);

    for (va, vb) in test_vectors {
        let mut row = Vec::with_capacity(2 * D);
        row.extend_from_slice(&va);
        row.extend_from_slice(&vb);
        let builder = RowEvalBuilder::new(row);

        let expected_mul =
            BinomialExtensionField::<F, D>::new(va) * BinomialExtensionField::<F, D>::new(vb);
        let expected_sq = BinomialExtensionField::<F, D>::new(va).square();

        let sym_eval_mul = resolve_symbolic_expressions(&prod_refs, &builder);
        let sym_eval_sq = resolve_symbolic_expressions(&sq_refs, &builder);
        assert_eq!(
            sym_eval_mul.as_slice(),
            expected_mul.as_basis_coefficients_slice()
        );
        assert_eq!(
            sym_eval_sq.as_slice(),
            expected_sq.as_basis_coefficients_slice()
        );

        let arith_eval_mul: [F; D] = array::from_fn(|i| arith_prod_roots[i].resolve(&builder));
        let arith_eval_sq: [F; D] = array::from_fn(|i| arith_sq_roots[i].resolve(&builder));
        assert_eq!(&arith_eval_mul, expected_mul.as_basis_coefficients_slice());
        assert_eq!(&arith_eval_sq, expected_sq.as_basis_coefficients_slice());
    }
}

#[test]
fn binomial_extension_algebra_min_rank_across_fields() {
    // D = 2: 4 -> 3 muls (-25.0%), square: 3 -> 2 muls (-33.3%)
    verify_binomial_air::<Goldilocks, 2>(4, 3, 2, (2, 0));
    verify_binomial_air::<Mersenne31, 2>(4, 3, 2, (2, 0));
    // D = 3: 9 -> 5 muls (-44.44%), square: 6 -> 5 squares
    verify_binomial_air::<Mersenne31, 3>(9, 5, 5, (0, 5));
    // D = 4: 16 -> 7 muls (-56.25%), square: 10 -> 7 squares (-30.0%)
    verify_binomial_air::<BabyBear, 4>(16, 7, 7, (0, 7));
    verify_binomial_air::<KoalaBear, 4>(16, 7, 7, (0, 7));
    // D = 5: 25 -> 9 muls (-64.00%), square: 15 -> 9 squares (-40.0%)
    verify_binomial_air::<BabyBear, 5>(25, 9, 9, (0, 9));
    verify_binomial_air::<Goldilocks, 5>(25, 9, 9, (0, 9));
    // D = 8: 64 -> 15 muls (-76.56%), square: 36 -> 15 squares (-58.33%)
    verify_binomial_air::<BabyBear, 8>(64, 15, 15, (0, 15));
    verify_binomial_air::<KoalaBear, 8>(64, 15, 15, (0, 15));
}

#[test]
fn cubic_trinomial_extension_algebra_min_rank_goldilocks() {
    type F = Goldilocks;
    let sym_a: [SymbolicExpression<F>; 3] =
        array::from_fn(|i| SymbolicVariable::new(BaseEntry::Main { offset: 0 }, i).into());
    let sym_b: [SymbolicExpression<F>; 3] =
        array::from_fn(|i| SymbolicVariable::new(BaseEntry::Main { offset: 0 }, 3 + i).into());

    // Before: trinomial_cubic_mul emits 9 quadratic SymbolicExpr::Mul nodes!
    let mut before_mul: [SymbolicExpression<F>; 3] = array::from_fn(|_| SymbolicExpression::ZERO);
    trinomial_cubic_mul(&sym_a, &sym_b, &mut before_mul);
    assert_eq!(count_symbolic_dag(&before_mul).quadratic_muls, 9);

    let mut before_sq: [SymbolicExpression<F>; 3] = array::from_fn(|_| SymbolicExpression::ZERO);
    cubic_square(&sym_a, &mut before_sq);
    assert_eq!(count_symbolic_dag(&before_sq).quadratic_muls, 6);

    // After: CubicTrinomialExtensionField<Goldilocks, SymbolicExpression<Goldilocks>> -> 5 muls!
    let ext_sym_a = CubicTrinomialExtensionField::<F, SymbolicExpression<F>>::new(sym_a);
    let ext_sym_b = CubicTrinomialExtensionField::<F, SymbolicExpression<F>>::new(sym_b);
    let prod_roots: [SymbolicExpression<F>; 3] = (ext_sym_a.clone() * ext_sym_b)
        .as_basis_coefficients_slice()
        .to_vec()
        .try_into()
        .unwrap();
    let sq_roots: [SymbolicExpression<F>; 3] = ext_sym_a
        .square()
        .as_basis_coefficients_slice()
        .to_vec()
        .try_into()
        .unwrap();

    assert_eq!(count_symbolic_dag(&prod_roots).quadratic_muls, 5);
    assert_eq!(count_symbolic_dag(&sq_roots).quadratic_muls, 5);

    // ArithmeticExpression<Goldilocks> -> 5 Mul nodes (0 scalar muls), 5 Square nodes!
    let arith_a: [ArithmeticExpression<F>; 3] =
        array::from_fn(|i| ArithmeticVariable::new(BaseEntry::Main { offset: 0 }, i).into());
    let arith_b: [ArithmeticExpression<F>; 3] =
        array::from_fn(|i| ArithmeticVariable::new(BaseEntry::Main { offset: 0 }, 3 + i).into());
    let ext_arith_a = CubicTrinomialExtensionField::<F, ArithmeticExpression<F>>::new(arith_a);
    let ext_arith_b = CubicTrinomialExtensionField::<F, ArithmeticExpression<F>>::new(arith_b);
    let arith_prod: [ArithmeticExpression<F>; 3] = (ext_arith_a.clone() * ext_arith_b)
        .as_basis_coefficients_slice()
        .to_vec()
        .try_into()
        .unwrap();
    let arith_sq: [ArithmeticExpression<F>; 3] = ext_arith_a
        .square()
        .as_basis_coefficients_slice()
        .to_vec()
        .try_into()
        .unwrap();

    let prod_stats = count_arithmetic_dag(&arith_prod);
    assert_eq!((prod_stats.muls, prod_stats.squares), (5, 0));
    let sq_stats = count_arithmetic_dag(&arith_sq);
    assert_eq!((sq_stats.muls, sq_stats.squares), (0, 5));

    let mut rng = SmallRng::seed_from_u64(0x901D_10C5);
    let prod_refs: [&SymbolicExpression<F>; 3] = array::from_fn(|i| &prod_roots[i]);
    for _ in 0..64 {
        let va: [F; 3] = array::from_fn(|_| rng.random());
        let vb: [F; 3] = array::from_fn(|_| rng.random());
        let builder = RowEvalBuilder::new(vec![va[0], va[1], va[2], vb[0], vb[1], vb[2]]);
        let expected =
            CubicTrinomialExtensionField::<F>::new(va) * CubicTrinomialExtensionField::<F>::new(vb);
        assert_eq!(
            resolve_symbolic_expressions(&prod_refs, &builder).as_slice(),
            expected.as_basis_coefficients_slice()
        );
        let arith_eval: [F; 3] = array::from_fn(|i| arith_prod[i].resolve(&builder));
        assert_eq!(&arith_eval, expected.as_basis_coefficients_slice());
    }
}

#[test]
fn quintic_trinomial_extension_algebra_min_rank_koalabear() {
    type F = KoalaBear;
    let sym_a: [SymbolicExpression<F>; 5] =
        array::from_fn(|i| SymbolicVariable::new(BaseEntry::Main { offset: 0 }, i).into());
    let sym_b: [SymbolicExpression<F>; 5] =
        array::from_fn(|i| SymbolicVariable::new(BaseEntry::Main { offset: 0 }, 5 + i).into());

    // Before: trinomial_quintic_mul emits 26 quadratic SymbolicExpr::Mul nodes
    // (25 convolution products + 1 unshared duplicate of c8 = a[4] * b[4])!
    let mut before_mul: [SymbolicExpression<F>; 5] = array::from_fn(|_| SymbolicExpression::ZERO);
    trinomial_quintic_mul(&sym_a, &sym_b, &mut before_mul);
    assert_eq!(count_symbolic_dag(&before_mul).quadratic_muls, 26);

    // Before: quintic_square emits 17 quadratic SymbolicExpr::Mul nodes
    // (15 products + unshared c7.dup() and c8.dup() duplicates)!
    let mut before_sq: [SymbolicExpression<F>; 5] = array::from_fn(|_| SymbolicExpression::ZERO);
    quintic_square(&sym_a, &mut before_sq);
    assert_eq!(count_symbolic_dag(&before_sq).quadratic_muls, 17);

    // After: QuinticTrinomialExtensionField<KoalaBear, SymbolicExpression<KoalaBear>> -> 9 muls!
    let ext_sym_a = QuinticTrinomialExtensionField::<F, SymbolicExpression<F>>::new(sym_a);
    let ext_sym_b = QuinticTrinomialExtensionField::<F, SymbolicExpression<F>>::new(sym_b);
    let prod_roots: [SymbolicExpression<F>; 5] = (ext_sym_a.clone() * ext_sym_b)
        .as_basis_coefficients_slice()
        .to_vec()
        .try_into()
        .unwrap();
    let sq_roots: [SymbolicExpression<F>; 5] = ext_sym_a
        .square()
        .as_basis_coefficients_slice()
        .to_vec()
        .try_into()
        .unwrap();

    assert_eq!(count_symbolic_dag(&prod_roots).quadratic_muls, 9);
    assert_eq!(count_symbolic_dag(&sq_roots).quadratic_muls, 9);

    let arith_a: [ArithmeticExpression<F>; 5] =
        array::from_fn(|i| ArithmeticVariable::new(BaseEntry::Main { offset: 0 }, i).into());
    let arith_b: [ArithmeticExpression<F>; 5] =
        array::from_fn(|i| ArithmeticVariable::new(BaseEntry::Main { offset: 0 }, 5 + i).into());
    let ext_arith_a = QuinticTrinomialExtensionField::<F, ArithmeticExpression<F>>::new(arith_a);
    let ext_arith_b = QuinticTrinomialExtensionField::<F, ArithmeticExpression<F>>::new(arith_b);
    let arith_prod: [ArithmeticExpression<F>; 5] = (ext_arith_a.clone() * ext_arith_b)
        .as_basis_coefficients_slice()
        .to_vec()
        .try_into()
        .unwrap();
    let arith_sq: [ArithmeticExpression<F>; 5] = ext_arith_a
        .square()
        .as_basis_coefficients_slice()
        .to_vec()
        .try_into()
        .unwrap();

    let prod_stats = count_arithmetic_dag(&arith_prod);
    assert_eq!((prod_stats.muls, prod_stats.squares), (9, 0));
    let sq_stats = count_arithmetic_dag(&arith_sq);
    assert_eq!((sq_stats.muls, sq_stats.squares), (0, 9));

    let mut rng = SmallRng::seed_from_u64(0xC0A1_ABEA);
    let prod_refs: [&SymbolicExpression<F>; 5] = array::from_fn(|i| &prod_roots[i]);
    for _ in 0..64 {
        let va: [F; 5] = array::from_fn(|_| rng.random());
        let vb: [F; 5] = array::from_fn(|_| rng.random());
        let mut row = Vec::with_capacity(10);
        row.extend_from_slice(&va);
        row.extend_from_slice(&vb);
        let builder = RowEvalBuilder::new(row);
        let expected = QuinticTrinomialExtensionField::<F>::new(va)
            * QuinticTrinomialExtensionField::<F>::new(vb);
        assert_eq!(
            resolve_symbolic_expressions(&prod_refs, &builder).as_slice(),
            expected.as_basis_coefficients_slice()
        );
        let arith_eval: [F; 5] = array::from_fn(|i| arith_prod[i].resolve(&builder));
        assert_eq!(&arith_eval, expected.as_basis_coefficients_slice());
    }
}

#[test]
fn qm31_min_rank_symbolic_and_arithmetic_ast_counts() {
    type F = Mersenne31;
    let sym_a: [SymbolicExpression<F>; 4] =
        array::from_fn(|i| SymbolicVariable::new(BaseEntry::Main { offset: 0 }, i).into());
    let sym_b: [SymbolicExpression<F>; 4] =
        array::from_fn(|i| SymbolicVariable::new(BaseEntry::Main { offset: 0 }, 4 + i).into());

    let mut prod_roots: [SymbolicExpression<F>; 4] = array::from_fn(|_| SymbolicExpression::ZERO);
    qm31_mul_min_rank::<F, _>(&sym_a, &sym_b, &mut prod_roots);
    assert_eq!(count_symbolic_dag(&prod_roots).quadratic_muls, 7);

    let mut sq_roots: [SymbolicExpression<F>; 4] = array::from_fn(|_| SymbolicExpression::ZERO);
    qm31_square_min_rank::<F, _>(&sym_a, &mut sq_roots);
    assert_eq!(count_symbolic_dag(&sq_roots).quadratic_muls, 6);

    let arith_a: [ArithmeticExpression<F>; 4] =
        array::from_fn(|i| ArithmeticVariable::new(BaseEntry::Main { offset: 0 }, i).into());
    let arith_b: [ArithmeticExpression<F>; 4] =
        array::from_fn(|i| ArithmeticVariable::new(BaseEntry::Main { offset: 0 }, 4 + i).into());
    let mut arith_prod: [ArithmeticExpression<F>; 4] =
        array::from_fn(|_| ArithmeticExpression::ZERO);
    qm31_mul_min_rank::<F, _>(&arith_a, &arith_b, &mut arith_prod);
    assert_eq!(count_arithmetic_dag(&arith_prod).muls, 7);

    let mut arith_sq: [ArithmeticExpression<F>; 4] = array::from_fn(|_| ArithmeticExpression::ZERO);
    qm31_square_min_rank::<F, _>(&arith_a, &mut arith_sq);
    assert_eq!(count_arithmetic_dag(&arith_sq).muls, 6);

    // End-to-end SymbolicAirBuilder + capture_arithmetic roundtrip check!
    let layout = AirLayout {
        main_width: 12,
        ..Default::default()
    };
    let mut sym_builder = SymbolicAirBuilder::<F, QM31>::new(layout);
    {
        use p3_air::WindowAccess;
        let main = sym_builder.main();
        let a: [SymbolicExpression<F>; 4] = array::from_fn(|i| main.current(i).unwrap().into());
        let b: [SymbolicExpression<F>; 4] = array::from_fn(|i| main.current(4 + i).unwrap().into());
        let c: [SymbolicExpression<F>; 4] = array::from_fn(|i| main.current(8 + i).unwrap().into());
        let mut ab: [SymbolicExpression<F>; 4] = array::from_fn(|_| SymbolicExpression::ZERO);
        qm31_mul_min_rank::<F, _>(&a, &b, &mut ab);
        sym_builder.assert_eq_arrays(ab, c);
    }
    let sym_constraints: [SymbolicExpression<F>; 4] = sym_builder
        .finish()
        .base_constraints()
        .to_vec()
        .try_into()
        .unwrap();
    assert_eq!(count_symbolic_dag(&sym_constraints).quadratic_muls, 7);

    let arith_captured = capture_arithmetic::<F, QM31>(layout, |builder| {
        use p3_air::WindowAccess;
        let main = builder.main();
        let a: [ArithmeticExpression<F>; 4] = array::from_fn(|i| main.current(i).unwrap().into());
        let b: [ArithmeticExpression<F>; 4] =
            array::from_fn(|i| main.current(4 + i).unwrap().into());
        let c: [ArithmeticExpression<F>; 4] =
            array::from_fn(|i| main.current(8 + i).unwrap().into());
        let mut ab: [ArithmeticExpression<F>; 4] = array::from_fn(|_| ArithmeticExpression::ZERO);
        qm31_mul_min_rank::<F, _>(&a, &b, &mut ab);
        builder.assert_eq_arrays(ab, c);
    });
    let lowered_qm31: [SymbolicExpression<F>; 4] = arith_captured
        .lower()
        .unwrap()
        .base_constraints()
        .to_vec()
        .try_into()
        .unwrap();
    assert_eq!(count_symbolic_dag(&lowered_qm31).quadratic_muls, 7);

    let mut rng = SmallRng::seed_from_u64(0x7FFF_FFFF);
    let prod_refs: [&SymbolicExpression<F>; 4] = array::from_fn(|i| &prod_roots[i]);
    let sq_refs: [&SymbolicExpression<F>; 4] = array::from_fn(|i| &sq_roots[i]);
    for _ in 0..64 {
        let va: [F; 4] = array::from_fn(|_| rng.random());
        let vb: [F; 4] = array::from_fn(|_| rng.random());
        let mut row = Vec::with_capacity(8);
        row.extend_from_slice(&va);
        row.extend_from_slice(&vb);
        let builder = RowEvalBuilder::new(row);
        let qa = <QM31 as BasedVectorSpace<F>>::from_basis_coefficients_slice(&va).unwrap();
        let qb = <QM31 as BasedVectorSpace<F>>::from_basis_coefficients_slice(&vb).unwrap();

        assert_eq!(
            resolve_symbolic_expressions(&prod_refs, &builder).as_slice(),
            <QM31 as BasedVectorSpace<F>>::as_basis_coefficients_slice(&(qa * qb))
        );
        assert_eq!(
            resolve_symbolic_expressions(&sq_refs, &builder).as_slice(),
            <QM31 as BasedVectorSpace<F>>::as_basis_coefficients_slice(&qa.square())
        );
        let arith_eval: [F; 4] = array::from_fn(|i| arith_prod[i].resolve(&builder));
        assert_eq!(
            &arith_eval,
            <QM31 as BasedVectorSpace<F>>::as_basis_coefficients_slice(&(qa * qb))
        );
    }
}

#[test]
fn complete_ast_node_census_before_vs_after() {
    fn row_binomial<F: BinomiallyExtendable<D>, const D: usize>(label: &str) {
        let w = <F as BinomiallyExtendable<D>>::W;
        let sym_a: [SymbolicExpression<F>; D] =
            array::from_fn(|i| SymbolicVariable::new(BaseEntry::Main { offset: 0 }, i).into());
        let sym_b: [SymbolicExpression<F>; D] =
            array::from_fn(|i| SymbolicVariable::new(BaseEntry::Main { offset: 0 }, D + i).into());

        let before_mul = count_symbolic_dag(&schoolbook_binomial_sym(&sym_a, &sym_b, w));
        let mut before_sq_roots: [SymbolicExpression<F>; D] =
            array::from_fn(|_| SymbolicExpression::ZERO);
        binomial_square(&sym_a, &mut before_sq_roots, w);
        let before_sq = count_symbolic_dag(&before_sq_roots);

        let ext_a = BinomialExtensionField::<F, D, SymbolicExpression<F>>::new(sym_a);
        let ext_b = BinomialExtensionField::<F, D, SymbolicExpression<F>>::new(sym_b);
        let after_mul_roots: [SymbolicExpression<F>; D] = (ext_a.clone() * ext_b)
            .as_basis_coefficients_slice()
            .to_vec()
            .try_into()
            .unwrap();
        let after_sq_roots: [SymbolicExpression<F>; D] = ext_a
            .square()
            .as_basis_coefficients_slice()
            .to_vec()
            .try_into()
            .unwrap();
        let after_mul = count_symbolic_dag(&after_mul_roots);
        let after_sq = count_symbolic_dag(&after_sq_roots);

        let arith_a: [ArithmeticExpression<F>; D] =
            array::from_fn(|i| ArithmeticVariable::new(BaseEntry::Main { offset: 0 }, i).into());
        let arith_b: [ArithmeticExpression<F>; D] = array::from_fn(|i| {
            ArithmeticVariable::new(BaseEntry::Main { offset: 0 }, D + i).into()
        });
        let ea = BinomialExtensionField::<F, D, ArithmeticExpression<F>>::new(arith_a);
        let eb = BinomialExtensionField::<F, D, ArithmeticExpression<F>>::new(arith_b);
        let am_roots: [ArithmeticExpression<F>; D] = (ea.clone() * eb)
            .as_basis_coefficients_slice()
            .to_vec()
            .try_into()
            .unwrap();
        let as_roots: [ArithmeticExpression<F>; D] = ea
            .square()
            .as_basis_coefficients_slice()
            .to_vec()
            .try_into()
            .unwrap();
        let am = count_arithmetic_dag(&am_roots);
        let asq = count_arithmetic_dag(&as_roots);

        std::println!(
            "{label:<36} | Mul Q-Muls: {:>2} -> {:>2} (Add/Sub {:>3} -> {:>3}) | Sq Q-Muls: {:>2} -> {:>2} | Arith Mul(m,sq,md): ({},{},{}) Sq: ({},{},{})",
            before_mul.quadratic_muls,
            after_mul.quadratic_muls,
            before_mul.add_subs,
            after_mul.add_subs,
            before_sq.quadratic_muls,
            after_sq.quadratic_muls,
            am.muls,
            am.squares,
            am.mixed_dots,
            asq.muls,
            asq.squares,
            asq.mixed_dots,
        );
    }

    row_binomial::<Mersenne31, 2>("Complex<Mersenne31> (D=2)");
    row_binomial::<Goldilocks, 2>("Goldilocks Binomial (D=2)");
    row_binomial::<Mersenne31, 3>("Mersenne31 Binomial (D=3)");
    row_binomial::<BabyBear, 4>("BabyBear Binomial (D=4)");
    row_binomial::<KoalaBear, 4>("KoalaBear Binomial (D=4)");
    row_binomial::<BabyBear, 5>("BabyBear Binomial (D=5)");
    row_binomial::<Goldilocks, 5>("Goldilocks Binomial (D=5)");
    row_binomial::<BabyBear, 8>("BabyBear Binomial (D=8)");
    row_binomial::<KoalaBear, 8>("KoalaBear Binomial (D=8)");

    // CubicTrinomial<Goldilocks>
    {
        type F = Goldilocks;
        let sym_a: [SymbolicExpression<F>; 3] =
            array::from_fn(|i| SymbolicVariable::new(BaseEntry::Main { offset: 0 }, i).into());
        let sym_b: [SymbolicExpression<F>; 3] =
            array::from_fn(|i| SymbolicVariable::new(BaseEntry::Main { offset: 0 }, 3 + i).into());
        let mut bm = array::from_fn(|_| SymbolicExpression::ZERO);
        trinomial_cubic_mul(&sym_a, &sym_b, &mut bm);
        let mut bs = array::from_fn(|_| SymbolicExpression::ZERO);
        cubic_square(&sym_a, &mut bs);
        let ea = CubicTrinomialExtensionField::<F, SymbolicExpression<F>>::new(sym_a);
        let eb = CubicTrinomialExtensionField::<F, SymbolicExpression<F>>::new(sym_b);
        let am: [SymbolicExpression<F>; 3] = (ea.clone() * eb)
            .as_basis_coefficients_slice()
            .to_vec()
            .try_into()
            .unwrap();
        let asq: [SymbolicExpression<F>; 3] = ea
            .square()
            .as_basis_coefficients_slice()
            .to_vec()
            .try_into()
            .unwrap();
        std::println!(
            "{:<36} | Mul Q-Muls: {:>2} -> {:>2} (Add/Sub {:>3} -> {:>3}) | Sq Q-Muls: {:>2} -> {:>2}",
            "Goldilocks CubicTrinomial (D=3)",
            count_symbolic_dag(&bm).quadratic_muls,
            count_symbolic_dag(&am).quadratic_muls,
            count_symbolic_dag(&bm).add_subs,
            count_symbolic_dag(&am).add_subs,
            count_symbolic_dag(&bs).quadratic_muls,
            count_symbolic_dag(&asq).quadratic_muls,
        );
    }

    // QuinticTrinomial<KoalaBear>
    {
        type F = KoalaBear;
        let sym_a: [SymbolicExpression<F>; 5] =
            array::from_fn(|i| SymbolicVariable::new(BaseEntry::Main { offset: 0 }, i).into());
        let sym_b: [SymbolicExpression<F>; 5] =
            array::from_fn(|i| SymbolicVariable::new(BaseEntry::Main { offset: 0 }, 5 + i).into());
        let mut bm = array::from_fn(|_| SymbolicExpression::ZERO);
        trinomial_quintic_mul(&sym_a, &sym_b, &mut bm);
        let mut bs = array::from_fn(|_| SymbolicExpression::ZERO);
        quintic_square(&sym_a, &mut bs);
        let ea = QuinticTrinomialExtensionField::<F, SymbolicExpression<F>>::new(sym_a);
        let eb = QuinticTrinomialExtensionField::<F, SymbolicExpression<F>>::new(sym_b);
        let am: [SymbolicExpression<F>; 5] = (ea.clone() * eb)
            .as_basis_coefficients_slice()
            .to_vec()
            .try_into()
            .unwrap();
        let asq: [SymbolicExpression<F>; 5] = ea
            .square()
            .as_basis_coefficients_slice()
            .to_vec()
            .try_into()
            .unwrap();
        std::println!(
            "{:<36} | Mul Q-Muls: {:>2} -> {:>2} (Add/Sub {:>3} -> {:>3}) | Sq Q-Muls: {:>2} -> {:>2}",
            "KoalaBear QuinticTrinomial (D=5)",
            count_symbolic_dag(&bm).quadratic_muls,
            count_symbolic_dag(&am).quadratic_muls,
            count_symbolic_dag(&bm).add_subs,
            count_symbolic_dag(&am).add_subs,
            count_symbolic_dag(&bs).quadratic_muls,
            count_symbolic_dag(&asq).quadratic_muls,
        );
    }

    // QM31 over Mersenne31
    {
        type F = Mersenne31;
        let a: [SymbolicExpression<F>; 4] =
            array::from_fn(|i| SymbolicVariable::new(BaseEntry::Main { offset: 0 }, i).into());
        let b: [SymbolicExpression<F>; 4] =
            array::from_fn(|i| SymbolicVariable::new(BaseEntry::Main { offset: 0 }, 4 + i).into());
        // Schoolbook QM31 (16 muls):
        let wb2 = b[2].clone() + b[2].clone() - b[3].clone();
        let wb3 = b[2].clone() + b[3].clone() + b[3].clone();
        let schoolbook_qm31 = [
            a[0].clone() * b[0].clone() - a[1].clone() * b[1].clone() + a[2].clone() * wb2.clone()
                - a[3].clone() * wb3.clone(),
            a[0].clone() * b[1].clone()
                + a[1].clone() * b[0].clone()
                + a[2].clone() * wb3
                + a[3].clone() * wb2,
            a[0].clone() * b[2].clone() - a[1].clone() * b[3].clone() + a[2].clone() * b[0].clone()
                - a[3].clone() * b[1].clone(),
            a[0].clone() * b[3].clone()
                + a[1].clone() * b[2].clone()
                + a[2].clone() * b[1].clone()
                + a[3].clone() * b[0].clone(),
        ];
        assert_eq!(count_symbolic_dag(&schoolbook_qm31).quadratic_muls, 16);

        let mut am = array::from_fn(|_| SymbolicExpression::ZERO);
        qm31_mul_min_rank::<F, _>(&a, &b, &mut am);
        let mut asq = array::from_fn(|_| SymbolicExpression::ZERO);
        qm31_square_min_rank::<F, _>(&a, &mut asq);
        std::println!(
            "{:<36} | Mul Q-Muls: {:>2} -> {:>2} (Add/Sub {:>3} -> {:>3}) | Sq Q-Muls: {:>2} -> {:>2}",
            "Mersenne31 QM31 Tower (D=4)",
            count_symbolic_dag(&schoolbook_qm31).quadratic_muls,
            count_symbolic_dag(&am).quadratic_muls,
            count_symbolic_dag(&schoolbook_qm31).add_subs,
            count_symbolic_dag(&am).add_subs,
            8,
            count_symbolic_dag(&asq).quadratic_muls,
        );
    }
}
