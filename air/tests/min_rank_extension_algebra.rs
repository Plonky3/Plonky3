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
    fn assert_binomial_counts<F: BinomiallyExtendable<D>, const D: usize>(
        expected_mul_q: (usize, usize),
        expected_sq_q: (usize, usize),
        expected_arith_mul: (usize, usize),
        expected_arith_sq: (usize, usize),
    ) {
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

        assert_eq!(
            (before_mul.quadratic_muls, after_mul.quadratic_muls),
            expected_mul_q
        );
        assert_eq!(
            (before_sq.quadratic_muls, after_sq.quadratic_muls),
            expected_sq_q
        );

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
        assert_eq!((am.muls, am.squares), expected_arith_mul);
        assert_eq!((asq.muls, asq.squares), expected_arith_sq);
    }

    assert_binomial_counts::<Mersenne31, 2>((4, 3), (3, 2), (3, 0), (2, 0));
    assert_binomial_counts::<Goldilocks, 2>((4, 3), (3, 2), (3, 0), (2, 0));
    assert_binomial_counts::<Mersenne31, 3>((9, 5), (9, 5), (5, 0), (0, 5));
    assert_binomial_counts::<BabyBear, 4>((16, 7), (10, 7), (7, 0), (0, 7));
    assert_binomial_counts::<KoalaBear, 4>((16, 7), (10, 7), (7, 0), (0, 7));
    assert_binomial_counts::<BabyBear, 5>((25, 9), (15, 9), (9, 0), (0, 9));
    assert_binomial_counts::<Goldilocks, 5>((25, 9), (15, 9), (9, 0), (0, 9));
    assert_binomial_counts::<BabyBear, 8>((64, 15), (36, 15), (15, 0), (0, 15));
    assert_binomial_counts::<KoalaBear, 8>((64, 15), (36, 15), (15, 0), (0, 15));
}

#[test]
fn extension_expr_and_arith_ext_min_rank_and_lowering() {
    use p3_air::arithmetic::{ArithmeticExpressionExt, ArithmeticVariableExt};
    use p3_air::{ExtEntry, SymbolicExpressionExt, SymbolicVariableExt};

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;

    let sym_ext_a: [SymbolicExpressionExt<F, EF>; 4] = array::from_fn(|i| {
        SymbolicVariableExt::<F, EF>::new(ExtEntry::Permutation { offset: 0 }, i).into()
    });
    let sym_ext_b: [SymbolicExpressionExt<F, EF>; 4] = array::from_fn(|i| {
        if i == 3 {
            // Mix a non-base extension constant (deg 0) with permutation variables (deg 1).
            SymbolicExpressionExt::from(EF::new([
                F::from_u8(2),
                F::from_u8(3),
                F::from_u8(5),
                F::from_u8(7),
            ]))
        } else {
            SymbolicVariable::new(BaseEntry::Main { offset: 0 }, i).into()
        }
    });

    let ea = BinomialExtensionField::<F, 4, SymbolicExpressionExt<F, EF>>::new(sym_ext_a);
    let eb = BinomialExtensionField::<F, 4, SymbolicExpressionExt<F, EF>>::new(sym_ext_b);
    let prod = ea.clone() * eb;
    let sq = ea.square();
    for c in prod.as_basis_coefficients_slice() {
        assert_eq!(c.degree_multiple(), 2);
        assert_eq!(c.poly_degree(16, &[]), 30);
    }
    for c in sq.as_basis_coefficients_slice() {
        assert_eq!(c.degree_multiple(), 2);
        assert_eq!(c.poly_degree(16, &[]), 30);
    }

    let arith_ext_a: [ArithmeticExpressionExt<F, EF>; 4] = array::from_fn(|i| {
        ArithmeticVariableExt::<F, EF>::new(ExtEntry::Permutation { offset: 0 }, i).into()
    });
    let arith_ext_b: [ArithmeticExpressionExt<F, EF>; 4] = array::from_fn(|i| {
        if i == 3 {
            ArithmeticExpressionExt::from_extension(EF::new([
                F::from_u8(2),
                F::from_u8(3),
                F::from_u8(5),
                F::from_u8(7),
            ]))
        } else {
            ArithmeticVariable::new(BaseEntry::Main { offset: 0 }, i).into()
        }
    });
    let aea = BinomialExtensionField::<F, 4, ArithmeticExpressionExt<F, EF>>::new(arith_ext_a);
    let aeb = BinomialExtensionField::<F, 4, ArithmeticExpressionExt<F, EF>>::new(arith_ext_b);
    let aprod = aea * aeb;
    let ext_roots: Vec<ArithmeticExpressionExt<F, EF>> =
        aprod.as_basis_coefficients_slice().to_vec();
    for c in &ext_roots {
        assert_eq!(c.degree_multiple(), 2);
        assert_eq!(c.poly_degree(16, &[]), 30);
    }

    let captured = CapturedArithmeticConstraints::<F, EF> {
        base: vec![],
        extension: ext_roots,
        layout: ConstraintLayout {
            base_indices: vec![],
            ext_indices: (0..4).collect(),
        },
    };
    let lowered = captured.lower().unwrap();
    for c in lowered.extension_constraints() {
        assert_eq!(c.degree_multiple(), 2);
        assert_eq!(c.poly_degree(16, &[]), 30);
    }
}

#[test]
fn stark_degree_soundness_mixed_degrees_selectors_flatten_and_next_row() {
    use p3_air::{
        BaseAir, NextRowColumns, NextRowTrace, SymbolicConstraints, UndeclaredNextRowRead,
    };

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;
    const N: usize = 16;

    // Mix degree-0 constant, degree-1 main column, degree-2 quadratic term, IsFirstRow,
    // IsTransition, and a period-4 periodic column!
    let c0 = SymbolicExpression::<F>::from(F::from_u8(9));
    let v0: SymbolicExpression<F> = SymbolicVariable::new(BaseEntry::Main { offset: 0 }, 0).into();
    let v1_next: SymbolicExpression<F> =
        SymbolicVariable::new(BaseEntry::Main { offset: 1 }, 1).into();
    let q2 = v0.clone() * v1_next.clone(); // deg 2
    let is_trans = SymbolicExpression::<F>::Leaf(BaseLeaf::IsTransition); // dm=0, trans_dm=1, poly=1
    let is_first = SymbolicExpression::<F>::Leaf(BaseLeaf::IsFirstRow); // dm=1, poly=N-1
    let per0: SymbolicExpression<F> = SymbolicVariable::new(BaseEntry::Periodic, 0).into(); // dm=1, poly=N - N/4 = 12

    let a = BinomialExtensionField::<F, 4, SymbolicExpression<F>>::new([
        c0,                      // dm=0, t_dm=0, poly=0
        v0.clone(),              // dm=1, t_dm=1, poly=15
        q2,                      // dm=2, t_dm=2, poly=30
        is_trans.clone() * per0, // dm=1, t_dm=2, poly=13
    ]);
    let b = BinomialExtensionField::<F, 4, SymbolicExpression<F>>::new([
        is_first, // dm=1, t_dm=1, poly=15
        is_trans, // dm=0, t_dm=1, poly=1
        SymbolicExpression::ZERO,
        SymbolicExpression::ONE,
    ]);
    let prod = a * b;
    let coords = prod.as_basis_coefficients_slice();

    // Exact maximum bounds across coordinates:
    // d_A(dm)=2, d_B(dm)=1 => max dm == 3
    // d_A(t_dm=1)=2, d_B(t_dm=1)=1 => max t_dm == 3
    // d_A(poly)=30, d_B(poly)=15 => max poly == 45
    assert_eq!(coords.iter().map(|c| c.degree_multiple()).max(), Some(3));
    assert_eq!(
        coords
            .iter()
            .map(|c| c.degree_multiple_with_transition(1))
            .max(),
        Some(3)
    );
    assert_eq!(
        coords.iter().map(|c| c.poly_degree(N, &[4])).max(),
        Some(45)
    );

    // Leading-term cancellation must NEVER under-report degree:
    let cancel_lhs = v0.clone() * v1_next.clone() + v0.clone();
    let cancel_rhs = -(v0 * v1_next);
    let cancelled = <SymbolicExpression<F> as p3_field::Algebra<F>>::linear_combinations::<2, 1>(
        &[cancel_lhs, cancel_rhs],
        &[[F::ONE, F::ONE]],
    );
    assert!(cancelled[0].degree_multiple() >= 1);

    // JSON wire-flattening roundtrip preserves shared DAG structure, degree, and evaluation:
    for coord in coords {
        let encoded = serde_json::to_string(coord).unwrap();
        let decoded: SymbolicExpression<F> = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded.degree_multiple(), coord.degree_multiple());
        assert_eq!(
            decoded.degree_multiple_with_transition(1),
            coord.degree_multiple_with_transition(1)
        );
        assert_eq!(decoded.poly_degree(N, &[4]), coord.poly_degree(N, &[4]));
    }

    // Next-row offset-1 read inside ExtensionAlgebra product MUST be caught when undeclared:
    struct DummyAir(Vec<usize>);
    impl BaseAir<F> for DummyAir {
        fn width(&self) -> usize {
            3
        }
        fn main_next_row_columns(&self) -> Vec<usize> {
            self.0.clone()
        }
    }

    let layout = AirLayout {
        main_width: 3,
        num_periodic_columns: 1,
        ..Default::default()
    };
    let mut builder = SymbolicAirBuilder::<F, EF>::new(layout);
    for coord in coords {
        builder.assert_zero(coord.clone());
    }
    let constraints: SymbolicConstraints<F, EF> = builder.finish();
    assert_eq!(
        constraints.check_next_row_reads(&NextRowColumns::from_air(&DummyAir(vec![0]))),
        Err(UndeclaredNextRowRead {
            trace: NextRowTrace::Main,
            column: 1,
        })
    );
    assert_eq!(
        constraints.check_next_row_reads(&NextRowColumns::from_air(&DummyAir(vec![1]))),
        Ok(())
    );
}

#[test]
fn linear_combinations_edge_cases_zero_empty_aliased_and_neg_one() {
    use p3_field::Algebra;

    type F = Mersenne31;

    // 1. Empty input slice (R = 0): exact degree-0 ZERO on both Symbolic and Arithmetic.
    let sym_empty =
        <SymbolicExpression<F> as Algebra<F>>::linear_combinations::<0, 3>(&[], &[[]; 3]);
    let arith_empty =
        <ArithmeticExpression<F> as Algebra<F>>::linear_combinations::<0, 3>(&[], &[[]; 3]);
    let empty_stats = count_symbolic_dag(&sym_empty);
    assert_eq!(
        empty_stats,
        SymDagStats {
            quadratic_muls: 0,
            scalar_muls: 0,
            add_subs: 0,
            negs: 0,
        }
    );
    for (s, a) in sym_empty.iter().zip(arith_empty.iter()) {
        assert_eq!(s.degree_multiple(), 0);
        assert_eq!(a.degree_multiple(), 0);
        assert_eq!(a.lower().degree_multiple(), 0);
    }

    // 2. All-zero row, single +1 / -1 (`F::NEG_ONE`) rows, and +/-1 combinations:
    //    `F::NEG_ONE` must emit Neg / Sub with ZERO constant leaf multiplications (`scalar_muls == 0`)!
    let sx0: SymbolicExpression<F> = SymbolicVariable::new(BaseEntry::Main { offset: 0 }, 0).into();
    let sx1: SymbolicExpression<F> = SymbolicVariable::new(BaseEntry::Main { offset: 0 }, 1).into();
    let sq2 = sx0.clone() * sx1.clone(); // deg 2

    let ax0: ArithmeticExpression<F> =
        ArithmeticVariable::new(BaseEntry::Main { offset: 0 }, 0).into();
    let ax1: ArithmeticExpression<F> =
        ArithmeticVariable::new(BaseEntry::Main { offset: 0 }, 1).into();
    let aq2 = ax0.clone() * ax1.clone(); // deg 2

    let weights = [
        [F::ZERO, F::ZERO, F::ZERO],      // all-zero row -> deg 0
        [F::ZERO, F::NEG_ONE, F::ZERO],   // single -1 -> -x1 (Neg, deg 1, 0 scalar muls)
        [F::ONE, F::ZERO, F::NEG_ONE],    // +x0 - q2 (Sub, deg 2, 0 scalar muls)
        [F::NEG_ONE, F::NEG_ONE, F::ONE], // -x0 - x1 + q2 (deg 2, 0 scalar muls)
    ];

    let sym_out = <SymbolicExpression<F> as Algebra<F>>::linear_combinations::<3, 4>(
        &[sx0, sx1, sq2.clone()],
        &weights,
    );
    assert_eq!(sym_out[0].degree_multiple(), 0);
    assert_eq!(sym_out[1].degree_multiple(), 1);
    assert_eq!(sym_out[2].degree_multiple(), 2);
    assert_eq!(sym_out[3].degree_multiple(), 2);
    let sym_lc_stats = count_symbolic_dag(&sym_out);
    assert_eq!(sym_lc_stats.quadratic_muls, 1);
    assert_eq!(sym_lc_stats.scalar_muls, 0);

    let arith_out = <ArithmeticExpression<F> as Algebra<F>>::linear_combinations::<3, 4>(
        &[ax0, ax1, aq2.clone()],
        &weights,
    );
    assert_eq!(arith_out[0].degree_multiple(), 0);
    assert_eq!(arith_out[1].degree_multiple(), 1);
    assert_eq!(arith_out[2].degree_multiple(), 2);
    assert_eq!(arith_out[3].degree_multiple(), 2);
    assert!(matches!(arith_out[1].node(), ArithmeticNode::Neg(_)));

    let lowered_lc: [SymbolicExpression<F>; 4] = CapturedArithmeticConstraints::<F, F> {
        base: arith_out.to_vec(),
        extension: vec![],
        layout: ConstraintLayout {
            base_indices: (0..4).collect(),
            ext_indices: vec![],
        },
    }
    .lower()
    .unwrap()
    .base_constraints()
    .to_vec()
    .try_into()
    .unwrap();
    assert_eq!(lowered_lc[0].degree_multiple(), 0);
    assert_eq!(lowered_lc[1].degree_multiple(), 1);
    assert_eq!(lowered_lc[2].degree_multiple(), 2);
    assert_eq!(lowered_lc[3].degree_multiple(), 2);
    let lowered_lc_stats = count_symbolic_dag(&lowered_lc);
    assert_eq!(lowered_lc_stats.quadratic_muls, 1);
    assert_eq!(lowered_lc_stats.scalar_muls, 0);

    // 3. Duplicate / aliased expressions in `values`:
    //    - Passing `[sq2.clone(), sq2]` (two root Mul values across 3 rows) shares each slot's Arc
    //      across all 3 rows (`2` quadratic muls instead of `6`, `0` scalar muls).
    //    - Passing `[parent.clone(), parent]` where `parent = sq2 + x0` shares the inner `Arc<Mul>`
    //      across both slots AND all 3 rows (`1` quadratic mul total on both Symbolic and lowered Arithmetic!).
    let alias_weights = [
        [F::ONE, F::ONE],
        [F::ONE, F::NEG_ONE],
        [F::NEG_ONE, F::NEG_ONE],
    ];
    let sym_dup_roots = <SymbolicExpression<F> as Algebra<F>>::linear_combinations::<2, 3>(
        &[sq2.clone(), sq2.clone()],
        &alias_weights,
    );
    assert_eq!(count_symbolic_dag(&sym_dup_roots).quadratic_muls, 2);
    assert_eq!(count_symbolic_dag(&sym_dup_roots).scalar_muls, 0);

    let arith_dup_roots = <ArithmeticExpression<F> as Algebra<F>>::linear_combinations::<2, 3>(
        &[aq2.clone(), aq2.clone()],
        &alias_weights,
    );
    let lowered_dup_roots: [SymbolicExpression<F>; 3] = CapturedArithmeticConstraints::<F, F> {
        base: arith_dup_roots.to_vec(),
        extension: vec![],
        layout: ConstraintLayout {
            base_indices: (0..3).collect(),
            ext_indices: vec![],
        },
    }
    .lower()
    .unwrap()
    .base_constraints()
    .to_vec()
    .try_into()
    .unwrap();
    assert_eq!(count_symbolic_dag(&lowered_dup_roots).quadratic_muls, 2);
    assert_eq!(count_symbolic_dag(&lowered_dup_roots).scalar_muls, 0);

    let sym_parent = sq2 + SymbolicVariable::new(BaseEntry::Main { offset: 0 }, 0);
    let arith_parent = aq2 + ArithmeticVariable::new(BaseEntry::Main { offset: 0 }, 0);

    let sym_alias = <SymbolicExpression<F> as Algebra<F>>::linear_combinations::<2, 3>(
        &[sym_parent.clone(), sym_parent],
        &alias_weights,
    );
    let arith_alias = <ArithmeticExpression<F> as Algebra<F>>::linear_combinations::<2, 3>(
        &[arith_parent.clone(), arith_parent],
        &alias_weights,
    );
    let lowered_alias: [SymbolicExpression<F>; 3] = CapturedArithmeticConstraints::<F, F> {
        base: arith_alias.to_vec(),
        extension: vec![],
        layout: ConstraintLayout {
            base_indices: (0..3).collect(),
            ext_indices: vec![],
        },
    }
    .lower()
    .unwrap()
    .base_constraints()
    .to_vec()
    .try_into()
    .unwrap();
    assert_eq!(count_symbolic_dag(&sym_alias).quadratic_muls, 1);
    assert_eq!(count_symbolic_dag(&sym_alias).scalar_muls, 0);
    assert_eq!(count_symbolic_dag(&lowered_alias).quadratic_muls, 1);
    assert_eq!(count_symbolic_dag(&lowered_alias).scalar_muls, 0);
    for r in sym_alias.iter().chain(lowered_alias.iter()) {
        assert_eq!(r.degree_multiple(), 2);
    }

    // Verify evaluation agreement across random rows on all edge-case outputs.
    let mut rng = SmallRng::seed_from_u64(0xED6E_CA5E);
    let sym_refs: [&SymbolicExpression<F>; 4] = array::from_fn(|i| &sym_out[i]);
    let low_refs: [&SymbolicExpression<F>; 4] = array::from_fn(|i| &lowered_lc[i]);
    let sym_alias_refs: [&SymbolicExpression<F>; 3] = array::from_fn(|i| &sym_alias[i]);
    let low_alias_refs: [&SymbolicExpression<F>; 3] = array::from_fn(|i| &lowered_alias[i]);
    for _ in 0..32 {
        let v0: F = rng.random();
        let v1: F = rng.random();
        let q = v0 * v1;
        let p_val = q + v0;
        let builder = RowEvalBuilder::new(vec![v0, v1]);
        let expected = [F::ZERO, -v1, v0 - q, -v0 - v1 + q];
        assert_eq!(resolve_symbolic_expressions(&sym_refs, &builder), expected);
        assert_eq!(resolve_symbolic_expressions(&low_refs, &builder), expected);
        let arith_eval: [F; 4] = array::from_fn(|i| arith_out[i].resolve(&builder));
        assert_eq!(arith_eval, expected);

        let expected_alias = [p_val + p_val, F::ZERO, -(p_val + p_val)];
        assert_eq!(
            resolve_symbolic_expressions(&sym_alias_refs, &builder),
            expected_alias
        );
        assert_eq!(
            resolve_symbolic_expressions(&low_alias_refs, &builder),
            expected_alias
        );
    }

    // 4. Complex<Mersenne31> (D = 2, W = -1 = F::NEG_ONE): both SymbolicExpression and lowered
    //    ArithmeticExpression emit 3 quadratic muls and ZERO scalar muls!
    let ca: [ArithmeticExpression<F>; 2] =
        array::from_fn(|i| ArithmeticVariable::new(BaseEntry::Main { offset: 0 }, i).into());
    let cb: [ArithmeticExpression<F>; 2] =
        array::from_fn(|i| ArithmeticVariable::new(BaseEntry::Main { offset: 0 }, 2 + i).into());
    let prod_c = BinomialExtensionField::<F, 2, ArithmeticExpression<F>>::new(ca)
        * BinomialExtensionField::<F, 2, ArithmeticExpression<F>>::new(cb);
    let lowered_c: [SymbolicExpression<F>; 2] = CapturedArithmeticConstraints::<F, F> {
        base: prod_c.as_basis_coefficients_slice().to_vec(),
        extension: vec![],
        layout: ConstraintLayout {
            base_indices: vec![0, 1],
            ext_indices: vec![],
        },
    }
    .lower()
    .unwrap()
    .base_constraints()
    .to_vec()
    .try_into()
    .unwrap();
    let stats_c = count_symbolic_dag(&lowered_c);
    assert_eq!(stats_c.quadratic_muls, 3);
    assert_eq!(stats_c.scalar_muls, 0);
}
