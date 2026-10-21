use p3_air::arithmetic::{
    ArithmeticExpression, ArithmeticExpressionExt, ArithmeticExtLeaf, ArithmeticNode,
    ArithmeticVariable, ArithmeticVariableExt, BaseOperand, ExtOperand,
};
use p3_air::{
    AirBuilder, AirLayout, ArithmeticAirBuilder, BaseEntry, ExtEntry, ExtLeaf, ExtensionBuilder,
    PermutationAirBuilder, SymbolicExpr, SymbolicExpressionExt, WindowAccess,
};
use p3_baby_bear::BabyBear as F;
use p3_field::extension::BinomialExtensionField;
use p3_field::{Algebra, BasedVectorSpace, PrimeCharacteristicRing};
use p3_matrix::dense::RowMajorMatrix;
use p3_mersenne_31::{Mersenne31, QM31};

#[test]
fn captures_square_sum_dot_and_scalar_coefficients() {
    let x = ArithmeticVariable::<F>::new(BaseEntry::Main { offset: 0 }, 0);
    let e: ArithmeticExpression<F> = x.into();
    assert!(matches!(e.square().node(), ArithmeticNode::Square(_)));
    assert!(matches!(
        ArithmeticExpression::sum_array::<2>(&[e.clone(), e.clone()]).node(),
        ArithmeticNode::Sum(_)
    ));
    assert!(matches!(
        ArithmeticExpression::dot_product(core::array::from_ref(&e), core::array::from_ref(&e))
            .node(),
        ArithmeticNode::Dot(_)
    ));
    let mixed = ArithmeticExpression::mixed_dot_product(&[e], &[F::TWO]);
    assert!(
        matches!(mixed.node(), ArithmeticNode::MixedDot(p) if matches!(p[0].1, BaseOperand::Scalar(_)))
    );
    assert_eq!((x * F::ZERO).degree_multiple(), 0);
}

#[test]
fn lowering_preserves_values_and_shared_squares() {
    let x = ArithmeticVariable::<F>::new(BaseEntry::Main { offset: 0 }, 0);
    let mut e = ArithmeticExpression::from(x);
    for _ in 0..12 {
        e = e.square();
    }
    assert_eq!(e.degree_multiple(), 4096);
    let old = e.lower();
    assert_eq!(old.degree_multiple(), 4096);
}

#[test]
fn replay_calls_square_on_target_expression() {
    let x = ArithmeticVariable::<F>::new(BaseEntry::Main { offset: 0 }, 0);
    let capture = p3_air::ArithmeticAirBuilder::<F>::new(p3_air::AirLayout {
        main_width: 1,
        ..Default::default()
    });
    assert!(matches!(
        ArithmeticExpression::from(x)
            .square()
            .resolve(&capture)
            .node(),
        ArithmeticNode::Square(_)
    ));
}

type EF = BinomialExtensionField<F, 4>;
type E = ArithmeticExpression<F>;
type EE = ArithmeticExpressionExt<F, EF>;

struct Concrete {
    main: RowMajorMatrix<F>,
    prep: RowMajorMatrix<F>,
    public: [F; 1],
    periodic: [F; 1],
    perm: RowMajorMatrix<EF>,
    challenges: [EF; 1],
    values: [EF; 1],
    selectors: [F; 3],
}
impl Concrete {
    fn new(seed: u32) -> Self {
        let f = |n| F::new(seed + n);
        let e = |n| EF::from_basis_coefficients_fn(|i| f(n + i as u32));
        Self {
            main: RowMajorMatrix::new(vec![f(1), f(2)], 1),
            prep: RowMajorMatrix::new(vec![f(3), f(4)], 1),
            public: [f(5)],
            periodic: [f(6)],
            perm: RowMajorMatrix::new(vec![e(7), e(8)], 1),
            challenges: [e(9)],
            values: [e(10)],
            selectors: [f(11), f(12), f(13)],
        }
    }
}
impl AirBuilder for Concrete {
    type F = F;
    type Expr = F;
    type Var = F;
    type MainWindow = RowMajorMatrix<F>;
    type PreprocessedWindow = RowMajorMatrix<F>;
    type PublicVar = F;
    type PeriodicVar = F;
    fn main(&self) -> Self::MainWindow {
        self.main.clone()
    }
    fn preprocessed(&self) -> &Self::PreprocessedWindow {
        &self.prep
    }
    fn public_values(&self) -> &[F] {
        &self.public
    }
    fn periodic_values(&self) -> &[F] {
        &self.periodic
    }
    fn is_first_row(&self) -> F {
        self.selectors[0]
    }
    fn is_last_row(&self) -> F {
        self.selectors[1]
    }
    fn is_transition(&self) -> F {
        self.selectors[2]
    }
    fn assert_zero<I: Into<F>>(&mut self, _: I) {
        unreachable!()
    }
}
impl ExtensionBuilder for Concrete {
    type EF = EF;
    type ExprEF = EF;
    type VarEF = EF;
    fn assert_zero_ext<I: Into<EF>>(&mut self, _: I) {
        unreachable!()
    }
}
impl PermutationAirBuilder for Concrete {
    type MP = RowMajorMatrix<EF>;
    type RandomVar = EF;
    type PermutationVar = EF;
    fn permutation(&self) -> Self::MP {
        self.perm.clone()
    }
    fn permutation_randomness(&self) -> &[EF] {
        &self.challenges
    }
    fn permutation_values(&self) -> &[EF] {
        &self.values
    }
}
fn eval_legacy(e: &SymbolicExpressionExt<F, EF>, b: &Concrete) -> EF {
    match e {
        SymbolicExpr::Leaf(ExtLeaf::Base(e)) => e.resolve(b).into(),
        SymbolicExpr::Leaf(ExtLeaf::ExtConstant(c)) => *c,
        SymbolicExpr::Leaf(ExtLeaf::ExtVariable(v)) => match v.entry {
            ExtEntry::Permutation { offset } => b.perm.values[offset],
            ExtEntry::Challenge => b.challenges[v.index],
            ExtEntry::PermutationValue => b.values[v.index],
        },
        SymbolicExpr::Add { x, y, .. } => eval_legacy(x, b) + eval_legacy(y, b),
        SymbolicExpr::Sub { x, y, .. } => eval_legacy(x, b) - eval_legacy(y, b),
        SymbolicExpr::Mul { x, y, .. } => eval_legacy(x, b) * eval_legacy(y, b),
        SymbolicExpr::Neg { x, .. } => -eval_legacy(x, b),
    }
}
const fn layout() -> AirLayout {
    AirLayout {
        main_width: 1,
        preprocessed_width: 1,
        num_public_values: 1,
        num_periodic_columns: 1,
        permutation_width: 1,
        num_permutation_challenges: 1,
        num_permutation_values: 1,
    }
}

#[test]
fn every_leaf_resolves_and_lowers_at_arbitrary_values() {
    let b = ArithmeticAirBuilder::<F, EF>::new(layout());
    let mut base = vec![b.is_first_row(), b.is_last_row(), b.is_transition()];
    for v in b
        .main()
        .values
        .iter()
        .chain(b.preprocessed().values.iter())
        .chain(b.public_values())
        .chain(b.periodic_values())
    {
        base.push((*v).into());
    }
    let mut ext: Vec<EE> = b
        .permutation()
        .values
        .iter()
        .chain(b.permutation_randomness())
        .chain(b.permutation_values())
        .map(|v| (*v).into())
        .collect();
    let c = EF::from_basis_coefficients_fn(|i| F::new(i as u32 + 17));
    ext.push(EE::from_extension(c));
    for seed in 0..17 {
        let concrete = Concrete::new(seed);
        let expected = [
            concrete.selectors.to_vec(),
            concrete.main.values.clone(),
            concrete.prep.values.clone(),
            concrete.public.to_vec(),
            concrete.periodic.to_vec(),
        ]
        .concat();
        for (e, expected) in base.iter().zip(expected) {
            assert_eq!(e.resolve(&concrete), expected);
            assert_eq!(e.lower().resolve(&concrete), expected);
        }
        let expected = [
            concrete.perm.values.clone(),
            concrete.challenges.to_vec(),
            concrete.values.to_vec(),
            vec![c],
        ]
        .concat();
        for (e, expected) in ext.iter().zip(expected) {
            assert_eq!(e.resolve(&concrete), expected);
            assert_eq!(eval_legacy(&e.lower(), &concrete), expected);
        }
    }
}

#[test]
fn arithmetic_assignments_and_mixed_domains_match_direct_field_arithmetic() {
    let b = ArithmeticAirBuilder::<F, EF>::new(layout());
    let x = b.main().values[0];
    let y = b.preprocessed().values[1];
    let z = b.permutation().values[0];
    let challenge = b.permutation_randomness()[0];
    let ec = EF::from_basis_coefficients_fn(|i| F::new(i as u32 + 2));
    let mut base = x + y;
    base *= F::TWO;
    base -= x;
    base += F::ONE;
    let mut ext = EE::from(z);
    ext += challenge;
    ext *= base.clone();
    ext -= ec;
    ext += F::TWO;
    let a = [ext.clone(), EE::from(z).square()];
    let s = EE::sum_array::<2>(&a);
    let d = EE::dot_product(&a, &[EE::from(challenge), EE::from_extension(ec)]);
    let mf = <EE as Algebra<F>>::mixed_dot_product(&a, &[F::TWO, F::new(3)]);
    let me = <EE as Algebra<EF>>::mixed_dot_product(&a, &[ec, EF::TWO]);
    let mb = <EE as Algebra<E>>::mixed_dot_product(&a, &[base.clone(), E::from(x)]);
    assert!(
        matches!(mf.node(),ArithmeticNode::MixedDot(p) if matches!(p[0].1,ExtOperand::BaseScalar(_)))
    );
    assert!(
        matches!(me.node(),ArithmeticNode::MixedDot(p) if matches!(p[0].1,ExtOperand::ExtensionScalar(_)))
    );
    assert!(
        matches!(mb.node(),ArithmeticNode::MixedDot(p) if matches!(p[0].1,ExtOperand::Base(_)))
    );
    assert!(
        matches!(EE::from(ec).node(),ArithmeticNode::Leaf(ArithmeticExtLeaf::Constant(v)) if *v==ec)
    );
    for seed in 0..17 {
        let c = Concrete::new(seed);
        let xv = c.main.values[0];
        let yv = c.prep.values[1];
        let zv = c.perm.values[0];
        let cv = c.challenges[0];
        let bv = (xv + yv) * F::TWO - xv + F::ONE;
        let ev = (zv + cv) * bv - ec + F::TWO;
        let square = zv.square();
        assert_eq!(base.resolve(&c), bv);
        assert_eq!(ext.resolve(&c), ev);
        for (e, v) in [
            (&s, ev + square),
            (&d, ev * cv + square * ec),
            (&mf, ev * F::TWO + square * F::new(3)),
            (&me, ev * ec + square * EF::TWO),
            (&mb, ev * bv + square * xv),
        ] {
            assert_eq!(e.resolve(&c), v);
            assert_eq!(eval_legacy(&e.lower(), &c), v);
        }
    }
}

#[test]
fn array_edges_and_runtime_fallbacks() {
    let x = ArithmeticVariable::<F>::new(BaseEntry::Main { offset: 0 }, 0);
    let e = E::from(x);
    for n in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 32, 64, 65] {
        let a = vec![e.clone(); n];
        let coeff = vec![F::TWO; n];
        let sum: E = a.iter().cloned().sum();
        let dot = E::batched_linear_combination(&a, &coeff);
        let c = Concrete::new(3);
        let v = c.main.values[0];
        assert_eq!(sum.resolve(&c), v * F::from_usize(n));
        assert_eq!(dot.resolve(&c), v * F::TWO * F::from_usize(n));
    }
    assert_eq!(E::sum_array::<0>(&[]).resolve(&Concrete::new(0)), F::ZERO);
    assert_eq!(
        E::dot_product::<0>(&[], &[]).resolve(&Concrete::new(0)),
        F::ZERO
    );
    assert_eq!(EE::sum_array::<0>(&[]).resolve(&Concrete::new(0)), EF::ZERO);
    assert_eq!(
        EE::dot_product::<0>(&[], &[]).resolve(&Concrete::new(0)),
        EF::ZERO
    );
    assert_eq!(
        E::sum_array::<1>(core::array::from_ref(&e)).resolve(&Concrete::new(0)),
        F::ONE
    );
    assert_eq!(
        E::dot_product(core::array::from_ref(&e), core::array::from_ref(&e))
            .resolve(&Concrete::new(0)),
        F::ONE
    );
}
#[test]
#[should_panic(expected = "sum_array length mismatch")]
fn bad_sum_length_panics() {
    let _ = E::sum_array::<2>(&[E::ONE]);
}
#[test]
#[should_panic(expected = "linear combination length mismatch")]
fn bad_lc_length_panics() {
    let _ = <E as Algebra<F>>::batched_linear_combination(&[E::ONE], &[]);
}
#[test]
#[should_panic(expected = "linear combination length mismatch")]
fn bad_ext_lc_length_panics() {
    let _ = <EE as Algebra<EF>>::batched_linear_combination(&[EE::ONE], &[]);
}

#[test]
fn conservative_degrees_include_periodic_transition_and_zero_coefficients() {
    let b = ArithmeticAirBuilder::<F, EF>::new(layout());
    let p = E::from(b.periodic_values()[0]);
    let x = E::from(b.main().values[0]);
    let tr = b.is_transition();
    let e = (p * x.clone()).square() * tr;
    assert_eq!(e.degree_multiple(), 4);
    assert_eq!(e.degree_multiple_with_transition(3), 7);
    assert_eq!(e.poly_degree(8, &[2]), 23);
    let mixed = E::mixed_dot_product(&[e.clone(), x.clone()], &[F::ZERO, F::ONE]);
    assert_eq!(mixed.degree_multiple(), 1);
    assert_eq!(mixed.poly_degree(8, &[2]), 7);
    let lifted: EE = e.into();
    assert_eq!(lifted.poly_degree(8, &[2]), 23);
    assert_eq!(lifted.degree_multiple_with_transition(3), 7);
    assert_eq!((x.clone() * F::ZERO).degree_multiple(), 0);
    assert_eq!((EE::from(x) * EF::ZERO).degree_multiple(), 0);
    assert_eq!(b.is_first_row().poly_degree(8, &[]), 7);
    assert_eq!(b.is_last_row().degree_multiple(), 1);
}
#[test]
#[should_panic(expected = "arithmetic degree overflow")]
fn degree_overflow_panics_instead_of_underbounding() {
    let mut e = E::from(ArithmeticVariable::new(BaseEntry::Main { offset: 0 }, 0));
    for _ in 0..usize::BITS {
        e = e.square();
    }
}

#[test]
fn captured_order_keeps_zero_and_duplicate_assertions() {
    let c = p3_air::capture_arithmetic::<F, EF>(layout(), |b| {
        let x = b.main().values[0];
        b.assert_zero(x);
        b.assert_zero_ext(EE::ZERO);
        b.assert_zero(x);
        b.assert_zero(F::ZERO);
        b.assert_zero_ext(EE::ZERO);
    });
    assert_eq!(c.base.len(), 3);
    assert_eq!(c.extension.len(), 2);
    assert_eq!(c.layout.base_indices, vec![0, 2, 3]);
    assert_eq!(c.layout.ext_indices, vec![1, 4]);
}

#[test]
fn replay_retains_mixed_domains_and_primitive_kinds() {
    let b = ArithmeticAirBuilder::<F, EF>::new(layout());
    let x = E::from(b.main().values[0]);
    let z = EE::from(b.permutation().values[0]);
    let s = E::sum_array::<2>(&[x.clone(), x.clone()]);
    assert!(matches!(s.resolve(&b).node(), ArithmeticNode::Sum(_)));
    let d = E::dot_product(core::array::from_ref(&x), core::array::from_ref(&x));
    assert!(matches!(d.resolve(&b).node(), ArithmeticNode::Dot(_)));
    let m = E::mixed_dot_product(core::array::from_ref(&x), &[F::TWO]);
    assert!(
        matches!(m.resolve(&b).node(),ArithmeticNode::MixedDot(p) if matches!(p[0].1,BaseOperand::Scalar(_)))
    );
    let me = <EE as Algebra<EF>>::mixed_dot_product(core::array::from_ref(&z), &[EF::TWO]);
    assert!(
        matches!(me.resolve(&b).node(),ArithmeticNode::MixedDot(p) if matches!(p[0].1,ExtOperand::ExtensionScalar(_)))
    );
    let mb = <EE as Algebra<E>>::mixed_dot_product(&[z], &[x]);
    assert!(
        matches!(mb.resolve(&b).node(),ArithmeticNode::MixedDot(p) if matches!(p[0].1,ExtOperand::Base(_)))
    );
}

// A real arithmetic ring with observable hook overrides: variant assertions alone
// would not detect replay that silently expands every operation to primitives.
mod hook_dispatch {
    use core::iter::{Product, Sum};
    use core::ops;
    use core::sync::atomic::{AtomicUsize, Ordering};

    use super::*;
    static SQUARE: AtomicUsize = AtomicUsize::new(0);
    static SUM: AtomicUsize = AtomicUsize::new(0);
    static DOT: AtomicUsize = AtomicUsize::new(0);
    static MIXED: AtomicUsize = AtomicUsize::new(0);
    static LOOKUPS: AtomicUsize = AtomicUsize::new(0);
    #[derive(Clone, Copy, Debug, Default, PartialEq)]
    struct Probe(F);
    impl From<F> for Probe {
        fn from(v: F) -> Self {
            Self(v)
        }
    }
    impl<T: Into<Self>> ops::Add<T> for Probe {
        type Output = Self;
        fn add(self, rhs: T) -> Self {
            Self(self.0 + rhs.into().0)
        }
    }
    impl<T: Into<Self>> ops::AddAssign<T> for Probe {
        fn add_assign(&mut self, rhs: T) {
            *self = *self + rhs;
        }
    }
    impl<T: Into<Self>> ops::Sub<T> for Probe {
        type Output = Self;
        fn sub(self, rhs: T) -> Self {
            Self(self.0 - rhs.into().0)
        }
    }
    impl<T: Into<Self>> ops::SubAssign<T> for Probe {
        fn sub_assign(&mut self, rhs: T) {
            *self = *self - rhs;
        }
    }
    impl<T: Into<Self>> ops::Mul<T> for Probe {
        type Output = Self;
        fn mul(self, rhs: T) -> Self {
            Self(self.0 * rhs.into().0)
        }
    }
    impl<T: Into<Self>> ops::MulAssign<T> for Probe {
        fn mul_assign(&mut self, rhs: T) {
            *self = *self * rhs;
        }
    }

    impl ops::Neg for Probe {
        type Output = Self;
        fn neg(self) -> Self {
            Self(-self.0)
        }
    }
    impl Sum for Probe {
        fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
            iter.fold(Self::ZERO, |a, b| a + b)
        }
    }
    impl Product for Probe {
        fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
            iter.fold(Self::ONE, |a, b| a * b)
        }
    }
    impl PrimeCharacteristicRing for Probe {
        type PrimeSubfield = <F as PrimeCharacteristicRing>::PrimeSubfield;
        const ZERO: Self = Self(F::ZERO);
        const ONE: Self = Self(F::ONE);
        const TWO: Self = Self(F::TWO);
        const NEG_ONE: Self = Self(F::NEG_ONE);
        fn from_prime_subfield(v: Self::PrimeSubfield) -> Self {
            Self(F::from_prime_subfield(v))
        }
        fn square(&self) -> Self {
            SQUARE.fetch_add(1, Ordering::SeqCst);
            Self(self.0.square())
        }
        fn sum_array<const N: usize>(v: &[Self]) -> Self {
            SUM.fetch_add(1, Ordering::SeqCst);
            assert_eq!(N, v.len());
            v.iter().copied().sum()
        }
        fn dot_product<const N: usize>(a: &[Self; N], b: &[Self; N]) -> Self {
            DOT.fetch_add(1, Ordering::SeqCst);
            a.iter().zip(b).map(|(a, b)| *a * *b).sum()
        }
    }
    impl Algebra<F> for Probe {
        fn mixed_dot_product<const N: usize>(a: &[Self; N], b: &[F; N]) -> Self {
            MIXED.fetch_add(1, Ordering::SeqCst);
            a.iter().zip(b).map(|(a, b)| *a * *b).sum()
        }
    }
    struct Builder {
        main: RowMajorMatrix<Probe>,
    }
    impl AirBuilder for Builder {
        type F = F;
        type Expr = Probe;
        type Var = Probe;
        type MainWindow = RowMajorMatrix<Probe>;
        type PreprocessedWindow = RowMajorMatrix<Probe>;
        type PublicVar = Probe;
        type PeriodicVar = Probe;
        fn main(&self) -> Self::MainWindow {
            LOOKUPS.fetch_add(1, Ordering::SeqCst);
            self.main.clone()
        }
        fn preprocessed(&self) -> &Self::PreprocessedWindow {
            &self.main
        }
        fn is_first_row(&self) -> Probe {
            Probe::ONE
        }
        fn is_last_row(&self) -> Probe {
            Probe::ZERO
        }
        fn is_transition(&self) -> Probe {
            Probe::ONE
        }
        fn assert_zero<I: Into<Probe>>(&mut self, _: I) {
            unreachable!()
        }
    }
    #[test]
    fn specialized_hooks_dispatch_and_shared_dags_are_memoized() {
        let b = Builder {
            main: RowMajorMatrix::new(vec![Probe(F::new(3)); 2], 1),
        };
        let e = E::from(ArithmeticVariable::new(BaseEntry::Main { offset: 0 }, 0));
        assert_eq!(e.square().resolve(&b).0, F::new(9));
        assert_eq!(SQUARE.load(Ordering::SeqCst), 1);
        assert_eq!(
            E::sum_array::<2>(&[e.clone(), e.clone()]).resolve(&b).0,
            F::new(6)
        );
        assert_eq!(SUM.load(Ordering::SeqCst), 1);
        assert_eq!(
            E::dot_product(core::array::from_ref(&e), core::array::from_ref(&e))
                .resolve(&b)
                .0,
            F::new(9)
        );
        assert_eq!(DOT.load(Ordering::SeqCst), 1);
        assert_eq!(
            E::mixed_dot_product(core::array::from_ref(&e), &[F::TWO])
                .resolve(&b)
                .0,
            F::new(6)
        );
        assert_eq!(MIXED.load(Ordering::SeqCst), 1);
        let squared = e.square().lower();
        assert!(matches!(squared,SymbolicExpr::Mul{x,y,..} if std::sync::Arc::ptr_eq(&x,&y)));
        let mut dag = e;
        for _ in 0..24 {
            dag = dag.clone() + dag;
        }
        LOOKUPS.store(0, Ordering::SeqCst);
        assert_eq!(dag.resolve(&b).0, F::new(3) * F::TWO.exp_u64(24));
        assert!(LOOKUPS.load(Ordering::SeqCst) < 100);
        // Lowering a multiply shared DAG must remain linear in its unique nodes.
        assert_eq!(dag.lower().degree_multiple(), 1);
    }
}

#[test]
fn identities_and_constant_batches_do_not_inflate_zero_coefficients() {
    let x = ArithmeticVariable::<F>::new(BaseEntry::Main { offset: 0 }, 0);
    let e = E::from(x);
    assert!(matches!(
        (e.clone() + F::ZERO).node(),
        ArithmeticNode::Leaf(_)
    ));
    assert!(matches!(
        (e.clone() * F::ONE).node(),
        ArithmeticNode::Leaf(_)
    ));
    let zero = E::sum_array::<2>(&[E::ONE, E::NEG_ONE]);
    assert_eq!((zero * e.clone()).degree_multiple(), 0);
    let c = EE::sum_array::<2>(&[EE::from_extension(EF::ONE), EE::from_extension(EF::NEG_ONE)]);
    assert_eq!((EE::from(e) * c).degree_multiple(), 0);
}

#[test]
fn ordinary_variable_operators_keep_operand_classes() {
    let x = ArithmeticVariable::<F>::new(BaseEntry::Main { offset: 0 }, 0);
    assert!(matches!((x + x).node(), ArithmeticNode::Add(_, _)));
    assert!(matches!((x - x).node(), ArithmeticNode::Sub(_, _)));
    assert!(matches!((x * x).node(), ArithmeticNode::Mul(_, _)));
    assert!(matches!(
        (x * F::TWO).node(),
        ArithmeticNode::Mul(_, BaseOperand::Scalar(_))
    ));
    let z = ArithmeticVariableExt::<F, EF>::new(ExtEntry::Permutation { offset: 0 }, 0);
    assert!(matches!((z + z).node(), ArithmeticNode::Add(_, _)));
    assert!(matches!(
        (z * F::TWO).node(),
        ArithmeticNode::Mul(_, ExtOperand::BaseScalar(_))
    ));
    assert!(matches!(
        (z * EF::TWO).node(),
        ArithmeticNode::Mul(_, ExtOperand::ExtensionScalar(_))
    ));
    assert!(matches!(
        (z * E::from(x)).node(),
        ArithmeticNode::Mul(_, ExtOperand::Base(_))
    ));
}
#[test]
fn same_type_dot_runtime_fallbacks_are_exact() {
    fn check<const N: usize>() {
        let x = E::from(ArithmeticVariable::new(BaseEntry::Main { offset: 0 }, 0));
        let a = core::array::from_fn::<_, N, _>(|_| x.clone());
        let e = E::dot_product(&a, &a);
        let c = Concrete::new(5);
        assert_eq!(e.resolve(&c), c.main.values[0].square() * F::from_usize(N));
        assert_eq!(e.lower().resolve(&c), e.resolve(&c));
    }
    check::<9>();
    check::<65>();
}
#[test]
fn complex_tower_constants_and_mixed_coefficients_are_supported() {
    type Tower = ArithmeticExpressionExt<Mersenne31, QM31>;
    let c = QM31::from_basis_coefficients_fn(|i| Mersenne31::new(i as u32 + 1));
    let z = ArithmeticVariableExt::<Mersenne31, QM31>::new(ExtEntry::Challenge, 0);
    let e = <Tower as Algebra<QM31>>::mixed_dot_product(&[z.into()], &[c]);
    assert!(
        matches!(e.node(),ArithmeticNode::MixedDot(p) if matches!(p[0].1,ExtOperand::ExtensionScalar(v) if v==c))
    );
    assert!(
        matches!(Tower::from(c).node(),ArithmeticNode::Leaf(ArithmeticExtLeaf::Constant(v)) if *v==c)
    );
}

#[test]
fn lowering_reuses_shared_base_graph_across_extension_references() {
    struct SharedBaseAir;
    impl p3_air::BaseAir<F> for SharedBaseAir {
        fn width(&self) -> usize {
            1
        }
    }
    impl<AB> p3_air::Air<AB> for SharedBaseAir
    where
        AB: PermutationAirBuilder<F = F, EF = EF>,
        AB::ExprEF: Algebra<F>,
    {
        fn eval(&self, b: &mut AB) {
            let x = b.main().current(0).unwrap();
            let y = b.main().next(0).unwrap();
            let z = b.permutation().current(0).unwrap();
            let mut base = AB::Expr::from(x);
            for _ in 0..32 {
                base = base * y + x;
            }
            let mut ext: AB::ExprEF = z.into();
            for _ in 0..64 {
                // Exercise both a base operand and a lifted base leaf.
                let lifted: AB::ExprEF = base.clone().into();
                ext = ext * base.clone() + lifted;
            }
            b.assert_zero_ext(ext);
        }
    }
    fn base_nodes(e: &SymbolicExpressionExt<F, EF>) -> usize {
        fn visit<L>(
            e: &SymbolicExpr<L>,
            seen: &mut std::collections::BTreeSet<usize>,
            leaf: &mut impl FnMut(&L),
        ) where
            L: p3_air::SymLeaf,
        {
            match e {
                SymbolicExpr::Leaf(l) => leaf(l),
                SymbolicExpr::Add { x, y, .. }
                | SymbolicExpr::Sub { x, y, .. }
                | SymbolicExpr::Mul { x, y, .. } => {
                    for child in [x, y] {
                        if seen.insert(std::sync::Arc::as_ptr(child) as usize) {
                            visit(child, seen, leaf);
                        }
                    }
                }
                SymbolicExpr::Neg { x, .. } => {
                    if seen.insert(std::sync::Arc::as_ptr(x) as usize) {
                        visit(x, seen, leaf);
                    }
                }
            }
        }
        let mut base = std::collections::BTreeSet::new();
        visit(e, &mut std::collections::BTreeSet::new(), &mut |leaf| {
            if let ExtLeaf::Base(e) = leaf {
                visit(e, &mut base, &mut |_| {});
            }
        });
        base.len()
    }
    let l = AirLayout {
        main_width: 1,
        permutation_width: 1,
        ..Default::default()
    };
    let arithmetic =
        p3_air::capture_arithmetic::<F, EF>(l, |b| p3_air::Air::eval(&SharedBaseAir, b));
    let legacy = p3_air::get_symbolic_constraints_extension::<F, EF, _>(&SharedBaseAir, l);
    let lowered_nodes = base_nodes(&arithmetic.extension[0].lower());
    let legacy_nodes = base_nodes(&legacy[0]);
    assert!(
        lowered_nodes <= legacy_nodes,
        "lowered {lowered_nodes}, legacy {legacy_nodes}"
    );
}

#[test]
fn deep_horner_traversals_fit_on_one_mib_stack() {
    let graphs = std::thread::Builder::new()
        .stack_size(1024 * 1024)
        .spawn(|| {
            let b = ArithmeticAirBuilder::<F, EF>::new(layout());
            let x = b.main().values[0];
            let y = b.main().values[1];
            let z = b.permutation().values[0];
            let mut base = E::from(x);
            let mut ext = EE::from(z);
            let concrete = Concrete::new(3);
            let mut base_value = concrete.main.values[0];
            let mut ext_value = concrete.perm.values[0];
            for _ in 0..3000 {
                base = base * y + x;
                ext = ext * y + z;
                base_value = base_value * concrete.main.values[1] + concrete.main.values[0];
                ext_value = ext_value * concrete.main.values[1] + concrete.perm.values[0];
            }
            assert_eq!(base.resolve(&concrete), base_value);
            assert_eq!(ext.resolve(&concrete), ext_value);
            assert_eq!(base.degree_multiple_with_transition(1), 3001);
            assert_eq!(ext.degree_multiple_with_transition(1), 3001);
            assert_eq!(base.poly_degree(8, &[2]), 3001 * 7);
            assert_eq!(ext.poly_degree(8, &[2]), 3001 * 7);
            let lowered_base = base.lower();
            let lowered_ext = ext.lower();
            assert_eq!(lowered_base.degree_multiple(), 3001);
            assert_eq!(lowered_ext.degree_multiple(), 3001);
            (base, ext, lowered_base, lowered_ext)
        })
        .unwrap()
        .join()
        .unwrap();
    // Arc graph destruction is recursive independently of the traversal APIs.
    std::thread::Builder::new()
        .stack_size(32 * 1024 * 1024)
        .spawn(move || drop(graphs))
        .unwrap()
        .join()
        .unwrap();
}

// Reusing a seeded program across three builders gives independent field-value
// expectations and legacy degree bounds for every arithmetic operand domain.
fn random_program<AB>(b: &AB, mut state: u64) -> (AB::Expr, AB::ExprEF)
where
    AB: PermutationAirBuilder<F = F, EF = EF>,
    AB::ExprEF: Algebra<F> + Algebra<AB::Var> + Algebra<AB::VarEF>,
{
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let x = b.main().current(0).unwrap();
    let y = b.main().next(0).unwrap();
    let z = b.permutation().current(0).unwrap();
    let base = vec![
        x.into(),
        y.into(),
        b.preprocessed().current(0).unwrap().into(),
        b.preprocessed().next(0).unwrap().into(),
        b.public_values()[0].into(),
        b.periodic_values()[0].into(),
        b.is_first_row(),
        b.is_last_row(),
        b.is_transition(),
        AB::Expr::ZERO,
        AB::Expr::ONE,
    ];
    let extension: Vec<AB::ExprEF> = vec![
        z.into(),
        b.permutation().next(0).unwrap().into(),
        b.permutation_randomness()[0].into(),
        b.permutation_values()[0].clone().into(),
        EF::from_basis_coefficients_fn(|i| F::from_usize(i + 1)).into(),
    ];
    let mut a = base[0].clone();
    let mut e = extension[0].clone();
    for step in 0..60 {
        let rhs = &base[next() as usize % base.len()];
        let erhs = &extension[next() as usize % extension.len()];
        let c = F::from_usize(next() as usize % 5);
        let ec = EF::from_basis_coefficients_fn(|i| c + F::from_usize(i));
        // Include all three runtime fallbacks even for an unlucky random seed.
        let op = if step >= 57 {
            12 + step - 57
        } else {
            next() as usize % 15
        };
        match op {
            0 => {
                a += rhs.clone();
                e += erhs.clone();
            }
            1 => {
                a -= rhs.clone();
                e -= erhs.clone();
            }
            2 => {
                a *= rhs.clone();
                e *= erhs.clone();
            }
            3 => {
                a = -a;
                e = -e;
            }
            4 => {
                a = a.square();
                e = e.square();
            }
            5 => {
                a += c;
                e += c;
            }
            6 => {
                a *= c;
                e *= ec;
            }
            7 => {
                a += x;
                e *= rhs.clone();
            }
            8 => {
                a = a * y - c;
                e += AB::ExprEF::from(ec);
            }
            9 => {
                a = <AB::Expr as Algebra<AB::Var>>::mixed_dot_product(&[a], &[x]);
                e = <AB::ExprEF as Algebra<AB::Var>>::mixed_dot_product(&[e], &[y]);
            }
            10 => {
                a = <AB::Expr as Algebra<F>>::mixed_dot_product(&[a], &[c]);
                e = <AB::ExprEF as Algebra<AB::VarEF>>::mixed_dot_product(&[e], &[z]);
            }
            11 => {
                a += rhs.clone();
                let lifted: AB::ExprEF = a.clone().into();
                e -= lifted;
            }
            _ => {
                fn batch<const N: usize, AB>(
                    a: &AB::Expr,
                    e: &AB::ExprEF,
                    rhs: &AB::Expr,
                    erhs: &AB::ExprEF,
                    c: F,
                    ec: EF,
                    kind: usize,
                ) -> (AB::Expr, AB::ExprEF)
                where
                    AB: PermutationAirBuilder<F = F, EF = EF>,
                    AB::ExprEF: Algebra<F>,
                {
                    let aa = core::array::from_fn::<_, N, _>(|i| {
                        if i == 0 { a.clone() } else { rhs.clone() }
                    });
                    let ee = core::array::from_fn::<_, N, _>(|i| {
                        if i == 0 { e.clone() } else { erhs.clone() }
                    });
                    let bb = core::array::from_fn::<_, N, _>(|_| rhs.clone());
                    let eb = core::array::from_fn::<_, N, _>(|_| erhs.clone());
                    match kind {
                        0 => (
                            AB::Expr::sum_array::<N>(&aa),
                            AB::ExprEF::sum_array::<N>(&ee),
                        ),
                        1 => (
                            AB::Expr::dot_product(&aa, &bb),
                            AB::ExprEF::dot_product(&ee, &eb),
                        ),
                        2 => (
                            <AB::Expr as Algebra<F>>::mixed_dot_product(&aa, &[c; N]),
                            <AB::ExprEF as Algebra<F>>::mixed_dot_product(&ee, &[c; N]),
                        ),
                        3 => (
                            <AB::Expr as Algebra<F>>::mixed_dot_product(&aa, &[c; N]),
                            <AB::ExprEF as Algebra<EF>>::mixed_dot_product(&ee, &[ec; N]),
                        ),
                        _ => (
                            AB::Expr::dot_product(&aa, &bb),
                            <AB::ExprEF as Algebra<AB::Expr>>::mixed_dot_product(&ee, &bb),
                        ),
                    }
                }
                let kind = next() as usize % 5;
                (a, e) = match op {
                    12 => batch::<9, AB>(&a, &e, rhs, erhs, c, ec, kind),
                    13 => batch::<17, AB>(&a, &e, rhs, erhs, c, ec, kind),
                    _ => batch::<33, AB>(&a, &e, rhs, erhs, c, ec, kind),
                };
            }
        }
    }
    (a, e)
}

#[test]
fn seeded_random_programs_match_legacy_values_and_degree_bounds() {
    fn symbolic_value<L: p3_air::SymLeaf, V: PrimeCharacteristicRing>(
        e: &SymbolicExpr<L>,
        memo: &mut std::collections::BTreeMap<usize, V>,
        leaf: &mut impl FnMut(&L) -> V,
    ) -> V {
        let key = core::ptr::from_ref(e) as usize;
        if let Some(value) = memo.get(&key) {
            return value.clone();
        }
        let value = match e {
            SymbolicExpr::Leaf(l) => leaf(l),
            SymbolicExpr::Add { x, y, .. } => {
                symbolic_value(x, memo, leaf) + symbolic_value(y, memo, leaf)
            }
            SymbolicExpr::Sub { x, y, .. } => {
                symbolic_value(x, memo, leaf) - symbolic_value(y, memo, leaf)
            }
            SymbolicExpr::Mul { x, y, .. } => {
                symbolic_value(x, memo, leaf) * symbolic_value(y, memo, leaf)
            }
            SymbolicExpr::Neg { x, .. } => -symbolic_value(x, memo, leaf),
        };
        memo.insert(key, value.clone());
        value
    }
    let arithmetic = ArithmeticAirBuilder::<F, EF>::new(layout());
    let legacy = p3_air::SymbolicAirBuilder::<F, EF>::new(layout());
    for seed in 1..=3000 {
        let concrete = Concrete::new(seed);
        let (base, ext) = random_program(&arithmetic, u64::from(seed));
        let (old_base, old_ext) = random_program(&legacy, u64::from(seed));
        let (base_value, ext_value) = random_program(&concrete, u64::from(seed));
        let lowered_base = base.lower();
        let lowered_ext = ext.lower();
        assert_eq!(base.resolve(&concrete), base_value, "base seed {seed}");
        assert_eq!(ext.resolve(&concrete), ext_value, "extension seed {seed}");
        for e in [&old_base, &lowered_base] {
            let value = symbolic_value(e, &mut Default::default(), &mut |l| {
                SymbolicExpr::Leaf(l.clone()).resolve(&concrete)
            });
            assert_eq!(value, base_value, "base seed {seed}");
        }
        for e in [&old_ext, &lowered_ext] {
            let mut base_memo = Default::default();
            let value = symbolic_value(e, &mut Default::default(), &mut |l| match l {
                ExtLeaf::Base(b) => symbolic_value(b, &mut base_memo, &mut |l| {
                    SymbolicExpr::Leaf(l.clone()).resolve(&concrete)
                })
                .into(),
                _ => eval_legacy(&SymbolicExpr::Leaf(l.clone()), &concrete),
            });
            assert_eq!(value, ext_value, "extension seed {seed}");
        }
        let check = |lowered: usize, arithmetic: usize, legacy: usize| {
            assert!(
                lowered <= arithmetic && arithmetic <= legacy,
                "seed {seed}: lowered {lowered}, arithmetic {arithmetic}, legacy {legacy}"
            );
        };
        check(
            lowered_base.degree_multiple(),
            base.degree_multiple(),
            old_base.degree_multiple(),
        );
        check(
            lowered_ext.degree_multiple(),
            ext.degree_multiple(),
            old_ext.degree_multiple(),
        );
        for transition in [1, 3] {
            check(
                lowered_base.degree_multiple_with_transition(transition),
                base.degree_multiple_with_transition(transition),
                old_base.degree_multiple_with_transition(transition),
            );
            check(
                lowered_ext.degree_multiple_with_transition(transition),
                ext.degree_multiple_with_transition(transition),
                old_ext.degree_multiple_with_transition(transition),
            );
        }
        for (trace_len, periods) in [(8, &[2][..]), (32, &[4][..])] {
            check(
                lowered_base.poly_degree(trace_len, periods),
                base.poly_degree(trace_len, periods),
                old_base.poly_degree(trace_len, periods),
            );
            check(
                lowered_ext.poly_degree(trace_len, periods),
                ext.poly_degree(trace_len, periods),
                old_ext.poly_degree(trace_len, periods),
            );
        }
    }
}
