#![cfg(feature = "zk")]

use std::borrow::Cow;
use std::sync::Arc;

use p3_air::{
    Air, AirBuilder, BaseAir, BaseEntry, BaseLeaf, BoundaryEnd, BoundaryPublic, ExtensionBuilder,
    SymbolicExpression, SymbolicVariable, WindowAccess,
};
use p3_binary_field::TowerLevel;
use p3_binary_pcs::zk::{
    ClassicalRomBudget, CurrentRowAirBuilder, ReferenceZkAir, Val, ZkAirConfig, ZkAirProof, ZkError,
};
use p3_field::{Field, PrimeCharacteristicRing};
use p3_matrix::dense::RowMajorMatrix;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

struct Product;
impl BaseAir<Val> for Product {
    fn width(&self) -> usize {
        2
    }
    fn num_public_values(&self) -> usize {
        1
    }
    fn main_next_row_columns(&self) -> Vec<usize> {
        vec![]
    }
}
impl<AB: AirBuilder<F = Val>> Air<AB> for Product {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let public: AB::Expr = builder.public_values()[0].into();
        builder.assert_zero(main.current_slice()[0] * main.current_slice()[1] - public);
    }
}
fn protocol() -> ReferenceZkAir {
    ReferenceZkAir::compile(&Product, ZkAirConfig::interactive(1, 1).unwrap()).unwrap()
}
fn witness() -> RowMajorMatrix<Val> {
    RowMajorMatrix::new(vec![Val::ONE, Val::ONE, Val::ONE, Val::ONE], 2)
}

#[test]
fn public_product_roundtrip_and_statement_binding() {
    let air = protocol();
    let mut rng = StdRng::seed_from_u64(9);
    let proof = air.prove(witness(), &[Val::ONE], &mut rng).unwrap();
    air.verify(&proof, &[Val::ONE]).unwrap();
    let encoded = postcard::to_allocvec(&proof).unwrap();
    let decoded: ZkAirProof = postcard::from_bytes(&encoded).unwrap();
    air.verify(&decoded, &[Val::ONE]).unwrap();
    assert!(air.verify(&proof, &[Val::ZERO]).is_err());
    let taller =
        ReferenceZkAir::compile(&Product, ZkAirConfig::interactive(2, 1).unwrap()).unwrap();
    assert!(taller.verify(&proof, &[Val::ONE]).is_err());
}

#[test]
fn distinct_private_witnesses_prove_the_same_public_product() {
    let air = protocol();
    let product = Val::from_repr(7);
    let first = RowMajorMatrix::new(vec![Val::ONE, product, Val::ONE, product], 2);
    let second = RowMajorMatrix::new(vec![product, Val::ONE, product, Val::ONE], 2);
    assert_ne!(first.values, second.values);
    let mut rng = StdRng::seed_from_u64(19);
    for trace in [first, second] {
        let proof = air.prove(trace, &[product], &mut rng).unwrap();
        let encoded = postcard::to_allocvec(&proof).unwrap();
        let decoded: ZkAirProof = postcard::from_bytes(&encoded).unwrap();
        air.verify(&decoded, &[product]).unwrap();
        assert!(air.verify(&decoded, &[Val::ZERO]).is_err());
    }
}

#[test]
fn invalid_witness_and_shapes_are_typed_errors() {
    let air = protocol();
    let mut rng = StdRng::seed_from_u64(7);
    assert!(matches!(
        air.prove(witness(), &[Val::ZERO], &mut rng),
        Err(ZkError::InvalidWitness(_))
    ));
    let mut zero_width = witness();
    zero_width.width = 0;
    let mut bad_len = witness();
    bad_len.values.pop();
    for trace in [zero_width, bad_len] {
        let mut rng = StdRng::seed_from_u64(10);
        let mut untouched = StdRng::seed_from_u64(10);
        assert!(matches!(
            air.prove(trace, &[Val::ONE], &mut rng),
            Err(ZkError::InvalidShape(_))
        ));
        assert_eq!(rng.random::<u64>(), untouched.random::<u64>());
    }
    let mut rng = StdRng::seed_from_u64(11);
    let mut untouched = StdRng::seed_from_u64(11);
    assert!(air.prove(witness(), &[], &mut rng).is_err());
    assert_eq!(rng.random::<u64>(), untouched.random::<u64>());
}

#[test]
fn independent_proofs_and_malformed_openings() {
    let air = protocol();
    let mut rng = StdRng::seed_from_u64(42);
    let proof = air.prove(witness(), &[Val::ONE], &mut rng).unwrap();
    let other = air.prove(witness(), &[Val::ONE], &mut rng).unwrap();
    assert_ne!(
        proof.opening.claimed_values[..2],
        other.opening.claimed_values[..2]
    );
    assert_ne!(
        proof.opening.claimed_values[0],
        proof.opening.claimed_values[1]
    );
    let mut mutations = vec![];
    let mut p = proof.clone();
    p.opening.claimed_values.pop();
    mutations.push(p);
    let mut p = proof.clone();
    p.opening.claimed_values.push(Val::ZERO);
    mutations.push(p);
    let mut p = proof.clone();
    p.opening.blended_coefficients[0].pop();
    mutations.push(p);
    let mut p = proof.clone();
    p.opening.claimed_values[2] += Val::ONE;
    mutations.push(p);
    let mut p = proof.clone();
    p.commitment.num_polynomials = 0;
    mutations.push(p);
    let mut p = proof.clone();
    p.commitment.message_len += 1;
    mutations.push(p);
    let mut p = proof.clone();
    p.opening.queries[0].h.values.clear();
    mutations.push(p);
    let mut p = proof.clone();
    p.opening.queries.clear();
    mutations.push(p);
    let mut p = proof.clone();
    p.commitment.root = p3_symmetric::MerkleCap::new(vec![[0; 32]]);
    mutations.push(p);
    let mut p = proof.clone();
    p.commitment.root = p3_symmetric::MerkleCap::new(vec![[0; 32]; 2]);
    mutations.push(p);
    let mut p = proof.clone();
    p.opening.mask_root = p3_symmetric::MerkleCap::new(vec![[0; 32]; 2]);
    mutations.push(p);
    let mut p = proof.clone();
    p.opening.blended_coefficients.push(vec![]);
    mutations.push(p);
    let mut p = proof;
    p.commitment.num_polynomials += 1;
    mutations.push(p);
    for p in mutations {
        let bytes = postcard::to_allocvec(&p).unwrap();
        let p = postcard::from_bytes(&bytes).unwrap();
        assert!(air.verify(&p, &[Val::ONE]).is_err());
    }
}

type Expr = SymbolicExpression<Val>;
fn variable(entry: BaseEntry, index: usize) -> Expr {
    SymbolicVariable::<Val>::new(entry, index).into()
}
fn current(index: usize) -> Expr {
    variable(BaseEntry::Main { offset: 0 }, index)
}

#[derive(Default)]
struct Fixture {
    constraints: Vec<Expr>,
    declaration: &'static str,
    boolean_columns: usize,
    assumes_boolean: bool,
    extension: bool,
}
impl BaseAir<Val> for Fixture {
    fn width(&self) -> usize {
        2
    }
    fn num_public_values(&self) -> usize {
        1
    }
    fn main_next_row_columns(&self) -> Vec<usize> {
        if self.declaration == "next" {
            vec![0]
        } else {
            vec![]
        }
    }
    fn preprocessed_width(&self) -> usize {
        usize::from(self.declaration == "preprocessed-width")
    }
    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<Val>> {
        (self.declaration == "preprocessed-data")
            .then(|| RowMajorMatrix::new_col(vec![Val::ZERO; 2]))
    }
    fn num_periodic_columns(&self) -> usize {
        usize::from(self.declaration == "periodic-count")
    }
    fn periodic_columns(&self) -> Cow<'_, [Vec<Val>]> {
        if self.declaration == "periodic-data" {
            Cow::Owned(vec![vec![Val::ONE]])
        } else {
            Cow::Borrowed(&[])
        }
    }
    fn periodic_periods(&self) -> Vec<usize> {
        if self.declaration == "periodic-periods" {
            vec![1]
        } else {
            vec![]
        }
    }
    fn periodic_statement(&self) -> Vec<u8> {
        if self.declaration == "periodic-statement" {
            vec![1]
        } else {
            vec![]
        }
    }
    fn preprocessed_next_row_columns(&self) -> Vec<usize> {
        if self.declaration == "preprocessed-next" {
            vec![0]
        } else {
            vec![]
        }
    }
    fn public_boundary_io(&self) -> &[BoundaryPublic] {
        const BOUNDARY: &[BoundaryPublic] = &[BoundaryPublic::new(0, BoundaryEnd::First, 0)];
        if self.declaration == "boundary" {
            BOUNDARY
        } else {
            &[]
        }
    }
    fn boolean_columns(&self) -> usize {
        self.boolean_columns
    }
    fn assumes_boolean_trace(&self) -> bool {
        self.assumes_boolean
    }
    // Deliberately incorrect hints must not affect AST-derived geometry.
    fn max_constraint_degree(&self) -> Option<usize> {
        Some(0)
    }
    fn num_constraints(&self) -> Option<usize> {
        Some(0)
    }
}
impl Air<CurrentRowAirBuilder> for Fixture {
    fn eval(&self, builder: &mut CurrentRowAirBuilder) {
        for expr in &self.constraints {
            builder.assert_zero(expr.clone());
        }
        if self.extension {
            builder.assert_zero_ext(Val::ZERO);
        }
    }
}
fn compile(fixture: &Fixture) -> Result<ReferenceZkAir, ZkError> {
    ReferenceZkAir::compile(fixture, ZkAirConfig::interactive(1, 1).unwrap())
}

#[test]
fn rejects_unsupported_declarations_even_without_constraints() {
    for declaration in [
        "next",
        "preprocessed-width",
        "preprocessed-data",
        "periodic-count",
        "periodic-data",
        "periodic-periods",
        "periodic-statement",
        "preprocessed-next",
        "boundary",
    ] {
        assert!(
            matches!(
                compile(&Fixture {
                    declaration,
                    ..Default::default()
                }),
                Err(ZkError::UnsupportedAir(_))
            ),
            "{declaration}"
        );
    }
    assert!(matches!(
        compile(&Fixture {
            extension: true,
            ..Default::default()
        }),
        Err(ZkError::UnsupportedAir(_))
    ));
}

#[test]
fn validates_actual_leaves_and_recursive_degree_ignoring_hints() {
    for expression in [
        variable(BaseEntry::Main { offset: 1 }, 0),
        variable(BaseEntry::Main { offset: 2 }, 0),
        current(2),
        variable(BaseEntry::Public, 1),
        variable(BaseEntry::Preprocessed { offset: 0 }, 0),
        variable(BaseEntry::Periodic, 0),
        Expr::Leaf(BaseLeaf::IsFirstRow),
        Expr::Leaf(BaseLeaf::IsLastRow),
        Expr::Leaf(BaseLeaf::IsTransition),
    ] {
        assert!(matches!(
            compile(&Fixture {
                constraints: vec![expression],
                ..Default::default()
            }),
            Err(ZkError::UnsupportedAir(_))
        ));
    }
    let mut expression = current(0);
    // Forge every cached degree to zero: actual degree five still rejects.
    for _ in 1..5 {
        expression = Expr::Mul {
            x: Arc::new(expression),
            y: Arc::new(current(0)),
            degree_multiple: 0,
        };
    }
    assert!(matches!(
        compile(&Fixture {
            constraints: vec![expression],
            ..Default::default()
        }),
        Err(ZkError::UnsupportedAir(_))
    ));
}

#[test]
fn boolean_declarations_are_enforced_and_metadata_checked() {
    let air = compile(&Fixture {
        boolean_columns: 1,
        ..Default::default()
    })
    .unwrap();
    let mut rng = StdRng::seed_from_u64(10);
    let proof = air.prove(witness(), &[Val::ONE], &mut rng).unwrap();
    air.verify(&proof, &[Val::ONE]).unwrap();
    let mut bad = witness();
    bad.values[0] = Val::from_repr(2);
    assert!(matches!(
        air.prove(bad, &[Val::ONE], &mut rng),
        Err(ZkError::InvalidWitness(_))
    ));
    for (boolean_columns, assumes_boolean) in [(3, false), (1, true)] {
        assert!(matches!(
            compile(&Fixture {
                boolean_columns,
                assumes_boolean,
                ..Default::default()
            }),
            Err(ZkError::UnsupportedAir(_))
        ));
    }
    let all_bits = compile(&Fixture {
        boolean_columns: 2,
        assumes_boolean: true,
        ..Default::default()
    })
    .unwrap();
    assert!(all_bits.verify(&proof, &[Val::ONE]).is_err());
}

#[test]
fn binds_expressions_and_proves_nonconstant_witnesses_with_same_statement() {
    let mut rng = StdRng::seed_from_u64(19);
    let air = protocol();
    let a = Val::from_repr(7);
    let trace = RowMajorMatrix::new(vec![a, a.inverse(), Val::ONE, Val::ONE], 2);
    let proof = air.prove(trace, &[Val::ONE], &mut rng).unwrap();
    air.verify(&proof, &[Val::ONE]).unwrap();
    let changed = compile(&Fixture {
        constraints: vec![current(0) * current(1) + current(0)],
        ..Default::default()
    })
    .unwrap();
    assert!(changed.verify(&proof, &[Val::ONE]).is_err());
    let quartic = current(0) * current(0) * current(0) * current(0) + Val::ONE;
    let quartic = compile(&Fixture {
        constraints: vec![quartic],
        ..Default::default()
    })
    .unwrap();
    let proof = quartic.prove(witness(), &[Val::ONE], &mut rng).unwrap();
    quartic.verify(&proof, &[Val::ONE]).unwrap();
}

#[test]
fn enforces_config_and_augmented_constraint_limits() {
    for (height, bits) in [(0, 1), (9, 1), (1, 0), (1, 121), (usize::MAX, usize::MAX)] {
        assert!(ZkAirConfig::interactive(height, bits).is_err());
    }
    assert!(
        compile(&Fixture {
            constraints: vec![Expr::ZERO; 32],
            boolean_columns: 1,
            ..Default::default()
        })
        .is_err()
    );
    let air = ReferenceZkAir::compile(&Product, ZkAirConfig::interactive(1, 96).unwrap()).unwrap();
    let mut rng = StdRng::seed_from_u64(21);
    let proof = air.prove(witness(), &[Val::ONE], &mut rng).unwrap();
    air.verify(&proof, &[Val::ONE]).unwrap();
}

#[test]
fn canonical_statement_ignores_pointer_sharing_and_cached_degrees() {
    let sum = Arc::new(current(0) + current(1));
    let shared = Expr::Mul {
        x: sum.clone(),
        y: sum,
        degree_multiple: 0,
    };
    let unshared = (current(0) + current(1)) * (current(0) + current(1));
    let first = compile(&Fixture {
        constraints: vec![shared],
        ..Default::default()
    })
    .unwrap();
    let second = compile(&Fixture {
        constraints: vec![unshared],
        ..Default::default()
    })
    .unwrap();
    let proof = first
        .prove(witness(), &[Val::ONE], &mut StdRng::seed_from_u64(50))
        .unwrap();
    second.verify(&proof, &[Val::ONE]).unwrap();
}

#[test]
fn rejects_expressions_and_batches_exceeding_resource_limits() {
    let mut deep = current(0);
    for _ in 0..128 {
        deep = Expr::Neg {
            x: Arc::new(deep),
            degree_multiple: 1,
        };
    }
    assert!(
        compile(&Fixture {
            constraints: vec![deep],
            ..Default::default()
        })
        .is_err()
    );
    let mut broad = Expr::ZERO;
    for _ in 0..12 {
        broad = Expr::Add {
            x: Arc::new(broad.clone()),
            y: Arc::new(broad),
            degree_multiple: 0,
        };
    }
    assert!(
        compile(&Fixture {
            constraints: vec![broad],
            ..Default::default()
        })
        .is_err()
    );
    let quartic = current(0) * current(0) * current(0) * current(0);
    let fixture = Fixture {
        constraints: vec![quartic; 32],
        ..Default::default()
    };
    assert!(ReferenceZkAir::compile(&fixture, ZkAirConfig::interactive(8, 96).unwrap()).is_err());
}

#[test]
fn explicit_security_policies_validate_declared_budgets() {
    for (target, queries, leaves) in [(0, 32, 49), (97, 32, 49), (80, 65, 49), (80, 32, 121)] {
        assert!(ClassicalRomBudget::new(target, queries, leaves).is_err());
    }
    for height in [0, 9, usize::MAX] {
        assert!(ZkAirConfig::interactive(height, 1).is_err());
        assert!(
            ZkAirConfig::classical_rom(height, ClassicalRomBudget::new(80, 32, 49).unwrap())
                .is_err()
        );
    }
    for bits in [0, 121, usize::MAX] {
        assert!(ZkAirConfig::interactive(1, bits).is_err());
    }
    assert!(ZkAirConfig::interactive(1, 120).is_ok());
    assert!(ReferenceZkAir::compile(&Product, ZkAirConfig::interactive(1, 120).unwrap()).is_ok());
    // The raw API range does not waive the committed-degree field check.
    assert!(ReferenceZkAir::compile(&Product, ZkAirConfig::interactive(8, 120).unwrap()).is_err());
    let budget = ClassicalRomBudget::new(80, 32, 49).unwrap();
    assert_eq!(
        (
            budget.target_bits(),
            budget.log_adversary_queries(),
            budget.log_total_salted_leaves()
        ),
        (80, 32, 49)
    );
    let air =
        ReferenceZkAir::compile(&Product, ZkAirConfig::classical_rom(2, budget).unwrap()).unwrap();
    let report = air.security_report();
    assert_eq!(
        (
            report.trace_height,
            report.width,
            report.constraints,
            report.batch_width
        ),
        (4, 2, 1, 3)
    );
    assert_eq!(
        (
            report.message_len,
            report.coefficient_len,
            report.codeword_len,
            report.num_queries,
            report.degree_bound
        ),
        (5, 466, 2048, 460, 8)
    );
    assert_eq!(
        (report.interactive_bits, report.salt_bits, report.hash_bits),
        (113, 256, 256)
    );
    let rom = report.classical_rom.unwrap();
    assert_eq!(rom.budget, budget);
    assert_eq!(rom.adversary_queries, 1u128 << 32);
    assert_eq!(rom.verifier_queries, 2 * 460 * 12 + 16 + 115);
    assert_eq!(rom.total_queries, (1u128 << 32) + rom.verifier_queries);
    assert_eq!(rom.log_total_queries, 33);
    assert_eq!(rom.total_salted_leaves, 1u128 << 49);
    assert_eq!(
        (
            rom.rbr_exponent,
            rom.graph_exponent,
            rom.salt_collision_exponent,
            rom.leaf_query_exponent
        ),
        (82, 185, 159, 174)
    );
    assert!(protocol().security_report().classical_rom.is_none());
}

#[test]
fn classical_profile_rejects_field_and_global_lifetime_underbudgets() {
    let config =
        ZkAirConfig::classical_rom(2, ClassicalRomBudget::new(96, 32, 49).unwrap()).unwrap();
    assert!(matches!(
        ReferenceZkAir::compile(&Product, config),
        Err(ZkError::InvalidConfig(
            "committed-degree AIR soundness bound exceeds target"
        ))
    ));
    // Both trees of even one complete proof must fit the declared global cap.
    let small = ZkAirConfig::classical_rom(1, ClassicalRomBudget::new(1, 0, 1).unwrap()).unwrap();
    assert!(ReferenceZkAir::compile(&Product, small).is_err());
    // Guard subtraction at the maximum budget and include verifier calls even at b=0.
    let extreme =
        ZkAirConfig::classical_rom(1, ClassicalRomBudget::new(1, 64, 120).unwrap()).unwrap();
    let report = ReferenceZkAir::compile(&Product, extreme)
        .unwrap()
        .security_report();
    let rom = report.classical_rom.unwrap();
    assert_eq!(rom.log_total_queries, 65);
    assert_eq!(rom.salt_collision_exponent, 17);
    let config = ZkAirConfig::classical_rom(1, ClassicalRomBudget::new(1, 0, 49).unwrap()).unwrap();
    let report = ReferenceZkAir::compile(&Product, config)
        .unwrap()
        .security_report();
    let rom = report.classical_rom.unwrap();
    assert!(rom.log_total_queries > 0);
    assert_eq!(rom.total_queries, 1 + rom.verifier_queries);
    assert!(rom.total_queries <= 1u128 << rom.log_total_queries);
    assert!(rom.total_queries > 1u128 << (rom.log_total_queries - 1));
    assert_eq!(report.num_queries, 4 * (1 + rom.log_total_queries + 2));
}

#[test]
fn classical_policy_and_each_declared_budget_bind_proof_transcript() {
    let configs = [(1, 0, 49), (1, 0, 50), (1, 1, 49), (2, 0, 49)];
    let protocols: Vec<_> = configs
        .into_iter()
        .map(|(t, b, a)| {
            ReferenceZkAir::compile(
                &Product,
                ZkAirConfig::classical_rom(1, ClassicalRomBudget::new(t, b, a).unwrap()).unwrap(),
            )
            .unwrap()
        })
        .collect();
    let first = &protocols[0];
    let mut rng = StdRng::seed_from_u64(70);
    let proof = first.prove(witness(), &[Val::ONE], &mut rng).unwrap();
    first.verify(&proof, &[Val::ONE]).unwrap();
    for other in &protocols[1..] {
        assert!(other.verify(&proof, &[Val::ONE]).is_err());
    }
    let interactive = ReferenceZkAir::compile(
        &Product,
        ZkAirConfig::interactive(1, first.security_report().interactive_bits).unwrap(),
    )
    .unwrap();
    assert_eq!(
        interactive.security_report().num_queries,
        first.security_report().num_queries
    );
    assert!(interactive.verify(&proof, &[Val::ONE]).is_err());
}

struct Products(usize);
impl BaseAir<Val> for Products {
    fn width(&self) -> usize {
        2 * self.0
    }
    fn num_public_values(&self) -> usize {
        self.0
    }
    fn main_next_row_columns(&self) -> Vec<usize> {
        vec![]
    }
}
impl<AB: AirBuilder<F = Val>> Air<AB> for Products {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        for j in 0..self.0 {
            let p: AB::Expr = builder.public_values()[j].into();
            builder.assert_zero(main.current_slice()[2 * j] * main.current_slice()[2 * j + 1] - p);
        }
    }
}

#[test]
fn accepted_finite_classical_profiles_preserve_combined_resource_caps() {
    for (height, pairs, m, l, n, v, bytes) in [
        (2, 1, 5, 466, 2048, 11171, 420000),
        (6, 1, 65, 526, 4096, 12091, 452320),
        (8, 1, 257, 718, 4096, 12091, 461536),
        (6, 4, 65, 526, 4096, 12091, 660832),
    ] {
        let config =
            ZkAirConfig::classical_rom(height, ClassicalRomBudget::new(80, 32, 49).unwrap())
                .unwrap();
        let air = ReferenceZkAir::compile(&Products(pairs), config).unwrap();
        let report = air.security_report();
        assert_eq!(
            (
                report.message_len,
                report.coefficient_len,
                report.codeword_len,
                report.num_queries
            ),
            (m, l, n, 460)
        );
        assert_eq!(report.classical_rom.unwrap().verifier_queries, v);
        assert_eq!(air.encoded_proof_len(), bytes);
    }
    let config =
        ZkAirConfig::classical_rom(8, ClassicalRomBudget::new(80, 32, 49).unwrap()).unwrap();
    let quartic = current(0) * current(0) * current(0) * current(0);
    let wide_batch = Fixture {
        constraints: vec![quartic; 32],
        ..Default::default()
    };
    assert!(ReferenceZkAir::compile(&wide_batch, config).is_err());
    // The largest declared leaf exponent is representable but cannot provide
    // an 80-bit ideal privacy guarantee under the conservative salt bound.
    let config =
        ZkAirConfig::classical_rom(1, ClassicalRomBudget::new(80, 32, 120).unwrap()).unwrap();
    assert!(matches!(
        ReferenceZkAir::compile(&Product, config),
        Err(ZkError::InvalidConfig(
            "classical ROM component exceeds target"
        ))
    ));
}

proptest::proptest! {
    #![proptest_config(proptest::test_runner::Config::with_cases(16))]
    #[test]
    fn varying_shapes_public_statements_and_transport_bind_the_same_relation(
        log_height in 1usize..=3, pairs in 1usize..=3, seed in proptest::prelude::any::<u64>(),
        public_repr in proptest::prelude::any::<u128>()
    ) {
        let mut rng=StdRng::seed_from_u64(seed); // test-only reproducibility
        let public: Vec<_>=(0..pairs).map(|j|Val::from_repr(public_repr.wrapping_add(j as u128))).collect();
        let mut values=vec![];
        for _ in 0..1<<log_height {
            for &product in &public {
                let factor=loop {let v:Val=rng.random();if v!=Val::ZERO {break v;}};
                values.extend([factor,product*factor.inverse()]);
            }
        }
        let air=ReferenceZkAir::compile(&Products(pairs),ZkAirConfig::interactive(log_height,1).unwrap()).unwrap();
        let proof=air.prove(RowMajorMatrix::new(values,2*pairs),&public,&mut rng).unwrap();
        let bytes=air.encode_proof(&proof).unwrap();
        let decoded=air.decode_proof(&bytes).unwrap();
        proptest::prop_assert!(air.encode_proof(&decoded).unwrap() == bytes);
        proptest::prop_assert!(air.verify_bytes(&bytes,&public).is_ok());
        let mut wrong_public=public.clone();wrong_public[0]+=Val::ONE;
        proptest::prop_assert!(air.verify_bytes(&bytes,&wrong_public).is_err());
        let mut corrupted=bytes;corrupted[64]^=1; // first public opening claim
        proptest::prop_assert!(air.verify_bytes(&corrupted,&public).is_err());
    }
}
