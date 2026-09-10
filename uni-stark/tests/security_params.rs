use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::DuplexChallenger;
use p3_circle::{CircleDomain, CirclePcs};
use p3_commit::ExtensionMmcs;
use p3_dft::Radix2DitParallel;
use p3_field::coset::TwoAdicMultiplicativeCoset;
use p3_field::extension::BinomialExtensionField;
use p3_field::{Field, PrimeCharacteristicRing};
use p3_fri::{FriParameters, HidingFriPcs, TwoAdicFriPcs};
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_mersenne_31::{Mersenne31, Poseidon2Mersenne31, QM31};
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use p3_uni_stark::{AirLayout, OpeningShape, StarkConfig, StarkSecurityParams, prove, verify};
use rand::SeedableRng;
use rand::rngs::StdRng;

struct SquareAir {
    degree_hint: Option<usize>,
}

impl<F> BaseAir<F> for SquareAir {
    fn width(&self) -> usize {
        2
    }

    fn main_next_row_columns(&self) -> Vec<usize> {
        vec![]
    }

    fn max_constraint_degree(&self) -> Option<usize> {
        self.degree_hint
    }
}

impl<AB: AirBuilder> Air<AB> for SquareAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let a = main.current(0).unwrap();
        let b = main.current(1).unwrap();
        builder.assert_eq(a * a, b);
    }
}

type Val = BabyBear;
type Challenge = BinomialExtensionField<Val, 4>;
type Perm = Poseidon2BabyBear<16>;
type Hash = PaddingFreeSponge<Perm, 16, 8, 8>;
type Compress = TruncatedPermutation<Perm, 2, 8, 16>;
type ValMmcs =
    MerkleTreeMmcs<<Val as Field>::Packing, <Val as Field>::Packing, Hash, Compress, 2, 8>;
type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
type Challenger = DuplexChallenger<Val, Perm, 16, 8>;
type Dft = Radix2DitParallel<Val>;

fn setup() -> (Perm, ValMmcs, FriParameters<ChallengeMmcs>) {
    let mut rng = StdRng::seed_from_u64(42);
    let perm = Perm::new_from_rng_128(&mut rng);
    let mmcs = ValMmcs::new(Hash::new(perm.clone()), Compress::new(perm.clone()), 0);
    let fri = FriParameters::new_testing(ChallengeMmcs::new(mmcs.clone()), 0);
    (perm, mmcs, fri)
}

fn trace() -> RowMajorMatrix<Val> {
    let values = (1..=16)
        .flat_map(|i| {
            let a = Val::from_usize(i);
            [a, a * a]
        })
        .collect();
    RowMajorMatrix::new(values, 2)
}

#[test]
fn opening_count_matches_proof_with_overestimated_degree_hint() {
    let (perm, mmcs, fri) = setup();
    let air = SquareAir {
        degree_hint: Some(5),
    };
    let params = StarkSecurityParams::from_air::<Val, Challenge, _>(
        fri.security_regime(),
        &air,
        AirLayout::from_air::<Val>(&air),
        TwoAdicMultiplicativeCoset::new(Val::ONE, 4).unwrap(),
        124,
        128,
        1,
        OpeningShape::new(),
        fri.grinding_sites(),
    );
    let pcs = TwoAdicFriPcs::new(Dft::default(), mmcs, fri);
    let config = StarkConfig::<_, Challenge, _>::new(pcs, Challenger::new(perm));
    let proof = prove(&config, &air, trace(), &[]).unwrap();
    verify(&config, &air, &proof, &[]).unwrap();

    // The degree-five hint commits four chunks, each four base-field columns wide.
    assert_eq!(proof.opened_values.quotient_chunks.len(), 4);
    let actual = proof.opened_values.trace_local.len()
        + proof
            .opened_values
            .quotient_chunks
            .iter()
            .map(Vec::len)
            .sum::<usize>();
    assert_eq!(actual, 18);
    assert_eq!(params.num_batched_functions, actual);
}

#[test]
fn opening_count_matches_hiding_proof() {
    let (perm, mmcs, fri) = setup();
    let air = SquareAir { degree_hint: None };
    let params = StarkSecurityParams::from_air::<Val, Challenge, _>(
        fri.security_regime(),
        &air,
        AirLayout::from_air::<Val>(&air),
        TwoAdicMultiplicativeCoset::new(Val::ONE, 4).unwrap(),
        124,
        128,
        1,
        OpeningShape::hiding(4),
        fri.grinding_sites(),
    );
    let pcs = HidingFriPcs::new(Dft::default(), mmcs, fri, 4, StdRng::seed_from_u64(43));
    let config = StarkConfig::<_, Challenge, _>::new(pcs, Challenger::new(perm));
    let proof = prove(&config, &air, trace(), &[]).unwrap();
    verify(&config, &air, &proof, &[]).unwrap();

    let public = proof.opened_values.trace_local.len()
        + proof
            .opened_values
            .quotient_chunks
            .iter()
            .map(Vec::len)
            .sum::<usize>()
        + proof.opened_values.random.as_ref().unwrap().len();
    let hiding: usize = proof
        .opening_proof
        .0
        .iter()
        .flatten()
        .flatten()
        .map(Vec::len)
        .sum();
    // Four random columns on the main, four quotient chunks, and randomizing matrix.
    assert_eq!((public, hiding), (22, 24));
    assert_eq!(params.num_batched_functions, public + hiding);
}

#[test]
fn opening_count_matches_circle_batching_degree() {
    type F = Mersenne31;
    type Perm = Poseidon2Mersenne31<16>;
    type Hash = PaddingFreeSponge<Perm, 16, 8, 8>;
    type Compress = TruncatedPermutation<Perm, 2, 8, 16>;
    type Mmcs = MerkleTreeMmcs<<F as Field>::Packing, <F as Field>::Packing, Hash, Compress, 2, 8>;
    let mut rng = StdRng::seed_from_u64(44);
    let perm = Perm::new_from_rng_128(&mut rng);
    let mmcs = Mmcs::new(Hash::new(perm.clone()), Compress::new(perm.clone()), 0);
    let fri = FriParameters::new_testing(ExtensionMmcs::<F, QM31, _>::new(mmcs.clone()), 0);
    let air = SquareAir { degree_hint: None };
    let params = StarkSecurityParams::from_air::<F, QM31, _>(
        fri.security_regime(),
        &air,
        AirLayout::from_air::<F>(&air),
        CircleDomain::standard(4),
        124,
        128,
        1,
        OpeningShape::Circle,
        fri.grinding_sites(),
    );
    let pcs = CirclePcs::<F, _, _>::new(mmcs, fri);
    let config = StarkConfig::<_, QM31, _>::new(pcs, DuplexChallenger::<F, _, 16, 8>::new(perm));
    let trace = RowMajorMatrix::new(vec![F::ONE; 32], 2);
    let proof = prove(&config, &air, trace, &[]).unwrap();
    verify(&config, &air, &proof, &[]).unwrap();

    let columns = proof.opened_values.trace_local.len()
        + proof
            .opened_values
            .quotient_chunks
            .iter()
            .map(Vec::len)
            .sum::<usize>();
    assert_eq!(columns, 6);
    // Circle's DEEP quotient consumes two powers per column and opening point.
    assert_eq!(params.num_batched_functions, 2 * columns);
}

#[test]
fn circle_transition_cubic_security_counts_all_committed_quotient_chunks() {
    struct TransitionCubicAir;
    impl BaseAir<Mersenne31> for TransitionCubicAir {
        fn width(&self) -> usize {
            1
        }
        fn main_next_row_columns(&self) -> Vec<usize> {
            vec![]
        }
    }
    impl<AB: AirBuilder<F = Mersenne31>> Air<AB> for TransitionCubicAir {
        fn eval(&self, builder: &mut AB) {
            let x: AB::Expr = builder.main().current_slice()[0].into();
            builder
                .when_transition()
                .assert_zero(x.clone() * x.clone() * x);
        }
    }
    type F = Mersenne31;
    type Perm = Poseidon2Mersenne31<16>;
    type Hash = PaddingFreeSponge<Perm, 16, 8, 8>;
    type Compress = TruncatedPermutation<Perm, 2, 8, 16>;
    type Mmcs = MerkleTreeMmcs<<F as Field>::Packing, <F as Field>::Packing, Hash, Compress, 2, 8>;
    let mut rng = StdRng::seed_from_u64(45);
    let perm = Perm::new_from_rng_128(&mut rng);
    let mmcs = Mmcs::new(Hash::new(perm.clone()), Compress::new(perm.clone()), 0);
    let mut fri = FriParameters::new_testing(ExtensionMmcs::<F, QM31, _>::new(mmcs.clone()), 0);
    fri.log_blowup = 2;
    let air = TransitionCubicAir;
    let params = StarkSecurityParams::from_air::<F, QM31, _>(
        fri.security_regime(),
        &air,
        AirLayout::from_air::<F>(&air),
        CircleDomain::standard(4),
        124,
        128,
        1,
        OpeningShape::Circle,
        fri.grinding_sites(),
    );
    let pcs = CirclePcs::<F, _, _>::new(mmcs, fri);
    let config = StarkConfig::<_, QM31, _>::new(pcs, DuplexChallenger::<F, _, 16, 8>::new(perm));
    let proof = prove(
        &config,
        &air,
        RowMajorMatrix::new(vec![F::ZERO; 16], 1),
        &[],
    )
    .unwrap();
    verify(&config, &air, &proof, &[]).unwrap();
    assert_eq!(proof.opened_values.quotient_chunks.len(), 4);
    assert_eq!(
        params.num_quotient_chunks,
        proof.opened_values.quotient_chunks.len()
    );
    assert_eq!(params.air_max_constraint_degree, 4);
    assert_eq!(params.num_batched_functions, 2 * (1 + 4 * 4));
}
