use core::borrow::Borrow;

use p3_air::{Air, AirBuilder, AirBuilderWithPublicValues, BaseAir, PairBuilder};
use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::DuplexChallenger;
use p3_commit::ExtensionMmcs;
use p3_dft::Radix2DitParallel;
use p3_field::extension::BinomialExtensionField;
use p3_field::{Field, PrimeField64};
use p3_fri::{TwoAdicFriPcs, create_test_fri_params};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use p3_uni_stark::{
    StarkConfig, prove_with_preprocessed, setup_preprocessed, verify_with_preprocessed,
};
use rand::SeedableRng;
use rand::rngs::SmallRng;

pub struct MulFibPAir {
    num_rows: usize,
    /// Index to tamper with in preprocessed trace (None = no tampering)
    tamper_index: Option<usize>,
}

impl MulFibPAir {
    pub const fn new(num_rows: usize) -> Self {
        Self {
            num_rows,
            tamper_index: None,
        }
    }

    pub const fn with_tampered_preprocessed(num_rows: usize, tamper_index: usize) -> Self {
        Self {
            num_rows,
            tamper_index: Some(tamper_index),
        }
    }
}

impl<F: PrimeField64> BaseAir<F> for MulFibPAir {
    fn width(&self) -> usize {
        NUM_COLS
    }
    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        Some(generate_preprocessed_trace::<F>(
            self.num_rows,
            self.tamper_index,
        ))
    }
}

impl<AB: AirBuilderWithPublicValues + PairBuilder> Air<AB> for MulFibPAir
where
    AB::F: PrimeField64,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let preprocessed = builder.preprocessed();

        let local_slice = main.row_slice(0).expect("Matrix is empty?");
        let next_slice = main.row_slice(1).expect("Matrix only has 1 row?");
        let prep_slice = preprocessed.row_slice(0).expect("Preprocessed is empty?");

        let local: &MulFibPairRow<AB::Var> = (*local_slice).borrow();
        let next: &MulFibPairRow<AB::Var> = (*next_slice).borrow();
        let prep: &PreprocessedRow<AB::Var> = (*prep_slice).borrow();

        let mut when_transition = builder.when_transition();

        // a' <- b
        when_transition.assert_eq(local.b.clone(), next.a.clone());

        // b' <- prod_coeff * a * b + sum_coeff * (a + b)
        let prod_term = prep.prod_coeff.clone() * local.a.clone() * local.b.clone();
        let sum_term = prep.sum_coeff.clone() * (local.a.clone() + local.b.clone());
        when_transition.assert_eq(prod_term + sum_term, next.b.clone());
    }
}

pub fn generate_trace_rows<F: PrimeField64>(a: u64, b: u64, n: usize) -> RowMajorMatrix<F> {
    assert!(n.is_power_of_two());

    let mut trace = RowMajorMatrix::new(F::zero_vec(n * NUM_COLS), NUM_COLS);
    let preprocessed = generate_preprocessed_trace::<F>(n, None);

    let (_, rows, _) = unsafe { trace.values.align_to_mut::<MulFibPairRow<F>>() };
    let (_, prep_rows, _) = unsafe { preprocessed.values.align_to::<PreprocessedRow<F>>() };
    assert_eq!(rows.len(), n);

    rows[0] = MulFibPairRow::new(F::from_u64(a), F::from_u64(b));

    for i in 1..n {
        rows[i].a = rows[i - 1].b;
        rows[i].b = prep_rows[i - 1].prod_coeff * rows[i - 1].a * rows[i - 1].b
            + prep_rows[i - 1].sum_coeff * (rows[i - 1].a + rows[i - 1].b);
    }

    trace
}

pub fn generate_preprocessed_trace<F: PrimeField64>(
    n: usize,
    tamper_index: Option<usize>,
) -> RowMajorMatrix<F> {
    assert!(n.is_power_of_two());

    let mut preprocessed = RowMajorMatrix::new(
        F::zero_vec(n * NUM_PREPROCESSED_COLS),
        NUM_PREPROCESSED_COLS,
    );

    let (_, rows, _) = unsafe { preprocessed.values.align_to_mut::<PreprocessedRow<F>>() };
    assert_eq!(rows.len(), n);

    rows.iter_mut().enumerate().for_each(|(i, row)| {
        row.prod_coeff = F::from_u64((i % 2) as u64);
        row.sum_coeff = F::from_u64(((i + 1) % 6) as u64);
    });

    if let Some(idx) = tamper_index.filter(|&i| i < n) {
        rows[idx].prod_coeff += F::ONE;
    }

    preprocessed
}

const NUM_COLS: usize = 2;
const NUM_PREPROCESSED_COLS: usize = 2;

pub struct MulFibPairRow<F> {
    pub a: F,
    pub b: F,
}

impl<F> MulFibPairRow<F> {
    const fn new(a: F, b: F) -> Self {
        Self { a, b }
    }
}

impl<F> Borrow<MulFibPairRow<F>> for [F] {
    fn borrow(&self) -> &MulFibPairRow<F> {
        debug_assert_eq!(self.len(), NUM_COLS);
        let (prefix, shorts, suffix) = unsafe { self.align_to::<MulFibPairRow<F>>() };
        debug_assert!(prefix.is_empty(), "Alignment should match");
        debug_assert!(suffix.is_empty(), "Alignment should match");
        debug_assert_eq!(shorts.len(), 1);
        &shorts[0]
    }
}

pub struct PreprocessedRow<F> {
    pub prod_coeff: F,
    pub sum_coeff: F,
}

impl<F> Borrow<PreprocessedRow<F>> for [F] {
    fn borrow(&self) -> &PreprocessedRow<F> {
        debug_assert_eq!(self.len(), NUM_PREPROCESSED_COLS);
        let (prefix, shorts, suffix) = unsafe { self.align_to::<PreprocessedRow<F>>() };
        debug_assert!(prefix.is_empty(), "Alignment should match");
        debug_assert!(suffix.is_empty(), "Alignment should match");
        debug_assert_eq!(shorts.len(), 1);
        &shorts[0]
    }
}

type Val = BabyBear;
type Perm = Poseidon2BabyBear<16>;
type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
type ValMmcs =
    MerkleTreeMmcs<<Val as Field>::Packing, <Val as Field>::Packing, MyHash, MyCompress, 8>;
type Challenge = BinomialExtensionField<Val, 4>;
type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
type Challenger = DuplexChallenger<Val, Perm, 16, 8>;
type Dft = Radix2DitParallel<Val>;
type Pcs = TwoAdicFriPcs<Val, Dft, ValMmcs, ChallengeMmcs>;
type MyConfig = StarkConfig<Pcs, Challenge, Challenger>;

fn setup_test_config() -> MyConfig {
    let mut rng = SmallRng::seed_from_u64(1);
    let perm = Perm::new_from_rng_128(&mut rng);
    let hash = MyHash::new(perm.clone());
    let compress = MyCompress::new(perm.clone());
    let val_mmcs = ValMmcs::new(hash, compress);
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
    let fri_params = create_test_fri_params(challenge_mmcs, 2);
    let pcs = Pcs::new(Dft::default(), val_mmcs, fri_params);
    let challenger = Challenger::new(perm);
    MyConfig::new(pcs, challenger)
}

#[test]
fn test_mul_fib_pair() {
    let num_rows = 1024;
    let config = setup_test_config();
    let trace = generate_trace_rows::<Val>(1, 1, num_rows);

    let air = MulFibPAir::new(num_rows);
    let degree_bits = 10; // log2(1024)
    let (preprocessed_prover_data, preprocessed_vk) =
        setup_preprocessed::<MyConfig, _>(&config, &air, degree_bits).unwrap();

    let proof = prove_with_preprocessed(&config, &air, trace, &[], Some(&preprocessed_prover_data));

    verify_with_preprocessed(&config, &air, &proof, &[], Some(&preprocessed_vk))
        .expect("verification failed");
}

#[test]
fn test_tampered_preprocessed_fails() {
    let num_rows = 1024;
    let config = setup_test_config();
    let trace = generate_trace_rows::<Val>(1, 1, num_rows);
    let air = MulFibPAir::new(num_rows);
    let degree_bits = 10; // log2(1024)

    // Prover uses the correct AIR for preprocessed setup.
    let (preprocessed_prover_data, _) =
        setup_preprocessed::<MyConfig, _>(&config, &air, degree_bits).unwrap();
    let proof = prove_with_preprocessed(&config, &air, trace, &[], Some(&preprocessed_prover_data));

    // Verifier uses a *tampered* AIR to derive the preprocessed commitment, which should
    // not match the one used in the proof.
    let tampered_air = MulFibPAir::with_tampered_preprocessed(num_rows, 3);
    let (_, tampered_preprocessed_vk) =
        setup_preprocessed::<MyConfig, _>(&config, &tampered_air, degree_bits).unwrap();

    let result = verify_with_preprocessed(
        &config,
        &tampered_air,
        &proof,
        &[],
        Some(&tampered_preprocessed_vk),
    );

    assert!(
        result.is_err(),
        "Verification should fail with tampered preprocessed columns"
    );
}
