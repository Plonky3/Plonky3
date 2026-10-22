//! A KoalaBear configuration with a Poseidon2 transcript, Merkle commitments and circuit chip.

#![allow(dead_code)]

use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_batch_stark::{BatchProof, ProverData, prove_batch, verify_batch};
use p3_challenger::DuplexChallenger;
use p3_circuit::{Circuit, Poseidon2Chip};
use p3_commit::ExtensionMmcs;
use p3_dft::Radix2DitParallel;
use p3_field::Field;
use p3_field::extension::BinomialExtensionField;
use p3_fri::{FriParameters, TwoAdicFriPcs};
use p3_koala_bear::{
    GenericPoseidon2LinearLayersKoalaBear, KOALABEAR_POSEIDON2_RC_16_EXTERNAL_FINAL,
    KOALABEAR_POSEIDON2_RC_16_EXTERNAL_INITIAL, KOALABEAR_POSEIDON2_RC_16_INTERNAL, KoalaBear,
    Poseidon2KoalaBear, default_koalabear_poseidon2_16,
};
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_poseidon2_air::RoundConstants;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use p3_uni_stark::StarkConfig;

/// The base field.
pub type F = KoalaBear;
/// The challenge field.
pub type EF = BinomialExtensionField<F, 4>;
/// The Poseidon2 permutation of width 16.
pub type Perm = Poseidon2KoalaBear<16>;
/// The leaf hash.
pub type Hash = PaddingFreeSponge<Perm, 16, 8, 8>;
/// The two-to-one compression.
pub type Compress = TruncatedPermutation<Perm, 2, 8, 16>;
/// Merkle commitments to base-field matrices.
pub type ValMmcs =
    MerkleTreeMmcs<<F as Field>::Packing, <F as Field>::Packing, Hash, Compress, 2, 8>;
/// Merkle commitments to challenge-field matrices.
pub type ChallengeMmcs = ExtensionMmcs<F, EF, ValMmcs>;
/// The Fiat-Shamir transcript.
pub type Challenger = DuplexChallenger<F, Perm, 16, 8>;
/// The FRI polynomial commitment scheme.
pub type Pcs = TwoAdicFriPcs<F, Radix2DitParallel<F>, ValMmcs, ChallengeMmcs>;
/// The STARK configuration.
pub type Config = StarkConfig<Pcs, EF, Challenger>;
/// The circuit's Poseidon2 chip.
pub type Chip = Poseidon2Chip<F, GenericPoseidon2LinearLayersKoalaBear, Perm, 16, 3, 0, 4, 20>;

/// The Poseidon2 chip matching the configuration's permutation.
pub fn chip() -> Chip {
    Chip::new(
        default_koalabear_poseidon2_16(),
        RoundConstants::new(
            KOALABEAR_POSEIDON2_RC_16_EXTERNAL_INITIAL,
            KOALABEAR_POSEIDON2_RC_16_INTERNAL,
            KOALABEAR_POSEIDON2_RC_16_EXTERNAL_FINAL,
        ),
    )
}

/// A configuration with `num_queries` FRI queries.
pub fn config(num_queries: usize) -> Config {
    let perm = default_koalabear_poseidon2_16();
    let mmcs = ValMmcs::new(Hash::new(perm.clone()), Compress::new(perm.clone()), 0);
    let fri = FriParameters {
        num_queries,
        ..FriParameters::new_testing(ChallengeMmcs::new(mmcs.clone()), 0)
    };
    Config::new(
        Pcs::new(Radix2DitParallel::default(), mmcs, fri),
        Challenger::new(perm),
    )
}

/// Prove a circuit and check the proof natively.
pub fn prove_and_verify(config: &Config, circuit: &Circuit<F, EF, Chip>) -> BatchProof<Config> {
    let instances = circuit.instances::<Config>();
    let prover_data = ProverData::from_instances(config, &instances).expect("prover data");
    let proof = prove_batch(config, &instances, &prover_data).expect("the circuit proves");
    verify_batch(
        config,
        &circuit.airs,
        &proof,
        &circuit.public_values(),
        &prover_data.common,
    )
    .expect("the circuit proof verifies");
    proof
}

/// `(a, b) -> (b, a + b)` per row, with the first pair and the last value public.
#[derive(Clone, Copy, Debug)]
pub struct Fibonacci;

impl<T> BaseAir<T> for Fibonacci {
    fn width(&self) -> usize {
        2
    }

    fn num_public_values(&self) -> usize {
        3
    }
}

impl<AB: AirBuilder> Air<AB> for Fibonacci {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (main.current_slice().to_vec(), main.next_slice().to_vec());
        let publics = builder.public_values().to_vec();
        let (a, b, result) = (publics[0], publics[1], publics[2]);

        builder.when_first_row().assert_eq(local[0], a);
        builder.when_first_row().assert_eq(local[1], b);
        builder.when_transition().assert_eq(next[0], local[1]);
        builder
            .when_transition()
            .assert_eq(next[1], local[0] + local[1]);
        builder.when_last_row().assert_eq(local[1], result);
    }
}

/// The trace from `(a, b)` over `rows` rows, with its public values.
pub fn fibonacci(a: F, b: F, rows: usize) -> (RowMajorMatrix<F>, Vec<F>) {
    let first = [a, b];
    let (mut a, mut b) = (a, b);
    let mut values = Vec::with_capacity(2 * rows);
    for _ in 0..rows {
        values.extend([a, b]);
        (a, b) = (b, a + b);
    }
    (RowMajorMatrix::new(values, 2), vec![first[0], first[1], a])
}
