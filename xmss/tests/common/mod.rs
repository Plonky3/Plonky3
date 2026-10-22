//! A KoalaBear STARK configuration with a Poseidon2 transcript and Merkle commitments.

#![allow(dead_code)]

use p3_batch_stark::{BatchProof, CommonData, ProverData, prove_batch, verify_batch};
use p3_challenger::DuplexChallenger;
use p3_circuit::Circuit;
use p3_circuit::tables::CircuitAir;
use p3_commit::ExtensionMmcs;
use p3_dft::Radix2DitParallel;
use p3_field::Field;
use p3_field::extension::BinomialExtensionField;
use p3_fri::{FriParameters, TwoAdicFriPcs};
use p3_koala_bear::{KoalaBear, Poseidon2KoalaBear, default_koalabear_poseidon2_16};
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use p3_uni_stark::StarkConfig;
use p3_xmss::Chip;

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

/// A configuration whose FRI parameters `fri` builds from the commitment scheme.
pub fn config(fri: impl FnOnce(ChallengeMmcs) -> FriParameters<ChallengeMmcs>) -> Config {
    let perm = default_koalabear_poseidon2_16();
    let mmcs = ValMmcs::new(Hash::new(perm.clone()), Compress::new(perm.clone()), 0);
    let fri = fri(ChallengeMmcs::new(mmcs.clone()));
    Config::new(
        Pcs::new(Radix2DitParallel::default(), mmcs, fri),
        Challenger::new(perm),
    )
}

/// A cheap configuration for tests, with two FRI queries.
pub fn test_config() -> Config {
    config(|mmcs| FriParameters::new_testing(mmcs, 0))
}

/// A circuit's proof, with what a parent needs to verify it.
pub struct Proven {
    /// The tables.
    pub airs: Vec<CircuitAir<F, EF, Chip>>,
    /// The verifier's data, including the preprocessed commitment.
    pub common: CommonData<Config>,
    /// The proof.
    pub proof: BatchProof<Config>,
    /// The public values of every table.
    pub public_values: Vec<Vec<F>>,
}

impl Proven {
    /// The proof of `circuit`, checked natively.
    pub fn new(config: &Config, circuit: Circuit<F, EF, Chip>) -> Self {
        let instances = circuit.instances::<Config>();
        let prover_data = ProverData::from_instances(config, &instances).expect("prover data");
        let proof = prove_batch(config, &instances, &prover_data).expect("the circuit proves");
        drop(instances);
        let public_values = circuit.public_values();
        verify_batch(
            config,
            &circuit.airs,
            &proof,
            &public_values,
            &prover_data.common,
        )
        .expect("the proof verifies");
        Self {
            airs: circuit.airs,
            common: prover_data.common,
            proof,
            public_values,
        }
    }

    /// The proof as a child of an aggregation node.
    pub fn child(&self) -> p3_circuit::recursion::Child<'_, Config, CircuitAir<F, EF, Chip>> {
        p3_circuit::recursion::Child {
            airs: &self.airs,
            common: &self.common,
            proof: &self.proof,
            public_values: &self.public_values,
        }
    }
}
