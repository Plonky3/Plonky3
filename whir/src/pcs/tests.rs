//! End-to-end tests exercising the WHIR PCS through the multilinear trait.

use alloc::vec;

use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::DuplexChallenger;
use p3_commit::MultilinearPcs;
use p3_dft::Radix2DFTSmallBatch;
use p3_field::Field;
use p3_field::extension::BinomialExtensionField;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use rand::SeedableRng;
use rand::rngs::SmallRng;

use crate::fiat_shamir::domain_separator::DomainSeparator;
use crate::parameters::{FoldingFactor, ProtocolParameters, SecurityAssumption, WhirConfig};
use crate::pcs::prover::WhirProver;
use crate::sumcheck::layout::{Layout, PrefixProver, SuffixProver, Witness};
use crate::sumcheck::tests::{random_table_specs, table_specs_to_tables};
use crate::sumcheck::{OpeningProtocol, TableSpec};

type F = BabyBear;
type EF = BinomialExtensionField<F, 4>;
type Perm = Poseidon2BabyBear<16>;

type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
type MyChallenger = DuplexChallenger<F, Perm, 16, 8>;

type PackedF = <F as Field>::Packing;
type MyMmcs = MerkleTreeMmcs<PackedF, PackedF, MyHash, MyCompress, 2, 8>;

type MyDft = Radix2DFTSmallBatch<F>;
type TestWhirPcs<L> = WhirProver<EF, F, MyDft, MyMmcs, MyChallenger, L>;

pub(crate) fn challenger() -> MyChallenger {
    let mut rng = SmallRng::seed_from_u64(1);
    let perm = Perm::new_from_rng_128(&mut rng);
    MyChallenger::new(perm)
}

#[allow(clippy::too_many_arguments)]
fn run_whir_pcs<L: Layout<F, EF>>(
    specs: &[TableSpec],
    folding_factor: FoldingFactor,
    soundness_type: SecurityAssumption,
    pow_bits: usize,
    rs_domain_initial_reduction_factor: usize,
) {
    let folding = folding_factor.at_round(0);
    let tables = table_specs_to_tables(specs);
    let witness = L::new_witness(tables, folding);
    let protocol = OpeningProtocol::new(specs.to_vec()).pad_to_min_num_variables(folding);
    assert_eq!(witness.table_shapes(), protocol.table_shapes());

    run_whir_pcs_lifecycle_with_witness::<L>(
        witness,
        protocol,
        folding_factor,
        soundness_type,
        pow_bits,
        rs_domain_initial_reduction_factor,
    );
}

#[allow(clippy::too_many_arguments)]
fn run_whir_pcs_lifecycle_with_witness<L: Layout<F, EF>>(
    witness: Witness<F>,
    protocol: OpeningProtocol,
    folding_factor: FoldingFactor,
    soundness_type: SecurityAssumption,
    pow_bits: usize,
    rs_domain_initial_reduction_factor: usize,
) {
    // Build Poseidon2-based hash and compression for the Merkle tree.
    let num_variables = witness.num_variables();
    let mut rng = SmallRng::seed_from_u64(1);
    let perm = Perm::new_from_rng_128(&mut rng);
    let merkle_hash = MyHash::new(perm.clone());
    let merkle_compress = MyCompress::new(perm);
    let mmcs = MyMmcs::new(merkle_hash, merkle_compress, 0);

    // Assemble the protocol parameters.
    // Security level 32 keeps the test fast; production would use 100-128.
    let params = ProtocolParameters {
        security_level: 32,
        pow_bits,
        rs_domain_initial_reduction_factor,
        folding_factor,
        soundness_type,
        starting_log_inv_rate: 1,
    };

    // Instantiate the PCS through the trait.
    let dft = MyDft::default();
    let config = WhirConfig::new(num_variables, params);
    let pcs = TestWhirPcs::<L>::new(config, dft, mmcs);

    // Prover
    let (commitment, proof) = {
        let mut challenger = challenger();
        let mut domain_separator = DomainSeparator::new(vec![]);
        pcs.add_domain_separator::<8>(&mut domain_separator);
        domain_separator.observe_domain_separator(&mut challenger);

        let (commitment, prover_data) =
            <TestWhirPcs<L> as MultilinearPcs<EF, MyChallenger>>::commit(
                &pcs,
                witness,
                &mut challenger,
            );
        let proof = <TestWhirPcs<L> as MultilinearPcs<EF, MyChallenger>>::open(
            &pcs,
            prover_data,
            protocol.clone(),
            &mut challenger,
        );
        (commitment, proof)
    };

    // Verifier
    {
        let mut challenger = challenger();
        let mut domain_separator = DomainSeparator::new(vec![]);
        pcs.add_domain_separator::<8>(&mut domain_separator);
        domain_separator.observe_domain_separator(&mut challenger);

        <TestWhirPcs<L> as MultilinearPcs<EF, MyChallenger>>::verify(
            &pcs,
            &commitment,
            &proof,
            &mut challenger,
            protocol,
        )
        .expect("verification failed");
    }
}

#[test]
fn test_whir_end_to_end() {
    const N: usize = 5;

    let folding_factors = [
        FoldingFactor::Constant(1),
        FoldingFactor::Constant(2),
        FoldingFactor::Constant(3),
        FoldingFactor::Constant(4),
        FoldingFactor::ConstantFromSecondRound(2, 1),
        FoldingFactor::ConstantFromSecondRound(3, 1),
        FoldingFactor::ConstantFromSecondRound(3, 2),
        FoldingFactor::ConstantFromSecondRound(5, 2),
    ];
    let soundness_type = [
        SecurityAssumption::JohnsonBound,
        SecurityAssumption::CapacityBound,
        SecurityAssumption::UniqueDecoding,
    ];
    let pow_bits = [0, 5, 10];
    let rs_domain_initial_reduction_factors = 1..=3;

    let mut rng = SmallRng::seed_from_u64(7);

    for rs_domain_initial_reduction_factor in rs_domain_initial_reduction_factors {
        for folding_factor in folding_factors {
            // Skip configurations where the first-round folding is smaller
            // than the initial domain reduction (would produce an empty domain).
            if folding_factor.at_round(0) < rs_domain_initial_reduction_factor {
                continue;
            }

            for soundness_type in soundness_type {
                for pow_bits in pow_bits {
                    for _ in 0..N {
                        let specs = random_table_specs(&mut rng, folding_factor.at_round(0));
                        run_whir_pcs::<PrefixProver<F, EF>>(
                            &specs,
                            folding_factor,
                            soundness_type,
                            pow_bits,
                            rs_domain_initial_reduction_factor,
                        );
                        run_whir_pcs::<SuffixProver<F, EF>>(
                            &specs,
                            folding_factor,
                            soundness_type,
                            pow_bits,
                            rs_domain_initial_reduction_factor,
                        );
                    }
                }
            }
        }
    }
}

mod keccak_tests {
    //! Same lifecycle test using Keccak-based Merkle trees over a different field.

    use alloc::vec;

    use p3_challenger::{HashChallenger, SerializingChallenger32};
    use p3_commit::MultilinearPcs;
    use p3_dft::Radix2DFTSmallBatch;
    use p3_field::extension::BinomialExtensionField;
    use p3_keccak::{Keccak256Hash, KeccakF};
    use p3_koala_bear::KoalaBear;
    use p3_merkle_tree::MerkleTreeMmcs;
    use p3_multilinear_util::poly::Poly;
    use p3_symmetric::{CompressionFunctionFromHasher, PaddingFreeSponge, SerializingHasher};
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    use crate::fiat_shamir::domain_separator::DomainSeparator;
    use crate::parameters::{FoldingFactor, ProtocolParameters, SecurityAssumption, WhirConfig};
    use crate::pcs::prover::WhirProver;
    use crate::sumcheck::layout::{Layout, PrefixProver, SuffixProver, Table};
    use crate::sumcheck::{OpeningProtocol, TableShape, TableSpec};

    type F = KoalaBear;
    type EF = BinomialExtensionField<F, 4>;

    type U64Hash = PaddingFreeSponge<KeccakF, 25, 17, 4>;
    type KeccakFieldHash = SerializingHasher<U64Hash>;
    type KeccakCompress = CompressionFunctionFromHasher<U64Hash, 2, 4>;

    type KeccakChallenger = SerializingChallenger32<F, HashChallenger<u8, Keccak256Hash, 32>>;
    type MyMmcs = MerkleTreeMmcs<F, u64, KeccakFieldHash, KeccakCompress, 2, 4>;
    type MyDft = Radix2DFTSmallBatch<F>;
    type TestWhirPcs<L> = WhirProver<EF, F, MyDft, MyMmcs, KeccakChallenger, L>;

    fn challenger() -> KeccakChallenger {
        KeccakChallenger::new(HashChallenger::<u8, Keccak256Hash, 32>::new(
            vec![],
            Keccak256Hash {},
        ))
    }

    /// Runs the full commit + open + verify lifecycle with Keccak Merkle trees.
    fn run_keccak_end_to_end<L: Layout<F, EF>>() {
        // Fixture: a single-table polynomial of arity 16 folded 4 vars at a time.
        const NUM_VARIABLES: usize = 16;
        const FOLDING: usize = 4;

        // Build one random table, stack it through the chosen layout mode.
        let mut rng = SmallRng::seed_from_u64(1);
        let table = Table::new(vec![Poly::<F>::rand(&mut rng, NUM_VARIABLES)]);
        let witness = L::new_witness(vec![table], FOLDING);
        // Public protocol: open the single column at one point.
        let protocol = OpeningProtocol::new(vec![TableSpec::new(
            TableShape::new(NUM_VARIABLES, 1),
            vec![vec![0]],
        )]);
        assert_eq!(witness.table_shapes(), protocol.table_shapes());

        // Wire Keccak-f as both the leaf-hash sponge and the 2-to-1 compressor.
        let u64_hash = U64Hash::new(KeccakF {});
        let merkle_hash = KeccakFieldHash::new(u64_hash);
        let merkle_compress = KeccakCompress::new(u64_hash);
        let mmcs = MyMmcs::new(merkle_hash, merkle_compress, 0);

        // Security level 32 keeps the test fast; not a production setting.
        let params = ProtocolParameters {
            security_level: 32,
            pow_bits: 0,
            rs_domain_initial_reduction_factor: 1,
            folding_factor: FoldingFactor::Constant(FOLDING),
            soundness_type: SecurityAssumption::CapacityBound,
            starting_log_inv_rate: 1,
        };
        let pcs = TestWhirPcs::<L>::new(
            WhirConfig::new(witness.num_variables(), params),
            MyDft::default(),
            mmcs,
        );

        // Prover side: seed the transcript with the protocol description, commit, open.
        let (commitment, proof) = {
            let mut prover_challenger = challenger();
            let mut domain_separator = DomainSeparator::new(vec![]);
            pcs.add_domain_separator::<4>(&mut domain_separator);
            domain_separator.observe_domain_separator(&mut prover_challenger);

            let (commitment, prover_data) = <TestWhirPcs<L> as MultilinearPcs<
                EF,
                KeccakChallenger,
            >>::commit(
                &pcs, witness, &mut prover_challenger
            );
            let proof = <TestWhirPcs<L> as MultilinearPcs<EF, KeccakChallenger>>::open(
                &pcs,
                prover_data,
                protocol.clone(),
                &mut prover_challenger,
            );
            (commitment, proof)
        };

        // Verifier side: replay the same transcript prefix from a fresh challenger.
        let mut verifier_challenger = challenger();
        let mut domain_separator = DomainSeparator::new(vec![]);
        pcs.add_domain_separator::<4>(&mut domain_separator);
        domain_separator.observe_domain_separator(&mut verifier_challenger);

        // Final assertion: the honest proof must verify under both layout modes.
        <TestWhirPcs<L> as MultilinearPcs<EF, KeccakChallenger>>::verify(
            &pcs,
            &commitment,
            &proof,
            &mut verifier_challenger,
            protocol,
        )
        .expect("keccak verification failed");
    }

    #[test]
    fn test_whir_keccak_end_to_end_suffix() {
        // Suffix mode binds the SVO suffix variables first.
        run_keccak_end_to_end::<SuffixProver<F, EF>>();
    }

    #[test]
    fn test_whir_keccak_end_to_end_prefix() {
        // Prefix mode binds the SVO prefix variables first; covers the other layout path.
        run_keccak_end_to_end::<PrefixProver<F, EF>>();
    }
}
