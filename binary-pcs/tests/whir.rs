//! End-to-end coverage for WHIR over the additive binary-field domain.

use p3_binary_field::{BinaryChallenger, Poly64, Poly192};
use p3_binary_pcs::whir::{BinaryWhirDomain, recommended_cap_height};
use p3_challenger::HashChallenger;
use p3_commit::MultilinearPcs;
use p3_field::PrimeCharacteristicRing;
use p3_keccak::Keccak256Hash;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_sumcheck::layout::{Layout, PrefixProver, Table};
use p3_sumcheck::{OpeningBatch, OpeningProtocol, TableShape, TableSpec};
use p3_symmetric::{CompressionFunctionFromHasher, SerializingHasher};
use p3_whir::{FoldingFactor, ProtocolParameters, SecurityAssumption, WhirConfig, WhirProver};
use rand::SeedableRng;
use rand::rngs::SmallRng;

type Hash = SerializingHasher<Keccak256Hash>;
type Compress = CompressionFunctionFromHasher<Keccak256Hash, 2, 32>;
type Mmcs = MerkleTreeMmcs<Poly64, u8, Hash, Compress, 2, 32>;
type Challenger = BinaryChallenger<Poly64, HashChallenger<u8, Keccak256Hash, 32>>;
type Domain = BinaryWhirDomain;
type Pcs = WhirProver<Poly192, Poly64, Domain, Mmcs, Challenger, PrefixProver<Poly64, Poly192>>;

const fn challenger() -> Challenger {
    Challenger::from_hasher(Vec::new(), Keccak256Hash)
}

const fn mmcs(cap_height: usize) -> Mmcs {
    Mmcs::new(
        Hash::new(Keccak256Hash),
        Compress::new(Keccak256Hash),
        cap_height,
    )
}

#[test]
fn additive_whir_commits_opens_and_verifies() {
    const NUM_VARIABLES: usize = 12;
    const FOLDING: usize = 3;

    let mut rng = SmallRng::seed_from_u64(0x00B1_A47E);
    let witness = PrefixProver::<Poly64, Poly192>::new_witness(
        vec![Table::rand(&mut rng, 2, NUM_VARIABLES)],
        FOLDING,
    );
    let protocol = OpeningProtocol::new(vec![TableSpec::new(
        TableShape::new(NUM_VARIABLES, 2),
        vec![OpeningBatch::new(vec![0, 1], vec![])],
    )]);
    let committed_variables = witness.num_variables();
    let domain = Domain::default();
    let config = WhirConfig::new_with_domain(
        committed_variables,
        ProtocolParameters {
            security_level: 32,
            pow_bits: 0,
            round_log_inv_rates: vec![],
            folding_factor: FoldingFactor::Constant(FOLDING),
            soundness_type: SecurityAssumption::JohnsonBound,
            starting_log_inv_rate: 1,
        },
        &domain,
    )
    .unwrap();
    let cap_height = recommended_cap_height(&config);
    let pcs = Pcs::new(config, domain, mmcs(cap_height));

    let mut prover_challenger = challenger();
    let (commitment, prover_data) = pcs.commit(witness, &mut prover_challenger).unwrap();
    let proof = pcs
        .open(prover_data, protocol.clone(), &mut prover_challenger)
        .unwrap();

    pcs.verify(&commitment, &proof, &mut challenger(), protocol)
        .unwrap();

    let mut tampered = proof;
    tampered.whir.final_poly.as_mut().unwrap().as_mut_slice()[0] += Poly192::ONE;
    assert!(
        pcs.verify(
            &commitment,
            &tampered,
            &mut challenger(),
            OpeningProtocol::new(vec![TableSpec::new(
                TableShape::new(NUM_VARIABLES, 2),
                vec![OpeningBatch::new(vec![0, 1], vec![])],
            )]),
        )
        .is_err()
    );
}
