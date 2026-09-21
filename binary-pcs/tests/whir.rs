//! End-to-end coverage for WHIR over the additive binary-field domain.

use p3_binary_field::{BinaryChallenger, Poly64, Poly192};
use p3_binary_pcs::whir::{BinaryWhirDomain, recommended_cap_height};
use p3_challenger::HashChallenger;
use p3_commit::MultilinearPcs;
use p3_field::PrimeCharacteristicRing;
use p3_keccak::Keccak256Hash;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_sumcheck::layout::{Layout, PrefixProver, SuffixProver, Table};
use p3_sumcheck::{OpeningBatch, OpeningProtocol, TableShape, TableSpec};
use p3_symmetric::{CompressionFunctionFromHasher, SerializingHasher};
use p3_whir::{
    FoldingFactor, ProtocolParameters, SecurityAssumption, VerifierError, WhirConfig,
    WhirConfigError, WhirProver,
};
use rand::SeedableRng;
use rand::rngs::SmallRng;

type Hash = SerializingHasher<Keccak256Hash>;
type Compress = CompressionFunctionFromHasher<Keccak256Hash, 2, 32>;
type Mmcs = MerkleTreeMmcs<Poly64, u8, Hash, Compress, 2, 32>;
type Challenger = BinaryChallenger<Poly64, HashChallenger<u8, Keccak256Hash, 32>>;
type Domain = BinaryWhirDomain;
type Pcs = WhirProver<Poly192, Poly64, Domain, Mmcs, Challenger, PrefixProver<Poly64, Poly192>>;
type SuffixPcs =
    WhirProver<Poly192, Poly64, Domain, Mmcs, Challenger, SuffixProver<Poly64, Poly192>>;

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
    let refused = pcs
        .verify(
            &commitment,
            &tampered,
            &mut challenger(),
            OpeningProtocol::new(vec![TableSpec::new(
                TableShape::new(NUM_VARIABLES, 2),
                vec![OpeningBatch::new(vec![0, 1], vec![])],
            )]),
        )
        .unwrap_err();

    // The final polynomial feeds the transcript, so moving it moves every position drawn after.
    //
    // The error carries no equality, so the variant is pinned whole and its message checked.
    assert!(
        matches!(
            refused,
            VerifierError::MerkleProofInvalid { position: 0, .. }
        ),
        "{refused:?}"
    );
    assert_eq!(
        refused.to_string(),
        "Merkle proof verification failed at position 0: \
         Extension field Merkle multiproof verification failed"
    );
}

#[test]
fn additive_whir_supports_suffix_binding_and_a_mixed_folding_schedule() {
    const NUM_VARIABLES: usize = 12;
    const FIRST_FOLDING: usize = 3;

    // Suffix binding reverses the fold point relative to the prefix layout.
    let mut rng = SmallRng::seed_from_u64(0x5AFF_1CED);
    let witness = SuffixProver::<Poly64, Poly192>::new_witness(
        vec![Table::rand(&mut rng, 2, NUM_VARIABLES)],
        FIRST_FOLDING,
    );
    let protocol = OpeningProtocol::new(vec![TableSpec::new(
        TableShape::new(NUM_VARIABLES, 2),
        vec![OpeningBatch::new(vec![0, 1], vec![])],
    )]);
    let domain = Domain::default();
    let config = WhirConfig::new_with_domain(
        witness.num_variables(),
        ProtocolParameters {
            security_level: 32,
            pow_bits: 0,
            round_log_inv_rates: vec![],
            folding_factor: FoldingFactor::ConstantFromSecondRound(FIRST_FOLDING, 2),
            soundness_type: SecurityAssumption::JohnsonBound,
            starting_log_inv_rate: 2,
        },
        &domain,
    )
    .unwrap();
    let cap_height = recommended_cap_height(&config);
    let pcs = SuffixPcs::new(config, domain, mmcs(cap_height));

    // Prover and verifier must agree across both layout and folding changes.
    let mut prover_challenger = challenger();
    let (commitment, prover_data) = pcs.commit(witness, &mut prover_challenger).unwrap();
    let proof = pcs
        .open(prover_data, protocol.clone(), &mut prover_challenger)
        .unwrap();
    pcs.verify(&commitment, &proof, &mut challenger(), protocol)
        .unwrap();
}

#[test]
fn additive_domain_rejects_the_capacity_regime() {
    // Capacity-rate list decoding is refuted on this characteristic-two domain.
    let error = WhirConfig::<Poly192, Poly64, Challenger>::new_with_domain(
        12,
        ProtocolParameters {
            security_level: 32,
            pow_bits: 0,
            round_log_inv_rates: vec![],
            folding_factor: FoldingFactor::Constant(3),
            soundness_type: SecurityAssumption::CapacityBound,
            starting_log_inv_rate: 1,
        },
        &Domain::default(),
    )
    .unwrap_err();

    assert!(matches!(
        error,
        WhirConfigError::UnsupportedSecurityAssumption {
            assumption: SecurityAssumption::CapacityBound
        }
    ));
}

#[test]
fn the_cap_height_fits_every_round_tree() {
    // A flat round rate shrinks the last tree far below the first round's folded domain.
    let config = WhirConfig::<Poly192, Poly64, Challenger>::new_with_domain(
        11,
        ProtocolParameters {
            security_level: 32,
            pow_bits: 0,
            round_log_inv_rates: vec![1],
            folding_factor: FoldingFactor::Constant(4),
            soundness_type: SecurityAssumption::JohnsonBound,
            starting_log_inv_rate: 2,
        },
        &Domain::default(),
    )
    .unwrap();

    let round = &config.round_parameters[0];
    let final_round = config.final_round_config();

    // The last phase asks for more queries than its folded domain holds, so it draws nothing.
    assert_eq!(round.num_queries, 35);
    assert_eq!(round.log_folded_domain_size, 9);
    assert_eq!(config.final_queries, 75);
    assert_eq!(final_round.log_folded_domain_size, 4);

    // Both trees carry the cap, so the shallower one bounds the deepest stratum of the other.
    let cap_height = recommended_cap_height(&config);
    assert_eq!(cap_height, 4);
    assert!(cap_height <= round.log_folded_domain_size);
    assert!(cap_height <= final_round.log_folded_domain_size);
}

/// A trace over a thirty-two-bit alphabet, committed at its own width.
mod small_field {
    use p3_binary_field::{BinaryChallenger, BinaryField32, BinaryField128};
    use p3_binary_pcs::whir::{BinaryWhirBudget, BinaryWhirDomain, BinaryWhirProfile, ProofShape};
    use p3_challenger::HashChallenger;
    use p3_commit::MultilinearPcs;
    use p3_keccak::Keccak256Hash;
    use p3_merkle_tree::MerkleTreeMmcs;
    use p3_sumcheck::layout::{Layout, SuffixProver, Table};
    use p3_sumcheck::{OpeningBatch, OpeningProtocol, TableShape, TableSpec};
    use p3_whir::WhirProver;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    use super::{Compress, Hash};

    type F = BinaryField32;
    type EF = BinaryField128;
    type NarrowMmcs = MerkleTreeMmcs<F, u8, Hash, Compress, 2, 32>;
    type NarrowChallenger = BinaryChallenger<F, HashChallenger<u8, Keccak256Hash, 32>>;
    type NarrowDomain = BinaryWhirDomain<F>;
    type NarrowPcs =
        WhirProver<EF, F, NarrowDomain, NarrowMmcs, NarrowChallenger, SuffixProver<F, EF>>;

    /// Bytes the encoder writes for one element of the narrow alphabet.
    const ALPHABET_BYTES: usize = 5;

    /// Bytes the encoder writes for one element of the challenge field instead.
    const CHALLENGE_BYTES: usize = 19;

    /// Bytes one Merkle digest occupies.
    const DIGEST_BYTES: usize = 32;

    #[test]
    fn a_small_field_trace_commits_at_its_own_width() {
        const NUM_VARIABLES: usize = 12;
        const FOLDING: usize = 3;

        let mut rng = SmallRng::seed_from_u64(0x5A11_F1E1);
        let witness = SuffixProver::<F, EF>::new_witness(
            vec![Table::rand(&mut rng, 1, NUM_VARIABLES)],
            FOLDING,
        );
        let protocol = OpeningProtocol::new(vec![TableSpec::new(
            TableShape::new(NUM_VARIABLES, 1),
            vec![OpeningBatch::new(vec![0], vec![])],
        )]);

        let domain = NarrowDomain::default();
        let profile = BinaryWhirProfile::proven_list_decoding(80, 2, FOLDING);
        let config = profile
            .config::<EF, F, NarrowChallenger, _>(NUM_VARIABLES, &domain)
            .unwrap();
        let shape = ProofShape::of(&config, 1);

        // The first codeword is opened at the alphabet's own width, not the challenge field's.
        let narrow = shape.max_bytes(ALPHABET_BYTES, CHALLENGE_BYTES, DIGEST_BYTES);
        let widened = shape.max_bytes(CHALLENGE_BYTES, CHALLENGE_BYTES, DIGEST_BYTES);
        assert!(narrow < widened);

        BinaryWhirBudget::PRODUCTION
            .check_shape(&shape, ALPHABET_BYTES, CHALLENGE_BYTES, DIGEST_BYTES)
            .unwrap();

        let cap_height = p3_binary_pcs::whir::recommended_cap_height(&config);
        let mmcs = NarrowMmcs::new(
            Hash::new(Keccak256Hash),
            Compress::new(Keccak256Hash),
            cap_height,
        );
        let pcs = NarrowPcs::new(config, domain, mmcs);

        let mut prover_challenger = NarrowChallenger::from_hasher(Vec::new(), Keccak256Hash);
        let (commitment, prover_data) = pcs.commit(witness, &mut prover_challenger).unwrap();
        let proof = pcs
            .open(prover_data, protocol.clone(), &mut prover_challenger)
            .unwrap();

        let mut verifier_challenger = NarrowChallenger::from_hasher(Vec::new(), Keccak256Hash);
        pcs.verify(&commitment, &proof, &mut verifier_challenger, protocol)
            .unwrap();

        let bytes = postcard::to_allocvec(&proof).unwrap().len();
        assert!(
            bytes <= narrow,
            "{bytes} bytes against an estimate of {narrow}"
        );
        eprintln!("small-field 2^{NUM_VARIABLES} over a 32-bit alphabet: {bytes} bytes");
    }
}
