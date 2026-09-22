//! A Boolean trace committed narrow and opened through WHIR over the additive binary domain.
//!
//! The fixture pins four things the integration exists for.
//!
//! ```text
//!     narrowness   one committed byte per eight trace bits
//!     regimes      three, each selected by name, none of them a default
//!     budget       a ceiling graded before proving and again after
//!     size         the proof against the folding-only path at one target
//! ```

use p3_binary_field::{BinaryChallenger, BinaryField128, Gf2, Ghash128, PackedGf2x64};
use p3_binary_pcs::whir::{
    BinaryWhirBudget, BinaryWhirProfile, BooleanWhirDomain, BooleanWhirError, BooleanWhirPcs,
    BooleanWhirProof, BooleanWhirProver, BooleanWhirTracePcs, BudgetError, recommended_cap_height,
};
use p3_binary_pcs::{
    BinaryPcsConfig, BinaryPcsParams, BitOpening, BitReadings, BooleanMultilinearPcs, BooleanPcs,
    BooleanTraceCommitmentError, GroupedCodewordMmcs,
};
use p3_challenger::HashChallenger;
use p3_commit::MultilinearPcs;
use p3_field::PrimeCharacteristicRing;
use p3_keccak::Keccak256Hash;
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;
use p3_sumcheck::layout::{Table, plan_stacked_layout};
use p3_sumcheck::ring_switch::bits::{BitPacking, BitRingSwitch, BitRingSwitchProofError};
use p3_sumcheck::{OpeningBatch, OpeningProtocol, PrescribedPointPcs, TableShape, TableSpec};
use p3_symmetric::{CompressionFunctionFromHasher, SerializingHasher};
use p3_whir::SecurityAssumption;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

type EF = BinaryField128;
type MyHash = SerializingHasher<Keccak256Hash>;
type MyCompress = CompressionFunctionFromHasher<Keccak256Hash, 2, 32>;
type MyMmcs = MerkleTreeMmcs<EF, u8, MyHash, MyCompress, 2, 32>;
type Grouped = GroupedCodewordMmcs<MyMmcs>;
type MyChallenger = BinaryChallenger<EF, HashChallenger<u8, Keccak256Hash, 32>>;
type Prover = BooleanWhirProver<EF, BooleanWhirDomain, MyMmcs, MyChallenger>;
type Pcs = BooleanWhirPcs<EF, BooleanWhirDomain, MyMmcs, MyChallenger>;
type TracePcs = BooleanWhirTracePcs<EF, BooleanWhirDomain, MyMmcs, MyChallenger>;

/// Bits the fixture commits, in log bits of one Boolean column.
const LOG_BITS: usize = 16;

/// Coordinates one element of the widest level absorbs.
const ABSORBED: usize = 7;

/// Target in bits for every error term both paths are graded against.
const SECURITY_LEVEL: usize = 100;

/// Base-two logarithm of the inverse code rate both paths encode at.
const LOG_INV_RATE: usize = 2;

/// Variables each proximity round eliminates.
const FOLDING: usize = 3;

/// Bytes one committed element of the widest tower level occupies in memory.
const ELEMENT_BYTES: usize = 16;

/// Bytes the encoder writes for one element, whose widest variable-length code is this long.
const ENCODED_ELEMENT_BYTES: usize = 19;

/// Bytes one Merkle digest occupies.
const DIGEST_BYTES: usize = 32;

const fn mmcs(cap_height: usize) -> MyMmcs {
    MyMmcs::new(
        MyHash::new(Keccak256Hash),
        MyCompress::new(Keccak256Hash),
        cap_height,
    )
}

const fn challenger() -> MyChallenger {
    MyChallenger::from_hasher(Vec::new(), Keccak256Hash)
}

/// A random bit-sliced witness covering the committed hypercube.
fn witness(seed: u64) -> Vec<PackedGf2x64> {
    let mut rng = SmallRng::seed_from_u64(seed);
    (0..1 << (LOG_BITS - 6))
        .map(|_| PackedGf2x64::new(rng.random::<u64>()))
        .collect()
}

/// The witness as a multilinear over every variable, one element per bit.
fn embedded(bits: &[PackedGf2x64]) -> Poly<EF> {
    Poly::new(
        bits.iter()
            .flat_map(|block| {
                (0..PackedGf2x64::WIDTH).map(move |lane| {
                    if block.get(lane) == Gf2::ONE {
                        EF::ONE
                    } else {
                        EF::ZERO
                    }
                })
            })
            .collect::<Vec<EF>>(),
    )
}

/// This many opening points, drawn from one seed.
fn points_at(seed: u64, count: usize) -> Vec<Point<EF>> {
    let mut rng = SmallRng::seed_from_u64(seed);
    (0..count)
        .map(|_| Point::<EF>::rand(&mut rng, LOG_BITS))
        .collect()
}

/// Two opening points the fixtures reuse.
fn points(seed: u64) -> Vec<Point<EF>> {
    points_at(seed, 2)
}

/// The proximity argument the fixtures wrap, built fresh from one regime.
fn whir_prover(profile: BinaryWhirProfile) -> Prover {
    let domain = BooleanWhirDomain::default();
    let config = profile
        .config::<EF, EF, MyChallenger, _>(LOG_BITS - ABSORBED, &domain)
        .unwrap();
    let cap_height = recommended_cap_height(&config);
    Prover::new(config, domain, mmcs(cap_height))
}

/// The WHIR-backed Boolean commitment for one regime.
fn whir_pcs(profile: BinaryWhirProfile) -> Pcs {
    BooleanWhirPcs::new(whir_prover(profile), LOG_BITS).unwrap()
}

/// The schedule this many surviving claims are discharged through, as the adapter builds it.
fn protocol(num_claims: usize) -> OpeningProtocol {
    OpeningProtocol::new(vec![TableSpec::new(
        TableShape::new(LOG_BITS - ABSORBED, 1),
        (0..num_claims)
            .map(|_| OpeningBatch::new(vec![0], Vec::new()))
            .collect(),
    )])
}

/// Each point as an opening asking for the current reading alone.
fn current_openings(points: &[Point<EF>]) -> Vec<BitOpening<EF>> {
    points
        .iter()
        .map(|point| BitOpening {
            point: point.clone(),
            row_variables: point.num_variables(),
            current: true,
            next: false,
        })
        .collect()
}

/// The folding-only Boolean commitment, graded at the same target and the same rate.
fn basefold_pcs() -> BooleanPcs<EF, Grouped, Grouped> {
    let config = BinaryPcsConfig::try_new_with_folding::<EF, EF>(
        LOG_BITS - ABSORBED,
        BinaryPcsParams {
            log_inv_rate: LOG_INV_RATE,
            pow_bits: 0,
            security_level: SECURITY_LEVEL,
        },
        FOLDING,
    )
    .unwrap();
    BooleanPcs::new(
        config,
        Grouped::for_folding(mmcs(0), &config),
        Grouped::for_folding(mmcs(0), &config),
        LOG_BITS,
    )
    .unwrap()
}

#[test]
fn a_boolean_opening_round_trips_through_whir() {
    let bits = witness(0x5711);
    let points = points(0x5712);
    let reference = embedded(&bits);

    for profile in [
        BinaryWhirProfile::unique_decoding(SECURITY_LEVEL, LOG_INV_RATE, FOLDING),
        BinaryWhirProfile::proven_list_decoding(SECURITY_LEVEL, LOG_INV_RATE, FOLDING),
    ] {
        let pcs = whir_pcs(profile);
        let regime = profile.assumption();

        let mut prover_chal = challenger();
        let (commitment, data) = pcs.commit_bits(&bits, &mut prover_chal).unwrap();
        let (values, proof) = pcs.open_at_points(data, &points, &mut prover_chal).unwrap();

        // The values answer for the bits themselves, not for the packing.
        for (point, &value) in points.iter().zip(&values) {
            assert_eq!(value, reference.eval_base(point), "{regime}");
        }

        let mut verifier_chal = challenger();
        pcs.observe_commitment(&commitment, &mut verifier_chal);
        pcs.verify_at_points(&commitment, &points, &values, &proof, &mut verifier_chal)
            .unwrap_or_else(|error| panic!("{regime}: {error:?}"));

        // A value the commitment does not hold must be refused.
        //
        // The reduction that reads the claim off the element is what refuses it.
        let mut tampered = values.clone();
        tampered[0] += EF::ONE;
        let mut verifier_chal = challenger();
        pcs.observe_commitment(&commitment, &mut verifier_chal);
        let refused = pcs
            .verify_at_points(&commitment, &points, &tampered, &proof, &mut verifier_chal)
            .unwrap_err();
        assert!(
            matches!(
                refused,
                BooleanWhirError::ReductionProof(BitRingSwitchProofError::ClaimMismatch)
            ),
            "{regime}: {refused:?}"
        );
    }
}

#[test]
fn the_commitment_holds_one_byte_per_eight_bits() {
    // A widened commitment would hold one element per bit, sixteen bytes each.
    // The packed one holds one element per one hundred and twenty-eight bits.
    let pcs = whir_pcs(BinaryWhirProfile::unique_decoding(
        SECURITY_LEVEL,
        LOG_INV_RATE,
        FOLDING,
    ));
    let shape = pcs.proof_shape(2, false);
    assert_eq!(pcs.committed_bytes(), (1 << LOG_BITS) / 8);

    let bits = witness(0x5713);
    let mut prover_chal = challenger();
    let (_, data) = pcs.commit_bits(&bits, &mut prover_chal).unwrap();

    // The retained table is the committed message, so its size is the commit path's width.
    let table = data.table(0);
    assert_eq!(table.num_polys(), 1);
    assert_eq!(table.num_variables(), LOG_BITS - ABSORBED);
    assert_eq!(
        (1usize << table.num_variables()) * ELEMENT_BYTES,
        pcs.committed_bytes()
    );

    // The openings never widen either: every opened element is one packed element.
    assert!(shape.opened_base_elements > 0);
}

#[test]
fn the_budget_grades_the_schedule_and_the_proof() {
    let profile = BinaryWhirProfile::proven_list_decoding(SECURITY_LEVEL, LOG_INV_RATE, FOLDING);
    let pcs = whir_pcs(profile);
    let shape = pcs.proof_shape(2, false);
    let budget = BinaryWhirBudget::PRODUCTION;

    // The derived schedule is not pinned here.
    //
    // It trades grinding against queries, and that trade has differed between builds.
    //
    // The shape's own arithmetic holds whichever way the trade lands.
    //
    // Three variables fold once, so every position opens eight off a tree of one depth.
    assert_eq!(shape.opened_base_elements, shape.stir_queries << FOLDING);
    assert_eq!(shape.opened_extension_elements, 0);
    assert_eq!(
        shape.merkle_digests,
        shape.stir_queries * (LOG_BITS - ABSORBED - FOLDING + LOG_INV_RATE)
    );
    assert!(shape.grinding_bits <= budget.max_grinding_bits);

    // What the claim count adds is fixed by the claims, not by the schedule.
    //
    // Each brings its opened value and one reduction, whose size the packing's arity fixes.
    let opening_only = pcs.proof_shape(0, false);
    assert_eq!(
        shape.sent_extension_elements - opening_only.sent_extension_elements,
        2 * (128 + 2 * (LOG_BITS - ABSORBED) + 2)
    );
    assert_eq!(shape.stir_queries, opening_only.stir_queries);

    budget
        .check_shape(
            &shape,
            ENCODED_ELEMENT_BYTES,
            ENCODED_ELEMENT_BYTES,
            DIGEST_BYTES,
        )
        .unwrap();

    let bits = witness(0x5714);
    let points = points(0x5715);
    let mut prover_chal = challenger();
    let (_, data) = pcs.commit_bits(&bits, &mut prover_chal).unwrap();
    let (_, proof) = pcs.open_at_points(data, &points, &mut prover_chal).unwrap();
    let bytes = postcard::to_allocvec(&proof).unwrap().len();

    // The schedule's own estimate must not be a fiction the real proof exceeds.
    let estimate = shape.max_bytes(ENCODED_ELEMENT_BYTES, ENCODED_ELEMENT_BYTES, DIGEST_BYTES);

    // The proof-of-work search runs in parallel and keeps whichever witness a worker reaches first.
    //
    // A different witness moves every later challenge, so the encoded length varies with the host.
    //
    // What must hold everywhere is that the estimate bounds the real proof without being fiction.
    assert!(
        bytes <= estimate,
        "the estimate understates the proof: {bytes}"
    );
    assert!(
        bytes * 3 > estimate,
        "the estimate is far too loose: {bytes}"
    );
    budget.check_bytes(bytes).unwrap();

    // A ceiling below what the schedule needs must refuse it rather than warn.
    //
    // The ceiling is read off the schedule that was derived, not off one host's figure.
    assert_eq!(
        BinaryWhirBudget {
            max_stir_queries: shape.stir_queries - 1,
            ..budget
        }
        .check_shape(
            &shape,
            ENCODED_ELEMENT_BYTES,
            ENCODED_ELEMENT_BYTES,
            DIGEST_BYTES
        ),
        Err(BudgetError::Queries {
            actual: shape.stir_queries,
            budget: shape.stir_queries - 1
        })
    );
    // The refusal is tied to the proof that was measured, not to one host's figure.
    assert_eq!(
        BinaryWhirBudget {
            max_proof_bytes: bytes - 1,
            ..budget
        }
        .check_bytes(bytes),
        Err(BudgetError::Bytes {
            actual: bytes,
            budget: bytes - 1
        })
    );
}

#[test]
fn the_estimate_covers_the_reductions_and_not_the_opening_alone() {
    // One reduction per claim rides along with the single opening, so the claim count moves it.
    let pcs = whir_pcs(BinaryWhirProfile::proven_list_decoding(
        SECURITY_LEVEL,
        LOG_INV_RATE,
        FOLDING,
    ));
    let two = pcs.proof_shape(2, false);
    let eight = pcs.proof_shape(8, false);
    assert_eq!(
        eight.sent_extension_elements - two.sent_extension_elements,
        6 * (128 + 2 * (LOG_BITS - ABSORBED) + 2)
    );
    // Sending carry and last triples the element each reduction puts on the wire.
    assert_eq!(
        pcs.proof_shape(2, true).sent_extension_elements - two.sent_extension_elements,
        2 * 2 * 128
    );

    // Enough claims that the reductions outweigh what the digest count over-estimates.
    //
    // Reading the schedule alone grows an estimate by one value, the proof by a reduction.
    //
    // At this many claims such an estimate has fallen behind the proof.
    const MANY: usize = 12;
    let bits = witness(0x5718);
    let mut prover_chal = challenger();
    let (_, data) = pcs.commit_bits(&bits, &mut prover_chal).unwrap();
    let (_, proof) = pcs
        .open_at_points(data, &points_at(0x5719, MANY), &mut prover_chal)
        .unwrap();
    let bytes = postcard::to_allocvec(&proof).unwrap().len();
    let estimate = pcs.proof_shape(MANY, false).max_bytes(
        ENCODED_ELEMENT_BYTES,
        ENCODED_ELEMENT_BYTES,
        DIGEST_BYTES,
    );
    assert!(
        bytes <= estimate,
        "the estimate understates the proof: {bytes} against {estimate}"
    );
}

#[test]
fn a_non_first_reduction_over_another_witness_leaves_a_claim_the_commitment_does_not_open() {
    // Invariant: the closing comparison is what ties the reductions to the commitment.
    //
    // Mutation: reduce the middle claim over witness A and open witness B.
    //
    //     - transcript  A's reduction runs on the sponge that absorbed B's root
    //     - reduction   replays and accepts, being a true proof about A
    //     - commitment  accepts, being a true opening of B
    //     - closing     the two surviving values at that point disagree
    //
    // A fresh sponge would split the transcripts at the first draw and pin nothing.
    let profile = BinaryWhirProfile::proven_list_decoding(SECURITY_LEVEL, LOG_INV_RATE, FOLDING);
    let pcs = whir_pcs(profile);
    let inner = whir_prover(profile);

    const FORGED: usize = 1;
    let points = points_at(0x5720, 3);
    let openings = current_openings(&points);

    // Witness A supplies the packing the middle reduction runs over, B the root and the opening.
    let (_, data_a) = pcs
        .commit_bits(&witness(0xAAAA), &mut challenger())
        .unwrap();
    let mut chal = challenger();
    let (root_b, data_b) = pcs.commit_bits(&witness(0xBBBB), &mut chal).unwrap();

    let packing_a = BitPacking::from_packed(data_a.table(0).poly(0)).unwrap();
    let packing_b = BitPacking::from_packed(data_b.table(0).poly(0)).unwrap();

    let mut reductions = Vec::with_capacity(points.len());
    let mut readings = Vec::with_capacity(points.len());
    let mut surviving_points = Vec::with_capacity(points.len());
    let mut surviving_values = Vec::with_capacity(points.len());
    for (index, point) in points.iter().enumerate() {
        let packing = if index == FORGED {
            &packing_a
        } else {
            &packing_b
        };
        let reduction = BitRingSwitch::new(point).unwrap();
        let (sent, surviving_point, surviving_value) =
            reduction.prove::<Ghash128, _, _>(packing, &mut chal);
        readings.push(BitReadings {
            current: Some(reduction.incoming_claim(&sent.tensor)),
            next: None,
        });
        reductions.push(sent);
        surviving_points.push(surviving_point);
        surviving_values.push(surviving_value);
    }

    // The commitment then opens B at all three surviving points.
    let opening = PrescribedPointPcs::<EF, MyChallenger>::open_at(
        &inner,
        data_b,
        &protocol(points.len()),
        &surviving_points,
        &mut chal,
    )
    .unwrap();

    // The first claim agrees, so checking only the first pair would accept this forgery.
    assert_eq!(opening.evals[0].current()[0], surviving_values[0]);
    // The middle claims about the same point disagree, which is what is caught below.
    assert_ne!(opening.evals[FORGED].current()[0], surviving_values[FORGED]);
    // The final claim also agrees, isolating the forged non-first index.
    assert_eq!(opening.evals[2].current()[0], surviving_values[2]);

    let proof = BooleanWhirProof {
        reductions,
        opening,
    };
    let mut verifier_chal = challenger();
    pcs.observe_commitment(&root_b, &mut verifier_chal);
    let refused = pcs
        .verify_readings(&root_b, &openings, &readings, &proof, &mut verifier_chal)
        .unwrap_err();
    assert!(
        matches!(refused, BooleanWhirError::SurvivingClaim),
        "{refused:?}"
    );

    // B's own reductions at the same points are accepted, so the rejection is the mismatch.
    let mut honest = challenger();
    let (root, data) = pcs.commit_bits(&witness(0xBBBB), &mut honest).unwrap();
    let (values, honest_proof) = pcs.open_at_points(data, &points, &mut honest).unwrap();
    let mut verifier_chal = challenger();
    pcs.observe_commitment(&root, &mut verifier_chal);
    pcs.verify_at_points(&root, &points, &values, &honest_proof, &mut verifier_chal)
        .unwrap();
}

#[test]
fn a_tampered_proximity_transcript_is_refused_by_the_opening() {
    // The reductions stay honest, so only the proximity argument can catch this.
    let pcs = whir_pcs(BinaryWhirProfile::proven_list_decoding(
        SECURITY_LEVEL,
        LOG_INV_RATE,
        FOLDING,
    ));
    let bits = witness(0x5721);
    let points = points(0x5722);

    let mut prover_chal = challenger();
    let (commitment, data) = pcs.commit_bits(&bits, &mut prover_chal).unwrap();
    let (values, proof) = pcs.open_at_points(data, &points, &mut prover_chal).unwrap();

    let mut tampered = proof.clone();
    tampered.opening.whir.initial_ood_answers[0] += EF::ONE;
    let mut verifier_chal = challenger();
    pcs.observe_commitment(&commitment, &mut verifier_chal);
    let refused = pcs
        .verify_at_points(&commitment, &points, &values, &tampered, &mut verifier_chal)
        .unwrap_err();
    assert!(
        matches!(refused, BooleanWhirError::Opening(_)),
        "{refused:?}"
    );

    // The untouched proof is accepted, so the rejection is the tampering and nothing else.
    let mut verifier_chal = challenger();
    pcs.observe_commitment(&commitment, &mut verifier_chal);
    pcs.verify_at_points(&commitment, &points, &values, &proof, &mut verifier_chal)
        .unwrap();
}

#[test]
fn the_whir_proof_is_smaller_than_the_folding_only_one_at_the_same_target() {
    let bits = witness(0x5716);
    let points = points(0x5717);

    let basefold = basefold_pcs();
    let mut chal = challenger();
    let (_, data) = basefold.commit_bits(&bits, &mut chal).unwrap();
    let (_, proof) = basefold.open_at_points(data, &points, &mut chal).unwrap();
    let basefold_bytes = postcard::to_allocvec(&proof).unwrap().len();

    let pcs = whir_pcs(BinaryWhirProfile::unique_decoding(
        SECURITY_LEVEL,
        LOG_INV_RATE,
        FOLDING,
    ));
    let mut chal = challenger();
    let (_, data) = pcs.commit_bits(&bits, &mut chal).unwrap();
    let (_, proof) = pcs.open_at_points(data, &points, &mut chal).unwrap();
    let whir_bytes = postcard::to_allocvec(&proof).unwrap().len();

    eprintln!(
        "boolean 2^{LOG_BITS} bits at {SECURITY_LEVEL} bits, unique decoding: \
         basefold {basefold_bytes} bytes, whir {whir_bytes} bytes"
    );
    assert!(
        whir_bytes < basefold_bytes,
        "whir {whir_bytes} bytes against basefold {basefold_bytes} bytes"
    );
}

#[test]
fn the_report_names_every_error_the_adapter_charges() {
    let pcs = whir_pcs(BinaryWhirProfile::proven_list_decoding(
        SECURITY_LEVEL,
        LOG_INV_RATE,
        FOLDING,
    ));
    let security = pcs.readings_security(2, false).unwrap();
    let labels: Vec<&str> = security.terms.iter().map(|term| term.label).collect();

    // The proximity argument's own budget, and the reduction the adapter adds on top of it.
    assert!(labels.contains(&p3_security::whir::WHIR_OPENING_LABEL));
    assert!(labels.contains(&p3_security::BIT_RING_SWITCH_LABEL));

    // The reduction's own term, picked by name rather than by being the weakest of the report.
    let reduction_bits = |successor_tensors| {
        pcs.readings_security(2, successor_tensors)
            .unwrap()
            .terms
            .iter()
            .find(|term| term.label == p3_security::BIT_RING_SWITCH_LABEL)
            .unwrap()
            .bits
            .bits()
    };

    // Carry and last are two more elements batched under the same draw, so the reduction is dearer.
    //
    // The weakest term is the proximity argument's either way, so a minimum would see no change.
    assert!(reduction_bits(true) < reduction_bits(false));

    // The reductions run while the commitment still admits a list of codewords.
    //
    // A prover therefore picks its member after seeing their challenges, so they pay for the list.
    assert!(security.log2_max_candidates > 0.0);
    let alone = p3_security::multilinear::bit_ring_switch_tensors_term(
        2,
        1,
        ABSORBED,
        LOG_BITS - ABSORBED,
        128,
    )
    .bits
    .bits();
    assert!((reduction_bits(false) - (alone - security.log2_max_candidates)).abs() < 1e-9);
}

#[test]
#[should_panic(expected = "soundness regime the domain rejects")]
fn a_regime_swapped_after_derivation_is_refused_at_construction() {
    // The regime is a security claim, so it must not be reachable by editing a derived schedule.
    let domain = BooleanWhirDomain::default();
    let mut config = BinaryWhirProfile::proven_list_decoding(SECURITY_LEVEL, LOG_INV_RATE, FOLDING)
        .config::<EF, EF, MyChallenger, _>(LOG_BITS - ABSORBED, &domain)
        .unwrap();
    config.params.soundness_type = SecurityAssumption::CapacityBound;
    let _ = Prover::new(config, domain, mmcs(0));
}

#[test]
fn a_boolean_trace_commits_and_opens_through_whir() {
    // Two tables of unequal height, the shape a batched prover actually stacks.
    let shapes = [TableShape::new(9, 3), TableShape::new(8, 2)];
    let (arity, _) = plan_stacked_layout(&shapes);

    let domain = BooleanWhirDomain::default();
    let config = BinaryWhirProfile::proven_list_decoding(SECURITY_LEVEL, LOG_INV_RATE, FOLDING)
        .config::<EF, EF, MyChallenger, _>(arity - ABSORBED, &domain)
        .unwrap();
    let cap_height = recommended_cap_height(&config);
    let bits = BooleanWhirPcs::new(Prover::new(config, domain, mmcs(cap_height)), arity).unwrap();
    let pcs = TracePcs::from_commitment(bits);

    let tables: Vec<Table<EF>> = shapes
        .iter()
        .enumerate()
        .map(|(index, shape)| {
            let rows = 1usize << shape.num_variables();
            let mut rng = SmallRng::seed_from_u64(0x7ACE_0000 + index as u64);
            let cells = (0..shape.width() * rows)
                .map(|_| EF::from_bool(rng.random::<bool>()))
                .collect();
            Table::new(RowMajorMatrix::new(cells, rows))
        })
        .collect();

    // The first table is read at the current row and one row ahead, the second at the current row.
    let protocol = OpeningProtocol::new(vec![
        TableSpec::new(
            shapes[0],
            vec![OpeningBatch::new(vec![0, 1, 2], vec![0, 1, 2])],
        ),
        TableSpec::new(shapes[1], vec![OpeningBatch::new(vec![0, 1], Vec::new())]),
    ]);

    let mut prover_chal = challenger();
    let (commitment, data) =
        MultilinearPcs::<EF, MyChallenger>::commit(&pcs, tables, &mut prover_chal).unwrap();
    let proof =
        MultilinearPcs::<EF, MyChallenger>::open(&pcs, data, protocol.clone(), &mut prover_chal)
            .unwrap();

    let mut verifier_chal = challenger();
    MultilinearPcs::<EF, MyChallenger>::verify(
        &pcs,
        &commitment,
        &proof,
        &mut verifier_chal,
        protocol.clone(),
    )
    .unwrap();

    // The schedule must be priced, or a security-checked caller would refuse to prove at all.
    let security = PrescribedPointPcs::<EF, MyChallenger>::prescribed_security(&pcs, &protocol)
        .expect("the trace opening is priced");
    // Several tables take the per-column route, which combines nothing and charges no batching.
    assert_eq!(
        security
            .terms
            .iter()
            .map(|term| term.label)
            .collect::<Vec<_>>(),
        vec![
            p3_security::whir::WHIR_OPENING_LABEL,
            p3_security::BIT_RING_SWITCH_LABEL
        ]
    );

    // A value the trace does not hold must be refused, by the reduction that reads the claim.
    let mut tampered = proof.clone();
    tampered.values[0] += EF::ONE;
    let refused = MultilinearPcs::<EF, MyChallenger>::verify(
        &pcs,
        &commitment,
        &tampered,
        &mut challenger(),
        protocol,
    )
    .unwrap_err();
    assert!(
        matches!(
            refused,
            BooleanTraceCommitmentError::Boolean(BooleanWhirError::ReductionProof(
                BitRingSwitchProofError::ClaimMismatch
            ))
        ),
        "{refused:?}"
    );
}

#[test]
fn one_table_read_whole_takes_the_batched_route_through_whir() {
    // One table whose every column is read at one point takes the combining route.
    //
    // That route draws a fresh challenge of its own, so it charges a batching term.
    let shape = TableShape::new(10, 4);
    let (arity, _) = plan_stacked_layout(&[shape]);

    let domain = BooleanWhirDomain::default();
    let config = BinaryWhirProfile::proven_list_decoding(SECURITY_LEVEL, LOG_INV_RATE, FOLDING)
        .config::<EF, EF, MyChallenger, _>(arity - ABSORBED, &domain)
        .unwrap();
    let cap_height = recommended_cap_height(&config);
    let bits = BooleanWhirPcs::new(Prover::new(config, domain, mmcs(cap_height)), arity).unwrap();
    let pcs = TracePcs::from_commitment(bits);

    let rows = 1usize << shape.num_variables();
    let mut rng = SmallRng::seed_from_u64(0x7ACE_B417);
    let cells = (0..shape.width() * rows)
        .map(|_| EF::from_bool(rng.random::<bool>()))
        .collect();
    let tables = vec![Table::new(RowMajorMatrix::new(cells, rows))];

    let protocol = OpeningProtocol::new(vec![TableSpec::new(
        shape,
        vec![OpeningBatch::new(vec![0, 1, 2, 3], Vec::new())],
    )]);

    let security = PrescribedPointPcs::<EF, MyChallenger>::prescribed_security(&pcs, &protocol)
        .expect("the trace opening is priced");
    assert_eq!(
        security
            .terms
            .iter()
            .map(|term| term.label)
            .collect::<Vec<_>>(),
        vec![
            p3_security::whir::WHIR_OPENING_LABEL,
            p3_security::BIT_RING_SWITCH_LABEL,
            p3_security::COLUMN_BATCH_LABEL
        ]
    );

    // The batching challenge is drawn before the proximity argument names one codeword.
    //
    // It therefore pays the same union bound the reduction pays.
    let charged = security.terms[2].bits.bits();
    let alone = p3_security::multilinear::column_batch_term(1, 2, 128)
        .bits
        .bits();
    assert!((charged - (alone - security.log2_max_candidates)).abs() < 1e-9);

    let mut prover_chal = challenger();
    let (commitment, data) =
        MultilinearPcs::<EF, MyChallenger>::commit(&pcs, tables, &mut prover_chal).unwrap();
    let proof =
        MultilinearPcs::<EF, MyChallenger>::open(&pcs, data, protocol.clone(), &mut prover_chal)
            .unwrap();
    MultilinearPcs::<EF, MyChallenger>::verify(
        &pcs,
        &commitment,
        &proof,
        &mut challenger(),
        protocol,
    )
    .unwrap();
}
