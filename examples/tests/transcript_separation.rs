//! Cross-protocol domain separation for the typed Fiat-Shamir layer.
//!
//! # Overview
//!
//! Fifteen protocols in this workspace seed their transcript from a domain separator.
//!
//! The version byte is a format version each protocol owns, so names carry the separation.
//!
//! ```text
//!     protocol_id = [ 1 | NAME | 0 .. 0 | NAME.len() ]
//!                     ^     ^                 ^
//!                     |     |                 disambiguates zero-padded prefixes
//!                     |     the only field that differs between protocols
//!                     the same byte for all fifteen
//! ```
//!
//! Separation therefore rests entirely on `NAME`, and this file is where that is checked.
//!
//! # Placement
//!
//! Every protocol crate depends on `p3-challenger`, so the check cannot live there.
//! `p3-examples` is a leaf: nothing depends on it, and it already pulls in most of the fifteen.

use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_batch_stark::BatchShape;
use p3_challenger::DuplexChallenger;
use p3_challenger::fs::{DomainSeparator, FieldUnit, PROTOCOL_ID_LEN};
use p3_challenger::testing::{SeedDigest, assert_seeds_pairwise_distinct, seed_digest};
use p3_circle::CirclePcsShape;
use p3_field::extension::BinomialExtensionField;
use p3_fri::{FriShape, PcsShape};
use p3_multi_stark::fractional_gkr::FractionGkrShape;
use p3_multi_stark::lookup::transcript::{LookupInstanceShape, LookupShape};
use p3_multi_stark::rounds::AirDegrees;
use p3_multi_stark::transcript::{MultiStarkInstanceShape, MultiStarkShape};
use p3_multi_stark::zerocheck::transcript::ZerocheckShape;
use p3_stir::{SecurityAssumption, StirInstanceShape, StirRoundShape, StirShape};
use p3_sumcheck::generic_degree::GenericDegreeShape;
use p3_sumcheck::strategy::Basis;
use p3_sumcheck::transcript::SumcheckShape;
use p3_sumcheck::zk::ZkSumcheckShape;
use p3_uni_stark::StarkShape;
use p3_whir::{
    FoldingFactor, ProtocolParameters, WhirConfig, WhirShape, ZkParameters, ZkWhirConfig,
    ZkWhirShape,
};

/// Base field every separator below is derived over.
///
/// One field for all fifteen, so nothing is separated by the field choice.
type F = BabyBear;

/// Extension field every separator below draws its challenges from.
type EF = BinomialExtensionField<F, 4>;

/// Permutation the WHIR configurations are parameterised by.
type Perm = Poseidon2BabyBear<16>;

/// Challenger the WHIR configurations are parameterised by.
///
/// WHIR carries the challenger in its config type, but seeding never touches a sponge.
type Ch = DuplexChallenger<F, Perm, 16, 8>;

/// Sponge alphabet every protocol in the workspace seeds through.
type Alphabet = FieldUnit<F>;

/// One labelled configuration of one protocol, reduced to the separator it derives.
type Case = (String, DomainSeparator<Alphabet>);

/// Number of protocols on the typed transcript layer.
///
/// A protocol added without an entry below leaves its name unchecked against the others.
const NUM_PROTOCOLS: usize = 15;

/// Configurations swept per protocol: one default, then two single-field moves of it.
///
/// The pairwise check is quadratic, so the sweep is a budget rather than a maximum.
///
/// ```text
///     15 protocols x 3 configurations = 45 seeds -> 990 pairs
/// ```
const CASES_PER_PROTOCOL: usize = 3;

/// Variable count both WHIR pipelines are configured at.
const WHIR_NUM_VARIABLES: usize = 16;

/// Log-inverse rate of the first committed WHIR codeword.
const WHIR_STARTING_LOG_INV_RATE: usize = 1;

/// Label one configuration of one protocol.
fn case(protocol: &str, configuration: &str, separator: DomainSeparator<Alphabet>) -> Case {
    (format!("{protocol}/{configuration}"), separator)
}

/// The protocol name stored inside an identifier, stripped of its zero padding.
fn protocol_name(separator: &DomainSeparator<Alphabet>) -> &[u8] {
    let id = separator.protocol_id();
    // The final byte is the name length, and the name starts right after the version.
    let len = usize::from(id[PROTOCOL_ID_LEN - 1]);
    &id[1..1 + len]
}

/// The uni-STARK cases: a plain single-table proof, then two single-field moves.
fn uni_stark_cases() -> Vec<Case> {
    let plain = StarkShape {
        log_ext_degree: 5,
        log_degree: 5,
        main_width: 2,
        preprocessed_width: 0,
        num_public_values: 1,
        num_periodic_columns: 0,
        num_quotient_chunks: 2,
        opens_main_next_row: true,
        opens_preprocessed_next_row: false,
        has_randomization: false,
        ood_pow_bits: 0,
    };

    let mut wider = plain.clone();
    wider.main_width += 1;

    let mut randomized = plain.clone();
    randomized.has_randomization = true;

    [("plain", plain), ("main_width", wider), ("zk", randomized)]
        .into_iter()
        .map(|(name, shape)| case("p3-uni-stark", name, shape.domain_separator::<F, EF>()))
        .collect()
}

/// The batch-STARK cases: a two-instance batch, then two single-field moves.
fn batch_stark_cases() -> Vec<Case> {
    let plain = BatchShape {
        trace_widths: vec![2, 3],
        public_value_counts: vec![0, 1],
        preprocessed_widths: vec![0, 0],
        has_preprocessed_commitment: false,
        num_lookup_instances: 0,
        lookup_pow_bits: 0,
        has_randomization_commitment: false,
        ood_pow_bits: 0,
    };

    let mut wider = plain.clone();
    wider.trace_widths[0] += 1;

    let mut preprocessed = plain.clone();
    preprocessed.has_preprocessed_commitment = true;

    [
        ("plain", plain),
        ("trace_widths", wider),
        ("preprocessed", preprocessed),
    ]
    .into_iter()
    .map(|(name, shape)| case("p3-batch-stark", name, shape.domain_separator::<F, EF>()))
    .collect()
}

/// The FRI low-degree-test cases: three commit rounds, then two single-field moves.
fn fri_cases() -> Vec<Case> {
    let plain = FriShape {
        log_arities: vec![3, 3, 2],
        final_poly_len: 1,
        commit_pow_bits: 0,
        query_pow_bits: 0,
        num_queries: 2,
        index_bits: 8,
        log_blowup: 1,
        max_log_arity: 3,
    };

    let mut wider_blowup = plain.clone();
    wider_blowup.log_blowup += 1;

    let mut more_queries = plain.clone();
    more_queries.num_queries += 1;

    [
        ("plain", plain),
        ("log_blowup", wider_blowup),
        ("num_queries", more_queries),
    ]
    .into_iter()
    .map(|(name, shape)| case("p3-fri", name, shape.domain_separator::<F, EF>()))
    .collect()
}

/// The FRI PCS cases: one commitment opened twice, then two single-field moves.
fn fri_pcs_cases() -> Vec<Case> {
    let plain = PcsShape {
        claimed_evaluation_counts: vec![vec![vec![3, 1]]],
        batch_pow_bits: 0,
    };

    let mut ground = plain.clone();
    ground.batch_pow_bits += 1;

    let mut wider = plain.clone();
    wider.claimed_evaluation_counts[0][0][0] += 1;

    [
        ("plain", plain),
        ("batch_pow_bits", ground),
        ("claim_widths", wider),
    ]
    .into_iter()
    .map(|(name, shape)| case("p3-fri-pcs", name, shape.domain_separator::<F, EF>()))
    .collect()
}

/// The Circle PCS cases: one commit round, then two single-field moves.
fn circle_pcs_cases() -> Vec<Case> {
    let plain = CirclePcsShape {
        opened_widths: vec![vec![vec![3, 1]]],
        num_commit_rounds: 1,
        commit_pow_bits: 0,
        query_pow_bits: 0,
        num_queries: 2,
        index_bits: 8,
        log_blowup: 1,
    };

    let mut longer = plain.clone();
    longer.num_commit_rounds += 1;

    let mut wider_blowup = plain.clone();
    wider_blowup.log_blowup += 1;

    [
        ("plain", plain),
        ("num_commit_rounds", longer),
        ("log_blowup", wider_blowup),
    ]
    .into_iter()
    .map(|(name, shape)| case("p3-circle-pcs", name, shape.domain_separator::<F, EF>()))
    .collect()
}

/// The STIR cases: one instance running one round, then two single-field moves.
fn stir_cases() -> Vec<Case> {
    let round = StirRoundShape {
        folding_pow_bits: 0,
        num_ood_samples: 1,
        pow_bits: 0,
        num_queries: 3,
        log_fold_domain_size: 6,
        log_degree: 8,
        log_domain_size: 9,
        log_folding_factor: 3,
        domain_shift: 31,
        eta_bits: 0.25_f64.to_bits(),
    };
    let instance = StirInstanceShape {
        rounds: vec![round],
        final_folding_pow_bits: 0,
        final_poly_len: 2,
        final_pow_bits: 0,
        final_queries: 2,
        final_log_domain_size: 5,
        log_starting_degree: 8,
        log_blowup: 1,
        log_folding_factor: 3,
        log_starting_folding_factor: 3,
        log_final_degree: 1,
        security_level: 100,
        max_pow_bits: 20,
        soundness_type: SecurityAssumption::JohnsonBound,
        final_eta_bits: 0.125_f64.to_bits(),
        max_log_final_poly_len: None,
    };
    let plain = StirShape {
        commits_initial: true,
        instances: vec![instance],
    };

    let mut committed_elsewhere = plain.clone();
    committed_elsewhere.commits_initial = false;

    let mut wider_blowup = plain.clone();
    wider_blowup.instances[0].log_blowup += 1;

    [
        ("plain", plain),
        ("commits_initial", committed_elsewhere),
        ("log_blowup", wider_blowup),
    ]
    .into_iter()
    .map(|(name, shape)| case("p3-stir", name, shape.domain_separator::<F, EF>()))
    .collect()
}

/// One code rate per intermediate round, growing with the folding schedule.
fn whir_round_rates(folding_factor: &FoldingFactor, starting_rate: usize) -> Vec<usize> {
    let schedule = folding_factor
        .compute_folding_schedule(WHIR_NUM_VARIABLES)
        .expect("the fixture schedule is valid");
    let mut rate = starting_rate;
    schedule[..schedule.len() - 1]
        .iter()
        .map(|folding| {
            rate += folding - 1;
            rate
        })
        .collect()
}

/// Baseline WHIR user parameters, shared by the plain and hiding pipelines.
fn whir_params() -> ProtocolParameters {
    let folding_factor = FoldingFactor::Constant(4);
    ProtocolParameters {
        security_level: 32,
        pow_bits: 12,
        round_log_inv_rates: whir_round_rates(&folding_factor, WHIR_STARTING_LOG_INV_RATE),
        folding_factor,
        soundness_type: SecurityAssumption::CapacityBound,
        starting_log_inv_rate: WHIR_STARTING_LOG_INV_RATE,
    }
}

/// The plain WHIR cases: the baseline parameters, then two single-field moves.
fn whir_cases() -> Vec<Case> {
    let mut stronger = whir_params();
    stronger.security_level += 1;

    let mut less_grinding = whir_params();
    less_grinding.pow_bits -= 1;

    [
        ("plain", whir_params()),
        ("security_level", stronger),
        ("pow_bits", less_grinding),
    ]
    .into_iter()
    .map(|(name, params)| {
        let config = WhirConfig::<EF, F, Ch>::new(WHIR_NUM_VARIABLES, params)
            .expect("the fixture parameters are valid");
        case(
            "p3-whir",
            name,
            WhirShape::new(&config, WHIR_NUM_OPENING_CLAIMS).domain_separator::<F, EF>(),
        )
    })
    .collect()
}

/// The hiding WHIR cases: the same plain parameters, then two moves of the mask.
fn zk_whir_cases() -> Vec<Case> {
    let plain = ZkParameters {
        ell_zk: 4,
        mask_log_inv_rate: 1,
    };

    let mut longer_mask = plain.clone();
    longer_mask.ell_zk += 1;

    let mut sparser_mask = plain.clone();
    sparser_mask.mask_log_inv_rate += 1;

    [
        ("plain", plain),
        ("ell_zk", longer_mask),
        ("mask_log_inv_rate", sparser_mask),
    ]
    .into_iter()
    .map(|(name, zk)| {
        let config = ZkWhirConfig::<EF, F, Ch>::new(WHIR_NUM_VARIABLES, whir_params(), zk)
            .expect("the fixture parameters are valid");
        case(
            "p3-whir-hvzk",
            name,
            ZkWhirShape::new(&config).domain_separator::<F, EF>(),
        )
    })
    .collect()
}

/// The multi-STARK zerocheck cases: two AIRs, then two single-field moves.
fn zerocheck_cases() -> Vec<Case> {
    let plain = ZerocheckShape {
        log_height: 10,
        lookup_point_len: 2,
        pow_bits: 4,
        air_degrees: vec![
            AirDegrees {
                constraints: 3,
                interactions: 2,
            },
            AirDegrees {
                constraints: 2,
                interactions: 0,
            },
        ],
    };

    let mut taller = plain.clone();
    taller.log_height += 1;

    let mut ground = plain.clone();
    ground.pow_bits += 1;

    [
        ("plain", plain),
        ("log_height", taller),
        ("pow_bits", ground),
    ]
    .into_iter()
    .map(|(name, shape)| {
        case(
            "p3-multi-stark-zerocheck",
            name,
            shape.domain_separator::<F, EF>(),
        )
    })
    .collect()
}

/// The multi-STARK lookup cases: two instances over two buses, then two single-field moves.
fn lookup_cases() -> Vec<Case> {
    let plain = LookupShape {
        num_variables: 7,
        max_width: 3,
        num_buses: 2,
        instances: vec![
            LookupInstanceShape {
                air_index: 0,
                num_variables: 5,
                base_offset: 0,
                bus_ids: vec![0, 1],
            },
            LookupInstanceShape {
                air_index: 2,
                num_variables: 4,
                base_offset: 64,
                bus_ids: vec![1],
            },
        ],
    };

    let mut more_buses = plain.clone();
    more_buses.num_buses += 1;

    let mut wider = plain.clone();
    wider.max_width += 1;

    [
        ("plain", plain),
        ("num_buses", more_buses),
        ("max_width", wider),
    ]
    .into_iter()
    .map(|(name, shape)| {
        case(
            "p3-multi-stark-lookup",
            name,
            shape.domain_separator::<F, EF>(),
        )
    })
    .collect()
}

/// The multi-STARK fractional-GKR cases: three layer counts.
fn fraction_gkr_cases() -> Vec<Case> {
    [3, 4, 5]
        .into_iter()
        .map(|num_variables| {
            let shape = FractionGkrShape { num_variables };
            (
                format!("p3-multi-stark-fraction-gkr/num_variables={num_variables}"),
                shape.domain_separator::<F, EF>(),
            )
        })
        .collect()
}

/// The generic-degree sumcheck cases: four rounds of degree three, then two moves.
fn sumcheck_generic_degree_cases() -> Vec<Case> {
    [
        ("plain", GenericDegreeShape::new(4, 3, 0)),
        ("num_rounds", GenericDegreeShape::new(5, 3, 0)),
        ("degree", GenericDegreeShape::new(4, 4, 0)),
    ]
    .into_iter()
    .map(|(name, shape)| {
        case(
            "p3-sumcheck-generic-degree",
            name,
            shape.domain_separator::<F, EF>(),
        )
    })
    .collect()
}

/// The quadratic sumcheck cases: four rounds in the evaluation basis, then two moves.
fn sumcheck_quadratic_cases() -> Vec<Case> {
    [
        ("plain", SumcheckShape::new(4, 0, Basis::Evaluation)),
        ("num_rounds", SumcheckShape::new(5, 0, Basis::Evaluation)),
        ("basis", SumcheckShape::new(4, 0, Basis::Projective)),
    ]
    .into_iter()
    .map(|(name, shape)| {
        case(
            "p3-sumcheck-quadratic",
            name,
            shape.domain_separator::<F, EF>(),
        )
    })
    .collect()
}

/// Every protocol's cases, the default configuration first in each group.
/// Number of opening claims the WHIR fixture runs with.
///
/// One point is drawn and one evaluation batch absorbed per claim.
const WHIR_NUM_OPENING_CLAIMS: usize = 1;

/// The multi-STARK statement cases: a two-instance batch, then two single-field moves.
fn multi_stark_cases() -> Vec<Case> {
    let plain = MultiStarkShape {
        instances: vec![
            MultiStarkInstanceShape {
                num_variables: 8,
                main_width: 3,
                preprocessed_width: 0,
                num_public_values: 2,
            },
            MultiStarkInstanceShape {
                num_variables: 6,
                main_width: 5,
                preprocessed_width: 2,
                num_public_values: 1,
            },
        ],
        pow_bits: 0,
    };

    let mut wider = plain.clone();
    wider.instances[0].main_width += 1;

    let mut ground = plain.clone();
    ground.pow_bits += 1;

    [
        ("plain", plain),
        ("main_width", wider),
        ("pow_bits", ground),
    ]
    .into_iter()
    .map(|(name, shape)| case("p3-multi-stark", name, shape.domain_separator::<F>()))
    .collect()
}

/// The hiding sumcheck cases: a plain masked batch, then two single-field moves.
fn zk_sumcheck_cases() -> Vec<Case> {
    let plain = ZkSumcheckShape::new_batching(3, 4, 0);
    let longer_mask = ZkSumcheckShape::new_batching(3, 5, 0);
    let ground = ZkSumcheckShape::new_batching(3, 4, 1);

    [
        ("plain", plain),
        ("ell_zk", longer_mask),
        ("pow_bits", ground),
    ]
    .into_iter()
    .map(|(name, shape)| case("p3-sumcheck-hvzk", name, shape.domain_separator::<F, EF>()))
    .collect()
}

fn protocols() -> Vec<Vec<Case>> {
    vec![
        uni_stark_cases(),
        batch_stark_cases(),
        fri_cases(),
        fri_pcs_cases(),
        circle_pcs_cases(),
        stir_cases(),
        whir_cases(),
        zk_whir_cases(),
        zerocheck_cases(),
        lookup_cases(),
        fraction_gkr_cases(),
        sumcheck_generic_degree_cases(),
        sumcheck_quadratic_cases(),
        multi_stark_cases(),
        zk_sumcheck_cases(),
    ]
}

/// One default configuration per protocol.
fn default_cases() -> Vec<Case> {
    protocols()
        .into_iter()
        .map(|group| {
            group
                .into_iter()
                .next()
                .expect("every protocol lists at least its default configuration")
        })
        .collect()
}

/// Every configuration of every protocol.
fn all_cases() -> Vec<Case> {
    protocols().into_iter().flatten().collect()
}

/// Digest every case's seed, keeping its label.
fn digested(cases: Vec<Case>) -> Vec<(String, SeedDigest)> {
    cases
        .into_iter()
        .map(|(label, separator)| (label, seed_digest(&separator)))
        .collect()
}

#[test]
fn every_protocol_is_listed_here() {
    // This compares two hand-maintained numbers against each other, and nothing wider.
    //
    //     builder added, count not bumped  ->  caught here
    //     count bumped, builder missing    ->  caught here
    //     new protocol, neither touched    ->  not caught
    //
    // A protocol on the typed layer that never joins the list is never compared at all.
    //
    // Enumerating them across a workspace at compile time has no clean form.
    //
    // So the list is a convention this test keeps consistent, not one it discovers.
    assert_eq!(default_cases().len(), NUM_PROTOCOLS);
}

#[test]
fn the_protocol_name_is_the_only_field_that_separates_two_protocols() {
    // The version byte is a format version, owned by one protocol and bumped by it alone.
    //
    //     [version | name | 0 .. 0 | name_len]
    //                ^^^^            ^^^^^^^^
    //                the only fields that may part two protocols
    //
    // So two protocols must stay apart with every version byte forced to agree.
    // A pair that only differs in that byte would collide the day either one bumps.
    let normalized: Vec<(String, [u8; PROTOCOL_ID_LEN])> = default_cases()
        .into_iter()
        .map(|(label, separator)| {
            let mut id = *separator.protocol_id();
            id[0] = 0;
            (label, id)
        })
        .collect();

    for (index, (left_label, left)) in normalized.iter().enumerate() {
        for (right_label, right) in &normalized[index + 1..] {
            assert_ne!(
                left, right,
                "`{left_label}` and `{right_label}` are parted only by their version byte",
            );
        }
    }
}

#[test]
fn a_shared_name_prefix_is_separated_by_the_name_length_byte() {
    // Two pairs of names stand in a prefix relation:
    //
    //     [1 | p3-fri       | 0 .. 0 |  6]
    //     [1 | p3-fri-pcs   | 0 .. 0 | 10]
    //
    //     [1 | p3-whir      | 0 .. 0 |  7]
    //     [1 | p3-whir-hvzk | 0 .. 0 | 12]
    //
    // Zero padding alone cannot tell a short name from a longer one starting with it.
    // The final byte holds the name length, and it is what keeps the two apart.
    let pairs = [
        (fri_cases(), fri_pcs_cases()),
        (whir_cases(), zk_whir_cases()),
    ];

    for (short_cases, long_cases) in pairs {
        let short = &short_cases[0].1;
        let long = &long_cases[0].1;

        // The prefix relation is real, so the name bytes alone do not separate them.
        let short_name = protocol_name(short);
        let long_name = protocol_name(long);
        assert!(long_name.starts_with(short_name));

        // The length byte differs, which is what makes the two identifiers differ.
        assert_ne!(
            short.protocol_id()[PROTOCOL_ID_LEN - 1],
            long.protocol_id()[PROTOCOL_ID_LEN - 1],
        );
    }
}

#[test]
fn no_two_configurations_of_any_two_protocols_share_a_seed() {
    // Invariant: the seed is unique across the whole cross-product, not merely across defaults.
    //
    // Each protocol contributes its default plus two single-field moves of it.
    // Pairwise over the union asks both questions at once:
    //
    //     within one protocol  ->  does each knob reach the seed, separately from the others
    //     across two protocols ->  can any configuration of one reach another's seed
    let seeds = digested(all_cases());
    assert_eq!(seeds.len(), CASES_PER_PROTOCOL * NUM_PROTOCOLS);
    assert_seeds_pairwise_distinct(&seeds);
}
