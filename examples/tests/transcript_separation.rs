//! Cross-protocol domain separation for the typed Fiat-Shamir layer.
//!
//! # Overview
//!
//! Twenty-five protocols in this workspace seed their transcript from a domain separator.
//!
//! The version byte is a format version each protocol owns, so names carry the separation.
//!
//! ```text
//!     protocol_id = [ 1 | NAME  | 0 .. 0 | NAME.len() ]
//!                       ^     ^                  ^
//!                       |     |                  disambiguates padded prefixes
//!                       |     the only field that differs between protocols
//!                       the same byte for all twenty-five
//! ```
//!
//! Separation therefore rests entirely on the name.
//!
//! This file is where that is checked.
//!
//! # Placement
//!
//! Every protocol crate depends on `p3-challenger`, so the check cannot live there.
//!
//! `p3-examples` is a leaf: nothing depends on it.
//!
//! It also already pulls in most of the twenty-five.

use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_batch_stark::BatchShape;
use p3_challenger::DuplexChallenger;
use p3_challenger::fs::{DomainSeparator, FieldUnit, PROTOCOL_ID_LEN};
use p3_challenger::testing::{
    SeedDigest, assert_seeds_pairwise_distinct, pow_difficulties, seed_digest,
};
use p3_circle::CirclePcsShape;
use p3_field::extension::BinomialExtensionField;
use p3_fri::{FriShape, PcsShape};
use p3_multi_stark::fractional_gkr::FractionGkrShape;
use p3_multi_stark::logup_star::transcript::{LogupStarShape, LogupStarTableShape};
use p3_multi_stark::lookup::transcript::{LookupInstanceShape, LookupShape};
use p3_multi_stark::rounds::AirDegrees;
use p3_multi_stark::transcript::{MultiStarkInstanceShape, MultiStarkShape};
use p3_multi_stark::zerocheck::transcript::ZerocheckShape;
use p3_security::fri::FriRegime;
use p3_security::grinding::{
    GrindingBudget, GrindingSites, RecordedGrind, UNPRICED_GRINDING_SITES, grinding_step,
    is_unpriced_grinding_site,
};
use p3_stir::pcs_transcript::{
    StirPcsBucketShape, StirPcsClaimShape, StirPcsCommitmentShape, StirPcsOpeningShape,
};
use p3_stir::{SecurityAssumption, StirInstanceShape, StirRoundShape, StirShape};
use p3_sumcheck::generic_degree::GenericDegreeShape;
use p3_sumcheck::ring_switch::RingSwitchShape;
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
/// One field for all twenty-five, so nothing is separated by the field choice.
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
const NUM_PROTOCOLS: usize = 25;

/// Configurations swept per protocol: one default, then two single-field moves of it.
///
/// The pairwise check is quadratic, so the sweep is a budget rather than a maximum.
///
/// A protocol with no configuration at all contributes one case instead of three.
///
/// ```text
///     23 protocols x 3 + 2 protocols x 1 = 71 seeds -> 2485 pairs
/// ```
const MAX_CASES_PER_PROTOCOL: usize = 3;

/// Phases whose description is fixed, so there is nothing to sweep.
///
/// A commitment is one Merkle root at every configuration.
///
/// These contribute one case each, and every other protocol contributes three.
const CONFIGURATION_FREE_PHASES: [&str; 2] =
    ["p3-sumcheck-layout-commitment", "p3-whir-hvzk-commitment"];

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
        batch_pow_bits: 0,
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
    zk_whir_parameters()
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

/// The three hiding parameter sets every hiding builder below sweeps.
///
/// ```text
///     plain              the baseline mask
///     ell_zk             one more mask coefficient
///     mask_log_inv_rate  one more halving of the mask rate
/// ```
fn zk_whir_parameters() -> [(&'static str, ZkParameters); 3] {
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
}

/// The masked base-case cases: the closing phase of each hiding configuration.
///
/// The base case runs under a seed of its own.
///
/// It is therefore a protocol of its own here.
fn zk_whir_base_case_cases() -> Vec<Case> {
    zk_whir_parameters()
        .into_iter()
        .map(|(name, zk)| {
            let config = ZkWhirConfig::<EF, F, Ch>::new(WHIR_NUM_VARIABLES, whir_params(), zk)
                .expect("the fixture parameters are valid");
            // The closing phase's own description hangs off the run's shape.
            let base = ZkWhirShape::new(&config).base_case;
            case("p3-whir-hvzk-base", name, base.domain_separator::<F, EF>())
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

/// The logUp* cases: two tables of different sizes, then two single-field moves.
fn logup_star_cases() -> Vec<Case> {
    let plain = LogupStarShape {
        tables: vec![
            LogupStarTableShape {
                num_variables: 5,
                width: 3,
                readers: vec![4, 3],
            },
            LogupStarTableShape {
                num_variables: 4,
                width: 2,
                readers: vec![3],
            },
        ],
        num_variables: 7,
    };

    let mut wider = plain.clone();
    wider.tables[0].width += 1;

    // Moving one reader from the first table to the second keeps every total unchanged.
    //
    //     plain:  [reader, reader | reader]
    //     split:  [reader         | reader, reader]
    let mut split = plain.clone();
    let moved = plain.tables[0].readers[1];
    split.tables[0].readers.pop();
    split.tables[1].readers.push(moved);

    [("plain", plain), ("width", wider), ("readers", split)]
        .into_iter()
        .map(|(name, shape)| {
            case(
                "p3-multi-stark-logup-star",
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

/// The ring-switching cases: three coordinate counts of the incoming evaluation point.
///
/// The point width is the reduction's only knob.
///
/// Everything else its description declares follows from the field pair.
fn ring_switch_cases() -> Vec<Case> {
    [6, 7, 8]
        .into_iter()
        .map(|num_variables| {
            let shape = RingSwitchShape::new(num_variables);
            (
                format!("p3-sumcheck-ring-switch/num_variables={num_variables}"),
                shape.domain_separator::<F, EF>(),
            )
        })
        .collect()
}

/// The STIR PCS commitment cases: one root, then two other group counts.
fn stir_pcs_commitment_cases() -> Vec<Case> {
    [1, 2, 3]
        .into_iter()
        .map(|num_roots| {
            let shape = StirPcsCommitmentShape::new(num_roots);
            (
                format!("p3-stir-pcs-commitment/num_roots={num_roots}"),
                shape.domain_separator::<F>(),
            )
        })
        .collect()
}

/// The stacked-layout commitment case: the phase that binds the committed root.
///
/// The phase has no configuration at all.
///
/// It contributes one case.
///
/// The suite still compares its name against every other.
fn layout_commitment_cases() -> Vec<Case> {
    vec![case(
        "p3-sumcheck-layout-commitment",
        "only",
        p3_sumcheck::layout::commitment_domain_separator::<F>(),
    )]
}

/// The hiding WHIR commitment case: the phase that binds the committed root.
///
/// Like the stacked-layout one, it has no configuration.
fn zk_whir_commitment_cases() -> Vec<Case> {
    vec![case(
        "p3-whir-hvzk-commitment",
        "only",
        p3_whir::transcript::zk::commitment_domain_separator::<F>(),
    )]
}

/// The hiding WHIR claim cases: one statement, then two moves of its shape.
///
/// ```text
///     claims  points     steps
///     1       16 wide    point(16), eval
///     2       16 wide    point(16), eval, point(16), eval
///     1       15 wide    point(15), eval
/// ```
fn zk_whir_claim_cases() -> Vec<Case> {
    [(1, WHIR_NUM_VARIABLES), (2, WHIR_NUM_VARIABLES), (1, 15)]
        .into_iter()
        .map(|(num_claims, num_variables)| {
            case(
                "p3-whir-hvzk-claims",
                &format!("claims={num_claims},vars={num_variables}"),
                p3_whir::transcript::zk::ZkClaimsShape::new(num_claims, num_variables)
                    .domain_separator::<F, EF>(),
            )
        })
        .collect()
}

/// The STIR PCS claim cases: one grouping, then two regroupings of the same widths.
///
/// All three flatten to the same widths.
///
/// ```text
///     one matrix, two points   [[[3, 3]]]
///     two matrices, one point  [[[3], [3]]]
///     two commitments          [[[3]], [[3]]]
/// ```
///
/// Only the containers part them, which is what makes them worth listing here.
fn stir_pcs_claim_cases() -> Vec<Case> {
    let groupings = [
        ("one_matrix_two_points", vec![vec![vec![3, 3]]]),
        ("two_matrices_one_point", vec![vec![vec![3], vec![3]]]),
        ("two_commitments", vec![vec![vec![3]], vec![vec![3]]]),
    ];

    groupings
        .into_iter()
        .map(|(name, claim_widths)| {
            let shape = StirPcsClaimShape { claim_widths };
            (
                format!("p3-stir-pcs-claims/{name}"),
                shape.domain_separator::<F, EF>(),
            )
        })
        .collect()
}

/// The STIR PCS opening cases: one merging bucket, then two single-field moves.
fn stir_pcs_opening_cases() -> Vec<Case> {
    let merging = StirPcsBucketShape {
        log_lde_height: 9,
        log_native_heights: vec![8, 6],
        log_first_fold_arity: 3,
        num_query_draws: 3,
    };

    // A bucket merging nothing draws no merging challenge.
    //
    // Its block sequence therefore differs from a merging one.
    let mut unmerged = merging.clone();
    unmerged.log_native_heights = vec![8];

    [
        ("one_merging_bucket", vec![merging.clone()]),
        ("one_unmerged_bucket", vec![unmerged]),
        ("two_merging_buckets", vec![merging.clone(), merging]),
    ]
    .into_iter()
    .map(|(name, buckets)| {
        let shape = StirPcsOpeningShape::new(buckets);
        (
            format!("p3-stir-pcs-opening/{name}"),
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
                main_next_row_columns: vec![0, 1, 2],
                preprocessed_next_row_columns: vec![],
            },
            MultiStarkInstanceShape {
                num_variables: 6,
                main_width: 5,
                preprocessed_width: 2,
                num_public_values: 1,
                main_next_row_columns: vec![0, 1, 2, 3, 4],
                preprocessed_next_row_columns: vec![0, 1],
            },
        ],
        pow_bits: 0,
        has_indexed: false,
    };

    let mut wider = plain.clone();
    wider.instances[0].main_width += 1;

    // A batch declaring an indexed read plays one more bracket.
    //
    // It is a different sequence rather than the same one relabelled.
    let mut indexed = plain.clone();
    indexed.has_indexed = true;

    [
        ("plain", plain),
        ("main_width", wider),
        ("indexed", indexed),
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

fn stir_pcs_batch_cases() -> Vec<Case> {
    [("plain", 1), ("pow_bits_2", 2), ("pow_bits_3", 3)]
        .into_iter()
        .map(|(name, bits)| {
            case(
                "p3-stir-pcs-batch",
                name,
                p3_stir::batch_domain_separator::<F, EF>(bits),
            )
        })
        .collect()
}

#[test]
fn circle_and_stir_patterns_match_grinding_budgets() {
    for (batch, commit, query) in [(0, 0, 0), (5, 3, 7)] {
        let shape = CirclePcsShape {
            opened_widths: vec![vec![vec![1]]],
            num_commit_rounds: 2,
            batch_pow_bits: batch,
            commit_pow_bits: commit,
            query_pow_bits: query,
            num_queries: 2,
            index_bits: 8,
            log_blowup: 1,
        };
        let sites = GrindingSites {
            batch_combination: batch,
            ..GrindingSites::NONE
        };
        let budget = GrindingBudget::from_sites(&sites).with_fri(&FriRegime {
            log_blowup: 1,
            num_queries: 2,
            log_final_poly_len: 0,
            max_log_arity: 1,
            commit_pow_bits: commit,
            query_pow_bits: query,
        });
        let recorded: Vec<_> = pow_difficulties(shape.domain_separator::<F, EF>().pattern())
            .into_iter()
            .map(|(label, bits)| RecordedGrind::new("p3-circle-pcs", label, bits))
            .collect();
        budget.check(&["p3-circle-pcs"], &recorded).unwrap();
        let recorded: Vec<_> = if batch == 0 {
            vec![]
        } else {
            pow_difficulties(p3_stir::batch_domain_separator::<F, EF>(batch).pattern())
                .into_iter()
                .map(|(label, bits)| RecordedGrind::new("p3-stir-pcs-batch", label, bits))
                .collect()
        };
        GrindingBudget::from_sites(&sites)
            .check(&["p3-stir-pcs-batch"], &recorded)
            .unwrap();
    }
}

fn protocols() -> Vec<Vec<Case>> {
    vec![
        uni_stark_cases(),
        batch_stark_cases(),
        fri_cases(),
        fri_pcs_cases(),
        circle_pcs_cases(),
        stir_cases(),
        stir_pcs_batch_cases(),
        layout_commitment_cases(),
        stir_pcs_commitment_cases(),
        stir_pcs_claim_cases(),
        stir_pcs_opening_cases(),
        whir_cases(),
        zk_whir_commitment_cases(),
        zk_whir_claim_cases(),
        zk_whir_cases(),
        zk_whir_base_case_cases(),
        zerocheck_cases(),
        lookup_cases(),
        logup_star_cases(),
        fraction_gkr_cases(),
        sumcheck_generic_degree_cases(),
        sumcheck_quadratic_cases(),
        multi_stark_cases(),
        zk_sumcheck_cases(),
        ring_switch_cases(),
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
    //     - builder added, count not bumped  ->  caught here
    //     - count bumped, builder missing    ->  caught here
    //     - new protocol, neither touched    ->  not caught
    //
    // A protocol on the typed layer that never joins the list is never compared at all.
    //
    // Enumerating them across a workspace at compile time has no clean form.
    //
    // So the list is a convention this test keeps consistent, not one it discovers.
    //
    // Three names cannot join, because their shapes are crate-private to `p3-sumcheck`.
    //
    // - `p3-sumcheck-layout-opening`
    // - `p3-sumcheck-layout-ood`
    // - `p3-sumcheck-layout-batching`
    //
    // Publishing them to reach this file would widen that crate's API for a test.
    //
    // They are compared against the crate's other names in its own suite instead.
    //
    // Two more cannot join for a different reason.
    //
    // `p3-binary-pcs` and `p3-sumcheck-bit-ring-switch` seed over a binary tower field.
    //
    // Their separators therefore have a different sponge alphabet, and a different type.
    //
    // Two protocols over different alphabets cannot collide on a sponge state anyway.
    //
    // Their own knobs are swept inside their own crates.
    assert_eq!(default_cases().len(), NUM_PROTOCOLS);
}

#[test]
fn the_protocol_name_is_the_only_field_that_separates_two_protocols() {
    // The version byte is a format version, owned by one protocol and bumped by it alone.
    //
    //     [ version | name  | 0 .. 0 | name_len ]
    //                  ^^^^              ^^^^^^^^
    //                  the only fields that may part two protocols
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
    // Several names stand in a prefix relation:
    //
    //     [1 | p3-fri                  | 0 .. 0 |  6]
    //     [1 | p3-fri-pcs              | 0 .. 0 | 10]
    //
    //     - [1 | p3-whir                 | 0 .. 0 |  7]
    //     - [1 | p3-whir-hvzk            | 0 .. 0 | 12]
    //     - [1 | p3-whir-hvzk-base       | 0 .. 0 | 17]
    //     - [1 | p3-whir-hvzk-claims     | 0 .. 0 | 19]
    //     - [1 | p3-whir-hvzk-commitment | 0 .. 0 | 23]
    //
    //     - [1 | p3-stir                 | 0 .. 0 |  7]
    //     - [1 | p3-stir-pcs-batch       | 0 .. 0 | 17]
    //     - [1 | p3-stir-pcs-claims      | 0 .. 0 | 18]
    //     - [1 | p3-stir-pcs-opening     | 0 .. 0 | 19]
    //     - [1 | p3-stir-pcs-commitment  | 0 .. 0 | 22]
    //
    // Zero padding alone cannot tell a short name from a longer one starting with it.
    //
    // The final byte holds the name length.
    //
    // That byte is what keeps the two apart.
    let pairs = [
        (fri_cases(), fri_pcs_cases()),
        (whir_cases(), zk_whir_cases()),
        (stir_cases(), stir_pcs_batch_cases()),
        (zk_whir_cases(), zk_whir_base_case_cases()),
        (zk_whir_cases(), zk_whir_claim_cases()),
        (zk_whir_cases(), zk_whir_commitment_cases()),
        (stir_cases(), stir_pcs_commitment_cases()),
        (stir_cases(), stir_pcs_claim_cases()),
        (stir_cases(), stir_pcs_opening_cases()),
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

    // Every protocol sweeps the full budget, except the two that have nothing to sweep.
    //
    // A configuration-free phase has exactly one seed to offer.
    // A fixed product would demand two duplicates of it.
    //
    // Pinning the count per protocol is what stops a sweep from quietly shrinking.
    // A shrunk sweep takes its per-knob coverage with it.
    let groups = protocols();
    assert_eq!(groups.len(), NUM_PROTOCOLS);
    for group in &groups {
        let protocol = group[0]
            .0
            .split('/')
            .next()
            .expect("a case label names its protocol");
        let expected = if CONFIGURATION_FREE_PHASES.contains(&protocol) {
            1
        } else {
            MAX_CASES_PER_PROTOCOL
        };
        assert_eq!(
            group.len(),
            expected,
            "{protocol} sweeps {} cases",
            group.len()
        );
    }
    assert_eq!(seeds.len(), groups.iter().map(Vec::len).sum::<usize>());

    assert_seeds_pairwise_distinct(&seeds);
}

/// One configuration per grinding protocol, every difficulty positive.
///
/// The sweep above picks configurations that separate seeds.
/// Most of them grind at zero bits.
///
/// A zero-bit step is elided from the pattern.
/// That sweep therefore cannot see the sites it never describes.
///
/// This one exists to make every grinding step visible at least once.
fn grinding_sweep() -> Vec<(String, DomainSeparator<Alphabet>)> {
    // STIR grinds at four sites: two per round, two in the closing phase.
    let stir = StirShape {
        commits_initial: true,
        instances: vec![StirInstanceShape {
            rounds: vec![StirRoundShape {
                folding_pow_bits: 3,
                num_ood_samples: 1,
                pow_bits: 4,
                num_queries: 3,
                log_fold_domain_size: 6,
                log_degree: 8,
                log_domain_size: 9,
                log_folding_factor: 3,
                domain_shift: 31,
                eta_bits: 0.25_f64.to_bits(),
            }],
            final_folding_pow_bits: 5,
            final_poly_len: 2,
            final_pow_bits: 6,
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
        }],
    };

    // Each WHIR pipeline grinds inside its own rounds.
    // The hiding one also grinds in its base case.
    let whir_config = WhirConfig::<EF, F, Ch>::new(WHIR_NUM_VARIABLES, whir_params())
        .expect("the fixture parameters are valid");
    let zk_config = ZkWhirConfig::<EF, F, Ch>::new(
        WHIR_NUM_VARIABLES,
        whir_params(),
        zk_whir_parameters()[0].1.clone(),
    )
    .expect("the fixture parameters are valid");
    let zk_shape = ZkWhirShape::new(&zk_config);

    vec![
        (String::from("p3-stir"), stir.domain_separator::<F, EF>()),
        (
            String::from("p3-whir"),
            WhirShape::new(&whir_config, WHIR_NUM_OPENING_CLAIMS).domain_separator::<F, EF>(),
        ),
        (
            String::from("p3-whir-hvzk"),
            zk_shape.domain_separator::<F, EF>(),
        ),
        (
            String::from("p3-whir-hvzk-base"),
            zk_shape.base_case.domain_separator::<F, EF>(),
        ),
        (
            String::from("p3-sumcheck-quadratic"),
            SumcheckShape::new(4, 2, Basis::Evaluation).domain_separator::<F, EF>(),
        ),
        (
            String::from("p3-sumcheck-hvzk"),
            ZkSumcheckShape::new_batching(3, 4, 2).domain_separator::<F, EF>(),
        ),
        (
            String::from("p3-sumcheck-generic-degree"),
            GenericDegreeShape::new(4, 3, 2).domain_separator::<F, EF>(),
        ),
    ]
}

#[test]
fn every_grinding_site_is_either_budgeted_or_priced_elsewhere() {
    // Invariant: a proof-of-work step is a soundness parameter in two places.
    //
    //     transcript  ->  the difficulty the pattern describes
    //     model       ->  the difficulty the security report credits
    //
    // A site in neither vocabulary is a difficulty nobody compares.
    //
    // That is how a grinding budget and a transcript drift apart unnoticed.
    //
    // Both vocabularies live in `p3-security`.
    // This walk compares the described steps against them, not against a local list.
    for group in protocols() {
        for (name, separator) in group {
            // Case labels are "protocol/configuration", and the name leads.
            let protocol = name
                .split('/')
                .next()
                .expect("a case label names its protocol");

            // A step may be described at zero difficulty.
            //
            // That is one of the zero-bit conventions, not an anomaly.
            //
            // So the difficulty itself is not read here.
            for (label, _bits) in pow_difficulties(separator.pattern()) {
                let budgeted = grinding_step(protocol, label).is_some();
                let priced_elsewhere = is_unpriced_grinding_site(protocol, label);

                assert!(
                    budgeted || priced_elsewhere,
                    "{protocol}/{label} grinds, but no vocabulary classifies it",
                );
                assert!(
                    !(budgeted && priced_elsewhere),
                    "{protocol}/{label} is both compared against the model and priced elsewhere",
                );
            }
        }
    }
}

#[test]
fn every_unpriced_grinding_site_is_described_by_the_protocol_that_owns_it() {
    // Invariant: the unpriced table names real steps.
    //
    // A stale row would exempt a site that no longer exists.
    // It would also hide a renamed one behind a classification that can never fire.
    //
    // Fixture state: every protocol below is swept at a positive difficulty.
    // Each of its grinding steps therefore reaches a pattern.
    let described: Vec<(String, String)> = grinding_sweep()
        .into_iter()
        .flat_map(|(protocol, separator)| {
            pow_difficulties(separator.pattern())
                .into_iter()
                .map(move |(label, _)| (protocol.clone(), String::from(label)))
        })
        .collect();

    for &(protocol, label) in &UNPRICED_GRINDING_SITES {
        // One protocol seeds over a binary tower field.
        //
        // Its separator has a different sponge alphabet, so it cannot join the sweep above.
        //
        // Its own crate runs both directions of this check instead.
        if protocol == "p3-binary-pcs" {
            continue;
        }

        assert!(
            described.contains(&(String::from(protocol), String::from(label))),
            "{protocol}/{label} is listed as priced elsewhere, but no pattern describes it",
        );
    }
}
