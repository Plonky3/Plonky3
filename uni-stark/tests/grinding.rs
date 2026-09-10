//! End-to-end coverage for the two grinding sites outside the low-degree test:
//! the PCS opening-batching challenge (`FriParameters::batch_proof_of_work_bits`)
//! and the DEEP out-of-domain point (`StarkGenericConfig::ood_proof_of_work_bits`).
//!
//! Both are proof-of-work phases whose only job is to add bits to a *proven*
//! soundness bound, so what has to be true of them is narrow and testable: a
//! proof produced with the grind verifies, a witness that does not satisfy the
//! predicate is rejected, and the bits reach the soundness report. The third is
//! the one worth stating explicitly — a grind the prover pays but the analysis
//! never credits is wasted work, and bits the analysis credits but no verifier
//! checks are worse than wasted.
//!
//! The last test in this file makes that third statement mechanical.
//!
//! It walks the three transcripts a uni-STARK proof is built from.
//!
//! Then it compares every recorded difficulty against the credited bits, site by site.

use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::DuplexChallenger;
use p3_challenger::testing::pow_difficulties;
use p3_commit::ExtensionMmcs;
use p3_dft::Radix2DitParallel;
use p3_field::extension::BinomialExtensionField;
use p3_field::{Field, PrimeCharacteristicRing, PrimeField64};
use p3_fri::{FriParameters, FriShape, PcsShape, TwoAdicFriPcs};
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_security::grinding::{GrindingBudget, GrindingSites, RecordedGrind};
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use p3_uni_stark::{
    InvalidProofShapeError, ProvenSecurity, StarkConfig, StarkGenericConfig, StarkSecurityParams,
    StarkShape, VerificationError, prove, verify,
};
use rand::SeedableRng;
use rand::rngs::SmallRng;

/// A minimal single-row AIR: enforces `a * a == b` per row.
struct SquareAir;

impl<F> BaseAir<F> for SquareAir {
    fn width(&self) -> usize {
        2
    }

    fn main_next_row_columns(&self) -> Vec<usize> {
        vec![]
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

fn generate_square_trace<F: PrimeField64>(n: usize) -> RowMajorMatrix<F> {
    assert!(n.is_power_of_two());
    let mut values = F::zero_vec(n * 2);
    for i in 0..n {
        let a = F::from_u64((i + 1) as u64);
        values[i * 2] = a;
        values[i * 2 + 1] = a * a;
    }
    RowMajorMatrix::new(values, 2)
}

type Val = BabyBear;
type Perm = Poseidon2BabyBear<16>;
type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
type ValMmcs =
    MerkleTreeMmcs<<Val as Field>::Packing, <Val as Field>::Packing, MyHash, MyCompress, 2, 8>;
type Challenge = BinomialExtensionField<Val, 4>;
type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
type Challenger = DuplexChallenger<Val, Perm, 16, 8>;
type Dft = Radix2DitParallel<Val>;
type Pcs = TwoAdicFriPcs<Val, Dft, ValMmcs, ChallengeMmcs>;
type MyConfig = StarkConfig<Pcs, Challenge, Challenger>;

/// Difficulties small enough to grind instantly, large enough that a wrong
/// witness is rejected with overwhelming probability.
const BATCH_POW_BITS: usize = 8;
const OOD_POW_BITS: usize = 8;

/// Generic in the commitment scheme so the same parameters can drive a proof and,
/// with `M = ()`, the transcript shapes the last test derives.
const fn fri_params<M>(batch_proof_of_work_bits: usize, challenge_mmcs: M) -> FriParameters<M> {
    FriParameters {
        log_blowup: 2,
        log_final_poly_len: 0,
        max_log_arity: 1,
        num_queries: 2,
        batch_proof_of_work_bits,
        commit_proof_of_work_bits: 1,
        query_proof_of_work_bits: 1,
        mmcs: challenge_mmcs,
    }
}

fn make_config(batch_pow_bits: usize, ood_pow_bits: usize) -> MyConfig {
    let mut rng = SmallRng::seed_from_u64(42);
    let perm = Perm::new_from_rng_128(&mut rng);
    let hash = MyHash::new(perm.clone());
    let compress = MyCompress::new(perm.clone());
    let val_mmcs = ValMmcs::new(hash, compress, 0);
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
    let pcs = Pcs::new(
        Dft::default(),
        val_mmcs,
        fri_params(batch_pow_bits, challenge_mmcs),
    );
    MyConfig::new(pcs, Challenger::new(perm)).with_ood_proof_of_work_bits(ood_pow_bits)
}

/// The default config grinds at neither new site, so an existing caller keeps
/// exactly the protocol it had.
#[test]
fn ungrounded_config_is_the_default() {
    let config = make_config(0, 0);
    assert_eq!(config.ood_proof_of_work_bits(), 0);

    let trace = generate_square_trace::<Val>(1 << 3);
    let proof = prove(&config, &SquareAir, trace, &[]);
    verify(&config, &SquareAir, &proof, &[]).expect("ungrounded proof verifies");
}

/// A proof produced with both grinds verifies against the same config.
#[test]
fn grinding_at_both_sites_round_trips() {
    let config = make_config(BATCH_POW_BITS, OOD_POW_BITS);
    let trace = generate_square_trace::<Val>(1 << 3);
    let proof = prove(&config, &SquareAir, trace, &[]);
    verify(&config, &SquareAir, &proof, &[]).expect("ground proof verifies");
}

/// A tampered out-of-domain witness is rejected, and rejected *as* a bad
/// witness rather than as a downstream constraint failure: the check runs
/// before `zeta` is sampled, so nothing past it is even reached.
#[test]
fn tampered_ood_pow_witness_is_rejected() {
    let config = make_config(0, OOD_POW_BITS);
    let trace = generate_square_trace::<Val>(1 << 3);
    let mut proof = prove(&config, &SquareAir, trace, &[]);
    proof.ood_pow_witness += Val::ONE;

    match verify(&config, &SquareAir, &proof, &[]) {
        Err(VerificationError::InvalidOodPowWitness) => {}
        other => panic!("expected InvalidOodPowWitness, got {other:?}"),
    }
}

#[test]
fn a_noncanonical_ood_pow_witness_at_zero_difficulty_is_rejected() {
    // Why: `check_witness` short-circuits at zero bits, and the step is elided.
    //
    //     ood_pow_bits = 0 -> the witness never reaches the sponge -> unbound
    //     ood_pow_bits > 0 -> absorbed, bits resampled             -> the grind binds it
    //
    // Left unbound, a third party rewrites the field and keeps a verifying proof.
    let config = make_config(0, 0);
    let trace = generate_square_trace::<Val>(1 << 3);
    let mut proof = prove(&config, &SquareAir, trace, &[]);

    // The honest prover writes zero when it pays no work, so zero is canonical.
    assert_eq!(proof.ood_pow_witness, Val::ZERO);

    // Any other value is a second encoding of the same statement.
    proof.ood_pow_witness = Val::ONE;

    match verify(&config, &SquareAir, &proof, &[]) {
        Err(VerificationError::InvalidProofShape(
            InvalidProofShapeError::NonCanonicalOodPowWitness,
        )) => {}
        other => panic!("expected NonCanonicalOodPowWitness, got {other:?}"),
    }
}

/// A verifier demanding more out-of-domain grinding than the prover paid
/// rejects. This is the failure mode of a prover/verifier config mismatch, and
/// it must be a rejection rather than a silently weaker proof.
#[test]
fn ood_pow_difficulty_mismatch_is_rejected() {
    let prover_config = make_config(0, 0);
    let trace = generate_square_trace::<Val>(1 << 3);
    let proof = prove(&prover_config, &SquareAir, trace, &[]);

    let verifier_config = make_config(0, 24);
    match verify(&verifier_config, &SquareAir, &proof, &[]) {
        Err(VerificationError::InvalidOodPowWitness) => {}
        other => panic!("expected InvalidOodPowWitness, got {other:?}"),
    }
}

/// The same for the batch site, which lives in the PCS: a verifier demanding
/// more than the prover paid rejects the opening argument.
#[test]
fn batch_pow_difficulty_mismatch_is_rejected() {
    let prover_config = make_config(0, 0);
    let trace = generate_square_trace::<Val>(1 << 3);
    let proof = prove(&prover_config, &SquareAir, trace, &[]);

    let verifier_config = make_config(24, 0);
    match verify(&verifier_config, &SquareAir, &proof, &[]) {
        Err(VerificationError::InvalidOpeningArgument(_)) => {}
        other => panic!("expected InvalidOpeningArgument, got {other:?}"),
    }
}

/// Reported level is a minimum over rounds, so a site only moves it while the
/// round it protects is the binding one. Each test below therefore picks a
/// shape where its round binds; `ground` grinds only at the named site.
fn level(params: &StarkSecurityParams, degree_bits: usize, sites: GrindingSites) -> usize {
    ProvenSecurity::compute_from_proof(degree_bits, &params.clone().with_grinding(sites))
        .security_bits()
}

/// Over a 128-bit field, batching 2^8 codewords puts the batch-combination
/// round below every other, and grinding before the batching challenge is then
/// the only thing that raises the reported level — no query count moves that
/// round, which is the whole reason the site exists.
#[test]
fn batch_grinding_raises_the_level_when_the_batch_round_binds() {
    let params = StarkSecurityParams::new(
        p3_security::fri::FriRegime {
            log_blowup: 3,
            num_queries: 64,
            log_final_poly_len: 0,
            max_log_arity: 3,
            commit_pow_bits: 0,
            query_pow_bits: 16,
        },
        128,
        128,
        512,
        9,
        2,
        1 << 8,
        8,
    );
    let degree_bits = 20;

    let ungrounded = level(&params, degree_bits, GrindingSites::NONE);
    let ground = level(
        &params,
        degree_bits,
        GrindingSites {
            batch_combination: 16,
            ..GrindingSites::NONE
        },
    );

    assert!(
        ground > ungrounded,
        "batch grinding bought nothing: {ungrounded} -> {ground}"
    );
}

/// With nothing batched there is no batch round at all, and enough commit- and
/// query-phase grinding to lift those rounds clear, the DEEP-ALI round is what
/// binds. Its error is `degree / |F|` — fixed by the AIR and the trace height —
/// so out-of-domain grinding is likewise the only lever on it.
#[test]
fn out_of_domain_grinding_raises_the_level_when_the_deep_round_binds() {
    let params = StarkSecurityParams::new(
        p3_security::fri::FriRegime {
            log_blowup: 3,
            num_queries: 200,
            log_final_poly_len: 0,
            max_log_arity: 1,
            commit_pow_bits: 20,
            query_pow_bits: 16,
        },
        128,
        128,
        1,
        9,
        2,
        1,
        8,
    );
    let degree_bits = 24;

    let ungrounded = level(&params, degree_bits, GrindingSites::NONE);
    let ground = level(
        &params,
        degree_bits,
        GrindingSites {
            out_of_domain: 16,
            ..GrindingSites::NONE
        },
    );

    assert!(
        ground > ungrounded,
        "out-of-domain grinding bought nothing: {ungrounded} -> {ground}"
    );
}

/// Whatever binds, grinding can only ever help — a site that lowered the
/// reported level would mean the boost had been wired to the wrong term.
#[test]
fn grinding_is_monotone_at_every_site() {
    let params = StarkSecurityParams::new(
        p3_security::fri::FriRegime {
            log_blowup: 3,
            num_queries: 64,
            log_final_poly_len: 0,
            max_log_arity: 3,
            commit_pow_bits: 0,
            query_pow_bits: 16,
        },
        128,
        128,
        512,
        9,
        2,
        1 << 8,
        8,
    );
    let degree_bits = 20;

    let mut previous = level(&params, degree_bits, GrindingSites::NONE);
    for bits in 0..=24 {
        let current = level(
            &params,
            degree_bits,
            GrindingSites {
                batch_combination: bits,
                out_of_domain: bits,
                lookup_challenge: bits,
            },
        );
        assert!(
            current >= previous,
            "grinding {bits} bits at every site lowered the level: {previous} -> {current}"
        );
        previous = current;
    }
}

/// The three protocols a uni-STARK proof over `p3-fri` layers, outermost first.
///
/// Each brackets the next, and each seeds its own transcript, so the same step
/// label in two of them is two different steps.
const UNI_STARK_STACK: [&str; 3] = ["p3-uni-stark", "p3-fri-pcs", "p3-fri"];

/// The grinding this parameter set credits.
///
/// Out-of-domain bits come from the config, the other four from the FRI parameters.
const fn credited(fri: &FriParameters<()>, ood_pow_bits: usize) -> GrindingBudget {
    GrindingBudget::from_sites(&GrindingSites {
        out_of_domain: ood_pow_bits,
        ..fri.grinding_sites()
    })
    .with_fri(&fri.security_regime())
}

/// Every grinding difficulty the three transcripts of one configuration demand.
///
/// Fixture state: one opening of width three, and a two-round folding schedule.
///
/// Nothing but the difficulties reaches the result, so those choices do not move it.
fn demanded(fri: &FriParameters<()>, ood_pow_bits: usize) -> Vec<RecordedGrind> {
    let stark = StarkShape {
        log_ext_degree: 5,
        log_degree: 5,
        main_width: 2,
        preprocessed_width: 0,
        num_public_values: 0,
        num_periodic_columns: 0,
        num_quotient_chunks: 2,
        opens_main_next_row: false,
        opens_preprocessed_next_row: false,
        has_randomization: false,
        ood_pow_bits,
    };
    let opening = PcsShape {
        claimed_evaluation_counts: vec![vec![vec![3]]],
        batch_pow_bits: fri.batch_proof_of_work_bits,
    };
    let ldt = FriShape::with_schedule(fri, vec![2, 2], 8);

    let patterns = [
        stark.pattern::<Val, Challenge>(),
        opening.pattern::<Val, Challenge>(),
        ldt.pattern::<Val, Challenge>(),
    ];

    UNI_STARK_STACK
        .iter()
        .zip(&patterns)
        .flat_map(|(protocol, pattern)| {
            pow_difficulties(pattern)
                .into_iter()
                .map(|(label, bits)| RecordedGrind::new(protocol, label, bits))
        })
        .collect()
}

#[test]
fn recorded_grinding_matches_what_the_security_model_credits() {
    // Property: recorded difficulty == credited difficulty, at every site.
    //
    // A parameter set that fails it reports bits nobody pays for, or pays for work
    // the report throws away.
    //
    // Sweep: every site at zero and at one positive difficulty.
    //
    // Both sides of the elided-at-zero convention are therefore exercised.
    for batch_pow_bits in [0, 10] {
        for ood_pow_bits in [0, 8] {
            for commit_pow_bits in [0, 4] {
                for query_pow_bits in [0, 16] {
                    let fri = FriParameters {
                        commit_proof_of_work_bits: commit_pow_bits,
                        query_proof_of_work_bits: query_pow_bits,
                        ..fri_params(batch_pow_bits, ())
                    };

                    credited(&fri, ood_pow_bits)
                        .check(&UNI_STARK_STACK, &demanded(&fri, ood_pow_bits))
                        .unwrap_or_else(|mismatch| {
                            panic!(
                                "batch={batch_pow_bits} ood={ood_pow_bits} \
                                 commit={commit_pow_bits} query={query_pow_bits}: {mismatch}"
                            )
                        });
                }
            }
        }
    }
}

#[test]
fn the_configuration_these_tests_prove_with_is_accounted_for() {
    // The config the other tests prove with is itself consistent.
    //
    // The numbers handed to the prover are the numbers the security report is built from.
    let fri = fri_params(BATCH_POW_BITS, ());
    let config = make_config(BATCH_POW_BITS, OOD_POW_BITS);
    let ood_pow_bits = config.ood_proof_of_work_bits();

    assert_eq!(ood_pow_bits, OOD_POW_BITS);
    credited(&fri, ood_pow_bits)
        .check(&UNI_STARK_STACK, &demanded(&fri, ood_pow_bits))
        .unwrap_or_else(|mismatch| panic!("{mismatch}"));
}
