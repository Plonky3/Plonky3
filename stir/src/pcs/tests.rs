use alloc::{format, vec};

use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::testing::seed_digest;
use p3_challenger::{CanSampleBits, DuplexChallenger, HashChallenger};
use p3_commit::ExtensionMmcs;
use p3_dft::Radix2DitParallel;
use p3_field::PrimeCharacteristicRing;
use p3_field::extension::BinomialExtensionField;
use p3_keccak::Keccak256Hash;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_security::whir::SecurityAssumption;
use p3_symmetric::{Hash, PaddingFreeSponge, TruncatedPermutation};
use proptest::prelude::*;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

use super::*;
use crate::prover::codeword_from_coeffs;
use crate::verifier::verify_stir_multi;

type EF = BinomialExtensionField<BabyBear, 4>;

#[test]
fn combine_coefficients_matches_definition_4_11() {
    // Definition 4.11 fixes `r_1 = 1` and `r_i = r^{(i-1) + Σ_{j<i}(d* - d_j)}`. Pinning
    // the exponents to that closed form rather than to the running-sum recurrence itself
    // is what keeps the two from drifting together: a wrong `r_i` still produces a
    // low-degree combined codeword, so prover and verifier would agree on a
    // miscomputation and STIR would accept it.
    //
    // The exponents are what make class `i` occupy the consecutive power block
    // `[e_i, e_i + gap_i]`, with the blocks tiling `[0, ell - 1]` without overlap. For
    // `d_i = [8, 4, 2]` and `d* = 8` the gaps are `[0, 4, 6]`, so the blocks are
    // `[0,0] | [1,5] | [6,12]` — exponents `[0, 1, 6]` and `ell = 13`.
    let r_comb = EF::from_u64(3);
    let coeffs = combine_coefficients(r_comb, 3, [3usize, 2, 1].into_iter());

    assert_eq!(
        coeffs,
        vec![(EF::ONE, 0), (r_comb.exp_u64(1), 4), (r_comb.exp_u64(6), 6),]
    );

    // The blocks tile exactly `Σᵢ (gapᵢ + 1)`, which is the `ell` the config's Combine
    // soundness accounting is charged at.
    let ell: usize = coeffs.iter().map(|&(_, gap)| gap + 1).sum();
    assert_eq!(ell, 13);
}

// A digest that the serializing challenger can absorb byte by byte.
fn digest(byte: u8) -> Hash<BabyBear, u8, 32> {
    Hash::from([byte; 32])
}

// The serializing backend, over the byte hasher the Keccak-Merkle configurations use.
fn byte_challenger() -> SerializingChallenger32<BabyBear, HashChallenger<u8, Keccak256Hash, 32>> {
    SerializingChallenger32::from_hasher(Vec::new(), Keccak256Hash {})
}

#[test]
fn the_group_count_separates_one_commitment_from_two() {
    // Invariant: the group count is absorbed as a length prefix, so a commitment cannot
    // be split or merged without changing the transcript.
    //
    //     one commitment : len=2 | root_a | root_b
    //     two commitments: len=1 | root_a | len=1 | root_b
    //
    // Without the prefix both flatten to `root_a | root_b` and sample the same challenge.
    let (root_a, root_b) = (digest(0xAA), digest(0xBB));

    let mut merged = byte_challenger();
    merged.observe(StirCommitment(vec![root_a, root_b]));

    let mut split = byte_challenger();
    split.observe(StirCommitment(vec![root_a]));
    split.observe(StirCommitment(vec![root_b]));

    assert_ne!(
        merged.sample_bits(24),
        split.sample_bits(24),
        "the length prefix must separate these two absorptions"
    );
}

#[test]
fn the_root_order_is_part_of_the_transcript() {
    // Two commitments over the same roots in opposite order must not collide: the roots
    // are absorbed in sequence, never as an order-insensitive set.
    let (root_a, root_b) = (digest(0xAA), digest(0xBB));

    let mut forward = byte_challenger();
    forward.observe(StirCommitment(vec![root_a, root_b]));

    let mut reversed = byte_challenger();
    reversed.observe(StirCommitment(vec![root_b, root_a]));

    assert_ne!(forward.sample_bits(24), reversed.sample_bits(24));
}

#[test]
fn every_backend_absorbs_a_commitment_through_the_shared_phase() {
    // Invariant: a backend's impl routes through the shared typed phase.
    //
    // It never writes out a hand-written variant of the sequence.
    //
    //     observe(commitment)  ==  the commitment phase over the same roots
    //
    // This is checked per backend.
    //
    // Prover and verifier share one challenger type.
    //
    // A drifting backend would make both of them wrong together.
    //
    // Fixture state: three digests absorbed as one commitment of three roots.
    let roots = vec![digest(0x01), digest(0x02), digest(0x03)];

    let mut via_commitment = byte_challenger();
    via_commitment.observe(StirCommitment(roots.clone()));

    let mut via_phase = byte_challenger();
    observe_commitment::<_, BabyBear, _>(&mut via_phase, roots);

    assert_eq!(via_commitment.sample_bits(24), via_phase.sample_bits(24));

    // The same obligation, for the duplex backend the Poseidon2 configurations use.
    let perm = TestPerm::new_from_rng_128(&mut SmallRng::seed_from_u64(1));
    let duplex_roots = vec![BabyBear::ONE, BabyBear::TWO];

    let mut duplex_via_commitment = TestChallenger::new(perm.clone());
    duplex_via_commitment.observe(StirCommitment(duplex_roots.clone()));

    let mut duplex_via_phase = TestChallenger::new(perm);
    observe_commitment::<_, BabyBear, _>(&mut duplex_via_phase, duplex_roots);

    assert_eq!(
        duplex_via_commitment.sample_bits(24),
        duplex_via_phase.sample_bits(24)
    );
}

#[test]
#[should_panic(expected = "classes must be sorted descending")]
fn combine_coefficients_rejects_a_class_above_d_star() {
    // `d_i > d*` would wrap the `d* - d_i` subtraction in release and yield a wrong
    // codeword rather than a failure, so the precondition is checked rather than assumed.
    let _ = combine_coefficients(EF::from_u64(3), 3, [3usize, 4].into_iter());
}

type TestVal = BabyBear;
type TestPerm = Poseidon2BabyBear<16>;
type TestHash = PaddingFreeSponge<TestPerm, 16, 8, 8>;
type TestCompress = TruncatedPermutation<TestPerm, 2, 8, 16>;
type TestPacked = <TestVal as Field>::Packing;
type TestValMmcs = MerkleTreeMmcs<TestPacked, TestPacked, TestHash, TestCompress, 2, 8>;
type TestStirMmcs = ExtensionMmcs<TestVal, EF, TestValMmcs>;
type TestChallenger = DuplexChallenger<TestVal, TestPerm, 16, 8>;
type TestPcs = TwoAdicStirPcs<
    TestVal,
    Radix2DitParallel<TestVal>,
    TestValMmcs,
    TestStirMmcs,
    EF,
    TestChallenger,
>;
type TestConfig = StirConfig<TestVal, EF, TestStirMmcs, TestChallenger>;

impl TestPcs {
    fn get_or_compute_stir_config(
        &self,
        log_degree: usize,
        combine: Option<(usize, u64)>,
    ) -> Arc<TestConfig> {
        self.get_or_try_compute_stir_config(log_degree, combine)
            .unwrap()
    }
}

/// Every value the schedule derives, in one comparable string. `StirConfig` has no
/// `PartialEq`, and only the derived schedule matters here — the `mmcs` field is cloned
/// straight from the shared parameters.
fn schedule_fingerprint(config: &TestConfig) -> alloc::string::String {
    format!(
        "{:?}|{}|{}|{}|{}|{}|{}|{}|{}|{:?}|{}|{}|{:?}",
        config.soundness_type,
        config.log_starting_degree,
        config.security_level,
        config.max_pow_bits,
        config.log_blowup,
        config.log_folding_factor,
        config.log_starting_folding_factor,
        config.log_final_degree,
        config.final_queries,
        config.final_eta,
        config.final_pow_bits,
        config.final_folding_pow_bits,
        config.round_configs,
    )
}

/// Add a random codeword of a bucket's own degree bound to its reduced opening.
///
/// # Overview
///
/// The sum is still a codeword of that degree bound.
///
/// So every proximity-test round accepts it.
///
/// Only the lane checks stand between such a prover and acceptance.
///
/// # Arguments
///
/// - `pcs`: the instance whose transform evaluates the perturbation.
/// - `reduced_openings`: the class map a prepared opening handed back.
/// - `log_native_h`: log of the bucket's only native height.
/// - `log_lde`: log of the shared domain the bucket runs on.
/// - `rng`: source of the perturbing polynomial's coefficients.
///
/// # Panics
///
/// When the map holds no class at that pair of heights.
fn add_low_degree_codeword(
    pcs: &TestPcs,
    reduced_openings: &mut alloc::collections::BTreeMap<(usize, usize), Vec<EF>>,
    log_native_h: usize,
    log_lde: usize,
    rng: &mut SmallRng,
) {
    // A random polynomial of the bucket's degree bound, on the bucket's shared coset.
    let mut coeffs: Vec<EF> = (0..1usize << log_native_h).map(|_| rng.random()).collect();
    coeffs.resize(1 << log_lde, EF::ZERO);
    let mut low_degree = codeword_from_coeffs(&pcs.dft, coeffs, TestVal::GENERATOR, log_lde);

    // Reduced openings are stored bit-reversed, and un-reversed on the way in.
    //
    // So the perturbation is reversed here, to land in natural order there.
    reverse_slice_index_bits(&mut low_degree);

    let ro = reduced_openings
        .get_mut(&(log_lde, log_native_h))
        .expect("the bucket's only native-height class");
    for (value, extra) in ro.iter_mut().zip(low_degree) {
        *value += extra;
    }
}

/// A PCS over `TestVal`/`EF` at the given layout and soundness knobs.
fn test_pcs_with(
    max_log_height_spread: usize,
    soundness_type: SecurityAssumption,
    security_level: usize,
) -> TestPcs {
    let mut rng = rand::rngs::SmallRng::seed_from_u64(11);
    let perm = TestPerm::new_from_rng_128(&mut rng);
    let val_mmcs = TestValMmcs::new(TestHash::new(perm.clone()), TestCompress::new(perm), 0);
    let stir = StirParameters {
        log_blowup: 1,
        log_folding_factor: 2,
        log_starting_folding_factor: 2,
        soundness_type,
        security_level,
        max_pow_bits: 0,
        mmcs: TestStirMmcs::new(val_mmcs.clone()),
    };
    TwoAdicStirPcs::new(Radix2DitParallel::default(), val_mmcs, stir)
        .with_max_log_height_spread(max_log_height_spread)
}

#[test]
fn batch_grinding_is_enforced_across_pcs_layouts() {
    for spread in [0, DEFAULT_MAX_LOG_HEIGHT_SPREAD] {
        let pcs = test_pcs_with(spread, SecurityAssumption::CapacityBound, 32)
            .with_batch_proof_of_work_bits(8);
        let mut rng = rand::rngs::SmallRng::seed_from_u64(42);
        let mut base = TestChallenger::new(TestPerm::new_from_rng_128(&mut rng));
        let domains = [6, 4].map(|log_h| pcs.natural_domain_for_degree(1 << log_h));
        let matrices = domains.map(|domain| {
            (
                domain,
                RowMajorMatrix::<TestVal>::rand(&mut rng, domain.size(), 3),
            )
        });
        let (commitment, data) = pcs.commit(matrices).unwrap();
        base.observe(commitment.clone());
        let point: EF = base.sample_algebra_element();
        let mut prover = base.clone();
        let (values, proof) = pcs
            .open(
                vec![OpeningRequest {
                    prover_data: &data,
                    points: vec![vec![point]; 2],
                }],
                &mut prover,
            )
            .unwrap();
        // Replay just the claim phase and the batching site.
        //
        // Neither reads anything the opening path settled.
        //
        // Moving the prover's grind before a claim, or after alpha, breaks this.
        let mut batch_replay = base.clone();
        observe_opened_values::<TestChallenger, TestVal, EF>(&mut batch_replay, &values);
        let _: EF = crate::batch_transcript::verify::<TestVal, EF, _, (), ()>(
            &mut batch_replay,
            8,
            proof.batch_pow_witness,
        )
        .expect("the witness must bind all claims before alpha is sampled");
        let claims: Vec<_> = vec![
            (
                commitment,
                domains
                    .into_iter()
                    .enumerate()
                    .map(|(i, domain)| (domain, vec![(point, values[0][i][0].clone())]))
                    .collect(),
            )
                .into(),
        ];
        let mut verifier = base.clone();
        pcs.verify(claims.clone(), &proof, &mut verifier).unwrap();
        assert_eq!(
            prover.sample_algebra_element::<EF>(),
            verifier.sample_algebra_element::<EF>(),
        );
        assert!(proof.batch_pow_witness.is_some());
        assert_eq!(proof.buckets.len(), if spread == 0 { 2 } else { 1 });
        let bytes = postcard::to_allocvec(&proof).unwrap();
        let decoded = postcard::from_bytes(&bytes).unwrap();
        pcs.verify(claims.clone(), &decoded, &mut base.clone())
            .unwrap();

        let mut missing = proof.clone();
        missing.batch_pow_witness = None;
        assert!(matches!(
            pcs.verify(claims.clone(), &missing, &mut base.clone()),
            Err(StirError::InvalidProofShape(
                ProofShapeError::BatchPowWitness { .. }
            ))
        ));

        // A verifier that skips the new PoW check may still reject at a later Merkle
        // check. Require the dedicated error to catch that missing enforcement.
        let mut invalid = proof.clone();
        let rejects_pow = (0..256).any(|candidate| {
            invalid.batch_pow_witness = Some(TestVal::from_u64(candidate));
            matches!(
                pcs.verify(claims.clone(), &invalid, &mut base.clone()),
                Err(StirError::InvalidBatchPowWitness { bits: 8 })
            )
        });
        assert!(
            rejects_pow,
            "an invalid batching witness must be rejected at its site"
        );

        let unground = pcs.clone().with_batch_proof_of_work_bits(0);
        assert_eq!(pcs.grinding_sites().batch_combination, 8);
        assert_eq!(unground.grinding_sites(), p3_security::GrindingSites::NONE);
        // A probe with no alpha or Combine error has no batch work to credit.
        assert_eq!(
            schedule_fingerprint(&pcs.get_or_compute_stir_config(6, None)),
            schedule_fingerprint(&unground.get_or_compute_stir_config(6, None)),
        );
        assert!(matches!(
            unground.verify(claims.clone(), &proof, &mut base.clone()),
            Err(StirError::InvalidProofShape(
                ProofShapeError::BatchPowWitness { .. }
            ))
        ));
        assert!(
            pcs.clone()
                .with_batch_proof_of_work_bits(7)
                .verify(claims, &proof, &mut base.clone())
                .is_err()
        );
    }
}

/// Group sizes of the plan for `log_native_heights`, in descending LDE height.
fn group_sizes(plan: &GroupPlan) -> Vec<usize> {
    (0..plan.log_lde_heights.len())
        .map(|g| plan.group_of_matrix.iter().filter(|&&x| x == g).count())
        .collect()
}

#[test]
fn groups_split_at_the_configured_spread() {
    let heights = [8usize, 6, 4];
    let cb = SecurityAssumption::CapacityBound;

    // Spread 2 rules out one group over all three, so two is the fewest available. Of the
    // two-group splits it does admit, `{8} | {6, 4}` keeps the 2^6 matrix off the 2^9
    // domain and is the cheaper one to extend.
    let plan = test_pcs_with(2, cb, 32).plan_groups(&heights);
    assert_eq!(plan.log_lde_heights, vec![9, 7]);
    assert_eq!(plan.group_of_matrix, vec![0, 1, 1]);

    // Spread 1 admits none of these pairs.
    let plan = test_pcs_with(1, cb, 32).plan_groups(&heights);
    assert_eq!(plan.log_lde_heights, vec![9, 7, 5]);
    assert_eq!(plan.group_of_matrix, vec![0, 1, 2]);
}

#[test]
fn a_group_spans_the_cheapest_admissible_run_not_the_widest() {
    // `[20, 17, 16, 15]` needs two groups either way — the full span is five octaves, past
    // the cap — but which two matters. Filling the tall group first puts the 2^17 matrix
    // on the 2^21 domain: a 16x blowup, and a `Combine` over two classes in the tall
    // bucket. Leaving it with the short heights costs it 2x instead, at the same group
    // count and with no `Combine` in the tall bucket at all.
    let pcs = test_pcs_with(
        DEFAULT_MAX_LOG_HEIGHT_SPREAD,
        SecurityAssumption::CapacityBound,
        16,
    );

    let plan = pcs.plan_groups(&[20, 17, 16, 15]);
    assert_eq!(plan.log_lde_heights, vec![21, 18]);
    assert_eq!(plan.group_of_matrix, vec![0, 1, 1, 1]);

    // Shapes lying wholly inside one band, or wholly outside it, have nothing to
    // redistribute: the widest run is also the cheapest once the group count is fixed.
    assert_eq!(pcs.plan_groups(&[20, 18, 17]).log_lde_heights, vec![21]);
    assert_eq!(
        pcs.plan_groups(&[20, 12, 12, 12, 12]).log_lde_heights,
        vec![21, 13]
    );
    assert_eq!(pcs.plan_groups(&[20, 10]).log_lde_heights, vec![21, 11]);
}

#[test]
fn zero_spread_is_the_per_height_class_layout() {
    // Every distinct native height on its own domain: no `Combine`, one STIR instance
    // each, and every matrix extended only by `log_blowup`.
    let heights = [8usize, 7, 6, 6];
    let plan = test_pcs_with(0, SecurityAssumption::CapacityBound, 32).plan_groups(&heights);

    assert_eq!(plan.log_lde_heights, vec![9, 8, 7]);
    assert_eq!(plan.group_of_matrix, vec![0, 1, 2, 2]);
    assert_eq!(group_sizes(&plan), vec![1, 1, 2]);
}

#[test]
fn spread_above_the_committed_range_is_a_single_shared_domain() {
    let heights = [8usize, 6, 4];
    let plan = test_pcs_with(64, SecurityAssumption::CapacityBound, 32).plan_groups(&heights);

    assert_eq!(plan.log_lde_heights, vec![9]);
    assert_eq!(plan.group_of_matrix, vec![0, 0, 0]);
}

#[test]
fn repeated_heights_share_one_class_and_one_group() {
    // Grouping is over *distinct* heights, so duplicates never open a new group and the
    // plan does not depend on how many matrices carry a given height.
    let cb = SecurityAssumption::CapacityBound;
    let pcs = test_pcs_with(2, cb, 32);

    let plan = pcs.plan_groups(&[8, 8, 6, 8]);
    assert_eq!(plan.log_lde_heights, vec![9]);
    assert_eq!(plan.group_of_matrix, vec![0, 0, 0, 0]);

    // Caller order does not matter either: a matrix follows its height.
    let plan = pcs.plan_groups(&[4, 8, 4, 6]);
    assert_eq!(plan.log_lde_heights, vec![9, 7]);
    assert_eq!(plan.group_of_matrix, vec![1, 0, 1, 1]);
}

#[test]
fn groups_shrink_when_combine_does_not_fit() {
    // JohnsonBound at 80 bits over a 124-bit challenge field: each height configures on
    // its own, but merging the two does not. The spread cap would allow one group, so
    // this is feasibility alone deciding — an infeasible parameter set degrades into more
    // STIR instances instead of failing.
    let jb = SecurityAssumption::JohnsonBound;
    let heights = [12usize, 11];

    let pcs = test_pcs_with(8, jb, 80);
    let ell = 2 * ((1u64 << 12) + 1) - ((1u64 << 12) + (1u64 << 11));
    assert!(TestConfig::try_new(12, pcs.stir.clone()).is_ok());
    assert!(TestConfig::try_new_with_combine(12, pcs.stir.clone(), 2, ell).is_err());

    let plan = pcs.plan_groups(&heights);
    assert_eq!(plan.log_lde_heights, vec![13, 12]);
    assert_eq!(plan.group_of_matrix, vec![0, 1]);

    // The same shape at a target the merge does fit stays in one group, so the split
    // above is not just the spread cap in disguise.
    let plan = test_pcs_with(8, jb, 32).plan_groups(&heights);
    assert_eq!(plan.log_lde_heights, vec![13]);
}

#[test]
fn batch_grinding_reduces_combine_queries_after_warming_a_clone() {
    let original = test_pcs_with(2, SecurityAssumption::CapacityBound, 72);
    let combine = TestPcs::combine_key(&[20, 19]);
    let before = original.get_or_compute_stir_config(20, combine);
    let grinded = original.clone().with_batch_proof_of_work_bits(8);
    let after = grinded.get_or_compute_stir_config(20, combine);
    assert!(after.round_configs[0].eta < before.round_configs[0].eta);
    assert!(after.round_configs[0].num_queries < before.round_configs[0].num_queries);
    assert_eq!(
        original
            .get_or_compute_stir_config(20, combine)
            .round_configs[0]
            .num_queries,
        before.round_configs[0].num_queries,
    );
}

#[test]
fn batch_grinding_makes_a_previously_split_group_feasible() {
    let mut original = test_pcs_with(2, SecurityAssumption::CapacityBound, 80);
    // Keep the later STIR rounds feasible independently of the PCS block.
    original.stir.max_pow_bits = 8;
    assert_eq!(original.plan_groups(&[20, 19]).log_lde_heights, [21, 20]);
    let grinded = original.with_batch_proof_of_work_bits(8);
    assert_eq!(grinded.plan_groups(&[20, 19]).log_lde_heights, [21]);
}

#[test]
fn opening_budgets_do_not_alias_probes_or_other_alpha_counts() {
    let pcs = test_pcs_with(2, SecurityAssumption::CapacityBound, 72);
    let probe = pcs.get_or_compute_stir_config(20, TestPcs::combine_key(&[20, 19]));
    let narrow = pcs
        .get_or_try_compute_pcs_config(20, &[(20, 1), (19, 1)])
        .unwrap();
    let wide = pcs
        .get_or_try_compute_pcs_config(20, &[(20, 1 << 20), (19, 1)])
        .unwrap();
    assert!(!Arc::ptr_eq(&probe, &narrow));
    assert!(wide.round_configs[0].eta > narrow.round_configs[0].eta);
    assert!(wide.round_configs[0].num_queries > narrow.round_configs[0].num_queries);
    assert!(Arc::ptr_eq(
        &wide,
        &pcs.get_or_try_compute_pcs_config(20, &[(20, 1 << 20), (19, 1)])
            .unwrap()
    ));
    assert!(
        pcs.get_or_try_compute_pcs_config(20, &[(20, 1 << 30), (19, 1)])
            .is_err()
    );
    // An opening can be infeasible although commitment-time grouping succeeded.
    assert_eq!(pcs.plan_groups(&[20, 19]).log_lde_heights, [21]);
}

#[test]
fn pooled_commitments_count_every_column_at_every_point() {
    let pcs = test_pcs_with(0, SecurityAssumption::CapacityBound, 32);
    let mut rng = SmallRng::seed_from_u64(6);
    let domain = pcs.natural_domain_for_degree(64);
    let (a, ad) = pcs
        .commit([(domain, RowMajorMatrix::<TestVal>::rand(&mut rng, 64, 3))])
        .unwrap();
    let (b, bd) = pcs
        .commit([(domain, RowMajorMatrix::<TestVal>::rand(&mut rng, 64, 5))])
        .unwrap();
    let mut base = TestChallenger::new(TestPerm::new_from_rng_128(&mut rng));
    base.observe(a.clone());
    base.observe(b.clone());
    let z: EF = base.sample_algebra_element();
    let zz: EF = base.sample_algebra_element();
    let requests = vec![
        OpeningRequest {
            prover_data: &ad,
            points: vec![vec![z, zz]],
        },
        OpeningRequest {
            prover_data: &bd,
            points: vec![vec![z]],
        },
    ];
    let mut prover = base.clone();
    let prepared = pcs.prepare_open(&requests, &mut prover).unwrap();
    // 3 columns at two points plus 5 columns at one point = 11 powers.
    let expected = pcs.get_or_try_compute_pcs_config(6, &[(6, 11)]).unwrap();
    assert!(Arc::ptr_eq(&prepared.stir_configs[0], &expected));
    let (values, proof) = pcs.prove_buckets(&[&ad, &bd], prepared, &mut prover);
    let claims = vec![
        (
            a,
            vec![(
                domain,
                vec![(z, values[0][0][0].clone()), (zz, values[0][0][1].clone())],
            )],
        )
            .into(),
        (b, vec![(domain, vec![(z, values[1][0][0].clone())])]).into(),
    ];
    // A fresh verifier must derive the same shape without relying on a prover-warmed cache.
    let verifier_pcs = pcs.with_batch_proof_of_work_bits(0);
    let mut verifier = base;
    verifier_pcs.verify(claims, &proof, &mut verifier).unwrap();
    assert_eq!(
        prover.sample_algebra_element::<EF>(),
        verifier.sample_algebra_element::<EF>()
    );
    assert!(
        verifier_pcs
            .config_cache
            .read()
            .contains_key(&(6, 1, 0, Some(vec![(6, 11)])))
    );
}

#[test]
fn an_opening_plan_pools_classes_and_offsets_alpha_in_claim_order() {
    let opened = |log_native_height, width, num_points| OpenedMatrix {
        log_native_height,
        width,
        num_points,
    };
    // Both commitments hold a group on the 2^7 domain, so their classes there pool.
    let a = OpenedCommitment {
        groups: GroupPlan {
            log_lde_heights: vec![7, 5],
            group_of_matrix: vec![0, 0, 1],
        },
        matrices: vec![opened(6, 3, 2), opened(5, 2, 1), opened(4, 1, 1)],
    };
    let b = OpenedCommitment {
        groups: GroupPlan {
            log_lde_heights: vec![7, 4],
            group_of_matrix: vec![0, 1],
        },
        matrices: vec![opened(6, 4, 2), opened(3, 2, 1)],
    };

    let plan = OpeningPlan::new(&[a, b]).unwrap();

    let slot = |log_lde_height, log_native_height, class, width, alpha_offset| MatrixSlot {
        log_lde_height,
        log_native_height,
        class,
        width,
        alpha_offset,
    };
    let input = |group, matrices: &[usize]| {
        Some(BucketInput {
            group,
            matrices: matrices.to_vec(),
        })
    };
    assert_eq!(
        plan,
        OpeningPlan {
            commitments: vec![
                CommitmentPlan {
                    num_groups: 2,
                    matrices: vec![
                        slot(7, 6, 0, 3, 0),
                        slot(7, 5, 1, 2, 0),
                        slot(5, 4, 0, 1, 0)
                    ],
                },
                CommitmentPlan {
                    num_groups: 2,
                    // A's 3 columns at 2 points already sit in class `(7, 6)`.
                    matrices: vec![slot(7, 6, 0, 4, 6), slot(4, 3, 0, 2, 0)],
                },
            ],
            buckets: vec![
                BucketPlan {
                    log_lde_height: 7,
                    classes: vec![(6, 14), (5, 2)],
                    inputs: vec![input(0, &[0, 1]), input(0, &[0])],
                },
                BucketPlan {
                    log_lde_height: 5,
                    classes: vec![(4, 1)],
                    inputs: vec![input(1, &[2]), None],
                },
                BucketPlan {
                    log_lde_height: 4,
                    classes: vec![(3, 2)],
                    inputs: vec![None, input(1, &[1])],
                },
            ],
        }
    );
    // Each later point of a matrix starts `width` powers past the one before it.
    assert_eq!(plan.commitments[1].matrices[0].alpha_exponent(1), 10);
}

#[test]
fn an_opening_plan_reports_an_alpha_power_overflow() {
    let opened = OpenedCommitment {
        groups: GroupPlan {
            log_lde_heights: vec![7],
            group_of_matrix: vec![0],
        },
        matrices: vec![OpenedMatrix {
            log_native_height: 6,
            width: usize::MAX,
            num_points: 2,
        }],
    };
    assert_eq!(
        OpeningPlan::new(&[opened]),
        Err(StirConfigError::PcsBatchMultiplicityOverflow)
    );
}

#[test]
fn prover_and_verifier_derive_the_same_opening_plan() {
    // Per-height domains, a partial merge, and a single shared domain: between them, a
    // class pooled across commitments, a merged bucket, and buckets one commitment skips.
    for (spread, num_buckets) in [(0, 4), (1, 3), (DEFAULT_MAX_LOG_HEIGHT_SPREAD, 1)] {
        let pcs = test_pcs_with(spread, SecurityAssumption::CapacityBound, 16);
        let mut rng = SmallRng::seed_from_u64(42);
        let commit = |shape: &[(usize, usize)], rng: &mut SmallRng| {
            let domains: Vec<_> = shape
                .iter()
                .map(|&(log_h, _)| pcs.natural_domain_for_degree(1 << log_h))
                .collect();
            let (commitment, data) = pcs
                .commit(domains.iter().zip(shape).map(|(&domain, &(_, width))| {
                    (
                        domain,
                        RowMajorMatrix::<TestVal>::rand(rng, domain.size(), width),
                    )
                }))
                .unwrap();
            (commitment, data, domains)
        };
        let (commit_a, data_a, domains_a) = commit(&[(6, 3), (5, 2), (4, 1)], &mut rng);
        let (commit_b, data_b, domains_b) = commit(&[(6, 4), (3, 2)], &mut rng);
        let mut challenger = TestChallenger::new(TestPerm::new_from_rng_128(&mut rng));
        challenger.observe(commit_a.clone());
        challenger.observe(commit_b.clone());
        let z1: EF = challenger.sample_algebra_element();
        let z2: EF = challenger.sample_algebra_element();
        let z3: EF = challenger.sample_algebra_element();
        let points = [
            vec![vec![z1, z2], vec![z1], vec![z2]],
            vec![vec![z1, z3], vec![z3]],
        ];

        let prepared = pcs
            .prepare_open(
                &[
                    OpeningRequest {
                        prover_data: &data_a,
                        points: points[0].clone(),
                    },
                    OpeningRequest {
                        prover_data: &data_b,
                        points: points[1].clone(),
                    },
                ],
                &mut challenger,
            )
            .unwrap();
        assert_eq!(prepared.plan.buckets.len(), num_buckets);

        let claims: Vec<_> = [(commit_a, domains_a), (commit_b, domains_b)]
            .into_iter()
            .zip(&points)
            .zip(&prepared.opened_values)
            .map(
                |(((commitment, domains), points), values)| CommitmentOpening {
                    commitment,
                    matrices: izip!(domains, points, values)
                        .map(|(domain, points, values)| MatrixOpening {
                            domain,
                            points: points
                                .iter()
                                .zip(values)
                                .map(|(&point, values)| PointOpening {
                                    point,
                                    values: values.clone(),
                                })
                                .collect(),
                        })
                        .collect(),
                },
            )
            .collect();
        assert_eq!(pcs.claimed_opening_plan(&claims).unwrap(), prepared.plan);
    }
}

#[test]
fn verifier_rejects_in_domain_points_and_inconsistent_widths_before_queries() {
    let pcs = test_pcs_with(2, SecurityAssumption::CapacityBound, 32);
    let mut rng = SmallRng::seed_from_u64(5);
    let domains = [6, 4].map(|h| pcs.natural_domain_for_degree(1 << h));
    let (commitment, _) = pcs
        .commit(domains.map(|d| (d, RowMajorMatrix::<TestVal>::rand(&mut rng, d.size(), 2))))
        .unwrap();
    let bad = EF::from(TestVal::GENERATOR * TestVal::two_adic_generator(7));
    let mut challenger = TestChallenger::new(TestPerm::new_from_rng_128(&mut rng));
    let claims = vec![
        (
            commitment.clone(),
            vec![
                (domains[0], vec![(EF::ZERO, vec![EF::ZERO; 2])]),
                (domains[1], vec![(bad, vec![EF::ZERO; 2])]),
            ],
        )
            .into(),
    ];
    let proof = StirPcsProof {
        batch_pow_witness: None,
        buckets: vec![],
    };
    assert!(matches!(
        pcs.verify(claims, &proof, &mut challenger),
        Err(StirError::OpeningPointInDomain {
            commitment: 0,
            matrix: 1,
            point: 0
        })
    ));
    let claims = vec![
        (
            commitment,
            vec![
                (
                    domains[0],
                    vec![(EF::ZERO, vec![EF::ZERO; 2]), (EF::ONE, vec![EF::ZERO; 3])],
                ),
                (domains[1], vec![(EF::ZERO, vec![EF::ZERO; 2])]),
            ],
        )
            .into(),
    ];
    assert!(matches!(
        pcs.verify(claims, &proof, &mut challenger),
        Err(StirError::InvalidOpeningWidth {
            commitment: 0,
            matrix: 0,
            point: 1
        })
    ));
}

#[test]
fn constant_only_buckets_cannot_claim_an_unenforced_native_degree() {
    let (pcs, params) = test_pcs_and_params();
    assert!(pcs.get_or_try_compute_pcs_config(1, &[(0, 2)]).is_err());
    assert!(matches!(
        TestConfig::try_new_with_pcs_batch(
            6,
            params,
            PcsBatch {
                classes: &[(6, 2)],
                combine: None,
                pow_bits: 32
            },
            StirOptions::default()
        ),
        Err(StirConfigError::InvalidPcsBatchPowBits { .. })
    ));
}

#[test]
fn pcs_schedules_meet_every_budget_without_lending_batch_work_to_later_rounds() {
    use crate::soundness::StirSoundness;
    for assumption in [
        SecurityAssumption::CapacityBound,
        SecurityAssumption::JohnsonBound,
    ] {
        for log_degree in [2, 8, 20] {
            for batch_bits in [16, 24] {
                for early_stop in [None, Some(2)] {
                    let mut pcs = test_pcs_with(2, assumption, 64)
                        .with_batch_proof_of_work_bits(batch_bits)
                        .with_options(StirOptions {
                            max_log_final_poly_len: early_stop,
                            ..Default::default()
                        });
                    pcs.stir.max_pow_bits = 8;
                    let classes = [(log_degree, 64), (log_degree - 1, 32)];
                    let config = pcs
                        .get_or_try_compute_pcs_config(log_degree, &classes)
                        .unwrap();
                    let target = 64. + libm::ceil(libm::log2((6 * config.num_rounds() + 4) as f64));
                    let field_bits = crate::pcs_budget::field_bits::<EF>(assumption);
                    let batch = PcsBatch {
                        classes: &classes,
                        combine: crate::pcs_budget::combine_requirement(log_degree, &classes)
                            .unwrap(),
                        pow_bits: batch_bits,
                    };
                    let first_eta = config
                        .round_configs
                        .first()
                        .map_or(config.final_eta, |r| r.eta);
                    assert!(
                        batch.algebraic_bits(assumption, field_bits, log_degree, 1, first_eta)
                            + batch_bits as f64
                            >= target - 1e-9
                    );
                    for r in &config.round_configs {
                        let rate = r.log_domain_size - r.log_degree;
                        assert!(
                            assumption.stir_query_pow_eligible_bits(
                                field_bits,
                                r.log_degree,
                                rate,
                                r.eta,
                                r.num_queries,
                                r.num_ood_samples
                            ) + r.pow_bits as f64
                                >= target - 1e-9
                        );
                        assert!(
                            assumption.stir_query_unprotected_bits(
                                field_bits,
                                r.log_degree,
                                rate,
                                r.eta,
                                r.num_queries,
                                r.num_ood_samples
                            ) >= target - 1e-9
                        );
                        assert!(
                            assumption.fold_algebraic_bits_at_log_eta(
                                field_bits,
                                r.log_degree,
                                rate,
                                libm::log2(r.eta)
                            ) + r.folding_pow_bits as f64
                                >= target - 1e-9
                        );
                    }
                    let (degree, rate) = config.round_configs.last().map_or((log_degree, 1), |r| {
                        let degree = r.log_degree - r.log_folding_factor;
                        (degree, r.log_domain_size - 1 - degree)
                    });
                    assert!(
                        assumption.stir_final_query_algebraic_bits(
                            rate,
                            config.final_eta,
                            config.final_queries
                        ) + config.final_pow_bits as f64
                            >= target - 1e-9
                    );
                    assert!(
                        assumption.fold_algebraic_bits_at_log_eta(
                            field_bits,
                            degree,
                            rate,
                            libm::log2(config.final_eta)
                        ) + config.final_folding_pow_bits as f64
                            >= target - 1e-9
                    );
                }
            }
        }
    }
}

#[test]
fn verifier_returns_a_config_error_for_an_infeasible_opening_shape() {
    let mut pcs = test_pcs_with(0, SecurityAssumption::CapacityBound, 110);
    pcs.stir.max_pow_bits = 20;
    let mut rng = SmallRng::seed_from_u64(19);
    let domain = pcs.natural_domain_for_degree(64);
    let (commitment, _) = pcs
        .commit([(domain, RowMajorMatrix::<TestVal>::rand(&mut rng, 64, 1))])
        .unwrap();
    let proof = StirPcsProof {
        batch_pow_witness: None,
        buckets: vec![(
            StirProof {
                initial_commitment: None,
                round_proofs: vec![],
                final_polynomial: vec![],
                final_folding_pow_witness: TestVal::ZERO,
                final_pow_witness: TestVal::ZERO,
                final_query_openings: None,
            },
            vec![None],
        )],
    };
    let claims = vec![
        (
            commitment,
            vec![(domain, vec![(EF::ONE, vec![EF::ZERO; 1024])])],
        )
            .into(),
    ];
    let mut challenger = TestChallenger::new(TestPerm::new_from_rng_128(&mut rng));
    assert!(matches!(
        pcs.verify(claims, &proof, &mut challenger),
        Err(StirError::Config(StirConfigError::EtaInfeasibleForTarget {
            label: "PCS joint alpha/Combine bound",
            ..
        }))
    ));
}

#[test]
fn batch_grinding_reduces_serialized_pcs_proofs() {
    let mut original = test_pcs_with(2, SecurityAssumption::CapacityBound, 92);
    original.stir.max_pow_bits = 8;
    let grinded = original.clone().with_batch_proof_of_work_bits(8);
    let mut rng = SmallRng::seed_from_u64(29);
    let domains = [10, 9, 8].map(|h| original.natural_domain_for_degree(1 << h));
    let matrices = domains.map(|d| (d, RowMajorMatrix::<TestVal>::rand(&mut rng, d.size(), 16)));
    let mut sizes = Vec::new();
    let mut queries = Vec::new();
    for pcs in [&original, &grinded] {
        let (commitment, data) = pcs.commit(matrices.clone()).unwrap();
        assert_eq!(commitment.len(), 1, "compare the same grouped layout");
        let mut base =
            TestChallenger::new(TestPerm::new_from_rng_128(&mut SmallRng::seed_from_u64(30)));
        base.observe(commitment.clone());
        let z: EF = base.sample_algebra_element();
        let mut prover = base.clone();
        let prepared = pcs
            .prepare_open(
                &[OpeningRequest {
                    prover_data: &data,
                    points: vec![vec![z]; 3],
                }],
                &mut prover,
            )
            .unwrap();
        queries.push(prepared.stir_configs[0].round_configs[0].num_queries);
        let (values, proof) = pcs.prove_buckets(&[&data], prepared, &mut prover);
        let claims = vec![
            (
                commitment,
                domains
                    .iter()
                    .enumerate()
                    .map(|(i, &d)| (d, vec![(z, values[0][i][0].clone())]))
                    .collect(),
            )
                .into(),
        ];
        let mut verifier = base;
        pcs.verify(claims, &proof, &mut verifier).unwrap();
        assert_eq!(
            prover.sample_algebra_element::<EF>(),
            verifier.sample_algebra_element::<EF>()
        );
        sizes.push(postcard::to_allocvec(&proof).unwrap().len());
    }
    assert!(queries[1] < queries[0], "first-round queries: {queries:?}");
    assert!(sizes[1] < sizes[0], "serialized proof bytes: {sizes:?}");
    std::println!("PCS batch grind 0 -> 8 bits: queries {queries:?}, proof bytes {sizes:?}");
}

#[test]
#[should_panic(expected = "opening point lies in its shared LDE domain")]
fn opening_inside_shared_domain_is_rejected_before_inversion() {
    let pcs = test_pcs_with(2, SecurityAssumption::CapacityBound, 32);
    let mut rng = SmallRng::seed_from_u64(5);
    let domains = [6, 4].map(|h| pcs.natural_domain_for_degree(1 << h));
    let (_, data) = pcs
        .commit(domains.map(|d| (d, RowMajorMatrix::<TestVal>::rand(&mut rng, d.size(), 2))))
        .unwrap();
    // In GEN*H_128, but outside the short matrix's GEN*H_16 domain.
    let bad = EF::from(TestVal::GENERATOR * TestVal::two_adic_generator(7));
    assert_ne!(
        (bad * TestVal::GENERATOR.inverse()).exp_power_of_2(4),
        EF::ONE
    );
    let mut challenger = TestChallenger::new(TestPerm::new_from_rng_128(&mut rng));
    pcs.prepare_open(
        &[OpeningRequest {
            prover_data: &data,
            points: vec![vec![EF::ZERO], vec![bad]],
        }],
        &mut challenger,
    )
    .unwrap();
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// The layout has to be a pure function of the multiset of native heights and this
    /// PCS's parameters: that is the assumption the verifier reconstructs it under, so a
    /// failure here is a verifier disagreeing with the prover for a reason no shape check
    /// in the proof could explain. The interesting inputs are multisets — duplicates,
    /// caller order, and heights sitting on a band edge — so this is sampled rather than
    /// enumerated.
    #[test]
    fn plan_groups_partitions_the_height_multiset(
        heights in prop::collection::vec(2usize..=12, 1..8),
        max_log_height_spread in 0usize..=8,
    ) {
        let pcs = test_pcs_with(
            max_log_height_spread,
            SecurityAssumption::CapacityBound,
            32,
        );
        let plan = pcs.plan_groups(&heights);

        prop_assert_eq!(plan.group_of_matrix.len(), heights.len());
        for pair in plan.log_lde_heights.windows(2) {
            prop_assert!(pair[0] > pair[1]);
        }

        // Distinct heights of each group, read back off the assignment.
        let mut members: Vec<Vec<usize>> = vec![Vec::new(); plan.log_lde_heights.len()];
        for (&h, &g) in heights.iter().zip(&plan.group_of_matrix) {
            if !members[g].contains(&h) {
                members[g].push(h);
            }
        }
        let mut distinct = heights.clone();
        distinct.sort_unstable();
        distinct.dedup();
        prop_assert_eq!(members.iter().map(Vec::len).sum::<usize>(), distinct.len());

        for (group, &log_lde_h) in members.iter_mut().zip(&plan.log_lde_heights) {
            prop_assert!(!group.is_empty());
            group.sort_unstable_by(|a, b| b.cmp(a));
            let (tallest, lowest) = (group[0], group[group.len() - 1]);
            // Each group sits on its own tallest member's domain, ...
            prop_assert_eq!(log_lde_h, tallest + pcs.stir.log_blowup);
            // ... and stays inside the band that member admits, which is what keeps the
            // band probe conservative for whatever union of classes a bucket pools.
            prop_assert!(tallest - lowest <= pcs.combine_band_width(tallest, lowest));
        }

        // Only the multiset decides: permuting the input moves matrices between slots but
        // not between heights, and repeating a height adds no class.
        let permuted: Vec<usize> = heights.iter().rev().copied().collect();
        let permuted_plan = pcs.plan_groups(&permuted);
        let permuted_back: Vec<usize> =
            permuted_plan.group_of_matrix.iter().rev().copied().collect();
        prop_assert_eq!(&permuted_plan.log_lde_heights, &plan.log_lde_heights);
        prop_assert_eq!(&permuted_back, &plan.group_of_matrix);

        let duplicated: Vec<usize> = heights.iter().chain(heights.iter()).copied().collect();
        let duplicated_plan = pcs.plan_groups(&duplicated);
        prop_assert_eq!(&duplicated_plan.log_lde_heights, &plan.log_lde_heights);
        prop_assert_eq!(
            &duplicated_plan.group_of_matrix[..heights.len()],
            &plan.group_of_matrix[..]
        );
    }
}

/// STIR parameters over the test types, at the given soundness knobs.
fn test_params(
    soundness_type: SecurityAssumption,
    security_level: usize,
    log_blowup: usize,
    max_pow_bits: usize,
) -> StirParameters<TestStirMmcs> {
    let mut rng = rand::rngs::SmallRng::seed_from_u64(11);
    let perm = TestPerm::new_from_rng_128(&mut rng);
    let val_mmcs = TestValMmcs::new(TestHash::new(perm.clone()), TestCompress::new(perm), 0);
    StirParameters {
        log_blowup,
        log_folding_factor: 2,
        log_starting_folding_factor: 2,
        soundness_type,
        security_level,
        max_pow_bits,
        mmcs: TestStirMmcs::new(val_mmcs),
    }
}

#[test]
fn band_feasibility_implies_subset_feasibility() {
    // The whole band's Combine error must bound every pooled class subset,
    // with or without the PCS grind. Alpha counts are checked separately at opening.
    for soundness_type in [
        SecurityAssumption::CapacityBound,
        SecurityAssumption::JohnsonBound,
    ] {
        for security_level in [32usize, 64, 80] {
            for log_blowup in [1usize, 2] {
                for (max_pow_bits, batch_bits) in [(0usize, 0), (16, 0), (16, 8)] {
                    let params =
                        test_params(soundness_type, security_level, log_blowup, max_pow_bits);
                    for log_d_star in [6usize, 10, 14] {
                        for w in 1..=6.min(log_d_star) {
                            let band: Vec<usize> =
                                (0..=w).map(|i| log_d_star - i).collect::<Vec<_>>();
                            let (n, ell) = TestPcs::combine_key(&band)
                                .expect("a band of width >= 1 holds two classes");
                            if TestConfig::try_new_with_pcs_batch(
                                log_d_star,
                                params.clone(),
                                PcsBatch {
                                    classes: &[],
                                    combine: Some((n, ell)),
                                    pow_bits: batch_bits,
                                },
                                StirOptions::default(),
                            )
                            .is_err()
                            {
                                // The band itself is infeasible, so it implies nothing.
                                continue;
                            }

                            for mask in 0u32..(1 << w) {
                                let mut subset = vec![log_d_star];
                                subset.extend(
                                    (0..w)
                                        .filter(|i| mask & (1 << i) != 0)
                                        .map(|i| log_d_star - 1 - i),
                                );
                                let Some((sub_n, sub_ell)) = TestPcs::combine_key(&subset) else {
                                    continue;
                                };
                                assert!(
                                    TestConfig::try_new_with_pcs_batch(
                                        log_d_star,
                                        params.clone(),
                                        PcsBatch {
                                            classes: &[],
                                            combine: Some((sub_n, sub_ell)),
                                            pow_bits: batch_bits
                                        },
                                        StirOptions::default(),
                                    )
                                    .is_ok(),
                                    "band [{}..{log_d_star}] configures but subset \
                                     {subset:?} does not, at {soundness_type:?} \
                                     security_level={security_level} \
                                     log_blowup={log_blowup} max_pow_bits={max_pow_bits} batch_bits={batch_bits}",
                                    log_d_star - w,
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}

fn test_pcs_and_params() -> (TestPcs, StirParameters<TestStirMmcs>) {
    let mut rng = rand::rngs::SmallRng::seed_from_u64(11);
    let perm = TestPerm::new_from_rng_128(&mut rng);
    let val_mmcs = TestValMmcs::new(TestHash::new(perm.clone()), TestCompress::new(perm), 0);
    let stir = StirParameters {
        log_blowup: 1,
        log_folding_factor: 2,
        log_starting_folding_factor: 2,
        soundness_type: SecurityAssumption::CapacityBound,
        security_level: 32,
        max_pow_bits: 0,
        mmcs: TestStirMmcs::new(val_mmcs.clone()),
    };
    (
        TwoAdicStirPcs::new(Radix2DitParallel::default(), val_mmcs, stir.clone()),
        stir,
    )
}

#[test]
fn pcs_rejects_infeasible_quotient_batching() {
    let mut pcs = test_pcs_with(0, SecurityAssumption::CapacityBound, 100);
    pcs.stir.max_pow_bits = 16;
    let mut rng = SmallRng::seed_from_u64(923);
    let mut challenger = TestChallenger::new(TestPerm::new_from_rng_128(&mut rng));
    let domain = <TestPcs as Pcs<EF, TestChallenger>>::natural_domain_for_degree(&pcs, 256);
    // Two points for 1024 columns means 2048 quotients in one alpha batch.
    let (commitment, data) = <TestPcs as Pcs<EF, TestChallenger>>::commit(
        &pcs,
        vec![(domain, RowMajorMatrix::<TestVal>::rand(&mut rng, 256, 1024))],
    )
    .unwrap();
    challenger.observe(commitment);
    let points = vec![
        challenger.sample_algebra_element(),
        challenger.sample_algebra_element(),
    ];
    let before: EF = challenger.clone().sample_algebra_element();
    let result = pcs.open(
        vec![OpeningRequest {
            prover_data: &data,
            points: vec![points],
        }],
        &mut challenger,
    );
    assert!(matches!(
        result,
        Err(StirConfigError::EtaInfeasibleForTarget { .. })
    ));
    assert_eq!(challenger.sample_algebra_element::<EF>(), before);
}

#[test]
fn pcs_quotient_batching_reaches_the_security_target() {
    let mut pcs = test_pcs_with(0, SecurityAssumption::CapacityBound, 100);
    pcs.stir.max_pow_bits = 16;
    let mut rng = SmallRng::seed_from_u64(924);
    let mut challenger = TestChallenger::new(TestPerm::new_from_rng_128(&mut rng));
    let domain = <TestPcs as Pcs<EF, TestChallenger>>::natural_domain_for_degree(&pcs, 256);
    let (commitment, data) = <TestPcs as Pcs<EF, TestChallenger>>::commit(
        &pcs,
        vec![(domain, RowMajorMatrix::<TestVal>::rand(&mut rng, 256, 32))],
    )
    .unwrap();
    challenger.observe(commitment);
    let points = [
        challenger.sample_algebra_element(),
        challenger.sample_algebra_element(),
    ];
    // Warming the same-degree cache with one point must not underbudget two points.
    for num_points in [1, 2] {
        let prepared = pcs
            .prepare_open(
                &[OpeningRequest {
                    prover_data: &data,
                    points: vec![points[..num_points].to_vec()],
                }],
                &mut challenger.clone(),
            )
            .unwrap();
        let config = &prepared.stir_configs[0];
        assert_eq!(config.quotient_batches, vec![(8, 32 * num_points)]);
        // Capacity bound, independently evaluated at the config's initial eta.
        //
        // The 123 is the rigorous field size.
        //
        // The extension has a 124-bit order, priced one bit below at floor(log2(|E|)).
        //
        // The capacity assumption charges no further reserve.
        //
        // This is therefore the same number the derivation itself uses.
        let batching_bits = 123. - libm::log2((32 * num_points - 1) as f64) - 8. - 2.
            + libm::log2(config.round_configs[0].eta);
        // Four folds have 22 error terms after adding PCS batching: ceil(log2 22)=5.
        assert!(
            batching_bits >= 105. - 1e-10,
            "batching only retains {batching_bits} bits"
        );
        assert!((config.initial_batching_error() - batching_bits).abs() < 1e-10);
    }
}

#[test]
fn cached_configs_match_a_fresh_derivation() {
    // An under-specified cache key is silent in the worst way: a proof produced under one
    // config and checked under another. Deriving the same shapes twice through the cache
    // and comparing against an uncached derivation is what catches it — in particular that
    // two buckets sharing a degree but differing in class count do not collide.
    let (pcs, stir) = test_pcs_and_params();

    let shapes: [(usize, Option<(usize, u64)>); 4] = [
        (8, None),
        // Same degree, but merging two classes: a degree-only key would alias these.
        (8, Some((2, 194))),
        (8, Some((3, 300))),
        (6, None),
    ];

    for (log_stir_degree, combine) in shapes {
        let expected = TestConfig::try_new_with_pcs_batch(
            log_stir_degree,
            stir.clone(),
            PcsBatch {
                classes: &[],
                combine,
                pow_bits: 0,
            },
            StirOptions::default(),
        )
        .expect("feasible shape");

        // Twice: the first call populates the entry, the second must return the same one.
        for round in 0..2 {
            let cached = pcs
                .get_or_try_compute_stir_config(log_stir_degree, combine)
                .expect("feasible shape");
            assert_eq!(
                schedule_fingerprint(&cached),
                schedule_fingerprint(&expected),
                "deg={log_stir_degree} combine={combine:?} round={round}"
            );
        }
    }

    // One entry per distinct shape, so nothing aliased and nothing was inserted twice.
    assert_eq!(pcs.config_cache.read().len(), shapes.len());
}

#[test]
fn cache_keeps_each_native_class_quotient_count() {
    let pcs = test_pcs_with(3, SecurityAssumption::CapacityBound, 32);
    let first = pcs
        .get_or_try_compute_pcs_config(8, &[(8, 512), (5, 2)])
        .unwrap();
    let second = pcs
        .get_or_try_compute_pcs_config(8, &[(8, 2), (5, 512)])
        .unwrap();
    assert!(!Arc::ptr_eq(&first, &second));
    assert_ne!(
        first.initial_batching_error(),
        second.initial_batching_error()
    );
    assert_eq!(first.quotient_batches, vec![(8, 512), (5, 2)]);
    assert_eq!(second.quotient_batches, vec![(8, 2), (5, 512)]);
}

#[test]
fn early_stop_options_keep_warm_clone_caches_and_proofs_independent() {
    let (original, params) = test_pcs_and_params();
    let shapes = [(8, None), (8, Some((2, 194)))];
    for (degree, combine) in shapes {
        assert_eq!(
            original
                .get_or_compute_stir_config(degree, combine)
                .num_rounds(),
            3
        );
    }
    let options = crate::StirOptions {
        max_log_final_poly_len: Some(4),
        ..Default::default()
    };
    let early = original.clone().with_options(options);
    assert_eq!(original.options(), crate::StirOptions::default());
    assert_eq!(early.options(), options);
    for (degree, combine) in shapes {
        let expected = TestConfig::try_new_with_pcs_batch(
            degree,
            params.clone(),
            PcsBatch {
                classes: &[],
                combine,
                pow_bits: 0,
            },
            options,
        )
        .unwrap();
        let actual = early.get_or_compute_stir_config(degree, combine);
        assert_eq!(actual.num_rounds(), 1);
        assert_eq!(
            schedule_fingerprint(&actual),
            schedule_fingerprint(&expected)
        );
        assert_eq!(
            original
                .get_or_compute_stir_config(degree, combine)
                .num_rounds(),
            3
        );
    }

    // Exercise ordinary buckets separately, then merge height classes through Combine.
    for spread in [0, 2] {
        for pcs in [original.clone(), early.clone()] {
            let pcs = pcs.with_max_log_height_spread(spread);
            let mut rng = rand::rngs::SmallRng::seed_from_u64(314);
            let perm = TestPerm::new_from_rng_128(&mut rng);
            let mut base = TestChallenger::new(perm);
            let domains: Vec<_> = [8, 6, 4]
                .map(|log_h| {
                    <TestPcs as Pcs<EF, TestChallenger>>::natural_domain_for_degree(
                        &pcs,
                        1 << log_h,
                    )
                })
                .into_iter()
                .collect();
            let inputs: Vec<_> = domains
                .iter()
                .zip([8, 6, 4])
                .map(|(&domain, log_h)| {
                    (
                        domain,
                        RowMajorMatrix::<TestVal>::rand(&mut rng, 1 << log_h, 2),
                    )
                })
                .collect();
            let (commit, data) =
                <TestPcs as Pcs<EF, TestChallenger>>::commit(&pcs, inputs).unwrap();
            base.observe(commit.clone());
            let zeta: EF = base.sample_algebra_element();
            let (values, proof) = <TestPcs as Pcs<EF, TestChallenger>>::open(
                &pcs,
                vec![OpeningRequest {
                    prover_data: &data,
                    points: vec![vec![zeta]; 3],
                }],
                &mut base.clone(),
            )
            .unwrap();
            let claims = vec![CommitmentOpening {
                commitment: commit,
                matrices: domains
                    .into_iter()
                    .enumerate()
                    .map(|(i, domain)| MatrixOpening {
                        domain,
                        points: vec![PointOpening {
                            point: zeta,
                            values: values[0][i][0].clone(),
                        }],
                    })
                    .collect(),
            }];
            <TestPcs as Pcs<EF, TestChallenger>>::verify(&pcs, claims, &proof, &mut base)
                .expect("ordinary and Combine proofs verify under their own options");
        }
    }
}

#[test]
fn compact_answers_preserve_pcs_values_and_warm_clone_transcripts() {
    for cap in [None, Some(4)] {
        for spread in [0, 2] {
            let (pcs, _) = test_pcs_and_params();
            let options = StirOptions {
                max_log_final_poly_len: cap,
                ..Default::default()
            };
            let pcs = pcs.with_max_log_height_spread(spread).with_options(options);
            let mut rng = rand::rngs::SmallRng::seed_from_u64(315);
            let perm = TestPerm::new_from_rng_128(&mut rng);
            let mut base = TestChallenger::new(perm);
            let domains: Vec<_> = [8, 6, 4]
                .map(|log_h| {
                    <TestPcs as Pcs<EF, TestChallenger>>::natural_domain_for_degree(
                        &pcs,
                        1 << log_h,
                    )
                })
                .into();
            let inputs: Vec<_> = domains
                .iter()
                .zip([8, 6, 4])
                .map(|(&domain, log_h)| {
                    (
                        domain,
                        RowMajorMatrix::<TestVal>::rand(&mut rng, 1 << log_h, 2),
                    )
                })
                .collect();
            let (commit, data) =
                <TestPcs as Pcs<EF, TestChallenger>>::commit(&pcs, inputs).unwrap();
            base.observe(commit.clone());
            let zeta: EF = base.sample_algebra_element();
            let mut full_p = base.clone();
            let (values, full_proof) = <TestPcs as Pcs<EF, TestChallenger>>::open(
                &pcs,
                vec![OpeningRequest {
                    prover_data: &data,
                    points: vec![vec![zeta]; 3],
                }],
                &mut full_p,
            )
            .unwrap();
            // Clone only after opening has warmed both ordinary and Combine schedules.
            let compact = pcs.clone().with_options(StirOptions {
                compact_answers: true,
                ..options
            });
            let mut compact_p = base.clone();
            let (compact_values, compact_proof) = <TestPcs as Pcs<EF, TestChallenger>>::open(
                &compact,
                vec![OpeningRequest {
                    prover_data: &data,
                    points: vec![vec![zeta]; 3],
                }],
                &mut compact_p,
            )
            .unwrap();
            assert_eq!(values, compact_values);
            let next: EF = full_p.sample_algebra_element();
            assert_eq!(next, compact_p.sample_algebra_element::<EF>());
            let claims = vec![CommitmentOpening {
                commitment: commit,
                matrices: domains
                    .into_iter()
                    .enumerate()
                    .map(|(i, domain)| MatrixOpening {
                        domain,
                        points: vec![PointOpening {
                            point: zeta,
                            values: values[0][i][0].clone(),
                        }],
                    })
                    .collect::<Vec<_>>(),
            }];
            for (pcs, proof) in [(&pcs, &full_proof), (&compact, &compact_proof)] {
                let mut v_ch = base.clone();
                <TestPcs as Pcs<EF, TestChallenger>>::verify(pcs, claims.clone(), proof, &mut v_ch)
                    .unwrap();
                assert_eq!(next, v_ch.sample_algebra_element::<EF>());
            }
            assert!(
                pcs.config_cache
                    .read()
                    .values()
                    .all(|c| !c.options().compact_answers)
            );
            assert!(
                compact
                    .config_cache
                    .read()
                    .values()
                    .all(|c| c.options().compact_answers)
            );
            assert!(
                postcard::to_allocvec(&compact_proof).unwrap().len()
                    < postcard::to_allocvec(&full_proof).unwrap().len()
            );
        }
    }
}

/// A prover that commits a perfectly low-degree codeword which is *not* the reduced
/// opening passes every STIR round — the lane checks are the only thing standing between
/// such a prover and acceptance. Adding a random low-degree codeword to `f_0` keeps STIR
/// happy and must be caught by the first lane the verifier compares.
#[test]
fn verify_rejects_a_low_degree_initial_oracle_that_is_not_the_reduced_opening() {
    let pcs = test_pcs_with(
        DEFAULT_MAX_LOG_HEIGHT_SPREAD,
        SecurityAssumption::CapacityBound,
        32,
    );
    let mut rng = rand::rngs::SmallRng::seed_from_u64(7);
    let perm = TestPerm::new_from_rng_128(&mut rng);
    let mut challenger = TestChallenger::new(perm);

    let log_h = 8;
    let log_lde = log_h + pcs.stir.log_blowup;
    let domain = <TestPcs as Pcs<EF, TestChallenger>>::natural_domain_for_degree(&pcs, 1 << log_h);
    let mat = RowMajorMatrix::<TestVal>::rand(&mut rng, 1 << log_h, 4);
    let (commit, data) =
        <TestPcs as Pcs<EF, TestChallenger>>::commit(&pcs, vec![(domain, mat)]).unwrap();
    challenger.observe(commit.clone());
    let zeta: EF = challenger.sample_algebra_element();

    let mut p_ch = challenger.clone();
    let mut prepared = pcs
        .prepare_open(
            &[OpeningRequest {
                prover_data: &data,
                points: vec![vec![zeta]],
            }],
            &mut p_ch,
        )
        .unwrap();
    // Fixture state: one commitment at one height.
    //
    // That is one bucket holding one class.
    assert_eq!(prepared.plan.buckets.len(), 1);

    // Mutation: the codeword the bucket will run on is no longer its reduced opening.
    //
    //     f_0 = reduced opening + a random codeword of the same degree bound
    add_low_degree_codeword(
        &pcs,
        &mut prepared.reduced_openings,
        log_h,
        log_lde,
        &mut rng,
    );

    let (opened_values, proof) = pcs.prove_buckets(&[&data], prepared, &mut p_ch);

    let mut v_ch = challenger;
    let claims = vec![(
        commit,
        vec![(domain, vec![(zeta, opened_values[0][0][0].clone())])],
    )];
    let err = <TestPcs as Pcs<EF, TestChallenger>>::verify(
        &pcs,
        claims.into_iter().map(Into::into).collect(),
        &proof,
        &mut v_ch,
    )
    .expect_err("a low-degree oracle that is not the reduced opening must be rejected");
    assert!(
        matches!(err, StirError::InitialOracleMismatch { log_height, .. } if log_height == log_lde),
        "expected InitialOracleMismatch, got {err:?}"
    );
}

/// A bucket with no intermediate rounds reads its initial oracle from the *final* round,
/// at the same arity, so the lane check has to bind there too. Same corruption as above,
/// on the schedule where the fold-domain/lane split has no intermediate round to hide in.
#[test]
fn verify_rejects_a_low_degree_initial_oracle_at_a_zero_round_bucket() {
    let pcs = test_pcs_with(
        DEFAULT_MAX_LOG_HEIGHT_SPREAD,
        SecurityAssumption::CapacityBound,
        16,
    );
    let mut rng = rand::rngs::SmallRng::seed_from_u64(13);
    let perm = TestPerm::new_from_rng_128(&mut rng);
    let mut challenger = TestChallenger::new(perm);

    // `log_stir_degree == log_folding_factor`, so STIR schedules no intermediate round.
    let log_h = 2;
    let log_lde = log_h + pcs.stir.log_blowup;
    let domain = <TestPcs as Pcs<EF, TestChallenger>>::natural_domain_for_degree(&pcs, 1 << log_h);
    let mat = RowMajorMatrix::<TestVal>::rand(&mut rng, 1 << log_h, 4);
    let (commit, data) =
        <TestPcs as Pcs<EF, TestChallenger>>::commit(&pcs, vec![(domain, mat)]).unwrap();
    challenger.observe(commit.clone());
    let zeta: EF = challenger.sample_algebra_element();

    let mut p_ch = challenger.clone();
    let mut prepared = pcs
        .prepare_open(
            &[OpeningRequest {
                prover_data: &data,
                points: vec![vec![zeta]],
            }],
            &mut p_ch,
        )
        .unwrap();
    assert_eq!(
        prepared.stir_configs[0].num_rounds(),
        0,
        "this test exists to cover the zero-intermediate-round schedule"
    );

    // Mutation: the same corruption, on a schedule with no intermediate round.
    add_low_degree_codeword(
        &pcs,
        &mut prepared.reduced_openings,
        log_h,
        log_lde,
        &mut rng,
    );

    let (opened_values, proof) = pcs.prove_buckets(&[&data], prepared, &mut p_ch);

    let mut v_ch = challenger;
    let claims = vec![(
        commit,
        vec![(domain, vec![(zeta, opened_values[0][0][0].clone())])],
    )];
    let err = <TestPcs as Pcs<EF, TestChallenger>>::verify(
        &pcs,
        claims.into_iter().map(Into::into).collect(),
        &proof,
        &mut v_ch,
    )
    .expect_err("a low-degree oracle that is not the reduced opening must be rejected");
    assert!(
        matches!(err, StirError::InitialOracleMismatch { log_height, .. } if log_height == log_lde),
        "expected InitialOracleMismatch, got {err:?}"
    );
}

#[test]
fn lanes_are_drawn_once_per_round_zero_draw() {
    let pcs = test_pcs_with(
        DEFAULT_MAX_LOG_HEIGHT_SPREAD,
        SecurityAssumption::CapacityBound,
        16,
    );
    let mut rng = rand::rngs::SmallRng::seed_from_u64(21);
    let perm = TestPerm::new_from_rng_128(&mut rng);
    let mut base = TestChallenger::new(perm);

    let log_h = 2;
    let domain = <TestPcs as Pcs<EF, TestChallenger>>::natural_domain_for_degree(&pcs, 1 << log_h);
    let mat = RowMajorMatrix::<TestVal>::rand(&mut rng, 1 << log_h, 2);
    let (commit, data) =
        <TestPcs as Pcs<EF, TestChallenger>>::commit(&pcs, vec![(domain, mat)]).unwrap();
    base.observe(commit);
    let zeta: EF = base.sample_algebra_element();

    // Invariant: one lane per first-round query draw, in draw order.
    //
    // Never one lane per distinct fiber.
    //
    // A repeated fiber would then contribute no fresh randomness.
    //
    // The per-draw product the round's query count is priced on would not hold.
    //
    // Fixture state: a zero-intermediate-round bucket, fold domain of two indices.
    //
    // A repeated fiber is forced there.
    //
    // The two counts actually differ.

    // The whole prover side, lanes included.
    let mut ch_full = base.clone();
    let prepared = pcs
        .prepare_open(
            &[OpeningRequest {
                prover_data: &data,
                points: vec![vec![zeta]],
            }],
            &mut ch_full,
        )
        .unwrap();
    let log_arity0 = prepared.stir_configs[0].log_starting_folding_factor;
    let final_queries = prepared.stir_configs[0].final_queries;
    let _ = pcs.prove_buckets(&[&data], prepared, &mut ch_full);

    // The same run, replayed step by step.
    //
    // Both the draws and the lanes are visible that way.
    let mut ch_draws = base;
    let PreparedOpen {
        plan,
        stir_configs,
        mut reduced_openings,
        ..
    } = pcs
        .prepare_open(
            &[OpeningRequest {
                prover_data: &data,
                points: vec![vec![zeta]],
            }],
            &mut ch_draws,
        )
        .unwrap();

    // The bucket phase's description, built exactly as the prover builds it.
    let shape = plan.transcript_shape(&stir_configs);
    let described_draws = shape.buckets[0].num_query_draws;
    assert_eq!(
        described_draws, final_queries,
        "a zero-round bucket reads its initial oracle through the final round's queries"
    );

    let configs: Vec<&TestConfig> = stir_configs.iter().map(AsRef::as_ref).collect();
    let mut transcript =
        OpeningProverTranscript::<TestChallenger, TestVal, EF>::new(&mut ch_draws, shape.clone());

    // One native height.
    //
    // The bucket merges nothing and draws no challenge.
    assert!(transcript.combination_challenge(0).is_none());
    let codeword = combined_bucket_codeword::<TestVal, EF>(
        &mut reduced_openings,
        plan.buckets[0].log_lde_height,
        &plan.buckets[0].log_native_heights(),
        None,
    );
    let results = transcript.delegate(|challenger| {
        prove_stir_multi_from_codewords(&configs, vec![codeword], &pcs.dft, challenger)
    });
    let lanes = transcript.lanes(0);
    transcript.finish();

    let draws = &results[0].1.draws;
    let unique = &results[0].1.unique_sorted;
    assert!(
        unique.len() < draws.len(),
        "the fold domain must be small enough that a fiber repeats, or the two lane \
         counts are indistinguishable"
    );

    // The described count is the draw count.
    //
    // One lane came out per draw.
    assert_eq!(described_draws, draws.len());
    assert_eq!(lanes.len(), draws.len());

    // The replay lands where the real prover landed.
    let after_full: EF = ch_full.sample_algebra_element();
    let after_replay: EF = ch_draws.sample_algebra_element();
    assert_eq!(after_full, after_replay);

    // Mutation: describe one lane per unique fiber instead of one per draw.
    //
    //     draws  : 2 draws of fiber 1  ->  2 lanes described
    //     unique : 1 distinct fiber    ->  1 lane described
    //
    // The count travels in the description.
    //
    // The two cannot share a seed.
    let mut per_unique = shape.clone();
    per_unique.buckets[0].num_query_draws = unique.len();
    assert_ne!(
        seed_digest(&shape.domain_separator::<TestVal, EF>()),
        seed_digest(&per_unique.domain_separator::<TestVal, EF>()),
    );

    // Two draws of one fiber with different lanes are two distinct positions and both get
    // opened; with the same lane they collapse to one opened row, but two lanes were
    // still drawn. `log_h + 1` is the LDE height, so the fold domain holds two indices.
    assert_eq!(
        query_positions(&[1, 1], &[0, 1], log_h + 1, log_arity0),
        [1, 3]
    );
    assert_eq!(
        query_positions(&[1, 1], &[1, 1], log_h + 1, log_arity0),
        [3]
    );
}

/// The lane the PCS compares must be the codeword position STIR actually authenticated.
///
/// `query_positions` maps `(fiber j, lane l)` to `p = j + l * 2^(log_h - log_arity0)`.
/// That is only right if it agrees with the layout the prover's fiber matrix commits and
/// with the subgroup point the verifier folds at, so it is pinned here against a real
/// STIR instance's own round-0 fibers rather than against itself.
#[test]
fn fiber_lanes_index_the_committed_codeword_positions() {
    let mut rng = rand::rngs::SmallRng::seed_from_u64(5);
    let perm = TestPerm::new_from_rng_128(&mut rng);
    let dft = Radix2DitParallel::<TestVal>::default();
    let params = test_params(SecurityAssumption::CapacityBound, 32, 1, 0);
    let config = TestConfig::try_new(6, params).expect("feasible shape");
    let log_h = config.log_starting_domain_size();
    let log_arity0 = config.log_starting_folding_factor;
    let fold_height0 = (1usize << log_h) >> log_arity0;

    let mut coeffs: Vec<EF> = (0..1usize << 6).map(|_| rng.random()).collect();
    coeffs.resize(1 << log_h, EF::ZERO);
    let codeword = codeword_from_coeffs(&dft, coeffs, TestVal::GENERATOR, log_h);

    let mut p_ch = TestChallenger::new(perm.clone());
    let results =
        prove_stir_multi_from_codewords(&[&config], vec![codeword.clone()], &dft, &mut p_ch);
    let proofs: Vec<_> = results.iter().map(|(proof, _)| proof).collect();
    let mut v_ch = TestChallenger::new(perm);
    let outputs =
        verify_stir_multi(&[&config], &proofs, &mut v_ch).expect("an honest proof verifies");

    let output = &outputs[0];
    assert!(!output.first_round_indices.is_empty());
    for (&j, fiber) in output
        .first_round_indices
        .iter()
        .zip(&output.first_round_fiber_evals)
    {
        assert_eq!(fiber.len(), 1 << log_arity0);
        for (lane, &value) in fiber.iter().enumerate() {
            let positions = query_positions(&[j], &[lane], log_h, log_arity0);
            assert_eq!(positions, [j + lane * fold_height0]);
            assert_eq!(split_position(positions[0], fold_height0), (j, lane));
            assert_eq!(value, codeword[positions[0]], "fiber {j}, lane {lane}");
        }
    }
}

#[test]
fn cache_errors_are_not_memoized() {
    // A garbage claim shape must not be able to occupy a cache slot permanently.
    let (pcs, _) = test_pcs_and_params();
    assert!(
        pcs.get_or_try_compute_stir_config(8, Some((2, 1))).is_err(),
        "ell below the class count must be rejected"
    );
    assert!(pcs.config_cache.read().is_empty());
}
