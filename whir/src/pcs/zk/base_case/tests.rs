//! Standalone Construction 7.2 tests over a directly-committed source.
//!
//! A directly-committed width-one codeword is the trivial virtual fold.
//! The closure-based source abstraction collapses to plain openings.

use alloc::vec;
use alloc::vec::Vec;
use core::slice::from_ref;

use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::DuplexChallenger;
use p3_commit::{BatchOpeningRef, ExtensionMmcs, Mmcs};
use p3_dft::Radix2DFTSmallBatch;
use p3_field::extension::BinomialExtensionField;
use p3_field::{Field, PrimeCharacteristicRing, dot_product};
use p3_matrix::Dimensions;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use p3_zk_codes::{ZkEncoding, ZkEncodingWithRandomness};
use proptest::prelude::*;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

use super::*;
use crate::pcs::proof::QueryOpening;
use crate::pcs::zk::committer::FoldedRsCode;
use crate::pcs::zk::mask::{MaskCodeShape, MaskGroupShape};
use crate::pcs::zk::proof::BaseCaseZkProof;

type F = BabyBear;
type EF = BinomialExtensionField<F, 4>;
type Perm = Poseidon2BabyBear<16>;
type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
type MyChallenger = DuplexChallenger<F, Perm, 16, 8>;
type PackedF = <F as Field>::Packing;
type MyMmcs = MerkleTreeMmcs<PackedF, PackedF, MyHash, MyCompress, 2, 8>;
type MyDft = Radix2DFTSmallBatch<F>;

fn setup() -> (
    MyChallenger,
    MyChallenger,
    ExtensionMmcs<F, EF, MyMmcs>,
    MyDft,
) {
    let perm = Perm::new_from_rng_128(&mut SmallRng::seed_from_u64(1));
    let prover_challenger = MyChallenger::new(perm.clone());
    let verifier_challenger = MyChallenger::new(perm.clone());
    let mmcs = MyMmcs::new(MyHash::new(perm.clone()), MyCompress::new(perm), 0);
    (
        prover_challenger,
        verifier_challenger,
        ExtensionMmcs::new(mmcs),
        MyDft::default(),
    )
}

/// Which committed oracle a run should desynchronize from its reveals.
///
/// The reveals and target are always built from the original messages, so
/// the joint target check still closes; only the committed codeword differs,
/// so verification proceeds to the spot-check equation under test.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Tamper {
    /// Commit and reveal the same messages (an honest run).
    None,
    /// Commit a source codeword that differs from the revealed message.
    SourceMessage,
    /// Commit a mask codeword that differs from the revealed message.
    MaskMessage,
}

/// One standalone run, returning everything a check needs.
#[allow(clippy::type_complexity)]
fn honest_run(
    seed: u64,
    num_masks: usize,
    pow_bits: usize,
    tamper: Tamper,
    source_override: Option<Vec<EF>>,
) -> (
    BaseCaseZkConfig<F>,
    ExtensionMmcs<F, EF, MyMmcs>,
    BaseCaseZkProof<F, EF, MyMmcs>,
    Vec<EF>,
    Vec<Vec<EF>>,
    Vec<<MyMmcs as Mmcs<F>>::Commitment>,
    EF,
    <MyMmcs as Mmcs<F>>::Commitment,
    MyChallenger,
) {
    let mut rng = SmallRng::seed_from_u64(seed);
    let (mut prover_challenger, verifier_challenger, extension_mmcs, dft) = setup();

    // Source: message 8, randomness 3, domain 16, committed directly.
    let code = FoldedRsCode::<F>::new(8, 3, 16);
    let source_message: Vec<EF> =
        source_override.unwrap_or_else(|| (0..code.message_len).map(|_| rng.random()).collect());
    let source_randomness: Vec<EF> = (0..code.randomness_len).map(|_| rng.random()).collect();
    // Under SourceMessage the committed codeword encodes a shifted message,
    // while the reveals and target keep the original.
    let mut committed_source = source_message.clone();
    if tamper == Tamper::SourceMessage {
        committed_source[0] += EF::ONE;
    }
    let source_codeword = code.encode_column(&dft, &committed_source, &source_randomness);
    let (source_commitment, source_data) = extension_mmcs.commit_matrix(source_codeword);

    // Carried masks as width-one groups with their own RS-ZK codes.
    let mask_groups: Vec<MaskGroupShape> = (0..num_masks)
        .map(|i| MaskGroupShape {
            shape: MaskCodeShape::new(4 + i, 2, 1),
            width: 1,
        })
        .collect();
    let mut mask_messages = Vec::new();
    let mut mask_randomness = Vec::new();
    let mut mask_covectors = Vec::new();
    let mut mask_commitments = Vec::new();
    let mut mask_data = Vec::new();
    for (i, group) in mask_groups.iter().enumerate() {
        let encoding = group.shape.encoding::<EF>();
        let message = encoding.sample_message(&mut rng);
        let randomness = encoding.sample_randomness(&mut rng);
        // Under MaskMessage the first committed mask encodes a shifted
        // message, while the reveals and target keep the original.
        let mut committed = message.clone();
        if tamper == Tamper::MaskMessage && i == 0 {
            committed[0] += EF::ONE;
        }
        let codeword = encoding.encode_with_randomness(&committed, &randomness);
        let (commitment, data) = extension_mmcs.commit_matrix(codeword);
        let covector: Vec<EF> = (0..group.shape.message_len).map(|_| rng.random()).collect();
        mask_messages.push(vec![message]);
        mask_randomness.push(vec![randomness]);
        mask_covectors.push(covector);
        mask_commitments.push(commitment);
        mask_data.push(data);
    }

    // Joint relation and its honest target.
    let source_covector: Vec<EF> = (0..code.message_len).map(|_| rng.random()).collect();
    let mut target = dot_product::<EF, _, _>(
        source_message.iter().copied(),
        source_covector.iter().copied(),
    );
    for (messages, covector) in mask_messages.iter().zip(&mask_covectors) {
        target += dot_product::<EF, _, _>(messages[0].iter().copied(), covector.iter().copied());
    }

    let config = BaseCaseZkConfig {
        code,
        mask_groups,
        num_queries: 4,
        mask_queries: 2,
        pow_bits,
    };
    let prover = BaseCaseZkProver {
        config: &config,
        extension_mmcs: &extension_mmcs,
    };

    let witnesses: Vec<MaskGroupWitness<'_, F, EF, MyMmcs>> = mask_messages
        .iter()
        .zip(&mask_randomness)
        .zip(&mask_covectors)
        .zip(&mask_data)
        .map(
            |(((messages, randomness), covector), data)| MaskGroupWitness {
                messages,
                randomness,
                covectors: from_ref(covector),
                data,
            },
        )
        .collect();

    let proof = prover.prove(
        &dft,
        &source_message,
        &source_randomness,
        &source_covector,
        &witnesses,
        |position| {
            let opening = extension_mmcs.open_batch(position, &source_data);
            QueryOpening::Extension {
                values: opening.opened_values.into_iter().next().unwrap(),
                proof: opening.opening_proof,
            }
        },
        &mut prover_challenger,
        &mut rng,
    );

    (
        config,
        extension_mmcs,
        proof,
        source_covector,
        mask_covectors,
        mask_commitments,
        target,
        source_commitment,
        verifier_challenger,
    )
}

/// Runs the verifier against a (possibly tampered) standalone proof.
#[allow(clippy::too_many_arguments)]
fn verify_run(
    config: &BaseCaseZkConfig<F>,
    extension_mmcs: &ExtensionMmcs<F, EF, MyMmcs>,
    proof: &BaseCaseZkProof<F, EF, MyMmcs>,
    source_covector: &[EF],
    mask_covectors: &[Vec<EF>],
    mask_commitments: &[<MyMmcs as Mmcs<F>>::Commitment],
    target: EF,
    source_commitment: &<MyMmcs as Mmcs<F>>::Commitment,
    mut challenger: MyChallenger,
) -> Result<(), BaseCaseZkError> {
    let verifier = BaseCaseZkVerifier {
        config,
        extension_mmcs,
    };
    let dims = [Dimensions {
        height: config.code.domain_size,
        width: 1,
    }];
    verifier.verify(
        proof,
        source_covector,
        mask_covectors,
        mask_commitments,
        target,
        |position, opening| {
            let QueryOpening::Extension { values, proof } = opening else {
                return Err(BaseCaseZkError::SourceOpeningRejected { position });
            };
            extension_mmcs
                .verify_batch(
                    source_commitment,
                    &dims,
                    position,
                    BatchOpeningRef {
                        opened_values: from_ref(values),
                        opening_proof: proof,
                    },
                )
                .map_err(|_| BaseCaseZkError::SourceOpeningRejected { position })?;
            Ok(values[0])
        },
        &mut challenger,
    )
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(16))]

    #[test]
    fn prop_base_case_completeness(seed in any::<u64>(), num_masks in 0usize..=3) {
        // Honest standalone runs accept across mask counts and seeds,
        // with and without grinding.
        let pow_bits = (seed % 2) as usize * 4;
        let (config, mmcs, proof, w, u, commits, target, source, challenger) =
            honest_run(seed, num_masks, pow_bits, Tamper::None, None);
        prop_assert!(
            verify_run(&config, &mmcs, &proof, &w, &u, &commits, target, &source, challenger)
                .is_ok()
        );
    }
}

#[test]
fn base_case_rejects_wrong_target() {
    // The joint linear identity pins the carried target.
    let (config, mmcs, proof, w, u, commits, target, source, challenger) =
        honest_run(3, 2, 0, Tamper::None, None);
    let err = verify_run(
        &config,
        &mmcs,
        &proof,
        &w,
        &u,
        &commits,
        target + EF::ONE,
        &source,
        challenger,
    )
    .unwrap_err();
    assert_eq!(err, BaseCaseZkError::TargetCheckFailed);
}

#[test]
fn base_case_rejects_tampered_blinded_message() {
    // The shifted reveal enters the joint linear identity directly,
    // so the target check fires before any spot check runs.
    let (config, mmcs, mut proof, w, u, commits, target, source, challenger) =
        honest_run(4, 1, 0, Tamper::None, None);
    proof.blinded_message[0] += EF::ONE;
    let err = verify_run(
        &config, &mmcs, &proof, &w, &u, &commits, target, &source, challenger,
    )
    .unwrap_err();
    assert_eq!(err, BaseCaseZkError::TargetCheckFailed);
}

#[test]
fn base_case_rejects_tampered_blinded_randomness() {
    // Invariant: a randomness-reveal shift is caught after the target check.
    //
    //     target identity   ->  unaffected (randomness is not in it)
    //     reveal absorbed   ->  before the spot positions are drawn
    //     failure           ->  diverged openings or encoding equation
    let (config, mmcs, mut proof, w, u, commits, target, source, challenger) =
        honest_run(5, 1, 0, Tamper::None, None);
    proof.blinded_randomness[0] += EF::ONE;
    let err = verify_run(
        &config, &mmcs, &proof, &w, &u, &commits, target, &source, challenger,
    )
    .unwrap_err();
    // A randomness-reveal shift is absorbed before the spot positions are
    // drawn, so the transcript diverges and a source opening is rejected.
    // Whether it surfaces as a diverged Merkle opening or the encoding
    // spot check depends on the transcript, so only rejection is asserted.
    assert!(matches!(
        err,
        BaseCaseZkError::SourceOpeningRejected { .. }
            | BaseCaseZkError::SourceSpotCheckFailed { .. },
    ));
}

#[test]
fn base_case_rejects_tampered_mask_reveal() {
    // The shifted mask reveal enters the joint linear identity directly,
    // so the target check fires before any spot check runs.
    let (config, mmcs, mut proof, w, u, commits, target, source, challenger) =
        honest_run(6, 2, 0, Tamper::None, None);
    proof.blinded_masks[1].message[0] += EF::ONE;
    let err = verify_run(
        &config, &mmcs, &proof, &w, &u, &commits, target, &source, challenger,
    )
    .unwrap_err();
    assert_eq!(err, BaseCaseZkError::TargetCheckFailed);
}

#[test]
fn base_case_rejects_unbound_source_reveal() {
    // The committed source encodes a shifted message; the reveals and
    // target keep the original.
    //
    //     target check  ->  passes (reveals and target agree)
    //     source check  ->  Enc(f*, r*)(z) != g(z) + gamma * f(z)
    //
    // This is the only branch that ties f* to the committed source.
    let (config, mmcs, proof, w, u, commits, target, source, challenger) =
        honest_run(30, 1, 0, Tamper::SourceMessage, None);
    let err = verify_run(
        &config, &mmcs, &proof, &w, &u, &commits, target, &source, challenger,
    )
    .unwrap_err();
    // The committed source genuinely differs from the reveal, so the source
    // spot check always fails; the position it lands on is incidental.
    assert!(matches!(err, BaseCaseZkError::SourceSpotCheckFailed { .. },));
}

#[test]
fn base_case_rejects_unbound_mask_reveal() {
    // The committed mask encodes a shifted message; the reveals and
    // target keep the original.
    //
    //     target check  ->  passes (reveals and target agree)
    //     source check  ->  passes (source is honest here)
    //     mask check    ->  Enc(xi*, r*)(y) != s'(y) + gamma * xi(y)
    //
    // This is the only branch that ties xi* to the committed mask.
    let (config, mmcs, proof, w, u, commits, target, source, challenger) =
        honest_run(31, 1, 0, Tamper::MaskMessage, None);
    let err = verify_run(
        &config, &mmcs, &proof, &w, &u, &commits, target, &source, challenger,
    )
    .unwrap_err();
    // The committed mask genuinely differs from the reveal, so the mask
    // spot check always fails in group 0; the position is incidental.
    assert!(matches!(
        err,
        BaseCaseZkError::MaskSpotCheckFailed { group: 0, .. },
    ));
}

#[test]
fn base_case_reveals_are_one_time_padded() {
    // Invariant (Lemma 7.3): the source reveal is a one-time pad.
    //
    //     reveal = g~ + gamma * secret
    //
    // The fresh mask g~ and the challenge gamma do not depend on the secret.
    // Two runs share the fresh-mask stream but differ in the secret at
    // coordinate 0 only:
    //
    //     reveal_a - reveal_b = gamma * (secret_a - secret_b)
    //     recovered g~        = reveal - gamma * secret  (equal across runs)
    let seed = 7;
    let message_len = 8_usize;

    // Two secrets differing only at coordinate 0.
    let secret_a: Vec<EF> = (0..message_len).map(|j| EF::from_u64(j as u64)).collect();
    let mut secret_b = secret_a.clone();
    secret_b[0] += EF::ONE;

    let (_, _, proof_a, ..) = honest_run(seed, 1, 0, Tamper::None, Some(secret_a.clone()));
    let (_, _, proof_b, ..) = honest_run(seed, 1, 0, Tamper::None, Some(secret_b.clone()));

    // The fresh mask is witness-independent, so its commitment matches.
    assert_eq!(
        postcard::to_allocvec(&proof_a.fresh_main_commitment).unwrap(),
        postcard::to_allocvec(&proof_b.fresh_main_commitment).unwrap(),
    );

    // Recover gamma from the one coordinate where the secrets differ:
    //
    //     reveal_a[0] - reveal_b[0] = gamma * (secret_a[0] - secret_b[0])
    let gamma = (proof_a.blinded_message[0] - proof_b.blinded_message[0])
        * (secret_a[0] - secret_b[0]).inverse();
    assert!(!gamma.is_zero(), "the pad must mix the secret");

    // The pad is present: the reveal is not the bare secret.
    assert_ne!(proof_a.blinded_message, secret_a);

    // Both runs recover the same fresh mask g~ = reveal - gamma * secret,
    // which holds only if the reveal is exactly g~ + gamma * secret.
    for i in 0..message_len {
        let g_from_a = proof_a.blinded_message[i] - gamma * secret_a[i];
        let g_from_b = proof_b.blinded_message[i] - gamma * secret_b[i];
        assert_eq!(g_from_a, g_from_b, "recovered fresh mask differs at {i}");
    }
}
