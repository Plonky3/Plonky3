use alloc::vec;

use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::DuplexChallenger;
use p3_dft::Radix2DFTSmallBatch;
use p3_field::PrimeCharacteristicRing;
use p3_field::extension::BinomialExtensionField;
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use p3_whir::parameters::{
    FoldingFactor, ProtocolParameters, SecurityAssumption, SumcheckStrategy,
};
use rand::SeedableRng;
use rand::rngs::SmallRng;

use super::*;
use crate::code::ReedSolomonCode;
use crate::finalize::WhirLimbAccumulatorBackend;
use crate::protocol::AccumulatorCommitmentBackend;
use crate::root_iop::{
    RootIopOpeningClaim, RootIopOpeningPoint, RootIopOpeningValue, RootIopOracleField,
};

type F = BabyBear;
type EF = BinomialExtensionField<F, 4>;
type Dft = Radix2DFTSmallBatch<F>;
type Perm = Poseidon2BabyBear<16>;
type TestChallenger = DuplexChallenger<F, Perm, 16, 8>;
type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
type PackedF = <F as Field>::Packing;
type MyMmcs = MerkleTreeMmcs<PackedF, PackedF, MyHash, MyCompress, 2, 8>;
type TestWhirPcs = WhirPcs<EF, F, MyMmcs, TestChallenger, Dft, 8>;

fn challenger() -> TestChallenger {
    let perm = Perm::new_from_rng_128(&mut SmallRng::seed_from_u64(2));
    DuplexChallenger::new(perm)
}

fn systematic_code() -> ReedSolomonCode<F, Dft> {
    ReedSolomonCode::new_systematic(2, 1, Dft::default())
}

fn coefficient_code() -> ReedSolomonCode<F, Dft> {
    ReedSolomonCode::new_coefficient(2, 1, Dft::default())
}

fn whir_pcs(num_variables: usize) -> TestWhirPcs {
    let perm = Perm::new_from_rng_128(&mut SmallRng::seed_from_u64(3));
    let mmcs = MyMmcs::new(MyHash::new(perm.clone()), MyCompress::new(perm), 0);
    let params = ProtocolParameters {
        security_level: 32,
        pow_bits: 0,
        rs_domain_initial_reduction_factor: 1,
        folding_factor: FoldingFactor::Constant(2),
        mmcs,
        soundness_type: SecurityAssumption::CapacityBound,
        starting_log_inv_rate: 1,
    };
    TestWhirPcs::new(
        num_variables,
        params,
        Dft::default(),
        SumcheckStrategy::Classic,
    )
}

#[test]
fn folded_oracle_eval_claim_becomes_linear_sigma() {
    let code = systematic_code();
    let compiler = NativeWarpWhirCompiler::new(&code);
    let witness = vec![
        F::from_u64(1),
        F::from_u64(2),
        F::from_u64(3),
        F::from_u64(4),
    ];
    let codeword = code.encode(&witness);
    let codeword_poly = Poly::new(codeword);
    let point = Point::new(vec![EF::from_u64(2), EF::from_u64(3), EF::from_u64(5)]);
    let value = codeword_poly.eval_base(&point);

    let claim = NativeWarpWhirEvalClaim::new(point, value);
    let constraint = compiler.eval_claim_constraint(&claim);

    assert!(constraint.verify_base(&codeword_poly));
}

#[test]
fn compiled_eval_claims_reduce_to_one_opening() {
    let code = systematic_code();
    let compiler = NativeWarpWhirCompiler::new(&code);
    let witness = vec![
        F::from_u64(1),
        F::from_u64(2),
        F::from_u64(3),
        F::from_u64(4),
    ];
    let codeword_poly = Poly::new(code.encode(&witness));
    let point0 = Point::new(vec![EF::from_u64(2), EF::from_u64(3), EF::from_u64(5)]);
    let point1 = Point::new(vec![EF::from_u64(7), EF::from_u64(11), EF::from_u64(13)]);
    let claims = eval_claims_from_parts(
        &[point0.clone(), point1.clone()],
        &[
            codeword_poly.eval_base(&point0),
            codeword_poly.eval_base(&point1),
        ],
    );
    let statement = compiler.eval_claim_statement(&claims);

    let mut prover_challenger = challenger();
    let mut verifier_challenger = challenger();
    let (proof, opening) = statement
        .prove_reduction_base::<F, _>(&codeword_poly, &mut prover_challenger, 0)
        .expect("honest WARP/WHIR reduction");
    let verified_opening = statement
        .verify_reduction::<F, _>(&proof, &mut verifier_challenger, 0)
        .expect("WARP/WHIR reduction verification");

    assert_eq!(opening, verified_opening);
    assert_eq!(opening.value, codeword_poly.eval_base(&opening.point));
}

#[test]
fn compiled_wrong_eval_claim_does_not_prove() {
    let code = systematic_code();
    let compiler = NativeWarpWhirCompiler::new(&code);
    let witness = vec![
        F::from_u64(1),
        F::from_u64(2),
        F::from_u64(3),
        F::from_u64(4),
    ];
    let codeword_poly = Poly::new(code.encode(&witness));
    let point = Point::new(vec![EF::from_u64(2), EF::from_u64(3), EF::from_u64(5)]);
    let bad_claim =
        NativeWarpWhirEvalClaim::new(point.clone(), codeword_poly.eval_base(&point) + EF::ONE);
    let statement = compiler.eval_claim_statement(&[bad_claim]);

    let err = statement
        .prove_reduction_base::<F, _>(&codeword_poly, &mut challenger(), 0)
        .expect_err("wrong claim should not produce an honest proof");
    assert!(matches!(
        err,
        LinearSigmaReductionError::UnsatisfiedStatement
    ));
}

#[test]
fn coefficient_rs_index_claim_uses_same_whir_initial_polynomial() {
    let code = coefficient_code();
    let compiler = NativeWarpWhirCompiler::new(&code);
    let witness = vec![
        F::from_u64(1),
        F::from_u64(2),
        F::from_u64(3),
        F::from_u64(4),
    ];
    let codeword = code.encode(&witness);
    let index = 5;
    let claim = RootIopOpeningClaim {
        claim_id: 0,
        oracle_id: 0,
        point: RootIopOpeningPoint::RsCodewordIndex(index),
        value: RootIopOpeningValue::Base(codeword[index]),
    };

    let statement = compiler
        .root_iop_base_message_claim_statement::<EF>(&[claim], 0)
        .expect("coefficient RS index claim compiles");

    let message_poly = Poly::new(witness);
    let mut prover_challenger = challenger();
    let mut verifier_challenger = challenger();
    let (proof, opening) = statement
        .prove_reduction_base::<F, _>(&message_poly, &mut prover_challenger, 0)
        .expect("honest coefficient RS-index reduction");
    let verified_opening = statement
        .verify_reduction::<F, _>(&proof, &mut verifier_challenger, 0)
        .expect("coefficient RS-index reduction verification");

    assert_eq!(opening, verified_opening);
    assert_eq!(opening.value, message_poly.eval_base(&opening.point));
}

#[test]
fn compiled_eval_claims_bind_to_whir_commitment() {
    let code = systematic_code();
    let compiler = NativeWarpWhirCompiler::new(&code);
    let witness = vec![
        F::from_u64(5),
        F::from_u64(7),
        F::from_u64(11),
        F::from_u64(13),
    ];
    let codeword = code.encode(&witness);
    let codeword_poly = Poly::new(codeword.clone());
    let point0 = Point::new(vec![EF::from_u64(2), EF::from_u64(3), EF::from_u64(5)]);
    let point1 = Point::new(vec![EF::from_u64(7), EF::from_u64(11), EF::from_u64(13)]);
    let claims = eval_claims_from_parts(
        &[point0.clone(), point1.clone()],
        &[
            codeword_poly.eval_base(&point0),
            codeword_poly.eval_base(&point1),
        ],
    );
    let statement = compiler.eval_claim_statement(&claims);
    let pcs = whir_pcs(code.log_codeword_len());

    let mut prover_challenger = challenger();
    let (commitment, prover_data) =
        pcs.commit_deferred(RowMajorMatrix::new(codeword, 1), &mut prover_challenger);
    let (opening, proof) = statement
        .prove_bound_deferred(&pcs, prover_data, &mut prover_challenger, 0)
        .expect("bound WARP/WHIR proof");

    let mut verifier_challenger = challenger();
    let verified_opening = statement
        .verify_bound_deferred(&pcs, &commitment, &proof, &mut verifier_challenger, 0)
        .expect("bound WARP/WHIR verification");

    assert_eq!(opening, verified_opening);
    assert_eq!(opening.value, codeword_poly.eval_base(&opening.point));
}

#[test]
fn root_iop_index_claim_compiles_with_warp_bit_order() {
    let code = systematic_code();
    let compiler = NativeWarpWhirCompiler::new(&code);
    let witness = vec![
        F::from_u64(1),
        F::from_u64(2),
        F::from_u64(3),
        F::from_u64(4),
    ];
    let codeword = code.encode(&witness);
    let index = 5;
    let claims: Vec<RootIopOpeningClaim<F, EF>> = vec![RootIopOpeningClaim {
        claim_id: 0,
        oracle_id: 9,
        point: RootIopOpeningPoint::<EF>::Index(index),
        value: RootIopOpeningValue::Base(codeword[index]),
    }];

    let statement = compiler
        .root_iop_claim_statement(&claims, 9, RootIopOracleField::Base)
        .expect("root-IOP base claim statement");

    assert!(statement.constraints.verify_base(&Poly::new(codeword)));
}

#[test]
fn root_iop_extension_mle_claim_compiles() {
    let code = systematic_code();
    let compiler = NativeWarpWhirCompiler::new(&code);
    let oracle = (0..code.codeword_len())
        .map(|i| EF::from_u64((3 * i + 5) as u64))
        .collect::<Vec<_>>();
    let poly = Poly::new(oracle.clone());
    let point = Point::new(vec![EF::from_u64(2), EF::from_u64(3), EF::from_u64(5)]);
    let value = poly.eval_ext::<F>(&point);
    let claims: Vec<RootIopOpeningClaim<F, EF>> = vec![RootIopOpeningClaim {
        claim_id: 0,
        oracle_id: 3,
        point: RootIopOpeningPoint::Mle(point.as_slice().to_vec()),
        value: RootIopOpeningValue::Extension(value),
    }];

    let statement = compiler
        .root_iop_claim_statement(&claims, 3, RootIopOracleField::Extension)
        .expect("root-IOP extension claim statement");

    assert!(statement.constraints.verify_ext(&poly));
}

#[test]
fn compiled_extension_claims_bind_to_whir_limb_backend() {
    let code = systematic_code();
    let compiler = NativeWarpWhirCompiler::new(&code);
    let oracle = (0..code.codeword_len())
        .map(|i| EF::from_u64((11 * i + 7) as u64))
        .collect::<Vec<_>>();
    let poly = Poly::new(oracle.clone());
    let point0 = Point::new(vec![EF::from_u64(2), EF::from_u64(3), EF::from_u64(5)]);
    let point1 = Point::new(vec![EF::from_u64(7), EF::from_u64(11), EF::from_u64(13)]);
    let claims = eval_claims_from_parts(
        &[point0.clone(), point1.clone()],
        &[poly.eval_ext::<F>(&point0), poly.eval_ext::<F>(&point1)],
    );
    let statement = compiler.eval_claim_statement(&claims);
    let pcs = whir_pcs(code.log_codeword_len());
    let backend =
        WhirLimbAccumulatorBackend::<F, EF, _, TestChallenger, Dft, 8>::new(&pcs, challenger());
    let (commitment, prover_data) = backend
        .commit(oracle.clone())
        .expect("WHIR limb accumulator commit");

    let mut prover_challenger = challenger();
    let (opening, proof) = statement
        .prove_bound_extension_points::<F, _, _>(
            &backend,
            &commitment,
            &prover_data,
            &oracle,
            &mut prover_challenger,
            0,
        )
        .expect("bound extension WARP/WHIR proof");

    let mut verifier_challenger = challenger();
    let verified_opening = statement
        .verify_bound_extension_points::<F, _, _>(
            &backend,
            &commitment,
            &proof,
            &mut verifier_challenger,
            0,
        )
        .expect("bound extension WARP/WHIR verification");

    assert_eq!(opening, verified_opening);
    assert_eq!(opening.value, poly.eval_ext::<F>(&opening.point));
}

#[test]
fn compiled_extension_claims_reject_tampered_reduction() {
    let code = systematic_code();
    let compiler = NativeWarpWhirCompiler::new(&code);
    let oracle = (0..code.codeword_len())
        .map(|i| EF::from_u64((13 * i + 17) as u64))
        .collect::<Vec<_>>();
    let poly = Poly::new(oracle.clone());
    let point = Point::new(vec![EF::from_u64(2), EF::from_u64(3), EF::from_u64(5)]);
    let statement = compiler.eval_claim_statement(&[NativeWarpWhirEvalClaim::new(
        point.clone(),
        poly.eval_ext::<F>(&point),
    )]);
    let pcs = whir_pcs(code.log_codeword_len());
    let backend =
        WhirLimbAccumulatorBackend::<F, EF, _, TestChallenger, Dft, 8>::new(&pcs, challenger());
    let (commitment, prover_data) = backend
        .commit(oracle.clone())
        .expect("WHIR limb accumulator commit");
    let (_, mut proof) = statement
        .prove_bound_extension_points::<F, _, _>(
            &backend,
            &commitment,
            &prover_data,
            &oracle,
            &mut challenger(),
            0,
        )
        .expect("bound extension proof");
    proof.reduction.oracle_eval += EF::ONE;

    let err = statement
        .verify_bound_extension_points::<F, _, _>(
            &backend,
            &commitment,
            &proof,
            &mut challenger(),
            0,
        )
        .expect_err("tampered reduction should be rejected");
    assert!(matches!(
        err,
        NativeWarpWhirCompilerError::Reduction(LinearSigmaReductionError::FinalCheckFailed)
    ));
}

#[test]
fn root_iop_transcript_claims_reduce_in_commitment_order() {
    let code = systematic_code();
    let compiler = NativeWarpWhirCompiler::new(&code);
    let base_witness = vec![
        F::from_u64(1),
        F::from_u64(2),
        F::from_u64(3),
        F::from_u64(4),
    ];
    let base_oracle = code.encode(&base_witness);
    let ext_oracle = (0..code.codeword_len())
        .map(|i| EF::from_u64((19 * i + 23) as u64))
        .collect::<Vec<_>>();
    let ext_poly = Poly::new(ext_oracle.clone());
    let ext_point = Point::new(vec![EF::from_u64(2), EF::from_u64(5), EF::from_u64(7)]);
    let commitments = vec![
        RootIopBoundCommitment {
            oracle_id: 0,
            log_len: code.log_codeword_len(),
            field: RootIopOracleField::Base,
            commitment: F::from_u64(101),
        },
        RootIopBoundCommitment {
            oracle_id: 1,
            log_len: code.log_codeword_len(),
            field: RootIopOracleField::Extension,
            commitment: F::from_u64(202),
        },
    ];
    let claims = vec![
        RootIopOpeningClaim {
            claim_id: 0,
            oracle_id: 0,
            point: RootIopOpeningPoint::Index(5),
            value: RootIopOpeningValue::Base(base_oracle[5]),
        },
        RootIopOpeningClaim {
            claim_id: 1,
            oracle_id: 1,
            point: RootIopOpeningPoint::Mle(ext_point.as_slice().to_vec()),
            value: RootIopOpeningValue::Extension(ext_poly.eval_ext::<F>(&ext_point)),
        },
    ];
    let transcript = RootIopBoundTranscript {
        oracles: vec![
            (
                commitments[0].clone(),
                RootIopOracleValues::Base(base_oracle),
            ),
            (
                commitments[1].clone(),
                RootIopOracleValues::Extension(ext_oracle),
            ),
        ],
        claims: claims.clone(),
    };

    let (prover_residuals, proof) = compiler
        .prove_root_iop_reductions(&transcript, &mut challenger(), 0)
        .expect("honest root-IOP reductions");
    let verifier_residuals = compiler
        .verify_root_iop_reductions(&commitments, &claims, &proof, &mut challenger(), 0)
        .expect("root-IOP reduction verification");

    assert_eq!(prover_residuals, verifier_residuals);
    assert_eq!(proof.oracles.len(), 2);
    assert_eq!(proof.oracles[0].oracle_id, 0);
    assert_eq!(proof.oracles[1].oracle_id, 1);
}

#[test]
fn root_iop_reductions_reject_tampered_public_claim() {
    let code = systematic_code();
    let compiler = NativeWarpWhirCompiler::new(&code);
    let witness = vec![
        F::from_u64(5),
        F::from_u64(7),
        F::from_u64(11),
        F::from_u64(13),
    ];
    let oracle = code.encode(&witness);
    let commitment = RootIopBoundCommitment {
        oracle_id: 0,
        log_len: code.log_codeword_len(),
        field: RootIopOracleField::Base,
        commitment: F::from_u64(303),
    };
    let claims = vec![RootIopOpeningClaim {
        claim_id: 0,
        oracle_id: 0,
        point: RootIopOpeningPoint::<EF>::Index(6),
        value: RootIopOpeningValue::Base(oracle[6]),
    }];
    let transcript = RootIopBoundTranscript {
        oracles: vec![(commitment.clone(), RootIopOracleValues::Base(oracle))],
        claims: claims.clone(),
    };
    let (_, proof) = compiler
        .prove_root_iop_reductions(&transcript, &mut challenger(), 0)
        .expect("honest root-IOP reduction");
    let mut tampered_claims = claims;
    tampered_claims[0].value = RootIopOpeningValue::Base(F::from_u64(999));

    assert!(
        compiler
            .verify_root_iop_reductions(
                &[commitment],
                &tampered_claims,
                &proof,
                &mut challenger(),
                0
            )
            .is_err()
    );
}

#[test]
fn root_iop_residuals_bind_to_whir_openings() {
    let code = systematic_code();
    let pcs = whir_pcs(code.log_codeword_len());
    let root_system = NativeWarpWhirRootProofSystem::new(&pcs, &code, challenger());
    let base_witness = vec![
        F::from_u64(1),
        F::from_u64(4),
        F::from_u64(9),
        F::from_u64(16),
    ];
    let base_oracle = code.encode(&base_witness);
    let ext_oracle = (0..code.codeword_len())
        .map(|i| EF::from_u64((29 * i + 31) as u64))
        .collect::<Vec<_>>();
    let ext_poly = Poly::new(ext_oracle.clone());
    let ext_point = Point::new(vec![EF::from_u64(3), EF::from_u64(5), EF::from_u64(11)]);
    let (base_commitment, base_prover_data) = root_system
        .commit_base_oracle(0, base_oracle.clone())
        .expect("base root oracle commit");
    let (extension_commitment, extension_prover_data) = root_system
        .commit_extension_oracle(1, ext_oracle.clone())
        .expect("extension root oracle commit");
    let commitments = vec![base_commitment.clone(), extension_commitment.clone()];
    let claims = vec![
        RootIopOpeningClaim {
            claim_id: 0,
            oracle_id: 0,
            point: RootIopOpeningPoint::<EF>::Index(3),
            value: RootIopOpeningValue::Base(base_oracle[3]),
        },
        RootIopOpeningClaim {
            claim_id: 1,
            oracle_id: 1,
            point: RootIopOpeningPoint::Mle(ext_point.as_slice().to_vec()),
            value: RootIopOpeningValue::Extension(ext_poly.eval_ext::<F>(&ext_point)),
        },
    ];
    let transcript = RootIopBoundTranscript {
        oracles: vec![
            (base_commitment, RootIopOracleValues::Base(base_oracle)),
            (
                extension_commitment,
                RootIopOracleValues::Extension(ext_oracle),
            ),
        ],
        claims: claims.clone(),
    };
    let proof = root_system
        .prove(
            &transcript,
            &[base_prover_data, extension_prover_data],
            &mut challenger(),
            0,
        )
        .expect("WHIR-bound root proof");

    let residuals = root_system
        .verify(&commitments, &claims, &proof, &mut challenger(), 0)
        .expect("WHIR-bound root proof verification");

    assert_eq!(residuals.len(), 2);
    assert_eq!(proof.reductions.oracles.len(), 2);
    assert_eq!(proof.openings.len(), 2);
}

#[test]
fn root_iop_whir_bound_proof_rejects_tampered_claim() {
    let code = systematic_code();
    let pcs = whir_pcs(code.log_codeword_len());
    let root_system = NativeWarpWhirRootProofSystem::new(&pcs, &code, challenger());
    let witness = vec![
        F::from_u64(2),
        F::from_u64(3),
        F::from_u64(5),
        F::from_u64(7),
    ];
    let oracle = code.encode(&witness);
    let (commitment, prover_data) = root_system
        .commit_base_oracle(0, oracle.clone())
        .expect("base root oracle commit");
    let claims = vec![RootIopOpeningClaim {
        claim_id: 0,
        oracle_id: 0,
        point: RootIopOpeningPoint::<EF>::Index(4),
        value: RootIopOpeningValue::Base(oracle[4]),
    }];
    let transcript = RootIopBoundTranscript {
        oracles: vec![(commitment.clone(), RootIopOracleValues::Base(oracle))],
        claims: claims.clone(),
    };
    let proof = root_system
        .prove(&transcript, &[prover_data], &mut challenger(), 0)
        .expect("WHIR-bound root proof");
    let mut tampered_claims = claims;
    tampered_claims[0].value = RootIopOpeningValue::Base(F::from_u64(1234));

    assert!(
        root_system
            .verify(
                &[commitment],
                &tampered_claims,
                &proof,
                &mut challenger(),
                0
            )
            .is_err()
    );
}

#[test]
fn message_domain_root_proof_batches_residual_openings() {
    let code = systematic_code();
    let pcs = whir_pcs(code.log_codeword_len());
    let message_pcs = whir_pcs(code.log_msg_len());
    let root_system = NativeWarpWhirRootProofSystem::new_with_base_message_pcs(
        &pcs,
        &message_pcs,
        &code,
        challenger(),
    );
    let base_message = vec![
        F::from_u64(2),
        F::from_u64(5),
        F::from_u64(8),
        F::from_u64(13),
    ];
    let base_codeword = code.encode(&base_message);
    let extension_message = (0..code.msg_len())
        .map(|i| EF::from_u64((17 * i + 19) as u64))
        .collect::<Vec<_>>();
    let extension_codeword = code.encode_algebra(&extension_message);
    let extension_poly = Poly::new(extension_codeword.clone());
    let extension_point = Point::new(vec![EF::from_u64(3), EF::from_u64(7), EF::from_u64(11)]);

    let (base_commitment, base_prover_data) = root_system
        .commit_base_message_oracle(0, base_codeword.clone(), base_message)
        .expect("base message root oracle commit");
    let (extension_commitment, extension_prover_data) = root_system
        .commit_extension_oracle(1, extension_codeword.clone())
        .expect("extension message root oracle commit");
    let commitments = vec![base_commitment.clone(), extension_commitment.clone()];
    let claims = vec![
        RootIopOpeningClaim {
            claim_id: 0,
            oracle_id: 0,
            point: RootIopOpeningPoint::<EF>::Index(4),
            value: RootIopOpeningValue::Base(base_codeword[4]),
        },
        RootIopOpeningClaim {
            claim_id: 1,
            oracle_id: 1,
            point: RootIopOpeningPoint::Mle(extension_point.as_slice().to_vec()),
            value: RootIopOpeningValue::Extension(extension_poly.eval_ext::<F>(&extension_point)),
        },
    ];
    let transcript = RootIopBoundTranscript {
        oracles: vec![
            (base_commitment, RootIopOracleValues::Base(base_codeword)),
            (
                extension_commitment,
                RootIopOracleValues::Extension(extension_codeword),
            ),
        ],
        claims: claims.clone(),
    };
    let proof = root_system
        .prove(
            &transcript,
            &[base_prover_data, extension_prover_data],
            &mut challenger(),
            0,
        )
        .expect("batched WHIR-bound root proof");

    assert!(proof.openings.is_empty());
    assert!(proof.batched_opening.is_none());
    assert!(proof.direct_batched_opening.is_some());
    let residuals = root_system
        .verify(&commitments, &claims, &proof, &mut challenger(), 0)
        .expect("batched WHIR-bound root proof verification");
    assert!(residuals.is_empty());
}

#[test]
fn message_domain_batched_root_proof_rejects_missing_claim_or_swapped_commitment() {
    let code = systematic_code();
    let pcs = whir_pcs(code.log_codeword_len());
    let message_pcs = whir_pcs(code.log_msg_len());
    let root_system = NativeWarpWhirRootProofSystem::new_with_base_message_pcs(
        &pcs,
        &message_pcs,
        &code,
        challenger(),
    );
    let base_message = vec![
        F::from_u64(2),
        F::from_u64(5),
        F::from_u64(8),
        F::from_u64(13),
    ];
    let base_codeword = code.encode(&base_message);
    let extension_message = (0..code.msg_len())
        .map(|i| EF::from_u64((17 * i + 19) as u64))
        .collect::<Vec<_>>();
    let extension_codeword = code.encode_algebra(&extension_message);

    let (base_commitment, base_prover_data) = root_system
        .commit_base_message_oracle(0, base_codeword.clone(), base_message)
        .expect("base message root oracle commit");
    let (extension_commitment, extension_prover_data) = root_system
        .commit_extension_oracle(1, extension_codeword.clone())
        .expect("extension message root oracle commit");
    let commitments = vec![base_commitment.clone(), extension_commitment.clone()];
    let claims = vec![
        RootIopOpeningClaim {
            claim_id: 0,
            oracle_id: 0,
            point: RootIopOpeningPoint::<EF>::Index(4),
            value: RootIopOpeningValue::Base(base_codeword[4]),
        },
        RootIopOpeningClaim {
            claim_id: 1,
            oracle_id: 1,
            point: RootIopOpeningPoint::<EF>::Index(5),
            value: RootIopOpeningValue::Extension(extension_codeword[5]),
        },
    ];
    let transcript = RootIopBoundTranscript {
        oracles: vec![
            (base_commitment, RootIopOracleValues::Base(base_codeword)),
            (
                extension_commitment,
                RootIopOracleValues::Extension(extension_codeword),
            ),
        ],
        claims: claims.clone(),
    };
    let proof = root_system
        .prove(
            &transcript,
            &[base_prover_data, extension_prover_data],
            &mut challenger(),
            0,
        )
        .expect("batched WHIR-bound root proof");

    let missing_claim = vec![claims[0].clone()];
    assert!(
        root_system
            .verify(&commitments, &missing_claim, &proof, &mut challenger(), 0)
            .is_err(),
        "dropping one recorded WARP root claim must be rejected"
    );

    let mut swapped_commitments = commitments.clone();
    swapped_commitments.swap(0, 1);
    assert!(
        root_system
            .verify(&swapped_commitments, &claims, &proof, &mut challenger(), 0)
            .is_err(),
        "changing root commitment order must be rejected"
    );

    let mut wrong_challenger = challenger();
    wrong_challenger.observe(F::ONE);
    assert!(
        root_system
            .verify(&commitments, &claims, &proof, &mut wrong_challenger, 0)
            .is_err(),
        "root proof must be bound to the Fiat-Shamir state"
    );
}

#[test]
fn shared_message_root_proof_binds_columns() {
    let code = systematic_code();
    let pcs = whir_pcs(code.log_codeword_len());
    let message_pcs = whir_pcs(code.log_msg_len());
    let root_system = NativeWarpWhirRootProofSystem::new_with_base_message_pcs(
        &pcs,
        &message_pcs,
        &code,
        challenger(),
    );
    let message0 = vec![
        F::from_u64(2),
        F::from_u64(3),
        F::from_u64(5),
        F::from_u64(7),
    ];
    let message1 = vec![
        F::from_u64(11),
        F::from_u64(13),
        F::from_u64(17),
        F::from_u64(19),
    ];
    let codeword0 = code.encode(&message0);
    let codeword1 = code.encode(&message1);
    let committed = root_system
        .commit_shared_base_message_oracles(vec![
            (0, codeword0.clone(), message0),
            (1, codeword1.clone(), message1),
        ])
        .expect("shared base message root oracle commit");
    let mut commitments = Vec::new();
    let mut prover_data = Vec::new();
    for (commitment, data) in committed {
        commitments.push(commitment);
        prover_data.push(data);
    }
    let claims = vec![
        RootIopOpeningClaim {
            claim_id: 0,
            oracle_id: 0,
            point: RootIopOpeningPoint::<EF>::Index(2),
            value: RootIopOpeningValue::Base(codeword0[2]),
        },
        RootIopOpeningClaim {
            claim_id: 1,
            oracle_id: 1,
            point: RootIopOpeningPoint::<EF>::Index(5),
            value: RootIopOpeningValue::Base(codeword1[5]),
        },
    ];
    let transcript = RootIopBoundTranscript {
        oracles: vec![
            (
                commitments[0].clone(),
                RootIopOracleValues::Base(codeword0.clone()),
            ),
            (
                commitments[1].clone(),
                RootIopOracleValues::Base(codeword1.clone()),
            ),
        ],
        claims: claims.clone(),
    };
    let proof = root_system
        .prove(&transcript, &prover_data, &mut challenger(), 0)
        .expect("shared WHIR-bound root proof");

    assert!(proof.openings.is_empty());
    assert!(proof.batched_opening.is_none());
    assert!(proof.direct_batched_opening.is_some());
    root_system
        .verify(&commitments, &claims, &proof, &mut challenger(), 0)
        .expect("shared WHIR-bound root proof verification");

    let mut malformed_commitments = commitments;
    if let NativeWarpWhirRootCommitment::BaseMessageShared { column, width, .. } =
        &mut malformed_commitments[1].commitment
    {
        *column = *width;
    } else {
        panic!("expected shared base commitment");
    }
    assert!(
        root_system
            .verify(
                &malformed_commitments,
                &claims,
                &proof,
                &mut challenger(),
                0
            )
            .is_err()
    );
}

#[test]
fn message_domain_batched_root_proof_rejects_tampered_virtual_eval() {
    let code = systematic_code();
    let pcs = whir_pcs(code.log_codeword_len());
    let message_pcs = whir_pcs(code.log_msg_len());
    let root_system = NativeWarpWhirRootProofSystem::new_with_base_message_pcs(
        &pcs,
        &message_pcs,
        &code,
        challenger(),
    );
    let base_message = vec![
        F::from_u64(3),
        F::from_u64(6),
        F::from_u64(10),
        F::from_u64(15),
    ];
    let base_codeword = code.encode(&base_message);
    let extension_message = (0..code.msg_len())
        .map(|i| EF::from_u64((23 * i + 29) as u64))
        .collect::<Vec<_>>();
    let extension_codeword = code.encode_algebra(&extension_message);
    let (base_commitment, base_prover_data) = root_system
        .commit_base_message_oracle(0, base_codeword.clone(), base_message)
        .expect("base message root oracle commit");
    let (extension_commitment, extension_prover_data) = root_system
        .commit_extension_oracle(1, extension_codeword.clone())
        .expect("extension message root oracle commit");
    let commitments = vec![base_commitment.clone(), extension_commitment.clone()];
    let claims = vec![
        RootIopOpeningClaim {
            claim_id: 0,
            oracle_id: 0,
            point: RootIopOpeningPoint::<EF>::Index(2),
            value: RootIopOpeningValue::Base(base_codeword[2]),
        },
        RootIopOpeningClaim {
            claim_id: 1,
            oracle_id: 1,
            point: RootIopOpeningPoint::<EF>::Index(5),
            value: RootIopOpeningValue::Extension(extension_codeword[5]),
        },
    ];
    let transcript = RootIopBoundTranscript {
        oracles: vec![
            (base_commitment, RootIopOracleValues::Base(base_codeword)),
            (
                extension_commitment,
                RootIopOracleValues::Extension(extension_codeword),
            ),
        ],
        claims: claims.clone(),
    };
    let mut proof = root_system
        .prove(
            &transcript,
            &[base_prover_data, extension_prover_data],
            &mut challenger(),
            0,
        )
        .expect("batched WHIR-bound root proof");
    proof
        .direct_batched_opening
        .as_mut()
        .expect("batched opening")
        .reduction
        .virtual_eval += EF::ONE;

    assert!(
        root_system
            .verify(&commitments, &claims, &proof, &mut challenger(), 0)
            .is_err()
    );
}

#[test]
fn systematic_message_claim_lifts_to_codeword_subspace() {
    let code = systematic_code();
    let compiler = NativeWarpWhirCompiler::new(&code);
    let witness = vec![
        F::from_u64(5),
        F::from_u64(7),
        F::from_u64(11),
        F::from_u64(13),
    ];
    let codeword = code.encode(&witness);
    let witness_poly = Poly::new(witness);
    let codeword_poly = Poly::new(codeword);
    let message_point = vec![EF::from_u64(17), EF::from_u64(19)];
    let value = witness_poly.eval_base(&Point::new(message_point.clone()));

    let constraint = compiler.systematic_message_eval_constraint(&message_point, value);

    assert!(constraint.verify_base(&codeword_poly));
}
