//! Tests for the HVZK code-switching round (Construction 9.7).
//!
//! These tests validate the mathematical invariants of Construction 9.7
//! from eprint 2026/391 using hand-constructed data over BabyBear.
//!
//! These tests use the real `p3-zk-codes` (`LinearZkEncoding`) and
//! `whir::utils` zero-evader APIs instead of hand-rolled stubs.

use alloc::vec;
use alloc::vec::Vec;

use p3_baby_bear::BabyBear;
use p3_dft::Radix2Dit;
use p3_field::extension::BinomialExtensionField;
use p3_field::{Field, PrimeCharacteristicRing};
use p3_zk_codes::{LinearZkEncoding, ReedSolomonZkEncoding, ZkEncoding, ZkEncodingWithRandomness};
use proptest::prelude::*;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

use super::{
    CodeSwitchError, RoundZkConfig, ZkMaskClaim, batched_claim, batching_coefficients,
    output_relation, private_ood_answer, private_ood_answers, simulated_verifier_view,
};
use crate::sumcheck::zk::test_helpers::{MyChallenger, MyMmcs, make_setup};
use crate::sumcheck::zk::{ZkVerifier, simulate_classic_unpacked};
use crate::utils::eval_ze_star_n;

type F = BabyBear;
type EF = BinomialExtensionField<F, 4>;

fn ef(v: u64) -> EF {
    EF::from(F::from_u64(v))
}

fn inner_product(a: &[EF], b: &[EF]) -> EF {
    a.iter().zip(b.iter()).map(|(x, y)| *x * *y).sum()
}

fn make_rs_encoding(msg_len: usize, t: usize, m: usize) -> ReedSolomonZkEncoding<F, Radix2Dit<F>> {
    ReedSolomonZkEncoding::new(t, msg_len, m, Radix2Dit::default())
}

fn solve_two_randomness_for_openings(
    encoding: &ReedSolomonZkEncoding<EF, Radix2Dit<EF>>,
    message: &[EF],
    positions: &[usize; 2],
    openings: &[EF],
) -> Vec<EF> {
    assert_eq!(openings.len(), 2);

    let delta = |index: usize| {
        let position = positions[index];
        let message_row = encoding.message_row(position);
        let message_value = inner_product(message, &message_row);
        let randomness_row = encoding.randomness_row(position);
        assert_eq!(randomness_row.len(), 2);
        (randomness_row, openings[index] - message_value)
    };
    let (row0, rhs0) = delta(0);
    let (row1, rhs1) = delta(1);

    let a = row0[0];
    let b = row0[1];
    let c = row1[0];
    let d = row1[1];
    let det = a * d - b * c;
    assert!(!det.is_zero(), "RS query rows should be independent");

    vec![(rhs0 * d - b * rhs1) / det, (a * rhs1 - rhs0 * c) / det]
}

fn solve_pad_for_private_ood(rho: EF, message: &[EF], source_randomness: &[EF], target: EF) -> EF {
    assert!(!rho.is_zero(), "programming formula requires nonzero rho");

    let message_eval = eval_ze_star_n(rho, message);
    let source_randomness_eval = eval_ze_star_n(rho, source_randomness);
    let shift = rho.exp_u64(message.len() as u64);
    let pad_scale = shift * rho.exp_u64(source_randomness.len() as u64);
    assert!(
        !pad_scale.is_zero(),
        "programming formula requires a nonzero pad coefficient",
    );

    (target - message_eval - shift * source_randomness_eval) / pad_scale
}

/// Lift a base-field row into extension-field elements for inner products.
fn lift_row(row: &[F]) -> Vec<EF> {
    row.iter().map(|&x| EF::from(x)).collect()
}

/// Construction 9.7 `mu'` identity test with `n = 0` (no prior auxiliary masks).
///
/// Uses `ReedSolomonZkEncoding` with `msg_len=4, t=2, m=8` and validates
/// the completeness equation using `LinearZkEncoding::message_row` /
/// `randomness_row` for `G^#` / `G^$` and `eval_ze_star_n` for OOD answers.
#[test]
fn test_construction_9_7_mu_prime_identity_n0() {
    let ell = 4;
    let r_len = 2;
    let s_pad_len = 1;
    let t_ood = 2;
    let t = 3;
    let iota = 1;
    let m = 8;

    let source_enc = make_rs_encoding(ell, r_len, m);

    let f: Vec<EF> = (1..=ell as u64).map(ef).collect();
    let r: Vec<EF> = (10..10 + r_len as u64).map(ef).collect();
    let s_pad: Vec<EF> = (20..20 + s_pad_len as u64).map(ef).collect();

    let f_base: Vec<F> = (1..=ell as u64).map(F::from_u64).collect();
    let r_base: Vec<F> = (10..10 + r_len as u64).map(F::from_u64).collect();
    let cw = source_enc.encode_with_randomness(&f_base, &r_base);
    let f_codeword: Vec<EF> = cw.values.iter().map(|&v| EF::from(v)).collect();

    let sl: Vec<EF> = (1..=ell as u64).map(|i| ef(i * 100)).collect();
    let mu = inner_product(&f, &sl);

    let rho_ood_points: Vec<EF> = (0..t_ood).map(|i| ef(50 + i as u64)).collect();

    let mut f_r_s: Vec<EF> = Vec::with_capacity(ell + r_len + s_pad_len);
    f_r_s.extend_from_slice(&f);
    f_r_s.extend_from_slice(&r);
    f_r_s.extend_from_slice(&s_pad);

    let y: Vec<EF> = rho_ood_points
        .iter()
        .map(|&rho| eval_ze_star_n(rho, &f_r_s))
        .collect();
    assert_eq!(y.len(), t_ood);

    let query_positions: Vec<usize> = vec![0, 2, 4];
    assert_eq!(query_positions.len(), t);

    let source_openings: Vec<EF> = query_positions.iter().map(|&pos| f_codeword[pos]).collect();

    let nu_dim = 1 + t_ood + t * iota;
    let rho_batch = ef(77);
    let nu = batching_coefficients(rho_batch, nu_dim);
    let claim = ZkMaskClaim {
        base_claim_coeff: nu[0],
        residual_sumcheck_scale: EF::ONE,
        ood_coeffs: nu[1..1 + t_ood].to_vec(),
        in_domain_coeffs: nu[1 + t_ood..].to_vec(),
    };

    let mu_prime = batched_claim(mu, &y, &source_openings, &claim).unwrap();
    let relation = output_relation::<F, EF, _>(
        &source_enc,
        &sl,
        &[],
        r_len,
        s_pad_len,
        &rho_ood_points,
        &query_positions,
        &claim,
    )
    .unwrap();

    let mut r_s_pad = r;
    r_s_pad.extend_from_slice(&s_pad);

    let mu_prime_from_relation = relation.evaluate(&f, &[], &r_s_pad).unwrap();

    assert_eq!(
        mu_prime, mu_prime_from_relation,
        "Construction 9.7 mu' identity failed: verifier-computed mu' != output relation linear form"
    );
}

/// Same identity test with `n = 2` (two prior auxiliary mask oracles).
#[test]
fn test_construction_9_7_mu_prime_identity_n2() {
    let ell = 3;
    let r_len = 2;
    let s_pad_len = 1;
    let t_ood = 1;
    let t = 2;
    let iota = 1;
    let n = 2;
    let m = 8;

    let source_enc = make_rs_encoding(ell, r_len, m);

    let f: Vec<EF> = (1..=ell as u64).map(|i| ef(i * 3)).collect();
    let r: Vec<EF> = (10..10 + r_len as u64).map(ef).collect();
    let s_pad: Vec<EF> = vec![ef(42)];

    let f_base: Vec<F> = (1..=ell as u64).map(|i| F::from_u64(i * 3)).collect();
    let r_base: Vec<F> = (10..10 + r_len as u64).map(F::from_u64).collect();
    let cw = source_enc.encode_with_randomness(&f_base, &r_base);
    let f_codeword: Vec<EF> = cw.values.iter().map(|&v| EF::from(v)).collect();

    let sl: Vec<EF> = (1..=ell as u64).map(|i| ef(i * 7)).collect();

    let mask_msg_len_zk = 2;
    let xi: Vec<Vec<EF>> = (0..n)
        .map(|k| {
            (0..mask_msg_len_zk)
                .map(|j| ef((k * 10 + j + 30) as u64))
                .collect()
        })
        .collect();
    let sl_aux: Vec<Vec<EF>> = (0..n)
        .map(|k| {
            (0..mask_msg_len_zk)
                .map(|j| ef((k * 5 + j + 60) as u64))
                .collect()
        })
        .collect();

    let mut mu = inner_product(&f, &sl);
    for i in 0..n {
        mu += inner_product(&xi[i], &sl_aux[i]);
    }

    let rho_ood_points = [ef(99)];
    let mut f_r_s: Vec<EF> = Vec::new();
    f_r_s.extend_from_slice(&f);
    f_r_s.extend_from_slice(&r);
    f_r_s.extend_from_slice(&s_pad);
    let y: Vec<EF> = rho_ood_points
        .iter()
        .map(|&rho| eval_ze_star_n(rho, &f_r_s))
        .collect();

    let query_positions: Vec<usize> = vec![1, 3];
    let source_openings: Vec<EF> = query_positions.iter().map(|&p| f_codeword[p]).collect();

    let nu_dim = 1 + t_ood + t * iota;
    let rho_batch = ef(55);
    let nu = batching_coefficients(rho_batch, nu_dim);
    let claim = ZkMaskClaim {
        base_claim_coeff: nu[0],
        residual_sumcheck_scale: EF::ONE,
        ood_coeffs: nu[1..1 + t_ood].to_vec(),
        in_domain_coeffs: nu[1 + t_ood..].to_vec(),
    };

    let mu_prime = batched_claim(mu, &y, &source_openings, &claim).unwrap();
    let aux_refs: Vec<&[EF]> = sl_aux.iter().map(Vec::as_slice).collect();
    let relation = output_relation::<F, EF, _>(
        &source_enc,
        &sl,
        &aux_refs,
        r_len,
        s_pad_len,
        &rho_ood_points,
        &query_positions,
        &claim,
    )
    .unwrap();

    let mut r_s_pad = r.clone();
    r_s_pad.extend_from_slice(&s_pad);
    let xi_refs: Vec<&[EF]> = xi.iter().map(Vec::as_slice).collect();

    let mu_prime_from_relation = relation.evaluate(&f, &xi_refs, &r_s_pad).unwrap();

    assert_eq!(
        mu_prime, mu_prime_from_relation,
        "Construction 9.7 mu' identity failed with n=2 auxiliary masks"
    );
}

/// Same identity with `iota = 2`, so queried source symbols expand to multiple
/// flattened generator rows `x_{i,l} = iota * x_i + l`.
///
/// Uses a larger domain (`m = 16`) to accommodate `msg_len + t = 6 < 16`.
#[test]
fn test_construction_9_7_mu_prime_identity_iota2() {
    let ell = 4;
    let r_len = 2;
    let s_pad_len = 2;
    let t_ood = 2;
    let t = 2;
    let iota = 2;
    let m = 16;

    let source_enc = make_rs_encoding(ell, r_len, m);

    let f: Vec<EF> = (1..=ell as u64).map(|i| ef(i * 2)).collect();
    let r: Vec<EF> = (0..r_len as u64).map(|i| ef(30 + i)).collect();
    let s_pad: Vec<EF> = (0..s_pad_len as u64).map(|i| ef(40 + i)).collect();

    let f_base: Vec<F> = (1..=ell as u64).map(|i| F::from_u64(i * 2)).collect();
    let r_base: Vec<F> = (0..r_len as u64).map(|i| F::from_u64(30 + i)).collect();
    let cw = source_enc.encode_with_randomness(&f_base, &r_base);
    let f_codeword: Vec<EF> = cw.values.iter().map(|&v| EF::from(v)).collect();

    let sl: Vec<EF> = (0..ell as u64).map(|i| ef(7 + i * 3)).collect();
    let mu = inner_product(&f, &sl);

    let rho_ood_points = [ef(6), ef(8)];
    let mut f_r_s = Vec::with_capacity(ell + r_len + s_pad_len);
    f_r_s.extend_from_slice(&f);
    f_r_s.extend_from_slice(&r);
    f_r_s.extend_from_slice(&s_pad);
    let y: Vec<EF> = rho_ood_points
        .iter()
        .map(|&rho| eval_ze_star_n(rho, &f_r_s))
        .collect();

    let query_symbols = [0_usize, 2_usize];
    let mut source_openings = Vec::with_capacity(t * iota);
    for &symbol in &query_symbols {
        for limb in 0..iota {
            source_openings.push(f_codeword[symbol * iota + limb]);
        }
    }

    let nu_dim = 1 + t_ood + t * iota;
    let nu = batching_coefficients(ef(13), nu_dim);
    let claim = ZkMaskClaim {
        base_claim_coeff: nu[0],
        residual_sumcheck_scale: EF::ONE,
        ood_coeffs: nu[1..1 + t_ood].to_vec(),
        in_domain_coeffs: nu[1 + t_ood..].to_vec(),
    };

    let mu_prime = batched_claim(mu, &y, &source_openings, &claim).unwrap();
    let query_positions: Vec<usize> = query_symbols
        .iter()
        .flat_map(|&symbol| (0..iota).map(move |limb| symbol * iota + limb))
        .collect();
    let relation = output_relation::<F, EF, _>(
        &source_enc,
        &sl,
        &[],
        r_len,
        s_pad_len,
        &rho_ood_points,
        &query_positions,
        &claim,
    )
    .unwrap();

    let mut r_s_pad = r.clone();
    r_s_pad.extend_from_slice(&s_pad);
    let mu_prime_from_relation = relation.evaluate(&f, &[], &r_s_pad).unwrap();

    assert_eq!(
        mu_prime, mu_prime_from_relation,
        "Construction 9.7 mu' identity failed for iota=2 flattened row indexing"
    );
}

/// Same identity when the inherited sumcheck claim has already been scaled
/// by the #1605 HVZK sumcheck challenge `eps`.
///
/// `ZkPrefixProver::into_sumcheck` folds `eps` into its residual handoff, so
/// the code-switch output relation must scale only the inherited source
/// covector. The fresh OOD and in-domain terms are batched by Construction 9.7
/// independently.
#[test]
fn test_construction_9_7_mu_prime_identity_eps_scaled_handoff() {
    let ell = 3;
    let r_len = 1;
    let s_pad_len = 1;
    let t_ood = 1;
    let t = 1;
    let iota = 1;
    let m = 8;

    let source_enc = make_rs_encoding(ell, r_len, m);

    let f: Vec<EF> = vec![ef(2), ef(5), ef(9)];
    let r: Vec<EF> = vec![ef(12)];
    let s_pad: Vec<EF> = vec![ef(20)];

    let f_base: Vec<F> = [2, 5, 9].into_iter().map(F::from_u64).collect();
    let r_base = vec![F::from_u64(12)];
    let cw = source_enc.encode_with_randomness(&f_base, &r_base);
    let f_codeword: Vec<EF> = cw.values.iter().map(|&v| EF::from(v)).collect();

    let eps = ef(19);
    let sl: Vec<EF> = vec![ef(4), ef(7), ef(11)];
    let mu = eps * inner_product(&f, &sl);

    let rho_ood_points = [ef(31)];
    let y = [private_ood_answer(
        rho_ood_points[0],
        &f,
        &[r.clone(), s_pad.clone()].concat(),
    )];

    let query_position = 2;
    let source_opening = f_codeword[query_position];

    let nu = batching_coefficients(ef(17), 1 + t_ood + t * iota);

    let claim = ZkMaskClaim {
        base_claim_coeff: nu[0],
        residual_sumcheck_scale: eps,
        ood_coeffs: vec![nu[1]],
        in_domain_coeffs: vec![nu[2]],
    };
    let mu_prime = batched_claim(mu, &y, &[source_opening], &claim).unwrap();

    let mut sl_prime = vec![EF::ZERO; ell];
    for (sp, s) in sl_prime.iter_mut().zip(&sl) {
        *sp += eps * nu[0] * *s;
    }

    let mut power = EF::ONE;
    for sp in sl_prime.iter_mut() {
        *sp += nu[1] * power;
        power *= rho_ood_points[0];
    }

    let g_sharp = lift_row(&source_enc.message_row(query_position));
    for (sp, gs) in sl_prime.iter_mut().zip(&g_sharp) {
        *sp += nu[2] * *gs;
    }

    let mask_msg_len = r_len + s_pad_len;
    let mut sl_mask = vec![EF::ZERO; mask_msg_len];
    let mut power = EF::ONE;
    for _ in 0..ell {
        power *= rho_ood_points[0];
    }
    for sm in sl_mask.iter_mut() {
        *sm += nu[1] * power;
        power *= rho_ood_points[0];
    }

    let g_dollar = lift_row(&source_enc.randomness_row(query_position));
    for (sm, gd) in sl_mask.iter_mut().zip(&g_dollar) {
        *sm += nu[2] * *gd;
    }

    let mut r_s_pad = r;
    r_s_pad.extend_from_slice(&s_pad);

    let mu_prime_from_relation = inner_product(&f, &sl_prime) + inner_product(&r_s_pad, &sl_mask);

    assert_eq!(
        mu_prime, mu_prime_from_relation,
        "Construction 9.7 must preserve the #1605 eps-scaled residual handoff"
    );
}

/// Same handoff as above, but with a prior auxiliary mask oracle.
///
/// The #1605 `eps` scale belongs to the source residual only. Carried mask
/// auxiliary covectors must be batched by `nu_1`, not by `nu_1 * eps`.
#[test]
fn test_eps_scaled_handoff_does_not_scale_auxiliary_covectors() {
    let ell = 3;
    let r_len = 1;
    let s_pad_len = 1;
    let t_ood = 1;
    let t = 1;
    let iota = 1;
    let m = 8;

    let source_enc = make_rs_encoding(ell, r_len, m);

    let f: Vec<EF> = vec![ef(3), ef(8), ef(13)];
    let r: Vec<EF> = vec![ef(21)];
    let s_pad: Vec<EF> = vec![ef(34)];

    let f_base: Vec<F> = [3, 8, 13].into_iter().map(F::from_u64).collect();
    let r_base = vec![F::from_u64(21)];
    let cw = source_enc.encode_with_randomness(&f_base, &r_base);
    let f_codeword: Vec<EF> = cw.values.iter().map(|&v| EF::from(v)).collect();

    let source_covector = vec![ef(5), ef(7), ef(11)];
    let auxiliary_witness = vec![ef(17), ef(19)];
    let auxiliary_covector = vec![ef(23), ef(29)];

    let eps = ef(31);
    let source_claim = inner_product(&f, &source_covector);
    let auxiliary_claim = inner_product(&auxiliary_witness, &auxiliary_covector);
    let inherited_claim = eps * source_claim + auxiliary_claim;

    let rho_ood_points = [ef(37)];
    let mut mask_message = r;
    mask_message.extend_from_slice(&s_pad);
    let y = private_ood_answers(&rho_ood_points, &f, &mask_message);

    let query_position = 4;
    let source_opening = f_codeword[query_position];
    let nu = batching_coefficients(ef(41), 1 + t_ood + t * iota);
    let claim = ZkMaskClaim {
        base_claim_coeff: nu[0],
        residual_sumcheck_scale: eps,
        ood_coeffs: vec![nu[1]],
        in_domain_coeffs: vec![nu[2]],
    };

    let mu_prime = batched_claim(inherited_claim, &y, &[source_opening], &claim).unwrap();
    let relation = output_relation::<F, EF, _>(
        &source_enc,
        &source_covector,
        &[&auxiliary_covector],
        r_len,
        s_pad_len,
        &rho_ood_points,
        &[query_position],
        &claim,
    )
    .unwrap();
    let mu_prime_from_relation = relation
        .evaluate(&f, &[&auxiliary_witness], &mask_message)
        .unwrap();

    let expected_auxiliary_covector: Vec<EF> =
        auxiliary_covector.iter().map(|&x| nu[0] * x).collect();

    assert_eq!(
        relation.auxiliary_covectors[0], expected_auxiliary_covector,
        "auxiliary covectors must not inherit the #1605 eps scale"
    );
    assert_eq!(
        mu_prime, mu_prime_from_relation,
        "eps-scaled source handoff must compose with unscaled auxiliary masks"
    );
}

/// Verify private zero-evader OOD answer via `padded_ood_t1` matches
/// the manual `eval_ze_star_n` on the concatenated vector.
#[test]
fn test_private_ood_answer_consistency() {
    let f = vec![ef(3), ef(7), ef(11)];
    let r = vec![ef(5), ef(9)];
    let s_pad = vec![ef(13)];

    let mut concat = Vec::new();
    concat.extend_from_slice(&f);
    concat.extend_from_slice(&r);
    concat.extend_from_slice(&s_pad);

    let rho = ef(17);

    let from_concat = eval_ze_star_n(rho, &concat);
    let from_padded = private_ood_answer(rho, &f, &[r.clone(), s_pad.clone()].concat());

    assert_eq!(
        from_concat, from_padded,
        "padded_ood_t1 vs eval_ze_star_n mismatch"
    );

    let r17 = rho;
    let expected = concat[0]
        + concat[1] * r17
        + concat[2] * r17 * r17
        + concat[3] * r17 * r17 * r17
        + concat[4] * r17 * r17 * r17 * r17
        + concat[5] * r17 * r17 * r17 * r17 * r17;

    assert_eq!(from_concat, expected, "OOD univariate evaluation mismatch");
}

/// Verify batching zero-evader produces correct powers.
#[test]
fn test_batching_zero_evader_powers() {
    let rho = ef(5);
    let nu = batching_coefficients(rho, 4);

    assert_eq!(nu[0], EF::ONE);
    assert_eq!(nu[1], rho);
    assert_eq!(nu[2], rho * rho);
    assert_eq!(nu[3], rho * rho * rho);
}

#[test]
fn test_private_ood_answers_matches_single_answer_helper() {
    let f = vec![ef(2), ef(4)];
    let mask = vec![ef(8), ef(16), ef(32)];
    let points = [ef(3), ef(5), ef(7)];

    let answers = private_ood_answers(&points, &f, &mask);
    let expected: Vec<EF> = points
        .iter()
        .map(|&rho| private_ood_answer(rho, &f, &mask))
        .collect();

    assert_eq!(answers, expected);
}

#[test]
fn test_round_zk_config_proof_field_overhead_matches_issue_bound() {
    let config = RoundZkConfig {
        target_query_budget: 4,
        mask_query_budget: 7,
        mask_message_len: 6,
        mask_randomness_len: 7,
        ood_samples: 5,
        mask_domain_size: 16,
        mask_width: 3,
        folded_mask_domain_gen: F::from_u64(11),
    };

    assert_eq!(
        config.mask_codeword_field_elements(),
        48,
        "mask codeword contribution should be m_zk * iota_zk"
    );
    assert_eq!(
        config.proof_field_overhead(),
        53,
        "Construction 9.7 proof overhead should be m_zk * iota_zk + t_ood"
    );
}

#[test]
fn test_simulated_verifier_view_matches_code_switch_relation() {
    let ell = 3;
    let r_len = 2;
    let s_pad_len = 1;
    let t_ood = 2;
    let t = 2;
    let iota = 1;
    let m = 8;

    let source_enc = make_rs_encoding(ell, r_len, m);

    let f: Vec<EF> = vec![ef(4), ef(6), ef(10)];
    let r: Vec<EF> = vec![ef(13), ef(17)];
    let s_pad: Vec<EF> = vec![ef(19)];
    let mut mask_message = r;
    mask_message.extend_from_slice(&s_pad);

    let f_base: Vec<F> = [4, 6, 10].into_iter().map(F::from_u64).collect();
    let r_base: Vec<F> = [13, 17].into_iter().map(F::from_u64).collect();
    let cw = source_enc.encode_with_randomness(&f_base, &r_base);
    let f_codeword: Vec<EF> = cw.values.iter().map(|&v| EF::from(v)).collect();

    let source_covector = vec![ef(23), ef(29), ef(31)];
    let eps = ef(47);
    let inherited_claim = eps * inner_product(&f, &source_covector);

    let rho_ood_points = [ef(37), ef(41)];
    let private_ood = private_ood_answers(&rho_ood_points, &f, &mask_message);

    let query_positions = [1_usize, 4_usize];
    let source_openings: Vec<EF> = query_positions
        .iter()
        .map(|&position| f_codeword[position])
        .collect();

    let nu = batching_coefficients(ef(43), 1 + t_ood + t * iota);
    let claim = ZkMaskClaim {
        base_claim_coeff: nu[0],
        residual_sumcheck_scale: eps,
        ood_coeffs: nu[1..1 + t_ood].to_vec(),
        in_domain_coeffs: nu[1 + t_ood..].to_vec(),
    };

    let view = simulated_verifier_view::<F, EF, _>(
        &source_enc,
        inherited_claim,
        &source_covector,
        &[],
        r_len,
        s_pad_len,
        &rho_ood_points,
        &query_positions,
        &private_ood,
        &source_openings,
        &claim,
    )
    .unwrap();
    let expected_mu = batched_claim(inherited_claim, &private_ood, &source_openings, &claim)
        .expect("valid simulated transcript dimensions");
    let relation_value = view
        .output_relation
        .evaluate(&f, &[], &mask_message)
        .unwrap();

    assert_eq!(view.private_ood_answers, private_ood);
    assert_eq!(view.source_openings, source_openings);
    assert_eq!(view.mu_prime, expected_mu);
    assert_eq!(
        relation_value, view.mu_prime,
        "deterministic simulator view must derive the same relation as the honest code-switch path"
    );
}

#[test]
fn test_zk_sumcheck_simulator_eps_handoff_to_code_switch_view() {
    let ell_zk = 4;
    let folding_factor = 2;
    let (perm, mmcs, sumcheck_encoding) = make_setup(91, ell_zk);
    let mut simulator_challenger = MyChallenger::new(perm.clone());
    let mut replay_challenger = MyChallenger::new(perm);
    let simulator_verifier = ZkVerifier::<F, EF>::new(&[]);
    let mut simulator_rng = SmallRng::seed_from_u64(92);
    let (zk_data, mask_commits, simulator_randomness) = simulate_classic_unpacked(
        &mut simulator_challenger,
        &simulator_verifier,
        folding_factor,
        0,
        &sumcheck_encoding,
        &mmcs,
        &mut simulator_rng,
    );
    let verifier_handoff = ZkVerifier::<F, EF>::new(&[])
        .into_sumcheck::<MyMmcs, _>(
            &zk_data,
            &mask_commits,
            ell_zk,
            folding_factor,
            0,
            &mut replay_challenger,
        )
        .expect("simulated #1605 transcript should verify");
    assert_eq!(verifier_handoff.randomness, simulator_randomness);
    assert!(
        !verifier_handoff.eps.is_zero(),
        "fixture seed should produce a nonzero eps for the code-switch source scale",
    );

    let ell = 3;
    let source_randomness_len = 2;
    let pad_len = 1;
    let source_enc = ReedSolomonZkEncoding::<EF, Radix2Dit<EF>>::new(
        source_randomness_len,
        ell,
        8,
        Radix2Dit::default(),
    );
    let source_message = vec![ef(4), ef(6), ef(10)];
    let source_randomness = vec![ef(13), ef(17)];
    let s_pad = vec![ef(19)];
    let mut mask_message = source_randomness.clone();
    mask_message.extend_from_slice(&s_pad);
    let source_codeword = source_enc.encode_with_randomness(&source_message, &source_randomness);
    let query_positions = [1_usize, 4_usize];
    let source_openings = query_positions
        .iter()
        .map(|&position| source_codeword.values[position])
        .collect::<Vec<_>>();

    let mut source_covector = EF::zero_vec(source_message.len());
    let (pivot, &pivot_value) = source_message
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_zero())
        .expect("fixture source message should contain a nonzero entry");
    source_covector[pivot] = verifier_handoff.claimed_residual / verifier_handoff.eps / pivot_value;

    let rho_ood_points = [ef(37), ef(41)];
    let private_ood = private_ood_answers(&rho_ood_points, &source_message, &mask_message);
    let nu = batching_coefficients(ef(43), 1 + rho_ood_points.len() + source_openings.len());
    let claim = ZkMaskClaim {
        base_claim_coeff: nu[0],
        residual_sumcheck_scale: verifier_handoff.eps,
        ood_coeffs: nu[1..1 + rho_ood_points.len()].to_vec(),
        in_domain_coeffs: nu[1 + rho_ood_points.len()..].to_vec(),
    };

    let view = simulated_verifier_view::<EF, EF, _>(
        &source_enc,
        verifier_handoff.claimed_residual,
        &source_covector,
        &[],
        source_randomness_len,
        pad_len,
        &rho_ood_points,
        &query_positions,
        &private_ood,
        &source_openings,
        &claim,
    )
    .expect("composed simulator view should have valid dimensions");
    let expected_mu = batched_claim(
        verifier_handoff.claimed_residual,
        &private_ood,
        &source_openings,
        &claim,
    )
    .expect("valid batched claim dimensions");
    let relation_value = view
        .output_relation
        .evaluate(&source_message, &[], &mask_message)
        .expect("composed simulator relation should evaluate");

    assert_eq!(view.mu_prime, expected_mu);
    assert_eq!(relation_value, view.mu_prime);
    assert_eq!(view.private_ood_answers.len(), rho_ood_points.len());
    assert_eq!(view.source_openings.len(), query_positions.len());
}

#[test]
fn test_composed_simulator_components_program_code_switch_view() {
    assert!(assert_composed_simulator_components_program_code_switch_view(101));
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(8))]

    #[test]
    fn prop_composed_simulator_components_program_code_switch_view(seed in 0_u64..64) {
        prop_assume!(assert_composed_simulator_components_program_code_switch_view(
            seed.wrapping_add(101),
        ));
    }
}

fn assert_composed_simulator_components_program_code_switch_view(seed: u64) -> bool {
    let ell_zk = 4;
    let folding_factor = 2;
    let (perm, mmcs, sumcheck_encoding) = make_setup(seed, ell_zk);
    let mut simulator_challenger = MyChallenger::new(perm.clone());
    let mut replay_challenger = MyChallenger::new(perm);
    let simulator_verifier = ZkVerifier::<F, EF>::new(&[]);
    let mut simulator_rng = SmallRng::seed_from_u64(seed.wrapping_add(1));
    let (zk_data, mask_commits, simulator_randomness) = simulate_classic_unpacked(
        &mut simulator_challenger,
        &simulator_verifier,
        folding_factor,
        0,
        &sumcheck_encoding,
        &mmcs,
        &mut simulator_rng,
    );
    let verifier_handoff = ZkVerifier::<F, EF>::new(&[])
        .into_sumcheck::<MyMmcs, _>(
            &zk_data,
            &mask_commits,
            ell_zk,
            folding_factor,
            0,
            &mut replay_challenger,
        )
        .expect("simulated #1605 transcript should verify");
    assert_eq!(verifier_handoff.randomness, simulator_randomness);
    if verifier_handoff.eps.is_zero() {
        return false;
    }

    let ell = 3;
    let source_randomness_len = 2;
    let pad_len = 1;
    let source_enc = ReedSolomonZkEncoding::<EF, Radix2Dit<EF>>::new(
        source_randomness_len,
        ell,
        8,
        Radix2Dit::default(),
    );
    let source_message = vec![ef(4), ef(6), ef(10)];
    let mut source_covector = EF::zero_vec(source_message.len());
    let (pivot, &pivot_value) = source_message
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_zero())
        .expect("fixture source message should contain a nonzero entry");
    source_covector[pivot] = verifier_handoff.claimed_residual / verifier_handoff.eps / pivot_value;

    let query_positions = [1_usize, 4_usize];
    let mut code_sim_rng = SmallRng::seed_from_u64(seed.wrapping_add(2));
    let simulated_source_openings = source_enc.simulate(&query_positions, &mut code_sim_rng);
    let source_randomness = solve_two_randomness_for_openings(
        &source_enc,
        &source_message,
        &query_positions,
        &simulated_source_openings,
    );
    let source_codeword = source_enc.encode_with_randomness(&source_message, &source_randomness);
    let honest_source_openings = query_positions
        .iter()
        .map(|&position| source_codeword.values[position])
        .collect::<Vec<_>>();
    assert_eq!(
        honest_source_openings, simulated_source_openings,
        "Sim_C' openings must be explainable by an honest RS randomness witness",
    );

    let rho_ood_points = [ef(37)];
    let mut ze_sim_rng = SmallRng::seed_from_u64(seed.wrapping_add(3));
    let simulated_private_ood = vec![ze_sim_rng.random::<EF>()];
    let s_pad = vec![solve_pad_for_private_ood(
        rho_ood_points[0],
        &source_message,
        &source_randomness,
        simulated_private_ood[0],
    )];
    let mut mask_message = source_randomness;
    mask_message.extend_from_slice(&s_pad);
    let honest_private_ood = private_ood_answers(&rho_ood_points, &source_message, &mask_message);
    assert_eq!(
        honest_private_ood, simulated_private_ood,
        "S_ze_ood output must be explainable by the Lemma 9.3 programmed pad witness",
    );

    let mask_enc = ReedSolomonZkEncoding::<EF, Radix2Dit<EF>>::new(
        source_randomness_len,
        mask_message.len(),
        8,
        Radix2Dit::default(),
    );
    let simulated_mask_openings = mask_enc.simulate(&query_positions, &mut code_sim_rng);
    assert_eq!(
        simulated_mask_openings.len(),
        query_positions.len(),
        "Sim_C_zk must produce one opening per mask query",
    );
    let mask_randomness = solve_two_randomness_for_openings(
        &mask_enc,
        &mask_message,
        &query_positions,
        &simulated_mask_openings,
    );
    let mask_codeword = mask_enc.encode_with_randomness(&mask_message, &mask_randomness);
    let honest_mask_openings = query_positions
        .iter()
        .map(|&position| mask_codeword.values[position])
        .collect::<Vec<_>>();
    assert_eq!(
        honest_mask_openings, simulated_mask_openings,
        "Sim_C_zk openings must be explainable by an honest RS randomness witness",
    );

    let nu = batching_coefficients(ef(43), 1 + rho_ood_points.len() + query_positions.len());
    let claim = ZkMaskClaim {
        base_claim_coeff: nu[0],
        residual_sumcheck_scale: verifier_handoff.eps,
        ood_coeffs: nu[1..1 + rho_ood_points.len()].to_vec(),
        in_domain_coeffs: nu[1 + rho_ood_points.len()..].to_vec(),
    };

    let simulated_view = simulated_verifier_view::<EF, EF, _>(
        &source_enc,
        verifier_handoff.claimed_residual,
        &source_covector,
        &[],
        source_randomness_len,
        pad_len,
        &rho_ood_points,
        &query_positions,
        &simulated_private_ood,
        &simulated_source_openings,
        &claim,
    )
    .expect("composed simulator view should have valid dimensions");

    assert_eq!(
        simulated_view
            .output_relation
            .evaluate(&source_message, &[], &mask_message)
            .expect("programmed witness should satisfy the output relation"),
        simulated_view.mu_prime,
    );

    true
}

#[test]
fn test_simulated_verifier_view_rejects_private_ood_count_mismatch() {
    let enc = make_rs_encoding(3, 2, 8);
    let claim = ZkMaskClaim {
        base_claim_coeff: ef(1),
        residual_sumcheck_scale: ef(1),
        ood_coeffs: vec![ef(2), ef(3)],
        in_domain_coeffs: vec![ef(4)],
    };

    let err = simulated_verifier_view::<F, EF, _>(
        &enc,
        ef(9),
        &[ef(1), ef(2), ef(3)],
        &[],
        2,
        1,
        &[ef(7), ef(8)],
        &[0],
        &[ef(10)],
        &[ef(11)],
        &claim,
    )
    .unwrap_err();

    assert_eq!(
        err,
        CodeSwitchError::PrivateOodAnswerCountMismatch {
            expected: 2,
            actual: 1
        }
    );
}

#[test]
fn test_batched_claim_rejects_ood_count_mismatch() {
    let claim = ZkMaskClaim {
        base_claim_coeff: ef(1),
        residual_sumcheck_scale: ef(1),
        ood_coeffs: vec![ef(2), ef(3)],
        in_domain_coeffs: vec![ef(4)],
    };

    let err = batched_claim(ef(9), &[ef(10)], &[ef(11)], &claim).unwrap_err();

    assert_eq!(
        err,
        CodeSwitchError::PrivateOodAnswerCountMismatch {
            expected: 2,
            actual: 1
        }
    );
}

#[test]
fn test_output_relation_rejects_query_count_mismatch() {
    let enc = make_rs_encoding(3, 2, 8);
    let claim = ZkMaskClaim {
        base_claim_coeff: ef(1),
        residual_sumcheck_scale: ef(1),
        ood_coeffs: vec![ef(2)],
        in_domain_coeffs: vec![ef(3), ef(4)],
    };

    let err = output_relation::<F, EF, _>(
        &enc,
        &[ef(1), ef(2), ef(3)],
        &[],
        2,
        1,
        &[ef(9)],
        &[0],
        &claim,
    )
    .unwrap_err();

    assert_eq!(
        err,
        CodeSwitchError::QueryPositionCountMismatch {
            expected: 2,
            actual: 1
        }
    );
}

#[test]
fn test_output_relation_rejects_source_covector_length_mismatch() {
    let enc = make_rs_encoding(3, 2, 8);
    let claim = ZkMaskClaim {
        base_claim_coeff: ef(1),
        residual_sumcheck_scale: ef(1),
        ood_coeffs: vec![ef(2)],
        in_domain_coeffs: vec![ef(3)],
    };

    let err = output_relation::<F, EF, _>(&enc, &[ef(1), ef(2)], &[], 2, 1, &[ef(9)], &[0], &claim)
        .unwrap_err();

    assert_eq!(
        err,
        CodeSwitchError::SourceCovectorLengthMismatch {
            expected: 3,
            actual: 2
        }
    );
}

/// Verify that `ReedSolomonZkEncoding::message_row` / `randomness_row`
/// decompose the codeword correctly: `cw[i] = <msg, G^#[i]> + <rand, G^$[i]>`.
#[test]
fn test_rs_row_decomposition_matches_encoding() {
    let msg_len = 4;
    let t = 2;
    let m = 8;
    let enc = make_rs_encoding(msg_len, t, m);

    let msg: Vec<F> = (1..=msg_len as u64).map(F::from_u64).collect();
    let rand: Vec<F> = (10..10 + t as u64).map(F::from_u64).collect();
    let cw = enc.encode_with_randomness(&msg, &rand);

    for i in 0..m {
        let m_dot: F = enc
            .message_row(i)
            .iter()
            .zip(&msg)
            .map(|(a, b)| *a * *b)
            .sum();
        let r_dot: F = enc
            .randomness_row(i)
            .iter()
            .zip(&rand)
            .map(|(a, b)| *a * *b)
            .sum();
        assert_eq!(
            cw.values[i],
            m_dot + r_dot,
            "Row decomposition failed at position {i}"
        );
    }
}
