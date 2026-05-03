//! Tests for the HVZK code-switching round (Construction 9.7).
//!
//! These tests validate the mathematical invariants of Construction 9.7
//! from eprint 2026/391 using hand-constructed data over BabyBear.
//!
//! As of #1584/#1585 merging, these tests use the real `p3-zk-codes`
//! (`LinearZkEncoding`) and `whir::utils` (`eval_ze_star_n`, `padded_ood_t1`)
//! APIs instead of hand-rolled stubs.

use alloc::vec;
use alloc::vec::Vec;

use p3_baby_bear::BabyBear;
use p3_dft::Radix2Dit;
use p3_field::PrimeCharacteristicRing;
use p3_field::extension::BinomialExtensionField;
use p3_zk_codes::{LinearZkEncoding, ReedSolomonZkEncoding, ZkEncodingWithRandomness};

use crate::utils::{eval_ze_star_n, padded_ood_t1};

type F = BabyBear;
type EF = BinomialExtensionField<F, 4>;

fn ef(v: u64) -> EF {
    EF::from(F::from_u64(v))
}

fn inner_product(a: &[EF], b: &[EF]) -> EF {
    a.iter().zip(b.iter()).map(|(x, y)| *x * *y).sum()
}

/// Returns `(1, rho, rho^2, ..., rho^{dim-1})`.
fn batching_zero_evader(rho: EF, dim: usize) -> Vec<EF> {
    let mut nu = Vec::with_capacity(dim);
    let mut power = EF::ONE;
    for _ in 0..dim {
        nu.push(power);
        power *= rho;
    }
    nu
}

fn make_rs_encoding(msg_len: usize, t: usize, m: usize) -> ReedSolomonZkEncoding<F, Radix2Dit<F>> {
    ReedSolomonZkEncoding::new(t, msg_len, m, Radix2Dit::default())
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
    let nu = batching_zero_evader(rho_batch, nu_dim);

    let mut mu_prime = nu[0] * mu;
    for i in 0..t_ood {
        mu_prime += nu[1 + i] * y[i];
    }
    for i in 0..t {
        for l in 0..iota {
            mu_prime += nu[1 + t_ood + i * iota + l] * source_openings[i * iota + l];
        }
    }

    let mut sl_prime = vec![EF::ZERO; ell];
    for j in 0..ell {
        sl_prime[j] += nu[0] * sl[j];
    }
    for i in 0..t_ood {
        let mut power = EF::ONE;
        for j in 0..ell {
            sl_prime[j] += nu[1 + i] * power;
            power *= rho_ood_points[i];
        }
    }
    for i in 0..t {
        for l in 0..iota {
            let g_sharp = lift_row(&source_enc.message_row(query_positions[i] + l));
            for j in 0..ell {
                sl_prime[j] += nu[1 + t_ood + i * iota + l] * g_sharp[j];
            }
        }
    }

    let mask_msg_len = r_len + s_pad_len;
    let mut sl_mask = vec![EF::ZERO; mask_msg_len];
    for i in 0..t_ood {
        let mut power = EF::ONE;
        for _ in 0..ell {
            power *= rho_ood_points[i];
        }
        for j in 0..mask_msg_len {
            sl_mask[j] += nu[1 + i] * power;
            power *= rho_ood_points[i];
        }
    }
    for i in 0..t {
        for l in 0..iota {
            let g_dollar = lift_row(&source_enc.randomness_row(query_positions[i] + l));
            for j in 0..r_len {
                sl_mask[j] += nu[1 + t_ood + i * iota + l] * g_dollar[j];
            }
        }
    }

    let mut r_s_pad = r.clone();
    r_s_pad.extend_from_slice(&s_pad);

    let mu_prime_from_relation = inner_product(&f, &sl_prime) + inner_product(&r_s_pad, &sl_mask);

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

    let rho_ood_points = vec![ef(99)];
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
    let nu = batching_zero_evader(rho_batch, nu_dim);

    let mut mu_prime = nu[0] * mu;
    for i in 0..t_ood {
        mu_prime += nu[1 + i] * y[i];
    }
    for i in 0..t {
        for l in 0..iota {
            mu_prime += nu[1 + t_ood + i * iota + l] * source_openings[i * iota + l];
        }
    }

    let mut sl_prime = vec![EF::ZERO; ell];
    for j in 0..ell {
        sl_prime[j] += nu[0] * sl[j];
    }
    for i in 0..t_ood {
        let mut power = EF::ONE;
        for j in 0..ell {
            sl_prime[j] += nu[1 + i] * power;
            power *= rho_ood_points[i];
        }
    }
    for i in 0..t {
        for l in 0..iota {
            let g_sharp = lift_row(&source_enc.message_row(query_positions[i] + l));
            for j in 0..ell {
                sl_prime[j] += nu[1 + t_ood + i * iota + l] * g_sharp[j];
            }
        }
    }

    let mut aux_contribution = EF::ZERO;
    for i in 0..n {
        aux_contribution += nu[0] * inner_product(&xi[i], &sl_aux[i]);
    }

    let mask_msg_len = r_len + s_pad_len;
    let mut sl_mask = vec![EF::ZERO; mask_msg_len];
    for i in 0..t_ood {
        let mut power = EF::ONE;
        for _ in 0..ell {
            power *= rho_ood_points[i];
        }
        for j in 0..mask_msg_len {
            sl_mask[j] += nu[1 + i] * power;
            power *= rho_ood_points[i];
        }
    }
    for i in 0..t {
        for l in 0..iota {
            let g_dollar = lift_row(&source_enc.randomness_row(query_positions[i] + l));
            for j in 0..r_len {
                sl_mask[j] += nu[1 + t_ood + i * iota + l] * g_dollar[j];
            }
        }
    }

    let mut r_s_pad = r.clone();
    r_s_pad.extend_from_slice(&s_pad);

    let mu_prime_from_relation =
        inner_product(&f, &sl_prime) + aux_contribution + inner_product(&r_s_pad, &sl_mask);

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

    let rho_ood_points = vec![ef(6), ef(8)];
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
    let nu = batching_zero_evader(ef(13), nu_dim);

    let mut mu_prime = nu[0] * mu;
    for i in 0..t_ood {
        mu_prime += nu[1 + i] * y[i];
    }
    for i in 0..t {
        for l in 0..iota {
            mu_prime += nu[1 + t_ood + i * iota + l] * source_openings[i * iota + l];
        }
    }

    let mut sl_prime = vec![EF::ZERO; ell];
    for j in 0..ell {
        sl_prime[j] += nu[0] * sl[j];
    }
    for i in 0..t_ood {
        let mut power = EF::ONE;
        for j in 0..ell {
            sl_prime[j] += nu[1 + i] * power;
            power *= rho_ood_points[i];
        }
    }
    for i in 0..t {
        for l in 0..iota {
            let flat_index = query_symbols[i] * iota + l;
            let g_sharp = lift_row(&source_enc.message_row(flat_index));
            for j in 0..ell {
                sl_prime[j] += nu[1 + t_ood + i * iota + l] * g_sharp[j];
            }
        }
    }

    let mask_msg_len = r_len + s_pad_len;
    let mut sl_mask = vec![EF::ZERO; mask_msg_len];
    for i in 0..t_ood {
        let mut power = EF::ONE;
        for _ in 0..ell {
            power *= rho_ood_points[i];
        }
        for j in 0..mask_msg_len {
            sl_mask[j] += nu[1 + i] * power;
            power *= rho_ood_points[i];
        }
    }
    for i in 0..t {
        for l in 0..iota {
            let flat_index = query_symbols[i] * iota + l;
            let g_dollar = lift_row(&source_enc.randomness_row(flat_index));
            for j in 0..r_len {
                sl_mask[j] += nu[1 + t_ood + i * iota + l] * g_dollar[j];
            }
        }
    }

    let mut r_s_pad = r.clone();
    r_s_pad.extend_from_slice(&s_pad);
    let mu_prime_from_relation = inner_product(&f, &sl_prime) + inner_product(&r_s_pad, &sl_mask);

    assert_eq!(
        mu_prime, mu_prime_from_relation,
        "Construction 9.7 mu' identity failed for iota=2 flattened row indexing"
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
    let from_padded = padded_ood_t1(rho, &f, &[r.clone(), s_pad.clone()].concat());

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
    let nu = batching_zero_evader(rho, 4);

    assert_eq!(nu[0], EF::ONE);
    assert_eq!(nu[1], rho);
    assert_eq!(nu[2], rho * rho);
    assert_eq!(nu[3], rho * rho * rho);
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
