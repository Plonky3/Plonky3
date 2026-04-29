//! Tests for the HVZK code-switching round (Construction 9.7).
//!
//! These tests validate the mathematical invariants of Construction 9.7
//! from eprint 2026/391 using hand-constructed data over BabyBear.
//! They have zero dependency on the upstream #1584/#1585/#1586 implementations
//! and can run against current `main`.

use alloc::vec;
use alloc::vec::Vec;

use p3_baby_bear::BabyBear;
use p3_field::PrimeCharacteristicRing;
use p3_field::extension::BinomialExtensionField;

type F = BabyBear;
type EF = BinomialExtensionField<F, 4>;

fn ef(v: u64) -> EF {
    EF::from(F::from_u64(v))
}

fn inner_product(a: &[EF], b: &[EF]) -> EF {
    a.iter().zip(b.iter()).map(|(x, y)| *x * *y).sum()
}

/// Compute `ze_ood(rho) * vec` where each row i of ze_ood is
/// `(rho_i^0, rho_i^1, ..., rho_i^{len-1})`.
fn apply_ood_zero_evader(rho_points: &[EF], vec: &[EF]) -> Vec<EF> {
    rho_points
        .iter()
        .map(|rho| {
            let mut power = EF::ONE;
            let mut acc = EF::ZERO;
            for v in vec {
                acc += power * *v;
                power *= *rho;
            }
            acc
        })
        .collect()
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

/// Trivial encoding for testing: `Enc(f, r) = f || r`.
/// `G^#[i, :]` is the `i`-th standard basis vector of length `ell`,
/// `G^$[i, :]` is the `(i - ell)`-th standard basis vector of length `r`.
struct TrivialEncoding {
    message_len: usize,
    randomness_len: usize,
}

impl TrivialEncoding {
    fn encode(&self, f: &[EF], r: &[EF]) -> Vec<EF> {
        assert_eq!(f.len(), self.message_len);
        assert_eq!(r.len(), self.randomness_len);
        let mut cw = f.to_vec();
        cw.extend_from_slice(r);
        cw
    }

    fn g_sharp_row(&self, symbol_index: usize) -> Vec<EF> {
        let mut row = vec![EF::ZERO; self.message_len];
        if symbol_index < self.message_len {
            row[symbol_index] = EF::ONE;
        }
        row
    }

    fn g_dollar_row(&self, symbol_index: usize) -> Vec<EF> {
        let mut row = vec![EF::ZERO; self.randomness_len];
        if symbol_index >= self.message_len {
            row[symbol_index - self.message_len] = EF::ONE;
        }
        row
    }
}

/// Construction 9.7 `mu'` identity test with `n = 0` (no prior auxiliary masks).
///
/// Validates the completeness equation:
///
/// ```text
/// mu' = nu_1 * mu
///     + sum_{i in [t_ood]} nu_{1+i} * y_i
///     + sum_{i in [t]} sum_{l in [iota]} nu_{1+t_ood+i*iota+l} * f(x_i)_l
/// ```
///
/// AND that `mu'` equals `<f, sl'> + <(r, s_pad), sl_mask>`.
#[test]
fn test_construction_9_7_mu_prime_identity_n0() {
    let ell = 4;
    let r_len = 2;
    let s_pad_len = 1;
    let t_ood = 2;
    let t = 3;
    let iota = 1;

    let source_enc = TrivialEncoding {
        message_len: ell,
        randomness_len: r_len,
    };

    let f: Vec<EF> = (1..=ell as u64).map(ef).collect();
    let r: Vec<EF> = (10..10 + r_len as u64).map(ef).collect();
    let s_pad: Vec<EF> = (20..20 + s_pad_len as u64).map(ef).collect();

    let f_codeword = source_enc.encode(&f, &r);

    let sl: Vec<EF> = (1..=ell as u64).map(|i| ef(i * 100)).collect();
    let mu = inner_product(&f, &sl);

    let rho_ood_points: Vec<EF> = (0..t_ood).map(|i| ef(50 + i as u64)).collect();

    let mut f_r_s: Vec<EF> = Vec::with_capacity(ell + r_len + s_pad_len);
    f_r_s.extend_from_slice(&f);
    f_r_s.extend_from_slice(&r);
    f_r_s.extend_from_slice(&s_pad);

    let y = apply_ood_zero_evader(&rho_ood_points, &f_r_s);
    assert_eq!(y.len(), t_ood);

    let query_positions: Vec<usize> = vec![0, 2, 4];
    assert_eq!(query_positions.len(), t);

    let source_openings: Vec<EF> = query_positions.iter().map(|&pos| f_codeword[pos]).collect();

    let nu_dim = 1 + t_ood + t * iota;
    let rho_batch = ef(77);
    let nu = batching_zero_evader(rho_batch, nu_dim);

    // Verifier computes mu'.
    let mut mu_prime = nu[0] * mu;
    for i in 0..t_ood {
        mu_prime += nu[1 + i] * y[i];
    }
    for i in 0..t {
        for l in 0..iota {
            mu_prime += nu[1 + t_ood + i * iota + l] * source_openings[i * iota + l];
        }
    }

    // Build output relation linear forms and verify identity.
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
            let g_sharp = source_enc.g_sharp_row(query_positions[i] + l);
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
            let g_dollar = source_enc.g_dollar_row(query_positions[i] + l);
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

    let source_enc = TrivialEncoding {
        message_len: ell,
        randomness_len: r_len,
    };

    let f: Vec<EF> = (1..=ell as u64).map(|i| ef(i * 3)).collect();
    let r: Vec<EF> = (10..10 + r_len as u64).map(ef).collect();
    let s_pad: Vec<EF> = vec![ef(42)];

    let f_codeword = source_enc.encode(&f, &r);

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
    let y = apply_ood_zero_evader(&rho_ood_points, &f_r_s);

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

    // Output relation decomposition.
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
            let g_sharp = source_enc.g_sharp_row(query_positions[i] + l);
            for j in 0..ell {
                sl_prime[j] += nu[1 + t_ood + i * iota + l] * g_sharp[j];
            }
        }
    }

    // Auxiliary mask contribution: nu_1 * sum_i <xi_i, sl_aux_i>.
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
            let g_dollar = source_enc.g_dollar_row(query_positions[i] + l);
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
#[test]
fn test_construction_9_7_mu_prime_identity_iota2() {
    let ell = 4;
    let r_len = 2;
    let s_pad_len = 2;
    let t_ood = 2;
    let t = 2;
    let iota = 2;

    let source_enc = TrivialEncoding {
        message_len: ell,
        randomness_len: r_len,
    };

    let f: Vec<EF> = (1..=ell as u64).map(|i| ef(i * 2)).collect();
    let r: Vec<EF> = (0..r_len as u64).map(|i| ef(30 + i)).collect();
    let s_pad: Vec<EF> = (0..s_pad_len as u64).map(|i| ef(40 + i)).collect();
    let f_codeword = source_enc.encode(&f, &r);

    let sl: Vec<EF> = (0..ell as u64).map(|i| ef(7 + i * 3)).collect();
    let mu = inner_product(&f, &sl);

    let rho_ood_points = vec![ef(6), ef(8)];
    let mut f_r_s = Vec::with_capacity(ell + r_len + s_pad_len);
    f_r_s.extend_from_slice(&f);
    f_r_s.extend_from_slice(&r);
    f_r_s.extend_from_slice(&s_pad);
    let y = apply_ood_zero_evader(&rho_ood_points, &f_r_s);

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
            let g_sharp = source_enc.g_sharp_row(flat_index);
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
            let g_dollar = source_enc.g_dollar_row(flat_index);
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

/// Verify private zero-evader OOD answer matches polynomial evaluation.
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
    let y = apply_ood_zero_evader(&[rho], &concat);

    // 3 + 7*17 + 11*17^2 + 5*17^3 + 9*17^4 + 13*17^5
    let r17 = rho;
    let expected = concat[0]
        + concat[1] * r17
        + concat[2] * r17 * r17
        + concat[3] * r17 * r17 * r17
        + concat[4] * r17 * r17 * r17 * r17
        + concat[5] * r17 * r17 * r17 * r17 * r17;

    assert_eq!(y[0], expected, "OOD univariate evaluation mismatch");
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
