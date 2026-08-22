//! STIR-specific soundness formulas and parameter derivation.

use p3_security::whir::SecurityAssumption;

pub(crate) trait StirSoundness {
    fn stir_num_ood_samples(&self) -> usize;

    fn stir_query_failure_base(&self, log_inv_rate: usize, eta: f64) -> f64;

    fn stir_eta_upper_bound(&self, log_inv_rate: usize) -> f64;

    fn stir_eta_is_valid(&self, log_inv_rate: usize, eta: f64) -> bool;

    fn stir_initial_eta(
        &self,
        pow_target_bits: usize,
        unprotected_target_bits: usize,
        log_degree: usize,
        log_inv_rate: usize,
        log_folding_factor: usize,
        field_size_bits: usize,
    ) -> f64;

    #[allow(clippy::too_many_arguments)]
    fn stir_recursive_eta(
        &self,
        pow_target_bits: usize,
        unprotected_target_bits: usize,
        log_degree: usize,
        log_inv_rate: usize,
        log_domain_size: usize,
        log_folding_factor: usize,
        field_size_bits: usize,
        prev_queries: usize,
    ) -> f64;

    fn stir_queries_for_base(&self, security_bits: usize, failure_base: f64) -> usize;

    fn fold_algebraic_bits_at_log_eta(
        &self,
        field_size_bits: usize,
        log_degree: usize,
        log_inv_rate: usize,
        log_eta: f64,
    ) -> f64;

    fn stir_query_pow_eligible_bits(
        &self,
        field_size_bits: usize,
        log_degree: usize,
        log_inv_rate: usize,
        eta: f64,
        num_queries: usize,
        num_ood_samples: usize,
    ) -> f64;

    fn stir_query_unprotected_bits(
        &self,
        field_size_bits: usize,
        log_degree: usize,
        log_inv_rate: usize,
        eta: f64,
        num_queries: usize,
        num_ood_samples: usize,
    ) -> f64;

    fn stir_final_query_algebraic_bits(
        &self,
        log_inv_rate: usize,
        eta: f64,
        num_queries: usize,
    ) -> f64;
}

fn rate_from_log_inv_rate(log_inv_rate: usize) -> f64 {
    libm::pow(2., -(log_inv_rate as f64))
}

fn log2_field_minus_domain(field_size_bits: usize, log_domain_size: usize) -> f64 {
    assert!(
        field_size_bits > log_domain_size,
        "challenge field must contain points outside the evaluation domain"
    );
    let ratio = libm::pow(2., log_domain_size as f64 - field_size_bits as f64);
    field_size_bits as f64 + libm::log2(1. - ratio)
}

fn query_count_from_failure_base(security_bits: usize, failure_base: f64) -> usize {
    assert!(
        failure_base > 0. && failure_base < 1.,
        "STIR query-count formula requires a failure base in (0, 1), got {failure_base}"
    );
    libm::ceil(security_bits as f64 / -libm::log2(failure_base)) as usize
}

fn minimum_eta_for_target(
    upper_bound: f64,
    target_bits: usize,
    mut bits_at_eta: impl FnMut(f64) -> f64,
    label: &str,
) -> f64 {
    let upper_bits = bits_at_eta(upper_bound);
    assert!(
        upper_bits >= target_bits as f64,
        "{label} reaches only {upper_bits:.4} bits at the largest permitted eta \
         ({upper_bound}); target is {target_bits} bits"
    );

    // Every bound used here is monotone in eta: a larger safety gap means a
    // smaller list and a smaller BCSS25 exceptional set. Keep `high` feasible
    // throughout so the returned value remains on the sound side of a step in
    // BCSS25's integer multiplicity.
    let mut low = 0.;
    let mut high = upper_bound;
    for _ in 0..80 {
        let mid = (low + high) / 2.;
        if bits_at_eta(mid) >= target_bits as f64 {
            high = mid;
        } else {
            low = mid;
        }
    }
    high
}

fn list_size_bits_at_log_eta(
    assumption: SecurityAssumption,
    log_degree: usize,
    log_inv_rate: usize,
    log_eta: f64,
) -> f64 {
    match assumption {
        SecurityAssumption::UniqueDecoding => 0.,
        SecurityAssumption::JohnsonBound => log_inv_rate as f64 / 2. - (1. + log_eta),
        SecurityAssumption::CapacityBound => (log_degree + log_inv_rate) as f64 - log_eta,
    }
}

fn prox_gaps_error_at_log_eta(
    assumption: SecurityAssumption,
    log_degree: usize,
    log_inv_rate: usize,
    field_size_bits: usize,
    num_functions: usize,
    log_eta: f64,
) -> f64 {
    assert!(
        num_functions >= 2,
        "num_functions must be >= 2 to compute proximity gaps error"
    );

    let exceptional_set_bits = match assumption {
        SecurityAssumption::UniqueDecoding => (log_degree + log_inv_rate) as f64,
        SecurityAssumption::JohnsonBound => {
            // BCSS25 Theorem 1.5, dominant term, at the protocol's actual eta:
            // m = max(ceil(sqrt(rho) / (2 eta)), 3).
            let log_sqrt_rho_over_2eta = -(log_inv_rate as f64) / 2. - 1. - log_eta;
            let m = libm::ceil(libm::pow(2., log_sqrt_rho_over_2eta)).max(3.);
            let log_n = (log_degree + log_inv_rate) as f64;
            let constant = libm::log2(2. * libm::pow(m + 0.5, 5.) / 3.);
            log_n + constant + 1.5 * log_inv_rate as f64
        }
        SecurityAssumption::CapacityBound => (log_degree + 2 * log_inv_rate) as f64 - log_eta,
    };

    field_size_bits as f64 - (exceptional_set_bits + libm::log2(num_functions as f64 - 1.))
}

fn ood_error_at_log_eta(
    assumption: SecurityAssumption,
    log_degree: usize,
    log_inv_rate: usize,
    field_size_bits: usize,
    ood_samples: usize,
    log_eta: f64,
) -> f64 {
    if matches!(assumption, SecurityAssumption::UniqueDecoding) {
        return 0.;
    }

    let list_size = list_size_bits_at_log_eta(assumption, log_degree, log_inv_rate, log_eta);
    let error = 2. * list_size + (log_degree * ood_samples) as f64;
    (ood_samples * field_size_bits) as f64 + 1. - error
}

fn fold_sumcheck_error_at_log_eta(
    assumption: SecurityAssumption,
    field_size_bits: usize,
    log_degree: usize,
    log_inv_rate: usize,
    log_eta: f64,
) -> f64 {
    let list_size = list_size_bits_at_log_eta(assumption, log_degree, log_inv_rate, log_eta);
    field_size_bits as f64 - (list_size + 1.)
}

fn queries_combination_error_at_log_eta(
    assumption: SecurityAssumption,
    field_size_bits: usize,
    log_degree: usize,
    log_inv_rate: usize,
    ood_samples: usize,
    num_queries: usize,
    log_eta: f64,
) -> f64 {
    let list_size = list_size_bits_at_log_eta(assumption, log_degree, log_inv_rate, log_eta);
    let log_combination = libm::log2((ood_samples + num_queries) as f64);
    field_size_bits as f64 - (log_combination + list_size + 1.)
}

fn shake_check_error(field_size_bits: usize, num_queries: usize, num_ood_samples: usize) -> f64 {
    let num_points = (num_queries + num_ood_samples) as f64;
    field_size_bits as f64 - libm::log2(2. * num_points).max(0.)
}

impl StirSoundness for SecurityAssumption {
    fn stir_num_ood_samples(&self) -> usize {
        match self {
            Self::JohnsonBound => 1,
            Self::CapacityBound => 2,
            Self::UniqueDecoding => {
                panic!("STIR's paper-backed parameter schedule does not support UniqueDecoding")
            }
        }
    }

    fn stir_query_failure_base(&self, log_inv_rate: usize, eta: f64) -> f64 {
        match self {
            Self::JohnsonBound => libm::sqrt(rate_from_log_inv_rate(log_inv_rate)) + eta,
            Self::CapacityBound => rate_from_log_inv_rate(log_inv_rate) + eta,
            Self::UniqueDecoding => {
                panic!("STIR's paper-backed parameter schedule does not support UniqueDecoding")
            }
        }
    }

    fn stir_eta_upper_bound(&self, log_inv_rate: usize) -> f64 {
        match self {
            Self::JohnsonBound => libm::sqrt(rate_from_log_inv_rate(log_inv_rate)) / 20.,
            Self::CapacityBound => rate_from_log_inv_rate(log_inv_rate) / 2.,
            Self::UniqueDecoding => {
                panic!("STIR's paper-backed parameter schedule does not support UniqueDecoding")
            }
        }
    }

    fn stir_eta_is_valid(&self, log_inv_rate: usize, eta: f64) -> bool {
        eta.is_finite() && eta > 0. && eta <= self.stir_eta_upper_bound(log_inv_rate)
    }

    fn stir_initial_eta(
        &self,
        pow_target_bits: usize,
        unprotected_target_bits: usize,
        log_degree: usize,
        log_inv_rate: usize,
        log_folding_factor: usize,
        field_size_bits: usize,
    ) -> f64 {
        let upper = self.stir_eta_upper_bound(log_inv_rate);
        let ood_samples = self.stir_num_ood_samples();

        let fold_eta = minimum_eta_for_target(
            upper,
            pow_target_bits,
            |eta| {
                self.fold_algebraic_bits_at_log_eta(
                    field_size_bits,
                    log_degree,
                    log_inv_rate,
                    libm::log2(eta),
                )
            },
            "initial STIR folding bound",
        );
        let ood_eta = minimum_eta_for_target(
            upper,
            unprotected_target_bits,
            |eta| {
                ood_error_at_log_eta(
                    *self,
                    log_degree,
                    log_inv_rate,
                    field_size_bits,
                    ood_samples,
                    libm::log2(eta),
                )
            },
            "initial STIR OOD bound",
        );

        let schedule_eta = match self {
            // The old BCIKS-form 1/7-power expression is intentionally not
            // retained here: validation uses BCSS25's O(n/eta^5) bound, so
            // deriving eta from the same bound avoids rejecting feasible JB
            // configurations at realistic security levels.
            Self::JohnsonBound => 0.,
            Self::CapacityBound => {
                let k = 1usize << log_folding_factor;
                let log_k_minus_1 = libm::log2((k - 1) as f64);
                let log_d_over_k = (log_degree - log_folding_factor) as f64;
                let log_eta_proxgap = pow_target_bits as f64
                    + log_k_minus_1
                    + log_d_over_k
                    + 2. * log_inv_rate as f64
                    - field_size_bits as f64;

                let rho = rate_from_log_inv_rate(log_inv_rate);
                let log_failure_base_max = libm::log2(1.5 * rho);
                let t_0_max = libm::ceil(pow_target_bits as f64 / -log_failure_base_max);
                let third_factor = (t_0_max + 1.) + (k - 1) as f64 / k as f64;
                let log_eta_combination =
                    pow_target_bits as f64 + 1. + log_degree as f64 + 2. * log_inv_rate as f64
                        - field_size_bits as f64
                        + libm::log2(third_factor);
                libm::pow(2., log_eta_proxgap.max(log_eta_combination))
            }
            Self::UniqueDecoding => {
                panic!("STIR's paper-backed parameter schedule does not support UniqueDecoding")
            }
        };

        schedule_eta.max(fold_eta).max(ood_eta)
    }

    fn stir_recursive_eta(
        &self,
        pow_target_bits: usize,
        unprotected_target_bits: usize,
        log_degree: usize,
        log_inv_rate: usize,
        log_domain_size: usize,
        log_folding_factor: usize,
        field_size_bits: usize,
        prev_queries: usize,
    ) -> f64 {
        let k = 1usize << log_folding_factor;
        let log_domain = log_domain_size as f64;
        let log_field_minus_domain = log2_field_minus_domain(field_size_bits, log_domain_size);

        let schedule_eta = match self {
            Self::JohnsonBound => {
                let log_ood_term = (unprotected_target_bits as f64 + log_degree as f64 - 3.
                    + log_inv_rate as f64
                    - log_field_minus_domain)
                    / 2.;
                libm::pow(2., log_ood_term)
            }
            Self::CapacityBound => {
                let log_term_1 = 1. - log_domain;
                let log_term_2 = log_domain
                    + (pow_target_bits as f64 + 2. * log_degree as f64
                        - 1.
                        - 2. * log_field_minus_domain)
                        / 2.;
                let third_factor = (prev_queries + 1) as f64 + (k - 1) as f64 / k as f64;
                let log_term_3 =
                    pow_target_bits as f64 + 1. + log_degree as f64 + 2. * log_inv_rate as f64
                        - field_size_bits as f64
                        + libm::log2(third_factor);
                libm::pow(2., log_term_1.max(log_term_2).max(log_term_3))
            }
            Self::UniqueDecoding => {
                panic!("STIR's paper-backed parameter schedule does not support UniqueDecoding")
            }
        };

        let upper = self.stir_eta_upper_bound(log_inv_rate);
        let fold_eta = minimum_eta_for_target(
            upper,
            pow_target_bits,
            |eta| {
                self.fold_algebraic_bits_at_log_eta(
                    field_size_bits,
                    log_degree,
                    log_inv_rate,
                    libm::log2(eta),
                )
            },
            "recursive STIR folding bound",
        );
        let ood_eta = minimum_eta_for_target(
            upper,
            unprotected_target_bits,
            |eta| {
                ood_error_at_log_eta(
                    *self,
                    log_degree,
                    log_inv_rate,
                    field_size_bits,
                    self.stir_num_ood_samples(),
                    libm::log2(eta),
                )
            },
            "recursive STIR OOD bound",
        );

        schedule_eta.max(fold_eta).max(ood_eta)
    }

    fn stir_queries_for_base(&self, security_bits: usize, failure_base: f64) -> usize {
        let _ = self;
        query_count_from_failure_base(security_bits, failure_base)
    }

    fn fold_algebraic_bits_at_log_eta(
        &self,
        field_size_bits: usize,
        log_degree: usize,
        log_inv_rate: usize,
        log_eta: f64,
    ) -> f64 {
        let prox_gaps = prox_gaps_error_at_log_eta(
            *self,
            log_degree,
            log_inv_rate,
            field_size_bits,
            2,
            log_eta,
        );
        let sumcheck = fold_sumcheck_error_at_log_eta(
            *self,
            field_size_bits,
            log_degree,
            log_inv_rate,
            log_eta,
        );
        prox_gaps.min(sumcheck)
    }

    fn stir_query_pow_eligible_bits(
        &self,
        field_size_bits: usize,
        log_degree: usize,
        log_inv_rate: usize,
        eta: f64,
        num_queries: usize,
        num_ood_samples: usize,
    ) -> f64 {
        let failure_base = self.stir_query_failure_base(log_inv_rate, eta);
        let query_failure = -(num_queries as f64) * libm::log2(failure_base);
        let combination = queries_combination_error_at_log_eta(
            *self,
            field_size_bits,
            log_degree,
            log_inv_rate,
            num_ood_samples,
            num_queries,
            libm::log2(eta),
        );
        query_failure.min(combination)
    }

    fn stir_query_unprotected_bits(
        &self,
        field_size_bits: usize,
        log_degree: usize,
        log_inv_rate: usize,
        eta: f64,
        num_queries: usize,
        num_ood_samples: usize,
    ) -> f64 {
        let ood = ood_error_at_log_eta(
            *self,
            log_degree,
            log_inv_rate,
            field_size_bits,
            num_ood_samples,
            libm::log2(eta),
        );
        let shake = shake_check_error(field_size_bits, num_queries, num_ood_samples);
        ood.min(shake)
    }

    fn stir_final_query_algebraic_bits(
        &self,
        log_inv_rate: usize,
        eta: f64,
        num_queries: usize,
    ) -> f64 {
        let failure_base = self.stir_query_failure_base(log_inv_rate, eta);
        -(num_queries as f64) * libm::log2(failure_base)
    }
}

/// Minimum `log2(eta)` the batch-degree-correction "Combine" step (§4.5) needs merging
/// `num_classes` height classes into a single codeword of degree `2^log_d_star` to reach
/// `target_bits` of algebraic security, at distance `delta := 1 - B*(rho) - eta` (`B*` per
/// the assumption: `sqrt(rho)` for `JohnsonBound`, `rho` for `CapacityBound`).
///
/// Using `eta` here — the *same* variable `stir_initial_eta` folds this into via `.max()` —
/// is what pins `delta_combine` to exactly the schedule's own round-0 `delta_0`: Combine's
/// guarantee ("if the merged codeword passes, each class has correlated agreement") is only
/// sound if it covers at least as far as STIR's own round-0 acceptance radius, and sharing
/// the variable makes that automatic rather than a separate value to keep in sync.
///
/// Both arms charge Lemma 7.3's round-by-round soundness of Construction 7.2 at the
/// degree-gap-inflated multiplicity `ell`, not the raw class count: `err*` is the §4.1
/// abstraction the lemma invokes regardless of which regime bounds it, so the conjectured
/// route (Conjecture 5.6, `CapacityBound`) inflates by `ell` exactly like the provable one
/// (Theorem 4.1, `JohnsonBound`).
/// - `CapacityBound`: `err*(d*, rho, delta, ell) = (ell-1)*d* / (eta*rho^2*|F|)` (`c1=c2=1`).
///   Lemma 7.3 also requires `delta < 1 - rho - 1/|L_0|`, which under `delta = 1 - rho - eta`
///   needs `eta >= 2/|L_0|` — the same margin `stir_recursive_eta`'s `CapacityBound` arm
///   keeps for later rounds.
/// - `JohnsonBound`: Theorem 4.1's "far"-case `err*(d*, rho, delta, ell) = (ell-1)*d*^2 /
///   (|F|*(2*eta)^7)` (Lemma 4.13's own route, valid only up to `delta < 1 - sqrt(rho)`, so
///   `eta` must additionally stay `<= sqrt(rho)/20` — `stir_eta_upper_bound` enforces this
///   once the returned value is `.max()`-folded into `stir_initial_eta`, checked by
///   `validate_eta` in `StirConfig::new`). This regime satisfies the `1/|L_0|` side
///   condition automatically.
///
/// `ell` is Lemma 4.13's error argument `num_classes·(d* + 1) − Σᵢ dᵢ` (dᵢ the untouched,
/// i.e. not further quotiented, degree bound of each class's reduced-opening).
pub(crate) fn combine_min_log_eta(
    assumption: SecurityAssumption,
    field_size_bits: usize,
    log_inv_rate: usize,
    log_d_star: usize,
    ell: u64,
    target_bits: usize,
) -> f64 {
    match assumption {
        SecurityAssumption::CapacityBound => {
            // bits = field_bits + log2(eta) + 2*log2(rho) - log2(ell-1) - log_d_star
            //      = field_bits + log2(eta) - 2*log_inv_rate - log2(ell-1) - log_d_star
            let log_ell_minus_1 = libm::log2((ell.saturating_sub(1)).max(1) as f64);
            let log_eta_conjecture =
                target_bits as f64 + 2. * log_inv_rate as f64 + log_ell_minus_1 + log_d_star as f64
                    - field_size_bits as f64;

            // Lemma 7.3's delta < 1 - rho - 1/|L_0| side condition, restated as a floor on
            // eta (|L_0| = 2^(log_d_star + log_inv_rate), round 0's domain size).
            let log_domain_size = (log_d_star + log_inv_rate) as f64;
            let log_eta_side_condition = 1. - log_domain_size;

            log_eta_conjecture.max(log_eta_side_condition)
        }
        SecurityAssumption::JohnsonBound => {
            // bits = field_bits + 7*(1 + log2(eta)) - log2(ell-1) - 2*log_d_star
            let log_ell_minus_1 = libm::log2((ell.saturating_sub(1)).max(1) as f64);
            (target_bits as f64 - field_size_bits as f64 + log_ell_minus_1 + 2. * log_d_star as f64)
                / 7.
                - 1.
        }
        SecurityAssumption::UniqueDecoding => {
            panic!("STIR's paper-backed parameter schedule does not support UniqueDecoding")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn johnson_initial_eta_is_derived_from_bcss25_validation_bound() {
        let jb = SecurityAssumption::JohnsonBound;

        for (target, field_bits) in [(100, 155), (128, 192)] {
            let eta = jb.stir_initial_eta(target, target, 20, 2, 2, field_bits);
            assert!(jb.stir_eta_is_valid(2, eta));
            assert!(
                jb.fold_algebraic_bits_at_log_eta(field_bits, 20, 2, libm::log2(eta))
                    >= target as f64
            );
            assert!(
                ood_error_at_log_eta(jb, 20, 2, field_bits, 1, libm::log2(eta)) >= target as f64
            );
        }
    }

    #[test]
    fn combine_min_log_eta_capacity_bound_matches_closed_form() {
        // bits(eta) = field_bits + log2(eta) - 2*log_inv_rate - log2(ell-1) - log_d_star, so
        // the minimum log2(eta) for a target is exactly solving that for log2(eta). Chosen so
        // the Lemma 7.3 side-condition floor (eta >= 2/|L_0| = 2^(1 - 21)) does not bind.
        let field_bits = 155;
        let log_inv_rate = 1;
        let log_d_star = 20;
        let ell = 1 << 20;
        let target_bits = 100;
        let log_eta = combine_min_log_eta(
            SecurityAssumption::CapacityBound,
            field_bits,
            log_inv_rate,
            log_d_star,
            ell,
            target_bits,
        );
        let bits = target_bits as f64
            - (field_bits as f64 + log_eta
                - 2. * log_inv_rate as f64
                - libm::log2((ell - 1) as f64)
                - log_d_star as f64);
        assert!(
            libm::fabs(bits) < 1e-9,
            "closed form does not round-trip: {bits}"
        );
    }

    #[test]
    fn combine_min_log_eta_capacity_bound_relaxes_with_smaller_ell() {
        // A smaller ell needs a smaller (more negative) minimum log2(eta). Both values stay
        // clear of the Lemma 7.3 side-condition floor (eta >= 2/|L_0| = 2^(1 - 21)).
        let larger =
            combine_min_log_eta(SecurityAssumption::CapacityBound, 155, 1, 20, 1 << 20, 100);
        let smaller =
            combine_min_log_eta(SecurityAssumption::CapacityBound, 155, 1, 20, 1 << 19, 100);
        assert!(smaller < larger);
    }

    #[test]
    fn combine_min_log_eta_johnson_bound_increases_with_more_groups() {
        // A larger ell (more/taller classes) demands a larger minimum eta.
        let fewer = combine_min_log_eta(SecurityAssumption::JohnsonBound, 192, 1, 20, 1 << 21, 100);
        let more = combine_min_log_eta(SecurityAssumption::JohnsonBound, 192, 1, 20, 1 << 22, 100);
        assert!(more > fewer);
    }

    #[test]
    #[should_panic(expected = "does not support UniqueDecoding")]
    fn combine_min_log_eta_rejects_unique_decoding() {
        combine_min_log_eta(SecurityAssumption::UniqueDecoding, 192, 1, 20, 4, 100);
    }

    #[test]
    fn query_pow_does_not_credit_ood_or_shake_checks() {
        let cb = SecurityAssumption::CapacityBound;
        let eligible = cb.stir_query_pow_eligible_bits(124, 20, 2, 0.01, 40, 2);
        let unprotected = cb.stir_query_unprotected_bits(124, 20, 2, 0.01, 40, 2);

        assert!(eligible.is_finite());
        assert!(unprotected.is_finite());
        assert_eq!(
            unprotected,
            ood_error_at_log_eta(cb, 20, 2, 124, 2, libm::log2(0.01))
                .min(shake_check_error(124, 40, 2))
        );
    }

    #[test]
    fn smaller_eta_reduces_list_driven_security() {
        let cb = SecurityAssumption::CapacityBound;
        let safe = libm::log2(cb.stir_eta_upper_bound(1));
        let smaller = safe - 20.;

        assert!(
            prox_gaps_error_at_log_eta(cb, 20, 1, 155, 2, smaller)
                < prox_gaps_error_at_log_eta(cb, 20, 1, 155, 2, safe)
        );
        assert!(
            ood_error_at_log_eta(cb, 20, 1, 155, 2, smaller)
                < ood_error_at_log_eta(cb, 20, 1, 155, 2, safe)
        );
    }
}
