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
