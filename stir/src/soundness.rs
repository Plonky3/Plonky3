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

    fn stir_combine_eta(
        &self,
        field_size_bits: usize,
        log_inv_rate: usize,
        log_d_star: usize,
        ell: u64,
        target_bits: usize,
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

/// Log2 of the exceptional set a proximity-gaps argument charges over a degree-`2^log_degree`
/// code at rate `2^-log_inv_rate`, evaluated at distance `delta = 1 - B*(rho) - eta`.
///
/// Every proximity-gaps-shaped error in this file is this quantity plus `log2(multiplicity - 1)`
/// subtracted from the field size. Both batching arguments STIR runs — the random linear
/// combination of `num_functions` oracles, and §7's `ell`-fold batch degree correction — differ
/// only in that multiplicity, so they share one bound per regime instead of each deriving its
/// own.
fn prox_gaps_exceptional_set_bits_at_log_eta(
    assumption: SecurityAssumption,
    log_degree: usize,
    log_inv_rate: usize,
    log_eta: f64,
) -> f64 {
    match assumption {
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

    let exceptional_set_bits =
        prox_gaps_exceptional_set_bits_at_log_eta(assumption, log_degree, log_inv_rate, log_eta);

    field_size_bits as f64 - (exceptional_set_bits + libm::log2(num_functions as f64 - 1.))
}

/// Algebraic bits §7 Construction 7.2's batch degree correction retains merging classes of total
/// multiplicity `ell` into a single codeword of degree `2^log_d_star`, at `delta = 1 - B*(rho) -
/// eta`.
///
/// This is [`prox_gaps_error_at_log_eta`] with the linear combination's `num_functions - 1`
/// replaced by Lemma 4.13's degree-gap-inflated `ell - 1`: `err*` is the §4.1 abstraction both
/// lemmas invoke, so the conjectured route (Conjecture 5.6, `CapacityBound`) and the provable one
/// (BCSS25 Theorem 1.5, `JohnsonBound`) each inflate by `ell` exactly as they do by the oracle
/// count.
fn combine_error_at_log_eta(
    assumption: SecurityAssumption,
    log_d_star: usize,
    log_inv_rate: usize,
    field_size_bits: usize,
    ell: u64,
    log_eta: f64,
) -> f64 {
    let exceptional_set_bits =
        prox_gaps_exceptional_set_bits_at_log_eta(assumption, log_d_star, log_inv_rate, log_eta);

    field_size_bits as f64
        - (exceptional_set_bits + libm::log2(ell.saturating_sub(1).max(1) as f64))
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

    /// Smallest `eta` at which §7 Construction 7.2's batch degree correction ("Combine")
    /// retains `target_bits` merging classes of total multiplicity `ell` into a single
    /// codeword of degree `2^log_d_star`.
    ///
    /// Returning `eta` rather than a bound on it lets `StirConfig` fold this into the
    /// schedule's round-0 `eta` with a plain `.max()`. Sharing that one variable is what pins
    /// `delta_combine` to exactly the schedule's own `delta_0`: Combine's guarantee ("if the
    /// merged codeword passes, each class has correlated agreement") is sound only if it
    /// reaches at least as far as STIR's round-0 acceptance radius, and one variable makes
    /// that automatic rather than a second value to keep in sync.
    ///
    /// Combine runs once, before the query phase's grind, so it is not PoW-eligible and
    /// `target_bits` must be the full buffered target, as for OOD and the shake check.
    ///
    /// `CapacityBound` additionally carries Lemma 7.3's `delta < 1 - rho - 1/|L_0|` side
    /// condition, which under `delta = 1 - rho - eta` is the floor `eta >= 2/|L_0|` — the same
    /// margin `stir_recursive_eta`'s `CapacityBound` arm keeps for later rounds.
    ///
    /// # Feasibility
    ///
    /// `ell = Σᵢ (gapᵢ + 1)` with distinct power-of-two `dᵢ`, so every `gapᵢ` is within a
    /// factor of two of `d*` and `log2(ell)` is close to `log_d_star` however tight the height
    /// spread is. Combine therefore costs roughly `2·log_d_star` bits of field regardless of
    /// how the classes are chosen, and is feasible only on wide challenge fields — see
    /// [`crate::StirParameters`] for the closed form. Under `JohnsonBound` it does not fit at
    /// production scale at all: the largest permitted `eta = sqrt(rho)/20` pins BCSS25's
    /// multiplicity at `m = 10`, which at `log_d_star = 20`, `log_inv_rate = 1` and a 155-bit
    /// challenge field retains only ~95 bits.
    ///
    /// # Panics
    ///
    /// If no permitted `eta` reaches `target_bits`.
    fn stir_combine_eta(
        &self,
        field_size_bits: usize,
        log_inv_rate: usize,
        log_d_star: usize,
        ell: u64,
        target_bits: usize,
    ) -> f64 {
        let eta = minimum_eta_for_target(
            self.stir_eta_upper_bound(log_inv_rate),
            target_bits,
            |eta| {
                combine_error_at_log_eta(
                    *self,
                    log_d_star,
                    log_inv_rate,
                    field_size_bits,
                    ell,
                    libm::log2(eta),
                )
            },
            "Combine batch-degree-correction bound",
        );

        match self {
            // `2/|L_0|` with `|L_0| = 2^(log_d_star + log_inv_rate)`, round 0's domain size.
            Self::CapacityBound => eta.max(libm::pow(2., 1. - (log_d_star + log_inv_rate) as f64)),
            // The Johnson regime satisfies the `1/|L_0|` side condition automatically.
            Self::JohnsonBound => eta,
            Self::UniqueDecoding => {
                panic!("STIR's paper-backed parameter schedule does not support UniqueDecoding")
            }
        }
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
    fn combine_error_matches_prox_gaps_error_at_matching_multiplicity() {
        // Combine charges the same exceptional set as the linear-combination proximity-gaps
        // argument, with `num_functions - 1` replaced by `ell - 1`. At `ell = num_functions`
        // the two must therefore agree exactly, in both regimes — this is what keeps the file
        // from deriving eta off two different Johnson-regime bounds.
        for assumption in [
            SecurityAssumption::JohnsonBound,
            SecurityAssumption::CapacityBound,
        ] {
            for log_inv_rate in [1, 2, 4] {
                for log_d_star in [10, 20, 24] {
                    for num_functions in [2usize, 3, 17, 1024] {
                        for log_eta in [-4.822, -8., -13.5] {
                            assert_eq!(
                                combine_error_at_log_eta(
                                    assumption,
                                    log_d_star,
                                    log_inv_rate,
                                    155,
                                    num_functions as u64,
                                    log_eta,
                                ),
                                prox_gaps_error_at_log_eta(
                                    assumption,
                                    log_d_star,
                                    log_inv_rate,
                                    155,
                                    num_functions,
                                    log_eta,
                                ),
                                "{assumption:?} lir={log_inv_rate} d*={log_d_star} \
                                 nf={num_functions} log_eta={log_eta}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn combine_eta_capacity_bound_matches_closed_form() {
        // bits(eta) = field_bits + log2(eta) - 2*log_inv_rate - log2(ell-1) - log_d_star, so
        // the minimum eta for a target is exactly solving that for log2(eta). Chosen so the
        // Lemma 7.3 side-condition floor (eta >= 2/|L_0| = 2^(1 - 21)) does not bind.
        let field_bits = 155;
        let log_inv_rate = 1;
        let log_d_star = 20;
        let ell = 1 << 20;
        let target_bits = 100;
        let log_eta = libm::log2(SecurityAssumption::CapacityBound.stir_combine_eta(
            field_bits,
            log_inv_rate,
            log_d_star,
            ell,
            target_bits,
        ));
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
    fn combine_eta_capacity_bound_relaxes_with_smaller_ell() {
        // A smaller ell needs a smaller minimum eta. Both values stay clear of the Lemma 7.3
        // side-condition floor (eta >= 2/|L_0| = 2^(1 - 21)).
        let cb = SecurityAssumption::CapacityBound;
        let larger = cb.stir_combine_eta(155, 1, 20, 1 << 20, 100);
        let smaller = cb.stir_combine_eta(155, 1, 20, 1 << 19, 100);
        assert!(smaller < larger);
    }

    #[test]
    fn combine_eta_capacity_bound_respects_side_condition_floor() {
        // At a tiny ell the conjectured term is far below the target's reach, so what binds is
        // Lemma 7.3's `delta < 1 - rho - 1/|L_0|` floor, `eta >= 2^(1 - (log_d_star + lir))`.
        let cb = SecurityAssumption::CapacityBound;
        let (log_d_star, log_inv_rate) = (20, 1);
        let eta = cb.stir_combine_eta(155, log_inv_rate, log_d_star, 2, 100);
        assert_eq!(eta, libm::pow(2., 1. - (log_d_star + log_inv_rate) as f64));
    }

    #[test]
    fn combine_eta_johnson_bound_increases_with_more_groups() {
        // A larger ell (more/taller classes) demands a larger minimum eta.
        let jb = SecurityAssumption::JohnsonBound;
        let fewer = jb.stir_combine_eta(192, 1, 20, 1 << 21, 100);
        let more = jb.stir_combine_eta(192, 1, 20, 1 << 22, 100);
        assert!(more > fewer);
    }

    #[test]
    fn combine_eta_johnson_bound_is_infeasible_at_production_scale() {
        // Documented consequence of deriving Combine from the same BCSS25 bound the rest of
        // the file validates against: at `d* = 2^20` on a 155-bit challenge field, the largest
        // permitted eta pins `m = 10` and retains well under the buffered 100-bit target.
        let jb = SecurityAssumption::JohnsonBound;
        let (log_inv_rate, log_d_star, ell) = (1, 20, (1u64 << 20) + (1 << 19) + 3);
        let upper = jb.stir_eta_upper_bound(log_inv_rate);
        let bits =
            combine_error_at_log_eta(jb, log_d_star, log_inv_rate, 155, ell, libm::log2(upper));
        assert!(
            bits < 100.,
            "expected JB + Combine to fall short, got {bits}"
        );
    }

    #[test]
    #[should_panic(expected = "does not support UniqueDecoding")]
    fn combine_eta_rejects_unique_decoding() {
        SecurityAssumption::UniqueDecoding.stir_combine_eta(192, 1, 20, 4, 100);
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
