//! Soundness analysis and security assumption models for Reed-Solomon proximity testing.

use alloc::format;
use alloc::string::String;
use core::f64::consts::LOG2_10;
use core::fmt::Display;
use core::str::FromStr;

use serde::Serialize;

/// Security assumption model for RS proximity testing, determining which proximity parameters and conjectures are assumed.
///
/// # References
///
/// The proximity gaps analysis is based on:
/// - **[BCI+20]**: Ben-Sasson, Carmon, Ishai, Kopparty, Saraf. "Proximity Gaps for Reed-Solomon Codes".
///   FOCS 2020. <https://eprint.iacr.org/2020/654>
/// - **[BCSS25]**: Ben-Sasson, Carmon, Haboeck, Kopparty, Saraf. "On Proximity Gaps for Reed-Solomon Codes".
///   <https://eprint.iacr.org/2025/2055>
///
/// The [BCSS25] paper significantly improves the Johnson bound proximity gaps from `O(n^2/eta^7)` to `O(n/eta^5)`,
/// enabling provable 128-bit security with smaller extension fields (e.g., degree-5 extension of KoalaBear).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum SecurityAssumption {
    /// Unique decoding assumes that the distance of each oracle is within the UDR of the code.
    /// We refer to this configuration as UD for short.
    /// This requires no conjectures.
    UniqueDecoding,

    /// Johnson bound assumes that the distance of each oracle is within the Johnson bound (1 - sqrt(rho) - eta).
    /// With eta = sqrt(rho)/20.
    /// We refer to this configuration as JB for short.
    /// This assumes that RS have mutual correlated agreement for proximity parameter up to (1 - sqrt(rho) - eta).
    JohnsonBound,

    /// Capacity bound assumes that the distance of each oracle is within the capacity bound (1 - rho - eta).
    /// With eta = rho/20.
    /// We refer to this configuration as CB for short.
    /// This requires conjecturing that RS codes are decodable up to capacity and have correlated agreement (mutual in WHIR) up to capacity.
    CapacityBound,
}

impl SecurityAssumption {
    fn rate_from_log_inv_rate(log_inv_rate: usize) -> f64 {
        libm::pow(2., -(log_inv_rate as f64))
    }

    fn log2_sum(a: f64, b: f64) -> f64 {
        if a.is_infinite() && a.is_sign_negative() {
            return b;
        }
        if b.is_infinite() && b.is_sign_negative() {
            return a;
        }
        if a >= b {
            a + libm::log2(1. + libm::pow(2., b - a))
        } else {
            b + libm::log2(1. + libm::pow(2., a - b))
        }
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

    /// Fixed number of OOD samples `s` from the paper's recommended parameter schedule.
    ///
    /// - Johnson-bound soundness uses the provable schedule with `s = 1`.
    /// - Capacity-bound soundness uses the conjectured schedule with `s = 2`.
    ///
    /// Unique decoding is not covered by the paper's recommended STIR schedule.
    #[must_use]
    pub fn stir_num_ood_samples(&self) -> usize {
        match self {
            Self::JohnsonBound => 1,
            Self::CapacityBound => 2,
            Self::UniqueDecoding => {
                panic!("STIR's paper-backed parameter schedule does not support UniqueDecoding")
            }
        }
    }

    /// The per-query failure base from the paper's recommended STIR schedule.
    ///
    /// For the provable regime this is `sqrt(rho) + eta_i`.
    /// For the conjectured regime this is `rho + eta_i`.
    #[must_use]
    pub fn stir_query_failure_base(&self, log_inv_rate: usize, eta: f64) -> f64 {
        match self {
            Self::JohnsonBound => libm::sqrt(Self::rate_from_log_inv_rate(log_inv_rate)) + eta,
            Self::CapacityBound => Self::rate_from_log_inv_rate(log_inv_rate) + eta,
            Self::UniqueDecoding => {
                panic!("STIR's paper-backed parameter schedule does not support UniqueDecoding")
            }
        }
    }

    /// The appendix-level upper bound imposed on the recursively chosen `eta_i`.
    ///
    /// Appendix C.1 shows the provable schedule under `eta_i <= sqrt(rho_i) / 20`.
    /// Appendix C.2 shows the conjectured schedule under `eta_i <= rho_i / 2`.
    #[must_use]
    pub fn stir_eta_upper_bound(&self, log_inv_rate: usize) -> f64 {
        match self {
            Self::JohnsonBound => libm::sqrt(Self::rate_from_log_inv_rate(log_inv_rate)) / 20.,
            Self::CapacityBound => Self::rate_from_log_inv_rate(log_inv_rate) / 2.,
            Self::UniqueDecoding => {
                panic!("STIR's paper-backed parameter schedule does not support UniqueDecoding")
            }
        }
    }

    /// Returns whether `eta` satisfies the appendix-level side condition needed by the
    /// paper's recommended STIR schedule.
    #[must_use]
    pub fn stir_eta_is_valid(&self, log_inv_rate: usize, eta: f64) -> bool {
        eta.is_finite() && eta > 0. && eta <= self.stir_eta_upper_bound(log_inv_rate)
    }

    /// Initial `eta_0` from §5.3's recommended STIR schedule.
    ///
    /// **Deviation from the paper.** The paper's formula solves only the prox-gap
    /// constraint; under eta-aware accounting that leaves `queries_combination_error`
    /// at round 0 below `target_bits` by `~log_2(t_0)` bits. We extend the formula to
    /// `max(prox_gap_term, joint_combination_term)`, mirroring `stir_recursive_eta`'s
    /// shape with `t_{-1}` replaced by a closed-form upper bound on `t_0` (computed at
    /// `eta = eta_upper_bound`, where `failure_base` and thus `t_0` are maximized).
    #[must_use]
    pub fn stir_initial_eta(
        &self,
        security_bits: usize,
        log_degree: usize,
        log_inv_rate: usize,
        log_folding_factor: usize,
        field_size_bits: usize,
    ) -> f64 {
        let k = 1usize << log_folding_factor;
        let log_k_minus_1 = libm::log2((k - 1) as f64);
        let log_d_over_k = (log_degree - log_folding_factor) as f64;
        let rho = Self::rate_from_log_inv_rate(log_inv_rate);

        let log_eta = match self {
            // η₀ := (2^λ (k - 1) (d/k)^2 / (2^7 |F|))^(1/7)
            Self::JohnsonBound => {
                let log_eta_proxgap = ((security_bits as f64) + log_k_minus_1 + 2. * log_d_over_k
                    - 7.
                    - field_size_bits as f64)
                    / 7.;

                // Round-0 analogue of `stir_recursive_eta`'s 7th-root term.
                // `failure_base_max = √ρ · 1.05` at the JB eta upper bound `√ρ/20`.
                let log_failure_base_max = libm::log2(1.05 * libm::sqrt(rho));
                let t_0_max = libm::ceil(security_bits as f64 / -log_failure_base_max);
                let log_t_term = libm::log2(t_0_max) + 2. * log_degree as f64;
                let log_k_term = log_k_minus_1 + 2. * log_d_over_k;
                let log_sum = Self::log2_sum(log_t_term, log_k_term);
                let log_eta_combination =
                    ((security_bits as f64) + 1. + log_sum - 7. - field_size_bits as f64) / 7.;

                log_eta_proxgap.max(log_eta_combination)
            }
            // With c₁ = c₂ = 1:
            // η₀ := 2^λ (k - 1) (d/k) / (ρ² |F|)
            Self::CapacityBound => {
                let log_eta_proxgap = (security_bits as f64)
                    + log_k_minus_1
                    + log_d_over_k
                    + 2. * (log_inv_rate as f64)
                    - field_size_bits as f64;

                // Round-0 analogue of `stir_recursive_eta`'s third term.
                // `failure_base_max = 1.5·ρ` at the CB eta upper bound `ρ/2`.
                let log_failure_base_max = libm::log2(1.5 * rho);
                let t_0_max = libm::ceil(security_bits as f64 / -log_failure_base_max);
                let third_factor = (t_0_max + 1.) + (k - 1) as f64 / k as f64;
                let log_eta_combination =
                    (security_bits as f64) + 1. + log_degree as f64 + 2. * (log_inv_rate as f64)
                        - field_size_bits as f64
                        + libm::log2(third_factor);

                log_eta_proxgap.max(log_eta_combination)
            }
            Self::UniqueDecoding => {
                panic!("STIR's paper-backed parameter schedule does not support UniqueDecoding")
            }
        };

        libm::pow(2., log_eta)
    }

    /// Recursive `eta_i` from §5.3's recommended STIR schedule.
    ///
    /// `prev_queries` is `t_{i-1}` from the previous stage.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn stir_recursive_eta(
        &self,
        security_bits: usize,
        log_degree: usize,
        log_inv_rate: usize,
        log_domain_size: usize,
        log_folding_factor: usize,
        field_size_bits: usize,
        prev_queries: usize,
    ) -> f64 {
        let k = 1usize << log_folding_factor;
        let log_domain = log_domain_size as f64;
        let log_field_minus_domain =
            Self::log2_field_minus_domain(field_size_bits, log_domain_size);

        match self {
            Self::JohnsonBound => {
                // max(
                //   sqrt(2^λ d_i / (8 ρ_i (|F| - |L_i|))),
                //   (2^(λ+1) (t_{i-1} d_i^2 + (k-1) (d_i / k)^2) / (2^7 |F|))^(1/7)
                // )
                let log_term_1 = ((security_bits as f64) + log_degree as f64 - 3.
                    + log_inv_rate as f64
                    - log_field_minus_domain)
                    / 2.;

                let log_prev_queries_piece =
                    libm::log2(prev_queries as f64) + 2. * log_degree as f64;
                let log_k_term =
                    libm::log2((k - 1) as f64) + 2. * (log_degree - log_folding_factor) as f64;
                let log_sum = Self::log2_sum(log_prev_queries_piece, log_k_term);
                let log_term_2 =
                    ((security_bits as f64) + 1. + log_sum - 7. - field_size_bits as f64) / 7.;

                libm::pow(2., log_term_1.max(log_term_2))
            }
            Self::CapacityBound => {
                // With c₁ = c₂ = c₃ = 1:
                // max(
                //   2ρ_i / d_i,
                //   (d_i / ρ_i) * sqrt(2^λ d_i^2 / (2 (|F| - |L_i|)^2)),
                //   2^(λ+1) d_i / (ρ_i² |F|)
                //     * ((t_{i-1} + 1) + (k - 1)/k)
                // )
                let log_term_1 = 1. - log_domain;

                let log_term_2 = log_domain
                    + ((security_bits as f64) + 2. * log_degree as f64
                        - 1.
                        - 2. * log_field_minus_domain)
                        / 2.;

                let third_factor = (prev_queries + 1) as f64 + (k - 1) as f64 / k as f64;
                let log_term_3 =
                    (security_bits as f64) + 1. + log_degree as f64 + 2. * (log_inv_rate as f64)
                        - field_size_bits as f64
                        + libm::log2(third_factor);

                libm::pow(2., log_term_1.max(log_term_2).max(log_term_3))
            }
            Self::UniqueDecoding => {
                panic!("STIR's paper-backed parameter schedule does not support UniqueDecoding")
            }
        }
    }

    /// STIR query count for a stage whose per-query failure base is known.
    ///
    /// This is the `ceil(target / -log2(base))` form used by §5.3.
    #[must_use]
    pub fn stir_queries_for_base(&self, security_bits: usize, failure_base: f64) -> usize {
        let _ = self;
        Self::query_count_from_failure_base(security_bits, failure_base)
    }

    /// In both JB and CB theorems such as list-size only hold for proximity parameters slightly below the bound.
    /// E.g. in JB proximity gaps holds for every delta in (0, 1 - sqrt(rho)).
    /// eta is the distance between the chosen proximity parameter and the bound.
    /// I.e. in JB delta = 1 - sqrt(rho) - eta and in CB delta = 1 - rho - eta.
    ///
    /// # Panics
    ///
    /// Panics when called with [`SecurityAssumption::UniqueDecoding`]: eta is
    /// undefined in UD (`delta = (1 - rho) / 2`, with no eta term). All
    /// current callers special-case the UD branch before reaching the
    /// eta-dependent arithmetic; the panic locks down that invariant so a
    /// future refactor that strays into the eta path under UD fails loudly.
    #[must_use]
    pub const fn log_eta(&self, log_inv_rate: usize) -> f64 {
        match self {
            // We don't use eta in UD
            Self::UniqueDecoding => panic!("log_eta is undefined for UniqueDecoding"),
            // Set as sqrt(rho)/20
            Self::JohnsonBound => -(0.5 * log_inv_rate as f64 + LOG2_10 + 1.),
            // Set as rho/20
            Self::CapacityBound => -(log_inv_rate as f64 + LOG2_10 + 1.),
        }
    }

    /// Given a RS code (specified by the log of the degree and log inv of the rate), compute the list size at the specified distance delta.
    #[must_use]
    pub const fn list_size_bits(&self, log_degree: usize, log_inv_rate: usize) -> f64 {
        match self {
            // In UD the list size is 1
            Self::UniqueDecoding => 0.,

            // By the JB, RS codes are (1 - sqrt(rho) - eta, (2*eta*sqrt(rho))^-1)-list decodable.
            Self::JohnsonBound => {
                let log_eta = self.log_eta(log_inv_rate);
                let log_inv_sqrt_rate: f64 = log_inv_rate as f64 / 2.;
                log_inv_sqrt_rate - (1. + log_eta)
            }

            // In CB we assume that RS codes are (1 - rho - eta, d/rho*eta)-list decodable (see Conjecture 5.6 in STIR).
            Self::CapacityBound => (log_degree + log_inv_rate) as f64 - self.log_eta(log_inv_rate),
        }
    }

    /// Like [`Self::list_size_bits`] but evaluated at an explicit `log_eta` rather than the
    /// appendix-level safe gap returned by [`Self::log_eta`].
    ///
    /// Use this when the protocol commits to a specific per-round `eta_i` (e.g. STIR §5.3's
    /// recommended schedule). The UD branch ignores `log_eta` because UD's `delta = (1 - rho) / 2`
    /// has no eta term — list size is always 1.
    #[must_use]
    pub const fn list_size_bits_at_log_eta(
        &self,
        log_degree: usize,
        log_inv_rate: usize,
        log_eta: f64,
    ) -> f64 {
        match self {
            Self::UniqueDecoding => 0.,
            Self::JohnsonBound => {
                let log_inv_sqrt_rate: f64 = log_inv_rate as f64 / 2.;
                log_inv_sqrt_rate - (1. + log_eta)
            }
            Self::CapacityBound => (log_degree + log_inv_rate) as f64 - log_eta,
        }
    }

    /// Given a RS code (specified by the log of the degree and log inv of the rate) a field_size
    /// and an arity, compute the proximity gaps error (in bits) at the specified distance.
    #[must_use]
    pub fn prox_gaps_error(
        &self,
        log_degree: usize,
        log_inv_rate: usize,
        field_size_bits: usize,
        num_functions: usize,
    ) -> f64 {
        assert!(
            num_functions >= 2,
            "num_functions must be >= 2 to compute proximity gaps error",
        );

        // Note that this does not include the field_size
        let error = match self {
            // In UD the error is |L|/|F| = d/(rho*|F|)
            Self::UniqueDecoding => (log_degree + log_inv_rate) as f64,

            // From Theorem 1.5 in [BCSS25] "On Proximity Gaps for Reed-Solomon Codes":
            //
            // For gamma < J(delta) - eta, the number of exceptional z's is bounded by:
            //   a > (2(m + 1/2)^5 + 3(m + 1/2)*gamma*rho) / (3*rho^(3/2)) * n + (m + 1/2) / sqrt(rho)
            //
            // With eta = sqrt(rho)/20 (safe gap), m = max(ceil(sqrt(rho)/(2*eta)), 3) = max(10, 3) = 10.
            //
            // Only the first (dominant) term is kept.
            // The second additive term (m + 1/2) / sqrt(rho) is O(1), negligible for large n.
            // Within the first term, the sub-term 3*(m + 1/2)*gamma*rho is also dropped
            // because 2*(m + 1/2)^5 dominates it when m = 10.
            //
            // This gives the approximation:
            //
            //   a ~ (2 * 10.5^5) / (3 * rho^(3/2)) * n
            //
            // In log form:
            //   log_2(a) = log_2(n) + log_2(2 * 10.5^5 / 3) + 1.5 * log_2(1/rho)
            //            = (log_degree + log_inv_rate) + 16.38 + 1.5 * log_inv_rate
            //            = log_degree + 2.5 * log_inv_rate + 16.38
            //
            // This improves over [BCI+20] which had:
            //   log_2(a) = 2*log_degree + 3.5*log_inv_rate + 23.24
            Self::JohnsonBound => {
                // n = 2^(log_degree + log_inv_rate)
                let log_n = (log_degree + log_inv_rate) as f64;

                // Constant from (2 * 10.5^5 / 3)
                let constant = libm::log2(2. * libm::pow(10.5, 5.) / 3.);

                // rho^(-3/2) contributes 1.5 * log_inv_rate
                let log_rho_neg_3_2 = 1.5 * log_inv_rate as f64;

                log_n + constant + log_rho_neg_3_2
            }

            // In CB we assume the error is degree/(eta*rho^2)
            Self::CapacityBound => {
                (log_degree + 2 * log_inv_rate) as f64 - self.log_eta(log_inv_rate)
            }
        };

        // Error is (num_functions - 1) * error/|F|;
        let num_functions_1_log = libm::log2(num_functions as f64 - 1.);
        field_size_bits as f64 - (error + num_functions_1_log)
    }

    /// Like [`Self::prox_gaps_error`] but evaluated at an explicit `log_eta`.
    ///
    /// For Johnson bound, the BCSS25 multiplicity is recomputed from the actual eta:
    /// `m = max(ceil(sqrt(rho) / (2 * eta)), 3)`. At `log_eta = log2(sqrt(rho) / 20)` this
    /// collapses to `m = 10`, matching the constant baked into [`Self::prox_gaps_error`].
    /// Smaller `eta` yields larger `m` and thus a larger exceptional-set lower bound (fewer
    /// bits of security).
    #[must_use]
    pub fn prox_gaps_error_at_log_eta(
        &self,
        log_degree: usize,
        log_inv_rate: usize,
        field_size_bits: usize,
        num_functions: usize,
        log_eta: f64,
    ) -> f64 {
        assert!(
            num_functions >= 2,
            "num_functions must be >= 2 to compute proximity gaps error",
        );

        let error = match self {
            Self::UniqueDecoding => (log_degree + log_inv_rate) as f64,
            Self::JohnsonBound => {
                // m = max(ceil(sqrt(rho) / (2 * eta)), 3).
                // log2(sqrt(rho) / (2 * eta)) = -log_inv_rate/2 - 1 - log_eta.
                let log_sqrt_rho_over_2eta = -(log_inv_rate as f64) / 2. - 1. - log_eta;
                let m_candidate = libm::ceil(libm::pow(2., log_sqrt_rho_over_2eta));
                let m = m_candidate.max(3.);
                let m_plus_half = m + 0.5;

                let log_n = (log_degree + log_inv_rate) as f64;
                let constant = libm::log2(2. * libm::pow(m_plus_half, 5.) / 3.);
                let log_rho_neg_3_2 = 1.5 * log_inv_rate as f64;
                log_n + constant + log_rho_neg_3_2
            }
            Self::CapacityBound => (log_degree + 2 * log_inv_rate) as f64 - log_eta,
        };

        let num_functions_1_log = libm::log2(num_functions as f64 - 1.);
        field_size_bits as f64 - (error + num_functions_1_log)
    }

    /// The query error is (1 - delta)^t where t is the number of queries.
    /// This computes log(1 - delta).
    /// - In UD, delta is (1 - rho)/2
    /// - In JB, delta is (1 - sqrt(rho) - eta)
    /// - In CB, delta is (1 - rho - eta)
    #[must_use]
    pub fn log_1_delta(&self, log_inv_rate: usize) -> f64 {
        let rate = 1. / f64::from(1 << log_inv_rate);

        let delta = match self {
            Self::UniqueDecoding => 0.5 * (1. - rate),
            Self::JohnsonBound => 1. - libm::sqrt(rate) - libm::pow(2., self.log_eta(log_inv_rate)),
            Self::CapacityBound => 1. - rate - libm::pow(2., self.log_eta(log_inv_rate)),
        };

        libm::log2(1. - delta)
    }

    /// Compute the number of queries needed to achieve the given security level.
    #[must_use]
    pub fn queries(&self, protocol_security_level: usize, log_inv_rate: usize) -> usize {
        let num_queries_f = -(protocol_security_level as f64) / self.log_1_delta(log_inv_rate);

        libm::ceil(num_queries_f) as usize
    }

    /// Compute the query soundness error (in bits) for the given number of queries.
    #[must_use]
    pub fn queries_error(&self, log_inv_rate: usize, num_queries: usize) -> f64 {
        let num_queries = num_queries as f64;

        -num_queries * self.log_1_delta(log_inv_rate)
    }

    /// Compute the error for the OOD samples of the protocol.
    ///
    /// See Lemma 4.5 in STIR.
    /// The error is list_size^2 * (degree/field_size_bits)^reps.
    /// NOTE: Here we are discounting the domain size as we assume it is negligible compared to the size of the field.
    #[must_use]
    pub const fn ood_error(
        &self,
        log_degree: usize,
        log_inv_rate: usize,
        field_size_bits: usize,
        ood_samples: usize,
    ) -> f64 {
        if matches!(self, Self::UniqueDecoding) {
            return 0.;
        }

        let list_size_bits = self.list_size_bits(log_degree, log_inv_rate);

        let error = 2. * list_size_bits + (log_degree * ood_samples) as f64;
        (ood_samples * field_size_bits) as f64 + 1. - error
    }

    /// Like [`Self::ood_error`] but evaluated at an explicit `log_eta`.
    #[must_use]
    pub const fn ood_error_at_log_eta(
        &self,
        log_degree: usize,
        log_inv_rate: usize,
        field_size_bits: usize,
        ood_samples: usize,
        log_eta: f64,
    ) -> f64 {
        if matches!(self, Self::UniqueDecoding) {
            return 0.;
        }

        let list_size_bits = self.list_size_bits_at_log_eta(log_degree, log_inv_rate, log_eta);

        let error = 2. * list_size_bits + (log_degree * ood_samples) as f64;
        (ood_samples * field_size_bits) as f64 + 1. - error
    }

    /// Computes the number of OOD samples required to achieve `security_level` bits of security.
    #[must_use]
    pub fn determine_ood_samples(
        &self,
        security_level: usize,
        log_degree: usize,
        log_inv_rate: usize,
        field_size_bits: usize,
    ) -> usize {
        if matches!(self, Self::UniqueDecoding) {
            return 0;
        }

        for ood_samples in 1..64 {
            if self.ood_error(log_degree, log_inv_rate, field_size_bits, ood_samples)
                >= security_level as f64
            {
                return ood_samples;
            }
        }

        panic!("Could not find an appropriate number of OOD samples");
    }

    /// Compute the sumcheck soundness term of the folding step (in bits).
    ///
    /// During folding, the verifier samples a random challenge and checks
    /// a degree-2 sumcheck identity. An adversary controlling a list of
    /// L codewords can bias the check with probability at most `L / |F|`.
    ///
    /// In log form:
    ///
    /// ```text
    /// bits_of_security = field_size_bits - (list_size_bits + 1)
    /// ```
    ///
    /// The `+1` accounts for the union bound over the list.
    #[must_use]
    pub const fn fold_sumcheck_error(
        &self,
        field_size_bits: usize,
        num_variables: usize,
        log_inv_rate: usize,
    ) -> f64 {
        let list_size = self.list_size_bits(num_variables, log_inv_rate);

        field_size_bits as f64 - (list_size + 1.)
    }

    /// Like [`Self::fold_sumcheck_error`] but evaluated at an explicit `log_eta`.
    #[must_use]
    pub const fn fold_sumcheck_error_at_log_eta(
        &self,
        field_size_bits: usize,
        num_variables: usize,
        log_inv_rate: usize,
        log_eta: f64,
    ) -> f64 {
        let list_size = self.list_size_bits_at_log_eta(num_variables, log_inv_rate, log_eta);

        field_size_bits as f64 - (list_size + 1.)
    }

    /// Compute the soundness error (in bits) of the query-combination step.
    ///
    /// After STIR queries and OOD samples, the verifier takes a random
    /// linear combination of all collected evaluations. An adversary must
    /// fool this combination for every codeword in the list, giving error:
    ///
    /// ```text
    /// error = (ood_samples + num_queries) * list_size / |F|
    /// ```
    ///
    /// In log form (all quantities in bits):
    ///
    /// ```text
    /// bits_of_security = field_size_bits - (log_2(ood + queries) + list_size_bits + 1)
    /// ```
    ///
    /// The `+1` accounts for the union bound over list elements.
    #[must_use]
    pub fn queries_combination_error(
        &self,
        field_size_bits: usize,
        num_variables: usize,
        log_inv_rate: usize,
        ood_samples: usize,
        num_queries: usize,
    ) -> f64 {
        let list_size = self.list_size_bits(num_variables, log_inv_rate);

        let log_combination = libm::log2((ood_samples + num_queries) as f64);

        field_size_bits as f64 - (log_combination + list_size + 1.)
    }

    /// Like [`Self::queries_combination_error`] but evaluated at an explicit `log_eta`.
    #[must_use]
    pub fn queries_combination_error_at_log_eta(
        &self,
        field_size_bits: usize,
        num_variables: usize,
        log_inv_rate: usize,
        ood_samples: usize,
        num_queries: usize,
        log_eta: f64,
    ) -> f64 {
        let list_size = self.list_size_bits_at_log_eta(num_variables, log_inv_rate, log_eta);

        let log_combination = libm::log2((ood_samples + num_queries) as f64);

        field_size_bits as f64 - (log_combination + list_size + 1.)
    }

    /// Compute the PoW difficulty needed for the folding step.
    ///
    /// The folding step has two independent error sources:
    /// - Proximity gaps: the probability that a far-from-RS function
    ///   survives the fold (depends on the code rate and list size).
    /// - Sumcheck: the probability that the sumcheck verifier accepts
    ///   a wrong claim (depends on the field size and list size).
    ///
    /// The overall folding error is limited by the weaker bound.
    /// PoW must bridge the gap to the target security level:
    ///
    /// ```text
    /// pow_bits = max(0, security_level - min(prox_gaps_error, sumcheck_error))
    /// ```
    ///
    /// Returns 0 when the algebraic bounds alone meet the target.
    #[must_use]
    pub fn folding_pow_bits(
        &self,
        security_level: usize,
        field_size_bits: usize,
        num_variables: usize,
        log_inv_rate: usize,
    ) -> f64 {
        let error = self.fold_algebraic_bits(field_size_bits, num_variables, log_inv_rate);
        0_f64.max(security_level as f64 - error)
    }

    /// Algebraic bits of security delivered by the STIR folding step.
    ///
    /// Returns the minimum (= worst) of the proximity-gaps error and the fold-sumcheck error,
    /// both expressed as bits of security (higher = better). The folding PoW must bridge from
    /// this value to `security_level`.
    #[must_use]
    pub fn fold_algebraic_bits(
        &self,
        field_size_bits: usize,
        num_variables: usize,
        log_inv_rate: usize,
    ) -> f64 {
        let prox_gaps = self.prox_gaps_error(num_variables, log_inv_rate, field_size_bits, 2);
        let sumcheck = self.fold_sumcheck_error(field_size_bits, num_variables, log_inv_rate);
        prox_gaps.min(sumcheck)
    }

    /// Like [`Self::fold_algebraic_bits`] but evaluated at an explicit `log_eta`.
    ///
    /// Use this when the protocol's per-round `eta_i` differs from [`Self::log_eta`] (STIR's
    /// §5.3 schedule does this; WHIR does not). Both children — proximity gaps and the
    /// fold-sumcheck list-size bound — are sensitive to `eta_i`, and using a fixed eta when the
    /// protocol commits to a smaller `eta_i` overstates security.
    #[must_use]
    pub fn fold_algebraic_bits_at_log_eta(
        &self,
        field_size_bits: usize,
        num_variables: usize,
        log_inv_rate: usize,
        log_eta: f64,
    ) -> f64 {
        let prox_gaps = self.prox_gaps_error_at_log_eta(
            num_variables,
            log_inv_rate,
            field_size_bits,
            2,
            log_eta,
        );
        let sumcheck = self.fold_sumcheck_error_at_log_eta(
            field_size_bits,
            num_variables,
            log_inv_rate,
            log_eta,
        );
        prox_gaps.min(sumcheck)
    }

    /// Algebraic bits of security delivered by an intermediate STIR query phase.
    ///
    /// Combines:
    /// - Query failure: `t · log2(1/(rho + eta))` bits (calibrated by `eta` and the query count).
    /// - OOD soundness: see [`Self::ood_error`].
    /// - Random-linear-combination soundness: see [`Self::queries_combination_error`].
    /// - Shake-check soundness: see [`Self::shake_check_error`].
    ///
    /// Returns the minimum (= the worst term, which dominates the per-round error).
    #[must_use]
    pub fn stir_query_algebraic_bits(
        &self,
        field_size_bits: usize,
        log_degree: usize,
        log_inv_rate: usize,
        eta: f64,
        num_queries: usize,
        num_ood_samples: usize,
    ) -> f64 {
        let failure_base = self.stir_query_failure_base(log_inv_rate, eta);
        assert!(
            failure_base > 0. && failure_base < 1.,
            "STIR query failure base must lie in (0, 1)"
        );
        let query_failure = -(num_queries as f64) * libm::log2(failure_base);
        // The OOD and combination terms must be evaluated at the same per-round eta the
        // protocol commits to (`stir_initial_eta` / `stir_recursive_eta`), not the appendix
        // safe gap. Using the safe gap when the protocol uses a much smaller `eta_i`
        // understates the list size and thus overstates security.
        let log_eta = libm::log2(eta);
        let ood = self.ood_error_at_log_eta(
            log_degree,
            log_inv_rate,
            field_size_bits,
            num_ood_samples,
            log_eta,
        );
        let combination = self.queries_combination_error_at_log_eta(
            field_size_bits,
            log_degree,
            log_inv_rate,
            num_ood_samples,
            num_queries,
            log_eta,
        );
        let shake = self.shake_check_error(field_size_bits, num_queries, num_ood_samples);
        query_failure.min(ood).min(combination).min(shake)
    }

    /// Schwartz-Zippel error (in bits) of the prover-assisted Ans/shake polynomial check.
    ///
    /// This implementation has the prover send `Ans` and a shake polynomial; the verifier
    /// then checks the rational identity
    /// `shake(rho) · Q(rho) = sum_i (Ans(rho) - val_i) · prod_{j != i} (rho - y_j)`
    /// at a transcript-derived random `rho`, where `Q = prod_i (X - y_i)` and the index `i`
    /// ranges over the `num_queries + num_ood_samples` interpolation points. The polynomial
    /// `shake · Q − sum_i (Ans − val_i) · prod_{j != i} (X − y_j)` has degree at most
    /// `2 · (num_queries + num_ood_samples) − 2`, so a malicious prover succeeds with
    /// probability at most that degree divided by `|F|`.
    ///
    /// This is an additional soundness term that the original STIR paper does not include
    /// — the paper's verifier interpolates `Ans` itself rather than receiving it from the
    /// prover. Including the shake-check error in [`Self::stir_query_algebraic_bits`] keeps
    /// the parameter accounting honest.
    #[must_use]
    pub fn shake_check_error(
        &self,
        field_size_bits: usize,
        num_queries: usize,
        num_ood_samples: usize,
    ) -> f64 {
        let total = (num_queries + num_ood_samples) as f64;
        // Conservative degree bound: 2 * num_points (covers the +O(1) slack).
        let log_deg = libm::log2(2.0 * total).max(0.0);
        field_size_bits as f64 - log_deg
    }

    /// Algebraic bits of security delivered by the STIR final query phase.
    ///
    /// The final round verifies queries directly against the sent final polynomial; there is no
    /// OOD or random-combination step. Returns the query-failure bits only.
    #[must_use]
    pub fn stir_final_query_algebraic_bits(
        &self,
        log_inv_rate: usize,
        eta: f64,
        num_queries: usize,
    ) -> f64 {
        let failure_base = self.stir_query_failure_base(log_inv_rate, eta);
        assert!(
            failure_base > 0. && failure_base < 1.,
            "STIR query failure base must lie in (0, 1)"
        );
        -(num_queries as f64) * libm::log2(failure_base)
    }
}

impl Display for SecurityAssumption {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(match self {
            Self::JohnsonBound => "JohnsonBound",
            Self::CapacityBound => "CapacityBound",
            Self::UniqueDecoding => "UniqueDecoding",
        })
    }
}

impl FromStr for SecurityAssumption {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "JohnsonBound" => Ok(Self::JohnsonBound),
            "CapacityBound" => Ok(Self::CapacityBound),
            "UniqueDecoding" => Ok(Self::UniqueDecoding),
            _ => Err(format!("Invalid soundness specification: {s}")),
        }
    }
}

#[cfg(test)]
#[allow(clippy::cast_lossless)]
mod tests {
    use alloc::string::ToString;

    use super::*;

    #[test]
    fn test_soundness_type_display() {
        assert_eq!(SecurityAssumption::JohnsonBound.to_string(), "JohnsonBound");
        assert_eq!(
            SecurityAssumption::CapacityBound.to_string(),
            "CapacityBound"
        );
        assert_eq!(
            SecurityAssumption::UniqueDecoding.to_string(),
            "UniqueDecoding"
        );
    }

    #[test]
    fn test_soundness_type_from_str() {
        assert_eq!(
            SecurityAssumption::from_str("JohnsonBound"),
            Ok(SecurityAssumption::JohnsonBound)
        );
        assert_eq!(
            SecurityAssumption::from_str("CapacityBound"),
            Ok(SecurityAssumption::CapacityBound)
        );
        assert_eq!(
            SecurityAssumption::from_str("UniqueDecoding"),
            Ok(SecurityAssumption::UniqueDecoding)
        );

        assert!(SecurityAssumption::from_str("InvalidType").is_err());
        assert!(SecurityAssumption::from_str("").is_err());
    }

    #[test]
    #[should_panic(expected = "num_functions must be >= 2")]
    fn prox_gaps_error_panics_when_num_functions_is_one() {
        let assumption = SecurityAssumption::UniqueDecoding;
        let _ = assumption.prox_gaps_error(1, 1, 64, 1);
    }

    #[test]
    #[should_panic(expected = "num_functions must be >= 2")]
    fn prox_gaps_error_panics_when_num_functions_is_zero() {
        let assumption = SecurityAssumption::UniqueDecoding;
        let _ = assumption.prox_gaps_error(1, 1, 64, 0);
    }

    #[test]
    fn test_ud_errors() {
        let assumption = SecurityAssumption::UniqueDecoding;

        let log_degree = 20;
        let degree = (1 << log_degree) as f64;
        let log_inv_rate = 2;
        let rate = 1. / (1 << log_inv_rate) as f64;

        let field_size_bits = 128;

        assert!(assumption.list_size_bits(log_degree, log_inv_rate) - 0. < 0.01);

        let computed_error =
            assumption.prox_gaps_error(log_degree, log_inv_rate, field_size_bits, 2);
        let real_error_non_log = degree / rate;
        let real_error = field_size_bits as f64 - real_error_non_log.log2();

        assert!((computed_error - real_error).abs() < 0.01);
    }

    #[test]
    fn test_jb_errors() {
        let assumption = SecurityAssumption::JohnsonBound;

        let log_degree = 20;
        let log_inv_rate = 2;
        let rate = 1. / (1 << log_inv_rate) as f64;

        let eta = rate.sqrt() / 20.;

        let field_size_bits = 128;

        let real_list_size = 1. / (2. * eta * rate.sqrt());
        let computed_list_size = assumption.list_size_bits(log_degree, log_inv_rate);
        assert!((real_list_size.log2() - computed_list_size).abs() < 0.01);

        let computed_error =
            assumption.prox_gaps_error(log_degree, log_inv_rate, field_size_bits, 2);

        let n = (1_u64 << (log_degree + log_inv_rate)) as f64;
        let rho = rate;
        let constant = 2. * 10.5_f64.powi(5) / 3.;
        let real_error_non_log = constant * n / rho.powf(1.5);
        let real_error = field_size_bits as f64 - real_error_non_log.log2();

        assert!(
            (computed_error - real_error).abs() < 0.01,
            "computed: {computed_error}, expected: {real_error}"
        );
    }

    #[test]
    fn test_cb_errors() {
        let assumption = SecurityAssumption::CapacityBound;

        let log_degree = 20;
        let degree = (1 << log_degree) as f64;
        let log_inv_rate = 2;
        let rate = 1. / (1 << log_inv_rate) as f64;

        let eta = rate / 20.;

        let field_size_bits = 128;

        let real_list_size = degree / (rate * eta);
        let computed_list_size = assumption.list_size_bits(log_degree, log_inv_rate);
        assert!((real_list_size.log2() - computed_list_size).abs() < 0.01);

        let computed_error =
            assumption.prox_gaps_error(log_degree, log_inv_rate, field_size_bits, 2);
        let real_error_non_log = degree / (eta * rate.powi(2));
        let real_error = field_size_bits as f64 - real_error_non_log.log2();

        assert!((computed_error - real_error).abs() < 0.01);
    }

    #[test]
    fn test_folding_pow_bits() {
        let field_size_bits = 64;
        let soundness = SecurityAssumption::CapacityBound;

        let pow_bits = soundness.folding_pow_bits(100, field_size_bits, 10, 5);

        assert!(pow_bits >= 0.);
    }

    #[test]
    #[should_panic(expected = "log_eta is undefined for UniqueDecoding")]
    fn log_eta_panics_for_unique_decoding() {
        // eta does not appear in the UD distance formula `delta = (1 - rho) / 2`.
        // Reading log_eta in the UD branch is a programmer error; the panic
        // locks that down so a future refactor that strays into the eta path
        // under UD fails loudly instead of silently propagating a bogus value.
        let _ = SecurityAssumption::UniqueDecoding.log_eta(5);
    }

    // BCSS25 vs BCI+20: Johnson-bound proximity gap improvement.
    //
    // Bounds on the size of the exceptional set |S|:
    //
    //     [BCI+20] Thm 5.1 :  |S| > (m + 1/2)^7 / 3      * n^2 / rho^{3/2}
    //     [BCSS25] Thm 1.5 :  |S| > 2 * (m + 1/2)^5 / 3  * n   / rho^{3/2}
    //
    // Safety choice eta = sqrt(rho) / 20  =>  multiplicity m = 10.
    //
    // Gain in log form:
    //
    //     gain = log_2((m + 1/2)^2 / 2) + log_2(n)
    //          = log_2(55.125)          + log_2(n)
    //         ~= log_2(n) + 5.78  bits.
    //
    // Tests in this section:
    // - strict improvement, with the exact analytical gap pinned;
    // - new bound clears `security_level - MAX_POW_BITS` over a 155-bit field;
    // - full WHIR security budget hits 128 bits at a reference config;
    // - curve folding costs log_2(M) bits per [BCSS25] Thm 4.2.

    /// Field size in bits used by every test in this section.
    ///
    /// Equals `5 * ceil(log_2(p_KoalaBear))` with `p_KoalaBear = 2^31 - 2^24 + 1`,
    /// i.e. a degree-5 extension of the KoalaBear prime field.
    ///
    /// Chosen because it is the smallest extension that gives the [BCSS25]
    /// bound enough headroom for 128-bit WHIR soundness in the regimes tested.
    const KOALABEAR_QUINTIC_BITS: usize = 155;

    /// Conventional ceiling for Fiat-Shamir grinding, in bits.
    ///
    /// Above roughly 30 bits, grinding becomes impractical for honest provers.
    ///
    /// Every algebraic bound contributing to the folding error must clear
    /// `security_level - MAX_POW_BITS` on its own.
    const MAX_POW_BITS: f64 = 30.0;

    /// Old prox-gap baseline used by the improvement test.
    ///
    /// # Overview
    ///
    /// [BCI+20] Theorem 5.1 at the safety choice `eta = sqrt(rho) / 20`
    /// (so `m = 10`):
    ///
    /// ```text
    ///     |S| > (m + 1/2)^7 / 3 * n^2 / rho^{3/2}.
    /// ```
    ///
    /// No leading factor of `2`: that factor belongs to the [BCSS25]
    /// statement only and would inflate this baseline.
    ///
    /// # Equivalence with [BCI+20] Theorem 1.2
    ///
    /// Substituting `n = (k + 1) / rho` rewrites the bound as
    /// `(k + 1)^2 * 10^7 / rho^{7/2}` — the [BCI+20] Theorem 1.2 form,
    /// up to a small leading constant.
    ///
    /// # Returns
    ///
    /// Field-size bits minus `log_2` of the lower bound on `|S|`.
    fn bci20_jb_prox_gaps_error(
        log_degree: usize,
        log_inv_rate: usize,
        field_size_bits: usize,
    ) -> f64 {
        // Multiplicity at the safety choice eta = sqrt(rho) / 20.
        const M_PLUS_HALF: f64 = 10.5;

        // n = 2^(log_degree + log_inv_rate), so log_2(n^2) collects
        // 2 * log_degree + 2 * log_inv_rate.
        let log_n_squared = 2.0 * (log_degree + log_inv_rate) as f64;

        // Leading constant of the bound; no extra factor of 2 here.
        let log_leading_constant = libm::log2(libm::pow(M_PLUS_HALF, 7.0) / 3.0);

        // The rho^{-3/2} term: log_2(rho^{-3/2}) = 3/2 * log_inv_rate.
        let log_rho_pow_neg_three_halves = 1.5 * log_inv_rate as f64;

        // Log_2 of the lower bound on |S|.
        let error_bits = log_n_squared + log_leading_constant + log_rho_pow_neg_three_halves;

        // Provable security in bits.
        field_size_bits as f64 - error_bits
    }

    #[test]
    fn jb_prox_gap_strictly_improves_over_old_bound() {
        // Invariant: the new bound exceeds the old one by exactly
        //
        //     gap = log_2((m + 1/2)^2 / 2) + log_2(n).
        //
        // At m = 10 (safety choice eta = sqrt(rho) / 20), the additive
        // constant is log_2(55.125) ~= 5.78.
        //
        // Fixture state:
        //
        //     log_degree   : 10 ..= 25   (polynomial degree 2^10 .. 2^25)
        //     log_inv_rate :  1 ..=  4   (rho = 1/2 .. 1/16)
        //     field        : 155-bit     (degree-5 KoalaBear extension)
        let jb = SecurityAssumption::JohnsonBound;

        // Ratio of leading constants between old and new bounds:
        //
        //     old   : (m + 1/2)^7 / 3
        //     new   : 2 * (m + 1/2)^5 / 3
        //     ratio = (m + 1/2)^2 / 2  =  55.125  at  m = 10.
        let leading_ratio_log = libm::log2(10.5_f64.powi(2) / 2.0);

        for log_degree in 10..=25 {
            for log_inv_rate in 1..=4 {
                // Provable security under the new bound.
                let new_bits =
                    jb.prox_gaps_error(log_degree, log_inv_rate, KOALABEAR_QUINTIC_BITS, 2);

                // Provable security under the old bound (local reference).
                let old_bits =
                    bci20_jb_prox_gaps_error(log_degree, log_inv_rate, KOALABEAR_QUINTIC_BITS);

                // Strict improvement: more provable security bits.
                assert!(
                    new_bits > old_bits,
                    "no improvement at log_degree={log_degree}, \
                     log_inv_rate={log_inv_rate}: new={new_bits:.4}, old={old_bits:.4}"
                );

                // Headline claim: gap = log_2(n) + log_2((m+1/2)^2 / 2).
                //
                // Both terms are exact under the implementation's choice of multiplicity.
                // The 1e-9 tolerance only absorbs floating-point rounding in pow / log2.
                let log_n = (log_degree + log_inv_rate) as f64;
                let observed = new_bits - old_bits;
                let expected = log_n + leading_ratio_log;

                assert!(
                    (observed - expected).abs() < 1e-9,
                    "gap mismatch at log_degree={log_degree}, \
                     log_inv_rate={log_inv_rate}: expected={expected:.6}, \
                     got={observed:.6}"
                );
            }
        }
    }

    #[test]
    fn jb_prox_gap_covers_security_level_minus_pow_over_koalabear_quintic() {
        // Invariant: the prox-gap bound alone clears
        // `security_level - MAX_POW_BITS`, so a feasible PoW budget
        // bridges the rest to the target security level.
        //
        // Fixture state:
        //
        //     security_level    = 128
        //     MAX_POW_BITS      =  30
        //     min_required_bits = 128 - 30 = 98
        //     field_size_bits   = 155
        //     log_inv_rate      : 1 ..= 2   (rho in {1/2, 1/4})
        //     log_degree        : 10 ..= 22
        let jb = SecurityAssumption::JohnsonBound;

        // f64 because every use site compares against a float bound.
        let security_level: f64 = 128.0;
        let min_required_bits = security_level - MAX_POW_BITS;

        for log_inv_rate in 1..=2 {
            for log_degree in 10..=22 {
                // 2-fold combination (line case, num_functions = 2).
                let prox_gap_bits =
                    jb.prox_gaps_error(log_degree, log_inv_rate, KOALABEAR_QUINTIC_BITS, 2);

                // Must clear 98 bits, leaving at most 30 bits for PoW.
                assert!(
                    prox_gap_bits > min_required_bits,
                    "prox-gap below {min_required_bits:.0} bits at \
                     log_degree={log_degree}, log_inv_rate={log_inv_rate}: \
                     got {prox_gap_bits:.2}"
                );
            }
        }
    }

    #[test]
    fn jb_full_security_budget_reaches_128_bits() {
        // Invariant: the full WHIR soundness budget reaches
        // `security_level` bits at a representative configuration.
        //
        // Components that may lean on PoW (>= security_level - MAX_POW_BITS):
        // - prox-gap
        // - sumcheck
        // - query-linear-combination
        //
        // Components that must self-sustain (>= security_level):
        // - out-of-domain sample
        // - FRI query phase
        //
        // PoW grinding must not exceed MAX_POW_BITS in total.
        //
        // Fixture state:
        //
        //     log_degree      = 20   (2^20 evaluation domain elements)
        //     log_inv_rate    =  2   (rho = 1/4)
        //     field_size_bits = 155  (degree-5 KoalaBear extension)
        //     security_level  = 128
        let jb = SecurityAssumption::JohnsonBound;

        // Passed as usize to sizing helpers; cast to f64 in asserts.
        let security_level: usize = 128;
        let min_with_pow = security_level as f64 - MAX_POW_BITS;
        let log_degree = 20;
        let log_inv_rate = 2;

        // FRI query count for the query phase to reach security_level alone.
        let num_queries = jb.queries(security_level, log_inv_rate);

        // OOD sample count for the OOD term to reach security_level alone.
        let ood_samples = jb.determine_ood_samples(
            security_level,
            log_degree,
            log_inv_rate,
            KOALABEAR_QUINTIC_BITS,
        );

        // Five algebraic error bounds at the chosen configuration.
        let prox_gap = jb.prox_gaps_error(log_degree, log_inv_rate, KOALABEAR_QUINTIC_BITS, 2);
        let sumcheck = jb.fold_sumcheck_error(KOALABEAR_QUINTIC_BITS, log_degree, log_inv_rate);
        let ood = jb.ood_error(
            log_degree,
            log_inv_rate,
            KOALABEAR_QUINTIC_BITS,
            ood_samples,
        );
        let query = jb.queries_error(log_inv_rate, num_queries);
        let combination = jb.queries_combination_error(
            KOALABEAR_QUINTIC_BITS,
            log_degree,
            log_inv_rate,
            ood_samples,
            num_queries,
        );

        // Three components that may lean on PoW.
        assert!(
            prox_gap >= min_with_pow,
            "prox-gap {prox_gap:.2} bits < {min_with_pow:.0}"
        );
        assert!(
            sumcheck >= min_with_pow,
            "sumcheck {sumcheck:.2} bits < {min_with_pow:.0}"
        );
        assert!(
            combination >= min_with_pow,
            "combination {combination:.2} bits < {min_with_pow:.0}"
        );

        // Two components that must reach the full target alone.
        assert!(
            ood >= security_level as f64,
            "OOD {ood:.2} bits < {security_level}"
        );
        assert!(
            query >= security_level as f64,
            "query {query:.2} bits < {security_level}"
        );

        // PoW closes the residual gap without exceeding the ceiling.
        let pow = jb.folding_pow_bits(
            security_level,
            KOALABEAR_QUINTIC_BITS,
            log_degree,
            log_inv_rate,
        );
        assert!(
            pow <= MAX_POW_BITS,
            "PoW grinding {pow:.2} bits > {MAX_POW_BITS:.0} cap"
        );
    }

    #[test]
    fn jb_prox_gap_scales_by_log_curve_degree() {
        // Invariant: combining `M + 1` functions costs exactly `log_2(M)`
        // bits of prox-gap vs. the line case (M = 1), per [BCSS25] Thm 4.2.
        //
        // Fixture state:  log_degree=20, log_inv_rate=2, field=155.
        //
        //     num_functions  |  M  |  expected loss
        //     ---------------+-----+----------------
        //          2         |  1  |  baseline
        //          3         |  2  |    1.0 bit
        //          5         |  4  |    2.0 bits
        //          9         |  8  |    3.0 bits
        let jb = SecurityAssumption::JohnsonBound;
        let log_degree = 20;
        let log_inv_rate = 2;

        // Baseline: 2-function combination (line, M = 1).
        let line_bits = jb.prox_gaps_error(log_degree, log_inv_rate, KOALABEAR_QUINTIC_BITS, 2);

        for (num_functions, expected_loss) in [(3_usize, 1.0_f64), (5, 2.0), (9, 3.0)] {
            // Prox-gap bits over a degree-M curve (M = num_functions - 1).
            let curve_bits = jb.prox_gaps_error(
                log_degree,
                log_inv_rate,
                KOALABEAR_QUINTIC_BITS,
                num_functions,
            );

            // Observed loss relative to the line case.
            let loss = line_bits - curve_bits;

            // Expected loss = log_2(M) = log_2(num_functions - 1).
            assert!(
                (loss - expected_loss).abs() < 1e-9,
                "curve scaling off at num_functions={num_functions}: \
                 expected log_2({}) = {expected_loss:.1} bits, got {loss:.6}",
                num_functions - 1
            );
        }
    }

    // ----- eta-parameterized variants ----------------------------------
    //
    // These tests pin two invariants of the `_at_log_eta` methods:
    //   (1) at the fixed safe-gap eta they must agree with the fixed-eta methods
    //       to within floating-point noise (callers must be free to migrate);
    //   (2) at a deliberately smaller eta they must report strictly fewer bits of
    //       security on every list-size-driven term (Codex flagged the missing
    //       sensitivity to per-round eta_i in STIR's parameter schedule).

    #[test]
    fn at_log_eta_collapses_to_fixed_at_safe_gap() {
        let log_degree = 20;
        let log_inv_rate = 2;
        let field = KOALABEAR_QUINTIC_BITS;
        let ood_samples = 2;
        let num_queries = 100;

        for assumption in [
            SecurityAssumption::JohnsonBound,
            SecurityAssumption::CapacityBound,
        ] {
            let safe = assumption.log_eta(log_inv_rate);

            let list_fixed = assumption.list_size_bits(log_degree, log_inv_rate);
            let list_eta = assumption.list_size_bits_at_log_eta(log_degree, log_inv_rate, safe);
            assert!(
                (list_fixed - list_eta).abs() < 1e-9,
                "{assumption}: list_size_bits mismatch at safe eta: fixed={list_fixed}, eta={list_eta}",
            );

            let prox_fixed = assumption.prox_gaps_error(log_degree, log_inv_rate, field, 2);
            let prox_eta =
                assumption.prox_gaps_error_at_log_eta(log_degree, log_inv_rate, field, 2, safe);
            assert!(
                (prox_fixed - prox_eta).abs() < 1e-9,
                "{assumption}: prox_gaps_error mismatch at safe eta: fixed={prox_fixed}, eta={prox_eta}",
            );

            let ood_fixed = assumption.ood_error(log_degree, log_inv_rate, field, ood_samples);
            let ood_eta =
                assumption.ood_error_at_log_eta(log_degree, log_inv_rate, field, ood_samples, safe);
            assert!(
                (ood_fixed - ood_eta).abs() < 1e-9,
                "{assumption}: ood_error mismatch at safe eta: fixed={ood_fixed}, eta={ood_eta}",
            );

            let comb_fixed = assumption.queries_combination_error(
                field,
                log_degree,
                log_inv_rate,
                ood_samples,
                num_queries,
            );
            let comb_eta = assumption.queries_combination_error_at_log_eta(
                field,
                log_degree,
                log_inv_rate,
                ood_samples,
                num_queries,
                safe,
            );
            assert!(
                (comb_fixed - comb_eta).abs() < 1e-9,
                "{assumption}: queries_combination_error mismatch at safe eta: fixed={comb_fixed}, eta={comb_eta}",
            );

            let sum_fixed = assumption.fold_sumcheck_error(field, log_degree, log_inv_rate);
            let sum_eta =
                assumption.fold_sumcheck_error_at_log_eta(field, log_degree, log_inv_rate, safe);
            assert!(
                (sum_fixed - sum_eta).abs() < 1e-9,
                "{assumption}: fold_sumcheck_error mismatch at safe eta: fixed={sum_fixed}, eta={sum_eta}",
            );

            let fold_fixed = assumption.fold_algebraic_bits(field, log_degree, log_inv_rate);
            let fold_eta =
                assumption.fold_algebraic_bits_at_log_eta(field, log_degree, log_inv_rate, safe);
            assert!(
                (fold_fixed - fold_eta).abs() < 1e-9,
                "{assumption}: fold_algebraic_bits mismatch at safe eta: fixed={fold_fixed}, eta={fold_eta}",
            );
        }
    }

    #[test]
    fn at_log_eta_shrinks_security_when_eta_is_smaller_cb() {
        // CB STIR §5.3 with KoalaBear quintic, log_degree=20, log_inv_rate=1, lambda=108.
        //
        //   stir_initial_eta(CB) = 2^(lambda + log(k-1) + log(d/k) + 2*log_inv_rate - field)
        //                        = 2^(108 + 1.585 + 18 + 2 - 155)  ≈  2^-25.4
        //
        // vs. the fixed safe gap log_eta = -(log_inv_rate + LOG2_10 + 1) ≈ -5.32.
        //
        // Codex's blocking finding: the fixed-eta accounting overstates security on
        // every list-size-driven term when the protocol commits to eta ≈ 2^-25.4.
        let cb = SecurityAssumption::CapacityBound;
        let log_degree = 20;
        let log_inv_rate = 1;
        let field = KOALABEAR_QUINTIC_BITS;
        let ood_samples = 2;
        let num_queries = 115;

        let safe = cb.log_eta(log_inv_rate);
        let small: f64 = -25.4;
        assert!(small < safe, "test premise: small eta must be < safe eta");

        // list_size grows linearly with -log_eta. The two numbers must straddle the
        // arithmetic gap quoted in the audit (~20 bits between log_eta values).
        let list_safe = cb.list_size_bits_at_log_eta(log_degree, log_inv_rate, safe);
        let list_small = cb.list_size_bits_at_log_eta(log_degree, log_inv_rate, small);
        assert!(
            list_small > list_safe + 19.,
            "list_size at small eta ({list_small}) should be > safe ({list_safe}) + 19",
        );
        assert!(
            list_small < list_safe + 21.,
            "list_size at small eta ({list_small}) should be < safe ({list_safe}) + 21",
        );

        // Every list-size-driven term must report fewer bits of security at the
        // smaller eta. This is the invariant the soundness accounting must preserve.
        let prox_safe = cb.prox_gaps_error_at_log_eta(log_degree, log_inv_rate, field, 2, safe);
        let prox_small = cb.prox_gaps_error_at_log_eta(log_degree, log_inv_rate, field, 2, small);
        assert!(
            prox_small < prox_safe,
            "prox_gaps_error must decrease when eta shrinks: safe={prox_safe}, small={prox_small}",
        );

        let ood_safe = cb.ood_error_at_log_eta(log_degree, log_inv_rate, field, ood_samples, safe);
        let ood_small =
            cb.ood_error_at_log_eta(log_degree, log_inv_rate, field, ood_samples, small);
        assert!(
            ood_small < ood_safe,
            "ood_error must decrease when eta shrinks: safe={ood_safe}, small={ood_small}",
        );

        let comb_safe = cb.queries_combination_error_at_log_eta(
            field,
            log_degree,
            log_inv_rate,
            ood_samples,
            num_queries,
            safe,
        );
        let comb_small = cb.queries_combination_error_at_log_eta(
            field,
            log_degree,
            log_inv_rate,
            ood_samples,
            num_queries,
            small,
        );
        assert!(
            comb_small < comb_safe,
            "queries_combination_error must decrease when eta shrinks: safe={comb_safe}, small={comb_small}",
        );

        let sum_safe = cb.fold_sumcheck_error_at_log_eta(field, log_degree, log_inv_rate, safe);
        let sum_small = cb.fold_sumcheck_error_at_log_eta(field, log_degree, log_inv_rate, small);
        assert!(
            sum_small < sum_safe,
            "fold_sumcheck_error must decrease when eta shrinks: safe={sum_safe}, small={sum_small}",
        );
    }
}
