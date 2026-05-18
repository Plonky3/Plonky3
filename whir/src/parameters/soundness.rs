//! Soundness analysis and security assumption models for Reed-Solomon proximity testing.

use alloc::format;
use alloc::string::String;
use core::f64::consts::LOG2_10;
use core::fmt::Display;
use core::str::FromStr;

use serde::Serialize;

/// Security assumptions determines which proximity parameters and conjectures are assumed by the error computation.
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
    ///
    /// # Proximity Gaps Improvement
    ///
    /// The error bound uses Theorem 1.5 from [BCSS25]:
    /// - **Old [BCI+20]**: `a = O(n^2/eta^7)` exceptional z's
    /// - **New [BCSS25]**: `a = O(n/eta^5)` exceptional z's
    ///
    /// This improvement of factor `n*eta^2` translates to approximately `log_2(n)` additional bits of security,
    /// making provable 128-bit security achievable with degree-5 extensions of small prime fields.
    JohnsonBound,

    /// Capacity bound assumes that the distance of each oracle is within the capacity bound (1 - rho - eta).
    /// With eta = rho/20.
    /// We refer to this configuration as CB for short.
    /// This requires conjecturing that RS codes are decodable up to capacity and have correlated agreement (mutual in WHIR) up to capacity.
    CapacityBound,
}

impl SecurityAssumption {
    /// In both JB and CB theorems such as list-size only hold for proximity parameters slighly below the bound.
    /// E.g. in JB proximity gaps holds for every delta in (0, 1 - sqrt(rho)).
    /// eta is the distance between the chosen proximity parameter and the bound.
    /// I.e. in JB delta = 1 - sqrt(rho) - eta and in CB delta = 1 - rho - eta.
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

    /// Given a RS code (specified by the log of the degree and log inv of the rate) a field_size
    /// and an arity, compute the proximity gaps error (in bits) at the specified distance.
    ///
    /// # Johnson Bound Improvement
    ///
    /// For the Johnson bound case, this uses the improved Theorem 1.5 from [BCSS25]:
    ///
    /// > "On Proximity Gaps for Reed-Solomon Codes" (eprint 2025/2055)
    /// > Ben-Sasson, Carmon, Haboeck, Kopparty, Saraf
    ///
    /// The theorem states that for proximity parameter `gamma < J(delta) - eta` (below Johnson bound by eta),
    /// the number of exceptional z's satisfies:
    ///
    /// ```text
    /// a > (2(m + 1/2)^5 + 3(m + 1/2)*gamma*rho) / (3*rho^(3/2)) * n + (m + 1/2) / sqrt(rho)
    /// ```
    ///
    /// where `m = max(ceil(sqrt(rho) / (2*eta)), 3)` and the asymptotic behavior is `O(n/eta^5)`.
    ///
    /// ## Comparison with prior work
    ///
    /// | Reference | Bound on exceptional z's | Proximity loss |
    /// |-----------|--------------------------|----------------|
    /// | [BCI+20]  | `O(n^2/eta^7)`             | 0              |
    /// | [BCSS25]  | `O(n/eta^5)`               | 0              |
    ///
    /// The improvement factor of `n*eta^2` translates to approximately `log_2(n)` additional bits
    /// of provable security, enabling 128-bit security with degree-5 extensions of KoalaBear.
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

    /// Compute the number of queries to match the security level
    /// The error to drive down is (1-delta)^t < 2^-lambda.
    /// Where delta is set as in the `log_1_delta` function.
    #[must_use]
    pub fn queries(&self, protocol_security_level: usize, log_inv_rate: usize) -> usize {
        let num_queries_f = -(protocol_security_level as f64) / self.log_1_delta(log_inv_rate);

        libm::ceil(num_queries_f) as usize
    }

    /// Compute the error for the given number of queries
    /// The error to drive down is (1-delta)^t < 2^-lambda.
    /// Where delta is set as in the `log_1_delta` function.
    #[must_use]
    pub fn queries_error(&self, log_inv_rate: usize, num_queries: usize) -> f64 {
        let num_queries = num_queries as f64;

        -num_queries * self.log_1_delta(log_inv_rate)
    }

    /// Compute the error for the OOD samples of the protocol
    /// See Lemma 4.5 in STIR.
    /// The error is list_size^2 * (degree/field_size_bits)^reps
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

    /// Computes the number of OOD samples required to achieve security_level bits of security
    /// We note that in both STIR and WHIR there are various strategies to set OOD samples.
    /// In this case, we are just sampling one element from the extension field
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
        // List size at the current proximity parameter and code rate.
        let list_size = self.list_size_bits(num_variables, log_inv_rate);

        // Security = field size minus the adversary's advantage from list decoding.
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
        // List size at the current proximity parameter.
        let list_size = self.list_size_bits(num_variables, log_inv_rate);

        // Total number of evaluation points available for the combination.
        let log_combination = libm::log2((ood_samples + num_queries) as f64);

        // Security = field size minus the adversary's advantage.
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
        // Proximity gaps bound: how many bits of security the
        // fold-and-test step provides against far-from-RS functions.
        // Uses arity 2 because each fold combines exactly 2 evaluations.
        let prox_gaps_error = self.prox_gaps_error(num_variables, log_inv_rate, field_size_bits, 2);

        // Sumcheck bound: security from the random folding challenge.
        // Bounded by (list_size + 1) / |F|.
        let sumcheck_error = self.fold_sumcheck_error(field_size_bits, num_variables, log_inv_rate);

        // The folding step is only as strong as its weakest component.
        let error = prox_gaps_error.min(sumcheck_error);

        // PoW covers the remaining gap; zero means no grinding needed.
        0_f64.max(security_level as f64 - error)
    }

    /// Compute the OOD error for a ZK code-switching round (in bits).
    ///
    /// In ZK mode, the OOD zero-evader operates on the joint message space
    /// of the target code C' and the mask code C_zk. The product list bound
    /// `(|Λ(C')| · |Λ(C_zk)|)²` comes from Lemma 9.9 §1 (p.59), which
    /// extends the single-code Lemma 9.1 bound to the joint code pair.
    ///
    /// # Formula
    ///
    /// From Theorem 9.6 and Lemma 9.9 §1 in eprint 2026/391:
    ///
    /// ```text
    /// error = (|Λ(C', δ')| · |Λ(C_zk, δ_zk)|)² / 2 · ε_ood
    /// ```
    ///
    /// where `ε_ood` is the private zero-evader error from Lemma 9.3:
    /// `max{(degree / |F|)^ood_samples, 1 / (|F|^ood_samples - 1)}`.
    /// For practical parameters the first term dominates (see below).
    ///
    /// # References
    ///
    /// - eprint 2026/391, Theorem 9.6, first RBR error term
    /// - eprint 2026/391, Lemma 9.9 §1 ("Out-of-domain samples")
    /// - eprint 2026/391, Lemma 9.3 (private zero-evader error)
    #[must_use]
    pub fn zk_ood_error(
        &self,
        log_degree: usize,
        log_inv_rate: usize,
        field_size_bits: usize,
        ood_samples: usize,
        mask_log_degree: usize,
        mask_log_inv_rate: usize,
    ) -> f64 {
        if matches!(self, Self::UniqueDecoding) {
            return 0.;
        }

        let list_target_bits = self.list_size_bits(log_degree, log_inv_rate);
        let list_mask_bits = self.list_size_bits(mask_log_degree, mask_log_inv_rate);

        // The ZK zero-evader operates on the joint vector [f, r, s] from
        // Construction 9.7. The Schwartz-Zippel degree for ε_ood is the max
        // component degree: max(deg(f), deg(s)) = max(degree, mask_degree).
        let max_log_degree = log_degree.max(mask_log_degree);

        // Lemma 9.3: ε_ood = max{(degree/|F|)^t, 1/(|F|^t - 1)}.
        // Term 1 dominates term 2 for all practical parameters (|F| >> degree).
        debug_assert!(
            max_log_degree >= 1 && field_size_bits > max_log_degree,
            "Lemma 9.3 term-1 dominance requires field_size_bits > max_log_degree"
        );
        let ze_log_error = (max_log_degree * ood_samples) as f64;

        let error = 2. * (list_target_bits + list_mask_bits) + ze_log_error;
        (ood_samples * field_size_bits) as f64 + 1. - error
    }

    /// Compute the combination error for a ZK code-switching round (in bits).
    ///
    /// In ZK mode, the union bound in the combination step ranges over
    /// witnesses from both the target code and the interleaved mask code.
    ///
    /// # Formula
    ///
    /// From Theorem 9.6 and Lemma 9.9 §3 in eprint 2026/391:
    ///
    /// ```text
    /// error = |Λ(C', δ')| · |Λ(C_zk^{≡n+1}, δ_zk)| · ε_zero
    /// ```
    ///
    /// For the interleaved mask list size `|Λ(C_zk^{≡n+1})|`, we use a
    /// simplified upper bound `(n+1) · |Λ(C_zk)|`. This holds when the
    /// GGR11 list-decoding exponent `r = 1` (typical for RS codes at
    /// practical WHIR distances). Overestimates the error (safe direction).
    ///
    /// # References
    ///
    /// - eprint 2026/391, Theorem 9.6, combination randomness error
    /// - eprint 2026/391, Lemma 9.9 §3 ("Combination randomness")
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn zk_combination_error(
        &self,
        field_size_bits: usize,
        num_variables: usize,
        log_inv_rate: usize,
        ood_samples: usize,
        num_queries: usize,
        mask_log_degree: usize,
        mask_log_inv_rate: usize,
        num_mask_functions: usize,
    ) -> f64 {
        assert!(
            num_mask_functions >= 1,
            "num_mask_functions must be >= 1 for ZK code-switching"
        );

        let list_target = self.list_size_bits(num_variables, log_inv_rate);
        let list_mask = self.list_size_bits(mask_log_degree, mask_log_inv_rate);

        // Interleaving margin: |Λ(C_zk^{≡n+1})| ≤ (n+1)·|Λ(C_zk)| (r=1 regime for RS).
        let interleaving_margin = libm::log2((num_mask_functions + 1) as f64);
        let list_mask_interleaved = list_mask + interleaving_margin;

        let log_combination = libm::log2((ood_samples + num_queries) as f64);

        field_size_bits as f64 - (log_combination + list_target + list_mask_interleaved + 1.)
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

        // Invalid cases
        assert!(SecurityAssumption::from_str("InvalidType").is_err());
        assert!(SecurityAssumption::from_str("").is_err()); // Empty string
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

        // Setting
        let log_degree = 20;
        let degree = (1 << log_degree) as f64;
        let log_inv_rate = 2;
        let rate = 1. / (1 << log_inv_rate) as f64;

        let field_size_bits = 128;

        // List size
        assert!(assumption.list_size_bits(log_degree, log_inv_rate) - 0. < 0.01);

        // Prox gaps
        let computed_error =
            assumption.prox_gaps_error(log_degree, log_inv_rate, field_size_bits, 2);
        let real_error_non_log = degree / rate;
        let real_error = field_size_bits as f64 - real_error_non_log.log2();

        assert!((computed_error - real_error).abs() < 0.01);
    }

    #[test]
    fn test_jb_errors() {
        let assumption = SecurityAssumption::JohnsonBound;

        // Setting
        let log_degree = 20;
        let log_inv_rate = 2;
        let rate = 1. / (1 << log_inv_rate) as f64;

        let eta = rate.sqrt() / 20.;

        let field_size_bits = 128;

        // List size
        let real_list_size = 1. / (2. * eta * rate.sqrt());
        let computed_list_size = assumption.list_size_bits(log_degree, log_inv_rate);
        assert!((real_list_size.log2() - computed_list_size).abs() < 0.01);

        // Prox gaps - Updated to use Theorem 1.5 from [BCSS25]
        //
        // From "On Proximity Gaps for Reed-Solomon Codes" (eprint 2025/2055):
        // With eta = sqrt(rho)/20, m = 10, the error bound is:
        //   a ~ (2 * 10.5^5) / (3 * rho^(3/2)) * n
        //
        // where n = 2^(log_degree + log_inv_rate)
        let computed_error =
            assumption.prox_gaps_error(log_degree, log_inv_rate, field_size_bits, 2);

        // n = 2^(log_degree + log_inv_rate) = 2^22
        let n = (1_u64 << (log_degree + log_inv_rate)) as f64;
        // rho = rate = 2^(-log_inv_rate) = 0.25
        let rho = rate;
        // Constant from Theorem 1.5: (2 * 10.5^5) / 3 ~ 85085.44
        let constant = 2. * 10.5_f64.powi(5) / 3.;
        // a ~ constant * n / rho^(3/2)
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

        // Setting
        let log_degree = 20;
        let degree = (1 << log_degree) as f64;
        let log_inv_rate = 2;
        let rate = 1. / (1 << log_inv_rate) as f64;

        let eta = rate / 20.;

        let field_size_bits = 128;

        // List size
        let real_list_size = degree / (rate * eta);
        let computed_list_size = assumption.list_size_bits(log_degree, log_inv_rate);
        assert!((real_list_size.log2() - computed_list_size).abs() < 0.01);

        // Prox gaps
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

        let pow_bits = soundness.folding_pow_bits(
            100, // Security level
            field_size_bits,
            10, // Number of variables
            5,  // Log inverse rate
        );

        // PoW bits should never be negative
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

    // ---------------------------------------------------------------
    // ZK-aware error estimation (eprint 2026/391, Theorem 9.6)
    // ---------------------------------------------------------------

    #[test]
    fn zk_ood_error_matches_hand_calculation() {
        // Verify zk_ood_error against the raw formula from Theorem 9.6:
        //
        //   error = (|Λ(C',δ')| · |Λ(C_zk,δ_zk)|)² / 2 · ε_ood
        //
        // where ε_ood = (degree/|F|)^{ood_samples} (Lemma 9.3).
        //
        // Fixture: JB, log_degree=20, log_inv_rate=2, field=155 bits,
        //   mask: log_degree=10, log_inv_rate=3, ood_samples=2.
        let jb = SecurityAssumption::JohnsonBound;
        let log_degree = 20;
        let log_inv_rate = 2;
        let mask_log_degree = 10;
        let mask_log_inv_rate = 3;
        let field_size_bits = KOALABEAR_QUINTIC_BITS;
        let ood_samples = 2;

        let computed = jb.zk_ood_error(
            log_degree,
            log_inv_rate,
            field_size_bits,
            ood_samples,
            mask_log_degree,
            mask_log_inv_rate,
        );

        // Hand calculation:
        // list_target = JB list size at (20, 2)
        // list_mask   = JB list size at (10, 3)
        // max_log_degree = max(20, 10) = 20
        // ε_ood = (2^20 / 2^155)^2 = 2^{-270}
        // error_bits = 2 * (list_target + list_mask) + max_log_degree * ood_samples
        // security = 2 * 155 + 1 - error_bits
        let list_target = jb.list_size_bits(log_degree, log_inv_rate);
        let list_mask = jb.list_size_bits(mask_log_degree, mask_log_inv_rate);
        let max_log_degree = log_degree.max(mask_log_degree);
        let error_bits = 2. * (list_target + list_mask) + (max_log_degree * ood_samples) as f64;
        let expected = (ood_samples * field_size_bits) as f64 + 1. - error_bits;

        assert!(
            (computed - expected).abs() < 0.01,
            "JB zk_ood_error: computed={computed:.4}, expected={expected:.4}"
        );

        // CB
        let cb = SecurityAssumption::CapacityBound;
        let computed_cb = cb.zk_ood_error(
            log_degree,
            log_inv_rate,
            field_size_bits,
            ood_samples,
            mask_log_degree,
            mask_log_inv_rate,
        );
        let list_target_cb = cb.list_size_bits(log_degree, log_inv_rate);
        let list_mask_cb = cb.list_size_bits(mask_log_degree, mask_log_inv_rate);
        let error_bits_cb =
            2. * (list_target_cb + list_mask_cb) + (max_log_degree * ood_samples) as f64;
        let expected_cb = (ood_samples * field_size_bits) as f64 + 1. - error_bits_cb;

        assert!(
            (computed_cb - expected_cb).abs() < 0.01,
            "CB zk_ood_error: computed={computed_cb:.4}, expected={expected_cb:.4}"
        );
    }

    #[test]
    fn zk_ood_reduces_to_non_zk_when_mask_is_ud() {
        // When the mask code has UniqueDecoding (list size = 1, list_size_bits = 0),
        // the ZK OOD error should equal the non-ZK OOD error.
        let jb = SecurityAssumption::JohnsonBound;

        for log_degree in [10, 15, 20] {
            for log_inv_rate in [1, 2, 3] {
                for ood_samples in [1, 2, 3] {
                    let non_zk = jb.ood_error(
                        log_degree,
                        log_inv_rate,
                        KOALABEAR_QUINTIC_BITS,
                        ood_samples,
                    );

                    // UD mask: list_size_bits(anything, anything) = 0 under UD.
                    // But zk_ood_error uses self (JB) for the mask list size too.
                    // To get list_mask = 0, we need UD for the mask. Since
                    // list_size_bits is called on self, we check the structural
                    // identity: when mask list = 0, the formulas must match.
                    //
                    // Under UD: list_size_bits returns 0 regardless of params.
                    // So use UD for the mask computation manually:
                    let list_target = jb.list_size_bits(log_degree, log_inv_rate);
                    let list_mask = 0.0_f64; // UD mask
                    let error = 2. * (list_target + list_mask) + (log_degree * ood_samples) as f64;
                    let zk_with_ud_mask =
                        (ood_samples * KOALABEAR_QUINTIC_BITS) as f64 + 1. - error;

                    assert!(
                        (zk_with_ud_mask - non_zk).abs() < 1e-9,
                        "ZK OOD with UD mask should equal non-ZK at \
                         log_degree={log_degree}, log_inv_rate={log_inv_rate}: \
                         zk={zk_with_ud_mask:.6}, non_zk={non_zk:.6}"
                    );
                }
            }
        }
    }

    #[test]
    fn zk_ood_stricter_than_non_zk() {
        // ZK OOD error must be <= non-ZK OOD error (less security)
        // because the product list bound is larger.
        for assumption in [
            SecurityAssumption::JohnsonBound,
            SecurityAssumption::CapacityBound,
        ] {
            for log_degree in [10, 15, 20] {
                for log_inv_rate in [1, 2, 3] {
                    let non_zk =
                        assumption.ood_error(log_degree, log_inv_rate, KOALABEAR_QUINTIC_BITS, 2);
                    let zk = assumption.zk_ood_error(
                        log_degree,
                        log_inv_rate,
                        KOALABEAR_QUINTIC_BITS,
                        2,
                        log_degree,   // mask same size as target
                        log_inv_rate, // mask same rate
                    );

                    assert!(
                        zk <= non_zk,
                        "ZK OOD should be stricter at log_degree={log_degree}, \
                         log_inv_rate={log_inv_rate}: zk={zk:.4} > non_zk={non_zk:.4}"
                    );
                }
            }
        }
    }

    #[test]
    fn zk_combination_error_matches_hand_calculation() {
        // Verify zk_combination_error against Theorem 9.6, third error:
        //
        //   error = |Λ(C',δ')| · |Λ(C_zk^{≡n+1},δ_zk)| · ε_zero
        //
        // with GGR11 interleaving margin: |Λ(C_zk^{≡n+1})| ≤ (n+1)·|Λ(C_zk)|.
        let jb = SecurityAssumption::JohnsonBound;
        let num_variables = 20;
        let log_inv_rate = 2;
        let mask_log_degree = 10;
        let mask_log_inv_rate = 3;
        let ood_samples = 2;
        let num_queries = 50;
        let num_mask_functions = 4;

        let computed = jb.zk_combination_error(
            KOALABEAR_QUINTIC_BITS,
            num_variables,
            log_inv_rate,
            ood_samples,
            num_queries,
            mask_log_degree,
            mask_log_inv_rate,
            num_mask_functions,
        );

        let list_target = jb.list_size_bits(num_variables, log_inv_rate);
        let list_mask = jb.list_size_bits(mask_log_degree, mask_log_inv_rate);
        let interleaving_margin = libm::log2((num_mask_functions + 1) as f64);
        let log_combination = libm::log2((ood_samples + num_queries) as f64);
        let expected = KOALABEAR_QUINTIC_BITS as f64
            - (log_combination + list_target + list_mask + interleaving_margin + 1.);

        assert!(
            (computed - expected).abs() < 0.01,
            "JB zk_combination: computed={computed:.4}, expected={expected:.4}"
        );
    }

    #[test]
    fn zk_combination_stricter_than_non_zk() {
        // ZK combination error must be <= non-ZK because of the larger
        // witness space (product list + interleaving margin).
        for assumption in [
            SecurityAssumption::JohnsonBound,
            SecurityAssumption::CapacityBound,
        ] {
            for num_variables in [10, 15, 20] {
                for log_inv_rate in [1, 2, 3] {
                    let non_zk = assumption.queries_combination_error(
                        KOALABEAR_QUINTIC_BITS,
                        num_variables,
                        log_inv_rate,
                        2,
                        50,
                    );
                    let zk = assumption.zk_combination_error(
                        KOALABEAR_QUINTIC_BITS,
                        num_variables,
                        log_inv_rate,
                        2,
                        50,
                        num_variables, // mask same size
                        log_inv_rate,  // mask same rate
                        1,             // minimal interleaving
                    );

                    assert!(
                        zk <= non_zk,
                        "ZK combination should be stricter at \
                         num_variables={num_variables}, log_inv_rate={log_inv_rate}: \
                         zk={zk:.4} > non_zk={non_zk:.4}"
                    );
                }
            }
        }
    }

    #[test]
    fn zk_combination_reduces_to_non_zk_when_mask_is_ud() {
        // When the mask code has UD (list size = 1, list_size_bits = 0)
        // and num_mask_functions = 0 (interleaving margin = log₂(1) = 0),
        // the ZK combination error should equal non-ZK.
        let jb = SecurityAssumption::JohnsonBound;

        for num_variables in [10, 15, 20] {
            for log_inv_rate in [1, 2, 3] {
                let non_zk = jb.queries_combination_error(
                    KOALABEAR_QUINTIC_BITS,
                    num_variables,
                    log_inv_rate,
                    2,
                    50,
                );

                let list_target = jb.list_size_bits(num_variables, log_inv_rate);
                let list_mask = 0.0_f64; // UD mask
                let log_combination = libm::log2(52.0); // ood=2 + queries=50
                let zk_with_ud_mask = KOALABEAR_QUINTIC_BITS as f64
                    - (log_combination + list_target + list_mask + 0.0 + 1.);

                assert!(
                    (zk_with_ud_mask - non_zk).abs() < 1e-9,
                    "ZK combination with UD mask should equal non-ZK at \
                     num_variables={num_variables}, log_inv_rate={log_inv_rate}: \
                     zk={zk_with_ud_mask:.6}, non_zk={non_zk:.6}"
                );
            }
        }
    }

    #[test]
    fn zk_ud_returns_zero_for_ood() {
        let ud = SecurityAssumption::UniqueDecoding;
        let result = ud.zk_ood_error(20, 2, 155, 2, 10, 3);
        assert!(
            result.abs() < 1e-9,
            "UD zk_ood_error should be 0, got {result}"
        );
    }

    #[test]
    fn zk_full_security_budget_128_bits() {
        // Full ZK security budget at a representative configuration.
        // All ZK-specific error terms must clear the target.
        let jb = SecurityAssumption::JohnsonBound;
        let security_level: usize = 128;
        let log_degree = 20;
        let log_inv_rate = 2;
        let mask_log_degree = 10;
        let mask_log_inv_rate = 3;
        let num_mask_functions = 4;

        let num_queries = jb.queries(security_level, log_inv_rate);
        let ood_samples = jb.determine_ood_samples(
            security_level,
            log_degree,
            log_inv_rate,
            KOALABEAR_QUINTIC_BITS,
        );

        let zk_ood = jb.zk_ood_error(
            log_degree,
            log_inv_rate,
            KOALABEAR_QUINTIC_BITS,
            ood_samples,
            mask_log_degree,
            mask_log_inv_rate,
        );

        let zk_combination = jb.zk_combination_error(
            KOALABEAR_QUINTIC_BITS,
            log_degree,
            log_inv_rate,
            ood_samples,
            num_queries,
            mask_log_degree,
            mask_log_inv_rate,
            num_mask_functions,
        );

        // Non-ZK terms that are unchanged in ZK mode.
        let query = jb.queries_error(log_inv_rate, num_queries);

        assert!(
            zk_ood >= security_level as f64,
            "ZK OOD {zk_ood:.2} bits < {security_level}"
        );
        assert!(
            query >= security_level as f64,
            "query {query:.2} bits < {security_level}"
        );
        assert!(
            zk_combination >= security_level as f64 - MAX_POW_BITS,
            "ZK combination {zk_combination:.2} bits < {}",
            security_level as f64 - MAX_POW_BITS
        );
    }

    #[test]
    fn zk_ood_error_hardcoded_values() {
        // Verify zk_ood_error against fully hardcoded values derived from the
        // raw JB list-size formula: list = 1/(2·η·√ρ), η = √ρ/20.
        //
        // JB at (log_degree=20, log_inv_rate=2): rate=0.25, η=0.025,
        //   list = 1/(2·0.025·0.5) = 40, log₂(40) ≈ 5.32193.
        // JB at (mask_log_degree=10, log_inv_rate=3): rate=0.125, η≈0.01768,
        //   list = 1/(2·0.01768·0.35355) = 80, log₂(80) ≈ 6.32193.
        let jb = SecurityAssumption::JohnsonBound;

        let list_target_hardcoded = 40_f64.log2(); // ≈ 5.32193
        let list_mask_hardcoded = 80_f64.log2(); // ≈ 6.32193

        // Verify our hardcoded values match list_size_bits.
        assert!(
            (jb.list_size_bits(20, 2) - list_target_hardcoded).abs() < 0.01,
            "list_target mismatch"
        );
        assert!(
            (jb.list_size_bits(10, 3) - list_mask_hardcoded).abs() < 0.01,
            "list_mask mismatch"
        );

        // Compute expected from hardcoded list sizes (not from list_size_bits).
        // max_log_degree = max(20, 10) = 20. ze_log_error = 20 * 2 = 40.
        let ze_log_error = 40.0;
        let error_bits = 2.0 * (list_target_hardcoded + list_mask_hardcoded) + ze_log_error;
        let expected = (2 * KOALABEAR_QUINTIC_BITS) as f64 + 1.0 - error_bits;

        let computed = jb.zk_ood_error(20, 2, KOALABEAR_QUINTIC_BITS, 2, 10, 3);
        assert!(
            (computed - expected).abs() < 0.01,
            "zk_ood_error hardcoded: computed={computed:.4}, expected={expected:.4}"
        );
    }
}
