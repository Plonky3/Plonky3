//! WHIR / STIR-style composition: error terms specific to RS-IOPs that
//! combine sumcheck-based folding with random-linear-combination queries.
//!
//! The underlying proximity-gap and query primitives live in
//! [`crate::assumption`]; this module orchestrates them into the
//! protocol-level error budget pieces (OOD sampling, fold sumcheck,
//! query-combination, folding PoW).

pub use crate::assumption::SecurityAssumption;

impl SecurityAssumption {
    /// OOD sampling error (in bits). See Lemma 4.5 in STIR:
    /// `error = L⁺² · (degree / |F|)^reps`.
    /// The domain size is discounted as negligible relative to `|F|`.
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

    /// Smallest number of OOD samples (from the extension field) needed
    /// for [`Self::ood_error`] to clear `security_level` bits.
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

    /// Sumcheck soundness term for the fold step (bits).
    ///
    /// During folding, the verifier samples a random challenge and
    /// checks a degree-2 sumcheck identity. An adversary controlling a
    /// list of L codewords can bias the check with probability at most
    /// `L / |F|`:
    ///
    /// `bits = field_size_bits − (list_size_bits + 1)`
    ///
    /// The `+1` is the union bound over the list.
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

    /// Query-combination soundness (bits). After STIR queries and OOD
    /// samples, the verifier takes a random linear combination of all
    /// evaluations; an adversary must fool this combination for every
    /// codeword in the list:
    ///
    /// `error = (ood_samples + num_queries) · L⁺ / |F|`,
    /// `bits  = field_size_bits − (log₂(ood + queries) + list_size_bits + 1)`.
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

    /// PoW bits needed at the fold step to bridge the weaker of the
    /// proximity-gap and fold-sumcheck bounds to `security_level`.
    /// Returns 0 when the algebraic bounds already meet the target.
    #[must_use]
    pub fn folding_pow_bits(
        &self,
        security_level: usize,
        field_size_bits: usize,
        num_variables: usize,
        log_inv_rate: usize,
    ) -> f64 {
        // Arity 2 because each fold combines exactly 2 evaluations.
        let prox_gaps_error = self.prox_gaps_error(num_variables, log_inv_rate, field_size_bits, 2);
        let sumcheck_error = self.fold_sumcheck_error(field_size_bits, num_variables, log_inv_rate);

        // The fold step is only as strong as its weakest component.
        let error = prox_gaps_error.min(sumcheck_error);

        // PoW closes the residual gap; zero means no grinding needed.
        0_f64.max(security_level as f64 - error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Field size in bits used by the WHIR-style budget regression test.
    /// 5 × ⌈log₂(p_KoalaBear)⌉ — degree-5 KoalaBear extension.
    const KOALABEAR_QUINTIC_BITS: usize = 155;

    /// Practical ceiling for Fiat-Shamir grinding. Above ~30 bits,
    /// grinding becomes impractical for honest provers.
    const MAX_POW_BITS: f64 = 30.0;

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
    fn jb_prox_gap_covers_security_level_minus_pow_over_koalabear_quintic() {
        // Prox-gap bound alone clears (security_level - MAX_POW_BITS),
        // so a feasible PoW budget bridges the rest.
        let jb = SecurityAssumption::JohnsonBound;
        let security_level: f64 = 128.0;
        let min_required_bits = security_level - MAX_POW_BITS;

        for log_inv_rate in 1..=2 {
            for log_degree in 10..=22 {
                let prox_gap_bits =
                    jb.prox_gaps_error(log_degree, log_inv_rate, KOALABEAR_QUINTIC_BITS, 2);

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
        // Full WHIR soundness budget reaches `security_level` at a
        // representative configuration.
        //
        // PoW-eligible components (>= security_level - MAX_POW_BITS):
        //   prox-gap, sumcheck, query-linear-combination
        // Self-sustaining components (>= security_level):
        //   out-of-domain sample, FRI query phase
        // Total grinding must not exceed MAX_POW_BITS.
        let jb = SecurityAssumption::JohnsonBound;
        let security_level: usize = 128;
        let min_with_pow = security_level as f64 - MAX_POW_BITS;
        let log_degree = 20;
        let log_inv_rate = 2;

        let num_queries = jb.queries(security_level, log_inv_rate);
        let ood_samples = jb.determine_ood_samples(
            security_level,
            log_degree,
            log_inv_rate,
            KOALABEAR_QUINTIC_BITS,
        );

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
        assert!(
            ood >= security_level as f64,
            "OOD {ood:.2} bits < {security_level}"
        );
        assert!(
            query >= security_level as f64,
            "query {query:.2} bits < {security_level}"
        );

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
}
