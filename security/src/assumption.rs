//! Security assumption (regime selector) and the Reed–Solomon proximity-gap
//! primitives that any RS-IOP can share.
//!
//! WHIR / STIR composition lives in [`crate::whir`]; FRI's per-regime error
//! orchestration lives in [`crate::fri`]. Both consume the methods on
//! [`SecurityAssumption`].
//!
//! # References
//! - **[BCI+20]** Ben-Sasson, Carmon, Ishai, Kopparty, Saraf.
//!   *Proximity Gaps for Reed-Solomon Codes*. FOCS 2020.
//!   <https://eprint.iacr.org/2020/654>
//! - **\[BCSS25\]** Ben-Sasson, Carmon, Haboeck, Kopparty, Saraf.
//!   *On Proximity Gaps for Reed-Solomon Codes*.
//!   <https://eprint.iacr.org/2025/2055>
//!
//! \[BCSS25\] improves the Johnson-bound proximity gap from `O(n²/η⁷)` to
//! `O(n/η⁵)`, enabling 128-bit provable security with degree-5 extensions
//! of small prime fields (e.g. KoalaBear).

use alloc::format;
use alloc::string::String;
use core::f64::consts::LOG2_10;
use core::fmt::Display;
use core::str::FromStr;

use serde::Serialize;

/// \[BCSS25\] Theorem 1.5 dominant term, in bits:
/// `log_2(2·(m + 1/2)⁵ / (3·ρ^{3/2}) · n)`. Shared by
/// [`SecurityAssumption::prox_gaps_error`] (fixed `m = 10`) and
/// [`SecurityAssumption::prox_gaps_error_jb_at_m`] (explicit `m`).
fn jb_prox_gaps_dominant_term_bits(log_degree: usize, log_inv_rate: usize, m: usize) -> f64 {
    let log_n = (log_degree + log_inv_rate) as f64;
    let constant = libm::log2(2. * libm::pow(m as f64 + 0.5, 5.) / 3.);
    let log_rho_neg_3_2 = 1.5 * log_inv_rate as f64;
    log_n + constant + log_rho_neg_3_2
}

/// Proximity regime selector for Reed–Solomon-based IOPs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum SecurityAssumption {
    /// Unique decoding: each oracle is within the UDR. No conjectures.
    UniqueDecoding,

    /// Johnson bound at `δ = 1 − √ρ − η`, with `η = √ρ / 20`. Requires
    /// mutual correlated agreement up to the Johnson bound.
    ///
    /// The proximity-gap error uses \[BCSS25\] Theorem 1.5:
    /// `a > (2(m + 1/2)⁵ + 3(m + 1/2)γρ) / (3ρ^{3/2}) · n + (m + 1/2)/√ρ`,
    /// asymptotically `O(n/η⁵)` — a `n·η²` improvement over [BCI+20].
    JohnsonBound,

    /// Capacity bound at `δ = 1 − ρ − η`, with `η = ρ / 20`. Requires
    /// conjecturing capacity-rate list decodability and correlated
    /// agreement up to capacity.
    CapacityBound,
}

impl SecurityAssumption {
    /// `log₂(η)`, where η is the safety gap below the regime's distance.
    ///
    /// # Panics
    /// Undefined for [`SecurityAssumption::UniqueDecoding`] (UD uses
    /// `δ = (1 − ρ)/2`, no η term). Callers must branch on UD first; the
    /// panic locks down that invariant.
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

    /// `log₂(η)`, or `0.` for [`Self::UniqueDecoding`], where no eta term applies and every
    /// `_at_log_eta` formula ignores the value in its UD branch.
    ///
    /// Public alongside the `_at_log_eta` family: a caller feeding those its own schedule of
    /// eta values still needs this convention to reproduce the fixed-eta entry points, and
    /// has no other way to spell "the regime's default" for `UniqueDecoding`.
    #[must_use]
    pub const fn log_eta_or_zero(&self, log_inv_rate: usize) -> f64 {
        match self {
            Self::UniqueDecoding => 0.,
            _ => self.log_eta(log_inv_rate),
        }
    }

    /// `log₂(L⁺)` for the regime's list size at an explicit `log_eta`, rather than the
    /// regime's own default safety margin ([`Self::log_eta`]).
    ///
    /// A caller deriving its own schedule of `eta` values across a protocol (STIR's per-round
    /// bisection, for instance) needs the list size at each of those working points, not just
    /// at the regime's fixed default; [`Self::list_size_bits`] is the
    /// `log_eta = self.log_eta(log_inv_rate)` specialization of this.
    #[must_use]
    pub const fn list_size_bits_at_log_eta(
        &self,
        log_degree: usize,
        log_inv_rate: usize,
        log_eta: f64,
    ) -> f64 {
        match self {
            // In UD the list size is 1
            Self::UniqueDecoding => 0.,

            // By the JB, RS codes are (1 - sqrt(rho) - eta, (2*eta*sqrt(rho))^-1)-list decodable.
            Self::JohnsonBound => {
                let log_inv_sqrt_rate: f64 = log_inv_rate as f64 / 2.;
                log_inv_sqrt_rate - (1. + log_eta)
            }

            // In CB we assume that RS codes are (1 - rho - eta, d/rho*eta)-list decodable (see Conjecture 5.6 in STIR).
            Self::CapacityBound => (log_degree + log_inv_rate) as f64 - log_eta,
        }
    }

    /// `log₂(L⁺)` for the regime's list size at distance δ.
    #[must_use]
    pub const fn list_size_bits(&self, log_degree: usize, log_inv_rate: usize) -> f64 {
        self.list_size_bits_at_log_eta(log_degree, log_inv_rate, self.log_eta_or_zero(log_inv_rate))
    }

    /// Proximity-gap error in bits for combining `num_functions` functions
    /// at the regime's distance.
    ///
    /// The Johnson-bound branch uses \[BCSS25\] Theorem 1.5 at the fixed
    /// safety choice `m = max(ceil(sqrt(rho)/(2*eta)), 3) = 10` (η = √ρ/20,
    /// see [`Self::log_eta`]). Only the dominant term
    /// `2·(m + 1/2)⁵ / (3·ρ^{3/2}) · n` is kept; the additive `(m + 1/2)/√ρ`
    /// and sub-dominant `3·(m + 1/2)·γ·ρ` terms are negligible at `m = 10`.
    /// Use [`Self::prox_gaps_error_jb_at_m`] when the surrounding regime
    /// decodes at a different explicit `m` (e.g. FRI's `best_m`) — the
    /// fixed `m = 10` here is a WHIR-style default, not necessarily the `m`
    /// the caller's list-decoding regime actually operates at.
    #[must_use]
    pub fn prox_gaps_error(
        &self,
        log_degree: usize,
        log_inv_rate: usize,
        field_size_bits: usize,
        num_functions: usize,
    ) -> f64 {
        self.prox_gaps_error_at_log_eta(
            log_degree,
            log_inv_rate,
            field_size_bits,
            num_functions,
            self.log_eta_or_zero(log_inv_rate),
        )
    }

    /// [`Self::prox_gaps_error`] at an explicit `log_eta`, rather than the regime's own
    /// default safety margin ([`Self::log_eta`]).
    ///
    /// A caller deriving its own schedule of `eta` values across a protocol (STIR's per-round
    /// bisection, for instance) needs the proximity-gap error at each of those working
    /// points, not just at the regime's fixed default. On [`SecurityAssumption::JohnsonBound`]
    /// this derives the \[BCSS25\] proximity parameter `m = max(ceil(√ρ / (2η)), 3)` from
    /// `log_eta` and defers to [`Self::prox_gaps_error_jb_at_m`] — at
    /// `log_eta = self.log_eta(log_inv_rate)` that derivation reduces to exactly `m = 10`,
    /// [`Self::prox_gaps_error`]'s fixed safety choice.
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

        match self {
            // In UD the error is |L|/|F| = d/(rho*|F|)
            Self::UniqueDecoding => {
                let error = (log_degree + log_inv_rate) as f64;
                let num_functions_1_log = libm::log2(num_functions as f64 - 1.);
                field_size_bits as f64 - (error + num_functions_1_log)
            }

            // From Theorem 1.5 in [BCSS25] "On Proximity Gaps for Reed-Solomon Codes":
            //
            // For gamma < J(delta) - eta, the number of exceptional z's is bounded by:
            //   a > (2(m + 1/2)^5 + 3(m + 1/2)*gamma*rho) / (3*rho^(3/2)) * n + (m + 1/2) / sqrt(rho)
            //
            // m = max(ceil(sqrt(rho)/(2*eta)), 3).
            Self::JohnsonBound => {
                let log_sqrt_rho_over_2eta = -(log_inv_rate as f64) / 2. - 1. - log_eta;
                let m = libm::ceil(libm::pow(2., log_sqrt_rho_over_2eta)).max(3.) as usize;
                Self::prox_gaps_error_jb_at_m(
                    log_degree,
                    log_inv_rate,
                    field_size_bits,
                    num_functions,
                    m,
                )
            }

            // In CB we assume the error is degree/(eta*rho^2)
            Self::CapacityBound => {
                let error = (log_degree + 2 * log_inv_rate) as f64 - log_eta;
                let num_functions_1_log = libm::log2(num_functions as f64 - 1.);
                field_size_bits as f64 - (error + num_functions_1_log)
            }
        }
    }

    /// Johnson-bound proximity-gap error (\[BCSS25\] Theorem 1.5, dominant
    /// term) at an explicit proximity parameter `m`, rather than the fixed
    /// `m = 10` safety choice [`Self::prox_gaps_error`] uses.
    ///
    /// Only the dominant term `2·(m + 1/2)⁵ / (3·ρ^{3/2}) · n` is kept; see
    /// [`Self::prox_gaps_error`] for the full derivation and the terms this
    /// drops. Those terms remain negligible for any `m` in FRI's searched
    /// range (`m ∈ [3, 1000]`): the dropped `3·(m + 1/2)·γ·ρ` sub-term is
    /// smaller than the kept `2·(m + 1/2)⁵` term by a factor of
    /// `2·(m + 1/2)⁴ / (3·γ)`, which grows with `m`.
    ///
    /// For use when the caller already knows the `m` the surrounding
    /// list-decoding regime decodes at (e.g. FRI's `best_m` from
    /// [`crate::fri::best_ldr_m`]) and needs the batch-combination term
    /// evaluated at that same radius rather than the WHIR-style fixed
    /// safety margin.
    #[must_use]
    pub fn prox_gaps_error_jb_at_m(
        log_degree: usize,
        log_inv_rate: usize,
        field_size_bits: usize,
        num_functions: usize,
        m: usize,
    ) -> f64 {
        assert!(
            num_functions >= 2,
            "num_functions must be >= 2 to compute proximity gaps error",
        );
        let error = jb_prox_gaps_dominant_term_bits(log_degree, log_inv_rate, m);
        let num_functions_1_log = libm::log2(num_functions as f64 - 1.);
        field_size_bits as f64 - (error + num_functions_1_log)
    }

    /// `log₂(1 − δ)` for the regime's distance δ.
    /// - UD: δ = (1 − ρ)/2
    /// - JB: δ = 1 − √ρ − η
    /// - CB: δ = 1 − ρ − η
    #[must_use]
    pub fn log_1_delta(&self, log_inv_rate: usize) -> f64 {
        let log_twenty_one_over_twenty = libm::log2(21. / 20.);
        match self {
            Self::UniqueDecoding => libm::log2(1. + libm::pow(2., -(log_inv_rate as f64))) - 1.,
            Self::JohnsonBound => log_twenty_one_over_twenty - 0.5 * log_inv_rate as f64,
            Self::CapacityBound => log_twenty_one_over_twenty - log_inv_rate as f64,
        }
    }

    /// Number of queries needed for `(1 − δ)^t < 2^{−λ}`.
    #[must_use]
    pub fn queries(&self, protocol_security_level: usize, log_inv_rate: usize) -> usize {
        let num_queries_f = -(protocol_security_level as f64) / self.log_1_delta(log_inv_rate);

        libm::ceil(num_queries_f) as usize
    }

    /// Bits of security from `num_queries` queries.
    #[must_use]
    pub fn queries_error(&self, log_inv_rate: usize, num_queries: usize) -> f64 {
        let num_queries = num_queries as f64;

        -num_queries * self.log_1_delta(log_inv_rate)
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

    /// Field size in bits used by the BCSS25 regression tests.
    ///
    /// Equals `5 * ceil(log_2(p_KoalaBear))` with `p_KoalaBear = 2^31 - 2^24 + 1`,
    /// i.e. a degree-5 extension of the KoalaBear prime field. The smallest
    /// extension that gives the [BCSS25] bound enough headroom for 128-bit
    /// WHIR soundness in the regimes tested.
    pub(crate) const KOALABEAR_QUINTIC_BITS: usize = 155;

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
    #[should_panic(expected = "log_eta is undefined for UniqueDecoding")]
    fn log_eta_panics_for_unique_decoding() {
        // eta does not appear in the UD distance formula `delta = (1 - rho) / 2`.
        // Reading log_eta in the UD branch is a programmer error; the panic
        // locks that down so a future refactor that strays into the eta path
        // under UD fails loudly instead of silently propagating a bogus value.
        let _ = SecurityAssumption::UniqueDecoding.log_eta(5);
    }

    #[test]
    fn log_one_minus_delta_is_stable_at_large_inverse_rates() {
        let log_twenty_one_over_twenty = libm::log2(21. / 20.);

        for log_inv_rate in [31, 32, 63] {
            let unique = SecurityAssumption::UniqueDecoding.log_1_delta(log_inv_rate);
            let johnson = SecurityAssumption::JohnsonBound.log_1_delta(log_inv_rate);
            let capacity = SecurityAssumption::CapacityBound.log_1_delta(log_inv_rate);

            assert!(unique.is_finite());
            assert!(johnson.is_finite());
            assert!(capacity.is_finite());
            assert!(
                (unique - (libm::log2(1. + libm::pow(2., -(log_inv_rate as f64))) - 1.)).abs()
                    < 1e-12
            );
            assert!(
                (johnson - (log_twenty_one_over_twenty - 0.5 * log_inv_rate as f64)).abs() < 1e-12
            );
            assert!((capacity - (log_twenty_one_over_twenty - log_inv_rate as f64)).abs() < 1e-12);
        }
    }

    /// Old prox-gap baseline used by the improvement test.
    ///
    /// [BCI+20] Theorem 5.1 at η = √ρ/20 (m = 10):
    /// `|S| > (m + 1/2)^7 / 3 · n^2 / ρ^{3/2}`.
    fn bci20_jb_prox_gaps_error(
        log_degree: usize,
        log_inv_rate: usize,
        field_size_bits: usize,
    ) -> f64 {
        const M_PLUS_HALF: f64 = 10.5;
        let log_n_squared = 2.0 * (log_degree + log_inv_rate) as f64;
        let log_leading_constant = libm::log2(libm::pow(M_PLUS_HALF, 7.0) / 3.0);
        let log_rho_pow_neg_three_halves = 1.5 * log_inv_rate as f64;
        let error_bits = log_n_squared + log_leading_constant + log_rho_pow_neg_three_halves;
        field_size_bits as f64 - error_bits
    }

    #[test]
    fn jb_prox_gap_strictly_improves_over_old_bound() {
        // gap = log_2((m + 1/2)^2 / 2) + log_2(n) bits over [BCI+20], with
        // log_2(55.125) ~= 5.78 at the safety choice m = 10.
        let jb = SecurityAssumption::JohnsonBound;
        let leading_ratio_log = libm::log2(10.5_f64.powi(2) / 2.0);

        for log_degree in 10..=25 {
            for log_inv_rate in 1..=4 {
                let new_bits =
                    jb.prox_gaps_error(log_degree, log_inv_rate, KOALABEAR_QUINTIC_BITS, 2);
                let old_bits =
                    bci20_jb_prox_gaps_error(log_degree, log_inv_rate, KOALABEAR_QUINTIC_BITS);

                assert!(
                    new_bits > old_bits,
                    "no improvement at log_degree={log_degree}, log_inv_rate={log_inv_rate}: \
                     new={new_bits:.4}, old={old_bits:.4}"
                );

                let log_n = (log_degree + log_inv_rate) as f64;
                let observed = new_bits - old_bits;
                let expected = log_n + leading_ratio_log;

                assert!(
                    (observed - expected).abs() < 1e-9,
                    "gap mismatch at log_degree={log_degree}, log_inv_rate={log_inv_rate}: \
                     expected={expected:.6}, got={observed:.6}"
                );
            }
        }
    }

    #[test]
    fn jb_prox_gap_scales_by_log_curve_degree() {
        // [BCSS25] Thm 4.2: combining M+1 functions costs log_2(M) bits.
        let jb = SecurityAssumption::JohnsonBound;
        let log_degree = 20;
        let log_inv_rate = 2;

        let line_bits = jb.prox_gaps_error(log_degree, log_inv_rate, KOALABEAR_QUINTIC_BITS, 2);

        for (num_functions, expected_loss) in [(3_usize, 1.0_f64), (5, 2.0), (9, 3.0)] {
            let curve_bits = jb.prox_gaps_error(
                log_degree,
                log_inv_rate,
                KOALABEAR_QUINTIC_BITS,
                num_functions,
            );

            let loss = line_bits - curve_bits;

            assert!(
                (loss - expected_loss).abs() < 1e-9,
                "curve scaling off at num_functions={num_functions}: \
                 expected log_2({}) = {expected_loss:.1} bits, got {loss:.6}",
                num_functions - 1
            );
        }
    }

    /// Field size used by the known-answer table below: a degree-5 extension of KoalaBear.
    const TABLE_FIELD_BITS: usize = 155;

    #[test]
    fn fixed_eta_formulas_match_their_recorded_values() {
        // These five entry points feed WHIR's, FRI's and STIR's derived parameters, so any
        // drift in them silently changes every protocol's security. Freezing the values is
        // the point: a property test would need the pre-refactor implementation as its
        // oracle, and that no longer exists.
        //
        // Columns: list_size_bits, prox_gaps_error(num_functions = 17),
        // ood_error(ood_samples = 2), fold_sumcheck_error,
        // queries_combination_error(ood_samples = 2, num_queries = 40).
        #[allow(clippy::type_complexity)]
        let table: [(SecurityAssumption, usize, usize, [f64; 5]); 45] = [
            (
                SecurityAssumption::JohnsonBound,
                1,
                10,
                [
                    4.321928094887362,
                    122.12337538682735,
                    282.35614381022526,
                    149.67807190511263,
                    144.28575448233389,
                ],
            ),
            (
                SecurityAssumption::JohnsonBound,
                1,
                20,
                [
                    4.321928094887362,
                    112.12337538682735,
                    262.35614381022526,
                    149.67807190511263,
                    144.28575448233389,
                ],
            ),
            (
                SecurityAssumption::JohnsonBound,
                1,
                24,
                [
                    4.321928094887362,
                    108.12337538682735,
                    254.35614381022526,
                    149.67807190511263,
                    144.28575448233389,
                ],
            ),
            (
                SecurityAssumption::JohnsonBound,
                2,
                10,
                [
                    5.321928094887362,
                    119.62337538682735,
                    280.35614381022526,
                    148.67807190511263,
                    143.28575448233389,
                ],
            ),
            (
                SecurityAssumption::JohnsonBound,
                2,
                20,
                [
                    5.321928094887362,
                    109.62337538682735,
                    260.35614381022526,
                    148.67807190511263,
                    143.28575448233389,
                ],
            ),
            (
                SecurityAssumption::JohnsonBound,
                2,
                24,
                [
                    5.321928094887362,
                    105.62337538682735,
                    252.35614381022526,
                    148.67807190511263,
                    143.28575448233389,
                ],
            ),
            (
                SecurityAssumption::JohnsonBound,
                3,
                10,
                [
                    6.321928094887362,
                    117.12337538682735,
                    278.35614381022526,
                    147.67807190511263,
                    142.28575448233389,
                ],
            ),
            (
                SecurityAssumption::JohnsonBound,
                3,
                20,
                [
                    6.321928094887362,
                    107.12337538682735,
                    258.35614381022526,
                    147.67807190511263,
                    142.28575448233389,
                ],
            ),
            (
                SecurityAssumption::JohnsonBound,
                3,
                24,
                [
                    6.321928094887362,
                    103.12337538682735,
                    250.35614381022526,
                    147.67807190511263,
                    142.28575448233389,
                ],
            ),
            (
                SecurityAssumption::JohnsonBound,
                4,
                10,
                [
                    7.321928094887362,
                    114.62337538682735,
                    276.35614381022526,
                    146.67807190511263,
                    141.28575448233389,
                ],
            ),
            (
                SecurityAssumption::JohnsonBound,
                4,
                20,
                [
                    7.321928094887362,
                    104.62337538682735,
                    256.35614381022526,
                    146.67807190511263,
                    141.28575448233389,
                ],
            ),
            (
                SecurityAssumption::JohnsonBound,
                4,
                24,
                [
                    7.321928094887362,
                    100.62337538682735,
                    248.35614381022526,
                    146.67807190511263,
                    141.28575448233389,
                ],
            ),
            (
                SecurityAssumption::JohnsonBound,
                8,
                10,
                [
                    11.321928094887362,
                    104.62337538682735,
                    268.35614381022526,
                    142.67807190511263,
                    137.28575448233389,
                ],
            ),
            (
                SecurityAssumption::JohnsonBound,
                8,
                20,
                [
                    11.321928094887362,
                    94.62337538682735,
                    248.35614381022526,
                    142.67807190511263,
                    137.28575448233389,
                ],
            ),
            (
                SecurityAssumption::JohnsonBound,
                8,
                24,
                [
                    11.321928094887362,
                    90.62337538682735,
                    240.35614381022526,
                    142.67807190511263,
                    137.28575448233389,
                ],
            ),
            (
                SecurityAssumption::CapacityBound,
                1,
                10,
                [
                    16.32192809488736,
                    133.67807190511263,
                    258.35614381022526,
                    137.67807190511263,
                    132.28575448233389,
                ],
            ),
            (
                SecurityAssumption::CapacityBound,
                1,
                20,
                [
                    26.32192809488736,
                    123.67807190511263,
                    218.35614381022526,
                    127.67807190511263,
                    122.28575448233389,
                ],
            ),
            (
                SecurityAssumption::CapacityBound,
                1,
                24,
                [
                    30.32192809488736,
                    119.67807190511263,
                    202.35614381022526,
                    123.67807190511263,
                    118.28575448233389,
                ],
            ),
            (
                SecurityAssumption::CapacityBound,
                2,
                10,
                [
                    18.32192809488736,
                    130.67807190511263,
                    254.35614381022526,
                    135.67807190511263,
                    130.28575448233389,
                ],
            ),
            (
                SecurityAssumption::CapacityBound,
                2,
                20,
                [
                    28.32192809488736,
                    120.67807190511263,
                    214.35614381022526,
                    125.67807190511263,
                    120.28575448233389,
                ],
            ),
            (
                SecurityAssumption::CapacityBound,
                2,
                24,
                [
                    32.32192809488736,
                    116.67807190511263,
                    198.35614381022526,
                    121.67807190511263,
                    116.28575448233389,
                ],
            ),
            (
                SecurityAssumption::CapacityBound,
                3,
                10,
                [
                    20.32192809488736,
                    127.67807190511263,
                    250.35614381022526,
                    133.67807190511263,
                    128.28575448233389,
                ],
            ),
            (
                SecurityAssumption::CapacityBound,
                3,
                20,
                [
                    30.32192809488736,
                    117.67807190511263,
                    210.35614381022526,
                    123.67807190511263,
                    118.28575448233389,
                ],
            ),
            (
                SecurityAssumption::CapacityBound,
                3,
                24,
                [
                    34.32192809488736,
                    113.67807190511263,
                    194.35614381022526,
                    119.67807190511263,
                    114.28575448233389,
                ],
            ),
            (
                SecurityAssumption::CapacityBound,
                4,
                10,
                [
                    22.32192809488736,
                    124.67807190511263,
                    246.35614381022526,
                    131.67807190511263,
                    126.28575448233389,
                ],
            ),
            (
                SecurityAssumption::CapacityBound,
                4,
                20,
                [
                    32.32192809488736,
                    114.67807190511263,
                    206.35614381022526,
                    121.67807190511263,
                    116.28575448233389,
                ],
            ),
            (
                SecurityAssumption::CapacityBound,
                4,
                24,
                [
                    36.32192809488736,
                    110.67807190511263,
                    190.35614381022526,
                    117.67807190511263,
                    112.28575448233389,
                ],
            ),
            (
                SecurityAssumption::CapacityBound,
                8,
                10,
                [
                    30.32192809488736,
                    112.67807190511263,
                    230.35614381022526,
                    123.67807190511263,
                    118.28575448233389,
                ],
            ),
            (
                SecurityAssumption::CapacityBound,
                8,
                20,
                [
                    40.32192809488736,
                    102.67807190511263,
                    190.35614381022526,
                    113.67807190511263,
                    108.28575448233389,
                ],
            ),
            (
                SecurityAssumption::CapacityBound,
                8,
                24,
                [
                    44.32192809488736,
                    98.67807190511263,
                    174.35614381022526,
                    109.67807190511263,
                    104.28575448233389,
                ],
            ),
            (
                SecurityAssumption::UniqueDecoding,
                1,
                10,
                [0.0, 140.0, 0.0, 154.0, 148.60768257722123],
            ),
            (
                SecurityAssumption::UniqueDecoding,
                1,
                20,
                [0.0, 130.0, 0.0, 154.0, 148.60768257722123],
            ),
            (
                SecurityAssumption::UniqueDecoding,
                1,
                24,
                [0.0, 126.0, 0.0, 154.0, 148.60768257722123],
            ),
            (
                SecurityAssumption::UniqueDecoding,
                2,
                10,
                [0.0, 139.0, 0.0, 154.0, 148.60768257722123],
            ),
            (
                SecurityAssumption::UniqueDecoding,
                2,
                20,
                [0.0, 129.0, 0.0, 154.0, 148.60768257722123],
            ),
            (
                SecurityAssumption::UniqueDecoding,
                2,
                24,
                [0.0, 125.0, 0.0, 154.0, 148.60768257722123],
            ),
            (
                SecurityAssumption::UniqueDecoding,
                3,
                10,
                [0.0, 138.0, 0.0, 154.0, 148.60768257722123],
            ),
            (
                SecurityAssumption::UniqueDecoding,
                3,
                20,
                [0.0, 128.0, 0.0, 154.0, 148.60768257722123],
            ),
            (
                SecurityAssumption::UniqueDecoding,
                3,
                24,
                [0.0, 124.0, 0.0, 154.0, 148.60768257722123],
            ),
            (
                SecurityAssumption::UniqueDecoding,
                4,
                10,
                [0.0, 137.0, 0.0, 154.0, 148.60768257722123],
            ),
            (
                SecurityAssumption::UniqueDecoding,
                4,
                20,
                [0.0, 127.0, 0.0, 154.0, 148.60768257722123],
            ),
            (
                SecurityAssumption::UniqueDecoding,
                4,
                24,
                [0.0, 123.0, 0.0, 154.0, 148.60768257722123],
            ),
            (
                SecurityAssumption::UniqueDecoding,
                8,
                10,
                [0.0, 133.0, 0.0, 154.0, 148.60768257722123],
            ),
            (
                SecurityAssumption::UniqueDecoding,
                8,
                20,
                [0.0, 123.0, 0.0, 154.0, 148.60768257722123],
            ),
            (
                SecurityAssumption::UniqueDecoding,
                8,
                24,
                [0.0, 119.0, 0.0, 154.0, 148.60768257722123],
            ),
        ];

        for (assumption, log_inv_rate, log_degree, expected) in table {
            let actual = [
                assumption.list_size_bits(log_degree, log_inv_rate),
                assumption.prox_gaps_error(log_degree, log_inv_rate, TABLE_FIELD_BITS, 17),
                assumption.ood_error(log_degree, log_inv_rate, TABLE_FIELD_BITS, 2),
                assumption.fold_sumcheck_error(TABLE_FIELD_BITS, log_degree, log_inv_rate),
                assumption.queries_combination_error(
                    TABLE_FIELD_BITS,
                    log_degree,
                    log_inv_rate,
                    2,
                    40,
                ),
            ];
            assert_eq!(
                actual, expected,
                "{assumption:?} log_inv_rate={log_inv_rate} log_degree={log_degree}"
            );
        }
    }

    #[test]
    fn jb_prox_gaps_default_eta_is_m_10() {
        // `prox_gaps_error`'s Johnson branch delegates to `prox_gaps_error_jb_at_m` through a
        // `f64 -> usize` round-trip whose input lands one ULP *below* 10 — on the correct side
        // of the `ceil`, but only just. Reassociating the `log_eta` expression could push the
        // residue the other way, making `m = 11` and moving every JB proximity-gap error by
        // about 0.65 bits, with nothing else in the suite noticing.
        let jb = SecurityAssumption::JohnsonBound;
        for log_inv_rate in 0..=32 {
            for log_degree in [10, 20, 24] {
                for num_functions in [2, 3, 17] {
                    assert_eq!(
                        jb.prox_gaps_error(log_degree, log_inv_rate, 128, num_functions),
                        SecurityAssumption::prox_gaps_error_jb_at_m(
                            log_degree,
                            log_inv_rate,
                            128,
                            num_functions,
                            10,
                        ),
                        "log_inv_rate={log_inv_rate} log_degree={log_degree} \
                         num_functions={num_functions}"
                    );
                }
            }
        }
    }
}
