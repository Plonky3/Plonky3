use alloc::{format, string::String};
use core::{
    f64::consts::LOG2_10,
    fmt::{Debug, Display},
    str::FromStr,
};

/// This module, used to determine proximity gaps in the relevant Reed-Solomon
/// codes based on various security assumptions (and therefore the necessary
/// number of queries and proof-of-work bits), is directly taken from the
/// co-author Giacomo Fenzi's
/// [implementation](https://github.com/WizardOfMenlo/stir-whir-scripts/blob/main/src/errors.rs)

/// Security assumptions determines which proximity parameters and conjectures are assumed by the error computation.
#[derive(Debug, Clone, Copy)]
pub enum SecurityAssumption {
    /// Unique decoding assumes that the distance of each oracle is within the UDR of the code.
    /// We refer to this configuration as UD for short.
    UniqueDecoding,

    /// Johnson bound assumes that the distance of each oracle is within the Johnson bound (1 - sqrt(rho)).
    /// We refer to this configuration as JB for short.
    /// This does not rely on any conjectures (in STIR).
    JohnsonBound,

    /// Capacity bound assumes that the distance of each oracle is within the capacity bound 1 - rho.
    /// We refer to this configuration as CB for short.
    /// This relies on the conjecture that RS codes are decodable up to capacity and have correlated agreement up to capacity.
    CapacityBound,
}

impl SecurityAssumption {
    /// In both JB and CB theorems such as list-size only hold for proximity parameters slighly below the bound.
    /// E.g. in JB proximity gaps holds for every delta in (0, 1 - sqrt(rho)).
    /// eta is the distance between the chosen proximity parameter and the bound.
    /// I.e. in JB delta = 1 - sqrt(rho) - eta and in CB delta = 1 - rho - eta.
    // TODO: Maybe it makes more sense to be multiplicative. I think this can be set in a better way.
    pub fn log_eta(&self, log_inv_rate: usize) -> f64 {
        // Editor's note: This is Giacom's original comment:
        //     Ask me how I did this? At the time, only God and I knew. Now only God knows
        //     I joke, I actually know but this is left for posterity.
        match self {
            // We don't use eta in UD
            Self::UniqueDecoding => 0., // TODO: Maybe just panic and avoid calling it in UD?
            // Set as sqrt(rho)/20
            Self::JohnsonBound => -(0.5 * log_inv_rate as f64 + LOG2_10 + 1.),
            // Set as rho/20
            Self::CapacityBound => -(log_inv_rate as f64 + LOG2_10 + 1.),
        }
    }

    /// Given a RS code (specified by the log of the degree and log inv of the rate), compute the list size at the specified distance delta.
    pub fn list_size_bits(&self, log_degree: usize, log_inv_rate: usize) -> f64 {
        let log_eta = self.log_eta(log_inv_rate);
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

    /// Given a RS code (specified by the log of the degree and log inv of the rate) a field_size and an arity, compute the proximity gaps error (in bits) at the specified distance
    pub fn prox_gaps_error(
        &self,
        log_degree: usize,
        log_inv_rate: usize,
        field_size_bits: usize,
        num_functions: usize,
    ) -> f64 {
        // The error computed here is from [BCIKS20] for the combination of two functions. Then we multiply it by the folding factor.
        let log_eta = self.log_eta(log_inv_rate);
        // Note that this does not include the field_size
        let error = match self {
            // In UD the error is |L|/|F| = d/ρ*|F|
            Self::UniqueDecoding => (log_degree + log_inv_rate) as f64,

            // In JB the error is degree^2/|F| * (2 * min{ 1 - sqrt(rho) - delta, sqrt(rho)/20 })^7
            // Since delta = 1 - sqrt(rho) - eta then 1 - sqrt(rho) - delta = eta
            // Thus the error is degree^2/|F| * (2 * min { eta, sqrt(rho)/20 })^7
            Self::JohnsonBound => {
                let numerator = (2 * log_degree) as f64;
                let sqrt_rho_20 = 1. + LOG2_10 + 0.5 * log_inv_rate as f64;
                numerator + 7. * (sqrt_rho_20.min(-log_eta) - 1.)
            }

            // In CB we assume the error is degree/eta*rho^2
            Self::CapacityBound => (log_degree + 2 * log_inv_rate) as f64 - log_eta,
        };

        // Error is  (num_functions - 1) * error/|F|;
        let num_functions_1_log = (num_functions as f64 - 1.).log2();
        field_size_bits as f64 - (error + num_functions_1_log as f64)
    }

    /// The query error is (1 - delta)^t where t is the number of queries.
    /// This computes log(1 - delta).
    /// In UD, delta is (1 - rho)/2
    /// In JB, delta is (1 - sqrt(rho) - eta)
    /// In CB, delta is (1 - rho - eta)
    pub fn log_1_delta(&self, log_inv_rate: usize) -> f64 {
        let log_eta = self.log_eta(log_inv_rate);
        let eta = 2_f64.powf(log_eta);
        let rate = 1. / (1 << log_inv_rate) as f64;

        let delta = match self {
            Self::UniqueDecoding => 0.5 * (1. - rate),
            Self::JohnsonBound => 1. - rate.sqrt() - eta,
            Self::CapacityBound => 1. - rate - eta,
        };

        (1. - delta).log2()
    }

    /// Compute the number of queries to match the security level
    /// The error to drive down is (1-delta)^t < 2^-lambda.
    /// Where delta is set as in the `log_1_delta` function.
    pub fn queries(&self, protocol_security_level: usize, log_inv_rate: usize) -> usize {
        let num_queries_f = -(protocol_security_level as f64) / self.log_1_delta(log_inv_rate);

        num_queries_f.ceil() as usize
    }

    /// Compute the error for the given number of queries
    /// The error to drive down is (1-delta)^t < 2^-lambda.
    /// Where delta is set as in the `log_1_delta` function.
    pub fn queries_error(&self, log_inv_rate: usize, num_queries: usize) -> f64 {
        let num_queries = num_queries as f64;

        -num_queries * self.log_1_delta(log_inv_rate)
    }

    /// Compute the error for the OOD samples of the protocol
    /// See Lemma 4.5 in STIR.
    /// The error is list_size^2 * (degree/field_size_bits)^reps
    /// NOTE: Here we are discounting the domain size as we assume it is negligible compared to the size of the field.
    pub fn ood_error(
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
    /// We note that there are various strategies to set OOD samples.
    /// In this case, we are just sampling one element from the extension field
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
}

impl Display for SecurityAssumption {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "{}",
            match &self {
                SecurityAssumption::UniqueDecoding => "UniqueDecoding",
                SecurityAssumption::JohnsonBound => "JohnsonBound",
                SecurityAssumption::CapacityBound => "CapacityBound",
            }
        )
    }
}

impl FromStr for SecurityAssumption {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s == "UniqueDecoding" {
            Ok(SecurityAssumption::UniqueDecoding)
        } else if s == "JohnsonBound" {
            Ok(SecurityAssumption::JohnsonBound)
        } else if s == "CapacityBound" {
            Ok(SecurityAssumption::CapacityBound)
        } else {
            Err(format!("Invalid soundness specification: {}", s))
        }
    }
}
