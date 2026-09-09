//! The PCS challenge block: alpha batching followed by every bucket's `Combine`.
//!
//! Claims and input commitments are fixed before the batch grind. No prover message
//! intervenes between alpha and the Combine challenges, so their errors add and the
//! grind is credited once to that sum. Later STIR challenges have their own budgets.
//!
//! For a batch failing the PCS's documented correlated-agreement/extraction
//! relation, fix a false native-height class before this block. If
//! alpha does not rescue it, Combine must rescue its bucket or that bucket remains
//! false for STIR. Acceptance requires every bucket to pass, so there is no union
//! penalty over buckets. Within a bucket we conservatively sum *all* alpha errors
//! and its Combine error rather than selecting the worst false class.

use p3_field::Field;

use crate::SecurityAssumption;
use crate::config::StirConfigError;
use crate::soundness::{StirSoundness, minimum_eta_for_target};

/// Opening multiplicities are the number of consecutive alpha powers per native
/// height: sum of matrix width times opening-point count, pooled across commitments.
/// An empty list is a commit-time, Combine-only feasibility probe. It must never be
/// used to derive an opening schedule.
#[derive(Clone, Copy)]
pub(crate) struct PcsBatch<'a> {
    pub classes: &'a [(usize, usize)],
    pub combine: Option<(usize, u64)>,
    pub pow_bits: usize,
}

impl PcsBatch<'_> {
    pub fn has_error(&self) -> bool {
        self.combine.is_some() || self.classes.iter().any(|&(_, count)| count > 1)
    }

    pub fn validate(&self, log_degree: usize) -> Result<(), StirConfigError> {
        if self
            .classes
            .iter()
            .any(|&(height, count)| height > log_degree || count == 0)
            || self.classes.windows(2).any(|pair| pair[0].0 <= pair[1].0)
            || self
                .classes
                .first()
                .is_some_and(|&(height, _)| height != log_degree)
        {
            return Err(StirConfigError::InvalidPcsBatch);
        }
        if let Some((num_classes, ell)) = self.combine {
            if num_classes < 2 {
                return Err(StirConfigError::CombineNeedsMultipleClasses { num_classes });
            }
            if ell < num_classes as u64 {
                return Err(StirConfigError::CombineMultiplicityTooSmall { num_classes, ell });
            }
        }
        if !self.classes.is_empty()
            && self.combine != combine_requirement(log_degree, self.classes)?
        {
            return Err(StirConfigError::InvalidPcsBatch);
        }
        Ok(())
    }

    /// Algebraic bits before grinding, at the *same* radius as STIR round zero.
    pub fn algebraic_bits(
        &self,
        assumption: SecurityAssumption,
        field_bits: usize,
        log_degree: usize,
        log_inv_rate: usize,
        eta: f64,
    ) -> f64 {
        if eta < minimum_eta(assumption, log_inv_rate) {
            // Fail closed before the shared Johnson helper's float-to-usize
            // conversion can saturate its multiplicity on very large fields.
            return 0.;
        }
        let log_domain = log_degree + log_inv_rate;
        let rho = libm::pow(2., -(log_inv_rate as f64));
        let agreement = match assumption {
            SecurityAssumption::CapacityBound => rho + eta,
            SecurityAssumption::JohnsonBound => libm::sqrt(rho) + eta,
            SecurityAssumption::UniqueDecoding => unreachable!("STIR rejects unique decoding"),
        };
        let mut bits = f64::INFINITY;
        for &(height, count) in self.classes {
            if count <= 1 {
                continue;
            }
            let class_log_inv_rate = log_domain - height;
            let class_rho = libm::pow(2., -(class_log_inv_rate as f64));
            let class_eta = match assumption {
                SecurityAssumption::CapacityBound => agreement - class_rho,
                SecurityAssumption::JohnsonBound => agreement - libm::sqrt(class_rho),
                SecurityAssumption::UniqueDecoding => unreachable!(),
            };
            // Avoid cancellation when eta is tiny and this is the tallest class.
            let class_eta = if height == log_degree { eta } else { class_eta };
            bits = add_error_bits(
                bits,
                assumption.prox_gaps_error_at_log_eta(
                    height,
                    class_log_inv_rate,
                    field_bits,
                    count,
                    libm::log2(class_eta),
                ),
            );
        }
        if let Some((_, ell)) = self.combine {
            // Derive with a u64 multiplicity even on 32-bit targets. The shared helper
            // is evaluated at multiplicity two, then scaled by the full ell - 1.
            let combine_bits = assumption.prox_gaps_error_at_log_eta(
                log_degree,
                log_inv_rate,
                field_bits,
                2,
                libm::log2(eta),
            ) - libm::log2((ell - 1) as f64);
            bits = add_error_bits(bits, combine_bits);
        }
        bits.max(0.)
    }

    pub fn eta(
        &self,
        assumption: SecurityAssumption,
        field_bits: usize,
        log_degree: usize,
        log_inv_rate: usize,
        target_bits: usize,
    ) -> Result<f64, StirConfigError> {
        if !self.has_error() {
            return Ok(0.);
        }
        let eta = minimum_eta_for_target(
            assumption.stir_eta_upper_bound(log_inv_rate),
            target_bits,
            |eta| {
                self.algebraic_bits(assumption, field_bits, log_degree, log_inv_rate, eta)
                    + self.pow_bits as f64
            },
            "PCS joint alpha/Combine bound",
        )?
        .max(minimum_eta(assumption, log_inv_rate));
        // Lemma 4.13 also requires delta < 1 - rho - 1/|L|.
        Ok(
            if self.combine.is_some() && assumption == SecurityAssumption::CapacityBound {
                eta.max(libm::pow(2., 1. - (log_degree + log_inv_rate) as f64))
            } else {
                eta
            },
        )
    }
}

pub(crate) fn combine_requirement(
    log_degree: usize,
    classes: &[(usize, usize)],
) -> Result<Option<(usize, u64)>, StirConfigError> {
    if classes.len() < 2 {
        return Ok(None);
    }
    let d = 1u64
        .checked_shl(u32::try_from(log_degree).map_err(|_| StirConfigError::InvalidPcsBatch)?)
        .ok_or(StirConfigError::InvalidPcsBatch)?;
    let mut ell = 0u64;
    for &(height, _) in classes {
        let native = 1u64
            .checked_shl(u32::try_from(height).map_err(|_| StirConfigError::InvalidPcsBatch)?)
            .ok_or(StirConfigError::InvalidPcsBatch)?;
        let contribution = d
            .checked_sub(native)
            .and_then(|gap| gap.checked_add(1))
            .ok_or(StirConfigError::InvalidPcsBatch)?;
        ell = ell
            .checked_add(contribution)
            .ok_or(StirConfigError::PcsBatchMultiplicityOverflow)?;
    }
    Ok(Some((classes.len(), ell)))
}

/// Keep BCSS25's integer multiplicity below usize::MAX in *every* PCS stage.
/// Use at most 2^(word_bits-1), which remains exactly representable in f64.
pub(crate) fn minimum_eta(assumption: SecurityAssumption, log_inv_rate: usize) -> f64 {
    if assumption == SecurityAssumption::JohnsonBound {
        libm::pow(2., -0.5 * log_inv_rate as f64 - usize::BITS as f64)
    } else {
        0.
    }
}

/// PCS schedules use a rigorous integer lower bound on log2(|E|), rather than
/// Field::bits()'s ceiling. Johnson reserves one further bit: twice BCSS25's
/// dominant term upper-bounds the full expression for m >= 3, rho <= 1, N >= 1.
/// Indeed the omitted terms divided by the dominant one sum to at most
/// 3 / (m + 1/2)^4 < 1 (using gamma <= 1). Applying this reserve to the field
/// size covers both the new block and the existing STIR proximity-gap calls.
pub(crate) fn field_bits<EF: Field>(assumption: SecurityAssumption) -> usize {
    let floor = EF::order().bits() as usize - 1;
    floor.saturating_sub(usize::from(assumption == SecurityAssumption::JohnsonBound))
}

/// Add probabilities represented as -log2(p), without underflow for large fields.
fn add_error_bits(a: f64, b: f64) -> f64 {
    if a == f64::INFINITY {
        return b;
    }
    if b == f64::INFINITY {
        return a;
    }
    let smaller = a.min(b);
    smaller - libm::log2(1. + libm::pow(2., smaller - a.max(b)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn joint_bound_adds_alpha_and_combine_at_the_shared_radius() {
        let batch = PcsBatch {
            classes: &[(8, 3), (7, 5)],
            combine: Some((2, 130)),
            pow_bits: 8,
        };
        // N=512, eta=.1: alpha coefficients 20480 and 163840/7,
        // Combine coefficient 1320960. The sum is 9553920/7, over |E|.
        let expected = 128. - libm::log2(9_553_920. / 7.);
        let bits = batch.algebraic_bits(SecurityAssumption::CapacityBound, 128, 8, 1, 0.1);
        assert!((bits - expected).abs() < 1e-12);
    }

    #[test]
    fn singleton_alpha_error_is_not_omitted() {
        let batch = PcsBatch {
            classes: &[(8, 17)],
            combine: None,
            pow_bits: 0,
        };
        // 16*256/(.5^2 * .125) = 2^17.
        assert_eq!(
            batch.algebraic_bits(SecurityAssumption::CapacityBound, 128, 8, 1, 0.125),
            111.
        );
        let one = PcsBatch {
            classes: &[(8, 1)],
            ..batch
        };
        assert!(!one.has_error());
        assert_eq!(
            one.algebraic_bits(SecurityAssumption::CapacityBound, 128, 8, 1, 0.125),
            f64::INFINITY
        );
    }

    #[test]
    fn grinding_is_applied_once_to_the_joint_probability() {
        let batch = PcsBatch {
            classes: &[(20, 64), (19, 64)],
            combine: Some((2, 524290)),
            pow_bits: 8,
        };
        for assumption in [
            SecurityAssumption::CapacityBound,
            SecurityAssumption::JohnsonBound,
        ] {
            let target = if assumption == SecurityAssumption::CapacityBound {
                106
            } else {
                100
            };
            let eta = batch.eta(assumption, 154, 20, 1, target).unwrap();
            let bits = batch.algebraic_bits(assumption, 154, 20, 1, eta);
            assert!(bits + 8. >= target as f64);
            // The returned value sits at the budget boundary, except for the CB floor.
            let below = batch.algebraic_bits(assumption, 154, 20, 1, eta * 0.99);
            assert!(below + 8. < target as f64);
            assert!(bits < target as f64, "this case needs the grind");
        }
    }

    #[test]
    fn impossible_opening_multiplicity_is_rejected_even_without_combine() {
        let batch = PcsBatch {
            classes: &[(20, 1 << 30)],
            combine: None,
            pow_bits: 0,
        };
        assert!(matches!(
            batch.eta(SecurityAssumption::CapacityBound, 123, 20, 1, 90),
            Err(StirConfigError::EtaInfeasibleForTarget {
                label: "PCS joint alpha/Combine bound",
                ..
            })
        ));
    }

    #[test]
    fn multiplicity_and_grinding_have_opposite_effects_on_eta() {
        for assumption in [
            SecurityAssumption::CapacityBound,
            SecurityAssumption::JohnsonBound,
        ] {
            let small = PcsBatch {
                classes: &[(20, 1 << 16), (19, 1 << 16)],
                combine: Some((2, 524290)),
                pow_bits: 8,
            };
            let wide = PcsBatch {
                classes: &[(20, 1 << 20), (19, 1 << 20)],
                ..small
            };
            let more_work = PcsBatch {
                pow_bits: 16,
                ..wide
            };
            let a = small.eta(assumption, 154, 20, 1, 100).unwrap();
            let b = wide.eta(assumption, 154, 20, 1, 100).unwrap();
            let c = more_work.eta(assumption, 154, 20, 1, 100).unwrap();
            assert!(b >= a);
            assert!(c <= b);
        }
    }

    #[test]
    fn johnson_large_fields_do_not_saturate_the_proximity_multiplicity() {
        let batch = PcsBatch {
            classes: &[(20, 64)],
            combine: None,
            pow_bits: 8,
        };
        let eta = batch
            .eta(SecurityAssumption::JohnsonBound, 510, 20, 1, 100)
            .unwrap();
        let m = libm::ceil(libm::sqrt(0.5) / (2. * eta));
        assert!(
            m < usize::MAX as f64,
            "m={m} would silently saturate in the shared helper"
        );
    }

    #[test]
    fn a_full_budget_cannot_omit_or_understate_its_combine() {
        let classes = &[(8, 2), (7, 2)];
        for combine in [None, Some((2, 2)), Some((3, 130))] {
            assert!(
                PcsBatch {
                    classes,
                    combine,
                    pow_bits: 0
                }
                .validate(8)
                .is_err()
            );
        }
        assert!(
            PcsBatch {
                classes,
                combine: Some((2, 130)),
                pow_bits: 0
            }
            .validate(8)
            .is_ok()
        );
    }
}
