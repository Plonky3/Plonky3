//! Folding factor strategies for variable reduction across WHIR rounds.

use alloc::vec::Vec;

use thiserror::Error;

/// Each WHIR step folds the polynomial, reducing the number of variables.
/// Once the number of variables drops to this threshold or below,
/// the prover sends the polynomial coefficients directly.
const MAX_NUM_VARIABLES_TO_SEND_COEFFS: usize = 6;

/// Errors that can occur when validating a folding factor.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum FoldingFactorError {
    /// The folding factor is larger than the number of variables.
    #[error(
        "Folding factor {0} is greater than the number of variables {1}. Polynomial too small, just send it directly."
    )]
    TooLarge(usize, usize),

    /// The folding factor cannot be zero.
    #[error("Folding factor shouldn't be zero.")]
    ZeroFactor,

    /// The explicit per-round folding factors fold too few variables to reach the direct-send threshold.
    #[error(
        "per-round folding factors leave {remaining} variables out of {num_variables}, above the direct-send threshold {threshold}; fold more variables"
    )]
    InsufficientFolding {
        num_variables: usize,
        remaining: usize,
        threshold: usize,
    },

    /// The explicit per-round schedule contains no folding factors.
    #[error("per-round folding schedule is empty; WHIR requires at least one fold")]
    EmptySchedule,
}

/// Defines the folding factor for polynomial commitments.
#[derive(Debug, Clone)]
pub enum FoldingFactor {
    /// A fixed folding factor used in all rounds.
    Constant(usize),
    /// Uses a different folding factor for the first round and a fixed one for the rest.
    ConstantFromSecondRound(usize, usize),
    /// Explicit folding factors for each pre-direct-send folding phase,
    /// including the initial fold.
    PerRound(Vec<usize>),
}

impl FoldingFactor {
    /// Retrieves the nominal configured folding factor for a given round.
    ///
    /// This does not apply the partial-final-fold clamp used by constant
    /// schedules. Use [`Self::compute_folding_schedule`] when the concrete
    /// derived factors matter.
    #[must_use]
    pub fn at_round(&self, round: usize) -> usize {
        match self {
            Self::Constant(factor) => *factor,
            Self::ConstantFromSecondRound(first_round_factor, factor) => {
                if round == 0 {
                    *first_round_factor
                } else {
                    *factor
                }
            }
            Self::PerRound(factors) => factors[round],
        }
    }

    /// Checks the validity of the folding factor against the number of variables.
    pub fn check_validity(&self, num_variables: usize) -> Result<(), FoldingFactorError> {
        self.compute_folding_schedule(num_variables).map(|_| ())
    }

    /// Derive the concrete folding factors used before the final direct-send phase.
    ///
    /// Constant schedules may use a smaller final fold when fewer than `factor`
    /// variables remain. Explicit per-round schedules stay exact and therefore
    /// still reject any factor larger than the current remainder.
    pub fn compute_folding_schedule(
        &self,
        num_variables: usize,
    ) -> Result<Vec<usize>, FoldingFactorError> {
        match self {
            Self::Constant(factor) => {
                if *factor == 0 {
                    return Err(FoldingFactorError::ZeroFactor);
                }
                if *factor > num_variables {
                    return Err(FoldingFactorError::TooLarge(*factor, num_variables));
                }
                let mut remaining = num_variables;
                let mut schedule = Vec::new();
                loop {
                    let round_factor = (*factor).min(remaining);
                    schedule.push(round_factor);
                    remaining -= round_factor;
                    if remaining <= MAX_NUM_VARIABLES_TO_SEND_COEFFS {
                        return Ok(schedule);
                    }
                }
            }
            Self::ConstantFromSecondRound(first_round_factor, factor) => {
                if *first_round_factor == 0 || *factor == 0 {
                    return Err(FoldingFactorError::ZeroFactor);
                }
                if *first_round_factor > num_variables {
                    return Err(FoldingFactorError::TooLarge(
                        *first_round_factor,
                        num_variables,
                    ));
                }

                let mut remaining = num_variables;
                let mut schedule = Vec::new();

                schedule.push(*first_round_factor);
                remaining -= *first_round_factor;
                while remaining > MAX_NUM_VARIABLES_TO_SEND_COEFFS {
                    let round_factor = (*factor).min(remaining);
                    schedule.push(round_factor);
                    remaining -= round_factor;
                }

                Ok(schedule)
            }
            Self::PerRound(factors) => {
                // WHIR always performs at least one fold before direct send,
                // so an empty explicit schedule can never run.
                if factors.is_empty() {
                    return Err(FoldingFactorError::EmptySchedule);
                }
                for &factor in factors {
                    if factor == 0 {
                        return Err(FoldingFactorError::ZeroFactor);
                    }
                    if factor > num_variables {
                        return Err(FoldingFactorError::TooLarge(factor, num_variables));
                    }
                }

                let mut remaining = num_variables;
                let mut schedule = Vec::new();
                for &factor in factors {
                    if factor > remaining {
                        return Err(FoldingFactorError::TooLarge(factor, remaining));
                    }
                    schedule.push(factor);
                    remaining -= factor;
                    if remaining <= MAX_NUM_VARIABLES_TO_SEND_COEFFS {
                        return Ok(schedule);
                    }
                }
                Err(FoldingFactorError::InsufficientFolding {
                    num_variables,
                    remaining,
                    threshold: MAX_NUM_VARIABLES_TO_SEND_COEFFS,
                })
            }
        }
    }

    /// Compute the round schedule: number of folding rounds and final-phase variables.
    ///
    /// # Errors
    ///
    /// Propagates every error of [`Self::compute_folding_schedule`]:
    ///
    /// - a zero folding factor,
    /// - a first fold larger than the variable count,
    /// - an empty explicit schedule,
    /// - an explicit round folding more variables than remain,
    /// - an explicit schedule stopping above the direct-send threshold.
    pub fn compute_number_of_rounds(
        &self,
        num_variables: usize,
    ) -> Result<(usize, usize), FoldingFactorError> {
        let schedule = self.compute_folding_schedule(num_variables)?;
        let folded: usize = schedule.iter().sum();
        Ok((schedule.len() - 1, num_variables - folded))
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;

    #[test]
    fn test_folding_factor_at_round() {
        let factor = FoldingFactor::Constant(4);
        assert_eq!(factor.at_round(0), 4);
        assert_eq!(factor.at_round(5), 4);

        let variable_factor = FoldingFactor::ConstantFromSecondRound(3, 5);
        assert_eq!(variable_factor.at_round(0), 3); // First round uses 3
        assert_eq!(variable_factor.at_round(1), 5); // Subsequent rounds use 5
        assert_eq!(variable_factor.at_round(10), 5);

        let per_round = FoldingFactor::PerRound(vec![3, 2, 1]);
        assert_eq!(per_round.at_round(0), 3);
        assert_eq!(per_round.at_round(1), 2);
        assert_eq!(per_round.at_round(2), 1);
    }

    #[test]
    fn test_folding_factor_check_validity() {
        // Valid cases
        assert!(FoldingFactor::Constant(2).check_validity(4).is_ok());
        assert!(
            FoldingFactor::ConstantFromSecondRound(2, 3)
                .check_validity(5)
                .is_ok()
        );

        // Invalid cases
        // Factor too large
        assert_eq!(
            FoldingFactor::Constant(5).check_validity(3),
            Err(FoldingFactorError::TooLarge(5, 3))
        );
        // Zero factor
        assert_eq!(
            FoldingFactor::Constant(0).check_validity(3),
            Err(FoldingFactorError::ZeroFactor)
        );
        // First round factor too large
        assert_eq!(
            FoldingFactor::ConstantFromSecondRound(4, 2).check_validity(3),
            Err(FoldingFactorError::TooLarge(4, 3))
        );
        // A later constant factor may be larger than the later remainder; the
        // derived schedule clamps only that final fold.
        assert_eq!(
            FoldingFactor::ConstantFromSecondRound(2, 5).compute_folding_schedule(4),
            Ok(vec![2])
        );
        // First round zero
        assert_eq!(
            FoldingFactor::ConstantFromSecondRound(0, 3).check_validity(4),
            Err(FoldingFactorError::ZeroFactor)
        );
        assert_eq!(
            FoldingFactor::PerRound(vec![2, 0]).check_validity(4),
            Err(FoldingFactorError::ZeroFactor)
        );
    }

    #[test]
    fn test_compute_number_of_rounds() {
        // Every valid schedule now yields `Ok((rounds, final_sumcheck_rounds))`.
        let constant_factor = 3;
        let factor = FoldingFactor::Constant(constant_factor);
        assert_eq!(
            factor.compute_number_of_rounds(MAX_NUM_VARIABLES_TO_SEND_COEFFS - 1),
            Ok((0, MAX_NUM_VARIABLES_TO_SEND_COEFFS - constant_factor - 1))
        );
        assert_eq!(
            factor.compute_number_of_rounds(MAX_NUM_VARIABLES_TO_SEND_COEFFS),
            Ok((0, MAX_NUM_VARIABLES_TO_SEND_COEFFS - constant_factor))
        );
        assert_eq!(
            factor.compute_number_of_rounds(MAX_NUM_VARIABLES_TO_SEND_COEFFS + 1),
            Ok((0, MAX_NUM_VARIABLES_TO_SEND_COEFFS - constant_factor + 1))
        );
        assert_eq!(
            factor.compute_number_of_rounds(MAX_NUM_VARIABLES_TO_SEND_COEFFS + constant_factor),
            Ok((0, MAX_NUM_VARIABLES_TO_SEND_COEFFS))
        );
        assert_eq!(
            factor.compute_number_of_rounds(MAX_NUM_VARIABLES_TO_SEND_COEFFS + constant_factor + 1),
            Ok((1, MAX_NUM_VARIABLES_TO_SEND_COEFFS - constant_factor + 1))
        );
        assert_eq!(
            factor.compute_number_of_rounds(
                MAX_NUM_VARIABLES_TO_SEND_COEFFS + constant_factor * 2 + 1
            ),
            Ok((2, MAX_NUM_VARIABLES_TO_SEND_COEFFS - constant_factor + 1))
        );

        let initial_factor = 4;
        let next_factor = 3;
        let variable_factor = FoldingFactor::ConstantFromSecondRound(initial_factor, next_factor);
        assert_eq!(
            variable_factor.compute_number_of_rounds(MAX_NUM_VARIABLES_TO_SEND_COEFFS - 1),
            Ok((0, MAX_NUM_VARIABLES_TO_SEND_COEFFS - initial_factor - 1))
        );
        assert_eq!(
            variable_factor.compute_number_of_rounds(MAX_NUM_VARIABLES_TO_SEND_COEFFS),
            Ok((0, MAX_NUM_VARIABLES_TO_SEND_COEFFS - initial_factor))
        );
        assert_eq!(
            variable_factor.compute_number_of_rounds(MAX_NUM_VARIABLES_TO_SEND_COEFFS + 1),
            Ok((0, MAX_NUM_VARIABLES_TO_SEND_COEFFS - initial_factor + 1))
        );
        assert_eq!(
            variable_factor
                .compute_number_of_rounds(MAX_NUM_VARIABLES_TO_SEND_COEFFS + initial_factor),
            Ok((0, MAX_NUM_VARIABLES_TO_SEND_COEFFS))
        );
        assert_eq!(
            variable_factor
                .compute_number_of_rounds(MAX_NUM_VARIABLES_TO_SEND_COEFFS + initial_factor + 1),
            Ok((1, MAX_NUM_VARIABLES_TO_SEND_COEFFS - next_factor + 1))
        );
        assert_eq!(
            variable_factor.compute_number_of_rounds(
                MAX_NUM_VARIABLES_TO_SEND_COEFFS + initial_factor + next_factor + 1
            ),
            Ok((2, MAX_NUM_VARIABLES_TO_SEND_COEFFS - next_factor + 1))
        );

        // PerRound([3, 2]) on 10 variables: 10 -3-> 7 -2-> 5 <= 6 threshold.
        assert_eq!(
            FoldingFactor::PerRound(vec![3, 2]).compute_number_of_rounds(10),
            Ok((1, 5))
        );
    }

    #[test]
    fn constant_schedule_allows_partial_final_fold() {
        let schedule = FoldingFactor::Constant(8);

        assert_eq!(schedule.compute_folding_schedule(15), Ok(vec![8, 7]));
        assert_eq!(schedule.compute_number_of_rounds(15), Ok((1, 0)));
        assert_eq!(schedule.check_validity(15), Ok(()));
    }

    #[test]
    fn constant_from_second_round_allows_partial_final_fold() {
        let schedule = FoldingFactor::ConstantFromSecondRound(8, 8);

        assert_eq!(schedule.compute_folding_schedule(15), Ok(vec![8, 7]));
        assert_eq!(schedule.compute_number_of_rounds(15), Ok((1, 0)));
        assert_eq!(schedule.check_validity(15), Ok(()));
    }

    #[test]
    fn compute_number_of_rounds_keeps_degenerate_guards() {
        assert_eq!(
            FoldingFactor::Constant(0).compute_number_of_rounds(15),
            Err(FoldingFactorError::ZeroFactor)
        );
        assert_eq!(
            FoldingFactor::Constant(16).compute_number_of_rounds(15),
            Err(FoldingFactorError::TooLarge(16, 15))
        );
        assert_eq!(
            FoldingFactor::ConstantFromSecondRound(0, 8).compute_number_of_rounds(15),
            Err(FoldingFactorError::ZeroFactor)
        );
        assert_eq!(
            FoldingFactor::ConstantFromSecondRound(8, 0).compute_number_of_rounds(15),
            Err(FoldingFactorError::ZeroFactor)
        );
        assert_eq!(
            FoldingFactor::ConstantFromSecondRound(16, 8).compute_number_of_rounds(15),
            Err(FoldingFactorError::TooLarge(16, 15))
        );
        assert_eq!(
            FoldingFactor::PerRound(vec![2, 0]).compute_number_of_rounds(15),
            Err(FoldingFactorError::ZeroFactor)
        );
        // An empty explicit schedule is rejected.
        // The guard fires even below the direct-send threshold.
        assert_eq!(
            FoldingFactor::PerRound(vec![]).compute_number_of_rounds(4),
            Err(FoldingFactorError::EmptySchedule)
        );
        assert_eq!(
            FoldingFactor::PerRound(vec![]).check_validity(15),
            Err(FoldingFactorError::EmptySchedule)
        );
    }

    #[test]
    fn computed_rounds_match_old_formula_when_old_formula_did_not_overfold() {
        fn old_constant(num_variables: usize, factor: usize) -> Option<(usize, usize)> {
            if factor == 0 || factor > num_variables {
                return None;
            }
            if num_variables <= MAX_NUM_VARIABLES_TO_SEND_COEFFS {
                return Some((0, num_variables - factor));
            }
            let num_rounds = (num_variables - MAX_NUM_VARIABLES_TO_SEND_COEFFS).div_ceil(factor);
            let folded = num_rounds.checked_mul(factor)?;
            if folded <= num_variables {
                Some((num_rounds - 1, num_variables - folded))
            } else {
                None
            }
        }

        fn old_constant_from_second_round(
            num_variables: usize,
            first_round_factor: usize,
            factor: usize,
        ) -> Option<(usize, usize)> {
            if first_round_factor == 0
                || factor == 0
                || first_round_factor > num_variables
                || factor > num_variables
            {
                return None;
            }
            let remaining = num_variables - first_round_factor;
            if remaining < MAX_NUM_VARIABLES_TO_SEND_COEFFS {
                return Some((0, remaining));
            }
            let num_rounds = (remaining - MAX_NUM_VARIABLES_TO_SEND_COEFFS).div_ceil(factor);
            let folded = num_rounds.checked_mul(factor)?;
            if folded <= remaining {
                Some((num_rounds, remaining - folded))
            } else {
                None
            }
        }

        for num_variables in 1..40 {
            for factor in 1..12 {
                if let Some(expected) = old_constant(num_variables, factor) {
                    assert_eq!(
                        FoldingFactor::Constant(factor).compute_number_of_rounds(num_variables),
                        Ok(expected),
                        "Constant({factor}) @ {num_variables}"
                    );
                }

                for first_round_factor in 1..12 {
                    if let Some(expected) =
                        old_constant_from_second_round(num_variables, first_round_factor, factor)
                    {
                        assert_eq!(
                            FoldingFactor::ConstantFromSecondRound(first_round_factor, factor)
                                .compute_number_of_rounds(num_variables),
                            Ok(expected),
                            "ConstantFromSecondRound({first_round_factor}, {factor}) @ {num_variables}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn per_round_factors_that_under_fold_error() {
        // Invariant: a per-round schedule that under-folds is rejected with an error, not a panic.
        //
        // Fixture state:
        //   num_variables = 20, threshold = MAX_NUM_VARIABLES_TO_SEND_COEFFS = 6
        //   PerRound([3, 2]) folds 3 + 2 = 5 variables in total
        //
        //     remaining:  20 -3-> 17 -2-> 15
        //     15 > 6  ->  schedule exhausted while still too wide
        let schedule = FoldingFactor::PerRound(vec![3, 2]);

        // The leftover variable count and threshold are reported for diagnosis.
        assert_eq!(
            schedule.compute_number_of_rounds(20),
            Err(FoldingFactorError::InsufficientFolding {
                num_variables: 20,
                remaining: 15,
                threshold: MAX_NUM_VARIABLES_TO_SEND_COEFFS,
            })
        );
    }

    #[test]
    fn per_round_factor_larger_than_remaining_errors() {
        // Invariant: a round cannot fold more variables than remain, even when each factor is in range.
        //
        // Fixture state:
        //   num_variables = 10
        //   PerRound([3, 9]): both factors are <= 10 individually
        //
        //     remaining:  10 -3-> 7 ; round 1 wants 9 > 7  ->  over-folds
        //
        // Without the cumulative guard this path would underflow `remaining`.
        let schedule = FoldingFactor::PerRound(vec![3, 9]);

        assert_eq!(
            schedule.compute_number_of_rounds(10),
            Err(FoldingFactorError::TooLarge(9, 7))
        );
    }
}
