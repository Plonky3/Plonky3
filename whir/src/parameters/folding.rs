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
}

/// Defines the folding factor for polynomial commitments.
#[derive(Debug, Clone)]
pub enum FoldingFactor {
    /// A fixed folding factor used in all rounds.
    Constant(usize),
    /// Uses a different folding factor for the first round and a fixed one for the rest.
    ConstantFromSecondRound(usize, usize),
    /// Explicit folding factors for each folding phase, including the initial
    /// fold and the final sumcheck fold.
    PerRound(Vec<usize>),
}

impl FoldingFactor {
    /// Retrieves the folding factor for a given round.
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
        match self {
            Self::Constant(factor) => {
                if *factor > num_variables {
                    // A folding factor cannot be greater than the number of available variables.
                    Err(FoldingFactorError::TooLarge(*factor, num_variables))
                } else if *factor == 0 {
                    // A folding factor of zero is invalid since folding must reduce variables.
                    Err(FoldingFactorError::ZeroFactor)
                } else {
                    Ok(())
                }
            }
            Self::ConstantFromSecondRound(first_round_factor, factor) => {
                if *first_round_factor > num_variables {
                    // The first round folding factor must not exceed the available variables.
                    Err(FoldingFactorError::TooLarge(
                        *first_round_factor,
                        num_variables,
                    ))
                } else if *factor > num_variables {
                    // Subsequent round folding factors must also not exceed the available
                    // variables.
                    Err(FoldingFactorError::TooLarge(*factor, num_variables))
                } else if *factor == 0 || *first_round_factor == 0 {
                    // Folding should occur at least once; zero is not valid.
                    Err(FoldingFactorError::ZeroFactor)
                } else {
                    Ok(())
                }
            }
            Self::PerRound(factors) => {
                for &factor in factors {
                    if factor > num_variables {
                        return Err(FoldingFactorError::TooLarge(factor, num_variables));
                    }
                    if factor == 0 {
                        return Err(FoldingFactorError::ZeroFactor);
                    }
                }
                Ok(())
            }
        }
    }

    /// Compute the round schedule: number of folding rounds and final-phase variables.
    ///
    /// # Errors
    ///
    /// An explicit per-round schedule errors if a round folds more variables than remain.
    /// It also errors if the factors together fail to reach the direct-send threshold.
    pub fn compute_number_of_rounds(
        &self,
        num_variables: usize,
    ) -> Result<(usize, usize), FoldingFactorError> {
        match self {
            Self::Constant(factor) => {
                if num_variables <= MAX_NUM_VARIABLES_TO_SEND_COEFFS {
                    // the first folding is mandatory in the current implem (TODO don't fold, send directly the polynomial)
                    return Ok((0, num_variables - factor));
                }
                // Starting from `num_variables`, each round reduces the number of variables by `factor`. As soon as the
                // number of variables is less of equal than `MAX_NUM_VARIABLES_TO_SEND_COEFFS`, we stop folding and the
                // prover sends directly the coefficients of the polynomial.
                let num_rounds =
                    (num_variables - MAX_NUM_VARIABLES_TO_SEND_COEFFS).div_ceil(*factor);
                let final_sumcheck_rounds = num_variables - num_rounds * factor;
                // The -1 accounts for the fact that the last round does not require another folding.
                Ok((num_rounds - 1, final_sumcheck_rounds))
            }
            Self::ConstantFromSecondRound(first_round_factor, factor) => {
                // Compute the number of variables remaining after the first round.
                let nv_except_first_round = num_variables - *first_round_factor;
                if nv_except_first_round < MAX_NUM_VARIABLES_TO_SEND_COEFFS {
                    // This case is equivalent to Constant(first_round_factor)
                    // the first folding is mandatory in the current implem (TODO don't fold, send directly the polynomial)
                    return Ok((0, nv_except_first_round));
                }
                // Starting from `num_variables`, the first round reduces the number of variables by `first_round_factor`,
                // and the next ones by `factor`. As soon as the number of variables is less of equal than
                // `MAX_NUM_VARIABLES_TO_SEND_COEFFS`, we stop folding and the prover sends directly the coefficients of the polynomial.
                let num_rounds =
                    (nv_except_first_round - MAX_NUM_VARIABLES_TO_SEND_COEFFS).div_ceil(*factor);
                let final_sumcheck_rounds = nv_except_first_round - num_rounds * factor;
                // No need to minus 1 because the initial round is already excepted out
                Ok((num_rounds, final_sumcheck_rounds))
            }
            Self::PerRound(factors) => {
                // Fold one explicit factor per round until the remainder reaches the threshold.
                let mut remaining = num_variables;
                for (i, &factor) in factors.iter().enumerate() {
                    // A round cannot fold more variables than remain.
                    //
                    //     remaining = 7, factor = 9  ->  over-folds
                    if factor > remaining {
                        return Err(FoldingFactorError::TooLarge(factor, remaining));
                    }
                    remaining -= factor;
                    // Threshold reached: `i` full rounds, `remaining` sent direct.
                    if remaining <= MAX_NUM_VARIABLES_TO_SEND_COEFFS {
                        return Ok((i, remaining));
                    }
                }
                // Factors exhausted but the polynomial is still above the threshold.
                //
                //     num_variables = 20, sum(factors) = 5  ->  remaining 15 > 6
                Err(FoldingFactorError::InsufficientFolding {
                    num_variables,
                    remaining,
                    threshold: MAX_NUM_VARIABLES_TO_SEND_COEFFS,
                })
            }
        }
    }

    /// Computes the total number of folding rounds over `n_rounds` iterations.
    #[must_use]
    pub fn total_number(&self, n_rounds: usize) -> usize {
        match self {
            Self::Constant(factor) => {
                // - Each round folds `factor` variables,
                // - There are `n_rounds + 1` iterations (including the original input size).
                *factor * (n_rounds + 1)
            }
            Self::ConstantFromSecondRound(first_round_factor, factor) => {
                // - The first round folds `first_round_factor` variables,
                // - Subsequent rounds fold `factor` variables each.
                *first_round_factor + *factor * n_rounds
            }
            Self::PerRound(factors) => factors.iter().take(n_rounds + 1).sum(),
        }
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
        // Second round factor too large
        assert_eq!(
            FoldingFactor::ConstantFromSecondRound(2, 5).check_validity(4),
            Err(FoldingFactorError::TooLarge(5, 4))
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

    #[test]
    fn test_total_number() {
        let factor = FoldingFactor::Constant(2);
        assert_eq!(factor.total_number(3), 8); // 2 * (3 + 1)

        let variable_factor = FoldingFactor::ConstantFromSecondRound(3, 2);
        assert_eq!(variable_factor.total_number(3), 9); // 3 + 2 * 3

        let per_round = FoldingFactor::PerRound(vec![3, 2, 1]);
        assert_eq!(per_round.total_number(1), 5);
    }
}
