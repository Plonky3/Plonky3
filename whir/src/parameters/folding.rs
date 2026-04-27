//! Folding factor strategies for variable reduction across WHIR rounds.

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
}

/// Defines the folding factor for polynomial commitments.
#[derive(Debug, Clone, Copy)]
pub enum FoldingFactor {
    /// A fixed folding factor used in all rounds.
    Constant(usize),
    /// Uses a different folding factor for the first round and a fixed one for the rest.
    ConstantFromSecondRound(usize, usize),
}

impl FoldingFactor {
    /// Retrieves the folding factor for a given round.
    #[must_use]
    pub const fn at_round(&self, round: usize) -> usize {
        match self {
            Self::Constant(factor) => *factor,
            Self::ConstantFromSecondRound(first_round_factor, factor) => {
                if round == 0 {
                    *first_round_factor
                } else {
                    *factor
                }
            }
        }
    }

    /// Checks the validity of the folding factor against the number of variables.
    pub const fn check_validity(&self, num_variables: usize) -> Result<(), FoldingFactorError> {
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
        }
    }

    /// Computes the number of WHIR rounds and the number of rounds in the final sumcheck.
    #[must_use]
    pub fn compute_number_of_rounds(&self, num_variables: usize) -> (usize, usize) {
        match self {
            Self::Constant(factor) => {
                if num_variables <= MAX_NUM_VARIABLES_TO_SEND_COEFFS {
                    // the first folding is mandatory in the current implem (TODO don't fold, send directly the polynomial)
                    return (0, num_variables - factor);
                }
                // Starting from `num_variables`, each round reduces the number of variables by `factor`. As soon as the
                // number of variables is less of equal than `MAX_NUM_VARIABLES_TO_SEND_COEFFS`, we stop folding and the
                // prover sends directly the coefficients of the polynomial.
                let num_rounds =
                    (num_variables - MAX_NUM_VARIABLES_TO_SEND_COEFFS).div_ceil(*factor);
                let final_sumcheck_rounds = num_variables - num_rounds * factor;
                // The -1 accounts for the fact that the last round does not require another folding.
                (num_rounds - 1, final_sumcheck_rounds)
            }
            Self::ConstantFromSecondRound(first_round_factor, factor) => {
                // Compute the number of variables remaining after the first round.
                let nv_except_first_round = num_variables - *first_round_factor;
                if nv_except_first_round < MAX_NUM_VARIABLES_TO_SEND_COEFFS {
                    // This case is equivalent to Constant(first_round_factor)
                    // the first folding is mandatory in the current implem (TODO don't fold, send directly the polynomial)
                    return (0, nv_except_first_round);
                }
                // Starting from `num_variables`, the first round reduces the number of variables by `first_round_factor`,
                // and the next ones by `factor`. As soon as the number of variables is less of equal than
                // `MAX_NUM_VARIABLES_TO_SEND_COEFFS`, we stop folding and the prover sends directly the coefficients of the polynomial.
                let num_rounds =
                    (nv_except_first_round - MAX_NUM_VARIABLES_TO_SEND_COEFFS).div_ceil(*factor);
                let final_sumcheck_rounds = nv_except_first_round - num_rounds * factor;
                // No need to minus 1 because the initial round is already excepted out
                (num_rounds, final_sumcheck_rounds)
            }
        }
    }

    /// Computes the total number of folding rounds over `n_rounds` iterations.
    #[must_use]
    pub const fn total_number(&self, n_rounds: usize) -> usize {
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
        }
    }
}

#[cfg(test)]
mod tests {
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
    }

    #[test]
    fn test_compute_number_of_rounds() {
        let constant_factor = 3;
        let factor = FoldingFactor::Constant(constant_factor);
        assert_eq!(
            factor.compute_number_of_rounds(MAX_NUM_VARIABLES_TO_SEND_COEFFS - 1),
            (0, MAX_NUM_VARIABLES_TO_SEND_COEFFS - constant_factor - 1)
        );
        assert_eq!(
            factor.compute_number_of_rounds(MAX_NUM_VARIABLES_TO_SEND_COEFFS),
            (0, MAX_NUM_VARIABLES_TO_SEND_COEFFS - constant_factor)
        );
        assert_eq!(
            factor.compute_number_of_rounds(MAX_NUM_VARIABLES_TO_SEND_COEFFS + 1),
            (0, MAX_NUM_VARIABLES_TO_SEND_COEFFS - constant_factor + 1)
        );
        assert_eq!(
            factor.compute_number_of_rounds(MAX_NUM_VARIABLES_TO_SEND_COEFFS + constant_factor),
            (0, MAX_NUM_VARIABLES_TO_SEND_COEFFS)
        );
        assert_eq!(
            factor.compute_number_of_rounds(MAX_NUM_VARIABLES_TO_SEND_COEFFS + constant_factor + 1),
            (1, MAX_NUM_VARIABLES_TO_SEND_COEFFS - constant_factor + 1)
        );
        assert_eq!(
            factor.compute_number_of_rounds(
                MAX_NUM_VARIABLES_TO_SEND_COEFFS + constant_factor * 2 + 1
            ),
            (2, MAX_NUM_VARIABLES_TO_SEND_COEFFS - constant_factor + 1)
        );

        let initial_factor = 4;
        let next_factor = 3;
        let variable_factor = FoldingFactor::ConstantFromSecondRound(initial_factor, next_factor);
        assert_eq!(
            variable_factor.compute_number_of_rounds(MAX_NUM_VARIABLES_TO_SEND_COEFFS - 1),
            (0, MAX_NUM_VARIABLES_TO_SEND_COEFFS - initial_factor - 1)
        );
        assert_eq!(
            variable_factor.compute_number_of_rounds(MAX_NUM_VARIABLES_TO_SEND_COEFFS),
            (0, MAX_NUM_VARIABLES_TO_SEND_COEFFS - initial_factor)
        );
        assert_eq!(
            variable_factor.compute_number_of_rounds(MAX_NUM_VARIABLES_TO_SEND_COEFFS + 1),
            (0, MAX_NUM_VARIABLES_TO_SEND_COEFFS - initial_factor + 1)
        );
        assert_eq!(
            variable_factor
                .compute_number_of_rounds(MAX_NUM_VARIABLES_TO_SEND_COEFFS + initial_factor),
            (0, MAX_NUM_VARIABLES_TO_SEND_COEFFS)
        );
        assert_eq!(
            variable_factor
                .compute_number_of_rounds(MAX_NUM_VARIABLES_TO_SEND_COEFFS + initial_factor + 1),
            (1, MAX_NUM_VARIABLES_TO_SEND_COEFFS - next_factor + 1)
        );
        assert_eq!(
            variable_factor.compute_number_of_rounds(
                MAX_NUM_VARIABLES_TO_SEND_COEFFS + initial_factor + next_factor + 1
            ),
            (2, MAX_NUM_VARIABLES_TO_SEND_COEFFS - next_factor + 1)
        );
    }

    #[test]
    fn test_total_number() {
        let factor = FoldingFactor::Constant(2);
        assert_eq!(factor.total_number(3), 8); // 2 * (3 + 1)

        let variable_factor = FoldingFactor::ConstantFromSecondRound(3, 2);
        assert_eq!(variable_factor.total_number(3), 9); // 3 + 2 * 3
    }
}
