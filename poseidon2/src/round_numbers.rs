//! As the security analysis of Poseidon2 is identical to that of Poseidon,
//! the relevant constraints regarding the number of full/partial rounds required can be found in
//! the original paper: `<https://eprint.iacr.org/2019/458.pdf>` and the associated codebase:
//! `<https://extgit.iaik.tugraz.at/krypto/hadeshash>` (See generate_params_poseidon.sage)
//!
//! These constraints are broken down into 6 equations:
//! statistical, interpolation, Gröbner 1, 2, 3 and
//! an extra constraint coming from the paper `<https://eprint.iacr.org/2023/537.pdf>`.
//!
//! For our parameters (M = 128, p > 2^30, WIDTH = t >= 8, D = alpha < 12),
//! the statistical constraint always simplifies to requiring RF >= 6.
//! Additionally p does not appear in Gröbner 3 or the constraint coming from `<https://eprint.iacr.org/2023/537.pdf>`.
//! The remaining 3 constraints all can be rearranged into the form:
//! F(RF, RP) >= G(p) where G is a function which is non-decreasing with respect to p.
//!
//! Thus, if some tuple (M, p, WIDTH, D, RF, RP) satisfies all constraints, then so will
//! the tuple (M, q, WIDTH, D, RF, RP) for any 2^30 < q < p.
//! Moreover if RF, RP are the "optimal" round numbers (Optimal meaning minimising the number of S-box operations we need to perform)
//! for two tuples (M, p, WIDTH, D) and (M, q, WIDTH, D), then
//! they will also be optimal for (M, r, WIDTH, D) for any q < r < p.
//!
//! We compute the optimal required number of external (full) and internal (partial) rounds using:
//! `<https://github.com/0xPolygonZero/hash-constants/blob/master/calc_round_numbers.py>`
//! Using the above analysis we can conclude that the round numbers are equal
//! for all 31 bit primes and 64 bit primes respectively.

use p3_field::PrimeField64;
use p3_util::relatively_prime_u64;
use thiserror::Error;

/// Reasons the precomputed 128-bit Poseidon2 round table cannot serve a parameter set.
#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
pub enum Poseidon2RoundNumbersError {
    /// The S-box exponent does not define a permutation of the field.
    #[error(
        "Poseidon2 S-box exponent {exponent} is not coprime to the field order minus one ({field_order_minus_one})"
    )]
    InvalidSboxExponent {
        /// Requested S-box exponent.
        exponent: u64,
        /// Multiplicative-group order the exponent must be coprime to.
        field_order_minus_one: u64,
    },
    /// No precomputed round count exists for this width and exponent.
    #[error(
        "no precomputed Poseidon2 round count for a {field_bits}-bit field with width {width} and S-box exponent {exponent}"
    )]
    UnsupportedWidthAndExponent {
        /// Bit length of the field order.
        field_bits: u32,
        /// Requested permutation width.
        width: usize,
        /// Requested S-box exponent.
        exponent: u64,
    },
    /// The table has not been computed for this field size.
    #[error("no precomputed Poseidon2 round counts for a {field_bits}-bit field")]
    UnsupportedFieldSize {
        /// Bit length of the field order.
        field_bits: u32,
    },
}

/// Total number of full rounds for 128-bit security.
///
/// The Poseidon paper's statistical attack analysis (Section 5.4) requires `RF ≥ 6`.
/// Adding the standard +2 security margin gives `RF = 8`.
/// This value is the same for all field sizes and widths at the 128-bit security level.
const FULL_ROUNDS_128: usize = 8;

/// Return the full and partial round counts needed for 128-bit security.
///
/// # Errors
///
/// Returns an error when `d` does not define a permutation over `F`.
///
/// Returns an error when the precomputed table has no matching parameter set.
pub const fn poseidon2_round_numbers_128<F: PrimeField64>(
    width: usize,
    d: u64,
) -> Result<(usize, usize), Poseidon2RoundNumbersError> {
    // Start by checking that d is a valid permutation.
    let field_order_minus_one = F::ORDER_U64 - 1;
    if !relatively_prime_u64(d, field_order_minus_one) {
        return Err(Poseidon2RoundNumbersError::InvalidSboxExponent {
            exponent: d,
            field_order_minus_one,
        });
    }

    // Next compute the number of bits in p.
    let prime_bit_number = F::ORDER_U64.ilog2() + 1;

    match prime_bit_number {
        31 => match (width, d) {
            (16, 3) => Ok((FULL_ROUNDS_128, 20)),
            (16, 5) => Ok((FULL_ROUNDS_128, 14)),
            (16, 7) => Ok((FULL_ROUNDS_128, 13)),
            (16, 9) => Ok((FULL_ROUNDS_128, 13)),
            (16, 11) => Ok((FULL_ROUNDS_128, 13)),
            (24, 3) => Ok((FULL_ROUNDS_128, 23)),
            (24, 5) => Ok((FULL_ROUNDS_128, 22)),
            (24, 7) => Ok((FULL_ROUNDS_128, 21)),
            (24, 9) => Ok((FULL_ROUNDS_128, 21)),
            (24, 11) => Ok((FULL_ROUNDS_128, 21)),
            (32, 3) => Ok((FULL_ROUNDS_128, 31)),
            (32, 5) => Ok((FULL_ROUNDS_128, 30)),
            (32, 7) => Ok((FULL_ROUNDS_128, 30)),
            (32, 9) => Ok((FULL_ROUNDS_128, 30)),
            (32, 11) => Ok((FULL_ROUNDS_128, 30)),
            _ => Err(Poseidon2RoundNumbersError::UnsupportedWidthAndExponent {
                field_bits: prime_bit_number,
                width,
                exponent: d,
            }),
        },
        64 => match (width, d) {
            (8, 3) => Ok((FULL_ROUNDS_128, 41)),
            (8, 5) => Ok((FULL_ROUNDS_128, 27)),
            (8, 7) => Ok((FULL_ROUNDS_128, 22)),
            (8, 9) => Ok((FULL_ROUNDS_128, 19)),
            (8, 11) => Ok((FULL_ROUNDS_128, 17)),
            (12, 3) => Ok((FULL_ROUNDS_128, 42)),
            (12, 5) => Ok((FULL_ROUNDS_128, 27)),
            (12, 7) => Ok((FULL_ROUNDS_128, 22)),
            (12, 9) => Ok((FULL_ROUNDS_128, 20)),
            (12, 11) => Ok((FULL_ROUNDS_128, 18)),
            (16, 3) => Ok((FULL_ROUNDS_128, 42)),
            (16, 5) => Ok((FULL_ROUNDS_128, 27)),
            (16, 7) => Ok((FULL_ROUNDS_128, 22)),
            (16, 9) => Ok((FULL_ROUNDS_128, 20)),
            (16, 11) => Ok((FULL_ROUNDS_128, 18)),
            _ => Err(Poseidon2RoundNumbersError::UnsupportedWidthAndExponent {
                field_bits: prime_bit_number,
                width,
                exponent: d,
            }),
        },
        _ => Err(Poseidon2RoundNumbersError::UnsupportedFieldSize {
            field_bits: prime_bit_number,
        }),
    }
}

#[cfg(test)]
mod tests {
    use alloc::format;
    use alloc::string::ToString;

    use p3_baby_bear::BabyBear;
    use p3_field::PrimeField64;

    use super::{Poseidon2RoundNumbersError, poseidon2_round_numbers_128};

    #[test]
    fn invalid_sbox_exponent_reports_the_exponent_and_group_order() {
        let error = poseidon2_round_numbers_128::<BabyBear>(16, 2).unwrap_err();
        assert_eq!(
            error,
            Poseidon2RoundNumbersError::InvalidSboxExponent {
                exponent: 2,
                field_order_minus_one: BabyBear::ORDER_U64 - 1,
            }
        );
        assert_eq!(
            error.to_string(),
            format!(
                "Poseidon2 S-box exponent 2 is not coprime to the field order minus one ({})",
                BabyBear::ORDER_U64 - 1
            )
        );
    }

    #[test]
    fn unsupported_parameters_report_every_table_key() {
        let error = poseidon2_round_numbers_128::<BabyBear>(15, 7).unwrap_err();
        assert_eq!(
            error,
            Poseidon2RoundNumbersError::UnsupportedWidthAndExponent {
                field_bits: 31,
                width: 15,
                exponent: 7,
            }
        );
        assert_eq!(
            error.to_string(),
            "no precomputed Poseidon2 round count for a 31-bit field with width 15 and S-box exponent 7"
        );
    }

    #[test]
    fn unsupported_field_size_message_reports_the_table_key() {
        let error = Poseidon2RoundNumbersError::UnsupportedFieldSize { field_bits: 48 };
        assert_eq!(
            error.to_string(),
            "no precomputed Poseidon2 round counts for a 48-bit field"
        );
    }
}
