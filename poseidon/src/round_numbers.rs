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
//!
//! Note: In Poseidon, full rounds are split into two halves (applied at the beginning and end),
//! so we return `half_num_full_rounds` instead of the total number of full rounds.

use p3_field::PrimeField64;
use p3_util::relatively_prime_u64;

/// Given a field, a width and an ALPHA return the number of half full rounds and partial rounds needed to achieve 128 bit security.
///
/// Returns `(half_num_full_rounds, num_partial_rounds)` where:
/// - `half_num_full_rounds` is half the number of full rounds (full rounds are applied twice: at the beginning and end)
/// - `num_partial_rounds` is the number of partial rounds
///
/// If alpha is not a valid permutation of the given field or the optimal parameters for that size of prime
/// have not been computed, an error is returned.
pub const fn poseidon_round_numbers_128<F: PrimeField64>(
    width: usize,
    alpha: u64,
) -> Result<(usize, usize), &'static str> {
    // Start by checking that alpha is a valid permutation.
    if !relatively_prime_u64(alpha, F::ORDER_U64 - 1) {
        return Err("Invalid permutation: gcd(alpha, F::ORDER_U64 - 1) must be 1");
    }

    // Next compute the number of bits in p.
    let prime_bit_number = F::ORDER_U64.ilog2() + 1;

    // Get the total number of full rounds and partial rounds (same as Poseidon2)
    let (total_full_rounds, num_partial_rounds) = match prime_bit_number {
        31 => match (width, alpha) {
            (16, 3) => (8, 20),
            (16, 5) => (8, 14),
            (16, 7) => (8, 13),
            (16, 9) => (8, 13),
            (16, 11) => (8, 13),
            (24, 3) => (8, 23),
            (24, 5) => (8, 22),
            (24, 7) => (8, 21),
            (24, 9) => (8, 21),
            (24, 11) => (8, 21),
            (32, 5) => (8, 28), // Added for width 32 with alpha 5 (Mersenne31)
            (32, 7) => (8, 22), // Added for width 32 with alpha 7 (BabyBear)
            _ => return Err("The given pair of width and alpha has not been checked for these fields"),
        },
        64 => match (width, alpha) {
            (8, 3) => (8, 41),
            (8, 5) => (8, 27),
            (8, 7) => (8, 22),
            (8, 9) => (8, 19),
            (8, 11) => (8, 17),
            (12, 3) => (8, 42),
            (12, 5) => (8, 27),
            (12, 7) => (8, 22),
            (12, 9) => (8, 20),
            (12, 11) => (8, 18),
            (16, 3) => (8, 42),
            (16, 5) => (8, 27),
            (16, 7) => (8, 22),
            (16, 9) => (8, 20),
            (16, 11) => (8, 18),
            _ => return Err("The given pair of width and alpha has not been checked for these fields"),
        },
        _ => return Err("The optimal parameters for that size of prime have not been computed."),
    };

    // In Poseidon, full rounds are split into two halves
    // half_num_full_rounds should be total_full_rounds / 2
    let half_num_full_rounds = total_full_rounds / 2;
    
    Ok((half_num_full_rounds, num_partial_rounds))
}

#[cfg(test)]
mod tests {
    use super::*;
    use p3_baby_bear::BabyBear;
    use p3_goldilocks::Goldilocks;
    use p3_mersenne_31::Mersenne31;

    #[test]
    fn test_baby_bear_round_numbers() {
        // Test all BabyBear configurations from the benchmark
        assert_eq!(
            poseidon_round_numbers_128::<BabyBear>(16, 7),
            Ok((4, 13))
        );
        assert_eq!(
            poseidon_round_numbers_128::<BabyBear>(24, 7),
            Ok((4, 21))
        );
        assert_eq!(
            poseidon_round_numbers_128::<BabyBear>(32, 7),
            Ok((4, 22))
        );
    }

    #[test]
    fn test_goldilocks_round_numbers() {
        // Test all Goldilocks configurations from the benchmark
        assert_eq!(
            poseidon_round_numbers_128::<Goldilocks>(8, 7),
            Ok((4, 22))
        );
        assert_eq!(
            poseidon_round_numbers_128::<Goldilocks>(12, 7),
            Ok((4, 22))
        );
        assert_eq!(
            poseidon_round_numbers_128::<Goldilocks>(16, 7),
            Ok((4, 22))
        );
    }

    #[test]
    fn test_mersenne31_round_numbers() {
        // Test all Mersenne31 configurations from the benchmark
        assert_eq!(
            poseidon_round_numbers_128::<Mersenne31>(16, 5),
            Ok((4, 14))
        );
        assert_eq!(
            poseidon_round_numbers_128::<Mersenne31>(32, 5),
            Ok((4, 28))
        );
    }

    #[test]
    fn test_invalid_alpha() {
        // Test with invalid alpha (not relatively prime to ORDER_U64 - 1)
        // For BabyBear, ORDER_U64 - 1 = 2013265918, which has factors
        // Alpha = 2 would not be relatively prime
        let result = poseidon_round_numbers_128::<BabyBear>(16, 2);
        assert!(result.is_err());
    }

    #[test]
    fn test_unsupported_combination() {
        // Test with unsupported width/alpha combination
        let result = poseidon_round_numbers_128::<BabyBear>(64, 7);
        assert!(result.is_err());
    }

    #[test]
    fn test_half_full_rounds_calculation() {
        // Verify that half_num_full_rounds is correctly calculated
        // Total full rounds should always be 8, so half should be 4
        let (half_full, _) = poseidon_round_numbers_128::<BabyBear>(16, 7).unwrap();
        assert_eq!(half_full, 4);
    }
}
