// We compute the optimal required number of external (full) and internal (partial) rounds using:
// See https://github.com/0xPolygonZero/hash-constants
// Note the the number of rounds required is the same for fields of approximately the same size.

use p3_field::PrimeField64;
// Mersenne31, KoalaBear and BabyBear fields
const PRIMES_31_BIT: [u64; 3] = [
    (1 << 31) - 1,
    (1 << 31) - (1 << 24) + 1,
    (1 << 31) - (1 << 27) + 1,
];

// Goldilocks and Crandall Primes
const PRIMES_64_BIT: [u64; 2] = [0xFFFFFFFF00000001, 0xFFFFFFFF70000001];

/// Given a field, a width and an alpha return the number of full and partial rounds needed to achieve 128 bit security.
pub fn poseidon2_round_numbers_128<F: PrimeField64>(width: usize, alpha: u64) -> (usize, usize) {
    assert!(PRIMES_31_BIT.contains(&F::ORDER_U64) || PRIMES_64_BIT.contains(&F::ORDER_U64));

    if PRIMES_31_BIT.contains(&F::ORDER_U64) {
        match (width, alpha) {
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
            _ => panic!("The given pair of width and alpha has not been checked for these fields"),
        }
    } else {
        match (width, alpha) {
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
            _ => panic!("The given pair of width and alpha has not been checked for these fields"),
        }
    }
}
