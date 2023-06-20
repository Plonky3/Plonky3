use ethereum_types::U256;
use gcd::Gcd;
use itertools::Itertools;
use modinverse::egcd;
use p3_field::PrimeField64;
use p3_util::{ceil_div_usize, log2_ceil_u64};
use sha3::{
    digest::{ExtendableOutput, Update, XofReader},
    Shake256,
};

pub fn binomial(n: usize, k: usize) -> U256 {
    let mut result = U256::one();
    for i in 0..k {
        result *= n - i;
        result /= i + 1;
    }

    result
}

fn get_alphas<F: PrimeField64>() -> (u64, u64) {
    let p = F::ORDER_U64;
    let alpha = (3..p).find(|&a| a.gcd(p - 1) == 1).unwrap();
    let alphainv = get_inverse::<F>(alpha);
    (alpha, alphainv)
}

fn get_inverse<F: PrimeField64>(alpha: u64) -> u64 {
    let p = F::ORDER_U64;
    let (_, alphainv, _) = egcd(alpha, p - 1);
    alphainv
}

fn shake256_hash(seed_bytes: &[u8]) -> Vec<u8> {
    let mut hasher = Shake256::default();
    hasher.update(seed_bytes);
    let mut reader = hasher.finalize_xof();
    let mut result = vec![0u8; 32];
    reader.read(&mut result);
    result
}

fn get_round_constants_rescue_prime<F: PrimeField64>(
    num_rounds: usize,
    width: usize,
    capacity: usize,
    sec_level: usize,
) -> Vec<F> {
    let p_bits = log2_ceil_u64(F::ORDER_U64) as usize;
    let bytes_per_int = ceil_div_usize(p_bits, 8) + 1;
    let num_bytes = bytes_per_int * 2 * width * num_rounds;
    let seed_string = format!(
        "Rescue-XLIX({},{},{},{}",
        F::ORDER_U64,
        width,
        capacity,
        sec_level,
    );
    let byte_string = shake256_hash(seed_string.as_bytes());

    byte_string
        .iter()
        .chunks(bytes_per_int)
        .into_iter()
        .map(|chunk| {
            let integer = chunk.collect_vec().iter().rev().fold(0, |acc, &byte| (acc << 8) + *byte as u64);
            F::from_canonical_u64(integer)
        })
        .collect()
}
