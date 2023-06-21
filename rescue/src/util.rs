use ethereum_types::U256;
use gcd::Gcd;
use modinverse::egcd;
use p3_field::PrimeField64;
use sha3::{
    digest::{ExtendableOutput, Update, XofReader},
    Shake256,
};

pub(crate) fn binomial(n: usize, k: usize) -> U256 {
    let mut result = U256::one();
    for i in 0..k {
        result *= n - i;
        result /= i + 1;
    }

    result
}

pub(crate) fn get_alphas<F: PrimeField64>() -> (u64, u64) {
    let p = F::ORDER_U64;
    let alpha = (3..p).find(|&a| a.gcd(p - 1) == 1).unwrap();
    let alphainv = get_inverse::<F>(alpha);
    (alpha, alphainv)
}

pub(crate) fn get_inverse<F: PrimeField64>(alpha: u64) -> u64 {
    let p = F::ORDER_U64 as i64;
    let (_, alphainv, _) = egcd(alpha as i64, p - 1);
    alphainv.rem_euclid(p-1) as u64
}

pub(crate) fn shake256_hash(seed_bytes: &[u8], num_bytes: usize) -> Vec<u8> {
    let mut hasher = Shake256::default();
    hasher.update(seed_bytes);
    let mut reader = hasher.finalize_xof();
    let mut result = vec![0u8; num_bytes];
    reader.read(&mut result);
    result
}
