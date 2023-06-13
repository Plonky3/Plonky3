use ethereum_types::U256;

pub fn binomial(n: usize, k: usize) -> U256 {
    let mut result = U256::one();
    for i in 0..k {
        result *= n - i;
        result /= i + 1;
    }

    result
}