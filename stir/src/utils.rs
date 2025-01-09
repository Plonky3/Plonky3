use p3_challenger::FieldChallenger;
use p3_field::Field;

pub(crate) fn compute_pow(security_level: usize, error: f64) -> f64 {
    0f64.max(security_level as f64 - error)
}

/// Samples an integer in the range `[0, 1 << log_range)`
pub(crate) fn sample_integer<F: Field>(
    challenger: &mut impl FieldChallenger<F>,
    log_range: usize,
) -> usize {
    let bits = challenger.sample_bits(log_range);
    bits.into_iter().fold(0, |acc, bit| acc * 2 + bit)
}
