use alloc::vec;
use alloc::vec::Vec;

use p3_field::TwoAdicField;

/// A generalization of even-odd decomposition.
pub fn decompose<F: TwoAdicField>(poly: Vec<F>, log_chunks: usize) -> Vec<Vec<F>> {
    let chunk_height = poly.len() >> log_chunks;
    vec![vec![F::ZERO; chunk_height], vec![F::ZERO; chunk_height]] // TODO: dummy data for now
}
