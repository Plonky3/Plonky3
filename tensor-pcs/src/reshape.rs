use p3_util::log2_strict_usize;

pub(crate) fn optimal_wraps(width: usize, height: usize) -> usize {
    let height_bits = log2_strict_usize(height);
    (0..height_bits)
        .min_by_key(|&wrap_bits| estimate_cost(width << wrap_bits, height >> wrap_bits))
        .unwrap()
}

fn estimate_cost(width: usize, height: usize) -> usize {
    // TODO: Better constants...
    width + height
}
