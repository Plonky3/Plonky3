/// Given a fri query index, return the index of its sibling.
/// Masks the lower `index_bits` bits, rotates left one bit, and flips bottom bit.
pub(crate) fn query_index_sibling(index: usize, index_bits: usize) -> usize {
    let top_bit = (index >> (index_bits - 1)) & 1;
    let bottom_bits_mask = (1 << (index_bits - 1)) - 1;
    (top_bit ^ 1) | ((index & bottom_bits_mask) << 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_query_index() {
        assert_eq!(query_index_sibling(0b0, 1), 0b1);
        assert_eq!(query_index_sibling(0b1, 1), 0b0);
        assert_eq!(query_index_sibling(0b0010, 4), 0b0101);
        assert_eq!(query_index_sibling(0b1001, 4), 0b0010);
        assert_eq!(query_index_sibling(0b1010, 4), 0b0100);
        assert_eq!(query_index_sibling(0b1011, 4), 0b0110);
        assert_eq!(query_index_sibling(0b1111011, 4), 0b0110);
    }
}
