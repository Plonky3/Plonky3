use alloc::vec::Vec;

pub trait CryptographicHasher<T, const OUT_WIDTH: usize> {
    fn hash(input: Vec<T>) -> [T; OUT_WIDTH];
}
