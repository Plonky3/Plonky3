use alloc::vec;
use alloc::vec::Vec;

use p3_symmetric::CryptographicHasher;

use crate::{CanObserve, CanSample};

#[derive(Clone, Debug)]
pub struct HashChallenger<T, H, const OUT_LEN: usize>
where
    T: Clone,
    H: CryptographicHasher<T, [T; OUT_LEN]>,
{
    input_buffer: Vec<T>,
    output_buffer: Vec<T>,
    hasher: H,
}

impl<T, H, const OUT_LEN: usize> HashChallenger<T, H, OUT_LEN>
where
    T: Clone,
    H: CryptographicHasher<T, [T; OUT_LEN]>,
{
    pub fn new(initial_state: Vec<T>, hasher: H) -> Self {
        Self {
            input_buffer: initial_state,
            output_buffer: vec![],
            hasher,
        }
    }

    fn flush(&mut self) {
        let inputs = self.input_buffer.drain(..);
        let output = self.hasher.hash_iter(inputs);

        self.output_buffer = output.to_vec();

        // Chaining values.
        self.input_buffer.extend(output.to_vec());
    }
}

impl<T, H, const OUT_LEN: usize> CanObserve<T> for HashChallenger<T, H, OUT_LEN>
where
    T: Clone,
    H: CryptographicHasher<T, [T; OUT_LEN]>,
{
    fn observe(&mut self, value: T) {
        // Any buffered output is now invalid.
        self.output_buffer.clear();

        self.input_buffer.push(value);
    }
}

impl<T, H, const N: usize, const OUT_LEN: usize> CanObserve<[T; N]>
    for HashChallenger<T, H, OUT_LEN>
where
    T: Clone,
    H: CryptographicHasher<T, [T; OUT_LEN]>,
{
    fn observe(&mut self, values: [T; N]) {
        for value in values {
            self.observe(value);
        }
    }
}

impl<T, H, const OUT_LEN: usize> CanSample<T> for HashChallenger<T, H, OUT_LEN>
where
    T: Clone,
    H: CryptographicHasher<T, [T; OUT_LEN]>,
{
    fn sample(&mut self) -> T {
        if self.output_buffer.is_empty() {
            self.flush();
        }
        self.output_buffer
            .pop()
            .expect("Output buffer should be non-empty")
    }
}

#[cfg(test)]
mod tests {
    use p3_field::AbstractField;
    use p3_goldilocks::Goldilocks;

    use super::*;

    const OUT_LEN: usize = 2;
    type F = Goldilocks;

    #[derive(Clone)]
    struct TestHasher {}

    impl CryptographicHasher<F, [F; OUT_LEN]> for TestHasher {
        /// A very simple hash iterator. From an input of type `IntoIterator<Item = Goldilocks>`,
        /// it outputs the sum of its elements and its length (as a field element).
        fn hash_iter<I>(&self, input: I) -> [F; OUT_LEN]
        where
            I: IntoIterator<Item = F>,
        {
            let (sum, len) = input
                .into_iter()
                .fold((F::zero(), 0_usize), |(acc_sum, acc_len), f| {
                    (acc_sum + f, acc_len + 1)
                });
            [sum, F::from_canonical_usize(len)]
        }

        /// A very simple slice hash iterator. From an input of type `IntoIterator<Item = &'a [Goldilocks]>`,
        /// it outputs the sum of its elements and its length (as a field element).
        fn hash_iter_slices<'a, I>(&self, input: I) -> [F; OUT_LEN]
        where
            I: IntoIterator<Item = &'a [F]>,
            F: 'a,
        {
            let (sum, len) =
                input
                    .into_iter()
                    .fold((F::zero(), 0_usize), |(acc_sum, acc_len), n| {
                        (
                            acc_sum + n.iter().fold(F::zero(), |acc, f| acc + *f),
                            acc_len + n.len(),
                        )
                    });
            [sum, F::from_canonical_usize(len)]
        }
    }

    #[test]
    fn test_hash_challenger() {
        let initial_state = (1..11_u8).map(F::from_canonical_u8).collect::<Vec<_>>();
        let test_hasher = TestHasher {};
        let mut hash_challenger = HashChallenger::new(initial_state.clone(), test_hasher);

        assert_eq!(hash_challenger.input_buffer, initial_state);
        assert_eq!(hash_challenger.output_buffer, vec![]);

        hash_challenger.flush();

        let expected_sum = F::from_canonical_u8(55);
        let expected_len = F::from_canonical_u8(10);
        assert_eq!(
            hash_challenger.input_buffer,
            vec![expected_sum, expected_len]
        );
        assert_eq!(
            hash_challenger.output_buffer,
            vec![expected_sum, expected_len]
        );

        let new_element = F::from_canonical_u8(11);
        hash_challenger.observe(new_element);
        assert_eq!(
            hash_challenger.input_buffer,
            vec![expected_sum, expected_len, new_element]
        );
        assert_eq!(hash_challenger.output_buffer, vec![]);

        let new_expected_len = 3;
        let new_expected_sum = 76;

        let new_element = hash_challenger.sample();
        assert_eq!(new_element, F::from_canonical_u8(new_expected_len));
        assert_eq!(
            hash_challenger.output_buffer,
            [F::from_canonical_u8(new_expected_sum)]
        )
    }
}
