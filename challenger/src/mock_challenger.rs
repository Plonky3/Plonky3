use alloc::vec::Vec;
use p3_field::Field;

use crate::{CanObserve, CanSample, CanSampleBits, FieldChallenger, GrindingChallenger};

/// Mocked Challenger structure which replies to item (e. g. field-element), bit
/// or grinding queries using preloaded answers (grinding is not checked). This
/// is useful, for instance, when constructing hand-crafted test cases for
/// Fiat-Shamired protocols. Replies are given in FIFO order, that is, the first
/// element of the provided vector is the first reply. The challenger panics if
/// asked for more items or bits than were preloaded.
#[derive(Clone, Debug)]
pub struct MockChallenger<T> {
    item_replies: Vec<T>,
    bit_replies: Vec<usize>,
}

pub type MockBitChallenger = MockChallenger<()>;

impl<T> MockChallenger<T> {
    /// Create a new MockChallenger with the given replies.
    /// - `item_replies` are the replies for the `FieldChallenger` interface
    /// - `bit_replies` are the replies for the `CanSampleBits` interface (the `bits` argument is ignored)
    pub fn new(item_replies: Vec<T>, bit_replies: Vec<usize>) -> Self {
        // We reverse here so that replies are obtained by pop()ing, as that is
        // cheaper than removing elements from the front of a Vector
        Self {
            item_replies: item_replies.into_iter().rev().collect(),
            bit_replies: bit_replies.into_iter().rev().collect(),
        }
    }
}

impl<T> CanSample<T> for MockChallenger<T> {
    fn sample(&mut self) -> T {
        self.item_replies.pop().expect("No item replies left")
    }
}

impl<S, T> CanObserve<S> for MockChallenger<T> {
    fn observe(&mut self, _: S) {}
}

impl<T> CanSampleBits<usize> for MockChallenger<T> {
    fn sample_bits(&mut self, _: usize) -> usize {
        self.bit_replies.pop().expect("No bit replies left")
    }
}

impl<F: Field> FieldChallenger<F> for MockChallenger<F> {}

impl<F: Field> GrindingChallenger for MockChallenger<F> {
    type Witness = F;

    fn grind(&mut self, _: usize) -> Self::Witness {
        F::ZERO
    }

    fn check_witness(&mut self, _: usize, _: Self::Witness) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use p3_field::FieldAlgebra;
    use p3_goldilocks::Goldilocks;

    #[test]
    fn test_mock_challenger() {
        let item_replies: Vec<_> = (0..10).map(Goldilocks::from_canonical_usize).collect();
        let bit_replies = vec![1, 2, 3, 4, 5, 6];

        let mut challenger = MockChallenger::new(item_replies.clone(), bit_replies.clone());

        // Test field element sampling
        item_replies
            .into_iter()
            .for_each(|item| assert_eq!(challenger.sample(), item));

        // Test bit sampling
        bit_replies
            .into_iter()
            .for_each(|item| assert_eq!(challenger.sample_bits(8), item));

        // Test grinding (remember this is not checked)
        assert_eq!(challenger.grind(4242), Goldilocks::ZERO);
        assert_eq!(
            challenger.check_witness(234, Goldilocks::from_canonical_usize(2424)),
            true
        );
    }

    #[test]
    #[should_panic(expected = "No item replies left")]
    fn test_mock_challenger_item_panic() {
        let field_els: Vec<_> = (42..47).map(Goldilocks::from_canonical_usize).collect();
        let mut challenger = MockChallenger::<Goldilocks>::new(field_els, vec![]);
        (0..6).for_each(|_| {
            challenger.sample();
        });
    }

    #[test]
    #[should_panic(expected = "No bit replies left")]
    fn test_mock_challenger_bits_panic() {
        let bit_els: Vec<_> = (2..5).collect();
        let mut challenger = MockChallenger::<Goldilocks>::new(vec![], bit_els);
        (0..4).for_each(|_| {
            challenger.sample_bits(8);
        });
    }
}
