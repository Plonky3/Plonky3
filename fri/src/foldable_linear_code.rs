use alloc::vec::Vec;
use core::{fmt::Debug, mem};
use itertools::{izip, Itertools};
use p3_field::Field;
use p3_util::{log2_strict_usize, split_bits, SliceExt};

pub trait FoldableLinearCode<F>: Sized + Clone + Eq + Debug {
    // fn new(log_blowup: usize, log_msg_len: usize) -> Self;

    fn log_blowup(&self) -> usize;
    fn log_msg_len(&self) -> usize;
    fn log_word_len(&self) -> usize {
        self.log_blowup() + self.log_msg_len()
    }

    fn blowup(&self) -> usize {
        1 << self.log_blowup()
    }
    fn msg_len(&self) -> usize {
        1 << self.log_msg_len()
    }
    fn word_len(&self) -> usize {
        1 << self.log_word_len()
    }

    fn encoded_at_point(&self, msg: &[F], index: usize) -> F;
    fn encode(&self, msg: &[F]) -> Vec<F> {
        (0..self.word_len())
            .map(|i| self.encoded_at_point(msg, i))
            .collect()
    }

    fn decode(&self, word: &[F]) -> Vec<F>;

    fn folded_code(&self) -> Self;
    fn fold_word_at_point(&self, beta: F, pair_index: usize, values: (F, F)) -> F;
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Codeword<F, C> {
    pub code: C,
    pub index: usize,
    pub word: Vec<F>,
}

impl<F: Field, C: FoldableLinearCode<F>> Codeword<F, C> {
    pub fn index_bits(&self) -> usize {
        self.code.log_word_len() - self.word.log_strict_len()
    }
    pub fn is_restricted(&self) -> bool {
        self.index_bits() > 0
    }

    pub fn restrict(&self, added_index_bits: usize, index: usize) -> Self {
        let new_index = (self.index << added_index_bits) | index;
        let new_word_bits = self.word.log_strict_len() - added_index_bits;
        let start = index << new_word_bits;
        let end = (index + 1) << new_word_bits;
        Codeword {
            code: self.code.clone(),
            index: new_index,
            word: self.word[start..end].to_vec(),
        }
    }
    pub fn expand(&self, mut siblings: Vec<F>) -> Self {
        assert_eq!(self.word.len(), 1);
        let removed_index_bits = log2_strict_usize(siblings.len() + 1);
        let (new_index, index_within_segment) = split_bits(self.index, removed_index_bits);
        siblings.insert(index_within_segment, self.word[0]);
        Codeword {
            code: self.code.clone(),
            index: new_index,
            word: siblings,
        }
    }

    pub fn decode(&self) -> Vec<F> {
        assert_eq!(self.index_bits(), 0);
        self.code.decode(&self.word)
    }

    pub fn fold(&mut self, beta: F) {
        assert_ne!(self.code.log_msg_len(), 0);
        let log_folded_word_len = self.word.log_strict_len() - 1;
        self.word = self
            .word
            .drain(..)
            .tuples()
            .enumerate()
            .map(|(i, values)| {
                self.code
                    .fold_word_at_point(beta, (self.index << log_folded_word_len) | i, values)
            })
            .collect();
        self.code = self.code.folded_code();
    }
    pub fn repeatedly_fold(&mut self, log_arity: usize, mut beta: F) {
        for _ in 0..log_arity {
            self.fold(beta);
            beta = beta.square();
        }
    }
    pub fn fold_to_log_word_len(&mut self, log_word_len: usize, beta: F) {
        self.repeatedly_fold(self.code.log_word_len() - log_word_len, beta);
    }
    pub fn fold_to_log_msg_len(&mut self, log_msg_len: usize, beta: F) {
        self.repeatedly_fold(self.code.log_msg_len() - log_msg_len, beta);
    }
    pub fn fold_to_point(&mut self, beta: F) {
        self.repeatedly_fold(self.word.log_strict_len(), beta);
    }

    pub fn sum_words_from_same_code(cws: &mut Vec<Self>) {
        for cw in mem::take(cws) {
            if let Some(cw2) = cws
                .iter_mut()
                .find(|cw2| cw2.code == cw.code && cw2.index == cw.index)
            {
                izip!(&mut cw2.word, cw.word).for_each(|(l, r)| *l += r);
            } else {
                cws.push(cw);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use p3_field::AbstractField;
    use p3_mersenne_31::Mersenne31;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha20Rng;

    use super::*;

    use std::marker::PhantomData;

    #[derive(Clone, PartialEq, Eq, Debug)]
    struct RepetitionCode<F: Field>(usize, usize, PhantomData<F>);

    impl<F: Field> FoldableLinearCode<F> for RepetitionCode<F> {
        fn log_blowup(&self) -> usize {
            self.0
        }
        fn log_msg_len(&self) -> usize {
            self.1
        }

        fn encoded_at_point(&self, msg: &[F], index: usize) -> F {
            msg[index % self.msg_len()]
        }
        fn decode(&self, word: &[F]) -> Vec<F> {
            word.chunks_exact(self.msg_len())
                .all_equal_value()
                .unwrap()
                .to_vec()
        }

        fn folded_code(&self) -> Self {
            Self(self.0, self.1 - 1, self.2)
        }
        fn fold_word_at_point(&self, beta: F, _pair_index: usize, (v0, v1): (F, F)) -> F {
            v0 + beta * v1
        }
    }

    fn m31s(iter: impl IntoIterator<Item = usize>) -> Vec<Mersenne31> {
        iter.into_iter()
            .map(|i| Mersenne31::from_canonical_usize(i))
            .collect()
    }

    #[test]
    fn test_foldable_linear_code() {
        let code: RepetitionCode<Mersenne31> = RepetitionCode(1, 2, PhantomData);
        let msg = m31s(0..code.msg_len());
        let word = code.encode(&msg);

        let cw = Codeword {
            code,
            index: 0,
            word,
        };
        assert_eq!(cw.index_bits(), 0);
        assert_eq!(cw.word, m31s([0, 1, 2, 3, 0, 1, 2, 3]));

        let lo = cw.restrict(1, 0);
        assert_eq!(lo.index_bits(), 1);
        assert_eq!(lo.word, m31s([0, 1, 2, 3]));

        assert_eq!(cw.decode(), m31s([0, 1, 2, 3]));

        let five = Mersenne31::from_canonical_usize(5);

        let mut fcw = cw.clone();
        fcw.fold(five);
        assert_eq!(fcw.decode(), m31s([(0 * 1) + (1 * 5), (2 * 1) + (3 * 5)]));

        let mut fcw = cw.clone();
        fcw.fold_to_log_msg_len(0, five);
        assert_eq!(fcw.code.log_msg_len(), 0);
        assert_eq!(fcw.code.log_word_len(), 1);
        assert_eq!(
            fcw.decode(),
            m31s([(0 * 1) + (1 * 5) + (2 * 25) + (3 * 125)])
        );

        let r = cw.restrict(cw.code.log_word_len(), 5);
        assert_eq!(r.word, m31s([1]));
        assert_eq!(r.expand(m31s([123])).word, m31s([123, 1]));
    }
}
