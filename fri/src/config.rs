use alloc::vec::Vec;
use core::{fmt::Debug, mem};
use itertools::{izip, Itertools};
use p3_util::{log2_strict_usize, split_bits, SliceExt};

use p3_field::Field;

#[derive(Debug)]
pub struct FriConfig<M> {
    pub log_blowup: usize,
    pub log_folding_arity: usize,
    pub log_max_final_poly_len: usize,

    pub num_queries: usize,
    pub proof_of_work_bits: usize,
    pub mmcs: M,
}

impl<M> FriConfig<M> {
    pub const fn blowup(&self) -> usize {
        1 << self.log_blowup
    }

    /// Returns the soundness bits of this FRI instance based on the
    /// [ethSTARK](https://eprint.iacr.org/2021/582) conjecture.
    ///
    /// Certain users may instead want to look at proven soundness, a more complex calculation which
    /// isn't currently supported by this crate.
    pub fn conjectured_soundness_bits(&self) -> usize {
        self.log_blowup * self.num_queries + self.proof_of_work_bits
    }
}

pub trait FoldableLinearCode<F: Field>: Sized + Clone + Eq + Debug {
    // fn new(log_blowup: usize, log_msg_len: usize) -> Self;

    fn log_blowup(&self) -> usize;
    fn log_msg_len(&self) -> usize;
    fn log_word_len(&self) -> usize {
        self.log_blowup() + self.log_msg_len()
    }

    fn encoded_at_point(&self, msg: &[F], index: usize) -> F;
    fn encode(&self, msg: &[F]) -> Vec<F>;
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
    pub fn segment(&self, index_bits: usize, index: usize) -> Self {
        let new_index = (self.index << index_bits) | index;
        let new_word_bits = self.word.log_strict_len() - index_bits;
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
        let added_bits = log2_strict_usize(siblings.len() + 1);
        let (new_index, index_within_segment) = split_bits(self.index, added_bits);
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

/*
pub struct CodewordSegment<F, C> {
    pub code: C,
    pub index: usize,
    pub segment: Vec<F>,
}

impl<F: Field, C: FoldableLinearCode<F>> CodewordSegment<F, C> {
    pub fn from_segment(log_blowup: usize, index: usize, segment: Vec<F>) -> Self {
        Self {
            code: C::new()
            index,
            segment,
        }
    }
}
    */
