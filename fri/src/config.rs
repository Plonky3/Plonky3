use alloc::vec::Vec;
use core::fmt::Debug;
use itertools::Itertools;
use p3_util::SliceExt;

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

pub trait FoldableLinearCode<F: Field>: Sized + Eq + Debug {
    fn new(log_height: usize, log_blowup: usize) -> Self;

    fn encode(&self, message: &[F]) -> Vec<F>;
    fn decode(&self, codeword: &[F]) -> Vec<F>;

    fn folded_code(&self) -> Self;
    fn fold_word_at_point(&self, beta: F, pair_index: usize, values: (F, F)) -> F;

    /*
    fn fold_word(&self, beta: F, codeword: &[F]) -> Vec<F> {
        codeword
            .iter()
            .copied()
            .tuples()
            .enumerate()
            .map(|(i, values)| self.fold_word_at_point(beta, i, values))
            .collect()
    }

    fn repeatedly_folded_code(&self, log_arity: usize) -> Self {
        assert_ne!(log_arity, 0);
        let mut fc = self.folded_code();
        for _ in 1..log_arity {
            fc = fc.folded_code();
        }
        fc
    }

    fn repeatedly_fold_word_at_point(&self, beta: F, reduced_index: usize, values: &[F]) -> F {
        todo!()
    }

    fn repeatedly_fold_word(&self, log_arity: usize, mut beta: F, codeword: &[F]) -> Vec<F> {
        assert_ne!(log_arity, 0);
        let mut fw = self.fold_word(beta, codeword);
        let mut fc = self.folded_code();
        for _ in 1..log_arity {
            beta = beta.square();
            fw = fc.fold_word(beta, &fw);
            fc = fc.folded_code();
        }
        fw
    }
    */

    /*
    fn fold_at_point(&self, mut beta: F, reduced_index: usize, values: &[F]) -> F {
        let log_arity = values.log_strict_len();
        assert_ne!(log_arity, 0);
        if log_arity == 1 {
            self.fold_at_point_once(beta, pair_index, values)
        }
    }

    fn fold_once(&self, beta: F, codeword: &mut Vec<F>) {
        *codeword = codeword
            .drain(..)
            .tuples::<(F, F)>()
            .enumerate()
            .map(|(pair_index, values)| self.fold_at_point_once(beta, pair_index, values))
            .collect();
    }

    fn fold(&self, log_arity: usize, mut beta: F, codeword: &mut Vec<F>) {
        for _ in 0..log_arity {
            self.fold_once(beta, codeword);
            beta = beta.square();
        }
    }
    */

    /*
    fn fold_at_point(&mut self, beta: F, reduced_index: usize, values: &[F]) -> F {
        let log_arity = values.log_len();

        todo!()
    }
    */
}

pub struct Codeword<F, C> {
    pub code: C,
    pub word: Vec<F>,
}

impl<F: Field, C: FoldableLinearCode<F>> Codeword<F, C> {
    pub fn fold(&mut self, beta: F) {
        self.word = self
            .word
            .drain(..)
            .tuples()
            .enumerate()
            .map(|(i, values)| self.code.fold_word_at_point(beta, i, values))
            .collect();
        self.code = self.code.folded_code();
    }
    pub fn repeatedly_fold(&mut self, log_arity: usize, mut beta: F) {
        for _ in 0..log_arity {
            self.fold(beta);
            beta = beta.square();
        }
    }
}

pub struct CodewordSegment<F, C> {
    pub code: C,
    pub index: usize,
    pub segment: Vec<F>,
}

impl<F: Field, C: FoldableLinearCode<F>> Codeword<F, C> {}
