use core::fmt::Debug;

use itertools::{izip, Itertools};
use p3_field::Field;
use p3_util::VecExt;

use crate::{CodeFamily, Codeword, FoldableCodeFamily};

#[derive(Debug)]
pub struct FriConfig<M> {
    pub log_folding_arity: usize,
    pub log_max_final_word_len: usize,

    pub num_queries: usize,
    pub proof_of_work_bits: usize,
    pub mmcs: M,
}

impl<M> FriConfig<M> {
    /*
    pub const fn blowup(&self) -> usize {
        1 << self.log_blowup
    }
    */

    /*
    /// Returns the soundness bits of this FRI instance based on the
    /// [ethSTARK](https://eprint.iacr.org/2021/582) conjecture.
    ///
    /// Certain users may instead want to look at proven soundness, a more complex calculation which
    /// isn't currently supported by this crate.
    pub fn conjectured_soundness_bits(&self) -> usize {
        self.log_blowup * self.num_queries + self.proof_of_work_bits
    }
    */

    pub fn fold_codewords<F, Code, Commit, E>(
        &self,
        mut codewords: Vec<Codeword<F, Code>>,
        mut commit: Commit,
    ) -> Result<Vec<Codeword<F, Code>>, E>
    where
        F: Field,
        Code: FoldableCodeFamily<F>,
        Commit: FnMut(&[(usize, Codeword<F, Code>)]) -> Result<F, E>,
    {
        while let Some(log_max_word_len) = codewords
            .iter()
            .map(|cw| cw.code.log_word_len())
            .max()
            .filter(|&l| l > self.log_max_final_word_len)
        {
            let log_folded_word_len = log_max_word_len - self.log_folding_arity;
            let to_fold = codewords
                .extract(|cw| cw.code.log_word_len() > log_folded_word_len)
                .map(|cw| (cw.code.log_word_len() - log_folded_word_len, cw))
                .collect_vec();

            let beta = commit(&to_fold)?;

            for (log_folding_arity, mut cw) in to_fold {
                cw.fold_repeatedly(log_folding_arity, beta);
                // todo: check is_linear
                if let Some(cw2) = codewords.iter_mut().find(|cw2| cw2.code == cw.code) {
                    izip!(&mut cw2.word, cw.word).for_each(|(l, r)| *l += r);
                } else {
                    codewords.push(cw);
                }
            }
        }
        Ok(codewords)
    }
}

pub struct CommitStep(Vec<Fold>);

pub struct Fold {
    pub log_arity: usize,
    pub add_to: Option<usize>,
}

#[derive(Clone, Debug)]
pub enum FriFoldingStrategy {
    ConstantArity(usize, usize),
}

impl FriFoldingStrategy {
    pub fn seq<F: Field, Code: FoldableCodeFamily<F>>(
        &self,
        mut codes: Vec<Code>,
    ) -> Vec<CommitStep> {
        match self {
            &FriFoldingStrategy::ConstantArity(log_arity, log_max_final_word_len) => {
                let mut steps = vec![];

                while let Some(log_word_len) = codes
                    .last()
                    .map(|c| c.log_word_len())
                    .filter(|&l| l > log_max_final_word_len)
                {
                    let log_folded_word_len = log_word_len - log_arity;

                    let pos = codes
                        .iter()
                        .position(|c| c.log_word_len() > log_folded_word_len)
                        .unwrap();

                    steps.push(CommitStep(
                        codes
                            .split_off(pos)
                            .into_iter()
                            .map(|c| {
                                let log_arity = c.log_word_len() - log_folded_word_len;
                                let fc = c.repeatedly_folded_code(log_arity);
                                let add_to = codes.iter().position(|c| c == &fc);
                                if add_to.is_none() {
                                    codes.push(fc);
                                }
                                Fold { log_arity, add_to }
                            })
                            .collect(),
                    ));
                }

                steps
            }
        }
    }
}
