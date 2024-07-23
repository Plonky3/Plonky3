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
        Commit: FnMut(usize, &mut [Codeword<F, Code>]) -> Result<F, E>,
    {
        while let Some(log_max_word_len) = codewords
            .iter()
            .map(|cw| cw.code.log_word_len())
            .max()
            .filter(|&l| l > self.log_max_final_word_len)
        {
            let log_folded_word_len = log_max_word_len - self.log_folding_arity;
            let mut to_fold = codewords
                .extract(|cw| cw.code.log_word_len() > log_folded_word_len)
                .collect_vec();

            let beta = commit(log_folded_word_len, &mut to_fold)?;

            for cw in to_fold {
                let log_folding_arity = cw.code.log_word_len() - log_folded_word_len;
                let fcw = cw.fold_repeatedly(log_folding_arity, beta);
                if let Some(cw2) = codewords.iter_mut().find(|cw2| cw2.code == fcw.code) {
                    izip!(&mut cw2.word, fcw.word).for_each(|(l, r)| *l += r);
                } else {
                    codewords.push(fcw);
                }
            }
        }
        Ok(codewords)
    }
}
