use core::fmt::Debug;

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
}
