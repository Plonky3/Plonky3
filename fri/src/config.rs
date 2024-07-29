#[derive(Debug)]
pub struct FriConfig<M> {
    pub log_blowup: usize,
    pub log_arity: usize,
    pub num_queries: usize,
    pub proof_of_work_bits: usize,
    pub mmcs: M,
}

impl<M> FriConfig<M> {
    pub const fn blowup(&self) -> usize {
        1 << self.log_blowup
    }
}
