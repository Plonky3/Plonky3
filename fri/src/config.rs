#[derive(Debug)]
pub struct FriConfig<M> {
    pub log_blowup: usize,
    pub log_final_poly_len: usize,
    pub num_queries: usize,
    pub proof_of_work_bits: usize,
    pub mmcs: M,
}

impl<M> FriConfig<M> {
    pub const fn blowup(&self) -> usize {
        1 << self.log_blowup
    }
    pub const fn final_poly_len(&self) -> usize {
        1 << self.log_final_poly_len
    }
}
