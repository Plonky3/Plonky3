use p3_fri::FriParameters;

/// Fill only the commitment scheme; keep every caller-selected FRI parameter.
#[allow(
    clippy::needless_pass_by_value,
    reason = "Consume the parameter bundle like the PCS constructor."
)]
pub(crate) const fn with_mmcs<M>(params: FriParameters<()>, mmcs: M) -> FriParameters<M> {
    // Exhaustive destructuring catches new protocol options at compile time.
    let FriParameters {
        log_blowup,
        log_final_poly_len,
        max_log_arity,
        num_queries,
        batch_proof_of_work_bits,
        commit_proof_of_work_bits,
        query_proof_of_work_bits,
        mmcs: (),
    } = params;
    FriParameters {
        log_blowup,
        log_final_poly_len,
        max_log_arity,
        num_queries,
        batch_proof_of_work_bits,
        commit_proof_of_work_bits,
        query_proof_of_work_bits,
        mmcs,
    }
}
