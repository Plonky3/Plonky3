use core::fmt::Debug;

/// Error during the verification process which causes the proof to be rejected
#[derive(Debug, PartialEq)]
pub enum VerificationError {
    /// The degree plus 1 of the final polynomial `p = g_{M + 1}` sent in plain
    /// is greater than the bound `starting_degree / product(folding_factors)`
    FinalPolynomialDegree,
    /// At least one of the Merkle proofs of the committed evaluations of `g_M`
    /// is invalid
    FinalQueryPath,
    /// The evaluations of the final polynomial `p = g_{M + 1}` sent in plain
    /// do not match the folded evaluations of `f_M`
    FinalPolynomialEvaluations,
    /// The proof of work for the final round `i = M + 1` is incorrect
    FinalProofOfWork,
    /// Invalid proof for the `i`-th full round (`1 <= i <= M`)
    Round(usize, FullRoundVerificationError),
}

/// Error during the verification of the `i`-th full round (`1 <= i <= M`)
#[derive(Debug, PartialEq)]
pub enum FullRoundVerificationError {
    /// Invalid proof of work for this round
    ProofOfWork,
    /// At least one of the Merkle proofs of the evaluations of `g_{i - 1}` at
    /// the queried indices is incorrect
    QueryPath,
    /// The degree of the polynomial `Ans_i` is greater than the length of the
    /// (de-duplicated) list of in-domain and out-of-domain sampled points for
    /// this round
    AnsPolynomialDegree,
    /// The folded evaluations of the previous round's polynomial `f_{i - 1}`
    /// do not match the purported evaluations of `g_i` interpolated by `Ans_i`
    AnsPolynomialEvaluations,
}
