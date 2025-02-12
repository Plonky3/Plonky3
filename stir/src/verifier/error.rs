use core::fmt::Debug;

#[derive(Debug, PartialEq)]
pub enum VerificationError {
    // The degree (plus 1) of the final polynomial p sent in plain is greater than
    // the bound starting_degree / product(folding_factors)
    FinalPolynomialDegree,
    // One of the Merkle proofs of the evaluations of g_M at the queried points
    // is incorrect
    FinalQueryPath,
    // The evaluations of the final polynomial p sent in plain do not match the
    // folded evaluations of f_M (checked through the shake-polynomial
    // evaluation)
    FinalPolynomialEvaluations,
    // The proof of work for round M + 1 is incorrect
    FinalProofOfWork,
    // Error when verifying a full round (round i for 1 <= i <= M)
    Round(usize, FullRoundVerificationError),
}

#[derive(Debug, PartialEq)]
pub enum FullRoundVerificationError {
    // The proof of work for this round is incorrect
    ProofOfWork,
    // One of the Merkle proofs of the evaluations of g_i at the queried indices
    // is incorrect
    QueryPath,
    // The degree of the polynomial f_i is greater than the the number of
    // (de-duplicated) OOD samples + queried indices for this round
    AnsPolynomialDegree,
    // The folded evaluations of the previous round's polynomial f_{i - 1} do
    // not match the evaluations of g_i (checked through the shake-polynomial
    // evaluation)
    AnsPolynomialEvaluations,
}
