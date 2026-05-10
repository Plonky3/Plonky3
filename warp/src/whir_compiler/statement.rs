use super::*;
use p3_whir::constraints::statement::LinearSigmaOpeningClaim;
use p3_whir::pcs::{WhirLinearSigmaError, WhirLinearSigmaProof};

/// One WARP evaluation claim against the folded RS codeword oracle.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NativeWarpWhirEvalClaim<EF> {
    /// Multilinear point in the codeword hypercube.
    pub point: Point<EF>,
    /// Claimed value of the folded codeword MLE at `point`.
    pub value: EF,
}

impl<EF> NativeWarpWhirEvalClaim<EF> {
    /// Create a new evaluation claim.
    pub const fn new(point: Point<EF>, value: EF) -> Self {
        Self { point, value }
    }
}

/// Statement emitted by the WARP layer for one folded oracle.
///
/// The constraints are linear Sigma constraints of the form
/// `sum_b a(b) * f_hat(b) = sigma`, ready for WHIR's constrained-RS compiler.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NativeWarpWhirOracleStatement<EF> {
    /// Linear Sigma constraints over the folded codeword oracle.
    pub constraints: LinearSigmaStatement<EF>,
}

impl<EF> NativeWarpWhirOracleStatement<EF> {
    /// Create a statement from constraints.
    pub const fn new(constraints: LinearSigmaStatement<EF>) -> Self {
        Self { constraints }
    }
}

impl<EF: Field> NativeWarpWhirOracleStatement<EF> {
    /// Prove the compiled WARP linear-Sigma statement against a base-field
    /// folded RS oracle, reducing it to one residual opening claim.
    pub fn prove_reduction_base<F, Challenger>(
        &self,
        oracle: &Poly<F>,
        challenger: &mut Challenger,
        pow_bits: usize,
    ) -> Result<
        (
            LinearSigmaReductionProof<F, EF>,
            LinearSigmaOpeningClaim<EF>,
        ),
        LinearSigmaReductionError,
    >
    where
        F: TwoAdicField,
        EF: ExtensionField<F> + TwoAdicField,
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        self.constraints
            .prove_reduction_base(oracle, challenger, pow_bits)
    }

    /// Prove the compiled WARP linear-Sigma statement against an
    /// extension-field folded RS oracle.
    pub fn prove_reduction_ext<F, Challenger>(
        &self,
        oracle: &Poly<EF>,
        challenger: &mut Challenger,
        pow_bits: usize,
    ) -> Result<
        (
            LinearSigmaReductionProof<F, EF>,
            LinearSigmaOpeningClaim<EF>,
        ),
        LinearSigmaReductionError,
    >
    where
        F: TwoAdicField,
        EF: ExtensionField<F> + TwoAdicField,
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        self.constraints
            .prove_reduction_ext::<F, _>(oracle, challenger, pow_bits)
    }

    /// Verify the compiled WARP linear-Sigma reduction and return the residual
    /// opening claim to be checked against the committed RS oracle.
    pub fn verify_reduction<F, Challenger>(
        &self,
        proof: &LinearSigmaReductionProof<F, EF>,
        challenger: &mut Challenger,
        pow_bits: usize,
    ) -> Result<LinearSigmaOpeningClaim<EF>, LinearSigmaReductionError>
    where
        F: TwoAdicField,
        EF: ExtensionField<F> + TwoAdicField,
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        self.constraints
            .verify_reduction(proof, challenger, pow_bits)
    }

    /// Prove this compiled WARP statement against an existing deferred WHIR
    /// commitment, including the residual WHIR opening that binds the
    /// sumcheck reduction to the committed oracle.
    pub fn prove_bound_deferred<F, MT, Challenger, Dft, const DIGEST_ELEMS: usize>(
        &self,
        pcs: &WhirPcs<EF, F, MT, Challenger, Dft, DIGEST_ELEMS>,
        prover_data: WhirDeferredProverData<F, EF, MT, DIGEST_ELEMS>,
        challenger: &mut Challenger,
        reduction_pow_bits: usize,
    ) -> Result<
        (LinearSigmaOpeningClaim<EF>, WhirLinearSigmaProof<F, EF, MT>),
        LinearSigmaReductionError,
    >
    where
        F: TwoAdicField + Ord,
        EF: ExtensionField<F> + TwoAdicField,
        MT: Mmcs<F>,
        Challenger:
            FieldChallenger<F> + GrindingChallenger<Witness = F> + CanObserve<MT::Commitment>,
        Dft: TwoAdicSubgroupDft<F>,
    {
        pcs.open_linear_sigma_deferred(
            prover_data,
            &self.constraints,
            challenger,
            reduction_pow_bits,
        )
    }

    /// Verify a bound WARP/WHIR proof against the supplied commitment.
    pub fn verify_bound_deferred<F, MT, Challenger, Dft, const DIGEST_ELEMS: usize>(
        &self,
        pcs: &WhirPcs<EF, F, MT, Challenger, Dft, DIGEST_ELEMS>,
        commitment: &MT::Commitment,
        proof: &WhirLinearSigmaProof<F, EF, MT>,
        challenger: &mut Challenger,
        reduction_pow_bits: usize,
    ) -> Result<LinearSigmaOpeningClaim<EF>, WhirLinearSigmaError>
    where
        F: TwoAdicField + Ord,
        EF: ExtensionField<F> + TwoAdicField,
        MT: Mmcs<F>,
        MT::Commitment: PartialEq,
        Challenger:
            FieldChallenger<F> + GrindingChallenger<Witness = F> + CanObserve<MT::Commitment>,
        Dft: TwoAdicSubgroupDft<F>,
    {
        pcs.verify_linear_sigma_deferred(
            commitment,
            &self.constraints,
            proof,
            challenger,
            reduction_pow_bits,
        )
    }
}
