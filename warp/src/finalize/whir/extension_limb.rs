//! Extension-field opening support over base-field WHIR commitments.
//!
//! WARP accumulator codewords live over the extension field `EF`, while the
//! current WHIR PCS commits base-field polynomials. The constructions here use
//! the fixed `F`-basis of `EF`: an extension polynomial `f` is represented by
//! limb polynomials `f_i` such that `f(x) = sum_i e_i f_i(x)`. Every limb is
//! committed/opened with WHIR, and verification recomposes the opened limb
//! values before comparing them to the WARP accumulator claims. This prevents
//! a prover from authenticating unrelated limb openings that do not reconstruct
//! the claimed extension-field value.

use super::*;

/// Adapter exposing a base-field multilinear PCS as an extension-field PCS.
///
/// WARP accumulator codewords live over `EF`, while WHIR's native PCS commits
/// base-field polynomials and evaluates them at extension-field points. This
/// adapter bridges that mismatch by decomposing every `EF` coefficient into
/// its fixed `F`-basis limbs, committing one limb polynomial per basis element,
/// and recomposing the opened limb evaluations in `EF`.
///
/// The proof carries the opened limb evaluations explicitly. That is necessary:
/// an evaluation of a base-field limb polynomial at an extension point is an
/// `EF` value, so the verifier cannot recover all limb evaluations from the
/// final recomposed `EF` claim alone.
#[derive(Clone, Debug)]
pub struct ExtensionLimbPcs<'a, F, EF, Inner>
where
    F: Field,
    EF: ExtensionField<F>,
{
    inner: &'a Inner,
    _ph: PhantomData<(F, EF)>,
}

impl<'a, F, EF, Inner> ExtensionLimbPcs<'a, F, EF, Inner>
where
    F: Field,
    EF: ExtensionField<F>,
{
    /// Wrap a base-field PCS whose challenge field is `EF`.
    pub const fn new(inner: &'a Inner) -> Self {
        Self {
            inner,
            _ph: PhantomData,
        }
    }
}

/// Prover data for [`ExtensionLimbPcs`].
pub struct ExtensionLimbPcsProverData<InnerData, Challenger> {
    limb_prover_data: Vec<InnerData>,
    limb_challengers: Vec<Challenger>,
}

/// Opening proof for [`ExtensionLimbPcs`].
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(
    bound = "EF: Serialize + serde::de::DeserializeOwned, InnerProof: Serialize + serde::de::DeserializeOwned"
)]
pub struct ExtensionLimbPcsProof<EF, InnerProof> {
    /// Opened values for each committed base-field limb polynomial.
    ///
    /// `limb_opened_values[i][p][j]` is the `EF` evaluation of limb `i`,
    /// polynomial `p`, at opening point `j`.
    pub limb_opened_values: Vec<MultilinearOpenedValues<EF>>,
    /// Inner PCS proofs, one per extension-field basis limb.
    pub limb_proofs: Vec<InnerProof>,
}

/// Verification failures for [`ExtensionLimbPcs`].
#[derive(Debug)]
pub enum ExtensionLimbPcsError<InnerError> {
    /// The commitment/proof did not contain one entry per extension basis limb.
    WrongLimbCount { expected: usize, actual: usize },
    /// A limb opening did not match the opening-claim shape.
    ShapeMismatch(&'static str),
    /// Limb openings did not recombine to the claimed extension-field values.
    RecompositionMismatch,
    /// The wrapped PCS rejected one limb opening.
    Inner { limb: usize, error: InnerError },
}

/// WHIR-backed commitment backend for WARP accumulator codewords.
///
/// The accumulator codeword lives over `EF`, while `p3-whir` commits one
/// base-field multilinear polynomial at a time. This backend decomposes the
/// accumulator into `EF` basis limbs, commits each limb once with
/// [`WhirPcs::commit_deferred`], and later proves VACC shift openings against
/// the saved WHIR prover data. The initial WHIR Merkle tree for a limb is not
/// rebuilt when a later WARP step asks for an opening.
#[derive(Debug)]
pub struct WhirLimbAccumulatorBackend<'a, F, EF, MT, Challenger, Dft, const DIGEST_ELEMS: usize>
where
    F: Field,
    EF: ExtensionField<F>,
    MT: p3_commit::Mmcs<F>,
{
    pcs: &'a WhirPcs<EF, F, MT, Challenger, Dft, DIGEST_ELEMS>,
    challenger_seed: Challenger,
    _ph: PhantomData<(F, EF)>,
}

impl<'a, F, EF, MT, Challenger, Dft, const DIGEST_ELEMS: usize>
    WhirLimbAccumulatorBackend<'a, F, EF, MT, Challenger, Dft, DIGEST_ELEMS>
where
    F: Field,
    EF: ExtensionField<F>,
    MT: p3_commit::Mmcs<F>,
{
    /// Create a WHIR-limb accumulator backend from a base-field WHIR PCS.
    ///
    /// `challenger_seed` is cloned and domain-separated per limb, so commit,
    /// open, and verify can reconstruct the same WHIR Fiat-Shamir transcript
    /// without borrowing WARP's step challenger after the step challenges have
    /// already been sampled.
    pub const fn new(
        pcs: &'a WhirPcs<EF, F, MT, Challenger, Dft, DIGEST_ELEMS>,
        challenger_seed: Challenger,
    ) -> Self {
        Self {
            pcs,
            challenger_seed,
            _ph: PhantomData,
        }
    }
}

/// Prover data for [`WhirLimbAccumulatorBackend`].
pub struct WhirLimbAccumulatorProverData<F, EF, MT, Challenger, const DIGEST_ELEMS: usize>
where
    F: Field,
    EF: ExtensionField<F>,
    MT: p3_commit::Mmcs<F>,
{
    limb_prover_data: Vec<WhirDeferredProverData<F, EF, MT, DIGEST_ELEMS>>,
    limb_challengers: Vec<Challenger>,
}

/// Opening proof for [`WhirLimbAccumulatorBackend`].
pub type WhirLimbAccumulatorOpeningProof<F, EF, MT> =
    ExtensionLimbPcsProof<EF, WhirProof<F, EF, MT>>;

impl<'a, F, EF, MT, Challenger, Dft, const DIGEST_ELEMS: usize>
    WhirLimbAccumulatorBackend<'a, F, EF, MT, Challenger, Dft, DIGEST_ELEMS>
where
    F: TwoAdicField + Ord,
    EF: ExtensionField<F> + TwoAdicField,
    MT: p3_commit::Mmcs<F>,
    MT::Commitment: Clone + PartialEq,
    Challenger: FieldChallenger<F>
        + GrindingChallenger<Witness = F>
        + CanObserve<F>
        + CanObserve<MT::Commitment>
        + Clone,
    Dft: TwoAdicSubgroupDft<F>,
{
    fn limb_challenger(&self, limb: usize) -> Challenger {
        extension_limb_challenger::<F, _>(&self.challenger_seed, limb)
    }

    /// Prove arbitrary multilinear openings against a previously committed
    /// accumulator codeword.
    pub fn prove_points(
        &self,
        prover_data: &WhirLimbAccumulatorProverData<F, EF, MT, Challenger, DIGEST_ELEMS>,
        opening_points: &[Vec<Point<EF>>],
    ) -> Result<
        (
            MultilinearOpenedValues<EF>,
            WhirLimbAccumulatorOpeningProof<F, EF, MT>,
        ),
        ExtensionLimbPcsError<p3_whir::pcs::verifier::errors::VerifierError>,
    >
    where
        WhirDeferredProverData<F, EF, MT, DIGEST_ELEMS>: Clone,
    {
        check_limb_count::<F, EF, p3_whir::pcs::verifier::errors::VerifierError>(
            prover_data.limb_prover_data.len(),
        )?;
        check_limb_count::<F, EF, p3_whir::pcs::verifier::errors::VerifierError>(
            prover_data.limb_challengers.len(),
        )?;

        let mut limb_opened_values = Vec::with_capacity(EF::DIMENSION);
        let mut limb_proofs = Vec::with_capacity(EF::DIMENSION);
        for limb in 0..EF::DIMENSION {
            let mut challenger = prover_data.limb_challengers[limb].clone();
            let (opened_values, proof) = self.pcs.open_deferred(
                prover_data.limb_prover_data[limb].clone(),
                opening_points,
                &mut challenger,
            );
            limb_opened_values.push(opened_values);
            limb_proofs.push(proof);
        }

        let opened_values = recompose_limb_opened_values::<F, EF>(&limb_opened_values).ok_or(
            ExtensionLimbPcsError::ShapeMismatch("limb opened values have inconsistent shape"),
        )?;

        Ok((
            opened_values,
            ExtensionLimbPcsProof {
                limb_opened_values,
                limb_proofs,
            },
        ))
    }

    /// Verify arbitrary multilinear openings against a WHIR-limb accumulator
    /// commitment.
    pub fn verify_points(
        &self,
        commitment: &[MT::Commitment],
        opening_claims: &[Vec<(Point<EF>, EF)>],
        proof: &WhirLimbAccumulatorOpeningProof<F, EF, MT>,
    ) -> Result<(), ExtensionLimbPcsError<p3_whir::pcs::verifier::errors::VerifierError>> {
        check_limb_count::<F, EF, p3_whir::pcs::verifier::errors::VerifierError>(commitment.len())?;
        check_limb_count::<F, EF, p3_whir::pcs::verifier::errors::VerifierError>(
            proof.limb_opened_values.len(),
        )?;
        check_limb_count::<F, EF, p3_whir::pcs::verifier::errors::VerifierError>(
            proof.limb_proofs.len(),
        )?;

        let recomposed = recompose_limb_opened_values::<F, EF>(&proof.limb_opened_values).ok_or(
            ExtensionLimbPcsError::ShapeMismatch("limb opened values have inconsistent shape"),
        )?;
        if !opening_claims_match_values(opening_claims, &recomposed) {
            return Err(ExtensionLimbPcsError::RecompositionMismatch);
        }

        for limb in 0..EF::DIMENSION {
            let limb_claims = limb_opening_claims(opening_claims, &proof.limb_opened_values[limb])?;
            let mut challenger = self.limb_challenger(limb);
            self.pcs
                .verify_deferred(
                    &commitment[limb],
                    &limb_claims,
                    &proof.limb_proofs[limb],
                    &mut challenger,
                )
                .map_err(|error| ExtensionLimbPcsError::Inner { limb, error })?;
        }

        Ok(())
    }
}

impl<'a, F, EF, MT, Challenger, Dft, const DIGEST_ELEMS: usize>
    AccumulatorCommitmentBackend<F, EF, Challenger>
    for WhirLimbAccumulatorBackend<'a, F, EF, MT, Challenger, Dft, DIGEST_ELEMS>
where
    F: TwoAdicField + Ord + Serialize + serde::de::DeserializeOwned,
    EF: ExtensionField<F> + TwoAdicField + Serialize + serde::de::DeserializeOwned,
    MT: p3_commit::Mmcs<F>,
    MT::Commitment: Clone + PartialEq + Serialize + serde::de::DeserializeOwned,
    WhirProof<F, EF, MT>: Clone + Serialize + serde::de::DeserializeOwned,
    WhirDeferredProverData<F, EF, MT, DIGEST_ELEMS>: Clone,
    Challenger: FieldChallenger<F>
        + GrindingChallenger<Witness = F>
        + CanObserve<F>
        + CanObserve<MT::Commitment>
        + Clone,
    Dft: TwoAdicSubgroupDft<F>,
{
    type Commitment = Vec<MT::Commitment>;
    type ProverData = WhirLimbAccumulatorProverData<F, EF, MT, Challenger, DIGEST_ELEMS>;
    type Proof = WhirLimbAccumulatorOpeningProof<F, EF, MT>;
    type Error = ExtensionLimbPcsError<p3_whir::pcs::verifier::errors::VerifierError>;

    fn commit(
        &self,
        codeword: Vec<EF>,
    ) -> Result<(Self::Commitment, Self::ProverData), Self::Error> {
        if codeword.len() != 1 << self.pcs.num_vars() {
            return Err(ExtensionLimbPcsError::ShapeMismatch(
                "accumulator codeword length does not match WHIR variable count",
            ));
        }

        let dimension = <EF as BasedVectorSpace<F>>::DIMENSION;
        let mut commitments = Vec::with_capacity(dimension);
        let mut limb_prover_data = Vec::with_capacity(dimension);
        let mut limb_challengers = Vec::with_capacity(dimension);

        for limb in 0..dimension {
            let limb_values = codeword
                .iter()
                .map(|value| <EF as BasedVectorSpace<F>>::as_basis_coefficients_slice(value)[limb])
                .collect::<Vec<_>>();
            let mut challenger = self.limb_challenger(limb);
            let (commitment, prover_data) = self
                .pcs
                .commit_deferred(RowMajorMatrix::new(limb_values, 1), &mut challenger);
            commitments.push(commitment);
            limb_prover_data.push(prover_data);
            limb_challengers.push(challenger);
        }

        Ok((
            commitments,
            WhirLimbAccumulatorProverData {
                limb_prover_data,
                limb_challengers,
            },
        ))
    }

    fn open(
        &self,
        prover_data: &Self::ProverData,
        index: usize,
    ) -> Result<(EF, Self::Proof), Self::Error> {
        let num_vars = self.pcs.num_vars();
        if index >= (1usize << num_vars) {
            return Err(ExtensionLimbPcsError::ShapeMismatch(
                "accumulator opening index out of range",
            ));
        }

        let opening_points = [vec![Point::new(boolean_index_point(index, num_vars))]];
        let (opened_values, proof) = self.prove_points(prover_data, &opening_points)?;
        let value = opened_values
            .first()
            .and_then(|values| values.first())
            .copied()
            .ok_or(ExtensionLimbPcsError::ShapeMismatch(
                "WHIR accumulator opening returned no value",
            ))?;
        Ok((value, proof))
    }

    fn observe_commitment(&self, challenger: &mut Challenger, commitment: &Self::Commitment) {
        for limb_commitment in commitment {
            challenger.observe(limb_commitment.clone());
        }
    }

    fn verify_opening(
        &self,
        commitment: &Self::Commitment,
        log_codeword_len: usize,
        index: usize,
        value: EF,
        proof: &Self::Proof,
    ) -> Result<(), Self::Error> {
        if log_codeword_len != self.pcs.num_vars() || index >= (1usize << log_codeword_len) {
            return Err(ExtensionLimbPcsError::ShapeMismatch(
                "accumulator opening shape does not match WHIR variable count",
            ));
        }

        let opening_claims = [vec![(
            Point::new(boolean_index_point(index, log_codeword_len)),
            value,
        )]];
        self.verify_points(commitment, &opening_claims, proof)
    }
}

impl<'a, F, EF, MT, Challenger, Dft, const DIGEST_ELEMS: usize>
    AccumulatorBatchOpeningBackend<F, EF, Challenger>
    for WhirLimbAccumulatorBackend<'a, F, EF, MT, Challenger, Dft, DIGEST_ELEMS>
where
    F: TwoAdicField + Ord + Serialize + serde::de::DeserializeOwned,
    EF: ExtensionField<F> + TwoAdicField + Serialize + serde::de::DeserializeOwned,
    MT: p3_commit::Mmcs<F>,
    MT::Commitment: Clone + PartialEq + Serialize + serde::de::DeserializeOwned,
    WhirProof<F, EF, MT>: Clone + Serialize + serde::de::DeserializeOwned,
    WhirDeferredProverData<F, EF, MT, DIGEST_ELEMS>: Clone,
    Challenger: FieldChallenger<F>
        + GrindingChallenger<Witness = F>
        + CanObserve<F>
        + CanObserve<MT::Commitment>
        + Clone,
    Dft: TwoAdicSubgroupDft<F>,
{
    type BatchProof = WhirLimbAccumulatorOpeningProof<F, EF, MT>;

    fn open_batch(
        &self,
        prover_data: &Self::ProverData,
        indices: &[usize],
    ) -> Result<(Vec<EF>, Self::BatchProof), Self::Error> {
        let num_vars = self.pcs.num_vars();
        let len = 1usize << num_vars;
        let points = indices
            .iter()
            .map(|&index| {
                if index >= len {
                    return Err(ExtensionLimbPcsError::ShapeMismatch(
                        "accumulator opening index out of range",
                    ));
                }
                Ok(Point::new(boolean_index_point(index, num_vars)))
            })
            .collect::<Result<Vec<_>, Self::Error>>()?;
        let opening_points = [points];
        let (opened_values, proof) = self.prove_points(prover_data, &opening_points)?;
        let values = opened_values
            .first()
            .ok_or(ExtensionLimbPcsError::ShapeMismatch(
                "WHIR accumulator opening returned no polynomial values",
            ))?
            .clone();
        if values.len() != indices.len() {
            return Err(ExtensionLimbPcsError::ShapeMismatch(
                "WHIR accumulator opening returned wrong number of values",
            ));
        }
        Ok((values, proof))
    }

    fn verify_batch_opening(
        &self,
        commitment: &Self::Commitment,
        log_codeword_len: usize,
        indices: &[usize],
        values: &[EF],
        proof: &Self::BatchProof,
    ) -> Result<(), Self::Error> {
        if log_codeword_len != self.pcs.num_vars() || indices.len() != values.len() {
            return Err(ExtensionLimbPcsError::ShapeMismatch(
                "accumulator opening shape does not match WHIR variable count",
            ));
        }
        let opening_claims = [indices
            .iter()
            .zip(values.iter())
            .map(|(&index, &value)| {
                if index >= (1usize << log_codeword_len) {
                    return Err(ExtensionLimbPcsError::ShapeMismatch(
                        "accumulator opening index out of range",
                    ));
                }
                Ok((
                    Point::new(boolean_index_point(index, log_codeword_len)),
                    value,
                ))
            })
            .collect::<Result<Vec<_>, Self::Error>>()?];
        self.verify_points(commitment, &opening_claims, proof)
    }
}

impl<'a, F, EF, MT, Challenger, Dft, const DIGEST_ELEMS: usize>
    AccumulatorPointOpeningBackend<F, EF, Challenger>
    for WhirLimbAccumulatorBackend<'a, F, EF, MT, Challenger, Dft, DIGEST_ELEMS>
where
    F: TwoAdicField + Ord,
    EF: ExtensionField<F> + TwoAdicField,
    MT: p3_commit::Mmcs<F>,
    MT::Commitment: PartialEq,
    Challenger:
        FieldChallenger<F> + GrindingChallenger<Witness = F> + CanObserve<MT::Commitment> + Clone,
    Dft: TwoAdicSubgroupDft<F>,
    WhirDeferredProverData<F, EF, MT, DIGEST_ELEMS>: Clone,
{
    type PointProof = WhirLimbAccumulatorOpeningProof<F, EF, MT>;
    type PointError = ExtensionLimbPcsError<p3_whir::pcs::verifier::errors::VerifierError>;

    fn num_vars(&self) -> usize {
        self.pcs.num_vars()
    }

    fn prove_points(
        &self,
        prover_data: &Self::ProverData,
        opening_points: &[Vec<Point<EF>>],
    ) -> Result<(MultilinearOpenedValues<EF>, Self::PointProof), Self::PointError> {
        WhirLimbAccumulatorBackend::prove_points(self, prover_data, opening_points)
    }

    fn verify_points(
        &self,
        commitment: &Self::Commitment,
        opening_claims: &[Vec<(Point<EF>, EF)>],
        proof: &Self::PointProof,
    ) -> Result<(), Self::PointError> {
        WhirLimbAccumulatorBackend::verify_points(self, commitment, opening_claims, proof)
    }
}

impl<'a, F, EF, Inner, Challenger> MultilinearPcs<EF, Challenger>
    for ExtensionLimbPcs<'a, F, EF, Inner>
where
    F: Field,
    EF: ExtensionField<F>,
    Inner: MultilinearPcs<EF, Challenger, Val = F>,
    Challenger: Clone + CanObserve<F>,
{
    type Val = EF;
    type Commitment = Vec<Inner::Commitment>;
    type ProverData = ExtensionLimbPcsProverData<Inner::ProverData, Challenger>;
    type Proof = ExtensionLimbPcsProof<EF, Inner::Proof>;
    type Error = ExtensionLimbPcsError<Inner::Error>;

    fn num_vars(&self) -> usize {
        self.inner.num_vars()
    }

    fn commit(
        &self,
        evaluations: RowMajorMatrix<Self::Val>,
        opening_points: &[Vec<Point<EF>>],
        challenger: &mut Challenger,
    ) -> (Self::Commitment, Self::ProverData) {
        let width = evaluations.width();
        let height = evaluations.height();
        let dimension = <EF as BasedVectorSpace<F>>::DIMENSION;
        let mut commitments = Vec::with_capacity(dimension);
        let mut limb_prover_data = Vec::with_capacity(dimension);
        let mut limb_challengers = Vec::with_capacity(dimension);

        for limb in 0..dimension {
            let mut limb_challenger = extension_limb_challenger::<F, _>(challenger, limb);
            let limb_values = evaluations
                .values
                .iter()
                .map(|value| <EF as BasedVectorSpace<F>>::as_basis_coefficients_slice(value)[limb])
                .collect::<Vec<_>>();
            let limb_matrix = RowMajorMatrix::new(limb_values, width);
            debug_assert_eq!(limb_matrix.height(), height);
            let (commitment, prover_data) =
                self.inner
                    .commit(limb_matrix, opening_points, &mut limb_challenger);
            commitments.push(commitment);
            limb_prover_data.push(prover_data);
            limb_challengers.push(limb_challenger);
        }

        (
            commitments,
            ExtensionLimbPcsProverData {
                limb_prover_data,
                limb_challengers,
            },
        )
    }

    fn open(
        &self,
        prover_data: Self::ProverData,
        _challenger: &mut Challenger,
    ) -> (MultilinearOpenedValues<EF>, Self::Proof) {
        let mut limb_opened_values = Vec::with_capacity(EF::DIMENSION);
        let mut limb_proofs = Vec::with_capacity(EF::DIMENSION);

        for (limb_data, mut limb_challenger) in prover_data
            .limb_prover_data
            .into_iter()
            .zip(prover_data.limb_challengers)
        {
            let (opened_values, proof) = self.inner.open(limb_data, &mut limb_challenger);
            limb_opened_values.push(opened_values);
            limb_proofs.push(proof);
        }

        let opened_values = recompose_limb_opened_values::<F, EF>(&limb_opened_values)
            .expect("inner PCS returned inconsistent limb opening shapes");
        (
            opened_values,
            ExtensionLimbPcsProof {
                limb_opened_values,
                limb_proofs,
            },
        )
    }

    fn verify(
        &self,
        commitment: &Self::Commitment,
        opening_claims: &[Vec<(Point<EF>, EF)>],
        proof: &Self::Proof,
        challenger: &mut Challenger,
    ) -> Result<(), Self::Error> {
        check_limb_count::<F, EF, Inner::Error>(commitment.len())?;
        check_limb_count::<F, EF, Inner::Error>(proof.limb_opened_values.len())?;
        check_limb_count::<F, EF, Inner::Error>(proof.limb_proofs.len())?;

        let recomposed = recompose_limb_opened_values::<F, EF>(&proof.limb_opened_values).ok_or(
            ExtensionLimbPcsError::ShapeMismatch("limb opened values have inconsistent shape"),
        )?;
        if !opening_claims_match_values(opening_claims, &recomposed) {
            return Err(ExtensionLimbPcsError::RecompositionMismatch);
        }

        for limb in 0..EF::DIMENSION {
            let limb_claims = limb_opening_claims(opening_claims, &proof.limb_opened_values[limb])?;
            let mut limb_challenger = extension_limb_challenger::<F, _>(challenger, limb);
            self.inner
                .verify(
                    &commitment[limb],
                    &limb_claims,
                    &proof.limb_proofs[limb],
                    &mut limb_challenger,
                )
                .map_err(|error| ExtensionLimbPcsError::Inner { limb, error })?;
        }

        Ok(())
    }
}

fn extension_limb_challenger<F, Challenger>(challenger: &Challenger, limb: usize) -> Challenger
where
    F: Field,
    Challenger: Clone + CanObserve<F>,
{
    let mut limb_challenger = challenger.clone();
    limb_challenger.observe(F::from_u64(domain::EXTENSION_LIMB_PCS));
    limb_challenger.observe(F::from_u64(limb as u64));
    limb_challenger
}

fn check_limb_count<F, EF, InnerError>(
    actual: usize,
) -> Result<(), ExtensionLimbPcsError<InnerError>>
where
    F: Field,
    EF: ExtensionField<F>,
{
    let dimension = <EF as BasedVectorSpace<F>>::DIMENSION;
    if actual != dimension {
        return Err(ExtensionLimbPcsError::WrongLimbCount {
            expected: dimension,
            actual,
        });
    }
    Ok(())
}

fn recompose_limb_opened_values<F, EF>(
    limb_opened_values: &[MultilinearOpenedValues<EF>],
) -> Option<MultilinearOpenedValues<EF>>
where
    F: Field,
    EF: ExtensionField<F>,
{
    let dimension = <EF as BasedVectorSpace<F>>::DIMENSION;
    if limb_opened_values.len() != dimension {
        return None;
    }
    if limb_opened_values.is_empty() {
        return Some(Vec::new());
    }

    let num_polys = limb_opened_values[0].len();
    let basis = (0..dimension)
        .map(|limb| {
            <EF as BasedVectorSpace<F>>::ith_basis_element(limb).expect("basis limb is in range")
        })
        .collect::<Vec<_>>();

    let mut recomposed = Vec::with_capacity(num_polys);
    for poly_idx in 0..num_polys {
        let num_points = limb_opened_values[0][poly_idx].len();
        let mut values = Vec::with_capacity(num_points);
        for point_idx in 0..num_points {
            let mut value = EF::ZERO;
            for limb in 0..dimension {
                if limb_opened_values[limb].len() != num_polys
                    || limb_opened_values[limb][poly_idx].len() != num_points
                {
                    return None;
                }
                value += limb_opened_values[limb][poly_idx][point_idx] * basis[limb];
            }
            values.push(value);
        }
        recomposed.push(values);
    }

    Some(recomposed)
}

fn opening_claims_match_values<EF: Field>(
    opening_claims: &[Vec<(Point<EF>, EF)>],
    opened_values: &MultilinearOpenedValues<EF>,
) -> bool {
    opening_claims.len() == opened_values.len()
        && opening_claims
            .iter()
            .zip(opened_values)
            .all(|(claims_for_poly, values_for_poly)| {
                claims_for_poly.len() == values_for_poly.len()
                    && claims_for_poly
                        .iter()
                        .zip(values_for_poly)
                        .all(|((_, claim), value)| claim == value)
            })
}

fn limb_opening_claims<EF: Field, InnerError>(
    opening_claims: &[Vec<(Point<EF>, EF)>],
    limb_values: &MultilinearOpenedValues<EF>,
) -> Result<Vec<Vec<(Point<EF>, EF)>>, ExtensionLimbPcsError<InnerError>> {
    if opening_claims.len() != limb_values.len() {
        return Err(ExtensionLimbPcsError::ShapeMismatch(
            "limb opened values do not match polynomial count",
        ));
    }

    opening_claims
        .iter()
        .zip(limb_values)
        .map(|(claims_for_poly, values_for_poly)| {
            if claims_for_poly.len() != values_for_poly.len() {
                return Err(ExtensionLimbPcsError::ShapeMismatch(
                    "limb opened values do not match opening point count",
                ));
            }
            Ok(claims_for_poly
                .iter()
                .zip(values_for_poly)
                .map(|((point, _), value)| (point.clone(), *value))
                .collect())
        })
        .collect()
}
