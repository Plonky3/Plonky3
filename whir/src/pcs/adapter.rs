//! Adapter implementing the multilinear PCS trait for the WHIR protocol.

use alloc::string::ToString;
use alloc::vec::Vec;
use core::marker::PhantomData;
use core::slice::from_ref;

use p3_challenger::{CanObserve, CanSampleUniformBits, FieldChallenger, GrindingChallenger};
use p3_commit::{BatchOpeningRef, Mmcs, MultilinearPcs};
use p3_dft::TwoAdicSubgroupDft;
use p3_field::{ExtensionField, Field, TwoAdicField, dot_product};
use p3_matrix::dense::DenseMatrix;
use p3_matrix::extension::FlatMatrixView;
use p3_matrix::{Dimensions, Matrix};
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;
use p3_zk_codes::{LinearZkEncoding, ZkEncoding, ZkEncodingWithRandomness};
use rand::distr::{Distribution, StandardUniform};
use rand::{Rng, RngExt};

use super::prover::WhirProver;
use super::verifier::WhirVerifier;
use super::verifier::errors::VerifierError;
use crate::pcs::code_switch::{
    CodeSwitchOutputRelation, ZkMaskClaim, batched_claim, batching_coefficients,
    private_ood_answers,
};
use crate::pcs::committer::writer::commit_extension;
use crate::pcs::proof::{PcsProof, QueryOpening, WhirInitialZkProof, WhirRoundZkProof};
use crate::pcs::utils::get_challenge_stir_queries;
use crate::sumcheck::OpeningProtocol;
use crate::sumcheck::layout::{Layout, PrefixProver, Verifier, Witness};
use crate::sumcheck::product_polynomial::ProductPolynomial;
use crate::sumcheck::strategy::{SumcheckProver, VariableOrder};
use crate::sumcheck::zk::{
    ZkPrefixProver, ZkSumcheckData, ZkSumcheckHandoff, ZkVerifier, ZkVerifierHandoff,
};

/// Prover-side handoff between the commit and open phases of the PCS.
///
/// # Lifecycle
///
/// - Built by the commit phase alongside the public commitment.
/// - Stored by the caller while the public transcript advances.
/// - Consumed by the opening phase; never reused afterwards.
pub struct WhirProverData<F, EF, MT, L>
where
    F: TwoAdicField,
    EF: ExtensionField<F>,
    MT: Mmcs<F>,
    L: Layout<F, EF>,
{
    /// Layout-mode prover holding the per-table opening claims accumulator.
    pub layout: L,
    /// Merkle prover data behind the initial commitment; reused to open STIR queries.
    pub merkle_data: MT::ProverData<DenseMatrix<F>>,
    /// Marker tying the data to its extension field; carries no runtime state.
    _marker: PhantomData<EF>,
}

/// Prefix-only ZK opening state after the initial HVZK sumcheck.
///
/// This is the dedicated API boundary for the ZK path. It deliberately does
/// not implement `MultilinearPcs::open`, because the ZK protocol needs an RNG
/// and an explicit mask encoding. The future `round_zk_prefix` flow consumes
/// `initial_handoff` together with `source_merkle_data`.
pub struct WhirZkPrefixOpenState<F, EF, Enc, MT>
where
    F: TwoAdicField,
    EF: ExtensionField<F>,
    Enc: ZkEncoding<F>,
    MT: Mmcs<F>,
{
    /// Partial proof containing public opening evaluations and the initial ZK sumcheck transcript.
    pub proof: PcsProof<F, EF, MT>,
    /// Typed Construction 6.3 handoff consumed by the first code-switch round.
    pub initial_handoff: ZkSumcheckHandoff<F, EF, Enc, MT>,
    /// Merkle prover data for the inherited source oracle queried by the first code-switch round.
    pub source_merkle_data: MT::ProverData<DenseMatrix<F>>,
}

/// Prover-side source relation consumed by one code-switch round.
///
/// The fields deliberately describe one coherent source object: the message
/// behind the source commitment, the covector defining the inherited linear
/// claim, and the row layout used to turn queried Merkle positions into
/// generator-matrix rows.
pub struct ZkCodeSwitchProverSource<EF>
where
    EF: Field,
{
    /// Source oracle message before encoding.
    pub message: Vec<EF>,
    /// Covector for the inherited source claim.
    pub covector: Vec<EF>,
    /// Inherited scalar claim handed off by the previous IOR.
    ///
    /// For the #1605 HVZK sumcheck this is `eps * <message, covector>` in
    /// the no-auxiliary case.
    pub inherited_claim: EF,
    /// Extra residual scale to apply when this source enters Construction 9.7.
    ///
    /// Use `eps` when carrying an unscaled residual relation into code-switch.
    /// Use `1` when `message` / `covector` already include the HVZK residual
    /// scaling, as with a source built from `ZkSumcheckHandoff::residual_prover`.
    pub residual_sumcheck_scale: EF,
    /// Randomness segment length of the source encoding.
    pub randomness_len: usize,
    /// Full encoded codeword domain size.
    pub domain_size: usize,
    /// Row width exponent used by the WHIR Merkle layout.
    pub folding_factor: usize,
}

/// Verifier-side source relation consumed by one code-switch round.
pub struct ZkCodeSwitchVerifierSource<F, EF, MT>
where
    F: TwoAdicField,
    EF: ExtensionField<F>,
    MT: Mmcs<F>,
{
    /// Source oracle commitment already bound in the transcript.
    pub commitment: MT::Commitment,
    /// Message length before encoding.
    pub message_len: usize,
    /// Covector for the inherited source claim.
    pub covector: Vec<EF>,
    /// Extra residual scale to apply when this source enters Construction 9.7.
    ///
    /// This mirrors the prover-side source convention. The round-0 verifier
    /// uses the initial #1605 `eps`; later sources built from an already scaled
    /// residual handoff should use `1`.
    pub residual_sumcheck_scale: EF,
    /// Randomness segment length of the source encoding.
    pub randomness_len: usize,
    /// Full encoded codeword domain size.
    pub domain_size: usize,
    /// Row width exponent used by the WHIR Merkle layout.
    pub folding_factor: usize,
}

/// Verifier-side relation metadata for a source committed through a linear ZK encoding.
pub struct ZkEncodedCodeSwitchVerifierSource<EF>
where
    EF: Field,
{
    /// Message length before source encoding.
    pub message_len: usize,
    /// Covector for the inherited source claim.
    pub covector: Vec<EF>,
    /// Extra residual scale to apply when this source enters Construction 9.7.
    pub residual_sumcheck_scale: EF,
    /// Full encoded source domain size.
    pub domain_size: usize,
    /// Source encoding randomness length carried in the code-switch mask message.
    pub randomness_len: usize,
}

/// Verifier-side state after replaying one prefix-ZK code-switch round.
pub struct WhirZkPrefixVerifierRoundState<EF>
where
    EF: Field,
{
    /// Typed nested ZK sumcheck handoff for the next round.
    pub handoff: ZkVerifierHandoff<EF>,
    /// Verifier-side source relation for the next ZK code-switch round, if one exists.
    pub next_source: Option<ZkEncodedCodeSwitchVerifierSource<EF>>,
}

/// Prover-side state after running the prefix-ZK code-switch round loop.
///
/// This is the Construction 9.7 output boundary for the dedicated ZK path.
/// The final handoff is still masked by the nested Construction 6.3 sumcheck,
/// so it must be composed with a ZK-aware downstream/base-case protocol. It is
/// intentionally not fed into the plain WHIR final-polynomial tail, which would
/// require unmasking data that the verifier should not learn.
pub struct WhirZkPrefixRoundsState<F, EF, Enc, MT>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
    Enc: ZkEncoding<F>,
    MT: Mmcs<F>,
{
    /// Partial proof with every intermediate ZK round populated.
    pub proof: PcsProof<F, EF, MT>,
    /// Typed nested ZK sumcheck handoff after the last code-switch round.
    pub handoff: ZkSumcheckHandoff<F, EF, Enc, MT>,
    /// Carried source for a further code-switch round. This is `None` once all
    /// intermediate rounds have been consumed.
    pub next_source: Option<ZkCodeSwitchProverSource<EF>>,
}

/// Prefix-ZK state after one code-switch round.
pub struct WhirZkPrefixRoundState<F, EF, Enc, MT>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
    Enc: ZkEncoding<F>,
    MT: Mmcs<F>,
{
    /// Partial proof with `rounds[round_index].zk` populated.
    pub proof: PcsProof<F, EF, MT>,
    /// Typed nested ZK sumcheck handoff for the next round.
    pub handoff: ZkSumcheckHandoff<F, EF, Enc, MT>,
    /// Merkle prover data for the newly committed folded oracle.
    pub target_merkle_data: <MT as Mmcs<F>>::ProverData<FlatMatrixView<F, EF, DenseMatrix<EF>>>,
    /// Typed source relation for the next ZK code-switch round, if one exists.
    pub next_source: Option<ZkCodeSwitchProverSource<EF>>,
}

/// Prefix-ZK state after a code-switch round whose source was linearly ZK-encoded over `EF`.
pub struct WhirZkPrefixEncodedRoundState<F, EF, Enc, MT>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
    Enc: ZkEncoding<F>,
    MT: Mmcs<F>,
{
    /// Partial proof with `rounds[round_index].zk` populated.
    pub proof: PcsProof<F, EF, MT>,
    /// Typed nested ZK sumcheck handoff for the next round.
    pub handoff: ZkSumcheckHandoff<F, EF, Enc, MT>,
    /// Source commitment produced and opened by this round.
    pub source_commitment: MT::Commitment,
    /// Typed source relation for the next ZK code-switch round, if one exists.
    pub next_source: Option<ZkCodeSwitchProverSource<EF>>,
}

/// Source-row provider for Construction 9.7 in-domain openings.
///
/// The row provider is the typed boundary that decides how a flattened
/// codeword position becomes `G_C^#` and `G_C^$`. First-round WHIR folded
/// targets have no encoding randomness, while later ZK-encoded sources must
/// provide real randomness rows through `LinearZkEncoding`.
pub trait ZkCodeSwitchSourceLayout<F: Field> {
    /// Message length before source encoding.
    fn message_len(&self) -> usize;

    /// Randomness segment length of the source encoding.
    fn randomness_len(&self) -> usize;

    /// Full source codeword domain size.
    fn domain_size(&self) -> usize;

    /// Row width exponent used by the Merkle layout.
    fn folding_factor(&self) -> usize;

    /// `G_C^#` row at the flattened codeword position.
    fn message_row(&self, position: usize) -> Vec<F>;

    /// `G_C^$` row at the flattened codeword position.
    fn randomness_row(&self, position: usize) -> Vec<F>;
}

/// Row provider for the folded extension oracle committed by WHIR itself.
#[derive(Debug, Clone, Copy)]
pub struct WhirFoldedSourceLayout {
    /// Message length before WHIR row packing.
    pub message_len: usize,
    /// Full encoded domain size.
    pub domain_size: usize,
    /// Row width exponent used by WHIR's Merkle layout.
    pub folding_factor: usize,
}

impl<F: TwoAdicField> ZkCodeSwitchSourceLayout<F> for WhirFoldedSourceLayout {
    fn message_len(&self) -> usize {
        self.message_len
    }

    fn randomness_len(&self) -> usize {
        0
    }

    fn domain_size(&self) -> usize {
        self.domain_size
    }

    fn folding_factor(&self) -> usize {
        self.folding_factor
    }

    fn message_row(&self, position: usize) -> Vec<F> {
        source_message_row(
            self.message_len,
            self.domain_size,
            self.folding_factor,
            position,
        )
    }

    fn randomness_row(&self, _position: usize) -> Vec<F> {
        Vec::new()
    }
}

/// Row provider for a source committed through a linear ZK encoding.
pub struct LinearZkSourceLayout<'a, Enc> {
    /// Source encoding whose generator rows are used for `G_C^#` / `G_C^$`.
    pub encoding: &'a Enc,
}

impl<F, Enc> ZkCodeSwitchSourceLayout<F> for LinearZkSourceLayout<'_, Enc>
where
    F: Field,
    Enc: LinearZkEncoding<F>,
{
    fn message_len(&self) -> usize {
        self.encoding.message_len()
    }

    fn randomness_len(&self) -> usize {
        self.encoding.randomness_len()
    }

    fn domain_size(&self) -> usize {
        self.encoding.codeword_len()
    }

    fn folding_factor(&self) -> usize {
        0
    }

    fn message_row(&self, position: usize) -> Vec<F> {
        self.encoding.message_row(position)
    }

    fn randomness_row(&self, position: usize) -> Vec<F> {
        self.encoding.randomness_row(position)
    }
}

fn source_rows_for_positions<F, L>(layout: &L, positions: &[usize]) -> (Vec<Vec<F>>, Vec<Vec<F>>)
where
    F: Field,
    L: ZkCodeSwitchSourceLayout<F>,
{
    let source_rows = positions
        .iter()
        .map(|&position| {
            let row = layout.message_row(position);
            assert_eq!(row.len(), layout.message_len());
            row
        })
        .collect::<Vec<_>>();
    let source_randomness_rows = positions
        .iter()
        .map(|&position| {
            let row = layout.randomness_row(position);
            assert_eq!(row.len(), layout.randomness_len());
            row
        })
        .collect::<Vec<_>>();
    (source_rows, source_randomness_rows)
}

fn encoded_source_domain_size<EF>(message_len: usize, randomness_len: usize) -> usize
where
    EF: TwoAdicField,
{
    let domain_size = (message_len + randomness_len).next_power_of_two();
    assert!(
        domain_size.ilog2() as usize <= EF::TWO_ADICITY,
        "encoded ZK source domain must fit the extension field two-adicity",
    );
    domain_size
}

fn code_switch_relation_weights<EF>(
    output_relation: CodeSwitchOutputRelation<EF>,
    folding_factor_next: usize,
) -> Vec<EF>
where
    EF: Field,
{
    let mut relation_weights = output_relation.source_covector;
    for covector in output_relation.auxiliary_covectors {
        relation_weights.extend(covector);
    }
    relation_weights.extend(output_relation.mask_covector);
    let relation_len = relation_weights
        .len()
        .next_power_of_two()
        .max(1usize << folding_factor_next);
    relation_weights.resize(relation_len, EF::ZERO);
    relation_weights
}

fn fold_prefix_covector<EF>(relation_weights: Vec<EF>, randomness: &Point<EF>) -> Vec<EF>
where
    EF: Field,
{
    let mut poly = Poly::new(relation_weights);
    for &gamma in randomness {
        poly.fix_prefix_var_mut(gamma);
    }
    poly.as_slice().to_vec()
}

fn source_message_row<F: TwoAdicField>(
    message_len: usize,
    domain_size: usize,
    folding_factor: usize,
    position: usize,
) -> Vec<F> {
    let row_width = 1usize << folding_factor;
    assert!(
        message_len.is_multiple_of(row_width),
        "source message length must be divisible by the WHIR row width",
    );
    let folded_domain_size = domain_size >> folding_factor;
    let row = position / row_width;
    let limb = position % row_width;
    assert!(row < folded_domain_size, "source query row out of range");

    let coeffs_per_limb = message_len / row_width;
    let point = F::two_adic_generator(folded_domain_size.ilog2() as usize).exp_u64(row as u64);
    let mut out = F::zero_vec(message_len);
    let mut power = F::ONE;
    for coeff in 0..coeffs_per_limb {
        out[limb * coeffs_per_limb + coeff] = power;
        power *= point;
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn code_switch_output_relation_from_rows<F, EF>(
    source_message_len: usize,
    source_covector: &[EF],
    auxiliary_covectors: &[Vec<EF>],
    source_randomness_len: usize,
    pad_len: usize,
    rho_ood_points: &[EF],
    source_rows: &[Vec<F>],
    source_randomness_rows: &[Vec<F>],
    claim: &ZkMaskClaim<EF>,
) -> CodeSwitchOutputRelation<EF>
where
    F: TwoAdicField,
    EF: ExtensionField<F>,
{
    assert_eq!(source_covector.len(), source_message_len);
    assert_eq!(rho_ood_points.len(), claim.ood_coeffs.len());
    assert_eq!(source_rows.len(), claim.in_domain_coeffs.len());
    assert_eq!(source_randomness_rows.len(), claim.in_domain_coeffs.len());
    assert!(claim.source_randomness_weights.is_empty());
    assert!(claim.pad_weights.is_empty());

    let mut source_covector_out = source_covector
        .iter()
        .map(|&x| claim.base_claim_coeff * claim.residual_sumcheck_scale * x)
        .collect::<Vec<_>>();
    let auxiliary_covectors_out = auxiliary_covectors
        .iter()
        .map(|covector| {
            covector
                .iter()
                .map(|&x| claim.base_claim_coeff * x)
                .collect()
        })
        .collect();
    let mut mask_covector = EF::zero_vec(source_randomness_len + pad_len);

    for (&rho, &coeff) in rho_ood_points.iter().zip(&claim.ood_coeffs) {
        let mut power = EF::ONE;
        for dst in &mut source_covector_out {
            *dst += coeff * power;
            power *= rho;
        }
        for dst in &mut mask_covector {
            *dst += coeff * power;
            power *= rho;
        }
    }

    for ((message_row, randomness_row), &coeff) in source_rows
        .iter()
        .zip(source_randomness_rows)
        .zip(&claim.in_domain_coeffs)
    {
        assert_eq!(message_row.len(), source_message_len);
        assert_eq!(randomness_row.len(), source_randomness_len);
        for (dst, &entry) in source_covector_out.iter_mut().zip(message_row) {
            *dst += coeff * EF::from(entry);
        }
        for (dst, &entry) in mask_covector
            .iter_mut()
            .take(source_randomness_len)
            .zip(randomness_row)
        {
            *dst += coeff * EF::from(entry);
        }
    }

    CodeSwitchOutputRelation {
        source_covector: source_covector_out,
        auxiliary_covectors: auxiliary_covectors_out,
        mask_covector,
    }
}

#[allow(clippy::too_many_arguments)]
fn code_switch_output_relation_from_ext_rows<EF>(
    source_message_len: usize,
    source_covector: &[EF],
    auxiliary_covectors: &[Vec<EF>],
    source_randomness_len: usize,
    pad_len: usize,
    rho_ood_points: &[EF],
    source_rows: &[Vec<EF>],
    source_randomness_rows: &[Vec<EF>],
    claim: &ZkMaskClaim<EF>,
) -> CodeSwitchOutputRelation<EF>
where
    EF: Field,
{
    assert_eq!(source_covector.len(), source_message_len);
    assert_eq!(rho_ood_points.len(), claim.ood_coeffs.len());
    assert_eq!(source_rows.len(), claim.in_domain_coeffs.len());
    assert_eq!(source_randomness_rows.len(), claim.in_domain_coeffs.len());
    assert!(claim.source_randomness_weights.is_empty());
    assert!(claim.pad_weights.is_empty());

    let mut source_covector_out = source_covector
        .iter()
        .map(|&x| claim.base_claim_coeff * claim.residual_sumcheck_scale * x)
        .collect::<Vec<_>>();
    let auxiliary_covectors_out = auxiliary_covectors
        .iter()
        .map(|covector| {
            covector
                .iter()
                .map(|&x| claim.base_claim_coeff * x)
                .collect()
        })
        .collect();
    let mut mask_covector = EF::zero_vec(source_randomness_len + pad_len);

    for (&rho, &coeff) in rho_ood_points.iter().zip(&claim.ood_coeffs) {
        let mut power = EF::ONE;
        for dst in &mut source_covector_out {
            *dst += coeff * power;
            power *= rho;
        }
        for dst in &mut mask_covector {
            *dst += coeff * power;
            power *= rho;
        }
    }

    for ((message_row, randomness_row), &coeff) in source_rows
        .iter()
        .zip(source_randomness_rows)
        .zip(&claim.in_domain_coeffs)
    {
        assert_eq!(message_row.len(), source_message_len);
        assert_eq!(randomness_row.len(), source_randomness_len);
        for (dst, &entry) in source_covector_out.iter_mut().zip(message_row) {
            *dst += coeff * entry;
        }
        for (dst, &entry) in mask_covector
            .iter_mut()
            .take(source_randomness_len)
            .zip(randomness_row)
        {
            *dst += coeff * entry;
        }
    }

    CodeSwitchOutputRelation {
        source_covector: source_covector_out,
        auxiliary_covectors: auxiliary_covectors_out,
        mask_covector,
    }
}

pub(crate) fn evaluate_zk_mask_residual<F, EF>(masks: &[Vec<F>], gammas: &[EF]) -> EF
where
    F: TwoAdicField,
    EF: ExtensionField<F>,
{
    assert_eq!(masks.len(), gammas.len());
    let k = masks.len();
    if k == 0 {
        return EF::ZERO;
    }

    let mut pow2 = Vec::with_capacity(k + 1);
    let mut power = F::ONE;
    for _ in 0..=k {
        pow2.push(power);
        power *= F::TWO;
    }

    let mut mask_evals_at_gamma = Vec::with_capacity(k);
    let mut sum_future_endpoints: F = masks
        .iter()
        .map(|mask| mask[0].double() + mask[1..].iter().copied().sum::<F>())
        .sum();
    let mut target = EF::ZERO;

    for (round_idx, (s_j, &gamma_j)) in masks.iter().zip(gammas).enumerate() {
        let j = round_idx + 1;
        let s_j_endpoints = s_j[0].double() + s_j[1..].iter().copied().sum::<F>();
        sum_future_endpoints -= s_j_endpoints;

        let h_size = s_j.len().max(3);
        let mut h = EF::zero_vec(h_size);
        let mult_live = pow2[k - j];
        for (i, &c) in s_j.iter().enumerate() {
            h[i] += mult_live * c;
        }

        let past_mask_sum: EF = mask_evals_at_gamma.iter().copied().sum();
        h[0] += past_mask_sum * mult_live;
        if j < k {
            h[0] += EF::from(pow2[k - j - 1] * sum_future_endpoints);
        }

        target = h
            .iter()
            .rev()
            .copied()
            .fold(EF::ZERO, |acc, coeff| acc * gamma_j + coeff);

        let s_j_at_gamma = s_j
            .iter()
            .rev()
            .copied()
            .map(EF::from)
            .fold(EF::ZERO, |acc, coeff| acc * gamma_j + coeff);
        mask_evals_at_gamma.push(s_j_at_gamma);
    }

    target
}

fn zk_mask_residual_covectors<F, EF>(masks: &[Vec<F>], gammas: &[EF]) -> Vec<Vec<EF>>
where
    F: TwoAdicField,
    EF: ExtensionField<F>,
{
    assert!(
        masks
            .iter()
            .all(|mask| mask.len() == masks.first().map_or(0, Vec::len))
    );
    zk_mask_residual_covectors_from_shape(masks.len(), masks.first().map_or(0, Vec::len), gammas)
}

fn zk_mask_residual_covectors_from_shape<F, EF>(
    mask_count: usize,
    mask_len: usize,
    gammas: &[EF],
) -> Vec<Vec<EF>>
where
    F: TwoAdicField,
    EF: ExtensionField<F>,
{
    let mut basis_masks = (0..mask_count)
        .map(|_| F::zero_vec(mask_len))
        .collect::<Vec<_>>();
    let mut covectors = (0..mask_count)
        .map(|_| EF::zero_vec(mask_len))
        .collect::<Vec<_>>();

    for mask_idx in 0..mask_count {
        for coeff_idx in 0..mask_len {
            basis_masks[mask_idx][coeff_idx] = F::ONE;
            covectors[mask_idx][coeff_idx] = evaluate_zk_mask_residual(&basis_masks, gammas);
            basis_masks[mask_idx][coeff_idx] = F::ZERO;
        }
    }

    covectors
}

impl<EF, F, Dft, MT, Challenger> WhirProver<EF, F, Dft, MT, Challenger, PrefixProver<F, EF>>
where
    F: TwoAdicField + Ord,
    EF: ExtensionField<F> + TwoAdicField,
    Dft: TwoAdicSubgroupDft<F>,
    MT: Mmcs<F>,
    Challenger: FieldChallenger<F>
        + GrindingChallenger<Witness = F>
        + CanSampleUniformBits<F>
        + CanObserve<MT::Commitment>,
{
    /// Start the prefix-only ZK WHIR opening flow.
    ///
    /// This records the same initial public opening claims as the plain
    /// `MultilinearPcs::open` path, then runs the #1605 HVZK sumcheck overlay
    /// and returns the typed handoff needed by Construction 9.7.
    ///
    /// The method intentionally stops before the WHIR round loop. The next
    /// implementation step is `round_zk_prefix`, which consumes the returned
    /// handoff and fills each `WhirRoundProof::zk` payload.
    pub fn begin_zk_prefix_open<Enc, R>(
        &self,
        mut prover_data: WhirProverData<F, EF, MT, PrefixProver<F, EF>>,
        protocol: &OpeningProtocol,
        challenger: &mut Challenger,
        encoding: Enc,
        rng: &mut R,
    ) -> WhirZkPrefixOpenState<F, EF, Enc, MT>
    where
        Enc: ZkEncoding<F>,
        Enc::Codeword: Matrix<F>,
        R: Rng,
        StandardUniform: Distribution<F>,
    {
        let zk_config = self
            .config
            .zk
            .as_ref()
            .expect("ZK prefix opening requires WhirConfig::with_zk_config");
        assert!(
            zk_config.only_prefix,
            "ZK WHIR currently supports only prefix layout"
        );
        assert_eq!(
            encoding.message_len(),
            zk_config.mask_message_len,
            "ZK encoding message length must match WhirZkConfig",
        );
        let first_round_zk = self
            .round_parameters
            .first()
            .and_then(|round| round.zk.as_ref())
            .expect("ZK prefix opening requires at least one ZK code-switch round");
        let required_query_bound = first_round_zk.mask_query_budget;
        assert!(
            encoding.query_bound() >= required_query_bound,
            "ZK encoding query bound must cover the derived mask query budget",
        );
        let expected_domain_size = first_round_zk.mask_domain_size;
        assert_eq!(
            encoding.codeword_len(),
            expected_domain_size,
            "ZK encoding codeword length must match the derived mask domain",
        );

        let mut whir_proof = self.config.empty_proof();
        tracing::info_span!("zk prefix ood claims").in_scope(|| {
            whir_proof.initial_ood_answers = (0..self.commitment_ood_samples)
                .map(|_| prover_data.layout.add_virtual_eval(challenger))
                .collect::<Vec<_>>();
        });

        let evals = protocol
            .iter_openings()
            .map(|(table_idx, polys)| prover_data.layout.eval(table_idx, polys, challenger))
            .collect::<Vec<_>>();

        let mut zk_sumcheck = ZkSumcheckData::<F, EF>::default();
        let zk_prover = ZkPrefixProver::new(prover_data.layout, encoding, self.mmcs.clone());
        let initial_handoff = zk_prover.into_sumcheck(
            &mut zk_sumcheck,
            self.starting_folding_pow_bits,
            challenger,
            rng,
        );
        whir_proof.initial_zk = Some(WhirInitialZkProof {
            zk_sumcheck,
            zk_sumcheck_mask_commitments: initial_handoff
                .mask_oracles
                .iter()
                .map(|(commitment, _)| commitment.clone())
                .collect(),
        });

        WhirZkPrefixOpenState {
            proof: PcsProof {
                whir: whir_proof,
                evals,
            },
            initial_handoff,
            source_merkle_data: prover_data.merkle_data,
        }
    }

    /// Prove one prefix-only ZK code-switching round.
    ///
    /// This is the first real Construction 9.7 wiring point: it consumes the
    /// typed Construction 6.3 handoff, commits the folded target oracle, commits
    /// a fresh mask oracle, records private OOD answers and source/mask openings,
    /// derives the `mu'` relation, and runs the next HVZK residual sumcheck.
    #[allow(clippy::too_many_lines)]
    pub fn round_zk_prefix<Enc, R>(
        &self,
        state: WhirZkPrefixOpenState<F, EF, Enc, MT>,
        round_index: usize,
        source_covector: &[EF],
        mask_encoding: &Enc,
        challenger: &mut Challenger,
        rng: &mut R,
    ) -> WhirZkPrefixRoundState<F, EF, Enc, MT>
    where
        Enc: ZkEncoding<F>,
        Enc::Codeword: Matrix<F>,
        R: Rng,
        StandardUniform: Distribution<F>,
    {
        assert_eq!(
            round_index, 0,
            "first ZK round helper currently handles round 0"
        );
        let handoff = state.initial_handoff;
        let source_message = handoff.residual_prover.evals().as_slice().to_vec();
        let num_variables =
            self.num_variables - self.params.folding_factor.total_number(round_index);
        let folding_factor_next = self.params.folding_factor.at_round(round_index + 1);
        let target_domain_size = self.inv_rate(round_index) * (1usize << num_variables);
        let source = ZkCodeSwitchProverSource {
            message: source_message,
            covector: source_covector.to_vec(),
            inherited_claim: handoff.residual_prover.claimed_sum(),
            residual_sumcheck_scale: handoff.eps,
            randomness_len: 0,
            domain_size: target_domain_size,
            folding_factor: folding_factor_next,
        };
        self.round0_zk_prefix_from_folded_source(
            state.proof,
            &handoff,
            &source,
            mask_encoding,
            challenger,
            rng,
        )
    }

    /// Prove one prefix-only ZK code-switching round from a folded WHIR source.
    ///
    /// This is the shared first folded-source Construction 9.7 consumer. It is
    /// intentionally not the later multi-round source consumer yet: the source
    /// commitment and in-domain openings below are still the freshly committed
    /// WHIR extension oracle, so accepting a linear-ZK source layout here would
    /// mix row semantics with the wrong Merkle opening semantics.
    #[allow(clippy::too_many_lines, clippy::too_many_arguments)]
    pub fn round0_zk_prefix_from_folded_source<Enc, R>(
        &self,
        mut proof: PcsProof<F, EF, MT>,
        handoff: &ZkSumcheckHandoff<F, EF, Enc, MT>,
        source: &ZkCodeSwitchProverSource<EF>,
        mask_encoding: &Enc,
        challenger: &mut Challenger,
        rng: &mut R,
    ) -> WhirZkPrefixRoundState<F, EF, Enc, MT>
    where
        Enc: ZkEncoding<F>,
        Enc::Codeword: Matrix<F>,
        R: Rng,
        StandardUniform: Distribution<F>,
    {
        let round_index = 0;
        assert!(
            self.config.zk.as_ref().is_some_and(|zk| zk.only_prefix),
            "ZK WHIR currently supports only prefix layout",
        );

        let round_params = &self.round_parameters[round_index];
        let round_zk = round_params
            .zk
            .as_ref()
            .expect("round0_zk_prefix_from_folded_source requires RoundConfig::zk");
        assert_eq!(
            mask_encoding.message_len(),
            round_zk.mask_message_len,
            "mask encoding message length must match the round ZK config",
        );
        assert!(
            mask_encoding.query_bound() >= round_zk.mask_query_budget,
            "mask encoding query bound must cover the round mask query budget",
        );
        assert_eq!(
            mask_encoding.codeword_len(),
            round_zk.mask_domain_size,
            "mask encoding domain must match the round ZK config",
        );

        let folded_evaluations = Poly::new(source.message.clone());
        let num_variables = folded_evaluations.num_variables();
        let folding_factor_next = source.folding_factor;
        assert_eq!(source.domain_size % (1usize << num_variables), 0);
        let inv_rate = source.domain_size >> num_variables;
        assert!(inv_rate > 0, "source domain must cover the source message");
        assert_eq!(
            source.randomness_len, 0,
            "ZK code-switch source randomness needs a source-oracle randomness handoff",
        );
        let source_layout = WhirFoldedSourceLayout {
            message_len: source.message.len(),
            domain_size: source.domain_size,
            folding_factor: source.folding_factor,
        };
        assert_eq!(
            source.message.len(),
            source.covector.len(),
            "source message and covector length must match",
        );
        assert_eq!(
            source.inherited_claim,
            handoff.residual_prover.claimed_sum(),
            "source handoff inherited claim must match the #1605 residual prover claim",
        );
        assert_eq!(
            source.inherited_claim,
            source.residual_sumcheck_scale
                * dot_product::<EF, _, _>(
                    source.message.iter().copied(),
                    source.covector.iter().copied(),
                ),
            "source handoff inherited claim must match the configured residual scale",
        );

        let (target_root, target_merkle_data) = commit_extension(
            crate::sumcheck::strategy::VariableOrder::Prefix,
            &self.dft,
            &self.extension_mmcs,
            &folded_evaluations,
            folding_factor_next,
            inv_rate,
        );
        challenger.observe(target_root.clone());
        proof.whir.rounds[round_index].commitment = Some(target_root);

        assert!(
            source.randomness_len <= round_zk.mask_message_len,
            "source randomness must fit in the round mask message",
        );
        let pad_len = round_zk.mask_message_len - source.randomness_len;
        let mask_message = (0..round_zk.mask_message_len)
            .map(|_| rng.random())
            .collect::<Vec<F>>();
        let mask_codeword = mask_encoding.encode(&mask_message, rng);
        let (mask_commitment, mask_prover_data) = self.mmcs.commit_matrix(mask_codeword);
        challenger.observe(mask_commitment.clone());

        let rho_ood_points = (0..round_zk.ood_samples)
            .map(|_| challenger.sample_algebra_element())
            .collect::<Vec<EF>>();
        let source_message = source.message.clone();
        let mask_message_ext = mask_message
            .iter()
            .copied()
            .map(EF::from)
            .collect::<Vec<_>>();
        let private_ood_answers =
            private_ood_answers(&rho_ood_points, &source_message, &mask_message_ext);
        challenger.observe_algebra_slice(&private_ood_answers);

        if round_params.pow_bits > 0 {
            proof.whir.rounds[round_index].pow_witness = challenger.grind(round_params.pow_bits);
        }
        challenger.sample();

        let row_indices = get_challenge_stir_queries::<Challenger, F, EF>(
            source.domain_size,
            source.folding_factor,
            round_zk.mask_query_budget,
            challenger,
        );
        let row_width = 1usize << source.folding_factor;
        let mut source_queries = Vec::with_capacity(row_indices.len());
        let mut source_openings = Vec::with_capacity(row_indices.len() * row_width);
        let mut query_positions = Vec::with_capacity(row_indices.len() * row_width);
        for &row in &row_indices {
            let opening = self.extension_mmcs.open_batch(row, &target_merkle_data);
            let values = opening.opened_values[0].clone();
            for (limb, &value) in values.iter().enumerate() {
                source_openings.push(value);
                query_positions.push(row * row_width + limb);
            }
            source_queries.push(QueryOpening::Extension {
                values,
                proof: opening.opening_proof,
            });
        }

        let mask_indices = get_challenge_stir_queries::<Challenger, F, EF>(
            round_zk.mask_domain_size,
            0,
            round_zk.mask_query_budget,
            challenger,
        );
        let mut mask_queries = Vec::with_capacity(mask_indices.len());
        for &position in &mask_indices {
            let opening = self.mmcs.open_batch(position, &mask_prover_data);
            mask_queries.push(QueryOpening::Base {
                values: opening.opened_values[0].clone(),
                proof: opening.opening_proof,
            });
        }

        let batching_dim = 1 + private_ood_answers.len() + source_openings.len();
        let batching_challenge = challenger.sample_algebra_element();
        let coeffs = batching_coefficients(batching_challenge, batching_dim);
        let claim = ZkMaskClaim {
            base_claim_coeff: coeffs[0],
            residual_sumcheck_scale: source.residual_sumcheck_scale,
            ood_coeffs: coeffs[1..1 + private_ood_answers.len()].to_vec(),
            in_domain_coeffs: coeffs[1 + private_ood_answers.len()..].to_vec(),
            source_randomness_weights: Vec::new(),
            pad_weights: Vec::new(),
        };
        let initial_gammas = handoff.randomness.iter().copied().collect::<Vec<_>>();
        let auxiliary_covectors =
            zk_mask_residual_covectors::<F, EF>(&handoff.mask_messages, &initial_gammas);
        let auxiliary_claim = handoff
            .mask_messages
            .iter()
            .zip(&auxiliary_covectors)
            .map(|(message, covector)| {
                dot_product::<EF, _, _>(
                    message.iter().copied().map(EF::from),
                    covector.iter().copied(),
                )
            })
            .sum::<EF>();
        let inherited_claim = handoff.residual_prover.claimed_sum() + auxiliary_claim;
        let mu_prime = batched_claim(
            inherited_claim,
            &private_ood_answers,
            &source_openings,
            &claim,
        )
        .expect("honest code-switch batching dimensions should match");
        let (source_rows, source_randomness_rows) =
            source_rows_for_positions::<F, _>(&source_layout, &query_positions);
        let output_relation = code_switch_output_relation_from_rows(
            source.message.len(),
            &source.covector,
            &auxiliary_covectors,
            source.randomness_len,
            pad_len,
            &rho_ood_points,
            &source_rows,
            &source_randomness_rows,
            &claim,
        );

        let mut relation_evals = source_message;
        for message in &handoff.mask_messages {
            relation_evals.extend(message.iter().copied().map(EF::from));
        }
        relation_evals.extend(mask_message_ext.iter().copied());
        let mut relation_weights = output_relation.source_covector;
        for covector in output_relation.auxiliary_covectors {
            relation_weights.extend(covector);
        }
        relation_weights.extend(output_relation.mask_covector);
        let relation_len = relation_evals
            .len()
            .next_power_of_two()
            .max(1usize << folding_factor_next);
        relation_evals.resize(relation_len, EF::ZERO);
        relation_weights.resize(relation_len, EF::ZERO);
        debug_assert_eq!(
            dot_product::<EF, _, _>(
                relation_evals.iter().copied(),
                relation_weights.iter().copied(),
            ),
            mu_prime,
            "code-switch output relation must evaluate to mu'",
        );
        let relation_poly = ProductPolynomial::new_unpacked(
            VariableOrder::Prefix,
            Poly::new(relation_evals),
            Poly::new(relation_weights),
        );
        let relation_prover = SumcheckProver::new(relation_poly, mu_prime);

        let mut zk_sumcheck = ZkSumcheckData::default();
        let next_handoff = relation_prover.into_zk_sumcheck(
            &mut zk_sumcheck,
            mask_encoding,
            &self.mmcs,
            folding_factor_next,
            round_params.folding_pow_bits,
            challenger,
            rng,
        );
        let next_source = if round_index + 1 < self.n_rounds() {
            let next_round_index = round_index + 1;
            let next_round_zk = self.round_parameters[next_round_index]
                .zk
                .as_ref()
                .expect("next ZK round source requires RoundConfig::zk");
            let next_message = next_handoff.residual_prover.evals().as_slice().to_vec();
            let next_covector = next_handoff.residual_prover.weights().as_slice().to_vec();
            assert_eq!(next_message.len(), next_covector.len());
            let next_randomness_len = next_round_zk.mask_query_budget;
            Some(ZkCodeSwitchProverSource {
                domain_size: encoded_source_domain_size::<EF>(
                    next_message.len(),
                    next_randomness_len,
                ),
                message: next_message,
                covector: next_covector,
                inherited_claim: next_handoff.residual_prover.claimed_sum(),
                residual_sumcheck_scale: EF::ONE,
                randomness_len: next_randomness_len,
                folding_factor: 0,
            })
        } else {
            None
        };
        proof.whir.rounds[round_index].zk = Some(WhirRoundZkProof {
            mask_commitment,
            private_ood_answers,
            source_queries,
            mask_queries,
            zk_sumcheck,
            zk_sumcheck_mask_commitments: next_handoff
                .mask_oracles
                .iter()
                .map(|(commitment, _)| commitment.clone())
                .collect(),
        });

        WhirZkPrefixRoundState {
            proof,
            handoff: next_handoff,
            target_merkle_data,
            next_source,
        }
    }

    /// Prove one prefix-only ZK code-switching round from a linearly ZK-encoded `EF` source.
    ///
    /// This is the multi-round consumer boundary: unlike the round-0 folded
    /// helper, the source commitment, source openings, and generator rows all
    /// come from `source_encoding`. The fresh code-switch mask is also an
    /// `EF` oracle so it can carry the source encoding randomness in its message
    /// prefix. The nested #1605 sumcheck still uses the base-field
    /// `sumcheck_mask_encoding`.
    #[allow(clippy::too_many_lines, clippy::too_many_arguments)]
    pub fn round_zk_prefix_from_encoded_source<SumcheckEnc, SourceEnc, CodeSwitchMaskEnc, R>(
        &self,
        mut proof: PcsProof<F, EF, MT>,
        handoff: &ZkSumcheckHandoff<F, EF, SumcheckEnc, MT>,
        round_index: usize,
        source: &ZkCodeSwitchProverSource<EF>,
        source_encoding: &SourceEnc,
        code_switch_mask_encoding: &CodeSwitchMaskEnc,
        sumcheck_mask_encoding: &SumcheckEnc,
        challenger: &mut Challenger,
        rng: &mut R,
    ) -> WhirZkPrefixEncodedRoundState<F, EF, SumcheckEnc, MT>
    where
        SumcheckEnc: ZkEncoding<F>,
        SumcheckEnc::Codeword: Matrix<F>,
        SourceEnc: LinearZkEncoding<EF> + ZkEncodingWithRandomness<EF>,
        SourceEnc::Codeword: Matrix<EF>,
        CodeSwitchMaskEnc: ZkEncoding<EF>,
        CodeSwitchMaskEnc::Codeword: Matrix<EF>,
        R: Rng,
        StandardUniform: Distribution<F> + Distribution<EF>,
    {
        assert!(
            round_index > 0,
            "encoded ZK source consumer starts after round 0"
        );
        assert!(
            self.config.zk.as_ref().is_some_and(|zk| zk.only_prefix),
            "ZK WHIR currently supports only prefix layout",
        );

        let round_params = &self.round_parameters[round_index];
        let round_zk = round_params
            .zk
            .as_ref()
            .expect("round_zk_prefix_from_encoded_source requires RoundConfig::zk");
        assert_eq!(
            source_encoding.message_len(),
            source.message.len(),
            "source encoding message length must match the carried source",
        );
        assert_eq!(
            source_encoding.codeword_len(),
            source.domain_size,
            "source encoding domain must match the carried source",
        );
        assert!(
            source_encoding.query_bound() >= round_zk.mask_query_budget,
            "source encoding query bound must cover the round source openings",
        );
        assert_eq!(
            source.folding_factor, 0,
            "linearly ZK-encoded source openings are single-position rows",
        );
        let source_randomness_len = source_encoding.randomness_len();
        assert_eq!(
            source.randomness_len, source_randomness_len,
            "carried source randomness length must match the source encoding",
        );
        assert_eq!(
            code_switch_mask_encoding.message_len(),
            round_zk.mask_message_len,
            "code-switch mask message length must match the round ZK config",
        );
        assert!(
            code_switch_mask_encoding.query_bound() >= round_zk.mask_query_budget,
            "code-switch mask query bound must cover the round mask query budget",
        );
        assert_eq!(
            code_switch_mask_encoding.codeword_len(),
            round_zk.mask_domain_size,
            "code-switch mask domain must match the round ZK config",
        );
        assert_eq!(
            sumcheck_mask_encoding.message_len(),
            round_zk.mask_message_len,
            "nested sumcheck mask message length must match the round ZK config",
        );
        assert!(
            sumcheck_mask_encoding.query_bound() >= round_zk.mask_query_budget,
            "nested sumcheck mask query bound must cover the round mask query budget",
        );
        assert_eq!(
            sumcheck_mask_encoding.codeword_len(),
            round_zk.mask_domain_size,
            "nested sumcheck mask domain must match the round ZK config",
        );
        assert_eq!(
            source.message.len(),
            source.covector.len(),
            "source message and covector length must match",
        );
        assert_eq!(
            source.inherited_claim,
            handoff.residual_prover.claimed_sum(),
            "source handoff inherited claim must match the #1605 residual prover claim",
        );
        assert_eq!(
            source.inherited_claim,
            source.residual_sumcheck_scale
                * dot_product::<EF, _, _>(
                    source.message.iter().copied(),
                    source.covector.iter().copied(),
                ),
            "source handoff inherited claim must match the configured residual scale",
        );

        let source_randomness = (0..source_randomness_len)
            .map(|_| rng.random())
            .collect::<Vec<EF>>();
        let source_codeword =
            source_encoding.encode_with_randomness(&source.message, &source_randomness);
        let (source_commitment, source_prover_data) =
            self.extension_mmcs.commit_matrix(source_codeword);
        challenger.observe(source_commitment.clone());
        proof.whir.rounds[round_index].commitment = Some(source_commitment.clone());

        assert!(
            source_randomness_len <= round_zk.mask_message_len,
            "source randomness must fit in the round mask message",
        );
        let pad_len = round_zk.mask_message_len - source_randomness_len;
        let mut mask_message = source_randomness;
        mask_message.extend((0..pad_len).map(|_| rng.random::<EF>()));
        let mask_codeword = code_switch_mask_encoding.encode(&mask_message, rng);
        let (mask_commitment, mask_prover_data) = self.extension_mmcs.commit_matrix(mask_codeword);
        challenger.observe(mask_commitment.clone());

        let rho_ood_points = (0..round_zk.ood_samples)
            .map(|_| challenger.sample_algebra_element())
            .collect::<Vec<EF>>();
        let source_message = source.message.clone();
        let private_ood_answers =
            private_ood_answers(&rho_ood_points, &source_message, &mask_message);
        challenger.observe_algebra_slice(&private_ood_answers);

        if round_params.pow_bits > 0 {
            proof.whir.rounds[round_index].pow_witness = challenger.grind(round_params.pow_bits);
        }
        challenger.sample();

        let source_positions = get_challenge_stir_queries::<Challenger, F, EF>(
            source.domain_size,
            0,
            round_zk.mask_query_budget,
            challenger,
        );
        let mut source_queries = Vec::with_capacity(source_positions.len());
        let mut source_openings = Vec::with_capacity(source_positions.len());
        for &position in &source_positions {
            let opening = self
                .extension_mmcs
                .open_batch(position, &source_prover_data);
            let values = opening.opened_values[0].clone();
            debug_assert_eq!(values.len(), 1);
            source_openings.push(values[0]);
            source_queries.push(QueryOpening::Extension {
                values,
                proof: opening.opening_proof,
            });
        }

        let mask_indices = get_challenge_stir_queries::<Challenger, F, EF>(
            round_zk.mask_domain_size,
            0,
            round_zk.mask_query_budget,
            challenger,
        );
        let mut mask_queries = Vec::with_capacity(mask_indices.len());
        for &position in &mask_indices {
            let opening = self.extension_mmcs.open_batch(position, &mask_prover_data);
            mask_queries.push(QueryOpening::Extension {
                values: opening.opened_values[0].clone(),
                proof: opening.opening_proof,
            });
        }

        let batching_dim = 1 + private_ood_answers.len() + source_openings.len();
        let batching_challenge = challenger.sample_algebra_element();
        let coeffs = batching_coefficients(batching_challenge, batching_dim);
        let claim = ZkMaskClaim {
            base_claim_coeff: coeffs[0],
            residual_sumcheck_scale: source.residual_sumcheck_scale,
            ood_coeffs: coeffs[1..1 + private_ood_answers.len()].to_vec(),
            in_domain_coeffs: coeffs[1 + private_ood_answers.len()..].to_vec(),
            source_randomness_weights: Vec::new(),
            pad_weights: Vec::new(),
        };
        let gammas = handoff.randomness.iter().copied().collect::<Vec<_>>();
        let auxiliary_covectors =
            zk_mask_residual_covectors::<F, EF>(&handoff.mask_messages, &gammas);
        let auxiliary_claim = handoff
            .mask_messages
            .iter()
            .zip(&auxiliary_covectors)
            .map(|(message, covector)| {
                dot_product::<EF, _, _>(
                    message.iter().copied().map(EF::from),
                    covector.iter().copied(),
                )
            })
            .sum::<EF>();
        let inherited_claim = handoff.residual_prover.claimed_sum() + auxiliary_claim;
        let mu_prime = batched_claim(
            inherited_claim,
            &private_ood_answers,
            &source_openings,
            &claim,
        )
        .expect("honest encoded code-switch batching dimensions should match");

        let source_layout = LinearZkSourceLayout {
            encoding: source_encoding,
        };
        let (source_rows, source_randomness_rows) =
            source_rows_for_positions::<EF, _>(&source_layout, &source_positions);
        let output_relation = code_switch_output_relation_from_ext_rows(
            source.message.len(),
            &source.covector,
            &auxiliary_covectors,
            source_randomness_len,
            pad_len,
            &rho_ood_points,
            &source_rows,
            &source_randomness_rows,
            &claim,
        );

        let mut relation_evals = source_message;
        for message in &handoff.mask_messages {
            relation_evals.extend(message.iter().copied().map(EF::from));
        }
        relation_evals.extend(mask_message.iter().copied());
        let mut relation_weights = output_relation.source_covector;
        for covector in output_relation.auxiliary_covectors {
            relation_weights.extend(covector);
        }
        relation_weights.extend(output_relation.mask_covector);
        let folding_factor_next = self.params.folding_factor.at_round(round_index + 1);
        let relation_len = relation_evals
            .len()
            .next_power_of_two()
            .max(1usize << folding_factor_next);
        relation_evals.resize(relation_len, EF::ZERO);
        relation_weights.resize(relation_len, EF::ZERO);
        debug_assert_eq!(
            dot_product::<EF, _, _>(
                relation_evals.iter().copied(),
                relation_weights.iter().copied(),
            ),
            mu_prime,
            "encoded code-switch output relation must evaluate to mu'",
        );
        let relation_poly = ProductPolynomial::new_unpacked(
            VariableOrder::Prefix,
            Poly::new(relation_evals),
            Poly::new(relation_weights),
        );
        let relation_prover = SumcheckProver::new(relation_poly, mu_prime);

        let mut zk_sumcheck = ZkSumcheckData::default();
        let next_handoff = relation_prover.into_zk_sumcheck(
            &mut zk_sumcheck,
            sumcheck_mask_encoding,
            &self.mmcs,
            folding_factor_next,
            round_params.folding_pow_bits,
            challenger,
            rng,
        );
        let next_source = if round_index + 1 < self.n_rounds() {
            let next_round_index = round_index + 1;
            let next_round_zk = self.round_parameters[next_round_index]
                .zk
                .as_ref()
                .expect("next ZK round source requires RoundConfig::zk");
            let next_message = next_handoff.residual_prover.evals().as_slice().to_vec();
            let next_covector = next_handoff.residual_prover.weights().as_slice().to_vec();
            assert_eq!(next_message.len(), next_covector.len());
            let next_randomness_len = next_round_zk.mask_query_budget;
            Some(ZkCodeSwitchProverSource {
                domain_size: encoded_source_domain_size::<EF>(
                    next_message.len(),
                    next_randomness_len,
                ),
                message: next_message,
                covector: next_covector,
                inherited_claim: next_handoff.residual_prover.claimed_sum(),
                residual_sumcheck_scale: EF::ONE,
                randomness_len: next_randomness_len,
                folding_factor: 0,
            })
        } else {
            None
        };
        proof.whir.rounds[round_index].zk = Some(WhirRoundZkProof {
            mask_commitment,
            private_ood_answers,
            source_queries,
            mask_queries,
            zk_sumcheck,
            zk_sumcheck_mask_commitments: next_handoff
                .mask_oracles
                .iter()
                .map(|(commitment, _)| commitment.clone())
                .collect(),
        });

        WhirZkPrefixEncodedRoundState {
            proof,
            handoff: next_handoff,
            source_commitment,
            next_source,
        }
    }

    /// Run all prefix-only ZK code-switching rounds and return the final ZK handoff.
    ///
    /// This is the loop-level Construction 9.7 driver. The returned handoff is
    /// the output implicit relation that the paper composes into the base-case
    /// IOPP. It deliberately does not route through the existing plain WHIR
    /// final-polynomial tail, because that tail checks an unmasked residual
    /// polynomial while the ZK handoff remains masked by Construction 6.3.
    #[allow(clippy::too_many_arguments)]
    pub fn prove_zk_prefix_rounds<
        SumcheckEnc,
        SourceEnc,
        CodeSwitchMaskEnc,
        MakeSumcheckEnc,
        MakeSourceEnc,
        MakeCodeSwitchMaskEnc,
        R,
    >(
        &self,
        state: WhirZkPrefixOpenState<F, EF, SumcheckEnc, MT>,
        source_covector: &[EF],
        mut sumcheck_mask_encoding_for: MakeSumcheckEnc,
        mut source_encoding_for: MakeSourceEnc,
        mut code_switch_mask_encoding_for: MakeCodeSwitchMaskEnc,
        challenger: &mut Challenger,
        rng: &mut R,
    ) -> WhirZkPrefixRoundsState<F, EF, SumcheckEnc, MT>
    where
        SumcheckEnc: ZkEncoding<F>,
        SumcheckEnc::Codeword: Matrix<F>,
        SourceEnc: LinearZkEncoding<EF> + ZkEncodingWithRandomness<EF>,
        SourceEnc::Codeword: Matrix<EF>,
        CodeSwitchMaskEnc: ZkEncoding<EF>,
        CodeSwitchMaskEnc::Codeword: Matrix<EF>,
        MakeSumcheckEnc: FnMut(usize) -> SumcheckEnc,
        MakeSourceEnc: FnMut(usize, &ZkCodeSwitchProverSource<EF>) -> SourceEnc,
        MakeCodeSwitchMaskEnc: FnMut(usize) -> CodeSwitchMaskEnc,
        R: Rng,
        StandardUniform: Distribution<F> + Distribution<EF>,
    {
        assert!(
            self.n_rounds() > 0,
            "ZK prefix round loop requires at least one code-switch round",
        );

        let round0_sumcheck_mask_encoding = sumcheck_mask_encoding_for(0);
        let first_round = self.round_zk_prefix(
            state,
            0,
            source_covector,
            &round0_sumcheck_mask_encoding,
            challenger,
            rng,
        );
        let mut proof = first_round.proof;
        let mut handoff = first_round.handoff;
        let mut next_source = first_round.next_source;

        for round_index in 1..self.n_rounds() {
            let source = next_source
                .take()
                .expect("previous ZK round must carry the next encoded source");
            let source_encoding = source_encoding_for(round_index, &source);
            let sumcheck_mask_encoding = sumcheck_mask_encoding_for(round_index);
            let code_switch_mask_encoding = code_switch_mask_encoding_for(round_index);
            let round_state = self.round_zk_prefix_from_encoded_source(
                proof,
                &handoff,
                round_index,
                &source,
                &source_encoding,
                &code_switch_mask_encoding,
                &sumcheck_mask_encoding,
                challenger,
                rng,
            );
            proof = round_state.proof;
            handoff = round_state.handoff;
            next_source = round_state.next_source;
        }

        WhirZkPrefixRoundsState {
            proof,
            handoff,
            next_source,
        }
    }

    /// Replay the initial ZK handoff and the first prefix-only ZK code-switch round.
    ///
    /// This verifier helper is intentionally scoped to the dedicated ZK API. It
    /// does not route through the plain PCS verifier, and it checks the source
    /// and mask openings carried by `WhirRoundZkProof`.
    #[allow(clippy::too_many_lines)]
    pub fn verify_round_zk_prefix(
        &self,
        proof: &PcsProof<F, EF, MT>,
        protocol: &OpeningProtocol,
        source: &ZkCodeSwitchVerifierSource<F, EF, MT>,
        challenger: &mut Challenger,
    ) -> Result<WhirZkPrefixVerifierRoundState<EF>, VerifierError> {
        let zk_config = self
            .config
            .zk
            .as_ref()
            .ok_or(VerifierError::ZkVerifierRequiresPrefixPath)?;
        assert!(
            zk_config.only_prefix,
            "ZK WHIR currently supports only prefix layout"
        );

        challenger.observe(source.commitment.clone());

        let mut initial_verifier = ZkVerifier::<F, EF>::new(&protocol.table_shapes());
        for &eval in &proof.whir.initial_ood_answers {
            initial_verifier.add_virtual_eval(eval, challenger);
        }
        if protocol.num_openings() != proof.evals.len() {
            return Err(VerifierError::OpeningBatchCountMismatch {
                expected: protocol.num_openings(),
                actual: proof.evals.len(),
            });
        }
        for ((table_idx, polys), evals) in protocol.iter_openings().zip(&proof.evals) {
            if polys.len() != evals.len() {
                return Err(VerifierError::OpeningBatchSizeMismatch {
                    table_idx,
                    expected: polys.len(),
                    actual: evals.len(),
                });
            }
            initial_verifier.add_claim(table_idx, polys, evals, challenger);
        }

        let initial_zk = proof
            .whir
            .initial_zk
            .as_ref()
            .ok_or(VerifierError::UnexpectedInitialZkPayloadInPlainProof)?;
        let initial_handoff = initial_verifier.into_sumcheck::<MT, _>(
            &initial_zk.zk_sumcheck,
            &initial_zk.zk_sumcheck_mask_commitments,
            zk_config.mask_message_len,
            self.params.folding_factor.at_round(0),
            self.starting_folding_pow_bits,
            challenger,
        )?;

        let round_index = 0;
        let round_params = &self.round_parameters[round_index];
        let round_zk = round_params
            .zk
            .as_ref()
            .expect("verify_round_zk_prefix requires RoundConfig::zk");
        let num_variables =
            self.num_variables - self.params.folding_factor.total_number(round_index);
        let target_domain_size = self.inv_rate(round_index) * (1usize << num_variables);
        assert_eq!(source.domain_size, target_domain_size);
        assert_eq!(
            source.folding_factor,
            self.params.folding_factor.at_round(round_index + 1),
        );
        assert_eq!(
            source.randomness_len, 0,
            "first ZK code-switch round only supports the target extension oracle; \
             nonzero source randomness needs an explicit randomness-row handoff",
        );
        let source_layout = WhirFoldedSourceLayout {
            message_len: source.message_len,
            domain_size: source.domain_size,
            folding_factor: source.folding_factor,
        };
        let round = proof
            .whir
            .rounds
            .get(round_index)
            .ok_or(VerifierError::InvalidRoundIndex { index: round_index })?;
        let target_commitment = round
            .commitment
            .clone()
            .ok_or(VerifierError::MissingRoundCommitment { round: round_index })?;
        challenger.observe(target_commitment.clone());

        let round_zk_proof = round
            .zk
            .as_ref()
            .ok_or(VerifierError::UnexpectedZkPayloadInPlainProof { round: 0 })?;
        challenger.observe(round_zk_proof.mask_commitment.clone());

        let rho_ood_points = (0..round_zk.ood_samples)
            .map(|_| challenger.sample_algebra_element())
            .collect::<Vec<EF>>();
        if round_zk_proof.private_ood_answers.len() != rho_ood_points.len() {
            return Err(VerifierError::StirQueryCountMismatch {
                round_index,
                expected: rho_ood_points.len(),
                actual: round_zk_proof.private_ood_answers.len(),
            });
        }
        challenger.observe_algebra_slice(&round_zk_proof.private_ood_answers);

        if round_params.pow_bits > 0
            && !challenger.check_witness(round_params.pow_bits, round.pow_witness)
        {
            return Err(VerifierError::InvalidPowWitness);
        }
        challenger.sample();

        let row_indices = get_challenge_stir_queries::<Challenger, F, EF>(
            source.domain_size,
            source.folding_factor,
            round_zk.mask_query_budget,
            challenger,
        );
        if round_zk_proof.source_queries.len() != row_indices.len() {
            return Err(VerifierError::StirQueryCountMismatch {
                round_index,
                expected: row_indices.len(),
                actual: round_zk_proof.source_queries.len(),
            });
        }
        let row_width = 1usize << source.folding_factor;
        let source_dimensions = [Dimensions {
            height: source.domain_size >> source.folding_factor,
            width: row_width,
        }];
        let mut source_openings = Vec::with_capacity(row_indices.len() * row_width);
        let mut query_positions = Vec::with_capacity(row_indices.len() * row_width);
        for (&row, query) in row_indices.iter().zip(&round_zk_proof.source_queries) {
            let QueryOpening::Extension { values, proof } = query else {
                return Err(VerifierError::MerkleProofInvalid {
                    position: row,
                    reason: "Expected extension-field target opening in first ZK round".into(),
                });
            };
            self.extension_mmcs
                .verify_batch(
                    &target_commitment,
                    &source_dimensions,
                    row,
                    BatchOpeningRef {
                        opened_values: from_ref(values),
                        opening_proof: proof,
                    },
                )
                .map_err(|_| VerifierError::MerkleProofInvalid {
                    position: row,
                    reason: "ZK target opening verification failed".into(),
                })?;
            for (limb, &value) in values.iter().enumerate() {
                source_openings.push(value);
                query_positions.push(row * row_width + limb);
            }
        }

        let mask_indices = get_challenge_stir_queries::<Challenger, F, EF>(
            round_zk.mask_domain_size,
            0,
            round_zk.mask_query_budget,
            challenger,
        );
        if round_zk_proof.mask_queries.len() != mask_indices.len() {
            return Err(VerifierError::StirQueryCountMismatch {
                round_index,
                expected: mask_indices.len(),
                actual: round_zk_proof.mask_queries.len(),
            });
        }
        let mask_dimensions = [Dimensions {
            height: round_zk.mask_domain_size,
            width: 1,
        }];
        for (&position, query) in mask_indices.iter().zip(&round_zk_proof.mask_queries) {
            let QueryOpening::Base { values, proof } = query else {
                return Err(VerifierError::MerkleProofInvalid {
                    position,
                    reason: "Expected base-field mask opening".into(),
                });
            };
            self.mmcs
                .verify_batch(
                    &round_zk_proof.mask_commitment,
                    &mask_dimensions,
                    position,
                    BatchOpeningRef {
                        opened_values: from_ref(values),
                        opening_proof: proof,
                    },
                )
                .map_err(|_| VerifierError::MerkleProofInvalid {
                    position,
                    reason: "ZK mask opening verification failed".into(),
                })?;
        }

        let batching_dim = 1 + round_zk_proof.private_ood_answers.len() + source_openings.len();
        let batching_challenge = challenger.sample_algebra_element();
        let coeffs = batching_coefficients(batching_challenge, batching_dim);
        let claim = ZkMaskClaim {
            base_claim_coeff: coeffs[0],
            residual_sumcheck_scale: source.residual_sumcheck_scale,
            ood_coeffs: coeffs[1..1 + round_zk_proof.private_ood_answers.len()].to_vec(),
            in_domain_coeffs: coeffs[1 + round_zk_proof.private_ood_answers.len()..].to_vec(),
            source_randomness_weights: Vec::new(),
            pad_weights: Vec::new(),
        };
        let mu_prime = batched_claim(
            initial_handoff.claimed_residual,
            &round_zk_proof.private_ood_answers,
            &source_openings,
            &claim,
        )
        .map_err(|err| VerifierError::MerkleProofInvalid {
            position: 0,
            reason: err.to_string(),
        })?;
        let pad_len = round_zk
            .mask_message_len
            .checked_sub(source.randomness_len)
            .ok_or_else(|| VerifierError::MerkleProofInvalid {
                position: 0,
                reason: "source encoding randomness exceeds round mask message length".into(),
            })?;
        let (source_rows, source_randomness_rows) =
            source_rows_for_positions::<F, _>(&source_layout, &query_positions);
        let initial_gammas = initial_handoff
            .randomness
            .iter()
            .copied()
            .collect::<Vec<_>>();
        let auxiliary_covectors = zk_mask_residual_covectors_from_shape::<F, EF>(
            initial_gammas.len(),
            initial_zk.zk_sumcheck.ell_zk,
            &initial_gammas,
        );
        let output_relation = code_switch_output_relation_from_rows(
            source.message_len,
            &source.covector,
            &auxiliary_covectors,
            source.randomness_len,
            pad_len,
            &rho_ood_points,
            &source_rows,
            &source_randomness_rows,
            &claim,
        );

        let folding_factor_next = self.params.folding_factor.at_round(round_index + 1);
        let relation_weights = code_switch_relation_weights(output_relation, folding_factor_next);
        let handoff = ZkVerifier::<F, EF>::verify_claim::<MT, _>(
            &round_zk_proof.zk_sumcheck,
            &round_zk_proof.zk_sumcheck_mask_commitments,
            round_zk.mask_message_len,
            folding_factor_next,
            round_params.folding_pow_bits,
            mu_prime,
            challenger,
        )
        .map_err(VerifierError::from)?;
        let next_source = if round_index + 1 < self.n_rounds() {
            let next_round_index = round_index + 1;
            let next_round_zk = self.round_parameters[next_round_index]
                .zk
                .as_ref()
                .expect("next ZK round source requires RoundConfig::zk");
            let next_covector = fold_prefix_covector(relation_weights, &handoff.randomness);
            let next_randomness_len = next_round_zk.mask_query_budget;
            Some(ZkEncodedCodeSwitchVerifierSource {
                domain_size: encoded_source_domain_size::<EF>(
                    next_covector.len(),
                    next_randomness_len,
                ),
                message_len: next_covector.len(),
                covector: next_covector,
                residual_sumcheck_scale: EF::ONE,
                randomness_len: next_randomness_len,
            })
        } else {
            None
        };

        Ok(WhirZkPrefixVerifierRoundState {
            handoff,
            next_source,
        })
    }

    /// Replay all prefix-only ZK code-switching rounds and return the final ZK handoff.
    ///
    /// The verifier derives each carried source from the previous public
    /// code-switch relation, so callers do not provide prover-side covectors for
    /// rounds `1..`. The returned claim is the masked Construction 6.3 handoff
    /// for the final code-switch output relation.
    pub fn verify_zk_prefix_rounds<SourceEnc, MakeSourceEnc>(
        &self,
        proof: &PcsProof<F, EF, MT>,
        protocol: &OpeningProtocol,
        source: &ZkCodeSwitchVerifierSource<F, EF, MT>,
        mut source_encoding_for: MakeSourceEnc,
        challenger: &mut Challenger,
    ) -> Result<WhirZkPrefixVerifierRoundState<EF>, VerifierError>
    where
        SourceEnc: LinearZkEncoding<EF>,
        MakeSourceEnc: FnMut(usize, &ZkEncodedCodeSwitchVerifierSource<EF>) -> SourceEnc,
    {
        if self.n_rounds() == 0 {
            return Err(VerifierError::ZkVerifierRequiresPrefixPath);
        }

        let first_round = self.verify_round_zk_prefix(proof, protocol, source, challenger)?;
        let mut handoff = first_round.handoff;
        let mut next_source = first_round.next_source;

        for round_index in 1..self.n_rounds() {
            let source = next_source
                .take()
                .ok_or_else(|| VerifierError::MerkleProofInvalid {
                    position: 0,
                    reason: "previous ZK round did not carry the next encoded source".into(),
                })?;
            let source_encoding = source_encoding_for(round_index, &source);
            let round_state = self.verify_round_zk_prefix_from_encoded_source(
                proof,
                &handoff,
                round_index,
                &source,
                &source_encoding,
                challenger,
            )?;
            handoff = round_state.handoff;
            next_source = round_state.next_source;
        }

        Ok(WhirZkPrefixVerifierRoundState {
            handoff,
            next_source,
        })
    }

    /// Replay one encoded-source prefix-only ZK code-switch round.
    ///
    /// The caller supplies the verifier handoff from the previous ZK sumcheck and
    /// a transcript positioned immediately after that handoff. The source
    /// commitment is read from `proof.whir.rounds[round_index].commitment`, and
    /// source rows are derived from `source_encoding`, keeping Merkle opening
    /// semantics and generator-row semantics aligned.
    #[allow(clippy::too_many_lines)]
    pub fn verify_round_zk_prefix_from_encoded_source<SourceEnc>(
        &self,
        proof: &PcsProof<F, EF, MT>,
        prior_handoff: &ZkVerifierHandoff<EF>,
        round_index: usize,
        source: &ZkEncodedCodeSwitchVerifierSource<EF>,
        source_encoding: &SourceEnc,
        challenger: &mut Challenger,
    ) -> Result<WhirZkPrefixVerifierRoundState<EF>, VerifierError>
    where
        SourceEnc: LinearZkEncoding<EF>,
    {
        assert!(
            round_index > 0,
            "encoded ZK source verifier starts after round 0"
        );
        let zk_config = self
            .config
            .zk
            .as_ref()
            .ok_or(VerifierError::ZkVerifierRequiresPrefixPath)?;
        assert!(
            zk_config.only_prefix,
            "ZK WHIR currently supports only prefix layout"
        );

        let round_params = &self.round_parameters[round_index];
        let round_zk = round_params
            .zk
            .as_ref()
            .expect("verify_round_zk_prefix_from_encoded_source requires RoundConfig::zk");
        assert_eq!(source_encoding.message_len(), source.message_len);
        assert_eq!(source_encoding.randomness_len(), source.randomness_len);
        assert_eq!(source_encoding.codeword_len(), source.domain_size);
        assert!(
            source_encoding.query_bound() >= round_zk.mask_query_budget,
            "source encoding query bound must cover the round source openings",
        );
        let pad_len = round_zk
            .mask_message_len
            .checked_sub(source.randomness_len)
            .ok_or_else(|| VerifierError::MerkleProofInvalid {
                position: 0,
                reason: "source encoding randomness exceeds round mask message length".into(),
            })?;

        let round = proof
            .whir
            .rounds
            .get(round_index)
            .ok_or(VerifierError::InvalidRoundIndex { index: round_index })?;
        let source_commitment = round
            .commitment
            .clone()
            .ok_or(VerifierError::MissingRoundCommitment { round: round_index })?;
        challenger.observe(source_commitment.clone());

        let round_zk_proof = round
            .zk
            .as_ref()
            .ok_or(VerifierError::UnexpectedZkPayloadInPlainProof { round: round_index })?;
        challenger.observe(round_zk_proof.mask_commitment.clone());

        let rho_ood_points = (0..round_zk.ood_samples)
            .map(|_| challenger.sample_algebra_element())
            .collect::<Vec<EF>>();
        if round_zk_proof.private_ood_answers.len() != rho_ood_points.len() {
            return Err(VerifierError::StirQueryCountMismatch {
                round_index,
                expected: rho_ood_points.len(),
                actual: round_zk_proof.private_ood_answers.len(),
            });
        }
        challenger.observe_algebra_slice(&round_zk_proof.private_ood_answers);

        if round_params.pow_bits > 0
            && !challenger.check_witness(round_params.pow_bits, round.pow_witness)
        {
            return Err(VerifierError::InvalidPowWitness);
        }
        challenger.sample();

        let source_positions = get_challenge_stir_queries::<Challenger, F, EF>(
            source.domain_size,
            0,
            round_zk.mask_query_budget,
            challenger,
        );
        if round_zk_proof.source_queries.len() != source_positions.len() {
            return Err(VerifierError::StirQueryCountMismatch {
                round_index,
                expected: source_positions.len(),
                actual: round_zk_proof.source_queries.len(),
            });
        }
        let source_dimensions = [Dimensions {
            height: source.domain_size,
            width: 1,
        }];
        let mut source_openings = Vec::with_capacity(source_positions.len());
        for (&position, query) in source_positions.iter().zip(&round_zk_proof.source_queries) {
            let QueryOpening::Extension { values, proof } = query else {
                return Err(VerifierError::MerkleProofInvalid {
                    position,
                    reason: "Expected extension-field encoded source opening".into(),
                });
            };
            if values.len() != 1 {
                return Err(VerifierError::MerkleProofInvalid {
                    position,
                    reason: "Encoded source opening must contain one value".into(),
                });
            }
            self.extension_mmcs
                .verify_batch(
                    &source_commitment,
                    &source_dimensions,
                    position,
                    BatchOpeningRef {
                        opened_values: from_ref(values),
                        opening_proof: proof,
                    },
                )
                .map_err(|_| VerifierError::MerkleProofInvalid {
                    position,
                    reason: "Encoded ZK source opening verification failed".into(),
                })?;
            source_openings.push(values[0]);
        }

        let mask_indices = get_challenge_stir_queries::<Challenger, F, EF>(
            round_zk.mask_domain_size,
            0,
            round_zk.mask_query_budget,
            challenger,
        );
        if round_zk_proof.mask_queries.len() != mask_indices.len() {
            return Err(VerifierError::StirQueryCountMismatch {
                round_index,
                expected: mask_indices.len(),
                actual: round_zk_proof.mask_queries.len(),
            });
        }
        let mask_dimensions = [Dimensions {
            height: round_zk.mask_domain_size,
            width: 1,
        }];
        for (&position, query) in mask_indices.iter().zip(&round_zk_proof.mask_queries) {
            let QueryOpening::Extension { values, proof } = query else {
                return Err(VerifierError::MerkleProofInvalid {
                    position,
                    reason: "Expected extension-field code-switch mask opening".into(),
                });
            };
            self.extension_mmcs
                .verify_batch(
                    &round_zk_proof.mask_commitment,
                    &mask_dimensions,
                    position,
                    BatchOpeningRef {
                        opened_values: from_ref(values),
                        opening_proof: proof,
                    },
                )
                .map_err(|_| VerifierError::MerkleProofInvalid {
                    position,
                    reason: "Encoded ZK mask opening verification failed".into(),
                })?;
        }

        let batching_dim = 1 + round_zk_proof.private_ood_answers.len() + source_openings.len();
        let batching_challenge = challenger.sample_algebra_element();
        let coeffs = batching_coefficients(batching_challenge, batching_dim);
        let claim = ZkMaskClaim {
            base_claim_coeff: coeffs[0],
            residual_sumcheck_scale: source.residual_sumcheck_scale,
            ood_coeffs: coeffs[1..1 + round_zk_proof.private_ood_answers.len()].to_vec(),
            in_domain_coeffs: coeffs[1 + round_zk_proof.private_ood_answers.len()..].to_vec(),
            source_randomness_weights: Vec::new(),
            pad_weights: Vec::new(),
        };
        let mu_prime = batched_claim(
            prior_handoff.claimed_residual,
            &round_zk_proof.private_ood_answers,
            &source_openings,
            &claim,
        )
        .map_err(|err| VerifierError::MerkleProofInvalid {
            position: 0,
            reason: err.to_string(),
        })?;

        let source_layout = LinearZkSourceLayout {
            encoding: source_encoding,
        };
        let (source_rows, source_randomness_rows) =
            source_rows_for_positions::<EF, _>(&source_layout, &source_positions);
        let gammas = prior_handoff.randomness.iter().copied().collect::<Vec<_>>();
        let prior_round =
            proof
                .whir
                .rounds
                .get(round_index - 1)
                .ok_or(VerifierError::InvalidRoundIndex {
                    index: round_index - 1,
                })?;
        let prior_round_zk =
            prior_round
                .zk
                .as_ref()
                .ok_or(VerifierError::UnexpectedZkPayloadInPlainProof {
                    round: round_index - 1,
                })?;
        let auxiliary_covectors = zk_mask_residual_covectors_from_shape::<F, EF>(
            gammas.len(),
            prior_round_zk.zk_sumcheck.ell_zk,
            &gammas,
        );
        let output_relation = code_switch_output_relation_from_ext_rows(
            source.message_len,
            &source.covector,
            &auxiliary_covectors,
            source.randomness_len,
            pad_len,
            &rho_ood_points,
            &source_rows,
            &source_randomness_rows,
            &claim,
        );

        let folding_factor_next = self.params.folding_factor.at_round(round_index + 1);
        let relation_weights = code_switch_relation_weights(output_relation, folding_factor_next);
        let handoff = ZkVerifier::<F, EF>::verify_claim::<MT, _>(
            &round_zk_proof.zk_sumcheck,
            &round_zk_proof.zk_sumcheck_mask_commitments,
            round_zk.mask_message_len,
            folding_factor_next,
            round_params.folding_pow_bits,
            mu_prime,
            challenger,
        )
        .map_err(VerifierError::from)?;
        let next_source = if round_index + 1 < self.n_rounds() {
            let next_round_index = round_index + 1;
            let next_round_zk = self.round_parameters[next_round_index]
                .zk
                .as_ref()
                .expect("next ZK round source requires RoundConfig::zk");
            let next_covector = fold_prefix_covector(relation_weights, &handoff.randomness);
            let next_randomness_len = next_round_zk.mask_query_budget;
            Some(ZkEncodedCodeSwitchVerifierSource {
                domain_size: encoded_source_domain_size::<EF>(
                    next_covector.len(),
                    next_randomness_len,
                ),
                message_len: next_covector.len(),
                covector: next_covector,
                residual_sumcheck_scale: EF::ONE,
                randomness_len: next_randomness_len,
            })
        } else {
            None
        };

        Ok(WhirZkPrefixVerifierRoundState {
            handoff,
            next_source,
        })
    }
}

impl<EF, F, Dft, MT, Challenger, L> MultilinearPcs<EF, Challenger>
    for WhirProver<EF, F, Dft, MT, Challenger, L>
where
    F: TwoAdicField + Ord,
    EF: ExtensionField<F> + TwoAdicField,
    Dft: TwoAdicSubgroupDft<F>,
    MT: Mmcs<F>,
    Challenger: FieldChallenger<F>
        + GrindingChallenger<Witness = F>
        + CanSampleUniformBits<F>
        + CanObserve<MT::Commitment>,
    L: Layout<F, EF>,
{
    type Commitment = MT::Commitment;
    type Val = F;
    type ProverData = WhirProverData<F, EF, MT, L>;
    type Proof = PcsProof<F, EF, MT>;
    type Error = VerifierError;
    type Witness = Witness<F>;
    type OpeningProtocol = OpeningProtocol;

    fn num_vars(&self) -> usize {
        self.config.num_variables
    }

    fn commit(
        &self,
        witness: Self::Witness,
        challenger: &mut Challenger,
    ) -> (Self::Commitment, Self::ProverData) {
        assert_eq!(witness.num_variables(), self.config.num_variables);
        let (layout, commitment, merkle_data) = L::commit(
            &self.dft,
            &self.mmcs,
            challenger,
            witness,
            self.config.folding_factor.at_round(0),
            self.config.starting_log_inv_rate,
        );
        (
            commitment,
            WhirProverData {
                layout,
                merkle_data,
                _marker: PhantomData,
            },
        )
    }

    fn open(
        &self,
        mut prover_data: Self::ProverData,
        protocol: Self::OpeningProtocol,
        challenger: &mut Challenger,
    ) -> Self::Proof {
        let mut whir_proof = self.config.empty_proof();
        tracing::info_span!("ood claims").in_scope(|| {
            whir_proof.initial_ood_answers = (0..self.commitment_ood_samples)
                .map(|_| prover_data.layout.add_virtual_eval(challenger))
                .collect::<Vec<_>>();
        });

        let evals = protocol
            .iter_openings()
            .map(|(table_idx, polys)| prover_data.layout.eval(table_idx, polys, challenger))
            .collect::<Vec<_>>();

        self.prove(
            &mut whir_proof,
            challenger,
            prover_data.layout,
            prover_data.merkle_data,
        );

        PcsProof {
            whir: whir_proof,
            evals,
        }
    }

    fn verify(
        &self,
        commitment: &Self::Commitment,
        proof: &Self::Proof,
        challenger: &mut Challenger,
        protocol: Self::OpeningProtocol,
    ) -> Result<(), Self::Error> {
        challenger.observe(commitment.clone());

        let mut layout_verifier = Verifier::<F, EF>::new(&protocol.table_shapes(), L::strategy());
        for &eval in &proof.whir.initial_ood_answers {
            layout_verifier.add_virtual_eval(eval, challenger);
        }
        if protocol.num_openings() != proof.evals.len() {
            return Err(VerifierError::OpeningBatchCountMismatch {
                expected: protocol.num_openings(),
                actual: proof.evals.len(),
            });
        }
        for ((table_idx, polys), evals) in protocol.iter_openings().zip(&proof.evals) {
            if polys.len() != evals.len() {
                return Err(VerifierError::OpeningBatchSizeMismatch {
                    table_idx,
                    expected: polys.len(),
                    actual: evals.len(),
                });
            }
            layout_verifier.add_claim(table_idx, polys, evals, challenger);
        }

        let alpha = challenger.sample_algebra_element();
        let constraint = layout_verifier.constraint(alpha);
        let mut claimed_eval = EF::ZERO;
        constraint.combine_evals(&mut claimed_eval);

        let verifier = WhirVerifier::new(&self.config, &self.mmcs, L::variable_order());
        verifier.verify(
            &proof.whir,
            challenger,
            commitment,
            constraint,
            claimed_eval,
        )?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_baby_bear::BabyBear;
    use p3_dft::Radix2Dit;
    use p3_field::PrimeCharacteristicRing;
    use p3_field::extension::BinomialExtensionField;
    use p3_zk_codes::{LinearZkEncoding, ReedSolomonZkEncoding};

    use super::*;

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;

    fn ef(x: u64) -> EF {
        EF::from_u64(x)
    }

    #[test]
    fn code_switch_rows_include_source_randomness_contribution() {
        let source_message = vec![ef(29), ef(31)];
        let source_covector = vec![ef(3), ef(5)];
        let mask_message = vec![ef(37), ef(41), ef(43)];
        let source_row = vec![F::from_u64(11), F::from_u64(13)];
        let randomness_row = vec![F::from_u64(17), F::from_u64(19)];
        let claim = ZkMaskClaim {
            base_claim_coeff: ef(2),
            residual_sumcheck_scale: ef(23),
            ood_coeffs: Vec::new(),
            in_domain_coeffs: vec![ef(7)],
            source_randomness_weights: Vec::new(),
            pad_weights: Vec::new(),
        };

        let relation = code_switch_output_relation_from_rows::<F, EF>(
            source_message.len(),
            &source_covector,
            &[],
            2,
            1,
            &[],
            core::slice::from_ref(&source_row),
            core::slice::from_ref(&randomness_row),
            &claim,
        );
        let value = relation
            .evaluate(&source_message, &[], &mask_message)
            .unwrap();

        let expected_inherited =
            claim.base_claim_coeff * claim.residual_sumcheck_scale * ef(29 * 3 + 31 * 5);
        let expected_query =
            claim.in_domain_coeffs[0] * (ef(29 * 11 + 31 * 13) + ef(37 * 17 + 41 * 19));
        assert_eq!(value, expected_inherited + expected_query);
    }

    #[test]
    fn linear_zk_source_layout_uses_encoding_rows() {
        let encoding = ReedSolomonZkEncoding::<F, Radix2Dit<F>>::new(2, 3, 8, Radix2Dit::default());
        let layout = LinearZkSourceLayout {
            encoding: &encoding,
        };
        let positions = vec![0, 5];

        assert_eq!(layout.message_len(), encoding.message_len());
        assert_eq!(layout.randomness_len(), encoding.randomness_len());
        assert_eq!(layout.domain_size(), encoding.codeword_len());
        assert_eq!(layout.folding_factor(), 0);

        let (message_rows, randomness_rows) =
            source_rows_for_positions::<F, _>(&layout, &positions);
        assert_eq!(message_rows[0], encoding.message_row(positions[0]));
        assert_eq!(message_rows[1], encoding.message_row(positions[1]));
        assert_eq!(randomness_rows[0], encoding.randomness_row(positions[0]));
        assert_eq!(randomness_rows[1], encoding.randomness_row(positions[1]));
    }
}
