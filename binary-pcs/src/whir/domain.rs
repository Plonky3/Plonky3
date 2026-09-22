//! Binary-field evaluation domains for the WHIR commitment scheme.

use p3_binary_dft::{AdditiveRsEncoder, LchNtt, PolyBasisNtt, domain_point, subspace_polynomial};
use p3_binary_field::{BinaryField32, BinaryField128, Poly64, TowerLevel};
use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_commit::Encoder;
use p3_field::{BasedVectorSpace, ExtensionField, Field};
use p3_matrix::dense::RowMajorMatrix;
use p3_multilinear_util::point::Point;
use p3_whir::transcript::query_draws;
use p3_whir::{SecurityAssumption, WhirConfig, WhirDomain, WhirQueryPoint};

/// Widest domain a machine word can index.
///
/// A query index is a word, so a wider subspace could never be addressed.
const MAX_LOG_DOMAIN_SIZE: usize = 64;

/// A binary alphabet the additive transform encodes over.
///
/// Two alphabets of the same width can still carry different codes, so each one names itself.
///
/// The label separates one code from every other inside a shared transcript.
pub trait BinaryWhirAlphabet: TowerLevel {
    /// Stable label bound into the Fiat--Shamir instance description.
    const DOMAIN_ID: &'static [u8];
}

impl BinaryWhirAlphabet for Poly64 {
    const DOMAIN_ID: &'static [u8] = b"p3-whir-domain:cantor-novel-basis-v1";
}

impl BinaryWhirAlphabet for BinaryField32 {
    const DOMAIN_ID: &'static [u8] = b"p3-whir-domain:cantor-tower-32-v1";
}

impl BinaryWhirAlphabet for BinaryField128 {
    const DOMAIN_ID: &'static [u8] = b"p3-whir-domain:cantor-tower-128-v1";
}

/// Returns a Merkle cap height that every round's tree can carry.
///
/// One cap node covers each deepest stratum, so authentication paths stop at a protocol-fixed layer.
///
/// A round that opens every position draws nothing and fixes no stratum.
///
/// All rounds share one commitment scheme, so the shallowest tree bounds the result.
#[must_use]
pub fn recommended_cap_height<EF, F, Challenger>(config: &WhirConfig<EF, F, Challenger>) -> usize
where
    F: Field,
    EF: ExtensionField<F>,
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    let mut deepest = 0;
    let mut shallowest = usize::MAX;

    for (log_folded_domain_size, queries) in config
        .round_parameters
        .iter()
        .map(|round| (round.log_folded_domain_size, round.num_queries))
        .chain(core::iter::once((
            config.final_round_config().log_folded_domain_size,
            config.terminal().num_queries,
        )))
    {
        shallowest = shallowest.min(log_folded_domain_size);

        // A domain too wide to size as a machine word can never be saturated.
        let draws = 1usize
            .checked_shl(log_folded_domain_size as u32)
            .map_or(queries, |folded_domain_size| {
                query_draws(folded_domain_size, queries)
            });

        if draws != 0 {
            deepest = deepest.max(draws.ilog2() as usize);
        }
    }

    deepest.min(shallowest)
}

/// WHIR's Reed--Solomon code over nested binary Cantor subspaces.
///
/// The alphabet is a type parameter, so a narrow trace is encoded at its own width.
///
/// A folded value wider than the alphabet is encoded as one column per coordinate.
///
/// Every butterfly therefore keeps its twiddle in the alphabet.
#[derive(Clone, Debug)]
pub struct BinaryWhirDomain<F = Poly64, Ntt = LchNtt<F>> {
    encoder: AdditiveRsEncoder<F, Ntt>,
}

impl<F, Ntt> BinaryWhirDomain<F, Ntt> {
    /// Build the domain around an additive transform implementation.
    pub const fn new(ntt: Ntt) -> Self {
        Self {
            encoder: AdditiveRsEncoder::new(ntt),
        }
    }
}

impl<F, Ntt: Default> Default for BinaryWhirDomain<F, Ntt> {
    fn default() -> Self {
        Self::new(Ntt::default())
    }
}

/// The largest subspace dimension an alphabet supports.
fn max_log_domain_size<F: TowerLevel>() -> usize {
    (1usize << F::LOG_BITS).min(MAX_LOG_DOMAIN_SIZE)
}

/// Encode a padded matrix whose entries are wider than the alphabet.
///
/// Each entry becomes one column per coordinate, so every butterfly stays in the alphabet.
fn encode_extension<F, EF, E>(
    encoder: &E,
    message: RowMajorMatrix<EF>,
    log_inv_rate: usize,
) -> RowMajorMatrix<EF>
where
    F: TowerLevel,
    EF: ExtensionField<F>,
    E: Encoder<F>,
{
    let width = message.width;
    let coefficients = EF::flatten_to_base(message.values);
    let encoded = encoder.encode_batch_padded(
        RowMajorMatrix::new(coefficients, width * <EF as BasedVectorSpace<F>>::DIMENSION),
        log_inv_rate,
    );
    RowMajorMatrix::new(EF::reconstitute_from_base(encoded.values), width)
}

/// Selector coordinates of one queried codeword position.
fn query_point<F: TowerLevel>(
    log_domain_size: usize,
    num_variables: usize,
    index: usize,
) -> WhirQueryPoint<F> {
    assert!(
        log_domain_size <= max_log_domain_size::<F>(),
        "additive domain exceeds the alphabet dimension"
    );
    assert!(
        log_domain_size == MAX_LOG_DOMAIN_SIZE || index < 1usize << log_domain_size,
        "domain index is out of range"
    );
    let point = domain_point(index);
    WhirQueryPoint::Multilinear(Point::new(
        (0..num_variables)
            .rev()
            .map(|j| subspace_polynomial(j, point))
            .collect(),
    ))
}

/// Whether a regime's distance assumption holds for a characteristic-two subspace domain.
///
/// Capacity is refuted there, so it is the one regime refused.
const fn supports(assumption: SecurityAssumption) -> bool {
    !matches!(assumption, SecurityAssumption::CapacityBound)
}

impl<Ntt> Encoder<Poly64> for BinaryWhirDomain<Poly64, Ntt>
where
    AdditiveRsEncoder<Poly64, Ntt>: Encoder<Poly64>,
{
    fn encode_batch(
        &self,
        message: RowMajorMatrix<Poly64>,
        log_inv_rate: usize,
    ) -> RowMajorMatrix<Poly64> {
        self.encoder.encode_batch(message, log_inv_rate)
    }

    fn encode_batch_padded(
        &self,
        message: RowMajorMatrix<Poly64>,
        log_inv_rate: usize,
    ) -> RowMajorMatrix<Poly64> {
        self.encoder.encode_batch_padded(message, log_inv_rate)
    }
}

impl<EF, Ntt> WhirDomain<Poly64, EF> for BinaryWhirDomain<Poly64, Ntt>
where
    EF: ExtensionField<Poly64>,
    Ntt: Sync,
    AdditiveRsEncoder<Poly64, Ntt>: Encoder<Poly64>,
{
    fn protocol_id(&self) -> &'static [u8] {
        <Poly64 as BinaryWhirAlphabet>::DOMAIN_ID
    }

    fn supports_security_assumption(&self, assumption: SecurityAssumption) -> bool {
        supports(assumption)
    }

    fn stratified_queries(&self) -> bool {
        true
    }

    fn max_log_domain_size(&self) -> usize {
        max_log_domain_size::<Poly64>()
    }

    fn encode_extension_batch_padded(
        &self,
        message: RowMajorMatrix<EF>,
        log_inv_rate: usize,
    ) -> RowMajorMatrix<EF> {
        encode_extension::<Poly64, EF, _>(&self.encoder, message, log_inv_rate)
    }

    fn query_point(
        &self,
        log_domain_size: usize,
        num_variables: usize,
        index: usize,
    ) -> WhirQueryPoint<Poly64> {
        query_point::<Poly64>(log_domain_size, num_variables, index)
    }
}

impl<Ntt> Encoder<BinaryField32> for BinaryWhirDomain<BinaryField32, Ntt>
where
    AdditiveRsEncoder<BinaryField32, Ntt>: Encoder<BinaryField32>,
{
    fn encode_batch(
        &self,
        message: RowMajorMatrix<BinaryField32>,
        log_inv_rate: usize,
    ) -> RowMajorMatrix<BinaryField32> {
        self.encoder.encode_batch(message, log_inv_rate)
    }

    fn encode_batch_padded(
        &self,
        message: RowMajorMatrix<BinaryField32>,
        log_inv_rate: usize,
    ) -> RowMajorMatrix<BinaryField32> {
        self.encoder.encode_batch_padded(message, log_inv_rate)
    }
}

impl<EF, Ntt> WhirDomain<BinaryField32, EF> for BinaryWhirDomain<BinaryField32, Ntt>
where
    EF: ExtensionField<BinaryField32>,
    Ntt: Sync,
    AdditiveRsEncoder<BinaryField32, Ntt>: Encoder<BinaryField32>,
{
    fn protocol_id(&self) -> &'static [u8] {
        <BinaryField32 as BinaryWhirAlphabet>::DOMAIN_ID
    }

    fn supports_security_assumption(&self, assumption: SecurityAssumption) -> bool {
        supports(assumption)
    }

    fn stratified_queries(&self) -> bool {
        true
    }

    fn max_log_domain_size(&self) -> usize {
        max_log_domain_size::<BinaryField32>()
    }

    fn encode_extension_batch_padded(
        &self,
        message: RowMajorMatrix<EF>,
        log_inv_rate: usize,
    ) -> RowMajorMatrix<EF> {
        encode_extension::<BinaryField32, EF, _>(&self.encoder, message, log_inv_rate)
    }

    fn query_point(
        &self,
        log_domain_size: usize,
        num_variables: usize,
        index: usize,
    ) -> WhirQueryPoint<BinaryField32> {
        query_point::<BinaryField32>(log_domain_size, num_variables, index)
    }
}

impl<Ntt> Encoder<BinaryField128> for BinaryWhirDomain<BinaryField128, Ntt>
where
    AdditiveRsEncoder<BinaryField128, Ntt>: Encoder<BinaryField128>,
{
    fn encode_batch(
        &self,
        message: RowMajorMatrix<BinaryField128>,
        log_inv_rate: usize,
    ) -> RowMajorMatrix<BinaryField128> {
        self.encoder.encode_batch(message, log_inv_rate)
    }

    fn encode_batch_padded(
        &self,
        message: RowMajorMatrix<BinaryField128>,
        log_inv_rate: usize,
    ) -> RowMajorMatrix<BinaryField128> {
        self.encoder.encode_batch_padded(message, log_inv_rate)
    }
}

impl<EF, Ntt> WhirDomain<BinaryField128, EF> for BinaryWhirDomain<BinaryField128, Ntt>
where
    EF: ExtensionField<BinaryField128>,
    Ntt: Sync,
    AdditiveRsEncoder<BinaryField128, Ntt>: Encoder<BinaryField128>,
{
    fn protocol_id(&self) -> &'static [u8] {
        <BinaryField128 as BinaryWhirAlphabet>::DOMAIN_ID
    }

    fn supports_security_assumption(&self, assumption: SecurityAssumption) -> bool {
        supports(assumption)
    }

    fn stratified_queries(&self) -> bool {
        true
    }

    fn max_log_domain_size(&self) -> usize {
        max_log_domain_size::<BinaryField128>()
    }

    fn encode_extension_batch_padded(
        &self,
        message: RowMajorMatrix<EF>,
        log_inv_rate: usize,
    ) -> RowMajorMatrix<EF> {
        encode_extension::<BinaryField128, EF, _>(&self.encoder, message, log_inv_rate)
    }

    fn query_point(
        &self,
        log_domain_size: usize,
        num_variables: usize,
        index: usize,
    ) -> WhirQueryPoint<BinaryField128> {
        query_point::<BinaryField128>(log_domain_size, num_variables, index)
    }
}

/// The additive domain a bit witness packed into the widest tower level commits over.
pub type BooleanWhirDomain = BinaryWhirDomain<BinaryField128, PolyBasisNtt>;

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use p3_binary_dft::NaiveAdditiveNtt;
    use p3_binary_field::Poly192;
    use p3_commit::Encoder;
    use p3_field::PrimeCharacteristicRing;
    use p3_matrix::Matrix;
    use p3_matrix::dense::RowMajorMatrix;
    use p3_multilinear_util::poly::Poly;
    use p3_sumcheck::constraints::statement::SelectStatement;
    use p3_whir::{WhirDomain, WhirQueryPoint};

    use super::{BinaryField32, BinaryField128, BinaryWhirDomain, Poly64};

    #[test]
    fn extension_encoding_matches_the_definition() {
        let width = 2;
        let log_message_height = 3;
        let log_inv_rate = 2;
        let mut values = Poly192::zero_vec(width << (log_message_height + log_inv_rate));
        for (index, value) in values[..width << log_message_height].iter_mut().enumerate() {
            *value = Poly192::new([
                Poly64::new(index as u64 + 1),
                Poly64::new((3 * index) as u64 + 2),
                Poly64::new((5 * index) as u64 + 4),
            ]);
        }
        let message = RowMajorMatrix::new(values, width);
        let fast_domain: BinaryWhirDomain<Poly64> = BinaryWhirDomain::default();
        let fast = WhirDomain::<Poly64, Poly192>::encode_extension_batch_padded(
            &fast_domain,
            message.clone(),
            log_inv_rate,
        );
        let reference = BinaryWhirDomain::<Poly64, _>::new(NaiveAdditiveNtt::<Poly64>::default());
        let reference = WhirDomain::<Poly64, Poly192>::encode_extension_batch_padded(
            &reference,
            message,
            log_inv_rate,
        );
        assert_eq!(fast, reference);
    }

    #[test]
    fn base_encoding_matches_the_reference_transform() {
        let message = RowMajorMatrix::new((0..16).map(|index| Poly64::new(index + 1)).collect(), 2);
        let fast_domain: BinaryWhirDomain<Poly64> = BinaryWhirDomain::default();
        let fast = fast_domain.encode_batch(message.clone(), 2);
        let reference = BinaryWhirDomain::<Poly64, _>::new(NaiveAdditiveNtt::<Poly64>::default())
            .encode_batch(message, 2);
        assert_eq!(fast, reference);
    }

    #[test]
    fn query_coordinates_evaluate_the_encoded_novel_basis_polynomial() {
        let coefficients = (0..8)
            .map(|index| Poly64::new((13 * index + 7) as u64))
            .collect::<Vec<_>>();
        let domain: BinaryWhirDomain<Poly64> = BinaryWhirDomain::default();

        // Every tested rate extends beyond the message subspace.
        for log_inv_rate in 1..=3 {
            let encoded =
                domain.encode_batch(RowMajorMatrix::new(coefficients.clone(), 1), log_inv_rate);

            for index in 0..encoded.height() {
                let WhirQueryPoint::Multilinear(point) =
                    WhirDomain::<Poly64, Poly192>::query_point(&domain, 3 + log_inv_rate, 3, index)
                else {
                    panic!("the additive domain must expose direct selector coordinates");
                };

                // The selector must recover the encoded row at every domain index.
                let poly = Poly::new(coefficients.clone());
                let mut statement = SelectStatement::initialize(3);
                statement.add_point_constraint(point.clone(), encoded.values[index]);
                assert!(statement.verify(&poly));

                // A changed codeword value must fail the same selector identity.
                let mut tampered = SelectStatement::initialize(3);
                tampered.add_point_constraint(point, encoded.values[index] + Poly64::ONE);
                assert!(!tampered.verify(&poly));
            }
        }
    }

    #[test]
    fn every_alphabet_reports_its_own_capacity_and_label() {
        let word: BinaryWhirDomain<BinaryField32> = BinaryWhirDomain::default();
        let wide = super::BooleanWhirDomain::default();
        let poly: BinaryWhirDomain<Poly64> = BinaryWhirDomain::default();

        assert_eq!(
            WhirDomain::<BinaryField32, BinaryField128>::max_log_domain_size(&word),
            32
        );
        // The subspace is wider than a machine word, so the index type bounds it.
        assert_eq!(
            WhirDomain::<BinaryField128, BinaryField128>::max_log_domain_size(&wide),
            64
        );
        assert_eq!(
            WhirDomain::<Poly64, Poly192>::max_log_domain_size(&poly),
            64
        );

        // A sixty-four-bit polynomial basis and a sixty-four-bit tower level are different codes.
        assert_ne!(
            WhirDomain::<Poly64, Poly192>::protocol_id(&poly),
            WhirDomain::<BinaryField128, BinaryField128>::protocol_id(&wide)
        );
    }

    #[test]
    fn a_narrow_alphabet_encodes_through_the_same_selector_identity() {
        let coefficients = (0..8)
            .map(|index| BinaryField32::from_u32(13 * index + 7))
            .collect::<Vec<_>>();
        let domain: BinaryWhirDomain<BinaryField32> = BinaryWhirDomain::default();
        let encoded = domain.encode_batch(RowMajorMatrix::new(coefficients.clone(), 1), 2);

        for index in 0..encoded.height() {
            let WhirQueryPoint::Multilinear(point) =
                WhirDomain::<BinaryField32, BinaryField128>::query_point(&domain, 5, 3, index)
            else {
                panic!("the additive domain must expose direct selector coordinates");
            };
            let poly = Poly::new(coefficients.clone());
            let mut statement = SelectStatement::initialize(3);
            statement.add_point_constraint(point, encoded.values[index]);
            assert!(statement.verify(&poly));
        }
    }
}
