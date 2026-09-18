//! Binary-field evaluation domain for the WHIR commitment scheme.

use p3_binary_dft::{AdditiveNtt, AdditiveRsEncoder, LchNtt, domain_point, subspace_polynomial};
use p3_binary_field::{Poly64, Poly192};
use p3_commit::Encoder;
use p3_field::BasedVectorSpace;
use p3_matrix::dense::RowMajorMatrix;
use p3_multilinear_util::point::Point;
use p3_whir::{WhirDomain, WhirQueryPoint};

/// Returns the Merkle cap height of the deepest query stratum.
///
/// One cap node covers each deepest stratum, so authentication paths stop at
/// a protocol-fixed layer.
#[must_use]
pub fn recommended_cap_height<Challenger>(
    config: &p3_whir::WhirConfig<Poly192, Poly64, Challenger>,
) -> usize {
    config
        .round_parameters
        .iter()
        .map(|round| round.num_queries)
        .chain(core::iter::once(config.final_queries))
        .filter(|&queries| queries != 0)
        .map(|queries| queries.ilog2() as usize)
        .max()
        .unwrap_or(0)
}

/// WHIR's Reed--Solomon code over nested 64-bit Cantor subspaces.
///
/// Folded 192-bit values are encoded as three 64-bit columns. Every butterfly
/// therefore keeps its twiddle in the base field.
#[derive(Clone, Debug, Default)]
pub struct BinaryWhirDomain<Ntt = LchNtt<Poly64>> {
    encoder: AdditiveRsEncoder<Poly64, Ntt>,
}

impl<Ntt> BinaryWhirDomain<Ntt> {
    /// Build the domain around an additive transform implementation.
    pub const fn new(ntt: Ntt) -> Self {
        Self {
            encoder: AdditiveRsEncoder::new(ntt),
        }
    }
}

impl<Ntt> Encoder<Poly64> for BinaryWhirDomain<Ntt>
where
    Ntt: AdditiveNtt<Poly64> + Sync,
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

impl<Ntt> WhirDomain<Poly64, Poly192> for BinaryWhirDomain<Ntt>
where
    Ntt: AdditiveNtt<Poly64> + Sync,
{
    fn protocol_id(&self) -> &'static [u8] {
        b"p3-whir-domain:cantor-novel-basis-v1"
    }

    fn stratified_queries(&self) -> bool {
        true
    }

    fn max_log_domain_size(&self) -> usize {
        64
    }

    fn encode_extension_batch_padded(
        &self,
        message: RowMajorMatrix<Poly192>,
        log_inv_rate: usize,
    ) -> RowMajorMatrix<Poly192> {
        let width = message.width;
        let coefficients = Poly192::flatten_to_base(message.values);
        let encoded = self.encoder.encode_batch_padded(
            RowMajorMatrix::new(
                coefficients,
                width * <Poly192 as BasedVectorSpace<Poly64>>::DIMENSION,
            ),
            log_inv_rate,
        );
        RowMajorMatrix::new(Poly192::reconstitute_from_base(encoded.values), width)
    }

    fn query_point(
        &self,
        log_domain_size: usize,
        num_variables: usize,
        index: usize,
    ) -> WhirQueryPoint<Poly64> {
        assert!(
            log_domain_size <= 64,
            "additive domain exceeds the field dimension"
        );
        assert!(
            log_domain_size == 64 || index < 1usize << log_domain_size,
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
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use p3_binary_dft::NaiveAdditiveNtt;
    use p3_commit::Encoder;
    use p3_field::PrimeCharacteristicRing;
    use p3_matrix::dense::RowMajorMatrix;
    use p3_whir::{WhirDomain, WhirQueryPoint};

    use super::{BinaryWhirDomain, Poly64, Poly192};

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
        let fast_domain: BinaryWhirDomain<p3_binary_dft::LchNtt<Poly64>> =
            BinaryWhirDomain::default();
        let fast = fast_domain.encode_extension_batch_padded(message.clone(), log_inv_rate);
        let reference = BinaryWhirDomain::new(NaiveAdditiveNtt::<Poly64>::default())
            .encode_extension_batch_padded(message, log_inv_rate);
        assert_eq!(fast, reference);
    }

    #[test]
    fn base_encoding_matches_the_reference_transform() {
        let message = RowMajorMatrix::new((0..16).map(|index| Poly64::new(index + 1)).collect(), 2);
        let fast_domain: BinaryWhirDomain<p3_binary_dft::LchNtt<Poly64>> =
            BinaryWhirDomain::default();
        let fast = fast_domain.encode_batch(message.clone(), 2);
        let reference =
            BinaryWhirDomain::new(NaiveAdditiveNtt::<Poly64>::default()).encode_batch(message, 2);
        assert_eq!(fast, reference);
    }

    #[test]
    fn query_coordinates_evaluate_the_encoded_novel_basis_polynomial() {
        let coefficients = (0..8)
            .map(|index| Poly64::new((13 * index + 7) as u64))
            .collect::<Vec<_>>();
        let domain: BinaryWhirDomain<p3_binary_dft::LchNtt<Poly64>> = BinaryWhirDomain::default();
        let encoded = domain.encode_batch(RowMajorMatrix::new(coefficients.clone(), 1), 0);

        for index in 0..8 {
            let WhirQueryPoint::Multilinear(point) = domain.query_point(3, 3, index) else {
                panic!("the additive domain must expose direct selector coordinates");
            };
            let mut values = coefficients.clone();
            for &coordinate in point.iter().rev() {
                for position in 0..values.len() / 2 {
                    values[position] = values[2 * position] + values[2 * position + 1] * coordinate;
                }
                values.truncate(values.len() / 2);
            }
            assert_eq!(values[0], encoded.values[index]);
        }
    }
}
