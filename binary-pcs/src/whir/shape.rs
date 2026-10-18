//! What a derived schedule fixes about the proofs it can produce.

use p3_binary_field::{BitCoordinates, TowerLevel};
use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_commit::Mmcs;
use p3_field::{ExtensionField, Field};
use p3_sumcheck::ring_switch::bits::BitTensor;
use p3_whir::WhirConfig;
use p3_whir::transcript::query_draws;
use thiserror::Error;

/// Why a proof payload cannot be bounded.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum ProofShapeError {
    /// The commitment backend does not describe its payload geometry.
    #[error("the MMCS does not support proof payload bounds for this geometry")]
    UnsupportedGeometry,
    /// A dimension or payload count cannot be represented.
    #[error("the proof payload bound overflows usize")]
    SizeOverflow,
}

fn checked(value: Option<usize>) -> Result<usize, ProofShapeError> {
    value.ok_or(ProofShapeError::SizeOverflow)
}

fn power_of_two(log: usize) -> Result<usize, ProofShapeError> {
    let shift = u32::try_from(log).map_err(|_| ProofShapeError::SizeOverflow)?;
    checked(1usize.checked_shl(shift))
}

/// Everything a schedule fixes about the size of its proofs and the work of checking one.
///
/// Each figure is read off the derived schedule and the claim count, never off a witness.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProofShape {
    /// Codeword positions opened across every round, one Merkle check each.
    pub stir_queries: usize,

    /// Alphabet elements opened from the first codeword.
    pub opened_base_elements: usize,

    /// Challenge-field elements opened from the folded codewords.
    pub opened_extension_elements: usize,

    /// Digests the proof carries, including folded codewords' commitments and opened paths.
    ///
    /// No prefix a path shares with another is discounted.
    pub merkle_digests: usize,

    /// Alphabet elements the proof carries outside the openings, which are its grinding witnesses.
    pub sent_base_elements: usize,

    /// Challenge-field elements the proof carries outside the openings.
    pub sent_extension_elements: usize,

    /// Hardest grinding any phase of the schedule demands.
    pub grinding_bits: usize,
}

impl ProofShape {
    /// Read the shape of every proximity opening a schedule can produce.
    ///
    /// The claim count fixes how many opened values travel beside the transcript.
    ///
    /// This bounds a single committed polynomial.
    /// Wider stacked layouts need their own bounds.
    /// The separately transmitted initial commitment and serializer framing are excluded.
    ///
    /// # Errors
    ///
    /// Returns an error for unsupported MMCS geometry or unrepresentable counts.
    pub fn of<EF, F, Challenger, MT>(
        config: &WhirConfig<EF, F, Challenger>,
        mmcs: &MT,
        opened_values: usize,
    ) -> Result<Self, ProofShapeError>
    where
        F: Field,
        EF: ExtensionField<F>,
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
        MT: Mmcs<F>,
    {
        let num_rounds = config.round_parameters().len();
        let final_poly_elements = power_of_two(config.final_sumcheck_rounds())?;
        let mut shape = Self {
            stir_queries: 0,
            opened_base_elements: 0,
            opened_extension_elements: 0,
            merkle_digests: 0,
            // One witness per folding round, one per folded codeword, and one closing the run.
            sent_base_elements: checked(
                config
                    .num_variables()
                    .checked_add(num_rounds)
                    .and_then(|n| n.checked_add(1)),
            )?,
            // Two coefficients bound one degree-two round, and one value closes the run.
            sent_extension_elements: checked(
                config
                    .num_variables()
                    .checked_mul(2)
                    .and_then(|n| n.checked_add(final_poly_elements))
                    .and_then(|n| n.checked_add(config.commitment_ood_samples()))
                    .and_then(|n| n.checked_add(opened_values)),
            )?,
            grinding_bits: config.max_pow_bits(),
        };

        let rounds = config
            .round_parameters()
            .iter()
            .map(|round| {
                (
                    round.log_folded_domain_size,
                    round.folding_factor,
                    round.num_queries,
                    round.ood_samples,
                )
            })
            .chain(core::iter::once({
                let final_round = config.final_round_config();
                (
                    final_round.log_folded_domain_size,
                    final_round.folding_factor,
                    config.terminal().num_queries,
                    0,
                )
            }));

        for (index, (log_folded, folding, queries, ood)) in rounds.enumerate() {
            let domain_size = power_of_two(log_folded)?;
            let geometry = mmcs
                .proof_geometry(log_folded)
                .ok_or(ProofShapeError::UnsupportedGeometry)?;
            // A round drawing nothing opens the whole folded domain instead.
            let positions = if query_draws(domain_size, queries) == 0 {
                domain_size
            } else {
                queries
            };
            let elements = checked(positions.checked_mul(power_of_two(folding)?))?;

            shape.stir_queries = checked(shape.stir_queries.checked_add(positions))?;
            if index == 0 {
                shape.opened_base_elements =
                    checked(shape.opened_base_elements.checked_add(elements))?;
            } else {
                shape.opened_extension_elements =
                    checked(shape.opened_extension_elements.checked_add(elements))?;
                // The initial commitment travels separately; every folded cap is in the proof.
                shape.merkle_digests = checked(
                    shape
                        .merkle_digests
                        .checked_add(geometry.commitment_digests),
                )?;
            }
            shape.merkle_digests = checked(
                shape
                    .merkle_digests
                    .checked_add(checked(positions.checked_mul(geometry.opening_digests))?),
            )?;
            shape.sent_extension_elements =
                checked(shape.sent_extension_elements.checked_add(ood))?;
        }

        Ok(shape)
    }

    /// Read the shape of every bit-witness opening a schedule can produce for this many claims.
    ///
    /// One batched reduction travels beside a proximity opening of its one surviving point.
    ///
    /// Each claim sends its elements by rows, one packing-level element per row.
    /// The batch sends one run of degree-two rounds and the value it survives with.
    ///
    /// # Errors
    ///
    /// Returns an error for unsupported MMCS geometry or unrepresentable counts.
    pub fn of_bit_readings<F, EF, Challenger, MT>(
        config: &WhirConfig<EF, F, Challenger>,
        mmcs: &MT,
        num_claims: usize,
        successor_tensors: bool,
    ) -> Result<Self, ProofShapeError>
    where
        F: TowerLevel,
        EF: BitCoordinates + ExtensionField<F>,
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
        MT: Mmcs<F>,
    {
        // The batch leaves one surviving claim, however many claims it folds.
        let surviving = num_claims.min(1);
        let mut shape = Self::of(config, mmcs, surviving)?;
        // The element alone, or the element with carry and last.
        let num_tensors: usize = if successor_tensors { 3 } else { 1 };
        let rows = checked(num_tensors.checked_mul(BitTensor::<EF, F>::DIMENSION))?;
        shape.sent_base_elements = checked(
            shape
                .sent_base_elements
                .checked_add(checked(num_claims.checked_mul(rows))?),
        )?;
        // A Boolean prefix only removes rounds, so the packing's arity bounds them.
        let rounds = checked(checked(config.num_variables().checked_mul(2))?.checked_add(1))?;
        shape.sent_extension_elements = checked(
            shape
                .sent_extension_elements
                .checked_add(checked(surviving.checked_mul(rounds))?),
        )?;
        Ok(shape)
    }

    /// Bytes a proof of this shape cannot exceed, given the width of each thing it carries.
    ///
    /// A serializer that pads a value beyond its own width invalidates the figure.
    ///
    /// The separately transmitted initial commitment is the caller's to budget.
    /// Framing the serializer adds around a list is also the caller's to allow for.
    ///
    /// # Errors
    ///
    /// Returns an error if the byte count cannot be represented.
    pub fn max_bytes(
        &self,
        base_element_bytes: usize,
        extension_element_bytes: usize,
        digest_bytes: usize,
    ) -> Result<usize, ProofShapeError> {
        let base = checked(
            self.opened_base_elements
                .checked_add(self.sent_base_elements)
                .and_then(|n| n.checked_mul(base_element_bytes)),
        )?;
        let extension = checked(
            self.opened_extension_elements
                .checked_add(self.sent_extension_elements)
                .and_then(|n| n.checked_mul(extension_element_bytes)),
        )?;
        let digests = checked(self.merkle_digests.checked_mul(digest_bytes))?;
        checked(
            base.checked_add(extension)
                .and_then(|n| n.checked_add(digests)),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::{ProofShape, ProofShapeError, power_of_two};

    // A hand-built shape whose byte total is computable without the implementation.
    const SHAPE: ProofShape = ProofShape {
        stir_queries: 40,
        opened_base_elements: 100,
        opened_extension_elements: 200,
        merkle_digests: 300,
        sent_base_elements: 10,
        sent_extension_elements: 50,
        grinding_bits: 16,
    };

    #[test]
    fn the_estimate_prices_each_part_at_its_own_width() {
        // (100 + 10)*4 + (200 + 50)*16 + 300*32 = 440 + 4000 + 9600.
        assert_eq!(SHAPE.max_bytes(4, 16, 32), Ok(14_040));
    }

    #[test]
    fn a_wider_alphabet_raises_the_estimate_by_exactly_the_alphabet_run() {
        // Only the first codeword and the grinding witnesses use the alphabet's width.
        assert_eq!(
            SHAPE.max_bytes(16, 16, 32).unwrap() - SHAPE.max_bytes(4, 16, 32).unwrap(),
            110 * (16 - 4)
        );
    }

    #[test]
    fn dimensions_must_fit_without_masking_the_shift() {
        assert_eq!(power_of_two(0), Ok(1));
        assert_eq!(
            power_of_two(usize::BITS as usize - 1),
            Ok(1usize << (usize::BITS - 1))
        );
        assert_eq!(
            power_of_two(usize::BITS as usize),
            Err(ProofShapeError::SizeOverflow)
        );
        assert_eq!(power_of_two(usize::MAX), Err(ProofShapeError::SizeOverflow));
    }
}
