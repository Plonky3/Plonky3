//! What a derived schedule fixes about the proofs it can produce.

use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::{ExtensionField, Field};
use p3_whir::WhirConfig;
use p3_whir::transcript::query_draws;

/// Everything a schedule fixes about the size of its proofs and the work of checking one.
///
/// Each figure is read off the derived schedule, so none of it depends on the witness.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProofShape {
    /// Codeword positions opened across every round, one Merkle check each.
    pub stir_queries: usize,

    /// Alphabet elements opened from the first codeword.
    pub opened_base_elements: usize,

    /// Challenge-field elements opened from the folded codewords.
    pub opened_extension_elements: usize,

    /// Authentication digests the openings need, counted without any shared prefix.
    pub merkle_digests: usize,

    /// Challenge-field elements the reduction sends outside the openings.
    pub sent_extension_elements: usize,

    /// Hardest grinding any phase of the schedule demands.
    pub grinding_bits: usize,
}

impl ProofShape {
    /// Read the shape of every proof a schedule can produce.
    #[must_use]
    pub fn of<EF, F, Challenger>(config: &WhirConfig<EF, F, Challenger>) -> Self
    where
        F: Field,
        EF: ExtensionField<F>,
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        let mut shape = Self {
            stir_queries: 0,
            opened_base_elements: 0,
            opened_extension_elements: 0,
            merkle_digests: 0,
            // Three coefficients bound one degree-two round, and one value closes the run.
            sent_extension_elements: 3 * config.num_variables
                + (1 << config.final_sumcheck_rounds)
                + config.commitment_ood_samples,
            grinding_bits: config.max_pow_bits(),
        };

        let rounds = config
            .round_parameters
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
                    config.final_queries,
                    0,
                )
            }));

        for (index, (log_folded, folding, queries, ood)) in rounds.enumerate() {
            // A round drawing nothing opens the whole folded domain instead.
            let positions = if query_draws(1usize << log_folded, queries) == 0 {
                1usize << log_folded
            } else {
                queries
            };
            let elements = positions << folding;

            shape.stir_queries += positions;
            if index == 0 {
                shape.opened_base_elements += elements;
            } else {
                shape.opened_extension_elements += elements;
            }
            shape.merkle_digests += positions * log_folded;
            shape.sent_extension_elements += ood;
        }

        shape
    }

    /// Bytes a proof of this shape cannot exceed, given the width of each thing it carries.
    ///
    /// A serializer that pads a value beyond its own width invalidates the figure.
    #[must_use]
    pub const fn max_bytes(
        &self,
        base_element_bytes: usize,
        extension_element_bytes: usize,
        digest_bytes: usize,
    ) -> usize {
        self.opened_base_elements * base_element_bytes
            + (self.opened_extension_elements + self.sent_extension_elements)
                * extension_element_bytes
            + self.merkle_digests * digest_bytes
    }
}

#[cfg(test)]
mod tests {
    use super::ProofShape;

    // A hand-built shape whose byte total is computable without the implementation.
    const SHAPE: ProofShape = ProofShape {
        stir_queries: 40,
        opened_base_elements: 100,
        opened_extension_elements: 200,
        merkle_digests: 300,
        sent_extension_elements: 50,
        grinding_bits: 16,
    };

    #[test]
    fn the_estimate_prices_each_part_at_its_own_width() {
        // 100*4 + (200 + 50)*16 + 300*32 = 400 + 4000 + 9600.
        assert_eq!(SHAPE.max_bytes(4, 16, 32), 14_000);
    }

    #[test]
    fn a_wider_alphabet_raises_the_estimate_by_exactly_the_opened_run() {
        // Only the first codeword is opened at the alphabet's width.
        assert_eq!(
            SHAPE.max_bytes(16, 16, 32) - SHAPE.max_bytes(4, 16, 32),
            100 * (16 - 4)
        );
    }
}
