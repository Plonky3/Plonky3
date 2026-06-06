//! HVZK base-case prover (Construction 7.2).

use alloc::vec::Vec;

use p3_challenger::{CanObserve, CanSampleUniformBits, FieldChallenger, GrindingChallenger};
use p3_commit::{ExtensionMmcs, Mmcs};
use p3_dft::TwoAdicSubgroupDft;
use p3_field::{ExtensionField, TwoAdicField, dot_product};
use p3_matrix::dense::RowMajorMatrix;
use p3_sumcheck::zk::stack_codewords;
use p3_zk_codes::{ZkEncoding, ZkEncodingWithRandomness};
use rand::distr::{Distribution, StandardUniform};
use rand::{Rng, RngExt};

use super::config::{BaseCaseZkConfig, MaskGroupWitness};
use crate::pcs::proof::QueryOpening;
use crate::pcs::utils::get_challenge_stir_queries;
use crate::pcs::zk::proof::{BaseCaseZkProof, BlindedMask, MaskOpeningPair};

/// HVZK base-case prover (Construction 7.2).
pub struct BaseCaseZkProver<'a, F, EF, MT>
where
    F: TwoAdicField,
    EF: ExtensionField<F>,
    MT: Mmcs<F>,
{
    /// Protocol shape shared with the verifier.
    pub config: &'a BaseCaseZkConfig<F>,
    /// Commitment scheme for the fresh masks and mask openings.
    pub extension_mmcs: &'a ExtensionMmcs<F, EF, MT>,
}

impl<F, EF, MT> BaseCaseZkProver<'_, F, EF, MT>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
    MT: Mmcs<F>,
    StandardUniform: Distribution<EF>,
{
    /// Runs the base case and writes the payload into `proof`.
    ///
    /// `open_source` opens the (virtual) source at a folded-domain position.
    /// The pipeline serves it by opening a leaf of the last oracle.
    #[allow(clippy::too_many_arguments)]
    pub fn prove<Dft, Challenger, R>(
        &self,
        dft: &Dft,
        source_message: &[EF],
        source_randomness: &[EF],
        source_covector: &[EF],
        masks: &[MaskGroupWitness<'_, F, EF, MT>],
        mut open_source: impl FnMut(usize) -> QueryOpening<F, EF, MT::Proof>,
        challenger: &mut Challenger,
        rng: &mut R,
    ) -> BaseCaseZkProof<F, EF, MT>
    where
        Dft: TwoAdicSubgroupDft<F>,
        Challenger: FieldChallenger<F>
            + GrindingChallenger<Witness = F>
            + CanSampleUniformBits<F>
            + CanObserve<MT::Commitment>,
        R: Rng,
    {
        let code = &self.config.code;
        assert_eq!(source_message.len(), code.message_len);
        assert_eq!(source_randomness.len(), code.randomness_len);
        assert_eq!(source_covector.len(), code.message_len);
        assert_eq!(masks.len(), self.config.mask_groups.len());

        // Fresh main mask in the folded source code.
        let fresh_message: Vec<EF> = (0..code.message_len).map(|_| rng.random()).collect();
        let fresh_randomness: Vec<EF> = (0..code.randomness_len).map(|_| rng.random()).collect();
        let codeword = code.encode_column(dft, &fresh_message, &fresh_randomness);
        let (fresh_main_commitment, fresh_main_data) = self.extension_mmcs.commit_matrix(codeword);
        challenger.observe(fresh_main_commitment.clone());

        // One fresh blind per carried mask, committed group-wise: the blinds
        // of a group stack into one matrix in the group's own code.
        let mut fresh_mask_commitments = Vec::with_capacity(masks.len());
        let mut fresh_groups = Vec::with_capacity(masks.len());
        for (group, witness) in self.config.mask_groups.iter().zip(masks) {
            let encoding = group.shape.encoding::<EF>();
            let mut blind_messages = Vec::with_capacity(group.width);
            let mut blind_randomness = Vec::with_capacity(group.width);
            let codewords: Vec<RowMajorMatrix<EF>> = (0..group.width)
                .map(|_| {
                    let message = encoding.sample_message(rng);
                    let randomness = encoding.sample_randomness(rng);
                    let codeword = encoding.encode_with_randomness(&message, &randomness);
                    blind_messages.push(message);
                    blind_randomness.push(randomness);
                    codeword
                })
                .collect();
            let (commitment, data) = self
                .extension_mmcs
                .commit_matrix(stack_codewords(&codewords));
            challenger.observe(commitment.clone());
            fresh_mask_commitments.push(commitment);
            fresh_groups.push((blind_messages, blind_randomness, data, witness));
        }

        // Fresh-side claim mu_g over the fresh main mask and every blind.
        let mut masked_claim = dot_product::<EF, _, _>(
            fresh_message.iter().copied(),
            source_covector.iter().copied(),
        );
        for (blind_messages, _, _, witness) in &fresh_groups {
            for (message, covector) in blind_messages.iter().zip(witness.covectors) {
                masked_claim +=
                    dot_product::<EF, _, _>(message.iter().copied(), covector.iter().copied());
            }
        }
        challenger.observe_algebra_element(masked_claim);

        // Blinding challenge.
        let gamma: EF = challenger.sample_algebra_element();

        // Reveal the one-time-padded source and masks.
        let blind = |fresh: &[EF], hidden: &[EF]| -> Vec<EF> {
            fresh
                .iter()
                .zip(hidden)
                .map(|(&fresh, &hidden)| fresh + gamma * hidden)
                .collect()
        };
        let blinded_message = blind(&fresh_message, source_message);
        let blinded_randomness = blind(&fresh_randomness, source_randomness);
        challenger.observe_algebra_slice(&blinded_message);
        challenger.observe_algebra_slice(&blinded_randomness);
        let mut blinded_masks = Vec::new();
        for (blind_messages, blind_randomness, _, witness) in &fresh_groups {
            for ((message, randomness), (hidden_message, hidden_randomness)) in blind_messages
                .iter()
                .zip(blind_randomness)
                .zip(witness.messages.iter().zip(witness.randomness))
            {
                let blinded = BlindedMask {
                    message: blind(message, hidden_message),
                    randomness: blind(randomness, hidden_randomness),
                };
                challenger.observe_algebra_slice(&blinded.message);
                challenger.observe_algebra_slice(&blinded.randomness);
                blinded_masks.push(blinded);
            }
        }

        // PoW before the spot checks.
        //
        //     pow_bits = 0  ->  no grind, zero witness on the wire
        let pow_witness = if self.config.pow_bits > 0 {
            challenger.grind(self.config.pow_bits)
        } else {
            F::ZERO
        };

        // Source spot checks: open the virtual source and the fresh mask.
        let positions = get_challenge_stir_queries::<Challenger, F, EF>(
            code.domain_size,
            0,
            self.config.num_queries,
            challenger,
        );
        let mut source_queries = Vec::with_capacity(positions.len());
        let mut fresh_main_queries = Vec::with_capacity(positions.len());
        for &position in &positions {
            source_queries.push(open_source(position));
            let opening = self.extension_mmcs.open_batch(position, &fresh_main_data);
            fresh_main_queries.push(QueryOpening::Extension {
                values: opening.opened_values.into_iter().next().unwrap(),
                proof: opening.opening_proof,
            });
        }

        // Mask spot checks: per group, open the carried oracle and its fresh
        // blind at shared positions; each row carries the whole group.
        let mut mask_queries = Vec::with_capacity(fresh_groups.len());
        for (group, (_, _, fresh_data, witness)) in
            self.config.mask_groups.iter().zip(&fresh_groups)
        {
            let positions = get_challenge_stir_queries::<Challenger, F, EF>(
                group.shape.domain_size,
                0,
                self.config.mask_queries,
                challenger,
            );
            let pairs = positions
                .iter()
                .map(|&position| {
                    let carried = self.extension_mmcs.open_batch(position, witness.data);
                    let fresh = self.extension_mmcs.open_batch(position, fresh_data);
                    MaskOpeningPair {
                        carried: QueryOpening::Extension {
                            values: carried.opened_values.into_iter().next().unwrap(),
                            proof: carried.opening_proof,
                        },
                        fresh: QueryOpening::Extension {
                            values: fresh.opened_values.into_iter().next().unwrap(),
                            proof: fresh.opening_proof,
                        },
                    }
                })
                .collect();
            mask_queries.push(pairs);
        }

        BaseCaseZkProof {
            fresh_main_commitment,
            fresh_mask_commitments,
            masked_claim,
            blinded_message,
            blinded_randomness,
            blinded_masks,
            pow_witness,
            source_queries,
            fresh_main_queries,
            mask_queries,
        }
    }
}
