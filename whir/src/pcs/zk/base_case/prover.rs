//! HVZK base-case prover (Construction 7.2).

use alloc::vec::Vec;

use p3_challenger::{CanObserve, CanSampleUniformBits, FieldChallenger, GrindingChallenger};
use p3_commit::{ExtensionMmcs, Mmcs};
use p3_dft::TwoAdicSubgroupDft;
use p3_field::{ExtensionField, TwoAdicField, dot_product};
use p3_zk_codes::{ZkEncoding, ZkEncodingWithRandomness};
use rand::distr::{Distribution, StandardUniform};
use rand::{Rng, RngExt};

use super::config::{BaseCaseZkConfig, MaskGroupWitness};
use super::{max_mask_domain_size, project_mask_positions};
use crate::pcs::proof::{QueryOpenings, SharedBatchOpening, SharedProofOpening};
use crate::pcs::utils::{get_challenge_iid_queries, get_challenge_stir_queries};
use crate::pcs::zk::proof::{BaseCaseZkProof, BlindedMask};

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
    /// Runs Construction 7.2 and returns the proof payload.
    ///
    /// # Moves
    ///
    /// ```text
    ///     1. commit fresh masks       g, s'_i
    ///     2. send fresh-side claim    mu_g
    ///     3. receive challenge        gamma
    ///     4. reveal one-time pads     f*, r*, xi*_i, r*_i
    ///     5. open spot-check positions
    /// ```
    ///
    /// # Arguments
    ///
    /// - `open_source`: opens the (virtual) source at the folded-domain positions.
    #[allow(clippy::too_many_arguments)]
    pub fn prove<Dft, Challenger, R>(
        &self,
        dft: &Dft,
        source_message: &[EF],
        source_randomness: &[EF],
        source_covector: &[EF],
        masks: &[MaskGroupWitness<'_, F, EF, MT>],
        open_source: impl FnOnce(&[usize]) -> QueryOpenings<F, EF, MT::MultiProof>,
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
        // The witness must fit the agreed folded source code exactly.
        assert_eq!(source_message.len(), code.message_len);
        assert_eq!(source_randomness.len(), code.randomness_len);
        assert_eq!(source_covector.len(), code.message_len);
        assert_eq!(masks.len(), self.config.mask_groups.len());

        // Move 1a: fresh main mask g = Enc(g~, r_g).
        //
        //     g~   ->  uniform vector, future pad of the source reveal f*
        //     r_g  ->  uniform vector, future pad of the randomness reveal r*
        //
        // Why the source's own code: the spot check later compares
        //
        //     Enc(f*, r*)(z)  vs  g(z) + gamma * f(z)
        //
        // and that equation needs all three words in one code.
        let fresh_message: Vec<EF> = (0..code.message_len).map(|_| rng.random()).collect();
        let fresh_randomness: Vec<EF> = (0..code.randomness_len).map(|_| rng.random()).collect();
        let codeword = code.encode_column(dft, &fresh_message, &fresh_randomness);
        let (fresh_main_commitment, fresh_main_data) = self.extension_mmcs.commit_matrix(codeword);
        // Bind the commitment before any challenge depends on it.
        challenger.observe(fresh_main_commitment.clone());

        // Move 1b: one fresh blind s'_i = Enc(s~'_i, r'_i) per carried mask.
        //
        // Each group keeps its own encoding and dimensions, mirroring the
        // carried masks. The mixed MMCS commits every encoded matrix under
        // one root:
        //
        //     group of width w  ->  w codewords stacked into one matrix
        //     all group matrices -> one mixed-height commitment
        let mut fresh_matrices = Vec::with_capacity(masks.len());
        let mut fresh_groups = Vec::with_capacity(masks.len());
        for (group, witness) in self.config.mask_groups.iter().zip(masks) {
            // Every member of a group shares the group's code.
            let encoding = group.shape.encoding::<EF>();
            let mut blind_messages = Vec::with_capacity(group.width);
            let mut blind_randomness = Vec::with_capacity(group.width);
            for _ in 0..group.width {
                // Uniform blind message s~'_i and fresh encoding
                // randomness r'_i, retained for the reveals below.
                let message = encoding.sample_message(rng);
                let randomness = encoding.sample_randomness(rng);
                blind_messages.push(message);
                blind_randomness.push(randomness);
            }
            // Row z of the stacked matrix holds position z of every blind.
            fresh_matrices
                .push(encoding.encode_batch_with_randomness(&blind_messages, &blind_randomness));
            fresh_groups.push((blind_messages, blind_randomness, witness));
        }
        let (fresh_mask_commitment, fresh_mask_data) = if fresh_matrices.is_empty() {
            (None, None)
        } else {
            let (commitment, data) = self.extension_mmcs.commit(fresh_matrices);
            challenger.observe(commitment.clone());
            (Some(commitment), Some(data))
        };

        // Move 2: the fresh-side claim.
        //
        //     mu_g = <g~, W> + sum_i <s~'_i, u_i>
        //
        // The relation evaluated on the fresh masks instead of the secrets.
        // Soundness hinges on mu_g being fixed before gamma is known.
        let mut masked_claim = dot_product::<EF, _, _>(
            fresh_message.iter().copied(),
            source_covector.iter().copied(),
        );
        for (blind_messages, _, witness) in &fresh_groups {
            for (message, covector) in blind_messages.iter().zip(witness.covectors) {
                masked_claim +=
                    dot_product::<EF, _, _>(message.iter().copied(), covector.iter().copied());
            }
        }
        challenger.observe_algebra_element(masked_claim);

        // Move 3: the blinding challenge, bound to every commitment above.
        let gamma: EF = challenger.sample_algebra_element();

        // Move 4: the one-time-pad reveals.
        //
        //     reveal = fresh + gamma * secret
        //
        // Uniform fresh, used once: the reveal is uniform and leaks nothing (Lemma 7.3).
        let blind = |fresh: &[EF], hidden: &[EF]| -> Vec<EF> {
            fresh
                .iter()
                .zip(hidden)
                .map(|(&fresh, &hidden)| fresh + gamma * hidden)
                .collect()
        };
        // Source reveals: f* = g~ + gamma * f and r* = r_g + gamma * r.
        let blinded_message = blind(&fresh_message, source_message);
        let blinded_randomness = blind(&fresh_randomness, source_randomness);
        challenger.observe_algebra_slice(&blinded_message);
        challenger.observe_algebra_slice(&blinded_randomness);
        // Mask reveals:
        // - xi*_i = s~'_i + gamma * xi_i,
        // - the analogous r*_i for each mask's encoding randomness.
        let mut blinded_masks = Vec::new();
        for (blind_messages, blind_randomness, witness) in &fresh_groups {
            for ((message, randomness), (hidden_message, hidden_randomness)) in blind_messages
                .iter()
                .zip(blind_randomness)
                .zip(witness.messages.iter().zip(witness.randomness))
            {
                let blinded = BlindedMask {
                    message: blind(message, hidden_message),
                    randomness: blind(randomness, hidden_randomness),
                };
                // Absorb each reveal before the spot positions are drawn.
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

        // Move 5a: source spot checks, t positions on the source domain.
        //
        // The verifier will recheck, per position z:
        //
        //     Enc(f*, r*)(z) = g(z) + gamma * f(z)
        //
        // so both committed sides are opened here.
        let positions = get_challenge_stir_queries::<Challenger, F>(
            code.domain_size,
            0,
            self.config.num_queries,
            challenger,
        );
        // f(z): leaves of the last committed oracle, virtually folded.
        let source_openings = open_source(&positions);
        // g(z): the fresh main mask, committed above.
        let fresh_main_openings =
            SharedProofOpening::open(self.extension_mmcs, &positions, &fresh_main_data);

        // Move 5b: mask spot checks from one global t_zk-query vector.
        //
        // The verifier will recheck, per position y and group member i:
        //
        //     Enc(xi*_i, r*_i)(y) = s'_i(y) + gamma * xi_i(y)
        //
        // A uniform index on the largest domain projects uniformly onto every
        // shorter power-of-two domain. Sampling with replacement preserves
        // independence after projection and reveals at most t_zk distinct
        // positions from any mask.
        let max_mask_domain = max_mask_domain_size(&self.config.mask_groups);
        let global_mask_positions = max_mask_domain.map(|domain_size| {
            get_challenge_iid_queries::<Challenger, F>(
                domain_size,
                self.config.mask_queries,
                challenger,
            )
        });
        let mut carried_mask_openings = Vec::with_capacity(fresh_groups.len());
        if let (Some(max_domain), Some(global_positions)) =
            (max_mask_domain, global_mask_positions.as_deref())
        {
            for (group, (_, _, witness)) in self.config.mask_groups.iter().zip(&fresh_groups) {
                let positions =
                    project_mask_positions(global_positions, max_domain, group.shape.domain_size);
                carried_mask_openings.push(SharedProofOpening::open(
                    self.extension_mmcs,
                    &positions,
                    witness.data,
                ));
            }
        }
        let fresh_mask_opening = global_mask_positions.as_deref().map(|positions| {
            SharedBatchOpening::open(
                self.extension_mmcs,
                positions,
                fresh_mask_data
                    .as_ref()
                    .expect("fresh data exists whenever mask positions do"),
            )
        });

        BaseCaseZkProof {
            fresh_main_commitment,
            fresh_mask_commitment,
            masked_claim,
            blinded_message,
            blinded_randomness,
            blinded_masks,
            pow_witness,
            source_openings,
            fresh_main_openings,
            carried_mask_openings,
            fresh_mask_opening,
        }
    }
}
