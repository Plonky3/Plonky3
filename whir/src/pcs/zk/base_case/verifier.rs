//! HVZK base-case verifier (Construction 7.2).

use alloc::vec;
use alloc::vec::Vec;
use core::iter::repeat_n;
use core::slice::from_ref;

use p3_challenger::{CanObserve, CanSampleUniformBits, FieldChallenger, GrindingChallenger};
use p3_commit::{BatchOpeningRef, ExtensionMmcs, Mmcs};
use p3_field::{ExtensionField, TwoAdicField, dot_product};
use p3_matrix::Dimensions;
use p3_util::log2_strict_usize;

use super::config::BaseCaseZkConfig;
use super::error::BaseCaseZkError;
use crate::pcs::proof::QueryOpening;
use crate::pcs::utils::get_challenge_stir_queries;
use crate::pcs::zk::proof::BaseCaseZkProof;
use crate::utils::padded_ood_t1;

/// HVZK base-case verifier (Construction 7.2).
pub struct BaseCaseZkVerifier<'a, F, EF, MT>
where
    F: TwoAdicField,
    EF: ExtensionField<F>,
    MT: Mmcs<F>,
{
    /// Protocol shape shared with the prover.
    pub config: &'a BaseCaseZkConfig<F>,
    /// Commitment scheme verifying the fresh masks and mask openings.
    pub extension_mmcs: &'a ExtensionMmcs<F, EF, MT>,
}

impl<F, EF, MT> BaseCaseZkVerifier<'_, F, EF, MT>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
    MT: Mmcs<F>,
{
    /// Replays Construction 7.2 against the carried claim.
    ///
    /// # Checks
    ///
    /// ```text
    ///     0. pin every length of the statement and the proof
    ///     1. replay the transcript    commitments, mu_g, gamma, reveals
    ///     2. target check             claim transfers onto the reveals
    ///     3. proof of work
    ///     4. source spot checks       reveals match the committed source
    ///     5. mask spot checks         reveals match the committed masks
    /// ```
    ///
    /// # Arguments
    ///
    /// - `mask_covectors`: flat in chronological mask order, tiled by the groups.
    /// - `verify_source`: authenticates one source opening, returns its folded value.
    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    pub fn verify<Challenger>(
        &self,
        proof: &BaseCaseZkProof<F, EF, MT>,
        source_covector: &[EF],
        mask_covectors: &[Vec<EF>],
        mask_commitments: &[MT::Commitment],
        target: EF,
        mut verify_source: impl FnMut(
            usize,
            &QueryOpening<F, EF, MT::Proof>,
        ) -> Result<EF, BaseCaseZkError>,
        challenger: &mut Challenger,
    ) -> Result<(), BaseCaseZkError>
    where
        Challenger: FieldChallenger<F>
            + GrindingChallenger<Witness = F>
            + CanSampleUniformBits<F>
            + CanObserve<MT::Commitment>,
    {
        let code = &self.config.code;
        let num_groups = self.config.mask_groups.len();
        // Total carried masks: the groups tile the flat mask list.
        let num_masks: usize = self
            .config
            .mask_groups
            .iter()
            .map(|group| group.width)
            .sum();

        // Check 0: pin every length before any transcript work.
        //
        //     count off    ->  MaskCountMismatch
        //     reveal off   ->  BlindedLengthMismatch
        //
        // Dot products zip silently, so every length is pinned up front.
        let count = |actual: usize, expected: usize| {
            if actual == expected {
                Ok(())
            } else {
                Err(BaseCaseZkError::MaskCountMismatch { expected, actual })
            }
        };
        // Top-level statement and proof vector counts.
        count(mask_covectors.len(), num_masks)?;
        count(mask_commitments.len(), num_groups)?;
        count(source_covector.len(), code.message_len)?;
        count(proof.blinded_masks.len(), num_masks)?;
        count(proof.fresh_mask_commitments.len(), num_groups)?;
        // Per-mask reveal and covector lengths against the group's code.
        let blinded = |kind, actual: usize, expected: usize| {
            if actual == expected {
                Ok(())
            } else {
                Err(BaseCaseZkError::BlindedLengthMismatch {
                    kind,
                    expected,
                    actual,
                })
            }
        };
        blinded("message", proof.blinded_message.len(), code.message_len)?;
        blinded(
            "randomness",
            proof.blinded_randomness.len(),
            code.randomness_len,
        )?;
        // One shape per carried mask, in group order; the counts pinned above
        // guarantee both zips cover every covector and reveal.
        let shapes = self
            .config
            .mask_groups
            .iter()
            .flat_map(|group| repeat_n(&group.shape, group.width));
        for ((covector, mask), shape) in mask_covectors.iter().zip(&proof.blinded_masks).zip(shapes)
        {
            // Covector width must match the member's message length.
            count(covector.len(), shape.message_len)?;
            // Reveals must match the member's message and randomness.
            blinded("mask message", mask.message.len(), shape.message_len)?;
            blinded(
                "mask randomness",
                mask.randomness.len(),
                shape.randomness_len,
            )?;
        }

        // Check 1: replay the prover's moves into the Fiat-Shamir sponge.
        //
        //     move 1  ->  fresh commitments g, s'_i
        //     move 2  ->  fresh-side claim mu_g
        //     move 3  ->  sample gamma (now bound to everything above)
        //     move 4  ->  reveals f*, r*, xi*_i, r*_i
        let fresh_main_commitment = &proof.fresh_main_commitment;
        challenger.observe(fresh_main_commitment.clone());
        for commitment in &proof.fresh_mask_commitments {
            challenger.observe(commitment.clone());
        }
        challenger.observe_algebra_element(proof.masked_claim);
        let gamma: EF = challenger.sample_algebra_element();
        challenger.observe_algebra_slice(&proof.blinded_message);
        challenger.observe_algebra_slice(&proof.blinded_randomness);
        for blinded in &proof.blinded_masks {
            challenger.observe_algebra_slice(&blinded.message);
            challenger.observe_algebra_slice(&blinded.randomness);
        }

        // Check 2: the joint target identity.
        //
        //     <f*, W> + sum_i <xi*_i, u_i>  =  mu_g + gamma * target
        //
        // By linearity, the identity holds when both claims hold.
        // A cheating mu_g was fixed before gamma.
        // A false claim therefore survives at most one gamma.
        let mut combined = dot_product::<EF, _, _>(
            proof.blinded_message.iter().copied(),
            source_covector.iter().copied(),
        );
        for (blinded, covector) in proof.blinded_masks.iter().zip(mask_covectors) {
            combined +=
                dot_product::<EF, _, _>(blinded.message.iter().copied(), covector.iter().copied());
        }
        if combined != proof.masked_claim + gamma * target {
            return Err(BaseCaseZkError::TargetCheckFailed);
        }

        // Check 3: proof of work before the spot positions are drawn.
        if self.config.pow_bits > 0
            && !challenger.check_witness(self.config.pow_bits, proof.pow_witness)
        {
            return Err(BaseCaseZkError::InvalidPowWitness);
        }

        // Check 4: source spot checks at t sampled positions.
        //
        // The target check used the reveals as bare vectors.
        // These checks tie them to the committed oracles, per position z:
        //
        //     Enc(f*, r*)(z) = g(z) + gamma * f(z)
        let positions = get_challenge_stir_queries::<Challenger, F>(
            code.domain_size,
            0,
            self.config.num_queries,
            challenger,
        );
        // One opening per sampled position, for the source and the fresh mask.
        let openings = |kind, actual: usize, expected: usize| {
            if actual == expected {
                Ok(())
            } else {
                Err(BaseCaseZkError::OpeningCountMismatch {
                    kind,
                    expected,
                    actual,
                })
            }
        };
        openings("source", proof.source_queries.len(), positions.len())?;
        openings(
            "fresh main",
            proof.fresh_main_queries.len(),
            positions.len(),
        )?;
        let fresh_main_dims = vec![Dimensions {
            height: code.domain_size,
            width: 1,
        }];
        for ((&position, source_opening), fresh_opening) in positions
            .iter()
            .zip(&proof.source_queries)
            .zip(&proof.fresh_main_queries)
        {
            // f(z): authenticate a leaf of the last oracle and fold it.
            let source_value = verify_source(position, source_opening)?;
            // g(z): authenticate the fresh main mask opening.
            let fresh_row = self.verify_row(
                fresh_main_commitment,
                &fresh_main_dims,
                position,
                fresh_opening,
                1,
                "fresh main",
            )?;
            // Enc(f*, r*)(z): re-encode the reveal at this position.
            let blinded_value =
                code.evaluate_at(position, &proof.blinded_message, &proof.blinded_randomness);
            if blinded_value != fresh_row[0] + gamma * source_value {
                return Err(BaseCaseZkError::SourceSpotCheckFailed { position });
            }
        }

        // Check 5: mask spot checks at t_zk positions per group.
        //
        // Same equation as check 4, per group member i and position y:
        //
        //     Enc(xi*_i, r*_i)(y) = s'_i(y) + gamma * xi_i(y)
        //
        // Positions are shared across a group: one opened row of each
        // oracle serves every member.
        openings("mask", proof.mask_queries.len(), num_groups)?;
        let mut mask_offset = 0;
        for (group_index, (group, pairs)) in self
            .config
            .mask_groups
            .iter()
            .zip(&proof.mask_queries)
            .enumerate()
        {
            let positions = get_challenge_stir_queries::<Challenger, F>(
                group.shape.domain_size,
                0,
                self.config.mask_queries,
                challenger,
            );
            openings("mask", pairs.len(), positions.len())?;
            let dims = vec![Dimensions {
                height: group.shape.domain_size,
                width: group.width,
            }];
            // Evaluation domain of the mask code over the extension field.
            let mask_gen = EF::two_adic_generator(log2_strict_usize(group.shape.domain_size));
            let blinded = &proof.blinded_masks[mask_offset..mask_offset + group.width];
            for (&position, pair) in positions.iter().zip(pairs) {
                // xi_i(y) for every member i: one row of the carried oracle.
                let carried_row = self.verify_row(
                    &mask_commitments[group_index],
                    &dims,
                    position,
                    &pair.carried,
                    group.width,
                    "carried mask",
                )?;
                // s'_i(y) for every member i: one row of the fresh blind.
                let fresh_row = self.verify_row(
                    &proof.fresh_mask_commitments[group_index],
                    &dims,
                    position,
                    &pair.fresh,
                    group.width,
                    "fresh mask",
                )?;
                // The field point behind position y.
                let point = mask_gen.exp_u64(position as u64);
                for ((blinded, &carried), &fresh) in
                    blinded.iter().zip(&carried_row).zip(&fresh_row)
                {
                    // Enc(xi*_i, r*_i)(y): re-encode member i's reveal.
                    let blinded_value = padded_ood_t1(point, &blinded.message, &blinded.randomness);
                    if blinded_value != fresh + gamma * carried {
                        return Err(BaseCaseZkError::MaskSpotCheckFailed {
                            group: group_index,
                            position,
                        });
                    }
                }
            }
            mask_offset += group.width;
        }

        Ok(())
    }

    /// Verifies one extension opening of the expected width and returns its row.
    fn verify_row(
        &self,
        commitment: &MT::Commitment,
        dims: &[Dimensions],
        position: usize,
        opening: &QueryOpening<F, EF, MT::Proof>,
        width: usize,
        kind: &'static str,
    ) -> Result<Vec<EF>, BaseCaseZkError> {
        // Mask oracles are extension-valued; any other variant is malformed.
        let QueryOpening::Extension { values, proof } = opening else {
            return Err(BaseCaseZkError::MerkleVerificationFailed { kind, position });
        };
        // Pin the row width locally before any caller indexes into it.
        if values.len() != width {
            return Err(BaseCaseZkError::MerkleVerificationFailed { kind, position });
        }
        self.extension_mmcs
            .verify_batch(
                commitment,
                dims,
                position,
                BatchOpeningRef {
                    opened_values: from_ref(values),
                    opening_proof: proof,
                },
            )
            .map_err(|_| BaseCaseZkError::MerkleVerificationFailed { kind, position })?;
        Ok(values.clone())
    }
}
