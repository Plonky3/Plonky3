//! HVZK base-case verifier (Construction 7.2).

use alloc::vec;
use alloc::vec::Vec;
use core::iter::repeat_n;

use p3_challenger::{CanObserve, CanSampleUniformBits, FieldChallenger, GrindingChallenger};
use p3_commit::{ExtensionMmcs, Mmcs};
use p3_field::{ExtensionField, TwoAdicField, dot_product};
use p3_matrix::Dimensions;
use p3_util::log2_strict_usize;

use super::config::BaseCaseZkConfig;
use super::error::BaseCaseZkError;
use crate::pcs::proof::{QueryOpenings, SharedProofOpening};
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
    /// - `verify_source`: authenticates the source openings, returns their folded values.
    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    pub fn verify<Challenger>(
        &self,
        proof: &BaseCaseZkProof<F, EF, MT>,
        source_covector: &[EF],
        mask_covectors: &[Vec<EF>],
        mask_commitments: &[MT::Commitment],
        target: EF,
        verify_source: impl FnOnce(
            &[usize],
            &QueryOpenings<F, EF, MT::MultiProof>,
        ) -> Result<Vec<EF>, BaseCaseZkError>,
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
        // One opened row per sampled position, for the source and the fresh mask.
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
        // f(z): authenticate the last oracle's leaves and fold them.
        let source_values = verify_source(&positions, &proof.source_openings)?;
        openings("source", source_values.len(), positions.len())?;
        // g(z): authenticate the fresh main mask openings.
        let fresh_main_dims = vec![Dimensions {
            height: code.domain_size,
            width: 1,
        }];
        let fresh_rows = self.verify_rows(
            fresh_main_commitment,
            &fresh_main_dims,
            &positions,
            &proof.fresh_main_openings,
            1,
            "fresh main",
        )?;
        for ((&position, source_value), fresh_row) in
            positions.iter().zip(&source_values).zip(fresh_rows)
        {
            // Enc(f*, r*)(z): re-encode the reveal at this position.
            let blinded_value =
                code.evaluate_at(position, &proof.blinded_message, &proof.blinded_randomness);
            if blinded_value != fresh_row[0] + gamma * *source_value {
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
        openings("mask", proof.mask_openings.len(), num_groups)?;
        let mut mask_offset = 0;
        for (group_index, (group, pair)) in self
            .config
            .mask_groups
            .iter()
            .zip(&proof.mask_openings)
            .enumerate()
        {
            let positions = get_challenge_stir_queries::<Challenger, F>(
                group.shape.domain_size,
                0,
                self.config.mask_queries,
                challenger,
            );
            let dims = vec![Dimensions {
                height: group.shape.domain_size,
                width: group.width,
            }];
            // xi_i(y) for every member i: rows of the carried oracle.
            let carried_rows = self.verify_rows(
                &mask_commitments[group_index],
                &dims,
                &positions,
                &pair.carried,
                group.width,
                "carried mask",
            )?;
            // s'_i(y) for every member i: rows of the fresh blind.
            let fresh_rows = self.verify_rows(
                &proof.fresh_mask_commitments[group_index],
                &dims,
                &positions,
                &pair.fresh,
                group.width,
                "fresh mask",
            )?;
            // Evaluation domain of the mask code over the extension field.
            let mask_gen = EF::two_adic_generator(log2_strict_usize(group.shape.domain_size));
            let blinded = &proof.blinded_masks[mask_offset..mask_offset + group.width];
            for ((&position, carried_row), fresh_row) in
                positions.iter().zip(carried_rows).zip(fresh_rows)
            {
                // The field point behind position y.
                let point = mask_gen.exp_u64(position as u64);
                for ((blinded, &carried), &fresh) in blinded.iter().zip(carried_row).zip(fresh_row)
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

    /// Verifies one extension multi-opening of the expected row width.
    ///
    /// Returns the opened rows, one per position.
    fn verify_rows<'p>(
        &self,
        commitment: &MT::Commitment,
        dims: &[Dimensions],
        positions: &[usize],
        opening: &'p SharedProofOpening<EF, MT::MultiProof>,
        width: usize,
        kind: &'static str,
    ) -> Result<&'p [Vec<EF>], BaseCaseZkError> {
        // One opened row per position; the multiproof binds rows to positions.
        if opening.rows.len() != positions.len() {
            return Err(BaseCaseZkError::OpeningCountMismatch {
                kind,
                expected: positions.len(),
                actual: opening.rows.len(),
            });
        }
        // Pin every row width locally before any caller indexes into it.
        if opening.rows.iter().any(|row| row.len() != width) {
            return Err(BaseCaseZkError::MerkleVerificationFailed { kind });
        }
        opening
            .verify(self.extension_mmcs, commitment, dims, positions)
            .map_err(|_| BaseCaseZkError::MerkleVerificationFailed { kind })?;
        Ok(&opening.rows)
    }
}
