//! HVZK base-case verifier (Construction 7.2).

use alloc::vec;
use alloc::vec::Vec;
use core::iter::repeat_n;

use p3_challenger::fs::TranscriptField;
use p3_challenger::{CanObserve, CanSampleUniformBits, FieldChallenger, GrindingChallenger};
use p3_commit::{ExtensionMmcs, Mmcs};
use p3_field::{ExtensionField, TwoAdicField, dot_product};
use p3_matrix::Dimensions;
use p3_util::log2_strict_usize;

use super::config::BaseCaseZkConfig;
use super::error::BaseCaseZkError;
use crate::pcs::proof::{QueryOpenings, SharedProofOpening};
use crate::pcs::zk::proof::BaseCaseZkProof;
use crate::transcript::zk::{ZkBaseCaseShape, ZkBaseCaseVerifierTranscript};
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
    ///     0. pin every length, and the witness a zero difficulty leaves unread
    ///     1. replay the transcript    commitments, mu_g, gamma, reveals
    ///     2. target check             claim transfers onto the reveals
    ///     3. proof of work
    ///     4. source spot checks       reveals match the committed source
    ///     5. mask spot checks         reveals match the committed masks
    /// ```
    ///
    /// # Transcript
    ///
    /// The base case is a protocol of its own.
    ///
    /// It therefore seeds a driver of its own.
    ///
    /// A rejection releases that driver before the error travels to the caller.
    ///
    /// # Arguments
    ///
    /// - `mask_covectors`: flat in chronological mask order, tiled by the groups.
    /// - `verify_source`: authenticates the source openings, returns their folded values.
    ///
    /// # Errors
    ///
    /// When any check below rejects the proof.
    #[allow(clippy::too_many_arguments)]
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
        F: TranscriptField,
        Challenger: FieldChallenger<F>
            + GrindingChallenger<Witness = F>
            + CanSampleUniformBits<F>
            + CanObserve<MT::Commitment>,
    {
        // One driver spans the whole base case.
        //
        // The description is therefore walked exactly once.
        let shape = ZkBaseCaseShape::new(self.config);
        let mut transcript =
            ZkBaseCaseVerifierTranscript::<Challenger, F, EF>::new(challenger, shape);

        // A rejection leaves the driver mid-description.
        //
        // Release it before the error travels out.
        let outcome = self.replay(
            &mut transcript,
            proof,
            source_covector,
            mask_covectors,
            mask_commitments,
            target,
            verify_source,
        );
        match outcome {
            Ok(()) => {
                transcript.finish();
                Ok(())
            }
            Err(error) => {
                transcript.abort();
                Err(error)
            }
        }
    }

    /// Walk every described step of the base case against the proof.
    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    fn replay<Challenger>(
        &self,
        transcript: &mut ZkBaseCaseVerifierTranscript<'_, Challenger, F, EF>,
        proof: &BaseCaseZkProof<F, EF, MT>,
        source_covector: &[EF],
        mask_covectors: &[Vec<EF>],
        mask_commitments: &[MT::Commitment],
        target: EF,
        verify_source: impl FnOnce(
            &[usize],
            &QueryOpenings<F, EF, MT::MultiProof>,
        ) -> Result<Vec<EF>, BaseCaseZkError>,
    ) -> Result<(), BaseCaseZkError>
    where
        F: TranscriptField,
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

        // Check 0: pin every count and covector width before any transcript work.
        //
        //     count off      ->  MaskCountMismatch
        //     covector off   ->  MaskCountMismatch
        //
        // Dot products zip silently.
        //
        // Every length they read is therefore pinned up front.
        //
        // The reveal lengths are pinned where each reveal is absorbed instead.
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
        // One code shape per carried mask, in group order.
        //
        // The counts pinned above make the zip cover every covector.
        let shapes = self
            .config
            .mask_groups
            .iter()
            .flat_map(|group| repeat_n(&group.shape, group.width));
        for (covector, shape) in mask_covectors.iter().zip(shapes) {
            // Covector width must match the member's message length.
            count(covector.len(), shape.message_len)?;
        }

        // Still check 0: a zero-difficulty grind leaves the witness unread, so pin it.
        //
        //     pow_bits = 0 -> prover emits zero, verifier reads nothing -> pin it here
        //     pow_bits > 0 -> prover grinds,     verifier resamples     -> check 3 pins it
        if self.config.pow_bits == 0 && proof.pow_witness != F::ZERO {
            return Err(BaseCaseZkError::NonCanonicalPowWitness);
        }

        // Check 1: replay the prover's moves into the Fiat-Shamir sponge.
        //
        //     move 1  ->  fresh commitments g, s'_i
        //     move 2  ->  fresh-side claim mu_g
        //     move 3  ->  sample gamma (now bound to everything above)
        //     move 4  ->  reveals f*, r*, xi*_i, r*_i
        let fresh_main_commitment = &proof.fresh_main_commitment;
        transcript.fresh_commitment(fresh_main_commitment.clone());
        for commitment in &proof.fresh_mask_commitments {
            transcript.blind_commitment(commitment.clone());
        }
        transcript.claim(proof.masked_claim);
        let gamma: EF = transcript.gamma();
        transcript.reveal(&proof.blinded_message, &proof.blinded_randomness)?;
        for blinded in &proof.blinded_masks {
            transcript.reveal(&blinded.message, &blinded.randomness)?;
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
        transcript.spot_check_pow(proof.pow_witness)?;

        // Check 4: source spot checks at t sampled positions.
        //
        // The target check used the reveals as bare vectors.
        // These checks tie them to the committed oracles, per position z:
        //
        //     Enc(f*, r*)(z) = g(z) + gamma * f(z)
        let positions = transcript.source_queries();
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
            let positions = transcript.mask_queries(group_index);
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
