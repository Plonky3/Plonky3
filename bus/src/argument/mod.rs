//! Commitment-independent reduction from a planned bus to terminal leaf claims.

use alloc::vec::Vec;

use p3_challenger::FieldChallenger;
use p3_challenger::fs::TranscriptField;
use p3_field::{ExtensionField, Field};
use serde::{Deserialize, Serialize};

mod error;
mod transcript;

pub use error::BusArgumentError;

use self::transcript::{BusProverTranscript, BusVerifierTranscript};
use crate::{BusDirection, BusPlan, ProductGkrOutput, ProductGkrProof};

/// Verifier challenges defining every bus leaf factor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BusChallenges<EF> {
    /// Point evaluating the padded tuple's multilinear extension.
    pub fingerprint: Vec<EF>,
    /// Random shift applied to every tuple fingerprint.
    pub offset: EF,
}

/// Product proof for one verifier-derived bus plan.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BusProof<EF> {
    /// Reduction of the push and pull product trees to one shared point.
    pub product: ProductGkrProof<EF>,
}

/// Unauthenticated terminal claims returned by the bus reduction.
#[must_use = "terminal claims remain unauthenticated until checked against PCS openings"]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BusReductionOutput<EF> {
    /// Challenges defining the reduced leaf polynomials.
    pub challenges: BusChallenges<EF>,
    /// Product roots, terminal point, and terminal leaf evaluations.
    pub product: ProductGkrOutput<EF>,
}

impl<EF: Field> BusReductionOutput<EF> {
    /// Combine the push and pull terminal identities with one random challenge.
    ///
    /// Identity padding is removed before batching.
    /// The push side has coefficient one.
    /// The pull side has the supplied coefficient.
    ///
    /// # Errors
    ///
    /// Returns an error unless the reduction contains exactly one value per direction.
    pub fn batched_terminal_claim(&self, direction_challenge: EF) -> Result<EF, BusArgumentError> {
        // Product trees are ordered push then pull by the checked public statement.
        let [push, pull] = self.product.values.as_slice() else {
            return Err(BusArgumentError::TerminalValueCount {
                expected: BusDirection::ALL.len(),
                actual: self.product.values.len(),
            });
        };

        // Subtracting one removes each tree's implicit identity-padded suffix.
        Ok((*push - EF::ONE) + direction_challenge * (*pull - EF::ONE))
    }
}

impl BusPlan {
    /// Prove the equality of this plan's materialized push and pull multisets.
    ///
    /// The returned terminal claims are not authenticated here.
    /// A caller must reconstruct them from commitment-bound evaluations.
    ///
    /// # Errors
    ///
    /// Returns an error for a wrong materialized shape or unequal products.
    pub fn prove<F, EF, Challenger>(
        &self,
        materialize: impl FnOnce(&BusChallenges<EF>) -> [Vec<EF>; 2],
        challenger: &mut Challenger,
    ) -> Result<(BusProof<EF>, BusReductionOutput<EF>), BusArgumentError>
    where
        F: TranscriptField,
        EF: ExtensionField<F>,
        Challenger: FieldChallenger<F>,
    {
        // Sample the leaf polynomial only after the surrounding commitment is bound.
        let mut transcript = BusProverTranscript::<_, F, EF>::new(challenger, self);
        let challenges = transcript.challenges();
        let [pushes, pulls] = materialize(&challenges);

        // Validate every derived witness length before product reduction allocates by shape.
        if let Err(error) = self
            .validate_leaf_count(BusDirection::Push, pushes.len())
            .and_then(|()| self.validate_leaf_count(BusDirection::Pull, pulls.len()))
        {
            transcript.abort();
            return Err(error);
        }

        // Shared-root encoding is a statement, so reject a false witness explicitly.
        let push_root = pushes.iter().copied().product::<EF>();
        let pull_root = pulls.iter().copied().product::<EF>();
        if push_root != pull_root {
            transcript.abort();
            return Err(BusArgumentError::UnbalancedProducts);
        }
        let (product, output) = transcript.product(|challenger| {
            ProductGkrProof::prove::<F, _>(
                &[pushes.as_slice(), pulls.as_slice()],
                self.product_shape(),
                challenger,
            )
        });
        transcript.finish();

        Ok((
            BusProof { product },
            BusReductionOutput {
                challenges,
                product: output,
            },
        ))
    }

    /// Verify this plan's product reduction and return unauthenticated terminal claims.
    ///
    /// # Errors
    ///
    /// Returns an error when the product proof is malformed or inconsistent.
    pub fn verify<F, EF, Challenger>(
        &self,
        proof: &BusProof<EF>,
        challenger: &mut Challenger,
    ) -> Result<BusReductionOutput<EF>, BusArgumentError>
    where
        F: TranscriptField,
        EF: ExtensionField<F>,
        Challenger: FieldChallenger<F>,
    {
        // Replay exactly the plan-derived challenge and product schedule.
        let mut transcript = BusVerifierTranscript::<_, F, EF>::new(challenger, self);
        let challenges = transcript.challenges();
        let product = match transcript.product(|challenger| {
            proof
                .product
                .verify::<F, _>(self.product_shape(), challenger)
        }) {
            Ok(product) => product,
            Err(error) => {
                transcript.abort();
                return Err(error.into());
            }
        };
        transcript.finish();

        Ok(BusReductionOutput {
            challenges,
            product,
        })
    }

    const fn validate_leaf_count(
        &self,
        direction: BusDirection,
        actual: usize,
    ) -> Result<(), BusArgumentError> {
        // Physical blocks exactly partition the explicit prefix on each side.
        let expected = self.security_geometry().non_padding_leaf_count(direction);
        if actual != expected {
            return Err(BusArgumentError::LeafCountMismatch {
                direction,
                expected,
                actual,
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;

    use super::*;

    /// Build one reduction output with a chosen terminal-value vector.
    fn output(values: Vec<BabyBear>) -> BusReductionOutput<BabyBear> {
        // Roots and points are irrelevant to this final direction-batching identity.
        BusReductionOutput {
            challenges: BusChallenges {
                fingerprint: Vec::new(),
                offset: BabyBear::ZERO,
            },
            product: ProductGkrOutput {
                roots: Vec::new(),
                point: Vec::new(),
                values,
            },
        }
    }

    #[test]
    fn terminal_claim_batches_push_then_pull_after_removing_identity_padding() {
        // The two tree evaluations are 5 and 7 before subtracting their identity suffixes.
        let reduction = output(vec![BabyBear::from_u8(5), BabyBear::from_u8(7)]);
        let challenge = BabyBear::from_u8(11);

        // Expected claim: (5 - 1) + 11 * (7 - 1).
        assert_eq!(
            reduction.batched_terminal_claim(challenge),
            Ok(BabyBear::from_u8(4) + challenge * BabyBear::from_u8(6))
        );
    }

    #[test]
    fn terminal_claim_rejects_every_noncanonical_value_count() {
        // Missing and extra values have no push-then-pull interpretation.
        for actual in [0, 1, 3] {
            let reduction = output(vec![BabyBear::ONE; actual]);
            assert_eq!(
                reduction.batched_terminal_claim(BabyBear::TWO),
                Err(BusArgumentError::TerminalValueCount {
                    expected: 2,
                    actual,
                })
            );
        }
    }
}
