//! Dense mask-side claim tracking.

use alloc::vec::Vec;

use p3_field::{Field, dot_product};

/// Dense linear claims carried against the mask oracles.
///
/// - One covector per mask oracle, in commitment order.
/// - The accumulated scale is folded into the covector entries directly.
#[derive(Debug, Clone, Default)]
pub struct MaskClaims<EF> {
    /// Per-oracle dense covectors over the mask messages.
    pub covectors: Vec<Vec<EF>>,
}

impl<EF: Field> MaskClaims<EF> {
    /// Starts with no mask oracles.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            covectors: Vec::new(),
        }
    }

    /// Scales every covector by `eps * 2^{-k}` after a `k`-round batch.
    ///
    /// # Why this scale
    ///
    /// The masked sumcheck carries the mask-claim total as a constant on
    /// each round's wire polynomial:
    ///
    /// ```text
    ///     eps  ->  the batch rescales the whole residual relation
    ///     2^-k ->  a constant loses a factor 2 per round, since h(0) + h(1) doubles it
    /// ```
    ///
    /// The covectors must absorb the same factor as the values they pair
    /// with, or the carried relation drifts.
    ///
    /// Returns the applied scale, so a running claim total can track it.
    pub fn absorb_sumcheck(&mut self, eps: EF, folding_factor: usize) -> EF {
        let scale = eps * EF::TWO.exp_u64(folding_factor as u64).inverse();
        for covector in &mut self.covectors {
            for entry in covector.iter_mut() {
                *entry *= scale;
            }
        }
        scale
    }

    /// Appends a fresh mask oracle claim.
    pub fn push(&mut self, covector: Vec<EF>) {
        self.covectors.push(covector);
    }

    /// Evaluates the aux total against concrete mask messages.
    ///
    /// Prover-side helper.
    /// The verifier never holds the messages.
    #[must_use]
    pub fn evaluate(&self, messages: &[Vec<EF>]) -> EF {
        assert_eq!(messages.len(), self.covectors.len());
        self.covectors
            .iter()
            .zip(messages)
            .map(|(covector, message)| {
                dot_product::<EF, _, _>(covector.iter().copied(), message.iter().copied())
            })
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;
    use p3_field::extension::BinomialExtensionField;

    use super::*;

    type EF = BinomialExtensionField<BabyBear, 4>;

    #[test]
    fn mask_claims_absorb_sumcheck_applies_eps_over_2k() {
        // After a k = 2 batch with eps = 8, scale = 8 / 4 = 2.
        let mut masks = MaskClaims::new();
        masks.push(vec![EF::from_u64(1), EF::from_u64(2)]);
        masks.push(vec![EF::from_u64(3)]);
        // The returned scale is eps / 2^k = 8 / 4 = 2.
        let scale = masks.absorb_sumcheck(EF::from_u64(8), 2);
        assert_eq!(scale, EF::from_u64(2));
        assert_eq!(masks.covectors[0], vec![EF::from_u64(2), EF::from_u64(4)]);
        assert_eq!(masks.covectors[1], vec![EF::from_u64(6)]);
    }
}
