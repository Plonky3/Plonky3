use alloc::vec::Vec;

use p3_commit::Pcs;
use serde::{Deserialize, Serialize};

use crate::security::{ConjecturedSecurity, LegacySecurity, ProvenSecurity, StarkSecurityParams};
use crate::{Com, StarkGenericConfig, Val};

type PcsProof<SC> = <<SC as StarkGenericConfig>::Pcs as Pcs<
    <SC as StarkGenericConfig>::Challenge,
    <SC as StarkGenericConfig>::Challenger,
>>::Proof;

#[derive(Serialize, Deserialize)]
#[serde(bound = "")]
pub struct Proof<SC: StarkGenericConfig> {
    pub commitments: Commitments<Com<SC>>,
    pub opened_values: OpenedValues<SC::Challenge>,
    pub opening_proof: PcsProof<SC>,
    pub degree_bits: usize,
    /// Proof of work for the phase before the out-of-domain point is sampled.
    ///
    /// Trivially valid (and unread) when
    /// [`StarkGenericConfig::ood_proof_of_work_bits`] is `0`.
    pub ood_pow_witness: Val<SC>,
}

impl<SC: StarkGenericConfig> Proof<SC> {
    /// Legacy conjectured security level (in bits).
    ///
    /// For historical comparison only: this may exceed [`ConjecturedSecurity`]
    /// and is not a soundness bound. Do not use it to size deployment parameters.
    ///
    /// See [`LegacySecurity`].
    pub fn legacy_security(&self, params: &StarkSecurityParams) -> LegacySecurity {
        LegacySecurity::compute_from_params(params, self.degree_bits)
    }

    /// Conjectured security level (in bits).
    ///
    /// See [`ConjecturedSecurity`].
    pub fn conjectured_security(&self, params: &StarkSecurityParams) -> ConjecturedSecurity {
        ConjecturedSecurity::compute_from_params(params, self.degree_bits)
    }

    /// Proven security level (in bits).
    ///
    /// See [`ProvenSecurity`].
    pub fn proven_security(&self, params: &StarkSecurityParams) -> ProvenSecurity {
        ProvenSecurity::compute_from_proof(self.degree_bits, params)
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Commitments<Com> {
    pub trace: Com,
    pub quotient_chunks: Com,
    pub random: Option<Com>,
}

#[derive(Debug)]
pub struct OpenedValues<Challenge> {
    pub trace_local: Vec<Challenge>,
    /// Main trace evaluated at `g * zeta`.
    ///
    /// `None` when the AIR has no transition constraints and does not access the next row.
    pub trace_next: Option<Vec<Challenge>>,
    /// Preprocessed trace openings.
    ///
    /// `None` when the AIR has no preprocessed columns.
    pub preprocessed: Option<PreprocessedOpenedValues<Challenge>>,
    pub quotient_chunks: Vec<Vec<Challenge>>,
    pub random: Option<Vec<Challenge>>,
}

/// The preprocessed trace evaluated at the out-of-domain point.
///
/// A present opening always carries the current row, so a next row without a current row
/// cannot be built. The next row is opened only when the AIR reads it, and then it is as
/// wide as the current row; the `Deserialize` impl of [`OpenedValues`] rejects anything else.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreprocessedOpenedValues<Challenge> {
    /// Preprocessed trace evaluated at `zeta`.
    pub local: Vec<Challenge>,
    /// Preprocessed trace evaluated at `g * zeta`.
    ///
    /// `None` when the AIR does not read the next preprocessed row.
    pub next: Option<Vec<Challenge>>,
}

impl<Challenge> OpenedValues<Challenge> {
    /// The preprocessed current-row opening, if the proof carries one.
    pub fn preprocessed_local(&self) -> Option<&[Challenge]> {
        self.preprocessed.as_ref().map(|p| p.local.as_slice())
    }

    /// The preprocessed next-row opening, if the proof carries one.
    pub fn preprocessed_next(&self) -> Option<&[Challenge]> {
        self.preprocessed.as_ref().and_then(|p| p.next.as_deref())
    }
}

/// The wire layout of [`OpenedValues`]: the two preprocessed rows travel as separate optional
/// fields, as they did before they were grouped, so encoded proofs keep their shape.
#[derive(Deserialize)]
#[serde(rename = "OpenedValues")]
#[serde(bound(deserialize = "Challenge: Deserialize<'de>"))]
struct OpenedValuesRepr<Challenge> {
    trace_local: Vec<Challenge>,
    trace_next: Option<Vec<Challenge>>,
    preprocessed_local: Option<Vec<Challenge>>,
    preprocessed_next: Option<Vec<Challenge>>,
    quotient_chunks: Vec<Vec<Challenge>>,
    random: Option<Vec<Challenge>>,
}

/// Borrowing twin of [`OpenedValuesRepr`] for serialization; slices encode like vectors.
#[derive(Serialize)]
#[serde(rename = "OpenedValues")]
#[serde(bound(serialize = "Challenge: Serialize"))]
struct OpenedValuesReprRef<'a, Challenge> {
    trace_local: &'a [Challenge],
    trace_next: Option<&'a [Challenge]>,
    preprocessed_local: Option<&'a [Challenge]>,
    preprocessed_next: Option<&'a [Challenge]>,
    quotient_chunks: &'a [Vec<Challenge>],
    random: Option<&'a [Challenge]>,
}

impl<Challenge: Serialize> Serialize for OpenedValues<Challenge> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        OpenedValuesReprRef {
            trace_local: &self.trace_local,
            trace_next: self.trace_next.as_deref(),
            preprocessed_local: self.preprocessed_local(),
            preprocessed_next: self.preprocessed_next(),
            quotient_chunks: &self.quotient_chunks,
            random: self.random.as_deref(),
        }
        .serialize(serializer)
    }
}

impl<'de, Challenge: Deserialize<'de>> Deserialize<'de> for OpenedValues<Challenge> {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        use serde::de::Error;

        let repr = OpenedValuesRepr::<Challenge>::deserialize(deserializer)?;
        let preprocessed = match (repr.preprocessed_local, repr.preprocessed_next) {
            (None, None) => None,
            (None, Some(_)) => {
                return Err(D::Error::custom(
                    "preprocessed next-row opening without a current-row opening",
                ));
            }
            (Some(local), next) => {
                if local.is_empty() {
                    return Err(D::Error::custom("empty preprocessed current-row opening"));
                }
                if let Some(next) = &next
                    && next.len() != local.len()
                {
                    return Err(D::Error::invalid_length(
                        next.len(),
                        &"a preprocessed next-row opening as wide as the current row",
                    ));
                }
                Some(PreprocessedOpenedValues { local, next })
            }
        };
        Ok(Self {
            trace_local: repr.trace_local,
            trace_next: repr.trace_next,
            preprocessed,
            quotient_chunks: repr.quotient_chunks,
            random: repr.random,
        })
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;

    use super::*;

    type F = BabyBear;

    fn opened_values(preprocessed: Option<PreprocessedOpenedValues<F>>) -> OpenedValues<F> {
        OpenedValues {
            trace_local: vec![F::ONE, F::TWO],
            trace_next: Some(vec![F::TWO, F::ONE]),
            preprocessed,
            quotient_chunks: vec![vec![F::ONE]],
            random: None,
        }
    }

    /// The grouped opening encodes exactly like the two separate optional rows it replaced.
    fn flat_encoding(local: Option<Vec<F>>, next: Option<Vec<F>>) -> Vec<u8> {
        #[derive(Serialize)]
        struct Flat {
            trace_local: Vec<F>,
            trace_next: Option<Vec<F>>,
            preprocessed_local: Option<Vec<F>>,
            preprocessed_next: Option<Vec<F>>,
            quotient_chunks: Vec<Vec<F>>,
            random: Option<Vec<F>>,
        }
        postcard::to_allocvec(&Flat {
            trace_local: vec![F::ONE, F::TWO],
            trace_next: Some(vec![F::TWO, F::ONE]),
            preprocessed_local: local,
            preprocessed_next: next,
            quotient_chunks: vec![vec![F::ONE]],
            random: None,
        })
        .unwrap()
    }

    #[test]
    fn preprocessed_opening_keeps_the_flat_wire_layout() {
        let cases = [
            (None, None, None),
            (
                Some(PreprocessedOpenedValues {
                    local: vec![F::ONE],
                    next: None,
                }),
                Some(vec![F::ONE]),
                None,
            ),
            (
                Some(PreprocessedOpenedValues {
                    local: vec![F::ONE, F::TWO],
                    next: Some(vec![F::TWO, F::ONE]),
                }),
                Some(vec![F::ONE, F::TWO]),
                Some(vec![F::TWO, F::ONE]),
            ),
        ];
        for (grouped, local, next) in cases {
            let values = opened_values(grouped.clone());
            let bytes = postcard::to_allocvec(&values).unwrap();
            assert_eq!(bytes, flat_encoding(local, next));
            let back: OpenedValues<F> = postcard::from_bytes(&bytes).unwrap();
            assert_eq!(back.preprocessed, grouped);
            assert_eq!(back.trace_local, values.trace_local);
        }
    }

    #[test]
    fn preprocessed_opening_rejects_shapes_the_prover_never_emits() {
        // A next row without a current row, an empty current row, and a next row narrower
        // than the current row each fail at the boundary instead of reaching the verifier.
        for (local, next) in [
            (None, Some(vec![F::ONE])),
            (Some(vec![]), None),
            (Some(vec![F::ONE, F::TWO]), Some(vec![F::ONE])),
        ] {
            let bytes = flat_encoding(local, next);
            assert!(postcard::from_bytes::<OpenedValues<F>>(&bytes).is_err());
        }
    }
}
