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

/// Every committed polynomial evaluated at the out-of-domain point.
///
/// The opening argument binds each vector here to the commitment it came from.
#[derive(Debug)]
pub struct OpenedValues<Challenge> {
    /// Main trace evaluated at `zeta`.
    pub trace_local: Vec<Challenge>,
    /// Main trace evaluated at `g * zeta`.
    ///
    /// `None` when the AIR has no transition constraints and does not access the next row.
    pub trace_next: Option<Vec<Challenge>>,
    /// Preprocessed trace openings, carried as one unit so the two rows cannot drift apart.
    ///
    /// `None` when the AIR has no preprocessed columns.
    pub preprocessed: Option<PreprocessedOpenedValues<Challenge>>,
    /// One evaluation vector per quotient chunk, in the order the chunks were committed.
    pub quotient_chunks: Vec<Vec<Challenge>>,
    /// Masking polynomial evaluated at `zeta`.
    ///
    /// `None` unless the commitment scheme proves in zero knowledge.
    pub random: Option<Vec<Challenge>>,
}

impl<Challenge> OpenedValues<Challenge> {
    /// The preprocessed current-row opening, if the proof carries one.
    #[must_use]
    pub fn preprocessed_local(&self) -> Option<&[Challenge]> {
        self.preprocessed.as_ref().map(|p| p.local.as_slice())
    }

    /// The preprocessed next-row opening, if the proof carries one.
    ///
    /// Absent when there is no preprocessed opening at all.
    ///
    /// Absent too when the AIR reads only the current preprocessed row.
    #[must_use]
    pub fn preprocessed_next(&self) -> Option<&[Challenge]> {
        self.preprocessed.as_ref().and_then(|p| p.next.as_deref())
    }
}

/// The wire layout of the openings, with the two preprocessed rows as separate optional fields.
///
/// The row container is left generic so one definition covers both directions.
///
/// Serializing fills it with slices borrowed from the openings.
///
/// Deserializing fills it with owned vectors.
///
/// A slice encodes exactly like the vector it borrows from.
#[derive(Serialize, Deserialize)]
#[serde(rename = "OpenedValues")]
struct OpenedValuesRepr<Row, Chunks> {
    /// Main trace evaluated at `zeta`.
    trace_local: Row,
    /// Main trace evaluated at `g * zeta`, absent when the AIR never reads the next row.
    trace_next: Option<Row>,
    /// Preprocessed trace evaluated at `zeta`, absent when the AIR has no preprocessed columns.
    preprocessed_local: Option<Row>,
    /// Preprocessed trace evaluated at `g * zeta`, absent when only the current row is read.
    preprocessed_next: Option<Row>,
    /// One evaluation vector per quotient chunk, in the order the chunks were committed.
    quotient_chunks: Chunks,
    /// Masking polynomial evaluated at `zeta`, absent outside zero knowledge.
    random: Option<Row>,
}

impl<Challenge: Serialize> Serialize for OpenedValues<Challenge> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        OpenedValuesRepr {
            trace_local: self.trace_local.as_slice(),
            trace_next: self.trace_next.as_deref(),
            preprocessed_local: self.preprocessed_local(),
            preprocessed_next: self.preprocessed_next(),
            quotient_chunks: self.quotient_chunks.as_slice(),
            random: self.random.as_deref(),
        }
        .serialize(serializer)
    }
}

impl<'de, Challenge: Deserialize<'de>> Deserialize<'de> for OpenedValues<Challenge> {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        use serde::de::Error;

        let repr =
            OpenedValuesRepr::<Vec<Challenge>, Vec<Vec<Challenge>>>::deserialize(deserializer)?;

        // Refuse the preprocessed shapes no prover emits, before the verifier sees them.
        //
        //     next without local -> a row no opening claim covers
        //     empty local        -> zero columns, yet still a present opening
        //     widths disagree    -> two rows that cannot stack into one constraint window
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

/// The preprocessed trace evaluated at the out-of-domain point.
///
/// The current row is unconditional.
///
/// A next row without a current row is therefore unrepresentable.
///
/// The next row is opened only when the AIR reads it.
///
/// It is then exactly as wide as the current row.
///
/// Decoding refuses anything else.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreprocessedOpenedValues<Challenge> {
    /// Preprocessed trace evaluated at `zeta`.
    pub local: Vec<Challenge>,
    /// Preprocessed trace evaluated at `g * zeta`.
    ///
    /// `None` when the AIR does not read the next preprocessed row.
    pub next: Option<Vec<Challenge>>,
}

#[cfg(test)]
mod tests {
    use alloc::string::ToString;
    use alloc::vec;

    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;

    use super::*;

    type F = BabyBear;

    // The six flat fields the encoded proof carries, kept here as the reference layout.
    #[derive(Serialize)]
    #[serde(rename = "OpenedValues")]
    struct Flat {
        trace_local: Vec<F>,
        trace_next: Option<Vec<F>>,
        preprocessed_local: Option<Vec<F>>,
        preprocessed_next: Option<Vec<F>>,
        quotient_chunks: Vec<Vec<F>>,
        random: Option<Vec<F>>,
    }

    // Spread a grouped opening back over the six flat fields, field for field.
    fn flatten(values: &OpenedValues<F>) -> Flat {
        Flat {
            trace_local: values.trace_local.clone(),
            trace_next: values.trace_next.clone(),
            preprocessed_local: values.preprocessed_local().map(<[F]>::to_vec),
            preprocessed_next: values.preprocessed_next().map(<[F]>::to_vec),
            quotient_chunks: values.quotient_chunks.clone(),
            random: values.random.clone(),
        }
    }

    // Build one opening around the preprocessed shape under test.
    fn opened_values(
        preprocessed: Option<PreprocessedOpenedValues<F>>,
        trace_next: Option<Vec<F>>,
        random: Option<Vec<F>>,
    ) -> OpenedValues<F> {
        OpenedValues {
            trace_local: vec![F::ONE, F::TWO],
            trace_next,
            preprocessed,
            quotient_chunks: vec![vec![F::ONE], vec![F::TWO]],
            random,
        }
    }

    // Encode the six flat fields directly, including the shapes the grouped type forbids.
    fn flat_encoding(local: Option<Vec<F>>, next: Option<Vec<F>>) -> Vec<u8> {
        postcard::to_allocvec(&Flat {
            trace_local: vec![F::ONE, F::TWO],
            trace_next: Some(vec![F::TWO, F::ONE]),
            preprocessed_local: local,
            preprocessed_next: next,
            quotient_chunks: vec![vec![F::ONE], vec![F::TWO]],
            random: None,
        })
        .unwrap()
    }

    #[test]
    fn preprocessed_opening_keeps_the_flat_wire_layout() {
        // The three shapes a prover can emit:
        //
        //     no preprocessed columns -> local absent, next absent
        //     current row only        -> local present, next absent
        //     both rows               -> local present, next of equal width
        let shapes = [
            None,
            Some(PreprocessedOpenedValues {
                local: vec![F::ONE],
                next: None,
            }),
            Some(PreprocessedOpenedValues {
                local: vec![F::ONE, F::TWO],
                next: Some(vec![F::TWO, F::ONE]),
            }),
        ];

        // Vary the two unrelated optional fields as well.
        // Their encoding is then pinned by the same comparison.
        for trace_next in [None, Some(vec![F::TWO, F::ONE])] {
            for random in [None, Some(vec![F::ONE])] {
                for shape in shapes.clone() {
                    let values = opened_values(shape.clone(), trace_next.clone(), random.clone());
                    let flat = flatten(&values);

                    // Postcard is the proof format.
                    // It writes the fields in order and never writes their names.
                    let bytes = postcard::to_allocvec(&values).unwrap();
                    assert_eq!(bytes, postcard::to_allocvec(&flat).unwrap());

                    // JSON is self-describing.
                    // This comparison is therefore what pins the six field names.
                    assert_eq!(
                        serde_json::to_string(&values).unwrap(),
                        serde_json::to_string(&flat).unwrap()
                    );

                    // Decoding the flat bytes rebuilds the grouped opening unchanged.
                    let back: OpenedValues<F> = postcard::from_bytes(&bytes).unwrap();
                    assert_eq!(back.preprocessed, shape);
                    assert_eq!(back.trace_local, values.trace_local);
                    assert_eq!(back.trace_next, values.trace_next);
                    assert_eq!(back.quotient_chunks, values.quotient_chunks);
                    assert_eq!(back.random, values.random);
                }
            }
        }
    }

    #[test]
    fn preprocessed_opening_rejects_a_next_row_without_a_current_row() {
        // Mutation: drop the current row and keep the next one.
        //
        //     local: absent
        //     next:  [1]      -> a row no opening claim covers
        let bytes = flat_encoding(None, Some(vec![F::ONE]));
        assert!(postcard::from_bytes::<OpenedValues<F>>(&bytes).is_err());
    }

    #[test]
    fn preprocessed_opening_rejects_an_empty_current_row() {
        // Mutation: keep the current row present but give it zero columns.
        //
        //     local: []       -> present, yet nothing in the opening argument binds it
        let bytes = flat_encoding(Some(vec![]), None);
        assert!(postcard::from_bytes::<OpenedValues<F>>(&bytes).is_err());
    }

    #[test]
    fn preprocessed_opening_rejects_rows_of_different_widths() {
        // Mutation: make the next row narrower than the current one.
        //
        //     local: [1, 2]
        //     next:  [1]      -> the two rows cannot stack into one constraint window
        let bytes = flat_encoding(Some(vec![F::ONE, F::TWO]), Some(vec![F::ONE]));
        assert!(postcard::from_bytes::<OpenedValues<F>>(&bytes).is_err());

        // A wider next row is refused the same way.
        let bytes = flat_encoding(Some(vec![F::ONE]), Some(vec![F::ONE, F::TWO]));
        assert!(postcard::from_bytes::<OpenedValues<F>>(&bytes).is_err());
    }

    #[test]
    fn preprocessed_opening_reports_which_shape_it_refused() {
        // Postcard erases custom messages.
        // The wording is pinned through JSON instead.
        let json = |local: Option<Vec<F>>, next: Option<Vec<F>>| {
            serde_json::to_string(&Flat {
                trace_local: vec![F::ONE],
                trace_next: None,
                preprocessed_local: local,
                preprocessed_next: next,
                quotient_chunks: vec![vec![F::ONE]],
                random: None,
            })
            .unwrap()
        };

        // A next row with no current row names the missing row.
        let err = serde_json::from_str::<OpenedValues<F>>(&json(None, Some(vec![F::ONE])))
            .expect_err("a next row without a current row must be refused");
        assert!(err.to_string().contains("without a current-row opening"));

        // A zero-column current row names its emptiness.
        let err = serde_json::from_str::<OpenedValues<F>>(&json(Some(vec![]), None))
            .expect_err("an empty current row must be refused");
        assert!(err.to_string().contains("empty preprocessed current-row"));

        // Disagreeing widths report the length that was wrong.
        let err = serde_json::from_str::<OpenedValues<F>>(&json(
            Some(vec![F::ONE, F::TWO]),
            Some(vec![F::ONE]),
        ))
        .expect_err("rows of different widths must be refused");
        assert!(err.to_string().contains("as wide as the current row"));
    }
}
