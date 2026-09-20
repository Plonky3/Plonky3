//! Evaluation of planned bus expressions at Boolean rows and extension-field points.

use alloc::vec::Vec;

use p3_air::symbolic::{BaseEntry, BaseLeaf, SymbolicExpr, SymbolicExpression};
use p3_field::{ExtensionField, Field};

mod error;

pub use error::BusEvaluationError;

use crate::multilinear::equality_weights_msb;
use crate::{BusActivation, BusChallenges, BusPlan, BusTupleSlot, SymbolicBusInteraction};

/// Values resolving every supported symbolic leaf at one common point.
#[derive(Clone, Copy, Debug)]
pub struct BusEvaluation<'a, F, EF> {
    /// Current-row main-column evaluations in column order.
    pub main: &'a [EF],
    /// Current-row preprocessed-column evaluations in column order.
    pub preprocessed: &'a [EF],
    /// Public values in declaration order.
    pub public: &'a [F],
    /// Multilinear first-row selector evaluation.
    pub is_first_row: EF,
    /// Multilinear last-row selector evaluation.
    pub is_last_row: EF,
    /// Multilinear transition selector evaluation.
    pub is_transition: EF,
}

impl<F, EF> BusEvaluation<'_, F, EF>
where
    F: Field,
    EF: ExtensionField<F>,
{
    /// Resolve one symbolic expression against commitment-bound evaluations.
    fn evaluate(&self, expression: &SymbolicExpression<F>) -> Result<EF, BusEvaluationError> {
        // Every leaf is checked independently because this API also accepts caller-built trees.
        match expression {
            SymbolicExpr::Leaf(BaseLeaf::Variable(variable)) => match variable.entry {
                BaseEntry::Main { offset: 0 } => {
                    self.main
                        .get(variable.index)
                        .copied()
                        .ok_or(BusEvaluationError::MainColumn {
                            column: variable.index,
                        })
                }
                BaseEntry::Main { offset } => Err(BusEvaluationError::MainOffset {
                    column: variable.index,
                    offset,
                }),
                BaseEntry::Preprocessed { offset: 0 } => {
                    self.preprocessed.get(variable.index).copied().ok_or(
                        BusEvaluationError::PreprocessedColumn {
                            column: variable.index,
                        },
                    )
                }
                BaseEntry::Preprocessed { offset } => Err(BusEvaluationError::PreprocessedOffset {
                    column: variable.index,
                    offset,
                }),
                BaseEntry::Public => self
                    .public
                    .get(variable.index)
                    .copied()
                    .map(Into::into)
                    .ok_or(BusEvaluationError::PublicValue {
                        index: variable.index,
                    }),
                BaseEntry::Periodic => Err(BusEvaluationError::PeriodicColumn {
                    column: variable.index,
                }),
            },
            SymbolicExpr::Leaf(BaseLeaf::IsFirstRow) => Ok(self.is_first_row),
            SymbolicExpr::Leaf(BaseLeaf::IsLastRow) => Ok(self.is_last_row),
            SymbolicExpr::Leaf(BaseLeaf::IsTransition) => Ok(self.is_transition),
            SymbolicExpr::Leaf(BaseLeaf::Constant(value)) => Ok((*value).into()),
            SymbolicExpr::Add { x, y, .. } => Ok(self.evaluate(x)? + self.evaluate(y)?),
            SymbolicExpr::Sub { x, y, .. } => Ok(self.evaluate(x)? - self.evaluate(y)?),
            SymbolicExpr::Neg { x, .. } => Ok(-self.evaluate(x)?),
            SymbolicExpr::Mul { x, y, .. } => Ok(self.evaluate(x)? * self.evaluate(y)?),
        }
    }
}

impl<EF: Field> BusChallenges<EF> {
    /// Equality-polynomial coefficients for the padded tuple slots.
    #[must_use]
    pub fn fingerprint_weights(&self) -> Vec<EF> {
        // Public tuple coordinates bind slot-index bits from most to least significant.
        equality_weights_msb(&self.fingerprint)
    }
}

/// One declaration's leaf factor reduced to the terms that read the trace.
///
/// Slot placement, the named-domain contribution, and every width check are settled once.
/// What remains per row is one payload evaluation and one inner product.
#[derive(Clone, Copy, Debug)]
pub struct BusFactorPlan<'a, F: Field, EF> {
    /// Declaration whose payload and activation expressions are resolved at each point.
    interaction: &'a SymbolicBusInteraction<F>,
    /// Tuple weight of each payload position, in declaration order.
    payload_weights: &'a [EF],
    /// Random shift already reduced by the fixed named-domain contribution.
    shifted_offset: EF,
}

impl<F, EF> BusFactorPlan<'_, F, EF>
where
    F: Field,
    EF: ExtensionField<F>,
{
    /// Evaluate the leaf factor at one resolved point.
    ///
    /// Boolean rows resolve in the base field and cost far less than a folded point.
    ///
    /// # Errors
    ///
    /// Returns an error when the supplied evaluation view omits a referenced value.
    pub fn evaluate<A>(&self, values: BusEvaluation<'_, F, A>) -> Result<EF, BusEvaluationError>
    where
        A: ExtensionField<F>,
        EF: ExtensionField<A>,
    {
        // Padding and named-domain slots are constant, so only payload slots are summed here.
        let mut fingerprint = EF::ZERO;
        for (expression, &weight) in self.interaction.fields.iter().zip(self.payload_weights) {
            fingerprint += weight * values.evaluate(expression)?;
        }
        let factor = self.shifted_offset - fingerprint;

        // Conditional rows interpolate between identity padding and the live factor.
        match &self.interaction.activation {
            BusActivation::Always => Ok(factor),
            BusActivation::Boolean(selector) => {
                let selector = values.evaluate(selector)?;
                Ok(EF::ONE + (factor - EF::ONE) * selector)
            }
        }
    }
}

impl BusPlan {
    /// Settle one declaration's slot placement against the sampled tuple weights.
    ///
    /// The result is reusable across every row of the owning table.
    ///
    /// # Errors
    ///
    /// Returns an error when the bus, the payload width, or the weight count is wrong.
    pub fn compile_factor<'a, F, EF>(
        &self,
        bus: usize,
        interaction: &'a SymbolicBusInteraction<F>,
        weights: &'a [EF],
        offset: EF,
    ) -> Result<BusFactorPlan<'a, F, EF>, BusEvaluationError>
    where
        F: Field,
        EF: ExtensionField<F>,
    {
        // The plan fixes both the bus-local payload width and the padded fingerprint width.
        let domain = self
            .domains()
            .get(bus)
            .ok_or_else(|| BusEvaluationError::UnknownBus {
                bus,
                num_buses: self.domains().len(),
            })?;
        if interaction.fields.len() != domain.payload_width {
            return Err(BusEvaluationError::PayloadWidth {
                bus,
                expected: domain.payload_width,
                actual: interaction.fields.len(),
            });
        }
        if weights.len() != self.fingerprint_width() {
            return Err(BusEvaluationError::FingerprintWidth {
                expected: self.fingerprint_width(),
                actual: weights.len(),
            });
        }

        // Every slot outside the payload prefix carries a value fixed for the whole proof.
        let domain_constant = weights
            .iter()
            .enumerate()
            .filter(|&(slot, _)| {
                matches!(
                    self.tuple_slot(bus, slot),
                    Some(BusTupleSlot::DomainBit(true))
                )
            })
            .map(|(_, &weight)| weight)
            .sum::<EF>();

        Ok(BusFactorPlan {
            interaction,
            // Payload position and tuple slot coincide over the leading payload slots.
            payload_weights: &weights[..domain.payload_width],
            shifted_offset: offset - domain_constant,
        })
    }

    /// Evaluate one planned declaration's selected leaf factor.
    ///
    /// # Errors
    ///
    /// Returns an error when the supplied evaluation view omits a referenced value.
    pub fn evaluate_factor<F, EF>(
        &self,
        bus: usize,
        interaction: &SymbolicBusInteraction<F>,
        values: BusEvaluation<'_, F, EF>,
        weights: &[EF],
        offset: EF,
    ) -> Result<EF, BusEvaluationError>
    where
        F: Field,
        EF: ExtensionField<F>,
    {
        // A single point pays for the placement it would otherwise reuse.
        self.compile_factor(bus, interaction, weights, offset)?
            .evaluate::<EF>(values)
    }
}

#[cfg(test)]
mod tests {
    use alloc::string::ToString;
    use alloc::vec;

    use p3_air::symbolic::{BaseEntry, SymbolicVariable};
    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;

    use super::*;
    use crate::{BusDirection, BusPlanInput};

    type F = BabyBear;

    /// Build the smallest plan that reserves one payload slot and one domain slot.
    fn plan() -> BusPlan {
        // A supported current-row expression establishes the public payload width.
        let interaction = SymbolicBusInteraction::<F> {
            bus_name: "memory".to_string(),
            direction: BusDirection::Push,
            fields: vec![SymbolicVariable::new(BaseEntry::Main { offset: 0 }, 0).into()],
            activation: BusActivation::Always,
        };
        BusPlan::build(&[BusPlanInput {
            log_height: 0,
            interactions: &[interaction],
        }])
        .unwrap()
        .unwrap()
    }

    /// Evaluate one caller-built expression through the public factor boundary.
    fn evaluate(expression: SymbolicExpression<F>) -> Result<F, BusEvaluationError> {
        // The adversarial declaration matches the planned width but not its access policy.
        let interaction = SymbolicBusInteraction::<F> {
            bus_name: "memory".to_string(),
            direction: BusDirection::Push,
            fields: vec![expression],
            activation: BusActivation::Always,
        };
        plan().evaluate_factor(
            0,
            &interaction,
            BusEvaluation {
                main: &[],
                preprocessed: &[],
                public: &[],
                is_first_row: F::ZERO,
                is_last_row: F::ZERO,
                is_transition: F::ZERO,
            },
            &[F::ONE, F::ZERO],
            F::ZERO,
        )
    }

    #[test]
    fn unsupported_symbolic_leaves_return_structured_errors() {
        // A next-row main access cannot be reconstructed from a current-row opening.
        let main = SymbolicVariable::new(BaseEntry::Main { offset: 1 }, 3).into();
        assert_eq!(
            evaluate(main),
            Err(BusEvaluationError::MainOffset {
                column: 3,
                offset: 1,
            })
        );

        // A next-row fixed access has the same missing-opening failure mode.
        let preprocessed = SymbolicVariable::new(BaseEntry::Preprocessed { offset: 2 }, 5).into();
        assert_eq!(
            evaluate(preprocessed),
            Err(BusEvaluationError::PreprocessedOffset {
                column: 5,
                offset: 2,
            })
        );

        // No periodic-column evaluation exists in the public evaluation view.
        let periodic = SymbolicVariable::new(BaseEntry::Periodic, 7).into();
        assert_eq!(
            evaluate(periodic),
            Err(BusEvaluationError::PeriodicColumn { column: 7 })
        );
    }

    #[test]
    fn fingerprint_coordinates_address_slots_most_significant_first() {
        // Slot order is 00, 01, 10, 11 for point coordinates a then b.
        let a = F::from_u8(2);
        let b = F::from_u8(3);
        let challenges = BusChallenges {
            fingerprint: vec![a, b],
            offset: F::ZERO,
        };

        // The first coordinate selects the two-slot half.
        // The second coordinate selects a slot inside that half.
        assert_eq!(
            challenges.fingerprint_weights(),
            vec![
                (F::ONE - a) * (F::ONE - b),
                (F::ONE - a) * b,
                a * (F::ONE - b),
                a * b,
            ]
        );
    }
}
