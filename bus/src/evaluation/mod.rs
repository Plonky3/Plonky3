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

impl BusPlan {
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

        // Evaluate payload expressions once, then place them into the padded tuple.
        let payload = interaction
            .fields
            .iter()
            .map(|expression| values.evaluate(expression))
            .collect::<Result<Vec<_>, _>>()?;
        let fingerprint = weights
            .iter()
            .enumerate()
            .map(|(slot, &weight)| {
                let value = match self
                    .tuple_slot(bus, slot)
                    .expect("weights have the planned fingerprint width")
                {
                    BusTupleSlot::Payload(index) => payload[index],
                    BusTupleSlot::DomainBit(true) => EF::ONE,
                    BusTupleSlot::DomainBit(false) | BusTupleSlot::Zero => EF::ZERO,
                };
                weight * value
            })
            .sum::<EF>();
        let factor = offset - fingerprint;

        // Conditional rows interpolate between identity padding and the live factor.
        match &interaction.activation {
            BusActivation::Always => Ok(factor),
            BusActivation::Boolean(selector) => {
                let selector = values.evaluate(selector)?;
                Ok(EF::ONE + selector * (factor - EF::ONE))
            }
        }
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
