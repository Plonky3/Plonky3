//! Evaluation of planned bus expressions at Boolean rows or extension-field points.

use alloc::vec::Vec;

use p3_air::symbolic::{BaseEntry, BaseLeaf, SymbolicExpr, SymbolicExpression};
use p3_field::{ExtensionField, Field};
use thiserror::Error;

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

/// A planned expression references a value absent from its evaluation view.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Error)]
pub enum BusEvaluationError {
    /// A main column lies outside the supplied committed opening.
    #[error("binary-bus main column {column} is not open")]
    MainColumn {
        /// Missing column index.
        column: usize,
    },
    /// A preprocessed column lies outside the supplied committed opening.
    #[error("binary-bus preprocessed column {column} is not open")]
    PreprocessedColumn {
        /// Missing column index.
        column: usize,
    },
    /// A public value lies outside the statement's public input.
    #[error("binary-bus public value {index} is absent")]
    PublicValue {
        /// Missing public-value index.
        index: usize,
    },
    /// The named-bus index lies outside the public plan.
    #[error("binary-bus domain index {bus} is outside {num_buses} planned domains")]
    UnknownBus {
        /// Rejected domain index.
        bus: usize,
        /// Number of domains in the plan.
        num_buses: usize,
    },
    /// The interaction width disagrees with its named bus.
    #[error("binary-bus domain {bus} has payload width {actual}, expected {expected}")]
    PayloadWidth {
        /// Planned domain index.
        bus: usize,
        /// Width fixed by the plan.
        expected: usize,
        /// Width supplied by the interaction.
        actual: usize,
    },
    /// The fingerprint weight table has the wrong width.
    #[error("binary-bus fingerprint has width {actual}, expected {expected}")]
    FingerprintWidth {
        /// Power-of-two width fixed by the plan.
        expected: usize,
        /// Number of supplied weights.
        actual: usize,
    },
}

impl<EF: Field> BusChallenges<EF> {
    /// Equality-polynomial coefficients for the padded tuple slots.
    #[must_use]
    pub fn fingerprint_weights(&self) -> Vec<EF> {
        // Expand one coordinate at a time in most-significant-variable-first order.
        let mut weights = alloc::vec![EF::ONE];
        for &coordinate in &self.fingerprint {
            let mut next = Vec::with_capacity(weights.len() * 2);
            for &weight in &weights {
                next.push(weight * (EF::ONE - coordinate));
                next.push(weight * coordinate);
            }
            weights = next;
        }
        weights
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
            .map(|expression| evaluate_expression(expression, values))
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
                let selector = evaluate_expression(selector, values)?;
                Ok(EF::ONE + selector * (factor - EF::ONE))
            }
        }
    }
}

fn evaluate_expression<F, EF>(
    expression: &SymbolicExpression<F>,
    values: BusEvaluation<'_, F, EF>,
) -> Result<EF, BusEvaluationError>
where
    F: Field,
    EF: ExtensionField<F>,
{
    // Resolve precisely the expression language accepted by bus planning.
    match expression {
        SymbolicExpr::Leaf(BaseLeaf::Variable(variable)) => match variable.entry {
            BaseEntry::Main { offset: 0 } => {
                values
                    .main
                    .get(variable.index)
                    .copied()
                    .ok_or(BusEvaluationError::MainColumn {
                        column: variable.index,
                    })
            }
            BaseEntry::Preprocessed { offset: 0 } => {
                values.preprocessed.get(variable.index).copied().ok_or(
                    BusEvaluationError::PreprocessedColumn {
                        column: variable.index,
                    },
                )
            }
            BaseEntry::Public => values
                .public
                .get(variable.index)
                .copied()
                .map(Into::into)
                .ok_or(BusEvaluationError::PublicValue {
                    index: variable.index,
                }),
            BaseEntry::Main { .. } | BaseEntry::Preprocessed { .. } | BaseEntry::Periodic => {
                unreachable!("bus planning rejects unsupported expression accesses")
            }
        },
        SymbolicExpr::Leaf(BaseLeaf::IsFirstRow) => Ok(values.is_first_row),
        SymbolicExpr::Leaf(BaseLeaf::IsLastRow) => Ok(values.is_last_row),
        SymbolicExpr::Leaf(BaseLeaf::IsTransition) => Ok(values.is_transition),
        SymbolicExpr::Leaf(BaseLeaf::Constant(value)) => Ok((*value).into()),
        SymbolicExpr::Add { x, y, .. } => {
            Ok(evaluate_expression(x, values)? + evaluate_expression(y, values)?)
        }
        SymbolicExpr::Sub { x, y, .. } => {
            Ok(evaluate_expression(x, values)? - evaluate_expression(y, values)?)
        }
        SymbolicExpr::Neg { x, .. } => Ok(-evaluate_expression(x, values)?),
        SymbolicExpr::Mul { x, y, .. } => {
            Ok(evaluate_expression(x, values)? * evaluate_expression(y, values)?)
        }
    }
}
