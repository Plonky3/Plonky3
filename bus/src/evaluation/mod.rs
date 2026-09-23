//! Evaluation of planned bus expressions at Boolean rows and extension-field points.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use p3_air::symbolic::{BaseEntry, BaseLeaf, SymbolicExpr, SymbolicExpression};
use p3_field::{ExtensionField, Field};
use p3_multilinear_util::point::Point;

mod error;

pub use error::BusEvaluationError;

use crate::{
    BusActivation, BusBoundary, BusChallenges, BusPlan, BusTupleSlot, SymbolicBusInteraction,
};

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

/// One node of a declaration's arithmetic graph, with operands given by position.
///
/// Every operand sits earlier in the node list, so one forward pass resolves all of them.
#[derive(Clone, Copy, Debug)]
enum BusNode<F> {
    /// Current-row main column at this index.
    Main(usize),
    /// Current-row preprocessed column at this index.
    Preprocessed(usize),
    /// Public value at this index.
    Public(usize),
    /// First-row selector.
    IsFirstRow,
    /// Last-row selector.
    IsLastRow,
    /// Transition selector.
    IsTransition,
    /// Literal field element.
    Constant(F),
    /// Sum of two earlier nodes.
    Add(usize, usize),
    /// Difference of two earlier nodes.
    Sub(usize, usize),
    /// Negation of one earlier node.
    Neg(usize),
    /// Product of two earlier nodes.
    Mul(usize, usize),
}

/// Resolve one symbolic leaf, rejecting the accesses no committed opening can answer.
const fn compile_leaf<F: Field>(leaf: &BaseLeaf<F>) -> Result<BusNode<F>, BusEvaluationError> {
    // Every leaf is checked independently because this API also accepts caller-built trees.
    Ok(match leaf {
        BaseLeaf::Variable(variable) => match variable.entry {
            BaseEntry::Main { offset: 0 } => BusNode::Main(variable.index),
            BaseEntry::Main { offset } => {
                return Err(BusEvaluationError::MainOffset {
                    column: variable.index,
                    offset,
                });
            }
            BaseEntry::Preprocessed { offset: 0 } => BusNode::Preprocessed(variable.index),
            BaseEntry::Preprocessed { offset } => {
                return Err(BusEvaluationError::PreprocessedOffset {
                    column: variable.index,
                    offset,
                });
            }
            BaseEntry::Public => BusNode::Public(variable.index),
            BaseEntry::Periodic => {
                return Err(BusEvaluationError::PeriodicColumn {
                    column: variable.index,
                });
            }
        },
        BaseLeaf::IsFirstRow => BusNode::IsFirstRow,
        BaseLeaf::IsLastRow => BusNode::IsLastRow,
        BaseLeaf::IsTransition => BusNode::IsTransition,
        BaseLeaf::Constant(value) => BusNode::Constant(*value),
    })
}

/// Flatten one symbolic tree into the shared node list, and return its root position.
///
/// Arithmetic nodes share their operands, so the declaration is a graph rather than a tree.
///
/// Recording each distinct arithmetic node once keeps every later pass linear in the graph size.
///
/// A leaf is copied at each use, which stays linear and keeps a small declaration off the lookup.
fn compile_expression<F: Field>(
    nodes: &mut Vec<BusNode<F>>,
    positions: &mut BTreeMap<*const SymbolicExpression<F>, usize>,
    root: &SymbolicExpression<F>,
) -> Result<usize, BusEvaluationError> {
    /// Record one operand and return its position in the shared node list.
    fn operand<F: Field>(
        nodes: &mut Vec<BusNode<F>>,
        positions: &BTreeMap<*const SymbolicExpression<F>, usize>,
        child: &SymbolicExpression<F>,
    ) -> Result<usize, BusEvaluationError> {
        match child {
            SymbolicExpr::Leaf(leaf) => {
                nodes.push(compile_leaf(leaf)?);
                Ok(nodes.len() - 1)
            }
            _ => Ok(positions[&core::ptr::from_ref(child)]),
        }
    }

    if let SymbolicExpr::Leaf(leaf) = root {
        nodes.push(compile_leaf(leaf)?);
        return Ok(nodes.len() - 1);
    }

    // An explicit worklist keeps a deep declaration off the call stack.
    // Each arithmetic node is visited once to schedule its operands and once to be recorded.
    let mut pending = alloc::vec![(root, false)];
    while let Some((expression, recording)) = pending.pop() {
        let key = core::ptr::from_ref(expression);
        if positions.contains_key(&key) {
            continue;
        }
        let (x, y) = match expression {
            SymbolicExpr::Add { x, y, .. }
            | SymbolicExpr::Sub { x, y, .. }
            | SymbolicExpr::Mul { x, y, .. } => (&**x, Some(&**y)),
            SymbolicExpr::Neg { x, .. } => (&**x, None),
            SymbolicExpr::Leaf(_) => unreachable!("leaves never reach the worklist"),
        };
        if !recording {
            pending.push((expression, true));
            for child in [Some(x), y].into_iter().flatten() {
                if !matches!(child, SymbolicExpr::Leaf(_)) {
                    pending.push((child, false));
                }
            }
            continue;
        }
        let node = match expression {
            SymbolicExpr::Add { .. } => BusNode::Add(
                operand(nodes, positions, x)?,
                operand(nodes, positions, y.unwrap())?,
            ),
            SymbolicExpr::Sub { .. } => BusNode::Sub(
                operand(nodes, positions, x)?,
                operand(nodes, positions, y.unwrap())?,
            ),
            SymbolicExpr::Mul { .. } => BusNode::Mul(
                operand(nodes, positions, x)?,
                operand(nodes, positions, y.unwrap())?,
            ),
            SymbolicExpr::Neg { .. } => BusNode::Neg(operand(nodes, positions, x)?),
            SymbolicExpr::Leaf(_) => unreachable!("leaves never reach the worklist"),
        };
        positions.insert(key, nodes.len());
        nodes.push(node);
    }
    Ok(positions[&core::ptr::from_ref(root)])
}

impl<EF: Field> BusChallenges<EF> {
    /// Equality-polynomial coefficients for the padded tuple slots.
    #[must_use]
    pub fn fingerprint_weights(&self) -> Vec<EF> {
        // Public tuple coordinates bind slot-index bits from most to least significant.
        Point::new(self.fingerprint.as_slice()).equality_weights_msb()
    }
}

/// One declaration's leaf factor reduced to the terms that read the trace.
///
/// Slot placement, the named-domain contribution, and every width check are settled once.
///
/// So is the arithmetic graph, which is flattened so that a shared node costs one evaluation.
///
/// What remains per row is one forward pass and one inner product.
#[derive(Clone, Debug)]
pub struct BusFactorPlan<'a, F: Field, EF> {
    /// Arithmetic graph in an order that resolves every operand before its user.
    nodes: Vec<BusNode<F>>,
    /// Node carrying each payload position, in declaration order.
    payload: Vec<usize>,
    /// Node carrying the conditional selector, when the declaration has one.
    activation: Option<usize>,
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
    /// The scratch buffer holds one value per graph node and is reused across rows.
    ///
    /// # Errors
    ///
    /// Returns an error when the supplied evaluation view omits a referenced value.
    pub fn evaluate<A>(
        &self,
        scratch: &mut Vec<A>,
        values: BusEvaluation<'_, F, A>,
    ) -> Result<EF, BusEvaluationError>
    where
        A: ExtensionField<F>,
        EF: ExtensionField<A>,
    {
        scratch.clear();
        scratch.reserve(self.nodes.len());
        for node in &self.nodes {
            let value = match *node {
                BusNode::Main(column) => *values
                    .main
                    .get(column)
                    .ok_or(BusEvaluationError::MainColumn { column })?,
                BusNode::Preprocessed(column) => *values
                    .preprocessed
                    .get(column)
                    .ok_or(BusEvaluationError::PreprocessedColumn { column })?,
                BusNode::Public(index) => values
                    .public
                    .get(index)
                    .copied()
                    .map(Into::into)
                    .ok_or(BusEvaluationError::PublicValue { index })?,
                BusNode::IsFirstRow => values.is_first_row,
                BusNode::IsLastRow => values.is_last_row,
                BusNode::IsTransition => values.is_transition,
                BusNode::Constant(value) => value.into(),
                BusNode::Add(x, y) => scratch[x] + scratch[y],
                BusNode::Sub(x, y) => scratch[x] - scratch[y],
                BusNode::Neg(x) => -scratch[x],
                BusNode::Mul(x, y) => scratch[x] * scratch[y],
            };
            scratch.push(value);
        }

        // Padding and named-domain slots are constant, so only payload slots are summed here.
        let mut fingerprint = EF::ZERO;
        for (&node, &weight) in self.payload.iter().zip(self.payload_weights) {
            fingerprint += weight * scratch[node];
        }
        let factor = self.shifted_offset - fingerprint;

        // Conditional rows interpolate between identity padding and the live factor.
        Ok(self
            .activation
            .map_or(factor, |node| EF::ONE + (factor - EF::ONE) * scratch[node]))
    }
}

impl BusPlan {
    /// Settle one declaration's slot placement against the sampled tuple weights.
    ///
    /// The result is reusable across every row of the owning table.
    ///
    /// # Errors
    ///
    /// Returns an error when the bus, the payload width, the weight count, or a leaf is wrong.
    pub fn compile_factor<'a, F, EF>(
        &self,
        bus: usize,
        interaction: &SymbolicBusInteraction<F>,
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

        // Both sides of one declaration share a node list, so a common factor is recorded once.
        let mut nodes = Vec::new();
        let mut positions = BTreeMap::new();
        let payload = interaction
            .fields
            .iter()
            .map(|expression| compile_expression(&mut nodes, &mut positions, expression))
            .collect::<Result<Vec<_>, _>>()?;
        let activation = match &interaction.activation {
            BusActivation::Always => None,
            // A boundary indicator is a backend leaf, so it joins the graph without an expression.
            BusActivation::Boundary(boundary) => {
                nodes.push(match boundary {
                    BusBoundary::First => BusNode::IsFirstRow,
                    BusBoundary::Last => BusNode::IsLastRow,
                });
                Some(nodes.len() - 1)
            }
            BusActivation::Boolean(selector) => {
                Some(compile_expression(&mut nodes, &mut positions, selector)?)
            }
        };

        Ok(BusFactorPlan {
            nodes,
            payload,
            activation,
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
            .evaluate::<EF>(&mut Vec::new(), values)
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
    fn a_deeply_shared_payload_resolves_once_per_distinct_node() {
        // A doubling chain of this depth has one node per level and two paths out of each.
        // Re-walking the graph per path would take about thirteen seconds here.
        const DEPTH: u32 = 32;

        let leaf = F::from_u8(5);
        let mut expression: SymbolicExpression<F> =
            SymbolicVariable::new(BaseEntry::Main { offset: 0 }, 0).into();
        for _ in 0..DEPTH {
            expression = expression.clone() + expression;
        }
        let interaction = SymbolicBusInteraction::<F> {
            bus_name: "memory".to_string(),
            direction: BusDirection::Push,
            fields: vec![expression],
            activation: BusActivation::Always,
        };

        // Weight one on the payload slot and a zero shift leave the negated payload.
        let factor = plan()
            .compile_factor(0, &interaction, &[F::ONE, F::ZERO], F::ZERO)
            .unwrap();
        let value = factor
            .evaluate(
                &mut vec![],
                BusEvaluation {
                    main: &[leaf],
                    preprocessed: &[],
                    public: &[],
                    is_first_row: F::ZERO,
                    is_last_row: F::ZERO,
                    is_transition: F::ZERO,
                },
            )
            .unwrap();

        // Doubling that many times multiplies the leaf by that power of two.
        assert_eq!(value, -(leaf * F::TWO.exp_u64(u64::from(DEPTH))));
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

#[cfg(test)]
mod boundary_tests {
    use alloc::string::ToString;
    use alloc::vec;

    use p3_air::symbolic::{BaseEntry, SymbolicVariable};
    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;

    use super::*;
    use crate::{BusDirection, BusPlanInput};

    type F = BabyBear;

    /// One declaration on a one-payload channel, activated however the caller asks.
    fn interaction(activation: BusActivation<SymbolicExpression<F>>) -> SymbolicBusInteraction<F> {
        SymbolicBusInteraction {
            bus_name: "state".to_string(),
            direction: BusDirection::Push,
            fields: vec![SymbolicVariable::new(BaseEntry::Main { offset: 0 }, 0).into()],
            activation,
        }
    }

    fn plan(interaction: &SymbolicBusInteraction<F>) -> BusPlan {
        BusPlan::build(&[BusPlanInput {
            log_height: 2,
            interactions: core::slice::from_ref(interaction),
        }])
        .unwrap()
        .unwrap()
    }

    /// The leaf factor of one declaration at one Boolean row of a four-row table.
    fn factor_at_row(interaction: &SymbolicBusInteraction<F>, row: usize) -> F {
        let plan = plan(interaction);
        let weights = alloc::vec![F::ONE; plan.fingerprint_width()];
        plan.evaluate_factor(
            0,
            interaction,
            BusEvaluation {
                main: &[F::from_usize(row + 1)],
                preprocessed: &[],
                public: &[],
                is_first_row: F::from_bool(row == 0),
                is_last_row: F::from_bool(row == 3),
                is_transition: F::from_bool(row < 3),
            },
            &weights,
            F::from_u64(17),
        )
        .unwrap()
    }

    #[test]
    fn a_boundary_declaration_is_the_identity_away_from_its_end() {
        for (boundary, live) in [(BusBoundary::First, 0usize), (BusBoundary::Last, 3)] {
            let declaration = interaction(BusActivation::Boundary(boundary));
            for row in 0..4 {
                let factor = factor_at_row(&declaration, row);
                if row == live {
                    // The live row contributes its shifted fingerprint, never the identity.
                    assert_ne!(factor, F::ONE);
                } else {
                    // Every other row multiplies the product tree by one.
                    assert_eq!(factor, F::ONE, "{boundary:?} leaked into row {row}");
                }
            }
        }
    }

    #[test]
    fn a_boundary_declaration_evaluates_as_its_selector_form_does() {
        // The proving path must not distinguish the first-class form from the hand-written one.
        let pairs = [
            (
                BusActivation::Boundary(BusBoundary::First),
                BusActivation::Boolean(SymbolicExpr::Leaf(BaseLeaf::<F>::IsFirstRow)),
            ),
            (
                BusActivation::Boundary(BusBoundary::Last),
                BusActivation::Boolean(SymbolicExpr::Leaf(BaseLeaf::<F>::IsLastRow)),
            ),
        ];
        for (boundary, selector) in pairs {
            let boundary = interaction(boundary);
            let selector = interaction(selector);
            for row in 0..4 {
                assert_eq!(factor_at_row(&boundary, row), factor_at_row(&selector, row));
            }
        }
    }

    #[test]
    fn a_boundary_declaration_reads_no_column_of_its_own() {
        // Compiling adds exactly one graph node for the indicator, and it names no column.
        let always = interaction(BusActivation::Always);
        let boundary = interaction(BusActivation::Boundary(BusBoundary::Last));
        let plan = plan(&boundary);
        let weights = alloc::vec![F::ONE; plan.fingerprint_width()];
        let compiled = plan
            .compile_factor(0, &boundary, &weights, F::ZERO)
            .unwrap();
        let baseline = plan.compile_factor(0, &always, &weights, F::ZERO).unwrap();
        assert_eq!(compiled.nodes.len(), baseline.nodes.len() + 1);
        assert!(matches!(compiled.nodes.last(), Some(BusNode::IsLastRow)));
    }
}
