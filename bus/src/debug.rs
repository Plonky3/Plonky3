//! Out-of-circuit diagnostics for binary-native bus declarations.

use alloc::string::String;
use alloc::vec::Vec;

use hashbrown::HashMap;
use p3_air::symbolic::{BaseEntry, BaseLeaf, SymbolicExpr, SymbolicExpression};
use p3_field::{ExtensionField, Field};
use p3_sumcheck::layout::Table;
use thiserror::Error;

use crate::{
    BusActivation, BusDirection, BusPlan, BusPlanError, BusPlanInput, BusSymbolicBuilder,
    SymbolicBusInteraction,
};

/// Maximum source locations retained for one tuple and direction.
///
/// The count remains exact after this diagnostic sample is full.
const LOCATION_LIMIT: usize = 4;

/// Concrete data required to replay one AIR's bus declarations.
#[derive(Clone, Copy, Debug)]
pub struct BusDebugInstance<'a, F: Field> {
    /// Committed trace columns in polynomial-major order.
    main: &'a Table<F>,
    /// Optional fixed trace columns in polynomial-major order.
    preprocessed: Option<&'a Table<F>>,
    /// Public inputs supplied to the AIR.
    public_values: &'a [F],
    /// Symbolic tuple declarations in AIR emission order.
    interactions: &'a [SymbolicBusInteraction<F>],
}

impl<'a, F: Field> BusDebugInstance<'a, F> {
    /// Pairs concrete AIR data with declarations from its symbolic profile.
    ///
    /// The profile must come from the same AIR and layout as the concrete data.
    #[must_use]
    pub fn new<EF: ExtensionField<F>>(
        main: &'a Table<F>,
        preprocessed: Option<&'a Table<F>>,
        public_values: &'a [F],
        profile: &'a BusSymbolicBuilder<F, EF>,
    ) -> Self {
        // The profile remains the sole source of symbolic declarations.
        // This prevents a caller from passing an unrelated raw interaction slice.
        Self {
            main,
            preprocessed,
            public_values,
            interactions: profile.interactions(),
        }
    }
}

/// Source of one tuple occurrence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BusDebugLocation {
    /// AIR position in statement order.
    pub air: usize,
    /// Declaration position within the AIR.
    pub declaration: usize,
    /// Row position within the owning trace.
    pub row: usize,
}

/// Exact occurrence count with a bounded source sample.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BusDebugOccurrence {
    /// Number of active rows carrying the tuple.
    pub count: usize,
    /// First source locations in deterministic traversal order.
    pub locations: Vec<BusDebugLocation>,
}

/// One tuple whose push and pull multiplicities differ.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BusImbalance<F> {
    /// Name of the independently balanced multiset.
    pub bus_name: String,
    /// Payload field elements in declaration order.
    pub tuple: Vec<F>,
    /// Produced occurrences and their first source locations.
    pub pushes: BusDebugOccurrence,
    /// Consumed occurrences and their first source locations.
    pub pulls: BusDebugOccurrence,
}

/// Deterministic list of unmatched bus tuples.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BusDebugReport<F> {
    /// Unmatched tuples in first-occurrence order.
    pub imbalances: Vec<BusImbalance<F>>,
}

impl<F> BusDebugReport<F> {
    /// Returns whether every named multiset balances exactly.
    #[must_use]
    pub const fn is_balanced(&self) -> bool {
        // An empty mismatch list is the exact balance condition.
        self.imbalances.is_empty()
    }
}

/// Invalid concrete data encountered during diagnostic replay.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum BusDebugError {
    /// The verifier-derived declaration layout is malformed.
    #[error(transparent)]
    Plan(#[from] BusPlanError),
    /// Fixed columns use a different height from committed columns.
    #[error("binary-bus AIR {air} preprocessed trace has height 2^{actual}, expected 2^{expected}")]
    PreprocessedHeightMismatch {
        /// AIR position in statement order.
        air: usize,
        /// Base-two logarithm of the committed trace height.
        expected: usize,
        /// Base-two logarithm of the fixed trace height.
        actual: usize,
    },
    /// An expression reads a missing committed column.
    #[error(
        "binary-bus AIR {air} declaration {declaration} row {row} reads main column {column}, but the trace has width {width}"
    )]
    MainColumnOutOfRange {
        /// AIR position in statement order.
        air: usize,
        /// Declaration position within the AIR.
        declaration: usize,
        /// Row position within the trace.
        row: usize,
        /// Missing column position.
        column: usize,
        /// Available column count.
        width: usize,
    },
    /// An expression reads fixed data that was not supplied.
    #[error(
        "binary-bus AIR {air} declaration {declaration} row {row} requires preprocessed column {column}, but no preprocessed trace was supplied"
    )]
    MissingPreprocessedTrace {
        /// AIR position in statement order.
        air: usize,
        /// Declaration position within the AIR.
        declaration: usize,
        /// Row position within the trace.
        row: usize,
        /// Requested column position.
        column: usize,
    },
    /// An expression reads a missing fixed column.
    #[error(
        "binary-bus AIR {air} declaration {declaration} row {row} reads preprocessed column {column}, but the trace has width {width}"
    )]
    PreprocessedColumnOutOfRange {
        /// AIR position in statement order.
        air: usize,
        /// Declaration position within the AIR.
        declaration: usize,
        /// Row position within the trace.
        row: usize,
        /// Missing column position.
        column: usize,
        /// Available column count.
        width: usize,
    },
    /// An expression reads a missing public input.
    #[error(
        "binary-bus AIR {air} declaration {declaration} row {row} reads public value {index}, but only {len} values were supplied"
    )]
    PublicValueOutOfRange {
        /// AIR position in statement order.
        air: usize,
        /// Declaration position within the AIR.
        declaration: usize,
        /// Row position within the trace.
        row: usize,
        /// Missing public-input position.
        index: usize,
        /// Available public-input count.
        len: usize,
    },
    /// A conditional declaration uses a value outside the Boolean set.
    #[error(
        "binary-bus AIR {air} declaration {declaration} row {row} has a non-Boolean activation"
    )]
    NonBooleanActivation {
        /// AIR position in statement order.
        air: usize,
        /// Declaration position within the AIR.
        declaration: usize,
        /// Row position within the trace.
        row: usize,
    },
    /// An exact tuple count cannot fit in the target pointer width.
    #[error(
        "binary-bus AIR {air} declaration {declaration} row {row} overflows the tuple occurrence count"
    )]
    OccurrenceCountOverflow {
        /// AIR position in statement order.
        air: usize,
        /// Declaration position within the AIR.
        declaration: usize,
        /// Row position within the trace.
        row: usize,
    },
}

/// Replays every declaration and reports unmatched tuples with their sources.
///
/// Named buses are checked independently.
///
/// Direction remains metadata rather than a field sign.
///
/// # Errors
///
/// Returns an error when the declaration layout or concrete inputs are malformed.
///
/// # Performance
///
/// - Memory is linear in the number of distinct active tuples.
/// - Output is linear in the number of unmatched tuples.
/// - The routine is intended for trusted development traces.
pub fn check_bus_balance<F>(
    instances: &[BusDebugInstance<'_, F>],
) -> Result<BusDebugReport<F>, BusDebugError>
where
    F: Field,
{
    // Build the same deterministic named-bus layout used by the proof protocol.
    let inputs = instances
        .iter()
        .map(|instance| BusPlanInput {
            log_height: instance.main.num_variables(),
            interactions: instance.interactions,
        })
        .collect::<Vec<_>>();
    let Some(plan) = BusPlan::build(&inputs)? else {
        return Ok(BusDebugReport {
            imbalances: Vec::new(),
        });
    };

    // Fixed and committed columns must describe the same row domain.
    for (air, instance) in instances.iter().enumerate() {
        if let Some(preprocessed) = instance.preprocessed
            && preprocessed.num_variables() != instance.main.num_variables()
        {
            return Err(BusDebugError::PreprocessedHeightMismatch {
                air,
                expected: instance.main.num_variables(),
                actual: preprocessed.num_variables(),
            });
        }
    }

    // Domain names are already sorted and assigned stable identities by the plan.
    let domain_indices = plan
        .domains()
        .iter()
        .enumerate()
        .map(|(index, domain)| (domain.name.as_str(), index))
        .collect::<HashMap<_, _>>();

    // A hash index provides constant-time grouping.
    // The parallel vector preserves deterministic first-occurrence order for diagnostics.
    let mut entry_indices = HashMap::<(usize, Vec<F>), usize>::new();
    let mut entries = Vec::<Entry<F>>::new();

    for (air, instance) in instances.iter().enumerate() {
        let height = 1usize << instance.main.num_variables();

        for row in 0..height {
            let evaluator = RowEvaluator {
                air,
                declaration: 0,
                row,
                height,
                main: instance.main,
                preprocessed: instance.preprocessed,
                public_values: instance.public_values,
            };

            for (declaration, interaction) in instance.interactions.iter().enumerate() {
                let evaluator = RowEvaluator {
                    declaration,
                    ..evaluator
                };

                // Inactive rows contribute the product identity and no multiset occurrence.
                if !activation_is_set(&interaction.activation, &evaluator)? {
                    continue;
                }

                // Resolve the exact payload tuple recorded by the symbolic AIR pass.
                let tuple = interaction
                    .fields
                    .iter()
                    .map(|field| evaluator.evaluate(field))
                    .collect::<Result<Vec<_>, _>>()?;
                let bus = domain_indices[interaction.bus_name.as_str()];
                let key = (bus, tuple.clone());

                // Reuse an existing tuple accumulator or append one deterministic entry.
                let entry = if let Some(&index) = entry_indices.get(&key) {
                    &mut entries[index]
                } else {
                    let index = entries.len();
                    entry_indices.insert(key, index);
                    entries.push(Entry::new(bus, tuple));
                    &mut entries[index]
                };

                entry.record(
                    interaction.direction,
                    BusDebugLocation {
                        air,
                        declaration,
                        row,
                    },
                )?;
            }
        }
    }

    // Equal multiplicities cancel as integers rather than as field elements.
    let imbalances = entries
        .into_iter()
        .filter(|entry| entry.pushes.count != entry.pulls.count)
        .map(|entry| BusImbalance {
            bus_name: plan.domains()[entry.bus].name.clone(),
            tuple: entry.tuple,
            pushes: entry.pushes,
            pulls: entry.pulls,
        })
        .collect();

    Ok(BusDebugReport { imbalances })
}

/// Integer multiplicities accumulated for one named tuple.
#[derive(Clone, Debug)]
struct Entry<F> {
    /// Stable position of the named multiset.
    bus: usize,
    /// Payload field elements in declaration order.
    tuple: Vec<F>,
    /// Produced multiplicity and source sample.
    pushes: BusDebugOccurrence,
    /// Consumed multiplicity and source sample.
    pulls: BusDebugOccurrence,
}

impl<F> Entry<F> {
    const fn new(bus: usize, tuple: Vec<F>) -> Self {
        // New tuples have not appeared on either side.
        Self {
            bus,
            tuple,
            pushes: BusDebugOccurrence {
                count: 0,
                locations: Vec::new(),
            },
            pulls: BusDebugOccurrence {
                count: 0,
                locations: Vec::new(),
            },
        }
    }

    fn record(
        &mut self,
        direction: BusDirection,
        location: BusDebugLocation,
    ) -> Result<(), BusDebugError> {
        // Direction selects an integer counter rather than a field sign.
        let occurrence = match direction {
            BusDirection::Push => &mut self.pushes,
            BusDirection::Pull => &mut self.pulls,
        };
        occurrence.count =
            occurrence
                .count
                .checked_add(1)
                .ok_or(BusDebugError::OccurrenceCountOverflow {
                    air: location.air,
                    declaration: location.declaration,
                    row: location.row,
                })?;

        // A bounded sample keeps diagnostics useful without retaining every repeated row.
        if occurrence.locations.len() < LOCATION_LIMIT {
            occurrence.locations.push(location);
        }
        Ok(())
    }
}

/// Concrete row context used to replay one symbolic expression.
#[derive(Clone, Copy)]
struct RowEvaluator<'a, F: Field> {
    /// AIR position in statement order.
    air: usize,
    /// Declaration position within the AIR.
    declaration: usize,
    /// Row position within the trace.
    row: usize,
    /// Number of rows in the trace.
    height: usize,
    /// Committed trace columns.
    main: &'a Table<F>,
    /// Optional fixed trace columns.
    preprocessed: Option<&'a Table<F>>,
    /// Public inputs supplied to the AIR.
    public_values: &'a [F],
}

impl<F: Field> RowEvaluator<'_, F> {
    fn evaluate(&self, expression: &SymbolicExpression<F>) -> Result<F, BusDebugError> {
        // Arithmetic nodes are replayed over the concrete row values.
        match expression {
            SymbolicExpr::Leaf(leaf) => self.evaluate_leaf(leaf),
            SymbolicExpr::Add { x, y, .. } => Ok(self.evaluate(x)? + self.evaluate(y)?),
            SymbolicExpr::Sub { x, y, .. } => Ok(self.evaluate(x)? - self.evaluate(y)?),
            SymbolicExpr::Neg { x, .. } => Ok(-self.evaluate(x)?),
            SymbolicExpr::Mul { x, y, .. } => Ok(self.evaluate(x)? * self.evaluate(y)?),
        }
    }

    fn evaluate_leaf(&self, leaf: &BaseLeaf<F>) -> Result<F, BusDebugError> {
        // Symbolic planning rejects next-row and periodic accesses before replay.
        match leaf {
            BaseLeaf::Variable(variable) => match variable.entry {
                BaseEntry::Main { offset: 0 } => {
                    if variable.index >= self.main.num_polys() {
                        return Err(BusDebugError::MainColumnOutOfRange {
                            air: self.air,
                            declaration: self.declaration,
                            row: self.row,
                            column: variable.index,
                            width: self.main.num_polys(),
                        });
                    }
                    Ok(self.main.poly(variable.index).as_slice()[self.row])
                }
                BaseEntry::Preprocessed { offset: 0 } => {
                    let Some(preprocessed) = self.preprocessed else {
                        return Err(BusDebugError::MissingPreprocessedTrace {
                            air: self.air,
                            declaration: self.declaration,
                            row: self.row,
                            column: variable.index,
                        });
                    };
                    if variable.index >= preprocessed.num_polys() {
                        return Err(BusDebugError::PreprocessedColumnOutOfRange {
                            air: self.air,
                            declaration: self.declaration,
                            row: self.row,
                            column: variable.index,
                            width: preprocessed.num_polys(),
                        });
                    }
                    Ok(preprocessed.poly(variable.index).as_slice()[self.row])
                }
                BaseEntry::Public => self.public_values.get(variable.index).copied().ok_or(
                    BusDebugError::PublicValueOutOfRange {
                        air: self.air,
                        declaration: self.declaration,
                        row: self.row,
                        index: variable.index,
                        len: self.public_values.len(),
                    },
                ),
                BaseEntry::Main { .. } | BaseEntry::Preprocessed { .. } | BaseEntry::Periodic => {
                    unreachable!("bus planning rejected an unsupported expression access")
                }
            },
            BaseLeaf::IsFirstRow => Ok(F::from_bool(self.row == 0)),
            BaseLeaf::IsLastRow => Ok(F::from_bool(self.row + 1 == self.height)),
            BaseLeaf::IsTransition => Ok(F::from_bool(self.row + 1 < self.height)),
            BaseLeaf::Constant(value) => Ok(*value),
        }
    }
}

fn activation_is_set<F: Field>(
    activation: &BusActivation<SymbolicExpression<F>>,
    evaluator: &RowEvaluator<'_, F>,
) -> Result<bool, BusDebugError> {
    // Unconditional declarations contribute on every row.
    let BusActivation::Boolean(expression) = activation else {
        return Ok(true);
    };

    // AIR semantics require a conditional activation to be exactly zero or one.
    let value = evaluator.evaluate(expression)?;
    if value == F::ZERO {
        Ok(false)
    } else if value == F::ONE {
        Ok(true)
    } else {
        Err(BusDebugError::NonBooleanActivation {
            air: evaluator.air,
            declaration: evaluator.declaration,
            row: evaluator.row,
        })
    }
}

#[cfg(test)]
mod tests {
    use alloc::string::ToString;
    use alloc::vec;

    use p3_air::symbolic::{BaseEntry, SymbolicVariable};
    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;
    use p3_matrix::dense::RowMajorMatrix;

    use super::*;

    type F = BabyBear;

    fn table(columns: &[&[u64]]) -> Table<F> {
        // Polynomial-major storage places each complete trace column contiguously.
        let width = columns[0].len();
        let values = columns
            .iter()
            .flat_map(|column| column.iter().copied().map(F::from_u64))
            .collect();
        Table::new(RowMajorMatrix::new(values, width))
    }

    fn current(index: usize) -> SymbolicExpression<F> {
        // Current-row references are the only trace access accepted by bus planning.
        SymbolicVariable::new(BaseEntry::Main { offset: 0 }, index).into()
    }

    fn interaction(
        name: &str,
        direction: BusDirection,
        fields: Vec<SymbolicExpression<F>>,
        activation: BusActivation<SymbolicExpression<F>>,
    ) -> SymbolicBusInteraction<F> {
        // Tests construct the same symbolic records an AIR emits through its builder.
        SymbolicBusInteraction {
            bus_name: name.to_string(),
            direction,
            fields,
            activation,
        }
    }

    #[test]
    fn balanced_mixed_height_instances_have_no_diagnostics() {
        // Fixture state:
        //     tall push rows : [3, 5, 7, 11]
        //     short pulls    : [3, 5]
        //     tall pulls     : [7, 11]
        let tall_push = table(&[&[3, 5, 7, 11]]);
        let short_pull = table(&[&[3, 5]]);
        let tall_pull = table(&[&[7, 11, 0, 0], &[1, 1, 0, 0]]);
        let pushes = [interaction(
            "memory",
            BusDirection::Push,
            vec![current(0)],
            BusActivation::Always,
        )];
        let pulls = [interaction(
            "memory",
            BusDirection::Pull,
            vec![current(0)],
            BusActivation::Always,
        )];
        let selected_pulls = [interaction(
            "memory",
            BusDirection::Pull,
            vec![current(0)],
            BusActivation::Boolean(current(1)),
        )];
        let instances = [
            BusDebugInstance {
                main: &tall_push,
                preprocessed: None,
                public_values: &[],
                interactions: &pushes,
            },
            BusDebugInstance {
                main: &short_pull,
                preprocessed: None,
                public_values: &[],
                interactions: &pulls,
            },
            BusDebugInstance {
                main: &tall_pull,
                preprocessed: None,
                public_values: &[],
                interactions: &selected_pulls,
            },
        ];

        // Every produced tuple has one matching consumed occurrence.
        let report = check_bus_balance(&instances).unwrap();
        assert!(report.is_balanced());
    }

    #[test]
    fn mismatch_reports_exact_counts_and_first_sources() {
        // Fixture state: tuple 9 is pushed twice and pulled once.
        let push = table(&[&[9, 9]]);
        let pull = table(&[&[9]]);
        let pushes = [interaction(
            "dispatch",
            BusDirection::Push,
            vec![current(0)],
            BusActivation::Always,
        )];
        let pulls = [interaction(
            "dispatch",
            BusDirection::Pull,
            vec![current(0)],
            BusActivation::Always,
        )];
        let instances = [
            BusDebugInstance {
                main: &push,
                preprocessed: None,
                public_values: &[],
                interactions: &pushes,
            },
            BusDebugInstance {
                main: &pull,
                preprocessed: None,
                public_values: &[],
                interactions: &pulls,
            },
        ];

        // The report names the bus, tuple, exact multiplicities, and emitting rows.
        let report = check_bus_balance(&instances).unwrap();
        assert_eq!(report.imbalances.len(), 1);
        let mismatch = &report.imbalances[0];
        assert_eq!(mismatch.bus_name, "dispatch");
        assert_eq!(mismatch.tuple, vec![F::from_u64(9)]);
        assert_eq!(mismatch.pushes.count, 2);
        assert_eq!(mismatch.pulls.count, 1);
        assert_eq!(mismatch.pushes.locations[0].row, 0);
        assert_eq!(mismatch.pushes.locations[1].row, 1);
        assert_eq!(mismatch.pulls.locations[0].air, 1);
    }

    #[test]
    fn named_buses_and_directions_never_cancel_in_the_field() {
        // Fixture state: the same tuple is pushed on two independently named buses.
        let trace = table(&[&[7]]);
        let interactions = [
            interaction(
                "a",
                BusDirection::Push,
                vec![current(0)],
                BusActivation::Always,
            ),
            interaction(
                "b",
                BusDirection::Push,
                vec![current(0)],
                BusActivation::Always,
            ),
        ];
        let instances = [BusDebugInstance {
            main: &trace,
            preprocessed: None,
            public_values: &[],
            interactions: &interactions,
        }];

        // Characteristic two cannot erase either structural direction or bus identity.
        let report = check_bus_balance(&instances).unwrap();
        assert_eq!(report.imbalances.len(), 2);
        assert_eq!(report.imbalances[0].bus_name, "a");
        assert_eq!(report.imbalances[1].bus_name, "b");
    }

    #[test]
    fn non_boolean_activation_is_rejected_at_its_source_row() {
        // Fixture state: the second row carries activation 2 instead of 0 or 1.
        let trace = table(&[&[4, 5], &[1, 2]]);
        let interactions = [interaction(
            "memory",
            BusDirection::Push,
            vec![current(0)],
            BusActivation::Boolean(current(1)),
        )];
        let instances = [BusDebugInstance {
            main: &trace,
            preprocessed: None,
            public_values: &[],
            interactions: &interactions,
        }];

        // Honest-prover diagnostics fail before constructing a misleading balance report.
        assert_eq!(
            check_bus_balance(&instances),
            Err(BusDebugError::NonBooleanActivation {
                air: 0,
                declaration: 0,
                row: 1,
            })
        );
    }
}
