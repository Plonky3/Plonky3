//! Out-of-circuit diagnostics for binary-native bus declarations.
//!
//! Replay reads declarations from a symbolic profile rather than from a concrete per-row builder.
//!
//! This lets a caller that holds a profile and its committed traces, but no longer the AIR itself, still be diagnosed.

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
    /// Both traces hold one polynomial per trace column, not one per trace row.
    ///
    /// # Errors
    ///
    /// Returns an error when a trace or the public inputs disagree with the profile's declared shape.
    pub fn new<EF: ExtensionField<F>>(
        main: &'a Table<F>,
        preprocessed: Option<&'a Table<F>>,
        public_values: &'a [F],
        profile: &'a BusSymbolicBuilder<F, EF>,
    ) -> Result<Self, BusDebugError> {
        // A transposed trace satisfies every table invariant, so the declared shape is the only guard.
        // Without it the replay would read trace rows as columns and report a fabricated imbalance.
        let layout = profile.layout();
        if main.num_polys() != layout.main_width {
            return Err(BusDebugError::MainWidthMismatch {
                expected: layout.main_width,
                actual: main.num_polys(),
            });
        }
        let preprocessed_polys = preprocessed.map_or(0, Table::num_polys);
        if preprocessed_polys != layout.preprocessed_width {
            return Err(BusDebugError::PreprocessedWidthMismatch {
                expected: layout.preprocessed_width,
                actual: preprocessed_polys,
            });
        }
        if public_values.len() != layout.num_public_values {
            return Err(BusDebugError::PublicValueCountMismatch {
                expected: layout.num_public_values,
                actual: public_values.len(),
            });
        }

        // The profile remains the sole source of symbolic declarations.
        // This prevents a caller from passing an unrelated raw interaction slice.
        Ok(Self {
            main,
            preprocessed,
            public_values,
            interactions: profile.interactions(),
        })
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
    /// The committed trace holds a different number of columns than the AIR declares.
    #[error("binary-bus main trace holds {actual} columns, but the AIR declares {expected}")]
    MainWidthMismatch {
        /// Column count declared by the AIR.
        expected: usize,
        /// Column count found in the supplied trace.
        actual: usize,
    },
    /// The fixed trace holds a different number of columns than the AIR declares.
    #[error(
        "binary-bus preprocessed trace holds {actual} columns, but the AIR declares {expected}"
    )]
    PreprocessedWidthMismatch {
        /// Column count declared by the AIR.
        expected: usize,
        /// Column count found in the supplied trace.
        actual: usize,
    },
    /// A different number of public inputs was supplied than the AIR declares.
    #[error("binary-bus instance supplies {actual} public values, but the AIR declares {expected}")]
    PublicValueCountMismatch {
        /// Public-input count declared by the AIR.
        expected: usize,
        /// Public-input count supplied by the caller.
        actual: usize,
    },
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

impl<F: Field> BusDebugReport<F> {
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
    /// - Time is linear in rows times distinct expression nodes, never in expression paths.
    /// - The routine is intended for trusted development traces.
    pub fn check(instances: &[BusDebugInstance<'_, F>]) -> Result<Self, BusDebugError> {
        // Build the same deterministic named-bus layout used by the proof protocol.
        let inputs = instances
            .iter()
            .map(|instance| BusPlanInput {
                log_height: instance.main.num_variables(),
                interactions: instance.interactions,
            })
            .collect::<Vec<_>>();
        let Some(plan) = BusPlan::build(&inputs)? else {
            return Ok(Self {
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

        // One hash index per named multiset provides constant-time grouping.
        // Keying per bus lets the probe borrow the tuple instead of cloning it on every occurrence.
        // The parallel vector preserves deterministic first-occurrence order for diagnostics.
        let mut entry_indices = alloc::vec![HashMap::<Vec<F>, usize>::new(); plan.domains().len()];
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

                    // Reuse an existing tuple accumulator or append one deterministic entry.
                    let entry = if let Some(&index) = entry_indices[bus].get(&tuple) {
                        &mut entries[index]
                    } else {
                        let index = entries.len();
                        entry_indices[bus].insert(tuple.clone(), index);
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

        Ok(Self { imbalances })
    }
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

/// One step of the iterative traversal over a shared expression graph.
enum Step<'a, F: Field> {
    /// Schedule a node, after a lookup in the completed-value index.
    Enter(&'a SymbolicExpression<F>),
    /// Combine the operand values a scheduled node's children already produced.
    Leave(&'a SymbolicExpression<F>),
}

impl<F: Field> RowEvaluator<'_, F> {
    fn evaluate(&self, expression: &SymbolicExpression<F>) -> Result<F, BusDebugError> {
        // Arithmetic nodes share their operands, so the expression is a graph rather than a tree.
        // Walking it per path costs time exponential in the depth, which a bit recomposition reaches immediately.
        // An explicit stack keyed on node identity evaluates each distinct node once and bounds the recursion depth.
        if let SymbolicExpr::Leaf(leaf) = expression {
            return self.evaluate_leaf(leaf);
        }

        let mut done = HashMap::<*const SymbolicExpression<F>, F>::new();
        let mut steps = alloc::vec![Step::Enter(expression)];
        let mut operands = Vec::<F>::new();

        while let Some(step) = steps.pop() {
            match step {
                Step::Enter(node) => {
                    if let Some(&value) = done.get(&core::ptr::from_ref(node)) {
                        operands.push(value);
                        continue;
                    }
                    match node {
                        SymbolicExpr::Leaf(leaf) => {
                            let value = self.evaluate_leaf(leaf)?;
                            done.insert(core::ptr::from_ref(node), value);
                            operands.push(value);
                        }
                        SymbolicExpr::Neg { x, .. } => {
                            steps.push(Step::Leave(node));
                            steps.push(Step::Enter(x));
                        }
                        SymbolicExpr::Add { x, y, .. }
                        | SymbolicExpr::Sub { x, y, .. }
                        | SymbolicExpr::Mul { x, y, .. } => {
                            steps.push(Step::Leave(node));
                            steps.push(Step::Enter(y));
                            steps.push(Step::Enter(x));
                        }
                    }
                }
                Step::Leave(node) => {
                    // Operands were pushed left before right, so the stack returns them in reverse.
                    let mut pop = || operands.pop().expect("a scheduled node left its operands");
                    let right = pop();
                    let value = match node {
                        SymbolicExpr::Neg { .. } => -right,
                        SymbolicExpr::Add { .. } => pop() + right,
                        SymbolicExpr::Sub { .. } => pop() - right,
                        SymbolicExpr::Mul { .. } => pop() * right,
                        SymbolicExpr::Leaf(_) => {
                            unreachable!("a leaf is completed when it is first entered")
                        }
                    };
                    done.insert(core::ptr::from_ref(node), value);
                    operands.push(value);
                }
            }
        }

        Ok(operands.pop().expect("the root node produced its value"))
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

    use p3_air::symbolic::{AirLayout, BaseEntry, SymbolicVariable};
    use p3_air::{Air, BaseAir, WindowAccess};
    use p3_baby_bear::BabyBear;
    use p3_binary_field::BinaryField128;
    use p3_field::PrimeCharacteristicRing;
    use p3_matrix::dense::RowMajorMatrix;

    use super::*;
    use crate::BusInteractionBuilder;

    type F = BabyBear;
    type B = BinaryField128;

    fn table<K: Field>(columns: &[&[u64]]) -> Table<K> {
        // Polynomial-major storage places each complete trace column contiguously.
        let width = columns[0].len();
        let values = columns
            .iter()
            .flat_map(|column| column.iter().copied().map(K::from_u64))
            .collect();
        Table::new(RowMajorMatrix::new(values, width))
    }

    fn current<K: Field>(index: usize) -> SymbolicExpression<K> {
        // Current-row references are the only trace access accepted by bus planning.
        SymbolicVariable::new(BaseEntry::Main { offset: 0 }, index).into()
    }

    fn fixed(index: usize) -> SymbolicExpression<F> {
        // Fixed columns are read through the same current-row access.
        SymbolicVariable::new(BaseEntry::Preprocessed { offset: 0 }, index).into()
    }

    fn public(index: usize) -> SymbolicExpression<F> {
        // Public inputs are shared by every row of the trace.
        SymbolicVariable::new(BaseEntry::Public, index).into()
    }

    fn interaction<K: Field>(
        name: &str,
        direction: BusDirection,
        fields: Vec<SymbolicExpression<K>>,
        activation: BusActivation<SymbolicExpression<K>>,
    ) -> SymbolicBusInteraction<K> {
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
        let tall_push = table::<F>(&[&[3, 5, 7, 11]]);
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
        let report = BusDebugReport::check(&instances).unwrap();
        assert!(report.is_balanced());
    }

    #[test]
    fn mismatch_reports_exact_counts_and_first_sources() {
        // Fixture state: tuple 9 is pushed twice and pulled once.
        let push = table::<F>(&[&[9, 9]]);
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
        let report = BusDebugReport::check(&instances).unwrap();
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
        // Fixture state: one tuple is pushed on two named buses, and a third bus sees a push and a pull.
        // The field has characteristic two, so a signed encoding would cancel all four records.
        let trace = table::<B>(&[&[7]]);
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
            interaction(
                "c",
                BusDirection::Push,
                vec![current(0)],
                BusActivation::Always,
            ),
            interaction(
                "c",
                BusDirection::Pull,
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

        // Bus identity survives: the two single-sided buses are reported separately.
        let report = BusDebugReport::check(&instances).unwrap();
        assert_eq!(report.imbalances.len(), 2);
        assert_eq!(report.imbalances[0].bus_name, "a");
        assert_eq!(report.imbalances[1].bus_name, "b");
        assert_eq!(report.imbalances[0].tuple, vec![B::from_u64(7)]);

        // Direction survives: the matched pair on the third bus cancels as integers, not as field elements.
        assert!(report.imbalances.iter().all(|entry| entry.bus_name != "c"));
    }

    // Two-column AIR whose payload column is pulled back on selected rows.
    struct PairAir;

    impl BaseAir<F> for PairAir {
        fn width(&self) -> usize {
            // One column carries payloads and one carries row activation.
            2
        }
    }

    impl<AB: BusInteractionBuilder<F = F>> Air<AB> for PairAir {
        fn eval(&self, builder: &mut AB) {
            let main = builder.main();
            let row = main.current_slice();
            let value: AB::Expr = row[0].into();
            let selector: AB::Expr = row[1].into();
            builder.push_bus_interaction(
                "pairs",
                BusDirection::Push,
                [value.clone()],
                BusActivation::Always,
            );
            builder.push_bus_interaction(
                "pairs",
                BusDirection::Pull,
                [value],
                BusActivation::Boolean(selector),
            );
        }
    }

    #[test]
    fn a_real_air_reaches_the_checker_through_its_symbolic_profile() {
        // Fixture state: payloads 1, 2, 3, 4 with the last row's pull switched off.
        let profile = BusSymbolicBuilder::<F>::from_air(&PairAir, AirLayout::from_air(&PairAir));
        let main = table::<F>(&[&[1, 2, 3, 4], &[1, 1, 1, 0]]);
        let instance = BusDebugInstance::new(&main, None, &[], &profile).unwrap();

        // The declaration order and the column mapping both come from the AIR, not from the test.
        let report = BusDebugReport::check(&[instance]).unwrap();
        assert_eq!(report.imbalances.len(), 1);
        assert_eq!(report.imbalances[0].tuple, vec![F::from_u64(4)]);
        assert_eq!(report.imbalances[0].pushes.count, 1);
        assert_eq!(report.imbalances[0].pulls.count, 0);
        assert_eq!(report.imbalances[0].pushes.locations[0].row, 3);
    }

    #[test]
    fn concrete_data_must_match_the_shape_the_profile_was_built_for() {
        // Fixture state: the same eight payload values laid out as four rows of two columns.
        let profile = BusSymbolicBuilder::<F>::from_air(&PairAir, AirLayout::from_air(&PairAir));
        let transposed = table::<F>(&[&[1, 1], &[2, 1], &[3, 1], &[4, 0]]);

        // Row-major data satisfies every table invariant, so only the declared width catches it.
        assert_eq!(
            BusDebugInstance::new(&transposed, None, &[], &profile).unwrap_err(),
            BusDebugError::MainWidthMismatch {
                expected: 2,
                actual: 4,
            }
        );

        // Fixed columns the AIR never declared are rejected for the same reason.
        let main = table::<F>(&[&[1, 2, 3, 4], &[1, 1, 1, 0]]);
        let extra = table::<F>(&[&[0, 0, 0, 0]]);
        assert_eq!(
            BusDebugInstance::new(&main, Some(&extra), &[], &profile).unwrap_err(),
            BusDebugError::PreprocessedWidthMismatch {
                expected: 0,
                actual: 1,
            }
        );

        // Public inputs the AIR never declared are rejected for the same reason.
        assert_eq!(
            BusDebugInstance::new(&main, None, &[F::ONE], &profile).unwrap_err(),
            BusDebugError::PublicValueCountMismatch {
                expected: 0,
                actual: 1,
            }
        );
    }

    #[test]
    fn shared_operands_are_replayed_once_per_node_rather_than_once_per_path() {
        // Fixture state: forty doublings share their operand, giving eighty nodes and 2^40 paths.
        const DEPTH: usize = 40;
        let mut field = current::<F>(0);
        let mut expected = F::ONE;
        for _ in 0..DEPTH {
            field = field.clone() + field;
            expected += expected;
        }

        let trace = table::<F>(&[&[1]]);
        let interactions = [interaction(
            "deep",
            BusDirection::Push,
            vec![field],
            BusActivation::Always,
        )];
        let instances = [BusDebugInstance {
            main: &trace,
            preprocessed: None,
            public_values: &[],
            interactions: &interactions,
        }];

        // A path-wise traversal would not finish; the reported payload is one doubled forty times.
        let report = BusDebugReport::check(&instances).unwrap();
        assert_eq!(report.imbalances[0].tuple, vec![expected]);
    }

    #[test]
    fn the_source_sample_is_bounded_while_the_count_stays_exact() {
        // Fixture state: eight rows all carrying the same payload, pushed and never pulled.
        let trace = table::<F>(&[&[6, 6, 6, 6, 6, 6, 6, 6]]);
        let interactions = [interaction(
            "repeat",
            BusDirection::Push,
            vec![current(0)],
            BusActivation::Always,
        )];
        let instances = [BusDebugInstance {
            main: &trace,
            preprocessed: None,
            public_values: &[],
            interactions: &interactions,
        }];

        // The count keeps rising after the sample fills, and the sample holds the first rows.
        let report = BusDebugReport::check(&instances).unwrap();
        let pushes = &report.imbalances[0].pushes;
        assert_eq!(pushes.count, 8);
        assert_eq!(pushes.locations.len(), LOCATION_LIMIT);
        assert_eq!(
            pushes
                .locations
                .iter()
                .map(|location| location.row)
                .collect::<Vec<_>>(),
            vec![0, 1, 2, 3],
        );
    }

    #[test]
    fn distinct_tuples_are_reported_in_first_occurrence_order() {
        // Fixture state: payload 9 appears before payload 4, and neither is alphabetically first.
        let trace = table::<F>(&[&[9, 4, 9, 4]]);
        let interactions = [interaction(
            "order",
            BusDirection::Push,
            vec![current(0)],
            BusActivation::Always,
        )];
        let instances = [BusDebugInstance {
            main: &trace,
            preprocessed: None,
            public_values: &[],
            interactions: &interactions,
        }];

        // Grouping is by hash, but the report follows the order the rows were scanned in.
        let report = BusDebugReport::check(&instances).unwrap();
        let tuples = report
            .imbalances
            .iter()
            .map(|entry| entry.tuple.clone())
            .collect::<Vec<_>>();
        assert_eq!(tuples, vec![vec![F::from_u64(9)], vec![F::from_u64(4)]]);
        assert_eq!(report.imbalances[0].pushes.count, 2);
    }

    #[test]
    fn row_selectors_follow_the_position_of_the_row_in_its_trace() {
        // Fixture state: four payload rows, each selector picking a different subset of them.
        let trace = table::<F>(&[&[10, 20, 30, 40]]);
        let selected = |leaf: BaseLeaf<F>| {
            let interactions = [interaction(
                "rows",
                BusDirection::Push,
                vec![current(0)],
                BusActivation::Boolean(SymbolicExpr::Leaf(leaf)),
            )];
            let instances = [BusDebugInstance {
                main: &trace,
                preprocessed: None,
                public_values: &[],
                interactions: &interactions,
            }];
            BusDebugReport::check(&instances)
                .unwrap()
                .imbalances
                .iter()
                .map(|entry| entry.tuple[0])
                .collect::<Vec<_>>()
        };

        // Each selector picks exactly the rows its name describes.
        assert_eq!(selected(BaseLeaf::IsFirstRow), vec![F::from_u64(10)]);
        assert_eq!(selected(BaseLeaf::IsLastRow), vec![F::from_u64(40)]);
        assert_eq!(
            selected(BaseLeaf::IsTransition),
            vec![F::from_u64(10), F::from_u64(20), F::from_u64(30)],
        );
    }

    #[test]
    fn fixed_columns_share_the_committed_row_domain() {
        // Fixture state: payloads pushed from the committed trace and pulled from the fixed one.
        let main = table::<F>(&[&[2, 3]]);
        let fixed_trace = table::<F>(&[&[2, 3]]);
        let pushes = [interaction(
            "fixed",
            BusDirection::Push,
            vec![current(0)],
            BusActivation::Always,
        )];
        let pulls = [interaction(
            "fixed",
            BusDirection::Pull,
            vec![fixed(0)],
            BusActivation::Always,
        )];
        let balanced = [
            BusDebugInstance {
                main: &main,
                preprocessed: None,
                public_values: &[],
                interactions: &pushes,
            },
            BusDebugInstance {
                main: &main,
                preprocessed: Some(&fixed_trace),
                public_values: &[],
                interactions: &pulls,
            },
        ];

        // Reading a fixed column yields the same values as the committed column it mirrors.
        assert!(BusDebugReport::check(&balanced).unwrap().is_balanced());

        // A fixed trace over a different row domain cannot be paired with the committed one.
        let short = table::<F>(&[&[2]]);
        let mismatched = [BusDebugInstance {
            main: &main,
            preprocessed: Some(&short),
            public_values: &[],
            interactions: &pulls,
        }];
        assert_eq!(
            BusDebugReport::check(&mismatched),
            Err(BusDebugError::PreprocessedHeightMismatch {
                air: 0,
                expected: 1,
                actual: 0,
            })
        );

        // A fixed column the caller never supplied is named rather than silently skipped.
        let absent = [BusDebugInstance {
            main: &main,
            preprocessed: None,
            public_values: &[],
            interactions: &pulls,
        }];
        assert_eq!(
            BusDebugReport::check(&absent),
            Err(BusDebugError::MissingPreprocessedTrace {
                air: 0,
                declaration: 0,
                row: 0,
                column: 0,
            })
        );

        // A fixed column past the end of the supplied trace is named the same way.
        let beyond = [interaction(
            "fixed",
            BusDirection::Pull,
            vec![fixed(2)],
            BusActivation::Always,
        )];
        let out_of_range = [BusDebugInstance {
            main: &main,
            preprocessed: Some(&fixed_trace),
            public_values: &[],
            interactions: &beyond,
        }];
        assert_eq!(
            BusDebugReport::check(&out_of_range),
            Err(BusDebugError::PreprocessedColumnOutOfRange {
                air: 0,
                declaration: 0,
                row: 0,
                column: 2,
                width: 1,
            })
        );
    }

    #[test]
    fn reads_past_the_supplied_data_are_named_rather_than_wrapped() {
        // Fixture state: a one-column trace and no public inputs at all.
        let trace = table::<F>(&[&[5]]);
        let beyond_main = [interaction(
            "range",
            BusDirection::Push,
            vec![current(3)],
            BusActivation::Always,
        )];
        let main_instances = [BusDebugInstance {
            main: &trace,
            preprocessed: None,
            public_values: &[],
            interactions: &beyond_main,
        }];
        assert_eq!(
            BusDebugReport::check(&main_instances),
            Err(BusDebugError::MainColumnOutOfRange {
                air: 0,
                declaration: 0,
                row: 0,
                column: 3,
                width: 1,
            })
        );

        let beyond_public = [interaction(
            "range",
            BusDirection::Push,
            vec![public(1)],
            BusActivation::Always,
        )];
        let public_instances = [BusDebugInstance {
            main: &trace,
            preprocessed: None,
            public_values: &[F::ONE],
            interactions: &beyond_public,
        }];
        assert_eq!(
            BusDebugReport::check(&public_instances),
            Err(BusDebugError::PublicValueOutOfRange {
                air: 0,
                declaration: 0,
                row: 0,
                index: 1,
                len: 1,
            })
        );
    }

    #[test]
    fn non_boolean_activation_is_rejected_at_its_source_row() {
        // Fixture state: the second row carries activation 2 instead of 0 or 1.
        let trace = table::<F>(&[&[4, 5], &[1, 2]]);
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
            BusDebugReport::check(&instances),
            Err(BusDebugError::NonBooleanActivation {
                air: 0,
                declaration: 0,
                row: 1,
            })
        );
    }
}
