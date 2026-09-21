//! Out-of-circuit diagnostics for binary-native bus declarations.
//!
//! Replay reads declarations from a symbolic profile rather than from a concrete per-row builder.
//!
//! This lets a caller that holds a profile and its committed traces, but no longer the AIR itself, still be diagnosed.

use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;

use hashbrown::HashMap;
use p3_air::symbolic::{BaseEntry, BaseLeaf, SymbolicExpr, SymbolicExpression};
use p3_field::{ExtensionField, Field};
use p3_sumcheck::layout::{ColumnView, Table};
use thiserror::Error;

use crate::{
    BusActivation, BusDirection, BusExpressionLocation, BusPlan, BusPlanError, BusPlanInput,
    BusSymbolicBuilder, SymbolicBusInteraction, UnsupportedBusAccess,
};

/// Concrete data required to replay one AIR's bus declarations.
#[derive(Clone, Copy)]
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

impl<F: Field> fmt::Debug for BusDebugInstance<'_, F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Rendering the traces themselves would dump every committed value.
        f.debug_struct("BusDebugInstance")
            .field(
                "main",
                &format_args!(
                    "2^{} x {}",
                    self.main.num_variables(),
                    self.main.num_polys()
                ),
            )
            .field("preprocessed", &self.preprocessed.map(Table::num_polys))
            .field("public_values", &self.public_values.len())
            .field("declarations", &self.interactions.len())
            .finish()
    }
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

/// Bounds a diagnostic run applies to the text it retains.
///
/// Both bounds shrink the report only, never the exact multiplicities it carries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BusDebugLimits {
    /// Maximum source rows retained for one tuple and one direction.
    pub locations: usize,
    /// Maximum unmatched tuples retained in the report.
    pub imbalances: usize,
}

impl BusDebugLimits {
    /// Source rows retained per tuple and direction unless the caller asks for more.
    pub const DEFAULT_LOCATIONS: usize = 4;

    /// Unmatched tuples retained unless the caller asks for more.
    pub const DEFAULT_IMBALANCES: usize = 64;
}

impl Default for BusDebugLimits {
    fn default() -> Self {
        // Both defaults keep a printed report readable on a terminal.
        Self {
            locations: Self::DEFAULT_LOCATIONS,
            imbalances: Self::DEFAULT_IMBALANCES,
        }
    }
}

/// Source of one tuple occurrence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct BusDebugLocation {
    /// AIR position in statement order.
    pub air: usize,
    /// Declaration position within the AIR.
    pub declaration: usize,
    /// Row position within the owning trace.
    pub row: usize,
}

/// Exact occurrence count with a bounded sample of excess sources.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct BusDebugOccurrence {
    /// Number of active declaration occurrences carrying the tuple.
    ///
    /// One row contributes once per declaration it activates, so a row can be counted more than once.
    pub count: usize,
    /// Sources of the occurrences this side holds in excess of the other side.
    ///
    /// The two sides are paired in traversal order, so this sample starts at the first unpaired occurrence.
    ///
    /// The smaller side is fully paired and therefore carries no sample at all.
    ///
    /// The sample is bounded, so it covers the whole excess only when it holds as many occurrences as the two counts differ by.
    ///
    /// When it does, removing those occurrences restores balance for this tuple, though with both sides non-empty no single one is the unique culprit.
    pub locations: Vec<BusDebugLocation>,
}

/// One tuple whose push and pull multiplicities differ.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct BusImbalance<F> {
    /// Name of the independently balanced multiset.
    pub bus_name: String,
    /// Payload field elements in declaration order.
    pub tuple: Vec<F>,
    /// Produced occurrences and the sources of any excess.
    pub pushes: BusDebugOccurrence,
    /// Consumed occurrences and the sources of any excess.
    pub pulls: BusDebugOccurrence,
}

/// Total unmatched tuples found in one named multiset.
///
/// The total is exact whether or not the report retained every tuple behind it.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct BusUnmatched {
    /// Name of the independently balanced multiset.
    pub bus_name: String,
    /// Unmatched tuples found, counting those the retained list dropped.
    pub unmatched: usize,
}

/// Deterministic list of unmatched bus tuples.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct BusDebugReport<F> {
    /// Retained unmatched tuples in first-occurrence order.
    pub imbalances: Vec<BusImbalance<F>>,
    /// Exact unmatched totals of every multiset that does not balance, in layout order.
    ///
    /// A multiset appears here even when the retained list dropped all of its tuples.
    pub buses: Vec<BusUnmatched>,
}

impl<F> BusDebugReport<F> {
    /// Returns whether every named multiset balances exactly.
    #[must_use]
    pub const fn is_balanced(&self) -> bool {
        // A multiset is listed exactly when it holds at least one unmatched tuple.
        self.buses.is_empty()
    }

    /// Returns the unmatched tuples dropped because the retained list was already full.
    #[must_use]
    pub fn unreported(&self) -> usize {
        // Every retained tuple is counted in exactly one of the totals.
        self.total_unmatched() - self.imbalances.len()
    }

    /// Returns the unmatched tuples found across every named multiset.
    #[must_use]
    pub fn total_unmatched(&self) -> usize {
        self.buses.iter().map(|bus| bus.unmatched).sum()
    }

    /// Returns whether unmatched tuples were dropped from the retained list.
    #[must_use]
    pub fn is_truncated(&self) -> bool {
        self.unreported() != 0
    }
}

impl<F: fmt::Display> fmt::Display for BusDebugReport<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_balanced() {
            return write!(f, "every named bus balances");
        }

        // Walking the totals rather than the retained list also reaches a multiset that retained nothing.
        for bus in &self.buses {
            let group = self
                .imbalances
                .iter()
                .filter(|entry| entry.bus_name == bus.bus_name);
            let shown = group.clone().count();
            write!(
                f,
                "bus {:?}: {} unmatched tuples",
                bus.bus_name, bus.unmatched
            )?;
            if shown != bus.unmatched {
                write!(f, " ({shown} shown)")?;
            }
            writeln!(f)?;
            for entry in group {
                write_imbalance(f, entry)?;
            }
        }
        Ok(())
    }
}

fn write_imbalance<F: fmt::Display>(
    f: &mut fmt::Formatter<'_>,
    imbalance: &BusImbalance<F>,
) -> fmt::Result {
    write!(f, "  (")?;
    for (slot, value) in imbalance.tuple.iter().enumerate() {
        if slot != 0 {
            write!(f, ", ")?;
        }
        write!(f, "{value}")?;
    }
    write!(
        f,
        ")  push {}  pull {}",
        imbalance.pushes.count, imbalance.pulls.count
    )?;

    // Exactly one side carries the excess, and only that side holds a sample.
    let (label, occurrence) = if imbalance.pushes.count > imbalance.pulls.count {
        ("push", &imbalance.pushes)
    } else {
        ("pull", &imbalance.pulls)
    };
    let excess = imbalance.pushes.count.abs_diff(imbalance.pulls.count);
    if !occurrence.locations.is_empty() {
        write!(f, "  excess {label} at ")?;
        for (index, location) in occurrence.locations.iter().enumerate() {
            if index != 0 {
                write!(f, ", ")?;
            }
            write!(
                f,
                "air {} decl {} row {}",
                location.air, location.declaration, location.row
            )?;
        }
        if occurrence.locations.len() < excess {
            write!(f, ", ...")?;
        }
    }
    writeln!(f)
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
        "binary-bus AIR {air} declaration {declaration} reads main column {column}, but the trace has width {width}"
    )]
    MainColumnOutOfRange {
        /// AIR position in statement order.
        air: usize,
        /// Declaration position within the AIR.
        declaration: usize,
        /// Missing column position.
        column: usize,
        /// Available column count.
        width: usize,
    },
    /// An expression reads fixed data that was not supplied.
    #[error(
        "binary-bus AIR {air} declaration {declaration} requires preprocessed column {column}, but no preprocessed trace was supplied"
    )]
    MissingPreprocessedTrace {
        /// AIR position in statement order.
        air: usize,
        /// Declaration position within the AIR.
        declaration: usize,
        /// Requested column position.
        column: usize,
    },
    /// An expression reads a missing fixed column.
    #[error(
        "binary-bus AIR {air} declaration {declaration} reads preprocessed column {column}, but the trace has width {width}"
    )]
    PreprocessedColumnOutOfRange {
        /// AIR position in statement order.
        air: usize,
        /// Declaration position within the AIR.
        declaration: usize,
        /// Missing column position.
        column: usize,
        /// Available column count.
        width: usize,
    },
    /// An expression reads a missing public input.
    #[error(
        "binary-bus AIR {air} declaration {declaration} reads public value {index}, but only {len} values were supplied"
    )]
    PublicValueOutOfRange {
        /// AIR position in statement order.
        air: usize,
        /// Declaration position within the AIR.
        declaration: usize,
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
    /// A padded trace whose declarations are unconditional records its padding tuple on every padding row.
    ///
    /// That contribution is real rather than an artefact, so an AIR has to gate its declarations off the padding.
    ///
    /// # Errors
    ///
    /// Returns an error when the declaration layout or concrete inputs are malformed.
    ///
    /// # Performance
    ///
    /// - Memory is linear in the number of distinct active tuples.
    /// - Output is bounded by the default limits.
    /// - Time is linear in rows times distinct expression nodes, never in expression paths.
    /// - Rows are scanned a second time only when something is unmatched.
    /// - The routine is intended for trusted development traces.
    pub fn check(instances: &[BusDebugInstance<'_, F>]) -> Result<Self, BusDebugError> {
        Self::check_with_limits(instances, BusDebugLimits::default())
    }

    /// Replays every declaration under caller-chosen retention bounds.
    ///
    /// Raising a bound widens the report without changing any reported count.
    ///
    /// # Errors
    ///
    /// Returns an error when the declaration layout or concrete inputs are malformed.
    pub fn check_with_limits(
        instances: &[BusDebugInstance<'_, F>],
        limits: BusDebugLimits,
    ) -> Result<Self, BusDebugError> {
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
                buses: Vec::new(),
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
        let domains = plan
            .domains()
            .iter()
            .enumerate()
            .map(|(index, domain)| (domain.name.as_str(), index))
            .collect::<HashMap<_, _>>();

        // One hash index per named multiset provides constant-time grouping.
        // Keying per bus lets the probe borrow the tuple instead of cloning it on every occurrence.
        let mut tuple_indices = alloc::vec![HashMap::<Vec<F>, usize>::new(); plan.domains().len()];
        let mut counts = Vec::<TupleCounts>::new();

        // The first scan retains no source rows, because which rows are unmatched is unknown until the totals are in.
        replay(
            instances,
            &domains,
            &mut |bus, tuple, direction, location| {
                let index = match tuple_indices[bus].get(tuple) {
                    Some(&index) => index,
                    None => {
                        let index = counts.len();
                        tuple_indices[bus].insert(tuple.to_vec(), index);
                        counts.push(TupleCounts {
                            bus,
                            pushes: 0,
                            pulls: 0,
                        });
                        index
                    }
                };
                let count = counts[index].side(direction);
                *count = count
                    .checked_add(1)
                    .ok_or(BusDebugError::OccurrenceCountOverflow {
                        air: location.air,
                        declaration: location.declaration,
                        row: location.row,
                    })?;
                Ok(())
            },
        )?;

        // Equal multiplicities cancel as integers rather than as field elements.
        let mut positions = HashMap::<usize, usize>::new();
        let mut imbalances = Vec::new();

        // The totals are accumulated before the cap applies, so truncation cannot shrink them.
        let mut unmatched = alloc::vec![0usize; plan.domains().len()];
        for (index, entry) in counts.iter().enumerate() {
            if entry.pushes == entry.pulls {
                continue;
            }
            unmatched[entry.bus] += 1;
            if imbalances.len() == limits.imbalances {
                continue;
            }
            positions.insert(index, imbalances.len());
            imbalances.push(BusImbalance {
                bus_name: plan.domains()[entry.bus].name.clone(),
                tuple: Vec::new(),
                pushes: BusDebugOccurrence {
                    count: entry.pushes,
                    locations: Vec::new(),
                },
                pulls: BusDebugOccurrence {
                    count: entry.pulls,
                    locations: Vec::new(),
                },
            });
        }
        drop(counts);
        let buses = plan
            .domains()
            .iter()
            .zip(&unmatched)
            .filter(|&(_, &unmatched)| unmatched != 0)
            .map(|(domain, &unmatched)| BusUnmatched {
                bus_name: domain.name.clone(),
                unmatched,
            })
            .collect::<Vec<_>>();
        if positions.is_empty() {
            return Ok(Self { imbalances, buses });
        }

        // The second scan samples only the occurrences a side holds beyond the pairing with the other side.
        let mut ordinals = alloc::vec![[0usize; 2]; imbalances.len()];
        replay(
            instances,
            &domains,
            &mut |bus, tuple, direction, location| {
                let Some(&index) = tuple_indices[bus].get(tuple) else {
                    return Ok(());
                };
                let Some(&position) = positions.get(&index) else {
                    return Ok(());
                };
                let imbalance = &mut imbalances[position];
                if imbalance.tuple.is_empty() {
                    imbalance.tuple = tuple.to_vec();
                }
                let matched = imbalance.pushes.count.min(imbalance.pulls.count);
                let side = side_index(direction);
                let ordinal = ordinals[position][side];
                ordinals[position][side] += 1;
                let occurrence = match direction {
                    BusDirection::Push => &mut imbalance.pushes,
                    BusDirection::Pull => &mut imbalance.pulls,
                };
                if ordinal >= matched && occurrence.locations.len() < limits.locations {
                    occurrence.locations.push(location);
                }
                Ok(())
            },
        )?;

        Ok(Self { imbalances, buses })
    }
}

/// Integer multiplicities accumulated for one named tuple.
struct TupleCounts {
    /// Stable position of the named multiset.
    bus: usize,
    /// Produced multiplicity.
    pushes: usize,
    /// Consumed multiplicity.
    pulls: usize,
}

impl TupleCounts {
    const fn side(&mut self, direction: BusDirection) -> &mut usize {
        // Direction selects an integer counter rather than a field sign.
        match direction {
            BusDirection::Push => &mut self.pushes,
            BusDirection::Pull => &mut self.pulls,
        }
    }
}

const fn side_index(direction: BusDirection) -> usize {
    match direction {
        BusDirection::Push => 0,
        BusDirection::Pull => 1,
    }
}

/// Walks every active declaration of every instance in deterministic order.
fn replay<F: Field>(
    instances: &[BusDebugInstance<'_, F>],
    domains: &HashMap<&str, usize>,
    visit: &mut impl FnMut(usize, &[F], BusDirection, BusDebugLocation) -> Result<(), BusDebugError>,
) -> Result<(), BusDebugError> {
    let mut slots = Vec::new();
    let mut tuple = Vec::new();

    for (air, instance) in instances.iter().enumerate() {
        // An AIR that declares nothing cannot contribute, so its rows are never visited.
        if instance.interactions.is_empty() {
            continue;
        }

        // Expression shape does not vary by row, so every declaration is flattened once per instance.
        let declarations = compile_instance(air, instance, domains)?;
        let height = 1usize << instance.main.num_variables();

        for row in 0..height {
            for (declaration, compiled) in declarations.iter().enumerate() {
                // Inactive rows contribute the product identity and no multiset occurrence.
                if let Some(activation) = &compiled.activation {
                    let value = activation.run(row, height, &mut slots);
                    if value == F::ZERO {
                        continue;
                    }
                    if value != F::ONE {
                        return Err(BusDebugError::NonBooleanActivation {
                            air,
                            declaration,
                            row,
                        });
                    }
                }

                tuple.clear();
                tuple.extend(
                    compiled
                        .fields
                        .iter()
                        .map(|field| field.run(row, height, &mut slots)),
                );
                visit(
                    compiled.bus,
                    &tuple,
                    compiled.direction,
                    BusDebugLocation {
                        air,
                        declaration,
                        row,
                    },
                )?;
            }
        }
    }
    Ok(())
}

/// One declaration whose expressions are resolved against concrete columns.
struct CompiledDeclaration<'a, F: Field> {
    /// Stable position of the named multiset.
    bus: usize,
    /// Side of the multiset equality receiving the tuple.
    direction: BusDirection,
    /// Row activation program, when the declaration is conditional.
    activation: Option<Program<'a, F>>,
    /// Payload programs in slot order.
    fields: Vec<Program<'a, F>>,
}

/// One resolved node of a flattened expression.
enum Op<'a, F: Field> {
    /// A whole trace column, indexed by row at evaluation time.
    ///
    /// The view decodes a cell from either dense or packed storage.
    Column(ColumnView<'a, F>),
    /// A value shared by every row.
    Constant(F),
    /// Indicator of the first row.
    FirstRow,
    /// Indicator of the last row.
    LastRow,
    /// Indicator of every row but the last.
    Transition,
    /// Additive inverse of an earlier node.
    Neg(usize),
    /// Sum of two earlier nodes.
    Add(usize, usize),
    /// Difference of two earlier nodes.
    Sub(usize, usize),
    /// Product of two earlier nodes.
    Mul(usize, usize),
}

/// An expression flattened into an order where every operand precedes its use.
struct Program<'a, F: Field>(Vec<Op<'a, F>>);

impl<F: Field> Program<'_, F> {
    fn run(&self, row: usize, height: usize, slots: &mut Vec<F>) -> F {
        // Operands always precede their use, so one forward pass over reusable storage suffices.
        slots.clear();
        for op in &self.0 {
            let value = match *op {
                Op::Column(column) => column.value(row),
                Op::Constant(value) => value,
                Op::FirstRow => F::from_bool(row == 0),
                Op::LastRow => F::from_bool(row + 1 == height),
                Op::Transition => F::from_bool(row + 1 < height),
                Op::Neg(x) => -slots[x],
                Op::Add(x, y) => slots[x] + slots[y],
                Op::Sub(x, y) => slots[x] - slots[y],
                Op::Mul(x, y) => slots[x] * slots[y],
            };
            slots.push(value);
        }
        *slots.last().expect("a flattened expression holds one node")
    }
}

/// Operator whose operand values are already on the stack.
enum Combine {
    /// Additive inverse of one operand.
    Neg,
    /// Sum of two operands.
    Add,
    /// Difference of two operands.
    Sub,
    /// Product of two operands.
    Mul,
}

/// One step of the iterative walk over a shared expression graph.
enum Step<'a, F: Field> {
    /// Schedule a node, after a lookup in the completed-node index.
    Enter(&'a SymbolicExpression<F>),
    /// Combine the operands a scheduled node's children already produced.
    Leave(Combine, &'a SymbolicExpression<F>),
}

fn compile_instance<'a, F: Field>(
    air: usize,
    instance: &BusDebugInstance<'a, F>,
    domains: &HashMap<&str, usize>,
) -> Result<Vec<CompiledDeclaration<'a, F>>, BusDebugError> {
    let main = instance.main.columns().collect::<Vec<_>>();
    let preprocessed = instance
        .preprocessed
        .map(|table| table.columns().collect::<Vec<_>>());

    instance
        .interactions
        .iter()
        .enumerate()
        .map(|(declaration, interaction)| {
            let compiler = Compiler {
                air,
                declaration,
                main: &main,
                preprocessed: preprocessed.as_deref(),
                public_values: instance.public_values,
            };
            let fields = interaction
                .fields
                .iter()
                .enumerate()
                .map(|(slot, field)| compiler.compile(BusExpressionLocation::Field(slot), field))
                .collect::<Result<Vec<_>, _>>()?;
            let activation = match &interaction.activation {
                BusActivation::Always => None,
                BusActivation::Boolean(expression) => {
                    Some(compiler.compile(BusExpressionLocation::Activation, expression)?)
                }
            };
            Ok(CompiledDeclaration {
                bus: domains[interaction.bus_name.as_str()],
                direction: interaction.direction,
                activation,
                fields,
            })
        })
        .collect()
}

/// Resolves symbolic leaves of one declaration against concrete data.
struct Compiler<'a, 'b, F: Field> {
    /// AIR position in statement order.
    air: usize,
    /// Declaration position within the AIR.
    declaration: usize,
    /// Committed trace columns.
    main: &'b [ColumnView<'a, F>],
    /// Optional fixed trace columns.
    preprocessed: Option<&'b [ColumnView<'a, F>]>,
    /// Public inputs supplied to the AIR.
    public_values: &'a [F],
}

impl<'a, F: Field> Compiler<'a, '_, F> {
    fn compile(
        &self,
        location: BusExpressionLocation,
        expression: &SymbolicExpression<F>,
    ) -> Result<Program<'a, F>, BusDebugError> {
        // Arithmetic nodes share their operands, so the expression is a graph rather than a tree.
        // Walking it per path costs time exponential in the depth, which a bit recomposition reaches immediately.
        let mut ops = Vec::new();
        let mut done = HashMap::<*const SymbolicExpression<F>, usize>::new();
        let mut steps = alloc::vec![Step::Enter(expression)];
        let mut operands = Vec::<usize>::new();

        while let Some(step) = steps.pop() {
            match step {
                Step::Enter(node) => {
                    if let Some(&slot) = done.get(&core::ptr::from_ref(node)) {
                        operands.push(slot);
                        continue;
                    }
                    match node {
                        SymbolicExpr::Leaf(leaf) => {
                            let op = self.compile_leaf(location, leaf)?;
                            ops.push(op);
                            done.insert(core::ptr::from_ref(node), ops.len() - 1);
                            operands.push(ops.len() - 1);
                        }
                        SymbolicExpr::Neg { x, .. } => {
                            steps.push(Step::Leave(Combine::Neg, node));
                            steps.push(Step::Enter(x));
                        }
                        SymbolicExpr::Add { x, y, .. } => {
                            steps.push(Step::Leave(Combine::Add, node));
                            steps.push(Step::Enter(y));
                            steps.push(Step::Enter(x));
                        }
                        SymbolicExpr::Sub { x, y, .. } => {
                            steps.push(Step::Leave(Combine::Sub, node));
                            steps.push(Step::Enter(y));
                            steps.push(Step::Enter(x));
                        }
                        SymbolicExpr::Mul { x, y, .. } => {
                            steps.push(Step::Leave(Combine::Mul, node));
                            steps.push(Step::Enter(y));
                            steps.push(Step::Enter(x));
                        }
                    }
                }
                Step::Leave(combine, node) => {
                    // Operands were pushed left before right, so the stack returns them in reverse.
                    let mut pop = || operands.pop().expect("a scheduled node left its operands");
                    let right = pop();
                    let op = match combine {
                        Combine::Neg => Op::Neg(right),
                        Combine::Add => Op::Add(pop(), right),
                        Combine::Sub => Op::Sub(pop(), right),
                        Combine::Mul => Op::Mul(pop(), right),
                    };
                    ops.push(op);
                    done.insert(core::ptr::from_ref(node), ops.len() - 1);
                    operands.push(ops.len() - 1);
                }
            }
        }

        Ok(Program(ops))
    }

    fn compile_leaf(
        &self,
        location: BusExpressionLocation,
        leaf: &BaseLeaf<F>,
    ) -> Result<Op<'a, F>, BusDebugError> {
        // Shape faults do not vary by row, so they are named once rather than once per row.
        match leaf {
            BaseLeaf::Variable(variable) => {
                match variable.entry {
                    BaseEntry::Main { offset: 0 } => self
                        .main
                        .get(variable.index)
                        .copied()
                        .map(Op::Column)
                        .ok_or(BusDebugError::MainColumnOutOfRange {
                            air: self.air,
                            declaration: self.declaration,
                            column: variable.index,
                            width: self.main.len(),
                        }),
                    BaseEntry::Preprocessed { offset: 0 } => {
                        let Some(preprocessed) = self.preprocessed else {
                            return Err(BusDebugError::MissingPreprocessedTrace {
                                air: self.air,
                                declaration: self.declaration,
                                column: variable.index,
                            });
                        };
                        preprocessed
                            .get(variable.index)
                            .copied()
                            .map(Op::Column)
                            .ok_or(BusDebugError::PreprocessedColumnOutOfRange {
                                air: self.air,
                                declaration: self.declaration,
                                column: variable.index,
                                width: preprocessed.len(),
                            })
                    }
                    BaseEntry::Public => self
                        .public_values
                        .get(variable.index)
                        .copied()
                        .map(Op::Constant)
                        .ok_or(BusDebugError::PublicValueOutOfRange {
                            air: self.air,
                            declaration: self.declaration,
                            index: variable.index,
                            len: self.public_values.len(),
                        }),
                    BaseEntry::Main { offset } => {
                        Err(self.unsupported(location, UnsupportedBusAccess::MainOffset(offset)))
                    }
                    BaseEntry::Preprocessed { offset } => Err(self
                        .unsupported(location, UnsupportedBusAccess::PreprocessedOffset(offset))),
                    BaseEntry::Periodic => {
                        Err(self.unsupported(location, UnsupportedBusAccess::Periodic))
                    }
                }
            }
            BaseLeaf::IsFirstRow => Ok(Op::FirstRow),
            BaseLeaf::IsLastRow => Ok(Op::LastRow),
            BaseLeaf::IsTransition => Ok(Op::Transition),
            BaseLeaf::Constant(value) => Ok(Op::Constant(*value)),
        }
    }

    fn unsupported(
        &self,
        location: BusExpressionLocation,
        access: UnsupportedBusAccess,
    ) -> BusDebugError {
        // Planning rejects these accesses first, so a report names them instead of panicking.
        BusPlanError::UnsupportedExpression {
            air: self.air,
            declaration: self.declaration,
            location,
            access,
        }
        .into()
    }
}

#[cfg(test)]
mod tests {
    use alloc::string::ToString;
    use alloc::{format, vec};

    use p3_air::symbolic::{AirLayout, BaseEntry, SymbolicVariable};
    use p3_air::{Air, BaseAir, WindowAccess};
    use p3_baby_bear::BabyBear;
    use p3_binary_field::BinaryField128;
    use p3_field::PrimeCharacteristicRing;
    use p3_matrix::dense::RowMajorMatrix;
    use proptest::prelude::*;

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
    fn mismatch_reports_exact_counts_and_unmatched_sources() {
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

        // The report names the bus, tuple and exact multiplicities.
        let report = BusDebugReport::check(&instances).unwrap();
        assert_eq!(report.imbalances.len(), 1);
        let mismatch = &report.imbalances[0];
        assert_eq!(mismatch.bus_name, "dispatch");
        assert_eq!(mismatch.tuple, vec![F::from_u64(9)]);
        assert_eq!(mismatch.pushes.count, 2);
        assert_eq!(mismatch.pulls.count, 1);

        // The first push pairs with the only pull, so the sample starts at the second push.
        assert_eq!(
            mismatch.pushes.locations,
            vec![BusDebugLocation {
                air: 0,
                declaration: 0,
                row: 1,
            }],
        );
        assert!(mismatch.pulls.locations.is_empty());
    }

    #[test]
    fn the_sample_skips_the_occurrences_that_cancel_against_the_other_side() {
        // Fixture state: tuple 9 is pushed eight times and pulled seven times.
        let push = table::<F>(&[&[9, 9, 9, 9, 9, 9, 9, 9]]);
        let pull_four = table::<F>(&[&[9, 9, 9, 9]]);
        let pull_two = table::<F>(&[&[9, 9]]);
        let pull_one = table::<F>(&[&[9]]);
        let pushes = [interaction(
            "dense",
            BusDirection::Push,
            vec![current(0)],
            BusActivation::Always,
        )];
        let pulls = [interaction(
            "dense",
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
                main: &pull_four,
                preprocessed: None,
                public_values: &[],
                interactions: &pulls,
            },
            BusDebugInstance {
                main: &pull_two,
                preprocessed: None,
                public_values: &[],
                interactions: &pulls,
            },
            BusDebugInstance {
                main: &pull_one,
                preprocessed: None,
                public_values: &[],
                interactions: &pulls,
            },
        ];

        // Seven of the eight pushes cancel, so only the eighth is named.
        let report = BusDebugReport::check(&instances).unwrap();
        let mismatch = &report.imbalances[0];
        assert_eq!((mismatch.pushes.count, mismatch.pulls.count), (8, 7));
        assert_eq!(
            mismatch.pushes.locations,
            vec![BusDebugLocation {
                air: 0,
                declaration: 0,
                row: 7,
            }],
        );
        assert!(mismatch.pulls.locations.is_empty());
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
        assert_eq!(pushes.locations.len(), BusDebugLimits::DEFAULT_LOCATIONS);
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

    #[test]
    fn padding_rows_contribute_like_any_other_row() {
        // Fixture state: two real payloads pulled from a trace padded to four rows with zeros.
        let push = table::<F>(&[&[5, 6]]);
        let pull = table::<F>(&[&[5, 6, 0, 0]]);
        let pushes = [interaction(
            "padded",
            BusDirection::Push,
            vec![current(0)],
            BusActivation::Always,
        )];
        let pulls = [interaction(
            "padded",
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

        // The in-circuit argument consumes the padding too, so the debugger reports it rather than hiding it.
        let report = BusDebugReport::check(&instances).unwrap();
        assert_eq!(report.imbalances.len(), 1);
        let padding = &report.imbalances[0];
        assert_eq!(padding.tuple, vec![F::ZERO]);
        assert_eq!((padding.pushes.count, padding.pulls.count), (0, 2));
        assert_eq!(
            padding
                .pulls
                .locations
                .iter()
                .map(|location| location.row)
                .collect::<Vec<_>>(),
            vec![2, 3],
        );
    }

    #[test]
    fn an_instance_without_declarations_contributes_nothing() {
        // Fixture state: a tall AIR declaring no tuple beside a balanced pair of short AIRs.
        let idle = table::<F>(&[&[1, 2, 3, 4]]);
        let push = table::<F>(&[&[7]]);
        let pull = table::<F>(&[&[7]]);
        let none: [SymbolicBusInteraction<F>; 0] = [];
        let pushes = [interaction(
            "quiet",
            BusDirection::Push,
            vec![current(0)],
            BusActivation::Always,
        )];
        let pulls = [interaction(
            "quiet",
            BusDirection::Pull,
            vec![current(0)],
            BusActivation::Always,
        )];
        let instances = [
            BusDebugInstance {
                main: &idle,
                preprocessed: None,
                public_values: &[],
                interactions: &none,
            },
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

        // The silent AIR neither contributes a tuple nor disturbs the positions of the others.
        assert!(BusDebugReport::check(&instances).unwrap().is_balanced());
    }

    #[test]
    fn the_retained_list_is_capped_and_reports_what_it_dropped() {
        // Fixture state: eight distinct payloads pushed with no pull side at all.
        let trace = table::<F>(&[&[1, 2, 3, 4, 5, 6, 7, 8]]);
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
        let limits = BusDebugLimits {
            locations: 1,
            imbalances: 3,
        };

        // Three tuples survive the cap and the remaining five are still counted in the total.
        let report = BusDebugReport::check_with_limits(&instances, limits).unwrap();
        assert_eq!(report.imbalances.len(), 3);
        assert_eq!(report.unreported(), 5);
        assert_eq!(report.total_unmatched(), 8);
        assert_eq!(
            report.buses,
            vec![BusUnmatched {
                bus_name: "order".to_string(),
                unmatched: 8,
            }],
        );
        assert!(report.is_truncated());
        assert!(!report.is_balanced());

        // The header carries the true total and says how much of it the body holds.
        assert_eq!(
            format!("{report}"),
            "bus \"order\": 8 unmatched tuples (3 shown)\n\
             \x20 (1)  push 1  pull 0  excess push at air 0 decl 0 row 0\n\
             \x20 (2)  push 1  pull 0  excess push at air 0 decl 0 row 1\n\
             \x20 (3)  push 1  pull 0  excess push at air 0 decl 0 row 2\n",
        );
    }

    #[test]
    fn a_balanced_statement_renders_as_one_line() {
        // Fixture state: one payload pushed and pulled once.
        let trace = table::<F>(&[&[3]]);
        let interactions = [
            interaction(
                "even",
                BusDirection::Push,
                vec![current(0)],
                BusActivation::Always,
            ),
            interaction(
                "even",
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

        let report = BusDebugReport::check(&instances).unwrap();
        assert_eq!(format!("{report}"), "every named bus balances");
    }

    #[test]
    fn a_fault_in_the_middle_instance_is_attributed_to_that_instance() {
        // Fixture state: four payloads pushed, split across two pulling AIRs, with the middle one wrong.
        let push = table::<F>(&[&[1, 2, 3, 4]]);
        let middle = table::<F>(&[&[1, 9]]);
        let last = table::<F>(&[&[3, 4]]);
        let pushes = [interaction(
            "split",
            BusDirection::Push,
            vec![current(0)],
            BusActivation::Always,
        )];
        let pulls = [interaction(
            "split",
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
                main: &middle,
                preprocessed: None,
                public_values: &[],
                interactions: &pulls,
            },
            BusDebugInstance {
                main: &last,
                preprocessed: None,
                public_values: &[],
                interactions: &pulls,
            },
        ];

        // The lost payload names the pushing AIR and the surplus one names the middle AIR.
        let report = BusDebugReport::check(&instances).unwrap();
        assert_eq!(report.imbalances.len(), 2);
        assert_eq!(report.imbalances[0].tuple, vec![F::from_u64(2)]);
        assert_eq!(
            report.imbalances[0].pushes.locations,
            vec![BusDebugLocation {
                air: 0,
                declaration: 0,
                row: 1,
            }],
        );
        assert_eq!(report.imbalances[1].tuple, vec![F::from_u64(9)]);
        assert_eq!(
            report.imbalances[1].pulls.locations,
            vec![BusDebugLocation {
                air: 1,
                declaration: 0,
                row: 1,
            }],
        );
    }

    #[test]
    fn two_declarations_producing_one_tuple_are_told_apart() {
        // Fixture state: a single row whose two columns hold the same payload, pushed once from each.
        let trace = table::<F>(&[&[7], &[7]]);
        let interactions = [
            interaction(
                "twin",
                BusDirection::Push,
                vec![current(0)],
                BusActivation::Always,
            ),
            interaction(
                "twin",
                BusDirection::Push,
                vec![current(1)],
                BusActivation::Always,
            ),
        ];
        let instances = [BusDebugInstance {
            main: &trace,
            preprocessed: None,
            public_values: &[],
            interactions: &interactions,
        }];

        // Both occurrences land on row zero, so only the declaration position separates them.
        let report = BusDebugReport::check(&instances).unwrap();
        assert_eq!(report.imbalances.len(), 1);
        assert_eq!(
            report.imbalances[0]
                .pushes
                .locations
                .iter()
                .map(|location| (location.declaration, location.row))
                .collect::<Vec<_>>(),
            vec![(0, 0), (1, 0)],
        );
    }

    #[test]
    fn an_unsupported_leaf_is_named_rather_than_panicked_on() {
        // Fixture state: a next-row access and a periodic access, both rejected by planning first.
        let column = [F::ZERO];
        let main = [ColumnView::Dense(column.as_slice())];
        let compiler = Compiler {
            air: 2,
            declaration: 1,
            main: &main,
            preprocessed: None,
            public_values: &[],
        };

        let next_row = BaseLeaf::Variable(SymbolicVariable::new(BaseEntry::Main { offset: 1 }, 0));
        assert_eq!(
            compiler
                .compile_leaf(BusExpressionLocation::Field(0), &next_row)
                .err()
                .unwrap(),
            BusDebugError::Plan(BusPlanError::UnsupportedExpression {
                air: 2,
                declaration: 1,
                location: BusExpressionLocation::Field(0),
                access: UnsupportedBusAccess::MainOffset(1),
            }),
        );

        let periodic = BaseLeaf::Variable(SymbolicVariable::new(BaseEntry::Periodic, 0));
        assert_eq!(
            compiler
                .compile_leaf(BusExpressionLocation::Activation, &periodic)
                .err()
                .unwrap(),
            BusDebugError::Plan(BusPlanError::UnsupportedExpression {
                air: 2,
                declaration: 1,
                location: BusExpressionLocation::Activation,
                access: UnsupportedBusAccess::Periodic,
            }),
        );
    }

    proptest! {
        #[test]
        fn a_single_mutated_row_is_named_by_the_report(
            log_height in 1usize..8,
            offset in 0usize..256,
            delta in 1u64..1000,
        ) {
            // Distinct payloads give every tuple multiplicity one, so the excess is attributable to one row.
            let height = 1usize << log_height;
            let row = offset % height;
            let values = (0..height as u64).map(|i| i * 4096 + 1).collect::<Vec<_>>();
            let push = table::<F>(&[&values]);

            let mut mutated = values.clone();
            mutated[row] += delta;
            let pull = table::<F>(&[&mutated]);

            let pushes = [interaction(
                "mutation",
                BusDirection::Push,
                vec![current(0)],
                BusActivation::Always,
            )];
            let pulls = [interaction(
                "mutation",
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

            // One payload lost its consumer and one gained a consumer that produces nothing.
            let report = BusDebugReport::check(&instances).unwrap();
            prop_assert_eq!(report.imbalances.len(), 2);

            let lost = &report.imbalances[0];
            prop_assert_eq!(&lost.tuple, &vec![F::from_u64(values[row])]);
            prop_assert_eq!((lost.pushes.count, lost.pulls.count), (1, 0));
            prop_assert_eq!(lost.pushes.locations[0].row, row);
            prop_assert_eq!(lost.pushes.locations[0].air, 0);

            let gained = &report.imbalances[1];
            prop_assert_eq!(&gained.tuple, &vec![F::from_u64(mutated[row])]);
            prop_assert_eq!((gained.pushes.count, gained.pulls.count), (0, 1));
            prop_assert_eq!(gained.pulls.locations[0].row, row);
            prop_assert_eq!(gained.pulls.locations[0].air, 1);
        }
    }

    #[test]
    fn packed_boolean_storage_is_read_without_materializing_a_dense_trace() {
        // Fixture state: two main columns and one activation column over eight logical rows.
        //
        // Bit i of a packed word is row i, so the dense spellings below are the words 77, 150 and 107.
        let packed_main = Table::<B>::from_packed_bits(RowMajorMatrix::new(vec![77, 150], 2), 3);
        let packed_fixed = Table::<B>::from_packed_bits(RowMajorMatrix::new(vec![107], 1), 3);
        let dense_main = table::<B>(&[&[1, 0, 1, 1, 0, 0, 1, 0], &[0, 1, 1, 0, 1, 0, 0, 1]]);
        let dense_fixed = table::<B>(&[&[1, 1, 0, 1, 0, 1, 1, 0]]);

        let interactions = [interaction(
            "bits",
            BusDirection::Push,
            vec![current(0), current(1)],
            BusActivation::Boolean(
                SymbolicVariable::<B>::new(BaseEntry::Preprocessed { offset: 0 }, 0).into(),
            ),
        )];
        let report = |main, preprocessed| {
            BusDebugReport::check(&[BusDebugInstance {
                main,
                preprocessed: Some(preprocessed),
                public_values: &[],
                interactions: &interactions,
            }])
            .unwrap()
        };

        // Rows 0, 1, 3, 5 and 6 are active, pushing (1,0) three times and (0,1) and (0,0) once.
        let packed = report(&packed_main, &packed_fixed);
        assert_eq!(
            packed.buses,
            vec![BusUnmatched {
                bus_name: "bits".to_string(),
                unmatched: 3,
            }],
        );
        assert_eq!(
            packed
                .imbalances
                .iter()
                .map(|entry| (entry.tuple.clone(), entry.pushes.count, entry.pulls.count))
                .collect::<Vec<_>>(),
            vec![
                (vec![B::ONE, B::ZERO], 3, 0),
                (vec![B::ZERO, B::ONE], 1, 0),
                (vec![B::ZERO, B::ZERO], 1, 0),
            ],
        );

        // Both storage layouts describe the same logical trace, so they must diagnose identically.
        assert_eq!(packed, report(&dense_main, &dense_fixed));
    }

    #[test]
    fn a_bus_whose_tuples_all_fall_past_the_cap_is_still_named() {
        // Fixture state: four unmatched tuples in the first AIR and two in the second.
        let first = table::<F>(&[&[1, 2, 3, 4]]);
        let second = table::<F>(&[&[5, 6]]);
        let alpha = [interaction(
            "alpha",
            BusDirection::Push,
            vec![current(0)],
            BusActivation::Always,
        )];
        let zulu = [interaction(
            "zulu",
            BusDirection::Push,
            vec![current(0)],
            BusActivation::Always,
        )];
        let instances = [
            BusDebugInstance {
                main: &first,
                preprocessed: None,
                public_values: &[],
                interactions: &alpha,
            },
            BusDebugInstance {
                main: &second,
                preprocessed: None,
                public_values: &[],
                interactions: &zulu,
            },
        ];
        let limits = BusDebugLimits {
            locations: 1,
            imbalances: 4,
        };

        // The first AIR fills the retained list, so nothing of the second survives it.
        let report = BusDebugReport::check_with_limits(&instances, limits).unwrap();
        assert_eq!(report.imbalances.len(), 4);
        assert!(
            report
                .imbalances
                .iter()
                .all(|entry| entry.bus_name == "alpha")
        );
        assert_eq!(
            report.buses,
            vec![
                BusUnmatched {
                    bus_name: "alpha".to_string(),
                    unmatched: 4,
                },
                BusUnmatched {
                    bus_name: "zulu".to_string(),
                    unmatched: 2,
                },
            ],
        );
        assert_eq!(report.unreported(), 2);

        // The bus with no retained entry is named with its own total rather than omitted.
        assert_eq!(
            format!("{report}"),
            "bus \"alpha\": 4 unmatched tuples\n\
             \x20 (1)  push 1  pull 0  excess push at air 0 decl 0 row 0\n\
             \x20 (2)  push 1  pull 0  excess push at air 0 decl 0 row 1\n\
             \x20 (3)  push 1  pull 0  excess push at air 0 decl 0 row 2\n\
             \x20 (4)  push 1  pull 0  excess push at air 0 decl 0 row 3\n\
             bus \"zulu\": 2 unmatched tuples (0 shown)\n",
        );
    }
}
