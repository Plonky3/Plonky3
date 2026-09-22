//! One byte string naming everything a table asserts.
//!
//! Two tables that differ anywhere a prover can observe produce different strings.
//!
//! Nothing here is ever decoded, so the string only has to be injective.

use alloc::vec::Vec;

use hashbrown::HashMap;
use p3_air::symbolic::{BaseEntry, BaseLeaf, ExtEntry, ExtLeaf, SymbolicExpr, SymbolicExpression};
use p3_air::{Air, BaseAir, BoundaryEnd};
use p3_field::{ExtensionField, Field, RawDataSerializable};
use p3_lookup::{InteractionSymbolicBuilder, TraceWindow};

/// Write everything one table fixes into a fresh byte string.
///
/// The two fields come first, so a table read over one pair never matches another.
///
/// A symbolic pass runs the evaluation alone, so what a table declares beside it goes in here:
///
/// - the cells the backend pins to public values;
/// - the periodic tables;
/// - the windows opened at the next row;
/// - the degree hint.
///
/// The contents of a fixed trace stay out, because the verifying key is what binds those.
///
/// Mutually exclusive interactions are left out, because this backend refuses them outright.
pub(super) fn encode<F, EF, A>(air: &A, builder: &InteractionSymbolicBuilder<F, EF>) -> Vec<u8>
where
    F: Field,
    EF: ExtensionField<F>,
    A: BaseAir<F> + Air<InteractionSymbolicBuilder<F, EF>>,
{
    let mut arena = Arena::default();
    arena.blob(&F::order().to_bytes_le());
    arena.count(EF::DIMENSION);
    arena.blob(&EF::algebra_id());

    let cells = air.public_boundary_io();
    arena.count(cells.len());
    for cell in cells {
        arena.count(cell.column);
        arena.byte(match cell.end {
            BoundaryEnd::First => 0,
            BoundaryEnd::Last => 1,
        });
        arena.count(cell.public_value);
    }

    let periodic = air.periodic_columns();
    arena.count(air.num_periodic_columns());
    arena.count(periodic.len());
    for column in periodic.iter() {
        arena.count(column.len());
        for &value in column {
            arena.scalar(value);
        }
    }

    arena.indices(&air.main_next_row_columns());
    arena.indices(&air.preprocessed_next_row_columns());
    match air.max_constraint_degree() {
        Some(degree) => {
            arena.byte(1);
            arena.count(degree);
        }
        None => arena.byte(0),
    }

    let base = builder.base_constraints();
    arena.count(base.len());
    for constraint in &base {
        arena.expression(constraint);
    }

    let extension = builder.extension_constraints();
    arena.count(extension.len());
    for constraint in &extension {
        arena.expression(constraint);
    }

    let global = builder.global_interactions();
    arena.count(global.len());
    for interaction in global {
        arena.text(&interaction.bus_name);
        arena.u32(interaction.count_weight);
        arena.count(interaction.fields.len());
        for field in &interaction.fields {
            arena.expression(field);
        }
        arena.expression(&interaction.count);
    }

    // A pair owns its multiplicity, so every one is copied out first and kept alive together.
    //
    // A node is named by where it sits in memory, and a temporary would let two share one address.
    let local = builder.local_interactions();
    let multiplicities = local
        .iter()
        .flat_map(|interaction| interaction.tuples.iter())
        .map(|(_, count)| count.clone().into_parts().0)
        .collect::<Vec<SymbolicExpression<F>>>();
    let mut multiplicities = multiplicities.iter();

    arena.count(local.len());
    for interaction in local {
        arena.count(interaction.tuples.len());
        for (fields, count) in &interaction.tuples {
            arena.count(fields.len());
            for field in fields {
                arena.expression(field);
            }
            arena.u32(count.weight());
            arena.expression(multiplicities.next().expect("one per tuple"));
        }
    }

    let reads = builder.indexed_reads();
    arena.count(reads.len());
    for read in reads {
        arena.text(&read.table);
        arena.count(read.position);
        arena.indices(&read.payload);
    }

    let tables = builder.indexed_tables();
    arena.count(tables.len());
    for table in tables {
        arena.text(&table.name);
        arena.byte(match table.window {
            TraceWindow::Main => 0,
            TraceWindow::Preprocessed => 1,
        });
        arena.indices(&table.columns);
    }

    arena.bytes
}

/// A growing byte string, plus where every expression node already written ended up.
///
/// Every token stands for one value.
///
/// A repeated subtree becomes a reference to the value it left behind, not a second copy.
#[derive(Default)]
struct Arena {
    bytes: Vec<u8>,
    written: HashMap<usize, u64>,
    values: u64,
}

impl Arena {
    fn byte(&mut self, value: u8) {
        self.bytes.push(value);
    }

    fn u32(&mut self, value: u32) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    /// Write a number seven bits at a time, smallest first, so short ones stay short.
    fn varint(&mut self, mut value: u64) {
        while value >= 0x80 {
            self.bytes.push((value as u8) | 0x80);
            value >>= 7;
        }
        self.bytes.push(value as u8);
    }

    fn count(&mut self, value: usize) {
        self.varint(value as u64);
    }

    fn blob(&mut self, value: &[u8]) {
        self.count(value.len());
        self.bytes.extend_from_slice(value);
    }

    fn text(&mut self, value: &str) {
        self.blob(value.as_bytes());
    }

    fn indices(&mut self, value: &[usize]) {
        self.count(value.len());
        for &index in value {
            self.count(index);
        }
    }

    fn scalar<G: RawDataSerializable>(&mut self, value: G) {
        self.bytes.extend(value.into_bytes());
    }

    /// Write every node of one expression, children before the node that uses them.
    fn expression<A: Leaf>(&mut self, expression: &SymbolicExpr<A>) {
        let key = core::ptr::from_ref(expression) as usize;
        if let Some(&position) = self.written.get(&key) {
            let back = self.values - position;
            self.byte(9);
            self.varint(back);
            self.values += 1;
            return;
        }

        match expression {
            SymbolicExpr::Leaf(leaf) => leaf.write(self),
            SymbolicExpr::Add { x, y, .. } => self.pair(5, x, y),
            SymbolicExpr::Sub { x, y, .. } => self.pair(6, x, y),
            SymbolicExpr::Mul { x, y, .. } => self.pair(7, x, y),
            SymbolicExpr::Neg { x, .. } => {
                self.expression(x);
                self.byte(8);
            }
        }

        self.written.insert(key, self.values);
        self.values += 1;
    }

    fn pair<A: Leaf>(&mut self, tag: u8, x: &SymbolicExpr<A>, y: &SymbolicExpr<A>) {
        self.expression(x);
        self.expression(y);
        self.byte(tag);
    }
}

/// What a leaf of one expression kind writes into the string.
trait Leaf {
    /// Write any subtree this leaf carries, then the leaf itself.
    fn write(&self, arena: &mut Arena);
}

impl<F: Field> Leaf for BaseLeaf<F> {
    fn write(&self, arena: &mut Arena) {
        match self {
            Self::Variable(variable) => {
                arena.byte(0);
                match variable.entry {
                    BaseEntry::Preprocessed { offset } => {
                        arena.byte(0);
                        arena.count(offset);
                    }
                    BaseEntry::Main { offset } => {
                        arena.byte(1);
                        arena.count(offset);
                    }
                    BaseEntry::Periodic => arena.byte(2),
                    BaseEntry::Public => arena.byte(3),
                }
                arena.count(variable.index);
            }
            Self::IsFirstRow => arena.byte(1),
            Self::IsLastRow => arena.byte(2),
            Self::IsTransition => arena.byte(3),
            Self::Constant(value) => {
                arena.byte(4);
                arena.scalar(*value);
            }
        }
    }
}

impl<F: Field, EF: ExtensionField<F>> Leaf for ExtLeaf<F, EF> {
    fn write(&self, arena: &mut Arena) {
        match self {
            Self::Base(expression) => {
                arena.expression(expression);
                arena.byte(10);
            }
            Self::ExtVariable(variable) => {
                arena.byte(11);
                match variable.entry {
                    ExtEntry::Permutation { offset } => {
                        arena.byte(0);
                        arena.count(offset);
                    }
                    ExtEntry::Challenge => arena.byte(1),
                    ExtEntry::PermutationValue => arena.byte(2),
                }
                arena.count(variable.index);
            }
            Self::ExtConstant(value) => {
                arena.byte(12);
                arena.scalar(*value);
            }
        }
    }
}
