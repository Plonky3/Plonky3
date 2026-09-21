//! One byte string naming everything a constraint system asserts.
//!
//! Two systems that differ anywhere a prover can observe produce different strings.
//!
//! Nothing here is ever decoded, so the string only has to be injective.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use p3_air::symbolic::{BaseEntry, BaseLeaf, ExtEntry, ExtLeaf, SymbolicExpr, SymbolicExpression};
use p3_field::{ExtensionField, Field, RawDataSerializable};
use p3_lookup::{InteractionSymbolicBuilder, TraceWindow};

/// Write the whole constraint system of one table into a fresh byte string.
///
/// The two fields come first, so a system read over one pair never matches another.
///
/// Mutually exclusive interactions are left out, because this backend refuses them outright.
pub(super) fn encode<F, EF>(builder: &InteractionSymbolicBuilder<F, EF>) -> Vec<u8>
where
    F: Field,
    EF: ExtensionField<F>,
{
    let mut arena = Arena::default();
    arena.blob(&F::order().to_bytes_le());
    arena.count(EF::DIMENSION);
    arena.blob(&EF::algebra_id());

    let base = builder.base_constraints();
    arena.count(base.len());
    for constraint in &base {
        arena.root(constraint);
    }

    let extension = builder.extension_constraints();
    arena.count(extension.len());
    for constraint in &extension {
        arena.root(constraint);
    }

    let global = builder.global_interactions();
    arena.count(global.len());
    for interaction in global {
        arena.text(&interaction.bus_name);
        arena.u32(interaction.count_weight);
        arena.count(interaction.fields.len());
        for field in &interaction.fields {
            arena.root(field);
        }
        arena.root(&interaction.count);
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
                arena.root(field);
            }
            arena.u32(count.weight());
            arena.root(multiplicities.next().expect("one per tuple"));
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

/// A growing byte string, plus the position of every expression node already written.
///
/// A repeated subtree is written once and named by its position afterwards.
#[derive(Default)]
struct Arena {
    bytes: Vec<u8>,
    written: BTreeMap<usize, u64>,
    next: u64,
}

impl Arena {
    fn byte(&mut self, value: u8) {
        self.bytes.push(value);
    }

    fn u32(&mut self, value: u32) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn u64(&mut self, value: u64) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn count(&mut self, value: usize) {
        self.u64(value as u64);
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

    /// Write one expression, then name the position its top node landed at.
    fn root<A: Leaf>(&mut self, expression: &SymbolicExpr<A>) {
        let position = self.expression(expression);
        self.u64(position);
    }

    /// Write every node of one expression that is not written already.
    ///
    /// Children come first, so a node only ever names positions that already exist.
    fn expression<A: Leaf>(&mut self, expression: &SymbolicExpr<A>) -> u64 {
        let key = core::ptr::from_ref(expression) as usize;
        if let Some(&position) = self.written.get(&key) {
            return position;
        }

        match expression {
            SymbolicExpr::Leaf(leaf) => leaf.write(self),
            SymbolicExpr::Add { x, y, .. } => self.pair(1, x, y),
            SymbolicExpr::Sub { x, y, .. } => self.pair(2, x, y),
            SymbolicExpr::Mul { x, y, .. } => self.pair(3, x, y),
            SymbolicExpr::Neg { x, .. } => {
                let x = self.expression(x);
                self.byte(4);
                self.u64(x);
            }
        }

        let position = self.next;
        self.next += 1;
        self.written.insert(key, position);
        position
    }

    fn pair<A: Leaf>(&mut self, tag: u8, x: &SymbolicExpr<A>, y: &SymbolicExpr<A>) {
        let x = self.expression(x);
        let y = self.expression(y);
        self.byte(tag);
        self.u64(x);
        self.u64(y);
    }
}

/// What a leaf of one expression kind writes into the string.
trait Leaf {
    /// Write any subtree this leaf carries, then the leaf itself.
    fn write(&self, arena: &mut Arena);
}

impl<F: Field> Leaf for BaseLeaf<F> {
    fn write(&self, arena: &mut Arena) {
        arena.byte(0);
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
                let position = arena.expression(expression);
                arena.byte(0);
                arena.byte(0);
                arena.u64(position);
            }
            Self::ExtVariable(variable) => {
                arena.byte(0);
                arena.byte(1);
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
                arena.byte(0);
                arena.byte(2);
                arena.scalar(*value);
            }
        }
    }
}
