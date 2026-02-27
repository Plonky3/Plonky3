use core::marker::PhantomData;

use crate::SymbolicExpression;
use crate::symbolic::expression_ext::SymbolicExpressionExt;

/// Entry kinds for base-field trace columns and public inputs.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum BaseEntry {
    Preprocessed { offset: usize },
    Main { offset: usize },
    Public,
}

/// Entry kinds for extension-field columns (permutation trace and challenges).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ExtEntry {
    Permutation { offset: usize },
    Challenge,
}

/// A variable within the evaluation window for base-field columns.
#[derive(Copy, Clone, Debug)]
pub struct SymbolicVariable<F> {
    pub entry: BaseEntry,
    pub index: usize,
    pub(crate) _phantom: PhantomData<F>,
}

impl<F> SymbolicVariable<F> {
    pub const fn new(entry: BaseEntry, index: usize) -> Self {
        Self {
            entry,
            index,
            _phantom: PhantomData,
        }
    }

    pub const fn degree_multiple(&self) -> usize {
        match self.entry {
            BaseEntry::Preprocessed { .. } | BaseEntry::Main { .. } => 1,
            BaseEntry::Public => 0,
        }
    }
}

impl_var_ops!(SymbolicVariable<F> => SymbolicExpression);

/// A variable within the evaluation window for extension-field columns.
#[derive(Copy, Clone, Debug)]
pub struct SymbolicVariableExt<F, EF> {
    pub entry: ExtEntry,
    pub index: usize,
    pub(crate) _phantom: PhantomData<(F, EF)>,
}

impl<F, EF> SymbolicVariableExt<F, EF> {
    pub const fn new(entry: ExtEntry, index: usize) -> Self {
        Self {
            entry,
            index,
            _phantom: PhantomData,
        }
    }

    pub const fn degree_multiple(&self) -> usize {
        match self.entry {
            ExtEntry::Permutation { .. } => 1,
            ExtEntry::Challenge => 0,
        }
    }
}

impl_var_ops!(SymbolicVariableExt<F, EF> => SymbolicExpressionExt);
