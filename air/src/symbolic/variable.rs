use core::marker::PhantomData;

/// Entry kinds for base-field trace columns and public inputs.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum BaseEntry {
    Preprocessed { offset: usize },
    Main { offset: usize },
    Periodic,
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
            BaseEntry::Preprocessed { .. }
            | BaseEntry::Main { .. }
            // TODO: Periodic columns use degree 1 as an approximation. In Winterfell's model,
            // a periodic column with period `p` over trace length `n` contributes degree `n/p - 1`.
            // See: https://github.com/facebook/winterfell/blob/main/air/src/air/transition/degree.rs
            | BaseEntry::Periodic => 1,
            BaseEntry::Public => 0,
        }
    }
}

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
