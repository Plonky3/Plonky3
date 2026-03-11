use core::marker::PhantomData;

/// Entry kinds for base-field trace columns and public inputs.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum BaseEntry {
    Preprocessed { offset: usize },
    Main { offset: usize },
    Periodic,
    Public,
}

/// Entry kinds for extension-field columns (permutation trace, challenges, and permutation values).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ExtEntry {
    Permutation { offset: usize },
    Challenge,
    PermutationValue,
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
            ExtEntry::Challenge | ExtEntry::PermutationValue => 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_field::extension::BinomialExtensionField;

    use super::*;

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;

    #[test]
    fn symbolic_variable_new_main() {
        // A main trace variable preserves its offset and column index.
        let var = SymbolicVariable::<F>::new(BaseEntry::Main { offset: 1 }, 3);
        assert_eq!(var.entry, BaseEntry::Main { offset: 1 });
        assert_eq!(var.index, 3);
    }

    #[test]
    fn symbolic_variable_new_preprocessed() {
        // A preprocessed trace variable preserves its offset and column index.
        let var = SymbolicVariable::<F>::new(BaseEntry::Preprocessed { offset: 0 }, 5);
        assert_eq!(var.entry, BaseEntry::Preprocessed { offset: 0 });
        assert_eq!(var.index, 5);
    }

    #[test]
    fn symbolic_variable_new_public() {
        // A public input variable has no offset, only a column index.
        let var = SymbolicVariable::<F>::new(BaseEntry::Public, 2);
        assert_eq!(var.entry, BaseEntry::Public);
        assert_eq!(var.index, 2);
    }

    #[test]
    fn symbolic_variable_degree_multiple_main() {
        // Main trace columns contribute degree 1.
        let var = SymbolicVariable::<F>::new(BaseEntry::Main { offset: 0 }, 0);
        assert_eq!(var.degree_multiple(), 1);
    }

    #[test]
    fn symbolic_variable_degree_multiple_preprocessed() {
        // Preprocessed trace columns contribute degree 1.
        let var = SymbolicVariable::<F>::new(BaseEntry::Preprocessed { offset: 0 }, 0);
        assert_eq!(var.degree_multiple(), 1);
    }

    #[test]
    fn symbolic_variable_degree_multiple_public() {
        // Public inputs are constant so they contribute degree 0.
        let var = SymbolicVariable::<F>::new(BaseEntry::Public, 0);
        assert_eq!(var.degree_multiple(), 0);
    }

    #[test]
    fn symbolic_variable_ext_new_permutation() {
        // A permutation variable preserves its offset and column index.
        let var = SymbolicVariableExt::<F, EF>::new(ExtEntry::Permutation { offset: 1 }, 7);
        assert_eq!(var.entry, ExtEntry::Permutation { offset: 1 });
        assert_eq!(var.index, 7);
    }

    #[test]
    fn symbolic_variable_ext_new_challenge() {
        // A challenge variable has no offset, only an index.
        let var = SymbolicVariableExt::<F, EF>::new(ExtEntry::Challenge, 4);
        assert_eq!(var.entry, ExtEntry::Challenge);
        assert_eq!(var.index, 4);
    }

    #[test]
    fn symbolic_variable_ext_degree_multiple_permutation() {
        // Permutation columns contribute degree 1.
        let var = SymbolicVariableExt::<F, EF>::new(ExtEntry::Permutation { offset: 0 }, 0);
        assert_eq!(var.degree_multiple(), 1);
    }

    #[test]
    fn symbolic_variable_ext_degree_multiple_challenge() {
        // Challenges are random constants so they contribute degree 0.
        let var = SymbolicVariableExt::<F, EF>::new(ExtEntry::Challenge, 0);
        assert_eq!(var.degree_multiple(), 0);
    }
}
