use core::marker::PhantomData;

use serde::{Deserialize, Serialize};

/// Entry kinds for base-field trace columns and public inputs.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BaseEntry {
    Preprocessed { offset: usize },
    Main { offset: usize },
    Periodic,
    Public,
}

/// Entry kinds for extension-field columns (permutation trace, challenges, and permutation values).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ExtEntry {
    Permutation { offset: usize },
    Challenge,
    PermutationValue,
}

/// A variable within the evaluation window for base-field columns.
#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub struct SymbolicVariable<F> {
    pub entry: BaseEntry,
    pub index: usize,
    #[serde(skip)]
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
            // Periodic columns are bounded by degree 1 here; the exact, trace-size-aware
            // degree is computed by [`Self::poly_degree`].
            BaseEntry::Preprocessed { .. } | BaseEntry::Main { .. } | BaseEntry::Periodic => 1,
            BaseEntry::Public => 0,
        }
    }

    /// Returns the exact degree of the polynomial this variable resolves to over a
    /// trace of length `trace_len`, given the period of each periodic column
    /// (indexed by periodic column index).
    ///
    /// Unlike [`Self::degree_multiple`], which measures degree in multiples of the
    /// degree-`(trace_len - 1)` trace polynomials, this returns the absolute
    /// polynomial degree, so it accounts for the reduced degree of periodic columns.
    pub fn poly_degree(&self, trace_len: usize, periodic_periods: &[usize]) -> usize {
        match self.entry {
            BaseEntry::Preprocessed { .. } | BaseEntry::Main { .. } => trace_len.saturating_sub(1),
            BaseEntry::Periodic => {
                // A periodic column of period `p` (with `p | trace_len`) is the evaluation
                // of `f'(X) = f(X^(trace_len/p))` with `deg f < p`, so its degree is
                // `(p - 1) * (trace_len / p) = trace_len - trace_len / p`.
                let period = periodic_periods
                    .get(self.index)
                    .copied()
                    .unwrap_or(trace_len)
                    .min(trace_len);
                match period {
                    0 => trace_len.saturating_sub(1),
                    p => trace_len - trace_len / p,
                }
            }
            BaseEntry::Public => 0,
        }
    }
}

/// A variable within the evaluation window for extension-field columns.
#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub struct SymbolicVariableExt<F, EF> {
    pub entry: ExtEntry,
    pub index: usize,
    #[serde(skip)]
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

    /// Returns the exact polynomial degree of this extension variable over a trace
    /// of length `trace_len`. See [`SymbolicVariable::poly_degree`].
    pub const fn poly_degree(&self, trace_len: usize) -> usize {
        match self.entry {
            ExtEntry::Permutation { .. } => trace_len.saturating_sub(1),
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
    fn symbolic_variable_poly_degree_main_and_preprocessed() {
        // Main and preprocessed columns are degree-`(N - 1)` trace polynomials.
        let main = SymbolicVariable::<F>::new(BaseEntry::Main { offset: 1 }, 0);
        let prep = SymbolicVariable::<F>::new(BaseEntry::Preprocessed { offset: 0 }, 0);
        assert_eq!(main.poly_degree(8, &[]), 7);
        assert_eq!(prep.poly_degree(8, &[]), 7);
    }

    #[test]
    fn symbolic_variable_poly_degree_public_is_constant() {
        // Public inputs are constants, regardless of trace length.
        let var = SymbolicVariable::<F>::new(BaseEntry::Public, 0);
        assert_eq!(var.poly_degree(8, &[]), 0);
    }

    #[test]
    fn symbolic_variable_poly_degree_periodic() {
        // A periodic column of period `p` over a trace of length `N` has degree
        // `N - N / p`, strictly below the `N - 1` of a regular column for `p < N`.
        let var = SymbolicVariable::<F>::new(BaseEntry::Periodic, 0);
        // Period 2 over N = 8: degree 8 - 4 = 4.
        assert_eq!(var.poly_degree(8, &[2]), 4);
        // Period 4 over N = 8: degree 8 - 2 = 6.
        assert_eq!(var.poly_degree(8, &[4]), 6);
        // Period equal to N behaves like a regular column: degree N - 1.
        assert_eq!(var.poly_degree(8, &[8]), 7);
        // Period 1 is a true constant column: degree 0.
        assert_eq!(var.poly_degree(8, &[1]), 0);
    }

    #[test]
    fn symbolic_variable_poly_degree_periodic_missing_period_is_conservative() {
        // With no period recorded for the column, fall back to a full-degree column.
        let var = SymbolicVariable::<F>::new(BaseEntry::Periodic, 3);
        assert_eq!(var.poly_degree(8, &[]), 7);
    }

    #[test]
    fn symbolic_variable_ext_poly_degree() {
        // Permutation columns are degree-`(N - 1)` polynomials; challenges are constants.
        let perm = SymbolicVariableExt::<F, EF>::new(ExtEntry::Permutation { offset: 0 }, 0);
        let challenge = SymbolicVariableExt::<F, EF>::new(ExtEntry::Challenge, 0);
        assert_eq!(perm.poly_degree(8), 7);
        assert_eq!(challenge.poly_degree(8), 0);
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
