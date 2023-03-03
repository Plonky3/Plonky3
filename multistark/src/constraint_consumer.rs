use hyperfield::field::{Field, FieldExtension};
use hyperfield::packed::PackedField;

pub struct ConstraintConsumer<F: Field, FE: FieldExtension<F>, P: PackedField<Scalar = F>> {
    /// Random value used to combine multiple constraints into one.
    alpha: FE,

    /// Running sum of constraints that have been emitted so far, scaled by powers of `alpha`.
    constraint_acc: FE,

    /// The evaluation of `X - g^(n-1)`.
    z_last: P,

    /// The evaluation of the Lagrange basis polynomial which is nonzero at the point associated
    /// with the first trace row, and zero at other points in the subgroup.
    lagrange_basis_first: P,

    /// The evaluation of the Lagrange basis polynomial which is nonzero at the point associated
    /// with the last trace row, and zero at other points in the subgroup.
    lagrange_basis_last: P,
}

impl<F: Field, FE: FieldExtension<F>, P: PackedField<Scalar = F>> ConstraintConsumer<F, FE, P> {
    pub fn new(alpha: FE, z_last: P, lagrange_basis_first: P, lagrange_basis_last: P) -> Self {
        Self {
            constraint_acc: FE::ZERO,
            alpha,
            z_last,
            lagrange_basis_first,
            lagrange_basis_last,
        }
    }

    pub fn accumulator(self) -> FE {
        self.constraint_acc
    }

    /// Add one constraint valid on all rows except the last.
    pub fn constraint_transition(&mut self, constraint: P) {
        self.constraint(constraint * self.z_last);
    }

    /// Add one constraint on all rows.
    pub fn constraint(&mut self, constraint: P) {
        // TODO: Could be more efficient if there's a packed version of FE. Use FE::Packing?
        for c in constraint.as_slice() {
            self.constraint_acc = (self.constraint_acc * self.alpha).add_base(*c);
        }
    }

    /// Add one constraint, but first multiply it by a filter such that it will only apply to the
    /// first row of the trace.
    pub fn constraint_first_row(&mut self, constraint: P) {
        self.constraint(constraint * self.lagrange_basis_first);
    }

    /// Add one constraint, but first multiply it by a filter such that it will only apply to the
    /// last row of the trace.
    pub fn constraint_last_row(&mut self, constraint: P) {
        self.constraint(constraint * self.lagrange_basis_last);
    }
}
