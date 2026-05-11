//! Iterative DAG traversal primitives for symbolic expression compilation.

/// Binary arithmetic operations on circuit expressions.
#[derive(Copy, Clone)]
pub(super) enum BinOp {
    /// Addition.
    Add,
    /// Subtraction.
    Sub,
    /// Multiplication.
    Mul,
}

/// Work items for the iterative DAG traversal stack.
///
/// Expression trees are flattened without recursion using an explicit
/// work stack paired with a value stack.
pub(super) enum Work<'a, E, K> {
    /// Decompose this expression node (handled by the caller's match).
    Eval(&'a E),
    /// Pop one operand and negate it (subtract from zero).
    BuildNeg(K),
    /// Pop two operands and combine them with the given binary operation.
    BuildBinary(K, BinOp),
}
