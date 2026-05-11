//! Expression graph construction and constant pooling.
//!
//! The [`ExpressionBuilder`] is the layer for building arithmetic circuits.
//!
//! It manages a directed acyclic graph (DAG) of expressions where
//! - nodes represent field operations,
//! - edges represent dependencies between expressions.

#![allow(unused_variables)]
#![allow(clippy::missing_const_for_fn)]

use alloc::string::String;
#[cfg(feature = "debugging")]
use alloc::vec;
use alloc::vec::Vec;
use core::hash::Hash;

use hashbrown::HashMap;
use p3_field::PrimeCharacteristicRing;

use crate::NpoTypeId;
#[cfg(feature = "debugging")]
use crate::alloc_entry::AllocationLog;
use crate::expr::{Expr, ExpressionGraph};
use crate::types::{ExprId, NonPrimitiveOpId};
#[cfg(feature = "debugging")]
use crate::{AllocationEntry, AllocationType};

/// Discriminant for binary operations used as CSE pool keys.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum BinOpKind {
    Add,
    Mul,
    Sub,
    Div,
}

/// CSE pool key for fused multiply-add: `a * b + c`.
///
/// `a` and `b` are sorted to share a single key for both `mul_add(a,b,c)` and `mul_add(b,a,c)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct MulAddKey {
    a: ExprId,
    b: ExprId,
    c: ExprId,
}

impl MulAddKey {
    fn new(a: ExprId, b: ExprId, c: ExprId) -> Self {
        if a.0 <= b.0 {
            Self { a, b, c }
        } else {
            Self { a: b, b: a, c }
        }
    }
}

/// CSE pool key for Horner accumulator step: `acc * alpha + p_at_z - p_at_x`.
///
/// None of the operands are commutative w.r.t. each other.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct HornerAccKey {
    acc: ExprId,
    alpha: ExprId,
    p_at_z: ExprId,
    p_at_x: ExprId,
}

/// Per-operation counters for profiling expression allocations.
///
/// This is only compiled when the `profiling` feature is enabled on the `p3-circuit` crate.
#[cfg(feature = "profiling")]
#[derive(Debug, Clone, Default)]
pub struct OpCounts {
    /// Number of public input expressions allocated.
    pub publics: u64,
    /// Number of constant expressions allocated.
    pub consts: u64,
    /// Number of addition expressions allocated.
    pub adds: u64,
    /// Number of subtraction expressions allocated.
    pub subs: u64,
    /// Number of multiplication expressions allocated.
    pub muls: u64,
    /// Number of division expressions allocated.
    pub divs: u64,
    /// Number of Horner accumulator expressions allocated.
    pub horner_accs: u64,
    /// Number of boolean check expressions allocated.
    pub bool_checks: u64,
    /// Number of fused multiply-add expressions allocated.
    pub mul_adds: u64,
    /// Number of non-primitive calls allocated, broken down by type.
    pub non_primitives: HashMap<NpoTypeId, u64>,
}

/// Internal profiling state tracking global and per-scope counts.
#[cfg(feature = "profiling")]
#[derive(Debug, Default)]
pub struct ProfilingState {
    pub(crate) global: OpCounts,
    pub(crate) per_scope: HashMap<String, OpCounts>,
    /// Dedicated scope stack for profiling so it can be enabled independently of debug assertions.
    pub(crate) scope_stack: Vec<String>,
}

#[cfg(feature = "profiling")]
impl ProfilingState {
    #[inline]
    fn bump_with(&mut self, mut f: impl FnMut(&mut OpCounts)) {
        // Global totals.
        f(&mut self.global);

        // Per-scope totals (if a scope is active).
        if let Some(scope) = self.scope_stack.last().cloned() {
            let entry = self.per_scope.entry(scope).or_default();
            f(entry);
        }
    }

    #[inline]
    fn bump_const(&mut self) {
        self.bump_with(|c| c.consts += 1);
    }

    #[inline]
    fn bump_public(&mut self) {
        self.bump_with(|c| c.publics += 1);
    }

    #[inline]
    fn bump_add(&mut self) {
        self.bump_with(|c| c.adds += 1);
    }

    #[inline]
    fn bump_sub(&mut self) {
        self.bump_with(|c| c.subs += 1);
    }

    #[inline]
    fn bump_mul(&mut self) {
        self.bump_with(|c| c.muls += 1);
    }

    #[inline]
    fn bump_div(&mut self) {
        self.bump_with(|c| c.divs += 1);
    }

    #[inline]
    fn bump_horner_acc(&mut self) {
        self.bump_with(|c| c.horner_accs += 1);
    }

    #[inline]
    fn bump_bool_check(&mut self) {
        self.bump_with(|c| c.bool_checks += 1);
    }

    #[inline]
    fn bump_mul_add(&mut self) {
        self.bump_with(|c| c.mul_adds += 1);
    }

    #[inline]
    fn bump_non_primitive(&mut self, op_type: NpoTypeId) {
        // Global totals.
        *self
            .global
            .non_primitives
            .entry(op_type.clone())
            .or_default() += 1;

        // Per-scope totals (if a scope is active).
        if let Some(scope) = self.scope_stack.last().cloned() {
            let entry = self.per_scope.entry(scope).or_default();
            *entry.non_primitives.entry(op_type).or_default() += 1;
        }
    }
}

/// Manages expression graph construction, constant pooling, and debug instrumentation.
///
/// The expression builder provides a high-level interface for constructing arithmetic
/// circuits as directed acyclic graphs (DAGs).
///
/// Each node in the graph represents a field operation or a special value
/// (constant, public input, witness hint).
#[derive(Debug)]
pub struct ExpressionBuilder<F> {
    /// The underlying expression graph storage.
    ///
    /// This graph holds all expression nodes in a flat vector, indexed by [`ExprId`].
    ///
    /// The graph is append-only: once an expression is added, it never moves or gets
    /// removed, ensuring stable handles.
    graph: ExpressionGraph<F>,

    /// Constant deduplication pool.
    ///
    /// Maps field values to their unique [`ExprId`] in the graph.
    ///
    /// When a constant is requested via [`define_const`](Self::define_const), this pool is checked first.
    ///
    /// If the value exists, the cached ID is returned immediately, avoiding duplicate nodes.
    const_pool: HashMap<F, ExprId>,

    /// Common sub-expression elimination pool for binary operations.
    ///
    /// Maps `(op_kind, lhs, rhs)` to the existing [`ExprId`] that computes
    /// the same result. For commutative operations (Add, Mul) the operands are
    /// sorted so that `add(a,b)` and `add(b,a)` share the same key.
    cse_pool: HashMap<(BinOpKind, ExprId, ExprId), ExprId>,

    /// CSE pool for fused multiply-add expressions (`a * b + c`).
    ///
    /// The `a` and `b` operands are normalized so commutative variants share an entry.
    mul_add_pool: HashMap<MulAddKey, ExprId>,

    /// CSE pool for Horner accumulator steps (`acc * alpha + p_at_z - p_at_x`).
    horner_acc_pool: HashMap<HornerAccKey, ExprId>,

    /// CSE pool for boolean-check expressions (`val * (val - 1) = 0`), keyed by the input.
    bool_check_pool: HashMap<ExprId, ExprId>,

    /// Pending equality constraints.
    ///
    /// Each entry `(a, b)` represents a constraint that expressions `a` and `b` must
    /// evaluate to the same value. These constraints are resolved during the lowering
    /// phase using Union-Find (DSU) to merge witness slots.
    ///
    /// Self-connections `(a, a)` are filtered out to avoid unnecessary work.
    pending_connects: Vec<(ExprId, ExprId)>,

    /// Complete allocation history for debugging.
    ///
    /// Tracks every expression added to the graph, including metadata:
    /// - Allocation type (Const, Add, Mul, etc.)
    /// - Human-readable label
    /// - Expression dependencies
    /// - Scope context
    ///
    /// **Only present in debug builds.**
    #[cfg(feature = "debugging")]
    allocation_log: AllocationLog,

    /// Hierarchical scope stack for organizing allocations.
    ///
    /// Users can push/pop named scopes to organize the allocation log into logical
    /// groups (e.g., "fibonacci_step", "mmcs_verify", etc.).
    ///
    /// The current scope is attached to each allocation.
    ///
    /// **Only present in debug builds.**
    #[cfg(feature = "debugging")]
    scope_stack: Vec<String>,

    /// Optional profiling state for counting allocations by type and scope.
    ///
    /// This is only compiled when the `profiling` feature is enabled to avoid
    /// any runtime overhead in normal builds.
    #[cfg(feature = "profiling")]
    profiling: ProfilingState,
}

impl<F> ExpressionBuilder<F>
where
    F: PrimeCharacteristicRing + Eq + Hash,
{
    #[inline]
    pub(crate) fn is_const_zero(&self, id: ExprId) -> bool {
        matches!(self.graph.get_expr(id), Expr::Const(val) if *val == F::ZERO)
    }

    #[inline]
    pub(crate) fn is_const_one(&self, id: ExprId) -> bool {
        matches!(self.graph.get_expr(id), Expr::Const(val) if *val == F::ONE)
    }

    #[inline]
    pub(crate) fn get_const_value(&self, id: ExprId) -> Option<F> {
        match self.graph.get_expr(id) {
            Expr::Const(val) => Some(val.dup()),
            _ => None,
        }
    }

    /// Creates a new expression builder with zero constant pre-allocated.
    ///
    /// The zero constant is always the first node in the graph, accessible via
    /// [`ExprId::ZERO`].
    ///
    /// # Postconditions
    ///
    /// After construction:
    /// - The graph contains exactly one node: `Expr::Const(F::ZERO)`
    /// - The constant pool contains one entry: `F::ZERO → ExprId::ZERO`
    /// - All other collections (pending_connects, hints_fillers) are empty
    pub fn new() -> Self {
        // Initialize an empty expression graph.
        let mut graph = ExpressionGraph::new();

        // Pre-allocate the zero constant as the first node.
        //
        // This ensures ExprId::ZERO (which is ExprId(0)) always refers to zero.
        let zero_val = F::ZERO;
        let zero_id = graph.add_expr(Expr::Const(zero_val.dup()));

        // Pre-populate the constant pool with zero.
        let const_pool = [(zero_val, zero_id)].into();

        Self {
            graph,
            const_pool,
            cse_pool: HashMap::new(),
            mul_add_pool: HashMap::new(),
            horner_acc_pool: HashMap::new(),
            bool_check_pool: HashMap::new(),
            pending_connects: Vec::new(),
            #[cfg(feature = "debugging")]
            allocation_log: AllocationLog::default(),
            #[cfg(feature = "debugging")]
            scope_stack: Vec::new(),
            #[cfg(feature = "profiling")]
            profiling: ProfilingState::default(),
        }
    }

    /// Adds a constant to the expression graph with automatic deduplication.
    ///
    /// If this constant value was previously added, returns the existing [`ExprId`]
    /// handle instead of creating a duplicate node. This ensures the graph contains
    /// at most one node per unique constant value.
    ///
    /// # Arguments
    ///
    /// - `val`: The constant field value to add
    /// - `label`: Human-readable label for debug logging
    ///
    /// # Returns
    ///
    /// An [`ExprId`] handle to the constant. If `val` was seen before, returns the
    /// cached ID. Otherwise, creates a new node and returns its ID.
    ///
    /// # Debug Behavior
    ///
    /// In debug builds, logs the allocation to the allocation log with:
    /// - Type: `AllocationType::Const`
    /// - Label: the provided label
    /// - Dependencies: empty (constants have no dependencies)
    /// - Scope: the current scope from the scope stack
    ///
    /// **Important**: Only new allocations are logged. Returning a cached constant
    /// does not create a new log entry.
    pub fn define_const(&mut self, val: F, label: &'static str) -> ExprId {
        // Check if this constant already exists in the pool.
        if let Some(&cached_id) = self.const_pool.get(&val) {
            // Found a cached entry. Return it immediately without allocating.
            return cached_id;
        }

        // This is a new constant. Add it to the expression graph.
        let expr_id = self.graph.add_expr(Expr::Const(val.dup()));

        // Insert into the constant pool for future lookups.
        self.const_pool.insert(val, expr_id);

        // Count constant allocations when profiling is enabled.
        #[cfg(feature = "profiling")]
        self.profiling.bump_const();

        // Log the allocation in debug builds only.
        //
        // When the `debugging` feature is disabled, this compiles to a no-op.
        #[cfg(feature = "debugging")]
        self.log_alloc(expr_id, label, || (AllocationType::Const, vec![]));
        #[cfg(not(feature = "debugging"))]
        self.log_alloc(expr_id, label, || ());

        expr_id
    }

    /// Adds a public input expression to the graph.
    ///
    /// Public inputs are values known to both the prover and verifier. They are
    /// identified by their position in the public input vector.
    ///
    /// **Important**: Unlike constants, public inputs are **not** deduplicated. Each
    /// call creates a new expression node, even if the position is the same. This is
    /// intentional: multiple references to the same public input are treated as
    /// independent expressions that happen to read from the same source.
    ///
    /// # Arguments
    ///
    /// - `pos`: Zero-based index into the public input vector
    /// - `label`: Human-readable label for debug logging
    ///
    /// # Returns
    ///
    /// A new [`ExprId`] handle to the public input expression.
    pub fn public(&mut self, pos: usize, label: &'static str) -> ExprId {
        // Create a new Public expression node in the graph.
        //
        // The `pos` field indicates which public input slot this expression reads from.
        let expr_id = self.graph.add_expr(Expr::Public(pos));

        // Count public input allocations when profiling is enabled.
        #[cfg(feature = "profiling")]
        self.profiling.bump_public();

        // Log the allocation in debug builds.
        //
        // Public inputs have no dependencies (they are leaf nodes in the expression DAG).
        #[cfg(feature = "debugging")]
        self.log_alloc(expr_id, label, || (AllocationType::Public, vec![]));
        #[cfg(not(feature = "debugging"))]
        self.log_alloc(expr_id, label, || ());

        expr_id
    }

    /// Allocates a private input expression at the given position.
    ///
    /// Private inputs are externally provided at runtime but do not create a Public table row.
    /// They are constrained by downstream use (e.g. MMCS Merkle path verification).
    pub fn private_input(&mut self, pos: usize, label: &'static str) -> ExprId {
        let expr_id = self.graph.add_expr(Expr::PrivateInput(pos));
        #[cfg(feature = "debugging")]
        self.log_alloc(expr_id, label, || (AllocationType::Public, vec![]));
        #[cfg(not(feature = "debugging"))]
        self.log_alloc(expr_id, label, || ());
        expr_id
    }

    /// Adds an addition expression to the graph.
    ///
    /// Represents the field addition operation: `result = lhs + rhs`.
    ///
    /// # Arguments
    ///
    /// - `lhs`: Left operand expression
    /// - `rhs`: Right operand expression
    /// - `label`: Human-readable label for debug logging
    ///
    /// # Returns
    ///
    /// An [`ExprId`] handle to the addition expression.
    pub fn add(&mut self, lhs: ExprId, rhs: ExprId, label: &'static str) -> ExprId {
        // x + 0 = x, 0 + x = x
        if self.is_const_zero(lhs) {
            return rhs;
        }
        if self.is_const_zero(rhs) {
            return lhs;
        }

        // Constant folding: const + const = const
        if let (Some(a), Some(b)) = (self.get_const_value(lhs), self.get_const_value(rhs)) {
            return self.define_const(a + b, label);
        }

        // CSE: commutative, so normalize operand order.
        let key = if lhs.0 <= rhs.0 {
            (BinOpKind::Add, lhs, rhs)
        } else {
            (BinOpKind::Add, rhs, lhs)
        };
        if let Some(&existing) = self.cse_pool.get(&key) {
            return existing;
        }

        #[cfg(feature = "profiling")]
        self.profiling.bump_add();

        let expr_id = self.add_bin_op(
            Expr::Add { lhs, rhs },
            label,
            #[cfg(feature = "debugging")]
            AllocationType::Add,
            lhs,
            rhs,
        );
        self.cse_pool.insert(key, expr_id);
        expr_id
    }

    /// Adds a subtraction expression to the graph.
    ///
    /// Represents the field subtraction operation: `result = lhs - rhs`.
    ///
    /// **Note**: During lowering, subtraction is encoded as an addition row in the ALU table:
    /// `lhs - rhs = result` becomes `result + rhs = lhs` with the `add` selector enabled.
    ///
    /// # Arguments
    ///
    /// - `lhs`: Left operand (minuend)
    /// - `rhs`: Right operand (subtrahend)
    /// - `label`: Human-readable label for debug logging
    ///
    /// # Returns
    ///
    /// An [`ExprId`] handle to the subtraction expression.
    pub fn sub(&mut self, lhs: ExprId, rhs: ExprId, label: &'static str) -> ExprId {
        // x - 0 = x
        if self.is_const_zero(rhs) {
            return lhs;
        }
        // x - x = 0
        if lhs == rhs {
            return ExprId::ZERO;
        }

        // Constant folding: const - const = const
        if let (Some(a), Some(b)) = (self.get_const_value(lhs), self.get_const_value(rhs)) {
            return self.define_const(a - b, label);
        }

        // CSE: non-commutative, operand order matters.
        let key = (BinOpKind::Sub, lhs, rhs);
        if let Some(&existing) = self.cse_pool.get(&key) {
            return existing;
        }

        #[cfg(feature = "profiling")]
        self.profiling.bump_sub();

        let expr_id = self.add_bin_op(
            Expr::Sub { lhs, rhs },
            label,
            #[cfg(feature = "debugging")]
            AllocationType::Sub,
            lhs,
            rhs,
        );
        self.cse_pool.insert(key, expr_id);
        expr_id
    }

    /// Adds a multiplication expression to the graph.
    ///
    /// Represents the field multiplication operation: `result = lhs * rhs`.
    ///
    /// # Arguments
    ///
    /// - `lhs`: Left operand
    /// - `rhs`: Right operand
    /// - `label`: Human-readable label for debug logging
    ///
    /// # Returns
    ///
    /// An [`ExprId`] handle to the multiplication expression.
    pub fn mul(&mut self, lhs: ExprId, rhs: ExprId, label: &'static str) -> ExprId {
        // x * 0 = 0, 0 * x = 0
        if self.is_const_zero(lhs) || self.is_const_zero(rhs) {
            return ExprId::ZERO;
        }

        // x * 1 = x, 1 * x = x
        if self.is_const_one(lhs) {
            return rhs;
        }
        if self.is_const_one(rhs) {
            return lhs;
        }

        // Constant folding: const * const = const
        if let (Some(a), Some(b)) = (self.get_const_value(lhs), self.get_const_value(rhs)) {
            return self.define_const(a * b, label);
        }

        // CSE: commutative, so normalize operand order.
        let key = if lhs.0 <= rhs.0 {
            (BinOpKind::Mul, lhs, rhs)
        } else {
            (BinOpKind::Mul, rhs, lhs)
        };
        if let Some(&existing) = self.cse_pool.get(&key) {
            return existing;
        }

        #[cfg(feature = "profiling")]
        self.profiling.bump_mul();

        let expr_id = self.add_bin_op(
            Expr::Mul { lhs, rhs },
            label,
            #[cfg(feature = "debugging")]
            AllocationType::Mul,
            lhs,
            rhs,
        );
        self.cse_pool.insert(key, expr_id);
        expr_id
    }

    /// Adds a division expression to the graph.
    ///
    /// Represents the field division operation: `result = lhs / rhs`.
    ///
    /// **Note**: During lowering, division is encoded as a multiplication row in the ALU table:
    /// `lhs / rhs = result` becomes `result * rhs = lhs` with the `mul` selector enabled.
    ///
    /// # Arguments
    ///
    /// - `lhs`: Left operand (dividend)
    /// - `rhs`: Right operand (divisor, must be non-zero)
    /// - `label`: Human-readable label for debug logging
    ///
    /// # Returns
    ///
    /// An [`ExprId`] handle to the division expression.
    pub fn div(&mut self, lhs: ExprId, rhs: ExprId, label: &'static str) -> ExprId {
        // x / 1 = x
        if self.is_const_one(rhs) {
            return lhs;
        }
        // 0 / x = 0
        if self.is_const_zero(lhs) {
            return ExprId::ZERO;
        }
        // x / x = 1
        if lhs == rhs {
            return self.define_const(F::ONE, label);
        }

        // CSE: non-commutative, operand order matters.
        let key = (BinOpKind::Div, lhs, rhs);
        if let Some(&existing) = self.cse_pool.get(&key) {
            return existing;
        }

        #[cfg(feature = "profiling")]
        self.profiling.bump_div();

        let expr_id = self.add_bin_op(
            Expr::Div { lhs, rhs },
            label,
            #[cfg(feature = "debugging")]
            AllocationType::Div,
            lhs,
            rhs,
        );
        self.cse_pool.insert(key, expr_id);
        expr_id
    }

    /// Adds a Horner accumulator step to the graph: result = acc * alpha + p_at_z - p_at_x.
    ///
    /// Lowers to a single HornerAcc ALU op with no intermediate witnesses.
    pub fn add_horner_acc(
        &mut self,
        acc: ExprId,
        alpha: ExprId,
        p_at_z: ExprId,
        p_at_x: ExprId,
        label: &'static str,
    ) -> ExprId {
        // Constant folding: all operands const collapses to a single Const node.
        if let (Some(va), Some(vb), Some(vz), Some(vx)) = (
            self.get_const_value(acc),
            self.get_const_value(alpha),
            self.get_const_value(p_at_z),
            self.get_const_value(p_at_x),
        ) {
            return self.define_const(va * vb + vz - vx, label);
        }

        let key = HornerAccKey {
            acc,
            alpha,
            p_at_z,
            p_at_x,
        };
        if let Some(&existing) = self.horner_acc_pool.get(&key) {
            return existing;
        }

        #[cfg(feature = "profiling")]
        self.profiling.bump_horner_acc();

        let expr_id = self.graph.add_expr(Expr::HornerAcc {
            acc,
            alpha,
            p_at_z,
            p_at_x,
        });

        #[cfg(feature = "debugging")]
        self.log_alloc(expr_id, label, || {
            (
                AllocationType::HornerAcc,
                vec![vec![acc], vec![alpha], vec![p_at_z], vec![p_at_x]],
            )
        });
        #[cfg(not(feature = "debugging"))]
        self.log_alloc(expr_id, label, || ());

        self.horner_acc_pool.insert(key, expr_id);
        expr_id
    }

    /// Adds a boolean check expression to the graph: asserts val ∈ {0, 1}.
    ///
    /// Lowers to a single BoolCheck ALU op with no intermediate witnesses.
    pub fn add_bool_check(&mut self, val: ExprId, label: &'static str) -> ExprId {
        // Constants in {0, 1} satisfy the predicate vacuously: skip the ALU row entirely
        // and return the input itself so downstream `connect(val, check)` is a no-op.
        if self.is_const_zero(val) || self.is_const_one(val) {
            return val;
        }

        if let Some(&existing) = self.bool_check_pool.get(&val) {
            return existing;
        }

        #[cfg(feature = "profiling")]
        self.profiling.bump_bool_check();

        let expr_id = self.graph.add_expr(Expr::BoolCheck { val });

        #[cfg(feature = "debugging")]
        self.log_alloc(expr_id, label, || {
            (AllocationType::BoolCheck, vec![vec![val]])
        });
        #[cfg(not(feature = "debugging"))]
        self.log_alloc(expr_id, label, || ());

        self.bool_check_pool.insert(val, expr_id);
        expr_id
    }

    /// Adds a fused multiply-add expression to the graph: result = a * b + c.
    ///
    /// Lowers to a single MulAdd ALU op with no intermediate witnesses.
    pub fn add_mul_add(&mut self, a: ExprId, b: ExprId, c: ExprId, label: &'static str) -> ExprId {
        // Constant folding: all operands const collapses to a single Const node.
        if let (Some(va), Some(vb), Some(vc)) = (
            self.get_const_value(a),
            self.get_const_value(b),
            self.get_const_value(c),
        ) {
            return self.define_const(va * vb + vc, label);
        }

        let key = MulAddKey::new(a, b, c);
        if let Some(&existing) = self.mul_add_pool.get(&key) {
            return existing;
        }

        #[cfg(feature = "profiling")]
        self.profiling.bump_mul_add();

        let expr_id = self.graph.add_expr(Expr::MulAdd { a, b, c });

        #[cfg(feature = "debugging")]
        self.log_alloc(expr_id, label, || {
            (AllocationType::MulAdd, vec![vec![a], vec![b], vec![c]])
        });
        #[cfg(not(feature = "debugging"))]
        self.log_alloc(expr_id, label, || ());

        self.mul_add_pool.insert(key, expr_id);
        expr_id
    }

    /// Adds a non-primitive output expression to the graph.
    ///
    /// This expression represents a value produced by a non-primitive operation.
    /// The `call` parameter is the `ExprId` of the `NonPrimitiveCall` node, making
    /// the dependency explicit in the DAG structure.
    pub fn add_non_primitive_output(
        &mut self,
        op_type: &NpoTypeId,
        call: ExprId,
        output_idx: u32,
        label: &'static str,
    ) -> ExprId {
        let expr_id = self
            .graph
            .add_expr(Expr::NonPrimitiveOutput { call, output_idx });

        #[cfg(feature = "debugging")]
        self.log_alloc(expr_id, label, || {
            (
                AllocationType::NonPrimitiveOp(op_type.clone()),
                vec![vec![call]],
            )
        });
        #[cfg(not(feature = "debugging"))]
        self.log_alloc(expr_id, label, || ());

        expr_id
    }

    /// Adds a non-primitive call anchor expression to the graph.
    ///
    /// This expression has no witness value, but provides an explicit point in the expression DAG
    /// for the lowerer to emit the non-primitive op in the correct execution order.
    ///
    /// The `inputs` parameter contains all input expressions (flattened), making dependencies
    /// explicit in the DAG structure. For stateful ops with chaining (e.g., `in_ctl=false`),
    /// `inputs` may be empty since chained values are not in the witness table.
    #[allow(unused)]
    pub fn add_non_primitive_call(
        &mut self,
        op_id: NonPrimitiveOpId,
        op_type: &NpoTypeId,
        inputs: Vec<ExprId>,
        label: &'static str,
    ) -> ExprId {
        #[cfg(feature = "debugging")]
        let dependencies: Vec<Vec<ExprId>> = inputs.iter().map(|&id| vec![id]).collect();

        let expr_id = self
            .graph
            .add_expr(Expr::NonPrimitiveCall { op_id, inputs });

        // Count non-primitive calls when profiling is enabled.
        #[cfg(feature = "profiling")]
        self.profiling.bump_non_primitive(op_type.clone());

        #[cfg(feature = "debugging")]
        self.log_alloc(expr_id, label, || {
            (
                AllocationType::NonPrimitiveOp(op_type.clone()),
                dependencies,
            )
        });
        #[cfg(not(feature = "debugging"))]
        self.log_alloc(expr_id, label, || ());

        expr_id
    }

    /// Internal helper for adding binary operations.
    ///
    /// # Arguments
    ///
    /// - `expr`: The binary expression variant to add (Add/Sub/Mul/Div)
    /// - `label`: Human-readable label for debug logging
    /// - `alloc_type`: Allocation type for debug logging (only exists in debug builds)
    /// - `lhs`: Left operand dependency
    /// - `rhs`: Right operand dependency
    ///
    /// # Returns
    ///
    /// An [`ExprId`] handle to the newly created expression.
    #[inline(always)]
    fn add_bin_op(
        &mut self,
        expr: Expr<F>,
        label: &'static str,
        #[cfg(feature = "debugging")] alloc_type: AllocationType,
        lhs: ExprId,
        rhs: ExprId,
    ) -> ExprId {
        // Add the expression to the graph.
        let expr_id = self.graph.add_expr(expr);

        // Log the allocation with dependencies.
        //
        // Binary operations have two dependencies: one for lhs, one for rhs.
        #[cfg(feature = "debugging")]
        self.log_alloc(expr_id, label, || (alloc_type, vec![vec![lhs], vec![rhs]]));
        #[cfg(not(feature = "debugging"))]
        self.log_alloc(expr_id, label, || ());

        expr_id
    }

    /// Enforces equality between two expressions.
    ///
    /// Adds a pending constraint that expressions `a` and `b` must evaluate to
    /// the same value. During the lowering phase, these constraints are resolved
    /// using Union-Find (DSU) to merge the witness slots for `a` and `b`.
    ///
    /// # Arguments
    ///
    /// - `a`: First expression
    /// - `b`: Second expression
    ///
    /// # Self-Connection Optimization
    ///
    /// If `a == b` (same expression), this is a no-op. No constraint is added
    /// because an expression is trivially equal to itself.
    ///
    /// # Constraint Resolution
    ///
    /// Constraints are not resolved immediately. They are stored in the
    /// `pending_connects` vector and processed during circuit compilation.
    pub fn connect(&mut self, a: ExprId, b: ExprId) {
        // Skip self-connections as they are trivially satisfied.
        if a != b {
            self.pending_connects.push((a, b));
        }
    }

    /// Returns an immutable reference to the underlying expression graph.
    ///
    /// The graph provides access to all expression nodes and their relationships.
    ///
    /// # Returns
    ///
    /// A reference to the [`ExpressionGraph`] containing all expressions.
    #[inline]
    pub const fn graph(&self) -> &ExpressionGraph<F> {
        &self.graph
    }

    /// Returns a slice of pending equality constraints.
    ///
    /// Each constraint `(a, b)` represents an assertion that expressions `a` and `b`
    /// must evaluate to the same value.
    ///
    /// # Returns
    ///
    /// A slice of `(ExprId, ExprId)` pairs representing pending connections.
    #[inline]
    pub fn pending_connects(&self) -> &[(ExprId, ExprId)] {
        &self.pending_connects
    }

    /// Centralized logging helper for debug builds.
    ///
    /// # Arguments
    ///
    /// - `id`: The expression ID being logged
    /// - `label`: Human-readable label
    /// - `info_fn`: Closure that produces allocation metadata **only when needed**
    #[cfg(feature = "debugging")]
    #[inline(always)]
    fn log_alloc<Info>(&mut self, id: ExprId, label: &'static str, info_fn: Info)
    where
        Info: FnOnce() -> (AllocationType, Vec<Vec<ExprId>>),
    {
        // Execute the closure to get allocation metadata.
        let (alloc_type, dependencies) = info_fn();

        // Capture the current scope from the stack.
        let scope = self.scope_stack.last().cloned();

        // Add an entry to the allocation log.
        self.allocation_log.push(AllocationEntry {
            expr_id: id,
            alloc_type,
            label,
            dependencies,
            scope,
        });
    }

    /// No-op logging helper for release builds.
    #[cfg(not(feature = "debugging"))]
    #[inline(always)]
    #[allow(clippy::needless_pass_by_ref_mut)]
    fn log_alloc<Info>(&mut self, _id: ExprId, _label: &'static str, _info_fn: Info)
    where
        Info: FnOnce(),
    {
        // Intentionally empty - compiles to nothing in release builds.
    }

    /// Logs a non-primitive operation allocation (debug builds only).
    ///
    /// Non-primitive operations (MMCS, FRI, Poseidon2, etc.) are not part of the
    /// standard expression types but need to be tracked in the allocation log.
    ///
    /// # Arguments
    ///
    /// - `op_id`: The non-primitive operation ID
    /// - `op_type`: The type of operation (e.g., `NpoTypeId::MmcsVerify`)
    /// - `input_deps`: Input expression dependencies for this operation
    /// - `output_deps`: Output expression dependencies for this operation
    /// - `label`: Human-readable label
    #[cfg(feature = "debugging")]
    pub fn log_non_primitive_op(
        &mut self,
        op_id: crate::types::NonPrimitiveOpId,
        op_type: crate::ops::NpoTypeId,
        input_deps: Vec<Vec<ExprId>>,
        output_deps: Vec<Vec<ExprId>>,
        label: &'static str,
    ) {
        // Capture the current scope.
        let scope = self.scope_stack.last().cloned();

        // Combine inputs and outputs for dependency tracking.
        // Use a separator to distinguish inputs from outputs in the log.
        let mut dependencies = input_deps;
        dependencies.extend(output_deps);

        // Add to allocation log.
        //
        // Non-primitive operations are stored with a special allocation type
        // that includes the operation variant.
        self.allocation_log.push(AllocationEntry {
            expr_id: ExprId(op_id.0),
            alloc_type: AllocationType::NonPrimitiveOp(op_type),
            label,
            dependencies,
            scope,
        });
    }

    /// Pushes a new scope onto the scope stack (debug builds only).
    ///
    /// Scopes provide hierarchical organization for the allocation log. Subsequent
    /// allocations will be tagged with this scope until it is popped.
    ///
    /// # Arguments
    ///
    /// - `scope`: Human-readable scope name
    pub fn push_scope(&mut self, scope: &str) {
        #[cfg(any(feature = "debugging", feature = "profiling"))]
        {
            let scope = String::from(scope);
            #[cfg(all(feature = "debugging", feature = "profiling"))]
            {
                // When both features are enabled, clone once and use for both
                let scope_clone = scope.clone();
                self.scope_stack.push(scope_clone);
                self.profiling.scope_stack.push(scope);
            }
            #[cfg(all(feature = "debugging", not(feature = "profiling")))]
            {
                self.scope_stack.push(scope);
            }
            #[cfg(all(feature = "profiling", not(feature = "debugging")))]
            {
                self.profiling.scope_stack.push(scope);
            }
        }
    }

    /// Pops the current scope from the scope stack (debug builds only).
    ///
    /// # Panics
    ///
    /// Panics if the scope stack is empty (mismatched push/pop).
    pub fn pop_scope(&mut self) {
        #[cfg(feature = "debugging")]
        self.scope_stack.pop();
        #[cfg(feature = "profiling")]
        self.profiling.scope_stack.pop();
    }

    /// Returns the current scope (debug builds only).
    ///
    /// Returns the name of the most recently pushed scope, or `None` if no scope is active.
    ///
    /// # Returns
    ///
    /// - `Some(&'static str)` - The name of the current scope
    /// - `None` - No active scope
    #[cfg(feature = "debugging")]
    pub fn current_scope(&self) -> Option<&str> {
        self.scope_stack.last().map(|s| s.as_str())
    }

    /// Returns a reference to the allocation log (debug builds only).
    ///
    /// Provides read-only access to all allocation entries recorded during circuit
    /// construction. Useful for testing and verifying allocation behavior.
    ///
    /// # Returns
    ///
    /// A slice of [`AllocationEntry`] containing all recorded allocations.
    #[cfg(feature = "debugging")]
    pub fn allocation_log(&self) -> &[AllocationEntry] {
        &self.allocation_log
    }

    /// Dumps the allocation log for specific `ExprId`s.
    ///
    /// If debug_assertions are not enabled, this is a no-op.
    pub fn dump_expr_ids(&self, expr_ids: &[ExprId]) {
        #[cfg(feature = "debugging")]
        self.allocation_log.dump_expr_ids(expr_ids);
    }

    /// Dumps the allocation log to stdout (debug builds only).
    ///
    /// Prints a formatted view of all allocations, including their types, labels,
    /// dependencies, and scopes. Useful for debugging circuit construction.
    ///
    /// # Output
    ///
    /// In debug builds, outputs a detailed allocation report. In release builds,
    /// this method does nothing.
    pub fn dump_allocation_log(&self) {
        #[cfg(feature = "debugging")]
        self.allocation_log.dump();
    }

    /// Lists all unique scopes in the allocation log.
    ///
    /// Returns a vector of scope names that appear in the allocation log,
    /// with duplicates removed.
    ///
    /// # Returns
    ///
    /// - **Debug builds**: Vector of unique scope names
    /// - **Release builds**: Empty vector (no scopes tracked)
    pub fn list_scopes(&self) -> Vec<String> {
        #[cfg(feature = "debugging")]
        {
            self.allocation_log.scopes()
        }
        #[cfg(not(feature = "debugging"))]
        {
            Vec::new()
        }
    }

    /// Returns the global and per-scope operation counts when profiling is enabled.
    #[cfg(feature = "profiling")]
    pub const fn profiling_counts(&self) -> (&OpCounts, &HashMap<String, OpCounts>) {
        (&self.profiling.global, &self.profiling.per_scope)
    }
}

impl<F> Default for ExpressionBuilder<F>
where
    F: PrimeCharacteristicRing + Eq + Hash,
{
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use p3_test_utils::baby_bear_params::BabyBear;

    use super::*;
    #[cfg(feature = "debugging")]
    use crate::alloc::string::ToString;

    #[test]
    fn test_new_builder_has_zero_constant() {
        // New builder should pre-allocate zero constant
        let builder = ExpressionBuilder::<BabyBear>::new();

        // Zero should be in the graph at ExprId::ZERO
        assert_eq!(builder.graph().nodes().len(), 1);
        match &builder.graph().nodes()[0] {
            Expr::Const(val) => assert_eq!(*val, BabyBear::ZERO),
            _ => panic!("Expected Const(0) at ExprId::ZERO"),
        }

        // Const pool should contain zero
        let mut expected_const_pool = HashMap::new();
        expected_const_pool.insert(BabyBear::ZERO, ExprId::ZERO);
        assert_eq!(builder.const_pool, expected_const_pool);

        // No pending connections
        assert!(builder.pending_connects.is_empty());
    }

    #[test]
    fn test_add_const_single() {
        // Adding a single constant should work
        let mut builder = ExpressionBuilder::<BabyBear>::new();

        let c1 = builder.define_const(BabyBear::ONE, "test_const");

        // Should have 2 nodes: zero + one
        assert_eq!(builder.graph().nodes().len(), 2);
        assert_eq!(c1, ExprId(1));

        // Verify the constant was added to graph
        match &builder.graph().nodes()[1] {
            Expr::Const(val) => assert_eq!(*val, BabyBear::ONE),
            _ => panic!("Expected Const(1)"),
        }

        // Const pool should have both zero and one
        assert_eq!(builder.const_pool.len(), 2);
    }

    #[test]
    fn test_add_const_deduplication() {
        // Adding the same constant twice should return same ExprId
        let mut builder = ExpressionBuilder::<BabyBear>::new();

        let c1 = builder.define_const(BabyBear::from_u64(42), "first");
        let c2 = builder.define_const(BabyBear::from_u64(42), "second");

        // Should return same ExprId
        assert_eq!(c1, c2);

        // Should only have 2 nodes (zero + 42), not 3
        assert_eq!(builder.graph().nodes().len(), 2);

        // Const pool should only have 2 entries
        assert_eq!(builder.const_pool.len(), 2);
    }

    #[test]
    fn test_add_const_multiple_different() {
        // Adding different constants should create distinct ExprIds
        let mut builder = ExpressionBuilder::<BabyBear>::new();

        let c1 = builder.define_const(BabyBear::from_u64(1), "one");
        let c2 = builder.define_const(BabyBear::from_u64(2), "two");
        let c3 = builder.define_const(BabyBear::from_u64(3), "three");

        // All should be different
        assert_ne!(c1, c2);
        assert_ne!(c2, c3);
        assert_ne!(c1, c3);

        // Should have 4 nodes: zero + 1 + 2 + 3
        assert_eq!(builder.graph().nodes().len(), 4);
        assert_eq!(builder.const_pool.len(), 4);
    }

    #[test]
    fn test_add_const_zero_deduplicates_with_prealloc() {
        // Adding zero constant should return pre-allocated ExprId::ZERO
        let mut builder = ExpressionBuilder::<BabyBear>::new();

        let zero = builder.define_const(BabyBear::ZERO, "explicit_zero");

        // Should return ExprId::ZERO (the pre-allocated one)
        assert_eq!(zero, ExprId::ZERO);

        // Should still have only 1 node
        assert_eq!(builder.graph().nodes().len(), 1);
        assert_eq!(builder.const_pool.len(), 1);
    }

    #[test]
    fn test_add_public_single() {
        // Adding a single public input
        let mut builder = ExpressionBuilder::<BabyBear>::new();

        let p0 = builder.public(0, "public_0");

        // Should have 2 nodes: zero + public
        assert_eq!(builder.graph().nodes().len(), 2);
        assert_eq!(p0, ExprId(1));

        // Verify it's a Public node
        match &builder.graph().nodes()[1] {
            Expr::Public(pos) => assert_eq!(*pos, 0),
            _ => panic!("Expected Public(0)"),
        }
    }

    #[test]
    fn test_add_public_multiple() {
        // Adding multiple public inputs with different positions
        let mut builder = ExpressionBuilder::<BabyBear>::new();

        let p0 = builder.public(0, "p0");
        let p1 = builder.public(1, "p1");
        let p2 = builder.public(2, "p2");

        // All should be different
        assert_ne!(p0, p1);
        assert_ne!(p1, p2);

        // Should have 4 nodes: zero + 3 publics
        assert_eq!(builder.graph().nodes().len(), 4);

        // Verify positions
        match &builder.graph().nodes()[1] {
            Expr::Public(pos) => assert_eq!(*pos, 0),
            _ => panic!("Expected Public(0)"),
        }
        match &builder.graph().nodes()[2] {
            Expr::Public(pos) => assert_eq!(*pos, 1),
            _ => panic!("Expected Public(1)"),
        }
        match &builder.graph().nodes()[3] {
            Expr::Public(pos) => assert_eq!(*pos, 2),
            _ => panic!("Expected Public(2)"),
        }
    }

    #[test]
    fn test_add_public_same_position_creates_different_nodes() {
        // Adding public inputs with same position creates different nodes
        let mut builder = ExpressionBuilder::<BabyBear>::new();

        let p0_a = builder.public(0, "first");
        let p0_b = builder.public(0, "second");

        // Should be different ExprIds (no deduplication for Public)
        assert_ne!(p0_a, p0_b);

        // Should have 3 nodes
        assert_eq!(builder.graph().nodes().len(), 3);
    }

    #[test]
    fn test_add_operation() {
        // Test Add operation with non-constant operands (constant folding applies to const+const)
        let mut builder = ExpressionBuilder::<BabyBear>::new();

        let a = builder.public(0, "a");
        let b = builder.public(1, "b");
        let _sum = builder.add(a, b, "sum");

        // Should have 4 nodes: zero + pub_a + pub_b + add
        assert_eq!(builder.graph().nodes().len(), 4);

        // Verify Add node
        match &builder.graph().nodes()[3] {
            Expr::Add { lhs, rhs } => {
                assert_eq!(*lhs, a);
                assert_eq!(*rhs, b);
            }
            _ => panic!("Expected Add operation"),
        }
    }

    #[test]
    fn test_add_const_folding() {
        // Test that const + const is folded at build time
        let mut builder = ExpressionBuilder::<BabyBear>::new();

        let a = builder.define_const(BabyBear::from_u64(2), "a");
        let b = builder.define_const(BabyBear::from_u64(3), "b");
        let sum = builder.add(a, b, "sum");

        // Should have 4 nodes: zero + 2 + 3 + 5 (folded constant, no Add node)
        assert_eq!(builder.graph().nodes().len(), 4);
        match &builder.graph().nodes()[sum.0 as usize] {
            Expr::Const(val) => assert_eq!(*val, BabyBear::from_u64(5)),
            _ => panic!("Expected constant folding to produce Const(5)"),
        }
    }

    #[test]
    fn test_sub_operation() {
        // Test Sub operation with non-constant operands
        let mut builder = ExpressionBuilder::<BabyBear>::new();

        let a = builder.public(0, "a");
        let b = builder.public(1, "b");
        let _diff = builder.sub(a, b, "diff");

        assert_eq!(builder.graph().nodes().len(), 4);

        match &builder.graph().nodes()[3] {
            Expr::Sub { lhs, rhs } => {
                assert_eq!(*lhs, a);
                assert_eq!(*rhs, b);
            }
            _ => panic!("Expected Sub operation"),
        }
    }

    #[test]
    fn test_sub_const_folding() {
        let mut builder = ExpressionBuilder::<BabyBear>::new();

        let a = builder.define_const(BabyBear::from_u64(5), "a");
        let b = builder.define_const(BabyBear::from_u64(3), "b");
        let diff = builder.sub(a, b, "diff");

        match &builder.graph().nodes()[diff.0 as usize] {
            Expr::Const(val) => assert_eq!(*val, BabyBear::from_u64(2)),
            _ => panic!("Expected constant folding to produce Const(2)"),
        }
    }

    #[test]
    fn test_sub_identity() {
        let mut builder = ExpressionBuilder::<BabyBear>::new();
        let a = builder.public(0, "a");
        let zero = builder.define_const(BabyBear::ZERO, "zero");

        // x - 0 = x
        assert_eq!(builder.sub(a, zero, "sub_zero"), a);
        // x - x = 0
        assert_eq!(builder.sub(a, a, "sub_self"), ExprId::ZERO);
    }

    #[test]
    fn test_mul_operation() {
        // Test Mul operation with non-constant operands
        let mut builder = ExpressionBuilder::<BabyBear>::new();

        let a = builder.public(0, "a");
        let b = builder.public(1, "b");
        let _prod = builder.mul(a, b, "prod");

        assert_eq!(builder.graph().nodes().len(), 4);

        match &builder.graph().nodes()[3] {
            Expr::Mul { lhs, rhs } => {
                assert_eq!(*lhs, a);
                assert_eq!(*rhs, b);
            }
            _ => panic!("Expected Mul operation"),
        }
    }

    #[test]
    fn test_mul_const_folding() {
        let mut builder = ExpressionBuilder::<BabyBear>::new();

        let a = builder.define_const(BabyBear::from_u64(7), "a");
        let b = builder.define_const(BabyBear::from_u64(6), "b");
        let prod = builder.mul(a, b, "prod");

        match &builder.graph().nodes()[prod.0 as usize] {
            Expr::Const(val) => assert_eq!(*val, BabyBear::from_u64(42)),
            _ => panic!("Expected constant folding to produce Const(42)"),
        }
    }

    #[test]
    fn test_div_operation() {
        // Test Div operation
        let mut builder = ExpressionBuilder::<BabyBear>::new();

        let a = builder.define_const(BabyBear::from_u64(10), "a");
        let b = builder.define_const(BabyBear::from_u64(2), "b");
        let _quot = builder.div(a, b, "quot");

        assert_eq!(builder.graph().nodes().len(), 4);

        match &builder.graph().nodes()[3] {
            Expr::Div { lhs, rhs } => {
                assert_eq!(*lhs, a);
                assert_eq!(*rhs, b);
            }
            _ => panic!("Expected Div operation"),
        }
    }

    #[test]
    fn test_div_identity() {
        let mut builder = ExpressionBuilder::<BabyBear>::new();
        let a = builder.public(0, "a");
        let one = builder.define_const(BabyBear::ONE, "one");
        let zero = ExprId::ZERO;

        // x / 1 = x
        assert_eq!(builder.div(a, one, "div_one"), a);
        // 0 / x = 0
        assert_eq!(builder.div(zero, a, "zero_div"), ExprId::ZERO);
        // x / x = 1
        let result = builder.div(a, a, "div_self");
        match builder.graph().get_expr(result) {
            Expr::Const(val) => assert_eq!(*val, BabyBear::ONE),
            _ => panic!("Expected div(x,x) to produce Const(1)"),
        }
    }

    #[test]
    fn test_nested_operations() {
        // Test nested operations with non-constant inputs: (a + b) * (c - d)
        let mut builder = ExpressionBuilder::<BabyBear>::new();

        let a = builder.public(0, "a");
        let b = builder.public(1, "b");
        let c = builder.public(2, "c");
        let d = builder.public(3, "d");

        let sum = builder.add(a, b, "sum");
        let diff = builder.sub(c, d, "diff");
        let _prod = builder.mul(sum, diff, "prod");

        // zero + 4 publics + 3 ops = 8 nodes
        assert_eq!(builder.graph().nodes().len(), 8);

        // Verify final operation references intermediate results
        match &builder.graph().nodes()[7] {
            Expr::Mul { lhs, rhs } => {
                assert_eq!(*lhs, sum);
                assert_eq!(*rhs, diff);
            }
            _ => panic!("Expected Mul operation"),
        }
    }

    #[test]
    fn test_nested_const_folding() {
        // Constant folding should chain: (1+2) * (3-4) = 3 * (-1) = -3
        let mut builder = ExpressionBuilder::<BabyBear>::new();

        let a = builder.define_const(BabyBear::from_u64(1), "a");
        let b = builder.define_const(BabyBear::from_u64(2), "b");
        let c = builder.define_const(BabyBear::from_u64(3), "c");
        let d = builder.define_const(BabyBear::from_u64(4), "d");

        // folded to Const(3) — same as c, deduplicated
        let sum = builder.add(a, b, "sum");
        // folded to Const(-1)
        let diff = builder.sub(c, d, "diff");
        // folded to Const(-3)
        let prod = builder.mul(sum, diff, "prod");

        // All intermediates are constant-folded
        match &builder.graph().nodes()[prod.0 as usize] {
            Expr::Const(val) => assert_eq!(*val, -BabyBear::from_u64(3)),
            _ => panic!("Expected full constant folding"),
        }
    }

    #[test]
    fn test_connect_different_expressions() {
        // Connecting different expressions should add to pending_connects
        let mut builder = ExpressionBuilder::<BabyBear>::new();

        let a = builder.define_const(BabyBear::ONE, "a");
        let b = builder.define_const(BabyBear::from_u64(2), "b");

        builder.connect(a, b);

        // Should have one pending connection
        assert_eq!(builder.pending_connects.len(), 1);
        assert_eq!(builder.pending_connects[0], (a, b));
    }

    #[test]
    fn test_connect_same_expression_no_op() {
        // Connecting an expression to itself should be a no-op
        let mut builder = ExpressionBuilder::<BabyBear>::new();

        let a = builder.define_const(BabyBear::ONE, "a");

        builder.connect(a, a);

        // Should not add to pending_connects
        assert!(builder.pending_connects.is_empty());
    }

    #[test]
    fn test_connect_multiple() {
        // Multiple connections should all be tracked
        let mut builder = ExpressionBuilder::<BabyBear>::new();

        let a = builder.define_const(BabyBear::from_u64(1), "a");
        let b = builder.define_const(BabyBear::from_u64(2), "b");
        let c = builder.define_const(BabyBear::from_u64(3), "c");

        builder.connect(a, b);
        builder.connect(b, c);

        assert_eq!(builder.pending_connects.len(), 2);
        assert_eq!(builder.pending_connects[0], (a, b));
        assert_eq!(builder.pending_connects[1], (b, c));
    }

    #[test]
    fn test_connect_with_operations() {
        // Can connect operation results
        let mut builder = ExpressionBuilder::<BabyBear>::new();

        let a = builder.define_const(BabyBear::ONE, "a");
        let b = builder.define_const(BabyBear::from_u64(2), "b");
        let sum = builder.add(a, b, "sum");
        let c = builder.public(0, "c");

        builder.connect(sum, c);

        assert_eq!(builder.pending_connects.len(), 1);
        assert_eq!(builder.pending_connects[0], (sum, c));
    }

    #[test]
    fn test_graph_accessor() {
        // graph() should return reference to underlying graph
        let mut builder = ExpressionBuilder::<BabyBear>::new();

        builder.define_const(BabyBear::ONE, "one");

        let graph = builder.graph();
        assert_eq!(graph.nodes().len(), 2); // zero + one
    }

    #[test]
    fn test_pending_connects_accessor() {
        // pending_connects() should return slice of connections
        let mut builder = ExpressionBuilder::<BabyBear>::new();

        let a = builder.define_const(BabyBear::ONE, "a");
        let b = builder.define_const(BabyBear::from_u64(2), "b");

        builder.connect(a, b);

        let connects = builder.pending_connects();
        assert_eq!(connects.len(), 1);
        assert_eq!(connects[0], (a, b));
    }

    #[test]
    #[cfg(feature = "debugging")]
    fn test_scope_stack() {
        // Test scope push/pop functionality
        let mut builder = ExpressionBuilder::<BabyBear>::new();

        // Initially no scope
        assert!(builder.current_scope().is_none());

        // Push a scope
        builder.push_scope("test_scope");
        assert_eq!(builder.current_scope(), Some("test_scope"));

        // Push nested scope
        builder.push_scope("nested_scope");
        assert_eq!(builder.current_scope(), Some("nested_scope"));

        // Pop scope
        builder.pop_scope();
        assert_eq!(builder.current_scope(), Some("test_scope"));

        // Pop last scope
        builder.pop_scope();
        assert!(builder.current_scope().is_none());
    }

    #[test]
    #[cfg(feature = "debugging")]
    fn test_allocation_log() {
        // Allocation log should track all allocations
        let mut builder = ExpressionBuilder::<BabyBear>::new();

        // Initial log has zero const
        assert_eq!(builder.allocation_log().len(), 0); // Zero is pre-allocated without logging

        // Add a const
        builder.define_const(BabyBear::ONE, "test_const");
        assert_eq!(builder.allocation_log().len(), 1);

        // Add a public
        builder.public(0, "test_public");
        assert_eq!(builder.allocation_log().len(), 2);

        // Add an operation
        let a = builder.define_const(BabyBear::from_u64(2), "a");
        let b = builder.define_const(BabyBear::from_u64(3), "b");
        builder.add(a, b, "sum");
        assert_eq!(builder.allocation_log().len(), 5); // +3 more (2 consts + 1 add)
    }

    #[test]
    #[cfg(feature = "debugging")]
    fn test_list_scopes() {
        // list_scopes should return unique scopes
        let mut builder = ExpressionBuilder::<BabyBear>::new();

        builder.push_scope("scope_a");
        builder.define_const(BabyBear::ONE, "in_a");

        builder.push_scope("scope_b");
        builder.define_const(BabyBear::from_u64(2), "in_b");

        builder.pop_scope();
        builder.define_const(BabyBear::from_u64(3), "in_a_again");

        let scopes = builder.list_scopes();
        assert!(scopes.contains(&("scope_a".to_string())));
        assert!(scopes.contains(&("scope_b".to_string())));
    }

    #[test]
    #[cfg(not(debug_assertions))]
    fn test_list_scopes_release() {
        // In release mode, list_scopes should return empty vec
        let builder = ExpressionBuilder::<BabyBear>::new();
        assert!(builder.list_scopes().is_empty());
    }

    #[test]
    fn test_cse_add_dedup() {
        let mut builder = ExpressionBuilder::<BabyBear>::new();
        let a = builder.public(0, "a");
        let b = builder.public(1, "b");

        let sum1 = builder.add(a, b, "sum1");
        let sum2 = builder.add(a, b, "sum2");

        assert_eq!(sum1, sum2, "identical add should return same ExprId");
        // zero + pub_a + pub_b + add = 4 nodes (no duplicate)
        assert_eq!(builder.graph().nodes().len(), 4);
    }

    #[test]
    fn test_cse_add_commutative() {
        let mut builder = ExpressionBuilder::<BabyBear>::new();
        let a = builder.public(0, "a");
        let b = builder.public(1, "b");

        let sum1 = builder.add(a, b, "a+b");
        let sum2 = builder.add(b, a, "b+a");

        assert_eq!(sum1, sum2, "add(a,b) and add(b,a) should share ExprId");
    }

    #[test]
    fn test_cse_mul_dedup() {
        let mut builder = ExpressionBuilder::<BabyBear>::new();
        let a = builder.public(0, "a");
        let b = builder.public(1, "b");

        let prod1 = builder.mul(a, b, "prod1");
        let prod2 = builder.mul(a, b, "prod2");

        assert_eq!(prod1, prod2, "identical mul should return same ExprId");
    }

    #[test]
    fn test_cse_mul_commutative() {
        let mut builder = ExpressionBuilder::<BabyBear>::new();
        let a = builder.public(0, "a");
        let b = builder.public(1, "b");

        let prod1 = builder.mul(a, b, "a*b");
        let prod2 = builder.mul(b, a, "b*a");

        assert_eq!(prod1, prod2, "mul(a,b) and mul(b,a) should share ExprId");
    }

    #[test]
    fn test_cse_sub_dedup() {
        let mut builder = ExpressionBuilder::<BabyBear>::new();
        let a = builder.public(0, "a");
        let b = builder.public(1, "b");

        let diff1 = builder.sub(a, b, "diff1");
        let diff2 = builder.sub(a, b, "diff2");

        assert_eq!(diff1, diff2, "identical sub should return same ExprId");
    }

    #[test]
    fn test_cse_sub_not_commutative() {
        let mut builder = ExpressionBuilder::<BabyBear>::new();
        let a = builder.public(0, "a");
        let b = builder.public(1, "b");

        let diff1 = builder.sub(a, b, "a-b");
        let diff2 = builder.sub(b, a, "b-a");

        assert_ne!(diff1, diff2, "sub(a,b) and sub(b,a) must be different");
    }

    #[test]
    fn test_cse_transitive_chain() {
        let mut builder = ExpressionBuilder::<BabyBear>::new();
        let a = builder.public(0, "a");
        let b = builder.public(1, "b");
        let c = builder.public(2, "c");

        let prod1 = builder.mul(a, b, "a*b");
        let sum1 = builder.add(prod1, c, "a*b+c");

        let prod2 = builder.mul(a, b, "a*b again");
        let sum2 = builder.add(prod2, c, "a*b+c again");

        assert_eq!(prod1, prod2, "mul should be CSE'd");
        assert_eq!(sum1, sum2, "transitive: add should also be CSE'd");
        // zero + 3 publics + 1 mul + 1 add = 6 nodes total
        assert_eq!(builder.graph().nodes().len(), 6);
    }

    #[test]
    fn test_cse_different_ops_same_operands() {
        let mut builder = ExpressionBuilder::<BabyBear>::new();
        let a = builder.public(0, "a");
        let b = builder.public(1, "b");

        let sum = builder.add(a, b, "a+b");
        let prod = builder.mul(a, b, "a*b");

        assert_ne!(sum, prod, "add and mul with same operands must differ");
    }

    #[test]
    fn test_cse_div_dedup() {
        let mut builder = ExpressionBuilder::<BabyBear>::new();
        let a = builder.public(0, "a");
        let b = builder.public(1, "b");

        let div1 = builder.div(a, b, "a/b");
        let div2 = builder.div(a, b, "a/b again");

        assert_eq!(div1, div2, "identical div should return same ExprId");
    }
}
