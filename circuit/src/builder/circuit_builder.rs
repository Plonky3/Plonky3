use alloc::boxed::Box;
use alloc::string::{String, ToString as _};
use alloc::sync::Arc;
use alloc::vec::Vec;
use alloc::{format, vec};
use core::hash::Hash;
use core::marker::PhantomData;

use hashbrown::HashMap;
use itertools::zip_eq;
use p3_field::{
    BasedVectorSpace, Dup, ExtensionField, Field, PrimeCharacteristicRing, PrimeField64,
};
use p3_symmetric::Permutation;

#[cfg(feature = "profiling")]
use super::OpCounts;
use super::compiler::{ExpressionLowerer, LoweringResult, Optimizer};
use super::npo::{NonPrimitiveOpParams, NonPrimitiveOperationData, NpoCircuitPlugin};
use super::{BuilderConfig, ExpressionBuilder, PublicInputTracker};
use crate::circuit::Circuit;
use crate::ops::poseidon2_perm::{
    Poseidon2CircuitPlugin, Poseidon2PermCallBase, Poseidon2PermExec,
};
use crate::ops::recompose::RecomposeCircuitPlugin;
use crate::ops::{
    HintExecutor, NpoConfig, NpoRegistry, NpoTypeId, Poseidon2Params, Poseidon2PermCall,
};
use crate::tables::TraceGeneratorFn;
use crate::types::{ExprId, NonPrimitiveOpId, WitnessAllocator, WitnessId};
use crate::{CircuitBuilderError, CircuitError};

/// Builder for constructing circuits.
pub struct CircuitBuilder<F: Field> {
    /// Expression graph builder
    expr_builder: ExpressionBuilder<F>,

    /// Public input tracker
    public_tracker: PublicInputTracker,
    /// Private input tracker
    private_input_tracker: PublicInputTracker,

    /// Witness index allocator
    witness_alloc: WitnessAllocator,

    /// Non-primitive operations (complex constraints that don't produce `ExprId`s)
    non_primitive_ops: Vec<NonPrimitiveOperationData<F>>,

    /// Builder configuration
    config: BuilderConfig,

    /// Registered non-primitive trace generators.
    non_primitive_trace_generators: HashMap<NpoTypeId, TraceGeneratorFn<F>>,

    /// Registered circuit-layer NPO plugins, keyed by type ID.
    npo_registry: NpoRegistry<F>,

    /// Tags for wires (ExprId) - enables probing values by name after execution.
    tag_to_expr: HashMap<String, ExprId>,

    /// Tags for non-primitive operations - enables setting private data by name.
    tag_to_op: HashMap<String, NonPrimitiveOpId>,

    /// Whether the recompose NPO table is enabled (bypasses ALU-based recompose).
    recompose_npo_enabled: bool,

    /// `recompose_base_coeffs_to_ext` outputs mapped back to their coefficient `ExprId`s.
    /// Lets `decompose_ext_to_base_coeffs` return those nodes without extra decomposition hints.
    ext_recompose_coeffs: HashMap<ExprId, Vec<ExprId>>,

    /// `select(b, t, s)` outputs mapped to their `(b, t, s)` triple.
    ///
    /// Lets `decompose_ext_to_base_coeffs` handle EF selects coefficient-wise when at least one
    /// input has known coefficient provenance (via `ext_recompose_coeffs`), avoiding D witness
    /// allocations for the known branch. The unknown branch is decomposed recursively.
    ext_select_sources: HashMap<ExprId, (ExprId, ExprId, ExprId)>,

    /// Routes decomposition reconnect through `recompose/coeff` when set (see
    /// [`Self::set_recompose_coeff_ctl_for_decompose_links`]).
    recompose_coeff_ctl_for_decompose_links: bool,
}

impl<F> Default for CircuitBuilder<F>
where
    F: Field + PrimeCharacteristicRing + Eq + Hash,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<F> CircuitBuilder<F>
where
    F: Field + PrimeCharacteristicRing + Eq + Hash,
{
    /// Creates a new circuit builder.
    pub fn new() -> Self {
        Self {
            expr_builder: ExpressionBuilder::new(),
            public_tracker: PublicInputTracker::new(),
            private_input_tracker: PublicInputTracker::new(),
            witness_alloc: WitnessAllocator::new(),
            non_primitive_ops: Vec::new(),
            config: BuilderConfig::new(),
            non_primitive_trace_generators: HashMap::new(),
            npo_registry: HashMap::new(),
            tag_to_expr: HashMap::new(),
            tag_to_op: HashMap::new(),
            recompose_npo_enabled: false,
            ext_recompose_coeffs: HashMap::new(),
            ext_select_sources: HashMap::new(),
            recompose_coeff_ctl_for_decompose_links: false,
        }
    }

    /// Toggle whether [`Self::decompose_ext_to_base_coeffs`] reconnects hinted coefficients via the
    /// `recompose/coeff` table (needed with D=1 Poseidon2 over a higher-degree extension field).
    pub const fn set_recompose_coeff_ctl_for_decompose_links(&mut self, enabled: bool) {
        self.recompose_coeff_ctl_for_decompose_links = enabled;
    }

    /// Register a circuit-layer NPO plugin.
    ///
    /// This:
    /// - stores the plugin in `npo_registry` for lowering
    /// - enables the op type in the builder config with the plugin's config
    /// - registers the plugin's trace generator
    pub fn register_npo(&mut self, plugin: impl NpoCircuitPlugin<F> + 'static) {
        let op_type: NpoTypeId = plugin.type_id();
        let plugin = Arc::new(plugin);

        // Enable op with plugin-provided config.
        let cfg = plugin.config();
        self.config.enable_op(op_type.clone(), cfg);

        // Register trace generator.
        self.non_primitive_trace_generators
            .insert(op_type.clone(), plugin.trace_generator());

        // Store plugin for lowering.
        self.npo_registry.insert(op_type, plugin);
    }

    /// Marks a non-primitive operation type as enabled.
    ///
    /// This only updates the configuration.
    ///
    /// The corresponding plugin must be registered separately for lowering to succeed.
    pub fn enable_op(&mut self, op: &NpoTypeId, cfg: NpoConfig) {
        self.config.enable_op(op.clone(), cfg);
    }

    /// Enables Poseidon2 permutation operations (one perm per table row).
    ///
    /// The current implementation only supports extension degree D=4 and WIDTH=16.
    ///
    /// # Arguments
    /// * `trace_generator` - Function to generate Poseidon2 trace from circuit and witness
    /// * `perm` - The Poseidon2 permutation to use for execution
    pub fn enable_poseidon2_perm<Config, P>(
        &mut self,
        trace_generator: TraceGeneratorFn<F>,
        perm: P,
    ) where
        Config: Poseidon2Params,
        F: Field + ExtensionField<Config::BaseField>,
        P: Permutation<[Config::BaseField; 16]> + Clone + Send + Sync + 'static,
    {
        let d = Config::D;
        let width_ext = Config::WIDTH_EXT;
        let width = Config::WIDTH;
        let exec: Poseidon2PermExec<F> = Arc::new(move |input: &[F]| {
            let mut base_input = vec![Config::BaseField::ZERO; width];
            for (i, ext_elem) in input.iter().enumerate() {
                let coeffs = ext_elem.as_basis_coefficients_slice();
                base_input[i * d..(i + 1) * d].copy_from_slice(coeffs);
            }
            let base_output = perm.permute(
                base_input
                    .try_into()
                    .expect("base_input length must equal WIDTH"),
            );
            let mut output = Vec::with_capacity(width_ext);
            for i in 0..width_ext {
                let coeffs = &base_output[i * d..(i + 1) * d];
                output.push(
                    F::from_basis_coefficients_slice(coeffs)
                        .expect("basis coefficients should be valid"),
                );
            }
            output
        });

        let plugin = Poseidon2CircuitPlugin::new(Config::CONFIG, exec, trace_generator);
        self.register_npo(plugin);
    }

    /// Enables Poseidon2 for configs with WIDTH=8 (e.g. Goldilocks).
    pub fn enable_poseidon2_perm_width_8<Config, P>(
        &mut self,
        trace_generator: TraceGeneratorFn<F>,
        perm: P,
    ) where
        Config: Poseidon2Params,
        F: Field + ExtensionField<Config::BaseField>,
        P: Permutation<[Config::BaseField; 8]> + Clone + Send + Sync + 'static,
    {
        assert!(
            Config::WIDTH == 8,
            "enable_poseidon2_perm_width_8 requires WIDTH=8"
        );
        let d = Config::D;
        let width_ext = Config::WIDTH_EXT;
        let exec: Poseidon2PermExec<F> = Arc::new(move |input: &[F]| {
            let mut base_input = vec![Config::BaseField::ZERO; 8];
            for (i, ext_elem) in input.iter().enumerate() {
                let coeffs = ext_elem.as_basis_coefficients_slice();
                base_input[i * d..(i + 1) * d].copy_from_slice(coeffs);
            }
            let base_output = perm.permute(
                base_input
                    .try_into()
                    .expect("base_input length must equal 8"),
            );
            let mut output = Vec::with_capacity(width_ext);
            for i in 0..width_ext {
                let coeffs = &base_output[i * d..(i + 1) * d];
                output.push(
                    F::from_basis_coefficients_slice(coeffs)
                        .expect("basis coefficients should be valid"),
                );
            }
            output
        });
        let plugin = Poseidon2CircuitPlugin::new(Config::CONFIG, exec, trace_generator);
        self.register_npo(plugin);
    }

    /// Enables the Poseidon2 permutation operation for base field challenges (D=1).
    ///
    /// This variant is for tests/circuits using base field as the challenge type.
    /// The permutation operates directly on 16 base field elements without packing.
    ///
    /// # Arguments
    /// * `trace_generator` - Function to generate Poseidon2 trace from circuit and witness
    /// * `perm` - The Poseidon2 permutation to use for execution
    pub fn enable_poseidon2_perm_base<Config, P>(
        &mut self,
        trace_generator: TraceGeneratorFn<F>,
        perm: P,
    ) where
        Config: Poseidon2Params,
        F: Field,
        P: Permutation<[F; 16]> + Clone + Send + Sync + 'static,
    {
        assert!(
            Config::D == 1,
            "enable_poseidon2_perm_base only supports extension degree D=1"
        );
        assert!(
            Config::WIDTH == 16,
            "enable_poseidon2_perm_base only supports WIDTH=16"
        );

        // For D=1, the exec closure converts slice to/from [F; 16]
        let exec: Poseidon2PermExec<F> = Arc::new(move |input: &[F]| {
            let arr: [F; 16] = input.try_into().expect("D=1 input must have 16 elements");
            perm.permute(arr).to_vec()
        });

        let plugin = Poseidon2CircuitPlugin::new(Config::CONFIG, exec, trace_generator);
        self.register_npo(plugin);
    }

    /// Enables the recompose NPO table, which replaces ALU-based `recompose_base_coeffs_to_ext`
    /// with a dedicated table that packs D base-field witnesses into one extension-field witness.
    pub fn enable_recompose<BF>(&mut self, trace_generator: TraceGeneratorFn<F>)
    where
        BF: PrimeField64,
        F: ExtensionField<BF>,
    {
        let d = <F as BasedVectorSpace<BF>>::DIMENSION;

        // Build a recompose closure that captures the BF→EF relationship via
        // `BasedVectorSpace<BF>`. Each input value is an EF element of the form
        // (c_i, 0, …, 0); we extract the BF scalar and reconstruct the full EF.
        #[allow(clippy::type_complexity)]
        let recompose_fn: Arc<dyn Fn(&[F]) -> F + Send + Sync> = Arc::new(|values: &[F]| {
            <F as BasedVectorSpace<BF>>::from_basis_coefficients_fn(|i| {
                <F as BasedVectorSpace<BF>>::as_basis_coefficients_slice(&values[i])[0]
            })
        });

        let plugin_std =
            RecomposeCircuitPlugin::new(d, trace_generator, Arc::clone(&recompose_fn), false);
        self.register_npo(plugin_std);
        let plugin_coeff = RecomposeCircuitPlugin::new(
            d,
            crate::ops::recompose::generate_recompose_coeff_trace::<BF, F>,
            Arc::clone(&recompose_fn),
            true,
        );
        self.register_npo(plugin_coeff);
        self.recompose_npo_enabled = true;
    }

    /// No-op recompose enablement: leaves `recompose_base_coeffs_to_ext` using
    /// the ALU fallback. Has the same signature as `enable_recompose` so it
    /// can be selected via a macro `$ident` parameter.
    pub fn noop_enable_recompose<BF>(&mut self, _trace_generator: TraceGeneratorFn<F>)
    where
        BF: PrimeField64,
        F: ExtensionField<BF>,
    {
    }

    /// Checks whether an op type is enabled on this builder.
    fn is_op_enabled(&self, op: &NpoTypeId) -> bool {
        self.config.is_op_enabled(op)
    }

    pub(crate) fn ensure_op_enabled(&self, op: &NpoTypeId) -> Result<(), CircuitBuilderError> {
        // Unconstrained operations are always enabled
        if !self.is_op_enabled(op) && *op != NpoTypeId::unconstrained() {
            return Err(CircuitBuilderError::OpNotAllowed { op: op.clone() });
        }
        Ok(())
    }

    /// Adds a public input to the circuit.
    ///
    /// Cost: 1 row in Public table + 1 row in witness table.
    pub fn public_input(&mut self) -> ExprId {
        self.alloc_public_input("")
    }

    /// Allocates a public input with a descriptive label.
    ///
    /// The label is logged in debug builds for easier debugging of public input ordering.
    ///
    /// Cost: 1 row in Public table + 1 row in witness table.
    pub fn alloc_public_input(&mut self, label: &'static str) -> ExprId {
        let pos = self.public_tracker.alloc();
        self.expr_builder.public(pos, label)
    }

    /// Allocates multiple public inputs with a descriptive label.
    pub fn alloc_public_inputs(&mut self, count: usize, label: &'static str) -> Vec<ExprId> {
        (0..count).map(|_| self.alloc_public_input(label)).collect()
    }

    /// Allocates a fixed-size array of public inputs with a descriptive label.
    pub fn alloc_public_input_array<const N: usize>(&mut self, label: &'static str) -> [ExprId; N] {
        core::array::from_fn(|_| self.alloc_public_input(label))
    }

    /// Returns the current public input count.
    pub const fn public_input_count(&self) -> usize {
        self.public_tracker.count()
    }

    /// Allocates a private input with a descriptive label.
    ///
    /// Private inputs are set at runtime via `set_private_inputs` but do not create a Public table row.
    pub fn alloc_private_input(&mut self, label: &'static str) -> ExprId {
        let pos = self.private_input_tracker.alloc();
        self.expr_builder.private_input(pos, label)
    }

    /// Allocates multiple private inputs with a descriptive label.
    pub fn alloc_private_inputs(&mut self, count: usize, label: &'static str) -> Vec<ExprId> {
        (0..count)
            .map(|_| self.alloc_private_input(label))
            .collect()
    }

    /// Allocates a fixed-size array of private inputs with a descriptive label.
    pub fn alloc_private_input_array<const N: usize>(
        &mut self,
        label: &'static str,
    ) -> [ExprId; N] {
        core::array::from_fn(|_| self.alloc_private_input(label))
    }

    /// Returns the current private input count.
    pub const fn private_input_count(&self) -> usize {
        self.private_input_tracker.count()
    }

    /// Adds a constant to the circuit (deduplicated).
    ///
    /// If this value was previously added, returns the original ExprId.
    /// Cost: 1 row in Const table + 1 row in witness table (only for new constants).
    pub fn define_const(&mut self, val: F) -> ExprId {
        self.alloc_const(val, "")
    }

    /// Allocates a constant with a descriptive label.
    ///
    /// Cost: 1 row in Const table + 1 row in witness table (only for new constants).
    pub fn alloc_const(&mut self, val: F, label: &'static str) -> ExprId {
        self.expr_builder.define_const(val, label)
    }

    /// Adds two expressions.
    ///
    /// Cost: 1 row in the ALU table (add selector) + 1 row in the witness table.
    pub fn add(&mut self, lhs: ExprId, rhs: ExprId) -> ExprId {
        self.alloc_add(lhs, rhs, "")
    }

    /// Adds two expressions with a descriptive label.
    ///
    /// Cost: 1 row in the ALU table (add selector) + 1 row in the witness table.
    pub fn alloc_add(&mut self, lhs: ExprId, rhs: ExprId, label: &'static str) -> ExprId {
        self.expr_builder.add(lhs, rhs, label)
    }

    /// Subtracts two expressions.
    ///
    /// Cost: 1 row in the ALU table (add selector) + 1 row in the witness table (encoded as result + rhs = lhs).
    pub fn sub(&mut self, lhs: ExprId, rhs: ExprId) -> ExprId {
        self.alloc_sub(lhs, rhs, "")
    }

    /// Subtracts two expressions with a descriptive label.
    ///
    /// Cost: 1 row in the ALU table (add selector) + 1 row in the witness table.
    pub fn alloc_sub(&mut self, lhs: ExprId, rhs: ExprId, label: &'static str) -> ExprId {
        self.expr_builder.sub(lhs, rhs, label)
    }

    /// Multiplies two expressions.
    ///
    /// Cost: 1 row in the ALU table (mul selector) + 1 row in the witness table.
    pub fn mul(&mut self, lhs: ExprId, rhs: ExprId) -> ExprId {
        self.alloc_mul(lhs, rhs, "")
    }

    /// Multiplies two expressions with a descriptive label.
    ///
    /// Cost: 1 row in the ALU table (mul selector) + 1 row in the witness table.
    pub fn alloc_mul(&mut self, lhs: ExprId, rhs: ExprId, label: &'static str) -> ExprId {
        self.expr_builder.mul(lhs, rhs, label)
    }

    /// Computes and returns `a * b + c`.
    ///
    /// This is a common fused operation in cryptographic circuits.
    ///
    /// # Arguments
    /// * `a`, `b`, `c`: The expressions to operate on.
    ///
    /// # Returns
    /// A new `ExprId` representing the result of `a * b + c`.
    ///
    /// # Cost
    /// 1 fused multiply-add constraint (single ALU row).
    pub fn mul_add(&mut self, a: ExprId, b: ExprId, c: ExprId) -> ExprId {
        self.expr_builder.add_mul_add(a, b, c, "mul_add")
    }

    /// Horner accumulator step: result = acc * alpha + p_at_z - p_at_x
    ///
    /// Emits a single HornerAcc ALU operation, avoiding intermediate witnesses
    /// that would arise from separate mul + add + sub.
    pub fn horner_acc_step(
        &mut self,
        acc: ExprId,
        alpha: ExprId,
        p_at_z: ExprId,
        p_at_x: ExprId,
    ) -> ExprId {
        self.expr_builder
            .add_horner_acc(acc, alpha, p_at_z, p_at_x, "")
    }

    /// Multiplies a slice of expressions together.
    ///
    /// # Arguments
    /// * `inputs`: A slice of `ExprId`s to multiply.
    ///
    /// # Returns
    /// A new `ExprId` representing the product of all inputs. Returns `1` if the slice is empty.
    ///
    /// # Cost
    /// `N-1` multiplication constraints, where `N` is the number of inputs.
    pub fn mul_many(&mut self, inputs: &[ExprId]) -> ExprId {
        // Handle edge cases for empty or single-element slices.
        if inputs.is_empty() {
            return self.define_const(F::ONE);
        }
        if inputs.len() == 1 {
            return inputs[0];
        }

        // Efficiently multiply all elements using a fold.
        inputs
            .iter()
            .skip(1)
            .fold(inputs[0], |acc, &x| self.mul(acc, x))
    }

    /// Computes the inner product (dot product) of two slices of expressions.
    ///
    /// Computes `∑ (a[i] * b[i])`.
    ///
    /// # Arguments
    /// * `a`: The first slice of `ExprId`s.
    /// * `b`: The second slice of `ExprId`s.
    ///
    /// # Panics
    /// Panics if the input slices `a` and `b` have different lengths.
    ///
    /// # Returns
    /// A new `ExprId` representing the inner product.
    ///
    /// # Cost
    /// `N` fused multiply-adds.
    pub fn inner_product(&mut self, a: &[ExprId], b: &[ExprId]) -> ExprId {
        let zero = self.define_const(F::ZERO);

        // Calculate the sum of element-wise products.
        zip_eq(a, b).fold(zero, |acc, (&x, &y)| self.mul_add(x, y, acc))
    }

    /// Divides two expressions.
    ///
    /// Cost: 1 row in the ALU table (mul selector) + 1 row in the witness table (encoded as rhs * out = lhs).
    pub fn div(&mut self, lhs: ExprId, rhs: ExprId) -> ExprId {
        self.alloc_div(lhs, rhs, "")
    }

    /// Divides two expressions with a descriptive label.
    ///
    /// Cost: 1 row in the ALU table (mul selector) + 1 row in the witness table.
    pub fn alloc_div(&mut self, lhs: ExprId, rhs: ExprId, label: &'static str) -> ExprId {
        self.expr_builder.div(lhs, rhs, label)
    }

    /// Asserts that an expression equals zero by connecting it to Const(0).
    ///
    /// Cost: Free in proving (implemented via connect).
    pub fn assert_zero(&mut self, expr: ExprId) {
        self.connect(expr, ExprId::ZERO);
    }

    /// Asserts that an expression is boolean: b ∈ {0,1}.
    ///
    /// Emits a single BoolCheck ALU op enforcing b · (b − 1) = 0.
    pub fn assert_bool(&mut self, b: ExprId) {
        let check = self.expr_builder.add_bool_check(b, "bool_check");
        self.connect(b, check);
    }

    /// Connects two expressions, enforcing a == b (by aliasing outputs).
    ///
    /// Cost: Free in proving (handled by IR optimization layer via witness slot aliasing).
    pub fn connect(&mut self, a: ExprId, b: ExprId) {
        if a == b {
            return;
        }
        // Never store coefficient provenance on `ExprId::ZERO`: it is shared circuit-wide
        // (e.g. `assert_zero`), so aliasing recompose coeffs to it would poison other uses.
        if a == ExprId::ZERO || b == ExprId::ZERO {
            self.ext_recompose_coeffs.remove(&a);
            self.ext_recompose_coeffs.remove(&b);
            self.ext_select_sources.remove(&a);
            self.ext_select_sources.remove(&b);
        } else {
            let ca = self.ext_recompose_coeffs.remove(&a);
            let cb = self.ext_recompose_coeffs.remove(&b);
            let merged = match (ca, cb) {
                (Some(va), Some(vb)) => {
                    debug_assert_eq!(
                        va, vb,
                        "connect: both sides carry recompose coeffs but they differ"
                    );
                    Some(va)
                }
                (Some(v), None) | (None, Some(v)) => Some(v),
                (None, None) => None,
            };
            if let Some(v) = merged {
                self.ext_recompose_coeffs.insert(a, v.clone());
                self.ext_recompose_coeffs.insert(b, v);
            }

            let sa = self.ext_select_sources.remove(&a);
            let sb = self.ext_select_sources.remove(&b);
            let merged_select = match (sa, sb) {
                (Some(va), Some(vb)) => {
                    debug_assert_eq!(
                        va, vb,
                        "connect: both sides carry ext_select_sources but they differ"
                    );
                    Some(va)
                }
                (Some(v), None) | (None, Some(v)) => Some(v),
                (None, None) => None,
            };
            if let Some(v) = merged_select {
                self.ext_select_sources.insert(a, v);
                self.ext_select_sources.insert(b, v);
            }
        }
        self.expr_builder.connect(a, b);
    }

    /// Selects between two values using selector `b`:
    /// result = s + b · (t − s).
    ///
    /// When `b` ∈ {0,1}, this returns `t` if b = 1, else `s` if b = 0.
    /// Call `assert_bool(b)` beforehand if you need booleanity enforced.
    /// Cost: 1 sub + 1 mul_add (0 if trivial).
    pub fn select(&mut self, b: ExprId, t: ExprId, s: ExprId) -> ExprId {
        // Trivial: both branches identical
        if t == s {
            return s;
        }
        // Selector is known constant
        if self.expr_builder.is_const_zero(b) {
            return s;
        }
        if self.expr_builder.is_const_one(b) {
            return t;
        }
        let t_minus_s = self.sub(t, s);
        let result = self.mul_add(b, t_minus_s, s);
        self.ext_select_sources.insert(result, (b, t, s));
        result
    }

    /// Exponentiates a base expression to a power of 2 (i.e. base^(2^power_log)), by squaring repeatedly.
    pub fn exp_power_of_2(&mut self, base: ExprId, power_log: usize) -> ExprId {
        let mut res = base;
        for _ in 0..power_log {
            let square = self.mul(res, res);
            res = square;
        }
        res
    }

    /// Pushes a non-primitive op and creates optional output nodes tied to the call.
    ///
    /// `output_labels` must have length equal to the op's output arity; each `Some(label)`
    /// creates an `Expr::NonPrimitiveOutput { call, output_idx }` node for that output index.
    /// The returned `Vec<Option<ExprId>>` is aligned with `output_labels`.
    pub fn push_non_primitive_op_with_outputs(
        &mut self,
        op_type: NpoTypeId,
        input_exprs: Vec<Vec<ExprId>>,
        output_labels: Vec<Option<&'static str>>,
        params: Option<NonPrimitiveOpParams<F>>,
        label: &'static str,
    ) -> (NonPrimitiveOpId, ExprId, Vec<Option<ExprId>>) {
        let op_id = NonPrimitiveOpId(self.non_primitive_ops.len() as u32);

        let flattened_inputs: Vec<ExprId> = input_exprs.iter().flatten().copied().collect();
        let call_expr_id =
            self.expr_builder
                .add_non_primitive_call(op_id, &op_type, flattened_inputs, label);

        let mut output_exprs: Vec<Vec<ExprId>> = vec![Vec::new(); output_labels.len()];
        let mut outputs: Vec<Option<ExprId>> = vec![None; output_labels.len()];
        for (i, maybe_label) in output_labels.into_iter().enumerate() {
            if let Some(out_label) = maybe_label {
                let out_expr_id = self.expr_builder.add_non_primitive_output(
                    &op_type,
                    call_expr_id,
                    i as u32,
                    out_label,
                );
                output_exprs[i] = vec![out_expr_id];
                outputs[i] = Some(out_expr_id);
            }
        }

        self.non_primitive_ops.push(NonPrimitiveOperationData {
            op_id,
            op_type,
            input_exprs,
            output_exprs,
            params,
        });

        (op_id, call_expr_id, outputs)
    }

    /// Pushes an unconstrained non-primitive op into the circuit and returns its output expressions.
    ///
    /// Each returned `ExprId` is an `Expr::NonPrimitiveOutput { call, output_idx }` node.
    /// The `call` ID points to the newly created `NonPrimitiveOpWithExecutor` entry, so the
    /// dependency is explicit in the computation DAG.
    ///
    /// This is used for creating new unconstrained wires assigned to a non-deterministic values
    /// computed by `hint`.
    pub(crate) fn push_unconstrained_op<H: HintExecutor<F> + 'static>(
        &mut self,
        input_exprs: Vec<Vec<ExprId>>,
        n_outputs: usize,
        hint: H,
        label: &'static str,
    ) -> (NonPrimitiveOpId, ExprId, Vec<Option<ExprId>>) {
        self.push_non_primitive_op_with_outputs(
            NpoTypeId::unconstrained(),
            input_exprs,
            (0..n_outputs).map(|_| Some(label)).collect(),
            Some(NonPrimitiveOpParams::Unconstrained {
                executor: Box::new(hint),
            }),
            label,
        )
    }

    /// Pushes a new scope onto the scope stack.
    ///
    /// All subsequent allocations will be tagged with this scope until
    /// `pop_scope` is called. Scopes can be nested.
    ///
    /// If the `debugging` feature is not enabled, this is a no-op.
    #[allow(warnings)]
    pub fn push_scope(&mut self, scope: &str) {
        #[cfg(feature = "debugging")]
        self.expr_builder.push_scope(scope);
    }

    /// Pops the current scope from the scope stack.
    ///
    /// If the `debugging` feature is not enabled, this is a no-op.
    #[allow(clippy::missing_const_for_fn)]
    pub fn pop_scope(&mut self) {
        #[cfg(feature = "debugging")]
        self.expr_builder.pop_scope();
    }

    /// Dumps the allocation log for specific `ExprId`s.
    ///
    /// If the `debugging` feature is not enabled, this is a no-op.
    #[allow(clippy::missing_const_for_fn)]
    pub fn dump_expr_ids(&self, expr_ids: &[ExprId]) {
        self.expr_builder.dump_expr_ids(expr_ids);
    }

    /// Dumps the allocation log.
    ///
    /// If debug_assertions are not enabled, this is a no-op.
    #[allow(clippy::missing_const_for_fn)]
    pub fn dump_allocation_log(&self) {
        self.expr_builder.dump_allocation_log();
    }

    /// Lists all unique scopes in the allocation log.
    ///
    /// Returns an empty vector if the `debugging` feature is not enabled.
    #[allow(clippy::missing_const_for_fn)]
    pub fn list_scopes(&self) -> Vec<String> {
        self.expr_builder.list_scopes()
    }

    /// Returns global operation counts collected during circuit construction when profiling is enabled.
    ///
    /// When the `profiling` feature is disabled, this method is not compiled.
    #[cfg(feature = "profiling")]
    pub const fn global_op_counts(&self) -> &OpCounts {
        let (global, _) = self.expr_builder.profiling_counts();
        global
    }

    /// Returns per-scope operation counts collected during circuit construction when profiling is enabled.
    ///
    /// The returned map is keyed by the scope names passed to `push_scope`.
    /// When the `profiling` feature is disabled, this method is not compiled.
    #[cfg(feature = "profiling")]
    pub const fn scope_op_counts(&self) -> &HashMap<String, OpCounts> {
        let (_, per_scope) = self.expr_builder.profiling_counts();
        per_scope
    }

    /// Convenience method logging global, per-scope, and per-non-primitive-id profiling information.
    ///
    /// When the `profiling` feature is disabled, this is a no-op.
    #[allow(clippy::missing_const_for_fn)]
    pub fn profile(&self) {
        #[cfg(feature = "profiling")]
        {
            let (global, per_scope) = self.expr_builder.profiling_counts();

            tracing::info!("[PROFILING] global: {:?}", global);
            for (scope, counts) in per_scope.iter() {
                tracing::info!("[PROFILING] scope: {:?}, counts: {:?}", scope, counts);
            }
        }
    }

    /// Tags an expression for value lookup via `Traces::probe()` later on during
    /// circuit execution.
    ///
    /// Tags must be unique within a circuit. Duplicate tags will return an error.
    ///
    /// Note that this is different from allocation labels for `ExprId`s, which are
    /// used purely for debugging purposes.
    ///
    /// # Example
    /// ```ignore
    /// let result = builder.add(a, b);
    /// builder.tag(result, "my-sum")?;
    /// // After execution:
    /// let value = traces.probe("my-sum").unwrap();
    /// ```
    pub fn tag(&mut self, expr: ExprId, tag: impl Into<String>) -> Result<(), CircuitBuilderError> {
        let tag = tag.into();
        if self.tag_to_expr.contains_key(&tag) || self.tag_to_op.contains_key(&tag) {
            return Err(CircuitBuilderError::DuplicateTag { tag });
        }
        self.tag_to_expr.insert(tag, expr);
        Ok(())
    }

    /// Tags a non-primitive operation for private data setting via tag later on during
    /// circuit execution.
    ///
    /// Tags must be unique within a circuit. Duplicate tags will return an error.
    ///
    /// Note that this is different from allocation labels for `ExprId`s, which are
    /// used purely for debugging purposes.
    ///
    /// # Example
    /// ```ignore
    /// let (op_id, outputs) = builder.add_poseidon2_perm(...)?;
    /// builder.tag_op(op_id, format!("fri-query-{}-depth-{}", i, j))?;
    /// // Before execution:
    /// runner.set_private_data_by_tag("fri-query-0-depth-1", data)?;
    /// ```
    pub fn tag_op(
        &mut self,
        op_id: NonPrimitiveOpId,
        tag: impl Into<String>,
    ) -> Result<(), CircuitBuilderError> {
        let tag = tag.into();
        if self.tag_to_expr.contains_key(&tag) || self.tag_to_op.contains_key(&tag) {
            return Err(CircuitBuilderError::DuplicateTag { tag });
        }
        self.tag_to_op.insert(tag, op_id);
        Ok(())
    }
}

impl<F> CircuitBuilder<F>
where
    F: Field + Clone + PartialEq + Eq + Hash,
{
    /// Builds the circuit into a Circuit with separate lowering and IR transformation stages.
    /// Returns an error if lowering fails due to an internal inconsistency.
    pub fn build(self) -> Result<Circuit<F>, CircuitBuilderError> {
        self.profile();

        let (circuit, _) = self.build_with_public_mapping()?;
        Ok(circuit)
    }

    /// Builds the circuit and returns both the circuit and the ExprId→WitnessId mapping for public inputs.
    #[allow(clippy::type_complexity)]
    pub fn build_with_public_mapping(
        self,
    ) -> Result<(Circuit<F>, HashMap<ExprId, WitnessId>), CircuitBuilderError> {
        // Stage 1: Lower expressions and non-primitives into a single op list
        for data in &self.non_primitive_ops {
            self.ensure_op_enabled(&data.op_type)?;
        }
        let lowerer = ExpressionLowerer::new(
            self.expr_builder.graph(),
            &self.non_primitive_ops,
            self.expr_builder.pending_connects(),
            self.public_tracker.count(),
            self.private_input_tracker.count(),
            self.witness_alloc,
            &self.npo_registry,
        );
        // Run the multi-phase lowering pipeline and destructure the result.
        let LoweringResult {
            ops,
            public_rows,
            private_input_rows,
            expr_to_widx,
            public_mappings,
            witness_count,
        } = lowerer.lower()?;

        // Stage 2: IR transformations and optimizations
        let (ops, rewrite) = Optimizer::optimize(ops);

        let resolve = |id: WitnessId| id.resolve(&rewrite);
        let expr_to_widx = expr_to_widx
            .into_iter()
            .map(|(e, w)| (e, resolve(w)))
            .collect();
        let public_rows = public_rows.into_iter().map(resolve).collect();
        let private_input_rows = private_input_rows.into_iter().map(resolve).collect();

        // Stage 3: Generate final circuit
        let mut circuit = Circuit::new(witness_count, expr_to_widx);
        circuit.ops = ops;
        circuit.public_rows = public_rows;
        circuit.private_input_rows = private_input_rows;
        circuit.private_flat_len = self.private_input_tracker.count();
        if !rewrite.is_empty() {
            circuit.witness_rewrite = Some(rewrite);
        }
        circuit.public_flat_len = self.public_tracker.count();
        circuit.enabled_ops = self.config.into_enabled_ops();
        circuit.non_primitive_trace_generators = self.non_primitive_trace_generators;
        let mut gen_order: Vec<_> = circuit
            .non_primitive_trace_generators
            .keys()
            .cloned()
            .collect();
        gen_order.sort();
        circuit.non_primitive_trace_generator_order = gen_order;

        // Transfer wire tags, converting ExprId to WitnessId
        for (tag, expr_id) in self.tag_to_expr {
            if let Some(&witness_id) = circuit.expr_to_widx.get(&expr_id) {
                circuit.tag_to_witness.insert(tag, witness_id);
            } else {
                return Err(CircuitBuilderError::MissingExprMapping {
                    expr_id,
                    context: tag,
                });
            }
        }

        // Transfer operation tags directly
        circuit.tag_to_op_id = self.tag_to_op;

        Ok((circuit, public_mappings))
    }

    /// Decomposes a field element into its little-endian binary representation.
    ///
    /// Given a target `x`, creates `n_bits` boolean witness targets representing
    /// the binary decomposition, and constrains them to reconstruct `x`.
    ///
    /// # Parameters
    /// - `x`: The field element to decompose.
    /// - `n_bits`: Number of bits in the decomposition (must be ≤ 64).
    ///
    /// # Returns
    /// A vector of `n_bits` boolean [`ExprId`]s
    /// ```text
    ///     [b_0, b_1, ..., b_{n-1}]
    /// ```
    /// such that:
    /// ```text
    ///     x = b_0·2^0 + b_1·2^1 + b_2·2^2 + ... + b_{n-1}·2^{n-1}.
    /// ```
    ///
    /// # Errors
    /// Returns [`CircuitError::BinaryDecompositionTooManyBits`] if `n_bits > 64`.
    ///
    /// # Cost
    /// `n_bits` witness hints + `n_bits` boolean constraints + reconstruction constraints.
    pub fn decompose_to_bits<BF>(
        &mut self,
        x: ExprId,
        n_bits: usize,
    ) -> Result<Vec<ExprId>, CircuitBuilderError>
    where
        F: ExtensionField<BF>,
        BF: PrimeField64,
    {
        self.push_scope("decompose_to_bits");

        // We cannot request more bits than the extension field can represent.
        if n_bits > F::bits() {
            return Err(CircuitBuilderError::BinaryDecompositionTooManyBits {
                expected: BF::bits(),
                n_bits,
            });
        }

        // Create bit witness variables
        let binary_decomposition_hint = BinaryDecompositionHint::new();
        let bits: Vec<ExprId> = self
            .push_unconstrained_op(
                vec![vec![x]],
                n_bits,
                binary_decomposition_hint,
                "decompose_to_bits",
            )
            .2
            .into_iter()
            .collect::<Option<Vec<_>>>()
            .ok_or(CircuitBuilderError::MissingOutput)?;

        // Constrain that the bits reconstruct to the original value.
        let reconstructed = self.reconstruct_index_from_bits(&bits)?;
        self.connect(x, reconstructed);

        self.pop_scope();
        Ok(bits)
    }

    /// Packs little-endian bits into an extension-field element, limb by limb.
    ///
    /// The input bits `[b_0, ..., b_{n-1}]` are in little-endian order. Let
    /// `W = BF::bits()`. Bits are processed in chunks of `W` bits. For chunk index `i`,
    /// the code computes:
    ///
    /// `limb_i = Σ b · 2^k`
    ///
    /// where the sum ranges over bits `b` in the chunk and `k` is the bit position
    /// within the chunk.
    ///
    /// Each `limb_i` is embedded into `F` using the canonical basis element `E_i`.
    /// The final value is `Σ limb_i · E_i`.
    ///
    /// # Parameters
    /// - `bits`: Boolean `ExprId`s in little-endian order.
    ///
    /// # Returns
    /// - `Ok(ExprId)` if `bits.len() <= F::bits()`, otherwise an error.
    ///
    /// # Cost
    /// `n` boolean constraints + `n` multiplications + `n` additions,
    /// where `n = bits.len()`.
    pub fn reconstruct_index_from_bits<BF>(
        &mut self,
        bits: &[ExprId],
    ) -> Result<ExprId, CircuitBuilderError>
    where
        F: ExtensionField<BF>,
        BF: Field,
    {
        self.push_scope("reconstruct_index_from_bits");

        if bits.len() > F::bits() {
            return Err(CircuitBuilderError::BinaryDecompositionTooManyBits {
                expected: F::bits(),
                n_bits: bits.len(),
            });
        }

        // Accumulator for the running sum.
        let mut acc = self.define_const(F::ZERO);

        for (i, chunk) in bits.chunks(BF::bits()).enumerate() {
            // The canonical basis element e_i.
            let mut e_i = vec![BF::ZERO; F::DIMENSION];
            e_i[i] = BF::ONE;
            let e_i =
                F::from_basis_coefficients_slice(&e_i).expect("`basis` is of size `F::DIMENSION`");
            for (j, &b) in chunk.iter().enumerate() {
                // Add the constant `2^j * e_i`
                let pow2 = self.define_const(e_i * BF::from_u64(1 << j));
                // Ensure each bit is boolean.
                self.assert_bool(b);

                // Add b_i · 2^j to the accumulator (at the corresponding limb).
                acc = self.mul_add(b, pow2, acc);
            }
        }

        self.pop_scope();
        Ok(acc)
    }

    /// Records that `result` is the recomposition of `coeffs` in the coefficient-provenance
    /// cache, without adding any circuit constraint. A subsequent
    /// [`Self::decompose_ext_to_base_coeffs`] on `result` can return `coeffs` directly.
    ///
    /// The caller must ensure `result` equals the field recomposition of `coeffs`. In debug
    /// builds, conflicting provenance for the same `ExprId` or use of `ExprId::ZERO` is
    /// rejected via `debug_assert`.
    pub fn hint_ext_recompose_coeffs(&mut self, result: ExprId, coeffs: &[ExprId]) {
        debug_assert_ne!(
            result,
            ExprId::ZERO,
            "hint_ext_recompose_coeffs: ExprId::ZERO is shared circuit-wide; do not attach provenance"
        );
        if let Some(existing) = self.ext_recompose_coeffs.get(&result) {
            debug_assert_eq!(
                existing.as_slice(),
                coeffs,
                "hint_ext_recompose_coeffs: conflicting coefficient provenance for the same ExprId"
            );
        }
        self.ext_recompose_coeffs.insert(result, coeffs.to_vec());
    }

    /// Recomposes D base field coefficients into an extension field element.
    ///
    /// Given coefficients `[c_0, c_1, ..., c_{D-1}]`, computes `x = sum(c_i * basis_i)`
    /// where `basis_i` is the i-th canonical basis element of the extension field.
    ///
    /// Each input coefficient should be a base field element embedded in the extension
    /// field (i.e., only the first basis component is non-zero).
    ///
    /// # Parameters
    /// - `coeffs`: Slice of D base field coefficient targets
    ///
    /// # Returns
    /// A single target representing the extension field element
    ///
    /// # Errors
    /// Returns error if `coeffs.len() != F::DIMENSION`
    ///
    /// # Cost
    /// When recompose NPO is enabled: 1 NPO row (zero ALU cost).
    /// Otherwise: D multiplications + (D-1) additions.
    ///
    /// The builder records the output in the coefficient-provenance cache so a later
    /// [`Self::decompose_ext_to_base_coeffs`] on that output can return the same `coeffs`
    /// without extra witness rows (this path **does** constrain recomposition via the recompose AIR).
    ///
    /// Uses the standard recompose AIR (no per-coefficient WitnessChecks receives). Prefer
    /// [`Self::recompose_base_coeffs_to_ext_with_coeff_lookups`] when the coefficient targets
    /// are read by a lower-degree Poseidon2 (e.g. D=1) after decomposition.
    pub fn recompose_base_coeffs_to_ext<BF>(
        &mut self,
        coeffs: &[ExprId],
    ) -> Result<ExprId, CircuitBuilderError>
    where
        BF: PrimeField64,
        F: ExtensionField<BF>,
    {
        self.recompose_base_coeffs_to_ext_impl::<BF>(coeffs, false)
    }

    /// Like [`Self::recompose_base_coeffs_to_ext`], but uses the `recompose/coeff` table so the
    /// BF coefficients appear on the WitnessChecks bus (per-coefficient receives).
    /// Required for soundness when those coefficients are consumed by a D=1 Poseidon2 (or similar)
    /// inside a higher-degree circuit.
    pub fn recompose_base_coeffs_to_ext_with_coeff_lookups<BF>(
        &mut self,
        coeffs: &[ExprId],
    ) -> Result<ExprId, CircuitBuilderError>
    where
        BF: PrimeField64,
        F: ExtensionField<BF>,
    {
        self.recompose_base_coeffs_to_ext_impl::<BF>(coeffs, true)
    }

    fn recompose_base_coeffs_to_ext_impl<BF>(
        &mut self,
        coeffs: &[ExprId],
        coeff_lookups: bool,
    ) -> Result<ExprId, CircuitBuilderError>
    where
        BF: PrimeField64,
        F: ExtensionField<BF>,
    {
        if coeffs.len() != F::DIMENSION {
            return Err(CircuitBuilderError::InvalidDimension {
                expected: F::DIMENSION,
                actual: coeffs.len(),
            });
        }

        // Constant fold: when every coefficient is a Const, build the recomposed EF value
        // directly and skip the NPO row (or the D mul_add chain). The coefficient Const ops
        // remain in the witness table for any other consumers.
        let bf_consts: Option<Vec<BF>> = coeffs
            .iter()
            .map(|&c| {
                self.expr_builder
                    .get_const_value(c)
                    .map(|ef| <F as BasedVectorSpace<BF>>::as_basis_coefficients_slice(&ef)[0])
            })
            .collect();
        if let Some(bf_values) = bf_consts {
            let folded =
                F::from_basis_coefficients_slice(&bf_values).expect("basis coefficients are valid");
            let result = self.alloc_const(folded, "recompose_const_fold");
            // Skip provenance for ZERO: it is shared circuit-wide and must not carry coeffs.
            if result != ExprId::ZERO {
                self.ext_recompose_coeffs.insert(result, coeffs.to_vec());
            }
            return Ok(result);
        }

        let result = if self.recompose_npo_enabled {
            self.recompose_via_npo(coeffs, coeff_lookups)?
        } else {
            self.push_scope("recompose_base_coeffs_to_ext");

            let mut acc = self.define_const(F::ZERO);

            for (i, &coeff) in coeffs.iter().enumerate() {
                let mut basis_coeffs = vec![BF::ZERO; F::DIMENSION];
                basis_coeffs[i] = BF::ONE;
                let basis_elem = F::from_basis_coefficients_slice(&basis_coeffs)
                    .expect("basis coefficients are valid");

                let basis_const = self.define_const(basis_elem);
                acc = self.mul_add(coeff, basis_const, acc);
            }

            self.pop_scope();
            acc
        };

        self.ext_recompose_coeffs.insert(result, coeffs.to_vec());
        Ok(result)
    }

    /// Recompose via the dedicated NPO table (zero ALU cost).
    fn recompose_via_npo(
        &mut self,
        coeffs: &[ExprId],
        coeff_lookups: bool,
    ) -> Result<ExprId, CircuitBuilderError> {
        self.push_scope("recompose_base_coeffs_to_ext");

        let op_type = if coeff_lookups {
            NpoTypeId::recompose_with_coeff_lookups()
        } else {
            NpoTypeId::recompose()
        };

        let (_, _call, outputs) = self.push_non_primitive_op_with_outputs(
            op_type,
            vec![coeffs.to_vec()],
            vec![Some("recompose_out")],
            Some(NonPrimitiveOpParams::Recompose),
            "recompose",
        );

        let result = outputs[0].ok_or(CircuitBuilderError::MissingOutput)?;
        self.pop_scope();
        Ok(result)
    }

    /// Decomposes an extension field element into its D base field coefficients.
    ///
    /// Given `x = c_0 + c_1*w + c_2*w^2 + ... + c_{D-1}*w^{D-1}`, returns `[c_0, c_1, ..., c_{D-1}]`
    /// as targets. Each coefficient target represents a base field element embedded in the
    /// extension field (i.e., only the first basis component is non-zero).
    ///
    /// # Parameters
    /// - `x`: The extension field element to decompose
    ///
    /// # Returns
    /// Vector of D targets, each representing a base field coefficient
    ///
    /// # Constraints Added
    /// - D witness allocations for coefficients (via `ExtDecompositionHint`)
    /// - 1 recomposition constraint: `sum(c_i * basis_i) == x`
    ///
    /// # Cost
    /// - If `x` is the output of `recompose_base_coeffs_to_ext` (same builder, modulo `connect`
    ///   to non-zero wires): **no extra rows** — returns the original coefficient `ExprId`s.
    /// - Otherwise: D Witness rows + D Mul rows + (D-1) Add rows (for the recomposition constraint)
    pub fn decompose_ext_to_base_coeffs<BF>(
        &mut self,
        x: ExprId,
    ) -> Result<Vec<ExprId>, CircuitBuilderError>
    where
        BF: PrimeField64,
        F: ExtensionField<BF>,
    {
        if let Some(coeffs) = self.ext_recompose_coeffs.get(&x) {
            debug_assert_eq!(coeffs.len(), F::DIMENSION);
            return Ok(coeffs.clone());
        }

        // Constant fold: when `x` is a Const, materialize each coefficient as a Const ExprId
        // and skip both the D witness allocations and the recomposition constraint.
        if let Some(ext_val) = self.expr_builder.get_const_value(x) {
            let bf_coeffs = <F as BasedVectorSpace<BF>>::as_basis_coefficients_slice(&ext_val);
            debug_assert_eq!(bf_coeffs.len(), F::DIMENSION);
            let mut embedded_buf = vec![BF::ZERO; F::DIMENSION];
            let coeffs: Vec<ExprId> = bf_coeffs
                .iter()
                .map(|&bf| {
                    embedded_buf[0] = bf;
                    let embedded_ef = F::from_basis_coefficients_slice(&embedded_buf)
                        .expect("embedded coefficients are valid");
                    self.alloc_const(embedded_ef, "decompose_const_fold")
                })
                .collect();
            // Cache the provenance so a later recompose on these coeffs returns the same `x`.
            if x != ExprId::ZERO {
                self.ext_recompose_coeffs.insert(x, coeffs.clone());
            }
            return Ok(coeffs);
        }

        // If x = select(b, t, s) and at least one input has known coefficient provenance,
        // decompose coefficient-wise: coeff[i] = select(b, t_coeff[i], s_coeff[i]).
        // For the input without provenance, decompose it recursively (may generate witnesses).
        // This saves D witness allocations for every input that IS in the provenance cache.
        if let Some((b, t, s)) = self.ext_select_sources.get(&x).copied() {
            let t_coeffs_opt = self.ext_recompose_coeffs.get(&t).cloned();
            let s_coeffs_opt = self.ext_recompose_coeffs.get(&s).cloned();
            if t_coeffs_opt.is_some() || s_coeffs_opt.is_some() {
                // Non-cached branches are decomposed WITHOUT the coeff-ctl flag: their
                // witnesses are not placed directly into Poseidon2 rate slots — only the
                // select results are. We create a ctl entry for x itself below (if needed),
                // so the WitnessChecks bus stays balanced.
                let saved_ctl = self.recompose_coeff_ctl_for_decompose_links;
                self.recompose_coeff_ctl_for_decompose_links = false;
                let t_coeffs = match t_coeffs_opt {
                    Some(c) => c,
                    None => self.decompose_ext_to_base_coeffs::<BF>(t)?,
                };
                let s_coeffs = match s_coeffs_opt {
                    Some(c) => c,
                    None => self.decompose_ext_to_base_coeffs::<BF>(s)?,
                };
                self.recompose_coeff_ctl_for_decompose_links = saved_ctl;
                debug_assert_eq!(t_coeffs.len(), F::DIMENSION);
                debug_assert_eq!(s_coeffs.len(), F::DIMENSION);
                let mut coeffs = Vec::with_capacity(F::DIMENSION);
                for (&tc, &sc) in t_coeffs.iter().zip(s_coeffs.iter()) {
                    coeffs.push(self.select(b, tc, sc));
                }
                if saved_ctl {
                    // The select coefficients are what enter Poseidon2 rate slots, so they
                    // need a coeff-ctl entry. Emit one for x now that the selects are known.
                    let reconstructed =
                        self.recompose_base_coeffs_to_ext_with_coeff_lookups::<BF>(&coeffs)?;
                    self.connect(x, reconstructed);
                    // connect() propagates ext_recompose_coeffs[reconstructed] → x
                } else {
                    self.ext_recompose_coeffs.insert(x, coeffs.clone());
                }
                return Ok(coeffs);
            }
        }

        self.push_scope("decompose_ext_to_base_coeffs");

        // Allocate D witness slots for coefficients using hint
        let ext_decomposition_hint = ExtDecompositionHint::<BF>::new();
        let coeffs: Vec<ExprId> = self
            .push_unconstrained_op(
                vec![vec![x]],
                F::DIMENSION,
                ext_decomposition_hint,
                "ext_decomposition",
            )
            .2
            .into_iter()
            .collect::<Option<Vec<_>>>()
            .ok_or(CircuitBuilderError::MissingOutput)?;

        // Constrain: sum(coeffs[i] * basis[i]) == x
        let reconstructed = if self.recompose_coeff_ctl_for_decompose_links {
            self.recompose_base_coeffs_to_ext_with_coeff_lookups::<BF>(&coeffs)?
        } else {
            self.recompose_base_coeffs_to_ext::<BF>(&coeffs)?
        };
        self.connect(x, reconstructed);

        self.pop_scope();
        Ok(coeffs)
    }

    /// Applies Poseidon2 permutation for the circuit challenger.
    ///
    /// Takes 4 extension element inputs and returns 4 extension element outputs.
    /// This operation is **CTL-verified** against the Poseidon2 AIR table for soundness.
    ///
    /// # CTL Verification
    /// - Inputs 0-3: CTL-verified against witness table
    /// - Outputs 0-1: CTL-verified against witness table (rate elements)
    /// - Outputs 2-3: NOT CTL-verified (capacity elements, constrained by Poseidon2 AIR)
    ///
    /// # Parameters
    /// - `config`: The Poseidon2 configuration to use
    /// - `inputs`: width_ext extension element targets (the sponge state)
    ///
    /// # Returns
    /// width_ext extension element targets (the permuted state)
    ///
    /// # Errors
    /// Returns error if the Poseidon2 operation is not enabled
    pub fn add_poseidon2_perm_for_challenger(
        &mut self,
        config: crate::ops::Poseidon2Config,
        inputs: &[ExprId],
    ) -> Result<Vec<ExprId>, CircuitBuilderError> {
        self.push_scope("poseidon2_perm_for_challenger");

        // Use add_poseidon2_perm with CTL verification for soundness
        // - All 4 inputs are CTL-verified
        // - Outputs 0-1 are CTL-verified (rate elements)
        // - Outputs 2-3 are returned but NOT CTL-verified (capacity elements)
        let width_ext = config.width_ext();
        let (_op_id, outputs) = self.add_poseidon2_perm(&Poseidon2PermCall {
            config,
            new_start: true, // Each challenger permutation is independent
            merkle_path: false,
            mmcs_bit: None,
            inputs: inputs.iter().map(|&x| Some(x)).collect(),
            out_ctl: vec![true; config.rate_ext()],
            return_all_outputs: true,
            mmcs_index_sum: None,
        })?;

        let output_exprs: Vec<ExprId> = (0..width_ext)
            .map(|i| outputs[i].ok_or(CircuitBuilderError::MissingOutput))
            .collect::<Result<Vec<_>, _>>()?;

        self.pop_scope();
        Ok(output_exprs)
    }

    /// Applies Poseidon2 permutation for the circuit challenger (base field, D=1).
    ///
    /// Takes 16 base field element inputs and returns 16 base field element outputs.
    /// This operation is **CTL-verified** against the Poseidon2 AIR table for soundness.
    ///
    /// # CTL Verification
    /// - Inputs 0-7: CTL-verified against witness table (rate)
    /// - Inputs 8-15: not CTL-verified; sponge `new_start` enforces zero via AIR, else chained
    /// - Outputs 0-7: CTL-verified against witness table (rate elements)
    /// - Outputs 8-15: NOT CTL-verified (capacity elements, constrained by Poseidon2 AIR)
    ///
    /// # Parameters
    /// - `config`: The Poseidon2 configuration to use (must be D=1)
    /// - `new_start`: When `true`, rate inputs (0-7) are CTL-verified; capacity slots (8-15)
    ///   should be `None` (zeros are enforced by the AIR). When `false`, rate is CTL-verified and
    ///   capacity `None` inherits from the previous row via the chain constraint.
    /// - `inputs`: Sixteen slots: `Some(expr)` for CTL-verified or witness-fed values, `None` for
    ///   zeroed or chain-inherited capacity per `new_start`.
    ///
    /// # Returns
    /// 16 base field element targets (the permuted state)
    ///
    /// # Errors
    /// Returns an error if the Poseidon2 operation is not enabled.
    pub fn add_poseidon2_perm_for_challenger_base(
        &mut self,
        config: crate::ops::Poseidon2Config,
        new_start: bool,
        inputs: [Option<ExprId>; 16],
    ) -> Result<[ExprId; 16], CircuitBuilderError> {
        self.push_scope("poseidon2_perm_for_challenger_base");

        // Rate outputs (0-7) are CTL-verified; capacity outputs (8-15) are chained.
        let (_op_id, outputs) = self.add_poseidon2_perm_base(&Poseidon2PermCallBase {
            config,
            new_start,
            inputs,
            out_ctl: [true; 8],
            return_all_outputs: true,
        })?;

        let output_exprs: [ExprId; 16] =
            core::array::from_fn(|i| outputs[i].expect("output should exist"));

        self.pop_scope();
        Ok(output_exprs)
    }
}

/// Witness hint for extension field decomposition.
///
/// At runtime, extracts the basis coefficients from an extension field element
/// and embeds each coefficient as an extension field element with zeroed higher coefficients.
#[derive(Debug, Clone)]
struct ExtDecompositionHint<BF: PrimeField64> {
    _phantom: PhantomData<BF>,
}

impl<BF: PrimeField64> ExtDecompositionHint<BF> {
    pub const fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<BF: PrimeField64, EF: ExtensionField<BF>> HintExecutor<EF> for ExtDecompositionHint<BF> {
    fn execute(
        &self,
        inputs: &[crate::WitnessId],
        outputs: &[crate::WitnessId],
        witness: &mut [Option<EF>],
    ) -> Result<(), CircuitError> {
        if inputs.len() != 1 {
            return Err(CircuitError::UnconstrainedOpInputLengthMismatch {
                op: "ExtDecompositionHint".to_string(),
                expected: 1,
                got: inputs.len(),
            });
        }

        if outputs.len() != EF::DIMENSION {
            return Err(CircuitError::UnconstrainedOpInputLengthMismatch {
                op: "ExtDecompositionHint".to_string(),
                expected: EF::DIMENSION,
                got: outputs.len(),
            });
        }

        let in_wid = inputs[0];
        let in_idx = in_wid.0 as usize;
        let ext_val = witness
            .get(in_idx)
            .and_then(|opt| opt.as_ref())
            .map(Dup::dup)
            .ok_or(CircuitError::WitnessNotSet { witness_id: in_wid })?;
        let coeffs = ext_val.as_basis_coefficients_slice();
        let witness_len = witness.len();
        let mut embedded_buf = vec![BF::ZERO; EF::DIMENSION];

        for (i, &out_wid) in outputs.iter().enumerate() {
            let coeff = coeffs
                .get(i)
                .ok_or(CircuitError::InvalidPreprocessedValues)?;
            embedded_buf[0] = *coeff;
            let embedded_ef = EF::from_basis_coefficients_slice(&embedded_buf)
                .expect("embedded coefficients are valid");

            let out_idx = out_wid.0 as usize;
            if out_idx >= witness_len {
                return Err(CircuitError::WitnessIdOutOfBounds {
                    witness_id: out_wid,
                });
            }
            let slot = &mut witness[out_idx];
            if let Some(existing) = slot.as_ref() {
                if *existing != embedded_ef {
                    return Err(CircuitError::WitnessConflict {
                        witness_id: out_wid,
                        existing: format!("{existing:?}"),
                        new: format!("{embedded_ef:?}"),
                        expr_ids: vec![],
                    });
                }
            } else {
                *slot = Some(embedded_ef);
            }
        }

        Ok(())
    }

    fn boxed(&self) -> alloc::boxed::Box<dyn HintExecutor<EF>> {
        Box::new(self.clone())
    }
}

/// Witness hint for binary decomposition of a field element.
///
/// At runtime:
/// - It extracts the canonical `u64` representation of the input field element,
/// - It fills the witness with its little-endian binary decomposition.
#[derive(Debug, Clone)]
struct BinaryDecompositionHint<BF: PrimeField64> {
    /// Phantom data for the base field type.
    _phantom: PhantomData<BF>,
}

impl<BF: PrimeField64> BinaryDecompositionHint<BF> {
    /// Creates a new binary decomposition hint.
    pub const fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<BF: PrimeField64, EF: ExtensionField<BF>> HintExecutor<EF> for BinaryDecompositionHint<BF> {
    fn execute(
        &self,
        inputs: &[crate::WitnessId],
        outputs: &[crate::WitnessId],
        witness: &mut [Option<EF>],
    ) -> Result<(), CircuitError> {
        if inputs.len() != 1 {
            return Err(CircuitError::UnconstrainedOpInputLengthMismatch {
                op: "BinaryDecompositionHint".to_string(),
                expected: 1,
                got: inputs.len(),
            });
        }

        let felt_bits = BF::bits();
        if outputs.len() > felt_bits * EF::DIMENSION {
            return Err(CircuitError::BinaryDecompositionTooManyBits {
                expected: felt_bits * EF::DIMENSION,
                n_bits: outputs.len(),
            });
        }

        let in_wid = inputs[0];
        let in_idx = in_wid.0 as usize;
        let ext_val = witness
            .get(in_idx)
            .and_then(|opt| opt.as_ref())
            .map(Dup::dup)
            .ok_or(CircuitError::WitnessNotSet { witness_id: in_wid })?;

        let witness_len = witness.len();
        let coeffs = ext_val.as_basis_coefficients_slice();
        let n_out = outputs.len();
        let mut o = 0usize;

        for coeff in coeffs {
            let val = coeff.as_canonical_u64();
            for i in 0..felt_bits {
                if o >= n_out {
                    return Ok(());
                }
                let bit = EF::from_bool(val >> i & 1 == 1);
                let out_wid = outputs[o];
                o += 1;

                let out_idx = out_wid.0 as usize;
                if out_idx >= witness_len {
                    return Err(CircuitError::WitnessIdOutOfBounds {
                        witness_id: out_wid,
                    });
                }
                let slot = &mut witness[out_idx];
                if let Some(existing) = slot.as_ref() {
                    if *existing != bit {
                        return Err(CircuitError::WitnessConflict {
                            witness_id: out_wid,
                            existing: format!("{existing:?}"),
                            new: format!("{bit:?}"),
                            expr_ids: vec![],
                        });
                    }
                } else {
                    *slot = Some(bit);
                }
            }
        }

        Ok(())
    }

    fn boxed(&self) -> alloc::boxed::Box<dyn HintExecutor<EF>> {
        Box::new(self.clone())
    }
}

#[cfg(test)]
mod tests {

    use p3_test_utils::baby_bear_params::{BabyBear, BinomialExtensionField};

    use super::*;

    #[test]
    fn test_new_builder_initialization() {
        let builder = CircuitBuilder::<BabyBear>::new();
        assert_eq!(builder.public_input_count(), 0);
    }

    #[test]
    fn test_default_same_as_new() {
        let builder1 = CircuitBuilder::<BabyBear>::new();
        let builder2 = CircuitBuilder::<BabyBear>::default();
        assert_eq!(builder1.public_input_count(), builder2.public_input_count());
    }

    #[test]
    fn test_add_public_input_single() {
        let mut builder = CircuitBuilder::<BabyBear>::new();
        builder.public_input();
        assert_eq!(builder.public_input_count(), 1);
    }

    #[test]
    fn test_alloc_public_inputs_multiple() {
        let mut builder = CircuitBuilder::<BabyBear>::new();
        let pis = builder.alloc_public_inputs(5, "batch");
        assert_eq!(pis.len(), 5);
        assert_eq!(builder.public_input_count(), 5);
    }

    #[test]
    fn test_alloc_public_input_array() {
        let mut builder = CircuitBuilder::<BabyBear>::new();
        let pis: [ExprId; 3] = builder.alloc_public_input_array("array");
        assert_eq!(pis.len(), 3);
        assert_eq!(builder.public_input_count(), 3);
    }

    #[test]
    fn test_public_input_count_increments() {
        let mut builder = CircuitBuilder::<BabyBear>::new();
        assert_eq!(builder.public_input_count(), 0);
        builder.public_input();
        assert_eq!(builder.public_input_count(), 1);
        builder.public_input();
        assert_eq!(builder.public_input_count(), 2);
    }

    #[test]
    fn test_add_const_deduplication() {
        let mut builder = CircuitBuilder::<BabyBear>::new();
        let c1 = builder.define_const(BabyBear::from_u64(99));
        let c2 = builder.define_const(BabyBear::from_u64(99));
        assert_eq!(c1, c2);
    }

    #[test]
    fn test_exp_power_of_2_zero() {
        let mut builder = CircuitBuilder::<BabyBear>::new();
        let base = builder.define_const(BabyBear::from_u64(5));
        let result = builder.exp_power_of_2(base, 0);
        assert_eq!(result, base);
    }

    #[test]
    fn test_select_operation() {
        let mut builder = CircuitBuilder::<BabyBear>::new();
        let b = builder.public_input();
        let t = builder.define_const(BabyBear::from_u64(10));
        let s = builder.define_const(BabyBear::from_u64(5));
        let _result = builder.select(b, t, s);
        // Should create: t_minus_s, scaled, and result
        assert_eq!(builder.public_input_count(), 1);
    }

    #[test]
    fn test_select_shortcuts() {
        let mut builder = CircuitBuilder::<BabyBear>::new();
        let t = builder.public_input();
        let s = builder.public_input();
        let zero = builder.define_const(BabyBear::ZERO);
        let one = builder.define_const(BabyBear::ONE);

        // select(b, t, t) = t (identical branches)
        assert_eq!(builder.select(zero, t, t), t);

        // select(0, t, s) = s
        assert_eq!(builder.select(zero, t, s), s);

        // select(1, t, s) = t
        assert_eq!(builder.select(one, t, s), t);
    }

    #[test]
    #[cfg(feature = "debugging")]
    fn test_scope_operations() {
        let mut builder = CircuitBuilder::<BabyBear>::new();
        builder.push_scope("test_scope");
        builder.define_const(BabyBear::ONE);
        builder.pop_scope();
        let scopes = builder.list_scopes();
        assert!(scopes.contains(&("test_scope".to_string())));
    }

    #[test]
    #[cfg(feature = "debugging")]
    fn test_list_scopes_release() {
        let builder = CircuitBuilder::<BabyBear>::new();
        assert!(builder.list_scopes().is_empty());
    }

    #[test]
    fn test_build_empty_circuit() {
        let builder = CircuitBuilder::<BabyBear>::new();
        let circuit = builder
            .build()
            .expect("Empty circuit should build successfully");

        assert_eq!(circuit.public_flat_len, 0);
        assert_eq!(circuit.witness_count, 1);
        assert_eq!(circuit.ops.len(), 1);
        assert!(circuit.public_rows.is_empty());
        assert!(circuit.enabled_ops.is_empty());

        match &circuit.ops[0] {
            crate::ops::Op::Const { out, val } => {
                assert_eq!(*out, WitnessId(0));
                assert_eq!(*val, BabyBear::ZERO);
            }
            _ => panic!("Expected Const operation at index 0"),
        }
    }

    #[test]
    fn test_build_with_public_inputs() {
        let mut builder = CircuitBuilder::<BabyBear>::new();
        builder.public_input();
        builder.public_input();
        let circuit = builder
            .build()
            .expect("Circuit with public inputs should build");

        assert_eq!(circuit.public_flat_len, 2);
        assert_eq!(circuit.public_rows.len(), 2);
        assert_eq!(circuit.witness_count, 3);
        assert_eq!(circuit.ops.len(), 3);

        match &circuit.ops[0] {
            crate::ops::Op::Const { out, val } => {
                assert_eq!(*out, WitnessId(0));
                assert_eq!(*val, BabyBear::ZERO);
            }
            _ => panic!("Expected Const at index 0"),
        }

        match &circuit.ops[1] {
            crate::ops::Op::Public { out, public_pos } => {
                assert_eq!(*out, WitnessId(1));
                assert_eq!(*public_pos, 0);
            }
            _ => panic!("Expected Public at index 1"),
        }

        match &circuit.ops[2] {
            crate::ops::Op::Public { out, public_pos } => {
                assert_eq!(*out, WitnessId(2));
                assert_eq!(*public_pos, 1);
            }
            _ => panic!("Expected Public at index 2"),
        }

        assert_eq!(circuit.public_rows[0], WitnessId(1));
        assert_eq!(circuit.public_rows[1], WitnessId(2));
    }

    #[test]
    fn test_build_with_constants() {
        let mut builder = CircuitBuilder::<BabyBear>::new();
        builder.define_const(BabyBear::from_u64(1));
        builder.define_const(BabyBear::from_u64(2));
        let circuit = builder
            .build()
            .expect("Circuit with constants should build");

        assert_eq!(circuit.public_flat_len, 0);
        assert!(circuit.public_rows.is_empty());
        assert_eq!(circuit.witness_count, 3);
        assert_eq!(circuit.ops.len(), 3);

        match &circuit.ops[0] {
            crate::ops::Op::Const { out, val } => {
                assert_eq!(*out, WitnessId(0));
                assert_eq!(*val, BabyBear::ZERO);
            }
            _ => panic!("Expected Const at index 0"),
        }

        match &circuit.ops[1] {
            crate::ops::Op::Const { out, val } => {
                assert_eq!(*out, WitnessId(1));
                assert_eq!(*val, BabyBear::from_u64(1));
            }
            _ => panic!("Expected Const at index 1"),
        }

        match &circuit.ops[2] {
            crate::ops::Op::Const { out, val } => {
                assert_eq!(*out, WitnessId(2));
                assert_eq!(*val, BabyBear::from_u64(2));
            }
            _ => panic!("Expected Const at index 2"),
        }
    }

    #[test]
    fn test_build_with_operations() {
        // Use a public input so constant folding doesn't eliminate the Add
        let mut builder = CircuitBuilder::<BabyBear>::new();
        let a = builder.public_input();
        let b = builder.define_const(BabyBear::from_u64(3));
        builder.add(a, b);
        let circuit = builder
            .build()
            .expect("Circuit with operations should build");

        // zero const + public + const(3) + add result = 4 witnesses
        assert_eq!(circuit.witness_count, 4);

        // Should contain an ALU Add op
        let has_alu_add = circuit.ops.iter().any(|op| {
            matches!(
                op,
                crate::ops::Op::Alu {
                    kind: crate::ops::AluOpKind::Add,
                    ..
                }
            )
        });
        assert!(has_alu_add, "Expected an ALU Add operation");
    }

    #[test]
    fn test_build_with_public_mapping() {
        let mut builder = CircuitBuilder::<BabyBear>::new();
        let p0 = builder.public_input();
        let p1 = builder.public_input();
        let (circuit, mapping) = builder
            .build_with_public_mapping()
            .expect("Circuit should build with public mapping");

        assert_eq!(circuit.public_flat_len, 2);
        assert_eq!(mapping.len(), 2);
        assert_eq!(mapping[&p0], WitnessId(1));
        assert_eq!(mapping[&p1], WitnessId(2));
    }

    #[test]
    fn test_build_with_connect_deduplication() {
        let mut builder = CircuitBuilder::<BabyBear>::new();
        let a = builder.define_const(BabyBear::from_u64(5));
        let b = builder.define_const(BabyBear::from_u64(5));
        builder.connect(a, b);
        let circuit = builder
            .build()
            .expect("Circuit with constraints should build");

        assert_eq!(circuit.witness_count, 2);
        assert_eq!(circuit.ops.len(), 2);
    }

    #[test]
    fn test_non_primitive_outputs_ordering_and_dedup() {
        use crate::ops::poseidon2_perm::Poseidon2PermExec;
        use crate::ops::{Poseidon2Config, Poseidon2PermCall};

        type Ext4 = BinomialExtensionField<BabyBear, 4>;

        let mut builder = CircuitBuilder::<Ext4>::new();
        let dummy_exec: Poseidon2PermExec<Ext4> =
            Arc::new(|_| panic!("should not be called in this test"));
        let plugin = Poseidon2CircuitPlugin::<Ext4>::new(
            Poseidon2Config::BabyBearD4Width16,
            dummy_exec,
            |_| Ok(None),
        );
        builder.register_npo(plugin);

        let z = builder.define_const(Ext4::ZERO);
        let (op_id, outputs) = builder
            .add_poseidon2_perm(&Poseidon2PermCall {
                config: Poseidon2Config::BabyBearD4Width16,
                new_start: true,
                merkle_path: false,
                mmcs_bit: None,
                inputs: vec![Some(z), Some(z), Some(z), Some(z)],
                out_ctl: vec![true, true],
                return_all_outputs: false,
                mmcs_index_sum: None,
            })
            .unwrap();

        let out0 = outputs[0].unwrap();
        let out1 = outputs[1].unwrap();

        let one = builder.define_const(Ext4::ONE);
        let sum0 = builder.add(out0, one);
        let sum1 = builder.add(out1, one);

        let circuit = builder.build().unwrap();

        // Non-primitive op emitted exactly once.
        let non_prims: Vec<_> = circuit
            .ops
            .iter()
            .enumerate()
            .filter_map(|(i, op)| match op {
                crate::ops::Op::NonPrimitiveOpWithExecutor { op_id: oid, .. } if *oid == op_id => {
                    Some(i)
                }
                _ => None,
            })
            .collect();
        assert_eq!(non_prims.len(), 1);
        let non_prim_pos = non_prims[0];

        // Exact Add matches (order of a/b may swap).
        let w_out0 = circuit.expr_to_widx[&out0];
        let w_out1 = circuit.expr_to_widx[&out1];
        let w_one = circuit.expr_to_widx[&one];
        let w_sum0 = circuit.expr_to_widx[&sum0];
        let w_sum1 = circuit.expr_to_widx[&sum1];

        let add0_pos = circuit
            .ops
            .iter()
            .position(|op| match op {
                crate::ops::Op::Alu {
                    kind: crate::ops::AluOpKind::Add,
                    a,
                    b,
                    out,
                    ..
                } => {
                    *out == w_sum0
                        && ((*a == w_out0 && *b == w_one) || (*a == w_one && *b == w_out0))
                }
                _ => false,
            })
            .unwrap();

        let add1_pos = circuit
            .ops
            .iter()
            .position(|op| match op {
                crate::ops::Op::Alu {
                    kind: crate::ops::AluOpKind::Add,
                    a,
                    b,
                    out,
                    ..
                } => {
                    *out == w_sum1
                        && ((*a == w_out1 && *b == w_one) || (*a == w_one && *b == w_out1))
                }
                _ => false,
            })
            .unwrap();

        assert!(non_prim_pos < add0_pos);
        assert!(non_prim_pos < add1_pos);
    }

    #[test]
    fn test_basic_tagging() {
        let mut builder = CircuitBuilder::<BabyBear>::new();
        let a = builder.define_const(BabyBear::from_u64(5));
        let b = builder.define_const(BabyBear::from_u64(7));
        let sum = builder.add(a, b);

        builder.tag(sum, "my-sum").unwrap();

        let circuit = builder.build().unwrap();
        let runner = circuit.runner();
        let traces = runner.run().unwrap();

        let probed = traces.probe("my-sum").unwrap();
        assert_eq!(*probed, BabyBear::from_u64(12));
    }

    #[test]
    fn test_tag_multiple_wires() {
        let mut builder = CircuitBuilder::<BabyBear>::new();
        let a = builder.define_const(BabyBear::from_u64(10));
        let b = builder.define_const(BabyBear::from_u64(20));
        let sum = builder.add(a, b);
        let prod = builder.mul(a, b);

        builder.tag(sum, "the-sum").unwrap();
        builder.tag(prod, "the-product").unwrap();

        let circuit = builder.build().unwrap();
        let runner = circuit.runner();
        let traces = runner.run().unwrap();

        assert_eq!(*traces.probe("the-sum").unwrap(), BabyBear::from_u64(30));
        assert_eq!(
            *traces.probe("the-product").unwrap(),
            BabyBear::from_u64(200)
        );
    }

    #[test]
    fn test_probe_unknown_tag() {
        let mut builder = CircuitBuilder::<BabyBear>::new();
        let a = builder.define_const(BabyBear::ONE);
        builder.tag(a, "known").unwrap();

        let circuit = builder.build().unwrap();
        let runner = circuit.runner();
        let traces = runner.run().unwrap();

        assert!(traces.probe("known").is_some());
        assert!(traces.probe("unknown").is_none());
    }

    #[test]
    fn test_duplicate_tag() {
        let mut builder = CircuitBuilder::<BabyBear>::new();
        let a = builder.define_const(BabyBear::ONE);
        let b = builder.define_const(BabyBear::from_u64(2));

        builder.tag(a, "same-tag").unwrap();
        let result = builder.tag(b, "same-tag");

        assert!(matches!(
            result,
            Err(CircuitBuilderError::DuplicateTag { tag }) if tag == "same-tag"
        ));
    }

    #[test]
    fn test_tag_with_dynamic_string() {
        let mut builder = CircuitBuilder::<BabyBear>::new();

        for i in 0..3 {
            let val = builder.define_const(BabyBear::from_u64(i as u64));
            builder.tag(val, format!("wire-{}", i)).unwrap();
        }

        let circuit = builder.build().unwrap();
        let runner = circuit.runner();
        let traces = runner.run().unwrap();

        for i in 0..3 {
            let tag = format!("wire-{}", i);
            assert_eq!(
                *traces.probe(&tag).unwrap(),
                BabyBear::from_u64(i as u64),
                "wire-{} should have value {}",
                i,
                i
            );
        }
    }

    #[test]
    fn test_connected_tags_resolve_after_optimization() {
        let mut builder = CircuitBuilder::<BabyBear>::new();

        let x = builder.public_input();
        let one = builder.define_const(BabyBear::ONE);
        let a = builder.add(x, one);
        let b = builder.add(x, one); // b == a

        builder.tag(a, "result-a").unwrap();
        builder.tag(b, "result-b").unwrap();

        // Connect them - the optimizer should alias one to the other
        builder.connect(a, b);

        let circuit = builder.build().unwrap();
        let mut runner = circuit.runner();
        runner.set_public_inputs(&[BabyBear::from_u64(5)]).unwrap();
        let traces = runner.run().unwrap();

        // Both tags should resolve to the same value (5 + 1 = 6)
        let expected = BabyBear::from_u64(6);
        assert_eq!(traces.probe("result-a"), Some(&expected));
        assert_eq!(traces.probe("result-b"), Some(&expected));
    }
}

#[cfg(test)]
mod proptests {
    use alloc::vec;
    use core::array;

    use itertools::Itertools;
    use p3_test_utils::baby_bear_params::{
        BabyBear, BasedVectorSpace, BinomialExtensionField, PrimeCharacteristicRing,
    };
    use proptest::prelude::*;

    use super::*;

    // Strategy for generating valid field elements
    fn field_element() -> impl Strategy<Value = BabyBear> {
        any::<u64>().prop_map(BabyBear::from_u64)
    }

    impl From<ExprId> for WitnessId {
        fn from(expr_id: ExprId) -> Self {
            Self(expr_id.0)
        }
    }

    proptest! {
        #[test]
        fn field_add_commutative(a in field_element(), b in field_element()) {
            let mut builder1 = CircuitBuilder::<BabyBear>::new();
            let ca = builder1.define_const(a);
            let cb = builder1.define_const(b);
            let sum1 = builder1.add(ca, cb);

            let mut builder2 = CircuitBuilder::<BabyBear>::new();
            let ca2 = builder2.define_const(a);
            let cb2 = builder2.define_const(b);
            let sum2 = builder2.add(cb2, ca2);

            let circuit1 = builder1.build().unwrap();
            let circuit2 = builder2.build().unwrap();

            let runner1 = circuit1.runner();
            let runner2 = circuit2.runner();

            let traces1 = runner1.run().unwrap();
            let traces2 = runner2.run().unwrap();

            prop_assert_eq!(
                traces1.witness_trace.get_value(sum1.into()),
                traces2.witness_trace.get_value(sum2.into()),
                "addition should be commutative"
            );
        }

        #[test]
        fn field_mul_commutative(a in field_element(), b in field_element()) {
            let mut builder1 = CircuitBuilder::<BabyBear>::new();
            let ca = builder1.define_const(a);
            let cb = builder1.define_const(b);
            let prod1 = builder1.mul(ca, cb);

            let mut builder2 = CircuitBuilder::<BabyBear>::new();
            let ca2 = builder2.define_const(a);
            let cb2 = builder2.define_const(b);
            let prod2 = builder2.mul(cb2, ca2);

            let circuit1 = builder1.build().unwrap();
            let circuit2 = builder2.build().unwrap();

            let runner1 = circuit1.runner();
            let runner2 = circuit2.runner();

            let traces1 = runner1.run().unwrap();
            let traces2 = runner2.run().unwrap();

            prop_assert_eq!(
                traces1.witness_trace.get_value(prod1.into()),
                traces2.witness_trace.get_value(prod2.into()),
                "multiplication should be commutative"
            );
        }

        #[test]
        fn field_add_identity(a in field_element()) {
            let mut builder = CircuitBuilder::<BabyBear>::new();
            let ca = builder.define_const(a);
            let zero = builder.define_const(BabyBear::ZERO);
            let result = builder.add(ca, zero);

            let circuit = builder.build().unwrap();
            let runner = circuit.runner();
            let traces = runner.run().unwrap();

            prop_assert_eq!(
                traces.witness_trace.get_value(result.into()).unwrap(),
                &a,
                "a + 0 = a"
            );
        }

        #[test]
        fn field_mul_identity(a in field_element()) {
            let mut builder = CircuitBuilder::<BabyBear>::new();
            let ca = builder.define_const(a);
            let one = builder.define_const(BabyBear::ONE);
            let result = builder.mul(ca, one);

            let circuit = builder.build().unwrap();
            let  runner = circuit.runner();
            let traces = runner.run().unwrap();

            prop_assert_eq!(
                traces.witness_trace.get_value(result.into()).unwrap(),
                &a,
                "a * 1 = a"
            );
        }

        #[test]
        fn field_add_sub(a in field_element(), b in field_element()) {
            let mut builder = CircuitBuilder::<BabyBear>::new();
            let ca = builder.define_const(a);
            let cb = builder.define_const(b);
            let diff = builder.sub(ca, cb);
            let result = builder.add(diff, cb);

            let circuit = builder.build().unwrap();
            let  runner = circuit.runner();
            let traces = runner.run().unwrap();

            prop_assert_eq!(
                traces.witness_trace.get_value(result.into()).unwrap(),
                &a,
                "(a - b) + b = a"
            );
        }

        #[test]
        fn field_mul_div(a in field_element(), b in field_element().prop_filter("b must be non-zero", |&x| x != BabyBear::ZERO)) {
            let mut builder = CircuitBuilder::<BabyBear>::new();
            let ca = builder.define_const(a);
            let cb = builder.define_const(b);
            let quot = builder.div(ca, cb);
            let result = builder.mul(quot, cb);

            let circuit = builder.build().unwrap();
            let  runner = circuit.runner();
            let traces = runner.run().unwrap();

            prop_assert_eq!(
                traces.witness_trace.get_value(result.into()).unwrap(),
                &a,
                "(a / b) * b = a"
            );
        }
    }

    #[test]
    fn test_mul_add() {
        // Test case 1: Basic computation (3 * 4 + 5 = 17)
        {
            let mut builder = CircuitBuilder::<BabyBear>::new();
            let a = builder.define_const(BabyBear::from_u64(3));
            let b = builder.define_const(BabyBear::from_u64(4));
            let c = builder.define_const(BabyBear::from_u64(5));
            let result = builder.mul_add(a, b, c);

            let circuit = builder.build().unwrap();
            let runner = circuit.runner();
            let traces = runner.run().unwrap();

            assert_eq!(
                traces.witness_trace.get_value(result.into()).unwrap(),
                &BabyBear::from_u64(17)
            );
        }

        // Test case 2: With zero product (0 * 7 + 9 = 9)
        {
            let mut builder = CircuitBuilder::<BabyBear>::new();
            let zero = builder.define_const(BabyBear::ZERO);
            let b = builder.define_const(BabyBear::from_u64(7));
            let c = builder.define_const(BabyBear::from_u64(9));
            let result = builder.mul_add(zero, b, c);

            let circuit = builder.build().unwrap();
            let runner = circuit.runner();
            let traces = runner.run().unwrap();

            assert_eq!(
                traces.witness_trace.get_value(result.into()).unwrap(),
                &BabyBear::from_u64(9)
            );
        }
    }

    #[test]
    fn test_mul_many() {
        // Test case 1: Empty slice returns 1 (multiplicative identity)
        {
            let mut builder = CircuitBuilder::<BabyBear>::new();
            let result = builder.mul_many(&[]);

            let circuit = builder.build().unwrap();
            let runner = circuit.runner();
            let traces = runner.run().unwrap();

            assert_eq!(
                traces.witness_trace.get_value(result.into()).unwrap(),
                &BabyBear::ONE
            );
        }

        // Test case 2: Multiple elements [2, 3, 4, 5] = 120
        {
            let mut builder = CircuitBuilder::<BabyBear>::new();
            let vals: Vec<ExprId> = vec![2, 3, 4, 5]
                .into_iter()
                .map(|v| builder.define_const(BabyBear::from_u64(v)))
                .collect();
            let result = builder.mul_many(&vals);

            let circuit = builder.build().unwrap();
            let runner = circuit.runner();
            let traces = runner.run().unwrap();

            assert_eq!(
                traces.witness_trace.get_value(result.into()).unwrap(),
                &BabyBear::from_u64(120)
            );
        }

        // Test case 3: With zero element [5, 0, 7] = 0
        {
            let mut builder = CircuitBuilder::<BabyBear>::new();
            let with_zero = vec![
                builder.define_const(BabyBear::from_u64(5)),
                builder.define_const(BabyBear::ZERO),
                builder.define_const(BabyBear::from_u64(7)),
            ];
            let result = builder.mul_many(&with_zero);

            let circuit = builder.build().unwrap();
            let runner = circuit.runner();
            let traces = runner.run().unwrap();

            assert_eq!(
                traces.witness_trace.get_value(result.into()).unwrap(),
                &BabyBear::ZERO
            );
        }
    }

    #[test]
    fn test_inner_product() {
        // Test case 1: Basic dot product [1,2,3] · [4,5,6] = 32
        {
            let mut builder = CircuitBuilder::<BabyBear>::new();
            let a: Vec<ExprId> = vec![1, 2, 3]
                .into_iter()
                .map(|v| builder.define_const(BabyBear::from_u64(v)))
                .collect();
            let b: Vec<ExprId> = vec![4, 5, 6]
                .into_iter()
                .map(|v| builder.define_const(BabyBear::from_u64(v)))
                .collect();
            let result = builder.inner_product(&a, &b);

            let circuit = builder.build().unwrap();
            let runner = circuit.runner();
            let traces = runner.run().unwrap();

            assert_eq!(
                traces.witness_trace.get_value(result.into()).unwrap(),
                &BabyBear::from_u64(32)
            );
        }

        // Test case 2: Empty vectors [] · [] = 0
        {
            let mut builder = CircuitBuilder::<BabyBear>::new();
            let empty_a: Vec<ExprId> = vec![];
            let empty_b: Vec<ExprId> = vec![];
            let result = builder.inner_product(&empty_a, &empty_b);

            let circuit = builder.build().unwrap();
            let runner = circuit.runner();
            let traces = runner.run().unwrap();

            assert_eq!(
                traces.witness_trace.get_value(result.into()).unwrap(),
                &BabyBear::ZERO
            );
        }

        // Test case 3: Zero vector [0,0,0] · [5,6,7] = 0
        {
            let mut builder = CircuitBuilder::<BabyBear>::new();
            let zeros: Vec<ExprId> = (0..3)
                .map(|_| builder.define_const(BabyBear::ZERO))
                .collect();
            let vals: Vec<ExprId> = vec![5, 6, 7]
                .into_iter()
                .map(|v| builder.define_const(BabyBear::from_u64(v)))
                .collect();
            let result = builder.inner_product(&zeros, &vals);

            let circuit = builder.build().unwrap();
            let runner = circuit.runner();
            let traces = runner.run().unwrap();

            assert_eq!(
                traces.witness_trace.get_value(result.into()).unwrap(),
                &BabyBear::ZERO
            );
        }
    }

    #[test]
    #[should_panic]
    fn test_inner_product_mismatched_lengths() {
        // Verify that inner_product panics with mismatched vector lengths
        let mut builder = CircuitBuilder::<BabyBear>::new();

        // Create vectors with different lengths: [1,2] vs [3,4,5]
        let a: Vec<ExprId> = vec![1, 2]
            .into_iter()
            .map(|v| builder.define_const(BabyBear::from_u64(v)))
            .collect();
        let b: Vec<ExprId> = vec![3, 4, 5]
            .into_iter()
            .map(|v| builder.define_const(BabyBear::from_u64(v)))
            .collect();

        // Should panic: lengths don't match (2 != 3)
        builder.inner_product(&a, &b);
    }

    proptest! {
        #[test]
        fn prop_mul_add_correctness(
            a in field_element(),
            b in field_element(),
            c in field_element()
        ) {
            // Build circuit with mul_add
            let mut builder = CircuitBuilder::<BabyBear>::new();
            let ca = builder.define_const(a);
            let cb = builder.define_const(b);
            let cc = builder.define_const(c);
            let result = builder.mul_add(ca, cb, cc);

            // Execute circuit
            let circuit = builder.build().unwrap();
            let runner = circuit.runner();
            let traces = runner.run().unwrap();

            // Compute expected value
            let expected = a * b + c;

            // Verify correctness
            prop_assert_eq!(
                traces.witness_trace.get_value(result.into()).unwrap(),
                &expected
            );
        }

        #[test]
        fn prop_mul_many_correctness(
            values in prop::collection::vec(field_element(), 0..8)
        ) {
            // Build circuit with mul_many
            let mut builder = CircuitBuilder::<BabyBear>::new();
            let expr_ids: Vec<ExprId> = values
                .iter()
                .map(|&v| builder.define_const(v))
                .collect();
            let result = builder.mul_many(&expr_ids);

            // Execute circuit
            let circuit = builder.build().unwrap();
            let runner = circuit.runner();
            let traces = runner.run().unwrap();

            // Compute expected product (empty → 1, otherwise fold multiply)
            let expected = if values.is_empty() {
                BabyBear::ONE
            } else {
                values.iter().fold(BabyBear::ONE, |acc, &x| acc * x)
            };

            // Verify correctness
            prop_assert_eq!(
                traces.witness_trace.get_value(result.into()).unwrap(),
                &expected
            );
        }

        #[test]
        fn prop_inner_product_correctness(
            values in prop::collection::vec((field_element(), field_element()), 0..8)
        ) {
            // Extract equal-length vectors from paired values
            let vec1: Vec<BabyBear> = values.iter().map(|(a, _)| *a).collect();
            let vec2: Vec<BabyBear> = values.iter().map(|(_, b)| *b).collect();

            // Build circuit with inner_product
            let mut builder = CircuitBuilder::<BabyBear>::new();
            let a: Vec<ExprId> = vec1.iter().map(|&v| builder.define_const(v)).collect();
            let b: Vec<ExprId> = vec2.iter().map(|&v| builder.define_const(v)).collect();
            let result = builder.inner_product(&a, &b);

            // Execute circuit
            let circuit = builder.build().unwrap();
            let runner = circuit.runner();
            let traces = runner.run().unwrap();

            // Compute expected dot product: Σ(a_i * b_i)
            let expected = vec1
                .iter()
                .zip(vec2.iter())
                .fold(BabyBear::ZERO, |acc, (&x, &y)| acc + x * y);

            // Verify correctness
            prop_assert_eq!(
                traces.witness_trace.get_value(result.into()).unwrap(),
                &expected
            );
        }
    }

    #[test]
    fn test_reconstruct_index_from_bits() {
        let mut builder = CircuitBuilder::<BabyBear>::new();

        // Test reconstructing the value 5 (binary: 101)
        let bit0 = builder.define_const(BabyBear::ONE); // 1
        let bit1 = builder.define_const(BabyBear::ZERO); // 0
        let bit2 = builder.define_const(BabyBear::ONE); // 1

        let bits = vec![bit0, bit1, bit2];
        let result = builder.reconstruct_index_from_bits(&bits).unwrap();

        // Connect result to a public input so we can verify its value
        let output = builder.public_input();
        builder.connect(result, output);

        // Build and run the circuit
        let circuit = builder.build().expect("Failed to build circuit");
        let mut runner = circuit.runner();

        // Set public inputs: the expected result value 5
        let expected_result = BabyBear::from_u64(5); // 1*1 + 0*2 + 1*4 = 5
        runner
            .set_public_inputs(&[expected_result])
            .expect("Failed to set public inputs");

        let traces = runner.run().expect("Failed to run circuit");

        // Just verify the calculation is correct - reconstruct gives us 5
        assert_eq!(traces.public_trace.values[0], BabyBear::from_u64(5));
    }

    type Ext4 = BinomialExtensionField<BabyBear, 4>;

    #[test]
    fn test_reconstruct_index_from_bits_ext_field() {
        let mut builder = CircuitBuilder::<Ext4>::new();

        // Test reconstructing a value from an alternating 124-bit pattern (0xAAAA…)
        let bits: [_; 124] = array::from_fn(|i| builder.define_const(Ext4::from_usize(i % 2)));

        let result = builder
            .reconstruct_index_from_bits::<BabyBear>(&bits)
            .unwrap();

        // Connect result to a public input so we can verify its value
        let output = builder.public_input();
        builder.connect(result, output);

        // Build and run the circuit
        let circuit = builder.build().expect("Failed to build circuit");
        let mut runner = circuit.runner();

        // Set public inputs: compute the expected result
        let expected_result = (0..Ext4::bits() as u64)
            .chunks(BabyBear::bits())
            .into_iter()
            .enumerate()
            .map(|(chunk_idx, chunk)| {
                chunk
                    .into_iter()
                    .map(|i| {
                        let mut pow2 =
                            [BabyBear::ZERO; <Ext4 as BasedVectorSpace<BabyBear>>::DIMENSION];
                        pow2[chunk_idx] = BabyBear::TWO.exp_u64(i % BabyBear::bits() as u64);
                        let power = Ext4::from_basis_coefficients_slice(&pow2).unwrap();
                        Ext4::from_u8((i % 2) as u8) * power
                    })
                    .sum()
            })
            .sum();

        runner
            .set_public_inputs(&[expected_result])
            .expect("Failed to set public inputs");

        let traces = runner.run().expect("Failed to run circuit");

        // Just verify the calculation is correct - reconstruct gives us 5
        assert_eq!(traces.public_trace.values[0], expected_result);
    }

    #[test]
    fn test_decompose_to_bits() {
        let mut builder = CircuitBuilder::<BabyBear>::new();

        // Create a target representing the value we want to decompose
        let value = builder.define_const(BabyBear::from_u64(6)); // Binary: 110

        // Decompose into 3 bits - this creates its own public inputs for the bits
        let bits = builder.decompose_to_bits::<BabyBear>(value, 3).unwrap();

        // Build and run the circuit
        let circuit = builder.build().expect("Failed to build circuit");
        let expr_to_widx = circuit.expr_to_widx.clone();
        let runner = circuit.runner();
        let traces = runner.run().expect("Failed to run circuit");

        // Verify the bits are correctly decomposed - 6 = [0,1,1] in little-endian
        let bit_values: Vec<BabyBear> = bits
            .iter()
            .map(|b| {
                let w = expr_to_widx.get(b).expect("bit expr mapped");
                *traces.witness_trace.get_value(*w).unwrap()
            })
            .collect();
        assert_eq!(bit_values[0], BabyBear::ZERO); // bit 0
        assert_eq!(bit_values[1], BabyBear::ONE); // bit 1
        assert_eq!(bit_values[2], BabyBear::ONE); // bit 2

        assert_eq!(bits.len(), 3);
    }

    #[test]
    fn test_decompose_to_bits_ext_field() {
        let mut builder = CircuitBuilder::<Ext4>::new();

        // Create a target representing the value we want to decompose
        let value = builder.define_const(
            Ext4::from_basis_coefficients_slice(&[
                BabyBear::from_u32(0x40000006), // Binary: 01100000 00000000 00000000 00000001
                BabyBear::from_u32(0x55555555), // Binary: 10101010 10101010 10101010 10101010
                BabyBear::from_u32(0x02000000), // Binary: 00000000 00000000 00000000 01000000
                BabyBear::ZERO,                 // Binary: 00000000 00000000 00000000 00000000
            ])
            .unwrap(),
        );

        // Decompose into 3 bits - this creates its own public inputs for the bits
        let bits = builder
            .decompose_to_bits::<BabyBear>(value, Ext4::bits())
            .unwrap();

        // Build and run the circuit
        let circuit = builder.build().expect("Failed to build circuit");
        let expr_to_widx = circuit.expr_to_widx.clone();
        let runner = circuit.runner();
        let traces = runner.run().expect("Failed to run circuit");

        // Verify the bits are correctly decomposed
        // Expected first limb binary decompostion
        let hex_0x40000006_bin = [
            0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 1,
        ]
        .map(Ext4::from_u8);
        // Expected second limb binary decompostion
        let hex_0x55555555_bin = [
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
            0, 1,
        ]
        .map(Ext4::from_u8);
        // Expected third limb binary decompostion
        let hex_0x02000000_bin = [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
            0, 0,
        ]
        .map(Ext4::from_u8);
        // Expected fourth limb binary decompostion
        let zero_bin = [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0,
        ]
        .map(Ext4::from_u8);

        let bit_values: Vec<Ext4> = bits
            .iter()
            .map(|b| {
                let w = expr_to_widx.get(b).expect("bit expr mapped");
                *traces.witness_trace.get_value(*w).unwrap()
            })
            .collect();
        let result = bit_values.chunks(31).collect::<Vec<&[Ext4]>>();
        assert_eq!(result[0], hex_0x40000006_bin);
        assert_eq!(result[1], hex_0x55555555_bin);
        assert_eq!(result[2], hex_0x02000000_bin);
        assert_eq!(result[3], zero_bin);
        assert_eq!(bits.len(), Ext4::bits());
    }

    #[test]
    fn test_recompose_base_coeffs_to_ext() {
        type Ext4 = BinomialExtensionField<BabyBear, 4>;

        let mut builder = CircuitBuilder::<Ext4>::new();

        let c0 = builder.define_const(Ext4::from(BabyBear::from_u64(1)));
        let c1 = builder.define_const(Ext4::from(BabyBear::from_u64(2)));
        let c2 = builder.define_const(Ext4::from(BabyBear::from_u64(3)));
        let c3 = builder.define_const(Ext4::from(BabyBear::from_u64(4)));

        let coeffs = [c0, c1, c2, c3];
        let recomposed = builder
            .recompose_base_coeffs_to_ext::<BabyBear>(&coeffs)
            .unwrap();

        let circuit = builder.build().expect("Failed to build circuit");
        let expr_to_widx = circuit.expr_to_widx.clone();
        let runner = circuit.runner();
        let traces = runner.run().expect("Failed to run circuit");

        let w = expr_to_widx.get(&recomposed).expect("recomposed mapped");
        let result = *traces.witness_trace.get_value(*w).unwrap();

        let expected = Ext4::from_basis_coefficients_slice(&[
            BabyBear::from_u64(1),
            BabyBear::from_u64(2),
            BabyBear::from_u64(3),
            BabyBear::from_u64(4),
        ])
        .unwrap();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_decompose_ext_to_base_coeffs() {
        type Ext4 = BinomialExtensionField<BabyBear, 4>;

        let mut builder = CircuitBuilder::<Ext4>::new();

        let ext_val = Ext4::from_basis_coefficients_slice(&[
            BabyBear::from_u64(5),
            BabyBear::from_u64(6),
            BabyBear::from_u64(7),
            BabyBear::from_u64(8),
        ])
        .unwrap();
        let x = builder.define_const(ext_val);

        let coeffs = builder.decompose_ext_to_base_coeffs::<BabyBear>(x).unwrap();

        assert_eq!(coeffs.len(), 4);

        let circuit = builder.build().expect("Failed to build circuit");
        let expr_to_widx = circuit.expr_to_widx.clone();
        let runner = circuit.runner();
        let traces = runner.run().expect("Failed to run circuit");

        for (i, coeff_expr) in coeffs.iter().enumerate() {
            let w = expr_to_widx.get(coeff_expr).expect("coeff mapped");
            let coeff_val = *traces.witness_trace.get_value(*w).unwrap();

            let expected_coeffs: &[BabyBear] = coeff_val.as_basis_coefficients_slice();
            assert_eq!(
                expected_coeffs[0],
                BabyBear::from_u64(5 + i as u64),
                "coefficient {} mismatch",
                i
            );
            for (j, coeff) in expected_coeffs.iter().enumerate().skip(1) {
                assert_eq!(
                    *coeff,
                    BabyBear::ZERO,
                    "coefficient {} should have zero at position {}",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn test_decompose_recompose_round_trip() {
        type Ext4 = BinomialExtensionField<BabyBear, 4>;

        let mut builder = CircuitBuilder::<Ext4>::new();

        let original = Ext4::from_basis_coefficients_slice(&[
            BabyBear::from_u64(123),
            BabyBear::from_u64(456),
            BabyBear::from_u64(789),
            BabyBear::from_u64(101112),
        ])
        .unwrap();
        let x = builder.define_const(original);

        let coeffs = builder.decompose_ext_to_base_coeffs::<BabyBear>(x).unwrap();
        let recomposed = builder
            .recompose_base_coeffs_to_ext::<BabyBear>(&coeffs)
            .unwrap();

        let circuit = builder.build().expect("Failed to build circuit");
        let expr_to_widx = circuit.expr_to_widx.clone();
        let runner = circuit.runner();
        let traces = runner.run().expect("Failed to run circuit");

        let w_orig = expr_to_widx.get(&x).expect("original mapped");
        let w_recomp = expr_to_widx.get(&recomposed).expect("recomposed mapped");

        let val_orig = *traces.witness_trace.get_value(*w_orig).unwrap();
        let val_recomp = *traces.witness_trace.get_value(*w_recomp).unwrap();

        assert_eq!(val_orig, original);
        assert_eq!(val_recomp, original);
        assert_eq!(val_orig, val_recomp);
    }

    #[test]
    fn test_decompose_reuses_recompose_coeffs() {
        type Ext4 = BinomialExtensionField<BabyBear, 4>;

        let mut builder = CircuitBuilder::<Ext4>::new();

        let c0 = builder.define_const(Ext4::from(BabyBear::from_u64(1)));
        let c1 = builder.define_const(Ext4::from(BabyBear::from_u64(2)));
        let c2 = builder.define_const(Ext4::from(BabyBear::from_u64(3)));
        let c3 = builder.define_const(Ext4::from(BabyBear::from_u64(4)));
        let coeffs_in = [c0, c1, c2, c3];

        let ext = builder
            .recompose_base_coeffs_to_ext::<BabyBear>(&coeffs_in)
            .unwrap();
        let coeffs_out = builder
            .decompose_ext_to_base_coeffs::<BabyBear>(ext)
            .unwrap();

        assert_eq!(coeffs_out, coeffs_in);
    }

    #[test]
    fn test_decompose_select_with_recompose_provenance() {
        type Ext4 = BinomialExtensionField<BabyBear, 4>;

        fn check_case(selector: BabyBear, expect_t: bool) {
            let mut builder = CircuitBuilder::<Ext4>::new();

            let b = builder.define_const(Ext4::from(selector));
            builder.assert_bool(b);

            let t0 = builder.define_const(Ext4::from(BabyBear::from_u64(10)));
            let t1 = builder.define_const(Ext4::from(BabyBear::from_u64(11)));
            let t2 = builder.define_const(Ext4::from(BabyBear::from_u64(12)));
            let t3 = builder.define_const(Ext4::from(BabyBear::from_u64(13)));
            let coeffs_t = [t0, t1, t2, t3];
            let ext_t = builder
                .recompose_base_coeffs_to_ext::<BabyBear>(&coeffs_t)
                .unwrap();

            let s0 = builder.define_const(Ext4::from(BabyBear::from_u64(20)));
            let s1 = builder.define_const(Ext4::from(BabyBear::from_u64(21)));
            let s2 = builder.define_const(Ext4::from(BabyBear::from_u64(22)));
            let s3 = builder.define_const(Ext4::from(BabyBear::from_u64(23)));
            let coeffs_s = [s0, s1, s2, s3];
            let ext_s = builder
                .recompose_base_coeffs_to_ext::<BabyBear>(&coeffs_s)
                .unwrap();

            let selected = builder.select(b, ext_t, ext_s);
            let coeffs = builder
                .decompose_ext_to_base_coeffs::<BabyBear>(selected)
                .unwrap();

            let circuit = builder.build().expect("Failed to build circuit");
            let expr_to_widx = circuit.expr_to_widx.clone();
            let runner = circuit.runner();
            let traces = runner.run().expect("Failed to run circuit");

            let src = if expect_t { coeffs_t } else { coeffs_s };
            for (i, coeff_expr) in coeffs.iter().enumerate() {
                let w = expr_to_widx.get(coeff_expr).expect("coeff mapped");
                let v = *traces.witness_trace.get_value(*w).unwrap();
                let w_src = expr_to_widx.get(&src[i]).expect("src coeff mapped");
                let v_src = *traces.witness_trace.get_value(*w_src).unwrap();
                assert_eq!(v, v_src, "coeff index {i}");
            }
        }

        check_case(BabyBear::ZERO, false);
        check_case(BabyBear::ONE, true);
    }

    #[test]
    fn test_recompose_invalid_dimension() {
        type Ext4 = BinomialExtensionField<BabyBear, 4>;

        let mut builder = CircuitBuilder::<Ext4>::new();

        let c0 = builder.define_const(Ext4::ONE);
        let c1 = builder.define_const(Ext4::ONE);
        let c2 = builder.define_const(Ext4::ONE);

        let result = builder.recompose_base_coeffs_to_ext::<BabyBear>(&[c0, c1, c2]);

        assert!(result.is_err());
        match result {
            Err(CircuitBuilderError::InvalidDimension { expected, actual }) => {
                assert_eq!(expected, 4);
                assert_eq!(actual, 3);
            }
            _ => panic!("Expected InvalidDimension error"),
        }
    }

    #[test]
    fn test_bool_check_fusion() {
        let mut builder = CircuitBuilder::<BabyBear>::new();

        let b = builder.public_input();
        builder.assert_bool(b);

        let circuit = builder.build().unwrap();

        let mut runner = circuit.runner();
        runner.set_public_inputs(&[BabyBear::ZERO]).unwrap();
        let traces = runner.run().unwrap();
        assert!(
            !traces.alu_trace.values.is_empty(),
            "ALU trace should not be empty"
        );

        let mut builder2 = CircuitBuilder::<BabyBear>::new();
        let b2 = builder2.public_input();
        builder2.assert_bool(b2);
        let circuit2 = builder2.build().unwrap();
        let mut runner2 = circuit2.runner();
        runner2.set_public_inputs(&[BabyBear::ONE]).unwrap();
        let traces2 = runner2.run().unwrap();
        assert!(
            !traces2.alu_trace.values.is_empty(),
            "ALU trace should not be empty"
        );
    }
}
