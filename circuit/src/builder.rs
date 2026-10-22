//! Building a circuit while computing its witness.

use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;

use hashbrown::HashMap;
use p3_field::{ExtensionField, PrimeCharacteristicRing, PrimeField64};

use crate::expr::{BaseRepr, ExtRepr, Node};
use crate::{Base, Circuit, Ext, PermutationChip, Wire};

/// Whether a wire carries a base-field or a challenge-field value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Kind {
    /// A base-field value; its higher limbs are zero.
    Base,
    /// A challenge-field value.
    Ext,
}

/// One circuit operation, naming the wires it reads and writes.
#[derive(Clone, Debug)]
pub(crate) enum Op {
    /// Writes a constant.
    Const { out: Wire },
    /// Writes the public input at `index`.
    Public { out: Wire, index: usize },
    /// Writes a value the prover chooses.
    Witness { out: Wire },
    /// `out = a * b + c`, over the base field when `ext` is unset.
    ///
    /// The row reads `out` instead of writing it when `out_read` is set.
    Alu {
        ext: bool,
        a: Wire,
        b: Wire,
        c: Wire,
        out: Wire,
        out_read: bool,
    },
    /// The challenge-field wire whose basis coefficients are the limb wires.
    ///
    /// The row writes `ext` from the limbs when `compose` is set.
    ///
    /// Otherwise it writes the limbs from `ext`.
    Recompose {
        ext: Wire,
        limbs: Vec<Wire>,
        compose: bool,
    },
    /// The permutation of the input wires, written to the outputs that are used.
    Permute {
        inputs: Vec<Wire>,
        outputs: Vec<Option<Wire>>,
    },
}

/// A circuit under construction, with the value of every wire.
///
/// Values are lowered into table rows only when an operation needs them as wires.
#[derive(Debug)]
pub struct CircuitBuilder<F, EF, P> {
    /// The permutation the circuit proves in its own table.
    pub(crate) chip: P,
    /// The kind of every wire.
    pub(crate) kinds: Vec<Kind>,
    /// The value of every wire.
    pub(crate) values: Vec<EF>,
    /// Operations in creation order.
    pub(crate) ops: Vec<Op>,
    /// Public inputs, in index order.
    pub(crate) publics: Vec<F>,
    /// Constant wires, one per distinct value.
    constants: HashMap<EF, Wire>,
    /// Lowered base-field nodes.
    lowered_base: Lowered<Base<F>>,
    /// Lowered challenge-field nodes.
    lowered_ext: Lowered<Ext<F, EF>>,
}

/// Lowered nodes with their wires, keyed by address and kept alive so addresses stay unique.
type Lowered<V> = HashMap<usize, (Arc<Node<V>>, Wire)>;

/// A failed circuit check, found while the witness was being computed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Unsatisfied;

impl<F, EF, P> CircuitBuilder<F, EF, P>
where
    F: PrimeField64,
    EF: ExtensionField<F>,
    P: PermutationChip<F>,
{
    /// An empty circuit proving permutations with `chip`.
    pub fn new(chip: P) -> Self {
        Self {
            chip,
            kinds: Vec::new(),
            values: Vec::new(),
            ops: Vec::new(),
            publics: Vec::new(),
            constants: HashMap::new(),
            lowered_base: HashMap::new(),
            lowered_ext: HashMap::new(),
        }
    }

    /// The permutation the circuit proves.
    pub const fn chip(&self) -> &P {
        &self.chip
    }

    fn new_wire(&mut self, kind: Kind, value: EF) -> Wire {
        // A wire's id is a field element on the bus, so two wires must never share one.
        let index = self.values.len();
        assert!((index as u64) < F::ORDER_U64, "fewer wires than field elements");
        let wire = Wire(u32::try_from(index).expect("fewer than 2^32 wires"));
        self.kinds.push(kind);
        self.values.push(value);
        wire
    }

    /// A public input of the circuit.
    pub fn public(&mut self, value: F) -> Base<F> {
        let out = self.new_wire(Kind::Base, EF::from(value));
        self.ops.push(Op::Public {
            out,
            index: self.publics.len(),
        });
        self.publics.push(value);
        Base::wire(out)
    }

    /// A base-field value the prover chooses.
    pub fn witness(&mut self, value: F) -> Base<F> {
        let out = self.new_wire(Kind::Base, EF::from(value));
        self.ops.push(Op::Witness { out });
        Base::wire(out)
    }

    /// A challenge-field value the prover chooses.
    pub fn witness_ext(&mut self, value: EF) -> Ext<F, EF> {
        let out = self.new_wire(Kind::Ext, value);
        self.ops.push(Op::Witness { out });
        Ext::wire(out)
    }

    fn constant_wire(&mut self, value: EF, kind: Kind) -> Wire {
        if let Some(&wire) = self.constants.get(&value) {
            return wire;
        }
        let out = self.new_wire(kind, value);
        self.ops.push(Op::Const { out });
        self.constants.insert(value, out);
        out
    }

    fn alu(&mut self, ext: bool, a: Wire, b: Wire, c: Wire, out: Option<Wire>) -> Wire {
        let value = self.values[a.index()] * self.values[b.index()] + self.values[c.index()];
        let kind = if ext { Kind::Ext } else { Kind::Base };
        let (out, out_read) =
            out.map_or_else(|| (self.new_wire(kind, value), false), |out| (out, true));
        self.ops.push(Op::Alu {
            ext,
            a,
            b,
            c,
            out,
            out_read,
        });
        out
    }

    /// The wire of a base-field value, lowering its graph into rows once.
    pub fn base_wire(&mut self, value: &Base<F>) -> Wire {
        match &value.0 {
            BaseRepr::Const(c) => self.constant_wire(EF::from(*c), Kind::Base),
            BaseRepr::Node(node) => self.lower_base_node(node, None),
        }
    }

    /// The wire of a challenge-field value, lowering its graph into rows once.
    pub fn ext_wire(&mut self, value: &Ext<F, EF>) -> Wire {
        match &value.0 {
            ExtRepr::Const(c) => self.constant_wire(*c, Kind::Ext),
            ExtRepr::Base(base) => self.base_wire(base),
            ExtRepr::Node(node) => self.lower_ext_node(node, None),
        }
    }

    // A graph can be deep, so lowering keeps its own stack instead of recursing.
    fn lower_base_node(&mut self, root: &Arc<Node<Base<F>>>, out: Option<Wire>) -> Wire {
        let key = |node: &Arc<Node<Base<F>>>| Arc::as_ptr(node) as usize;
        let mut stack = vec![(Arc::clone(root), false)];
        while let Some((node, ready)) = stack.pop() {
            if self.lowered_base.contains_key(&key(&node)) {
                continue;
            }
            let children: Vec<&Base<F>> = match &*node {
                Node::Wire(_) => Vec::new(),
                Node::Add(a, b) | Node::Sub(a, b) | Node::Mul(a, b) => vec![a, b],
            };
            let pending: Vec<_> = children
                .iter()
                .filter_map(|child| match &child.0 {
                    BaseRepr::Node(n) if !self.lowered_base.contains_key(&key(n)) => {
                        Some(Arc::clone(n))
                    }
                    _ => None,
                })
                .collect();
            if !ready && !pending.is_empty() {
                stack.push((node, true));
                stack.extend(pending.into_iter().map(|n| (n, false)));
                continue;
            }
            // Only the root may write into a wire the caller names.
            let target = if Arc::ptr_eq(&node, root) { out } else { None };
            let wire = match &*node {
                Node::Wire(wire) => {
                    if let Some(target) = target {
                        let (zero, one) = (self.zero(), self.one());
                        self.alu(false, *wire, one, zero, Some(target));
                    }
                    *wire
                }
                Node::Add(a, b) => {
                    let (a, b) = (self.base_wire(a), self.base_wire(b));
                    let one = self.one();
                    self.alu(false, a, one, b, target)
                }
                Node::Sub(a, b) => {
                    let (a, b) = (self.base_wire(a), self.base_wire(b));
                    let neg_one = self.constant_wire(EF::NEG_ONE, Kind::Base);
                    self.alu(false, b, neg_one, a, target)
                }
                Node::Mul(a, b) => {
                    let (a, b) = (self.base_wire(a), self.base_wire(b));
                    let zero = self.zero();
                    self.alu(false, a, b, zero, target)
                }
            };
            self.lowered_base.insert(key(&node), (node, wire));
        }
        self.lowered_base[&key(root)].1
    }

    fn lower_ext_node(&mut self, root: &Arc<Node<Ext<F, EF>>>, out: Option<Wire>) -> Wire {
        let key = |node: &Arc<Node<Ext<F, EF>>>| Arc::as_ptr(node) as usize;
        let mut stack = vec![(Arc::clone(root), false)];
        while let Some((node, ready)) = stack.pop() {
            if self.lowered_ext.contains_key(&key(&node)) {
                continue;
            }
            let children: Vec<&Ext<F, EF>> = match &*node {
                Node::Wire(_) => Vec::new(),
                Node::Add(a, b) | Node::Sub(a, b) | Node::Mul(a, b) => vec![a, b],
            };
            let pending: Vec<_> = children
                .iter()
                .filter_map(|child| match &child.0 {
                    ExtRepr::Node(n) if !self.lowered_ext.contains_key(&key(n)) => {
                        Some(Arc::clone(n))
                    }
                    _ => None,
                })
                .collect();
            if !ready && !pending.is_empty() {
                stack.push((node, true));
                stack.extend(pending.into_iter().map(|n| (n, false)));
                continue;
            }
            let target = if Arc::ptr_eq(&node, root) { out } else { None };
            let wire = match &*node {
                Node::Wire(wire) => {
                    if let Some(target) = target {
                        let (zero, one) = (self.zero(), self.one());
                        self.alu(true, *wire, one, zero, Some(target));
                    }
                    *wire
                }
                Node::Add(a, b) => {
                    let (a, b) = (self.ext_wire(a), self.ext_wire(b));
                    let one = self.one();
                    self.alu(true, a, one, b, target)
                }
                Node::Sub(a, b) => {
                    let (a, b) = (self.ext_wire(a), self.ext_wire(b));
                    let neg_one = self.constant_wire(EF::NEG_ONE, Kind::Base);
                    self.alu(true, b, neg_one, a, target)
                }
                Node::Mul(a, b) => {
                    let (a, b) = (self.ext_wire(a), self.ext_wire(b));
                    let zero = self.zero();
                    self.alu(true, a, b, zero, target)
                }
            };
            self.lowered_ext.insert(key(&node), (node, wire));
        }
        self.lowered_ext[&key(root)].1
    }

    fn zero(&mut self) -> Wire {
        self.constant_wire(EF::ZERO, Kind::Base)
    }

    fn one(&mut self) -> Wire {
        self.constant_wire(EF::ONE, Kind::Base)
    }

    /// The value of a base-field value.
    pub fn value(&self, value: &Base<F>) -> F {
        match &value.0 {
            BaseRepr::Const(c) => *c,
            BaseRepr::Node(node) => self.eval_base(node),
        }
    }

    /// The value of a challenge-field value.
    pub fn value_ext(&self, value: &Ext<F, EF>) -> EF {
        match &value.0 {
            ExtRepr::Const(c) => *c,
            ExtRepr::Base(base) => EF::from(self.value(base)),
            ExtRepr::Node(node) => self.eval_ext(node),
        }
    }

    // A lowered node already carries its value on its wire.
    fn eval_base(&self, root: &Arc<Node<Base<F>>>) -> F {
        let mut memo: HashMap<usize, F> = HashMap::new();
        let mut stack = vec![(Arc::clone(root), false)];
        let key = |node: &Arc<Node<Base<F>>>| Arc::as_ptr(node) as usize;
        while let Some((node, ready)) = stack.pop() {
            if memo.contains_key(&key(&node)) {
                continue;
            }
            if let Some((_, wire)) = self.lowered_base.get(&key(&node)) {
                memo.insert(key(&node), self.base_value_of(*wire));
                continue;
            }
            let operand = |memo: &HashMap<usize, F>, v: &Base<F>| match &v.0 {
                BaseRepr::Const(c) => Some(*c),
                BaseRepr::Node(n) => memo.get(&key(n)).copied(),
            };
            let value = match &*node {
                Node::Wire(wire) => Some(self.base_value_of(*wire)),
                Node::Add(a, b) | Node::Sub(a, b) | Node::Mul(a, b) => {
                    match (operand(&memo, a), operand(&memo, b)) {
                        (Some(x), Some(y)) => Some(match &*node {
                            Node::Add(..) => x + y,
                            Node::Sub(..) => x - y,
                            _ => x * y,
                        }),
                        _ => None,
                    }
                }
            };
            if let Some(value) = value {
                memo.insert(key(&node), value);
            } else {
                assert!(!ready, "children are evaluated before their parent");
                stack.push((Arc::clone(&node), true));
                if let Node::Add(a, b) | Node::Sub(a, b) | Node::Mul(a, b) = &*node {
                    for child in [a, b] {
                        if let BaseRepr::Node(n) = &child.0 {
                            stack.push((Arc::clone(n), false));
                        }
                    }
                }
            }
        }
        memo[&key(root)]
    }

    fn eval_ext(&self, root: &Arc<Node<Ext<F, EF>>>) -> EF {
        let mut memo: HashMap<usize, EF> = HashMap::new();
        let mut stack = vec![(Arc::clone(root), false)];
        let key = |node: &Arc<Node<Ext<F, EF>>>| Arc::as_ptr(node) as usize;
        while let Some((node, ready)) = stack.pop() {
            if memo.contains_key(&key(&node)) {
                continue;
            }
            if let Some((_, wire)) = self.lowered_ext.get(&key(&node)) {
                memo.insert(key(&node), self.values[wire.index()]);
                continue;
            }
            let operand = |memo: &HashMap<usize, EF>, v: &Ext<F, EF>| match &v.0 {
                ExtRepr::Const(c) => Some(*c),
                ExtRepr::Base(base) => Some(EF::from(self.value(base))),
                ExtRepr::Node(n) => memo.get(&key(n)).copied(),
            };
            let value = match &*node {
                Node::Wire(wire) => Some(self.values[wire.index()]),
                Node::Add(a, b) | Node::Sub(a, b) | Node::Mul(a, b) => {
                    match (operand(&memo, a), operand(&memo, b)) {
                        (Some(x), Some(y)) => Some(match &*node {
                            Node::Add(..) => x + y,
                            Node::Sub(..) => x - y,
                            _ => x * y,
                        }),
                        _ => None,
                    }
                }
            };
            if let Some(value) = value {
                memo.insert(key(&node), value);
            } else {
                assert!(!ready, "children are evaluated before their parent");
                stack.push((Arc::clone(&node), true));
                if let Node::Add(a, b) | Node::Sub(a, b) | Node::Mul(a, b) = &*node {
                    for child in [a, b] {
                        if let ExtRepr::Node(n) = &child.0 {
                            stack.push((Arc::clone(n), false));
                        }
                    }
                }
            }
        }
        memo[&key(root)]
    }

    fn base_value_of(&self, wire: Wire) -> F {
        self.values[wire.index()].as_basis_coefficients_slice()[0]
    }

    /// Require a base-field value to be zero.
    ///
    /// # Errors
    ///
    /// When the value is not zero, so no witness satisfies the circuit.
    pub fn assert_zero(&mut self, value: &Base<F>) -> Result<(), Unsatisfied> {
        // The rows are emitted either way, so the circuit never depends on a value.
        let satisfied = self.value(value).is_zero();
        let zero = self.zero();
        match &value.0 {
            BaseRepr::Const(_) => {}
            BaseRepr::Node(node) => {
                // A node lowered elsewhere already has a wire; its zero check is one more row.
                if self
                    .lowered_base
                    .contains_key(&(Arc::as_ptr(node) as usize))
                {
                    let (wire, one) = (self.base_wire(value), self.one());
                    self.alu(false, wire, one, zero, Some(zero));
                } else {
                    self.lower_base_node(node, Some(zero));
                }
            }
        }
        if satisfied { Ok(()) } else { Err(Unsatisfied) }
    }

    /// Require a challenge-field value to be zero.
    ///
    /// # Errors
    ///
    /// When the value is not zero, so no witness satisfies the circuit.
    pub fn assert_zero_ext(&mut self, value: &Ext<F, EF>) -> Result<(), Unsatisfied> {
        let satisfied = self.value_ext(value).is_zero();
        let zero = self.zero();
        match &value.0 {
            ExtRepr::Const(_) => {}
            ExtRepr::Base(base) => return self.assert_zero(base),
            ExtRepr::Node(node) => {
                if self.lowered_ext.contains_key(&(Arc::as_ptr(node) as usize)) {
                    let (wire, one) = (self.ext_wire(value), self.one());
                    self.alu(true, wire, one, zero, Some(zero));
                } else {
                    self.lower_ext_node(node, Some(zero));
                }
            }
        }
        if satisfied { Ok(()) } else { Err(Unsatisfied) }
    }

    /// Require a base-field value to be zero or one.
    pub fn assert_bool(&mut self, value: &Base<F>) {
        // A bit is its own square, so one row `x = x * x + 0` checks it.
        let wire = self.base_wire(value);
        let zero = self.zero();
        self.alu(false, wire, wire, zero, Some(wire));
    }

    /// The inverse of a challenge-field value, supplied by the prover and checked by one row.
    ///
    /// # Errors
    ///
    /// When the value is zero.
    pub fn inverse_ext(&mut self, value: &Ext<F, EF>) -> Result<Ext<F, EF>, Unsatisfied> {
        let inverse = self.value_ext(value).try_inverse().ok_or(Unsatisfied)?;
        let hint = self.witness_ext(inverse);
        let (a, b) = (self.ext_wire(value), self.ext_wire(&hint));
        let (zero, one) = (self.zero(), self.one());
        self.alu(true, a, b, zero, Some(one));
        Ok(hint)
    }

    /// The canonical bits of a base-field value, least significant first.
    ///
    /// Every bit is checked boolean and the bits recompose the value.
    ///
    /// Their integer is also checked to lie below the modulus.
    pub fn canonical_bits(&mut self, value: &Base<F>) -> Vec<Base<F>> {
        let canonical = self.value(value).as_canonical_u64();
        let n = 64 - (F::ORDER_U64 - 1).leading_zeros() as usize;
        let bits: Vec<Base<F>> = (0..n)
            .map(|i| self.witness(F::from_bool((canonical >> i) & 1 == 1)))
            .collect();
        for bit in &bits {
            self.assert_bool(bit);
        }

        // Horner from the top bit: acc = 2 * acc + bit, ending on the value itself.
        let target = self.base_wire(value);
        let two = self.constant_wire(EF::TWO, Kind::Base);
        let mut acc = self.base_wire(&bits[n - 1]);
        for (i, bit) in bits.iter().enumerate().rev().skip(1) {
            let bit = self.base_wire(bit);
            acc = self.alu(false, acc, two, bit, (i == 0).then_some(target));
        }
        if n == 1 {
            let (zero, one) = (self.zero(), self.one());
            self.alu(false, acc, one, zero, Some(target));
        }

        // The modulus is `2^n - 2^k + 1` for every two-adic prime used here.
        //
        // So `p - 1` in binary is `n - k` ones followed by `k` zeros.
        //
        // An integer above `p - 1` therefore has all its top `n - k` bits set and a low bit set.
        let p_minus_one = F::ORDER_U64 - 1;
        let k = p_minus_one.trailing_zeros() as usize;
        assert_eq!(
            p_minus_one,
            (u64::MAX >> (64 - n)) & !((1u64 << k) - 1),
            "the modulus has the form 2^n - 2^k + 1"
        );
        let top: Base<F> = bits[k..].iter().cloned().product();
        let low: Base<F> = bits[..k].iter().cloned().sum();
        self.assert_zero(&(top * low))
            .expect("a canonical decomposition passes its own range check");
        bits
    }

    /// The challenge-field value with the given basis coefficients.
    pub fn compose(&mut self, limbs: &[Base<F>]) -> Ext<F, EF> {
        assert_eq!(limbs.len(), EF::DIMENSION);
        let values: Vec<F> = limbs.iter().map(|limb| self.value(limb)).collect();
        let limbs: Vec<Wire> = limbs.iter().map(|limb| self.base_wire(limb)).collect();
        let value = EF::from_basis_coefficients_slice(&values).expect("one limb per coefficient");
        let ext = self.new_wire(Kind::Ext, value);
        self.ops.push(Op::Recompose {
            ext,
            limbs,
            compose: true,
        });
        Ext::wire(ext)
    }

    /// The basis coefficients of a challenge-field value.
    pub fn decompose(&mut self, value: &Ext<F, EF>) -> Vec<Base<F>> {
        if let ExtRepr::Base(base) = &value.0 {
            let mut limbs = vec![Base::ZERO; EF::DIMENSION];
            limbs[0] = base.clone();
            return limbs;
        }
        if let Some(constant) = value.as_const() {
            return constant
                .as_basis_coefficients_slice()
                .iter()
                .map(|&c| Base::from(c))
                .collect();
        }
        let ext = self.ext_wire(value);
        let coefficients = self.values[ext.index()];
        let limbs: Vec<Wire> = coefficients
            .as_basis_coefficients_slice()
            .iter()
            .map(|&c| self.new_wire(Kind::Base, EF::from(c)))
            .collect();
        self.ops.push(Op::Recompose {
            ext,
            limbs: limbs.clone(),
            compose: false,
        });
        limbs.into_iter().map(Base::wire).collect()
    }

    /// The permutation of a state, with every output available as a value.
    ///
    /// Outputs the circuit never reads cost nothing beyond their trace cells.
    pub fn permute(&mut self, state: &[Base<F>]) -> Vec<Base<F>> {
        assert_eq!(state.len(), P::WIDTH);
        let values: Vec<F> = state.iter().map(|x| self.value(x)).collect();
        let inputs: Vec<Wire> = state.iter().map(|x| self.base_wire(x)).collect();
        let outputs: Vec<Wire> = self
            .chip
            .permute(&values)
            .into_iter()
            .map(|value| self.new_wire(Kind::Base, EF::from(value)))
            .collect();
        self.ops.push(Op::Permute {
            inputs,
            outputs: outputs.iter().copied().map(Some).collect(),
        });
        outputs.into_iter().map(Base::wire).collect()
    }

    /// The finished circuit, together with the witness of this run.
    pub fn build(self) -> Circuit<F, EF, P> {
        Circuit::new(self)
    }
}
