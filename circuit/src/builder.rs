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
pub(crate) enum Op<F> {
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
    /// One Merkle compression of `digest` with a sibling only this row holds.
    ///
    /// `digest` is the right child when `bit` is one, and the left child otherwise.
    MerkleStep {
        digest: Vec<Wire>,
        sibling: Vec<F>,
        bit: Wire,
        outputs: Vec<Wire>,
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
    pub(crate) ops: Vec<Op<F>>,
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

/// The address that identifies a node while it is alive.
fn key<V>(node: &Arc<Node<V>>) -> usize {
    Arc::as_ptr(node) as usize
}

/// A value kind the builder lowers into rows of its ALU table.
pub(crate) trait Lowerable<F, EF, P>: Sized {
    /// Whether the rows belong to the challenge-field table.
    const EXT: bool;

    /// The graph node behind the value, unless it is a constant or a lifted base value.
    fn node(&self) -> Option<&Arc<Node<Self>>>;

    /// The wire of the value.
    fn wire(&self, builder: &mut CircuitBuilder<F, EF, P>) -> Wire;

    /// The value, in the challenge field.
    fn value(&self, builder: &CircuitBuilder<F, EF, P>) -> EF;

    /// The nodes of this kind that have a wire.
    fn lowered(builder: &CircuitBuilder<F, EF, P>) -> &Lowered<Self>;

    /// The nodes of this kind that have a wire, for insertion.
    fn lowered_mut(builder: &mut CircuitBuilder<F, EF, P>) -> &mut Lowered<Self>;
}

impl<F: PrimeField64, EF: ExtensionField<F>, P: PermutationChip<F>> Lowerable<F, EF, P>
    for Base<F>
{
    const EXT: bool = false;

    fn node(&self) -> Option<&Arc<Node<Self>>> {
        match &self.0 {
            BaseRepr::Node(node) => Some(node),
            BaseRepr::Const(_) => None,
        }
    }

    fn wire(&self, builder: &mut CircuitBuilder<F, EF, P>) -> Wire {
        builder.base_wire(self)
    }

    fn value(&self, builder: &CircuitBuilder<F, EF, P>) -> EF {
        EF::from(builder.value(self))
    }

    fn lowered(builder: &CircuitBuilder<F, EF, P>) -> &Lowered<Self> {
        &builder.lowered_base
    }

    fn lowered_mut(builder: &mut CircuitBuilder<F, EF, P>) -> &mut Lowered<Self> {
        &mut builder.lowered_base
    }
}

impl<F: PrimeField64, EF: ExtensionField<F>, P: PermutationChip<F>> Lowerable<F, EF, P>
    for Ext<F, EF>
{
    const EXT: bool = true;

    fn node(&self) -> Option<&Arc<Node<Self>>> {
        match &self.0 {
            ExtRepr::Node(node) => Some(node),
            ExtRepr::Const(_) | ExtRepr::Base(_) => None,
        }
    }

    fn wire(&self, builder: &mut CircuitBuilder<F, EF, P>) -> Wire {
        builder.ext_wire(self)
    }

    fn value(&self, builder: &CircuitBuilder<F, EF, P>) -> EF {
        builder.value_ext(self)
    }

    fn lowered(builder: &CircuitBuilder<F, EF, P>) -> &Lowered<Self> {
        &builder.lowered_ext
    }

    fn lowered_mut(builder: &mut CircuitBuilder<F, EF, P>) -> &mut Lowered<Self> {
        &mut builder.lowered_ext
    }
}

/// The ALU rows that lower one node.
enum Row<'a, V> {
    /// The node is a wire already.
    Wire(Wire),
    /// `a * b`.
    Mul(&'a V, &'a V),
    /// `a - b`.
    Sub(&'a V, &'a V),
    /// A sum, flattened through every inner sum and product that nothing else holds.
    ///
    /// Each product costs one row, and each plain term after the first costs one row.
    Sum {
        /// The plain terms.
        plain: Vec<&'a V>,
        /// The products, as their two factors.
        products: Vec<(&'a V, &'a V)>,
    },
}

impl<'a, V> Row<'a, V> {
    /// The rows of a node.
    fn of<F, EF, P>(node: &'a Node<V>) -> Self
    where
        V: Lowerable<F, EF, P>,
    {
        // A node held by nothing but its parent has no other reader, so it needs no wire.
        let sole = |v: &'a V| v.node().filter(|n| Arc::strong_count(n) == 1).map(|n| &**n);
        match node {
            Node::Wire(wire) => Self::Wire(*wire),
            Node::Mul(a, b) => Self::Mul(a, b),
            Node::Sub(a, b) => Self::Sub(a, b),
            Node::Add(x, y) => {
                let (mut plain, mut products) = (Vec::new(), Vec::new());
                let mut terms = vec![y, x];
                while let Some(term) = terms.pop() {
                    match sole(term) {
                        Some(Node::Add(a, b)) => terms.extend([b, a]),
                        Some(Node::Mul(a, b)) => products.push((a, b)),
                        _ => plain.push(term),
                    }
                }
                Self::Sum { plain, products }
            }
        }
    }

    /// The values the rows read.
    fn operands(&self) -> Vec<&'a V> {
        match self {
            Self::Wire(_) => Vec::new(),
            Self::Mul(a, b) | Self::Sub(a, b) => vec![a, b],
            Self::Sum { plain, products } => plain
                .iter()
                .copied()
                .chain(products.iter().flat_map(|&(a, b)| [a, b]))
                .collect(),
        }
    }
}

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
            BaseRepr::Node(node) => self.lower(node, None),
        }
    }

    /// The wire of a challenge-field value, lowering its graph into rows once.
    pub fn ext_wire(&mut self, value: &Ext<F, EF>) -> Wire {
        match &value.0 {
            ExtRepr::Const(c) => self.constant_wire(*c, Kind::Ext),
            ExtRepr::Base(base) => self.base_wire(base),
            ExtRepr::Node(node) => self.lower(node, None),
        }
    }

    // A graph can be deep, so lowering keeps its own stack instead of recursing.
    fn lower<V: Lowerable<F, EF, P>>(&mut self, root: &Arc<Node<V>>, out: Option<Wire>) -> Wire {
        let mut stack = vec![(Arc::clone(root), false)];
        while let Some((node, ready)) = stack.pop() {
            if V::lowered(self).contains_key(&key(&node)) {
                continue;
            }
            if !ready {
                let pending: Vec<_> = Row::of(&node)
                    .operands()
                    .into_iter()
                    .filter_map(V::node)
                    .filter(|n| !V::lowered(self).contains_key(&key(n)))
                    .cloned()
                    .collect();
                if !pending.is_empty() {
                    stack.push((node, true));
                    stack.extend(pending.into_iter().map(|n| (n, false)));
                    continue;
                }
            }
            let op = Row::of(&node);
            // Only the root may write into a wire the caller names.
            let target = if Arc::ptr_eq(&node, root) { out } else { None };
            let wire = match op {
                Row::Wire(wire) => {
                    if let Some(target) = target {
                        let (zero, one) = (self.zero(), self.one());
                        self.alu(V::EXT, wire, one, zero, Some(target));
                    }
                    wire
                }
                Row::Mul(a, b) => {
                    let (a, b, zero) = (a.wire(self), b.wire(self), self.zero());
                    self.alu(V::EXT, a, b, zero, target)
                }
                Row::Sum { plain, products } => {
                    // The first plain term seeds the sum, and every other term adds one row.
                    let one = self.one();
                    let mut plain = plain.into_iter().map(|term| term.wire(self));
                    let mut sum = plain.next();
                    let mut steps: Vec<(Wire, Wire)> = plain.map(|term| (term, one)).collect();
                    for (a, b) in products {
                        steps.push((a.wire(self), b.wire(self)));
                    }
                    let last = steps.len().checked_sub(1);
                    for (i, (a, b)) in steps.into_iter().enumerate() {
                        let c = sum.unwrap_or_else(|| self.zero());
                        let out = if Some(i) == last { target } else { None };
                        sum = Some(self.alu(V::EXT, a, b, c, out));
                    }
                    sum.expect("a sum has two terms")
                }
                Row::Sub(a, b) => {
                    let (a, b) = (a.wire(self), b.wire(self));
                    let neg_one = self.constant_wire(EF::NEG_ONE, Kind::Base);
                    self.alu(V::EXT, b, neg_one, a, target)
                }
            };
            V::lowered_mut(self).insert(key(&node), (node, wire));
        }
        V::lowered(self)[&key(root)].1
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
            BaseRepr::Node(node) => self.eval(node).as_basis_coefficients_slice()[0],
        }
    }

    /// The value of a challenge-field value.
    pub fn value_ext(&self, value: &Ext<F, EF>) -> EF {
        match &value.0 {
            ExtRepr::Const(c) => *c,
            ExtRepr::Base(base) => EF::from(self.value(base)),
            ExtRepr::Node(node) => self.eval(node),
        }
    }

    // A lowered node already carries its value on its wire.
    fn eval<V: Lowerable<F, EF, P>>(&self, root: &Arc<Node<V>>) -> EF {
        let mut memo: HashMap<usize, EF> = HashMap::new();
        let mut stack = vec![(Arc::clone(root), false)];
        while let Some((node, ready)) = stack.pop() {
            if memo.contains_key(&key(&node)) {
                continue;
            }
            if let Some((_, wire)) = V::lowered(self).get(&key(&node)) {
                memo.insert(key(&node), self.values[wire.index()]);
                continue;
            }
            let operand = |memo: &HashMap<usize, EF>, v: &V| {
                v.node()
                    .map_or_else(|| Some(v.value(self)), |n| memo.get(&key(n)).copied())
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
                    stack.extend(
                        [a, b]
                            .into_iter()
                            .filter_map(V::node)
                            .cloned()
                            .map(|n| (n, false)),
                    );
                }
            }
        }
        memo[&key(root)]
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
                    self.lower(node, Some(zero));
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
                    self.lower(node, Some(zero));
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

    /// The parent digest of `digest` and `sibling`, compressed by the permutation.
    ///
    /// `digest` is the right child when `bit` is one, and the left child otherwise.
    ///
    /// The sibling is a prover-chosen value that only this row reads, so it costs no other row.
    ///
    /// # Panics
    ///
    /// When `bit` is not boolean, or the children do not fit in the state.
    pub fn merkle_step(
        &mut self,
        digest: &[Base<F>],
        sibling: &[F],
        bit: &Base<F>,
    ) -> Vec<Base<F>> {
        let size = digest.len();
        assert_eq!(sibling.len(), size, "both children have the digest size");
        assert!(2 * size <= P::WIDTH, "both children fit in the state");
        let digest_values: Vec<F> = digest.iter().map(|x| self.value(x)).collect();
        let right = match self.value(bit) {
            b if b == F::ZERO => false,
            b if b == F::ONE => true,
            _ => panic!("a direction bit is boolean"),
        };

        // The state is `left || right`, padded with zeros, as a truncated permutation lays it out.
        let (left, right) = if right {
            (sibling, &digest_values[..])
        } else {
            (&digest_values[..], sibling)
        };
        let mut state = vec![F::ZERO; P::WIDTH];
        state[..size].copy_from_slice(left);
        state[size..2 * size].copy_from_slice(right);

        let digest: Vec<Wire> = digest.iter().map(|x| self.base_wire(x)).collect();
        let bit = self.base_wire(bit);
        let outputs: Vec<Wire> = self.chip.permute(&state)[..size]
            .iter()
            .map(|&value| self.new_wire(Kind::Base, EF::from(value)))
            .collect();
        self.ops.push(Op::MerkleStep {
            digest,
            sibling: sibling.to_vec(),
            bit,
            outputs: outputs.clone(),
        });
        outputs.into_iter().map(Base::wire).collect()
    }

    /// The finished circuit, together with the witness of this run.
    pub fn build(self) -> Circuit<F, EF, P> {
        Circuit::new(self)
    }
}

#[cfg(test)]
mod tests {
    use p3_field::extension::BinomialExtensionField;
    use p3_koala_bear::{
        GenericPoseidon2LinearLayersKoalaBear, KOALABEAR_POSEIDON2_RC_16_EXTERNAL_FINAL,
        KOALABEAR_POSEIDON2_RC_16_EXTERNAL_INITIAL, KOALABEAR_POSEIDON2_RC_16_INTERNAL, KoalaBear,
        Poseidon2KoalaBear, default_koalabear_poseidon2_16,
    };
    use p3_poseidon2_air::RoundConstants;

    use super::*;
    use crate::Poseidon2Chip;

    type F = KoalaBear;
    type Chip = Poseidon2Chip<
        F,
        GenericPoseidon2LinearLayersKoalaBear,
        Poseidon2KoalaBear<16>,
        16,
        3,
        0,
        4,
        20,
    >;

    fn builder() -> CircuitBuilder<F, BinomialExtensionField<F, 4>, Chip> {
        let constants = RoundConstants::new(
            KOALABEAR_POSEIDON2_RC_16_EXTERNAL_INITIAL,
            KOALABEAR_POSEIDON2_RC_16_INTERNAL,
            KOALABEAR_POSEIDON2_RC_16_EXTERNAL_FINAL,
        );
        CircuitBuilder::new(Chip::new(default_koalabear_poseidon2_16(), constants))
    }

    fn alu_rows<P>(builder: &CircuitBuilder<F, BinomialExtensionField<F, 4>, P>) -> usize {
        builder
            .ops
            .iter()
            .filter(|op| matches!(op, Op::Alu { .. }))
            .count()
    }

    #[test]
    fn a_sum_of_products_takes_one_row_per_product() {
        let mut builder = builder();
        let values: Vec<Base<F>> = (1..=17).map(|i| builder.witness(F::new(i))).collect();

        // 1 + sum_i a_i * b_i over eight pairs, then one row asserting the known total.
        let sum = values[1..].chunks(2).fold(values[0].clone(), |acc, pair| {
            acc + pair[0].clone() * pair[1].clone()
        });
        let total = 1 + (0..8).map(|i| (2 * i + 2) * (2 * i + 3)).sum::<u32>();
        builder.assert_zero(&(sum - F::new(total))).unwrap();
        assert_eq!(alu_rows(&builder), 8 + 1);
    }

    #[test]
    fn a_shared_product_keeps_its_own_row() {
        let mut builder = builder();
        let (a, b, c) = (
            builder.witness(F::new(2)),
            builder.witness(F::new(3)),
            builder.witness(F::new(4)),
        );

        // The product is read twice, so it gets a wire, and each sum adds one row.
        let product = a * b;
        builder
            .assert_zero(&(product.clone() + c.clone() - F::new(10)))
            .unwrap();
        builder.assert_zero(&(product + c - F::new(10))).unwrap();
        assert_eq!(alu_rows(&builder), 1 + 2 * 2);
    }
}
