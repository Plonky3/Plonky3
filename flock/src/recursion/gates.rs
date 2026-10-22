//! The circuit under construction, its advice and revealed bits, and the field gadgets on wires.
//!
//! A product in `GF(2^64)` is a Karatsuba carry-less product to the single bit, 729 ANDs, then a free reduction.
//!
//! A product in `GF(2^192)` is a three-term Karatsuba over the 64-bit limbs, six such products or 4,374 ANDs.
//!
//! Every linear map (constant products, squares, reductions) is read off the native field on basis vectors, so it costs XORs only.

use alloc::collections::VecDeque;
use alloc::vec;
use alloc::vec::Vec;

use p3_binary_field::{Poly64, Poly192};
use p3_field::{BasedVectorSpace, PrimeCharacteristicRing};

use crate::circuit::{Builder, Wire};

/// A field whose elements are bit vectors in a fixed coordinate order.
pub trait BinaryElement:
    PrimeCharacteristicRing<PrimeSubfield = p3_binary_field::Gf2>
    + Copy
    + PartialEq
    + Send
    + Sync
    + 'static
{
    /// Bits of an element.
    const BITS: usize;

    /// Bit `i` of the element.
    fn bit(&self, i: usize) -> bool;

    /// The element with the given bits.
    fn from_bits(bits: impl Fn(usize) -> bool) -> Self;
}

impl BinaryElement for Poly64 {
    const BITS: usize = 64;

    fn bit(&self, i: usize) -> bool {
        (self.to_bits() >> i) & 1 == 1
    }

    fn from_bits(bits: impl Fn(usize) -> bool) -> Self {
        Self::new((0..64).fold(0, |acc, i| acc | (u64::from(bits(i)) << i)))
    }
}

impl BinaryElement for Poly192 {
    const BITS: usize = 192;

    fn bit(&self, i: usize) -> bool {
        limbs_of(self)[i / 64].bit(i % 64)
    }

    fn from_bits(bits: impl Fn(usize) -> bool) -> Self {
        Self::new(core::array::from_fn(|l| {
            Poly64::from_bits(|i| bits(64 * l + i))
        }))
    }
}

/// The three committed-field coordinates of a challenge-field element.
pub(crate) fn limbs_of(value: &Poly192) -> [Poly64; 3] {
    let limbs = <Poly192 as BasedVectorSpace<Poly64>>::as_basis_coefficients_slice(value);
    [limbs[0], limbs[1], limbs[2]]
}

/// One byte of a transcript or a digest: its wires, low bit first, and its native value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Byte {
    /// The wires, low bit first.
    pub wires: [Wire; 8],
    /// The native value.
    pub value: u8,
}

/// A draw of the native transcript, waiting for the verifier to take it.
#[derive(Clone, Debug)]
pub(crate) enum Sample {
    /// One committed-field element, as its 64 wires and native value.
    Value(Vec<Wire>, Poly64),
    /// A query index, as its bits, least significant first, with their native values.
    Index(Vec<(Wire, bool)>),
}

/// Products and gates spent so far in one part of the circuit.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Cost {
    /// The part of the verifier.
    pub component: &'static str,
    /// Committed products.
    pub ands: usize,
    /// Gates, XORs included.
    pub gates: usize,
    /// Products in `GF(2^64)`.
    pub mul64: usize,
    /// Products of a `GF(2^192)` element by a `GF(2^64)` one.
    pub mul_mixed: usize,
    /// Products in `GF(2^192)`.
    pub mul192: usize,
}

/// The circuit under construction, with the native values of its advice and revealed bits.
#[derive(Debug)]
pub struct Gates {
    /// The gate list.
    pub(crate) builder: Builder,
    /// Native advice bits, packed 64 per word.
    pub(crate) advice: Vec<u64>,
    /// The revealed bits the statement pins, packed 64 per word.
    pub(crate) revealed: Vec<u64>,
    /// Bytes the native transcript absorbs next, in absorb order.
    pub(crate) pending: VecDeque<Byte>,
    /// Samples the native transcript drew and the verifier has not yet taken.
    pub(crate) sampled: VecDeque<Sample>,
    /// Whether the transcript under construction absorbs its seed.
    pub(crate) seeding: bool,
    /// Digests a seed holds as constants that are wires here: each native digest with its wires.
    pub(crate) substitutions: Vec<([u8; 32], [Byte; 32])>,
    /// Cost of every finished component, in order.
    pub(crate) costs: Vec<Cost>,
    /// The running component's counters, its products and gates as offsets.
    pub(crate) current: Cost,
    /// The OR of every asserted bit: the statement pins it to zero.
    pub(crate) fail: Wire,
    /// Linear maps already lowered, by the native constant that defines them.
    linear: Vec<(LinearKey, Vec<Vec<u32>>)>,
}

/// The native definition of a linear map on `GF(2^192)` or `GF(2^64)`.
#[derive(Clone, Copy, Debug, PartialEq)]
enum LinearKey {
    /// `x -> c x` in `GF(2^64)`.
    Mul64(Poly64),
    /// `x -> c x` in `GF(2^192)`.
    Mul192(Poly192),
    /// `x -> x^2` in `GF(2^64)`.
    Square64,
    /// `x -> x^2` in `GF(2^192)`.
    Square192,
}

impl Gates {
    /// An empty circuit with no declared port.
    pub(crate) fn new() -> Self {
        Self {
            builder: Builder::new(&[], &[]),
            advice: Vec::new(),
            revealed: Vec::new(),
            pending: VecDeque::new(),
            sampled: VecDeque::new(),
            seeding: false,
            substitutions: Vec::new(),
            costs: Vec::new(),
            current: Cost {
                component: "setup",
                ..Cost::default()
            },
            fail: Wire::ZERO,
            linear: Vec::new(),
        }
    }

    /// Require a bit to be zero: it joins the OR the statement pins to zero, one product.
    pub(crate) fn assert_zero(&mut self, wire: Wire) {
        self.fail = self.builder.or(self.fail, wire);
    }

    /// Close the running component and open `component`, returning the closed one's name.
    pub(crate) fn component(&mut self, component: &'static str) -> &'static str {
        let (ands, gates) = (self.builder.num_ands(), self.builder.num_gates());
        let mut done = core::mem::replace(
            &mut self.current,
            Cost {
                component,
                ands,
                gates,
                ..Cost::default()
            },
        );
        done.ands = ands - done.ands;
        done.gates = gates - done.gates;
        match self
            .costs
            .iter_mut()
            .find(|c| c.component == done.component)
        {
            Some(c) => {
                c.ands += done.ands;
                c.gates += done.gates;
                c.mul64 += done.mul64;
                c.mul_mixed += done.mul_mixed;
                c.mul192 += done.mul192;
            }
            None => self.costs.push(done.clone()),
        }
        done.component
    }

    /// Fresh advice wires carrying the given bits.
    pub(crate) fn advice(&mut self, bits: impl IntoIterator<Item = bool>) -> Vec<Wire> {
        let bits: Vec<bool> = bits.into_iter().collect();
        let first = self.builder.advice_bits();
        let wires = self.builder.advice(bits.len());
        push_bits(&mut self.advice, first, &bits);
        wires
    }

    /// Reveal a wire, the statement pinning it to `value`.
    pub(crate) fn reveal(&mut self, wire: Wire, value: bool) {
        let first = self.builder.revealed_bits();
        self.builder.reveal(wire);
        push_bits(&mut self.revealed, first, &[value]);
    }

    /// The wire of a constant bit.
    pub(crate) const fn constant(&self, bit: bool) -> Wire {
        if bit { self.builder.one() } else { Wire::ZERO }
    }

    /// The wires of a constant element.
    pub(crate) fn constant_bits<E: BinaryElement>(&self, value: &E) -> Vec<Wire> {
        (0..E::BITS).map(|i| self.constant(value.bit(i))).collect()
    }

    /// `x + y`, bit by bit.
    pub(crate) fn xor(&mut self, x: &[Wire], y: &[Wire]) -> Vec<Wire> {
        self.builder.xor_word(x, y).to_vec()
    }

    /// `bit * x`, bit by bit.
    pub(crate) fn scale(&mut self, bit: Wire, x: &[Wire]) -> Vec<Wire> {
        x.iter().map(|&x| self.builder.and(bit, x)).collect()
    }

    /// The leaf operands of an operand's Karatsuba tree: low half, high half, their sum, recursively to single bits.
    ///
    /// A 64-bit operand has 729 leaves, and the leaf products of two operands are their carry-less product before recombination.
    pub(crate) fn expand(&mut self, x: &[Wire]) -> Vec<Wire> {
        let n = x.len();
        if n == 1 {
            return x.to_vec();
        }
        let h = n / 2;
        let (x0, x1) = x.split_at(h);
        let mut padded = x0.to_vec();
        padded.resize(n - h, Wire::ZERO);
        let sum = self.xor(&padded, x1);
        let mut out = self.expand(x0);
        out.extend(self.expand(x1));
        out.extend(self.expand(&sum));
        out
    }

    /// The leaf products of two expansions: one product per leaf.
    pub(crate) fn leaf_products(&mut self, x: &[Wire], y: &[Wire]) -> Vec<Wire> {
        x.iter()
            .zip(y)
            .map(|(&x, &y)| self.builder.and(x, y))
            .collect()
    }

    /// The carry-less product of two `n`-bit operands from the sum of their leaf products, `2n - 1` bits.
    pub(crate) fn recombine(&mut self, leaves: &[Wire], n: usize) -> Vec<Wire> {
        if n == 1 {
            return leaves.to_vec();
        }
        if leaves.iter().all(|w| w.is_zero()) {
            return vec![Wire::ZERO; 2 * n - 1];
        }
        let h = n / 2;
        let (lh, lr) = (leaf_count(h), leaf_count(n - h));
        let low = self.recombine(&leaves[..lh], h);
        let high = self.recombine(&leaves[lh..lh + lr], n - h);
        let mut middle = self.recombine(&leaves[lh + lr..], n - h);
        // The middle term is (x0 + x1)(y0 + y1) + x0 y0 + x1 y1.
        for part in [&low, &high] {
            for (i, &w) in part.iter().enumerate() {
                middle[i] = self.builder.xor(middle[i], w);
            }
        }
        let mut out = vec![Wire::ZERO; 2 * n - 1];
        out[..low.len()].copy_from_slice(&low);
        out[2 * h..].copy_from_slice(&high);
        for (i, &w) in middle.iter().enumerate() {
            out[h + i] = self.builder.xor(out[h + i], w);
        }
        out
    }

    /// Reduce a 127-bit carry-less product into `GF(2^64)`, with `x^k` for `k >= 64` read off the native field.
    pub(crate) fn reduce64(&mut self, wide: &[Wire]) -> Vec<Wire> {
        let mut out = wide[..64].to_vec();
        for (k, &w) in wide.iter().enumerate().skip(64) {
            if w.is_zero() {
                continue;
            }
            let power = Poly64::new(1 << (k - 63)) * Poly64::new(1 << 63);
            for (j, o) in out.iter_mut().enumerate() {
                if power.bit(j) {
                    *o = self.builder.xor(*o, w);
                }
            }
        }
        out
    }

    /// The `GF(2^64)` value of summed leaf products.
    pub(crate) fn finish64(&mut self, leaves: &[Wire]) -> Vec<Wire> {
        let wide = self.recombine(leaves, 64);
        self.reduce64(&wide)
    }

    /// The `GF(2^192)` value of summed leaf products, by product slot.
    ///
    /// Slots `0..6` are the Karatsuba products `a0 b0, a1 b1, a2 b2, (a0+a1)(b0+b1), (a0+a2)(b0+b2), (a1+a2)(b1+b2)`; slots `6..9` add into the coefficients of `1, y, y^2` alone.
    pub(crate) fn finish192(&mut self, slots: &[Option<Vec<Wire>>; 9]) -> Vec<Wire> {
        // Slots feeding each coefficient of y^0 .. y^4, before reduction.
        const TERMS: [&[usize]; 5] = [&[0, 6], &[3, 0, 1, 7], &[4, 0, 2, 1, 8], &[5, 1, 2], &[2]];
        // y^3 and y^4 rewritten in the basis 1, y, y^2 by the native field: each limb is zero or one.
        let y = Poly192::new([Poly64::ZERO, Poly64::ONE, Poly64::ZERO]);
        let powers = [y.cube(), y.cube() * y];
        let mut incidence = [[false; 9]; 3];
        for (l, row) in incidence.iter_mut().enumerate() {
            let mut toggle = |d: usize| {
                for &t in TERMS[d] {
                    row[t] ^= true;
                }
            };
            toggle(l);
            for (power, d) in powers.iter().zip(3..) {
                let c = limbs_of(power)[l];
                assert!(
                    c == Poly64::ZERO || c == Poly64::ONE,
                    "the reduction of y^3 and y^4 has bit limbs"
                );
                if c == Poly64::ONE {
                    toggle(d);
                }
            }
        }
        // Recombination is linear, so each limb recombines the sum of its slots' leaves once.
        let mut out = Vec::with_capacity(192);
        for row in incidence {
            let mut leaves: Option<Vec<Wire>> = None;
            for (slot, &used) in slots.iter().zip(&row) {
                if let (true, Some(v)) = (used, slot) {
                    leaves = Some(leaves.map_or_else(|| v.clone(), |acc| self.xor(&acc, v)));
                }
            }
            out.extend(leaves.map_or_else(|| vec![Wire::ZERO; 64], |l| self.finish64(&l)));
        }
        out
    }

    /// The image of `x` under the linear map `key` names, its matrix read off the native field.
    fn linear(&mut self, key: LinearKey, x: &[Wire]) -> Vec<Wire> {
        let rows = match self.linear.iter().find(|(k, _)| *k == key) {
            Some((_, rows)) => rows.clone(),
            None => {
                let rows = match key {
                    LinearKey::Mul64(c) => matrix::<Poly64>(|e| c * e),
                    LinearKey::Mul192(c) => matrix::<Poly192>(|e| c * e),
                    LinearKey::Square64 => matrix::<Poly64>(|e| e.square()),
                    LinearKey::Square192 => matrix::<Poly192>(|e| e.square()),
                };
                if self.linear.len() >= 64 {
                    self.linear.remove(0);
                }
                self.linear.push((key, rows.clone()));
                rows
            }
        };
        rows.iter()
            .map(|row| {
                row.iter()
                    .fold(Wire::ZERO, |acc, &i| self.builder.xor(acc, x[i as usize]))
            })
            .collect()
    }

    /// `c x` in `GF(2^64)`, free.
    pub(crate) fn mul64_const(&mut self, c: Poly64, x: &[Wire]) -> Vec<Wire> {
        self.linear(LinearKey::Mul64(c), x)
    }

    /// `c x` in `GF(2^192)`, free.
    pub(crate) fn mul192_const(&mut self, c: Poly192, x: &[Wire]) -> Vec<Wire> {
        self.linear(LinearKey::Mul192(c), x)
    }

    /// `x^2` in `GF(2^64)`, free.
    pub(crate) fn square64(&mut self, x: &[Wire]) -> Vec<Wire> {
        self.linear(LinearKey::Square64, x)
    }

    /// `x^2` in `GF(2^192)`, free.
    pub(crate) fn square192(&mut self, x: &[Wire]) -> Vec<Wire> {
        self.linear(LinearKey::Square192, x)
    }
}

/// Leaves of the Karatsuba tree of an `n`-bit operand.
pub(crate) const fn leaf_count(n: usize) -> usize {
    if n == 1 {
        1
    } else {
        leaf_count(n / 2) + 2 * leaf_count(n - n / 2)
    }
}

/// Row `j` of a linear map's matrix: the inputs whose image has bit `j` set.
fn matrix<E: BinaryElement>(map: impl Fn(E) -> E) -> Vec<Vec<u32>> {
    let mut rows = vec![Vec::new(); E::BITS];
    for i in 0..E::BITS {
        let image = map(E::from_bits(|b| b == i));
        for (j, row) in rows.iter_mut().enumerate() {
            if image.bit(j) {
                row.push(i as u32);
            }
        }
    }
    rows
}

/// Write `bits` into packed words from bit `first` on, growing the words as needed.
fn push_bits(words: &mut Vec<u64>, first: usize, bits: &[bool]) {
    words.resize((first + bits.len()).div_ceil(64), 0);
    for (i, &bit) in bits.iter().enumerate() {
        let at = first + i;
        words[at / 64] |= u64::from(bit) << (at % 64);
    }
}
