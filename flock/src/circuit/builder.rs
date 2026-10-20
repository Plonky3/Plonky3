//! Gate lists under construction.

use alloc::vec;
use alloc::vec::Vec;
use core::mem;
use core::ops::Range;

use super::{Circuit, Gate, Wire, Word, slot};

/// A gate list under construction.
#[derive(Debug)]
pub struct Builder {
    /// Gates so far.
    gates: Vec<Gate>,
    /// Slot of the next product.
    next_slot: usize,
    /// The constant wire.
    one: Wire,
    /// Each input port's wires, low bit first.
    inputs: Vec<Vec<Wire>>,
    /// Each output port's first slot and width in bits.
    outputs: Vec<(usize, usize)>,
    /// Whether each output slot is already committed, from the first output slot on.
    driven: Vec<bool>,
    /// Words the input ports span.
    n_input_words: usize,
}

impl Builder {
    /// A circuit with input and output ports of the given widths in bits, each rounded up to whole words.
    #[must_use]
    pub fn new(input_bits: &[usize], output_bits: &[usize]) -> Self {
        let inputs: Vec<Range<usize>> = input_bits.iter().map(|&bits| 0..bits).collect();
        Self::with_input_ranges(&inputs, output_bits)
    }

    /// Input ports given as the range of their free bits; bits below a range are structural zeros.
    #[must_use]
    pub fn with_input_ranges(inputs: &[Range<usize>], output_bits: &[usize]) -> Self {
        // Ports are whole words; the constant sits right after them.
        let words = |bits: usize| bits.div_ceil(64);
        let n_input_words = inputs.iter().map(|r| words(r.end)).sum::<usize>();
        let n_output_words = output_bits.iter().map(|&b| words(b)).sum::<usize>();
        let const_pos = 64 * (n_input_words + n_output_words);
        let mut c = Self {
            gates: Vec::new(),
            next_slot: const_pos + 1,
            one: Wire::ZERO,
            inputs: Vec::new(),
            outputs: Vec::new(),
            driven: vec![false; 64 * n_output_words],
            n_input_words,
        };
        c.one = c.push(Gate::Free(slot(const_pos)));

        // A free wire per bit of each input range.
        let mut base = 0;
        for bits in inputs {
            let wires = (0..bits.end)
                .map(|i| {
                    if bits.contains(&i) {
                        c.push(Gate::Free(slot(base + i)))
                    } else {
                        Wire::ZERO
                    }
                })
                .collect();
            c.inputs.push(wires);
            base += 64 * words(bits.end);
        }

        // Output slots, filled later by copies or products.
        for &bits in output_bits {
            c.outputs.push((base, bits));
            base += 64 * words(bits);
        }
        c
    }

    /// Append a gate, returning its wire.
    fn push(&mut self, gate: Gate) -> Wire {
        self.gates.push(gate);
        Wire::gate(slot(self.gates.len() - 1))
    }

    /// The constant one.
    #[must_use]
    pub const fn one(&self) -> Wire {
        self.one
    }

    /// An input port's wires, low bit first.
    #[must_use]
    pub fn input(&self, port: usize) -> &[Wire] {
        &self.inputs[port]
    }

    /// `x + y`, free.
    pub fn xor(&mut self, x: Wire, y: Wire) -> Wire {
        match (x.index(), y.index()) {
            (Some(x), Some(y)) => self.push(Gate::Xor(x, y)),
            // A structural zero leaves the other wire.
            _ if x.is_zero() => y,
            _ => x,
        }
    }

    /// `1 + x`, free.
    pub fn not(&mut self, x: Wire) -> Wire {
        self.xor(x, self.one)
    }

    /// `x * y`: one committed product, unless an operand is a constant.
    pub fn and(&mut self, x: Wire, y: Wire) -> Wire {
        // The constant one is the identity.
        if x == self.one {
            return y;
        }
        if y == self.one {
            return x;
        }
        let (Some(x), Some(y)) = (x.index(), y.index()) else {
            return Wire::ZERO;
        };
        let s = slot(self.next_slot);
        self.next_slot += 1;
        self.push(Gate::And(x, y, s))
    }

    /// `x OR y = x + y + x y`: one product.
    pub fn or(&mut self, x: Wire, y: Wire) -> Wire {
        let both = self.and(x, y);
        let either = self.xor(x, y);
        self.xor(either, both)
    }

    /// `if s { x } else { y } = y + s (x + y)`: one product.
    pub fn mux(&mut self, s: Wire, x: Wire, y: Wire) -> Wire {
        let d = self.xor(x, y);
        let picked = self.and(s, d);
        self.xor(picked, y)
    }

    /// The low `bits` bits of `value` as constant wires, free.
    #[must_use]
    pub fn literal(&self, value: u64, bits: usize) -> Word {
        (0..bits)
            .map(|i| {
                if (value >> i) & 1 == 1 {
                    self.one
                } else {
                    Wire::ZERO
                }
            })
            .collect()
    }

    /// `x ^ y`, free.
    pub fn xor_word(&mut self, x: &[Wire], y: &[Wire]) -> Word {
        x.iter().zip(y).map(|(&x, &y)| self.xor(x, y)).collect()
    }

    /// `!x`, free.
    pub fn not_word(&mut self, x: &[Wire]) -> Word {
        x.iter().map(|&x| self.not(x)).collect()
    }

    /// `x & y`, one product per bit that is not constant.
    pub fn and_word(&mut self, x: &[Wire], y: &[Wire]) -> Word {
        x.iter().zip(y).map(|(&x, &y)| self.and(x, y)).collect()
    }

    /// `!x & y`, one product per bit: Keccak's chi.
    pub fn andn_word(&mut self, x: &[Wire], y: &[Wire]) -> Word {
        let not_x = self.not_word(x);
        self.and_word(&not_x, y)
    }

    /// SHA-256's choice `(e & f) ^ (!e & g) = g ^ e (f ^ g)`, one product per bit.
    pub fn ch(&mut self, e: &[Wire], f: &[Wire], g: &[Wire]) -> Word {
        let fg = self.xor_word(f, g);
        let picked = self.and_word(e, &fg);
        self.xor_word(&picked, g)
    }

    /// SHA-256's majority `b ^ (a ^ b)(b ^ c)`, one product per bit.
    pub fn maj(&mut self, a: &[Wire], b: &[Wire], c: &[Wire]) -> Word {
        let ab = self.xor_word(a, b);
        let bc = self.xor_word(b, c);
        let both = self.and_word(&ab, &bc);
        self.xor_word(&both, b)
    }

    /// The sum of any number of words modulo `2^n`, by column compression.
    ///
    /// A full adder costs one product, its majority; carries out of the top bit are never made.
    ///
    /// Constant bits fold: two carry a one up, and one beside a single bit is a free increment.
    ///
    /// # Panics
    ///
    /// - The operands differ in width.
    pub fn wrapping_sum(&mut self, operands: &[&[Wire]]) -> Word {
        let n = operands.first().map_or(0, |w| w.len());
        assert!(
            operands.iter().all(|w| w.len() == n),
            "operands share one width"
        );
        let one = self.one;
        let mut sum = Vec::with_capacity(n);

        // Each column's variable bits and constant ones, carries included.
        let (mut carries, mut carried_ones) = (Vec::new(), 0usize);
        for j in 0..n {
            let mut bits: Vec<Wire> = mem::take(&mut carries);
            let mut ones = mem::take(&mut carried_ones);
            for w in operands {
                match w[j] {
                    bit if bit.is_zero() => {}
                    bit if bit == one => ones += 1,
                    bit => bits.push(bit),
                }
            }

            // The top column only XORs: its carries fall off the modulus.
            if j + 1 == n {
                let mut s = if ones % 2 == 1 { one } else { Wire::ZERO };
                for bit in bits {
                    s = self.xor(s, bit);
                }
                sum.push(s);
                break;
            }

            // Two constant ones are a constant carry.
            carried_ones = ones / 2;
            ones %= 2;

            // Full adders: three bits become their sum here and their majority above, for one product.
            while bits.len() >= 3 {
                let at = bits.len() - 3;
                let (x, y, z) = (bits[at], bits[at + 1], bits[at + 2]);
                bits.truncate(at);
                let xz = self.xor(x, z);
                let yz = self.xor(y, z);
                let both = self.and(xz, yz);
                carries.push(self.xor(both, z));
                bits.push(self.xor(xz, y));
            }

            // At most two bits and one constant are left.
            let s = match (bits.as_slice(), ones) {
                ([], 0) => Wire::ZERO,
                ([], _) => one,
                (&[x], 0) => x,
                // `x + 1`: the sum is `!x`, the carry `x`, both free.
                (&[x], _) => {
                    carries.push(x);
                    self.not(x)
                }
                // A half adder: the carry is the product.
                (&[x, y], 0) => {
                    carries.push(self.and(x, y));
                    self.xor(x, y)
                }
                // `x + y + 1`: the carry is `x OR y`, one product.
                (&[x, y], _) => {
                    carries.push(self.or(x, y));
                    let xy = self.xor(x, y);
                    self.not(xy)
                }
                _ => unreachable!("full adders leave at most two bits"),
            };
            sum.push(s);
        }
        sum.into_iter().collect()
    }

    /// Commit a wire as one bit of an output port.
    ///
    /// # Panics
    ///
    /// - The bit is outside the port, or already committed.
    pub fn output(&mut self, port: usize, bit: usize, wire: Wire) {
        let s = self.commit_output(port, bit);
        // A structural zero needs no gate: its empty row forces zero.
        if let Some(wire) = wire.index() {
            self.push(Gate::Copy(wire, s));
        }
    }

    /// `x * y`, committed directly as one bit of an output port.
    ///
    /// # Panics
    ///
    /// - The bit is outside the port, or already committed.
    pub fn and_output(&mut self, port: usize, bit: usize, x: Wire, y: Wire) -> Wire {
        let s = self.commit_output(port, bit);
        let (Some(x), Some(y)) = (x.index(), y.index()) else {
            return Wire::ZERO;
        };
        self.push(Gate::And(x, y, s))
    }

    /// Claim one bit of an output port, returning its slot.
    ///
    /// A slot holds one row, so a second commit would silently drop the first.
    fn commit_output(&mut self, port: usize, bit: usize) -> u32 {
        let (base, bits) = self.outputs[port];
        assert!(bit < bits, "output port {port} has {bits} bits");
        let index = base + bit - 64 * self.n_input_words;
        assert!(
            !core::mem::replace(&mut self.driven[index], true),
            "output port {port}, bit {bit} is already committed"
        );
        slot(base + bit)
    }

    /// The finished circuit, padded to a power of two bits.
    #[must_use]
    pub fn finish(self) -> Circuit {
        let useful_bits = self.next_slot;
        let output_words: usize = self
            .outputs
            .iter()
            .map(|&(_, bits)| bits.div_ceil(64))
            .sum();
        Circuit::new(
            self.gates,
            64 * (self.n_input_words + output_words),
            // At least one packed word per instance.
            useful_bits.next_power_of_two().trailing_zeros().max(6) as usize,
            useful_bits,
            self.n_input_words,
        )
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::*;

    /// Run a circuit on one instance and return its output words.
    fn run(c: &Circuit, inputs: &[u64]) -> Vec<u64> {
        let words = (1 << c.k_log()) / 64;
        let (mut z, mut az, mut bz) = (vec![0; words], vec![0; words], vec![0; words]);
        c.witness_instance(inputs, &mut z, &mut az, &mut bz);
        c.output_words(&z).to_vec()
    }

    /// Commit a word as the low bits of an output port.
    fn output_word(c: &mut Builder, port: usize, offset: usize, word: &[Wire]) {
        for (bit, &w) in word.iter().enumerate() {
            c.output(port, offset + bit, w);
        }
    }

    /// A circuit adding four 32-bit inputs and a constant, packed two per word.
    fn sum_circuit(constant: u32) -> Circuit {
        let mut c = Builder::new(&[64, 64], &[32]);
        let ws: Vec<Word> = (0..2).flat_map(|p| Word::split_u32(c.input(p))).collect();
        let k = c.literal(u64::from(constant), 32);
        let ops: Vec<&[Wire]> = ws.iter().map(|w| &w[..]).chain([&k[..]]).collect();
        let s = c.wrapping_sum(&ops);
        output_word(&mut c, 0, 0, &s);
        c.finish()
    }

    #[test]
    fn a_two_operand_sum_costs_one_product_per_carry() {
        // Invariant: 31 products for 32 bits, the top carry never made.
        let mut c = Builder::new(&[64], &[32]);
        let ws = Word::split_u32(c.input(0));
        let s = c.wrapping_sum(&[&ws[0], &ws[1]]);
        output_word(&mut c, 0, 0, &s);
        assert_eq!(c.finish().num_ands(), 31);

        // A three-operand sum fuses into 61 products, one fewer than two chained sums.
        let mut c = Builder::new(&[64, 64], &[32]);
        let ws: Vec<Word> = (0..2).flat_map(|p| Word::split_u32(c.input(p))).collect();
        let s = c.wrapping_sum(&[&ws[0], &ws[1], &ws[2]]);
        output_word(&mut c, 0, 0, &s);
        assert_eq!(c.finish().num_ands(), 61);
    }

    #[test]
    fn a_product_with_a_constant_is_free() {
        // Invariant: the constant one is the identity and the structural zero absorbs.
        let mut c = Builder::new(&[64], &[64]);
        let (x, one) = (c.input(0)[0], c.one());
        assert_eq!(c.and(x, one), x);
        assert_eq!(c.and(one, x), x);
        assert_eq!(c.and(Wire::ZERO, x), Wire::ZERO);
        let xy = c.and(x, c.input(0)[1]);
        c.output(0, 0, xy);
        assert_eq!(c.finish().num_ands(), 1);
    }

    #[test]
    fn sums_wrap_modulo_two_to_the_thirty_two() {
        // Invariant: the column compressor is the wrapping sum, constants included.
        let mut rng = SmallRng::seed_from_u64(0xADD);
        for k in [0, u32::MAX, 1, rng.random()] {
            let c = sum_circuit(k);
            for _ in 0..16 {
                let x: [u32; 4] = rng.random();
                let pack = |lo: u32, hi: u32| u64::from(lo) | (u64::from(hi) << 32);
                let out = run(&c, &[pack(x[0], x[1]), pack(x[2], x[3])]);
                let expected = x.iter().fold(k, |s, &v| s.wrapping_add(v));
                assert_eq!(out[0], u64::from(expected), "constant {k}, operands {x:?}");
            }
        }
    }

    #[test]
    fn word_operations_match_the_native_ones() {
        // Invariant: ch, maj, and-not, rotations and the shift are the native operations.
        let mut c = Builder::new(&[64, 32], &[64, 64, 64]);
        let ws: Vec<Word> = (0..2).flat_map(|p| Word::split_u32(c.input(p))).collect();
        let outs = [
            c.ch(&ws[0], &ws[1], &ws[2]),
            c.maj(&ws[0], &ws[1], &ws[2]),
            c.andn_word(&ws[0], &ws[1]),
            ws[0].rotate_right(7),
            ws[1].rotate_left(3),
            &ws[2] >> 10,
        ];
        for (i, out) in outs.iter().enumerate() {
            output_word(&mut c, i / 2, 32 * (i % 2), out);
        }
        let c = c.finish();
        assert_eq!(c.num_ands(), 3 * 32);

        let mut rng = SmallRng::seed_from_u64(0x3A1);
        for _ in 0..64 {
            let [e, f, g]: [u32; 3] = rng.random();
            let out = run(&c, &[u64::from(e) | (u64::from(f) << 32), u64::from(g)]);
            let expected = [
                (e & f) ^ (!e & g),
                (e & f) ^ (e & g) ^ (f & g),
                !e & f,
                e.rotate_right(7),
                f.rotate_left(3),
                g >> 10,
            ];
            for (i, &v) in expected.iter().enumerate() {
                assert_eq!((out[i / 2] >> (32 * (i % 2))) as u32, v, "operation {i}");
            }
        }
    }
}
