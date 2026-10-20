//! The gate walk compiled for speed: dead XORs dropped and wires packed into few registers.

use alloc::vec;
use alloc::vec::Vec;

use crate::circuit::{Circuit, Gate};

/// Register that takes values nobody reads.
const SINK: u32 = 0;

/// One step of a compiled walk, on 64 lanes per register.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Op {
    /// `r[dst] = r[x] ^ r[y]`.
    Xor { dst: u32, x: u32, y: u32 },
    /// Row `r[x] * r[y]` at a slot, and `r[dst]` the product.
    And { dst: u32, x: u32, y: u32, slot: u32 },
    /// Row `r[x] * 1` at a slot, and `r[dst] = r[x]`.
    Copy { dst: u32, x: u32, slot: u32 },
    /// Row `v * 1` at an input or constant slot, and `r[dst] = v`.
    Free { dst: u32, slot: u32 },
}

/// A circuit's gates as register operations.
#[derive(Clone, Debug)]
pub(super) struct Program {
    /// Operations in gate order.
    ops: Vec<Op>,
    /// Registers the walk needs, the sink included.
    registers: usize,
    /// Slots whose right factor is the constant, port and copy rows.
    constant_rows: Vec<u32>,
    /// Slot of the constant.
    const_pos: usize,
    /// Bits per instance.
    slots: usize,
    /// Slots the input ports span.
    input_slots: usize,
}

impl Program {
    /// Compile a circuit: each wire lives in a register from its gate to its last reader.
    pub(super) fn new(circuit: &Circuit) -> Self {
        let gates = circuit.gates();

        // The last gate reading each wire.
        let mut last_use: Vec<u32> = vec![0; gates.len()];
        for (i, &gate) in gates.iter().enumerate() {
            let i = i as u32;
            match gate {
                Gate::Xor(x, y) | Gate::And(x, y, _) => {
                    last_use[x as usize] = i;
                    last_use[y as usize] = i;
                }
                Gate::Copy(x, _) => last_use[x as usize] = i,
                Gate::Free(_) => {}
            }
        }

        // Allocate on the fly: operands free their registers before the result takes one.
        let mut reg: Vec<u32> = vec![SINK; gates.len()];
        let mut free: Vec<u32> = Vec::new();
        let mut registers = 1u32;
        let mut ops = Vec::with_capacity(gates.len());
        let mut constant_rows = Vec::new();
        for (i, &gate) in gates.iter().enumerate() {
            let mut release = |w: u32, reg: &[u32]| {
                if last_use[w as usize] as usize == i && reg[w as usize] != SINK {
                    free.push(reg[w as usize]);
                }
            };
            match gate {
                Gate::Xor(x, y) | Gate::And(x, y, _) => {
                    release(x, &reg);
                    // A wire read twice by one gate frees once.
                    if y != x {
                        release(y, &reg);
                    }
                }
                Gate::Copy(x, _) => release(x, &reg),
                Gate::Free(_) => {}
            }
            let dst = if last_use[i] as usize > i {
                free.pop().unwrap_or_else(|| {
                    registers += 1;
                    registers - 1
                })
            } else {
                SINK
            };
            reg[i] = dst;
            let r = |w: u32| reg[w as usize];
            match gate {
                // An unread XOR does nothing.
                Gate::Xor(_, _) if dst == SINK => {}
                Gate::Xor(x, y) => ops.push(Op::Xor {
                    dst,
                    x: r(x),
                    y: r(y),
                }),
                Gate::And(x, y, slot) => ops.push(Op::And {
                    dst,
                    x: r(x),
                    y: r(y),
                    slot,
                }),
                Gate::Copy(x, slot) => {
                    constant_rows.push(slot);
                    ops.push(Op::Copy { dst, x: r(x), slot });
                }
                Gate::Free(slot) => {
                    constant_rows.push(slot);
                    ops.push(Op::Free { dst, slot });
                }
            }
        }
        Self {
            ops,
            registers: registers as usize,
            constant_rows,
            const_pos: circuit.const_pos(),
            slots: 1 << circuit.k_log(),
            input_slots: 64 * circuit.n_input_words(),
        }
    }

    /// Scratch for one worker's walks.
    pub(super) fn lanes(&self) -> Lanes {
        let mut b = vec![0; self.slots];
        for &s in &self.constant_rows {
            b[s as usize] = u64::MAX;
        }
        Lanes {
            registers: vec![0; self.registers],
            inputs: vec![0; self.input_slots],
            a: vec![0; self.slots],
            b,
        }
    }

    /// Walk once for 64 instances, bit `l` of every word being instance `l`.
    ///
    /// Reads `lanes.inputs` by slot and leaves the rows' factors in `lanes.a` and `lanes.b`.
    pub(super) fn run(&self, lanes: &mut Lanes) {
        let Lanes {
            registers: r,
            inputs,
            a,
            b,
        } = lanes;
        for &op in &self.ops {
            match op {
                Op::Xor { dst, x, y } => r[dst as usize] = r[x as usize] ^ r[y as usize],
                Op::And { dst, x, y, slot } => {
                    let (x, y) = (r[x as usize], r[y as usize]);
                    a[slot as usize] = x;
                    b[slot as usize] = y;
                    r[dst as usize] = x & y;
                }
                Op::Copy { dst, x, slot } => {
                    let v = r[x as usize];
                    a[slot as usize] = v;
                    r[dst as usize] = v;
                }
                Op::Free { dst, slot } => {
                    let s = slot as usize;
                    let v = if s == self.const_pos {
                        u64::MAX
                    } else {
                        inputs[s]
                    };
                    a[s] = v;
                    r[dst as usize] = v;
                }
            }
        }
    }
}

/// One worker's scratch for the 64-instance walk.
pub(super) struct Lanes {
    /// Live wires.
    registers: Vec<u64>,
    /// Input bits by slot.
    pub(super) inputs: Vec<u64>,
    /// `A z` by slot.
    pub(super) a: Vec<u64>,
    /// `B z` by slot.
    pub(super) b: Vec<u64>,
}
