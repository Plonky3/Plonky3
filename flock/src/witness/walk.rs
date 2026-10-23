//! The gate walk compiled for speed: XORs folded into the linear forms that read them, live wires packed into few registers.
//!
//! An operation reads linear forms, XORs of registers, so an XOR read once costs no operation of its own.
//!
//! The rows are the gate list's: a form is the sum its XOR tree computes, so `A` and `B` are unchanged.

use alloc::vec;
use alloc::vec::Vec;

use crate::circuit::{Gate, GateList};

/// Register that takes values nobody reads.
const SINK: u32 = 0;

/// Operations between two checkpoints of the backward walk.
const CHECKPOINT: usize = 1 << 10;

/// Terms past which an XOR is kept as its own operation.
const INLINE_CAP: usize = 1 << 12;

/// Header kind of an uncommitted sum.
const LIN: u32 = 0;

/// Header kind of a committed product.
const AND: u32 = 1;

/// Header kind of a committed affine wire.
const COPY: u32 = 2;

/// Header kind of a committed free wire.
const FREE: u32 = 3;

/// Tag of a wire folded into its readers, on its slab entry.
const INLINED: u32 = 1 << 31;

/// A wire whose XOR tree cancels to zero.
const EMPTY: u32 = u32::MAX;

/// One decoded operation, its forms as register lists.
#[derive(Clone, Copy, Debug)]
enum Op<'a> {
    /// `r[dst] = sum r[terms]`.
    Lin { dst: u32, terms: &'a [u32] },
    /// Row `(sum r[a]) * (sum r[b])` at a slot, and `r[dst]` the product.
    And {
        dst: u32,
        slot: u32,
        a: &'a [u32],
        b: &'a [u32],
    },
    /// Row `(sum r[a]) * 1` at a slot, and `r[dst]` the sum.
    Copy { dst: u32, slot: u32, a: &'a [u32] },
    /// Row `v * 1` at an input or constant slot, and `r[dst] = v`.
    Free { dst: u32, slot: u32 },
}

/// Flag of a one-word header holding the destination: products then take the slot after the previous one.
const SHORT: u32 = 1 << 31;

/// Destination registers a one-word header can name.
const SHORT_DST: u32 = 1 << 24;

/// Terms of a short sum or copy, and twice the bits of a short product's two counts.
const SHORT_COUNT: usize = 32;

/// Terms of each form of a short product.
const SHORT_FORM: usize = 4;

/// Decode the operation at `at`, returning it and the offset of the next one; `next_slot` follows the products.
///
/// A short header is the flag, the kind, five bits of counts (a product's two counts of two bits) and the destination.
///
/// A long header is the kind, then a sum's or copy's count, or a product's counts of 14 and 15 bits; the destination and slot follow.
#[inline(always)]
fn decode<'c>(code: &'c [u32], at: usize, next_slot: &mut u32) -> (Op<'c>, usize) {
    let header = code[at];
    let kind = (header >> 29) & 3;
    if header & SHORT != 0 {
        let dst = header & (SHORT_DST - 1);
        let n = ((header >> 24) & 0x1F) as usize;
        return match kind {
            LIN => (
                Op::Lin {
                    dst,
                    terms: &code[at + 1..at + 1 + n],
                },
                at + 1 + n,
            ),
            AND => {
                let (na, nb) = (n >> 2, n & 3);
                let slot = *next_slot;
                *next_slot += 1;
                let a = &code[at + 1..at + 1 + na];
                let b = &code[at + 1 + na..at + 1 + na + nb];
                (Op::And { dst, slot, a, b }, at + 1 + na + nb)
            }
            COPY => (
                Op::Copy {
                    dst,
                    slot: code[at + 1],
                    a: &code[at + 2..at + 2 + n],
                },
                at + 2 + n,
            ),
            _ => (
                Op::Free {
                    dst,
                    slot: code[at + 1],
                },
                at + 2,
            ),
        };
    }
    let dst = code[at + 1];
    match kind {
        LIN => {
            let n = (header & 0x1FFF_FFFF) as usize;
            (
                Op::Lin {
                    dst,
                    terms: &code[at + 2..at + 2 + n],
                },
                at + 2 + n,
            )
        }
        AND => {
            let (na, nb) = (
                ((header >> 15) & 0x3FFF) as usize,
                (header & 0x7FFF) as usize,
            );
            let slot = code[at + 2];
            *next_slot = slot + 1;
            let a = &code[at + 3..at + 3 + na];
            let b = &code[at + 3 + na..at + 3 + na + nb];
            (Op::And { dst, slot, a, b }, at + 3 + na + nb)
        }
        COPY => {
            let n = (header & 0x1FFF_FFFF) as usize;
            let slot = code[at + 2];
            let a = &code[at + 3..at + 3 + n];
            (Op::Copy { dst, slot, a }, at + 3 + n)
        }
        _ => (
            Op::Free {
                dst,
                slot: code[at + 2],
            },
            at + 3,
        ),
    }
}

/// Every term of an operation, in code order: the positions a register allocator rewrites.
#[inline(always)]
const fn term_range(code: &[u32], at: usize) -> core::ops::Range<usize> {
    let header = code[at];
    if header & SHORT != 0 {
        let n = ((header >> 24) & 0x1F) as usize;
        return match (header >> 29) & 3 {
            LIN => at + 1..at + 1 + n,
            AND => at + 1..at + 1 + (n >> 2) + (n & 3),
            COPY => at + 2..at + 2 + n,
            _ => at + 2..at + 2,
        };
    }
    match (header >> 29) & 3 {
        LIN => at + 2..at + 2 + (header & 0x1FFF_FFFF) as usize,
        AND => at + 3..at + 3 + ((header >> 15) & 0x3FFF) as usize + (header & 0x7FFF) as usize,
        COPY => at + 3..at + 3 + (header & 0x1FFF_FFFF) as usize,
        _ => at + 3..at + 3,
    }
}

/// A circuit's gates as register operations on linear forms.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Program {
    /// Operations in gate order, each a header, the destination register and a slot unless the header holds them, then its terms.
    code: Vec<u32>,
    /// Code offset of every `CHECKPOINT`-th operation, with the slot a short product there takes.
    checkpoints: Vec<(usize, u32)>,
    /// Operations.
    ops: usize,
    /// Terms of every form.
    terms: usize,
    /// Committed products.
    ands: usize,
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

/// XOR expansions waiting for their readers: the sorted operations they sum, and the reads left.
#[derive(Default)]
struct Slab {
    /// Entries, reused once their reads are done.
    entries: Vec<Entry>,
    /// Entries free for reuse, the most recently freed first.
    spare: Vec<u32>,
}

/// Terms an entry holds in place.
const SMALL: usize = 6;

/// One expansion: up to `SMALL` terms in place, more in a buffer.
#[derive(Default)]
struct Entry {
    /// Terms.
    len: u32,
    /// Reads left.
    reads: u32,
    /// The terms, when they fit.
    small: [u32; SMALL],
    /// The terms, when they do not.
    big: Vec<u32>,
}

impl Slab {
    /// Store an expansion read `reads` times, returning its tagged index; a long `terms` swaps buffers with the entry.
    fn insert(&mut self, terms: &mut Vec<u32>, reads: u32) -> u32 {
        let i = self.spare.pop().unwrap_or_else(|| {
            self.entries.push(Entry::default());
            (self.entries.len() - 1) as u32
        });
        let entry = &mut self.entries[i as usize];
        entry.len = terms.len() as u32;
        entry.reads = reads;
        if terms.len() <= SMALL {
            entry.small[..terms.len()].copy_from_slice(terms);
        } else {
            core::mem::swap(&mut entry.big, terms);
        }
        INLINED | i
    }

    /// Append the expansion of a wire mapped to `m` to `out`, consuming one read.
    #[inline]
    fn read(&mut self, m: u32, out: &mut Vec<u32>) {
        if m == EMPTY {
            return;
        }
        if m & INLINED == 0 {
            out.push(m);
            return;
        }
        let i = m & !INLINED;
        let entry = &mut self.entries[i as usize];
        entry.reads -= 1;
        let len = entry.len as usize;
        if len <= SMALL {
            out.extend_from_slice(&entry.small[..len]);
        } else if entry.reads == 0 && out.is_empty() {
            // The last read takes the buffer rather than a copy.
            core::mem::swap(&mut entry.big, out);
        } else {
            out.extend_from_slice(&entry.big);
        }
        if entry.reads == 0 {
            self.spare.push(i);
        }
    }
}

/// The symmetric difference of two sorted lists into `out`: their sum over `GF(2)`.
///
/// A single term past the other list's last is appended in place, so a chain of XORs costs linear time.
fn xor_sorted(x: &mut Vec<u32>, y: &mut Vec<u32>, out: &mut Vec<u32>) {
    if let &[t] = y.as_slice()
        && x.last().is_none_or(|&l| l < t)
    {
        x.push(t);
        core::mem::swap(x, out);
        return;
    }
    if let &[t] = x.as_slice()
        && y.last().is_none_or(|&l| l < t)
    {
        y.push(t);
        core::mem::swap(y, out);
        return;
    }
    out.clear();
    let (mut i, mut j) = (0, 0);
    while i < x.len() && j < y.len() {
        match x[i].cmp(&y[j]) {
            core::cmp::Ordering::Less => {
                out.push(x[i]);
                i += 1;
            }
            core::cmp::Ordering::Greater => {
                out.push(y[j]);
                j += 1;
            }
            core::cmp::Ordering::Equal => {
                i += 1;
                j += 1;
            }
        }
    }
    out.extend_from_slice(&x[i..]);
    out.extend_from_slice(&y[j..]);
}

/// Append an operation with its destination left zero: a short header when its counts allow, else a long one.
///
/// A product's slot is left out when it follows the previous product's; `next_slot` follows the products.
///
/// The code grows by an eighth at a time, so its slack stays small.
fn emit(
    code: &mut Vec<u32>,
    kind: u32,
    slot: Option<u32>,
    terms: [&[u32]; 2],
    next_slot: &mut u32,
) {
    let extra = 3 + terms[0].len() + terms[1].len();
    if code.capacity() - code.len() < extra {
        code.reserve_exact(extra.max(code.len() / 8));
    }
    let (na, nb) = (terms[0].len(), terms[1].len());
    let short = match kind {
        AND => na < SHORT_FORM && nb < SHORT_FORM && slot == Some(*next_slot),
        _ => na < SHORT_COUNT,
    };
    if kind == AND {
        *next_slot = slot.expect("a product's slot") + 1;
    }
    if short {
        let n = if kind == AND { (na << 2) | nb } else { na };
        code.push(SHORT | (kind << 29) | ((n as u32) << 24));
        if matches!(kind, COPY | FREE) {
            code.extend(slot);
        }
    } else {
        let counts = if kind == AND { (na << 15) | nb } else { na };
        code.extend([(kind << 29) | counts as u32, 0]);
        code.extend(slot);
    }
    code.extend_from_slice(terms[0]);
    code.extend_from_slice(terms[1]);
}

/// Pass 1 of the compiler: operations on operation indices, each gate mapped to its operation or to its folded expansion.
///
/// An XOR is folded when it is read once, or when copying its terms into every reader costs no more than keeping it.
///
/// Returns the code, the operations and the products.
fn fold_xors(gates: impl Iterator<Item = Gate>, reads: &[u8]) -> (Vec<u32>, usize, usize) {
    let mut map: Vec<u32> = vec![0; reads.len()];
    let mut slab = Slab::default();
    let mut code: Vec<u32> = Vec::new();
    let (mut x, mut y, mut sum) = (Vec::new(), Vec::new(), Vec::new());
    let mut ops = 0u32;
    let mut ands = 0;
    let mut next_slot = 0;
    for (i, gate) in gates.enumerate() {
        x.clear();
        y.clear();
        match gate {
            Gate::Xor(a, b) => {
                slab.read(map[a as usize], &mut x);
                slab.read(map[b as usize], &mut y);
                let n = reads[i] as usize;
                // An unread XOR only consumes its operands' reads.
                if n == 0 {
                    continue;
                }
                xor_sorted(&mut x, &mut y, &mut sum);
                let t = sum.len();
                // A sum of at most one term is that term; a longer one is copied into its readers when that is no dearer.
                if t <= 1 {
                    map[i] = sum.first().copied().unwrap_or(EMPTY);
                    continue;
                }
                if t <= INLINE_CAP && (n == 1 || t * n <= t + n) {
                    map[i] = slab.insert(&mut sum, n as u32);
                    continue;
                }
                emit(&mut code, LIN, None, [&sum, &[]], &mut next_slot);
            }
            Gate::And(a, b, s) => {
                slab.read(map[a as usize], &mut x);
                slab.read(map[b as usize], &mut y);
                ands += 1;
                emit(&mut code, AND, Some(s), [&x, &y], &mut next_slot);
            }
            Gate::Copy(a, s) => {
                slab.read(map[a as usize], &mut x);
                emit(&mut code, COPY, Some(s), [&x, &[]], &mut next_slot);
            }
            Gate::Free(s) => emit(&mut code, FREE, Some(s), [&[], &[]], &mut next_slot),
        }
        map[i] = ops;
        ops += 1;
    }
    (code, ops as usize, ands)
}

impl Program {
    /// Compile a gate list: XORs fold into their readers, then each operation's value lives in a register up to its last reader.
    ///
    /// With `consume`, the list is emptied segment by segment as it is read.
    pub(crate) fn new(
        gates: &mut GateList,
        consume: bool,
        const_pos: usize,
        k_log: usize,
        input_slots: usize,
    ) -> Self {
        // The list counted each gate's readers as it grew; the folding needs the count up to two.
        let reads = gates.take_reads();
        let (mut code, ops, ands) = if consume {
            fold_xors(gates.drain(), &reads)
        } else {
            fold_xors(gates.iter(), &reads)
        };
        drop(reads);

        // The last operation reading each one, itself when none does.
        let mut last_use: Vec<u32> = (0..ops as u32).collect();
        let mut at = 0;
        for k in 0..ops as u32 {
            let terms = term_range(&code, at);
            for p in terms.clone() {
                last_use[code[p] as usize] = k;
            }
            at = terms.end;
        }

        // Pass 2: registers, rewritten in place; operands free theirs before the result takes one.
        let mut reg: Vec<u32> = vec![SINK; ops];
        let mut free: Vec<u32> = Vec::new();
        let mut registers = 1u32;
        let mut checkpoints = Vec::with_capacity(ops.div_ceil(CHECKPOINT));
        let mut constant_rows = Vec::new();
        let mut terms = 0;
        let (mut at, mut next_slot) = (0, 0);
        for k in 0..ops {
            if k % CHECKPOINT == 0 {
                checkpoints.push((at, next_slot));
            }
            let range = term_range(&code, at);
            terms += range.len();
            for p in range.clone() {
                let o = code[p] as usize;
                code[p] = reg[o];
                // A form reading one operation twice frees it once.
                if last_use[o] as usize == k {
                    last_use[o] = 0;
                    free.push(reg[o]);
                }
            }
            let dst = if last_use[k] as usize > k {
                free.pop().unwrap_or_else(|| {
                    registers += 1;
                    registers - 1
                })
            } else {
                SINK
            };
            reg[k] = dst;
            if code[at] & SHORT != 0 {
                assert!(dst < SHORT_DST, "registers fit a short header");
                code[at] |= dst;
            } else {
                code[at + 1] = dst;
            }
            match decode(&code, at, &mut next_slot).0 {
                Op::Copy { slot, .. } | Op::Free { slot, .. } => constant_rows.push(slot),
                _ => {}
            }
            at = range.end;
        }
        code.shrink_to_fit();
        Self {
            code,
            checkpoints,
            ops,
            terms,
            ands,
            registers: registers as usize,
            constant_rows,
            const_pos,
            slots: 1 << k_log,
            input_slots,
        }
    }

    /// Committed products.
    pub(crate) const fn ands(&self) -> usize {
        self.ands
    }

    /// Operations the walk runs.
    pub(crate) const fn ops(&self) -> usize {
        self.ops
    }

    /// Terms of every linear form.
    pub(crate) const fn terms(&self) -> usize {
        self.terms
    }

    /// Registers a walk holds at once.
    pub(crate) const fn registers(&self) -> usize {
        self.registers
    }

    /// Every operation in order.
    #[inline(always)]
    fn for_each(&self, mut f: impl FnMut(Op<'_>)) {
        let (mut at, mut next_slot) = (0, 0);
        while at < self.code.len() {
            let (op, next) = decode(&self.code, at, &mut next_slot);
            f(op);
            at = next;
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
        let xor = |r: &[u64], terms: &[u32]| terms.iter().fold(0, |acc, &t| acc ^ r[t as usize]);
        self.for_each(|op| match op {
            Op::Lin { dst, terms } => r[dst as usize] = xor(r, terms),
            Op::And {
                dst,
                slot,
                a: x,
                b: y,
            } => {
                let (x, y) = (xor(r, x), xor(r, y));
                a[slot as usize] = x;
                b[slot as usize] = y;
                r[dst as usize] = x & y;
            }
            Op::Copy { dst, slot, a: x } => {
                let v = xor(r, x);
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
        });
    }

    /// Every committed row's `(A_0 w, B_0 w)` entries, `w(s)` the entry of slot `s`, by one forward walk.
    pub(crate) fn walk_rows_by<V: p3_field::PrimeCharacteristicRing>(
        &self,
        w: impl Fn(usize) -> V,
        mut row: impl FnMut(usize, V, V),
    ) {
        let wc = w(self.const_pos);
        let mut r = vec![V::ZERO; self.registers];
        let sum = |r: &[V], terms: &[u32]| {
            terms
                .iter()
                .fold(V::ZERO, |acc, &t| acc + r[t as usize].dup())
        };
        self.for_each(|op| match op {
            Op::Lin { dst, terms } => r[dst as usize] = sum(&r, terms),
            Op::And { dst, slot, a, b } => {
                row(slot as usize, sum(&r, a), sum(&r, b));
                r[dst as usize] = w(slot as usize);
            }
            Op::Copy { dst, slot, a } => {
                row(slot as usize, sum(&r, a), wc.dup());
                r[dst as usize] = w(slot as usize);
            }
            Op::Free { dst, slot } => {
                let ws = w(slot as usize);
                row(slot as usize, ws.dup(), wc.dup());
                r[dst as usize] = ws;
            }
        });
    }

    /// The forward walk on `L` column vectors at once.
    ///
    /// `column(s)` is the lanes' entry at slot `s`, and `row(s, a, b)` receives every committed row's `A` and `B` entries.
    pub(crate) fn walk_rows_lanes<V, const L: usize>(
        &self,
        column: impl Fn(usize) -> [V; L],
        mut row: impl FnMut(usize, &[V; L], &[V; L]),
    ) where
        V: Copy + Default + core::ops::Add<Output = V>,
    {
        let wc = column(self.const_pos);
        let zero = [V::default(); L];
        let mut r = vec![zero; self.registers];
        let sum = |r: &[[V; L]], terms: &[u32]| {
            terms.iter().fold(zero, |acc, &t| {
                let v = r[t as usize];
                core::array::from_fn(|l| acc[l] + v[l])
            })
        };
        self.for_each(|op| match op {
            Op::Lin { dst, terms } => r[dst as usize] = sum(&r, terms),
            Op::And { dst, slot, a, b } => {
                row(slot as usize, &sum(&r, a), &sum(&r, b));
                r[dst as usize] = column(slot as usize);
            }
            Op::Copy { dst, slot, a } => {
                row(slot as usize, &sum(&r, a), &wc);
                r[dst as usize] = column(slot as usize);
            }
            Op::Free { dst, slot } => {
                let w = column(slot as usize);
                row(slot as usize, &w, &wc);
                r[dst as usize] = w;
            }
        });
    }

    /// `(a A_0 + b B_0)^T u` by one backward walk, `u(s)` the weight of row `s`: `out(c, v)` receives every part `v` of column `c`.
    ///
    /// A register's adjoint collects its readers' weights and is taken where the operation writing it is met.
    pub(crate) fn fold_transpose_by<V: p3_field::Field>(
        &self,
        a: V,
        b: V,
        u: impl Fn(usize) -> V,
        mut out: impl FnMut(usize, V),
    ) {
        let c = self.const_pos;
        // The lincheck weighs `A` by one: its product is the weight itself.
        let scale_a = |us: V| if a == V::ONE { us } else { a * us };
        let mut adj = vec![V::ZERO; self.registers];
        let mut starts: Vec<(usize, u32)> = Vec::with_capacity(CHECKPOINT);
        let spread = |adj: &mut [V], terms: &[u32], g: V| {
            for &t in terms {
                adj[t as usize] += g;
            }
        };
        let ends = self.checkpoints[1..]
            .iter()
            .map(|&(at, _)| at)
            .chain([self.code.len()]);
        let chunks: Vec<((usize, u32), usize)> =
            self.checkpoints.iter().copied().zip(ends).collect();
        for &((first, slot), end) in chunks.iter().rev() {
            // Each operation's start and the slot it would take, then the operations backward.
            starts.clear();
            let (mut at, mut next_slot) = (first, slot);
            while at < end {
                starts.push((at, next_slot));
                at = decode(&self.code, at, &mut next_slot).1;
            }
            for &(at, mut next_slot) in starts.iter().rev() {
                let (op, _) = decode(&self.code, at, &mut next_slot);
                let dst = match op {
                    Op::Lin { dst, .. }
                    | Op::And { dst, .. }
                    | Op::Copy { dst, .. }
                    | Op::Free { dst, .. } => dst as usize,
                };
                // The written value's adjoint is complete; its register starts afresh for the operands.
                let g = core::mem::replace(&mut adj[dst], V::ZERO);
                match op {
                    Op::Lin { terms, .. } => spread(&mut adj, terms, g),
                    Op::And {
                        slot, a: x, b: y, ..
                    } => {
                        let s = slot as usize;
                        let us = u(s);
                        out(s, g);
                        spread(&mut adj, x, scale_a(us));
                        spread(&mut adj, y, b * us);
                    }
                    Op::Copy { slot, a: x, .. } => {
                        let s = slot as usize;
                        let us = u(s);
                        out(s, g);
                        spread(&mut adj, x, scale_a(us));
                        out(c, b * us);
                    }
                    Op::Free { slot, .. } => {
                        let s = slot as usize;
                        let us = u(s);
                        out(s, g + scale_a(us));
                        out(c, b * us);
                    }
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
