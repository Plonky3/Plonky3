//! Gate lists packed into words: a gate whose operands sit close behind it takes one word.
//!
//! The words live in fixed segments, so a list grows without copying and a compiler can free it as it reads.

use alloc::vec::Vec;

use super::Gate;

/// Gates between two recorded positions.
pub(crate) const CHUNK: usize = 1 << 12;

/// Words of one segment.
const SEGMENT: usize = 1 << 20;

/// An XOR of two near wires: one word.
const XOR_NEAR: u32 = 0;

/// An XOR of any two wires: the tag, then both operands.
const XOR_FAR: u32 = 1;

/// A product of two near wires: one word, then the slot.
const AND_NEAR: u32 = 2;

/// A product of any two wires: the tag, both operands, the slot.
const AND_FAR: u32 = 3;

/// An affine wire: the tag, the operand, the slot.
const COPY: u32 = 4;

/// A free wire: the tag, the slot.
const FREE: u32 = 5;

/// Distances below which a first operand is near.
const NEAR_X: u32 = 1 << 14;

/// Distances below which a second operand is near.
const NEAR_Y: u32 = 1 << 15;

/// A gate list, each gate one to four words; operands near a gate are stored as distances back from it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct GateList {
    /// The packed gates, no gate straddling two segments.
    segments: Vec<Vec<u32>>,
    /// Gates.
    len: usize,
    /// Segment and offset of every `CHUNK`-th gate.
    chunks: Vec<(usize, usize)>,
    /// Readers of each gate, counted up to 255.
    reads: Vec<u8>,
}

/// Words of a gate from its first word.
const fn width(word: u32) -> usize {
    match word >> 29 {
        XOR_NEAR => 1,
        AND_NEAR | FREE => 2,
        XOR_FAR | COPY => 3,
        _ => 4,
    }
}

/// Decode the gate of index `i` from its words.
#[inline]
fn decode(i: u32, w: &[u32]) -> Gate {
    let near = |d: u32| (i - ((d >> 15) & 0x3FFF), i - (d & 0x7FFF));
    match w[0] >> 29 {
        XOR_NEAR => {
            let (x, y) = near(w[0]);
            Gate::Xor(x, y)
        }
        XOR_FAR => Gate::Xor(w[1], w[2]),
        AND_NEAR => {
            let (x, y) = near(w[0]);
            Gate::And(x, y, w[1])
        }
        AND_FAR => Gate::And(w[1], w[2], w[3]),
        COPY => Gate::Copy(w[1], w[2]),
        _ => Gate::Free(w[1]),
    }
}

impl GateList {
    /// Append a gate, returning its index.
    pub(crate) fn push(&mut self, gate: Gate) -> u32 {
        let i = u32::try_from(self.len).expect("gate indices fit in 32 bits");
        let near =
            |x: u32, y: u32| (i - x < NEAR_X && i - y < NEAR_Y).then(|| ((i - x) << 15) | (i - y));
        let mut count = |w: u32| self.reads[w as usize] = self.reads[w as usize].saturating_add(1);
        match gate {
            Gate::Xor(x, y) | Gate::And(x, y, _) => {
                count(x);
                count(y);
            }
            Gate::Copy(x, _) => count(x),
            Gate::Free(_) => {}
        }
        self.reads.push(0);
        let mut words = [0; 4];
        let n = match gate {
            Gate::Xor(x, y) => match near(x, y) {
                Some(d) => {
                    words[0] = (XOR_NEAR << 29) | d;
                    1
                }
                None => {
                    words[..3].copy_from_slice(&[XOR_FAR << 29, x, y]);
                    3
                }
            },
            Gate::And(x, y, s) => match near(x, y) {
                Some(d) => {
                    words[..2].copy_from_slice(&[(AND_NEAR << 29) | d, s]);
                    2
                }
                None => {
                    words = [AND_FAR << 29, x, y, s];
                    4
                }
            },
            Gate::Copy(x, s) => {
                words[..3].copy_from_slice(&[COPY << 29, x, s]);
                3
            }
            Gate::Free(s) => {
                words[..2].copy_from_slice(&[FREE << 29, s]);
                2
            }
        };
        if self
            .segments
            .last()
            .is_none_or(|seg| seg.len() + n > SEGMENT)
        {
            self.segments.push(Vec::with_capacity(SEGMENT));
        }
        let seg = self.segments.len() - 1;
        if self.len.is_multiple_of(CHUNK) {
            self.chunks.push((seg, self.segments[seg].len()));
        }
        self.segments[seg].extend_from_slice(&words[..n]);
        self.len += 1;
        i
    }

    /// Gates.
    pub(crate) const fn len(&self) -> usize {
        self.len
    }

    /// Readers of each gate, counted up to 255, taken out of the list.
    pub(crate) fn take_reads(&mut self) -> Vec<u8> {
        core::mem::take(&mut self.reads)
    }

    /// Every gate in order.
    pub(crate) fn iter(&self) -> Iter<'_> {
        Iter {
            segments: &self.segments,
            seg: 0,
            at: 0,
            index: 0,
            end: self.len as u32,
        }
    }

    /// The gates in runs of `CHUNK`, each run read on its own.
    pub(crate) fn runs(&self) -> Vec<Iter<'_>> {
        self.chunks
            .iter()
            .enumerate()
            .map(|(c, &(seg, at))| Iter {
                segments: &self.segments,
                seg,
                at,
                index: (c * CHUNK) as u32,
                end: ((c + 1) * CHUNK).min(self.len) as u32,
            })
            .collect()
    }

    /// Every gate in order, each segment freed once read; the list is left empty.
    pub(crate) fn drain(&mut self) -> Drain {
        let len = core::mem::take(&mut self.len);
        self.chunks = Vec::new();
        Drain {
            segments: core::mem::take(&mut self.segments).into_iter(),
            current: Vec::new(),
            at: 0,
            index: 0,
            end: len as u32,
        }
    }

    /// Rewrite every slot through `f`.
    pub(crate) fn relocate(&mut self, f: impl Fn(u32) -> u32) {
        for seg in &mut self.segments {
            let mut at = 0;
            while at < seg.len() {
                let slot = match seg[at] >> 29 {
                    XOR_NEAR | XOR_FAR => None,
                    AND_NEAR | FREE => Some(at + 1),
                    COPY => Some(at + 2),
                    _ => Some(at + 3),
                };
                if let Some(p) = slot {
                    seg[p] = f(seg[p]);
                }
                at += width(seg[at]);
            }
        }
    }
}

/// The gates of a list from some gate on.
#[derive(Clone, Debug)]
pub(crate) struct Iter<'a> {
    /// The packed gates.
    segments: &'a [Vec<u32>],
    /// Segment of the next gate.
    seg: usize,
    /// Offset of the next gate in its segment.
    at: usize,
    /// Index of the next gate.
    index: u32,
    /// Index past the last gate to yield.
    end: u32,
}

impl Iterator for Iter<'_> {
    type Item = Gate;

    #[inline]
    fn next(&mut self) -> Option<Gate> {
        if self.index == self.end {
            return None;
        }
        let mut seg = &self.segments[self.seg];
        if self.at == seg.len() {
            self.seg += 1;
            self.at = 0;
            seg = &self.segments[self.seg];
        }
        let w = &seg[self.at..];
        let gate = decode(self.index, w);
        self.at += width(w[0]);
        self.index += 1;
        Some(gate)
    }
}

/// The gates of a drained list, each segment freed once read.
#[derive(Debug)]
pub(crate) struct Drain {
    /// Segments not yet reached.
    segments: alloc::vec::IntoIter<Vec<u32>>,
    /// The segment being read.
    current: Vec<u32>,
    /// Offset of the next gate in it.
    at: usize,
    /// Index of the next gate.
    index: u32,
    /// Gates.
    end: u32,
}

impl Iterator for Drain {
    type Item = Gate;

    #[inline]
    fn next(&mut self) -> Option<Gate> {
        if self.index == self.end {
            return None;
        }
        if self.at == self.current.len() {
            // The read segment is dropped as the next one takes its place.
            self.current = self.segments.next()?;
            self.at = 0;
        }
        let w = &self.current[self.at..];
        let gate = decode(self.index, w);
        self.at += width(w[0]);
        self.index += 1;
        Some(gate)
    }
}
