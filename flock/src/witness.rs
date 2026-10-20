//! Packed witness tables for a batch of circuit instances, filled 64 instances per gate walk.

use alloc::vec;
use alloc::vec::Vec;

use p3_binary_field::PackedGf2x64;
use p3_maybe_rayon::prelude::*;

use crate::circuit::{Circuit, Gate};

/// Instances one gate walk covers: one per bit of a word.
const LANES: usize = 64;

/// One batch's packed tables, 64 bits per word, instance-major.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Witness {
    /// Witness bits.
    z: Vec<u64>,
    /// Left factor of every row, `A z`.
    az: Vec<u64>,
    /// Right factor of every row, `B z`.
    bz: Vec<u64>,
    /// Instances, as a power of two.
    log_instances: usize,
}

impl Witness {
    /// The witness of `2^log_instances` instances of a circuit, 64 per gate walk.
    ///
    /// Instances past the rows take the padding row, so the constant is one everywhere.
    ///
    /// # Panics
    ///
    /// - There are more rows than instances.
    pub fn generate<S: Sync>(
        circuit: &Circuit,
        log_instances: usize,
        rows: &[S],
        padding: &S,
        write_input: impl Fn(&S, &mut [u64]) + Sync,
    ) -> Self {
        assert!(rows.len() <= 1 << log_instances, "more rows than instances");
        let words = (1usize << circuit.k_log()) / 64;
        let total = words << log_instances;
        let (mut z, mut az, mut bz) = (vec![0; total], vec![0; total], vec![0; total]);

        // A batch below 64 instances is walked in full and stored in part.
        let lanes = LANES.min(1 << log_instances);
        let n_in = circuit.n_input_words();
        let live_words = circuit.useful_bits().div_ceil(64);
        let group = lanes * words;

        // Each group of instances owns one chunk of every table.
        z.par_chunks_exact_mut(group)
            .zip(az.par_chunks_exact_mut(group))
            .zip(bz.par_chunks_exact_mut(group))
            .enumerate()
            .for_each_init(
                || Lanes::new(circuit),
                |s, (g, ((z, az), bz))| {
                    // Phase 1: each lane's input words.
                    for (l, row_words) in
                        s.rows.chunks_exact_mut(n_in.max(1)).take(LANES).enumerate()
                    {
                        if n_in > 0 {
                            write_input(rows.get(g * lanes + l).unwrap_or(padding), row_words);
                        }
                    }

                    // Phase 2: input word w of 64 lanes, transposed, is its bits by slot.
                    for (w, by_slot) in s.inputs.as_chunks_mut::<LANES>().0[..n_in]
                        .iter_mut()
                        .enumerate()
                    {
                        *by_slot = core::array::from_fn(|l| s.rows[l * n_in + w]);
                        PackedGf2x64::transpose(PackedGf2x64::from_bits_slice_mut(by_slot));
                    }

                    // Phase 3: one gate walk for all lanes.
                    walk_lanes(circuit, s);

                    // Phase 4: back to instance-major, one transpose per 64-slot group.
                    for (by_slot, out) in [(&s.z, &mut *z), (&s.a, &mut *az), (&s.b, &mut *bz)] {
                        for (w, block) in by_slot.as_chunks::<LANES>().0[..live_words]
                            .iter()
                            .enumerate()
                        {
                            let mut m = *block;
                            PackedGf2x64::transpose(PackedGf2x64::from_bits_slice_mut(&mut m));
                            for (l, &word) in m[..lanes].iter().enumerate() {
                                out[l * words + w] = word;
                            }
                        }
                    }
                },
            );
        Self {
            z,
            az,
            bz,
            log_instances,
        }
    }

    /// Witness bits.
    #[must_use]
    pub fn z(&self) -> &[u64] {
        &self.z
    }

    /// Move the bits out, leaving only `A z` and `B z` to read.
    pub(crate) fn take_z(&mut self) -> Vec<u64> {
        core::mem::take(&mut self.z)
    }

    /// Left factor of every row, `A z`.
    #[must_use]
    pub fn az(&self) -> &[u64] {
        &self.az
    }

    /// Right factor of every row, `B z`.
    #[must_use]
    pub fn bz(&self) -> &[u64] {
        &self.bz
    }

    /// Instances, as a power of two.
    #[must_use]
    pub const fn log_instances(&self) -> usize {
        self.log_instances
    }

    /// Tables built some other way, for cross-checks.
    #[cfg(test)]
    pub(crate) const fn from_tables(
        z: Vec<u64>,
        az: Vec<u64>,
        bz: Vec<u64>,
        log_instances: usize,
    ) -> Self {
        Self {
            z,
            az,
            bz,
            log_instances,
        }
    }

    /// The three tables, to tamper with.
    #[cfg(test)]
    pub(crate) const fn tables_mut(&mut self) -> [&mut Vec<u64>; 3] {
        [&mut self.z, &mut self.az, &mut self.bz]
    }
}

/// Walk the gates once for 64 instances, bit `l` of every word being instance `l`.
fn walk_lanes(circuit: &Circuit, lanes: &mut Lanes) {
    let Lanes {
        inputs,
        wires,
        z,
        a,
        b,
        ..
    } = lanes;
    let const_pos = circuit.const_pos();
    wires.clear();
    for &gate in circuit.gates() {
        let v = match gate {
            Gate::Free(s) => {
                let s = s as usize;
                let v = if s == const_pos { u64::MAX } else { inputs[s] };
                (z[s], a[s], b[s]) = (v, v, u64::MAX);
                v
            }
            Gate::Xor(x, y) => wires[x as usize] ^ wires[y as usize],
            Gate::And(x, y, s) => {
                let (x, y) = (wires[x as usize], wires[y as usize]);
                let s = s as usize;
                (z[s], a[s], b[s]) = (x & y, x, y);
                x & y
            }
            Gate::Copy(x, s) => {
                let v = wires[x as usize];
                let s = s as usize;
                (z[s], a[s], b[s]) = (v, v, u64::MAX);
                v
            }
        };
        wires.push(v);
    }
}

/// One worker's scratch for the 64-instance walk.
struct Lanes {
    /// The lanes' input words, one row after another.
    rows: Vec<u64>,
    /// Input bits by slot.
    inputs: Vec<u64>,
    /// Every gate's value.
    wires: Vec<u64>,
    /// `z` by slot.
    z: Vec<u64>,
    /// `A z` by slot.
    a: Vec<u64>,
    /// `B z` by slot.
    b: Vec<u64>,
}

impl Lanes {
    /// Zeroed scratch for one circuit.
    fn new(circuit: &Circuit) -> Self {
        let slots = 1 << circuit.k_log();
        Self {
            rows: vec![0; LANES * circuit.n_input_words().max(1)],
            inputs: vec![0; slots],
            wires: Vec::with_capacity(circuit.gates().len()),
            z: vec![0; slots],
            a: vec![0; slots],
            b: vec![0; slots],
        }
    }
}
