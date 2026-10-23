//! Packed witness tables for a batch of circuit instances, by the gate walk or a word-level generator.

mod walk;

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;

use p3_binary_field::PackedGf2x64;
use p3_maybe_rayon::prelude::*;

use self::walk::Lanes;
pub(crate) use self::walk::Program;
use crate::circuit::Circuit;
use crate::stream::Stream;

/// Instances one gate walk covers: one per bit of a word.
const LANES: usize = 64;

/// Slot groups transposed together: one cache line of each lane.
const BLOCK: usize = 8;

/// Instances a word-level generator fills before streaming them out.
const INSTANCE_GROUP: usize = 8;

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
        let n_in = circuit.n_input_words();
        let live_words = circuit.useful_bits().div_ceil(64);
        let program = circuit.program();
        Self::fill(
            words,
            log_instances,
            LANES,
            || (vec![0; LANES * n_in], program.lanes()),
            |(input_rows, s): &mut (Vec<u64>, Lanes), first, az, bz| {
                // A batch below 64 instances is walked in full and stored in part.
                let lanes = az.len() / words;

                // Phase 1: each lane's input words.
                if n_in > 0 {
                    for (l, row_words) in input_rows.chunks_exact_mut(n_in).enumerate() {
                        write_input(rows.get(first + l).unwrap_or(padding), row_words);
                    }
                }

                // Phase 2: input word w of 64 lanes, transposed, is its bits by slot.
                for (w, by_slot) in s.inputs.as_chunks_mut::<LANES>().0[..n_in]
                    .iter_mut()
                    .enumerate()
                {
                    *by_slot = core::array::from_fn(|l| input_rows[l * n_in + w]);
                    PackedGf2x64::transpose(PackedGf2x64::from_bits_slice_mut(by_slot));
                }

                // Phase 3: one gate walk for all lanes.
                program.run(s);

                // Phase 4: back to instance-major, eight 64-slot groups at a time.
                //
                // Whole lines per lane: single words at a power-of-two stride thrash the cache.
                for (by_slot, out) in [(&s.a, az), (&s.b, bz)] {
                    for first_word in (0..words).step_by(BLOCK) {
                        let n = BLOCK.min(words - first_word);
                        let mut block = [[0; LANES]; BLOCK];
                        for (j, m) in block[..n].iter_mut().enumerate() {
                            let w = first_word + j;
                            if w < live_words {
                                m.copy_from_slice(&by_slot[w * LANES..(w + 1) * LANES]);
                                PackedGf2x64::transpose(PackedGf2x64::from_bits_slice_mut(m));
                            }
                        }
                        for l in 0..lanes {
                            let at = l * words + first_word;
                            for (j, m) in block[..n].iter().enumerate() {
                                out[at + j] = m[l];
                            }
                        }
                    }
                }
            },
        )
    }

    /// The witness of instances of `2^k_log` bits from a word-level generator, one instance at a time.
    ///
    /// The generator writes every word of one instance's `A z` and `B z`.
    ///
    /// # Panics
    ///
    /// - There are more rows than instances.
    pub(crate) fn by_instance<S: Sync>(
        k_log: usize,
        log_instances: usize,
        rows: &[S],
        padding: &S,
        instance: impl Fn(&S, &mut [u64], &mut [u64]) + Sync,
    ) -> Self {
        assert!(rows.len() <= 1 << log_instances, "more rows than instances");
        let words = (1usize << k_log) / 64;
        Self::fill(
            words,
            log_instances,
            INSTANCE_GROUP,
            || (),
            |(), first, az, bz| {
                for (i, (az, bz)) in az
                    .chunks_exact_mut(words)
                    .zip(bz.chunks_exact_mut(words))
                    .enumerate()
                {
                    instance(rows.get(first + i).unwrap_or(padding), az, bz);
                }
            },
        )
    }

    /// Fill `group` instances of `words` words at a time in a worker's scratch, then stream them out.
    ///
    /// `fill(state, first, az, bz)` writes every word of the group's `A z` and `B z`, instance-major.
    fn fill<St>(
        words: usize,
        log_instances: usize,
        group: usize,
        init: impl Fn() -> St + Sync + Send,
        fill: impl Fn(&mut St, usize, &mut [u64], &mut [u64]) + Sync + Send,
    ) -> Self {
        let total = words << log_instances;
        let chunk = group.min(1 << log_instances) * words;
        let mut z = Box::<[u64]>::new_uninit_slice(total);
        let mut az = Box::<[u64]>::new_uninit_slice(total);
        let mut bz = Box::<[u64]>::new_uninit_slice(total);

        // Each group of instances owns one chunk of every table.
        z.par_chunks_mut(chunk)
            .zip(az.par_chunks_mut(chunk))
            .zip(bz.par_chunks_mut(chunk))
            .enumerate()
            .for_each_init(
                || (vec![0; chunk], vec![0; chunk], init()),
                |(sa, sb, st), (g, ((z, az), bz))| {
                    fill(st, g * (chunk / words), sa, sb);
                    // Every row is a product, so `z = A z AND B z` bitwise.
                    let stream = Stream::new();
                    stream.write_and(z, sa, sb);
                    stream.write(az, sa);
                    stream.write(bz, sb);
                },
            );

        // SAFETY: the chunks tile each table and every chunk was written in full.
        let (z, az, bz) = unsafe { (z.assume_init(), az.assume_init(), bz.assume_init()) };
        Self {
            z: z.into_vec(),
            az: az.into_vec(),
            bz: bz.into_vec(),
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
